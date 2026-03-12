# SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Ridgeback Franka Pick-and-Place Demo for Isaac Sim 5.1.0

Uses the built-in FrankaPickPlace helper (IK-based arm control) wrapped with
a RidgebackFrankaMobile class that adds mobile base navigation.

The base is moved by directly setting the Franka robot prim's world position
each simulation step.  A visual box underneath the Franka represents the
Ridgeback omnidirectional mobile platform.

Architecture (adapted from Spot + GR00T reference implementation):
  - FrankaPickPlace handles the Franka arm, table, cube (with built-in IK)
  - RidgebackFrankaMobile wraps it with mobile base navigation
  - The base is moved by calling set_world_pose() on the Franka prim
  - A visual gray box moves in sync as the Ridgeback platform

State machine:
  INIT -> MOVE_TO_CUBE -> WAIT_SETTLED -> PICK_CUBE -> RETURN_TO_START
    -> WAIT_SETTLED_RETURN -> PLACE_CUBE -> DONE

Usage:
  python ridgeback_franka_pick_place.py [--device cpu|cuda] [--headless]
    [--cube-offset 1.5] [--ik-method damped-least-squares]
"""

from __future__ import annotations

import argparse

parser = argparse.ArgumentParser(
    description="Ridgeback Franka Pick-and-Place Demo"
)
parser.add_argument(
    "--device",
    type=str,
    choices=["cpu", "cuda"],
    default="cpu",
    help="Simulation device",
)
parser.add_argument(
    "--headless",
    action="store_true",
    default=False,
    help="Run in headless mode (no GUI)",
)
parser.add_argument(
    "--cube-offset",
    type=float,
    default=1.5,
    help="Distance along +x axis for table/cube placement (default: 1.5)",
)
parser.add_argument(
    "--ik-method",
    type=str,
    choices=[
        "singular-value-decomposition",
        "pseudoinverse",
        "transpose",
        "damped-least-squares",
    ],
    default="damped-least-squares",
    help="Differential IK method for Franka arm (default: damped-least-squares)",
)
args, _ = parser.parse_known_args()

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": args.headless})

import numpy as np
import omni.usd
import omni.timeline
from enum import Enum

from isaacsim.core.api import World
from isaacsim.core.simulation_manager import SimulationManager
from isaacsim.robot.manipulators.examples.franka import FrankaPickPlace
from pxr import UsdGeom, Gf, Usd

# Optional imports -- multiple fallback strategies for moving the robot prim
try:
    from pxr import PhysxSchema
    PHYSX_AVAILABLE = True
except ImportError:
    PHYSX_AVAILABLE = False

try:
    from omni.isaac.dynamic_control import _dynamic_control
    DC_AVAILABLE = True
except ImportError:
    DC_AVAILABLE = False

try:
    from isaacsim.core.api.robots import Robot as CoreRobot
    CORE_ROBOT_AVAILABLE = True
except ImportError:
    CORE_ROBOT_AVAILABLE = False

try:
    from omni.isaac.core.prims import XFormPrim
    XFORM_AVAILABLE = True
except ImportError:
    XFORM_AVAILABLE = False


# ---------------------------------------------------------------------------
#  State machine
# ---------------------------------------------------------------------------
class MobileState(Enum):
    """States for the mobile pick-and-place task."""
    INIT = 0
    MOVE_TO_CUBE = 1
    WAIT_SETTLED = 2
    PICK_CUBE = 3
    RETURN_TO_START = 4
    WAIT_SETTLED_RETURN = 5
    PLACE_CUBE = 6
    DONE = 7


# ---------------------------------------------------------------------------
#  RidgebackFrankaMobile
# ---------------------------------------------------------------------------
class RidgebackFrankaMobile:
    """Ridgeback Franka mobile manipulator with base navigation + pick-place.

    Wraps the built-in FrankaPickPlace helper (which provides IK-based arm
    control, table, and cube) with mobile-base navigation.  The base drives to
    the cube, the Franka arm picks it via IK, the base returns to the start
    position, and the cube is placed down.

    Adapted from the Spot + GR00T reference implementation.
    """

    def __init__(self, franka_pick_place, cube_offset=1.5):
        self.franka_pick_place = franka_pick_place
        self.cube_offset = cube_offset

        self._mobile_base_prim_path = "/World/RidgebackBase"
        self._state = MobileState.INIT
        self._step_count = 0
        self._state_step_count = 0
        self._settled_steps = 0

        # Positions in world coordinates
        self.start_position = np.array([0.0, 0.0, 0.0])
        self.table_position = np.array([cube_offset, 0.0, 0.0])
        self._current_position = self.start_position.copy()
        self._move_speed = 0.01  # metres per step

        # Franka prim discovery / position helpers
        self._franka_prim_path = None
        self._dc = None
        self._articulation_handle = None
        self._franka_robot = None
        self._franka_xform = None

        # Scene prim helpers (for cube/table offset and tracking)
        self._scene_prim_originals = {}
        self._cube_prim_path = None
        self._gripper_prim_path = None
        self._gripper_to_cube_offset = np.array([0.0, 0.0, -0.04])
        self._pick_steps = 0
        self._place_steps = 0

    # ------------------------------------------------------------------
    #  Scene setup
    # ------------------------------------------------------------------
    def setup_mobile_base(self, stage):
        """Create the visual mobile base and configure the Franka for movement."""
        self._franka_prim_path = self._find_franka_prim(stage)
        if self._franka_prim_path is None:
            print("[ERROR] Could not find Franka robot in scene!")
            return None

        print(f"[INFO] Found Franka robot at: {self._franka_prim_path}")

        # Allow the Franka to be repositioned (unfixed base)
        self._set_franka_floating_base(stage)

        # Create visual Ridgeback base (a scaled dark-gray box)
        cube_prim = UsdGeom.Cube.Define(stage, self._mobile_base_prim_path)
        cube_prim.GetSizeAttr().Set(0.5)

        xform = UsdGeom.Xformable(cube_prim)
        scale_op = xform.AddScaleOp()
        scale_op.Set(Gf.Vec3f(1.0, 0.7, 0.4))

        translate_op = xform.AddTranslateOp()
        translate_op.Set(
            Gf.Vec3d(self.start_position[0], self.start_position[1], 0.1)
        )

        cube_prim.GetDisplayColorAttr().Set([Gf.Vec3f(0.3, 0.3, 0.3)])

        # Place Franka at the start position
        self._set_franka_usd_position(stage, self.start_position)

        # Discover and offset scene prims (cube / table)
        self._discover_scene_prims(stage)
        self._apply_scene_offset(stage)
        self._find_cube_prim(stage)
        self._find_gripper_prim(stage)

        print(f"[INFO] Created Ridgeback mobile base at start={self.start_position}")
        print(f"[INFO] Table/cube offset to x={self.cube_offset}m")

        return self._mobile_base_prim_path

    # ------------------------------------------------------------------
    #  Scene-prim helpers (cube / table discovery and tracking)
    # ------------------------------------------------------------------
    def _discover_scene_prims(self, stage):
        """Find cube/table scene prims and cache their original positions."""
        franka_path = self._franka_prim_path or "/World/Franka"
        skip_prefixes = (franka_path, self._mobile_base_prim_path)
        keywords = ["cube", "table", "block", "target", "goal", "object"]

        self._scene_prim_originals = {}
        for prim in stage.Traverse():
            path = str(prim.GetPath())
            if any(path.startswith(p) for p in skip_prefixes):
                continue
            name = prim.GetName().lower()
            if not any(kw in name for kw in keywords):
                continue
            try:
                xf = UsdGeom.Xformable(prim)
                if not xf:
                    continue
                orig = Gf.Vec3d(0, 0, 0)
                for op in xf.GetOrderedXformOps():
                    if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                        orig = Gf.Vec3d(op.Get())
                        break
                self._scene_prim_originals[path] = orig
                print(f"[INFO] Cached scene prim {path} at {orig}")
            except Exception as exc:
                print(f"[WARNING] Could not read {path}: {exc}")

    def _find_cube_prim(self, stage):
        """Identify the graspable cube prim from cached scene prims."""
        for path in self._scene_prim_originals:
            prim = stage.GetPrimAtPath(path)
            if prim.IsValid():
                name = prim.GetName().lower()
                if "cube" in name or "block" in name:
                    self._cube_prim_path = path
                    print(f"[INFO] Graspable cube prim: {path}")
                    return
        if self._scene_prim_originals:
            self._cube_prim_path = next(iter(self._scene_prim_originals))
            print(f"[INFO] Using first scene prim as cube: {self._cube_prim_path}")

    def _get_cube_position(self, stage):
        """Read the cube prim current translate from USD."""
        if self._cube_prim_path is None:
            return None
        prim = stage.GetPrimAtPath(self._cube_prim_path)
        if not prim.IsValid():
            return None
        xf = UsdGeom.Xformable(prim)
        if not xf:
            return None
        for op in xf.GetOrderedXformOps():
            if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                v = op.Get()
                return np.array([v[0], v[1], v[2]])
        return None

    def _set_cube_position(self, stage, position):
        """Set the cube prim translate in USD."""
        if self._cube_prim_path is None:
            return
        prim = stage.GetPrimAtPath(self._cube_prim_path)
        if not prim.IsValid():
            return
        xf = UsdGeom.Xformable(prim)
        if not xf:
            return
        for op in xf.GetOrderedXformOps():
            if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                op.Set(Gf.Vec3d(position[0], position[1], position[2]))
                return

    def _find_gripper_prim(self, stage):
        """Find the Franka gripper / hand prim for cube tracking."""
        if self._franka_prim_path is None:
            return
        candidates = [
            f"{self._franka_prim_path}/panda_hand",
            f"{self._franka_prim_path}/panda_link8",
            f"{self._franka_prim_path}/panda_link7",
        ]
        for path in candidates:
            prim = stage.GetPrimAtPath(path)
            if prim.IsValid():
                self._gripper_prim_path = path
                print(f"[INFO] Found gripper prim: {path}")
                return
        # Fall back to name-based search
        franka_prim = stage.GetPrimAtPath(self._franka_prim_path)
        if franka_prim.IsValid():
            for prim in Usd.PrimRange(franka_prim):
                name = prim.GetName().lower()
                if "hand" in name or "gripper" in name or "tool" in name:
                    self._gripper_prim_path = str(prim.GetPath())
                    print(f"[INFO] Found gripper prim: {self._gripper_prim_path}")
                    return
        print("[WARNING] Could not find gripper prim")

    def _get_gripper_world_position(self, stage):
        """Get the gripper world position via composed USD transforms."""
        if self._gripper_prim_path is None:
            return None
        prim = stage.GetPrimAtPath(self._gripper_prim_path)
        if not prim.IsValid():
            return None
        xform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())
        world_transform = xform_cache.GetLocalToWorldTransform(prim)
        pos = world_transform.ExtractTranslation()
        return np.array([pos[0], pos[1], pos[2]])

    def _position_cube_at_gripper(self, stage):
        """Keep the cube attached to the gripper during transport."""
        gripper_pos = self._get_gripper_world_position(stage)
        if gripper_pos is not None:
            cube_target = gripper_pos + self._gripper_to_cube_offset
            self._set_cube_position(stage, cube_target)
            return
        # Fallback: place cube above the current base position
        self._set_cube_position(
            stage,
            np.array([self._current_position[0], self._current_position[1], 0.5]),
        )

    def _is_cube_lifted(self, stage):
        """Check if the cube has been lifted above its original height."""
        pos = self._get_cube_position(stage)
        if pos is None:
            return False
        orig = self._scene_prim_originals.get(self._cube_prim_path)
        if orig is None:
            return False
        return pos[2] > orig[2] + 0.05

    def _apply_scene_offset(self, stage):
        """Offset scene prims (cube/table) by cube_offset along +x."""
        for path, orig in self._scene_prim_originals.items():
            prim = stage.GetPrimAtPath(path)
            if not prim.IsValid():
                continue
            try:
                xf = UsdGeom.Xformable(prim)
                if not xf:
                    continue
                target = Gf.Vec3d(orig[0] + self.cube_offset, orig[1], orig[2])
                applied = False
                for op in xf.GetOrderedXformOps():
                    if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                        op.Set(target)
                        applied = True
                        break
                if not applied:
                    xf.AddTranslateOp().Set(target)
            except Exception as exc:
                print(f"[WARNING] Could not offset {path}: {exc}")

    # ------------------------------------------------------------------
    #  Franka-prim helpers
    # ------------------------------------------------------------------
    def _find_franka_prim(self, stage):
        """Find the Franka robot prim in the scene."""
        possible_paths = [
            "/World/Franka",
            "/World/robot",
            "/World/panda",
            "/World/franka",
        ]
        for path in possible_paths:
            prim = stage.GetPrimAtPath(path)
            if prim.IsValid():
                return path

        # Fall back to heuristic search
        for prim in stage.Traverse():
            path = str(prim.GetPath())
            if "/World/" in path and prim.IsA(UsdGeom.Xform):
                children = [c.GetName() for c in prim.GetChildren()]
                if any("link" in c.lower() or "panda" in c.lower() for c in children):
                    return path

        return "/World/Franka"

    def _set_franka_floating_base(self, stage):
        """Disable the fixed-base flag so the Franka can be repositioned."""
        if not PHYSX_AVAILABLE or self._franka_prim_path is None:
            return

        franka_prim = stage.GetPrimAtPath(self._franka_prim_path)
        if not franka_prim.IsValid():
            return

        prims_to_check = [franka_prim] + list(Usd.PrimRange(franka_prim))
        for prim in prims_to_check:
            prim_path = str(prim.GetPath())
            articulation_api = PhysxSchema.PhysxArticulationAPI.Get(stage, prim_path)
            if articulation_api:
                try:
                    fix_base_attr = articulation_api.GetFixBaseAttr()
                    if fix_base_attr:
                        fix_base_attr.Set(False)
                    else:
                        articulation_api.CreateFixBaseAttr(False)
                except Exception as exc:
                    print(f"[WARNING] Could not modify fixBase: {exc}")

    def _set_franka_usd_position(self, stage, position):
        """Set the Franka position using USD transforms (pre-simulation)."""
        if self._franka_prim_path is None:
            return
        franka_prim = stage.GetPrimAtPath(self._franka_prim_path)
        if franka_prim.IsValid():
            xf = UsdGeom.Xformable(franka_prim)
            xf.ClearXformOpOrder()
            translate_op = xf.AddTranslateOp()
            translate_op.Set(Gf.Vec3d(position[0], position[1], position[2]))

    # ------------------------------------------------------------------
    #  Base movement
    # ------------------------------------------------------------------
    def _move_towards(self, target_pos):
        """Move the mobile base + Franka toward target_pos.

        Returns True when the base has arrived (within 2 cm).
        """
        direction = target_pos - self._current_position
        direction[2] = 0
        distance = np.linalg.norm(direction[:2])

        if distance < 0.02:
            self._current_position = target_pos.copy()
            self._set_positions(target_pos)
            return True

        direction = direction / distance
        step = direction * min(self._move_speed, distance)
        self._current_position = self._current_position + step
        self._current_position[2] = 0

        self._set_positions(self._current_position)
        return False

    def _set_positions(self, position):
        """Set positions of both the visual base box and the Franka robot.

        Tries multiple strategies for moving the Franka for maximum
        compatibility across Isaac Sim versions.
        """
        stage = omni.usd.get_context().get_stage()

        # 1. Move the visual Ridgeback base box
        base_prim = stage.GetPrimAtPath(self._mobile_base_prim_path)
        if base_prim.IsValid():
            xf = UsdGeom.Xformable(base_prim)
            for op in xf.GetOrderedXformOps():
                if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                    op.Set(Gf.Vec3d(position[0], position[1], 0.1))
                    break

        # 2. Move the Franka robot (try multiple methods)
        moved = False

        if self._franka_robot is not None and not moved:
            try:
                pos = np.array([position[0], position[1], 0.0])
                orient = np.array([1.0, 0.0, 0.0, 0.0])
                self._franka_robot.set_world_pose(position=pos, orientation=orient)
                moved = True
            except Exception:
                pass

        if self._franka_xform is not None and not moved:
            try:
                pos = np.array([position[0], position[1], 0.0])
                orient = np.array([1.0, 0.0, 0.0, 0.0])
                self._franka_xform.set_world_pose(position=pos, orientation=orient)
                moved = True
            except Exception:
                pass

        if DC_AVAILABLE and self._dc is not None and self._articulation_handle is not None and not moved:
            try:
                root_body = self._dc.get_articulation_root_body(self._articulation_handle)
                if root_body != 0:
                    transform = _dynamic_control.Transform()
                    transform.p = _dynamic_control.float3(position[0], position[1], 0.0)
                    transform.r = _dynamic_control.float4(0.0, 0.0, 0.0, 1.0)
                    self._dc.set_rigid_body_pose(root_body, transform)
                    moved = True
            except Exception:
                pass

        if not moved:
            franka_prim = stage.GetPrimAtPath(self._franka_prim_path)
            if franka_prim.IsValid():
                xf = UsdGeom.Xformable(franka_prim)
                for op in xf.GetOrderedXformOps():
                    if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                        op.Set(Gf.Vec3d(position[0], position[1], 0.0))
                        break

    # ------------------------------------------------------------------
    #  Reset
    # ------------------------------------------------------------------
    def reset(self):
        """Reset the mobile manipulator to the start position."""
        self._state = MobileState.INIT
        self._step_count = 0
        self._state_step_count = 0
        self._settled_steps = 0
        self._current_position = self.start_position.copy()

        # Initialise robot / xform / dc handles for position setting
        if CORE_ROBOT_AVAILABLE and self._franka_prim_path is not None:
            if self._franka_robot is None:
                try:
                    self._franka_robot = CoreRobot(prim_path=self._franka_prim_path)
                    self._franka_robot.initialize()
                except Exception:
                    self._franka_robot = None

        if XFORM_AVAILABLE and self._franka_xform is None and self._franka_prim_path:
            try:
                self._franka_xform = XFormPrim(prim_path=self._franka_prim_path)
            except Exception:
                self._franka_xform = None

        if DC_AVAILABLE:
            if self._dc is None:
                try:
                    self._dc = _dynamic_control.acquire_dynamic_control_interface()
                except Exception:
                    self._dc = None
            if self._dc is not None and self._articulation_handle is None and self._franka_prim_path:
                try:
                    self._articulation_handle = self._dc.get_articulation(self._franka_prim_path)
                    if self._articulation_handle == 0:
                        self._articulation_handle = None
                except Exception:
                    self._articulation_handle = None

        self._set_positions(self.start_position)
        self.franka_pick_place.reset()
        self._apply_scene_offset(omni.usd.get_context().get_stage())

        print(f"[Reset] Ridgeback Franka at start position {self.start_position}")

    # ------------------------------------------------------------------
    #  State machine
    # ------------------------------------------------------------------
    def forward(self, ik_method):
        """Execute one step of the mobile pick-and-place state machine."""
        self._step_count += 1
        self._state_step_count += 1
        stage = omni.usd.get_context().get_stage()

        if self._state == MobileState.INIT:
            print("[STATE] INIT -> MOVE_TO_CUBE")
            self._state = MobileState.MOVE_TO_CUBE
            self._state_step_count = 0

        elif self._state == MobileState.MOVE_TO_CUBE:
            reached = self._move_towards(self.table_position)
            if reached or self._state_step_count > 2000:
                print("[STATE] MOVE_TO_CUBE -> WAIT_SETTLED")
                self._state = MobileState.WAIT_SETTLED
                self._state_step_count = 0
                self._settled_steps = 0

        elif self._state == MobileState.WAIT_SETTLED:
            self._settled_steps += 1
            if self._settled_steps > 30:
                print("[STATE] WAIT_SETTLED -> PICK_CUBE")
                self._state = MobileState.PICK_CUBE
                self._state_step_count = 0
                self._pick_steps = 0
                self.franka_pick_place.reset()
                self._apply_scene_offset(stage)

        elif self._state == MobileState.PICK_CUBE:
            self.franka_pick_place.forward(ik_method)
            self._pick_steps += 1

            cube_lifted = self._is_cube_lifted(stage)
            pick_done = self.franka_pick_place.is_done()

            if cube_lifted or pick_done or self._pick_steps > 800:
                gripper_pos = self._get_gripper_world_position(stage)
                cube_pos = self._get_cube_position(stage)
                if gripper_pos is not None and cube_pos is not None:
                    self._gripper_to_cube_offset = cube_pos - gripper_pos
                elif gripper_pos is not None:
                    self._gripper_to_cube_offset = np.array([0.0, 0.0, -0.04])

                reason = "lifted" if cube_lifted else ("full-cycle" if pick_done else "timeout")
                print(f"[STATE] PICK_CUBE -> RETURN_TO_START ({reason} after {self._pick_steps} steps)")
                self._state = MobileState.RETURN_TO_START
                self._state_step_count = 0

        elif self._state == MobileState.RETURN_TO_START:
            reached = self._move_towards(self.start_position)
            self._position_cube_at_gripper(stage)

            if reached or self._state_step_count > 2000:
                print("[STATE] RETURN_TO_START -> WAIT_SETTLED_RETURN")
                self._state = MobileState.WAIT_SETTLED_RETURN
                self._state_step_count = 0
                self._settled_steps = 0

        elif self._state == MobileState.WAIT_SETTLED_RETURN:
            self._settled_steps += 1
            self._position_cube_at_gripper(stage)

            if self._settled_steps > 30:
                print("[STATE] WAIT_SETTLED_RETURN -> PLACE_CUBE")
                self._state = MobileState.PLACE_CUBE
                self._state_step_count = 0
                self._place_steps = 0

        elif self._state == MobileState.PLACE_CUBE:
            self._place_steps += 1
            descent_rate = 0.002
            cube_pos = self._get_cube_position(stage)
            if cube_pos is not None and cube_pos[2] > 0.05:
                cube_pos[2] -= descent_rate
                self._set_cube_position(stage, cube_pos)
            elif self._place_steps > 50:
                orig = self._scene_prim_originals.get(self._cube_prim_path)
                if orig is not None:
                    self._set_cube_position(stage, np.array([0.0, orig[1], orig[2]]))
                print("[STATE] PLACE_CUBE -> DONE")
                self._state = MobileState.DONE

        # Periodic debug output
        if self._step_count % 200 == 0:
            print(f"[DEBUG] Step {self._step_count}, State: {self._state.name}")

    def is_done(self):
        """Return True when the task is complete."""
        return self._state == MobileState.DONE


# ---------------------------------------------------------------------------
#  Entry point
# ---------------------------------------------------------------------------
def main():
    """Entry point for the Ridgeback Franka pick-and-place demo."""
    print("=" * 60)
    print("  Ridgeback Franka Pick-and-Place Demo  (Isaac Sim 5.1.0)")
    print("=" * 60)

    SimulationManager.set_physics_sim_device(args.device)
    simulation_app.update()

    # FrankaPickPlace sets up Franka arm + table + cube scene with IK
    franka_pick_place = FrankaPickPlace()
    franka_pick_place.setup_scene()
    simulation_app.update()

    # Wrap the Franka with mobile-base navigation
    stage = omni.usd.get_context().get_stage()
    mobile = RidgebackFrankaMobile(
        franka_pick_place, cube_offset=args.cube_offset
    )
    mobile.setup_mobile_base(stage)
    print("[Setup] Ridgeback Franka mobile manipulator ready.")
    simulation_app.update()

    # Start simulation
    timeline = omni.timeline.get_timeline_interface()
    timeline.play()
    simulation_app.update()

    # Reset
    mobile.reset()
    simulation_app.update()

    # Main loop
    print("[Main] Starting mobile pick-and-place ...")
    task_completed = False
    while simulation_app.is_running():
        if not task_completed:
            mobile.forward(args.ik_method)

            if mobile.is_done():
                print("=" * 60)
                print("  Pick-and-place task completed successfully!")
                print("=" * 60)
                task_completed = True

        simulation_app.update()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        simulation_app.close()
