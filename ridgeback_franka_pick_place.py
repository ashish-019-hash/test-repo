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
a RidgebackFrankaMobile class that adds mobile base navigation using the
actual Ridgeback Franka USD model from Isaac Sim assets.

Architecture:
  - FrankaPickPlace handles IK-based arm control, table, and cube scene setup
  - RidgebackFrankaMobile wraps it with the Ridgeback Franka USD model
  - The standalone Franka from FrankaPickPlace is hidden; the Ridgeback Franka
    USD model is displayed instead
  - Arm joint positions are synced from the IK-controlled Franka to the
    Ridgeback Franka each simulation step
  - Base is moved by calling set_world_pose() on the Ridgeback Franka prim

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
from isaacsim.core.api.robots import Robot
from isaacsim.core.simulation_manager import SimulationManager
from isaacsim.robot.manipulators.examples.franka import FrankaPickPlace
from isaacsim.storage.native import get_assets_root_path
import isaacsim.core.utils.stage as stage_utils
from pxr import UsdGeom, Gf, Usd, Sdf

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
    from omni.isaac.core.prims import XFormPrim
    XFORM_AVAILABLE = True
except ImportError:
    XFORM_AVAILABLE = False

# Ridgeback Franka USD asset path (from Isaac Sim asset library)
RIDGEBACK_FRANKA_USD_SUBPATH = "/Isaac/Robots/Clearpath/RidgebackFranka/ridgeback_franka.usd"

# Joint names shared between the standalone Franka and the Ridgeback Franka
FRANKA_ARM_JOINT_NAMES = [
    "panda_joint1",
    "panda_joint2",
    "panda_joint3",
    "panda_joint4",
    "panda_joint5",
    "panda_joint6",
    "panda_joint7",
]
FRANKA_GRIPPER_JOINT_NAMES = [
    "panda_finger_joint1",
    "panda_finger_joint2",
]
# Ridgeback-specific dummy joints for base movement
RIDGEBACK_BASE_JOINTS = [
    "dummy_base_prismatic_x_joint",
    "dummy_base_prismatic_y_joint",
    "dummy_base_revolute_z_joint",
]


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
    control, table, and cube) with mobile-base navigation using the actual
    Ridgeback Franka USD model from Isaac Sim assets.

    The standalone Franka from FrankaPickPlace is hidden. The Ridgeback Franka
    USD model is loaded as the visible robot. Each simulation step, arm joint
    positions are synced from the IK-controlled Franka to the Ridgeback Franka.
    """

    def __init__(self, franka_pick_place, cube_offset=1.5):
        self.franka_pick_place = franka_pick_place
        self.cube_offset = cube_offset

        self._ridgeback_prim_path = "/World/RidgebackFranka"
        self._state = MobileState.INIT
        self._step_count = 0
        self._state_step_count = 0
        self._settled_steps = 0

        # Positions in world coordinates
        self.start_position = np.array([0.0, 0.0, 0.0])
        self.table_position = np.array([cube_offset, 0.0, 0.0])
        self._current_position = self.start_position.copy()
        self._move_speed = 0.01  # metres per step

        # FrankaPickPlace's standalone Franka (hidden, used for IK)
        self._franka_prim_path = None

        # Ridgeback Franka robot handle
        self._ridgeback_robot = None
        self._ridgeback_arm_indices = []
        self._ridgeback_gripper_indices = []
        self._ridgeback_base_x_idx = -1
        self._ridgeback_base_y_idx = -1
        self._ridgeback_base_yaw_idx = -1

        # FrankaPickPlace Franka robot handle (for reading IK joint positions)
        self._ik_franka_robot = None
        self._ik_arm_indices = []
        self._ik_gripper_indices = []

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
        """Load the Ridgeback Franka USD and configure the scene."""
        # Find the standalone Franka that FrankaPickPlace created
        self._franka_prim_path = self._find_franka_prim(stage)
        if self._franka_prim_path is None:
            print("[ERROR] Could not find FrankaPickPlace's Franka in scene!")
            return None

        print(f"[INFO] Found FrankaPickPlace Franka at: {self._franka_prim_path}")

        # Load the Ridgeback Franka USD from Isaac Sim assets
        assets_root_path = get_assets_root_path()
        ridgeback_usd_path = assets_root_path + RIDGEBACK_FRANKA_USD_SUBPATH
        stage_utils.add_reference_to_stage(ridgeback_usd_path, self._ridgeback_prim_path)
        print(f"[INFO] Loaded Ridgeback Franka USD at: {self._ridgeback_prim_path}")

        # Position the Ridgeback Franka at the start position
        self._set_ridgeback_usd_position(stage, self.start_position)

        # Hide the standalone Franka (FrankaPickPlace's robot) so only
        # the Ridgeback Franka is visible
        self._hide_franka_prim(stage)

        # Discover and offset scene prims (cube / table)
        self._discover_scene_prims(stage)
        self._apply_scene_offset(stage)
        self._find_cube_prim(stage)
        self._find_gripper_prim(stage)

        print(f"[INFO] Ridgeback Franka at start={self.start_position}")
        print(f"[INFO] Table/cube offset to x={self.cube_offset}m")

        return self._ridgeback_prim_path

    def _hide_franka_prim(self, stage):
        """Make the standalone Franka from FrankaPickPlace invisible."""
        if self._franka_prim_path is None:
            return
        franka_prim = stage.GetPrimAtPath(self._franka_prim_path)
        if not franka_prim.IsValid():
            return
        try:
            imageable = UsdGeom.Imageable(franka_prim)
            if imageable:
                imageable.MakeInvisible()
                print(f"[INFO] Hidden FrankaPickPlace Franka at {self._franka_prim_path}")
        except Exception as exc:
            print(f"[WARNING] Could not hide Franka prim: {exc}")

    def _set_ridgeback_usd_position(self, stage, position):
        """Set the Ridgeback Franka position using USD transforms (pre-sim)."""
        prim = stage.GetPrimAtPath(self._ridgeback_prim_path)
        if not prim.IsValid():
            return
        xf = UsdGeom.Xformable(prim)
        xf.ClearXformOpOrder()
        translate_op = xf.AddTranslateOp()
        translate_op.Set(Gf.Vec3d(position[0], position[1], position[2]))

    # ------------------------------------------------------------------
    #  Joint sync: copy arm positions from IK Franka -> Ridgeback Franka
    # ------------------------------------------------------------------
    def _init_joint_handles(self):
        """Initialize Robot handles and discover joint indices."""
        # Ridgeback Franka robot handle
        if self._ridgeback_robot is None:
            try:
                self._ridgeback_robot = Robot(
                    prim_path=self._ridgeback_prim_path,
                    name="ridgeback_franka",
                )
                self._ridgeback_robot.initialize()
                print(f"[INFO] Ridgeback DOFs: {self._ridgeback_robot.dof_names}")
                print(f"[INFO] Ridgeback Num DOFs: {self._ridgeback_robot.num_dof}")
            except Exception as exc:
                print(f"[WARNING] Could not init Ridgeback Robot: {exc}")
                self._ridgeback_robot = None

        # Discover Ridgeback joint indices
        if self._ridgeback_robot is not None and not self._ridgeback_arm_indices:
            try:
                for name in FRANKA_ARM_JOINT_NAMES:
                    idx = self._ridgeback_robot.get_dof_index(name)
                    self._ridgeback_arm_indices.append(idx)
                for name in FRANKA_GRIPPER_JOINT_NAMES:
                    idx = self._ridgeback_robot.get_dof_index(name)
                    self._ridgeback_gripper_indices.append(idx)
                self._ridgeback_base_x_idx = self._ridgeback_robot.get_dof_index(
                    "dummy_base_prismatic_x_joint"
                )
                self._ridgeback_base_y_idx = self._ridgeback_robot.get_dof_index(
                    "dummy_base_prismatic_y_joint"
                )
                self._ridgeback_base_yaw_idx = self._ridgeback_robot.get_dof_index(
                    "dummy_base_revolute_z_joint"
                )
                print(f"[INFO] Ridgeback arm indices: {self._ridgeback_arm_indices}")
                print(f"[INFO] Ridgeback base X={self._ridgeback_base_x_idx}, "
                      f"Y={self._ridgeback_base_y_idx}, Yaw={self._ridgeback_base_yaw_idx}")
            except Exception as exc:
                print(f"[WARNING] Could not discover Ridgeback joints: {exc}")

        # FrankaPickPlace's standalone Franka robot handle (for reading IK results)
        if self._ik_franka_robot is None and self._franka_prim_path:
            try:
                self._ik_franka_robot = Robot(
                    prim_path=self._franka_prim_path,
                    name="ik_franka",
                )
                self._ik_franka_robot.initialize()
                print(f"[INFO] IK Franka DOFs: {self._ik_franka_robot.dof_names}")
            except Exception as exc:
                print(f"[WARNING] Could not init IK Franka Robot: {exc}")
                self._ik_franka_robot = None

        # Discover IK Franka joint indices
        if self._ik_franka_robot is not None and not self._ik_arm_indices:
            try:
                for name in FRANKA_ARM_JOINT_NAMES:
                    idx = self._ik_franka_robot.get_dof_index(name)
                    self._ik_arm_indices.append(idx)
                for name in FRANKA_GRIPPER_JOINT_NAMES:
                    idx = self._ik_franka_robot.get_dof_index(name)
                    self._ik_gripper_indices.append(idx)
                print(f"[INFO] IK Franka arm indices: {self._ik_arm_indices}")
            except Exception as exc:
                print(f"[WARNING] Could not discover IK Franka joints: {exc}")

    def _sync_arm_joints(self):
        """Copy arm + gripper joint positions from IK Franka to Ridgeback Franka."""
        if self._ik_franka_robot is None or self._ridgeback_robot is None:
            return
        if not self._ik_arm_indices or not self._ridgeback_arm_indices:
            return

        try:
            # Read current joint positions from the IK-controlled Franka
            ik_positions = self._ik_franka_robot.get_joint_positions()
            if ik_positions is None:
                return

            # Read current Ridgeback joint positions
            rb_positions = self._ridgeback_robot.get_joint_positions()
            if rb_positions is None:
                return

            # Copy arm joint values
            new_positions = rb_positions.copy()
            for ik_idx, rb_idx in zip(self._ik_arm_indices, self._ridgeback_arm_indices):
                new_positions[rb_idx] = ik_positions[ik_idx]

            # Copy gripper joint values
            for ik_idx, rb_idx in zip(self._ik_gripper_indices, self._ridgeback_gripper_indices):
                new_positions[rb_idx] = ik_positions[ik_idx]

            # Apply to Ridgeback Franka
            self._ridgeback_robot.set_joint_positions(new_positions)
        except Exception as exc:
            if self._step_count % 500 == 0:
                print(f"[WARNING] Joint sync failed: {exc}")

    # ------------------------------------------------------------------
    #  Scene-prim helpers (cube / table discovery and tracking)
    # ------------------------------------------------------------------
    def _discover_scene_prims(self, stage):
        """Find cube/table scene prims and cache their original positions."""
        franka_path = self._franka_prim_path or "/World/Franka"
        skip_prefixes = (franka_path, self._ridgeback_prim_path)
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
        """Find the Ridgeback Franka gripper / hand prim for cube tracking."""
        # Search under the Ridgeback Franka prim (not the hidden standalone Franka)
        search_root = self._ridgeback_prim_path
        candidates = [
            f"{search_root}/panda_hand",
            f"{search_root}/panda_link8",
            f"{search_root}/panda_link7",
        ]
        for path in candidates:
            prim = stage.GetPrimAtPath(path)
            if prim.IsValid():
                self._gripper_prim_path = path
                print(f"[INFO] Found gripper prim: {path}")
                return
        # Fall back to name-based search under Ridgeback
        root_prim = stage.GetPrimAtPath(search_root)
        if root_prim.IsValid():
            for prim in Usd.PrimRange(root_prim):
                name = prim.GetName().lower()
                if "hand" in name or "gripper" in name or "tool" in name:
                    self._gripper_prim_path = str(prim.GetPath())
                    print(f"[INFO] Found gripper prim: {self._gripper_prim_path}")
                    return
        # Also try under the IK Franka as fallback
        if self._franka_prim_path:
            fallback_candidates = [
                f"{self._franka_prim_path}/panda_hand",
                f"{self._franka_prim_path}/panda_link8",
                f"{self._franka_prim_path}/panda_link7",
            ]
            for path in fallback_candidates:
                prim = stage.GetPrimAtPath(path)
                if prim.IsValid():
                    self._gripper_prim_path = path
                    print(f"[INFO] Found gripper prim (IK Franka fallback): {path}")
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
        """Find the standalone Franka robot prim created by FrankaPickPlace."""
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
                if any("panda" in c.lower() for c in children):
                    # Make sure it's not our Ridgeback Franka
                    if path != self._ridgeback_prim_path:
                        return path

        return "/World/Franka"

    # ------------------------------------------------------------------
    #  Base movement
    # ------------------------------------------------------------------
    def _move_towards(self, target_pos):
        """Move the Ridgeback Franka base toward target_pos.

        Uses the dummy prismatic joints on the Ridgeback Franka for base
        movement.  Returns True when the base has arrived (within 2 cm).
        """
        direction = target_pos - self._current_position
        direction[2] = 0
        distance = np.linalg.norm(direction[:2])

        if distance < 0.02:
            self._current_position = target_pos.copy()
            self._set_base_position(target_pos)
            return True

        direction = direction / distance
        step = direction * min(self._move_speed, distance)
        self._current_position = self._current_position + step
        self._current_position[2] = 0

        self._set_base_position(self._current_position)
        return False

    def _set_base_position(self, position):
        """Move the Ridgeback Franka base to the given world position.

        Tries multiple strategies:
        1. Dummy prismatic joint commands on the Ridgeback Franka articulation
        2. set_world_pose() on the Ridgeback Robot handle
        3. Direct USD transform update as fallback
        """
        moved = False

        # Strategy 1: Use dummy prismatic joints (most natural for Ridgeback)
        if (self._ridgeback_robot is not None
                and self._ridgeback_base_x_idx >= 0
                and self._ridgeback_base_y_idx >= 0):
            try:
                joint_positions = self._ridgeback_robot.get_joint_positions()
                if joint_positions is not None:
                    new_positions = joint_positions.copy()
                    # The dummy prismatic joints move the base relative to spawn
                    # Since we spawned at start_position, the joint value IS the
                    # world position (start_position is [0,0,0])
                    new_positions[self._ridgeback_base_x_idx] = position[0]
                    new_positions[self._ridgeback_base_y_idx] = position[1]
                    self._ridgeback_robot.set_joint_positions(new_positions)
                    moved = True
            except Exception:
                pass

        # Strategy 2: set_world_pose on the Robot handle
        if not moved and self._ridgeback_robot is not None:
            try:
                pos = np.array([position[0], position[1], 0.0])
                orient = np.array([1.0, 0.0, 0.0, 0.0])
                self._ridgeback_robot.set_world_pose(position=pos, orientation=orient)
                moved = True
            except Exception:
                pass

        # Strategy 3: Direct USD transform update
        if not moved:
            stage = omni.usd.get_context().get_stage()
            prim = stage.GetPrimAtPath(self._ridgeback_prim_path)
            if prim.IsValid():
                xf = UsdGeom.Xformable(prim)
                for op in xf.GetOrderedXformOps():
                    if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                        op.Set(Gf.Vec3d(position[0], position[1], 0.0))
                        break

        # Also move the hidden IK Franka to the same position so that
        # FrankaPickPlace's IK computations are in the correct frame
        self._move_ik_franka(position)

    def _move_ik_franka(self, position):
        """Move the hidden IK Franka to match the Ridgeback's position."""
        if self._franka_prim_path is None:
            return
        stage = omni.usd.get_context().get_stage()

        if self._ik_franka_robot is not None:
            try:
                pos = np.array([position[0], position[1], 0.0])
                orient = np.array([1.0, 0.0, 0.0, 0.0])
                self._ik_franka_robot.set_world_pose(position=pos, orientation=orient)
                return
            except Exception:
                pass

        # Fallback: USD transform
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

        # Initialize joint handles
        self._init_joint_handles()

        # Reset Ridgeback base position
        self._set_base_position(self.start_position)

        # Reset FrankaPickPlace (arm IK)
        self.franka_pick_place.reset()
        self._apply_scene_offset(omni.usd.get_context().get_stage())

        # Sync arm joints after reset
        self._sync_arm_joints()

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
            # Sync arm joints during movement so Ridgeback arm stays in position
            self._sync_arm_joints()
            if reached or self._state_step_count > 2000:
                print("[STATE] MOVE_TO_CUBE -> WAIT_SETTLED")
                self._state = MobileState.WAIT_SETTLED
                self._state_step_count = 0
                self._settled_steps = 0

        elif self._state == MobileState.WAIT_SETTLED:
            self._settled_steps += 1
            self._sync_arm_joints()
            if self._settled_steps > 30:
                print("[STATE] WAIT_SETTLED -> PICK_CUBE")
                self._state = MobileState.PICK_CUBE
                self._state_step_count = 0
                self._pick_steps = 0
                self.franka_pick_place.reset()
                self._apply_scene_offset(stage)

        elif self._state == MobileState.PICK_CUBE:
            # FrankaPickPlace runs IK on the hidden Franka
            self.franka_pick_place.forward(ik_method)
            self._pick_steps += 1

            # Sync arm positions to make the Ridgeback Franka arm mirror the IK
            self._sync_arm_joints()

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
            self._sync_arm_joints()

            if reached or self._state_step_count > 2000:
                print("[STATE] RETURN_TO_START -> WAIT_SETTLED_RETURN")
                self._state = MobileState.WAIT_SETTLED_RETURN
                self._state_step_count = 0
                self._settled_steps = 0

        elif self._state == MobileState.WAIT_SETTLED_RETURN:
            self._settled_steps += 1
            self._position_cube_at_gripper(stage)
            self._sync_arm_joints()

            if self._settled_steps > 30:
                print("[STATE] WAIT_SETTLED_RETURN -> PLACE_CUBE")
                self._state = MobileState.PLACE_CUBE
                self._state_step_count = 0
                self._place_steps = 0

        elif self._state == MobileState.PLACE_CUBE:
            self._place_steps += 1
            self._sync_arm_joints()
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
            print(f"[DEBUG] Step {self._step_count}, State: {self._state.name}, "
                  f"Pos: ({self._current_position[0]:.2f}, {self._current_position[1]:.2f})")

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

    # Wrap the Franka with mobile-base navigation using actual Ridgeback USD
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
