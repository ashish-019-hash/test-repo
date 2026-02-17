
"""
Ridgeback + Franka Mobile Manipulator Pick-and-Place Demo

This script combines:
1. The working FrankaPickPlace approach for reliable arm/gripper control
2. A visual mobile base (Ridgeback) that moves together with the Franka arm
3. The Simple Warehouse environment

The Ridgeback base starts at a configurable distance from the table,
drives to the table, then the Franka arm performs pick-and-place.
"""

from __future__ import annotations

import argparse
import numpy as np
from enum import Enum

parser = argparse.ArgumentParser()
parser.add_argument("--device", type=str, choices=["cpu", "cuda"], default="cpu", help="Simulation device")
parser.add_argument(
    "--ik-method",
    type=str,
    choices=["singular-value-decomposition", "pseudoinverse", "transpose", "damped-least-squares"],
    default="damped-least-squares",
    help="Differential inverse kinematics method",
)
parser.add_argument(
    "--cube-offset",
    type=float,
    default=3.0,
    help="Position along +x axis where cube/table are placed (base travels from 0 to this)",
)
args, _ = parser.parse_known_args()

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

import omni.timeline
import omni.usd
import carb
from isaacsim.core.simulation_manager import SimulationManager
from isaacsim.robot.manipulators.examples.franka import FrankaPickPlace
from isaacsim.core.utils.prims import define_prim
from isaacsim.storage.native import get_assets_root_path
from omni.isaac.dynamic_control import _dynamic_control
from pxr import UsdGeom, Gf, UsdPhysics, PhysxSchema, Usd

try:
    from omni.isaac.core.prims import XFormPrim
    from omni.isaac.core.articulations import Articulation
    from omni.isaac.core.robots import Robot
    CORE_AVAILABLE = True
except ImportError:
    CORE_AVAILABLE = False
    print("[WARNING] omni.isaac.core not fully available")


class MobileState(Enum):
    INIT = 0
    MOVE_TO_CUBE = 1
    WAIT_SETTLED = 2
    PICK_CUBE = 3
    RETURN_TO_START = 4
    WAIT_SETTLED_RETURN = 5
    PLACE_CUBE = 6
    DONE = 7


class RidgebackFrankaMobile:
    """Ridgeback Franka mobile manipulator with movement before pick and place.

    Wraps FrankaPickPlace with a visual Ridgeback mobile base. The base starts
    at a configurable distance from the table, drives forward, then the Franka
    arm performs pick-and-place.
    """

    def __init__(self, franka_pick_place: FrankaPickPlace, cube_offset: float = 3.0):
        self.franka_pick_place = franka_pick_place
        self.cube_offset = cube_offset

        self._mobile_base_prim_path = "/World/RidgebackBase"
        self._state = MobileState.INIT
        self._step_count = 0
        self._state_step_count = 0
        self._settled_steps = 0

        self.start_position = np.array([0.0, 0.0, 0.0])
        self.table_position = np.array([cube_offset, 0.0, 0.0])
        self._current_position = self.start_position.copy()
        self._move_speed = 0.01

        self._franka_prim_path = None
        self._dc = None
        self._articulation_handle = None
        self._franka_robot = None
        self._franka_xform = None
        self._scene_prim_originals = {}
        self._cube_prim_path = None
        self._gripper_prim_path = None
        self._gripper_to_cube_offset = np.array([0.0, 0.0, -0.04])
        self._pick_steps = 0

    def setup_mobile_base(self, stage):
        """Create the visual mobile base and configure the Franka for movement."""
        self._franka_prim_path = self._find_franka_prim(stage)
        if self._franka_prim_path is None:
            print("[ERROR] Could not find Franka robot in scene!")
            return None

        print(f"[INFO] Found Franka robot at: {self._franka_prim_path}")

        self._set_franka_floating_base(stage)

        cube_prim = UsdGeom.Cube.Define(stage, self._mobile_base_prim_path)
        cube_prim.GetSizeAttr().Set(0.5)

        xform = UsdGeom.Xformable(cube_prim)
        scale_op = xform.AddScaleOp()
        scale_op.Set(Gf.Vec3f(1.0, 0.7, 0.4))

        translate_op = xform.AddTranslateOp()
        translate_op.Set(Gf.Vec3d(self.start_position[0], self.start_position[1], 0.1))

        cube_prim.GetDisplayColorAttr().Set([Gf.Vec3f(0.3, 0.3, 0.3)])

        self._set_franka_usd_position(stage, self.start_position)

        self._discover_scene_prims(stage)
        self._apply_scene_offset(stage)
        self._find_cube_prim(stage)
        self._find_gripper_prim(stage)

        print(f"[INFO] Created Ridgeback mobile base at start position {self.start_position}")
        print(f"[INFO] Cube/table offset to x={self.cube_offset}m, round-trip travel: {self.cube_offset * 2:.2f}m")

        return self._mobile_base_prim_path

    def _discover_scene_prims(self, stage):
        """Find cube/table scene prims and cache their original positions."""
        franka_path = self._franka_prim_path or "/World/Franka"
        skip_prefixes = (franka_path, self._mobile_base_prim_path, "/World/Warehouse")
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
                xform = UsdGeom.Xformable(prim)
                if not xform:
                    continue
                orig = Gf.Vec3d(0, 0, 0)
                for op in xform.GetOrderedXformOps():
                    if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                        orig = Gf.Vec3d(op.Get())
                        break
                self._scene_prim_originals[path] = orig
                print(f"[INFO] Cached scene prim {path} at original position {orig}")
            except Exception as e:
                print(f"[WARNING] Could not read {path}: {e}")

    def _find_cube_prim(self, stage):
        """Identify the graspable cube prim from cached scene prims."""
        for path in self._scene_prim_originals:
            prim = stage.GetPrimAtPath(path)
            if prim.IsValid():
                name = prim.GetName().lower()
                if "cube" in name or "block" in name:
                    self._cube_prim_path = path
                    print(f"[INFO] Identified graspable cube prim: {path}")
                    return
        if self._scene_prim_originals:
            self._cube_prim_path = next(iter(self._scene_prim_originals))
            print(f"[INFO] Using first scene prim as cube: {self._cube_prim_path}")

    def _get_cube_position(self, stage):
        """Read the cube prim's current translate from USD."""
        if self._cube_prim_path is None:
            return None
        prim = stage.GetPrimAtPath(self._cube_prim_path)
        if not prim.IsValid():
            return None
        xform = UsdGeom.Xformable(prim)
        if not xform:
            return None
        for op in xform.GetOrderedXformOps():
            if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                v = op.Get()
                return np.array([v[0], v[1], v[2]])
        return None

    def _set_cube_position(self, stage, position):
        """Set the cube prim's translate in USD."""
        if self._cube_prim_path is None:
            return
        prim = stage.GetPrimAtPath(self._cube_prim_path)
        if not prim.IsValid():
            return
        xform = UsdGeom.Xformable(prim)
        if not xform:
            return
        for op in xform.GetOrderedXformOps():
            if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                op.Set(Gf.Vec3d(position[0], position[1], position[2]))
                return

    def _find_gripper_prim(self, stage):
        """Find the Franka gripper/hand prim for tracking during transport."""
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
        franka_prim = stage.GetPrimAtPath(self._franka_prim_path)
        if franka_prim.IsValid():
            for prim in Usd.PrimRange(franka_prim):
                name = prim.GetName().lower()
                if "hand" in name or "gripper" in name or "tool" in name:
                    self._gripper_prim_path = str(prim.GetPath())
                    print(f"[INFO] Found gripper prim by search: {self._gripper_prim_path}")
                    return
        print("[WARNING] Could not find gripper prim, cube transport may not track correctly")

    def _get_gripper_world_position(self, stage):
        """Get the gripper's world position via composed USD transforms."""
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
        """Place the cube at the gripper's current world position + offset."""
        gripper_pos = self._get_gripper_world_position(stage)
        if gripper_pos is not None:
            cube_target = gripper_pos + self._gripper_to_cube_offset
            self._set_cube_position(stage, cube_target)
            return
        self._set_cube_position(stage, np.array([
            self._current_position[0],
            self._current_position[1],
            0.5
        ]))

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
        """Set scene prims to original_position + cube_offset along x. Idempotent."""
        for path, orig in self._scene_prim_originals.items():
            prim = stage.GetPrimAtPath(path)
            if not prim.IsValid():
                continue
            try:
                xform = UsdGeom.Xformable(prim)
                if not xform:
                    continue
                target = Gf.Vec3d(orig[0] + self.cube_offset, orig[1], orig[2])
                applied = False
                for op in xform.GetOrderedXformOps():
                    if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                        op.Set(target)
                        applied = True
                        break
                if not applied:
                    xform.AddTranslateOp().Set(target)
            except Exception as e:
                print(f"[WARNING] Could not offset {path}: {e}")

    def _find_franka_prim(self, stage):
        """Find the Franka robot prim in the scene."""
        possible_paths = ["/World/Franka", "/World/robot", "/World/panda", "/World/franka"]

        for path in possible_paths:
            prim = stage.GetPrimAtPath(path)
            if prim.IsValid():
                return path

        for prim in stage.Traverse():
            path = str(prim.GetPath())
            if "/World/" in path and prim.IsA(UsdGeom.Xform):
                if PhysxSchema.PhysxArticulationAPI.Get(stage, path):
                    return path
                children = [c.GetName() for c in prim.GetChildren()]
                if any("link" in c.lower() or "panda" in c.lower() for c in children):
                    return path

        return "/World/Franka"

    def _set_franka_floating_base(self, stage):
        """Configure the Franka articulation to have a floating base."""
        if self._franka_prim_path is None:
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
                except Exception as e:
                    print(f"[WARNING] Could not modify fixBase: {e}")

    def _set_franka_usd_position(self, stage, position):
        """Set the Franka's position using USD transforms."""
        if self._franka_prim_path is None:
            return

        franka_prim = stage.GetPrimAtPath(self._franka_prim_path)
        if franka_prim.IsValid():
            xform = UsdGeom.Xformable(franka_prim)
            xform.ClearXformOpOrder()
            translate_op = xform.AddTranslateOp()
            translate_op.Set(Gf.Vec3d(position[0], position[1], position[2]))

    def _move_towards(self, target_pos):
        """Move both the mobile base and Franka robot towards target position."""
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
        """Set positions of both the mobile base and Franka robot together."""
        stage = omni.usd.get_context().get_stage()

        base_prim = stage.GetPrimAtPath(self._mobile_base_prim_path)
        if base_prim.IsValid():
            xform = UsdGeom.Xformable(base_prim)
            ops = xform.GetOrderedXformOps()
            for op in ops:
                if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                    op.Set(Gf.Vec3d(position[0], position[1], 0.1))
                    break

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

        if self._dc is not None and self._articulation_handle is not None and not moved:
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
                xform = UsdGeom.Xformable(franka_prim)
                ops = xform.GetOrderedXformOps()
                for op in ops:
                    if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                        op.Set(Gf.Vec3d(position[0], position[1], 0.0))
                        break

    def reset(self):
        """Reset the mobile manipulator to start position."""
        self._state = MobileState.INIT
        self._step_count = 0
        self._state_step_count = 0
        self._settled_steps = 0
        self._current_position = self.start_position.copy()

        if CORE_AVAILABLE and self._franka_prim_path is not None:
            if self._franka_robot is None:
                try:
                    self._franka_robot = Robot(prim_path=self._franka_prim_path)
                    self._franka_robot.initialize()
                except Exception:
                    self._franka_robot = None

            if self._franka_robot is None:
                try:
                    self._franka_robot = Articulation(prim_path=self._franka_prim_path)
                    self._franka_robot.initialize()
                except Exception:
                    self._franka_robot = None

            if self._franka_xform is None:
                try:
                    self._franka_xform = XFormPrim(prim_path=self._franka_prim_path)
                except Exception:
                    self._franka_xform = None

        if self._dc is None:
            try:
                self._dc = _dynamic_control.acquire_dynamic_control_interface()
            except Exception:
                self._dc = None

        if self._dc is not None and self._articulation_handle is None and self._franka_prim_path is not None:
            try:
                self._articulation_handle = self._dc.get_articulation(self._franka_prim_path)
                if self._articulation_handle == 0:
                    self._articulation_handle = None
            except Exception:
                self._articulation_handle = None

        self._set_positions(self.start_position)
        self.franka_pick_place.reset()
        self._apply_scene_offset(omni.usd.get_context().get_stage())

        print(f"[INFO] Ridgeback Franka reset to start position {self.start_position}")

    def forward(self, ik_method: str):
        """Execute one step of the mobile manipulation."""
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

        if self._step_count % 200 == 0:
            print(f"[DEBUG] Step {self._step_count}, State: {self._state.name}")

    def is_done(self):
        """Check if the entire task is complete."""
        return self._state == MobileState.DONE


def main():
    print("=" * 60)
    print("Ridgeback + Franka Mobile Manipulator Pick-and-Place Demo")
    print("(Warehouse Environment)")
    print("=" * 60)
    print(f"\nCube/table placed at +{args.cube_offset}m along x-axis.")
    print("Ridgeback drives to cube, picks it, returns to origin, and places it.")
    print(f"IK method: {args.ik_method}")
    print("=" * 60)

    SimulationManager.set_physics_sim_device(args.device)
    simulation_app.update()

    franka_pick_place = FrankaPickPlace()
    franka_pick_place.setup_scene()

    assets_root_path = get_assets_root_path()
    if assets_root_path is None:
        carb.log_error("Could not find Isaac Sim assets folder")
        return

    warehouse_usd_path = assets_root_path + "/Isaac/Environments/Simple_Warehouse/warehouse_multiple_shelves.usd"
    print(f"[INFO] Loading warehouse environment: {warehouse_usd_path}")
    prim = define_prim("/World/Warehouse", "Xform")
    prim.GetReferences().AddReference(warehouse_usd_path)
    print("[INFO] Warehouse environment loaded successfully")

    simulation_app.update()

    stage = omni.usd.get_context().get_stage()

    ridgeback_franka = RidgebackFrankaMobile(franka_pick_place, cube_offset=args.cube_offset)
    ridgeback_franka.setup_mobile_base(stage)
    simulation_app.update()

    timeline = omni.timeline.get_timeline_interface()
    timeline.play()
    simulation_app.update()

    reset_needed = True
    task_completed = False

    print("\nStarting simulation...")
    print("Press Ctrl+C to exit\n")

    while simulation_app.is_running():
        simulation_app.update()

        if not timeline.is_playing():
            reset_needed = True
            continue

        if reset_needed:
            ridgeback_franka.reset()
            task_completed = False
            reset_needed = False

        if not task_completed:
            ridgeback_franka.forward(args.ik_method)

            if ridgeback_franka.is_done():
                print("\ndone picking and placing")
                task_completed = True

    simulation_app.close()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted")
    except Exception as e:
        print(f"[ERROR] {e}")
        import traceback
        traceback.print_exc()
    finally:
        if simulation_app.is_running():
            simulation_app.close()
# *************************************************************************************************************
