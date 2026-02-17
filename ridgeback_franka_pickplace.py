
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
import os
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
    "--start-distance",
    type=float,
    default=1.0,
    help="Distance in meters from table where Ridgeback starts",
)
parser.add_argument(
    "--assets-path",
    type=str,
    default=None,
    help="Local path to Isaac Sim assets root (e.g. /home/user/isaac_assets). If not set, uses default Nucleus/S3 server.",
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
    MOVE_TO_TABLE = 1
    WAIT_SETTLED = 2
    PICK_PLACE = 3
    DONE = 4


class RidgebackFrankaMobile:
    """Ridgeback Franka mobile manipulator with movement before pick and place.

    Wraps FrankaPickPlace with a visual Ridgeback mobile base. The base starts
    at a configurable distance from the table, drives forward, then the Franka
    arm performs pick-and-place.
    """

    def __init__(self, franka_pick_place: FrankaPickPlace, start_distance: float = 1.0):
        self.franka_pick_place = franka_pick_place
        self.start_distance = start_distance

        self._mobile_base_prim_path = "/World/RidgebackBase"
        self._state = MobileState.INIT
        self._step_count = 0
        self._state_step_count = 0
        self._settled_steps = 0

        # self.start_position = np.array([-start_distance, 0.0, 0.0])
        self.start_position = np.array([-start_distance, -2.0, 0.0])
        # self.table_position = np.array([0.0, 0.0, 0.0])
        self.table_position = np.array([0.0, -2.0, 0.0])
        self._current_position = self.start_position.copy()
        self._move_speed = 0.002
        self._max_move_speed = 0.002
        self._min_move_speed = 0.0003
        self._accel_distance = 0.3

        self._base_height = 0.2

        self._franka_prim_path = None
        self._dc = None
        self._articulation_handle = None
        self._franka_robot = None
        self._franka_xform = None

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
        translate_op.Set(Gf.Vec3d(self.start_position[0], self.start_position[1], self._base_height / 2.0))

        cube_prim.GetDisplayColorAttr().Set([Gf.Vec3f(0.3, 0.3, 0.3)])

        self._set_franka_usd_position(stage, self.start_position)

        print(f"[INFO] Created Ridgeback mobile base at start position {self.start_position}")

        return self._mobile_base_prim_path

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
            translate_op.Set(Gf.Vec3d(position[0], position[1], self._base_height))

    def _move_towards(self, target_pos):
        """Move both the mobile base and Franka robot towards target position with smooth easing."""
        direction = target_pos - self._current_position
        direction[2] = 0
        distance = np.linalg.norm(direction[:2])

        if distance < 0.005:
            self._current_position = target_pos.copy()
            self._set_positions(target_pos)
            return True

        total_distance = np.linalg.norm((target_pos - self.start_position)[:2])
        traveled = total_distance - distance

        ease_in = min(1.0, traveled / self._accel_distance) if self._accel_distance > 0 else 1.0
        ease_out = min(1.0, distance / self._accel_distance) if self._accel_distance > 0 else 1.0
        ease_factor = min(ease_in, ease_out)
        ease_factor = ease_factor * ease_factor * (3.0 - 2.0 * ease_factor)

        speed = self._min_move_speed + (self._max_move_speed - self._min_move_speed) * ease_factor

        direction = direction / distance
        step = direction * min(speed, distance)
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
                    op.Set(Gf.Vec3d(position[0], position[1], self._base_height / 2.0))
                    break

        moved = False

        if self._franka_robot is not None and not moved:
            try:
                pos = np.array([position[0], position[1], self._base_height])
                orient = np.array([1.0, 0.0, 0.0, 0.0])
                self._franka_robot.set_world_pose(position=pos, orientation=orient)
                moved = True
            except Exception:
                pass

        if self._franka_xform is not None and not moved:
            try:
                pos = np.array([position[0], position[1], self._base_height])
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
                    transform.p = _dynamic_control.float3(position[0], position[1], self._base_height)
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
                        op.Set(Gf.Vec3d(position[0], position[1], self._base_height))
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

        print(f"[INFO] Ridgeback Franka reset to start position {self.start_position}")

    def forward(self, ik_method: str):
        """Execute one step of the mobile manipulation."""
        self._step_count += 1
        self._state_step_count += 1

        if self._state == MobileState.INIT:
            print("[STATE] INIT -> MOVE_TO_TABLE")
            self._state = MobileState.MOVE_TO_TABLE
            self._state_step_count = 0

        elif self._state == MobileState.MOVE_TO_TABLE:
            reached = self._move_towards(self.table_position)

            if reached or self._state_step_count > 1000:
                print("[STATE] MOVE_TO_TABLE -> WAIT_SETTLED")
                self._state = MobileState.WAIT_SETTLED
                self._state_step_count = 0
                self._settled_steps = 0

        elif self._state == MobileState.WAIT_SETTLED:
            self._settled_steps += 1

            if self._settled_steps > 30:
                print("[STATE] WAIT_SETTLED -> PICK_PLACE")
                self._state = MobileState.PICK_PLACE
                self._state_step_count = 0
                self.franka_pick_place.reset()

        elif self._state == MobileState.PICK_PLACE:
            self.franka_pick_place.forward(ik_method)

            if self.franka_pick_place.is_done():
                print("[STATE] PICK_PLACE -> DONE")
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
    print(f"\nRidgeback starts {args.start_distance}m from the table,")
    print("drives to the table, then Franka performs pick-and-place.")
    print(f"IK method: {args.ik_method}")
    print("=" * 60)

    SimulationManager.set_physics_sim_device(args.device)
    simulation_app.update()

    if args.assets_path:
        import carb.settings
        settings = carb.settings.get_settings()
        settings.set("/persistent/isaac/asset_root/default", args.assets_path)
        settings.set("/persistent/isaac/asset_root/cloud", args.assets_path)
        settings.set("/persistent/isaac/asset_root/nvidia", args.assets_path)
        print(f"[INFO] Using local assets path: {args.assets_path}")
    else:
        try:
            test_path = get_assets_root_path()
            print(f"[INFO] Assets root path: {test_path}")
        except RuntimeError:
            import carb.settings
            settings = carb.settings.get_settings()
            isaac_sim_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            local_candidates = [
                os.path.join(isaac_sim_dir, "data", "Assets"),
                os.path.expanduser("~/isaac_assets"),
                "/home/ashish/isaac_assets",
            ]
            found_local = None
            for candidate in local_candidates:
                if os.path.isdir(candidate):
                    found_local = candidate
                    break
            if found_local:
                settings.set("/persistent/isaac/asset_root/default", found_local)
                settings.set("/persistent/isaac/asset_root/cloud", found_local)
                settings.set("/persistent/isaac/asset_root/nvidia", found_local)
                print(f"[INFO] Remote assets unavailable. Using local path: {found_local}")
            else:
                print("[ERROR] Cannot reach remote asset server and no local assets found.")
                print("  Please provide a local assets path with --assets-path /path/to/assets")
                print("  Or download assets: omni_asset_download --path ~/isaac_assets")
                simulation_app.close()
                return

    franka_pick_place = FrankaPickPlace()
    franka_pick_place.setup_scene()

    stage = omni.usd.get_context().get_stage()
    cube_prim_paths = ["/World/Cube", "/World/CuboidTarget", "/World/target", "/World/cube"]
    cube_y_offset = -2.0
    for cube_path in cube_prim_paths:
        cube_prim = stage.GetPrimAtPath(cube_path)
        if cube_prim.IsValid():
            xformable = UsdGeom.Xformable(cube_prim)
            existing_ops = xformable.GetOrderedXformOps()
            translate_op = None
            for op in existing_ops:
                if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                    translate_op = op
                    break
            if translate_op:
                current = translate_op.Get()
                translate_op.Set(Gf.Vec3d(current[0], cube_y_offset, current[2]))
            else:
                translate_op = xformable.AddTranslateOp()
                translate_op.Set(Gf.Vec3d(0.0, cube_y_offset, 0.0))
            print(f"[INFO] Repositioned cube at {cube_path} to Y={cube_y_offset}")
            break
    else:
        for prim in stage.Traverse():
            prim_path = str(prim.GetPath())
            if "/World/" in prim_path and ("cube" in prim_path.lower() or "target" in prim_path.lower() or "object" in prim_path.lower()):
                xformable = UsdGeom.Xformable(prim)
                existing_ops = xformable.GetOrderedXformOps()
                translate_op = None
                for op in existing_ops:
                    if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                        translate_op = op
                        break
                if translate_op:
                    current = translate_op.Get()
                    translate_op.Set(Gf.Vec3d(current[0], cube_y_offset, current[2]))
                else:
                    translate_op = xformable.AddTranslateOp()
                    translate_op.Set(Gf.Vec3d(0.0, cube_y_offset, 0.0))
                print(f"[INFO] Repositioned object at {prim_path} to Y={cube_y_offset}")
                break

    try:
        assets_root_path = get_assets_root_path()
    except RuntimeError:
        assets_root_path = args.assets_path
    if assets_root_path is None:
        carb.log_error("Could not find Isaac Sim assets folder")
        return

    warehouse_usd_path = assets_root_path + "/Isaac/Environments/Simple_Warehouse/warehouse_multiple_shelves.usd"
    print(f"[INFO] Loading warehouse environment: {warehouse_usd_path}")
    prim = define_prim("/World/Warehouse", "Xform")
    prim.GetReferences().AddReference(warehouse_usd_path)
    print("[INFO] Warehouse environment loaded successfully")

    simulation_app.update()

    ridgeback_franka = RidgebackFrankaMobile(franka_pick_place, start_distance=args.start_distance)
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
