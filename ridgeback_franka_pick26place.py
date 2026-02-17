
"""
Ridgeback + Franka Mobile Manipulator Pick-and-Place Demo (Extended)

This script combines:
1. The working FrankaPickPlace approach for reliable arm/gripper control
2. A visual mobile base (Ridgeback) that moves together with the Franka arm
3. The Simple Warehouse environment

Extended behavior:
- The Franka arm picks a cube from the table and places it (first cycle).
- The cube is then relocated to a configurable far position.
- The Ridgeback walks to the far position, the arm picks the cube (second cycle).
- The Ridgeback walks back to the original table position and the cube is
  placed at the original starting position.
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
    "--start-distance",
    type=float,
    default=1.0,
    help="Distance in meters from table where Ridgeback starts",
)
parser.add_argument(
    "--place-distance",
    type=float,
    default=2.0,
    help="Distance in meters from the table where the cube is placed further away",
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
    RELOCATE_CUBE_FAR = 4
    MOVE_TO_FAR = 5
    WAIT_SETTLED_FAR = 6
    PICK_PLACE_FAR = 7
    MOVE_TO_ORIGIN = 8
    WAIT_SETTLED_ORIGIN = 9
    DONE = 10


class RidgebackFrankaMobile:
    """Ridgeback Franka mobile manipulator with extended pick-and-place.

    Wraps FrankaPickPlace with a visual Ridgeback mobile base. The workflow is:
    1. Drive to the table, pick up the cube, and place it (first cycle).
    2. The cube is relocated to a configurable far position.
    3. Drive to the far position, pick up the cube (second cycle).
    4. Drive back to the original table position where the cube is placed.
    """

    def __init__(
        self,
        franka_pick_place: FrankaPickPlace,
        start_distance: float = 1.0,
        place_distance: float = 2.0,
    ):
        self.franka_pick_place = franka_pick_place
        self.start_distance = start_distance
        self.place_distance = place_distance

        self._mobile_base_prim_path = "/World/RidgebackBase"
        self._state = MobileState.INIT
        self._step_count = 0
        self._state_step_count = 0
        self._settled_steps = 0

        self.start_position = np.array([-start_distance, 0.0, 0.0])
        self.table_position = np.array([0.0, 0.0, 0.0])
        self.far_position = np.array([place_distance, 0.0, 0.0])
        self._current_position = self.start_position.copy()
        self._move_speed = 0.005

        self._franka_prim_path = None
        self._dc = None
        self._articulation_handle = None
        self._franka_robot = None
        self._franka_xform = None

        self._cube_prim_path = None
        self._original_cube_position = None

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

        self._cube_prim_path = self._find_cube_prim(stage)
        if self._cube_prim_path:
            self._original_cube_position = self._get_prim_world_position(
                stage, self._cube_prim_path
            )
            print(
                f"[INFO] Found scene cube at: {self._cube_prim_path}, "
                f"position: {self._original_cube_position}"
            )

        print(f"[INFO] Created Ridgeback mobile base at start position {self.start_position}")
        print(f"[INFO] Far placement position: {self.far_position}")

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

    def _find_cube_prim(self, stage):
        """Find the pick-and-place cube prim in the scene."""
        candidate_paths = [
            "/World/Cube",
            "/World/cube",
            "/World/target",
            "/World/Target",
            "/World/object",
            "/World/Object",
        ]

        for path in candidate_paths:
            prim = stage.GetPrimAtPath(path)
            if prim.IsValid():
                return path

        for prim in stage.Traverse():
            path = str(prim.GetPath())
            if self._mobile_base_prim_path in path:
                continue
            if self._franka_prim_path and self._franka_prim_path in path:
                continue
            name_lower = prim.GetName().lower()
            if name_lower in ("cube", "target", "object", "block"):
                return path
            if prim.IsA(UsdGeom.Cube) and "/World/" in path:
                if "ridgeback" not in path.lower() and "warehouse" not in path.lower():
                    return path

        return None

    def _get_prim_world_position(self, stage, prim_path):
        """Get the world position of a prim."""
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            return np.array([0.0, 0.0, 0.0])

        xform = UsdGeom.Xformable(prim)
        world_transform = xform.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        translation = world_transform.ExtractTranslation()
        return np.array([translation[0], translation[1], translation[2]])

    def _move_cube_to_position(self, position):
        """Relocate the scene cube to a new world position via USD."""
        if self._cube_prim_path is None:
            print("[WARNING] No cube prim found, cannot relocate")
            return

        stage = omni.usd.get_context().get_stage()
        prim = stage.GetPrimAtPath(self._cube_prim_path)
        if not prim.IsValid():
            print(f"[WARNING] Cube prim {self._cube_prim_path} is invalid")
            return

        xform = UsdGeom.Xformable(prim)
        xform.ClearXformOpOrder()
        translate_op = xform.AddTranslateOp()
        cube_height = position[2] if position[2] != 0 else 0.025
        translate_op.Set(
            Gf.Vec3d(float(position[0]), float(position[1]), float(cube_height))
        )

        if self._dc is not None:
            try:
                body_handle = self._dc.get_rigid_body(self._cube_prim_path)
                if body_handle != 0:
                    transform = _dynamic_control.Transform()
                    transform.p = _dynamic_control.float3(
                        float(position[0]), float(position[1]), float(cube_height)
                    )
                    transform.r = _dynamic_control.float4(0.0, 0.0, 0.0, 1.0)
                    self._dc.set_rigid_body_pose(body_handle, transform)
                    self._dc.set_rigid_body_linear_velocity(
                        body_handle, _dynamic_control.float3(0, 0, 0)
                    )
                    self._dc.set_rigid_body_angular_velocity(
                        body_handle, _dynamic_control.float3(0, 0, 0)
                    )
            except Exception as e:
                print(f"[WARNING] Could not set cube rigid body pose: {e}")

        print(f"[INFO] Moved cube to position ({position[0]:.2f}, {position[1]:.2f}, {cube_height:.3f})")

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

        if self._cube_prim_path is None:
            stage = omni.usd.get_context().get_stage()
            self._cube_prim_path = self._find_cube_prim(stage)
            if self._cube_prim_path:
                self._original_cube_position = self._get_prim_world_position(
                    stage, self._cube_prim_path
                )

        if self._original_cube_position is not None:
            self._move_cube_to_position(self._original_cube_position)

        self._set_positions(self.start_position)
        self.franka_pick_place.reset()

        print(f"[INFO] Ridgeback Franka reset to start position {self.start_position}")

    def forward(self, ik_method: str):
        """Execute one step of the extended mobile manipulation.

        State machine flow:
        INIT -> MOVE_TO_TABLE -> WAIT_SETTLED -> PICK_PLACE (first cycle at table)
             -> RELOCATE_CUBE_FAR -> MOVE_TO_FAR -> WAIT_SETTLED_FAR
             -> PICK_PLACE_FAR (second cycle at far position)
             -> MOVE_TO_ORIGIN -> WAIT_SETTLED_ORIGIN -> DONE
        """
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
                print("[STATE] PICK_PLACE -> RELOCATE_CUBE_FAR")
                self._state = MobileState.RELOCATE_CUBE_FAR
                self._state_step_count = 0

        elif self._state == MobileState.RELOCATE_CUBE_FAR:
            cube_height = 0.025
            if self._original_cube_position is not None:
                cube_height = self._original_cube_position[2]
            far_cube_pos = np.array([
                self.far_position[0],
                self.far_position[1],
                cube_height,
            ])
            self._move_cube_to_position(far_cube_pos)
            print(f"[STATE] RELOCATE_CUBE_FAR -> MOVE_TO_FAR (cube at {far_cube_pos})")
            self._state = MobileState.MOVE_TO_FAR
            self._state_step_count = 0

        elif self._state == MobileState.MOVE_TO_FAR:
            reached = self._move_towards(self.far_position)

            if reached or self._state_step_count > 2000:
                print("[STATE] MOVE_TO_FAR -> WAIT_SETTLED_FAR")
                self._state = MobileState.WAIT_SETTLED_FAR
                self._state_step_count = 0
                self._settled_steps = 0

        elif self._state == MobileState.WAIT_SETTLED_FAR:
            self._settled_steps += 1

            if self._settled_steps > 30:
                print("[STATE] WAIT_SETTLED_FAR -> PICK_PLACE_FAR")
                self._state = MobileState.PICK_PLACE_FAR
                self._state_step_count = 0
                self.franka_pick_place.reset()

        elif self._state == MobileState.PICK_PLACE_FAR:
            self.franka_pick_place.forward(ik_method)

            if self.franka_pick_place.is_done():
                if self._original_cube_position is not None:
                    self._move_cube_to_position(self._original_cube_position)
                print("[STATE] PICK_PLACE_FAR -> MOVE_TO_ORIGIN")
                self._state = MobileState.MOVE_TO_ORIGIN
                self._state_step_count = 0

        elif self._state == MobileState.MOVE_TO_ORIGIN:
            reached = self._move_towards(self.table_position)

            if reached or self._state_step_count > 2000:
                print("[STATE] MOVE_TO_ORIGIN -> WAIT_SETTLED_ORIGIN")
                self._state = MobileState.WAIT_SETTLED_ORIGIN
                self._state_step_count = 0
                self._settled_steps = 0

        elif self._state == MobileState.WAIT_SETTLED_ORIGIN:
            self._settled_steps += 1

            if self._settled_steps > 30:
                print("[STATE] WAIT_SETTLED_ORIGIN -> DONE")
                self._state = MobileState.DONE

        if self._step_count % 200 == 0:
            print(f"[DEBUG] Step {self._step_count}, State: {self._state.name}")

    def is_done(self):
        """Check if the entire task is complete."""
        return self._state == MobileState.DONE


def main():
    print("=" * 60)
    print("Ridgeback + Franka Mobile Manipulator Pick-and-Place Demo")
    print("(Extended: place further, walk, retrieve, return)")
    print("=" * 60)
    print(f"\nRidgeback starts {args.start_distance}m from the table,")
    print(f"Cube will be placed {args.place_distance}m from the table,")
    print("Robot walks to retrieve it, then returns to the original position.")
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

    ridgeback_franka = RidgebackFrankaMobile(
        franka_pick_place,
        start_distance=args.start_distance,
        place_distance=args.place_distance,
    )
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
                print("\nDone: cube retrieved from far position and placed at original location")
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
