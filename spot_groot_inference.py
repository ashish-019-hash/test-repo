
from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})

import carb
import numpy as np
import os
import io
import argparse
import time
from pathlib import Path
from enum import Enum
import omni.appwindow
import omni.usd
import omni.timeline

from isaacsim.core.api import World
from isaacsim.core.utils.prims import define_prim
from isaacsim.storage.native import get_assets_root_path
from isaacsim.robot.manipulators.examples.franka import FrankaPickPlace
from omni.isaac.dynamic_control import _dynamic_control
from omni.isaac.sensor import Camera
from pxr import UsdGeom, Gf, UsdPhysics, PhysxSchema, Usd

import zmq
import msgpack

try:
    from omni.isaac.core.prims import XFormPrim
    from omni.isaac.core.articulations import Articulation
    from omni.isaac.core.robots import Robot
    CORE_AVAILABLE = True
except ImportError:
    CORE_AVAILABLE = False
    print("[WARNING] omni.isaac.core not fully available")


class MsgSerializer:
    @staticmethod
    def to_bytes(data):
        return msgpack.packb(data, default=MsgSerializer._encode)

    @staticmethod
    def from_bytes(data):
        return msgpack.unpackb(data, object_hook=MsgSerializer._decode)

    @staticmethod
    def _decode(obj):
        if not isinstance(obj, dict):
            return obj
        if "__ndarray_class__" in obj:
            return np.load(io.BytesIO(obj["as_npy"]), allow_pickle=False)
        return obj

    @staticmethod
    def _encode(obj):
        if isinstance(obj, np.ndarray):
            output = io.BytesIO()
            np.save(output, obj, allow_pickle=False)
            return {"__ndarray_class__": True, "as_npy": output.getvalue()}
        return obj


class SimplePolicyClient:
    def __init__(self, host="localhost", port=5555, timeout_ms=30000):
        self.context = zmq.Context()
        self.host = host
        self.port = port
        self.socket = self.context.socket(zmq.REQ)
        self.socket.setsockopt(zmq.RCVTIMEO, timeout_ms)
        self.socket.setsockopt(zmq.SNDTIMEO, timeout_ms)
        self.socket.connect(f"tcp://{host}:{port}")

    def ping(self):
        try:
            request = {"endpoint": "ping"}
            self.socket.send(MsgSerializer.to_bytes(request))
            message = self.socket.recv()
            response = MsgSerializer.from_bytes(message)
            return isinstance(response, dict) and response.get("status") == "ok"
        except zmq.error.ZMQError:
            return False

    def get_action(self, observation):
        request = {
            "endpoint": "get_action",
            "data": {"observation": observation},
        }
        self.socket.send(MsgSerializer.to_bytes(request))
        message = self.socket.recv()
        response = MsgSerializer.from_bytes(message)
        if isinstance(response, dict) and "error" in response:
            raise RuntimeError(f"Server error: {response['error']}")
        return tuple(response)

    def reset(self):
        request = {"endpoint": "reset", "data": {"options": None}}
        self.socket.send(MsgSerializer.to_bytes(request))
        self.socket.recv()

    def close(self):
        self.socket.close()
        self.context.term()


CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
DEFAULT_CAMERA_PRIM = "/World/Spot/body/frontleft_fisheye"

FORWARD_SPEED = 0.5
QUERY_INTERVAL_SECONDS = 0.3
WARMUP_SECONDS = 3.0


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

    Wraps FrankaPickPlace with a visual Ridgeback mobile base. The base drives
    to the cube, the Franka arm picks it, then the base returns to start.
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
        skip_prefixes = (franka_path, self._mobile_base_prim_path, "/World/Warehouse", "/World/Spot")
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


class SpotGR00TRunner(object):
    def __init__(
        self,
        franka_pick_place,
        physics_dt,
        render_dt,
        task_description,
        server_host="localhost",
        server_port=5555,
        forward_speed=FORWARD_SPEED,
        query_interval=QUERY_INTERVAL_SECONDS,
        camera_prim=DEFAULT_CAMERA_PRIM,
        cube_offset=3.0,
        ik_method="damped-least-squares",
    ):
        self._world = World(
            stage_units_in_meters=1.0,
            physics_dt=physics_dt,
            rendering_dt=render_dt,
        )

        assets_root_path = get_assets_root_path()
        if assets_root_path is None:
            carb.log_error("Could not find Isaac Sim assets folder")

        prim = define_prim("/World/Warehouse", "Xform")
        asset_path = assets_root_path + "/Isaac/Environments/Simple_Warehouse/warehouse_multiple_shelves.usd"
        prim.GetReferences().AddReference(asset_path)

        BASE_DIR = Path(__file__).resolve().parent.parent
        import sys
        sys.path.insert(0, os.path.join(str(BASE_DIR), "applications"))
        from spot_policy import SpotFlatTerrainPolicy
        policy_path = os.path.join(BASE_DIR, "policies/spot/models", "spot_policy.pt")
        policy_params_path = os.path.join(BASE_DIR, "policies/spot/params", "env.yaml")
        usd_path = os.path.join(BASE_DIR, "assets", "spot.usd")

        self._spot = SpotFlatTerrainPolicy(
            prim_path="/World/Spot",
            name="Spot",
            usd_path=usd_path,
            policy_path=policy_path,
            policy_params_path=policy_params_path,
            position=np.array([3.5, 3, 0.8]),
            orientation=np.array([0.707, 0, 0, -0.707]),
        )

        self._cube_offset = cube_offset
        self._ik_method = ik_method

        stage = omni.usd.get_context().get_stage()
        self._ridgeback_franka = RidgebackFrankaMobile(
            franka_pick_place, cube_offset=cube_offset
        )
        self._ridgeback_franka.setup_mobile_base(stage)
        print("[Spot] Ridgeback Franka mobile manipulator added to scene.")

        self._camera_prim = camera_prim
        self._setup_camera()

        self._task_description = task_description
        self._forward_speed = forward_speed

        self._policy = SimplePolicyClient(host=server_host, port=server_port)
        print(f"[Spot] Connecting to GR00T policy server at {server_host}:{server_port}...")
        if not self._policy.ping():
            raise RuntimeError(
                f"Cannot connect to GR00T policy server at {server_host}:{server_port}. "
                "Make sure the server is running."
            )
        print("[Spot] Connected to GR00T policy server successfully.")

        self._timeline = omni.timeline.get_timeline_interface()

        self._physics_step_count = 0
        self._query_count = 0
        self._object_detected = False
        self._last_query_time = 0.0
        self._query_interval = query_interval
        self._warmup_seconds = WARMUP_SECONDS
        self._start_time = 0.0
        self._camera_ready = False

        self._pick_place_active = False
        self._pick_place_done = False

        self.needs_reset = False
        self.first_step = True

    def _setup_camera(self):
        print(f"[Camera] Setting up camera at: {self._camera_prim}")
        self._camera = Camera(
            prim_path=self._camera_prim,
            resolution=(CAMERA_WIDTH, CAMERA_HEIGHT),
            frequency=30,
        )
        cam_prim = omni.usd.get_context().get_stage().GetPrimAtPath(self._camera_prim)
        xformable = UsdGeom.Xformable(cam_prim)
        xformable.AddRotateXYZOp(opSuffix="tilt").Set((-25.0, 0.0, 0.0))

    def _get_joint_state(self):
        try:
            spot_articulation = Articulation("/World/Spot")
            joint_positions = spot_articulation.get_joint_positions()
            joint_velocities = spot_articulation.get_joint_velocities()
            if joint_positions is None:
                joint_positions = np.zeros(12)
            if joint_velocities is None:
                joint_velocities = np.zeros(12)
            return joint_positions, joint_velocities
        except Exception:
            return np.zeros(12), np.zeros(12)

    def _get_camera_image(self):
        try:
            rgba = self._camera.get_rgba()
            if rgba is not None and rgba.shape[0] > 0:
                rgb = rgba[:, :, :3]
                if not self._camera_ready and np.any(rgb > 0):
                    self._camera_ready = True
                    print(f"[Camera] Camera is now producing valid frames (shape={rgb.shape}, max_val={rgb.max()})")
                return rgb
        except Exception as e:
            print(f"[Camera] Error getting image: {e}")
        return np.zeros((CAMERA_HEIGHT, CAMERA_WIDTH, 3), dtype=np.uint8)

    def _build_observation(self):
        rgb = self._get_camera_image()
        joint_positions, joint_velocities = self._get_joint_state()

        obs = {
            "video.front_camera": rgb.astype(np.uint8),
            "state.joint_positions": joint_positions.astype(np.float32),
            "state.joint_velocities": joint_velocities.astype(np.float32),
            "annotation.human.action.task_description": self._task_description,
        }
        return obs

    def _query_groot(self):
        obs = self._build_observation()
        rgb = obs["video.front_camera"]
        print(f"[GR00T] Sending query #{self._query_count + 1} | camera: shape={rgb.shape}, mean={rgb.mean():.1f}")
        try:
            result = self._policy.get_action(obs)
            self._query_count += 1

            if isinstance(result, (list, tuple)) and len(result) == 2:
                action, info = result
            else:
                print(f"[GR00T] Unexpected response type: {type(result)}")
                return

            detected = False
            if isinstance(info, dict):
                detected = info.get("object_detected", False)
                ref_nov = info.get("ref_novelty", 0.0)
                frame_nov = info.get("frame_novelty", 0.0)
                print(f"[GR00T Query #{self._query_count}] ref_novelty={ref_nov:.4f}, frame_novelty={frame_nov:.4f}")

            if detected:
                self._object_detected = True
                print(f">>> [GR00T Query #{self._query_count}] OBJECT DETECTED -> STOPPING <<<")
            else:
                self._object_detected = False
                print(f"[GR00T Query #{self._query_count}] No object -> MOVING")

        except Exception as e:
            print(f"[GR00T] Query FAILED: {e}")
            import traceback
            traceback.print_exc()

    def setup(self) -> None:
        self._appwindow = omni.appwindow.get_default_app_window()
        self._input = carb.input.acquire_input_interface()
        self._keyboard = self._appwindow.get_keyboard()
        self._sub_keyboard = self._input.subscribe_to_keyboard_events(
            self._keyboard, self._sub_keyboard_event
        )
        self._world.add_physics_callback("spot_groot_forward", callback_fn=self.on_physics_step)

    def on_physics_step(self, step_size) -> None:
        if self.first_step:
            self._spot.initialize()
            self._camera.initialize()
            self._start_time = time.time()
            self._last_query_time = self._start_time
            self.first_step = False
            print("[Spot] Initialized. Warming up camera...")
            return

        if self.needs_reset:
            return

        self._physics_step_count += 1
        now = time.time()
        elapsed = now - self._start_time

        if elapsed < self._warmup_seconds:
            self._spot.forward(step_size, np.array([self._forward_speed, 0.0, 0.0]))
            if self._physics_step_count % 200 == 0:
                print(f"[Spot] Warming up... {elapsed:.1f}s / {self._warmup_seconds}s")
            return

        if not self._pick_place_active:
            if (now - self._last_query_time) >= self._query_interval:
                self._last_query_time = now
                self._query_groot()

        if self._object_detected and not self._pick_place_active and not self._pick_place_done:
            self._pick_place_active = True
            self._ridgeback_franka.reset()
            print("[Spot] Object detected! Triggering Ridgeback Franka pick-and-place...")

        if self._pick_place_active or (self._object_detected and not self._pick_place_done):
            self._spot.forward(step_size, np.zeros(3))
        else:
            self._spot.forward(step_size, np.array([self._forward_speed, 0.0, 0.0]))

    def run(self) -> None:
        print("")
        print("=" * 60)
        print("  Spot + GR00T N1 Visual Novelty Detection")
        print("  with Ridgeback Franka Pick-and-Place")
        print("=" * 60)
        print(f"  Task: {self._task_description}")
        print(f"  Forward speed: {self._forward_speed}")
        print(f"  Query interval: {self._query_interval}s")
        print(f"  Warmup period: {self._warmup_seconds}s")
        print(f"  IK method: {self._ik_method}")
        print("  Detection: GR00T backbone feature novelty")
        print("  Flow: Spot walks -> detects object -> stops")
        print("        -> Ridgeback Franka picks and places cube")
        print("  Press SPACE to reset, ESC to quit.")
        print("=" * 60)
        print("")

        while simulation_app.is_running():
            simulation_app.update()

            if not self._timeline.is_playing():
                self.needs_reset = True
                continue

            if self.needs_reset:
                self._policy.reset()
                self._ridgeback_franka.reset()
                self._object_detected = False
                self._pick_place_active = False
                self._pick_place_done = False
                self._physics_step_count = 0
                self._query_count = 0
                self._camera_ready = False
                self.needs_reset = False
                self.first_step = True
                print("[Spot] Episode reset. Spot will start moving forward again.")
                continue

            if self._pick_place_active:
                self._ridgeback_franka.forward(self._ik_method)
                if self._ridgeback_franka.is_done():
                    self._pick_place_active = False
                    self._pick_place_done = True
                    self._object_detected = False
                    self._policy.reset()
                    self._start_time = time.time()
                    self._query_count = 0
                    self._camera_ready = False
                    print("[Spot] Pick-and-place complete! Spot resuming walk with GR00T queries. Press SPACE to reset for another cycle.")
        return

    def _sub_keyboard_event(self, event, *args, **kwargs) -> bool:
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            if event.input.name == "SPACE":
                print("[Spot] Resetting episode...")
                self.needs_reset = True
            elif event.input.name == "ESCAPE":
                simulation_app.close()
        return True


def _restyle_cube_as_pipe(stage):
    """Make cube transparent, add upright cylinder child as visual exhaust pipe."""
    from pxr import UsdShade, Sdf
    for prim in stage.Traverse():
        path = str(prim.GetPath())
        if any(skip in path for skip in ["/Franka", "/Warehouse", "/Spot", "/Ridgeback"]):
            continue
        if not prim.IsA(UsdGeom.Cube):
            continue
        name = prim.GetName().lower()
        if not any(kw in name for kw in ["cube", "block", "target", "object"]):
            continue

        hide_mat_path = f"{path}/HideMaterial"
        hide_material = UsdShade.Material.Define(stage, hide_mat_path)
        hide_shader = UsdShade.Shader.Define(stage, f"{hide_mat_path}/Shader")
        hide_shader.CreateIdAttr("OmniPBR")
        hide_shader.CreateInput("diffuse_color_constant", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.0, 0.0, 0.0))
        hide_shader.CreateInput("opacity_constant", Sdf.ValueTypeNames.Float).Set(0.0)
        hide_shader.CreateInput("enable_opacity", Sdf.ValueTypeNames.Bool).Set(True)
        hide_material.CreateSurfaceOutput().ConnectToSource(hide_shader.ConnectableAPI(), "surface")
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(hide_material)

        cyl_path = f"{path}/CylinderVisual"
        cyl = UsdGeom.Cylinder.Define(stage, cyl_path)
        cyl.GetRadiusAttr().Set(0.65)
        cyl.GetHeightAttr().Set(1.8)
        cyl.GetAxisAttr().Set("Z")
        cyl.GetDisplayColorAttr().Set([Gf.Vec3f(0.45, 0.25, 0.12)])

        pipe_mat_path = f"{cyl_path}/PipeMaterial"
        pipe_material = UsdShade.Material.Define(stage, pipe_mat_path)
        pipe_shader = UsdShade.Shader.Define(stage, f"{pipe_mat_path}/Shader")
        pipe_shader.CreateIdAttr("OmniPBR")
        pipe_shader.CreateInput("diffuse_color_constant", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.45, 0.25, 0.12))
        pipe_shader.CreateInput("reflection_roughness_constant", Sdf.ValueTypeNames.Float).Set(0.55)
        pipe_shader.CreateInput("metallic_constant", Sdf.ValueTypeNames.Float).Set(0.75)
        pipe_shader.CreateInput("specular_level", Sdf.ValueTypeNames.Float).Set(0.35)
        pipe_material.CreateSurfaceOutput().ConnectToSource(pipe_shader.ConnectableAPI(), "surface")
        UsdShade.MaterialBindingAPI.Apply(cyl.GetPrim()).Bind(pipe_material)

        print(f"[INFO] Hidden cube at {path}, added upright cylinder visual at {cyl_path}")
        return


def main():
    parser = argparse.ArgumentParser(
        description="Spot + GR00T N1 Object Detection with Ridgeback Franka Pick-and-Place"
    )
    parser.add_argument(
        "--task",
        type=str,
        default="move forward and stop when you see an object on the floor",
    )
    parser.add_argument("--server-host", type=str, default="localhost")
    parser.add_argument("--server-port", type=int, default=5555)
    parser.add_argument("--forward-speed", type=float, default=FORWARD_SPEED)
    parser.add_argument("--query-interval", type=float, default=QUERY_INTERVAL_SECONDS)
    parser.add_argument("--camera-prim", type=str, default=DEFAULT_CAMERA_PRIM,
                        help="Prim path of Spot's camera to use")
    parser.add_argument("--cube-offset", type=float, default=3.0,
                        help="Position along +x axis where cube/table are placed for Ridgeback Franka")
    parser.add_argument(
        "--ik-method",
        type=str,
        choices=["singular-value-decomposition", "pseudoinverse", "transpose", "damped-least-squares"],
        default="damped-least-squares",
        help="Differential inverse kinematics method for Franka arm",
    )
    args = parser.parse_args()

    physics_dt = 1 / 200.0
    render_dt = 1 / 60.0

    franka_pick_place = FrankaPickPlace()
    franka_pick_place.setup_scene()
    _restyle_cube_as_pipe(omni.usd.get_context().get_stage())
    simulation_app.update()

    runner = SpotGR00TRunner(
        franka_pick_place=franka_pick_place,
        physics_dt=physics_dt,
        render_dt=render_dt,
        task_description=args.task,
        server_host=args.server_host,
        server_port=args.server_port,
        forward_speed=args.forward_speed,
        query_interval=args.query_interval,
        camera_prim=args.camera_prim,
        cube_offset=args.cube_offset,
        ik_method=args.ik_method,
    )
    simulation_app.update()

    timeline = omni.timeline.get_timeline_interface()
    timeline.play()
    simulation_app.update()

    runner.setup()
    simulation_app.update()
    runner.run()
    simulation_app.close()


if __name__ == "__main__":
    main()
