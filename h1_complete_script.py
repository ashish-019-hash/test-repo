
from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})

import carb
import math
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
from isaacsim.robot.policy.examples.robots import H1FlatTerrainPolicy
from isaacsim.sensors.camera import Camera
from omni.isaac.dynamic_control import _dynamic_control
from pxr import UsdGeom, Gf, UsdPhysics, PhysxSchema, Usd

import zmq
import msgpack

import omni.kit.app
import omni.graph.core as og

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
    def __init__(self, host="localhost", port=5555, timeout_ms=10000):
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
DEFAULT_CAMERA_PRIM = "/World/EyeCamera"

FORWARD_SPEED = 1.0
QUERY_INTERVAL_SECONDS = 0.3
WARMUP_SECONDS = 3.0
ROBOT_STOP_DISTANCE = 1.5  # meters from object where H1 should stop
ROBOT_SETTLED_THRESHOLD = 0.005  # max position change (meters) per step to consider H1 "stopped"
ROBOT_SETTLED_FRAMES = 100  # number of consecutive frames H1 must be still before Franka triggers
FRANKA_DELAY_SECONDS = 80.0  # 2.5-minute delay before starting Franka pick-and-place after H1 settles

CAMERA_SMOOTHING = 0.05  # EMA alpha for stabilized camera (0.01=smooth, 0.2=responsive)
CAMERA_PITCH = -25.0  # degrees (negative = look down)
H1_EYE_LINK = "d435_rgb_module_link"  # H1's head camera link name
ROS2_CAMERA_TOPIC = "/h1/camera/image_raw"
ROS2_PUBLISH_RATE_HZ = 20  # publish camera images at ~20 Hz


# ---- Stabilized camera helper functions (from H1 robot script) ----

def _quat_to_yaw(q):
    """Extract yaw angle from quaternion (w, x, y, z)."""
    w, x, y, z = q[0], q[1], q[2], q[3]
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def _angle_lerp(a, b, t):
    """Shortest-path interpolation between two angles."""
    diff = (b - a + math.pi) % (2 * math.pi) - math.pi
    return a + t * diff


def _rotation_matrix_to_quat(R):
    """Convert a 3x3 rotation matrix to quaternion (w, x, y, z)."""
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0:
        s = 0.5 / math.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    return np.array([w, x, y, z])


def _look_at_quat(forward_dir, up_dir=np.array([0.0, 0.0, 1.0])):
    """
    Compute a quaternion (w, x, y, z) for a USD camera to look along forward_dir.
    USD cameras look down their local -Z axis, with +Y as local up.
    """
    fwd = forward_dir / np.linalg.norm(forward_dir)
    cam_z = -fwd
    cam_x = np.cross(up_dir, cam_z)
    if np.linalg.norm(cam_x) < 1e-6:
        cam_x = np.array([1.0, 0.0, 0.0])
    cam_x = cam_x / np.linalg.norm(cam_x)
    cam_y = np.cross(cam_z, cam_x)
    cam_y = cam_y / np.linalg.norm(cam_y)
    R = np.column_stack([cam_x, cam_y, cam_z])
    return _rotation_matrix_to_quat(R)


def _get_link_world_pose(stage, link_path):
    """
    Get the world-space position and orientation of a USD prim link.
    Returns (position_np, quat_np) where quat is (w, x, y, z).
    """
    prim = stage.GetPrimAtPath(link_path)
    if not prim.IsValid():
        return None, None
    xformable = UsdGeom.Xformable(prim)
    world_transform = xformable.ComputeLocalToWorldTransform(0)
    translation = world_transform.ExtractTranslation()
    rotation = world_transform.ExtractRotationQuat()
    pos = np.array([translation[0], translation[1], translation[2]])
    imag = rotation.GetImaginary()
    quat = np.array([rotation.GetReal(), imag[0], imag[1], imag[2]])
    return pos, quat


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
        skip_prefixes = (franka_path, self._mobile_base_prim_path, "/World/Warehouse", "/World/H1")
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
                # Reset FrankaPickPlace state machine to Phase 4 (move to target)
                # so it executes phases 4 (move), 5 (release), 6 (retract)
                self.franka_pick_place._event = 4
                self.franka_pick_place._step = 0

        elif self._state == MobileState.PLACE_CUBE:
            self._place_steps += 1
            # Delegate to FrankaPickPlace phases 4-6 (move to target, release, retract)
            # This reuses the exact same placement logic as pick_place.py
            self.franka_pick_place.forward(ik_method)

            # Keep cube kinematically attached to gripper while arm moves to target (phase 4)
            # Once gripper starts opening (phase 5+), stop tracking so cube stays in place
            if self.franka_pick_place._event < 5:
                self._position_cube_at_gripper(stage)

            if self.franka_pick_place.is_done():
                print("[STATE] PLACE_CUBE -> DONE")
                self._state = MobileState.DONE

        if self._step_count % 200 == 0:
            print(f"[DEBUG] Step {self._step_count}, State: {self._state.name}")

    def is_done(self):
        """Check if the entire task is complete."""
        return self._state == MobileState.DONE


class H1GR00TRunner(object):
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
        ros2_camera_topic=ROS2_CAMERA_TOPIC,
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

        # --- H1 Unitree humanoid robot ---
        self._h1 = H1FlatTerrainPolicy(
            prim_path="/World/H1",
            name="H1",
            usd_path=assets_root_path + "/Isaac/Robots/Unitree/H1/h1.usd",
            position=np.array([3.5, 5, 1.05]),
            orientation=np.array([0.707, 0, 0, -0.707]),
        )
        self._h1_prim_path = "/World/H1"
        self._h1_eye_link_path = f"{self._h1_prim_path}/{H1_EYE_LINK}"

        # --- Second H1 Unitree humanoid robot (opposite side) ---
        self._h1_2 = H1FlatTerrainPolicy(
            prim_path="/World/H1_2",
            name="H1_2",
            usd_path=assets_root_path + "/Isaac/Robots/Unitree/H1/h1.usd",
            position=np.array([3.5, -5, 1.05]),
            orientation=np.array([0.707, 0, 0, 0.707]),
        )
        self._h1_2_prim_path = "/World/H1_2"

        # Distance-based H1_2 movement state (mirrors H1 logic)
        self._h1_2_reached_object = False
        self._h1_2_stopping = False
        self._h1_2_last_position = None
        self._h1_2_settled_count = 0

        self._cube_offset = cube_offset
        self._ik_method = ik_method

        self._stage = omni.usd.get_context().get_stage()
        self._ridgeback_franka = RidgebackFrankaMobile(
            franka_pick_place, cube_offset=cube_offset
        )
        self._ridgeback_franka.setup_mobile_base(self._stage)
        print("[H1] Ridgeback Franka mobile manipulator added to scene.")

        self._camera_prim = camera_prim
        self._setup_camera()

        self._task_description = task_description
        self._forward_speed = forward_speed

        self._policy = SimplePolicyClient(host=server_host, port=server_port)
        print(f"[H1] Connecting to GR00T policy server at {server_host}:{server_port}...")
        if not self._policy.ping():
            raise RuntimeError(
                f"Cannot connect to GR00T policy server at {server_host}:{server_port}. "
                "Make sure the server is running."
            )
        print("[H1] Connected to GR00T policy server successfully.")

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

        # Distance-based H1 movement: walk toward object, stop when close
        # Object position = cube's original X (0.5) + cube_offset along X, Y=0
        self._object_position = np.array([cube_offset + 0.5, 0.0, 0.0])
        self._robot_reached_object = False  # True when H1 is within ROBOT_STOP_DISTANCE
        self._robot_stopping = False  # True when stop command sent, waiting for H1 to settle
        self._robot_last_position = None
        self._robot_settled_count = 0

        # Stabilized eye camera state (EMA smoothing)
        self._smooth_camera_pos = None
        self._smooth_camera_yaw = None

        self._waiting_for_franka_delay = False
        self._franka_delay_start_time = 0.0
        self.needs_reset = False
        self.first_step = True

        # --- ROS 2 camera publisher (Isaac Sim ROS2 bridge) ---
        self._ros2_camera_topic = ros2_camera_topic
        self._ros2_camera_graph_path = "/World/ROS2CameraGraph"
        self._ros2_camera_graph_built = False
        print(f"[ROS2] Will publish camera images to: {self._ros2_camera_topic}")

    def _setup_camera(self):
        """Set up stabilized eye-level camera that tracks H1's head link."""
        print(f"[Camera] Setting up stabilized eye camera at: {self._camera_prim}")
        # Create the Camera sensor for image capture
        self._camera = Camera(
            prim_path=self._camera_prim,
            position=np.array([3.5, 8.0, 1.7]),  # initial pos near H1 head height
            orientation=np.array([0.5, -0.5, 0.5, -0.5]),
            resolution=(CAMERA_WIDTH, CAMERA_HEIGHT),
            frequency=20,
        )
        # Get the USD prim for direct transform updates each frame
        self._eye_camera_prim = self._stage.GetPrimAtPath(self._camera_prim)

        # Add a small visual marker on the head camera link
        try:
            head_cam_mesh = UsdGeom.Cube.Define(self._stage, self._h1_eye_link_path + "/CameraVis")
            head_cam_mesh.GetSizeAttr().Set(0.04)
            head_cam_mesh.GetDisplayColorAttr().Set([Gf.Vec3f(0.1, 0.1, 0.1)])
            xf = UsdGeom.Xformable(head_cam_mesh)
            xf.AddTranslateOp().Set(Gf.Vec3d(0.02, 0.0, 0.0))
        except Exception as e:
            print(f"[Camera] Could not add head camera visualization: {e}")

    def _setup_ros2_camera_graph(self):
        """Create an OmniGraph that publishes the camera via ROS 2 bridge."""
        if self._ros2_camera_graph_built:
            return

        # Enable ROS2 extensions (names vary across Isaac Sim versions)
        try:
            ext_mgr = omni.kit.app.get_app().get_extension_manager()
            for ext_name in [
                "isaacsim.ros2.nodes",
                "isaacsim.ros2.bridge",
                "omni.isaac.ros2_bridge",
            ]:
                try:
                    ext_mgr.set_extension_enabled_immediate(ext_name, True)
                except Exception:
                    pass
        except Exception as e:
            print(f"[ROS2] Could not enable ROS2 extensions: {e}")

        # Get render product path from Camera
        render_product_path = None
        try:
            if hasattr(self._camera, "get_render_product_path"):
                render_product_path = self._camera.get_render_product_path()
            elif hasattr(self._camera, "render_product_path"):
                render_product_path = self._camera.render_product_path
            elif hasattr(self._camera, "_render_product_path"):
                render_product_path = self._camera._render_product_path
        except Exception:
            render_product_path = None

        if not render_product_path:
            print("[ROS2] Could not find camera render product path; skipping ROS2 camera publishing.")
            return

        # Parse full topic into nodeNamespace + topicName
        topic = (self._ros2_camera_topic or "").lstrip("/")
        parts = [p for p in topic.split("/") if p]
        if not parts:
            node_namespace = ""
            topic_name = "rgb"
        else:
            node_namespace = "/".join(parts[:-1])
            topic_name = parts[-1]

        # Approximate publish rate using render ticks (~60Hz)
        frame_skip = max(int(round(60.0 / float(ROS2_PUBLISH_RATE_HZ))) - 1, 0)

        try:
            og.Controller.edit(
                {"graph_path": self._ros2_camera_graph_path, "evaluator_name": "execution"},
                {
                    og.Controller.Keys.CREATE_NODES: [
                        ("OnPlaybackTick", "omni.graph.action.OnPlaybackTick"),
                        ("ROS2Camera", "isaacsim.ros2.bridge.ROS2CameraHelper"),
                    ],
                    og.Controller.Keys.CONNECT: [
                        ("OnPlaybackTick.outputs:tick", "ROS2Camera.inputs:execIn"),
                    ],
                    og.Controller.Keys.SET_VALUES: [
                        ("ROS2Camera.inputs:enabled", True),
                        ("ROS2Camera.inputs:renderProductPath", render_product_path),
                        ("ROS2Camera.inputs:type", "rgb"),
                        ("ROS2Camera.inputs:nodeNamespace", node_namespace),
                        ("ROS2Camera.inputs:topicName", topic_name),
                        ("ROS2Camera.inputs:frameId", "h1_eye_camera"),
                        ("ROS2Camera.inputs:frameSkipCount", frame_skip),
                    ],
                },
            )
            self._ros2_camera_graph_built = True
            full_topic = "/" + "/".join([p for p in [node_namespace, topic_name] if p])
            print(f"[ROS2] ROS2CameraHelper graph created. Publishing on: {full_topic}")
        except Exception as e:
            print(f"[ROS2] Failed to create ROS2 camera graph: {e}")

    def _update_stabilized_camera(self):
        """
        Update the stabilized eye camera to track H1's head position and base yaw.
        Uses EMA smoothing for stable first-person images without walking shake.
        """
        # Get the eye link position (head height)
        eye_pos, _ = _get_link_world_pose(self._stage, self._h1_eye_link_path)
        if eye_pos is None:
            return

        # Get yaw from the robot base (forward direction)
        try:
            base_pos, base_quat = self._h1.robot.get_world_pose()
        except Exception:
            return

        target_pos = eye_pos.copy()
        target_yaw = _quat_to_yaw(base_quat)

        # Initialize on first call
        if self._smooth_camera_pos is None:
            self._smooth_camera_pos = target_pos.copy()
            self._smooth_camera_yaw = target_yaw

        alpha = CAMERA_SMOOTHING

        # Exponential smoothing on position (XY smoothed, Z smoothed separately)
        self._smooth_camera_pos[0] += alpha * (target_pos[0] - self._smooth_camera_pos[0])
        self._smooth_camera_pos[1] += alpha * (target_pos[1] - self._smooth_camera_pos[1])
        # Z: heavier smoothing to reduce vertical bobbing
        z_alpha = alpha * 0.3
        self._smooth_camera_pos[2] += z_alpha * (target_pos[2] - self._smooth_camera_pos[2])

        # Smooth yaw
        self._smooth_camera_yaw = _angle_lerp(self._smooth_camera_yaw, target_yaw, alpha)

        # Compute forward direction from smoothed yaw with pitch (tilt down)
        pitch_rad = math.radians(CAMERA_PITCH)
        cos_pitch = math.cos(pitch_rad)
        forward_dir = np.array([
            math.cos(self._smooth_camera_yaw) * cos_pitch,
            math.sin(self._smooth_camera_yaw) * cos_pitch,
            math.sin(pitch_rad),
        ])

        # Compute camera quaternion
        cam_quat = _look_at_quat(forward_dir, up_dir=np.array([0.0, 0.0, 1.0]))

        # Apply to camera prim
        try:
            self._eye_camera_prim.GetAttribute("xformOp:translate").Set(
                Gf.Vec3d(float(self._smooth_camera_pos[0]),
                          float(self._smooth_camera_pos[1]),
                          float(self._smooth_camera_pos[2]))
            )
            self._eye_camera_prim.GetAttribute("xformOp:orient").Set(
                Gf.Quatd(float(cam_quat[0]), float(cam_quat[1]),
                          float(cam_quat[2]), float(cam_quat[3]))
            )
        except Exception as e:
            if self._physics_step_count % 1000 == 0:
                print(f"[Camera] Error updating stabilized camera: {e}")

    def _get_joint_state(self):
        try:
            joint_positions = self._h1.robot.get_joint_positions()
            joint_velocities = self._h1.robot.get_joint_velocities()
            if joint_positions is None:
                joint_positions = np.zeros(19)
            if joint_velocities is None:
                joint_velocities = np.zeros(19)
            return joint_positions, joint_velocities
        except Exception:
            return np.zeros(19), np.zeros(19)

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
        self._world.add_physics_callback("h1_groot_forward", callback_fn=self.on_physics_step)

    def on_physics_step(self, step_size) -> None:
        if self.first_step:
            self._h1.initialize()
            self._h1_2.initialize()
            self._camera.initialize()
            self._setup_ros2_camera_graph()
            self._start_time = time.time()
            self._last_query_time = self._start_time
            self.first_step = False
            print("[H1] Initialized. Warming up camera...")
            print("[H1_2] Initialized (opposite side).")
            return

        if self.needs_reset:
            return

        self._physics_step_count += 1
        now = time.time()
        elapsed = now - self._start_time

        # Update stabilized eye camera each frame
        self._update_stabilized_camera()


        if elapsed < self._warmup_seconds:
            self._h1.forward(step_size, np.array([self._forward_speed, 0.0, 0.0]))
            self._h1_2.forward(step_size, np.array([self._forward_speed, 0.0, 0.0]))
            if self._physics_step_count % 200 == 0:
                print(f"[H1] Warming up... {elapsed:.1f}s / {self._warmup_seconds}s")
            return

        # --- Distance-based H1 movement with steering ---
        # Get H1's current world position and orientation
        h1_pos = None
        h1_orient = None
        try:
            h1_pos, h1_orient = self._h1.robot.get_world_pose()
        except Exception:
            pass

        if h1_pos is not None and not self._robot_reached_object and not self._pick_place_active and not self._pick_place_done:
            distance_to_object = np.linalg.norm(h1_pos[:2] - self._object_position[:2])
            if self._physics_step_count % 200 == 0:
                print(f"[H1] Walking toward object... distance={distance_to_object:.2f}m (stop at {ROBOT_STOP_DISTANCE}m)")
            if distance_to_object <= ROBOT_STOP_DISTANCE:
                self._robot_reached_object = True
                self._robot_stopping = True
                self._robot_settled_count = 0
                self._robot_last_position = None
                print(f"[H1] Reached object vicinity (distance={distance_to_object:.2f}m <= {ROBOT_STOP_DISTANCE}m). Stopping...")

        # GR00T queries continue running in parallel for detection logging
        if not self._pick_place_active and not self._robot_stopping:
            if (now - self._last_query_time) >= self._query_interval:
                self._last_query_time = now
                self._query_groot()

        # While H1 is stopping, monitor its position to detect when it has physically settled
        if self._robot_stopping and not self._pick_place_active:
            if h1_pos is not None and self._robot_last_position is not None:
                pos_delta = np.linalg.norm(h1_pos[:2] - self._robot_last_position[:2])
                if pos_delta < ROBOT_SETTLED_THRESHOLD:
                    self._robot_settled_count += 1
                else:
                    self._robot_settled_count = 0
                if self._robot_settled_count >= ROBOT_SETTLED_FRAMES:
                    # H1 has physically stopped — start delay before triggering Franka
                    self._robot_stopping = False
                    self._waiting_for_franka_delay = True
                    self._franka_delay_start_time = time.time()
                    print(f"[H1] H1 has stopped (settled for {ROBOT_SETTLED_FRAMES} frames). Waiting {FRANKA_DELAY_SECONDS}s (2.5 min) before starting Franka pick-and-place...")
                elif self._physics_step_count % 100 == 0:
                    print(f"[H1] Waiting for H1 to stop... settled_count={self._robot_settled_count}/{ROBOT_SETTLED_FRAMES}, delta={pos_delta:.4f}")
            if h1_pos is not None:
                self._robot_last_position = h1_pos.copy()

        # Wait for 2.5-minute delay before starting Franka pick-and-place
        if self._waiting_for_franka_delay and not self._pick_place_active:
            delay_elapsed = time.time() - self._franka_delay_start_time
            if delay_elapsed >= FRANKA_DELAY_SECONDS:
                self._waiting_for_franka_delay = False
                self._pick_place_active = True
                self._ridgeback_franka.reset()
                print(f"[H1] 2.5-minute delay complete. Triggering Ridgeback Franka pick-and-place...")
            elif self._physics_step_count % 200 == 0:
                remaining = FRANKA_DELAY_SECONDS - delay_elapsed
                print(f"[H1] Waiting for Franka delay... {delay_elapsed:.1f}s / {FRANKA_DELAY_SECONDS}s ({remaining:.1f}s remaining)")

        # Movement control: walk forward with yaw steering, or stop
        if self._pick_place_active or self._robot_stopping or self._robot_reached_object or self._waiting_for_franka_delay:
            self._h1.forward(step_size, np.zeros(3))
        elif self._pick_place_done:
            # Task done — walk straight forward without steering (no target to aim at)
            self._h1.forward(step_size, np.array([self._forward_speed, 0.0, 0.0]))
        else:
            # Compute yaw correction to steer H1 straight toward the object
            yaw_cmd = 0.0
            if h1_pos is not None and h1_orient is not None:
                # Desired heading from H1 to object
                dx = self._object_position[0] - h1_pos[0]
                dy = self._object_position[1] - h1_pos[1]
                desired_yaw = np.arctan2(dy, dx)

                # Current yaw from quaternion [w, x, y, z]
                current_yaw = _quat_to_yaw(h1_orient)

                # Yaw error (normalized to [-pi, pi])
                yaw_error = desired_yaw - current_yaw
                yaw_error = (yaw_error + np.pi) % (2 * np.pi) - np.pi

                # Proportional yaw correction
                yaw_cmd = 2.0 * yaw_error  # Kp = 2.0
                yaw_cmd = np.clip(yaw_cmd, -1.5, 1.5)

            self._h1.forward(step_size, np.array([self._forward_speed, 0.0, yaw_cmd]))

        # --- Distance-based H1_2 movement (mirrors H1 logic) ---
        h1_2_pos = None
        h1_2_orient = None
        try:
            h1_2_pos, h1_2_orient = self._h1_2.robot.get_world_pose()
        except Exception:
            pass

        # Check if H1_2 has reached the object
        if h1_2_pos is not None and not self._h1_2_reached_object and not self._pick_place_active and not self._pick_place_done:
            distance_to_object_2 = np.linalg.norm(h1_2_pos[:2] - self._object_position[:2])
            if self._physics_step_count % 200 == 0:
                print(f"[H1_2] Walking toward object... distance={distance_to_object_2:.2f}m (stop at {ROBOT_STOP_DISTANCE}m)")
            if distance_to_object_2 <= ROBOT_STOP_DISTANCE:
                self._h1_2_reached_object = True
                self._h1_2_stopping = True
                self._h1_2_settled_count = 0
                self._h1_2_last_position = None
                print(f"[H1_2] Reached object vicinity (distance={distance_to_object_2:.2f}m <= {ROBOT_STOP_DISTANCE}m). Stopping...")

        # While H1_2 is stopping, monitor its position to detect when it has physically settled
        if self._h1_2_stopping and not self._pick_place_active:
            if h1_2_pos is not None and self._h1_2_last_position is not None:
                pos_delta_2 = np.linalg.norm(h1_2_pos[:2] - self._h1_2_last_position[:2])
                if pos_delta_2 < ROBOT_SETTLED_THRESHOLD:
                    self._h1_2_settled_count += 1
                else:
                    self._h1_2_settled_count = 0
                if self._h1_2_settled_count >= ROBOT_SETTLED_FRAMES:
                    self._h1_2_stopping = False
                    print(f"[H1_2] H1_2 has stopped (settled for {ROBOT_SETTLED_FRAMES} frames). Waiting for pick-and-place to complete...")
                elif self._physics_step_count % 100 == 0:
                    print(f"[H1_2] Waiting for H1_2 to stop... settled_count={self._h1_2_settled_count}/{ROBOT_SETTLED_FRAMES}, delta={pos_delta_2:.4f}")
            if h1_2_pos is not None:
                self._h1_2_last_position = h1_2_pos.copy()

        # H1_2 movement control: walk forward with yaw steering, or stop
        if self._pick_place_active or self._h1_2_stopping or self._h1_2_reached_object or self._waiting_for_franka_delay:
            self._h1_2.forward(step_size, np.zeros(3))
        elif self._pick_place_done:
            # Task done — walk straight forward without steering
            self._h1_2.forward(step_size, np.array([self._forward_speed, 0.0, 0.0]))
        else:
            # Compute yaw correction to steer H1_2 toward the object
            yaw_cmd_2 = 0.0
            if h1_2_pos is not None and h1_2_orient is not None:
                dx_2 = self._object_position[0] - h1_2_pos[0]
                dy_2 = self._object_position[1] - h1_2_pos[1]
                desired_yaw_2 = np.arctan2(dy_2, dx_2)
                current_yaw_2 = _quat_to_yaw(h1_2_orient)
                yaw_error_2 = desired_yaw_2 - current_yaw_2
                yaw_error_2 = (yaw_error_2 + np.pi) % (2 * np.pi) - np.pi
                yaw_cmd_2 = 2.0 * yaw_error_2  # Kp = 2.0
                yaw_cmd_2 = np.clip(yaw_cmd_2, -1.5, 1.5)

            self._h1_2.forward(step_size, np.array([self._forward_speed, 0.0, yaw_cmd_2]))

    def run(self) -> None:
        print("")
        print("=" * 60)
        print("  H1 Unitree + GR00T N1 Visual Novelty Detection")
        print("  with Ridgeback Franka Pick-and-Place")
        print("=" * 60)
        print(f"  Task: {self._task_description}")
        print(f"  Forward speed: {self._forward_speed}")
        print(f"  Query interval: {self._query_interval}s")
        print(f"  Warmup period: {self._warmup_seconds}s")
        print(f"  IK method: {self._ik_method}")
        print("  Detection: GR00T backbone feature novelty")
        print("  Flow: H1 walks -> approaches object -> stops")
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
                self._robot_reached_object = False
                self._robot_stopping = False
                self._robot_last_position = None
                self._robot_settled_count = 0
                self._h1_2_reached_object = False
                self._h1_2_stopping = False
                self._h1_2_last_position = None
                self._h1_2_settled_count = 0
                self._smooth_camera_pos = None
                self._smooth_camera_yaw = None
                self._physics_step_count = 0
                self._query_count = 0
                self._camera_ready = False
                self._waiting_for_franka_delay = False
                self._franka_delay_start_time = 0.0
                self.needs_reset = False
                self.first_step = True
                print("[H1] Episode reset. H1 will start moving forward again.")
                continue

            if self._pick_place_active:
                self._ridgeback_franka.forward(self._ik_method)
                if self._ridgeback_franka.is_done():
                    self._pick_place_active = False
                    self._pick_place_done = True  # prevents distance check from re-triggering
                    self._object_detected = False
                    self._robot_reached_object = False  # allows H1 to walk forward
                    self._robot_stopping = False
                    self._robot_last_position = None
                    self._robot_settled_count = 0
                    self._h1_2_reached_object = False  # allows H1_2 to walk forward
                    self._h1_2_stopping = False
                    self._h1_2_last_position = None
                    self._h1_2_settled_count = 0
                    self._waiting_for_franka_delay = False
                    self._franka_delay_start_time = 0.0
                    self._policy.reset()
                    self._start_time = time.time()
                    self._query_count = 0
                    self._camera_ready = False
                    print("[H1] Pick-and-place complete! H1 and H1_2 resuming walk forward.")

        return

    def _sub_keyboard_event(self, event, *args, **kwargs) -> bool:
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            if event.input.name == "SPACE":
                print("[H1] Resetting episode...")
                self.needs_reset = True
            elif event.input.name == "ESCAPE":
                simulation_app.close()
        return True


def main():
    parser = argparse.ArgumentParser(
        description="H1 Unitree + GR00T N1 Object Detection with Ridgeback Franka Pick-and-Place"
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
                        help="Prim path of H1's stabilized eye camera")
    parser.add_argument("--cube-offset", type=float, default=3.0,
                        help="Position along +x axis where cube/table are placed for Ridgeback Franka")
    parser.add_argument("--ros2-camera-topic", type=str, default=ROS2_CAMERA_TOPIC,
                        help="ROS 2 topic name for publishing camera images")
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
    simulation_app.update()

    runner = H1GR00TRunner(
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
        ros2_camera_topic=args.ros2_camera_topic,
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
