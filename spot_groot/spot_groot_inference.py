from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})

import carb
import numpy as np
import os
import io
import argparse
import time
from pathlib import Path
import omni.appwindow

from isaacsim.core.api import World
from isaacsim.core.utils.prims import define_prim
from isaacsim.storage.native import get_assets_root_path

import zmq
import msgpack


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


FORWARD_SPEED = -0.5
QUERY_INTERVAL_SECONDS = 0.5
WARMUP_SECONDS = 3.0


class SpotGR00TRunner(object):
    def __init__(
        self,
        physics_dt,
        render_dt,
        task_description,
        server_host="localhost",
        server_port=5555,
        forward_speed=FORWARD_SPEED,
        query_interval=QUERY_INTERVAL_SECONDS,
        camera_prim=DEFAULT_CAMERA_PRIM,
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
            position=np.array([1, 0, 0.8]),
            orientation=np.array([0, 0, 0, 1]),
        )

        self._add_cube_obstacle()
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

        self._physics_step_count = 0
        self._query_count = 0
        self._object_detected = False
        self._last_query_time = 0.0
        self._query_interval = query_interval
        self._warmup_seconds = WARMUP_SECONDS
        self._start_time = 0.0
        self._camera_ready = False

        self.needs_reset = False
        self.first_step = True

    def _add_cube_obstacle(self):
        from pxr import UsdGeom, UsdPhysics, Gf
        stage = self._world.stage
        cube_path = "/World/ObstacleCube"
        cube_prim = stage.DefinePrim(cube_path, "Cube")
        UsdGeom.Xformable(cube_prim).AddTranslateOp().Set(Gf.Vec3d(4.0, 0.0, 0.15))
        UsdGeom.Xformable(cube_prim).AddScaleOp().Set(Gf.Vec3d(0.05, 0.05, 0.05))
        UsdGeom.Gprim(cube_prim).CreateDisplayColorAttr([(1.0, 0.2, 0.2)])
        UsdPhysics.CollisionAPI.Apply(cube_prim)
        UsdPhysics.RigidBodyAPI.Apply(cube_prim)
        mass_api = UsdPhysics.MassAPI.Apply(cube_prim)
        mass_api.CreateMassAttr(1.0)

    def _setup_camera(self):
        from omni.isaac.sensor import Camera
        print(f"[Camera] Setting up camera at: {self._camera_prim}")
        self._camera = Camera(
            prim_path=self._camera_prim,
            resolution=(CAMERA_WIDTH, CAMERA_HEIGHT),
            frequency=30,
        )

    def _get_joint_state(self):
        from omni.isaac.core.articulations import Articulation
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
            self._world.reset(True)
            self._policy.reset()
            self._object_detected = False
            self._physics_step_count = 0
            self._query_count = 0
            self._camera_ready = False
            self.needs_reset = False
            self.first_step = True
            print("[Spot] Episode reset. Spot will start moving forward again.")
            return

        self._physics_step_count += 1
        now = time.time()
        elapsed = now - self._start_time

        if elapsed < self._warmup_seconds:
            self._spot.forward(step_size, np.array([self._forward_speed, 0.0, 0.0]))
            if self._physics_step_count % 200 == 0:
                print(f"[Spot] Warming up... {elapsed:.1f}s / {self._warmup_seconds}s")
            return

        if (now - self._last_query_time) >= self._query_interval:
            self._last_query_time = now
            self._query_groot()

        if self._object_detected:
            self._spot.forward(step_size, np.zeros(3))
        else:
            self._spot.forward(step_size, np.array([self._forward_speed, 0.0, 0.0]))

    def run(self) -> None:
        print("")
        print("=" * 50)
        print("  Spot + GR00T N1 Visual Novelty Detection")
        print("=" * 50)
        print(f"  Task: {self._task_description}")
        print(f"  Forward speed: {self._forward_speed}")
        print(f"  Query interval: {self._query_interval}s")
        print(f"  Warmup period: {self._warmup_seconds}s")
        print("  Detection: GR00T backbone feature novelty")
        print("  Press SPACE to reset, ESC to quit.")
        print("=" * 50)
        print("")

        while simulation_app.is_running():
            self._world.step(render=True)
            if self._world.is_stopped():
                self.needs_reset = True
        return

    def _sub_keyboard_event(self, event, *args, **kwargs) -> bool:
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            if event.input.name == "SPACE":
                print("[Spot] Resetting episode...")
                self.needs_reset = True
            elif event.input.name == "ESCAPE":
                simulation_app.close()
        return True


def main():
    parser = argparse.ArgumentParser(description="Spot + GR00T N1 Object Detection in Isaac Sim")
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
    args = parser.parse_args()

    physics_dt = 1 / 200.0
    render_dt = 1 / 60.0

    runner = SpotGR00TRunner(
        physics_dt=physics_dt,
        render_dt=render_dt,
        task_description=args.task,
        server_host=args.server_host,
        server_port=args.server_port,
        forward_speed=args.forward_speed,
        query_interval=args.query_interval,
        camera_prim=args.camera_prim,
    )
    simulation_app.update()
    runner._world.reset()
    simulation_app.update()
    runner.setup()
    simulation_app.update()
    runner.run()
    simulation_app.close()


if __name__ == "__main__":
    main()
