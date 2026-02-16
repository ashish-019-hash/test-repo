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

from __future__ import annotations

import argparse

parser = argparse.ArgumentParser(description="Ridgeback Franka Pick-and-Place in Warehouse Environment")
parser.add_argument("--device", type=str, choices=["cpu", "cuda"], default="cpu", help="Simulation device")
parser.add_argument(
    "--ik-method",
    type=str,
    choices=["singular-value-decomposition", "pseudoinverse", "transpose", "damped-least-squares"],
    default="damped-least-squares",
    help="Differential inverse kinematics method",
)
parser.add_argument("--headless", action="store_true", default=False, help="Run in headless mode")
args, _ = parser.parse_known_args()

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": args.headless})

import numpy as np
import omni.timeline
import omni.usd
from isaacsim.core.api import World
from isaacsim.core.api.objects import DynamicCuboid, FixedCuboid
from isaacsim.core.simulation_manager import SimulationManager
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.api.robots import Robot
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.storage.native import get_assets_root_path
from pxr import UsdGeom


ROBOT_PRIM_PATH = "/World/RidgebackFranka"
WAREHOUSE_PRIM_PATH = "/World/Warehouse"
TARGET_OBJECT_PATH = "/World/TargetCube"
PLACE_TARGET_PATH = "/World/PlaceTarget"

ROBOT_INITIAL_POS = np.array([2.0, 2.0, 0.0])
OBJECT_POSITION = np.array([5.0, 2.0, 0.30])
PLACE_POSITION = np.array([5.0, 3.0, 0.30])

RIDGEBACK_WHEEL_DOF_NAMES = [
    "front_left_wheel",
    "front_right_wheel",
    "rear_left_wheel",
    "rear_right_wheel",
]

FRANKA_ARM_DOF_NAMES = [
    "panda_joint1", "panda_joint2", "panda_joint3", "panda_joint4",
    "panda_joint5", "panda_joint6", "panda_joint7",
]

FRANKA_GRIPPER_DOF_NAMES = ["panda_finger_joint1", "panda_finger_joint2"]

FRANKA_HOME_POSITIONS = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])
GRIPPER_OPEN = np.array([0.04, 0.04])
GRIPPER_CLOSE = np.array([0.0, 0.0])


class RobotState:
    NAVIGATE = "navigate"
    ALIGN = "align"
    PICK = "pick"
    LIFT = "lift"
    MOVE_TO_PLACE = "move_to_place"
    PLACE = "place"
    OPEN_GRIPPER = "open_gripper"
    RETREAT = "retreat"
    DONE = "done"


class MecanumController:
    def __init__(self, wheel_radius=0.0762, wheel_base=0.572, track_width=0.37476):
        self._r = wheel_radius
        self._lx = wheel_base / 2.0
        self._ly = track_width / 2.0

    def compute_wheel_velocities(self, vx, vy, omega):
        fl = (vx - vy - (self._lx + self._ly) * omega) / self._r
        fr = (vx + vy + (self._lx + self._ly) * omega) / self._r
        rl = (vx + vy - (self._lx + self._ly) * omega) / self._r
        rr = (vx - vy + (self._lx + self._ly) * omega) / self._r
        return np.array([fl, fr, rl, rr])


class NavigationController:
    def __init__(self, linear_speed=0.5, angular_speed=1.0, position_threshold=0.7, angle_threshold=0.15):
        self._linear_speed = linear_speed
        self._angular_speed = angular_speed
        self._position_threshold = position_threshold
        self._angle_threshold = angle_threshold

    def compute_command(self, current_pos, current_yaw, target_pos):
        dx = target_pos[0] - current_pos[0]
        dy = target_pos[1] - current_pos[1]
        distance = np.sqrt(dx * dx + dy * dy)

        if distance < self._position_threshold:
            return np.array([0.0, 0.0, 0.0]), True

        target_yaw = np.arctan2(dy, dx)
        yaw_error = target_yaw - current_yaw
        yaw_error = np.arctan2(np.sin(yaw_error), np.cos(yaw_error))

        if abs(yaw_error) > self._angle_threshold:
            omega = np.clip(self._angular_speed * yaw_error, -self._angular_speed, self._angular_speed)
            return np.array([0.05, 0.0, omega]), False

        vx = np.clip(self._linear_speed * min(distance, 1.0), 0.1, self._linear_speed)
        omega = np.clip(0.5 * yaw_error, -0.3, 0.3)
        return np.array([vx, 0.0, omega]), False


class PickPlaceStateMachine:
    def __init__(self, pick_pos, place_pos, lift_height=0.15):
        self._pick_pos = np.array(pick_pos)
        self._place_pos = np.array(place_pos)
        self._lift_height = lift_height
        self._phase = "approach"
        self._phase_counter = 0
        self._phases = [
            "approach", "descend", "close_gripper", "lift",
            "move_to_place", "lower", "open_gripper", "retreat", "done",
        ]

    @property
    def phase(self):
        return self._phase

    @property
    def is_done(self):
        return self._phase == "done"

    def get_target_position(self, ee_pos):
        if self._phase == "approach":
            t = self._pick_pos.copy()
            t[2] += self._lift_height
            return t
        elif self._phase == "descend":
            return self._pick_pos.copy()
        elif self._phase == "close_gripper":
            return self._pick_pos.copy()
        elif self._phase == "lift":
            t = self._pick_pos.copy()
            t[2] += self._lift_height
            return t
        elif self._phase == "move_to_place":
            t = self._place_pos.copy()
            t[2] += self._lift_height
            return t
        elif self._phase == "lower":
            return self._place_pos.copy()
        elif self._phase == "open_gripper":
            return self._place_pos.copy()
        elif self._phase == "retreat":
            t = self._place_pos.copy()
            t[2] += self._lift_height
            return t
        return ee_pos

    def should_close_gripper(self):
        return self._phase == "close_gripper"

    def should_open_gripper(self):
        return self._phase == "open_gripper"

    def is_gripper_phase(self):
        return self._phase in ("close_gripper", "open_gripper")

    def advance(self, ee_pos, threshold=0.03):
        target = self.get_target_position(ee_pos)
        dist = np.linalg.norm(ee_pos[:3] - target[:3])

        if self.is_gripper_phase():
            self._phase_counter += 1
            if self._phase_counter > 80:
                self._phase_counter = 0
                self._next_phase()
        elif dist < threshold:
            self._phase_counter += 1
            if self._phase_counter > 15:
                self._phase_counter = 0
                self._next_phase()
        else:
            self._phase_counter = 0

    def _next_phase(self):
        idx = self._phases.index(self._phase)
        if idx < len(self._phases) - 1:
            self._phase = self._phases[idx + 1]
            print(f"  Pick-place phase: {self._phase}")


class RidgebackFrankaPickPlace:
    def __init__(self):
        self._assets_root_path = get_assets_root_path()
        if self._assets_root_path is None:
            raise RuntimeError("Could not find Isaac Sim assets root path")

        self._world = World(stage_units_in_meters=1.0)
        self._stage = omni.usd.get_context().get_stage()

        self._state = RobotState.NAVIGATE
        self._nav_controller = NavigationController(
            linear_speed=0.5,
            angular_speed=1.2,
            position_threshold=0.7,
        )
        self._mecanum = MecanumController()
        self._pick_place_sm = None
        self._robot = None

        self._wheel_dof_indices = []
        self._arm_dof_indices = []
        self._gripper_dof_indices = []

    def setup_scene(self):
        print("Setting up warehouse environment...")
        warehouse_usd = self._assets_root_path + "/Isaac/Environments/Simple_Warehouse/warehouse_multiple_shelves.usd"
        add_reference_to_stage(usd_path=warehouse_usd, prim_path=WAREHOUSE_PRIM_PATH)
        print(f"  Loaded warehouse: {warehouse_usd}")

        print("Setting up Ridgeback Franka robot...")
        robot_usd = self._assets_root_path + "/Isaac/Robots/Clearpath/RidgebackFranka/ridgeback_franka.usd"
        add_reference_to_stage(usd_path=robot_usd, prim_path=ROBOT_PRIM_PATH)
        self._robot = self._world.scene.add(
            Robot(
                prim_path=ROBOT_PRIM_PATH,
                name="ridgeback_franka",
                position=ROBOT_INITIAL_POS,
            )
        )
        print(f"  Robot at: {ROBOT_INITIAL_POS}")

        print("Setting up target objects...")
        self._target_cube = self._world.scene.add(
            DynamicCuboid(
                prim_path=TARGET_OBJECT_PATH,
                name="target_cube",
                position=OBJECT_POSITION,
                size=0.05,
                color=np.array([1.0, 0.0, 0.0]),
                mass=0.02,
            )
        )
        self._place_marker = self._world.scene.add(
            FixedCuboid(
                prim_path=PLACE_TARGET_PATH,
                name="place_marker",
                position=np.array([PLACE_POSITION[0], PLACE_POSITION[1], 0.01]),
                size=0.06,
                color=np.array([0.0, 1.0, 0.0]),
            )
        )
        print("Scene setup complete.")

    def initialize(self):
        print("Initializing simulation and robot...")
        self._world.reset()

        for name in RIDGEBACK_WHEEL_DOF_NAMES:
            try:
                idx = self._robot.get_dof_index(name)
                self._wheel_dof_indices.append(idx)
            except Exception:
                print(f"  Warning: wheel joint '{name}' not found")

        for name in FRANKA_ARM_DOF_NAMES:
            try:
                idx = self._robot.get_dof_index(name)
                self._arm_dof_indices.append(idx)
            except Exception:
                print(f"  Warning: arm joint '{name}' not found")

        for name in FRANKA_GRIPPER_DOF_NAMES:
            try:
                idx = self._robot.get_dof_index(name)
                self._gripper_dof_indices.append(idx)
            except Exception:
                print(f"  Warning: gripper joint '{name}' not found")

        print(f"  Wheel DOF indices: {self._wheel_dof_indices}")
        print(f"  Arm DOF indices: {self._arm_dof_indices}")
        print(f"  Gripper DOF indices: {self._gripper_dof_indices}")

        self._pick_place_sm = PickPlaceStateMachine(
            pick_pos=OBJECT_POSITION,
            place_pos=PLACE_POSITION,
        )

        self._set_arm_positions(FRANKA_HOME_POSITIONS)
        self._set_gripper(GRIPPER_OPEN)
        print("Robot initialized.")

    def _get_robot_pose(self):
        pos, orient = self._robot.get_world_pose()
        w, x, y, z = orient[0], orient[1], orient[2], orient[3]
        yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
        return pos, yaw

    def _get_ee_position(self):
        for suffix in ["/panda_hand", "/panda_link8"]:
            prim = self._stage.GetPrimAtPath(ROBOT_PRIM_PATH + suffix)
            if prim.IsValid():
                xformable = UsdGeom.Xformable(prim)
                transform = xformable.ComputeLocalToWorldTransform(0)
                t = transform.ExtractTranslation()
                return np.array([t[0], t[1], t[2]])
        pos, _ = self._get_robot_pose()
        return pos + np.array([0.3, 0.0, 0.9])

    def _set_arm_positions(self, positions):
        if not self._arm_dof_indices:
            return
        current = self._robot.get_joint_positions()
        if current is None:
            return
        new_positions = current.copy()
        for i, idx in enumerate(self._arm_dof_indices):
            if i < len(positions):
                new_positions[idx] = positions[i]
        self._robot.set_joint_positions(new_positions)

    def _set_gripper(self, positions):
        if not self._gripper_dof_indices:
            return
        current = self._robot.get_joint_positions()
        if current is None:
            return
        new_positions = current.copy()
        for i, idx in enumerate(self._gripper_dof_indices):
            if i < len(positions):
                new_positions[idx] = positions[i]
        self._robot.set_joint_positions(new_positions)

    def _apply_arm_target(self, target_positions):
        if not self._arm_dof_indices:
            return
        current = self._robot.get_joint_positions()
        if current is None:
            return
        joint_positions = current.copy()
        for i, idx in enumerate(self._arm_dof_indices):
            if i < len(target_positions):
                joint_positions[idx] = target_positions[i]
        self._robot.apply_action(ArticulationAction(joint_positions=joint_positions))

    def _apply_gripper_target(self, target_positions):
        if not self._gripper_dof_indices:
            return
        current = self._robot.get_joint_positions()
        if current is None:
            return
        joint_positions = current.copy()
        for i, idx in enumerate(self._gripper_dof_indices):
            if i < len(target_positions):
                joint_positions[idx] = target_positions[i]
        self._robot.apply_action(ArticulationAction(joint_positions=joint_positions))

    def _apply_base_velocity(self, vx, vy, omega):
        if not self._wheel_dof_indices:
            return
        wheel_vels = self._mecanum.compute_wheel_velocities(vx, vy, omega)
        n_dofs = self._robot.num_dof
        joint_velocities = np.zeros(n_dofs)
        for i, idx in enumerate(self._wheel_dof_indices):
            if i < len(wheel_vels):
                joint_velocities[idx] = wheel_vels[i]
        self._robot.apply_action(ArticulationAction(joint_velocities=joint_velocities))

    def _stop_base(self):
        self._apply_base_velocity(0.0, 0.0, 0.0)

    def _compute_arm_ik(self, target_relative_pos):
        x = np.clip(target_relative_pos[0], 0.1, 0.85)
        y = np.clip(target_relative_pos[1], -0.5, 0.5)
        z = np.clip(target_relative_pos[2], 0.0, 0.8)

        reach = np.sqrt(x * x + y * y)

        q1 = np.arctan2(y, x)
        q2 = -0.785 + (z - 0.3) * 0.8
        q3 = 0.0
        q4 = -2.356 + (0.55 - reach) * 1.2
        q5 = 0.0
        q6 = 1.571 + (z - 0.3) * 0.4
        q7 = 0.785

        positions = np.array([q1, q2, q3, q4, q5, q6, q7])
        return np.clip(positions, -2.8973, 2.8973)

    def forward(self):
        if self._state == RobotState.NAVIGATE:
            self._step_navigate()
        elif self._state == RobotState.ALIGN:
            self._step_align()
        elif self._state in (RobotState.PICK, RobotState.LIFT, RobotState.MOVE_TO_PLACE,
                             RobotState.PLACE, RobotState.OPEN_GRIPPER, RobotState.RETREAT):
            self._step_pick_place()
        elif self._state == RobotState.DONE:
            self._stop_base()

    def _step_navigate(self):
        base_pos, base_yaw = self._get_robot_pose()
        nav_target = OBJECT_POSITION[:2] - np.array([0.45, 0.0])

        cmd, reached = self._nav_controller.compute_command(
            current_pos=base_pos[:2],
            current_yaw=base_yaw,
            target_pos=nav_target,
        )

        if reached:
            print("Navigation complete - near target object")
            self._stop_base()
            self._state = RobotState.ALIGN
            self._align_counter = 0
        else:
            self._apply_base_velocity(cmd[0], cmd[1], cmd[2])

    def _step_align(self):
        self._align_counter = getattr(self, '_align_counter', 0) + 1
        self._stop_base()
        self._apply_arm_target(FRANKA_HOME_POSITIONS)
        self._apply_gripper_target(GRIPPER_OPEN)

        if self._align_counter > 90:
            print("Alignment complete - starting pick-and-place")
            self._state = RobotState.PICK
            self._align_counter = 0

    def _step_pick_place(self):
        if self._pick_place_sm.is_done:
            print("Pick-and-place complete!")
            self._state = RobotState.DONE
            return

        self._stop_base()

        ee_pos = self._get_ee_position()
        target_pos = self._pick_place_sm.get_target_position(ee_pos)
        base_pos, base_yaw = self._get_robot_pose()

        relative_target = target_pos - base_pos
        cos_yaw = np.cos(-base_yaw)
        sin_yaw = np.sin(-base_yaw)
        local_x = relative_target[0] * cos_yaw - relative_target[1] * sin_yaw
        local_y = relative_target[0] * sin_yaw + relative_target[1] * cos_yaw
        local_z = relative_target[2] - base_pos[2]
        local_target = np.array([local_x, local_y, local_z])

        joint_targets = self._compute_arm_ik(local_target)
        self._apply_arm_target(joint_targets)

        if self._pick_place_sm.should_close_gripper():
            self._apply_gripper_target(GRIPPER_CLOSE)
        elif self._pick_place_sm.should_open_gripper():
            self._apply_gripper_target(GRIPPER_OPEN)

        self._pick_place_sm.advance(ee_pos)

    def is_done(self):
        return self._state == RobotState.DONE

    def reset(self):
        self._state = RobotState.NAVIGATE
        self._pick_place_sm = PickPlaceStateMachine(
            pick_pos=OBJECT_POSITION,
            place_pos=PLACE_POSITION,
        )
        self._world.reset()
        self._wheel_dof_indices = []
        self._arm_dof_indices = []
        self._gripper_dof_indices = []
        for name in RIDGEBACK_WHEEL_DOF_NAMES:
            try:
                self._wheel_dof_indices.append(self._robot.get_dof_index(name))
            except Exception:
                pass
        for name in FRANKA_ARM_DOF_NAMES:
            try:
                self._arm_dof_indices.append(self._robot.get_dof_index(name))
            except Exception:
                pass
        for name in FRANKA_GRIPPER_DOF_NAMES:
            try:
                self._gripper_dof_indices.append(self._robot.get_dof_index(name))
            except Exception:
                pass
        self._set_arm_positions(FRANKA_HOME_POSITIONS)
        self._set_gripper(GRIPPER_OPEN)

    def step_world(self):
        self._world.step(render=True)


def main():
    print("=" * 60)
    print("Ridgeback Franka Pick-and-Place in Warehouse Environment")
    print("=" * 60)

    SimulationManager.set_physics_sim_device(args.device)
    simulation_app.update()

    controller = RidgebackFrankaPickPlace()
    controller.setup_scene()

    omni.timeline.get_timeline_interface().play()
    simulation_app.update()

    controller.initialize()

    task_completed = False
    step_count = 0

    print(f"\nStarting execution (IK: {args.ik_method}, Device: {args.device})")
    print("-" * 60)

    while simulation_app.is_running():
        if not task_completed:
            controller.forward()
            step_count += 1

            if step_count % 300 == 0:
                pos, yaw = controller._get_robot_pose()
                print(f"  Step {step_count}: state={controller._state}, "
                      f"pos=({pos[0]:.2f}, {pos[1]:.2f}), yaw={np.degrees(yaw):.1f} deg")

        if controller.is_done() and not task_completed:
            print("-" * 60)
            print(f"Task complete! Total steps: {step_count}")
            task_completed = True

        controller.step_world()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        print(f"Error: {e}")
        traceback.print_exc()
    finally:
        simulation_app.close()
