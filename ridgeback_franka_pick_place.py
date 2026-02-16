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
from isaacsim.core.api.objects import DynamicCuboid, FixedCuboid
from isaacsim.core.prims import XFormPrim
from isaacsim.core.simulation_manager import SimulationManager
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.storage.native import get_assets_root_path
from pxr import Gf, UsdGeom, UsdPhysics


class RobotState:
    NAVIGATE = "navigate"
    ALIGN = "align"
    PICK = "pick"
    LIFT = "lift"
    NAVIGATE_TO_PLACE = "navigate_to_place"
    PLACE = "place"
    DONE = "done"


class NavigationController:
    def __init__(self, linear_speed=0.5, angular_speed=1.0, position_threshold=0.8, angle_threshold=0.1):
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
            return np.array([0.0, 0.0, omega]), False

        vx = np.clip(self._linear_speed * distance, 0.0, self._linear_speed)
        return np.array([vx, 0.0, 0.0]), False


class SimpleIKController:
    def __init__(self, method="damped-least-squares"):
        self._method = method
        self._damping = 0.05

    def compute_joint_velocities(self, jacobian, ee_pos, ee_orient, target_pos, target_orient):
        pos_error = target_pos - ee_pos
        orient_error = self._orientation_error(ee_orient, target_orient)
        error = np.concatenate([pos_error, orient_error])

        if self._method == "damped-least-squares":
            jt = jacobian.T
            jjt = jacobian @ jt
            damped = jjt + (self._damping ** 2) * np.eye(jjt.shape[0])
            joint_velocities = jt @ np.linalg.solve(damped, error)
        elif self._method == "pseudoinverse":
            joint_velocities = np.linalg.pinv(jacobian) @ error
        elif self._method == "transpose":
            joint_velocities = jacobian.T @ error
        elif self._method == "singular-value-decomposition":
            u, s, vt = np.linalg.svd(jacobian, full_matrices=False)
            s_inv = np.where(s > 1e-5, 1.0 / s, 0.0)
            joint_velocities = vt.T @ np.diag(s_inv) @ u.T @ error
        else:
            joint_velocities = np.linalg.pinv(jacobian) @ error

        return joint_velocities

    def _orientation_error(self, current_quat, target_quat):
        q_c = current_quat
        q_t = target_quat
        q_c_inv = np.array([-q_c[0], -q_c[1], -q_c[2], q_c[3]])
        error_quat = self._quat_multiply(q_t, q_c_inv)
        if error_quat[3] < 0:
            error_quat = -error_quat
        return error_quat[:3] * 2.0

    def _quat_multiply(self, q1, q2):
        x1, y1, z1, w1 = q1
        x2, y2, z2, w2 = q2
        return np.array([
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        ])


class PickPlaceStateMachine:
    def __init__(self, pick_pos, place_pos, lift_height=0.15):
        self._pick_pos = np.array(pick_pos)
        self._place_pos = np.array(place_pos)
        self._lift_height = lift_height
        self._phase = "approach"
        self._phase_counter = 0
        self._phases = ["approach", "pick", "close_gripper", "lift", "move_to_place", "place", "open_gripper", "done"]

    @property
    def phase(self):
        return self._phase

    @property
    def is_done(self):
        return self._phase == "done"

    def get_target_position(self, ee_pos):
        if self._phase == "approach":
            target = self._pick_pos.copy()
            target[2] += self._lift_height
            return target
        elif self._phase == "pick":
            return self._pick_pos.copy()
        elif self._phase == "close_gripper":
            return self._pick_pos.copy()
        elif self._phase == "lift":
            target = self._pick_pos.copy()
            target[2] += self._lift_height
            return target
        elif self._phase == "move_to_place":
            target = self._place_pos.copy()
            target[2] += self._lift_height
            return target
        elif self._phase == "place":
            return self._place_pos.copy()
        elif self._phase == "open_gripper":
            return self._place_pos.copy()
        return ee_pos

    def should_close_gripper(self):
        return self._phase == "close_gripper"

    def should_open_gripper(self):
        return self._phase == "open_gripper"

    def advance(self, ee_pos, threshold=0.02):
        target = self.get_target_position(ee_pos)
        dist = np.linalg.norm(ee_pos - target)

        if self._phase in ["close_gripper", "open_gripper"]:
            self._phase_counter += 1
            if self._phase_counter > 60:
                self._phase_counter = 0
                self._next_phase()
        elif dist < threshold:
            self._phase_counter += 1
            if self._phase_counter > 10:
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

        self._stage = omni.usd.get_context().get_stage()

        self._robot_prim_path = "/World/RidgebackFranka"
        self._warehouse_prim_path = "/World/Warehouse"
        self._target_object_path = "/World/TargetCube"
        self._place_target_path = "/World/PlaceTarget"

        self._robot_initial_pos = np.array([-4.0, -4.0, 0.0])
        self._object_position = np.array([0.0, 0.0, 0.05])
        self._place_position = np.array([1.0, 0.5, 0.05])

        self._nav_target_offset = np.array([-0.5, 0.0, 0.0])

        self._state = RobotState.NAVIGATE
        self._nav_controller = NavigationController(
            linear_speed=0.8,
            angular_speed=1.5,
            position_threshold=0.6,
        )
        self._ik_controller = None
        self._pick_place_sm = None

        self._franka_arm_joint_names = [
            "panda_joint1", "panda_joint2", "panda_joint3", "panda_joint4",
            "panda_joint5", "panda_joint6", "panda_joint7",
        ]
        self._franka_gripper_joint_names = [
            "panda_finger_joint1", "panda_finger_joint2",
        ]
        self._ridgeback_wheel_joint_names = [
            "front_left_wheel", "front_right_wheel",
            "rear_left_wheel", "rear_right_wheel",
        ]

        self._franka_home_positions = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])
        self._gripper_open_positions = np.array([0.04, 0.04])
        self._gripper_close_positions = np.array([0.0, 0.0])

        self._arm_joint_indices = None
        self._gripper_joint_indices = None
        self._wheel_joint_indices = None
        self._robot_articulation = None

    def setup_scene(self):
        print("Setting up warehouse environment...")
        warehouse_usd = self._assets_root_path + "/Isaac/Environments/Simple_Warehouse/warehouse_multiple_shelves.usd"
        add_reference_to_stage(usd_path=warehouse_usd, prim_path=self._warehouse_prim_path)
        print(f"  Loaded warehouse from: {warehouse_usd}")

        print("Setting up Ridgeback Franka robot...")
        robot_usd = self._assets_root_path + "/Isaac/Robots/Clearpath/RidgebackFranka/ridgeback_franka.usd"
        robot_prim = add_reference_to_stage(usd_path=robot_usd, prim_path=self._robot_prim_path)

        xform = UsdGeom.Xformable(robot_prim)
        xform.ClearXformOpOrder()
        translate_op = xform.AddTranslateOp()
        translate_op.Set(Gf.Vec3d(
            float(self._robot_initial_pos[0]),
            float(self._robot_initial_pos[1]),
            float(self._robot_initial_pos[2]),
        ))
        print(f"  Robot placed at: {self._robot_initial_pos}")

        print("Setting up target object...")
        self._target_cube = DynamicCuboid(
            prim_path=self._target_object_path,
            name="target_cube",
            position=np.array([self._object_position[0], self._object_position[1], self._object_position[2] + 0.5]),
            size=0.05,
            color=np.array([1.0, 0.0, 0.0]),
            mass=0.1,
        )

        self._place_marker = FixedCuboid(
            prim_path=self._place_target_path,
            name="place_marker",
            position=np.array([self._place_position[0], self._place_position[1], 0.01]),
            size=0.06,
            color=np.array([0.0, 1.0, 0.0]),
        )

        print("Scene setup complete.")

    def initialize(self):
        print("Initializing robot articulation...")
        from pxr import PhysxSchema

        robot_prim = self._stage.GetPrimAtPath(self._robot_prim_path)
        if not robot_prim.IsValid():
            raise RuntimeError(f"Robot prim not found at {self._robot_prim_path}")

        self._find_joint_indices()
        self._ik_controller = SimpleIKController(method=args.ik_method)
        self._pick_place_sm = PickPlaceStateMachine(
            pick_pos=self._object_position,
            place_pos=self._place_position,
        )
        print("Robot initialized.")

    def _find_joint_indices(self):
        from pxr import UsdPhysics as UsdPhys
        self._arm_joints = {}
        self._gripper_joints = {}
        self._wheel_joints = {}

        for prim in self._stage.Traverse():
            if prim.IsA(UsdPhys.RevoluteJoint) or prim.IsA(UsdPhys.PrismaticJoint):
                name = prim.GetName()
                if name in self._franka_arm_joint_names:
                    self._arm_joints[name] = prim.GetPath().pathString
                elif name in self._franka_gripper_joint_names:
                    self._gripper_joints[name] = prim.GetPath().pathString
                elif name in self._ridgeback_wheel_joint_names:
                    self._wheel_joints[name] = prim.GetPath().pathString

        print(f"  Found {len(self._arm_joints)} arm joints, {len(self._gripper_joints)} gripper joints, {len(self._wheel_joints)} wheel joints")

    def get_robot_base_pose(self):
        xform_prim = self._stage.GetPrimAtPath(self._robot_prim_path)
        if not xform_prim.IsValid():
            return np.zeros(3), 0.0

        xformable = UsdGeom.Xformable(xform_prim)
        transform = xformable.ComputeLocalToWorldTransform(0)
        translation = transform.ExtractTranslation()
        pos = np.array([translation[0], translation[1], translation[2]])

        rotation = transform.ExtractRotationMatrix()
        forward = rotation.GetColumn(0)
        yaw = np.arctan2(forward[1], forward[0])

        return pos, yaw

    def get_ee_pose(self):
        ee_paths = [
            self._robot_prim_path + "/panda_hand",
            self._robot_prim_path + "/panda_link8",
            self._robot_prim_path + "/panda_link7",
        ]
        for path in ee_paths:
            prim = self._stage.GetPrimAtPath(path)
            if prim.IsValid():
                xformable = UsdGeom.Xformable(prim)
                transform = xformable.ComputeLocalToWorldTransform(0)
                translation = transform.ExtractTranslation()
                pos = np.array([translation[0], translation[1], translation[2]])
                rotation = transform.ExtractRotation()
                quat = rotation.GetQuat()
                q_imag = quat.GetImaginary()
                q_real = quat.GetReal()
                orient = np.array([q_imag[0], q_imag[1], q_imag[2], q_real])
                return pos, orient

        base_pos, _ = self.get_robot_base_pose()
        return base_pos + np.array([0.0, 0.0, 0.5]), np.array([0.0, 0.0, 0.0, 1.0])

    def apply_base_velocity(self, cmd):
        vx, vy, omega = cmd
        robot_prim = self._stage.GetPrimAtPath(self._robot_prim_path)
        if not robot_prim.IsValid():
            return

        xformable = UsdGeom.Xformable(robot_prim)
        transform = xformable.ComputeLocalToWorldTransform(0)
        translation = transform.ExtractTranslation()
        current_pos = np.array([translation[0], translation[1], translation[2]])

        rotation = transform.ExtractRotationMatrix()
        forward = rotation.GetColumn(0)
        current_yaw = np.arctan2(forward[1], forward[0])

        dt = 1.0 / 60.0
        new_yaw = current_yaw + omega * dt
        dx = vx * np.cos(new_yaw) * dt - vy * np.sin(new_yaw) * dt
        dy = vx * np.sin(new_yaw) * dt + vy * np.cos(new_yaw) * dt

        new_pos = current_pos + np.array([dx, dy, 0.0])

        xformable.ClearXformOpOrder()

        translate_op = xformable.AddTranslateOp()
        translate_op.Set(Gf.Vec3d(float(new_pos[0]), float(new_pos[1]), float(new_pos[2])))

        rotate_op = xformable.AddRotateZOp()
        rotate_op.Set(float(np.degrees(new_yaw)))

    def apply_arm_positions(self, positions):
        for i, joint_name in enumerate(self._franka_arm_joint_names):
            if joint_name in self._arm_joints:
                joint_path = self._arm_joints[joint_name]
                joint_prim = self._stage.GetPrimAtPath(joint_path)
                if joint_prim.IsValid():
                    drive_api = UsdPhysics.DriveAPI.Get(joint_prim, "angular")
                    if drive_api:
                        drive_api.GetTargetPositionAttr().Set(float(np.degrees(positions[i])))

    def apply_gripper_positions(self, positions):
        for i, joint_name in enumerate(self._franka_gripper_joint_names):
            if joint_name in self._gripper_joints:
                joint_path = self._gripper_joints[joint_name]
                joint_prim = self._stage.GetPrimAtPath(joint_path)
                if joint_prim.IsValid():
                    drive_api = UsdPhysics.DriveAPI.Get(joint_prim, "linear")
                    if drive_api:
                        drive_api.GetTargetPositionAttr().Set(float(positions[i] * 100.0))

    def forward(self):
        if self._state == RobotState.NAVIGATE:
            self._step_navigate()
        elif self._state == RobotState.ALIGN:
            self._step_align()
        elif self._state in [RobotState.PICK, RobotState.LIFT, RobotState.PLACE]:
            self._step_pick_place()
        elif self._state == RobotState.NAVIGATE_TO_PLACE:
            self._step_navigate_to_place()

    def _step_navigate(self):
        base_pos, base_yaw = self.get_robot_base_pose()
        nav_target = self._object_position + self._nav_target_offset

        cmd, reached = self._nav_controller.compute_command(
            current_pos=base_pos[:2],
            current_yaw=base_yaw,
            target_pos=nav_target[:2],
        )

        if reached:
            print("Navigation complete - reached near object")
            self._state = RobotState.ALIGN
            self.apply_base_velocity(np.array([0.0, 0.0, 0.0]))
            self.apply_arm_positions(self._franka_home_positions)
            self.apply_gripper_positions(self._gripper_open_positions)
        else:
            self.apply_base_velocity(cmd)

    def _step_align(self):
        self._align_counter = getattr(self, '_align_counter', 0) + 1
        self.apply_arm_positions(self._franka_home_positions)
        self.apply_gripper_positions(self._gripper_open_positions)

        if self._align_counter > 60:
            print("Alignment complete - starting pick-and-place")
            self._state = RobotState.PICK
            self._align_counter = 0

    def _step_pick_place(self):
        if self._pick_place_sm.is_done:
            print("Pick-and-place complete!")
            self._state = RobotState.DONE
            return

        ee_pos, ee_orient = self.get_ee_pose()
        target_pos = self._pick_place_sm.get_target_position(ee_pos)

        direction = target_pos - ee_pos
        distance = np.linalg.norm(direction)

        if distance > 0.001:
            step_size = min(0.002, distance)
            move = (direction / distance) * step_size
            new_arm_target = ee_pos + move

            base_pos, _ = self.get_robot_base_pose()
            arm_relative_target = new_arm_target - base_pos

            joint_positions = self._compute_arm_ik(arm_relative_target)
            self.apply_arm_positions(joint_positions)

        if self._pick_place_sm.should_close_gripper():
            self.apply_gripper_positions(self._gripper_close_positions)
        elif self._pick_place_sm.should_open_gripper():
            self.apply_gripper_positions(self._gripper_open_positions)

        self._pick_place_sm.advance(ee_pos)

    def _step_navigate_to_place(self):
        base_pos, base_yaw = self.get_robot_base_pose()
        place_nav_target = self._place_position + self._nav_target_offset

        cmd, reached = self._nav_controller.compute_command(
            current_pos=base_pos[:2],
            current_yaw=base_yaw,
            target_pos=place_nav_target[:2],
        )

        if reached:
            print("Reached place location")
            self._state = RobotState.PLACE
        else:
            self.apply_base_velocity(cmd)

    def _compute_arm_ik(self, target_relative_pos):
        target = np.array([
            np.clip(target_relative_pos[0], 0.2, 0.8),
            np.clip(target_relative_pos[1], -0.5, 0.5),
            np.clip(target_relative_pos[2], 0.02, 0.8),
        ])

        x, y, z = target
        reach = np.sqrt(x * x + y * y)

        q1 = np.arctan2(y, x)
        q2 = -0.785 + (z - 0.3) * 0.5
        q3 = 0.0
        q4 = -2.356 + (0.5 - reach) * 1.0
        q5 = 0.0
        q6 = 1.571 + (z - 0.3) * 0.3
        q7 = 0.785

        positions = np.array([q1, q2, q3, q4, q5, q6, q7])
        positions = np.clip(positions, -2.8, 2.8)
        return positions

    def is_done(self):
        return self._state == RobotState.DONE

    def reset(self):
        self._state = RobotState.NAVIGATE
        self._pick_place_sm = PickPlaceStateMachine(
            pick_pos=self._object_position,
            place_pos=self._place_position,
        )

        xform_prim = self._stage.GetPrimAtPath(self._robot_prim_path)
        if xform_prim.IsValid():
            xformable = UsdGeom.Xformable(xform_prim)
            xformable.ClearXformOpOrder()
            translate_op = xformable.AddTranslateOp()
            translate_op.Set(Gf.Vec3d(
                float(self._robot_initial_pos[0]),
                float(self._robot_initial_pos[1]),
                float(self._robot_initial_pos[2]),
            ))

        self.apply_arm_positions(self._franka_home_positions)
        self.apply_gripper_positions(self._gripper_open_positions)


def main():
    print("=" * 60)
    print("Ridgeback Franka Pick-and-Place in Warehouse Environment")
    print("=" * 60)

    SimulationManager.set_physics_sim_device(args.device)
    simulation_app.update()

    ridgeback_franka = RidgebackFrankaPickPlace()
    ridgeback_franka.setup_scene()

    omni.timeline.get_timeline_interface().play()
    simulation_app.update()

    ridgeback_franka.initialize()

    reset_needed = True
    task_completed = False

    print("\nStarting Ridgeback Franka pick-and-place execution")
    print(f"  IK Method: {args.ik_method}")
    print(f"  Device: {args.device}")
    print("-" * 60)

    step_count = 0
    while simulation_app.is_running():
        if SimulationManager.is_simulating() and not task_completed:
            if reset_needed:
                ridgeback_franka.reset()
                reset_needed = False

            ridgeback_franka.forward()
            step_count += 1

            if step_count % 300 == 0:
                base_pos, base_yaw = ridgeback_franka.get_robot_base_pose()
                print(f"  Step {step_count}: state={ridgeback_franka._state}, "
                      f"base_pos=({base_pos[0]:.2f}, {base_pos[1]:.2f}), "
                      f"yaw={np.degrees(base_yaw):.1f}°")

        if ridgeback_franka.is_done() and not task_completed:
            print("-" * 60)
            print("Task complete: Ridgeback Franka pick-and-place finished!")
            print(f"Total simulation steps: {step_count}")
            task_completed = True

        simulation_app.update()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        print(f"Error: {e}")
        traceback.print_exc()
    finally:
        simulation_app.close()
