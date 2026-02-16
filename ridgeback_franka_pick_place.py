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

parser = argparse.ArgumentParser()
parser.add_argument("--device", type=str, choices=["cpu", "cuda"], default="cpu", help="Simulation device")
parser.add_argument(
    "--ik-method",
    type=str,
    choices=["singular-value-decomposition", "pseudoinverse", "transpose", "damped-least-squares"],
    default="damped-least-squares",
    help="Differential inverse kinematics method",
)
args, _ = parser.parse_known_args()

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

from typing import List, Optional

import isaacsim.core.experimental.utils.stage as stage_utils
import numpy as np
import omni.timeline
import warp as wp
from isaacsim.core.experimental.materials import PreviewSurfaceMaterial
from isaacsim.core.experimental.objects import Cube
from isaacsim.core.experimental.prims import Articulation, GeomPrim, RigidPrim
from isaacsim.core.experimental.utils.impl.transform import quaternion_conjugate, quaternion_multiplication
from isaacsim.core.simulation_manager import SimulationManager
from isaacsim.storage.native import get_assets_root_path

RIDGEBACK_FRANKA_USD_PATH = "/Isaac/Robots/Clearpath/RidgebackFranka/ridgeback_franka.usd"

RIDGEBACK_BASE_HEIGHT = 0.278

RIDGEBACK_WHEEL_DOF_NAMES = [
    "front_left_wheel",
    "front_right_wheel",
    "rear_left_wheel",
    "rear_right_wheel",
]

FRANKA_ARM_DOF_NAMES = [
    "panda_joint1",
    "panda_joint2",
    "panda_joint3",
    "panda_joint4",
    "panda_joint5",
    "panda_joint6",
    "panda_joint7",
]

FRANKA_GRIPPER_DOF_NAMES = [
    "panda_finger_joint1",
    "panda_finger_joint2",
]

FRANKA_ARM_DEFAULT_POSITIONS = [0.012, -0.568, 0.0, -2.811, 0.0, 3.037, 0.741]

FRANKA_GRIPPER_OPEN_POSITIONS = [0.04, 0.04]

FRANKA_GRIPPER_CLOSED_POSITIONS = [0.0, 0.0]


class RidgebackFrankaExperimental(Articulation):
    """Ridgeback Franka robot controller with inverse kinematics and gripper control.

    This class inherits from Articulation and provides high-level control commands
    for the Clearpath Ridgeback mobile base with a Franka Emika Panda arm mounted on top.
    It handles the combined articulation (mobile base wheels + arm joints + gripper joints),
    exposing the arm IK and gripper control while keeping the base stationary or
    allowing explicit base repositioning.
    """

    def __init__(
        self,
        robot_path: str = "/World/robot",
        create_robot: bool = True,
        end_effector_link: Optional[RigidPrim] = None,
    ):
        """Initialize the Ridgeback Franka controller.

        Args:
            robot_path: USD path where the robot should be created or exists.
            create_robot: Whether to create a new robot from USD assets.
            end_effector_link: The end effector rigid body link. If None, creates from robot_path.
        """
        if create_robot:
            stage_utils.add_reference_to_stage(
                usd_path=get_assets_root_path() + RIDGEBACK_FRANKA_USD_PATH,
                path=robot_path,
            )

        super().__init__(robot_path)

        if end_effector_link is None:
            self.end_effector_link = RigidPrim(f"{robot_path}/panda_hand")
        else:
            self.end_effector_link = end_effector_link

        self._robot_path = robot_path
        self._arm_dof_indices = None
        self._gripper_dof_indices = None
        self._wheel_dof_indices = None
        self._num_arm_dofs = len(FRANKA_ARM_DOF_NAMES)

        if create_robot:
            self._set_default_state()

        self.end_effector_link_index = self.get_link_indices("panda_hand").list()[0]

        self.gripper_open_position = np.array([FRANKA_GRIPPER_OPEN_POSITIONS])
        self.gripper_closed_position = np.array([FRANKA_GRIPPER_CLOSED_POSITIONS])

    def _set_default_state(self) -> None:
        """Set the robot to its default state with the arm in a ready pose and gripper open."""
        all_dof_names = self.get_dof_names()
        num_dofs = self.get_dof_count()
        default_positions = np.zeros(num_dofs)

        for i, name in enumerate(all_dof_names):
            for j, arm_name in enumerate(FRANKA_ARM_DOF_NAMES):
                if arm_name in name:
                    default_positions[i] = FRANKA_ARM_DEFAULT_POSITIONS[j]
                    break
            for j, gripper_name in enumerate(FRANKA_GRIPPER_DOF_NAMES):
                if gripper_name in name:
                    default_positions[i] = FRANKA_GRIPPER_OPEN_POSITIONS[j]
                    break

        self.set_default_state(dof_positions=default_positions.tolist())

    def _resolve_dof_indices(self) -> None:
        """Resolve DOF indices for the arm, gripper, and wheel joints."""
        all_dof_names = self.get_dof_names()

        arm_indices = []
        for arm_name in FRANKA_ARM_DOF_NAMES:
            for i, name in enumerate(all_dof_names):
                if arm_name in name:
                    arm_indices.append(i)
                    break
        self._arm_dof_indices = arm_indices

        gripper_indices = []
        for gripper_name in FRANKA_GRIPPER_DOF_NAMES:
            for i, name in enumerate(all_dof_names):
                if gripper_name in name:
                    gripper_indices.append(i)
                    break
        self._gripper_dof_indices = gripper_indices

        wheel_indices = []
        for wheel_name in RIDGEBACK_WHEEL_DOF_NAMES:
            for i, name in enumerate(all_dof_names):
                if wheel_name in name:
                    wheel_indices.append(i)
                    break
        self._wheel_dof_indices = wheel_indices

    @property
    def arm_dof_indices(self):
        if self._arm_dof_indices is None:
            self._resolve_dof_indices()
        return self._arm_dof_indices

    @property
    def gripper_dof_indices(self):
        if self._gripper_dof_indices is None:
            self._resolve_dof_indices()
        return self._gripper_dof_indices

    @property
    def wheel_dof_indices(self):
        if self._wheel_dof_indices is None:
            self._resolve_dof_indices()
        return self._wheel_dof_indices

    def differential_inverse_kinematics(
        self,
        jacobian_end_effector: np.ndarray,
        current_position: np.ndarray,
        current_orientation: np.ndarray,
        goal_position: np.ndarray,
        goal_orientation: Optional[np.ndarray] = None,
        method: str = "damped-least-squares",
        method_cfg: dict[str, float] = None,
    ) -> np.ndarray:
        """Compute differential inverse kinematics for the arm joints only.

        Args:
            jacobian_end_effector: The Jacobian matrix of the end-effector (arm columns only).
            current_position: The current position of the end-effector.
            current_orientation: The current orientation as quaternion [w, x, y, z].
            goal_position: The desired position of the end-effector.
            goal_orientation: The desired orientation as quaternion [w, x, y, z].
            method: IK method to use.
            method_cfg: Configuration for the selected method.

        Returns:
            Delta joint positions for the arm joints.
        """
        if method_cfg is None:
            method_cfg = {"scale": 1.0, "damping": 0.05, "min_singular_value": 1e-5}
        scale = method_cfg.get("scale", 1.0)

        goal_orientation = current_orientation if goal_orientation is None else goal_orientation

        goal_quat_wp = wp.from_numpy(goal_orientation, dtype=wp.float32)
        current_quat_wp = wp.from_numpy(current_orientation, dtype=wp.float32)

        current_quat_conjugate_wp = quaternion_conjugate(current_quat_wp)
        q_wp = quaternion_multiplication(goal_quat_wp, current_quat_conjugate_wp)
        q_np = q_wp.numpy()

        error = np.expand_dims(
            np.concatenate([goal_position - current_position, q_np[:, 1:] * np.sign(q_np[:, [0]])], axis=-1), axis=2
        )

        if method == "singular-value-decomposition":
            min_singular_value = method_cfg.get("min_singular_value", 1e-5)
            U, S, Vh = np.linalg.svd(jacobian_end_effector)
            inv_s = np.where(S > min_singular_value, 1.0 / S, np.zeros_like(S))
            pseudoinverse = (
                np.swapaxes(Vh, 1, 2)[:, :, :6] @ np.diagflat(inv_s) @ np.swapaxes(U, 1, 2)
            )
            return (scale * pseudoinverse @ error).squeeze(-1)
        elif method == "pseudoinverse":
            pseudoinverse = np.linalg.pinv(jacobian_end_effector)
            return (scale * pseudoinverse @ error).squeeze(-1)
        elif method == "transpose":
            transpose = np.swapaxes(jacobian_end_effector, 1, 2)
            return (scale * transpose @ error).squeeze(-1)
        elif method == "damped-least-squares":
            damping = method_cfg.get("damping", 0.05)
            transpose = np.swapaxes(jacobian_end_effector, 1, 2)
            lmbda = np.eye(jacobian_end_effector.shape[1]) * (damping**2)
            return (
                scale * transpose @ np.linalg.inv(jacobian_end_effector @ transpose + lmbda) @ error
            ).squeeze(-1)
        else:
            raise ValueError(f"Invalid IK method: {method}")

    def get_current_state(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get current robot state including DOF positions and end effector pose.

        Returns:
            Tuple of (current_dof_positions, current_ee_position, current_ee_orientation).
        """
        current_dof_positions = self.get_dof_positions().numpy()
        current_end_effector_position, current_end_effector_orientation = self.end_effector_link.get_world_poses()
        current_end_effector_position = current_end_effector_position.numpy()
        current_end_effector_orientation = current_end_effector_orientation.numpy()
        return current_dof_positions, current_end_effector_position, current_end_effector_orientation

    def set_end_effector_pose(
        self,
        position: np.ndarray,
        orientation: np.ndarray,
        ik_method: str = "damped-least-squares",
    ) -> None:
        """Set the end effector to a specific pose using IK on the arm joints only.

        The mobile base remains stationary; only the 7 Franka arm joints are actuated.

        Args:
            position: Target position [x, y, z].
            orientation: Target orientation as quaternion [w, x, y, z].
            ik_method: The inverse kinematics method to use.
        """
        current_dof_positions, current_ee_position, current_ee_orientation = self.get_current_state()

        if position.ndim == 1:
            position = position.reshape(1, -1)

        jacobian_matrices = self.get_jacobian_matrices().numpy()

        arm_col_indices = self.arm_dof_indices
        jacobian_end_effector = jacobian_matrices[:, self.end_effector_link_index - 1, :, :]
        jacobian_arm = jacobian_end_effector[:, :, arm_col_indices]

        delta_dof_positions = self.differential_inverse_kinematics(
            jacobian_end_effector=jacobian_arm,
            current_position=current_ee_position,
            current_orientation=current_ee_orientation,
            goal_position=position,
            goal_orientation=orientation,
            method=ik_method,
        )

        current_arm_positions = current_dof_positions[:, arm_col_indices]
        arm_position_targets = current_arm_positions + delta_dof_positions
        self.set_dof_position_targets(arm_position_targets, dof_indices=self.arm_dof_indices)

    def open_gripper(self) -> None:
        """Open the gripper to the default open position."""
        self.set_dof_position_targets(self.gripper_open_position, dof_indices=self.gripper_dof_indices)

    def close_gripper(self) -> None:
        """Close the gripper to the default closed position."""
        self.set_dof_position_targets(self.gripper_closed_position, dof_indices=self.gripper_dof_indices)

    def set_gripper_position(self, position: np.ndarray) -> None:
        """Set gripper to a specific position.

        Args:
            position: Gripper position [finger1, finger2] where 0.0 is closed and 0.04 is open.
        """
        if position.ndim == 1:
            position = position.reshape(1, -1)
        self.set_dof_position_targets(position, dof_indices=self.gripper_dof_indices)

    def get_downward_orientation(self) -> np.ndarray:
        """Get the standard downward-facing orientation for the end effector.

        Returns:
            Quaternion [w, x, y, z] representing downward-facing orientation.
        """
        return np.array([[0.0, 1.0, 0.0, 0.0]])

    def reset_to_default_pose(self) -> None:
        """Reset the robot to its default pose with open gripper.

        Resets the arm joints to their default configuration and the gripper to open.
        Wheel joints are set to zero velocity.
        """
        num_dofs = self.get_dof_count()
        all_dof_names = self.get_dof_names()
        default_positions = np.zeros((1, num_dofs))

        for i, name in enumerate(all_dof_names):
            for j, arm_name in enumerate(FRANKA_ARM_DOF_NAMES):
                if arm_name in name:
                    default_positions[0, i] = FRANKA_ARM_DEFAULT_POSITIONS[j]
                    break
            for j, gripper_name in enumerate(FRANKA_GRIPPER_DOF_NAMES):
                if gripper_name in name:
                    default_positions[0, i] = FRANKA_GRIPPER_OPEN_POSITIONS[j]
                    break

        self.set_dof_positions(default_positions)
        self.set_dof_position_targets(default_positions)


class RidgebackFrankaPickPlace:
    """Pick-and-place controller for the Ridgeback Franka mobile manipulator.

    Uses the Clearpath Ridgeback mobile base with a Franka Emika Panda arm
    mounted on top. The mobile base remains stationary while the arm performs
    the pick-and-place operation via differential inverse kinematics.

    State machine phases:
        0: Move to x,y position above cube
        1: Approach down to cube
        2: Close gripper to grasp
        3: Lift cube upward
        4: Move cube to target location
        5: Open gripper to release
        6: Move up and away
    """

    def __init__(self, events_dt: Optional[List[float]] = None):
        """Initialize the RidgebackFrankaPickPlace controller.

        Args:
            events_dt: List of step counts for each phase. If None, uses default values.
        """
        self.cube = None
        self.robot = None

        self.events_dt = events_dt
        if self.events_dt is None:
            self.events_dt = [
                80,
                50,
                25,
                50,
                100,
                25,
                25,
            ]
        self._event = 0
        self._step = 0

    def setup_scene(
        self,
        cube_initial_position: Optional[np.ndarray] = None,
        cube_initial_orientation: Optional[np.ndarray] = None,
        cube_size: Optional[np.ndarray] = None,
        target_position: Optional[np.ndarray] = None,
        offset: Optional[np.ndarray] = None,
    ) -> None:
        """Set up the scene with the Ridgeback Franka robot and a cube.

        Positions are adjusted to account for the elevated arm mounting on the
        Ridgeback platform.

        Args:
            cube_initial_position: Initial cube position [x, y, z].
            cube_initial_orientation: Initial cube orientation as quaternion [w, x, y, z].
            cube_size: Cube dimensions [w, h, d].
            target_position: Target position for placing [x, y, z].
            offset: Additional offset to apply to target position [x, y, z].
        """
        self.cube_initial_position = cube_initial_position
        self.cube_initial_orientation = cube_initial_orientation
        self.target_position = target_position
        self.cube_size = cube_size
        self.offset = offset

        if self.cube_size is None:
            self.cube_size = np.array([0.0515, 0.0515, 0.0515])
        if self.cube_initial_position is None:
            self.cube_initial_position = np.array([0.5, 0.0, RIDGEBACK_BASE_HEIGHT + 0.0258])
        if self.cube_initial_orientation is None:
            self.cube_initial_orientation = np.array([1, 0, 0, 0])
        if self.target_position is None:
            self.target_position = np.array([-0.3, -0.3, RIDGEBACK_BASE_HEIGHT + 0.12])
        if self.offset is None:
            self.offset = np.array([0.0, 0.0, 0.0])
        self.target_position = self.target_position + self.offset

        self.robot = RidgebackFrankaExperimental(robot_path="/World/robot", create_robot=True)
        self.end_effector_link = self.robot.end_effector_link

        stage_utils.add_reference_to_stage(
            usd_path=get_assets_root_path() + "/Isaac/Environments/Grid/default_environment.usd",
            path="/World/ground",
        )

        visual_material = PreviewSurfaceMaterial("/Visual_materials/blue")
        visual_material.set_input_values("diffuseColor", [0.0, 0.0, 1.0])

        cube_shape = Cube(
            paths="/World/Cube",
            positions=self.cube_initial_position,
            orientations=self.cube_initial_orientation,
            sizes=[1.0],
            scales=self.cube_size,
            reset_xform_op_properties=True,
        )

        GeomPrim(paths=cube_shape.paths, apply_collision_apis=True)
        self.cube = RigidPrim(paths=cube_shape.paths)
        cube_shape.apply_visual_materials(visual_material)

    def forward(self, ik_method: str = "damped-least-squares") -> bool:
        """Execute one step of the pick-and-place operation.

        Args:
            ik_method: The inverse kinematics method to use.

        Returns:
            True if a step was executed, False if the sequence is complete.
        """
        if self.is_done():
            return False

        goal_orientation = self.robot.get_downward_orientation()

        if self._event == 0:
            if self._step == 0:
                print("Phase 0: Moving to x,y position above cube...")
            cube_pos = self.cube.get_world_poses()[0].numpy()
            goal_position = np.array([cube_pos[0, 0], cube_pos[0, 1], cube_pos[0, 2] + 0.2])
            self.robot.set_end_effector_pose(position=goal_position, orientation=goal_orientation, ik_method=ik_method)
            self._step += 1
            if self._step >= self.events_dt[0]:
                self._event += 1
                self._step = 0

        elif self._event == 1:
            if self._step == 0:
                print("Phase 1: Approaching cube...")
            cube_pos = self.cube.get_world_poses()[0].numpy()
            goal_position = cube_pos + np.array([0.0, 0.0, 0.1])
            self.robot.set_end_effector_pose(position=goal_position, orientation=goal_orientation, ik_method=ik_method)
            self._step += 1
            if self._step >= self.events_dt[1]:
                self._event += 1
                self._step = 0

        elif self._event == 2:
            if self._step == 0:
                print("Phase 2: Closing gripper...")
            self.robot.close_gripper()
            self._step += 1
            if self._step >= self.events_dt[2]:
                self._event += 1
                self._step = 0

        elif self._event == 3:
            if self._step == 0:
                print("Phase 3: Lifting cube...")
            _, current_position, _ = self.robot.get_current_state()
            goal_position = current_position + np.array([0.0, 0.0, 0.2])
            self.robot.set_end_effector_pose(position=goal_position, orientation=goal_orientation, ik_method=ik_method)
            self._step += 1
            if self._step >= self.events_dt[3]:
                self._event += 1
                self._step = 0

        elif self._event == 4:
            if self._step == 0:
                print("Phase 4: Moving cube to target...")
            self.robot.set_end_effector_pose(
                position=self.target_position, orientation=goal_orientation, ik_method=ik_method
            )
            self._step += 1
            if self._step >= self.events_dt[4]:
                self._event += 1
                self._step = 0

        elif self._event == 5:
            if self._step == 0:
                print("Phase 5: Opening gripper...")
            self.robot.open_gripper()
            self._step += 1
            if self._step >= self.events_dt[5]:
                self._event += 1
                self._step = 0

        elif self._event == 6:
            if self._step == 0:
                print("Phase 6: Moving up...")
            cube_pos = self.cube.get_world_poses()[0].numpy()
            goal_position = cube_pos + np.array([0.0, 0.0, 0.3])
            self.robot.set_end_effector_pose(position=goal_position, orientation=goal_orientation, ik_method=ik_method)
            self._step += 1
            if self._step >= self.events_dt[6]:
                self._event += 1
                self._step = 0

        return True

    def is_done(self) -> bool:
        """Check if the pick-and-place sequence is complete."""
        return self._event >= len(self.events_dt)

    def reset(self, cube_position: Optional[np.ndarray] = None, cube_orientation: Optional[np.ndarray] = None):
        """Reset the entire pick-and-place system to initial state.

        Args:
            cube_position: Optional new position for the cube.
            cube_orientation: Optional new orientation for the cube.
        """
        self.reset_robot()
        self.reset_cube(position=cube_position, orientation=cube_orientation)

    def reset_robot(self):
        """Reset the robot to its default state."""
        if self.robot is not None:
            self.robot.reset_to_default_pose()
            self._event = 0
            self._step = 0

    def reset_cube(self, position: Optional[np.ndarray] = None, orientation: Optional[np.ndarray] = None):
        """Reset the cube to its initial position and orientation.

        Args:
            position: Optional new position for the cube.
            orientation: Optional new orientation for the cube.
        """
        if self.cube is not None:
            reset_position = position if position is not None else self.cube_initial_position
            reset_orientation = orientation if orientation is not None else self.cube_initial_orientation
            self.cube.set_world_poses(positions=reset_position, orientations=reset_orientation)


def main():
    print("Starting Ridgeback Franka Pick-and-Place Demo")
    SimulationManager.set_physics_sim_device(args.device)
    simulation_app.update()

    pick_place = RidgebackFrankaPickPlace()
    pick_place.setup_scene()

    omni.timeline.get_timeline_interface().play()
    simulation_app.update()

    reset_needed = True
    task_completed = False

    print("Starting pick-and-place execution")
    while simulation_app.is_running():
        if SimulationManager.is_simulating() and not task_completed:
            if reset_needed:
                pick_place.reset()
                reset_needed = False

            pick_place.forward(args.ik_method)

        if pick_place.is_done() and not task_completed:
            print("Done picking and placing with Ridgeback Franka")
            task_completed = True

        simulation_app.update()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {e}")
    finally:
        simulation_app.close()
