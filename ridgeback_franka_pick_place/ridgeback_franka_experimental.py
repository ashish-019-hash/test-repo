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

from typing import Optional

import isaacsim.core.experimental.utils.stage as stage_utils
import numpy as np
import warp as wp
from isaacsim.core.experimental.prims import Articulation, RigidPrim
from isaacsim.core.experimental.utils.impl.transform import quaternion_conjugate, quaternion_multiplication
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
