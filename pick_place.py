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

ARM_JOINT_NAMES = [f"panda_joint{i}" for i in range(1, 8)]
FINGER_JOINT_NAMES = ["panda_finger_joint1", "panda_finger_joint2"]
BASE_JOINT_NAMES = ["dummy_base_prismatic_x_joint", "dummy_base_prismatic_y_joint", "dummy_base_revolute_z_joint"]
ARM_DEFAULT_POSITIONS = [0.012, -0.568, 0.0, -2.811, 0.0, 3.037, 0.741]
FINGER_OPEN_POSITIONS = [0.04, 0.04]
FINGER_CLOSED_POSITIONS = [0.0, 0.0]


class RidgebackFrankaExperimental(Articulation):
    """Ridgeback Franka robot controller with inverse kinematics and gripper control.

    This class inherits from Articulation and provides high-level control commands
    for the Ridgeback Franka robot (Clearpath Ridgeback mobile base with a Franka
    Emika Panda arm). It dynamically discovers DOF indices from joint names so it
    works regardless of the USD's internal joint ordering.
    """

    def __init__(
        self,
        robot_path: str = "/World/robot",
        create_robot: bool = True,
        end_effector_link: Optional[RigidPrim] = None,
    ):
        if create_robot:
            stage_utils.add_reference_to_stage(
                usd_path=get_assets_root_path() + "/Isaac/Robots/Clearpath/RidgebackFranka/ridgeback_franka.usd",
                path=robot_path,
            )

        super().__init__(robot_path)

        all_dof_names = self.dof_names
        num_dofs = self.num_dofs
        print(f"Articulation has {num_dofs} DOFs:")
        for i, name in enumerate(all_dof_names):
            print(f"  DOF {i}: {name}")

        all_link_names = self.link_names
        all_link_paths = self.link_paths[0]
        print(f"Articulation has {len(all_link_names)} links:")
        for i, (name, path) in enumerate(zip(all_link_names, all_link_paths)):
            print(f"  Link {i}: {name} -> {path}")

        panda_hand_path = None
        for name, path in zip(all_link_names, all_link_paths):
            if name == "panda_hand":
                panda_hand_path = path
                break
        if panda_hand_path is None:
            raise RuntimeError(f"Could not find 'panda_hand' link. Available links: {all_link_names}")
        print(f"End effector USD path: {panda_hand_path}")

        if end_effector_link is None:
            self.end_effector_link = RigidPrim(panda_hand_path)
        else:
            self.end_effector_link = end_effector_link

        self._arm_dof_indices = []
        for arm_name in ARM_JOINT_NAMES:
            for i, name in enumerate(all_dof_names):
                if name == arm_name:
                    self._arm_dof_indices.append(i)
                    break

        self._finger_dof_indices = []
        for finger_name in FINGER_JOINT_NAMES:
            for i, name in enumerate(all_dof_names):
                if name == finger_name:
                    self._finger_dof_indices.append(i)
                    break

        self._base_dof_indices = []
        for base_name in BASE_JOINT_NAMES:
            for i, name in enumerate(all_dof_names):
                if name == base_name:
                    self._base_dof_indices.append(i)
                    break

        print(f"Arm DOF indices: {self._arm_dof_indices}")
        print(f"Finger DOF indices: {self._finger_dof_indices}")
        print(f"Base DOF indices: {self._base_dof_indices}")

        if create_robot:
            default_positions = [0.0] * num_dofs
            for idx, pos in zip(self._arm_dof_indices, ARM_DEFAULT_POSITIONS):
                default_positions[idx] = pos
            for idx, pos in zip(self._finger_dof_indices, FINGER_OPEN_POSITIONS):
                default_positions[idx] = pos
            self.set_default_state(dof_positions=default_positions)
            self._default_positions = default_positions

        self.end_effector_link_index = self.get_link_indices("panda_hand").list()[0]
        print(f"End effector link index: {self.end_effector_link_index}")

        if self._base_dof_indices:
            base_stiffnesses = np.array([[1e6] * len(self._base_dof_indices)])
            base_dampings = np.array([[1e5] * len(self._base_dof_indices)])
            self.set_dof_gains(
                stiffnesses=base_stiffnesses,
                dampings=base_dampings,
                dof_indices=self._base_dof_indices,
            )
            print(f"Locked base joints with stiffness={1e6}, damping={1e5}")

        self.gripper_open_position = np.array([FINGER_OPEN_POSITIONS])
        self.gripper_closed_position = np.array([FINGER_CLOSED_POSITIONS])
        self._num_total_dofs = num_dofs
        self._jacobian_detected = False
        self._debug_step = 0

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
            pseudoinverse = np.swapaxes(Vh, 1, 2)[:, :, :6] @ np.diagflat(inv_s) @ np.swapaxes(U, 1, 2)
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
            return (scale * transpose @ np.linalg.inv(jacobian_end_effector @ transpose + lmbda) @ error).squeeze(-1)
        else:
            raise ValueError(f"Invalid IK method: {method}")

    def get_current_state(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        current_dof_positions = self.get_dof_positions().numpy()
        current_end_effector_position, current_end_effector_orientation = self.end_effector_link.get_world_poses()
        current_end_effector_position = current_end_effector_position.numpy()
        current_end_effector_orientation = current_end_effector_orientation.numpy()

        return current_dof_positions, current_end_effector_position, current_end_effector_orientation

    def _print_joint_gains(self) -> None:
        stiffnesses, dampings = self.get_dof_gains()
        stiffnesses_np = stiffnesses.numpy()[0]
        dampings_np = dampings.numpy()[0]
        all_dof_names = self.dof_names
        print("\nJoint drive gains:")
        for i, name in enumerate(all_dof_names):
            print(f"  DOF {i} ({name}): stiffness={stiffnesses_np[i]:.1f}, damping={dampings_np[i]:.1f}")
        return stiffnesses_np, dampings_np

    def _ensure_arm_gains(self) -> None:
        stiffnesses_np, dampings_np = self._print_joint_gains()
        arm_stiffness_values = [stiffnesses_np[i] for i in self._arm_dof_indices]
        if any(s < 1.0 for s in arm_stiffness_values):
            print("\nWARNING: Arm joint stiffnesses are too low! Setting appropriate gains.")
            arm_stiffnesses = np.array([[400.0, 400.0, 400.0, 400.0, 400.0, 400.0, 400.0]])
            arm_dampings = np.array([[80.0, 80.0, 80.0, 80.0, 80.0, 80.0, 80.0]])
            self.set_dof_gains(
                stiffnesses=arm_stiffnesses,
                dampings=arm_dampings,
                dof_indices=self._arm_dof_indices,
            )
            print("Set arm joint gains: stiffness=400, damping=80")
            self._print_joint_gains()

    def _find_jacobian_row(self, jacobian_matrices, current_ee_pos, current_ee_ori,
                           target_pos, target_ori, ik_method) -> int:
        n_rows = jacobian_matrices.shape[1]
        pos_error = target_pos - current_ee_pos
        pos_error_norm = np.linalg.norm(pos_error)
        if pos_error_norm < 1e-6:
            return self._jac_link_row
        pos_error_dir = pos_error / pos_error_norm

        print(f"\nSearching for correct Jacobian row...")
        print(f"  Current EE pos: {current_ee_pos[0]}")
        print(f"  Target pos:     {target_pos[0]}")
        print(f"  Pos error:      {pos_error[0]} (norm={pos_error_norm:.4f})")

        best_row = self._jac_link_row
        best_score = -np.inf

        for row in range(n_rows):
            j_arm = jacobian_matrices[:, row, :, self._jac_arm_cols]
            j_norm = np.linalg.norm(j_arm)
            if j_norm < 0.01:
                continue

            try:
                delta_q = self.differential_inverse_kinematics(
                    j_arm, current_ee_pos, current_ee_ori,
                    target_pos, target_ori, ik_method
                )
            except Exception:
                continue

            predicted_v = (j_arm[0] @ delta_q[0])[:3]
            pv_norm = np.linalg.norm(predicted_v)
            if pv_norm < 1e-8:
                continue

            alignment = np.dot(predicted_v / pv_norm, pos_error_dir[0])
            link_name = self.link_names[row] if row < len(self.link_names) else "?"
            print(f"  Row {row:2d} ({link_name:22s}): J_norm={j_norm:.3f}, align={alignment:+.4f}")

            if alignment > best_score:
                best_score = alignment
                best_row = row

        link_name = self.link_names[best_row] if best_row < len(self.link_names) else "?"
        print(f"  => Best row: {best_row} ({link_name}), alignment={best_score:+.4f}")
        orig_name = self.link_names[self._jac_link_row] if self._jac_link_row < len(self.link_names) else "?"
        print(f"  => Original row: {self._jac_link_row} ({orig_name})")
        return best_row

    def set_end_effector_pose(
        self,
        position: np.ndarray,
        orientation: np.ndarray,
        ik_method: str = "damped-least-squares",
    ) -> None:
        current_dof_positions, current_end_effector_position, current_end_effector_orientation = (
            self.get_current_state()
        )

        if position.ndim == 1:
            position = position.reshape(1, -1)

        jacobian_matrices = self.get_jacobian_matrices().numpy()

        if not self._jacobian_detected:
            self._ensure_arm_gains()

            jac_shape = jacobian_matrices.shape
            n_links = len(self.link_names)
            n_dofs = self._num_total_dofs
            print(f"\nJacobian shape: {jac_shape}")
            print(f"  num_links={n_links}, num_dofs={n_dofs}")
            if jac_shape[1] == n_links - 1 and jac_shape[3] == n_dofs:
                self._is_floating_base = False
                self._jac_link_row = self.end_effector_link_index - 1
                self._jac_arm_cols = self._arm_dof_indices
                print(f"  Detected FIXED base: jac_link_row={self._jac_link_row}, jac_arm_cols={self._jac_arm_cols}")
            elif jac_shape[1] == n_links and jac_shape[3] == n_dofs + 6:
                self._is_floating_base = True
                self._jac_link_row = self.end_effector_link_index
                self._jac_arm_cols = [idx + 6 for idx in self._arm_dof_indices]
                print(f"  Detected FLOATING base: jac_link_row={self._jac_link_row}, jac_arm_cols={self._jac_arm_cols}")
            else:
                print(f"  WARNING: Unexpected Jacobian shape. Trying fixed-base indexing.")
                self._is_floating_base = False
                self._jac_link_row = self.end_effector_link_index - 1
                self._jac_arm_cols = self._arm_dof_indices

            verified_row = self._find_jacobian_row(
                jacobian_matrices, current_end_effector_position,
                current_end_effector_orientation, position, orientation, ik_method
            )
            if verified_row != self._jac_link_row:
                print(f"  *** CORRECTING Jacobian row from {self._jac_link_row} to {verified_row} ***")
                self._jac_link_row = verified_row

            self._jacobian_detected = True

        jacobian_end_effector = jacobian_matrices[:, self._jac_link_row, :, :]
        jacobian_arm = jacobian_end_effector[:, :, self._jac_arm_cols]

        delta_dof_positions = self.differential_inverse_kinematics(
            jacobian_end_effector=jacobian_arm,
            current_position=current_end_effector_position,
            current_orientation=current_end_effector_orientation,
            goal_position=position,
            goal_orientation=orientation,
            method=ik_method,
        )

        max_delta = 0.1
        delta_dof_positions = np.clip(delta_dof_positions, -max_delta, max_delta)

        if self._debug_step < 3:
            print(f"\nIK step {self._debug_step}:")
            print(f"  EE pos:    {current_end_effector_position[0]}")
            print(f"  Target:    {position[0]}")
            print(f"  Pos error: {(position - current_end_effector_position)[0]}")
            print(f"  Delta q:   {delta_dof_positions[0]}")
            self._debug_step += 1

        current_arm_positions = current_dof_positions[:, self._arm_dof_indices]
        dof_position_targets = current_arm_positions + delta_dof_positions
        self.set_dof_position_targets(dof_position_targets, dof_indices=self._arm_dof_indices)

    def open_gripper(self) -> None:
        self.set_dof_position_targets(self.gripper_open_position, dof_indices=self._finger_dof_indices)

    def close_gripper(self) -> None:
        self.set_dof_position_targets(self.gripper_closed_position, dof_indices=self._finger_dof_indices)

    def set_gripper_position(self, position: np.ndarray) -> None:
        if position.ndim == 1:
            position = position.reshape(1, -1)
        self.set_dof_position_targets(position, dof_indices=self._finger_dof_indices)

    def get_downward_orientation(self) -> np.ndarray:
        return np.array([[0.0, 1.0, 0.0, 0.0]])

    def reset_to_default_pose(self) -> None:
        default_positions = np.array([self._default_positions])
        self.set_dof_positions(default_positions)
        self.set_dof_position_targets(default_positions)


class RidgebackFrankaPickPlace:
    """Simple, direct Ridgeback Franka pick-and-place controller.

    Adapts the Franka pick-and-place for the Ridgeback Franka mobile manipulator.
    The Ridgeback base remains stationary while the Franka arm performs the
    pick-and-place task using inverse kinematics.
    """

    def __init__(self, events_dt: Optional[List[float]] = None):
        self.cube = None
        self.robot = None

        self.events_dt = events_dt
        if self.events_dt is None:
            self.events_dt = [
                60,
                40,
                20,
                40,
                80,
                20,
                20,
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
        self.cube_initial_position = cube_initial_position
        self.cube_initial_orientation = cube_initial_orientation
        self.target_position = target_position
        self.cube_size = cube_size
        self.offset = offset

        if self.cube_size is None:
            self.cube_size = np.array([0.0515, 0.0515, 0.0515])
        if self.cube_initial_position is None:
            self.cube_initial_position = np.array([0.5, 0.0, 0.0258])
        if self.cube_initial_orientation is None:
            self.cube_initial_orientation = np.array([1, 0, 0, 0])
        if self.target_position is None:
            self.target_position = np.array([-0.3, -0.3, 0.12])
        if self.offset is None:
            self.offset = np.array([0.0, 0.0, 0.0])
        self.target_position = self.target_position + self.offset

        stage_utils.create_new_stage(template="sunlight")

        self.robot = RidgebackFrankaExperimental(robot_path="/World/robot", create_robot=True)
        self.end_effector_link = self.robot.end_effector_link

        ground_plane = stage_utils.add_reference_to_stage(
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
                print("Phase 4: Moving cube...")

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
        if self._event >= len(self.events_dt):
            return True
        else:
            return False

    def reset(self, cube_position: Optional[np.ndarray] = None, cube_orientation: Optional[np.ndarray] = None):
        print("Resetting pick-and-place system...")
        self.reset_robot()
        self.reset_cube(position=cube_position, orientation=cube_orientation)
        print("Pick-and-place system reset complete")

    def reset_robot(self):
        if self.robot is not None:
            self.robot.reset_to_default_pose()
            self._event = 0
            self._step = 0
            print("Robot reset to default state")
        else:
            print("Warning: Ridgeback Franka controller not initialized, cannot reset")

    def reset_cube(self, position: Optional[np.ndarray] = None, orientation: Optional[np.ndarray] = None):
        if self.cube is not None:
            reset_position = position if position is not None else self.cube_initial_position
            reset_orientation = orientation if orientation is not None else self.cube_initial_orientation

            self.cube.set_world_poses(
                positions=reset_position.reshape(1, -1), orientations=reset_orientation.reshape(1, -1)
            )

            print(f"Cube reset to position: {reset_position}")
        else:
            print("Warning: Cube not initialized, cannot reset")


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
            print("done picking and placing")
            task_completed = True

        simulation_app.update()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Error: {e}")
    finally:
        simulation_app.close()
