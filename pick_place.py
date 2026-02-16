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
args, _ = parser.parse_known_args()

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

from typing import List, Optional

import isaacsim.core.experimental.utils.stage as stage_utils
import numpy as np
import omni.timeline
from isaacsim.core.experimental.materials import PreviewSurfaceMaterial
from isaacsim.core.experimental.objects import Cube
from isaacsim.core.experimental.prims import Articulation, GeomPrim, RigidPrim
from isaacsim.core.simulation_manager import SimulationManager
from isaacsim.robot_motion.motion_generation.lula.kinematics import LulaKinematicsSolver
from isaacsim.storage.native import get_assets_root_path
import isaacsim.robot_motion.motion_generation.interface_config_loader as interface_config_loader

ARM_JOINT_NAMES = [f"panda_joint{i}" for i in range(1, 8)]
FINGER_JOINT_NAMES = ["panda_finger_joint1", "panda_finger_joint2"]
BASE_JOINT_KEYWORDS = ["dummy_base"]
ARM_DEFAULT_POSITIONS = [0.012, -0.568, 0.0, -2.811, 0.0, 3.037, 0.741]
FINGER_OPEN_POSITIONS = [0.04, 0.04]
FINGER_CLOSED_POSITIONS = [0.0, 0.0]
LULA_EE_FRAME = "panda_hand"


class RidgebackFrankaExperimental(Articulation):
    """Ridgeback Franka robot controller using Lula IK solver.

    Uses Isaac Sim's built-in Lula kinematics solver (URDF-based CCD+BFGS)
    instead of PhysX Jacobian-based differential IK. This avoids issues with
    the floating-base Jacobian column ordering that differs between PhysX
    internal tree-traversal order and the articulation's dof_names order.
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

        panda_hand_path = None
        arm_base_path = None
        for name, path in zip(all_link_names, all_link_paths):
            if name == "panda_hand":
                panda_hand_path = path
            if name == "panda_link0":
                arm_base_path = path

        if panda_hand_path is None:
            panda_hand_path = f"{robot_path}/panda_hand"
        if arm_base_path is None:
            arm_base_path = f"{robot_path}/panda_link0"

        if end_effector_link is None:
            self.end_effector_link = RigidPrim(panda_hand_path)
        else:
            self.end_effector_link = end_effector_link

        self._arm_base_link = RigidPrim(arm_base_path)

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
        for i, name in enumerate(all_dof_names):
            if any(kw in name for kw in BASE_JOINT_KEYWORDS):
                self._base_dof_indices.append(i)

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

        if self._base_dof_indices:
            base_stiffnesses = np.array([[1e6] * len(self._base_dof_indices)])
            base_dampings = np.array([[1e5] * len(self._base_dof_indices)])
            self.set_dof_gains(
                stiffnesses=base_stiffnesses,
                dampings=base_dampings,
                dof_indices=self._base_dof_indices,
            )
            print(f"Locked base joints with stiffness={1e6}, damping={1e5}")

        kinematics_config = interface_config_loader.load_supported_lula_kinematics_solver_config("Franka")
        self._lula_solver = LulaKinematicsSolver(**kinematics_config)
        lula_joint_names = self._lula_solver.get_joint_names()
        lula_frames = self._lula_solver.get_all_frame_names()
        print(f"Lula solver joints: {lula_joint_names}")
        print(f"Lula solver frames: {lula_frames}")
        self._base_pose_set = False

        self.gripper_open_position = np.array([FINGER_OPEN_POSITIONS])
        self.gripper_closed_position = np.array([FINGER_CLOSED_POSITIONS])
        self._num_total_dofs = num_dofs

    def _ensure_base_pose(self) -> None:
        if self._base_pose_set:
            return
        base_pos, base_orient = self._arm_base_link.get_world_poses()
        base_pos_np = base_pos.numpy()[0]
        base_orient_np = base_orient.numpy()[0]
        self._lula_solver.set_robot_base_pose(base_pos_np, base_orient_np)
        print(f"Lula robot base pose set: pos={base_pos_np}, orient={base_orient_np}")
        self._base_pose_set = True

    def get_current_state(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        current_dof_positions = self.get_dof_positions().numpy()
        current_end_effector_position, current_end_effector_orientation = self.end_effector_link.get_world_poses()
        current_end_effector_position = current_end_effector_position.numpy()
        current_end_effector_orientation = current_end_effector_orientation.numpy()
        return current_dof_positions, current_end_effector_position, current_end_effector_orientation

    def set_end_effector_pose(
        self,
        position: np.ndarray,
        orientation: np.ndarray = None,
    ) -> None:
        self._ensure_base_pose()

        all_dof_positions = self.get_dof_positions().numpy()
        arm_positions = all_dof_positions[0, self._arm_dof_indices]

        if position.ndim > 1:
            position = position.flatten()

        target_orientation = None
        if orientation is not None:
            if orientation.ndim > 1:
                orientation = orientation.flatten()
            target_orientation = orientation

        ik_result, success = self._lula_solver.compute_inverse_kinematics(
            frame_name=LULA_EE_FRAME,
            target_position=position,
            target_orientation=target_orientation,
            warm_start=arm_positions,
        )

        if success:
            targets = np.array(ik_result).reshape(1, -1)
            self.set_dof_position_targets(targets, dof_indices=self._arm_dof_indices)

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
        self._base_pose_set = False


class RidgebackFrankaPickPlace:
    """Simple, direct Ridgeback Franka pick-and-place controller.

    Adapts the Franka pick-and-place for the Ridgeback Franka mobile manipulator.
    The Ridgeback base remains stationary while the Franka arm performs the
    pick-and-place task using Lula inverse kinematics.
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

    def forward(self) -> bool:
        if self.is_done():
            return False

        goal_orientation = self.robot.get_downward_orientation()

        if self._event == 0:
            if self._step == 0:
                print("Phase 0: Moving to x,y position above cube...")

            cube_pos = self.cube.get_world_poses()[0].numpy()
            goal_position = np.array([cube_pos[0, 0], cube_pos[0, 1], cube_pos[0, 2] + 0.2])

            self.robot.set_end_effector_pose(position=goal_position, orientation=goal_orientation)

            self._step += 1
            if self._step >= self.events_dt[0]:
                self._event += 1
                self._step = 0

        elif self._event == 1:
            if self._step == 0:
                print("Phase 1: Approaching cube...")

            cube_pos = self.cube.get_world_poses()[0].numpy()
            goal_position = cube_pos + np.array([0.0, 0.0, 0.1])

            self.robot.set_end_effector_pose(position=goal_position, orientation=goal_orientation)

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

            self.robot.set_end_effector_pose(position=goal_position, orientation=goal_orientation)

            self._step += 1
            if self._step >= self.events_dt[3]:
                self._event += 1
                self._step = 0

        elif self._event == 4:
            if self._step == 0:
                print("Phase 4: Moving cube...")

            self.robot.set_end_effector_pose(
                position=self.target_position, orientation=goal_orientation
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

            self.robot.set_end_effector_pose(position=goal_position, orientation=goal_orientation)

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
    print("Starting Ridgeback Franka Pick-and-Place Demo (Lula IK)")
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

            pick_place.forward()

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
