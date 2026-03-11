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

import omni.timeline
from isaacsim.core.simulation_manager import SimulationManager

import isaacsim.core.experimental.utils.stage as stage_utils
import numpy as np
from isaacsim.core.experimental.materials import PreviewSurfaceMaterial
from isaacsim.core.experimental.objects import Cube
from isaacsim.core.experimental.prims import GeomPrim, RigidPrim
from isaacsim.robot.manipulators.examples.franka import FrankaExperimental
from isaacsim.storage.native import get_assets_root_path

from typing import List, Optional


class RidgebackFrankaExperimental(FrankaExperimental):
    """Ridgeback Franka mobile manipulator controller.

    Extends FrankaExperimental to use the Ridgeback Franka USD asset
    (Clearpath Ridgeback mobile base + Franka Emika Panda arm).

    The Ridgeback Franka articulation has 12 DOFs:
      - 3 base DOFs (x, y, yaw) for the Ridgeback holonomic base
      - 7 arm DOFs (panda_joint1 .. panda_joint7) for the Franka arm
      - 2 gripper DOFs (panda_finger_joint1, panda_finger_joint2)

    This subclass overrides IK and gripper methods to account for the
    3 base DOFs that precede the arm joints.
    """

    # Number of mobile base DOFs that precede the arm joints
    BASE_DOF_COUNT = 3

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
            end_effector_link: The end effector rigid body link.
        """
        if create_robot:
            # Load Ridgeback Franka USD from Isaac Sim assets
            stage_utils.add_reference_to_stage(
                usd_path=get_assets_root_path()
                + "/Isaac/Robots/Clearpath/RidgebackFranka/ridgeback_franka.usd",
                path=robot_path,
            )

        # Initialize Articulation directly (skip FrankaExperimental's __init__
        # which would load the standard Franka USD and use 9-DOF defaults)
        from isaacsim.core.experimental.prims import Articulation

        Articulation.__init__(self, robot_path)

        # Set up end effector link
        if end_effector_link is None:
            self.end_effector_link = RigidPrim(f"{robot_path}/panda_hand")
        else:
            self.end_effector_link = end_effector_link

        if create_robot:
            # Default state: 12 DOFs = [3 base + 7 arm + 2 gripper]
            self.set_default_state(
                dof_positions=[
                    0.0, 0.0, 0.0,                                       # Ridgeback base (x, y, yaw)
                    0.012, -0.568, 0.0, -2.811, 0.0, 3.037, 0.741,       # Franka arm joints
                    0.04, 0.04,                                           # Gripper fingers (open)
                ]
            )

        self.end_effector_link_index = self.get_link_indices("panda_hand").list()[0]

        # Gripper positions
        self.gripper_open_position = np.array([[0.04, 0.04]])
        self.gripper_closed_position = np.array([[0.0, 0.0]])

    def set_end_effector_pose(self, position, orientation, ik_method="damped-least-squares"):
        """Set the end effector pose using IK on the 7 arm joints only.

        The Jacobian columns are sliced to extract only the arm DOFs
        (indices 3-9), skipping the 3 base DOFs.
        """
        current_dof_positions, current_ee_position, current_ee_orientation = (
            self.get_current_state()
        )

        if position.ndim == 1:
            position = position.reshape(1, -1)

        jacobian_matrices = self.get_jacobian_matrices().numpy()
        # Extract Jacobian columns for the 7 arm joints only (skip 3 base DOFs)
        jacobian_end_effector = jacobian_matrices[
            :, self.end_effector_link_index - 1, :, self.BASE_DOF_COUNT : self.BASE_DOF_COUNT + 7
        ]

        delta_dof_positions = self.differential_inverse_kinematics(
            jacobian_end_effector=jacobian_end_effector,
            current_position=current_ee_position,
            current_orientation=current_ee_orientation,
            goal_position=position,
            goal_orientation=orientation,
            method=ik_method,
        )

        # Apply IK delta to arm joints only (DOF indices 3 through 9)
        arm_start = self.BASE_DOF_COUNT
        arm_end = arm_start + 7
        dof_position_targets = current_dof_positions[:, arm_start:arm_end] + delta_dof_positions
        self.set_dof_position_targets(dof_position_targets, dof_indices=list(range(arm_start, arm_end)))

    def open_gripper(self):
        """Open the gripper (DOF indices 10, 11)."""
        finger_indices = [self.BASE_DOF_COUNT + 7, self.BASE_DOF_COUNT + 8]
        self.set_dof_position_targets(self.gripper_open_position, dof_indices=finger_indices)

    def close_gripper(self):
        """Close the gripper (DOF indices 10, 11)."""
        finger_indices = [self.BASE_DOF_COUNT + 7, self.BASE_DOF_COUNT + 8]
        self.set_dof_position_targets(self.gripper_closed_position, dof_indices=finger_indices)


class RidgebackFrankaPickPlace:
    """Pick-and-place controller for the Ridgeback Franka mobile manipulator.

    Adapts the standard FrankaPickPlace logic to use the Ridgeback Franka,
    which adds a Clearpath Ridgeback mobile base beneath the Franka arm.
    The mobile base remains stationary during pick-and-place; only the arm moves.
    """

    def __init__(self, events_dt: Optional[List[float]] = None):
        self.cube = None
        self.robot = None
        self.events_dt = events_dt
        if self.events_dt is None:
            self.events_dt = [
                60,  # Phase 0: Move to x,y position above cube
                40,  # Phase 1: Approach down to cube
                20,  # Phase 2: Close gripper to grasp
                40,  # Phase 3: Lift cube upward
                80,  # Phase 4: Move cube to target location
                20,  # Phase 5: Open gripper to release
                20,  # Phase 6: Move up and away
            ]
        self._event = 0
        self._step = 0

    def setup_scene(
        self,
        cube_initial_position=None,
        cube_initial_orientation=None,
        cube_size=None,
        target_position=None,
        offset=None,
    ):
        """Set up the scene with the Ridgeback Franka robot and a cube."""
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

        # Use RidgebackFrankaExperimental instead of FrankaExperimental
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

    def forward(self, ik_method="damped-least-squares"):
        """Execute one step of the pick-and-place operation."""
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

    def is_done(self):
        """Check if the pick-and-place sequence is complete."""
        return self._event >= len(self.events_dt)

    def reset(self, cube_position=None, cube_orientation=None):
        """Reset the pick-and-place system to initial state."""
        print("Resetting pick-and-place system...")
        self.reset_robot()
        self.reset_cube(position=cube_position, orientation=cube_orientation)
        print("Pick-and-place system reset complete")

    def reset_robot(self):
        """Reset the robot to its default state."""
        if self.robot is not None:
            # Reset all 12 DOFs: [3 base + 7 arm + 2 gripper]
            default_positions = np.array([[
                0.0, 0.0, 0.0,                                       # Ridgeback base
                0.012, -0.568, 0.0, -2.811, 0.0, 3.037, 0.741,       # Franka arm
                0.04, 0.04,                                           # Gripper (open)
            ]])
            self.robot.set_dof_positions(default_positions)
            self.robot.set_dof_position_targets(default_positions)
            self._event = 0
            self._step = 0
            print("Ridgeback Franka reset to default state")
        else:
            print("Warning: Ridgeback Franka controller not initialized, cannot reset")

    def reset_cube(self, position=None, orientation=None):
        """Reset the cube to its initial position and orientation."""
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
    print("Starting Simple Ridgeback Franka Pick-and-Place Demo")
    SimulationManager.set_physics_sim_device(args.device)
    simulation_app.update()

    pick_place = RidgebackFrankaPickPlace()
    pick_place.setup_scene()

    # Play the simulation.
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

            # Execute one step of the pick-and-place operation
            pick_place.forward(args.ik_method)

        if pick_place.is_done() and not task_completed:
            print("done picking and placing")
            task_completed = True

        simulation_app.update()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {e}")
    finally:
        simulation_app.close()
