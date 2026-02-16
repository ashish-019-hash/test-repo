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

from typing import List, Optional

import isaacsim.core.experimental.utils.stage as stage_utils
import numpy as np
from isaacsim.core.experimental.materials import PreviewSurfaceMaterial
from isaacsim.core.experimental.objects import Cube
from isaacsim.core.experimental.prims import GeomPrim, RigidPrim
from isaacsim.storage.native import get_assets_root_path

from ridgeback_franka_pick_place.ridgeback_franka_experimental import (
    RIDGEBACK_BASE_HEIGHT,
    RidgebackFrankaExperimental,
)


class RidgebackFrankaPickPlace:
    """Pick-and-place controller for the Ridgeback Franka mobile manipulator.

    Uses the Clearpath Ridgeback mobile base with a Franka Emika Panda arm
    mounted on top. The mobile base remains stationary while the arm performs
    the pick-and-place operation via differential inverse kinematics.
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
        print("Resetting Ridgeback Franka pick-and-place system...")
        self.reset_robot()
        self.reset_cube(position=cube_position, orientation=cube_orientation)
        print("Ridgeback Franka pick-and-place system reset complete")

    def reset_robot(self):
        """Reset the robot to its default state."""
        if self.robot is not None:
            self.robot.reset_to_default_pose()
            self._event = 0
            self._step = 0
            print("Ridgeback Franka reset to default state")
        else:
            print("Warning: Ridgeback Franka controller not initialized, cannot reset")

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
            print(f"Cube reset to position: {reset_position}")
        else:
            print("Warning: Cube not initialized, cannot reset")
