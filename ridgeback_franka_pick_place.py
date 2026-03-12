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

"""
Ridgeback Franka Pick-and-Place Demo for Isaac Sim 5.1.0

This script demonstrates a mobile manipulation pick-and-place task using the
Clearpath Ridgeback omnidirectional mobile base with a Franka Emika Panda arm
mounted on top.

The ridgeback_franka.usd asset uses dummy prismatic/revolute joints for the
base (not individual wheel joints):
  - dummy_base_prismatic_x_joint  (base X translation)
  - dummy_base_prismatic_y_joint  (base Y translation)
  - dummy_base_revolute_z_joint   (base yaw rotation)
  - panda_joint1..7               (Franka arm)
  - panda_finger_joint1/2         (gripper)

The workflow:
  1. Navigate the Ridgeback base to the pick location (near a table with a cube).
  2. Use the Franka arm to pick up the cube from the table.
  3. Navigate the Ridgeback base to the place location.
  4. Use the Franka arm to place the cube at the target position.

Usage:
  python ridgeback_franka_pick_place.py [--device cpu|cuda] [--headless]
"""

from __future__ import annotations

import argparse

parser = argparse.ArgumentParser(description="Ridgeback Franka Pick-and-Place Demo")
parser.add_argument(
    "--device",
    type=str,
    choices=["cpu", "cuda"],
    default="cpu",
    help="Simulation device",
)
parser.add_argument(
    "--headless",
    action="store_true",
    default=False,
    help="Run in headless mode (no GUI)",
)
args, _ = parser.parse_known_args()

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": args.headless})

import numpy as np
import omni.timeline
from isaacsim.core.api import World
from isaacsim.core.api.objects import DynamicCuboid, FixedCuboid
from isaacsim.core.api.robots import Robot
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.core.simulation_manager import SimulationManager
from isaacsim.storage.native import get_assets_root_path
import isaacsim.core.utils.stage as stage_utils

# -- Joint names (from the ridgeback_franka.usd asset) ----------------------
# The Ridgeback base is controlled via dummy prismatic/revolute joints
# (not individual wheel joints). This simplifies base navigation to direct
# position control on X, Y, and yaw.
#
# Full DOF list (12 total):
#   panda_joint2, panda_joint3, panda_joint1, panda_joint4, panda_joint5,
#   panda_joint6, dummy_base_revolute_z_joint, panda_joint7,
#   dummy_base_prismatic_y_joint, dummy_base_prismatic_x_joint,
#   panda_finger_joint1, panda_finger_joint2

# Franka arm joint names (7 DOF)
FRANKA_ARM_JOINT_NAMES = [
    "panda_joint1",
    "panda_joint2",
    "panda_joint3",
    "panda_joint4",
    "panda_joint5",
    "panda_joint6",
    "panda_joint7",
]
FRANKA_GRIPPER_JOINT_NAMES = [
    "panda_finger_joint1",
    "panda_finger_joint2",
]

# -- Task configuration -----------------------------------------------------
# Starting position for the Ridgeback (away from the table)
ROBOT_START_POSITION = np.array([-1.5, 0.0, 0.0])

# Table position and dimensions
TABLE_POSITION = np.array([0.5, 0.0, 0.25])
TABLE_SCALE = np.array([0.5, 0.8, 0.5])

# Cube (object to pick) initial position on the table
CUBE_POSITION = np.array([0.5, 0.0, 0.55])
CUBE_SIZE = 0.05

# Pick approach position for the base (close enough for the arm to reach)
PICK_BASE_XY = np.array([-0.2, 0.0])  # x, y for the base

# Place target position for the cube
PLACE_CUBE_POSITION = np.array([0.5, 0.8, 0.55])

# Place approach position for the base
PLACE_BASE_XY = np.array([-0.2, 0.8])  # x, y for the base

# Navigation parameters
POSITION_TOLERANCE = 0.05       # meters -- close enough to target
BASE_MOVE_STEP_SIZE = 0.005     # meters per sim step for smooth base motion

# Franka arm home (stowed) joint positions
FRANKA_HOME_POSITIONS = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])

# Gripper open/close widths
GRIPPER_OPEN = 0.04    # meters (each finger)
GRIPPER_CLOSED = 0.0   # meters (each finger)


class RidgebackFrankaPickPlace:
    """
    Orchestrates a pick-and-place task using the Ridgeback Franka mobile manipulator.

    The Ridgeback base is controlled by setting positions on the dummy prismatic
    joints (X/Y) and the dummy revolute joint (yaw). The Franka arm and gripper
    are controlled via joint position commands.

    State machine:
      NAVIGATE_TO_PICK  -> Move the base near the pick location
      PRE_GRASP         -> Move arm to pre-grasp pose above the cube
      GRASP_APPROACH    -> Lower the arm to the grasp pose
      CLOSE_GRIPPER     -> Close the gripper to grasp the cube
      LIFT              -> Lift the cube
      NAVIGATE_TO_PLACE -> Move the base near the place location
      PRE_PLACE         -> Move arm to pre-place pose above the target
      PLACE_APPROACH    -> Lower the arm to place the cube
      OPEN_GRIPPER      -> Open the gripper to release the cube
      RETREAT           -> Lift the arm back up
      DONE              -> Task complete
    """

    NAVIGATE_TO_PICK = 0
    PRE_GRASP = 1
    GRASP_APPROACH = 2
    CLOSE_GRIPPER = 3
    LIFT = 4
    NAVIGATE_TO_PLACE = 5
    PRE_PLACE = 6
    PLACE_APPROACH = 7
    OPEN_GRIPPER = 8
    RETREAT = 9
    DONE = 10

    STATE_NAMES = [
        "NAVIGATE_TO_PICK",
        "PRE_GRASP",
        "GRASP_APPROACH",
        "CLOSE_GRIPPER",
        "LIFT",
        "NAVIGATE_TO_PLACE",
        "PRE_PLACE",
        "PLACE_APPROACH",
        "OPEN_GRIPPER",
        "RETREAT",
        "DONE",
    ]

    def __init__(self):
        self._state = self.NAVIGATE_TO_PICK
        self._robot = None
        self._cube = None
        self._world = None
        self._base_x_idx = -1
        self._base_y_idx = -1
        self._base_yaw_idx = -1
        self._arm_joint_indices = []
        self._gripper_joint_indices = []
        self._wait_steps = 0

    def setup_scene(self):
        """Build the simulation world: ground plane, table, cube, and robot."""
        if World.instance():
            World.instance().clear_instance()

        self._world = World()
        self._world.initialize_physics()

        # Ground plane
        self._world.scene.add_default_ground_plane()

        # Load the Ridgeback Franka USD from Isaac Sim assets
        assets_root_path = get_assets_root_path()
        ridgeback_franka_path = (
            assets_root_path
            + "/Isaac/Robots/Clearpath/RidgebackFranka/ridgeback_franka.usd"
        )
        robot_prim_path = "/World/RidgebackFranka"
        stage_utils.add_reference_to_stage(ridgeback_franka_path, robot_prim_path)

        self._robot = self._world.scene.add(
            Robot(
                prim_path=robot_prim_path,
                name="ridgeback_franka",
                position=ROBOT_START_POSITION,
            )
        )

        # Table (static box)
        self._world.scene.add(
            FixedCuboid(
                prim_path="/World/Table",
                name="table",
                position=TABLE_POSITION,
                scale=TABLE_SCALE,
                color=np.array([0.5, 0.3, 0.1]),
            )
        )

        # Cube to pick up (dynamic)
        self._cube = self._world.scene.add(
            DynamicCuboid(
                prim_path="/World/Cube",
                name="pick_cube",
                position=CUBE_POSITION,
                size=CUBE_SIZE,
                color=np.array([1.0, 0.0, 0.0]),
                mass=0.1,
            )
        )

        # Place target visual indicator (static, semi-transparent)
        self._world.scene.add(
            FixedCuboid(
                prim_path="/World/PlaceTarget",
                name="place_target",
                position=PLACE_CUBE_POSITION,
                scale=np.array([CUBE_SIZE, CUBE_SIZE, 0.005]),
                color=np.array([0.0, 1.0, 0.0]),
            )
        )

        print("[Setup] Scene created with Ridgeback Franka, table, and cube.")

    def reset(self):
        """Reset the world, discover joint indices, and initialise the robot."""
        self._world.reset()

        # Print discovered DOF names for debugging
        print(f"[Reset] DOF names: {self._robot.dof_names}")
        print(f"[Reset] Num DOFs:  {self._robot.num_dof}")

        # Discover base joint indices (dummy prismatic/revolute)
        self._base_x_idx = self._robot.get_dof_index(
            "dummy_base_prismatic_x_joint"
        )
        self._base_y_idx = self._robot.get_dof_index(
            "dummy_base_prismatic_y_joint"
        )
        self._base_yaw_idx = self._robot.get_dof_index(
            "dummy_base_revolute_z_joint"
        )

        # Discover arm joint indices
        self._arm_joint_indices = []
        for name in FRANKA_ARM_JOINT_NAMES:
            idx = self._robot.get_dof_index(name)
            self._arm_joint_indices.append(idx)

        self._gripper_joint_indices = []
        for name in FRANKA_GRIPPER_JOINT_NAMES:
            idx = self._robot.get_dof_index(name)
            self._gripper_joint_indices.append(idx)

        # Move arm to home position and open gripper
        self._set_arm_positions(FRANKA_HOME_POSITIONS)
        self._set_gripper(GRIPPER_OPEN)

        self._state = self.NAVIGATE_TO_PICK
        self._wait_steps = 0

        print("[Reset] Robot initialised. Starting pick-and-place task.")
        print(
            f"[Reset] Base joints: x={self._base_x_idx}, "
            f"y={self._base_y_idx}, yaw={self._base_yaw_idx}"
        )
        print(f"[Reset] Arm joints: {self._arm_joint_indices}")
        print(f"[Reset] Gripper joints: {self._gripper_joint_indices}")

    def forward(self):
        """Execute one step of the pick-and-place state machine."""
        # Handle wait states (used for gripper actions and settling)
        if self._wait_steps > 0:
            self._wait_steps -= 1
            return

        if self._state == self.NAVIGATE_TO_PICK:
            self._navigate_base(PICK_BASE_XY, next_state=self.PRE_GRASP)

        elif self._state == self.PRE_GRASP:
            # Move arm to pre-grasp pose above the cube
            pre_grasp_positions = np.array(
                [0.0, -0.4, 0.0, -1.8, 0.0, 1.4, 0.785]
            )
            self._set_arm_positions(pre_grasp_positions)
            self._set_gripper(GRIPPER_OPEN)
            self._wait_steps = 60  # wait for arm to settle
            self._transition(self.GRASP_APPROACH)

        elif self._state == self.GRASP_APPROACH:
            # Lower arm to grasp position
            grasp_positions = np.array(
                [0.0, 0.0, 0.0, -1.5, 0.0, 1.5, 0.785]
            )
            self._set_arm_positions(grasp_positions)
            self._wait_steps = 60
            self._transition(self.CLOSE_GRIPPER)

        elif self._state == self.CLOSE_GRIPPER:
            self._set_gripper(GRIPPER_CLOSED)
            self._wait_steps = 30
            self._transition(self.LIFT)

        elif self._state == self.LIFT:
            # Lift the cube by returning to pre-grasp pose
            lift_positions = np.array(
                [0.0, -0.4, 0.0, -1.8, 0.0, 1.4, 0.785]
            )
            self._set_arm_positions(lift_positions)
            self._wait_steps = 60
            self._transition(self.NAVIGATE_TO_PLACE)

        elif self._state == self.NAVIGATE_TO_PLACE:
            self._navigate_base(PLACE_BASE_XY, next_state=self.PRE_PLACE)

        elif self._state == self.PRE_PLACE:
            # Move arm to pre-place pose
            pre_place_positions = np.array(
                [0.0, -0.4, 0.0, -1.8, 0.0, 1.4, 0.785]
            )
            self._set_arm_positions(pre_place_positions)
            self._wait_steps = 60
            self._transition(self.PLACE_APPROACH)

        elif self._state == self.PLACE_APPROACH:
            # Lower arm to place position
            place_positions = np.array(
                [0.0, 0.0, 0.0, -1.5, 0.0, 1.5, 0.785]
            )
            self._set_arm_positions(place_positions)
            self._wait_steps = 60
            self._transition(self.OPEN_GRIPPER)

        elif self._state == self.OPEN_GRIPPER:
            self._set_gripper(GRIPPER_OPEN)
            self._wait_steps = 30
            self._transition(self.RETREAT)

        elif self._state == self.RETREAT:
            # Return arm to home position
            self._set_arm_positions(FRANKA_HOME_POSITIONS)
            self._wait_steps = 60
            self._transition(self.DONE)

    def is_done(self):
        """Return True when the task is complete."""
        return self._state == self.DONE and self._wait_steps <= 0

    # -- Private helpers ----------------------------------------------------

    def _transition(self, new_state):
        """Transition to a new state with logging."""
        print(
            f"[State] {self.STATE_NAMES[self._state]} -> "
            f"{self.STATE_NAMES[new_state]}"
        )
        self._state = new_state

    def _get_base_xy(self):
        """Read current base X, Y position from the prismatic joint values."""
        joint_positions = self._robot.get_joint_positions()
        base_x = joint_positions[self._base_x_idx]
        base_y = joint_positions[self._base_y_idx]
        return np.array([base_x, base_y])

    def _navigate_base(self, target_xy, next_state):
        """
        Move the Ridgeback base toward target_xy (2D) by incrementally
        setting the dummy prismatic joint positions each step.
        Transitions to next_state once within tolerance.
        """
        current_xy = self._get_base_xy()
        error = target_xy - current_xy
        distance = np.linalg.norm(error)

        if distance < POSITION_TOLERANCE:
            self._transition(next_state)
            return

        # Move a small step toward the target for smooth motion
        direction = error / distance
        step = direction * min(BASE_MOVE_STEP_SIZE, distance)
        new_xy = current_xy + step

        # Build a full joint position array, preserving all current positions
        full_positions = self._robot.get_joint_positions().copy()
        full_positions[self._base_x_idx] = new_xy[0]
        full_positions[self._base_y_idx] = new_xy[1]

        action = ArticulationAction(joint_positions=full_positions)
        self._robot.apply_action(action)

    def _set_arm_positions(self, joint_positions):
        """Command the Franka arm joints to target positions."""
        full_positions = self._robot.get_joint_positions()
        if full_positions is None:
            return
        full_positions = full_positions.copy()

        for i, idx in enumerate(self._arm_joint_indices):
            if i < len(joint_positions):
                full_positions[idx] = joint_positions[i]

        action = ArticulationAction(joint_positions=full_positions)
        self._robot.apply_action(action)

    def _set_gripper(self, width):
        """Command the gripper fingers to a target width."""
        full_positions = self._robot.get_joint_positions()
        if full_positions is None:
            return
        full_positions = full_positions.copy()

        for idx in self._gripper_joint_indices:
            full_positions[idx] = width

        action = ArticulationAction(joint_positions=full_positions)
        self._robot.apply_action(action)


def main():
    """Entry point for the Ridgeback Franka pick-and-place demo."""
    print("=" * 60)
    print("  Ridgeback Franka Pick-and-Place Demo  (Isaac Sim 5.1.0)")
    print("=" * 60)

    SimulationManager.set_physics_sim_device(args.device)
    simulation_app.update()

    task = RidgebackFrankaPickPlace()
    task.setup_scene()

    # Start simulation
    omni.timeline.get_timeline_interface().play()
    simulation_app.update()

    reset_needed = True
    task_completed = False

    print("[Main] Starting pick-and-place execution ...")
    while simulation_app.is_running():
        if SimulationManager.is_simulating() and not task_completed:
            if reset_needed:
                task.reset()
                reset_needed = False

            task.forward()

        if task.is_done() and not task_completed:
            print("=" * 60)
            print("  Pick-and-place task completed successfully!")
            print("=" * 60)
            task_completed = True

        simulation_app.update()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        simulation_app.close()
