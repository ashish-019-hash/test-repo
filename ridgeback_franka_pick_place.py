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
from isaacsim.robot.wheeled_robots.controllers.holonomic_controller import (
    HolonomicController,
)
from isaacsim.storage.native import get_assets_root_path
import isaacsim.core.utils.stage as stage_utils

# ── Ridgeback mecanum wheel configuration ──────────────────────────────────
# The Clearpath Ridgeback has 4 mecanum wheels in a rectangular arrangement.
# Approximate parameters based on the Ridgeback specifications:
#   - Wheel radius: ~0.076 m (7.6 cm)
#   - Robot half-length (front-back): ~0.272 m
#   - Robot half-width (left-right): ~0.2745 m
RIDGEBACK_WHEEL_RADIUS = [0.076, 0.076, 0.076, 0.076]
RIDGEBACK_WHEEL_POSITIONS = [
    [0.272, -0.2745, 0.0],   # front-left
    [0.272, 0.2745, 0.0],    # front-right
    [-0.272, -0.2745, 0.0],  # rear-left
    [-0.272, 0.2745, 0.0],   # rear-right
]
RIDGEBACK_WHEEL_ORIENTATIONS = [
    [0, 0, 0, 1],   # front-left
    [0, 0, 0, 1],   # front-right
    [0, 0, 0, 1],   # rear-left
    [0, 0, 0, 1],   # rear-right
]
RIDGEBACK_MECANUM_ANGLES = [45.0, -45.0, -45.0, 45.0]  # degrees

# Ridgeback wheel joint names (typical for the ridgeback_franka.usd asset)
RIDGEBACK_WHEEL_JOINT_NAMES = [
    "front_left_wheel",
    "front_right_wheel",
    "rear_left_wheel",
    "rear_right_wheel",
]

# Franka arm joint names (7 DOF + 2 gripper fingers)
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

# ── Task configuration ─────────────────────────────────────────────────────
# Starting position for the Ridgeback (away from the table)
ROBOT_START_POSITION = np.array([-1.5, 0.0, 0.0])

# Table position and dimensions
TABLE_POSITION = np.array([0.5, 0.0, 0.25])
TABLE_SCALE = np.array([0.5, 0.8, 0.5])

# Cube (object to pick) initial position on the table
CUBE_POSITION = np.array([0.5, 0.0, 0.55])
CUBE_SIZE = 0.05

# Pick approach position for the base (close enough for the arm to reach)
PICK_BASE_POSITION = np.array([-0.2, 0.0])  # x, y for the base

# Place target position for the cube
PLACE_CUBE_POSITION = np.array([0.5, 0.8, 0.55])

# Place approach position for the base
PLACE_BASE_POSITION = np.array([-0.2, 0.8])  # x, y for the base

# Navigation tolerances
POSITION_TOLERANCE = 0.05  # meters
BASE_LINEAR_SPEED = 0.3    # m/s
BASE_ANGULAR_SPEED = 0.0   # rad/s (no rotation needed for straight-line moves)

# Franka arm home (stowed) joint positions
FRANKA_HOME_POSITIONS = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])

# Gripper open/close widths
GRIPPER_OPEN = 0.04    # meters (each finger)
GRIPPER_CLOSED = 0.0   # meters (each finger)

# Pre-grasp and grasp heights (relative to the Franka base on the Ridgeback)
PRE_GRASP_HEIGHT_OFFSET = 0.15  # approach from above
GRASP_HEIGHT_OFFSET = 0.0       # at the cube level


class RidgebackFrankaPickPlace:
    """
    Orchestrates a pick-and-place task using the Ridgeback Franka mobile manipulator.

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
        self._base_controller = None
        self._wheel_joint_indices = []
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
        """Reset the world, discover joint indices, and initialise controllers."""
        self._world.reset()

        # Discover joint indices for wheels, arm, and gripper
        self._wheel_joint_indices = []
        for name in RIDGEBACK_WHEEL_JOINT_NAMES:
            idx = self._robot.get_dof_index(name)
            self._wheel_joint_indices.append(idx)

        self._arm_joint_indices = []
        for name in FRANKA_ARM_JOINT_NAMES:
            idx = self._robot.get_dof_index(name)
            self._arm_joint_indices.append(idx)

        self._gripper_joint_indices = []
        for name in FRANKA_GRIPPER_JOINT_NAMES:
            idx = self._robot.get_dof_index(name)
            self._gripper_joint_indices.append(idx)

        # Create the holonomic base controller
        self._base_controller = HolonomicController(
            name="ridgeback_base_controller",
            wheel_radius=RIDGEBACK_WHEEL_RADIUS,
            wheel_positions=RIDGEBACK_WHEEL_POSITIONS,
            wheel_orientations=RIDGEBACK_WHEEL_ORIENTATIONS,
            mecanum_angles=RIDGEBACK_MECANUM_ANGLES,
        )

        # Move arm to home position and open gripper
        self._set_arm_positions(FRANKA_HOME_POSITIONS)
        self._set_gripper(GRIPPER_OPEN)

        self._state = self.NAVIGATE_TO_PICK
        self._wait_steps = 0

        print("[Reset] Robot initialised. Starting pick-and-place task.")
        print(
            f"[Reset] Wheel joints: {self._wheel_joint_indices}, "
            f"Arm joints: {self._arm_joint_indices}, "
            f"Gripper joints: {self._gripper_joint_indices}"
        )

    def forward(self):
        """Execute one step of the pick-and-place state machine."""
        # Handle wait states (used for gripper actions and settling)
        if self._wait_steps > 0:
            self._wait_steps -= 1
            return

        if self._state == self.NAVIGATE_TO_PICK:
            self._navigate_base(PICK_BASE_POSITION, next_state=self.PRE_GRASP)

        elif self._state == self.PRE_GRASP:
            self._stop_base()
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
            self._navigate_base(PLACE_BASE_POSITION, next_state=self.PRE_PLACE)

        elif self._state == self.PRE_PLACE:
            self._stop_base()
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

    # ── Private helpers ────────────────────────────────────────────────────

    def _transition(self, new_state):
        """Transition to a new state with logging."""
        print(
            f"[State] {self.STATE_NAMES[self._state]} -> "
            f"{self.STATE_NAMES[new_state]}"
        )
        self._state = new_state

    def _navigate_base(self, target_xy, next_state):
        """
        Drive the Ridgeback base toward target_xy (2D).
        Transitions to next_state once within tolerance.
        """
        robot_position, _ = self._robot.get_world_pose()
        current_xy = robot_position[:2]
        error = target_xy - current_xy
        distance = np.linalg.norm(error)

        if distance < POSITION_TOLERANCE:
            self._stop_base()
            self._transition(next_state)
            return

        # Compute velocity direction
        direction = error / distance
        vx = direction[0] * BASE_LINEAR_SPEED
        vy = direction[1] * BASE_LINEAR_SPEED
        command = [vx, vy, BASE_ANGULAR_SPEED]

        # Get wheel velocity commands from the holonomic controller
        actions = self._base_controller.forward(command)

        # Apply wheel velocities
        full_velocities = np.zeros(self._robot.num_dof)
        wheel_velocities = actions.joint_velocities
        for i, idx in enumerate(self._wheel_joint_indices):
            if i < len(wheel_velocities):
                full_velocities[idx] = wheel_velocities[i]

        action = ArticulationAction(joint_velocities=full_velocities)
        self._robot.apply_action(action)

    def _stop_base(self):
        """Stop all wheel joints."""
        full_velocities = np.zeros(self._robot.num_dof)
        for idx in self._wheel_joint_indices:
            full_velocities[idx] = 0.0
        action = ArticulationAction(joint_velocities=full_velocities)
        self._robot.apply_action(action)

    def _set_arm_positions(self, joint_positions):
        """Command the Franka arm joints to target positions."""
        full_positions = np.zeros(self._robot.num_dof)
        # Read current positions so we don't disturb other joints
        current_positions = self._robot.get_joint_positions()
        if current_positions is not None:
            full_positions[:] = current_positions

        for i, idx in enumerate(self._arm_joint_indices):
            if i < len(joint_positions):
                full_positions[idx] = joint_positions[i]

        action = ArticulationAction(joint_positions=full_positions)
        self._robot.apply_action(action)

    def _set_gripper(self, width):
        """Command the gripper fingers to a target width."""
        full_positions = np.zeros(self._robot.num_dof)
        current_positions = self._robot.get_joint_positions()
        if current_positions is not None:
            full_positions[:] = current_positions

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
