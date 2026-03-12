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

Loads the actual Ridgeback Franka USD model from Isaac Sim assets and
performs a mobile pick-and-place task.  The Ridgeback base navigates to
a table, the Franka arm picks a cube, the base returns to the start,
and the cube is placed down.

The Ridgeback Franka USD has 12 DOFs:
  - panda_joint1..7               (7-DOF Franka arm)
  - panda_finger_joint1/2         (gripper)
  - dummy_base_prismatic_x_joint  (base X translation)
  - dummy_base_prismatic_y_joint  (base Y translation)
  - dummy_base_revolute_z_joint   (base yaw)

State machine:
  NAVIGATE_TO_PICK -> PRE_GRASP -> GRASP_APPROACH -> CLOSE_GRIPPER
    -> LIFT -> NAVIGATE_TO_PLACE -> PRE_PLACE -> OPEN_GRIPPER
    -> RETREAT -> DONE

Usage:
  python ridgeback_franka_pick_place.py [--device cpu|cuda] [--headless]
"""

from __future__ import annotations

import argparse

parser = argparse.ArgumentParser(
    description="Ridgeback Franka Pick-and-Place Demo"
)
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
import omni.usd
import omni.timeline
from isaacsim.core.api import World
from isaacsim.core.api.objects import DynamicCuboid, FixedCuboid
from isaacsim.core.api.robots import Robot
from isaacsim.core.simulation_manager import SimulationManager
from isaacsim.storage.native import get_assets_root_path
import isaacsim.core.utils.stage as stage_utils

# ---------------------------------------------------------------------------
#  Joint names (from ridgeback_franka.usd)
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
#  Scene configuration
# ---------------------------------------------------------------------------
# Robot starts at origin; table is placed 0.6m in front (+x)
ROBOT_START_POSITION = np.array([0.0, 0.0, 0.0])

# Table position and dimensions
TABLE_POSITION = np.array([0.65, 0.0, 0.25])
TABLE_SCALE = np.array([0.5, 0.8, 0.5])

# Cube (object to pick) on the table
CUBE_POSITION = np.array([0.5, 0.0, 0.55])
CUBE_SIZE = 0.05

# Place target position (to the side)
PLACE_POSITION = np.array([0.5, 0.6, 0.55])

# Base approach positions (dummy prismatic joint values)
# joint value = desired world X/Y since robot starts at origin
PICK_BASE_XY = np.array([0.0, 0.0])       # arm can reach table from origin
PLACE_BASE_XY = np.array([0.0, 0.6])      # slide sideways to place location

# Navigation parameters
POSITION_TOLERANCE = 0.02       # metres
BASE_MOVE_STEP_SIZE = 0.02      # metres per sim step

# Franka arm home (stowed) joint positions
FRANKA_HOME = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])

# Gripper open/close widths (per finger)
GRIPPER_OPEN = 0.04
GRIPPER_CLOSED = 0.001

# Arm poses for pick-and-place (joint angles for panda_joint1..7)
# These are tuned so the gripper reaches forward and down toward the table.
#   j1: base rotation
#   j2: shoulder lift (+ = forward tilt)
#   j3: elbow rotation
#   j4: elbow flex (more negative = more bent)
#   j5: forearm rotation
#   j6: wrist flex (higher = more wrist bend)
#   j7: wrist rotation
ARM_PRE_GRASP = np.array([0.0, 0.15, 0.0, -1.8, 0.0, 2.0, 0.785])
ARM_GRASP = np.array([0.0, 0.45, 0.0, -1.45, 0.0, 1.95, 0.785])
ARM_LIFT = np.array([0.0, -0.2, 0.0, -2.0, 0.0, 1.8, 0.785])
ARM_PRE_PLACE = np.array([0.0, 0.15, 0.0, -1.8, 0.0, 2.0, 0.785])
ARM_PLACE = np.array([0.0, 0.45, 0.0, -1.45, 0.0, 1.95, 0.785])

# Settle time (sim steps) after arm/gripper commands
ARM_SETTLE_STEPS = 60
GRIPPER_SETTLE_STEPS = 30


# ---------------------------------------------------------------------------
#  State machine
# ---------------------------------------------------------------------------
class RidgebackFrankaPickPlace:
    """Pick-and-place using the Ridgeback Franka mobile manipulator.

    Loads the actual Ridgeback Franka USD model.  The Ridgeback base is
    controlled via dummy prismatic joints.  The Franka arm is controlled
    via direct joint position commands.

    State machine:
      NAVIGATE_TO_PICK  -> Move base near pick location
      PRE_GRASP         -> Arm to pre-grasp above cube
      GRASP_APPROACH    -> Lower arm to grasp
      CLOSE_GRIPPER     -> Close gripper
      LIFT              -> Lift cube
      NAVIGATE_TO_PLACE -> Move base near place location
      PRE_PLACE         -> Arm to pre-place above target
      OPEN_GRIPPER      -> Open gripper to release
      RETREAT           -> Arm to home
      DONE              -> Complete
    """

    NAVIGATE_TO_PICK = 0
    PRE_GRASP = 1
    GRASP_APPROACH = 2
    CLOSE_GRIPPER = 3
    LIFT = 4
    NAVIGATE_TO_PLACE = 5
    PRE_PLACE = 6
    OPEN_GRIPPER = 7
    RETREAT = 8
    DONE = 9

    STATE_NAMES = [
        "NAVIGATE_TO_PICK",
        "PRE_GRASP",
        "GRASP_APPROACH",
        "CLOSE_GRIPPER",
        "LIFT",
        "NAVIGATE_TO_PLACE",
        "PRE_PLACE",
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
        """Build the simulation world: ground plane, table, cube, Ridgeback Franka."""
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

        # Place target visual indicator
        self._world.scene.add(
            FixedCuboid(
                prim_path="/World/PlaceTarget",
                name="place_target",
                position=PLACE_POSITION + np.array([0.0, 0.0, -0.025]),
                scale=np.array([CUBE_SIZE, CUBE_SIZE, 0.005]),
                color=np.array([0.0, 1.0, 0.0]),
            )
        )

        print("[Setup] Scene created with Ridgeback Franka, table, and cube.")

    def reset(self):
        """Reset the world, discover joint indices, and initialise the robot."""
        self._world.reset()

        # Print discovered DOFs
        print(f"[Reset] DOF names: {self._robot.dof_names}")
        print(f"[Reset] Num DOFs:  {self._robot.num_dof}")

        # Discover base joint indices
        self._base_x_idx = self._robot.get_dof_index("dummy_base_prismatic_x_joint")
        self._base_y_idx = self._robot.get_dof_index("dummy_base_prismatic_y_joint")
        self._base_yaw_idx = self._robot.get_dof_index("dummy_base_revolute_z_joint")

        # Discover arm joint indices
        self._arm_joint_indices = []
        for name in FRANKA_ARM_JOINT_NAMES:
            self._arm_joint_indices.append(self._robot.get_dof_index(name))

        self._gripper_joint_indices = []
        for name in FRANKA_GRIPPER_JOINT_NAMES:
            self._gripper_joint_indices.append(self._robot.get_dof_index(name))

        # Move arm to home and open gripper
        self._set_arm_positions(FRANKA_HOME)
        self._set_gripper(GRIPPER_OPEN)

        self._state = self.NAVIGATE_TO_PICK
        self._wait_steps = 0

        print("[Reset] Robot initialised.")
        print(f"[Reset] Base joints: X={self._base_x_idx}, "
              f"Y={self._base_y_idx}, Yaw={self._base_yaw_idx}")
        print(f"[Reset] Arm indices: {self._arm_joint_indices}")
        print(f"[Reset] Gripper indices: {self._gripper_joint_indices}")

    def forward(self):
        """Execute one step of the pick-and-place state machine."""
        # Handle wait states (settling)
        if self._wait_steps > 0:
            self._wait_steps -= 1
            return

        if self._state == self.NAVIGATE_TO_PICK:
            self._navigate_base(PICK_BASE_XY, next_state=self.PRE_GRASP)

        elif self._state == self.PRE_GRASP:
            self._set_arm_positions(ARM_PRE_GRASP)
            self._set_gripper(GRIPPER_OPEN)
            self._wait_steps = ARM_SETTLE_STEPS
            self._transition(self.GRASP_APPROACH)

        elif self._state == self.GRASP_APPROACH:
            self._set_arm_positions(ARM_GRASP)
            self._wait_steps = ARM_SETTLE_STEPS
            self._transition(self.CLOSE_GRIPPER)

        elif self._state == self.CLOSE_GRIPPER:
            self._set_gripper(GRIPPER_CLOSED)
            self._wait_steps = GRIPPER_SETTLE_STEPS
            self._transition(self.LIFT)

        elif self._state == self.LIFT:
            self._set_arm_positions(ARM_LIFT)
            self._wait_steps = ARM_SETTLE_STEPS
            self._transition(self.NAVIGATE_TO_PLACE)

        elif self._state == self.NAVIGATE_TO_PLACE:
            self._navigate_base(PLACE_BASE_XY, next_state=self.PRE_PLACE)

        elif self._state == self.PRE_PLACE:
            self._set_arm_positions(ARM_PRE_PLACE)
            self._wait_steps = ARM_SETTLE_STEPS
            self._transition(self.OPEN_GRIPPER)

        elif self._state == self.OPEN_GRIPPER:
            self._set_gripper(GRIPPER_OPEN)
            self._wait_steps = GRIPPER_SETTLE_STEPS
            self._transition(self.RETREAT)

        elif self._state == self.RETREAT:
            self._set_arm_positions(FRANKA_HOME)
            self._wait_steps = ARM_SETTLE_STEPS
            self._transition(self.DONE)

    def is_done(self):
        """Return True when the task is complete."""
        return self._state == self.DONE and self._wait_steps <= 0

    # -- Private helpers ---------------------------------------------------

    def _transition(self, new_state):
        """Transition to a new state with logging."""
        print(f"[State] {self.STATE_NAMES[self._state]} -> "
              f"{self.STATE_NAMES[new_state]}")
        self._state = new_state

    def _get_base_xy(self):
        """Read current base X, Y from the dummy prismatic joints."""
        joint_positions = self._robot.get_joint_positions()
        return np.array([
            joint_positions[self._base_x_idx],
            joint_positions[self._base_y_idx],
        ])

    def _navigate_base(self, target_xy, next_state):
        """Move the Ridgeback base toward target_xy by stepping the dummy
        prismatic joints.  Transitions to next_state once within tolerance."""
        current_xy = self._get_base_xy()
        error = target_xy - current_xy
        distance = np.linalg.norm(error)

        if distance < POSITION_TOLERANCE:
            print(f"[Nav] Arrived at ({current_xy[0]:.2f}, {current_xy[1]:.2f})")
            self._wait_steps = 30  # settle
            self._transition(next_state)
            return

        # Step toward target
        direction = error / distance
        step = direction * min(BASE_MOVE_STEP_SIZE, distance)
        new_xy = current_xy + step

        joint_positions = self._robot.get_joint_positions().copy()
        joint_positions[self._base_x_idx] = new_xy[0]
        joint_positions[self._base_y_idx] = new_xy[1]
        self._robot.set_joint_positions(joint_positions)

    def _set_arm_positions(self, arm_positions):
        """Set the 7 Franka arm joint positions without disturbing other joints."""
        joint_positions = self._robot.get_joint_positions().copy()
        for i, idx in enumerate(self._arm_joint_indices):
            joint_positions[idx] = arm_positions[i]
        self._robot.set_joint_positions(joint_positions)

    def _set_gripper(self, width):
        """Set gripper finger joint positions."""
        joint_positions = self._robot.get_joint_positions().copy()
        for idx in self._gripper_joint_indices:
            joint_positions[idx] = width
        self._robot.set_joint_positions(joint_positions)


# ---------------------------------------------------------------------------
#  Entry point
# ---------------------------------------------------------------------------
def main():
    """Entry point for the Ridgeback Franka pick-and-place demo."""
    print("=" * 60)
    print("  Ridgeback Franka Pick-and-Place Demo  (Isaac Sim 5.1.0)")
    print("=" * 60)

    SimulationManager.set_physics_sim_device(args.device)
    simulation_app.update()

    task = RidgebackFrankaPickPlace()
    task.setup_scene()
    simulation_app.update()

    # Start simulation
    timeline = omni.timeline.get_timeline_interface()
    timeline.play()
    simulation_app.update()

    # Reset
    task.reset()
    simulation_app.update()

    # Main loop
    print("[Main] Starting pick-and-place ...")
    while simulation_app.is_running():
        if not task.is_done():
            task.forward()
        else:
            print("=" * 60)
            print("  Pick-and-place task completed successfully!")
            print("=" * 60)
            break

        simulation_app.update()

    # Keep sim running after completion so user can inspect
    while simulation_app.is_running():
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
