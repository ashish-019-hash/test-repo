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


# ---------------------------------------------------------------------------
# Ridgeback Franka workspace parameters
# ---------------------------------------------------------------------------
# The Clearpath Ridgeback mobile base elevates the Franka arm base ~0.5 m
# above the ground plane.  The standard Franka demo places the cube on the
# ground (z ~ 0.026), but with the elevated base that position is near the
# edge of the arm's downward workspace.  The damped-least-squares IK then
# produces near-zero joint deltas and the arm appears frozen.
#
# Fix: add a table and raise the cube / target into the comfortable zone.
#
# Workspace derivation (Ridgeback + Franka kinematic chain):
#   Ridgeback body height    ~0.278 m
#   Franka mounting offset   ~0.05 m
#   Franka link0 to shoulder ~0.333 m
#   Total arm-base height    ~0.66 m
#   Franka max reach          0.855 m
#   Comfortable working zone  arm-base +/- 0.35 m  =>  ~0.3 - 1.0 m
# ---------------------------------------------------------------------------
TABLE_HEIGHT = 0.4                     # 40 cm tall table
TABLE_HALF_HEIGHT = TABLE_HEIGHT / 2.0
CUBE_HALF_SIZE = 0.0515 / 2.0         # half of 5.15 cm cube

# Franka arm default joint positions (7 joints)
FRANKA_ARM_DEFAULT = [0.012, -0.568, 0.0, -2.811, 0.0, 3.037, 0.741]
# Franka finger default positions (2 fingers, open)
FRANKA_FINGER_DEFAULT = [0.04, 0.04]


class RidgebackFrankaExperimental(FrankaExperimental):
    """Ridgeback Franka mobile manipulator controller.

    Extends FrankaExperimental to use the Ridgeback Franka USD asset
    (Clearpath Ridgeback mobile base + Franka Emika Panda arm).

    DOF indices are auto-detected by matching joint names (panda_joint*,
    panda_finger_joint*) so the code works regardless of how many base
    DOFs the Ridgeback USD exposes or what order they appear in.
    """

    def __init__(
        self,
        robot_path: str = "/World/robot",
        create_robot: bool = True,
        end_effector_link: Optional[RigidPrim] = None,
    ):
        if create_robot:
            stage_utils.add_reference_to_stage(
                usd_path=get_assets_root_path()
                + "/Isaac/Robots/Clearpath/RidgebackFranka/ridgeback_franka.usd",
                path=robot_path,
            )

        # Initialize Articulation directly (skip FrankaExperimental.__init__
        # which would load the standard Franka USD and hardcode 9-DOF defaults)
        from isaacsim.core.experimental.prims import Articulation

        Articulation.__init__(self, robot_path)

        # --- Auto-detect DOF indices by name --------------------------------
        dof_names = self.get_dof_names()
        self._arm_dof_indices = []
        self._finger_dof_indices = []
        for i, name in enumerate(dof_names):
            if "panda_joint" in name and "finger" not in name:
                self._arm_dof_indices.append(i)
            elif "panda_finger" in name:
                self._finger_dof_indices.append(i)

        print(f"[RidgebackFranka] Total DOFs: {len(dof_names)}")
        print(f"[RidgebackFranka] DOF names: {dof_names}")
        print(f"[RidgebackFranka] Arm DOF indices:    {self._arm_dof_indices} "
              f"(count={len(self._arm_dof_indices)})")
        print(f"[RidgebackFranka] Finger DOF indices: {self._finger_dof_indices} "
              f"(count={len(self._finger_dof_indices)})")

        if len(self._arm_dof_indices) != 7:
            print(f"[RidgebackFranka] WARNING: expected 7 arm DOFs, "
                  f"found {len(self._arm_dof_indices)}")
        if len(self._finger_dof_indices) != 2:
            print(f"[RidgebackFranka] WARNING: expected 2 finger DOFs, "
                  f"found {len(self._finger_dof_indices)}")

        # --- End-effector link ----------------------------------------------
        if end_effector_link is None:
            ee_path = self._find_prim_path(robot_path, "panda_hand")
            self.end_effector_link = RigidPrim(ee_path)
        else:
            self.end_effector_link = end_effector_link

        # --- Default state (dynamically built) ------------------------------
        if create_robot:
            default_positions = [0.0] * len(dof_names)
            for i, idx in enumerate(self._arm_dof_indices):
                if i < len(FRANKA_ARM_DEFAULT):
                    default_positions[idx] = FRANKA_ARM_DEFAULT[i]
            for i, idx in enumerate(self._finger_dof_indices):
                if i < len(FRANKA_FINGER_DEFAULT):
                    default_positions[idx] = FRANKA_FINGER_DEFAULT[i]
            self.set_default_state(dof_positions=default_positions)
            print(f"[RidgebackFranka] Default positions: {default_positions}")

        self.end_effector_link_index = self.get_link_indices("panda_hand").list()[0]
        print(f"[RidgebackFranka] EE link index: {self.end_effector_link_index}")

        # Gripper targets
        self.gripper_open_position = np.array([[0.04, 0.04]])
        self.gripper_closed_position = np.array([[0.0, 0.0]])

        # Ensure PD drives exist on arm and finger joints
        if create_robot:
            self._configure_arm_drives(robot_path)

        # IK step counter for diagnostics
        self._ik_debug_count = 0

    @staticmethod
    def _find_prim_path(robot_path, link_name):
        """Find the full USD prim path for a named link under robot_path.

        The Ridgeback Franka USD may nest panda links under sub-prims
        (e.g., /World/robot/panda/panda_hand) rather than directly under
        the robot root (/World/robot/panda_hand). This method searches
        the USD hierarchy to find the actual path.
        """
        import omni.usd
        from pxr import Usd

        stage = omni.usd.get_context().get_stage()

        # Try direct path first (works for standard Franka USD)
        direct_path = f"{robot_path}/{link_name}"
        prim = stage.GetPrimAtPath(direct_path)
        if prim.IsValid():
            print(f"[RidgebackFranka] Found {link_name} at direct path: {direct_path}")
            return direct_path

        # Search recursively through the robot's USD hierarchy
        print(f"[RidgebackFranka] {link_name} not at {direct_path}, searching hierarchy...")
        robot_prim = stage.GetPrimAtPath(robot_path)
        if robot_prim.IsValid():
            for descendant in Usd.PrimRange(robot_prim):
                if descendant.GetName() == link_name:
                    found_path = str(descendant.GetPath())
                    print(f"[RidgebackFranka] Found {link_name} at: {found_path}")
                    return found_path

        print(f"[RidgebackFranka] WARNING: {link_name} not found, using: {direct_path}")
        return direct_path

    @staticmethod
    def _configure_arm_drives(robot_path):
        """Ensure arm and finger joints have position drives with adequate stiffness.

        The standard Franka USD ships with position drives (stiffness ~400 Nm/rad).
        The Ridgeback Franka USD may differ.  Without proper drives,
        set_dof_position_targets() has no effect and the arm will not move.
        """
        try:
            import omni.usd
            from pxr import Usd, UsdPhysics

            stage = omni.usd.get_context().get_stage()
            robot_prim = stage.GetPrimAtPath(robot_path)
            if not robot_prim.IsValid():
                return

            arm_joint_names = {f"panda_joint{i}" for i in range(1, 8)}
            finger_joint_names = {"panda_finger_joint1", "panda_finger_joint2"}
            target_joints = arm_joint_names | finger_joint_names

            for prim in Usd.PrimRange(robot_prim):
                name = prim.GetName()
                if name not in target_joints:
                    continue

                # angular for revolute arm joints, linear for prismatic fingers
                drive_type = "linear" if name in finger_joint_names else "angular"

                # Apply DriveAPI if missing
                if not UsdPhysics.DriveAPI.Get(prim, drive_type):
                    UsdPhysics.DriveAPI.Apply(prim, drive_type)
                    print(f"[RidgebackFranka] Applied {drive_type} drive to {name}")

                drive = UsdPhysics.DriveAPI.Get(prim, drive_type)
                stiffness = drive.GetStiffnessAttr().Get()
                damping = drive.GetDampingAttr().Get()

                # Only override if stiffness is missing or zero
                if stiffness is None or stiffness == 0:
                    if name in finger_joint_names:
                        drive.GetStiffnessAttr().Set(1e4)
                        drive.GetDampingAttr().Set(1e2)
                    else:
                        drive.GetStiffnessAttr().Set(400.0)
                        drive.GetDampingAttr().Set(40.0)
                    print(f"[RidgebackFranka] Configured drive for {name}: "
                          f"stiffness={drive.GetStiffnessAttr().Get()}, "
                          f"damping={drive.GetDampingAttr().Get()}")
                else:
                    print(f"[RidgebackFranka] {name} drive OK: "
                          f"stiffness={stiffness}, damping={damping}")

        except Exception as e:
            print(f"[RidgebackFranka] Could not configure drives: {e}")

    def set_end_effector_pose(self, position, orientation, ik_method="damped-least-squares"):
        """Move the end effector toward *position* / *orientation* via IK.

        Uses auto-detected arm DOF indices for Jacobian slicing so the
        code works regardless of the DOF ordering in the USD.
        """
        current_dof_positions, current_ee_position, current_ee_orientation = (
            self.get_current_state()
        )

        if position.ndim == 1:
            position = position.reshape(1, -1)

        jacobian_matrices = self.get_jacobian_matrices().numpy()

        # Slice Jacobian columns for the 7 arm DOFs only (auto-detected indices)
        arm_idx = self._arm_dof_indices
        jacobian_end_effector = jacobian_matrices[
            :, self.end_effector_link_index - 1, :, arm_idx
        ]

        delta_dof_positions = self.differential_inverse_kinematics(
            jacobian_end_effector=jacobian_end_effector,
            current_position=current_ee_position,
            current_orientation=current_ee_orientation,
            goal_position=position,
            goal_orientation=orientation,
            method=ik_method,
        )

        # Apply IK delta to arm joints only
        dof_position_targets = current_dof_positions[:, arm_idx] + delta_dof_positions
        self.set_dof_position_targets(dof_position_targets, dof_indices=arm_idx)

        # Diagnostics (first 3 IK steps)
        if self._ik_debug_count < 3:
            n = self._ik_debug_count
            print(f"[IK {n}] Jacobian shape={jacobian_matrices.shape}  "
                  f"EE link idx={self.end_effector_link_index}  arm_idx={arm_idx}")
            print(f"[IK {n}] EE pos={current_ee_position}  Goal={position}")
            print(f"[IK {n}] |J_ee|={np.abs(jacobian_end_effector).max():.6f}  "
                  f"|delta|={np.abs(delta_dof_positions).max():.6f}")
            print(f"[IK {n}] arm targets={dof_position_targets}")
            self._ik_debug_count += 1

    def open_gripper(self):
        """Open the gripper using auto-detected finger DOF indices."""
        self.set_dof_position_targets(
            self.gripper_open_position, dof_indices=self._finger_dof_indices
        )

    def close_gripper(self):
        """Close the gripper using auto-detected finger DOF indices."""
        self.set_dof_position_targets(
            self.gripper_closed_position, dof_indices=self._finger_dof_indices
        )


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
        """Set up the scene with the Ridgeback Franka, a table, and a cube.

        The table provides a surface at TABLE_HEIGHT so the cube is within the
        arm's comfortable workspace (the Ridgeback elevates the arm ~0.5 m).
        """
        self.cube_initial_position = cube_initial_position
        self.cube_initial_orientation = cube_initial_orientation
        self.target_position = target_position
        self.cube_size = cube_size
        self.offset = offset

        if self.cube_size is None:
            self.cube_size = np.array([0.0515, 0.0515, 0.0515])
        if self.cube_initial_position is None:
            # Cube sits on the table surface (table top = TABLE_HEIGHT)
            self.cube_initial_position = np.array([0.5, 0.0, TABLE_HEIGHT + CUBE_HALF_SIZE])
        if self.cube_initial_orientation is None:
            self.cube_initial_orientation = np.array([1, 0, 0, 0])
        if self.target_position is None:
            # Target: on the other side, also at table height
            self.target_position = np.array([-0.3, -0.3, TABLE_HEIGHT + 0.12])
        if self.offset is None:
            self.offset = np.array([0.0, 0.0, 0.0])
        self.target_position = self.target_position + self.offset

        stage_utils.create_new_stage(template="sunlight")

        # Robot
        self.robot = RidgebackFrankaExperimental(
            robot_path="/World/robot", create_robot=True
        )
        self.end_effector_link = self.robot.end_effector_link

        # Ground plane
        stage_utils.add_reference_to_stage(
            usd_path=get_assets_root_path()
            + "/Isaac/Environments/Grid/default_environment.usd",
            path="/World/ground",
        )

        # --- Table ---
        # A static box that gives the cube a surface to rest on.
        # 60 cm x 80 cm x TABLE_HEIGHT, centred 30 cm in front of the robot.
        table_material = PreviewSurfaceMaterial("/Visual_materials/wood")
        table_material.set_input_values("diffuseColor", [0.55, 0.35, 0.17])

        table_shape = Cube(
            paths="/World/Table",
            positions=np.array([0.3, 0.0, TABLE_HALF_HEIGHT]),
            orientations=np.array([1, 0, 0, 0]),
            sizes=[1.0],
            scales=np.array([0.6, 0.8, TABLE_HEIGHT]),
            reset_xform_op_properties=True,
        )
        GeomPrim(paths=table_shape.paths, apply_collision_apis=True)
        table_shape.apply_visual_materials(table_material)
        print(f"[Scene] Table added (top surface at z={TABLE_HEIGHT})")

        # --- Cube ---
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

        print(f"[Scene] Cube at {self.cube_initial_position}")
        print(f"[Scene] Target at {self.target_position}")

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
        """Reset all DOFs to their default values.

        Builds the default array dynamically from auto-detected indices
        so it works regardless of the DOF count or ordering.
        """
        if self.robot is None:
            print("WARNING: robot not initialised")
            return

        # Build default array dynamically from auto-detected indices
        num_dofs = len(self.robot.get_dof_names())
        defaults = np.zeros((1, num_dofs))
        for i, idx in enumerate(self.robot._arm_dof_indices):
            if i < len(FRANKA_ARM_DEFAULT):
                defaults[0, idx] = FRANKA_ARM_DEFAULT[i]
        for i, idx in enumerate(self.robot._finger_dof_indices):
            if i < len(FRANKA_FINGER_DEFAULT):
                defaults[0, idx] = FRANKA_FINGER_DEFAULT[i]

        self.robot.set_dof_positions(defaults)
        self.robot.set_dof_position_targets(defaults)
        self.robot._ik_debug_count = 0
        self._event = 0
        self._step = 0
        print(f"Robot reset ({num_dofs} DOFs)")

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
