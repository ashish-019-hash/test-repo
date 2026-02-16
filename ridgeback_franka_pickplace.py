# SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import numpy as np

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
from omni.isaac.core import World
from omni.isaac.core.objects import DynamicCuboid
from omni.isaac.core.robots import Robot
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.utils.types import ArticulationAction
from omni.isaac.core.utils.prims import get_prim_at_path
from isaacsim.core.simulation_manager import SimulationManager
from pxr import UsdGeom, Usd
import carb
import omni.usd


def find_end_effector_path(base_path):
    """Recursively search for end effector prim"""
    stage = omni.usd.get_context().get_stage()
    
    # Common end effector names to search for
    possible_names = ["panda_hand", "panda_link8", "panda_rightfinger", "hand", "end_effector", "ee_link"]
    
    def search_recursive(prim_path, depth=0, max_depth=10):
        if depth > max_depth:
            return None
            
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            return None
        
        # Check if this prim matches any end effector names
        prim_name = prim.GetName()
        for possible_name in possible_names:
            if possible_name in prim_name.lower():
                print(f"Found potential end effector: {prim.GetPath()}")
                return str(prim.GetPath())
        
        # Search children
        for child in prim.GetChildren():
            result = search_recursive(str(child.GetPath()), depth + 1, max_depth)
            if result:
                return result
                
        return None
    
    return search_recursive(base_path)


def print_prim_tree(prim_path, depth=0, max_depth=5):
    """Print USD prim hierarchy for debugging"""
    if depth > max_depth:
        return
        
    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath(prim_path)
    
    if not prim.IsValid():
        return
    
    indent = "  " * depth
    print(f"{indent}{prim.GetName()} ({prim.GetTypeName()})")
    
    for child in prim.GetChildren():
        print_prim_tree(str(child.GetPath()), depth + 1, max_depth)


class RidgebackFrankaPickPlace:
    def __init__(self):
        self._world = World(stage_units_in_meters=1.0)
        self._robot = None
        self._cube = None
        self._current_step = 0
        self._state = "moving_to_pick"
        self._gripper_dof_indices = None
        self._arm_dof_indices = None
        self._end_effector_prim_path = None
        self._ee_offset = np.array([0.0, 0.0, 0.0])
        
        # Pick and place waypoints
        self._waypoints = []
        self._current_waypoint = 0
        
    def setup_scene(self):
        # Add ground plane
        self._world.scene.add_default_ground_plane()
        
        # Get assets root path
        assets_root_path = get_assets_root_path()
        if assets_root_path is None:
            carb.log_error("Could not find Isaac Sim assets folder")
            return
            
        # Load Ridgeback Franka USD
        usd_path = f"{assets_root_path}/Isaac/Robots/Clearpath/RidgebackFranka/ridgeback_franka.usd"
        
        print(f"Loading robot from: {usd_path}")
        add_reference_to_stage(usd_path=usd_path, prim_path="/World/RidgebackFranka")
        
        # Use generic Robot class
        self._robot = self._world.scene.add(
            Robot(
                prim_path="/World/RidgebackFranka",
                name="ridgeback_franka",
            )
        )
        
        # Print the prim hierarchy to find the correct end effector path
        print("\n=== Robot Prim Hierarchy ===")
        print_prim_tree("/World/RidgebackFranka", max_depth=4)
        print("=" * 50)
        
        # Try to find end effector automatically
        self._end_effector_prim_path = find_end_effector_path("/World/RidgebackFranka")
        
        if not self._end_effector_prim_path:
            # Fallback to common paths
            possible_paths = [
                "/World/RidgebackFranka/panda_hand",
                "/World/RidgebackFranka/panda/panda_hand",
                "/World/RidgebackFranka/panda/panda_link8",
                "/World/RidgebackFranka/franka/panda_hand",
            ]
            for path in possible_paths:
                prim = get_prim_at_path(path)
                if prim and prim.IsValid():
                    self._end_effector_prim_path = path
                    print(f"Found end effector at: {path}")
                    break
        
        if self._end_effector_prim_path:
            print(f"\nUsing end effector: {self._end_effector_prim_path}")
        else:
            print("\nWARNING: Could not find end effector prim!")
        
        # Add cube to pick - positioned relative to robot
        self._cube = self._world.scene.add(
            DynamicCuboid(
                prim_path="/World/Cube",
                name="cube",
                position=np.array([0.6, 0.0, 0.025]),
                scale=np.array([0.05, 0.05, 0.05]),
                color=np.array([0.0, 0.5, 1.0]),
            )
        )
        
        # Define pick and place waypoints (world coordinates)
        cube_pos = np.array([0.6, 0.0, 0.025])
        
        # Approach position (above cube)
        approach_offset = np.array([0.0, 0.0, 0.12])
        # Pick position (lowered so fingers can wrap around the cube)
        pick_offset = np.array([0.0, 0.0, 0.04])
        # Lift position (lift cube up)
        lift_offset = np.array([0.0, 0.0, 0.25])
        # Place position (lowered to match pick height)
        place_pos = np.array([0.3, 0.4, 0.04])
        # Place approach
        place_approach = place_pos + np.array([0.0, 0.0, 0.12])
        
        self._waypoints = [
            ("approach", cube_pos + approach_offset),
            ("pick", cube_pos + pick_offset),
            ("lift", cube_pos + lift_offset),
            ("place_approach", place_approach),
            ("place", place_pos),
        ]
        
    def reset(self):
        self._world.reset()
        self._current_step = 0
        self._current_waypoint = 0
        self._state = "moving_to_pick"
        
        # Reset cube position
        if self._cube:
            self._cube.set_world_pose(position=np.array([0.6, 0.0, 0.025]))
        
        # Find DOF indices on first reset
        if self._robot and self._gripper_dof_indices is None:
            dof_names = self._robot.dof_names
            print(f"\nAvailable DOF names ({len(dof_names)} total):")
            for i, name in enumerate(dof_names):
                print(f"  [{i}] {name}")
            
            # Find arm DOFs (Franka/Panda joints) - should be joints 1-7
            self._arm_dof_indices = []
            for i, name in enumerate(dof_names):
                # Look specifically for panda_joint1 through panda_joint7
                if 'panda_joint' in name.lower() and 'finger' not in name.lower():
                    # Extract joint number
                    try:
                        joint_num = int(name.lower().split('panda_joint')[1].split('_')[0])
                        if 1 <= joint_num <= 7:
                            self._arm_dof_indices.append(i)
                    except:
                        pass
            
            # Sort arm DOF indices to ensure they're in order
            self._arm_dof_indices = sorted(self._arm_dof_indices)
                    
            # Find gripper DOFs
            self._gripper_dof_indices = []
            for i, name in enumerate(dof_names):
                if 'finger' in name.lower():
                    self._gripper_dof_indices.append(i)
                    
            print(f"\nArm DOF indices: {self._arm_dof_indices}")
            print(f"Gripper DOF indices: {self._gripper_dof_indices}")
            
            # Verify we have the right number of arm DOFs
            if len(self._arm_dof_indices) != 7:
                print(f"WARNING: Expected 7 arm DOFs, found {len(self._arm_dof_indices)}")
            
            # Set initial arm configuration (home position)
            if len(self._arm_dof_indices) >= 7:
                home_joints = [0.0, 0.8, 0.0, -1.4, 0.0, 3.74, 0.785]
                self._set_arm_joints(home_joints)
            
            # Open gripper initially
            self._open_gripper()
            
    def get_end_effector_position(self):
        """Get end effector position in world coordinates"""
        if not self._end_effector_prim_path:
            return None
            
        stage = omni.usd.get_context().get_stage()
        ee_prim = stage.GetPrimAtPath(self._end_effector_prim_path)
        
        if not ee_prim or not ee_prim.IsValid():
            return None
            
        xformable = UsdGeom.Xformable(ee_prim)
        world_transform = xformable.ComputeLocalToWorldTransform(0)
        translation = world_transform.ExtractTranslation()
        
        ee_position = np.array([translation[0], translation[1], translation[2]])
        return ee_position + self._ee_offset
            
    def forward(self, ik_method):
        if not self._robot:
            return
        
        # Get end effector pose
        ee_position = self.get_end_effector_position()
        
        if ee_position is None:
            # Fallback: just use timer-based states
            self._simple_forward()
            return
        
        # State machine for pick and place
        if self._state == "moving_to_pick":
            if self._current_waypoint < 2:  # Approach and pick waypoints
                waypoint_name, target_pos = self._waypoints[self._current_waypoint]
                
                # Move towards target
                distance = np.linalg.norm(ee_position - target_pos)
                
                if distance > 0.01:
                    self._move_to_target(ee_position, target_pos, ik_method)
                else:
                    print(f"Reached waypoint: {waypoint_name} (distance: {distance:.4f}m)")
                    self._current_waypoint += 1
                    self._current_step = 0
                    
                    # Close gripper when at pick position
                    if waypoint_name == "pick":
                        self._state = "closing_gripper"
                        
        elif self._state == "closing_gripper":
            self._close_gripper()
            self._current_step += 1
            
            if self._current_step > 30:
                self._state = "lifting"
                self._current_step = 0
                print("Gripper closed, lifting...")
                
        elif self._state == "lifting":
            waypoint_name, target_pos = self._waypoints[2]  # Lift waypoint
            distance = np.linalg.norm(ee_position - target_pos)
            
            if distance > 0.01:
                self._move_to_target(ee_position, target_pos, ik_method)
            else:
                print(f"Reached waypoint: {waypoint_name} (distance: {distance:.4f}m)")
                self._current_waypoint = 3
                self._state = "moving_to_place"
                
        elif self._state == "moving_to_place":
            if self._current_waypoint < len(self._waypoints):
                waypoint_name, target_pos = self._waypoints[self._current_waypoint]
                distance = np.linalg.norm(ee_position - target_pos)
                
                if distance > 0.01:
                    self._move_to_target(ee_position, target_pos, ik_method)
                else:
                    print(f"Reached waypoint: {waypoint_name} (distance: {distance:.4f}m)")
                    self._current_waypoint += 1
                    self._current_step = 0
                    
                    # Open gripper at place position
                    if waypoint_name == "place":
                        self._state = "opening_gripper"
                        
        elif self._state == "opening_gripper":
            self._open_gripper()
            self._current_step += 1
            
            if self._current_step > 30:
                self._state = "done"
                print("Task completed!")
    
    def _simple_forward(self):
        """Fallback method without end effector tracking"""
        if self._state == "moving_to_pick":
            if len(self._arm_dof_indices) >= 7:
                target_joints = [0.0, 1.0, 0.0, -1.5, 0.0, 3.64, 0.785]
                self._set_arm_joints(target_joints)
                
            self._current_step += 1
            
            if self._current_step > 150:
                self._state = "closing_gripper"
                self._current_step = 0
                print("Reached pick position")
                
        elif self._state == "closing_gripper":
            self._close_gripper()
            self._current_step += 1
            
            if self._current_step > 40:
                self._state = "lifting"
                self._current_step = 0
                print("Gripper closed")
                
        elif self._state == "lifting":
            if len(self._arm_dof_indices) >= 7:
                target_joints = [0.0, 0.9, 0.0, -1.5, 0.0, 3.74, 0.785]
                self._set_arm_joints(target_joints)
                
            self._current_step += 1
            
            if self._current_step > 80:
                self._state = "moving_to_place"
                self._current_step = 0
                print("Lifted object")
                
        elif self._state == "moving_to_place":
            if len(self._arm_dof_indices) >= 7:
                target_joints = [0.785, 0.9, 0.0, -1.5, 0.0, 3.74, 0.785]
                self._set_arm_joints(target_joints)
                
            self._current_step += 1
            
            if self._current_step > 150:
                self._state = "opening_gripper"
                self._current_step = 0
                print("Reached place position")
                
        elif self._state == "opening_gripper":
            self._open_gripper()
            self._current_step += 1
            
            if self._current_step > 40:
                self._state = "done"
                print("Released object")
                
    def _move_to_target(self, current_pos, target_pos, ik_method):
        """Move end effector towards target using numerical Jacobian with null-space orientation bias"""
        if not self._robot or not self._arm_dof_indices or len(self._arm_dof_indices) != 7:
            return
            
        # Compute position error
        position_error = target_pos - current_pos
        
        # Proportional control with higher gain for better convergence
        max_velocity = 0.3  # m/s
        velocity_command = np.clip(position_error * 5.0, -max_velocity, max_velocity)
        
        # Compute numerical Jacobian
        current_joint_positions = self._robot.get_joint_positions()
        jacobian = self._compute_numerical_jacobian(current_joint_positions)
        
        if jacobian is not None:
            # Compute joint velocities using specified IK method
            if ik_method == "pseudoinverse":
                J_pinv = np.linalg.pinv(jacobian)
                joint_velocities = J_pinv @ velocity_command
            elif ik_method == "transpose":
                J_pinv = jacobian.T
                joint_velocities = J_pinv @ velocity_command
            elif ik_method == "damped-least-squares":
                lambda_damping = 0.05
                J_pinv = jacobian.T @ np.linalg.inv(jacobian @ jacobian.T + lambda_damping**2 * np.eye(3))
                joint_velocities = J_pinv @ velocity_command
            else:  # singular-value-decomposition
                U, s, Vh = np.linalg.svd(jacobian)
                s_inv = np.array([1.0/si if si > 0.001 else 0.0 for si in s])
                J_pinv = Vh.T @ np.diag(s_inv) @ U.T
                joint_velocities = J_pinv @ velocity_command
            
            # Null-space projection to bias gripper orientation downward
            # For Franka Panda, gripper faces down when J2 + J4 + J6 ~ pi
            desired_config = np.array([0.0, 1.0, 0.0, -1.5, 0.0, 3.64, 0.785])
            current_arm = np.array([current_joint_positions[idx] for idx in self._arm_dof_indices[:7]])
            config_error = desired_config - current_arm
            
            null_space_gain = 2.0
            if ik_method != "transpose":
                N = np.eye(7) - np.linalg.pinv(jacobian) @ jacobian
            else:
                N = np.eye(7) - jacobian.T @ np.linalg.pinv(jacobian.T)
            null_space_velocity = N @ (config_error * null_space_gain)
            joint_velocities = joint_velocities + null_space_velocity
            
            # Integrate velocities to get target positions
            dt = 1.0 / 60.0  # Assuming 60 Hz
            target_joint_positions = current_joint_positions.copy()
            
            # Franka Panda joint limits
            joint_limits_lower = [-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973]
            joint_limits_upper = [2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973]
            
            for i, idx in enumerate(self._arm_dof_indices[:7]):
                new_pos = current_joint_positions[idx] + joint_velocities[i] * dt
                target_joint_positions[idx] = np.clip(new_pos, joint_limits_lower[i], joint_limits_upper[i])
            
            # Apply action
            action = ArticulationAction(joint_positions=target_joint_positions)
            self._robot.apply_action(action)
    
    def _compute_numerical_jacobian(self, current_positions):
        """Compute numerical Jacobian using finite differences"""
        epsilon = 1e-4
        ee_pos_0 = self.get_end_effector_position()
        
        if ee_pos_0 is None:
            return None
            
        jacobian = np.zeros((3, 7))
        
        for i, idx in enumerate(self._arm_dof_indices[:7]):
            # Perturb joint i
            perturbed_positions = current_positions.copy()
            perturbed_positions[idx] += epsilon
            
            # Set perturbed positions
            action = ArticulationAction(joint_positions=perturbed_positions)
            self._robot.apply_action(action)
            
            # Wait for update
            self._world.step(render=False)
            
            # Get new end effector position
            ee_pos_perturbed = self.get_end_effector_position()
            
            if ee_pos_perturbed is not None:
                # Compute finite difference
                jacobian[:, i] = (ee_pos_perturbed - ee_pos_0) / epsilon
            
            # Restore original positions
            action = ArticulationAction(joint_positions=current_positions)
            self._robot.apply_action(action)
            
        return jacobian
                
    def _set_arm_joints(self, target_positions):
        """Set arm joint positions directly"""
        if self._robot and self._arm_dof_indices and len(self._arm_dof_indices) >= len(target_positions):
            current_positions = self._robot.get_joint_positions()
            for i, idx in enumerate(self._arm_dof_indices[:len(target_positions)]):
                current_positions[idx] = target_positions[i]
            action = ArticulationAction(joint_positions=current_positions)
            self._robot.apply_action(action)
                
    def _close_gripper(self):
        """Close the gripper"""
        if self._robot and self._gripper_dof_indices:
            current_positions = self._robot.get_joint_positions()
            for idx in self._gripper_dof_indices:
                current_positions[idx] = 0.0  # Closed position
            action = ArticulationAction(joint_positions=current_positions)
            self._robot.apply_action(action)
            
    def _open_gripper(self):
        """Open the gripper"""
        if self._robot and self._gripper_dof_indices:
            current_positions = self._robot.get_joint_positions()
            for idx in self._gripper_dof_indices:
                current_positions[idx] = 0.04  # Open position
            action = ArticulationAction(joint_positions=current_positions)
            self._robot.apply_action(action)
    
    def is_done(self):
        return self._state == "done"


def main():
    print("Starting Ridgeback Franka Pick-and-Place Demo")
    SimulationManager.set_physics_sim_device(args.device)
    simulation_app.update()

    pick_place = RidgebackFrankaPickPlace()
    pick_place.setup_scene()
    
    # Wait for scene to be ready
    for _ in range(10):
        simulation_app.update()

    # Play the simulation
    omni.timeline.get_timeline_interface().play()
    
    # Wait for physics to stabilize
    for _ in range(20):
        simulation_app.update()

    reset_needed = True
    task_completed = False

    print("\nStarting pick-and-place execution")
    while simulation_app.is_running():
        if SimulationManager.is_simulating() and not task_completed:
            if reset_needed:
                pick_place.reset()
                reset_needed = False

            # Execute one step of the pick-and-place operation
            pick_place.forward(args.ik_method)

        if pick_place.is_done() and not task_completed:
            print("\n=== TASK COMPLETED SUCCESSFULLY ===")
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
