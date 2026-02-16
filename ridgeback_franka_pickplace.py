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
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.utils.types import ArticulationAction
from omni.isaac.core.utils.prims import get_prim_at_path
from isaacsim.core.simulation_manager import SimulationManager
from pxr import UsdGeom
import carb
import omni.usd

try:
    from omni.isaac.motion_generation import (
        ArticulationMotionPolicy,
        RmpFlow,
        interface_config_loader,
    )
    RMP_AVAILABLE = True
except ImportError:
    try:
        from isaacsim.robot.motion_generation import (
            ArticulationMotionPolicy,
            RmpFlow,
            interface_config_loader,
        )
        RMP_AVAILABLE = True
    except ImportError:
        RMP_AVAILABLE = False
        print("WARNING: motion_generation not available, using fallback joint control")


def find_end_effector_path(base_path):
    stage = omni.usd.get_context().get_stage()
    possible_names = ["panda_hand", "panda_link8", "panda_rightfinger", "hand", "end_effector", "ee_link"]

    def search_recursive(prim_path, depth=0, max_depth=10):
        if depth > max_depth:
            return None
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            return None
        prim_name = prim.GetName()
        for possible_name in possible_names:
            if possible_name in prim_name.lower():
                print(f"Found potential end effector: {prim.GetPath()}")
                return str(prim.GetPath())
        for child in prim.GetChildren():
            result = search_recursive(str(child.GetPath()), depth + 1, max_depth)
            if result:
                return result
        return None

    return search_recursive(base_path)


def print_prim_tree(prim_path, depth=0, max_depth=5):
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
        self._rmpflow = None
        self._art_motion_policy = None
        self._rmp_initialized = False
        self._waypoints = []
        self._current_waypoint = 0
        self._gripper_down_orientation = np.array([0.0, 1.0, 0.0, 0.0])

    def setup_scene(self):
        self._world.scene.add_default_ground_plane()
        assets_root_path = get_assets_root_path()
        if assets_root_path is None:
            carb.log_error("Could not find Isaac Sim assets folder")
            return

        usd_path = f"{assets_root_path}/Isaac/Robots/Clearpath/RidgebackFranka/ridgeback_franka.usd"
        print(f"Loading robot from: {usd_path}")
        add_reference_to_stage(usd_path=usd_path, prim_path="/World/RidgebackFranka")

        from omni.isaac.core.robots import Robot
        self._robot = self._world.scene.add(
            Robot(prim_path="/World/RidgebackFranka", name="ridgeback_franka")
        )

        print("\n=== Robot Prim Hierarchy ===")
        print_prim_tree("/World/RidgebackFranka", max_depth=4)
        print("=" * 50)

        self._end_effector_prim_path = find_end_effector_path("/World/RidgebackFranka")
        if not self._end_effector_prim_path:
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

        cube_pos = np.array([0.5, 0.0, 0.025])
        self._cube = self._world.scene.add(
            DynamicCuboid(
                prim_path="/World/Cube",
                name="cube",
                position=cube_pos,
                scale=np.array([0.05, 0.05, 0.05]),
                color=np.array([0.0, 0.5, 1.0]),
            )
        )

        approach_offset = np.array([0.0, 0.0, 0.15])
        pick_offset = np.array([0.0, 0.0, 0.04])
        lift_offset = np.array([0.0, 0.0, 0.25])
        place_pos = np.array([0.3, 0.3, 0.04])
        place_approach = place_pos + np.array([0.0, 0.0, 0.15])

        self._waypoints = [
            ("approach", cube_pos + approach_offset),
            ("pick", cube_pos + pick_offset),
            ("lift", cube_pos + lift_offset),
            ("place_approach", place_approach),
            ("place", place_pos),
        ]

    def _init_rmpflow(self):
        if not RMP_AVAILABLE:
            print("RmpFlow not available, using fallback joint control")
            return False
        try:
            rmp_config = interface_config_loader.load_supported_motion_policy_config(
                "Franka", "RMPflow"
            )
            self._rmpflow = RmpFlow(**rmp_config)
            self._art_motion_policy = ArticulationMotionPolicy(
                robot_articulation=self._robot,
                motion_policy=self._rmpflow,
                default_physics_dt=1.0 / 60.0,
            )
            self._rmp_initialized = True
            print("RmpFlow IK solver initialized successfully")
            return True
        except Exception as e:
            print(f"RmpFlow init failed: {e}")
            print("Falling back to direct joint control")
            self._rmpflow = None
            self._art_motion_policy = None
            return False

    def reset(self):
        self._world.reset()
        self._current_step = 0
        self._current_waypoint = 0
        self._state = "moving_to_pick"

        if self._cube:
            self._cube.set_world_pose(position=np.array([0.5, 0.0, 0.025]))

        if self._robot and self._gripper_dof_indices is None:
            dof_names = self._robot.dof_names
            print(f"\nAvailable DOF names ({len(dof_names)} total):")
            for i, name in enumerate(dof_names):
                print(f"  [{i}] {name}")

            self._arm_dof_indices = []
            for i, name in enumerate(dof_names):
                if "panda_joint" in name.lower() and "finger" not in name.lower():
                    try:
                        joint_num = int(name.lower().split("panda_joint")[1].split("_")[0])
                        if 1 <= joint_num <= 7:
                            self._arm_dof_indices.append(i)
                    except (ValueError, IndexError):
                        pass
            self._arm_dof_indices = sorted(self._arm_dof_indices)

            self._gripper_dof_indices = []
            for i, name in enumerate(dof_names):
                if "finger" in name.lower():
                    self._gripper_dof_indices.append(i)

            print(f"\nArm DOF indices: {self._arm_dof_indices}")
            print(f"Gripper DOF indices: {self._gripper_dof_indices}")

            if len(self._arm_dof_indices) != 7:
                print(f"WARNING: Expected 7 arm DOFs, found {len(self._arm_dof_indices)}")

        if not self._rmp_initialized:
            self._init_rmpflow()

        if len(self._arm_dof_indices) >= 7:
            home_joints = [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785]
            self._set_arm_joints(home_joints)

        self._open_gripper()

    def get_end_effector_pose(self):
        if not self._end_effector_prim_path:
            return None, None
        stage = omni.usd.get_context().get_stage()
        ee_prim = stage.GetPrimAtPath(self._end_effector_prim_path)
        if not ee_prim or not ee_prim.IsValid():
            return None, None
        xformable = UsdGeom.Xformable(ee_prim)
        world_transform = xformable.ComputeLocalToWorldTransform(0)
        translation = world_transform.ExtractTranslation()
        rotation = world_transform.ExtractRotationQuat()
        ee_position = np.array([translation[0], translation[1], translation[2]])
        imaginary = rotation.GetImaginary()
        ee_orientation = np.array([rotation.GetReal(), imaginary[0], imaginary[1], imaginary[2]])
        return ee_position, ee_orientation

    def forward(self, ik_method):
        if not self._robot:
            return
        if self._rmp_initialized and self._rmpflow is not None:
            self._rmpflow_forward()
        else:
            self._simple_forward()

    def _rmpflow_forward(self):
        ee_position, _ = self.get_end_effector_pose()

        if self._state == "moving_to_pick":
            if self._current_waypoint < 2:
                waypoint_name, target_pos = self._waypoints[self._current_waypoint]
                self._rmpflow.set_end_effector_target(
                    target_position=target_pos,
                    target_orientation=self._gripper_down_orientation,
                )
                actions = self._art_motion_policy.get_next_articulation_action()
                self._apply_arm_action(actions)

                if ee_position is not None:
                    distance = np.linalg.norm(ee_position - target_pos)
                    if distance < 0.02:
                        print(f"Reached waypoint: {waypoint_name} (distance: {distance:.4f}m)")
                        self._current_waypoint += 1
                        self._current_step = 0
                        if waypoint_name == "pick":
                            self._state = "closing_gripper"
                else:
                    self._current_step += 1
                    if self._current_step > 200:
                        self._current_waypoint += 1
                        self._current_step = 0
                        if self._current_waypoint >= 2:
                            self._state = "closing_gripper"

        elif self._state == "closing_gripper":
            self._close_gripper()
            self._current_step += 1
            if self._current_step > 40:
                self._state = "lifting"
                self._current_step = 0
                print("Gripper closed, lifting...")

        elif self._state == "lifting":
            _, target_pos = self._waypoints[2]
            self._rmpflow.set_end_effector_target(
                target_position=target_pos,
                target_orientation=self._gripper_down_orientation,
            )
            actions = self._art_motion_policy.get_next_articulation_action()
            self._apply_arm_action(actions)

            if ee_position is not None:
                distance = np.linalg.norm(ee_position - target_pos)
                if distance < 0.02:
                    print(f"Reached lift waypoint (distance: {distance:.4f}m)")
                    self._current_waypoint = 3
                    self._state = "moving_to_place"
                    self._current_step = 0
            else:
                self._current_step += 1
                if self._current_step > 150:
                    self._current_waypoint = 3
                    self._state = "moving_to_place"
                    self._current_step = 0

        elif self._state == "moving_to_place":
            if self._current_waypoint < len(self._waypoints):
                waypoint_name, target_pos = self._waypoints[self._current_waypoint]
                self._rmpflow.set_end_effector_target(
                    target_position=target_pos,
                    target_orientation=self._gripper_down_orientation,
                )
                actions = self._art_motion_policy.get_next_articulation_action()
                self._apply_arm_action(actions)

                if ee_position is not None:
                    distance = np.linalg.norm(ee_position - target_pos)
                    if distance < 0.02:
                        print(f"Reached waypoint: {waypoint_name} (distance: {distance:.4f}m)")
                        self._current_waypoint += 1
                        self._current_step = 0
                        if waypoint_name == "place":
                            self._state = "opening_gripper"
                else:
                    self._current_step += 1
                    if self._current_step > 200:
                        self._current_waypoint += 1
                        self._current_step = 0
                        if self._current_waypoint >= len(self._waypoints):
                            self._state = "opening_gripper"

        elif self._state == "opening_gripper":
            self._open_gripper()
            self._current_step += 1
            if self._current_step > 40:
                self._state = "done"
                print("Task completed!")

    def _apply_arm_action(self, actions):
        if actions is None:
            return
        current_positions = self._robot.get_joint_positions()
        if actions.joint_positions is not None:
            target_positions = current_positions.copy()
            rmp_positions = actions.joint_positions
            for i, idx in enumerate(self._arm_dof_indices[:7]):
                if i < len(rmp_positions) and rmp_positions[i] is not None:
                    target_positions[idx] = rmp_positions[i]
            action = ArticulationAction(joint_positions=target_positions)
            self._robot.apply_action(action)

    def _simple_forward(self):
        if self._state == "moving_to_pick":
            if len(self._arm_dof_indices) >= 7:
                pick_joints = self._interpolate_joints(
                    [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785],
                    [0.0, 0.15, 0.0, -1.13, 0.0, 1.28, 0.785],
                    self._current_step, 150,
                )
                self._set_arm_joints(pick_joints)
            self._current_step += 1
            if self._current_step > 150:
                self._state = "closing_gripper"
                self._current_step = 0
                print("Reached pick position")

        elif self._state == "closing_gripper":
            self._close_gripper()
            self._current_step += 1
            if self._current_step > 50:
                self._state = "lifting"
                self._current_step = 0
                print("Gripper closed")

        elif self._state == "lifting":
            if len(self._arm_dof_indices) >= 7:
                lift_joints = self._interpolate_joints(
                    [0.0, 0.15, 0.0, -1.13, 0.0, 1.28, 0.785],
                    [0.0, -0.3, 0.0, -1.8, 0.0, 1.571, 0.785],
                    self._current_step, 100,
                )
                self._set_arm_joints(lift_joints)
            self._current_step += 1
            if self._current_step > 100:
                self._state = "moving_to_place"
                self._current_step = 0
                print("Lifted object")

        elif self._state == "moving_to_place":
            if len(self._arm_dof_indices) >= 7:
                place_joints = self._interpolate_joints(
                    [0.0, -0.3, 0.0, -1.8, 0.0, 1.571, 0.785],
                    [0.785, 0.15, 0.0, -1.13, 0.0, 1.28, 0.785],
                    self._current_step, 150,
                )
                self._set_arm_joints(place_joints)
            self._current_step += 1
            if self._current_step > 150:
                self._state = "opening_gripper"
                self._current_step = 0
                print("Reached place position")

        elif self._state == "opening_gripper":
            self._open_gripper()
            self._current_step += 1
            if self._current_step > 50:
                self._state = "done"
                print("Released object")

    def _interpolate_joints(self, start, end, step, total_steps):
        t = min(step / max(total_steps, 1), 1.0)
        t_smooth = t * t * (3.0 - 2.0 * t)
        return [s + (e - s) * t_smooth for s, e in zip(start, end)]

    def _set_arm_joints(self, target_positions):
        if self._robot and self._arm_dof_indices and len(self._arm_dof_indices) >= len(target_positions):
            current_positions = self._robot.get_joint_positions()
            for i, idx in enumerate(self._arm_dof_indices[:len(target_positions)]):
                current_positions[idx] = target_positions[i]
            action = ArticulationAction(joint_positions=current_positions)
            self._robot.apply_action(action)

    def _close_gripper(self):
        if self._robot and self._gripper_dof_indices:
            current_positions = self._robot.get_joint_positions()
            for idx in self._gripper_dof_indices:
                current_positions[idx] = 0.0
            action = ArticulationAction(joint_positions=current_positions)
            self._robot.apply_action(action)

    def _open_gripper(self):
        if self._robot and self._gripper_dof_indices:
            current_positions = self._robot.get_joint_positions()
            for idx in self._gripper_dof_indices:
                current_positions[idx] = 0.04
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

    for _ in range(10):
        simulation_app.update()

    omni.timeline.get_timeline_interface().play()

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
