#! /usr/bin/python
# -*- coding: utf-8 -*-

from coppeliasim_remote_api.bluezero import b0RemoteApi
from coppeliasim_master.msg import CoppeliaSimSynchronous
from coppeliasim_interface.srv import GetGrasp, GetGraspResponse
from std_srvs.srv import Empty, EmptyResponse
from rl_task.srv import RLStep, RLStepResponse
from rl_task.srv import RLReset, RLResetResponse

from geometry_msgs.msg import Point, PoseStamped, TransformStamped, Transform, Point32, Pose
from visualization_msgs.msg import Marker
from sensor_msgs.msg import JointState
from moveit_msgs.msg import RobotState, RobotTrajectory, DisplayTrajectory
from trajectory_msgs.msg import JointTrajectory
from std_msgs.msg import Bool, Float64MultiArray

import rospy
import numpy as np
import time
import random
import string
import trimesh
import tf
import tf.transformations

import moveit_commander


class MotionPlanner(object):
    def __init__(self, group_name, pose_reference_frame):
        self.group_name = group_name
        self.pose_reference_frame = pose_reference_frame
        self.env_reset_counter = 0

        self.move_group = moveit_commander.MoveGroupCommander(self.group_name)
        #self.scene = moveit_commander.PlanningSceneInterface()
        self.move_group.set_planning_time(1)

        self.move_group.set_pose_reference_frame(self.pose_reference_frame)

        self.gripper_controller = rospy.Publisher(
            '/gripper_control', Bool, queue_size=1)

        # Methods for calling services
        self.get_grasp_srv = rospy.ServiceProxy('get_grasp', GetGrasp)
        self.scene_reset_srv = rospy.ServiceProxy('scene_reset', Empty)
        # Create services
        self.do_rl_step = rospy.Service(
            "/rl_step_srv", RLStep, self.env_step_cb)
        self.do_rl_step = rospy.Service(
            "/rl_reset_srv", RLReset, self.env_reset_cb)
        # Ros subscribers
        self.force_torque_sub = rospy.Subscriber(
            "/force_torque", Float64MultiArray, self.update_state_cb, queue_size=1)

        # sleep
        rospy.sleep(1)
        self.get_current_pose()

    def get_current_pose(self):
        """Wrapper for MoveGroupCommander.get_current_pose()
        Even the pose reference frame has been set, the MoveGroupCommander.get_current_pose()
        does not return a current pose correctly. Here we enforce to transform the
        current pose from the world frame to the reference frame.
        Returns:
            `geometry_msgs/PoseStamped`: Current end-effector pose on the pose reference frame.
        """
        return self.move_group.get_current_pose()

    def go_home(self, goal="home_pose"):
        """
        Move the robot to "home" pose.
        """
        self.move_group.clear_pose_targets()
        self.move_group.set_start_state_to_current_state()
        self.move_group.set_named_target(goal)
        self.move_group.plan()
        self.move_group.go(wait=True)
        self.move_group.stop()

    def go_to_pick(self, gripper_controller):
        " Robot moves to pick position"
        gripper_controller.publish(Bool(data=False))
        rospy.sleep(1)
        matrix = motion_planner.get_grasp
        if matrix is None:
            return False

        q = tf.transformations.quaternion_from_matrix(matrix)

        beggin_translation = np.array([0, 0, -0.1, 1])
        beggin_translation = np.dot(matrix, beggin_translation)

        self.move_group.clear_pose_targets()

        p_0 = Pose()
        p_0.position.x = beggin_translation[0]
        p_0.position.y = beggin_translation[1]
        p_0.position.z = beggin_translation[2]
        p_0.orientation.x = q[0]
        p_0.orientation.y = q[1]
        p_0.orientation.z = q[2]
        p_0.orientation.w = q[3]

        self.move_group.set_start_state_to_current_state()
        self.move_group.set_pose_target(p_0)
        approach_plan = self.move_group.plan()
        if len(approach_plan.joint_trajectory.points) == 0:
            rospy.logwarn(
                '[{}] Picking approach pose motion plan failed!'.format(self.group_name))
            return False

        p_1 = Pose()
        p_1.position.x = matrix[0][3]
        p_1.position.y = matrix[1][3]
        p_1.position.z = matrix[2][3]  # +0.5
        p_1.orientation.x = q[0]
        p_1.orientation.y = q[1]
        p_1.orientation.z = q[2]
        p_1.orientation.w = q[3]

        last_joint_state = JointState()
        last_joint_state.header = approach_plan.joint_trajectory.header
        last_joint_state.name = approach_plan.joint_trajectory.joint_names
        last_joint_state.position = approach_plan.joint_trajectory.points[-1].positions
        moveit_robot_state = RobotState()
        moveit_robot_state.joint_state = last_joint_state
        self.move_group.set_start_state(moveit_robot_state)
        self.move_group.set_pose_target(p_1)
        pickup_plan = self.move_group.plan()

        if len(pickup_plan.joint_trajectory.points) == 0:
            rospy.logwarn(
                '[{}] Picking pickup pose motion plan failed!'.format(self.group_name))
            return False

        gripper_controller.publish(Bool(data=False))
        self.move_group.execute(approach_plan, wait=True)
        self.move_group.stop()
        self.move_group.execute(pickup_plan, wait=True)
        self.move_group.stop()
        rospy.sleep(1)
        gripper_controller.publish(Bool(data=True))
        rospy.sleep(1)

        return True

    def update_state_cb(self, msg):
        self.force_torque = np.array(
            [msg.data[3], msg.data[4], msg.data[5], msg.data[0], msg.data[1], msg.data[2]])
        print(self.force_torque)
        return None

    def env_reset_cb(self, msg):
        self.env_step_counter = 0
        self.env_reset_counter += 1
        if self.env_reset_counter % 10 == 0:
            motion_planner.go_home("out_of_way")
            motion_planner.scene_reset_srv()
            rospy.sleep(1)
        response = RLResetResponse()
        # Get one grasp
        success = None
        while success is not True:
            for i in range(10):
                success = motion_planner.go_to_pick(self.gripper_controller)
                if success:
                    rospy.sleep(0.3)
                    if (np.array(self.force_torque) == 0).all(-1):
                        continue
                    break
            else:
                if success == False:
                    motion_planner.go_home("out_of_way")
                    motion_planner.scene_reset_srv()
                    rospy.sleep(1)
                continue


        state = np.append(self.force_torque[0:3], self.force_torque[3:6])
        rospy.logwarn("-----------------------------------------")
        rospy.logwarn(success)
        rospy.logwarn(self.force_torque)
        rospy.logwarn("-----------------------------------------")
        response.state = [self.force_torque[3], self.force_torque[4], self.force_torque[5],
                          self.force_torque[0], self.force_torque[1], self.force_torque[2]]
        return response

    def env_step_cb(self, msg):
        self.env_step_counter += 1
        response = RLStepResponse()
        info = "nothing"

        # Perform the given action
        action = msg.action
        success = self.move_offset(action)
        if success == False:
            response.info = "fail"

        # Calculate the reward
        state = [self.force_torque[3], self.force_torque[4], self.force_torque[5], self.force_torque[0], self.force_torque[1], self.force_torque[2]]
        
        print(state)

        # Return the result
        response.state = list(state)
        response.reward = 0.0

        if self.env_step_counter > 20:
            response.done = True
        else:
            response.done = False

        if response.info == "fail":
            response.done = True

        response.info = info
        if response.done:
            self.go_home()
            self.gripper_controller.publish(Bool(data=False))
            rospy.sleep(0.2)
        return response

    def move_offset(self, offset):
        """
        Move the robot to the current pose plus the offset.
        Args:
            offset (`geometry_msgs/Point`): Offset to add to the current pose.
        """
        # set planning start state
        self.move_group.clear_pose_targets()
        self.move_group.set_start_state_to_current_state()
        cur_pose = self.move_group.get_current_pose()

        frame_tf = tf.transformations.quaternion_matrix([cur_pose.pose.orientation.x,
                                                         cur_pose.pose.orientation.y,
                                                         cur_pose.pose.orientation.z,
                                                         cur_pose.pose.orientation.w])

        for i in range(1, 6):
            if offset is not None:
                rot_axis = np.array([offset[1], offset[2], offset[3]])
                rot_axis = rot_axis/np.linalg.norm(rot_axis)

                offset_matrix = tf.transformations.rotation_matrix(
                    offset[0]/i, rot_axis)
            else:
                offset_matrix = np.identity(4)
            combined_matrix = np.dot(frame_tf, offset_matrix)
            combined_quat = tf.transformations.quaternion_from_matrix(
                combined_matrix)

            new_pose = cur_pose
            # (0.4 - cur_pose.pose.position.x)/4
            new_pose.pose.position.x += (0.4 - cur_pose.pose.position.x)/6
            new_pose.pose.position.y += (0 - cur_pose.pose.position.y)/6
            new_pose.pose.position.z += 0.02  # 0.02
            new_pose.pose.orientation.x = combined_quat[0]
            new_pose.pose.orientation.y = combined_quat[1]
            new_pose.pose.orientation.z = combined_quat[2]
            new_pose.pose.orientation.w = combined_quat[3]

            # plan from current pose to approach pose
            waypoints = []
            # waypoints.append(self.get_current_pose().pose)
            waypoints.append(new_pose.pose)
            self.move_group.set_pose_target(waypoints[0])
            offset_move_plan = self.move_group.plan()

            if len(offset_move_plan.joint_trajectory.points) <= 1:
                rospy.logwarn(
                    '[{}] Place approach pose motion plan failed!'.format(self.group_name))
                continue
            elif len(offset_move_plan.joint_trajectory.points) > 35:
                rospy.logwarn(
                    '[{}] Number of trajectory points'.format(len(offset_move_plan.joint_trajectory.points)))
                continue
            else:
                self.execute_plan(offset_move_plan)
                rospy.sleep(0.1)
                return True
        else:
            return False

    def execute_plan(self, plan, auto=True):
        """
        A wrapper for MoveIt function move_group.execute(). Makes the robot do the executed the given plan.
        --------------
        plan : RobotTrajectory() (MoveIt RobotTrajectory MSG)
            A motion plan to be executed
        auto : `Bool`
            If true, robot will be executed with out user confirmation.
        --------------
        """
        if plan == None:
            raise ValueError("Plan is non existant.")

        #self.display_planned_path(plan)

        if auto is True:
            self.move_group.execute(plan, wait=True)
            self.move_group.stop()
        elif auto is False:
            # rospy.loginfo("Excute the plan")
            self.confirm_to_execution()
            self.move_group.execute(plan, wait=True)
            self.move_group.stop()
            rospy.loginfo("Finish execute")
        plan = None

    @property
    def get_grasp(self):
        for i in range(10):
            grasp_loc = self.get_grasp_srv(1)
            grasp_loc = grasp_loc.grasp_tf
            grasp_loc = np.reshape(np.array(grasp_loc), (4, 4))
            if (grasp_loc == 0).all():
                continue
            else:
                return grasp_loc
        else:
            return None


if __name__ == "__main__":
    rospy.init_node('motion_planner')

    motion_planner = MotionPlanner("panda_arm", "panda_1_link0")
    motion_planner.go_home("out_of_way")

    motion_planner.scene_reset_srv()
    motion_planner.move_group.set_max_velocity_scaling_factor(1)

    rospy.spin()
    