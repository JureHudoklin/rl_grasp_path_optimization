#! /usr/bin/python
# -*- coding: utf-8 -*-

from coppeliasim_remote_api.bluezero import b0RemoteApi
from coppeliasim_master.msg import CoppeliaSimSynchronous
from coppeliasim_interface.srv import GetGrasp, GetGraspResponse
from std_srvs.srv import Empty, EmptyResponse

from geometry_msgs.msg import Point, PoseStamped, TransformStamped, Transform, Point32, Pose
from visualization_msgs.msg import Marker
from moveit_msgs.msg import RobotState, RobotTrajectory, DisplayTrajectory
from trajectory_msgs.msg import JointTrajectory

import rospy
import numpy as np
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

        self.move_group = moveit_commander.MoveGroupCommander(self.group_name)
        #self.scene = moveit_commander.PlanningSceneInterface()

        self.move_group.set_pose_reference_frame(self.pose_reference_frame)

        self.get_grasp_srv = rospy.ServiceProxy('get_grasp', GetGrasp)
        self.scene_reset_srv = rospy.ServiceProxy('scene_reset', Empty)

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

    def go_home(self):
        """
        Move the robot to "home" pose.
        """
        self.move_group.clear_pose_targets()
        self.move_group.set_start_state_to_current_state()
        self.move_group.set_named_target("home_pose")
        self.move_group.plan()
        self.move_group.go(wait=True)
        self.move_group.stop()

    def go_to_pick(self, matrix):
        # matrix = np.array([[1 ,0 ,0 ,0],
        #                    [0 ,1 ,0 ,0],
        #                    [0 ,0 ,-1 ,0],
        #                    [0 ,0 ,0 ,1]])
        q = tf.transformations.quaternion_from_matrix(matrix)

        beggin_translation = np.array([0,0,-0.1, 1])
        beggin_translation = np.dot(matrix, beggin_translation)

        p_0 = Pose()
        p_0.position.x = beggin_translation[0]
        p_0.position.y = beggin_translation[1]
        p_0.position.z = beggin_translation[2]
        p_0.orientation.x = q[0]
        p_0.orientation.y = q[1]
        p_0.orientation.z = q[2]
        p_0.orientation.w = q[3]

        p_1= Pose()
        p_1.position.x = matrix[0][3]
        p_1.position.y = matrix[1][3]
        p_1.position.z = matrix[2][3]  # +0.5
        p_1.orientation.x = q[0]
        p_1.orientation.y = q[1]
        p_1.orientation.z = q[2]
        p_1.orientation.w = q[3]

        self.move_group.clear_pose_targets()
        self.move_group.set_start_state_to_current_state()
        self.move_group.set_pose_target(p_0)
        self.move_group.go(wait=True)

        self.move_group.clear_pose_targets()
        self.move_group.set_start_state_to_current_state()
        self.move_group.set_pose_target(p_1)
        self.move_group.go(wait=True)

    @property
    def get_grasp(self):
        for i in range(10):
            grasp_loc = motion_planner.get_grasp_srv(1)
            grasp_loc = grasp_loc.grasp_tf
            grasp_loc = np.reshape(np.array(grasp_loc), (4, 4))
            if (grasp_loc==0).all():
                continue
            else:
                return grasp_loc
        else:
            return None
        

if __name__ == "__main__":
    rospy.init_node('motion_planner')

    motion_planner = MotionPlanner("panda_arm", "panda_1_link0")
    rospy.sleep(5)
    motion_planner.go_home()

    while not rospy.is_shutdown():
        motion_planner.scene_reset_srv()
        rospy.sleep(1)
        grasp_tf = motion_planner.get_grasp
        if grasp_tf is None:
            continue

        motion_planner.go_to_pick(grasp_tf)
        exit()
        rospy.sleep(5)


    rospy.spin()
