#! /usr/bin/python
# -*- coding: utf-8 -*-

from coppeliasim_remote_api.bluezero import b0RemoteApi
from coppeliasim_master.msg import CoppeliaSimSynchronous
from std_msgs.msg import Bool
from std_srvs.srv import Empty, EmptyResponse
from coppeliasim_interface.srv import GetGrasp, GetGraspResponse

import rospy
import numpy as np
import random
import string
import trimesh

import trimesh_visualization as tv


def id_generator():
    id = ''.join(
        random.choice(string.ascii_uppercase + string.digits)
        for _ in range(8))
    return id


def vector_with_noise(a_v, noise_cone_angle):
    """
    Adds some noise to the input unit vector. The output vector will be inside the defined noise cone from original vector.
    --------------
    a_v : np.array(3,)
        Input unit vector
    noise_cone_angle : float 
        Angle in radians
    --------------
    vector_with_noise : np.array(3,)
        Original vector with noise applied.
    """
    # Create a noise vector in zero Position of length 1
    noise_max = np.tan(noise_cone_angle)
    noise_x = random.uniform(-noise_max, noise_max)
    noise_y = random.uniform(-noise_max, noise_max)
    vector_with_noise = np.array([noise_x, noise_y, 1])
    vector_with_noise = vector_with_noise/np.linalg.norm(vector_with_noise)

    # Calculate then rotation matrix from (0,0,1) to approach vector
    rot_matrix = trimesh.geometry.align_vectors(
        np.array([0, 0, 1]), a_v, return_angle=False)

    # Rotate the vector
    vector_with_noise = np.dot(rot_matrix[0:3, 0:3], vector_with_noise)

    return vector_with_noise

class ObjectMatrixCallback:
    def __init__(self, tf, name):
        self.tf = tf
        self.name = name

    def __call__(self, msg):
        msg = np.array(msg[1])
        msg = np.reshape(msg, (3,4))
        self.tf[self.name][0:3,0:4] = msg
        self.tf[self.name][3, 0:4] = np.array([0,0,0,1])

class Scene(object):
    """Represents a scene, which is a collection of objects and their poses."""

    def __init__(self):
        """Create a scene object."""
        self._objects = {}
        self._poses = {}
        self._support_objects = []
        self.gripper_path = "/home/jure/reinforcement_ws/src/coppeliasim/coppeliasim_base/coppeliasim_interface/src/gripper_collision.stl"
        self.box_path_2 = "/home/jure/reinforcement_ws/src/coppeliasim/coppeliasim_base/coppeliasim_interface/src/link_box.stl"
        self.box_path_1 = "/home/jure/reinforcement_ws/src/coppeliasim/coppeliasim_base/coppeliasim_interface/src/link_box.stl"

        self.box_pose_2 = np.array([[0.0, -1.0, 0.0, 0.0],
                                  [1.0, 0.0, 0.0, 0.5],
                                  [0.0, 0.0, 1.0, 0.0],
                                  [0.0, 0.0, 0.0, 1.0]])
        self.box_pose_1 = np.array([[1.0, 0.0, 0.0, 0.5],
                                  [0.0, 1.0, 0.0, 0.0],
                                  [0.0, 0.0, 1.0, 0.0],
                                  [0.0, 0.0, 0.0, 1.0]])

        self.box_mesh_2 = trimesh.load(self.box_path_2, force="mesh")
        self.box_mesh_2 = self.box_mesh_2.apply_scale(0.001)

        self.box_mesh_1 = trimesh.load(self.box_path_1, force="mesh")
        self.box_mesh_1 = self.box_mesh_1.apply_scale(0.001)

        self.collision_manager = trimesh.collision.CollisionManager()

        tf = trimesh.geometry.align_vectors(
            np.array([0, 0, 1]), np.array([0, 0, -1]))
        self.gripper_mesh = trimesh.load(self.gripper_path, force="mesh")
        #self.gripper_mesh.apply_transform(tf)
        self.gripper_mesh = self.gripper_mesh.apply_scale(0.001)

    def add_object(self, obj_id, obj_mesh, pose, support=False):
        """
        Add a named object mesh to the scene.
        --------------
        Args:
            obj_id (str): Name of the object.
            obj_mesh (trimesh.Trimesh): Mesh of the object to be added.
            pose (np.ndarray): Homogenous 4x4 matrix describing the objects pose in scene coordinates.
            support (bool, optional): Indicates whether this object has support surfaces for other objects. Defaults to False.
        --------------
        """
        self._objects[obj_id] = obj_mesh
        self._poses[obj_id] = pose
        if support:
            self._support_objects.append(obj_mesh)

        self.collision_manager.add_object(
            name=obj_id, mesh=obj_mesh, transform=pose)

    def is_colliding(self, mesh, transform, eps=1e-6):
        """
        Whether given mesh collides with scene
        --------------
        Arguments:
            mesh {trimesh.Trimesh} -- mesh 
            transform {np.ndarray} -- mesh transform
        Keyword Arguments:
            eps {float} -- minimum distance detected as collision (default: {1e-6})
        --------------
        Returns:
            [bool] -- colliding or not
        """
        # dist = self.collision_manager.min_distance_single(
        #     mesh, transform=transform)
        is_collision = self.collision_manager.in_collision_single(
            mesh, transform=transform)
        return not is_collision
        #return dist < eps

    def reset(self):
        """
        --------------
        Reset, i.e. remove scene objects
        --------------
        """
        for name in self._objects:
            print(name)
            self.collision_manager.remove_object(name)
        self._objects = {}
        self._poses = {}
        self._support_objects = []

    def as_trimesh_scene(self, display=False):
        """
        Return trimesh scene representation.
        --------------
        Keyword args:
            display (bool->default is False) : Wheather to display the trimesh scene
        --------------
        Returns:
            trimesh.Scene: Scene representation.
        """
        trimesh_scene = trimesh.scene.Scene()
        for obj_id, obj_mesh in self._objects.items():
            trimesh_scene.add_geometry(
                obj_mesh,
                node_name=obj_id,
                geom_name=obj_id,
                transform=self._poses[obj_id],
            )

        if display:
            trimesh_scene.show()

        return trimesh_scene

class CoppeliaSimInterface(object):

    def __init__(self, client, client_ID, object_list):
        self.client = client
        self.client_ID = client_ID

        self.gripper_path = "/home/jure/reinforcement_ws/src/coppeliasim/coppeliasim_base/coppeliasim_interface/src/gripper_collision.stl"
        meshes_dir = "/home/jure/reinforcement_ws/src/coppeliasim/coppeliasim_base/coppeliasim_interface/src/meshes/"


        # Object list
        self.objects = {name: self.client.simxGetObjectHandle(
            name, self.client.simxServiceCall())[1] for name in object_list}
        self.object_meshes = {name: trimesh.load_mesh(
            meshes_dir+name+".obj") for name in object_list}

        # Object transformations from world frame
        self.object_tf = {name: np.zeros((4, 4)) for name in object_list}
        for name in object_list:
            self.client.simxGetObjectMatrix(
                self.objects[name],
                -1, 
                self.client.simxDefaultSubscriber(ObjectMatrixCallback(self.object_tf, name)))

        # Make trimesh scene
        self.tri_scene = Scene()

        # Ros Subscigers
        #self.get_grasp_sub = rospy.Subscriber("get_grasp", Bool, self.get_grasp_cb, queue_size=1)

        # Ros Service
        self.get_grasp_srv = rospy.Service("get_grasp", GetGrasp, self.get_grasp_srv_cb)
        self.randomly_place_objects_srv = rospy.Service(
            "scene_reset", Empty, self.scene_reset_srv_cb)


        # Synchronous operation
        self.do_next_step = True
        self.client.simxGetSimulationStepStarted(
            self.client.simxDefaultSubscriber(self.simulation_step_start_cb))

        self.coppeliasim_synchronous_trigger_pub = rospy.Publisher(
            "/coppeliasim_synchronous", CoppeliaSimSynchronous, queue_size=100)

        self.coppeliasim_synchronous_trigger_sub = rospy.Subscriber(
            "/coppeliasim_synchronous", CoppeliaSimSynchronous, callback=self.coppeliasim_synchronous_cb)

        rospy.sleep(0.2)

    def get_grasp_srv_cb(self, request):
        rospy.loginfo("Getting a viable grasp")
        self.tri_scene.reset()
        self.tri_scene.add_object('box_1', self.tri_scene.box_mesh_1,
                        self.tri_scene.box_pose_1)
        self.tri_scene.add_object('box_2', self.tri_scene.box_mesh_2,
                                  self.tri_scene.box_pose_2)
        for object_id in self.objects.keys():
            self.tri_scene.add_object(object_id, self.object_meshes[object_id], np.array(self.object_tf[object_id]))

        response = GetGraspResponse()

        random_object = random.choice(self.objects.keys())

        samples, face_idx = trimesh.sample.sample_surface_even(
            self.object_meshes[random_object], count=100)
        normals = self.object_meshes[random_object].face_normals[face_idx]

        samples = trimesh.transform_points(
            samples, self.object_tf[random_object], translate=True)
        normals = trimesh.transform_points(
            normals, self.object_tf[random_object], translate=False)

        for sample, a_v in zip(samples, normals):
            if a_v[2] < 0:
                continue
            for i in range(10):
                a_v_noise = vector_with_noise(a_v, 0.6)
                tf = np.array(trimesh.geometry.align_vectors(np.array([0,0,-1]), a_v_noise))
                tf[0:3,3] = sample
                grasp_viable = self.tri_scene.is_colliding(self.tri_scene.gripper_mesh, tf)
                
                if grasp_viable:
                    ##################
                    my_scene = tv.VisualizationObject()
                    my_scene.plot_coordinate_system(scale = 0.01)
                    self.tri_scene.add_object("gripper", self.tri_scene.gripper_mesh, tf)
                    my_scene.display(self.tri_scene.as_trimesh_scene(display=False))
                    ##################
                    rospy.loginfo("Found a viable grasp")
                    response.grasp_tf = list(tf.flatten())
                    return response
        else:
            rospy.logwarn("No viable grasp found")
            return response

    def scene_reset_srv_cb(self, request):
        rospy.loginfo("Resetting the scene")
        for object in self.objects:
            x = random.uniform(0.33, 0.67)
            y = random.uniform(-0.27, 0.27)
            z = random.uniform(0.3, 0.5)
            self.client.simxSetObjectPosition(
                self.objects[object],
                -1,
                [x, y, z],
                self.client.simxServiceCall())

        return EmptyResponse()

    def coppeliasim_synchronous_done(self):
        """
        Call this function when the client has finished doing all the calculations.
        To call this the object must have the attributes:
            - self.sequence_number
            - self.client_ID
        --------------
        Returns
        --------------
        None : None
        """
        if not hasattr(self, "sequence_number"):
            return
        # Publish the trigger to all other synchrnous clients
        trigger_msg = CoppeliaSimSynchronous()
        trigger_msg.stamp = rospy.Time.now()
        trigger_msg.sequence_number = self.sequence_number
        trigger_msg.client_ID = self.client_ID
        self.coppeliasim_synchronous_trigger_pub.publish(trigger_msg)
        return

    def coppeliasim_synchronous_cb(self, msg):
        """
        Callback for synchronous operation. If message is received from "coppeliasim_master", 
        "self.do_next_step" is set to True
        and "self.sequence_number" is updated.
        --------------
        msg : CoppeliaSimSynchronous 
            ros message containing a time stamp,  sequence_number and client_ID
        Returns
        --------------
        Bool : True/False
            True if message from "coppeliasim_master"
        """
        if msg.client_ID == "coppeliasim_master":
            self.sequence_number = msg.sequence_number
            self.do_next_step = True
            return True
        else:
            return False

    def simulation_step_start_cb(self, msg):
        self.fake_client_activation()

    def simulation_step_done_cb(self):
        1
        # HERE do what you want to do in simulation step
        

    def fake_client_activation(self):
        """
        Fake client activation:
        If CoppeliaSim can not detect client's activation, it automately
        deativate the client. So we fool the simulation that the client
        is active.
        """
        self.client.simxSetIntSignal('a', 0, client.simxDefaultPublisher())

    def step_simulation(self):
        """
        Perform one step of the simulation.
        --------------
        Returns
        --------------
        None : None
        """
        while not self.do_next_step:
            self.client.simxSpinOnce()
            # Check if simulation on. Makes for a clean exit
            if hasattr(self, "sequence_number"):
                if self.sequence_number == -1:
                    rospy.signal_shutdown("Stopping ROS: Manipulator Driver")
                    return
            rospy.sleep(0.002)

        self.simulation_step_done_cb()

        self.do_next_step = False
        self.coppeliasim_synchronous_done()






if __name__ == '__main__':
    # init ros
    rospy.init_node('coppeliasim_interface')

    object_list = ["object_"+str(i) for i in range(0,17)]

    # init sim
    client_ID = 'b0RemoteApi_interface_client_{}'.format(id_generator())
    client = b0RemoteApi.RemoteApiClient(client_ID, 'b0RemoteApi')
    sim = CoppeliaSimInterface(client, client_ID, object_list)

    rospy.loginfo("EPick ClientStart")  


    while not rospy.is_shutdown():
        sim.step_simulation()
