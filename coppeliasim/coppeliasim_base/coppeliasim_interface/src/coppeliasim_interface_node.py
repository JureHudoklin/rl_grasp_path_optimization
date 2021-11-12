
from coppeliasim_remote_api.bluezero import b0RemoteApi
from coppeliasim_master.msg import CoppeliaSimSynchronous
from std_msgs.msg import Bool

import rospy
import numpy as np
import random
import string
import trimesh


def id_generator():
    id = ''.join(
        random.choice(string.ascii_uppercase + string.digits)
        for _ in range(8))
    return id

class ObjectMatrixCallback:
    def __init__(self, tf, name):
        self.tf = tf
        self.name = name

    def __call__(self, msg):
        self.tf[self.name] = msg[1]


def CoppeliaSimInterface(object):

    def __init__(self, client, client_ID, object_list):
        self.client = client
        self.client_ID = client_ID

        # Object list
        self.objects = {name: self.client.simxGetObjectHandle(
            name, self.client.simxServiceCall())[1] for name in object_list}

        # Object transformations from world frame
        self.object_tf = {name: np.zeros((4, 4)) for name in object_list}
        for name in object_list:
            self.client.simxGetObjectMatrix(
                self.obj_handles[name], -1, 
                self.client.simxDefaultSubscriber(ObjectMatrixCallback(self.object_tf, name)))


        # Ros Subscigers
        self.get_grasp_sub = rospy.Subscriber("get_grasp", Bool, self.get_grasp_cb, queue_size=1)


        # Synchronous operation
        self.client.simxGetSimulationStepStarted(
            self.client.simxDefaultSubscriber(self.simulation_step_start_cb))

        self.coppeliasim_synchronous_trigger_pub = rospy.Publisher(
            "/coppeliasim_synchronous", CoppeliaSimSynchronous, queue_size=100)

        self.coppeliasim_synchronous_trigger_sub = rospy.Subscriber(
            "/coppeliasim_synchronous", CoppeliaSimSynchronous, callback=self.coppeliasim_synchronous_cb)

        rospy.sleep(0.2)

    def get_grasp_cb(self, msg):
        

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


class Scene(object):
    """Represents a scene, which is a collection of objects and their poses."""
    def __init__(self) -> None:
        """Create a scene object."""
        self._objects = {}
        self._poses = {}
        self._support_objects = []
        self. gripper_path = "/home/jure/reinforcement_ws/src/coppeliasim/coppeliasim_base/coppeliasim_interface/src/EPick_extend_sg_collision.stl"

        self.collision_manager = trimesh.collision.CollisionManager()

        tf = trimesh.geometry.align_vectors(np.array([0, 0, 1]), np.array([0, 0, -1]))
        self.gripper_mesh = trimesh.load(gripper_path, force="mesh")
        self.gripper_mesh.apply_transform(tf)
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

    def as_trimesh_scene(self, display = False):
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



if __name__ == '__main__':
    # init ros
    rospy.init_node('coppeliasim_interface')

    object_list = []

    # init sim
    client_ID = 'b0RemoteApi_interface_client_{}'.format(id_generator())
    client = b0RemoteApi.RemoteApiClient(client_ID, 'b0RemoteApi')
    sim = CoppeliaSimInterface(client, client_ID, object_list)

    rospy.loginfo("EPick ClientStart")  


    while not rospy.is_shutdown():
        sim.step_simulation()
