#! /usr/bin/python2
# -*- coding: utf-8 -*-

import rospy
import string
import random
import numpy as np

from coppeliasim_remote_api.bluezero import b0RemoteApi
from coppeliasim_master.msg import CoppeliaSimSynchronous
from std_msgs.msg import Int32, Float64MultiArray
from std_msgs.msg import Bool

import simple_gripper as sg
 

def id_generator():
    id = ''.join(
        random.choice(string.ascii_uppercase + string.digits)
        for _ in range(8))
    return id


class CoppeliaSimGripperDriver(object):
    def __init__(self, client, client_ID, gripper_contact):
        self.client = client
        self.client_ID = client_ID

        self.client.do_next_step = False
        self.client.runInSynchronousMode = True

        self.gripper_contact = gripper_contact
        self.simData = gripper_contact.simData

        # Initialise
        self.client.simxGetSimulationStepStarted(
            self.client.simxDefaultSubscriber(self.simulation_step_start_cb))

        # set ros publishers
        self.coppeliasim_synchrnous_trigger_pub = rospy.Publisher(
            "/coppeliasim_synchronous", CoppeliaSimSynchronous, queue_size=100)
        self.gripper_status_pub= rospy.Publisher('gripper_status',Int32, queue_size=10)
        self.force_torque_pub = rospy.Publisher(
            'force_torque', Float64MultiArray, queue_size=10)

        # set ros subscribers
        self.coppeliasim_synchrnous_trigger_sub = rospy.Subscriber(
            "/coppeliasim_synchronous", CoppeliaSimSynchronous, callback=self.coppeliasim_synchronous_cb)
        self.gripper_control_sub = rospy.Subscriber(
            "/gripper_control", Bool, callback=self.gripper_control_cb)

        rospy.sleep(0.2)
        rospy.loginfo("Gripper interface turned on!")

    def coppeliasim_synchronous_cb(self, msg):
        """
        Callback for synchronous operation. If message is received from "coppeliasim_master", 
        "self.client.do_next_step" is set to True
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
            self.client.do_next_step=True
            return True
        else:
            return False

    def gripper_control_cb(self, msg):
        """
        Callback for vacuum control (rostopic: "gripper_control")
        --------------
        msg : Bool
            True for vacuum on and False for vacuum off
        Returns
        --------------
        None : None
        """
        if msg.data == True:
            self.simData.vacuum_on = True
            print("vacuum on TRUE")
        else:
            self.simData.vacuum_on = False

        return None

    def simulation_step_start_cb(self, msg):
        self.fake_client_activation()

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
        self.coppeliasim_synchrnous_trigger_pub.publish(trigger_msg)
        return

    def fake_client_activation(self):
        """
        Fake client activation:
        If CoppeliaSim can not detect client's activation, it automately
        deativate the client. So we fool the simulation that the client
        is active.
        """
        self.client.simxSetIntSignal('a', 0, client.simxDefaultPublisher())

    def force_torque_publish(self):
        """
        Publish the force torque to rostopic: "force_torque"
        """
        force_torque = Float64MultiArray()
        force_torque.data = [
            self.gripper_contact.force[0],
            self.gripper_contact.force[1],
            self.gripper_contact.force[2],
            self.gripper_contact.torque[0],
            self.gripper_contact.torque[1],
            self.gripper_contact.torque[2]
        ]
        self.force_torque_pub.publish(force_torque)

    def step_simulation(self):
        while not self.client.do_next_step:
            self.client.simxSpinOnce()
            # Check if simulation on. Makes for a clean exit
            if hasattr(self, "sequence_number"):
                if self.sequence_number == -1:
                    rospy.signal_shutdown("Stoping ROS: Gripper")
                    return
            rospy.sleep(0.002)
        # ------------------ What to do in sim step -------------------
        # Select simple or complex model of simulation
        sg.epick_simple_sim_step(self.client, self.gripper_contact)
        # Publish the force torque
        self.force_torque_publish()
        # -------------------------------------------------------------

        self.client.do_next_step = False
        # Signal to master that we finished all calculations
        self.coppeliasim_synchronous_done()


if __name__ == '__main__':
    # init ros
    rospy.init_node('coppeliasim_simple_gripper')

    # init sim
    client_ID = 'b0RemoteApi_simple_gripper_{}'.format(
        id_generator())
    client = b0RemoteApi.RemoteApiClient(client_ID, 'b0RemoteApi')

    client.do_next_step = False
    client.runInSynchronousMode = True

    simData = sg.SimulationData(client)
    gripper_contact = sg.GripperContact(simData)

    sim = CoppeliaSimGripperDriver(client, client_ID, gripper_contact)

    print("EPick ClientStart")

    while not rospy.is_shutdown():
        sim.step_simulation()
