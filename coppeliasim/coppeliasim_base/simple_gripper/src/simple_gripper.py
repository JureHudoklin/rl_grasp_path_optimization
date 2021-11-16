#! /usr/bin/python2
# -*- coding: utf-8 -*-

from __future__ import print_function

from coppeliasim_remote_api.bluezero import b0RemoteApi

import time

import numpy as np
import scipy
import trimesh

class SimulationData():
    """
    --------------
    This class contains the object matrices, positions, ... It also creates all the subscribers and publishers we use for exchanging data
    --------------
    """

    def __init__(self, client, seal_infinite_strength = True):

        # Get the object handles
        link_gripper= client.simxGetObjectHandle(
            "grasp_point_visual", client.simxServiceCall())[1]
        j_y = client.simxGetObjectHandle(
            "EPpick_joint_y", client.simxServiceCall())[1]
        j_x = client.simxGetObjectHandle(
            "EPpick_joint_x", client.simxServiceCall())[1]
        proximitySensor = client.simxGetObjectHandle(
            "EPick_proximity", client.simxServiceCall())[1]
        forceSensor = client.simxGetObjectHandle(
            "EPick_ForceSensor", client.simxServiceCall())[1]


        # Seth the required parameters as attributes
        # Object handle for the gripper body
        self.link_gripper = link_gripper
        self.j_y = j_y  # Object handle for the joint y
        self.j_x = j_x  # Object handle for the joint x
        self.forceSensor = forceSensor  # Object handle for the Force sensor
        self.proximitySensor = proximitySensor

        self.seal_infinite_strength = seal_infinite_strength

        self.client = client
        self.contact_object = -1
        self.vacuum_on = None

        # Create publishers and subscribers for b0 API
        self.create_subscribers()

    def create_subscribers(self):
        """
        Creates subscriber for all the calls to VREP. We use different subscriber for everything so that the code runs faster 
        (service calls are slow and defult subscriber can get cluttered)
        --------------
        contact : SuctionContact
            
        Returns
        --------------
        None : None
            Subscribers are stored as object attributes
        """
       
        # GETTING OBJECT MATRICES ----------------------------------------------------
        # Get transformation matrix from world frame to suction cup frame  -- cup_WORLD_f
        self.client.simxGetObjectMatrix(self.link_gripper, -1, self.client.simxDefaultSubscriber(self.update_gripper_WORLD_f, publishInterval=1))

        # Object matrix from link_gripper to forceSensor frame  -- fs_SC_f
        self.client.simxGetObjectMatrix(self.forceSensor, self.link_gripper,
                                        self.client.simxDefaultSubscriber(self.update_fs_GRIPPER_f, publishInterval=1))
        
        # FORCE SENSOR -------------------------------------------------------------------
        self.client.simxReadForceSensor(self.forceSensor, self.client.simxDefaultSubscriber(
            self.update_forceSensor, publishInterval=1))

        # PROXIMITY SENSOR ---------------------------------------------------------------
        self.client.simxReadProximitySensor(self.proximitySensor, self.client.simxDefaultSubscriber(
            self.update_proximitySensor, publishInterval=1))

        


    def update_gripper_WORLD_f(self, msg):
        if msg[0] == True:
            cup_tf = np.array(msg[1])
            cup_tf = np.reshape(cup_tf, (3, 4))
            self.gripper_WORLD_f = (msg[0], cup_tf)

    def update_forceSensor(self, msg):
        if msg[0] == True:
            self.external_forceMom = msg

    def update_fs_GRIPPER_f(self, msg):
        if msg[0] == True:
            cup_tf = np.array(msg[1])
            cup_tf = np.reshape(cup_tf, (3, 4))
            self.fs_GRIPPER_f = (msg[0], cup_tf)

    def update_proximitySensor(self, msg):
        if msg[0] == True:
            self.proximity_sensor = msg



class GripperContact():
    """
    --------------
    Very simple suction gripper. This class wont do any seal analysis. 
    An object will be grasped as long as the gripper is in contact with it.
    --------------
    """

    def __init__(self, simData):
        self.seal_formed = False
        self.contact_object = -1
        self.simData = simData

    def formSeal(self, max_dist=0.060):
        """
        Formes a seal if an object is close enough to the proximity sensor of the suction cup .
        --------------
        *args
        detection_distance : float
            At what distance from the suction do we consider that object and suction cup are in contact.
        
        Returns:
        --------------
        success : Bool
            Returns True if a seal can be formed and False otherwise.
        """
        print("-----------FORMIN SEAL -------------------")
        print(self.simData.proximity_sensor)
        # Check to see if we ever got any response from the proximity sensor
        if not hasattr(self.simData, "proximity_sensor") or (self.simData.proximity_sensor[1] == 0):
            return False
        print("prox sensor: ", self.simData.proximity_sensor[2])
        print("Detection distance:", max_dist)
        print(self.simData.proximity_sensor[2] - max_dist)
        print(self.simData.proximity_sensor[2] < max_dist)
        print("-----------------------------------------")
        # If the distance to the object is "small enough" we can make a seal
        if self.simData.proximity_sensor[2] < max_dist:
            self.contact_object = self.simData.proximity_sensor[4]
            print("RETURNING TRUE FROM FROM SEAL")
            return True
        else:
            return False

    def evaluateForces(self, vacuum):
        """
        Evaluates whether the suction cup can withstand the outside forces.
        As force limits the euqations from Dex-Net 3.0 are used 
        --------------
        vacuum : float
            pressure difference between inside and outside in N/mm^2 (MPa)
        Returns:
        --------------
        success : Bool
            Returns True if it can withstand the force and False otherwise
        """
        coef_friction = 0.5
        model_r = 16
        # Vacuum force
        suction_cup_area = np.pi*32**2
        vacuum_force = suction_cup_area*vacuum

        # External forces
        f_ext, mom_ext = self.get_external_force()

        max_wrench = {"f_x": coef_friction*(vacuum_force-f_ext[2]),
                      "f_y": coef_friction*(vacuum_force-f_ext[2]),
                      "f_z": vacuum_force,
                      "t_x": np.pi*model_r*0.005,
                      "t_y": np.pi*model_r*0.005,
                      "t_z": model_r*coef_friction*(vacuum_force-f_ext[2])/1000}

        # Check if all conditions apply:
        # Friction
        if not (np.sqrt(3)*abs(f_ext[0]) <= max_wrench["f_x"]):
            return False
        elif not (np.sqrt(3)*abs(f_ext[1]) <= max_wrench["f_y"]):
            return False
        elif not abs(f_ext[2]) <= max_wrench["f_z"]:
            return False
        elif not (np.sqrt(2)*abs(mom_ext[0]) <= max_wrench["t_x"]):
            return False
        elif not (np.sqrt(2)*abs(mom_ext[1]) <= max_wrench["t_y"]):
            return False
        elif not (np.sqrt(3)*abs(mom_ext[2]) <= max_wrench["t_z"]):
            return False

        # Passed all conditions -> return True
        return True

    def resetContact(self, client):
        """
        The function does all the cleanup everytime the contact needs to be reset
        --------------
        Returns:
        --------------
        None : None
        """
        self.seal_formed = False
        self.contact_object = -1
        self.simData.vacuum_on = False
        self.proximity_sensor = None
        args = []
        ret = client.simxCallScriptFunction(
            'brakeDummyLink@link_gripper_respondable', 'sim.scripttype_childscript', args, client.simxDefaultPublisher())

    def get_external_force(self):
        """
        Gets the external force and moment from the force sensor, formats it to the SC frame and returns the force and torque vector in SC frame.
        --------------
        self.simData : SimulationData
            Simulation data containing the latest force, moment and transformation matrices
        --------------
        f_dir : np.array(3,)
            A force vector in SC frame originating from the contact point
        m_vec : np.array(3,)
            A moment vector in SC frame
        """

        # External forces acting on an object. We must transform them to SC frame
        if (self.simData.external_forceMom[0] == False) or (self.simData.external_forceMom[1] == 0) or (self.simData.external_forceMom[1] == 2):
            self.simData.external_forceMom = [
                True, 1, np.array([0, 0, 0]), np.array([0, 0, 0])]

        # Brake them down into appropriate arrays
        external_force = np.array(self.simData.external_forceMom[2])
        external_moment = np.array(self.simData.external_forceMom[3])
        # Transform everything to SC frame
        tf_force = self.simData.fs_GRIPPER_f[1]
        f_dir = np.dot(tf_force[:, 0:3], external_force)
        m_vec = np.dot(tf_force[:, 0:3], external_moment)
        print(f_dir, m_vec)
        return f_dir, m_vec


def epick_simple_sim_step(client, gripperContact):
    """
    Performs one step of EPick gripper simulation with very simple logic
    --------------
    contact_obj : object we are contacting; int
    interPnts_formated : perimiter points, ordered, written in mm and SC frame; np.array(m, 3)
    client : client object
    model : suction_cup object model that contains suction cup parameters
    simData : an object that contains necessary simulation data
    contact : contact class
    --------------
    None : None
    """
    simData = gripperContact.simData

    if gripperContact.seal_formed == False:
        if simData.vacuum_on:
            seal_formed = gripperContact.formSeal()
            gripperContact.seal_formed = seal_formed
            if seal_formed == True:
                print(" ATTACHING THE OBJECTS TOGETHER")
                #Attach two objects together
                args = gripperContact.contact_object
                ret = client.simxCallScriptFunction(
                    'createDummyLink@link_gripper_respondable', 'sim.scripttype_childscript', args, client.simxDefaultPublisher())
            else:
                1
                #simData.vacuum_on = False

    if gripperContact.seal_formed == True:
        # Check if the vacuum state has changed if 0 brake the contact
        # Check wheather forces broke the seat
        if simData.vacuum_on == False:
            print("Turning vacuum off.")
            gripperContact.resetContact(client)
            return None

        # Analyze the acting forces
        vacuum = 0.07
        gripperContact.get_external_force()
        if simData.seal_infinite_strength != True:
            contact_state = gripperContact.evaluateForces(vacuum)
            if contact_state == False:
                print("Seal was broken. Turning vacuum off.")
                gripperContact.resetContact(client)
                return None

    return None
