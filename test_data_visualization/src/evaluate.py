#!/usr/bin/env python3

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

DATA = "/home/jure/reinforcement_ws/src/test_data_visualization/data"

def calculate_reward_v3(force, torque):
    model_r = 0.032
    coef_friction = 0.5
    vacuum_force = 100-abs(force[2])
    f_tan = np.sqrt(force[0]**2 + force[1]**2)
    t_tan = np.sqrt(torque[0]**2 + torque[1]**2)

    alpha_f = np.arctan2(force[1], force[0])
    alpha_t = np.arctan2(torque[1], torque[0])

    f_tan_ratio = f_tan/(vacuum_force*coef_friction)
    t_tan_ratio = t_tan/(np.pi*model_r*5)

    x_dir = f_tan_ratio*np.cos(alpha_f)
    y_dir = f_tan_ratio*np.sin(alpha_f)
    z_dir = force[2]/vacuum_force
    x_torque = t_tan_ratio*np.cos(alpha_t)
    y_torque = t_tan_ratio*np.sin(alpha_t)
    z_torque = torque[2]/np.abs(model_r*coef_friction*(vacuum_force))

    state = np.array([x_dir, y_dir, z_dir, x_torque, y_torque, z_torque])

    reward = 5-np.abs(x_dir)-np.abs(y_dir) - \
        np.abs(x_torque)-np.abs(y_torque)-np.abs(z_torque)
    return state


def evaluate_test_1(dir_simple, dir_complex, save = False):
    files_dir_1 = os.listdir(dir_simple)
    files_dir_2 = os.listdir(dir_complex)

    # Read both directories
    arrays_complex = []
    arrays_simple = []

    for file in files_dir_1:
        arr = np.load(os.path.join(dir_simple,file))
        if len(arr) > 20:
            arrays_simple.append(arr)
    for file in files_dir_2:
        arr = np.load(os.path.join(dir_complex, file))
        if len(arr) > 20:
            arrays_complex.append(arr)

    # Format force and torque to states
    states_simple = []
    for arr in arrays_simple:
        gripper_states = []
        for state in arr:
            gripper_state = calculate_reward_v3(state[0:3], state[3:6])
            gripper_states.append(gripper_state)
        states_simple.append(gripper_states)
    
    states_complex = []
    for arr in arrays_complex:
        gripper_states = []
        for state in arr:
            gripper_state = calculate_reward_v3(state[0:3], state[3:6])
            gripper_states.append(gripper_state)
        states_complex.append(gripper_states)
    
    states_simple = np.array(states_simple)
    states_complex = np.array(states_complex)

    average_simple = np.average(np.abs(states_simple), axis=0)
    average_complex = np.average(np.abs(states_complex), axis=0)

    plt.title(" Timestep Penalty")
    plt.plot(np.sum(average_simple[0:20], axis=1), label="Simple Model")
    plt.plot(np.sum(average_complex[0:20], axis=1), label="Complex Model")
    plt.xlabel("Timestep [/]")
    plt.ylabel("Avg. Penalty [/]")
    plt.legend()
    if save:
        plt.savefig(os.path.join(DATA, "Images/test_1_timestep_penalty.png"))
    else:
        plt.show()
    return

def evaluate_test_2(dir, sample, save = False):
    data = np.load(os.path.join(dir, sample))
    data = np.array(data)

    gripper_states = []
    for state in data:
        gripper_state = calculate_reward_v3(state[0:3], state[3:6])
        gripper_states.append(gripper_state)
    
    gripper_states = np.array(gripper_states)

    plt.title("Test 2: No object colission")
    plt.plot(gripper_states[:,0], label = "f_x")
    plt.plot(gripper_states[:, 1], label="f_y")
    plt.plot(gripper_states[:, 2], label="f_z")
    plt.plot(gripper_states[:, 3], label="t_x")
    plt.plot(gripper_states[:, 4], label="t_y")
    plt.plot(gripper_states[:, 5], label="t_z")
    plt.xlabel("Timestep [/]")
    plt.ylabel("Penalty [/]")
    plt.legend()
    if save:
        plt.savefig(os.path.join(DATA, "Images/test_2_collision.png"))
    else:
        plt.show()
    return


if __name__ == '__main__':
    TEST_1_DIR = os.path.join(DATA, "Test_1_ObjectOrientation")
    TEST_2_DIR = os.path.join(DATA, "Test_2_Cluttered")

    test_1_simple = os.path.join(TEST_1_DIR, "Simple/M1")
    test_1_complex = os.path.join(TEST_1_DIR, "Complex/M1")
    evaluate_test_1(test_1_simple, test_1_complex)

    test_2_dir = os.path.join(TEST_2_DIR, "Simple/M2")
    sample = "CoppeliaSim_2.npy"
    evaluate_test_2(test_2_dir, sample)
    