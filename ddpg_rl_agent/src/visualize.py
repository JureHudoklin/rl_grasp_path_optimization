import numpy as np
import matplotlib.pyplot as plt

my_arr = np.load("/home/jure/reinforcement_ws/src/ddpg_rl_agent/src/coopelia_test5.npz.npy")


def calculate_reward_v3(force, torque):
    model_r = 0.032
    coef_friction = 0.5
    vacuum_force = 100-abs(force[2])
    print(force, "force")
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



gripper_states = []
for state in my_arr:
    gripper_state = calculate_reward_v3(state[0:3], state[3:6])
    gripper_states.append(gripper_state)

print(gripper_states)
plt.plot(np.array(gripper_states))



print(my_arr)
#plt.plot(my_arr)
plt.show()
