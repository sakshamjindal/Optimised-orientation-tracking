#!/usr/bin/env python
# coding: utf-8

# In[1]:


# %load_ext autoreload
# %autoreload 2


# In[2]:


import jax
jax.config.update("jax_enable_x64", True)


# In[14]:


import jax
import jax.numpy as jnp
import numpy as nnp
import os
import sys
import pickle
import matplotlib.pyplot as plt
from tqdm import tqdm

from core.optimiser import Optimiser, RiemannSGDOptimiser
from core.cost import cost_function
from core.utils import imu_preprocessing, read_data, vicon_processing, get_eulers_from_quaternions, generate_panaroma
from core.model import euler_integration


# In[4]:


def test_orientation_tracking(data_path, imu_file_name):
    """
    Runs the orientation tracking algorithm on the given data
    Args:
        data_path (str): path to the data folder
        imu_file_name (str): name of the IMU data file
    """

    imu_path = os.path.join(data_path, "imu")
    cam_path = os.path.join(data_path, "cam")


# In[5]:

    file_number = imu_file_name.split("imuRaw")[1].split(".")[0]

    # In[61]:


    # make director if not exists set_name + file_number
    result_path = "results/{}_{}".format(os.path.basename(data_path), file_number)
    if not os.path.exists(result_path):
        os.makedirs(result_path)


    # In[6]:


    print(imu_file_name)

    imu_file_path = os.path.join(imu_path, imu_file_name)
    cam_file_path = os.path.join(cam_path, "cam" + file_number + ".p")

    assert os.path.exists(imu_file_path), "File not found: {}".format(imu_file_path)
    
    imu_vals, imu_ts = imu_preprocessing(imu_file_path)

    qs = euler_integration(imu_vals, imu_ts)

    # In[7]:


    q = qs.copy()

    optimiser = RiemannSGDOptimiser(q, imu_vals, imu_ts, learning_rate=0.01)

    iter_num = 0

    logs = {
        "cost": [],
        "cost_motion": [],
        "cost_observation": [],
    }

    for i in range(50):
        optimiser.update()
        optimiser.project()
        q = optimiser.params
        cost, (cost_logs) = optimiser.cost, optimiser.cost_logs
        cost_motion, cost_observation = cost_logs["cost_motion"]/1000000000.0, cost_logs["cost_observation"]/10000.0
        print("iter: {}, cost: {:.4f}, cost_motion: {:.4f}, cost_observation: {:.4f}".format(i, cost, cost_motion, cost_observation))
        logs["cost"].append(cost)
        logs["cost_motion"].append(cost_motion)
        logs["cost_observation"].append(cost_observation)


    # In[104]:


    # save logs to file
    import pickle
    with open(os.path.join(result_path, "logs.json"), "wb") as f:
        pickle.dump(logs, f)


    # In[83]:

    # plot between cost and iterations
    plt.plot(logs["cost"], label="Cost", color = "red")
    plt.xlabel("Iterations")
    plt.ylabel("Cost")
    plt.title("Cost vs Iterations")
    plt.tight_layout()
    plt.savefig(os.path.join(result_path, "cost_vs_iterations.png"))


    # In[39]:


    euler_from_q = get_eulers_from_quaternions(q)
    euler_from_qs = get_eulers_from_quaternions(qs)


    # In[78]:


    fig, axs = plt.subplots(3, 1, figsize=(15, 10))

    # plot the first column
    axs[0].plot(imu_ts, euler_from_qs[:, 0], label="Integration", color = "red", linestyle = "dotted", linewidth = 2)
    axs[0].plot(imu_ts, euler_from_q[:, 0], label="Optimised", color = "green", linewidth = 2.5)
    axs[0].set_xlabel('Time (s)')
    axs[0].set_ylabel('Roll (rad)')
    axs[0].title.set_text("Orientation Tracking - Roll")
    axs[0].legend()

    # plot the second column
    axs[1].plot(imu_ts, euler_from_qs[:, 1], label="Integration", color = "red", linestyle = "dotted", linewidth = 2)
    axs[1].plot(imu_ts, euler_from_q[:, 1], label="Optimised", color = "green", linewidth = 2.5)
    axs[1].set_xlabel('Time (s)')
    axs[1].set_ylabel('Pitch (rad)')
    axs[1].title.set_text("Orientation Tracking - Pitch")
    axs[1].legend()

    # plot the third column
    axs[2].plot(imu_ts, euler_from_qs[:, 2], label="Integration", color = "red", linestyle = "dotted", linewidth = 2)
    axs[2].plot(imu_ts, euler_from_q[:, 2], label="Optimised", color = "green", linewidth = 2.5)
    axs[2].set_xlabel('Time (s)')
    axs[2].set_ylabel('Yaw (rad)')
    axs[2].title.set_text("Orientation Tracking - Yaw")
    axs[2].legend()

    # very tight layout and save the figure
    plt.tight_layout()
    plt.savefig("{}/orientation_horizontal.png".format(result_path))
    plt.show(block=False)

    fig, axs = plt.subplots(3, 1, figsize=(5, 10))

    # plot the first column
    axs[0].plot(imu_ts, euler_from_qs[:, 0], label="Integration", color = "red", linestyle = "dotted", linewidth = 2)
    axs[0].plot(imu_ts, euler_from_q[:, 0], label="Optimised", color = "green", linewidth = 2.5)
    axs[0].set_xlabel('Time (s)')
    axs[0].title.set_text("Orientation Tracking - Roll (rad)")
    axs[0].legend()

    # plot the second column
    axs[1].plot(imu_ts, euler_from_qs[:, 1], label="Integration", color = "red", linestyle = "dotted", linewidth = 2)
    axs[1].plot(imu_ts, euler_from_q[:, 1], label="Optimised", color = "green", linewidth = 2.5)
    axs[1].set_xlabel('Time (s)')
    axs[1].title.set_text("Orientation Tracking - Pitch (rad)")
    axs[1].legend()

    # plot the third column
    axs[2].plot(imu_ts, euler_from_qs[:, 2], label="Integration", color = "red", linestyle = "dotted", linewidth = 2)
    axs[2].plot(imu_ts, euler_from_q[:, 2], label="Optimised", color = "green", linewidth = 2.5)
    axs[2].set_xlabel('Time (s)')
    axs[2].title.set_text("Orientation Tracking - Yaw (rad)")
    axs[2].legend()

    # very tight layout and save the figure
    plt.tight_layout()
    plt.savefig("{}/orientation_vertical.png".format(result_path))
    plt.show(block=False)


    # In[45]:


    if os.path.exists(cam_file_path):
        panorama_image_optimised = generate_panaroma(cam_file_path, q, imu_ts)
        panorama_image_integration = generate_panaroma(cam_file_path, qs, imu_ts)
        
        plt.tight_layout()
        plt.imsave("{}/pan_img_optimised.png".format(result_path), panorama_image_optimised.astype(nnp.uint8))
        plt.imshow(panorama_image_optimised.astype(nnp.uint8))

        plt.tight_layout()
        plt.imsave("{}/pan_img_integration.png".format(result_path), panorama_image_integration.astype(nnp.uint8))
        plt.imshow(panorama_image_integration.astype(nnp.uint8))