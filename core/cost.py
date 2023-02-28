from tqdm import tqdm
import jax.numpy as jnp
from .model import observation_model, motion_model, loss_observation, loss_motion, initialise_w


def cost_function(params, imu_vals, imu_ts, return_all=False):
    """
    Calculates the cost function.
    Args:
        params (jnp.array): Array of shape (num_time_stamps, 4) containing the quaternion values.
        imu_vals (np.array): Array of shape (num_time_stamps, 6) containing the IMU values.
        imu_ts (np.array): Array of shape (num_time_stamps,) containing the IMU timestamps.
        return_all (bool): If True, returns the cost and the cost logs. If False, returns only the cost.
    Returns:
        cost (float): The cost value.
        cost_logs (dict): A dictionary containing the cost logs.
    """
    
    cost_observation = 0
    cost_motion = 0
    num_time_stamps = params.shape[0]

    for i in (range(1, num_time_stamps)): 
        q_current = params[i]
        q_prev = params[i-1]

        ws_at_prev = initialise_w(i, imu_vals)
        dt = imu_ts[i] - imu_ts[i-1]

        obs_acceleration = (observation_model(q_current))
        cost_observation += loss_observation(obs_acceleration[1:], imu_vals[i, :3])

        q_motion = motion_model(ws_at_prev, dt, q_prev)               
        cost_motion += loss_motion(q_current, q_motion)
      
    cost = 0.5*cost_motion + 0.5*cost_observation

    cost_logs = {
        "cost_observation": (cost_observation*10000).astype("int").item(),
        "cost_motion": (cost_motion*1000000000).astype("int").item(),
    }
    
    if return_all:
        return (cost, cost_logs)

    return cost