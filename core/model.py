
import jax.numpy as jnp
import jax
import numpy as np
from tqdm import tqdm

from .quartenions import norm, log, exp, q_mult , q_inverse
from transforms3d.euler import quat2euler, mat2euler


@jax.jit
def observation_model(q):
    """Returns the observation model for a given quaternion."""
    return q_mult(q_mult(q_inverse(q), jnp.array([0, 0, 0, 1])), q)

@jax.jit
def loss_observation(obs_acceleration, a_t):
    """Returns the loss for the observation model."""
    return norm(obs_acceleration - a_t)**2

@jax.jit
def motion_model(w, dt, prev_q):
    """Returns the motion model for a given quaternion."""
    return (q_mult)(prev_q, exp(w*dt/2))

@jax.jit
def loss_motion(q_current, q_motion):
    """Returns the loss for the motion model."""
    return norm(2*log(q_mult(q_inverse(q_current), q_motion)))**2

@jax.jit
def initialise_w(index, imu_vals):
    """Initialises the angular velocity vector for the motion model."""
    w = jnp.zeros(4)
    w = w.at[1:4].set(imu_vals[index-1, 3:])
    return w

@jax.jit
def update_qs(q, q_update, index):
    """Updates the quaternion array."""
    return q.at[index].set(q_update)


def euler_integration(imu_vals, imu_ts):
    """
    Performs euler integration on the IMU values.
    Args:
        imu_vals (np.array): Array of shape (num_time_stamps, 6) containing the IMU values.
        imu_ts (np.array): Array of shape (num_time_stamps,) containing the IMU timestamps.
    Returns:
        qs (jnp.array): Array of shape (num_time_stamps, 4) containing the quaternion values.
    """

    num_time_stamps = len(imu_ts)

    qs = jnp.zeros((num_time_stamps, 4))
    qs = qs.at[0].set([1, 0, 0, 0])

    euler_from_q = np.zeros((num_time_stamps, 3))

    for i in (range(1, num_time_stamps)):
        q_prev = qs[i-1]
        ws_at_prev = initialise_w(i, imu_vals)
        dt = imu_ts[i] - imu_ts[i-1]
        
        q_motion = motion_model(ws_at_prev, dt, q_prev)
        qs = update_qs(qs, q_motion, i)

    return qs