import pickle
import sys
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm
from transforms3d.euler import quat2euler, mat2euler, quat2mat


Vref = 3300
v_sensitivity = 3.33
acc_sensitivity = 300
scale_factor_acc = Vref/(1023*acc_sensitivity)
scale_factor_vel = (jnp.pi/180)*(Vref/(1023*v_sensitivity))

def read_data(fname):
    """
    Reads the data from the pickle file.
    Args:
        fname (str): path to the pickle file
    Returns:
        d (dict): dictionary containing the data
    """
    
    d = []
    with open(fname, 'rb') as f:
        if sys.version_info[0] < 3:
            d = pickle.load(f)
        else:
            d = pickle.load(f, encoding='latin1')  # need for python 3

    return d

def imu_preprocessing(imu_file_path):
    """
    Preprocesses the IMU data
    Args:
        imu_file_path (str): path to the IMU data file
    Returns:
        imu_vals (np.ndarray): array of shape (num_time_stamps, 6) containing the IMU values
        imu_ts (np.ndarray): array of shape (num_time_stamps,) containing the IMU timestamps
    """

    imu_data = read_data(imu_file_path)
    imu_vals, imu_ts = imu_data["vals"].astype("float64"), imu_data["ts"]
    imu_vals = imu_vals.T
    imu_ts = imu_ts.T

    # do scaling, bias correction for accelerations
    ## bias correction and scaling
    imu_vals = imu_vals
    imu_vals[:, :3] = (imu_vals[:, :3] - imu_vals[:500, :3].mean(axis=0))*scale_factor_acc
    ## IMU Ax and Ay direction is flipped (due to device design), so positive acceleration in 
    ## body frame will result in negative acceleration reported by the IMU 
    imu_vals[:, 0] = -imu_vals[:, 0]
    imu_vals[:, 1] = -imu_vals[:, 1]
    imu_vals[:, 2] = imu_vals[:, 2] + 1

    # do scaling, bias correction for velocity
    imu_vals[:, 3:] = (imu_vals[:, 3:] - imu_vals[:500, 3:].mean(axis=0))
    ## change order Wz, Wx, Wy --> Wx, Wy, Wz
    imu_vals[:, [3, 4, 5]] = imu_vals[:, [4, 5, 3]]*scale_factor_vel
    return imu_vals, imu_ts


def vicon_processing(vicon_file_path):
    """
    Preprocesses the Vicon data
    Args:
        vicon_file_path (str): path to the Vicon data file
    Returns:
        vicon_vals (jnp.ndarray): array of shape (num_time_stamps, 3, 3) containing the Vicon values
        vicon_ts (jnp.ndarray): array of shape (num_time_stamps,) containing the Vicon timestamps
    """

    vicon_data = read_data(vicon_file_path)
    vicon_vals, vicon_ts = vicon_data["rots"], vicon_data["ts"]
    vicon_ts = vicon_ts.reshape(-1)

    euler_from_rot = np.zeros((len(vicon_ts), 3))
    for i in range(len(vicon_ts)):
        euler_from_rot[i] = mat2euler(vicon_vals[:, :, i]) 

    return euler_from_rot, vicon_ts

def get_eulers_from_quaternions(q):
    """
    Converts the quaternions to euler angles
    Args:
        q (jnp.ndarray): array of shape (num_time_stamps, 4) containing the quaternions
    Returns:
        euler_from_q (jnp.ndarray): array of shape (num_time_stamps, 3) containing the euler angles
    """

    num_time_stamps = q.shape[0]

    euler_from_q = np.zeros((num_time_stamps, 3))

    for i in (range(1, num_time_stamps)):
        euler_from_q[i] = quat2euler(q[i])

    return euler_from_q

def degree_to_rad(angle):
    """
    Converts the angle from degrees to radians
    Args:
        angle (float): angle in degrees
    Returns:
        angle (float): angle in radians
    """
    return angle*np.pi/180

def spherical_coords_to_cartesian_coords(theta, phi, r):
    """
    Convert spherical coordinates to cartesian coordinates
    Args:
        theta: latitude
        phi: longitude
        r: radius
    Returns:
        x: x coordinate
        y: y coordinate
        z: z coordinate
    """
    x = r * np.cos(theta) * np.cos(phi)
    y = -r * np.cos(theta) * np.sin(phi)
    z = -r * np.sin(theta)
    return x, y, z

def cartesian_coords_to_spherical_coords(x, y, z):
    """
    Convert cartesian coordinates to spherical coordinates
    Args:
        x: x coordinate
        y: y coordinate
        z: z coordinate
    Returns:
        theta: latitude
        phi: longitude
        r: radius
    """
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arcsin(-z/r)
    phi = np.arctan2(-y, x)
    return theta, phi, r

def generate_panaroma(cam_file_path, q, imu_ts):
    """
    Generates the panorama image
    Args:
        cam_file_path (str): path to the camera data file
        q (jnp.ndarray): array of shape (num_time_stamps, 4) containing the quaternions
        imu_ts (np.ndarray): array of shape (num_time_stamps,) containing the IMU timestamps
    Returns:
        panorama_image (np.ndarray): array of shape (720, 1080, 3) containing the panorama image
    """

    # read the camera data
    cam_data = read_data(cam_file_path)
    cam_vals, cam_ts = cam_data["cam"], cam_data["ts"]
    H, W, C, n_images = cam_vals.shape
    H, W, C, n_images = int(H), int(W), int(C), int(n_images)

    # initialize the panorama image
    panorama_height, panorama_width = 720, 1080
    panorama_image = np.zeros((int(panorama_height), int(panorama_width ), 3))

    # create a meshgrid of the sphereical coordinates latitude varies from -22.5 to 22.5 and longitude varies from -30 to 30
    theta_left =  degree_to_rad(-22.5)
    theta_right = degree_to_rad(22.5)
    phi_top = degree_to_rad(-30)
    phi_bottom = degree_to_rad(30)
    theta = np.linspace(theta_left, theta_right, H)
    phi = np.linspace(phi_top, phi_bottom, W)
    theta, phi = np.meshgrid(theta, phi)
    theta, phi = theta.T, phi.T
    r = np.ones((H, W))
    spherical_coords = np.stack((theta, phi, r), axis=-1)

    # convert spherical coordinates to cartesian coordinates
    cartesian_coords = spherical_coords_to_cartesian_coords(
                            spherical_coords[:, :, 0], spherical_coords[:, :, 1], spherical_coords[:, :, 2]
                        )
    cartesian_coords = np.stack(cartesian_coords, axis=-1)

    # iterate over all the images
    for i in tqdm(range(n_images)):

        # use camera timestamp to find closest-in-the-past timestamp for filtered IMU data
        cam_time = cam_ts[0][i]

        imu_closest_index = np.argmax(imu_ts>cam_time)

        # eucledean distance between camera and world frame
        w_R_c = quat2mat(q[imu_closest_index])
        # Pw = w_R_c * Pc + w_t_c
        cartesian_coords_w = w_R_c @ cartesian_coords.reshape((-1, 3)).T + np.repeat(np.array([0, 0, 0.1]).reshape((-1, 1)), H*W, axis=1)
        cartesian_coords_w = cartesian_coords_w.T.reshape((H, W, 3))

        # convert cartesian coordinates to spherical coordinates
        spherical_coords_w = cartesian_coords_to_spherical_coords(
            cartesian_coords_w[:, :, 0], cartesian_coords_w[:, :, 1], cartesian_coords_w[:, :, 2]
        )

        # select points on unit radius sphere
        spherical_coords_w = np.stack(spherical_coords_w, axis=-1)
        spherical_coords_w = spherical_coords_w[:, :, 0:2]

        # project spherical coords to image plane
        spherical_coords_w[:, :, 0] = (np.pi/2 + spherical_coords_w[:, :, 0]) / np.pi
        spherical_coords_w[:, :, 1] = (np.pi + spherical_coords_w[:, :, 1]) / (2 * np.pi)
        
        # scale to image size
        spherical_coords_w[:, :, 0] = spherical_coords_w[:, :, 0]*panorama_height
        spherical_coords_w[:, :, 1] = spherical_coords_w[:, :, 1]*panorama_width

        # convert to ints for indexing
        cylindrical_coords_w = spherical_coords_w.astype(int)

        # copy image to panorama
        panorama_image = panorama_image.astype(int)
        panorama_image[cylindrical_coords_w[:, :, 0], cylindrical_coords_w[:, :, 1]] = cam_vals[:, :, 0:3, i]

    return panorama_image
        
