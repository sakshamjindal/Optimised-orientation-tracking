from jax import value_and_grad
from .cost import cost_function
import jax.numpy as jnp

class Optimiser():
    def __init__(self, params, imu_vals, imu_ts, learning_rate=0.01):
        """
        Base class for the optimisers.
        Args:
            params (jnp.ndarray): the parameters that are to be optimised (quaternions)
            imu_vals (np.ndarray): the IMU values that are used to calculate the cost function.
            imu_ts (np.ndarray): the time stamps of the IMU values.
            learning_rate (float): the learning rate of the optimiser.
        """     

        self.params = params
        self.imu_vals = imu_vals
        self.imu_ts = imu_ts
        self.learning_rate = learning_rate

    def update(self):
        """
        Updates the parameters of the optimiser.
        """
        cost_grad = self.calculate_gradients()
        self.params = self.params - self.learning_rate * cost_grad

    def project(self):
        """
        Projects the parameters of the optimiser to the unit sphere.
        """
        self.params = (self.params/jnp.linalg.norm(self.params, axis=1).reshape(-1, 1))

    def calculate_gradients(self):
        """
        Calculates the gradients of the cost function.
        Returns:
            cost_grad (jnp.ndarray): the gradients of the cost function.
        """
        (self.cost, self.cost_logs), cost_grad = value_and_grad(cost_function, has_aux=True)(
                                                        self.params, 
                                                        self.imu_vals, 
                                                        self.imu_ts,
                                                        return_all=True
                                                    )
        return cost_grad


class RiemannSGDOptimiser(Optimiser):
    
    def __init__(self, params, imu_vals, imu_ts, learning_rate=0.01):
        """
        Riemannian Stochastic Gradient Descent Optimiser.
        Args:
            params (jnp.ndarray): the parameters that are to be optimised (quaternions)
            imu_vals (np.ndarray): the IMU values that are used to calculate the cost function.
            imu_ts (np.ndarray): the time stamps of the IMU values.
            learning_rate (float): the learning rate of the optimiser.
        Returns:
            RiemannSGDOptimiser: the Riemannian Stochastic Gradient Descent Optimiser.
        """
        super().__init__(params, imu_vals, imu_ts, learning_rate)

    def update(self):
        """
        Updates the parameters of the optimiser.
        """
        cost_grad = self.calculate_gradients()
        grad_vec = -self.learning_rate*cost_grad
        tangent_vec =  grad_vec - (jnp.sum((grad_vec*self.params), axis=1).reshape(-1, 1))*self.params
        tangent_vec_norm = jnp.linalg.norm(tangent_vec, axis=1).reshape(-1, 1)
        tangent_vec = tangent_vec/tangent_vec_norm
        self.params = self.params*jnp.cos(tangent_vec_norm) + tangent_vec*jnp.sin(tangent_vec_norm)
