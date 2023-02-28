import jax
import jax.numpy as jnp

@jax.jit
def norm(q):
    """
    Calculates the norm of a quaternion.
    """
    return jnp.sqrt(jnp.sum(q**2))

@jax.jit
def exp(q, eps = 0.0001): 
    """
    Performs the exponential map on a quaternion.
    Args:
        q = [qs, qv] (jnp.array): quarternion on which exponential map is calculated.
        eps (float): amount of perturbation added to the quartenions to avoid numerical instabilities.
    Returns:
        new_q (jnp.array): quarternion that is calculated from the exponential map.
    """
    qs, qv = q[0:1], q[1:]
    new_qs = jnp.exp(qs)*jnp.cos(norm(qv))
    new_qv = (jnp.exp(qs)*(qv/(norm(qv)))*jnp.sin(norm(qv)))
    return jnp.concatenate([new_qs, new_qv])

@jax.jit
def log(q, eps = 0.0001):
    """
    Performs the logarithm map on a quaternion.
    Args:
        q = [qs, qv] (jnp.array): quarternion on which logarithm map is calculated.
        eps (float): amount of perturbation added to the quartenions to avoid numerical instabilities.
    Returns:
        new_q (jnp.array): quarternion that is calculated from the logarithm map.
    """
    q = q + eps
    qs, qv = q[0:1], q[1:]
    new_qs = jnp.array([jnp.log(norm(q))])
    new_qv = (qv/(jnp.linalg.norm(qv))*jnp.arccos(qs/norm(q)))

    return jnp.concatenate([new_qs, new_qv])

@jax.jit
def conjugate(q):
    """
    Performs the conjugate on a quaternion.
    Args:
        q: (jnp.array) The quaternion whose conjugate is calculated.
    Returns:
        new_q: (jnp.array) The quaternion that is calculated from the conjugate.
    """
    qs, qv = q[0:1], q[1:]
    new_qs = qs
    new_qv = -qv
    return jnp.concatenate([new_qs, new_qv])

@jax.jit
def q_inverse(q):
    """
    Performs the inverse on a quaternion.
    Args:
        q: (jnp.array) The quaternion whose inverse is calculated.
    Returns:
        new_q: (jnp.array) The quaternion that is calculated from the inverse.
    """
    return conjugate(q)/(norm(q))**2


@jax.jit
def q_mult(q1, q2):

    """
    Performs the multiplication on two quaternions.
    Args:
        q1: (jnp.array) The first quaternion.
        q2: (jnp.array) The second quaternion.
    Returns:
        new_q: (jnp.array) The quaternion that is calculated from the multiplication.
    """

    qs1, qv1 = q1[0:1], q1[1:]
    qs2, qv2 = q2[0:1], q2[1:]
    new_qs = qs1*qs2 - jnp.dot(qv1, qv2)
    new_qv = qs1*qv2 + qs2*qv1 + jnp.cross(qv1, qv2)

    return jnp.concatenate([new_qs, new_qv])

if __name__ == "__main__":

    def run_quaternion_tests():
        test_q = jnp.array([1, 1, 0.000, 0.0000]).astype("float64")
        print(conjugate(test_q))
        print(log(test_q))
        print(exp(test_q))
        print(norm(test_q))

    run_quaternion_tests()



            
