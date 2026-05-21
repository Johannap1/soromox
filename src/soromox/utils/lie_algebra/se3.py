__all__ = [
    "tilde_SE3",
    "hat_SE3",
    "exp_SE3",
    "log_SE3",
    "exp_gn_SE3",
    "adjoint_se3",
    "coadjoint_se3",
    "Adjoint_g_SE3",
    "Adjoint_g_inv_SE3",
    "Adjoint_gi_se3",
    "Adjoint_gi_se3_inv",
    "Tangent_gi_se3",
    "Tangent_derivative_gi_se3",
]

import jax.numpy as jnp
from jax import Array, lax


def _rotational_strain_magnitude(xi: Array, eps: float | Array) -> Array:
    """
    Computes the magnitude of the rotational strain component of a 6D vector.

    Args:
        xi (Array): shape (6,) or (6, 1)
            Screw coordinates [omega, v] in se(3).
            The first three elements correspond to the angular component,
            and the last three elements correspond to the linear component.

    Returns:
        theta: scalar
            Magnitude of the rotational strain component.
    """
    k = xi[:3]

    # not differentiable in reverse mode at k=0
    # theta = jnp.linalg.norm(k)

    # differentiable in reverse mode at k=0
    theta_sq = jnp.dot(k, k)
    theta = lax.cond(
        theta_sq <= eps**2,
        lambda _: jnp.zeros((), dtype=xi.dtype),
        lambda _: jnp.sqrt(theta_sq),
        operand=None,
    )
    return theta


def tilde_SE3(vec3: Array) -> Array:
    """
    Computes the tilde operator of SE(3) for a 3D vector.

    Args:
        vec3 (Array): shape (3,) or (3, 1)
            3D vector [x, y, z].

    Returns:
        tilde: shape (3, 3)
            A 3x3 matrix representing the tilde operator of the input vector.
    """
    vec3 = vec3.reshape(-1)  # Ensure vec3 is a 1D array

    # Extract components of the vector
    x, y, z = vec3.flatten()

    # Construct the tilde operator
    tilde = jnp.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return tilde


def hat_SE3(vec6: Array) -> Array:
    """
    Computes the hat operator for a 6D vector of se(3).

    Args:
        vec6 (Array): shape (6,) or (6, 1)
            Screw coordinates [omega, v] in se(3).
            The first three elements correspond to the angular component,
            and the last three elements correspond to the linear component.

    Returns:
        hat: shape (4, 4)
            Matrix representation of the Lie algebra element.
    """
    vec6 = vec6.reshape(-1)  # Ensure vec6 is a 1D array

    ang = vec6[:3].reshape((3, 1))  # Angular as a (3,1) vector
    lin = vec6[3:].reshape((3, 1))  # Linear as a (3,1) vector

    angtilde = tilde_SE3(ang)  # Tilde operator for angular part

    hat = jnp.block([[angtilde, lin], [jnp.zeros((1, 3)), jnp.zeros((1, 1))]])

    return hat


def exp_SE3(vec6: Array) -> Array:
    """
    Construct an SE(3) homogeneous transform from raw pose coordinates.

    This helper keeps the historical ``exp_SE3`` name, but it is not the
    matrix exponential of ``hat_SE3(vec6)`` for a general twist. The rotational
    coordinates are interpreted as ZYX Euler angles and the translational
    entries are inserted directly into the homogeneous transform:
    ``vec6 = [phi, theta, psi, x, y, z]`` maps to
    ``[[R_zyx(phi, theta, psi), [x, y, z]], [0, 0, 0, 1]]``.

    Use :func:`exp_gn_SE3` when ``vec6`` represents a Lie-algebra twist whose
    translational component must be integrated by the SE(3) exponential.

    Args:
        vec6 (Array): shape (6,) or (6, 1)
            Pose coordinates ``[phi, theta, psi, x, y, z]`` with ZYX Euler
            convention for the rotation part and raw translation ``(x, y, z)``.

    Returns:
        g: shape (4, 4)
            Homogeneous pose with ZYX Euler rotation and translation
            ``[x, y, z]``.
    """
    vec6 = vec6.reshape(-1)  # Ensure vec6 is a 1D array

    phi = vec6[0]
    theta = vec6[1]
    psi = vec6[2]
    cosphi, sinphi = jnp.cos(phi), jnp.sin(phi)
    costheta, sintheta = jnp.cos(theta), jnp.sin(theta)
    cospsi, sinpsi = jnp.cos(psi), jnp.sin(psi)

    p = vec6[3:].reshape((3, 1))

    Rphi = jnp.array([[cosphi, -sinphi, 0], [sinphi, cosphi, 0], [0, 0, 1]])
    Rtheta = jnp.array([[1, 0, 0], [0, costheta, -sintheta], [0, sintheta, costheta]])
    Rpsi = jnp.array([[cospsi, -sinpsi, 0], [sinpsi, cospsi, 0], [0, 0, 1]])
    # Combine the rotations
    R = Rpsi @ Rtheta @ Rphi  # Rotation matrix

    g = jnp.block([[R, p], [jnp.zeros((1, 3)), jnp.ones((1, 1))]])

    return g


def log_SE3(g: Array, eps: float | Array) -> Array:
    """
    Computes the logarithm map from SE(3) to se(3), i.e., extracts the twist from a transformation matrix.

    Args:
        g (Array): shape (4, 4)
            Homogeneous transform in SE(3).
        eps (Union[float, Array]): tolerance to avoid division by zero in small angle approximations.

    Returns:
        log: shape (6,)
            Twist coordinates corresponding to ``g``.
    """
    R = g[:3, :3]
    p = g[:3, 3].reshape((3, 1))

    # Compute the rotation angle
    trace_R = jnp.trace(R)
    skew_part = R - R.T
    skew_norm_sq = jnp.sum(jnp.square(skew_part))
    cos_theta = (trace_R - 1.0) * 0.5
    cos_theta = jnp.clip(cos_theta, -1.0, 1.0)
    sin_theta = jnp.sqrt(jnp.maximum(0.0, skew_norm_sq) * 0.125)
    eps_scalar = 1e8 * jnp.asarray(eps, dtype=R.dtype)
    is_small_angle = sin_theta <= jnp.sin(eps_scalar)
    # jax.debug.print("sin(theta): {sinth}, sin(eps): {seps}", sinth=sin_theta, seps=jnp.sin(eps_scalar))
    # jax.debug.print("is_small_angle: {b}", b=is_small_angle)

    theta = lax.cond(
        is_small_angle,
        lambda _: jnp.zeros((), dtype=R.dtype),
        lambda _: jnp.arctan2(sin_theta, cos_theta),
        operand=None,
    )

    def _omega_hat_small(args):
        skew, _ = args
        return 0.5 * skew

    def _omega_hat_general(args):
        skew, angle = args
        sin_theta = jnp.sin(angle)
        factor = angle / (2.0 * sin_theta)
        return factor * skew

    omega_hat = lax.cond(
        is_small_angle,
        _omega_hat_small,
        _omega_hat_general,
        (skew_part, theta),
    )

    omega = jnp.array(
        [omega_hat[2, 1], omega_hat[0, 2], omega_hat[1, 0]], dtype=R.dtype
    ).reshape((3, 1))

    # Compute V inverse (Jacobian inverse)
    omega_tilde = omega_hat

    def _compute_V_inv_small(args):
        omega_local, _ = args
        omega_sq = omega_local @ omega_local
        return jnp.eye(3, dtype=R.dtype) - 0.5 * omega_local + (1.0 / 12.0) * omega_sq

    def _compute_V_inv_general(args):
        omega_local, angle = args
        sin_theta = jnp.sin(angle)
        cos_theta_local = jnp.cos(angle)
        omega_sq = omega_local @ omega_local
        A = jnp.eye(3, dtype=R.dtype) - 0.5 * omega_local
        B = (1.0 / (angle**2)) * (
            1.0 - (angle * sin_theta) / (2.0 * (1.0 - cos_theta_local))
        )
        return A + B * omega_sq

    V_inv = lax.cond(
        is_small_angle,
        _compute_V_inv_small,
        _compute_V_inv_general,
        (omega_tilde, theta),
    )

    v = V_inv @ p

    log = jnp.vstack([omega, v]).reshape(-1)
    return log


def exp_gn_SE3(vec6: Array, eps: float | Array) -> Array:
    """
    Function to compute the exponential map of the Magnus expansion.

    Args:
        vec6 (Array): shape (6,) or (6, 1)
            Screw coordinates used in the Magnus expansion.
        eps (Union[float, Array]): small value to avoid division by zero in the series expansion.
    Returns:
        g (Array): shape (4, 4)
            Homogeneous transform obtained from the Magnus expansion.
    """
    theta = _rotational_strain_magnitude(vec6, eps)
    vec6_hat = hat_SE3(vec6)  # Compute the hat

    costheta = jnp.cos(theta)
    sintheta = jnp.sin(theta)

    g = lax.cond(
        theta <= eps,
        lambda _: (
            jnp.eye(4)  # Avoid division by zero
            + vec6_hat
            + 1 / 2 * jnp.linalg.matrix_power(vec6_hat, 2)
            + 1 / 6 * jnp.linalg.matrix_power(vec6_hat, 3)
        ),
        lambda _: (
            jnp.eye(4)
            + vec6_hat
            + 1
            / jnp.power(theta, 2)
            * (1 - costheta)
            * jnp.linalg.matrix_power(vec6_hat, 2)
            + 1
            / jnp.power(theta, 3)
            * (theta - sintheta)
            * jnp.linalg.matrix_power(vec6_hat, 3)
        ),
        operand=None,
    )

    return g


def adjoint_se3(vec6: Array) -> Array:
    """
    Computes the adjoint representation of a vector of se(3).

    Args:
        vec6 (Array): shape (6,) or (6, 1)
            Screw coordinates [omega, v] in se(3).
            The first three elements correspond to the angular component,
            and the last three elements correspond to the linear component.

    Returns:
        Array: shape (6, 6)
            Adjoint representation of ``vec6``.
    """
    vec6 = vec6.reshape(-1)  # Ensure vec6 is a 1D array

    ang = vec6[:3].reshape((3, 1))  # Angular as a (3,1) vector
    lin = vec6[3:].reshape((3, 1))  # Linear as a (3,1) vector

    angtilde = tilde_SE3(ang)  # Tilde operator for angular part
    lintilde = tilde_SE3(lin)  # Tilde operator for linear part

    ad = jnp.block([[angtilde, jnp.zeros((3, 3))], [lintilde, angtilde]])

    return ad


def coadjoint_se3(vec6: Array) -> Array:
    """
    Computes the co-adjoint representation of a vector of se(3).

    Args:
        vec6 (Array): shape (6,) or (6, 1)
            Screw coordinates [omega, v] in se(3).
            The first three elements correspond to the angular component,
            and the last three elements correspond to the linear component.

    Returns:
        Array: shape (6, 6)
            Co-adjoint representation of ``vec6``.
    """
    vec6 = vec6.reshape(-1)  # Ensure vec6 is a 1D array

    ang = vec6[:3].reshape((3, 1))  # Angular as a (3,1) vector
    lin = vec6[3:].reshape((3, 1))  # Linear as a (3,1) vector

    angtilde = tilde_SE3(ang)  # Tilde operator for angular part
    lintilde = tilde_SE3(lin)  # Tilde operator for linear part

    coad = jnp.block([[angtilde, lintilde], [jnp.zeros((3, 3)), angtilde]])

    return coad


def Adjoint_g_SE3(mat4: Array) -> Array:
    """
    Computes the adjoint representation of a 4x4 matrix.

    Args:
        mat4 (Array): shape (4, 4)
            Homogeneous transform in SE(3).

    Returns:
        Array: shape (6, 6)
            Adjoint matrix associated with ``mat4``.
    """
    R = mat4[:3, :3]  # Extract the angular part (top-left 3x3 block)
    t = mat4[:3, 3].reshape((3, 1))  # Extract the linear part (top-right column)

    ttilde = tilde_SE3(t)  # Tilde operator for linear part

    Ad = jnp.block([[R, jnp.zeros((3, 3))], [ttilde @ R, R]])

    return Ad


def Adjoint_g_inv_SE3(mat4: Array) -> Array:
    """
    Computes the adjoint representation of a 4x4 matrix.

    Args:
        mat4 (Array): shape (4, 4)
            Homogeneous transform in SE(3).

    Returns:
        Array: shape (6, 6)
            Inverse adjoint matrix associated with ``mat4``.
    """
    R = mat4[:3, :3]  # Extract the angular part (top-left 3x3 block)
    t = mat4[:3, 3].reshape((3, 1))  # Extract the linear part (top-right column)

    ttilde = tilde_SE3(t)  # Tilde operator for linear part
    R_inv = jnp.transpose(R)  # Since R is a rotation matrix, R^-1=R^T

    # Construct the inverse Adjoint matrix
    Ad_inv = jnp.block([[R_inv, jnp.zeros((3, 3))], [-R_inv @ ttilde, R_inv]])

    return Ad_inv


def Adjoint_gi_se3(
    xi_i: Array,
    s_i: Array,
    eps: float | Array,
) -> Array:
    """
    Computes the adjoint representation of a position of a points at s_i (local curvilinear coordinate)
    along a rod in SE(3) deformed ine the current segment according to a strain vector xi_i.

    Args:
        xi_i (Array): shape (6,) or (6, 1)
            Constant strain vector in the segment, [omega, v].
        s_i (Array): scalar arclength position along the segment.
        eps (Union[float, Array]): small value to avoid division by zero in the series expansion.

    Returns:
        Array: shape (6, 6)
            Adjoint matrix evaluated at ``s_i``.
    """
    theta = _rotational_strain_magnitude(xi_i, eps)
    adjoint_xi_i = adjoint_se3(xi_i)

    def _series_branch() -> Array:
        return jnp.eye(6) + s_i * adjoint_xi_i

    def _general_branch(theta_val: Array) -> Array:
        cos_theta = jnp.cos(s_i * theta_val)
        sin_theta = jnp.sin(s_i * theta_val)

        adjoint_xi_i_square = adjoint_xi_i @ adjoint_xi_i
        adjoint_xi_i_cube = adjoint_xi_i_square @ adjoint_xi_i
        adjoint_xi_i_quad = adjoint_xi_i_cube @ adjoint_xi_i

        term1 = (
            (3 * sin_theta - s_i * theta_val * cos_theta)
            / (2 * theta_val)
            * adjoint_xi_i
        )
        term2 = (
            (4 - 4 * cos_theta - s_i * theta_val * sin_theta)
            / (2 * theta_val**2)
            * adjoint_xi_i_square
        )
        term3 = (
            (sin_theta - s_i * theta_val * cos_theta)
            / (2 * theta_val**3)
            * adjoint_xi_i_cube
        )
        term4 = (
            (2 - 2 * cos_theta - s_i * theta_val * sin_theta)
            / (2 * theta_val**4)
            * adjoint_xi_i_quad
        )

        return jnp.eye(6) + term1 + term2 + term3 + term4

    Ad = lax.cond(
        theta <= eps,
        lambda _: _series_branch(),
        lambda _: _general_branch(theta),
        operand=None,
    )

    return Ad


def Adjoint_gi_se3_inv(
    xi_i: Array,
    s_i: Array,
    eps: float | Array,
) -> Array:
    """
    Computes the adjoint representation of a position of a points at s_i (local curvilinear coordinate)
    along a rod in SE(3) deformed ine the current segment according to a strain vector xi_i.

    Args:
        xi_i (Array): shape (6,) or (6, 1)
            Constant strain vector in the segment, [omega, v].
        s_i (Array): scalar arclength position along the segment.
        eps (Union[float, Array]): small value to avoid division by zero in the series expansion.

    Returns:
        Array: shape (6, 6)
            Inverse adjoint matrix evaluated at ``s_i``.
    """
    Ad = Adjoint_gi_se3(
        xi_i, s_i, eps=eps
    )  # Adjoint representation of the input vector

    # Extract R and -Jt from the Adjoint matrix
    R = Ad[:3, :3]
    ttildeR = Ad[3:, :3]

    # Compute the inverse using the Schur complement
    R_inv = jnp.transpose(R)  # Since R is a rotation matrix, R^-1=R^T
    ttilde = ttildeR @ R_inv  # Compute the tilde operator for the linear part
    # Construct the inverse Adjoint matrix
    Ad_inv = jnp.block([[R_inv, jnp.zeros((3, 3))], [-R_inv @ ttilde, R_inv]])

    return Ad_inv


def Tangent_gi_se3(
    xi_i: Array,
    s_i: Array,
    eps: float | Array,
) -> Array:
    """
    Computes the tangent representation of a position of a points at s_i (local curvilinear coordinate)
    along a rod in SE(3) deformed in the current segment according to a strain vector xi_i.

    Args:
        xi_i (Array): shape (6,) or (6, 1)
            Constant strain vector in the segment, [omega, v].
        s_i (Array): scalar arclength position along the segment.
        eps (Union[float, Array]): small value to avoid division by zero in the series expansion.

    Returns:
        T (Array): shape (6, 6)
            A 6x6 matrix representing the tangent transformation of the input screw vector at the specified position.
    """
    theta = _rotational_strain_magnitude(xi_i, eps)
    adjoint_xi_i = adjoint_se3(xi_i)

    def _series_branch() -> Array:
        return s_i * jnp.eye(6) + 0.5 * s_i**2 * adjoint_xi_i

    def _general_branch(theta_val: Array) -> Array:
        cos_theta = jnp.cos(s_i * theta_val)
        sin_theta = jnp.sin(s_i * theta_val)

        adjoint_xi_i_square = adjoint_xi_i @ adjoint_xi_i
        adjoint_xi_i_cube = adjoint_xi_i_square @ adjoint_xi_i
        adjoint_xi_i_quad = adjoint_xi_i_cube @ adjoint_xi_i

        term1 = (
            (4 - 4 * cos_theta - s_i * theta_val * sin_theta)
            / (2 * theta_val**2)
            * adjoint_xi_i
        )
        term2 = (
            (4 * s_i * theta_val - 5 * sin_theta + s_i * theta_val * cos_theta)
            / (2 * theta_val**3)
            * adjoint_xi_i_square
        )
        term3 = (
            (2 - 2 * cos_theta - s_i * theta_val * sin_theta)
            / (2 * theta_val**4)
            * adjoint_xi_i_cube
        )
        term4 = (
            (2 * s_i * theta_val - 3 * sin_theta + s_i * theta_val * cos_theta)
            / (2 * theta_val**5)
            * adjoint_xi_i_quad
        )

        return s_i * jnp.eye(6) + term1 + term2 + term3 + term4

    T = lax.cond(
        theta <= eps,
        lambda _: _series_branch(),
        lambda _: _general_branch(theta),
        operand=None,
    )

    return T


def Tangent_derivative_gi_se3(
    xi_i: Array,
    xid_i: Array,
    s_i: Array,
    eps: float | Array,
) -> Array:
    """
    Computes the tangent derivative representation of a position of a points at s_i (local curvilinear coordinate)
    along a rod in SE(3) deformed in the current segment according to a strain vector xi_i and its derivative xid_i.

    Args:
        xi_i (Array): shape (6,) or (6, 1)
            Constant strain vector in the segment, [omega, v].
        xid_i (Array): shape (6,) or (6, 1)
            Time derivative of the strain vector.
        s_i (Array): scalar arclength position along the segment.
        eps (Union[float, Array]): small value to avoid division by zero in the series expansion.

    Returns:
        Td (Array): shape (6, 6)
            A 6x6 matrix representing the tangent derivative transformation of the input screw vector at the specified position.
    """
    k = xi_i[:3]
    kd = xid_i[:3]

    theta = _rotational_strain_magnitude(xi_i, eps)
    adjoint_xi_i = adjoint_se3(xi_i)
    adjoint_xid_i = adjoint_se3(xid_i)

    def _adjoint_powers_with_derivatives(max_power: int):
        current = jnp.eye(6)
        dot_current = jnp.zeros_like(adjoint_xi_i)
        powers = [current]
        dot_powers = [dot_current]

        for _ in range(max_power):
            dot_next = dot_current @ adjoint_xi_i + current @ adjoint_xid_i
            current = current @ adjoint_xi_i
            powers.append(current)
            dot_powers.append(dot_next)
            dot_current = dot_next

        return powers, dot_powers

    def _series_branch() -> Array:
        return 0.5 * s_i**2 * adjoint_xid_i

    def _general_branch(theta_val: Array) -> Array:
        cos_theta = jnp.cos(s_i * theta_val)
        sin_theta = jnp.sin(s_i * theta_val)

        powers, dot_powers = _adjoint_powers_with_derivatives(4)

        adjoint_xi_i_square = powers[2]
        adjoint_xi_i_cube = powers[3]
        adjoint_xi_i_quad = powers[4]
        adjoint_dot_xi_i = dot_powers[1]
        adjoint_dot_xi_i_square = dot_powers[2]
        adjoint_dot_xi_i_cube = dot_powers[3]
        adjoint_dot_xi_i_quad = dot_powers[4]

        thetad = jnp.dot(kd, k) / theta_val

        coeff_theta_common = (
            -8
            + (8 - s_i**2 * theta_val**2) * cos_theta
            + 5 * s_i * theta_val * sin_theta
        )
        coeff_theta_alt = (
            -8 * s_i * theta_val
            + (15 - s_i**2 * theta_val**2) * sin_theta
            - 7 * s_i * theta_val * cos_theta
        )

        coeff1_theta = thetad / (2 * theta_val**3) * coeff_theta_common
        coeff1 = (4 - 4 * cos_theta - s_i * theta_val * sin_theta) / (2 * theta_val**2)

        coeff2_theta = thetad / (2 * theta_val**4) * coeff_theta_alt
        coeff2 = (4 * s_i * theta_val - 5 * sin_theta + s_i * theta_val * cos_theta) / (
            2 * theta_val**3
        )

        coeff3_theta = thetad / (2 * theta_val**5) * coeff_theta_common
        coeff3 = (2 - 2 * cos_theta - s_i * theta_val * sin_theta) / (2 * theta_val**4)

        coeff4_theta = thetad / (2 * theta_val**6) * coeff_theta_alt
        coeff4 = (2 * s_i * theta_val - 3 * sin_theta + s_i * theta_val * cos_theta) / (
            2 * theta_val**5
        )

        term1 = coeff1_theta * adjoint_xi_i + coeff1 * adjoint_dot_xi_i
        term2 = coeff2_theta * adjoint_xi_i_square + coeff2 * adjoint_dot_xi_i_square
        term3 = coeff3_theta * adjoint_xi_i_cube + coeff3 * adjoint_dot_xi_i_cube
        term4 = coeff4_theta * adjoint_xi_i_quad + coeff4 * adjoint_dot_xi_i_quad

        return term1 + term2 + term3 + term4

    Td = lax.cond(
        theta <= eps,
        lambda _: _series_branch(),
        lambda _: _general_branch(theta),
        operand=None,
    )
    return Td

def coadjointbar_se3(vec6: Array) -> Array:
    """
    Computes the co-adjoint-bar representation of a vector of se(3).

    """
    vec6 = vec6.reshape(-1)  # Ensure vec6 is a 1D array

    ang = vec6[:3].reshape((3, 1))  # Angular as a (3,1) vector
    lin = vec6[3:].reshape((3, 1))  # Linear as a (3,1) vector

    angtilde = tilde_SE3(ang)  # Tilde operator for angular part
    lintilde = tilde_SE3(lin)  # Tilde operator for linear part

    coadbar = -jnp.block([[angtilde, lintilde], [lintilde, jnp.zeros((3, 3))]])

    return coadbar

def gvs_SoftJointDifferentialKinematics_Z4(
        eps,
        H,
        Phi_Z1,
        Phi_Z2,
        xi_star_Z1,
        xi_star_Z2,
        q,
        qd,
        qdd) -> tuple[
            Array,
            Array,
            Array,
            Array,
            Array,
            Array,
            Array,
            Array,
            Array,
            Array,
            Array,
            Array
            ]:
    
    I_theta = jnp.diag(jnp.array([1, 1, 1, 0, 0, 0]))
    xi_Z1 = Phi_Z1 @ q + xi_star_Z1
    xi_Z2 = Phi_Z2 @ q + xi_star_Z2
    xid_Z1 = Phi_Z1 @ qd
    xid_Z2 = Phi_Z2 @ qd
    xidd_Z1 = Phi_Z1 @ qdd
    xidd_Z2 = Phi_Z2 @ qdd

    Omega = (H/2) * (xi_Z1 + xi_Z2) + (jnp.sqrt(3) * H**2 / 12) * (adjoint_se3(xi_Z1) @ xi_Z2)

    Z = (H/2) * (Phi_Z1 + Phi_Z2) + (jnp.sqrt(3) * H**2 / 12) * (
        adjoint_se3(xi_Z1) @ Phi_Z2 - adjoint_se3(xi_Z2) @ Phi_Z1
        )
    
    Omegad  = Z @ qd

    Zd      = (jnp.sqrt(3) * H**2 / 12) * (adjoint_se3(xid_Z1) @ Phi_Z2 - adjoint_se3(xid_Z2) @ Phi_Z1)

    Zdd     = (jnp.sqrt(3) * H**2 / 12) * (adjoint_se3(xidd_Z1) @ Phi_Z2 - adjoint_se3(xidd_Z2) @ Phi_Z1)

    adjOmegap  = jnp.zeros((24, 6))
    adjOmegapd = jnp.zeros((24, 6))

    k = Omega[:3]
    kd = Omegad[:3]
    theta = _rotational_strain_magnitude(Omega, eps)
    thetad = (kd @ k) / theta

    Omegahat = hat_SE3(Omega)
    Omegahatp2 = Omegahat @ Omegahat
    Omegahatp3 = Omegahatp2 @ Omegahat

    adjOmegap = adjOmegap.at[0:6, :].set(adjoint_se3(Omega))
    adjOmegap = adjOmegap.at[6:12, :].set(adjOmegap[0:6, :] @ adjOmegap[0:6, :])
    adjOmegap = adjOmegap.at[12:18, :].set(adjOmegap[6:12, :] @ adjOmegap[0:6, :])
    adjOmegap = adjOmegap.at[18:24, :].set(adjOmegap[12:18, :] @ adjOmegap[0:6, :])

    adjOmegapd = adjOmegapd.at[0:6, :].set(adjoint_se3(Omegad))
    adjOmegapd = adjOmegapd.at[6:12, :].set(adjOmegapd[0:6, :] @ adjOmegap[0:6, :] +
    adjOmegap[0:6, :] @ adjOmegapd[0:6, :]
    )
    adjOmegapd = adjOmegapd.at[12:18, :].set(adjOmegapd[6:12, :] @ adjOmegap[0:6, :] +
    adjOmegap[6:12, :] @ adjOmegapd[0:6, :]
    )
    adjOmegapd = adjOmegapd.at[18:24, :].set(adjOmegapd[12:18, :] @ adjOmegap[0:6, :] +
    adjOmegap[12:18, :] @ adjOmegapd[0:6, :]
    )

    def _g_series_branch() -> tuple[Array, Array, Array, Array, Array, Array]:
        g = jnp.eye(4) + Omegahat + Omegahatp2 / 2 + Omegahatp3 / 6
        f = jnp.array([1/2, 1/6, 1/24, 1/120])
        fd  = jnp.zeros((4,))
        fdd = jnp.zeros((4,))

        T = (
            jnp.eye(6)
            + f[0] * adjOmegap[0:6, :]
            + f[1] * adjOmegap[6:12, :]
            + f[2] * adjOmegap[12:18, :]
            + f[3] * adjOmegap[18:24, :]
        )

        Td = (
            f[0] * adjOmegapd[0:6, :]
            + f[1] * adjOmegapd[6:12, :]
            + f[2] * adjOmegapd[12:18, :]
            + f[3] * adjOmegapd[18:24, :]
        )

        return (f, fd, fdd, g, T, Td)
    
    def _g_general_branch(theta: Array) -> tuple[Array, Array, Array, Array, Array, Array]:
        tp2        = theta * theta
        tp3        = tp2 * theta
        tp4        = tp3 * theta
        tp5        = tp4 * theta
        tp6        = tp5 * theta
        tp7        = tp6 * theta
        costheta = jnp.cos(theta)
        sintheta = jnp.sin(theta)

        t1 = theta * sintheta
        t2 = theta * costheta
        t3 = -8 + (8 - tp2) * costheta + 5 * t1
        t4 = -8 * theta + (15 - tp2) * sintheta - 7 * t2
        t3d = 5 * sintheta + sintheta * (tp2 - 8) + 3 * theta * costheta
        t4d = -8 + 5 * theta * sintheta + (15 - tp2) * costheta - 7 * costheta

        f = jnp.array([
            (4 - 4 * costheta - t1) / (2 * tp2),
            (4 * theta - 5 * sintheta + t2) / (2 * tp3),
            (2 - 2 * costheta - t1) / (2 * tp4),
            (2 * theta - 3 * sintheta + t2) / (2 * tp5),
        ])

        fd = jnp.array([
            t3 / (2 * tp3),
            t4 / (2 * tp4),
            t3 / (2 * tp5),
            t4 / (2 * tp6),
        ])

        fdd = jnp.array([
            (theta * t3d - 3 * t3) / (2 * tp4),
            (theta * t4d - 4 * t4) / (2 * tp5),
            (theta * t3d - 5 * t3) / (2 * tp6),
            (theta * t4d - 6 * t4) / (2 * tp7),
        ])

        g = jnp.eye(4) + Omegahat + ((1 - costheta) / tp2) * Omegahatp2 + ((theta - sintheta) / tp3) * Omegahatp3
        
        T = (
            jnp.eye(6)
            + f[0] * adjOmegap[0:6, :]
            + f[1] * adjOmegap[6:12, :]
            + f[2] * adjOmegap[12:18, :]
            + f[3] * adjOmegap[18:24, :]
        )

        Td = (
            fd[0] * thetad * adjOmegap[0:6, :] + f[0] * adjOmegapd[0:6, :]
            + fd[1] * thetad * adjOmegap[6:12, :] + f[1] * adjOmegapd[6:12, :]
            + fd[2] * thetad * adjOmegap[12:18, :] + f[2] * adjOmegapd[12:18, :]
            + fd[3] * thetad * adjOmegap[18:24, :] + f[3] * adjOmegapd[18:24, :]
        )

        return (f, fd, fdd, g, T, Td)

    f, fd, fdd, g, T, Td = lax.cond(
        theta <= eps,
        lambda _: _g_series_branch(),
        lambda _: _g_general_branch(theta),
        operand=None,
    )

    S  = T @ Z
    TZd = T @ Zd
    Sd = Td @ Z + TZd
    dSdq_qd  = TZd
    dSdq_qdd = T @ Zdd
    dSddq_qd = Td @ Zd

    Zqdd = Z @ qdd
    Zdqd = Zd @ qd

    def _block6(A: Array, k: int) -> Array:
        # k = 0,1,2,3 gives rows [0:6], [6:12], [12:18], [18:24]
        return A[k * 6:(k + 1) * 6, :]

    def _S_series_branch() -> tuple[Array, Array, Array]:
        dSdq_qd  = TZd
        dSdq_qdd = T @ Zdd
        dSddq_qd = Td @ Zd
        # dSdq_qd = jnp.zeros_like(Z)
        # dSdq_qdd = jnp.zeros_like(Z)
        # dSddq_qd = jnp.zeros_like(Z)
        for r in range(4):
            for u in range(1, r + 2):
                if u == 1:
                    adjOmegap1 = jnp.eye(6)
                else:
                    adjOmegap1 = _block6(adjOmegap, u - 2)
                
                if u == r + 1:
                    adjOmegap2 = jnp.eye(6)
                else:
                    adjOmegap2 = _block6(adjOmegap, r - u)
                
                dSdq_qd = dSdq_qd - f[r] * (adjOmegap1 @ adjoint_se3(adjOmegap2 @ Omegad) @ Z)
                dSdq_qdd = dSdq_qdd - f[r] * (adjOmegap1 @ adjoint_se3(adjOmegap2 @ Zqdd) @ Z)
                dSddq_qd = dSddq_qd - f[r] * (adjOmegap1 @ adjoint_se3(adjOmegap2 @ Zdqd) @ Z) - f[r] * (
                    adjOmegap1 @ adjoint_se3(adjOmegap2 @ Omegad) @ Zd
                    )
                                
                for p in range(1, u):
                    if p == 1:
                        adjOmegap3 = jnp.eye(6)
                    else:
                        adjOmegap3 = _block6(adjOmegap, p - 2)

                    if p == u - 1:
                        adjOmegap4 = jnp.eye(6)
                    else:
                        adjOmegap4 = _block6(adjOmegap, u - p - 2)

                    dSddq_qd = dSddq_qd - f[r] * (
                        adjOmegap3
                        @ adjoint_se3(adjOmegap4 @ _block6(adjOmegapd, 0) @ adjOmegap2 @ Omegad)
                        @ Z
                    )
                
                for p in range(1, r - u + 2):
                    if p == 1:
                        adjOmegap3 = jnp.eye(6)
                    else:
                        adjOmegap3 = _block6(adjOmegap, p - 2)

                    if p == r - u + 1:
                        adjOmegap4 = jnp.eye(6)
                    else:
                        adjOmegap4 = _block6(adjOmegap, r - u - p)

                    dSddq_qd = dSddq_qd - f[r] * (
                        adjOmegap1
                        @ _block6(adjOmegapd, 0)
                        @ adjOmegap3
                        @ adjoint_se3(adjOmegap4 @ Omegad)
                        @ Z
                    )
        return (dSdq_qd, dSdq_qdd, dSddq_qd)
    
    def _S_general_branch(theta: Array) -> tuple[Array, Array, Array]:
        # dSdq_qd = jnp.zeros_like(Z)
        # dSdq_qdd = jnp.zeros_like(Z)
        # dSddq_qd = jnp.zeros_like(Z)

        dSdq_qd  = TZd
        dSdq_qdd = T @ Zdd
        dSddq_qd = Td @ Zd
        for r in range(4):
            adjr = _block6(adjOmegap, r)
            adjrd = _block6(adjOmegapd, r)

            dSdq_qd = dSdq_qd + (1 / theta) * fd[r] * jnp.outer(
                adjr @ Omegad,
                Omega @ (I_theta @ Z),
            )

            dSdq_qdd = dSdq_qdd + (1 / theta) * fd[r] * jnp.outer(
                adjr @ Zqdd,
                Omega @ (I_theta @ Z),
            )

            term1 = (1 / theta) * fd[r] * jnp.outer(
                adjr @ Zdqd,
                Omega @ (I_theta @ Z),
            )

            term2 = (1 / theta) * jnp.outer(
                ((fdd[r] * thetad) * adjr + fd[r] * adjrd) @ Omegad,
                Omega @ (I_theta @ Z),
            )

            term3 = (1 / theta) * fd[r] * jnp.outer(
                adjr @ Omegad,
                (Omegad - (thetad / theta) * Omega) @ (I_theta @ Z) + Omega @ (I_theta @ Zd),
            )

            dSddq_qd = dSddq_qd + term1 + term2 + term3
            for u in range(1, r + 2):
                if u == 1:
                    adjOmegap1 = jnp.eye(6)
                else:
                    adjOmegap1 = _block6(adjOmegap, u - 2)

                if u == r + 1:
                    adjOmegap2 = jnp.eye(6)
                else:
                    adjOmegap2 = _block6(adjOmegap, r - u)
                
                dSdq_qd = dSdq_qd - f[r] * (adjOmegap1 @ adjoint_se3(adjOmegap2 @ Omegad) @ Z)
                dSdq_qdd = dSdq_qdd - f[r] * (adjOmegap1 @ adjoint_se3(adjOmegap2 @ Zqdd) @ Z)

                dSddq_qd = (
                    dSddq_qd
                    - f[r] * (adjOmegap1 @ adjoint_se3(adjOmegap2 @ Zdqd) @ Z)
                    - fd[r] * thetad * (adjOmegap1 @ adjoint_se3(adjOmegap2 @ Omegad) @ Z)
                    - f[r] * (adjOmegap1 @ adjoint_se3(adjOmegap2 @ Omegad) @ Zd)
                )

                for p in range(1, u):
                    if p == 1:
                        adjOmegap3 = jnp.eye(6)
                    else:
                        adjOmegap3 = _block6(adjOmegap, p - 2)

                    if p == u - 1:
                        adjOmegap4 = jnp.eye(6)
                    else:
                        adjOmegap4 = _block6(adjOmegap, u - p - 2)

                    dSddq_qd = dSddq_qd - f[r] * (
                        adjOmegap3 @ adjoint_se3(adjOmegap4 @ _block6(adjOmegapd, 0) @ adjOmegap2 @ Omegad) @ Z)
                                            
                for p in range(1, r - u + 2):
                    if p == 1:
                        adjOmegap3 = jnp.eye(6)
                    else:
                        adjOmegap3 = _block6(adjOmegap, p - 2)

                    if p == r - u + 1:
                        adjOmegap4 = jnp.eye(6)
                    else:
                        adjOmegap4 = _block6(adjOmegap, r - u - p)

                    dSddq_qd = dSddq_qd - f[r] * (adjOmegap1 @ _block6(adjOmegapd, 0) @ adjOmegap3
                        @ adjoint_se3(adjOmegap4 @ Omegad) @ Z
                    )


        return (dSdq_qd, dSdq_qdd, dSddq_qd)

    dSdq_qd, dSdq_qdd, dSddq_qd = lax.cond(
        theta <= eps,
        lambda _: _S_series_branch(),
        lambda _: _S_general_branch(theta),
        operand=None,
    )

    return (Omega, Z, g, T, S, Sd, f, fd, adjOmegap, dSdq_qd, dSdq_qdd, dSddq_qd)


def pcs_SoftJointDifferentialKinematics(
        eps,
        H,
        Phi,
        xi_star,
        q,
        qd,
        qdd) -> tuple[
            Array,
            Array,
            Array,
            Array,
            Array,
            Array,
            Array,
            Array,
            Array,
            Array,
            Array,
            Array
            ]:
    
    I_theta = jnp.diag(jnp.array([1, 1, 1, 0, 0, 0]))
    xi = Phi @ q + xi_star
    Omega = H * xi
    Z = H * Phi    
    Omegad  = Z @ qd

    Zd      = jnp.zeros_like(Z)
    Zdd     = jnp.zeros_like(Z)

    adjOmegap  = jnp.zeros((24, 6))
    adjOmegapd = jnp.zeros((24, 6))

    k = Omega[:3]
    kd = Omegad[:3]
    theta = _rotational_strain_magnitude(Omega, eps)
    thetad = (kd @ k) / theta

    Omegahat = hat_SE3(Omega)
    Omegahatp2 = Omegahat @ Omegahat
    Omegahatp3 = Omegahatp2 @ Omegahat

    adjOmegap = adjOmegap.at[0:6, :].set(adjoint_se3(Omega))
    adjOmegap = adjOmegap.at[6:12, :].set(adjOmegap[0:6, :] @ adjOmegap[0:6, :])
    adjOmegap = adjOmegap.at[12:18, :].set(adjOmegap[6:12, :] @ adjOmegap[0:6, :])
    adjOmegap = adjOmegap.at[18:24, :].set(adjOmegap[12:18, :] @ adjOmegap[0:6, :])

    adjOmegapd = adjOmegapd.at[0:6, :].set(adjoint_se3(Omegad))
    adjOmegapd = adjOmegapd.at[6:12, :].set(adjOmegapd[0:6, :] @ adjOmegap[0:6, :] +
    adjOmegap[0:6, :] @ adjOmegapd[0:6, :]
    )
    adjOmegapd = adjOmegapd.at[12:18, :].set(adjOmegapd[6:12, :] @ adjOmegap[0:6, :] +
    adjOmegap[6:12, :] @ adjOmegapd[0:6, :]
    )
    adjOmegapd = adjOmegapd.at[18:24, :].set(adjOmegapd[12:18, :] @ adjOmegap[0:6, :] +
    adjOmegap[12:18, :] @ adjOmegapd[0:6, :]
    )

    def _g_series_branch() -> tuple[Array, Array, Array, Array, Array, Array]:
        g = jnp.eye(4) + Omegahat + Omegahatp2 / 2 + Omegahatp3 / 6
        f = jnp.array([1/2, 1/6, 1/24, 1/120])
        fd  = jnp.zeros((4,))
        fdd = jnp.zeros((4,))

        T = (
            jnp.eye(6)
            + f[0] * adjOmegap[0:6, :]
            + f[1] * adjOmegap[6:12, :]
            + f[2] * adjOmegap[12:18, :]
            + f[3] * adjOmegap[18:24, :]
        )

        Td = (
            f[0] * adjOmegapd[0:6, :]
            + f[1] * adjOmegapd[6:12, :]
            + f[2] * adjOmegapd[12:18, :]
            + f[3] * adjOmegapd[18:24, :]
        )

        return (f, fd, fdd, g, T, Td)
    
    def _g_general_branch(theta: Array) -> tuple[Array, Array, Array, Array, Array, Array]:
        tp2        = theta * theta
        tp3        = tp2 * theta
        tp4        = tp3 * theta
        tp5        = tp4 * theta
        tp6        = tp5 * theta
        tp7        = tp6 * theta
        costheta = jnp.cos(theta)
        sintheta = jnp.sin(theta)

        t1 = theta * sintheta
        t2 = theta * costheta
        t3 = -8 + (8 - tp2) * costheta + 5 * t1
        t4 = -8 * theta + (15 - tp2) * sintheta - 7 * t2
        t3d = 5 * sintheta + sintheta * (tp2 - 8) + 3 * theta * costheta
        t4d = -8 + 5 * theta * sintheta + (15 - tp2) * costheta - 7 * costheta

        f = jnp.array([
            (4 - 4 * costheta - t1) / (2 * tp2),
            (4 * theta - 5 * sintheta + t2) / (2 * tp3),
            (2 - 2 * costheta - t1) / (2 * tp4),
            (2 * theta - 3 * sintheta + t2) / (2 * tp5),
        ])

        fd = jnp.array([
            t3 / (2 * tp3),
            t4 / (2 * tp4),
            t3 / (2 * tp5),
            t4 / (2 * tp6),
        ])

        fdd = jnp.array([
            (theta * t3d - 3 * t3) / (2 * tp4),
            (theta * t4d - 4 * t4) / (2 * tp5),
            (theta * t3d - 5 * t3) / (2 * tp6),
            (theta * t4d - 6 * t4) / (2 * tp7),
        ])

        g = jnp.eye(4) + Omegahat + ((1 - costheta) / tp2) * Omegahatp2 + ((theta - sintheta) / tp3) * Omegahatp3
        
        T = (
            jnp.eye(6)
            + f[0] * adjOmegap[0:6, :]
            + f[1] * adjOmegap[6:12, :]
            + f[2] * adjOmegap[12:18, :]
            + f[3] * adjOmegap[18:24, :]
        )

        Td = (
            fd[0] * thetad * adjOmegap[0:6, :] + f[0] * adjOmegapd[0:6, :]
            + fd[1] * thetad * adjOmegap[6:12, :] + f[1] * adjOmegapd[6:12, :]
            + fd[2] * thetad * adjOmegap[12:18, :] + f[2] * adjOmegapd[12:18, :]
            + fd[3] * thetad * adjOmegap[18:24, :] + f[3] * adjOmegapd[18:24, :]
        )

        return (f, fd, fdd, g, T, Td)

    f, fd, fdd, g, T, Td = lax.cond(
        theta <= eps,
        lambda _: _g_series_branch(),
        lambda _: _g_general_branch(theta),
        operand=None,
    )

    S  = T @ Z
    Sd = Td @ Z
    Zqdd = Z @ qdd

    def _block6(A: Array, k: int) -> Array:
        # k = 0,1,2,3 gives rows [0:6], [6:12], [12:18], [18:24]
        return A[k * 6:(k + 1) * 6, :]

    def _S_series_branch() -> tuple[Array, Array, Array]:
        dSdq_qd  = jnp.zeros_like(Z)
        dSdq_qdd = jnp.zeros_like(Z)
        dSddq_qd = jnp.zeros_like(Z)
        for r in range(4):
            for u in range(1, r + 2):
                if u == 1:
                    adjOmegap1 = jnp.eye(6)
                else:
                    adjOmegap1 = _block6(adjOmegap, u - 2)
                
                if u == r + 1:
                    adjOmegap2 = jnp.eye(6)
                else:
                    adjOmegap2 = _block6(adjOmegap, r - u)
                
                dSdq_qd = dSdq_qd - f[r] * (adjOmegap1 @ adjoint_se3(adjOmegap2 @ Omegad) @ Z)
                dSdq_qdd = dSdq_qdd - f[r] * (adjOmegap1 @ adjoint_se3(adjOmegap2 @ Zqdd) @ Z)
                
                for p in range(1, u):
                    if p == 1:
                        adjOmegap3 = jnp.eye(6)
                    else:
                        adjOmegap3 = _block6(adjOmegap, p - 2)

                    if p == u - 1:
                        adjOmegap4 = jnp.eye(6)
                    else:
                        adjOmegap4 = _block6(adjOmegap, u - p - 2)

                    dSddq_qd = dSddq_qd - f[r] * (
                        adjOmegap3
                        @ adjoint_se3(adjOmegap4 @ _block6(adjOmegapd, 0) @ adjOmegap2 @ Omegad)
                        @ Z
                    )
                
                for p in range(1, r - u + 2):
                    if p == 1:
                        adjOmegap3 = jnp.eye(6)
                    else:
                        adjOmegap3 = _block6(adjOmegap, p - 2)

                    if p == r - u + 1:
                        adjOmegap4 = jnp.eye(6)
                    else:
                        adjOmegap4 = _block6(adjOmegap, r - u - p)

                    dSddq_qd = dSddq_qd - f[r] * (
                        adjOmegap1
                        @ _block6(adjOmegapd, 0)
                        @ adjOmegap3
                        @ adjoint_se3(adjOmegap4 @ Omegad)
                        @ Z
                    )
        return (dSdq_qd, dSdq_qdd, dSddq_qd)
    
    def _S_general_branch(theta: Array) -> tuple[Array, Array, Array]:
        dSdq_qd  = jnp.zeros_like(Z)
        dSdq_qdd = jnp.zeros_like(Z)
        dSddq_qd = jnp.zeros_like(Z)
        for r in range(4):
            adjr = _block6(adjOmegap, r)
            adjrd = _block6(adjOmegapd, r)

            dSdq_qd = dSdq_qd + (1 / theta) * fd[r] * jnp.outer(
                adjr @ Omegad,
                Omega @ (I_theta @ Z),
            )

            dSdq_qdd = dSdq_qdd + (1 / theta) * fd[r] * jnp.outer(
                adjr @ Zqdd,
                Omega @ (I_theta @ Z),
            )

            term1 = (1 / theta) * jnp.outer(
                ((fdd[r] * thetad) * adjr + fd[r] * adjrd) @ Omegad,
                Omega @ (I_theta @ Z),
            )

            term2 = (1 / theta) * fd[r] * jnp.outer(
                adjr @ Omegad,
                (Omegad - (thetad / theta) * Omega) @ (I_theta @ Z) + Omega @ (I_theta @ Zd),
            )

            dSddq_qd = dSddq_qd + term1 + term2
            for u in range(1, r + 2):
                if u == 1:
                    adjOmegap1 = jnp.eye(6)
                else:
                    adjOmegap1 = _block6(adjOmegap, u - 2)

                if u == r + 1:
                    adjOmegap2 = jnp.eye(6)
                else:
                    adjOmegap2 = _block6(adjOmegap, r - u)
                
                dSdq_qd = dSdq_qd - f[r] * (adjOmegap1 @ adjoint_se3(adjOmegap2 @ Omegad) @ Z)
                dSdq_qdd = dSdq_qdd - f[r] * (adjOmegap1 @ adjoint_se3(adjOmegap2 @ Zqdd) @ Z)
                dSddq_qd = (dSddq_qd- fd[r] * thetad * (adjOmegap1 @ adjoint_se3(adjOmegap2 @ Omegad) @ Z))

                for p in range(1, u):
                    if p == 1:
                        adjOmegap3 = jnp.eye(6)
                    else:
                        adjOmegap3 = _block6(adjOmegap, p - 2)

                    if p == u - 1:
                        adjOmegap4 = jnp.eye(6)
                    else:
                        adjOmegap4 = _block6(adjOmegap, u - p - 2)

                    dSddq_qd = dSddq_qd - f[r] * (
                        adjOmegap3 @ adjoint_se3(adjOmegap4 @ _block6(adjOmegapd, 0) @ adjOmegap2 @ Omegad) @ Z)
                
                for p in range(1, r - u + 2):
                    if p == 1:
                        adjOmegap3 = jnp.eye(6)
                    else:
                        adjOmegap3 = _block6(adjOmegap, p - 2)

                    if p == r - u + 1:
                        adjOmegap4 = jnp.eye(6)
                    else:
                        adjOmegap4 = _block6(adjOmegap, r - u - p)

                    dSddq_qd = dSddq_qd - f[r] * (adjOmegap1 @ _block6(adjOmegapd, 0) @ adjOmegap3
                        @ adjoint_se3(adjOmegap4 @ Omegad) @ Z
                    )

        return (dSdq_qd, dSdq_qdd, dSddq_qd)

    dSdq_qd, dSdq_qdd, dSddq_qd = lax.cond(
        theta <= eps,
        lambda _: _S_series_branch(),
        lambda _: _S_general_branch(theta),
        operand=None,
    )

    return (Omega, Z, g, T, S, Sd, f, fd, adjOmegap, dSdq_qd, dSdq_qdd, dSddq_qd)
