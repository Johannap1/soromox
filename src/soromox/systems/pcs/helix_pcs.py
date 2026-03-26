__all__ = ["HelixPCS"]

from typing import Any

import jax.numpy as jnp
from jax import Array

from .pcs import PCS
from .tendon_actuated_pcs import TendonActuatedPCS


class HelixPCS(TendonActuatedPCS):
    """
    Tendon-actuated PCS model for continuum soft robots based on trimmed helicoids.

    The model reuses the full tendon-actuated PCS kinematics and dynamics while
    overriding the cross-sectional and constitutive quantities so that mass and
    stiffness are derived from a reduced annular-shell approximation of a trimmed-
    helicoid structure, often referred to as a Helix soft robot.

    The connection between the trimmed-helicoid design parameters and the equivalent
    Cosserat-rod parameters, such as axial and bending stiffness, is based on the
    empirical relationships reported in the reference below.

    References:
        Guan, Q., Stella, F., Della Santina, C., Leng, J., & Hughes, J. (2023).
        Trimmed helicoids: an architectured soft structure yielding soft robots
        with high precision, large workspace, and compliant interactions. npj
        Robotics, 1(1), 4.
    """

    w: Array
    t: Array
    fill_factor: Array
    num_fill_strands: Array
    nu: Array

    A_geom: Array
    I_geom: Array
    A_eff: Array
    I_eff: Array
    J_eff: Array
    lambda_stiff: Array
    K_axial: Array
    K_bend: Array
    E_ax: Array
    E_bend: Array
    G_ax: Array
    G_bend: Array
    p0_params: Array

    def __init__(self, params: dict[str, Array], *args, **kwargs: Any):
        num_segments = self._infer_num_segments(params)
        super().__init__(num_segments=num_segments, params=params, *args, **kwargs)

    @staticmethod
    def _infer_num_segments(params: dict[str, Array]) -> int:
        try:
            L = params["L"]
        except KeyError as exc:
            raise KeyError("Parameter 'L' is required in params dictionary.") from exc

        L = jnp.asarray(L, dtype=jnp.float64)
        if L.ndim != 1:
            raise ValueError(f"L must be a 1D array, got shape {L.shape}")
        if L.size < 1:
            raise ValueError("L must contain at least one segment length.")
        return int(L.shape[0])

    def _broadcast_segment_param(
        self,
        value: Array | float | int | list[float] | list[int],
        name: str,
        *,
        dtype: Any,
    ) -> Array:
        arr = jnp.asarray(value, dtype=dtype)
        if arr.ndim == 0 or arr.size == 1:
            return jnp.full((self.num_segments,), arr.reshape(()), dtype=dtype)
        if arr.shape != (self.num_segments,):
            raise ValueError(
                f"{name} must be a scalar or have shape ({self.num_segments},), got {arr.shape}"
            )
        return arr

    def _build_base_params(self, params: dict[str, Array]) -> dict[str, Array]:
        eps = 1e1 * float(jnp.finfo(jnp.float64).eps)
        p0 = jnp.asarray(
            params.get("p0", jnp.array([jnp.pi / 2, jnp.pi / 2, 0.0, 0.0, 0.0, 0.0])),
            dtype=jnp.float64,
        )
        if p0.shape != (6,):
            raise ValueError(f"p0 must have shape (6,), got {p0.shape}")

        r = self._broadcast_segment_param(params["r"], "r", dtype=jnp.float64)
        w = self._broadcast_segment_param(params["w"], "w", dtype=jnp.float64)
        t = self._broadcast_segment_param(params["t"], "t", dtype=jnp.float64)
        rho = self._broadcast_segment_param(
            params.get("rho", 1200.0), "rho", dtype=jnp.float64
        )
        nu = self._broadcast_segment_param(
            params.get("nu", 0.45), "nu", dtype=jnp.float64
        )
        num_fill_strands = self._broadcast_segment_param(
            params.get("num_fill_strands", 1.0),
            "num_fill_strands",
            dtype=jnp.float64,
        )

        if jnp.any(r <= 0.0):
            raise ValueError("r must be strictly positive.")
        if jnp.any(w <= 0.0):
            raise ValueError("w must be strictly positive.")
        if jnp.any(w > r):
            raise ValueError("w must be less than or equal to r.")
        if jnp.any(t <= 0.0):
            raise ValueError("t must be strictly positive.")
        if jnp.any(num_fill_strands <= 0.0):
            raise ValueError("num_fill_strands must be strictly positive.")
        if jnp.any(nu <= -1.0):
            raise ValueError("nu must be greater than -1.")

        fill_factor_param = params.get("fill_factor")
        if fill_factor_param is None:
            fill_factor = jnp.clip(num_fill_strands * w / (2.0 * jnp.pi * r), eps, 1.0)
        else:
            fill_factor = self._broadcast_segment_param(
                fill_factor_param, "fill_factor", dtype=jnp.float64
            )
        if jnp.any(fill_factor <= 0.0) or jnp.any(fill_factor > 1.0):
            raise ValueError("fill_factor must satisfy 0 < fill_factor <= 1.")

        A_geom = jnp.pi * (r**2 - (r - w) ** 2)
        I_geom = (jnp.pi / 4.0) * (r**4 - (r - w) ** 4)
        A_eff = fill_factor * A_geom
        I_eff = (
            fill_factor - jnp.sin(fill_factor * 2.0 * jnp.pi) / (2.0 * jnp.pi)
        ) * I_geom
        J_eff = 2.0 * I_eff

        r_mm = r * 1e3
        w_mm = w * 1e3
        t_mm = t * 1e3

        lambda_stiff = 12.76 * (w_mm / r_mm) ** 3.42 + 1.58
        K_axial_per_mm = 0.05 * t_mm**3.16
        K_axial = 1e3 * K_axial_per_mm
        K_bend = 1e-3 * (K_axial_per_mm * r_mm**2 / lambda_stiff)

        L_total = jnp.sum(jnp.asarray(params["L"], dtype=jnp.float64))
        E_ax = K_axial * L_total / A_eff
        E_bend = K_bend * L_total**3 / (3.0 * I_eff)
        G_ax = E_ax / (2.0 * (1.0 + nu))
        G_bend = E_bend / (2.0 * (1.0 + nu))

        self.w = w
        self.t = t
        self.fill_factor = fill_factor
        self.num_fill_strands = num_fill_strands
        self.nu = nu
        self.A_geom = A_geom
        self.I_geom = I_geom
        self.A_eff = A_eff
        self.I_eff = I_eff
        self.J_eff = J_eff
        self.lambda_stiff = lambda_stiff
        self.K_axial = K_axial
        self.K_bend = K_bend
        self.E_ax = E_ax
        self.E_bend = E_bend
        self.G_ax = G_ax
        self.G_bend = G_bend
        self.p0_params = p0

        base_params = dict(params)
        base_params["p0"] = p0
        base_params["r"] = r
        base_params["rho"] = rho
        base_params["E"] = E_bend
        base_params["G"] = G_bend
        return base_params

    def _set_params(self, params: dict[str, Array]) -> None:
        for key in ("r", "w", "t"):
            if key not in params:
                raise KeyError(
                    f"Parameter '{key}' is required for the trimmed-helicoid HelixPCS model."
                )

        base_params = self._build_base_params(params)
        PCS._set_params(self, base_params)

    def update_params(self, params: dict[str, Array]) -> "HelixPCS":
        current_params = {
            "p0": self.p0_params,
            "L": self.L,
            "r": self.r,
            "rho": self.rho,
            "g": self.g[3:],
            "D": self.D,
            "w": self.w,
            "t": self.t,
            "fill_factor": self.fill_factor,
            "num_fill_strands": self.num_fill_strands,
            "nu": self.nu,
        }
        current_params.update(params)

        strain_selector = jnp.any(self.B_xi != 0.0, axis=1)
        passive_tendon_params = {
            "k_pt": jnp.diag(self.K_pt),
            "d_pt": jnp.diag(self.D_pt),
            "l_pt0": self.l_pt0,
        }

        return HelixPCS(
            current_params,
            order_gauss=self.num_gauss_points - 2,
            strain_selector=strain_selector,
            xi_ref=self.xi_ref,
            active_tendon_routing_basis={
                "d_s": self.active_d_s,
                "dd_s_ds": self.active_dd_s_ds,
            },
            active_tendon_routing_params=self.active_tendon_routing_params,
            passive_tendon_routing_basis={
                "d_s": self.passive_d_s,
                "dd_s_ds": self.passive_dd_s_ds,
            },
            passive_tendon_routing_params=self.passive_tendon_routing_params,
            passive_tendon_params=passive_tendon_params,
        )

    def _local_cross_sectional_area(self, i: Array) -> Array:
        return self.A_eff[i]

    def _local_second_moment_of_area(self, i: Array) -> Array:
        return jnp.array([self.J_eff[i], self.I_eff[i], self.I_eff[i]])

    def _local_stiffness_matrix(self, i: Array) -> Array:
        A_i = self._local_cross_sectional_area(i)
        I_i = self._local_second_moment_of_area(i)

        S_i = self.L[i] * jnp.diag(
            jnp.array(
                [
                    self.G_bend[i] * I_i[0],
                    self.E_bend[i] * I_i[1],
                    self.E_bend[i] * I_i[2],
                    self.E_ax[i] * A_i,
                    self.G_bend[i] * A_i,
                    self.G_bend[i] * A_i,
                ]
            )
        )
        return S_i
