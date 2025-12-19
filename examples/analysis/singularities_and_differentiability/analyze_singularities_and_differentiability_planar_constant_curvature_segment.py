"""
Compare singularity handling and differentiability for a planar constant-curvature
segment across three implementations:
1) JSRM planar PCS (regularized bending strain).
2) JSRM planar PCS without epsilon regularization.
3) soromox PlanarPCS with only the bending strain active.

The script evaluates forward kinematics, the Jacobian, and the inertia matrix
over a grid of bending configurations and visualizes both the raw quantities
and their first-order gradients obtained via autograd.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict

import jax
import jax.numpy as jnp
from jax import Array
import matplotlib.pyplot as plt

import jsrm
from jsrm.systems import planar_pcs as jsrm_planar_pcs
from soromox.systems.planar_pcs import PlanarPCS

jax.config.update("jax_enable_x64", True)

plt.rcParams.update(
    {
        "text.usetex": True,
        "text.latex.preamble": r"\usepackage{amsmath}",
        "font.family": "serif",
        "font.size": 12,
    }
)


@dataclass
class Implementation:
    name: str
    forward_fn: Callable[[Array], Array]
    jacobian_fn: Callable[[Array], Array]
    inertia_fn: Callable[[Array], Array]
    color: str
    linestyle: str = "-"
    linewidth: float = 2.0
    marker: str | None = None
    markevery: int | None = None
    zorder: int = 2
    alpha: float = 1.0


def _style_axis(ax, xlabel: str, ylabel: str, title: str | None = None) -> None:
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.7)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(True)


def _pad_limits(y: Array) -> tuple[float, float]:
    y_min = float(jnp.min(y))
    y_max = float(jnp.max(y))
    span = y_max - y_min
    pad = 0.05 * span if span > 0 else max(1e-3, 0.1 * abs(y_max))
    return y_min - pad, y_max + pad


def _nan_to_inf(x: Array) -> Array:
    return jnp.nan_to_num(x, nan=jnp.inf)


def build_soromox_impl(
    params: Dict[str, Array], color: str
) -> Implementation:
    strain_selector = jnp.array([True, False, False])
    robot = PlanarPCS(
        num_segments=1,
        params=params,
        strain_selector=strain_selector,
    )
    total_length = float(params["L"].sum())

    def forward(q: Array) -> Array:
        q_vec = jnp.asarray(q, dtype=jnp.float64).reshape(-1)
        return robot.forward_kinematics(q_vec, total_length)

    def jacobian(q: Array) -> Array:
        q_vec = jnp.asarray(q, dtype=jnp.float64).reshape(-1)
        J = robot.jacobian(q_vec, total_length)
        return J[:, 0]

    def inertia(q: Array) -> Array:
        q_vec = jnp.asarray(q, dtype=jnp.float64).reshape(-1)
        B = robot.inertia_matrix(q_vec)
        return B[0, 0]

    return Implementation(
        name="SoRoMoX",
        forward_fn=forward,
        jacobian_fn=jacobian,
        inertia_fn=inertia,
        color=color,
        linestyle="-",
        linewidth=1.6,
        alpha=0.7,
        zorder=1,
    )


def build_jsrm_impl(
    params: Dict[str, Array],
    eps: float,
    label: str,
    color: str,
    linestyle: str = "-",
    linewidth: float = 1.8,
    marker: str | None = None,
    markevery: int | None = None,
    zorder: int = 3,
) -> Implementation:
    strain_selector = jnp.array([True, False, False])
    dill_path = (
        Path(jsrm.__file__).resolve().parent
        / "symbolic_expressions"
        / "planar_pcs_ns-1.dill"
    )
    _, forward_kinematics_fn, dynamical_matrices_fn, auxiliary_fns = (
        jsrm_planar_pcs.factory(
            filepath=dill_path,
            strain_selector=strain_selector,
            global_eps=eps,
        )
    )
    jacobian_fn = auxiliary_fns["jacobian_fn"]
    total_length = float(params["l"].sum())

    def forward(q: Array) -> Array:
        q_vec = jnp.asarray(q, dtype=jnp.float64).reshape(-1)
        chi = forward_kinematics_fn(params, q_vec, total_length)
        chi = jnp.stack([chi[2] + jnp.pi / 2, chi[0], chi[1]])  # theta, x, y
        return chi

    def jacobian(q: Array) -> Array:
        q_vec = jnp.asarray(q, dtype=jnp.float64).reshape(-1)
        J = jacobian_fn(params, q_vec, total_length)
        J_reordered = J[[2, 0, 1], :]  # theta, x, y rows
        return J_reordered[:, 0]

    def inertia(q: Array) -> Array:
        q_vec = jnp.asarray(q, dtype=jnp.float64).reshape(-1)
        B, _, _, _, _, _ = dynamical_matrices_fn(
            params, q_vec, jnp.zeros_like(q_vec)
        )
        return B[0, 0]

    return Implementation(
        name=label,
        forward_fn=forward,
        jacobian_fn=jacobian,
        inertia_fn=inertia,
        color=color,
        linestyle=linestyle,
        linewidth=linewidth,
        marker=marker,
        markevery=markevery,
        zorder=zorder,
    )


def evaluate_impl(impl: Implementation, q_values: Array) -> Dict[str, Array]:
    chi_raw = jax.vmap(impl.forward_fn)(q_values)
    chi_nan_mask = jnp.isnan(chi_raw)
    chi_vals = _nan_to_inf(chi_raw)
    jac_vals = _nan_to_inf(jax.vmap(impl.jacobian_fn)(q_values))
    inertia_vals = _nan_to_inf(jax.vmap(impl.inertia_fn)(q_values))

    grad_chi = _nan_to_inf(
        jnp.stack(
        [
            jax.vmap(jax.grad(lambda qq, idx=idx: impl.forward_fn(qq)[idx]))(
                q_values
            )
            for idx in range(3)
        ],
        axis=1,
    )
    )
    grad_inertia = _nan_to_inf(jax.vmap(jax.grad(impl.inertia_fn))(q_values))

    return {
        "chi": chi_vals,
        "chi_nan_mask": chi_nan_mask,
        "jac": jac_vals,
        "inertia": inertia_vals,
        "grad_chi": grad_chi,
        "grad_inertia": grad_inertia,
    }


def plot_forward(impls, results, q_values: Array) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharex=True)
    soromox_chi = results["SoRoMoX"]["chi"]
    ylabels = [
        r"$\theta(s=L)\;[\mathrm{rad}]$",
        r"$p_\mathrm{x}(s=L)\;[\mathrm{m}]$",
        r"$p_\mathrm{y}(s=L)\;[\mathrm{m}]$",
    ]
    for comp_idx in range(3):
        ax = axes[comp_idx]
        ylims = _pad_limits(soromox_chi[:, comp_idx])
        for impl in impls:
            ax.plot(
                q_values,
                results[impl.name]["chi"][:, comp_idx],
                label=impl.name,
                color=impl.color,
                linestyle=impl.linestyle,
                linewidth=impl.linewidth,
                marker=impl.marker,
                markevery=impl.markevery,
                zorder=impl.zorder,
                alpha=impl.alpha,
            )
            chi_nan_mask = results[impl.name].get("chi_nan_mask")
            if chi_nan_mask is not None:
                _plot_nan_marker(
                    ax,
                    q_values,
                    chi_nan_mask[:, comp_idx],
                    ylims,
                    impl,
                )
        ax.set_ylim(*ylims)
        _style_axis(
            ax,
            xlabel=r"$q = \kappa\,[\mathrm{m}^{-1}]$",
            ylabel=ylabels[comp_idx],
        )
    axes[0].legend(loc="best")
    fig.suptitle("Forward kinematics at $s=L$")
    fig.tight_layout()


def plot_forward_x_only(impls, results, q_values: Array) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    soromox_x = results["SoRoMoX"]["chi"][:, 1]
    ylims = _pad_limits(soromox_x)
    for impl in impls:
        ax.plot(
            q_values,
            results[impl.name]["chi"][:, 1],
            label=impl.name,
            color=impl.color,
            linestyle=impl.linestyle,
            linewidth=impl.linewidth,
            marker=impl.marker,
            markevery=impl.markevery,
            zorder=impl.zorder,
            alpha=impl.alpha,
        )
        chi_nan_mask = results[impl.name].get("chi_nan_mask")
        if chi_nan_mask is not None:
            _plot_nan_marker(ax, q_values, chi_nan_mask[:, 1], ylims, impl)
    ax.set_ylim(*ylims)
    _style_axis(
        ax,
        xlabel=r"$q = \kappa\,[\mathrm{m}^{-1}]$",
        ylabel=r"$p_\mathrm{x}(s=L)\;[\mathrm{m}]$",
        title="Forward kinematics (x only)",
    )
    ax.legend(loc="best")
    fig.tight_layout()


def _plot_nan_marker(
    ax: plt.Axes,
    q_values: Array,
    nan_mask: Array,
    ylims: tuple[float, float],
    impl: Implementation,
) -> None:
    if not bool(jnp.any(nan_mask)):
        return
    q_nan = jnp.asarray(q_values)[nan_mask]
    y_pos = ylims[1] - 0.04 * (ylims[1] - ylims[0])
    ax.scatter(
        q_nan,
        jnp.full_like(q_nan, y_pos),
        marker="X",
        s=60,
        color=impl.color,
        edgecolors="k",
        linewidths=0.6,
        zorder=impl.zorder + 1,
        label=f"{impl.name} NaN",
    )


def plot_jacobian(impls, results, q_values: Array) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharex=True)
    soromox_jac = results["SoRoMoX"]["jac"]
    ylabels = [
        r"$\partial \theta / \partial q$",
        r"$\partial x / \partial q$",
        r"$\partial y / \partial q$",
    ]
    for comp_idx in range(3):
        ax = axes[comp_idx]
        ylims = _pad_limits(soromox_jac[:, comp_idx])
        for impl in impls:
            ax.plot(
                q_values,
                results[impl.name]["jac"][:, comp_idx],
                label=impl.name,
                color=impl.color,
                linestyle=impl.linestyle,
                linewidth=impl.linewidth,
                marker=impl.marker,
                markevery=impl.markevery,
                zorder=impl.zorder,
                alpha=impl.alpha,
            )
        ax.set_ylim(*ylims)
        _style_axis(
            ax,
            xlabel=r"$q = \kappa\,[\mathrm{m}^{-1}]$",
            ylabel=ylabels[comp_idx],
        )
    axes[0].legend(loc="best")
    fig.suptitle("Jacobian at $s=L$")
    fig.tight_layout()


def plot_inertia(impls, results, q_values: Array) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    soromox_inertia = results["SoRoMoX"]["inertia"]
    ylims = _pad_limits(soromox_inertia)
    for impl in impls:
        ax.plot(
            q_values,
            results[impl.name]["inertia"],
            label=impl.name,
            color=impl.color,
            linestyle=impl.linestyle,
            linewidth=impl.linewidth,
            marker=impl.marker,
            markevery=impl.markevery,
            zorder=impl.zorder,
            alpha=impl.alpha,
        )
    ax.set_ylim(*ylims)
    _style_axis(
        ax,
        xlabel=r"$q = \kappa\,[\mathrm{m}^{-1}]$",
        ylabel=r"$B_{qq}$",
        title="Inertia matrix entry",
    )
    ax.legend(loc="best")
    fig.tight_layout()


def plot_gradients(impls, results, q_values: Array) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    soromox_grad_chi = results["SoRoMoX"]["grad_chi"]
    soromox_grad_inertia = results["SoRoMoX"]["grad_inertia"]
    grad_labels = [
        (0, 0, r"$\nabla_q \theta$", 0),
        (0, 1, r"$\nabla_q x$", 1),
        (1, 0, r"$\nabla_q y$", 2),
    ]
    for row, col, ylabel, comp_idx in grad_labels:
        ax = axes[row, col]
        ylims = _pad_limits(soromox_grad_chi[:, comp_idx])
        for impl in impls:
            ax.plot(
                q_values,
                results[impl.name]["grad_chi"][:, comp_idx],
                label=impl.name,
                color=impl.color,
                linestyle=impl.linestyle,
                linewidth=impl.linewidth,
                marker=impl.marker,
                markevery=impl.markevery,
                zorder=impl.zorder,
                alpha=impl.alpha,
            )
        ax.set_ylim(*ylims)
        _style_axis(
            ax,
            xlabel=r"$q = \kappa\,[\mathrm{m}^{-1}]$",
            ylabel=ylabel,
        )
    # inertia gradient in the last subplot
    ax_inertia = axes[1, 1]
    ylims_inertia = _pad_limits(soromox_grad_inertia)
    for impl in impls:
        ax_inertia.plot(
            q_values,
            results[impl.name]["grad_inertia"],
            label=impl.name,
            color=impl.color,
            linestyle=impl.linestyle,
            linewidth=impl.linewidth,
            marker=impl.marker,
            markevery=impl.markevery,
            zorder=impl.zorder,
            alpha=impl.alpha,
        )
    ax_inertia.set_ylim(*ylims_inertia)
    _style_axis(
        ax_inertia,
        xlabel=r"$q = \kappa\,[\mathrm{m}^{-1}]$",
        ylabel=r"$\nabla_q B$",
    )
    axes[0, 0].legend(loc="best")
    fig.suptitle("Autograd gradients w.r.t. bending strain")
    fig.tight_layout()


def main() -> None:
    # Shared physical parameters
    length = 0.25  # [m]
    radius = 0.02  # [m]
    density = 1070.0  # [kg/m^3]
    params_soromox: Dict[str, Array] = {
        "th0": jnp.array(jnp.pi / 2),
        "L": jnp.array([length]),
        "r": jnp.array([radius]),
        "rho": jnp.array([density]),
        "g": jnp.array([0.0, 9.81]),
        "E": jnp.array([2e3]),
        "G": jnp.array([1e3]),
    }
    params_soromox["D"] = jnp.zeros((3, 3))

    params_jsrm: Dict[str, Array] = {
        "th0": jnp.array(0.0),
        "l": params_soromox["L"],
        "r": params_soromox["r"],
        "rho": params_soromox["rho"],
        "g": params_soromox["g"],
        "E": params_soromox["E"],
        "G": params_soromox["G"],
        "D": params_soromox["D"],
    }

    impls = [
        build_jsrm_impl(
            params=params_jsrm,
            eps=1e-6,
            label=r"JSRM ($\varepsilon>0$)",
            color="C0",
            linestyle="--",
            linewidth=2.6,
            marker="o",
            markevery=7000,
            zorder=3,
        ),
        build_jsrm_impl(
            params=params_jsrm,
            eps=0.0,
            label=r"JSRM ($\varepsilon=0$)",
            color="C1",
            linestyle=":",
            linewidth=2.4,
            marker="s",
            markevery=7000,
            zorder=4,
        ),
        build_soromox_impl(params=params_soromox, color="C2"),
    ]

    q_values = jnp.linspace(-2e-6, 2e-6, 100001)  # includes q=0 exactly
    if float(jnp.min(jnp.abs(q_values))) > 0.0:
        q_values = jnp.sort(jnp.concatenate([q_values, jnp.array([0.0])]))

    chi0_eps = impls[0].forward_fn(jnp.array([0.0]))
    print("Forward kinematics at q=0 with JSRM (eps>0):", chi0_eps)

    chi0_no_eps = impls[1].forward_fn(jnp.array([0.0]))
    print("Forward kinematics at q=0 with JSRM (eps=0):", chi0_no_eps)

    chi0_soromox = impls[2].forward_fn(jnp.array([0.0]))
    print("Forward kinematics at q=0 with SoRoMoX:", chi0_soromox)

    results = {impl.name: evaluate_impl(impl, q_values) for impl in impls}

    chi_eps = results["JSRM ($\\varepsilon>0$)"]["chi"]
    print("chi_eps:", chi_eps)
    print("min of y (eps>0):", jnp.min(chi_eps[:, 1]))
    print("max of y (eps>0):", jnp.max(chi_eps[:, 1]))

    chi_no_eps = results["JSRM ($\\varepsilon=0$)"]["chi"]
    print("min of y (eps=0):", jnp.min(chi_no_eps[:, 1]))
    print("max of y (eps=0):", jnp.max(chi_no_eps[:, 1]))

    plot_forward_x_only(impls, results, q_values)
    plot_forward(impls, results, q_values)
    # plot_jacobian(impls, results, q_values)
    # plot_inertia(impls, results, q_values)
    # plot_gradients(impls, results, q_values)

    plt.show()


if __name__ == "__main__":
    main()
