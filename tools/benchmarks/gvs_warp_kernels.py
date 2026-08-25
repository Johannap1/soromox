# ruff: noqa: I001, UP018
"""Experimental fixed-shape Warp kernels for the four-segment GVS benchmark.

These kernels intentionally target the benchmark geometry (four constant-strain
segments, six link cells, and 24 active coordinates).  They are prototypes used
to measure launch topology before committing to a general public API. Explicit
``int(0)`` initializers are retained because Warp's DSL uses them to type mutable
loop variables.
"""

from __future__ import annotations

import warp as wp

wp.set_module_options({"enable_backward": False})


NUM_SEGMENTS = 4
NUM_CELLS = 6
NUM_QUADRATURE_CELLS = 5
NUM_DOFS = 24
SPATIAL_DIM = 6
MAX_DOF = 6


@wp.func
def _cross_component(
    ax: wp.float64,
    ay: wp.float64,
    az: wp.float64,
    bx: wp.float64,
    by: wp.float64,
    bz: wp.float64,
    component: int,
) -> wp.float64:
    result = wp.float64(0.0)
    if component == 0:
        result = ay * bz - az * by
    elif component == 1:
        result = az * bx - ax * bz
    else:
        result = ax * by - ay * bx
    return result


@wp.func
def _source_jacobian_velocity(
    adjoint: wp.array3d(dtype=wp.float64),
    jacobian: wp.array3d(dtype=wp.float64),
    qd: wp.array2d(dtype=wp.float64),
    environment: int,
    row: int,
) -> wp.float64:
    value = wp.float64(0.0)
    for k in range(SPATIAL_DIM):
        inner = wp.float64(0.0)
        for column in range(NUM_DOFS):
            inner += jacobian[environment, k, column] * qd[environment, column]
        value += adjoint[environment, row, k] * inner
    return value


@wp.kernel(enable_backward=False)
def fused_cell_advance_kernel(
    adjoint: wp.array3d(dtype=wp.float64),
    tangent_active: wp.array3d(dtype=wp.float64),
    link_velocity: wp.array2d(dtype=wp.float64),
    step_velocity: wp.array2d(dtype=wp.float64),
    tangent_velocity_dot: wp.array2d(dtype=wp.float64),
    qd: wp.array2d(dtype=wp.float64),
    jacobian_previous: wp.array3d(dtype=wp.float64),
    jacobian_dot_qd_previous: wp.array2d(dtype=wp.float64),
    velocity_previous: wp.array2d(dtype=wp.float64),
    gravity_previous: wp.array2d(dtype=wp.float64),
    jacobian_next: wp.array3d(dtype=wp.float64),
    jacobian_dot_qd_next: wp.array2d(dtype=wp.float64),
    velocity_next: wp.array2d(dtype=wp.float64),
    gravity_next: wp.array2d(dtype=wp.float64),
):
    """Advance one cell with one thread per output Jacobian entry."""

    linear_index = wp.tid()
    entries_per_environment = SPATIAL_DIM * NUM_DOFS
    environment = linear_index // entries_per_environment
    entry = linear_index - environment * entries_per_environment
    row = entry // NUM_DOFS
    column = entry - row * NUM_DOFS

    jacobian_value = wp.float64(0.0)
    for k in range(SPATIAL_DIM):
        jacobian_value += adjoint[environment, row, k] * (
            jacobian_previous[environment, k, column]
            + tangent_active[environment, k, column]
        )
    jacobian_next[environment, row, column] = jacobian_value

    if column == 0:
        velocity_value = wp.float64(0.0)
        gravity_value = wp.float64(0.0)
        derivative_value = wp.float64(0.0)
        for k in range(SPATIAL_DIM):
            velocity_value += adjoint[environment, row, k] * (
                velocity_previous[environment, k] + link_velocity[environment, k]
            )
            gravity_value += (
                adjoint[environment, row, k] * gravity_previous[environment, k]
            )
            derivative_value += adjoint[environment, row, k] * (
                jacobian_dot_qd_previous[environment, k]
                + tangent_velocity_dot[environment, k]
            )

        eta0 = _source_jacobian_velocity(adjoint, jacobian_previous, qd, environment, 0)
        eta1 = _source_jacobian_velocity(adjoint, jacobian_previous, qd, environment, 1)
        eta2 = _source_jacobian_velocity(adjoint, jacobian_previous, qd, environment, 2)
        eta3 = _source_jacobian_velocity(adjoint, jacobian_previous, qd, environment, 3)
        eta4 = _source_jacobian_velocity(adjoint, jacobian_previous, qd, environment, 4)
        eta5 = _source_jacobian_velocity(adjoint, jacobian_previous, qd, environment, 5)

        omega_cross = _cross_component(
            step_velocity[environment, 0],
            step_velocity[environment, 1],
            step_velocity[environment, 2],
            eta0,
            eta1,
            eta2,
            row % 3,
        )
        bracket = omega_cross
        if row >= 3:
            bracket = _cross_component(
                step_velocity[environment, 3],
                step_velocity[environment, 4],
                step_velocity[environment, 5],
                eta0,
                eta1,
                eta2,
                row - 3,
            )
            bracket += _cross_component(
                step_velocity[environment, 0],
                step_velocity[environment, 1],
                step_velocity[environment, 2],
                eta3,
                eta4,
                eta5,
                row - 3,
            )

        jacobian_dot_qd_next[environment, row] = derivative_value - bracket
        velocity_next[environment, row] = velocity_value
        gravity_next[environment, row] = gravity_value


def _fused_cell_advance(
    adjoint: wp.array3d(dtype=wp.float64),
    tangent_active: wp.array3d(dtype=wp.float64),
    link_velocity: wp.array2d(dtype=wp.float64),
    step_velocity: wp.array2d(dtype=wp.float64),
    tangent_velocity_dot: wp.array2d(dtype=wp.float64),
    qd: wp.array2d(dtype=wp.float64),
    jacobian_previous: wp.array3d(dtype=wp.float64),
    jacobian_dot_qd_previous: wp.array2d(dtype=wp.float64),
    velocity_previous: wp.array2d(dtype=wp.float64),
    gravity_previous: wp.array2d(dtype=wp.float64),
    jacobian_next: wp.array3d(dtype=wp.float64),
    jacobian_dot_qd_next: wp.array2d(dtype=wp.float64),
    velocity_next: wp.array2d(dtype=wp.float64),
    gravity_next: wp.array2d(dtype=wp.float64),
):
    wp.launch(
        fused_cell_advance_kernel,
        dim=jacobian_previous.size,
        inputs=[
            adjoint,
            tangent_active,
            link_velocity,
            step_velocity,
            tangent_velocity_dot,
            qd,
            jacobian_previous,
            jacobian_dot_qd_previous,
            velocity_previous,
            gravity_previous,
        ],
        outputs=[
            jacobian_next,
            jacobian_dot_qd_next,
            velocity_next,
            gravity_next,
        ],
        block_dim=128,
    )


fused_cell_advance = wp.jax_callable(_fused_cell_advance, num_outputs=4)


@wp.func
def _segment_source_jacobian(
    jacobian_initial: wp.array3d(dtype=wp.float64),
    jacobian_states: wp.array4d(dtype=wp.float64),
    environment: int,
    cell: int,
    row: int,
    column: int,
) -> wp.float64:
    if cell == 0:
        return jacobian_initial[environment, row, column]
    return jacobian_states[environment, cell - 1, row, column]


@wp.func
def _segment_source_vector(
    initial: wp.array2d(dtype=wp.float64),
    states: wp.array3d(dtype=wp.float64),
    environment: int,
    cell: int,
    row: int,
) -> wp.float64:
    if cell == 0:
        return initial[environment, row]
    return states[environment, cell - 1, row]


@wp.kernel(enable_backward=False)
def segment_cell_advance_kernel(
    adjoint: wp.array4d(dtype=wp.float64),
    tangent_active: wp.array4d(dtype=wp.float64),
    link_velocity: wp.array3d(dtype=wp.float64),
    step_velocity: wp.array3d(dtype=wp.float64),
    tangent_velocity_dot: wp.array3d(dtype=wp.float64),
    qd: wp.array2d(dtype=wp.float64),
    jacobian_initial: wp.array3d(dtype=wp.float64),
    jacobian_dot_qd_initial: wp.array2d(dtype=wp.float64),
    velocity_initial: wp.array2d(dtype=wp.float64),
    gravity_initial: wp.array2d(dtype=wp.float64),
    cell: int,
    jacobian_states: wp.array4d(dtype=wp.float64),
    jacobian_dot_qd_states: wp.array3d(dtype=wp.float64),
    velocity_states: wp.array3d(dtype=wp.float64),
    gravity_states: wp.array3d(dtype=wp.float64),
):
    """Advance one segment cell; six launches are captured as one JAX graph."""

    linear_index = wp.tid()
    entries_per_environment = SPATIAL_DIM * NUM_DOFS
    environment = linear_index // entries_per_environment
    entry = linear_index - environment * entries_per_environment
    row = entry // NUM_DOFS
    column = entry - row * NUM_DOFS

    jacobian_value = wp.float64(0.0)
    for k in range(SPATIAL_DIM):
        jacobian_value += adjoint[environment, cell, row, k] * (
            _segment_source_jacobian(
                jacobian_initial,
                jacobian_states,
                environment,
                cell,
                k,
                column,
            )
            + tangent_active[environment, cell, k, column]
        )
    jacobian_states[environment, cell, row, column] = jacobian_value

    if column == 0:
        eta0 = wp.float64(0.0)
        eta1 = wp.float64(0.0)
        eta2 = wp.float64(0.0)
        eta3 = wp.float64(0.0)
        eta4 = wp.float64(0.0)
        eta5 = wp.float64(0.0)
        for k in range(SPATIAL_DIM):
            inner = wp.float64(0.0)
            for source_column in range(NUM_DOFS):
                inner += (
                    _segment_source_jacobian(
                        jacobian_initial,
                        jacobian_states,
                        environment,
                        cell,
                        k,
                        source_column,
                    )
                    * qd[environment, source_column]
                )
            eta0 += adjoint[environment, cell, 0, k] * inner
            eta1 += adjoint[environment, cell, 1, k] * inner
            eta2 += adjoint[environment, cell, 2, k] * inner
            eta3 += adjoint[environment, cell, 3, k] * inner
            eta4 += adjoint[environment, cell, 4, k] * inner
            eta5 += adjoint[environment, cell, 5, k] * inner

        velocity_value = wp.float64(0.0)
        gravity_value = wp.float64(0.0)
        derivative_value = wp.float64(0.0)
        for k in range(SPATIAL_DIM):
            velocity_value += adjoint[environment, cell, row, k] * (
                _segment_source_vector(
                    velocity_initial,
                    velocity_states,
                    environment,
                    cell,
                    k,
                )
                + link_velocity[environment, cell, k]
            )
            gravity_value += adjoint[environment, cell, row, k] * (
                _segment_source_vector(
                    gravity_initial,
                    gravity_states,
                    environment,
                    cell,
                    k,
                )
            )
            derivative_value += adjoint[environment, cell, row, k] * (
                _segment_source_vector(
                    jacobian_dot_qd_initial,
                    jacobian_dot_qd_states,
                    environment,
                    cell,
                    k,
                )
                + tangent_velocity_dot[environment, cell, k]
            )

        component = row % 3
        bracket = _cross_component(
            step_velocity[environment, cell, 0],
            step_velocity[environment, cell, 1],
            step_velocity[environment, cell, 2],
            eta0,
            eta1,
            eta2,
            component,
        )
        if row >= 3:
            bracket = _cross_component(
                step_velocity[environment, cell, 3],
                step_velocity[environment, cell, 4],
                step_velocity[environment, cell, 5],
                eta0,
                eta1,
                eta2,
                component,
            )
            bracket += _cross_component(
                step_velocity[environment, cell, 0],
                step_velocity[environment, cell, 1],
                step_velocity[environment, cell, 2],
                eta3,
                eta4,
                eta5,
                component,
            )

        jacobian_dot_qd_states[environment, cell, row] = derivative_value - bracket
        velocity_states[environment, cell, row] = velocity_value
        gravity_states[environment, cell, row] = gravity_value


def _segment_recurrence(
    adjoint: wp.array4d(dtype=wp.float64),
    tangent_active: wp.array4d(dtype=wp.float64),
    link_velocity: wp.array3d(dtype=wp.float64),
    step_velocity: wp.array3d(dtype=wp.float64),
    tangent_velocity_dot: wp.array3d(dtype=wp.float64),
    qd: wp.array2d(dtype=wp.float64),
    jacobian_initial: wp.array3d(dtype=wp.float64),
    jacobian_dot_qd_initial: wp.array2d(dtype=wp.float64),
    velocity_initial: wp.array2d(dtype=wp.float64),
    gravity_initial: wp.array2d(dtype=wp.float64),
    jacobian_states: wp.array4d(dtype=wp.float64),
    jacobian_dot_qd_states: wp.array3d(dtype=wp.float64),
    velocity_states: wp.array3d(dtype=wp.float64),
    gravity_states: wp.array3d(dtype=wp.float64),
):
    for cell in range(NUM_CELLS):
        wp.launch(
            segment_cell_advance_kernel,
            dim=jacobian_initial.size,
            inputs=[
                adjoint,
                tangent_active,
                link_velocity,
                step_velocity,
                tangent_velocity_dot,
                qd,
                jacobian_initial,
                jacobian_dot_qd_initial,
                velocity_initial,
                gravity_initial,
                cell,
            ],
            outputs=[
                jacobian_states,
                jacobian_dot_qd_states,
                velocity_states,
                gravity_states,
            ],
            block_dim=128,
        )


segment_recurrence = wp.jax_callable(_segment_recurrence, num_outputs=4)


def _rejected_monolithic_cooperative_segment_dynamics_kernel(
    adjoint: wp.array2d(dtype=wp.float64),
    tangent_active: wp.array2d(dtype=wp.float64),
    link_velocity: wp.array2d(dtype=wp.float64),
    step_velocity: wp.array2d(dtype=wp.float64),
    tangent_velocity_dot: wp.array2d(dtype=wp.float64),
    qd: wp.array2d(dtype=wp.float64),
    jacobian_initial: wp.array2d(dtype=wp.float64),
    jacobian_dot_qd_initial: wp.array2d(dtype=wp.float64),
    velocity_initial: wp.array2d(dtype=wp.float64),
    gravity_initial: wp.array2d(dtype=wp.float64),
    weights: wp.array(dtype=wp.float64),
    masses: wp.array2d(dtype=wp.float64),
    jacobian_tip: wp.array2d(dtype=wp.float64),
    jacobian_dot_qd_tip: wp.array2d(dtype=wp.float64),
    velocity_tip: wp.array2d(dtype=wp.float64),
    gravity_tip: wp.array2d(dtype=wp.float64),
    inertia: wp.array2d(dtype=wp.float64),
    coriolis_qd: wp.array2d(dtype=wp.float64),
    gravity_force: wp.array2d(dtype=wp.float64),
):
    """Advance one complete segment cooperatively inside one CUDA block.

    The leading launch dimension identifies an environment; the implicit final
    tiled-launch dimension assigns a block of threads to that environment.  The
    six causal cells are advanced in order, while Warp tile operations spread
    each 6x24 Jacobian update and each quadrature contraction over the block.
    Only the segment tip and accumulated dynamics terms reach global memory.
    """

    environment, _lane = wp.tid()
    jacobian = wp.tile_load(
        jacobian_initial,
        shape=(SPATIAL_DIM, NUM_DOFS),
        offset=(environment * SPATIAL_DIM, 0),
        storage="shared",
    )
    jacobian_dot_qd = wp.tile_load(
        jacobian_dot_qd_initial,
        shape=(SPATIAL_DIM, 1),
        offset=(environment * SPATIAL_DIM, 0),
        storage="shared",
    )
    velocity = wp.tile_load(
        velocity_initial,
        shape=(SPATIAL_DIM, 1),
        offset=(environment * SPATIAL_DIM, 0),
        storage="shared",
    )
    gravity = wp.tile_load(
        gravity_initial,
        shape=(SPATIAL_DIM, 1),
        offset=(environment * SPATIAL_DIM, 0),
        storage="shared",
    )
    generalized_velocity = wp.tile_load(
        qd,
        shape=(NUM_DOFS, 1),
        offset=(environment * NUM_DOFS, 0),
        storage="shared",
    )
    inertia_segment = wp.tile_zeros(
        shape=(NUM_DOFS, NUM_DOFS), dtype=wp.float64, storage="shared"
    )
    coriolis_segment = wp.tile_zeros(
        shape=(NUM_DOFS, 1), dtype=wp.float64, storage="shared"
    )
    gravity_segment = wp.tile_zeros(
        shape=(NUM_DOFS, 1), dtype=wp.float64, storage="shared"
    )

    for cell in range(NUM_CELLS):
        cell_matrix_row = (environment * NUM_CELLS + cell) * SPATIAL_DIM
        cell_vector_row = cell_matrix_row
        adjoint_cell = wp.tile_load(
            adjoint,
            shape=(SPATIAL_DIM, SPATIAL_DIM),
            offset=(cell_matrix_row, 0),
        )
        tangent_cell = wp.tile_load(
            tangent_active,
            shape=(SPATIAL_DIM, NUM_DOFS),
            offset=(cell_matrix_row, 0),
        )
        link_velocity_cell = wp.tile_load(
            link_velocity,
            shape=(SPATIAL_DIM, 1),
            offset=(cell_vector_row, 0),
        )
        step_velocity_cell = wp.tile_load(
            step_velocity,
            shape=(SPATIAL_DIM, 1),
            offset=(cell_vector_row, 0),
        )
        tangent_velocity_dot_cell = wp.tile_load(
            tangent_velocity_dot,
            shape=(SPATIAL_DIM, 1),
            offset=(cell_vector_row, 0),
        )

        source_jacobian_velocity = wp.tile_zeros(
            shape=(SPATIAL_DIM, 1), dtype=wp.float64
        )
        wp.tile_matmul(jacobian, generalized_velocity, source_jacobian_velocity)
        transported_source_velocity = wp.tile_zeros(
            shape=(SPATIAL_DIM, 1), dtype=wp.float64
        )
        wp.tile_matmul(
            adjoint_cell,
            source_jacobian_velocity,
            transported_source_velocity,
        )

        jacobian_next = wp.tile_zeros(shape=(SPATIAL_DIM, NUM_DOFS), dtype=wp.float64)
        wp.tile_matmul(adjoint_cell, jacobian + tangent_cell, jacobian_next)
        velocity_next = wp.tile_zeros(shape=(SPATIAL_DIM, 1), dtype=wp.float64)
        wp.tile_matmul(adjoint_cell, velocity + link_velocity_cell, velocity_next)
        gravity_next = wp.tile_zeros(shape=(SPATIAL_DIM, 1), dtype=wp.float64)
        wp.tile_matmul(adjoint_cell, gravity, gravity_next)
        jacobian_dot_qd_next = wp.tile_zeros(shape=(SPATIAL_DIM, 1), dtype=wp.float64)
        wp.tile_matmul(
            adjoint_cell,
            jacobian_dot_qd + tangent_velocity_dot_cell,
            jacobian_dot_qd_next,
        )

        step0 = wp.tile_extract(step_velocity_cell, 0, 0)
        step1 = wp.tile_extract(step_velocity_cell, 1, 0)
        step2 = wp.tile_extract(step_velocity_cell, 2, 0)
        step3 = wp.tile_extract(step_velocity_cell, 3, 0)
        step4 = wp.tile_extract(step_velocity_cell, 4, 0)
        step5 = wp.tile_extract(step_velocity_cell, 5, 0)
        eta0 = wp.tile_extract(transported_source_velocity, 0, 0)
        eta1 = wp.tile_extract(transported_source_velocity, 1, 0)
        eta2 = wp.tile_extract(transported_source_velocity, 2, 0)
        eta3 = wp.tile_extract(transported_source_velocity, 3, 0)
        eta4 = wp.tile_extract(transported_source_velocity, 4, 0)
        eta5 = wp.tile_extract(transported_source_velocity, 5, 0)
        bracket = wp.tile_zeros(
            shape=(SPATIAL_DIM, 1), dtype=wp.float64, storage="shared"
        )
        bracket0 = step1 * eta2 - step2 * eta1
        bracket1 = step2 * eta0 - step0 * eta2
        bracket2 = step0 * eta1 - step1 * eta0
        bracket3 = step4 * eta2 - step5 * eta1 + step1 * eta5 - step2 * eta4
        bracket4 = step5 * eta0 - step3 * eta2 + step2 * eta3 - step0 * eta5
        bracket5 = step3 * eta1 - step4 * eta0 + step0 * eta4 - step1 * eta3
        wp.tile_assign(
            bracket,
            wp.tile_full(
                shape=(1, 1), value=bracket0, dtype=wp.float64, storage="register"
            ),
            offset=(0, 0),
        )
        wp.tile_assign(
            bracket,
            wp.tile_full(
                shape=(1, 1), value=bracket1, dtype=wp.float64, storage="register"
            ),
            offset=(1, 0),
        )
        wp.tile_assign(
            bracket,
            wp.tile_full(
                shape=(1, 1), value=bracket2, dtype=wp.float64, storage="register"
            ),
            offset=(2, 0),
        )
        wp.tile_assign(
            bracket,
            wp.tile_full(
                shape=(1, 1), value=bracket3, dtype=wp.float64, storage="register"
            ),
            offset=(3, 0),
        )
        wp.tile_assign(
            bracket,
            wp.tile_full(
                shape=(1, 1), value=bracket4, dtype=wp.float64, storage="register"
            ),
            offset=(4, 0),
        )
        wp.tile_assign(
            bracket,
            wp.tile_full(
                shape=(1, 1), value=bracket5, dtype=wp.float64, storage="register"
            ),
            offset=(5, 0),
        )
        jacobian_dot_qd_next -= bracket

        wp.tile_assign(jacobian, jacobian_next, offset=(0, 0))
        wp.tile_assign(jacobian_dot_qd, jacobian_dot_qd_next, offset=(0, 0))
        wp.tile_assign(velocity, velocity_next, offset=(0, 0))
        wp.tile_assign(gravity, gravity_next, offset=(0, 0))

        if cell < NUM_QUADRATURE_CELLS:
            mass_row = cell * SPATIAL_DIM
            mass = wp.tile_load(
                masses,
                shape=(SPATIAL_DIM, 1),
                offset=(mass_row, 0),
            )
            weight = weights[cell]
            momentum = mass * velocity

            omega0 = wp.tile_extract(velocity, 0, 0)
            omega1 = wp.tile_extract(velocity, 1, 0)
            omega2 = wp.tile_extract(velocity, 2, 0)
            linear0 = wp.tile_extract(velocity, 3, 0)
            linear1 = wp.tile_extract(velocity, 4, 0)
            linear2 = wp.tile_extract(velocity, 5, 0)
            moment0 = wp.tile_extract(momentum, 0, 0)
            moment1 = wp.tile_extract(momentum, 1, 0)
            moment2 = wp.tile_extract(momentum, 2, 0)
            force0 = wp.tile_extract(momentum, 3, 0)
            force1 = wp.tile_extract(momentum, 4, 0)
            force2 = wp.tile_extract(momentum, 5, 0)
            coadjoint = wp.tile_zeros(
                shape=(SPATIAL_DIM, 1), dtype=wp.float64, storage="shared"
            )
            coadjoint0 = (
                omega1 * moment2
                - omega2 * moment1
                + linear1 * force2
                - linear2 * force1
            )
            coadjoint1 = (
                omega2 * moment0
                - omega0 * moment2
                + linear2 * force0
                - linear0 * force2
            )
            coadjoint2 = (
                omega0 * moment1
                - omega1 * moment0
                + linear0 * force1
                - linear1 * force0
            )
            coadjoint3 = omega1 * force2 - omega2 * force1
            coadjoint4 = omega2 * force0 - omega0 * force2
            coadjoint5 = omega0 * force1 - omega1 * force0
            wp.tile_assign(
                coadjoint,
                wp.tile_full(
                    shape=(1, 1),
                    value=coadjoint0,
                    dtype=wp.float64,
                    storage="register",
                ),
                offset=(0, 0),
            )
            wp.tile_assign(
                coadjoint,
                wp.tile_full(
                    shape=(1, 1),
                    value=coadjoint1,
                    dtype=wp.float64,
                    storage="register",
                ),
                offset=(1, 0),
            )
            wp.tile_assign(
                coadjoint,
                wp.tile_full(
                    shape=(1, 1),
                    value=coadjoint2,
                    dtype=wp.float64,
                    storage="register",
                ),
                offset=(2, 0),
            )
            wp.tile_assign(
                coadjoint,
                wp.tile_full(
                    shape=(1, 1),
                    value=coadjoint3,
                    dtype=wp.float64,
                    storage="register",
                ),
                offset=(3, 0),
            )
            wp.tile_assign(
                coadjoint,
                wp.tile_full(
                    shape=(1, 1),
                    value=coadjoint4,
                    dtype=wp.float64,
                    storage="register",
                ),
                offset=(4, 0),
            )
            wp.tile_assign(
                coadjoint,
                wp.tile_full(
                    shape=(1, 1),
                    value=coadjoint5,
                    dtype=wp.float64,
                    storage="register",
                ),
                offset=(5, 0),
            )

            weighted_mass = mass * weight
            weighted_jacobian = jacobian * wp.tile_broadcast(
                weighted_mass, shape=(SPATIAL_DIM, NUM_DOFS)
            )
            wp.tile_matmul(
                wp.tile_transpose(jacobian),
                weighted_jacobian,
                inertia_segment,
            )
            wrench = mass * jacobian_dot_qd + coadjoint
            wp.tile_matmul(
                wp.tile_transpose(jacobian),
                wrench * weight,
                coriolis_segment,
            )
            wp.tile_matmul(
                wp.tile_transpose(jacobian),
                gravity * weighted_mass,
                gravity_segment,
                alpha=-1.0,
            )

    wp.tile_store(jacobian_tip, jacobian, offset=(environment * SPATIAL_DIM, 0))
    wp.tile_store(
        jacobian_dot_qd_tip,
        jacobian_dot_qd,
        offset=(environment * SPATIAL_DIM, 0),
    )
    wp.tile_store(velocity_tip, velocity, offset=(environment * SPATIAL_DIM, 0))
    wp.tile_store(gravity_tip, gravity, offset=(environment * SPATIAL_DIM, 0))
    wp.tile_store(inertia, inertia_segment, offset=(environment * NUM_DOFS, 0))
    wp.tile_store(coriolis_qd, coriolis_segment, offset=(environment * NUM_DOFS, 0))
    wp.tile_store(gravity_force, gravity_segment, offset=(environment * NUM_DOFS, 0))


def _cooperative_segment_dynamics(
    adjoint: wp.array2d(dtype=wp.float64),
    tangent_active: wp.array2d(dtype=wp.float64),
    link_velocity: wp.array2d(dtype=wp.float64),
    step_velocity: wp.array2d(dtype=wp.float64),
    tangent_velocity_dot: wp.array2d(dtype=wp.float64),
    qd: wp.array2d(dtype=wp.float64),
    jacobian_initial: wp.array2d(dtype=wp.float64),
    jacobian_dot_qd_initial: wp.array2d(dtype=wp.float64),
    velocity_initial: wp.array2d(dtype=wp.float64),
    gravity_initial: wp.array2d(dtype=wp.float64),
    weights: wp.array(dtype=wp.float64),
    masses: wp.array2d(dtype=wp.float64),
    jacobian_tip: wp.array2d(dtype=wp.float64),
    jacobian_dot_qd_tip: wp.array2d(dtype=wp.float64),
    velocity_tip: wp.array2d(dtype=wp.float64),
    gravity_tip: wp.array2d(dtype=wp.float64),
    inertia: wp.array2d(dtype=wp.float64),
    coriolis_qd: wp.array2d(dtype=wp.float64),
    gravity_force: wp.array2d(dtype=wp.float64),
):
    batch_size = qd.shape[0] // NUM_DOFS
    wp.launch_tiled(
        _rejected_monolithic_cooperative_segment_dynamics_kernel,
        dim=batch_size,
        inputs=[
            adjoint,
            tangent_active,
            link_velocity,
            step_velocity,
            tangent_velocity_dot,
            qd,
            jacobian_initial,
            jacobian_dot_qd_initial,
            velocity_initial,
            gravity_initial,
            weights,
            masses,
        ],
        outputs=[
            jacobian_tip,
            jacobian_dot_qd_tip,
            velocity_tip,
            gravity_tip,
            inertia,
            coriolis_qd,
            gravity_force,
        ],
        block_dim=128,
    )


# This exact monolithic tile experiment is intentionally not registered as a
# Warp kernel.  It is correct on Warp's CPU lowering but its CUDA compilation
# did not complete after six minutes.  The split recurrence/assembly variant
# below preserves the cooperative recurrence and is practical to compile.


@wp.kernel(enable_backward=False)
def cooperative_segment_recurrence_kernel(
    adjoint: wp.array2d(dtype=wp.float64),
    tangent_active: wp.array2d(dtype=wp.float64),
    link_velocity: wp.array2d(dtype=wp.float64),
    step_velocity: wp.array2d(dtype=wp.float64),
    tangent_velocity_dot: wp.array2d(dtype=wp.float64),
    qd: wp.array2d(dtype=wp.float64),
    jacobian_initial: wp.array2d(dtype=wp.float64),
    jacobian_dot_qd_initial: wp.array2d(dtype=wp.float64),
    velocity_initial: wp.array2d(dtype=wp.float64),
    gravity_initial: wp.array2d(dtype=wp.float64),
    jacobian_states: wp.array2d(dtype=wp.float64),
    jacobian_dot_qd_states: wp.array2d(dtype=wp.float64),
    velocity_states: wp.array2d(dtype=wp.float64),
    gravity_states: wp.array2d(dtype=wp.float64),
):
    """Advance all six cells for one environment in one cooperative block."""

    environment, _lane = wp.tid()
    jacobian = wp.tile_load(
        jacobian_initial,
        shape=(SPATIAL_DIM, NUM_DOFS),
        offset=(environment * SPATIAL_DIM, 0),
        storage="shared",
    )
    jacobian_dot_qd = wp.tile_load(
        jacobian_dot_qd_initial,
        shape=(SPATIAL_DIM, 1),
        offset=(environment * SPATIAL_DIM, 0),
        storage="shared",
    )
    velocity = wp.tile_load(
        velocity_initial,
        shape=(SPATIAL_DIM, 1),
        offset=(environment * SPATIAL_DIM, 0),
        storage="shared",
    )
    gravity = wp.tile_load(
        gravity_initial,
        shape=(SPATIAL_DIM, 1),
        offset=(environment * SPATIAL_DIM, 0),
        storage="shared",
    )
    generalized_velocity = wp.tile_load(
        qd,
        shape=(NUM_DOFS, 1),
        offset=(environment * NUM_DOFS, 0),
        storage="shared",
    )

    for cell in range(NUM_CELLS):
        cell_matrix_row = (environment * NUM_CELLS + cell) * SPATIAL_DIM
        adjoint_cell = wp.tile_load(
            adjoint,
            shape=(SPATIAL_DIM, SPATIAL_DIM),
            offset=(cell_matrix_row, 0),
        )
        tangent_cell = wp.tile_load(
            tangent_active,
            shape=(SPATIAL_DIM, NUM_DOFS),
            offset=(cell_matrix_row, 0),
        )
        link_velocity_cell = wp.tile_load(
            link_velocity,
            shape=(SPATIAL_DIM, 1),
            offset=(cell_matrix_row, 0),
        )
        step_velocity_cell = wp.tile_load(
            step_velocity,
            shape=(SPATIAL_DIM, 1),
            offset=(cell_matrix_row, 0),
        )
        tangent_velocity_dot_cell = wp.tile_load(
            tangent_velocity_dot,
            shape=(SPATIAL_DIM, 1),
            offset=(cell_matrix_row, 0),
        )

        source_jacobian_velocity = wp.tile_matmul(jacobian, generalized_velocity)
        transported_source_velocity = wp.tile_matmul(
            adjoint_cell, source_jacobian_velocity
        )
        jacobian_next = wp.tile_matmul(adjoint_cell, jacobian + tangent_cell)
        velocity_next = wp.tile_matmul(adjoint_cell, velocity + link_velocity_cell)
        gravity_next = wp.tile_matmul(adjoint_cell, gravity)
        jacobian_dot_qd_next = wp.tile_matmul(
            adjoint_cell, jacobian_dot_qd + tangent_velocity_dot_cell
        )

        step0 = wp.tile_extract(step_velocity_cell, 0, 0)
        step1 = wp.tile_extract(step_velocity_cell, 1, 0)
        step2 = wp.tile_extract(step_velocity_cell, 2, 0)
        step3 = wp.tile_extract(step_velocity_cell, 3, 0)
        step4 = wp.tile_extract(step_velocity_cell, 4, 0)
        step5 = wp.tile_extract(step_velocity_cell, 5, 0)
        eta0 = wp.tile_extract(transported_source_velocity, 0, 0)
        eta1 = wp.tile_extract(transported_source_velocity, 1, 0)
        eta2 = wp.tile_extract(transported_source_velocity, 2, 0)
        eta3 = wp.tile_extract(transported_source_velocity, 3, 0)
        eta4 = wp.tile_extract(transported_source_velocity, 4, 0)
        eta5 = wp.tile_extract(transported_source_velocity, 5, 0)
        bracket = wp.tile_zeros(
            shape=(SPATIAL_DIM, 1), dtype=wp.float64, storage="shared"
        )
        wp.tile_assign(
            bracket,
            wp.tile_full(
                shape=(1, 1),
                value=step1 * eta2 - step2 * eta1,
                dtype=wp.float64,
            ),
            offset=(0, 0),
        )
        wp.tile_assign(
            bracket,
            wp.tile_full(
                shape=(1, 1),
                value=step2 * eta0 - step0 * eta2,
                dtype=wp.float64,
            ),
            offset=(1, 0),
        )
        wp.tile_assign(
            bracket,
            wp.tile_full(
                shape=(1, 1),
                value=step0 * eta1 - step1 * eta0,
                dtype=wp.float64,
            ),
            offset=(2, 0),
        )
        wp.tile_assign(
            bracket,
            wp.tile_full(
                shape=(1, 1),
                value=(step4 * eta2 - step5 * eta1 + step1 * eta5 - step2 * eta4),
                dtype=wp.float64,
            ),
            offset=(3, 0),
        )
        wp.tile_assign(
            bracket,
            wp.tile_full(
                shape=(1, 1),
                value=(step5 * eta0 - step3 * eta2 + step2 * eta3 - step0 * eta5),
                dtype=wp.float64,
            ),
            offset=(4, 0),
        )
        wp.tile_assign(
            bracket,
            wp.tile_full(
                shape=(1, 1),
                value=(step3 * eta1 - step4 * eta0 + step0 * eta4 - step1 * eta3),
                dtype=wp.float64,
            ),
            offset=(5, 0),
        )
        jacobian_dot_qd_next -= bracket

        wp.tile_assign(jacobian, jacobian_next, offset=(0, 0))
        wp.tile_assign(jacobian_dot_qd, jacobian_dot_qd_next, offset=(0, 0))
        wp.tile_assign(velocity, velocity_next, offset=(0, 0))
        wp.tile_assign(gravity, gravity_next, offset=(0, 0))

        state_row = (environment * NUM_CELLS + cell) * SPATIAL_DIM
        wp.tile_store(jacobian_states, jacobian, offset=(state_row, 0))
        wp.tile_store(jacobian_dot_qd_states, jacobian_dot_qd, offset=(state_row, 0))
        wp.tile_store(velocity_states, velocity, offset=(state_row, 0))
        wp.tile_store(gravity_states, gravity, offset=(state_row, 0))


@wp.kernel(enable_backward=False)
def cooperative_segment_assembly_kernel(
    jacobian_states: wp.array2d(dtype=wp.float64),
    jacobian_dot_qd_states: wp.array2d(dtype=wp.float64),
    velocity_states: wp.array2d(dtype=wp.float64),
    gravity_states: wp.array2d(dtype=wp.float64),
    weights: wp.array(dtype=wp.float64),
    masses: wp.array2d(dtype=wp.float64),
    inertia: wp.array2d(dtype=wp.float64),
    coriolis_qd: wp.array2d(dtype=wp.float64),
    gravity_force: wp.array2d(dtype=wp.float64),
):
    """Assemble one segment with one thread per 24x24 inertia entry."""

    linear_index = wp.tid()
    entries_per_environment = NUM_DOFS * NUM_DOFS
    environment = linear_index // entries_per_environment
    entry = linear_index - environment * entries_per_environment
    coordinate = entry // NUM_DOFS
    other_coordinate = entry - coordinate * NUM_DOFS

    inertia_value = wp.float64(0.0)
    coriolis_value = wp.float64(0.0)
    gravity_value = wp.float64(0.0)
    for cell in range(NUM_QUADRATURE_CELLS):
        weight = weights[cell]
        state_base = (environment * NUM_CELLS + cell) * SPATIAL_DIM
        for row in range(SPATIAL_DIM):
            state_row = state_base + row
            mass = masses[cell * SPATIAL_DIM + row, 0]
            jacobian_value = jacobian_states[state_row, coordinate]
            inertia_value += (
                jacobian_value
                * weight
                * mass
                * jacobian_states[state_row, other_coordinate]
            )

            if other_coordinate == 0:
                omega0 = velocity_states[state_base + 0, 0]
                omega1 = velocity_states[state_base + 1, 0]
                omega2 = velocity_states[state_base + 2, 0]
                linear0 = velocity_states[state_base + 3, 0]
                linear1 = velocity_states[state_base + 4, 0]
                linear2 = velocity_states[state_base + 5, 0]
                moment0 = masses[cell * SPATIAL_DIM + 0, 0] * omega0
                moment1 = masses[cell * SPATIAL_DIM + 1, 0] * omega1
                moment2 = masses[cell * SPATIAL_DIM + 2, 0] * omega2
                force0 = masses[cell * SPATIAL_DIM + 3, 0] * linear0
                force1 = masses[cell * SPATIAL_DIM + 4, 0] * linear1
                force2 = masses[cell * SPATIAL_DIM + 5, 0] * linear2
                coadjoint_value = _cross_component(
                    omega0,
                    omega1,
                    omega2,
                    moment0,
                    moment1,
                    moment2,
                    row % 3,
                )
                if row < 3:
                    coadjoint_value += _cross_component(
                        linear0,
                        linear1,
                        linear2,
                        force0,
                        force1,
                        force2,
                        row,
                    )
                else:
                    coadjoint_value = _cross_component(
                        omega0,
                        omega1,
                        omega2,
                        force0,
                        force1,
                        force2,
                        row - 3,
                    )
                wrench = mass * jacobian_dot_qd_states[state_row, 0] + coadjoint_value
                coriolis_value += jacobian_value * weight * wrench
                gravity_value -= (
                    jacobian_value * weight * mass * gravity_states[state_row, 0]
                )

    inertia[environment * NUM_DOFS + coordinate, other_coordinate] = inertia_value
    if other_coordinate == 0:
        coriolis_qd[environment * NUM_DOFS + coordinate, 0] = coriolis_value
        gravity_force[environment * NUM_DOFS + coordinate, 0] = gravity_value


def _cooperative_segment_split(
    adjoint: wp.array2d(dtype=wp.float64),
    tangent_active: wp.array2d(dtype=wp.float64),
    link_velocity: wp.array2d(dtype=wp.float64),
    step_velocity: wp.array2d(dtype=wp.float64),
    tangent_velocity_dot: wp.array2d(dtype=wp.float64),
    qd: wp.array2d(dtype=wp.float64),
    jacobian_initial: wp.array2d(dtype=wp.float64),
    jacobian_dot_qd_initial: wp.array2d(dtype=wp.float64),
    velocity_initial: wp.array2d(dtype=wp.float64),
    gravity_initial: wp.array2d(dtype=wp.float64),
    weights: wp.array(dtype=wp.float64),
    masses: wp.array2d(dtype=wp.float64),
    jacobian_states: wp.array2d(dtype=wp.float64),
    jacobian_dot_qd_states: wp.array2d(dtype=wp.float64),
    velocity_states: wp.array2d(dtype=wp.float64),
    gravity_states: wp.array2d(dtype=wp.float64),
    inertia: wp.array2d(dtype=wp.float64),
    coriolis_qd: wp.array2d(dtype=wp.float64),
    gravity_force: wp.array2d(dtype=wp.float64),
):
    batch_size = qd.shape[0] // NUM_DOFS
    wp.launch_tiled(
        cooperative_segment_recurrence_kernel,
        dim=batch_size,
        inputs=[
            adjoint,
            tangent_active,
            link_velocity,
            step_velocity,
            tangent_velocity_dot,
            qd,
            jacobian_initial,
            jacobian_dot_qd_initial,
            velocity_initial,
            gravity_initial,
        ],
        outputs=[
            jacobian_states,
            jacobian_dot_qd_states,
            velocity_states,
            gravity_states,
        ],
        block_dim=128,
    )
    wp.launch(
        cooperative_segment_assembly_kernel,
        dim=batch_size * NUM_DOFS * NUM_DOFS,
        inputs=[
            jacobian_states,
            jacobian_dot_qd_states,
            velocity_states,
            gravity_states,
            weights,
            masses,
        ],
        outputs=[inertia, coriolis_qd, gravity_force],
        block_dim=128,
    )


cooperative_segment_dynamics = wp.jax_callable(
    _cooperative_segment_split, num_outputs=7
)


@wp.kernel(enable_backward=False)
def contract_cell_terms_kernel(
    tangent: wp.array3d(dtype=wp.float64),
    tangent_dot: wp.array3d(dtype=wp.float64),
    adjoint: wp.array3d(dtype=wp.float64),
    magnus_basis: wp.array3d(dtype=wp.float64),
    magnus_basis_dot: wp.array3d(dtype=wp.float64),
    qd_link: wp.array3d(dtype=wp.float64),
    gather_indices: wp.array2d(dtype=wp.int32),
    gather_mask: wp.array2d(dtype=wp.float64),
    tangent_active: wp.array3d(dtype=wp.float64),
    link_velocity: wp.array2d(dtype=wp.float64),
    step_velocity: wp.array2d(dtype=wp.float64),
    tangent_velocity_dot: wp.array2d(dtype=wp.float64),
):
    """Parallel contraction of local cell terms over environments and cells."""

    linear_index = wp.tid()
    entries_per_cell = SPATIAL_DIM * NUM_DOFS
    work_item = linear_index // entries_per_cell
    entry = linear_index - work_item * entries_per_cell
    work_items_per_environment = NUM_SEGMENTS * NUM_CELLS
    environment = work_item // work_items_per_environment
    environment_remainder = work_item - environment * work_items_per_environment
    segment = environment_remainder // NUM_CELLS
    row = entry // NUM_DOFS
    column = entry - row * NUM_DOFS

    active_value = wp.float64(0.0)
    for local_column in range(MAX_DOF):
        if gather_indices[segment, local_column] == column and gather_mask[
            segment, local_column
        ] != wp.float64(0.0):
            local_value = wp.float64(0.0)
            for k in range(SPATIAL_DIM):
                local_value += (
                    tangent[work_item, row, k]
                    * (magnus_basis[work_item, k, local_column])
                )
            active_value += local_value * gather_mask[segment, local_column]
    tangent_active[work_item, row, column] = active_value

    if column == 0:
        link_value = wp.float64(0.0)
        tangent_dot_value = wp.float64(0.0)
        for local_column in range(MAX_DOF):
            local_value = wp.float64(0.0)
            local_dot_value = wp.float64(0.0)
            for k in range(SPATIAL_DIM):
                local_value += (
                    tangent[work_item, row, k]
                    * (magnus_basis[work_item, k, local_column])
                )
                local_dot_value += (
                    tangent_dot[work_item, row, k]
                    * magnus_basis[work_item, k, local_column]
                    + tangent[work_item, row, k]
                    * magnus_basis_dot[work_item, k, local_column]
                )
            link_value += local_value * qd_link[environment, segment, local_column]
            tangent_dot_value += (
                local_dot_value * qd_link[environment, segment, local_column]
            )

        step_value = wp.float64(0.0)
        for k in range(SPATIAL_DIM):
            link_k = wp.float64(0.0)
            for local_column in range(MAX_DOF):
                local_k = wp.float64(0.0)
                for inner in range(SPATIAL_DIM):
                    local_k += (
                        tangent[work_item, k, inner]
                        * (magnus_basis[work_item, inner, local_column])
                    )
                link_k += local_k * qd_link[environment, segment, local_column]
            step_value += adjoint[work_item, row, k] * link_k

        link_velocity[work_item, row] = link_value
        step_velocity[work_item, row] = step_value
        tangent_velocity_dot[work_item, row] = tangent_dot_value


def _contract_cell_terms(
    tangent: wp.array3d(dtype=wp.float64),
    tangent_dot: wp.array3d(dtype=wp.float64),
    adjoint: wp.array3d(dtype=wp.float64),
    magnus_basis: wp.array3d(dtype=wp.float64),
    magnus_basis_dot: wp.array3d(dtype=wp.float64),
    qd_link: wp.array3d(dtype=wp.float64),
    gather_indices: wp.array2d(dtype=wp.int32),
    gather_mask: wp.array2d(dtype=wp.float64),
    tangent_active: wp.array3d(dtype=wp.float64),
    link_velocity: wp.array2d(dtype=wp.float64),
    step_velocity: wp.array2d(dtype=wp.float64),
    tangent_velocity_dot: wp.array2d(dtype=wp.float64),
):
    wp.launch(
        contract_cell_terms_kernel,
        dim=tangent_active.size,
        inputs=[
            tangent,
            tangent_dot,
            adjoint,
            magnus_basis,
            magnus_basis_dot,
            qd_link,
            gather_indices,
            gather_mask,
        ],
        outputs=[
            tangent_active,
            link_velocity,
            step_velocity,
            tangent_velocity_dot,
        ],
        block_dim=128,
    )


contract_cell_terms = wp.jax_callable(_contract_cell_terms, num_outputs=4)


@wp.func
def _raw_local_basis_value(
    tangent: wp.array3d(dtype=wp.float64),
    magnus_basis: wp.array3d(dtype=wp.float64),
    work_item: int,
    row: int,
    local_column: int,
) -> wp.float64:
    value = wp.float64(0.0)
    k = int(0)
    while k < SPATIAL_DIM:
        value += tangent[work_item, row, k] * magnus_basis[work_item, k, local_column]
        k += 1
    return value


@wp.func
def _raw_active_basis_value(
    tangent: wp.array3d(dtype=wp.float64),
    magnus_basis: wp.array3d(dtype=wp.float64),
    gather_indices: wp.array2d(dtype=wp.int32),
    gather_mask: wp.array2d(dtype=wp.float64),
    work_item: int,
    segment: int,
    row: int,
    column: int,
) -> wp.float64:
    value = wp.float64(0.0)
    local_column = int(0)
    while local_column < MAX_DOF:
        if gather_indices[segment, local_column] == column:
            value += (
                _raw_local_basis_value(
                    tangent, magnus_basis, work_item, row, local_column
                )
                * gather_mask[segment, local_column]
            )
        local_column += 1
    return value


@wp.func
def _raw_link_velocity_value(
    tangent: wp.array3d(dtype=wp.float64),
    magnus_basis: wp.array3d(dtype=wp.float64),
    qd_link: wp.array3d(dtype=wp.float64),
    work_item: int,
    environment: int,
    segment: int,
    row: int,
) -> wp.float64:
    value = wp.float64(0.0)
    local_column = int(0)
    while local_column < MAX_DOF:
        value += (
            _raw_local_basis_value(tangent, magnus_basis, work_item, row, local_column)
            * qd_link[environment, segment, local_column]
        )
        local_column += 1
    return value


@wp.func
def _raw_tangent_dot_velocity_value(
    tangent: wp.array3d(dtype=wp.float64),
    tangent_dot_values: wp.array3d(dtype=wp.float64),
    magnus_basis: wp.array3d(dtype=wp.float64),
    magnus_basis_dot: wp.array3d(dtype=wp.float64),
    qd_link: wp.array3d(dtype=wp.float64),
    work_item: int,
    environment: int,
    segment: int,
    row: int,
) -> wp.float64:
    value = wp.float64(0.0)
    local_column = int(0)
    while local_column < MAX_DOF:
        local_value = wp.float64(0.0)
        k = int(0)
        while k < SPATIAL_DIM:
            local_value += (
                tangent_dot_values[work_item, row, k]
                * magnus_basis[work_item, k, local_column]
                + tangent[work_item, row, k]
                * magnus_basis_dot[work_item, k, local_column]
            )
            k += 1
        value += local_value * qd_link[environment, segment, local_column]
        local_column += 1
    return value


@wp.func
def _raw_step_velocity_value(
    adjoint: wp.array3d(dtype=wp.float64),
    tangent: wp.array3d(dtype=wp.float64),
    magnus_basis: wp.array3d(dtype=wp.float64),
    qd_link: wp.array3d(dtype=wp.float64),
    work_item: int,
    environment: int,
    segment: int,
    row: int,
) -> wp.float64:
    value = wp.float64(0.0)
    k = int(0)
    while k < SPATIAL_DIM:
        value += adjoint[work_item, row, k] * _raw_link_velocity_value(
            tangent,
            magnus_basis,
            qd_link,
            work_item,
            environment,
            segment,
            k,
        )
        k += 1
    return value


@wp.func
def _state_jacobian_velocity(
    adjoint: wp.array3d(dtype=wp.float64),
    jacobian_states: wp.array4d(dtype=wp.float64),
    qd: wp.array2d(dtype=wp.float64),
    work_item: int,
    environment: int,
    state_index: int,
    row: int,
) -> wp.float64:
    value = wp.float64(0.0)
    k = int(0)
    while k < SPATIAL_DIM:
        inner = wp.float64(0.0)
        column = int(0)
        while column < NUM_DOFS:
            inner += (
                jacobian_states[environment, state_index, k, column]
                * qd[environment, column]
            )
            column += 1
        value += adjoint[work_item, row, k] * inner
        k += 1
    return value


@wp.kernel(enable_backward=False)
def persistent_raw_dynamics_kernel(
    joint_adjoint: wp.array3d(dtype=wp.float64),
    joint_adjoint_dot: wp.array3d(dtype=wp.float64),
    joint_tangent_active: wp.array3d(dtype=wp.float64),
    joint_tangent_dot_qd: wp.array2d(dtype=wp.float64),
    joint_velocity: wp.array2d(dtype=wp.float64),
    cell_adjoint: wp.array3d(dtype=wp.float64),
    cell_tangent: wp.array3d(dtype=wp.float64),
    cell_tangent_dot: wp.array3d(dtype=wp.float64),
    cell_magnus_basis: wp.array3d(dtype=wp.float64),
    cell_magnus_basis_dot: wp.array3d(dtype=wp.float64),
    qd: wp.array2d(dtype=wp.float64),
    qd_link: wp.array3d(dtype=wp.float64),
    gather_indices: wp.array2d(dtype=wp.int32),
    gather_mask: wp.array2d(dtype=wp.float64),
    weights: wp.array2d(dtype=wp.float64),
    masses: wp.array3d(dtype=wp.float64),
    gravity_base: wp.array(dtype=wp.float64),
    jacobian_states: wp.array4d(dtype=wp.float64),
    jacobian_dot_qd_states: wp.array3d(dtype=wp.float64),
    velocity_states: wp.array3d(dtype=wp.float64),
    gravity_states: wp.array3d(dtype=wp.float64),
    inertia: wp.array3d(dtype=wp.float64),
    coriolis_qd: wp.array2d(dtype=wp.float64),
    gravity_force: wp.array2d(dtype=wp.float64),
):
    """Persistent one-thread-per-environment recurrence and dynamics assembly."""

    environment = wp.tid()
    row = int(0)
    while row < NUM_DOFS:
        coriolis_qd[environment, row] = wp.float64(0.0)
        gravity_force[environment, row] = wp.float64(0.0)
        column = int(0)
        while column < NUM_DOFS:
            inertia[environment, row, column] = wp.float64(0.0)
            column += 1
        row += 1

    segment = int(0)
    while segment < NUM_SEGMENTS:
        joint_work_item = environment * NUM_SEGMENTS + segment
        joint_state = segment * (NUM_CELLS + 1)
        previous_tip_state = (segment - 1) * (NUM_CELLS + 1) + NUM_CELLS

        spatial_row = int(0)
        while spatial_row < SPATIAL_DIM:
            column = int(0)
            while column < NUM_DOFS:
                value = wp.float64(0.0)
                k = int(0)
                while k < SPATIAL_DIM:
                    previous_value = wp.float64(0.0)
                    if segment > 0:
                        previous_value = jacobian_states[
                            environment, previous_tip_state, k, column
                        ]
                    value += joint_adjoint[joint_work_item, spatial_row, k] * (
                        previous_value
                        + joint_tangent_active[joint_work_item, k, column]
                    )
                    k += 1
                jacobian_states[environment, joint_state, spatial_row, column] = value
                column += 1

            derivative_value = wp.float64(0.0)
            velocity_value = wp.float64(0.0)
            gravity_value = wp.float64(0.0)
            k = int(0)
            while k < SPATIAL_DIM:
                previous_derivative = wp.float64(0.0)
                previous_velocity = wp.float64(0.0)
                previous_gravity = gravity_base[k]
                previous_jacobian_qd = wp.float64(0.0)
                if segment > 0:
                    previous_derivative = jacobian_dot_qd_states[
                        environment, previous_tip_state, k
                    ]
                    previous_velocity = velocity_states[
                        environment, previous_tip_state, k
                    ]
                    previous_gravity = gravity_states[
                        environment, previous_tip_state, k
                    ]
                    source_column = int(0)
                    while source_column < NUM_DOFS:
                        previous_jacobian_qd += (
                            jacobian_states[
                                environment, previous_tip_state, k, source_column
                            ]
                            * qd[environment, source_column]
                        )
                        source_column += 1
                derivative_value += joint_adjoint[joint_work_item, spatial_row, k] * (
                    previous_derivative + joint_tangent_dot_qd[joint_work_item, k]
                )
                derivative_value += (
                    joint_adjoint_dot[joint_work_item, spatial_row, k]
                    * previous_jacobian_qd
                )
                velocity_value += joint_adjoint[joint_work_item, spatial_row, k] * (
                    previous_velocity + joint_velocity[joint_work_item, k]
                )
                gravity_value += (
                    joint_adjoint[joint_work_item, spatial_row, k] * previous_gravity
                )
                k += 1
            jacobian_dot_qd_states[environment, joint_state, spatial_row] = (
                derivative_value
            )
            velocity_states[environment, joint_state, spatial_row] = velocity_value
            gravity_states[environment, joint_state, spatial_row] = gravity_value
            spatial_row += 1

        cell = int(0)
        while cell < NUM_CELLS:
            cell_work_item = (
                environment * NUM_SEGMENTS * NUM_CELLS + segment * NUM_CELLS + cell
            )
            source_state = joint_state + cell
            destination_state = source_state + 1

            spatial_row = int(0)
            while spatial_row < SPATIAL_DIM:
                column = int(0)
                while column < NUM_DOFS:
                    value = wp.float64(0.0)
                    k = int(0)
                    while k < SPATIAL_DIM:
                        value += cell_adjoint[cell_work_item, spatial_row, k] * (
                            jacobian_states[environment, source_state, k, column]
                            + _raw_active_basis_value(
                                cell_tangent,
                                cell_magnus_basis,
                                gather_indices,
                                gather_mask,
                                cell_work_item,
                                segment,
                                k,
                                column,
                            )
                        )
                        k += 1
                    jacobian_states[
                        environment, destination_state, spatial_row, column
                    ] = value
                    column += 1

                derivative_value = wp.float64(0.0)
                velocity_value = wp.float64(0.0)
                gravity_value = wp.float64(0.0)
                k = int(0)
                while k < SPATIAL_DIM:
                    derivative_value += cell_adjoint[cell_work_item, spatial_row, k] * (
                        jacobian_dot_qd_states[environment, source_state, k]
                        + _raw_tangent_dot_velocity_value(
                            cell_tangent,
                            cell_tangent_dot,
                            cell_magnus_basis,
                            cell_magnus_basis_dot,
                            qd_link,
                            cell_work_item,
                            environment,
                            segment,
                            k,
                        )
                    )
                    velocity_value += cell_adjoint[cell_work_item, spatial_row, k] * (
                        velocity_states[environment, source_state, k]
                        + _raw_link_velocity_value(
                            cell_tangent,
                            cell_magnus_basis,
                            qd_link,
                            cell_work_item,
                            environment,
                            segment,
                            k,
                        )
                    )
                    gravity_value += (
                        cell_adjoint[cell_work_item, spatial_row, k]
                        * gravity_states[environment, source_state, k]
                    )
                    k += 1

                eta0 = _state_jacobian_velocity(
                    cell_adjoint,
                    jacobian_states,
                    qd,
                    cell_work_item,
                    environment,
                    source_state,
                    0,
                )
                eta1 = _state_jacobian_velocity(
                    cell_adjoint,
                    jacobian_states,
                    qd,
                    cell_work_item,
                    environment,
                    source_state,
                    1,
                )
                eta2 = _state_jacobian_velocity(
                    cell_adjoint,
                    jacobian_states,
                    qd,
                    cell_work_item,
                    environment,
                    source_state,
                    2,
                )
                eta3 = _state_jacobian_velocity(
                    cell_adjoint,
                    jacobian_states,
                    qd,
                    cell_work_item,
                    environment,
                    source_state,
                    3,
                )
                eta4 = _state_jacobian_velocity(
                    cell_adjoint,
                    jacobian_states,
                    qd,
                    cell_work_item,
                    environment,
                    source_state,
                    4,
                )
                eta5 = _state_jacobian_velocity(
                    cell_adjoint,
                    jacobian_states,
                    qd,
                    cell_work_item,
                    environment,
                    source_state,
                    5,
                )
                component = spatial_row % 3
                step0 = _raw_step_velocity_value(
                    cell_adjoint,
                    cell_tangent,
                    cell_magnus_basis,
                    qd_link,
                    cell_work_item,
                    environment,
                    segment,
                    0,
                )
                step1 = _raw_step_velocity_value(
                    cell_adjoint,
                    cell_tangent,
                    cell_magnus_basis,
                    qd_link,
                    cell_work_item,
                    environment,
                    segment,
                    1,
                )
                step2 = _raw_step_velocity_value(
                    cell_adjoint,
                    cell_tangent,
                    cell_magnus_basis,
                    qd_link,
                    cell_work_item,
                    environment,
                    segment,
                    2,
                )
                bracket = _cross_component(
                    step0, step1, step2, eta0, eta1, eta2, component
                )
                if spatial_row >= 3:
                    step3 = _raw_step_velocity_value(
                        cell_adjoint,
                        cell_tangent,
                        cell_magnus_basis,
                        qd_link,
                        cell_work_item,
                        environment,
                        segment,
                        3,
                    )
                    step4 = _raw_step_velocity_value(
                        cell_adjoint,
                        cell_tangent,
                        cell_magnus_basis,
                        qd_link,
                        cell_work_item,
                        environment,
                        segment,
                        4,
                    )
                    step5 = _raw_step_velocity_value(
                        cell_adjoint,
                        cell_tangent,
                        cell_magnus_basis,
                        qd_link,
                        cell_work_item,
                        environment,
                        segment,
                        5,
                    )
                    bracket = _cross_component(
                        step3, step4, step5, eta0, eta1, eta2, component
                    ) + _cross_component(
                        step0, step1, step2, eta3, eta4, eta5, component
                    )

                jacobian_dot_qd_states[environment, destination_state, spatial_row] = (
                    derivative_value - bracket
                )
                velocity_states[environment, destination_state, spatial_row] = (
                    velocity_value
                )
                gravity_states[environment, destination_state, spatial_row] = (
                    gravity_value
                )
                spatial_row += 1

            if cell < NUM_QUADRATURE_CELLS:
                coordinate = int(0)
                while coordinate < NUM_DOFS:
                    c_value = wp.float64(0.0)
                    g_value = wp.float64(0.0)
                    spatial_row = int(0)
                    while spatial_row < SPATIAL_DIM:
                        omega0 = velocity_states[environment, destination_state, 0]
                        omega1 = velocity_states[environment, destination_state, 1]
                        omega2 = velocity_states[environment, destination_state, 2]
                        linear0 = velocity_states[environment, destination_state, 3]
                        linear1 = velocity_states[environment, destination_state, 4]
                        linear2 = velocity_states[environment, destination_state, 5]
                        moment0 = masses[segment, cell, 0] * omega0
                        moment1 = masses[segment, cell, 1] * omega1
                        moment2 = masses[segment, cell, 2] * omega2
                        force0 = masses[segment, cell, 3] * linear0
                        force1 = masses[segment, cell, 4] * linear1
                        force2 = masses[segment, cell, 5] * linear2
                        coadjoint_value = _cross_component(
                            omega0,
                            omega1,
                            omega2,
                            moment0,
                            moment1,
                            moment2,
                            spatial_row % 3,
                        )
                        if spatial_row < 3:
                            coadjoint_value += _cross_component(
                                linear0,
                                linear1,
                                linear2,
                                force0,
                                force1,
                                force2,
                                spatial_row,
                            )
                        else:
                            coadjoint_value = _cross_component(
                                omega0,
                                omega1,
                                omega2,
                                force0,
                                force1,
                                force2,
                                spatial_row - 3,
                            )
                        wrench = (
                            masses[segment, cell, spatial_row]
                            * jacobian_dot_qd_states[
                                environment, destination_state, spatial_row
                            ]
                            + coadjoint_value
                        )
                        jacobian_value = jacobian_states[
                            environment,
                            destination_state,
                            spatial_row,
                            coordinate,
                        ]
                        c_value += jacobian_value * weights[segment, cell] * wrench
                        g_value -= (
                            jacobian_value
                            * weights[segment, cell]
                            * masses[segment, cell, spatial_row]
                            * gravity_states[
                                environment, destination_state, spatial_row
                            ]
                        )
                        spatial_row += 1
                    coriolis_qd[environment, coordinate] += c_value
                    gravity_force[environment, coordinate] += g_value

                    other_coordinate = int(0)
                    while other_coordinate < NUM_DOFS:
                        b_value = wp.float64(0.0)
                        spatial_row = int(0)
                        while spatial_row < SPATIAL_DIM:
                            b_value += (
                                jacobian_states[
                                    environment,
                                    destination_state,
                                    spatial_row,
                                    coordinate,
                                ]
                                * weights[segment, cell]
                                * masses[segment, cell, spatial_row]
                                * jacobian_states[
                                    environment,
                                    destination_state,
                                    spatial_row,
                                    other_coordinate,
                                ]
                            )
                            spatial_row += 1
                        inertia[environment, coordinate, other_coordinate] += b_value
                        other_coordinate += 1
                    coordinate += 1
            cell += 1
        segment += 1


def _persistent_raw_dynamics(
    joint_adjoint: wp.array3d(dtype=wp.float64),
    joint_adjoint_dot: wp.array3d(dtype=wp.float64),
    joint_tangent_active: wp.array3d(dtype=wp.float64),
    joint_tangent_dot_qd: wp.array2d(dtype=wp.float64),
    joint_velocity: wp.array2d(dtype=wp.float64),
    cell_adjoint: wp.array3d(dtype=wp.float64),
    cell_tangent: wp.array3d(dtype=wp.float64),
    cell_tangent_dot: wp.array3d(dtype=wp.float64),
    cell_magnus_basis: wp.array3d(dtype=wp.float64),
    cell_magnus_basis_dot: wp.array3d(dtype=wp.float64),
    qd: wp.array2d(dtype=wp.float64),
    qd_link: wp.array3d(dtype=wp.float64),
    gather_indices: wp.array2d(dtype=wp.int32),
    gather_mask: wp.array2d(dtype=wp.float64),
    weights: wp.array2d(dtype=wp.float64),
    masses: wp.array3d(dtype=wp.float64),
    gravity_base: wp.array(dtype=wp.float64),
    jacobian_states: wp.array4d(dtype=wp.float64),
    jacobian_dot_qd_states: wp.array3d(dtype=wp.float64),
    velocity_states: wp.array3d(dtype=wp.float64),
    gravity_states: wp.array3d(dtype=wp.float64),
    inertia: wp.array3d(dtype=wp.float64),
    coriolis_qd: wp.array2d(dtype=wp.float64),
    gravity_force: wp.array2d(dtype=wp.float64),
):
    wp.launch(
        persistent_raw_dynamics_kernel,
        dim=qd.shape[0],
        inputs=[
            joint_adjoint,
            joint_adjoint_dot,
            joint_tangent_active,
            joint_tangent_dot_qd,
            joint_velocity,
            cell_adjoint,
            cell_tangent,
            cell_tangent_dot,
            cell_magnus_basis,
            cell_magnus_basis_dot,
            qd,
            qd_link,
            gather_indices,
            gather_mask,
            weights,
            masses,
            gravity_base,
        ],
        outputs=[
            jacobian_states,
            jacobian_dot_qd_states,
            velocity_states,
            gravity_states,
            inertia,
            coriolis_qd,
            gravity_force,
        ],
        block_dim=128,
    )


persistent_raw_dynamics = wp.jax_callable(_persistent_raw_dynamics, num_outputs=7)
