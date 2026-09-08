__all__ = [
    "GeneralizedForceTerms",
    "MechanicalDynamicsTerms",
    "MechanicalState",
    "MechanicalStateRate",
]
from typing import Generic, TypeVar

import equinox as eqx
from jax import Array

AuxT = TypeVar("AuxT")


class MechanicalState(eqx.Module, Generic[AuxT]):
    configuration: Array  # (..., nq)
    velocity: Array  # (..., nv)
    auxiliary: AuxT


class MechanicalStateRate(eqx.Module, Generic[AuxT]):
    configuration_rate: Array  # (..., nq)
    acceleration: Array  # (..., nv)
    auxiliary_rate: AuxT


class GeneralizedForceTerms(eqx.Module):
    """Terms use the convention M @ acceleration = net_force."""

    actuation: Array  # +tau_u
    external: Array  # +tau_ext, including composed environment loads
    convective: Array  # +C(q, v) @ v on the left-hand side
    gravity: Array  # +G(q) on the left-hand side
    elastic: Array  # +K(q, z) on the left-hand side
    dissipative: Array  # +D(q, v, z) on the left-hand side

    @property
    def net_force(self) -> Array:
        return (
            self.actuation
            + self.external
            - self.convective
            - self.gravity
            - self.elastic
            - self.dissipative
        )


class MechanicalDynamicsTerms(eqx.Module, Generic[AuxT]):
    """Mechanical equation evaluated at one state."""

    mass_matrix: Array  # (..., nv, nv)
    configuration_rate: Array  # N(q) @ v, shape (..., nq)
    forces: GeneralizedForceTerms
    auxiliary_rate: AuxT
