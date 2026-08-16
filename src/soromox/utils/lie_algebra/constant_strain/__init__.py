r"""Closed-form Lie operators for constant-strain rod segments.

The algebra-specific :mod:`.se2` and :mod:`.se3` namespaces provide concise
public functions for planar and spatial rod kinematics. Both modules return
:class:`ConstantStrainOperators` from their bundled ``operators`` evaluator.
"""

from . import se2, se3
from ._shared import ConstantStrainOperators

# Compatibility aliases for the API exposed by the former constant_strain.py
# module. Prefer constant_strain.se2.* and constant_strain.se3.* in new code.
adjoint_se2 = se2.adjoint
adjoint_inverse_se2 = se2.adjoint_inverse
operators_se2 = se2.operators
tangent_se2 = se2.tangent
tangent_derivative_se2 = se2.tangent_derivative

adjoint_se3 = se3.adjoint
adjoint_inverse_se3 = se3.adjoint_inverse
operators_se3 = se3.operators
tangent_se3 = se3.tangent
tangent_derivative_se3 = se3.tangent_derivative

__all__ = [
    "ConstantStrainOperators",
    "se2",
    "se3",
    "adjoint_se2",
    "adjoint_inverse_se2",
    "operators_se2",
    "tangent_se2",
    "tangent_derivative_se2",
    "adjoint_se3",
    "adjoint_inverse_se3",
    "operators_se3",
    "tangent_se3",
    "tangent_derivative_se3",
]
