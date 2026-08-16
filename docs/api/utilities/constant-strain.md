# Constant Strain

The constant-strain package evaluates the Lie operators used to propagate
kinematics along rod segments with constant body strain. Planar and spatial
operators have separate namespaces, so their function names remain concise:

```python
from soromox.utils.lie_algebra.constant_strain import se2, se3

planar_tangent = se2.tangent(xi_planar, arc_length, eps)
spatial_operators = se3.operators(xi_spatial, arc_length, eps, xid_spatial)
```

Use a single-operator function when only one result is required. Use
`operators` when several related expressions are needed at the same strain and
arclength; it shares intermediate work and returns a fixed named bundle.

## Compatibility API

Before constant strain was split into algebra-specific modules, its public
functions carried suffixes such as `tangent_se2` and `adjoint_se3`. Those names
remain available from `soromox.utils.lie_algebra.constant_strain` for source
compatibility and refer to the same function objects as the module-scoped API.
New code should prefer `constant_strain.se2.*` or `constant_strain.se3.*`.

| Compatibility name | Preferred name |
| --- | --- |
| `adjoint_se2` | `se2.adjoint` |
| `adjoint_inverse_se2` | `se2.adjoint_inverse` |
| `operators_se2` | `se2.operators` |
| `tangent_se2` | `se2.tangent` |
| `tangent_derivative_se2` | `se2.tangent_derivative` |
| `adjoint_se3` | `se3.adjoint` |
| `adjoint_inverse_se3` | `se3.adjoint_inverse` |
| `operators_se3` | `se3.operators` |
| `tangent_se3` | `se3.tangent` |
| `tangent_derivative_se3` | `se3.tangent_derivative` |

## Operator Bundle

::: soromox.utils.lie_algebra.constant_strain.ConstantStrainOperators

## SE(2)

::: soromox.utils.lie_algebra.constant_strain.se2

## SE(3)

::: soromox.utils.lie_algebra.constant_strain.se3
