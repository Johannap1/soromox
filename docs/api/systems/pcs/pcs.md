# Spatial PCS and PCS-based systems

The general Piecewise Constant Strain (PCS) implementation provides the core modeling framework for continuum soft robots, based on the discrete Cosserat approach proposed by Renda et al. (2018).

## Overview

This module contains the fundamental spatial PCS implementation and the
PCS-based `PlanarHSA` specialization. It provides the core mathematical
framework for modeling continuum robots using piecewise constant strain
assumptions, following the discrete Cosserat approach for multisection soft
manipulator dynamics.

## API Reference

::: soromox.systems.pcs.pcs
    options:
      show_root_heading: true
      show_source: false
      heading_level: 3
      group_by_category: true
      docstring_section_style: table
      members_order: source
      members: [PCS]

## References

The PCS (Piecewise Constant Strain) model was originally proposed in:

Renda, F., Boyer, F., Dias, J., & Seneviratne, L. (2018). Discrete cosserat approach for multisection soft manipulator dynamics. *IEEE Transactions on Robotics*, 34(6), 1518-1533.
