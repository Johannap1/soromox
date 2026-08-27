# Backend Configuration

!!! note "Usually no direct interaction is required"
    Keep each system's defaults unless application-level profiling demonstrates
    that a different PCS Warp block size is beneficial on the target hardware.

::: soromox.systems.execution.config
    options:
      show_root_heading: true
      show_source: false
      heading_level: 2
      members:
        - PCSBackendParams
        - DEFAULT_PLANAR_PCS_BLOCK_DIM
        - DEFAULT_PCS_BLOCK_DIM
