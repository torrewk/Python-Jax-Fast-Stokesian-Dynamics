"""Enums for different simulation cases"""

from enum import IntEnum


class HydrodynamicInteraction(IntEnum):
    """Hydrodynamic interaction levels.
    
    Determines the level of hydrodynamic interactions between particles.
    """
    BROWNIAN = 0          # Brownian Dynamics (no hydrodynamic interactions)
    RPY = 1              # Rotne-Prager-Yamakawa approximation
    STOKESIAN = 2        # Full Stokesian Dynamics


class BoundaryConditions(IntEnum):
    """Boundary condition types.
    
    Determines the type of boundary conditions for hydrodynamic interactions.
    """
    PERIODIC = 0         # Periodic boundary conditions
    OPEN = 1            # Open boundary conditions (infinite domain)


class ThermalTestType(IntEnum):
    """Thermal fluctuation test types.
    
    Flag used to test thermal fluctuation calculations.
    """
    NONE = 0            # No thermal test
    FARFIELD_REAL = 1   # Test far-field real-space thermal fluctuations
    LUBRICATION = 2     # Test lubrication thermal fluctuations


class BuoyancyFlag(IntEnum):
    """Buoyancy settings.
    
    Controls whether gravitational/buoyancy forces are applied to particles.
    """
    OFF = 0             # No buoyancy forces
    ON = 1              # Apply buoyancy forces (gravity in z-direction)


class StorageFlags(IntEnum):
    """Storage flags for output data.
    
    Controls what data to store during simulation.
    """
    DISABLED = 0        # Don't store this data type
    ENABLED = 1         # Store this data type