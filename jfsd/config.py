from pathlib import Path
from typing import NamedTuple, Optional

import numpy as np

try:
    import tomllib
except ImportError:
    import tomli as tomllib  # type: ignore  # noqa

from jfsd.utils import create_hardsphere_configuration


class JfsdConfiguration:
    def __init__(self, general, initialization, physics, box, seeds, output, start_configuration):
        self.general = general
        self.physics = physics
        self.box = box
        self.initialization = initialization
        self.output = output
        self.seeds = seeds
        self.start_configuration = start_configuration

    @classmethod
    def from_toml(cls, config_fp, start_configuration: Optional[Path] = None):
        # def _parse_config(config):
        # final_dict = {}
        # for key, value in config.items():
        # if isinstance(value, Mapping):
        # final_dict.update(_parse_config(value))
        # else:
        # final_dict[key] = value
        with open(config_fp, "rb") as handle:
            config_data = tomllib.load(handle)
        print(config_data.get("initialization", {}))
        new_config = cls(
            General(**config_data.pop("general")),
            Initialization(**config_data.pop("initialization", {})),
            Physics(**config_data.pop("physics", {})),
            Box(**config_data.pop("box", {})),
            Seeds(**config_data.pop("seeds", {})),
            Output(**config_data.pop("output", {})),
            start_configuration,
        )
        if len(config_data) > 0:
            raise ValueError(f"Unknown configuration directive(s) detected: {list(config_data)}")
        return new_config

    @property
    def parameters(self):
        print(self.start_configuration)
        params = {}
        params.update(self.general.get_parameters())
        params.update(
            self.initialization.get_parameters(
                self.box.lx, self.general.n_particles, self.start_configuration
            )
        )
        if self.general.n_particles is None:
            n_particles = params["positions"].shape[0]
            params["num_particles"] = n_particles
        else:
            n_particles = self.general.n_particles
        params.update(self.physics.get_parameters(n_particles))
        params.update(self.box.get_parameters())
        params.update(self.output.get_parameters())
        params.update(self.seeds.get_parameters())
        return params


class General(NamedTuple):
    n_steps: int
    n_particles: Optional[int] = None
    dt: float = 0.01

    def get_parameters(self):
        return {
            "num_steps": self.n_steps,
            "num_particles": self.n_particles,
            "time_step": self.dt,
        }


class Initialization(NamedTuple):
    position_source_type: str = "random_hardsphere"
    init_seed: int = 210398423

    def get_parameters(self, box_x, n_particles, numpy_file=None):
        if self.position_source_type == "random_hardsphere":
            if n_particles is None:
                raise ValueError(
                    "Please supply a number of particles or initial positions file"
                    " and set source_type to 'file'."
                )
            if numpy_file is not None:
                raise ValueError(
                    "Starting configuration was supplied while 'random_hardsphere' "
                    "source_type was selected. Leave the starting configuration empty "
                    "or switch to 'file' source type."
                )
            positions = create_hardsphere_configuration(box_x, n_particles, self.init_seed, 0.001)
        elif self.position_source_type == "file":
            if n_particles is not None:
                raise ValueError(
                    "An initial position file has been supplied, remove the number of"
                    " particles from the configuration file."
                )
            if numpy_file is None:
                raise ValueError("Please supply the numpy file if using the source_type 'file'.")
            positions = np.load(numpy_file)
        else:
            raise ValueError(f"Unknown source_type {self.position_source_type}")
        return {
            "positions": positions,
            "particle_radius": 1,  # Colloid radius
        }


class Vector(NamedTuple):
    x: float
    y: float
    z: float


class Physics(NamedTuple):
    dynamics_type: str = "brownian"
    boundary_conditions: str = "periodic"
    kT: float = 0.005305165
    interaction_strength: float = 0.0
    interaction_cutoff: float = 0.0
    shear_rate: float = 0.0
    shear_frequency: float = 0.0
    friction_coefficient: float = 0.0
    friction_range: float = 0.0
    constant_force: Vector = (0.0, 0.0, 0.0)
    constant_torque: Vector = (0.0, 0.0, 0.0)
    buoyancy: bool = False

    def get_parameters(self, n_particles):
        if self.dynamics_type.lower() == "brownian":
            HIs_flag = 0
        elif self.dynamics_type.lower() == "rpy":
            HIs_flag = 1
        elif self.dynamics_type.lower() == "stokesian":
            HIs_flag = 2
        else:
            raise ValueError(
                f"Unknown dynamics type: {self.dynamics_type}, choose from:"
                " brownian, rpy or stokesian."
            )

        if self.boundary_conditions.lower() == "periodic":
            boundary_flag = 0
        elif self.boundary_conditions.lower() == "open":
            boundary_flag = 1
        else:
            raise ValueError(
                f"Unknown boundary conditions: {self.boundary_conditions}, choose from:"
                " periodic or open."
            )

        constant_forces = np.zeros((n_particles, 3))
        constant_forces[:, :] = self.constant_force
        constant_torques = np.zeros((n_particles, 3))
        constant_torques[:, :] = self.constant_torque
        return {
            "hydrodynamic_interaction_flag": HIs_flag,
            "boundary_flag": boundary_flag,
            "temperature": self.kT,
            "interaction_strength": self.interaction_strength,
            "interaction_cutoff": self.interaction_cutoff,
            "shear_rate_0": self.shear_rate,
            "shear_frequency": self.shear_frequency,
            "friction_coefficient": self.friction_coefficient,
            "friction_range": self.friction_range,
            "constant_applied_forces": constant_forces,
            "constant_applied_torques": constant_torques,
            "buoyancy_flag": int(self.buoyancy),
            "ewald_xi": 0.5,  # Ewald parameter
            "error_tolerance": 0.001,  # Error tolerance
        }


class Box(NamedTuple):
    lx: int = 20
    ly: int = 20
    lz: int = 20
    max_strain: float = 0.5

    def get_parameters(self):
        return dict(zip(self._fields, self))


class Seeds(NamedTuple):
    rfd: int = 9237412
    ffwave: int = 30498522
    ffreal: int = 57239485
    nf: int = 2343095

    def get_parameters(self):
        return {f"seed_{f}": getattr(self, f) for f in self._fields}


class Output(NamedTuple):
    store_stresslet: bool = False
    store_velocity: bool = False
    store_orientation: bool = False
    writing_period: int = 10
    thermal_fluctuation_test: str = "none"

    def get_parameters(self):
        _rf_convert = {
            "none": 0,
            "far-field": 1,
            "lubrication": 2,
        }
        try:
            thermal_fluct = _rf_convert[self.thermal_fluctuation_test]
        except KeyError:
            raise ValueError(
                "Wrong parameter for thermal fluctuation test, choose one"
                " of none, far-field or lubrication"
            )
        return {
            "store_stresslet": self.store_stresslet,
            "store_velocity": self.store_velocity,
            "store_orientation": self.store_orientation,
            "writing_period": self.writing_period,
            "thermal_test_flag": thermal_fluct,
        }

    # N = n_particles
    # Nsteps = n_steps
    # (dynamics_type, T, U, U_cutoff, shear_rate, shear_frequency, alpha_friction,
    #     h0_friction) = physics_options
    # (_, seed_RFD, seed_ffwave, seed_ffreal, seed_nf) = seeds
    # (stresslet_flag, velocity_flag, orient_flag, writing_period) = output_options

    # if dynamics_type == "brownian":
    #     HIs_flag = 0
    # elif dynamics_type == "rpy":
    #     HIs_flag = 1
    # elif dynamics_type == "stokesian":
    #     HIs_flag = 2
    # else:
    #     raise ValueError(f"Unknown dynamics type: {dynamics_type}")
