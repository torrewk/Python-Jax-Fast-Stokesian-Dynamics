from typing import NamedTuple

import numpy as np
import tomllib

from jfsd.utils import create_hardsphere_configuration


class JfsdConfiguration():
    def __init__(self, general, initialization, physics, box, seeds, output):
        self.general = general
        self.physics = physics
        self.box = box
        self.initialization = initialization
        self.output = output
        self.seeds = seeds

    @classmethod
    def from_toml(cls, config_fp):
        # def _parse_config(config):
        # final_dict = {}
        # for key, value in config.items():
            # if isinstance(value, Mapping):
                # final_dict.update(_parse_config(value))
            # else:
                # final_dict[key] = value
        with open(config_fp, "rb") as handle:
            config_data = tomllib.load(handle)
        return cls(
            General(**config_data["general"]),
            Initialization(**config_data["initialization"]),
            Physics(**config_data["physics"]),
            Box(**config_data["box"]),
            Seeds(**config_data["seeds"]),
            Output(**config_data["output"])
        )


    @property
    def parameters(self):
        params = {}
        params.update(self.general.get_parameters())
        params.update(self.physics.get_parameters())
        params.update(self.box.get_parameters())
        params.update(self.initialization.get_parameters(self.box.Lx,
                                                         self.general.n_particles))
        params.update(self.output.get_parameters())
        params.update(self.seeds.get_parameters())
        return params

class General(NamedTuple):
    n_steps: int
    n_particles: int
    dt: float

    def get_parameters(self):
        return {
            "NSteps": self.n_steps,
            "N": self.n_particles,
            "dt": self.dt
        }

class Initialization(NamedTuple):
    position_source_type: str
    init_seed: int

    def get_parameters(self, box_x, n_particles, numpy_file = None):
        if self.position_source_type == "random_hardsphere":
            positions = create_hardsphere_configuration(box_x, n_particles, self.init_seed, 0.001)
        elif self.position_source_type == "file":
            if numpy_file is None:
                raise ValueError("Please supply the numpy file if using the source_type 'file'.")
            positions = np.load(numpy_file)
        else:
            raise ValueError(f"Unknown source_type {self.position_source_type}")
        return {
            "positions": positions
        }

class Vector(NamedTuple):
    x: float
    y: float
    z: float

class Physics(NamedTuple):
    dynamics_type: str
    kT: float
    interaction_strength: float
    interaction_cutoff: float
    shear_rate: float
    shear_frequency: float
    friction_coefficient: float
    friction_range: float
    constant_force: Vector
    constant_torque: Vector
    buoyancy: bool

    def get_parameters(self):
        if self.dynamics_type.lower() == "brownian":
            HIs_flag = 0
        elif self.dynamics_type.lower() == "rpy":
            HIs_flag = 1
        elif self.dynamics_type.lower() == "stokesian":
            HIs_flag = 2
        else:
            raise ValueError(f"Unknown dynamics type: {self.dynamics_type}, choose from:"
                             " brownian, rpy or stokesian.")
        return {
            "HIs_flag": HIs_flag,
            "T": self.kT,
            "U": self.interaction_strength,
            "U_cutoff": self.interaction_cutoff,
            "shear_rate_0": self.shear_rate,
            "shear_freq": self.shear_frequency,
            "alpha_friction": self.friction_coefficient,
            "h0_friction": self.friction_range,
            "constant_applied_forces": np.array(self.constant_force),
            "constant_applied_torques": np.array(self.constant_torque),
            "buoyancy_flag": int(self.buoyancy),
            "xi": 0.5,  # Ewald parameter
            "error": 0.001  # Error tolerance
        }


class Box(NamedTuple):
    Lx: int
    Ly: int
    Lz: int

    def get_parameters(self):
        return dict(zip(self._fields, self))

class Seeds(NamedTuple):
    RFD: int
    ffwave: int
    ffreal: int
    nf: int

    def get_parameters(self):
        return {"seed_{f}": getattr(self, f) for f in self._fields}

class Output(NamedTuple):
    store_stresslet: bool
    store_velocity: bool
    store_orientation: bool
    writing_period: int
    thermal_fluctuation_test: str

    def get_parameters(self):
        _rf_convert = {
            "none": 0,
            "far-field": 1,
            "lubrication": 2,
        }
        try:
            thermal_fluct = _rf_convert[self.thermal_fluctuation_test]
        except KeyError as exc:
            raise ValueError("Wrong parameter for thermal fluctuation test, choose one"
                             " of none, far-field or lubrication")
        return {
            "stresslet_flag": self.store_stresslet,
            "velocity_flag": self.store_velocity,
            "orientation_flag": self.store_orientation,
            "writing_period": self.writing_period,
            "thermal_test_flag": self.thermal_fluctuation_test,
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