"""
I/O utilities for simulation input / output.
"""

# Standard library imports
import glob
import os
import re
import sys
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

# Third-party imports
import numpy as np
from loguru import logger

# JAX imports with fallback
try:
    import jax.numpy as jnp
    from jax.typing import ArrayLike
except ImportError:
    jnp = np
    ArrayLike = Any


# =============================================================================
# Logging Setup and Utilities
# =============================================================================

def get_next_log_file(base_path: str, max_backups: int = 10) -> str:
    """Generate the next available log file name by appending a number."""
    dir_path = os.path.dirname(base_path)
    if dir_path:
        os.makedirs(dir_path, exist_ok=True)
    
    base_name = os.path.basename(base_path)
    name, ext = os.path.splitext(base_name)
    
    # Get existing log files and rotate if original exists
    pattern = os.path.join(dir_path, f"{name}_*{ext}")
    existing_files = glob.glob(pattern)
    original_path = os.path.join(dir_path, base_name)
    
    if os.path.exists(original_path):
        # Find the next available number
        nums = [0]
        for f in existing_files:
            match = re.search(f"{name}_([0-9]+){ext}$", f)
            if match:
                nums.append(int(match.group(1)))
        
        next_num = max(nums) + 1
        new_path = os.path.join(dir_path, f"{name}_{next_num}{ext}")
        
        try:
            os.rename(original_path, new_path)
        except OSError as e:
            logger.error(f"Failed to rotate log file: {e}")
            
        # Prune old backups
        if max_backups is not None:
            _prune_old_backups(dir_path, name, ext, max_backups)
    
    return original_path


def _prune_old_backups(dir_path: str, name: str, ext: str, max_backups: int):
    """Remove old backup files beyond max_backups limit."""
    backup_pattern = os.path.join(dir_path, f"{name}_*{ext}")
    backup_files = glob.glob(backup_pattern)
    backups = []
    
    for bf in backup_files:
        match = re.search(f"{name}_([0-9]+){ext}$", bf)
        if match:
            backups.append((int(match.group(1)), bf))
    
    backups.sort(key=lambda x: x[0])
    if len(backups) > max_backups:
        for _, old_file in backups[:-max_backups]:
            try:
                os.remove(old_file)
            except OSError as e:
                logger.error(f"Failed to remove old log backup {old_file}: {e}")


def setup_logging(output_dir: Optional[str] = None, 
                 debug_enabled: bool = False, 
                 max_backups: int = 10):
    """Set up logging configuration with console and file handlers."""
    # Determine log directory
    log_dir = str(Path(output_dir)) if output_dir else "logs"
    log_file = os.path.join(log_dir, "simulation.log")
    debug_file = os.path.join(log_dir, "debug.log")
    
    # Create logs directory
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
    
    # Remove default logger and add basic console handler
    logger.remove()
    logger.add(sys.stderr, level="WARNING")
    
    # Handle log file rotation
    try:
        log_file = get_next_log_file(log_file, max_backups)
        if debug_enabled:
            debug_file = get_next_log_file(debug_file, max_backups)
    except Exception as e:
        logger.error(f"Failed to set up log rotation: {e}")
        return

    logger.info("Logging system initialized")
    
    # Setup console handlers
    _setup_console_handlers()
    
    # Setup file handlers
    _setup_file_handlers(log_file, debug_file, debug_enabled)


def _setup_console_handlers():
    """Setup console logging handlers."""
    # Clean format for info messages
    logger.add(
        sys.stdout,
        format="{message}",
        level="INFO",
        filter=lambda record: record["level"].name in ["INFO", "SUCCESS"]
    )
    
    # Detailed format for warnings and errors
    logger.add(
        sys.stderr,
        format="<level>{level:<8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:{line} | {message}",
        level="WARNING",
        filter=lambda record: record["level"].name in ["WARNING", "ERROR", "CRITICAL"]
    )


def _setup_file_handlers(log_file: str, debug_file: str, debug_enabled: bool):
    """Setup file logging handlers."""
    file_format = "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level:<8} | {name}:{function}:{line} | {message}"
    
    # Main log file
    logger.add(log_file, format=file_format, level="INFO", enqueue=True, catch=True)
    
    # Debug log file (optional)
    if debug_enabled:
        logger.add(debug_file, format=file_format, level="DEBUG", enqueue=True, catch=True)
        logger.debug("Debug logging enabled")


def setup_simulation_logging(output_dir: Optional[str] = None, debug_enabled: bool = False):
    """Setup logging for simulation with the specified output directory."""
    setup_logging(output_dir=output_dir, debug_enabled=debug_enabled)


def log_and_exit(message: str, code: int = 1):
    """Log an error message and exit the script with a status code."""
    logger.error(message)
    sys.exit(code)


# =============================================================================
# Thermodynamic Output Class
# =============================================================================

class ThermoOutput:
    """LAMMPS-style thermodynamic output for simulation monitoring."""
    
    def __init__(self, 
                 thermo_period: int = 100,
                 columns: Optional[List[str]] = None,
                 width: int = 12):
        """Initialize thermodynamic output."""
        self.thermo_period = thermo_period
        self.width = width
        self.start_time = None
        self.step_times = []
        
        # Default columns similar to LAMMPS thermo output
        self.columns = columns or ['Step', 'Time', 'Temp', 'CPU', 'TotTime']
        self.data_history = {col: [] for col in self.columns}
        self.header_printed = False
        
    def print_header(self):
        """Print the header for thermodynamic output."""
        total_width = len(self.columns) * (self.width + 1) - 1
        print("-" * total_width)
        
        header_line = ""
        for i, col in enumerate(self.columns):
            if i > 0:
                header_line += " "
            header_line += f"{col:>{self.width}}"
        print(header_line)
        print("-" * total_width)
        self.header_printed = True
        
    def update(self, 
               step: int, 
               time_step: float,
               temperature: float = 0.0,
               positions: Optional[ArrayLike] = None,
               velocities: Optional[ArrayLike] = None,
               forces: Optional[ArrayLike] = None,
               **kwargs):
        """
        Update and potentially print thermodynamic data.
        
        Parameters
        ----------
        step : int
            Current simulation step
        time_step : float
            Simulation time step
        temperature : float
            System temperature
        positions : ArrayLike, optional
            Particle positions
        velocities : ArrayLike, optional
            Particle velocities
        forces : ArrayLike, optional
            Forces on particles
        **kwargs
            Additional thermodynamic quantities
        """
        current_time = time.time()
        
        if self.start_time is None:
            self.start_time = current_time
            
        # Calculate timing statistics
        step_duration = current_time - self.step_times[-1] if self.step_times else 0.0
        self.step_times.append(current_time)
        
        # Keep only recent step times for performance calculation
        if len(self.step_times) > 1000:
            self.step_times = self.step_times[-1000:]
        
        # Calculate thermodynamic quantities
        thermo_data = self._calculate_thermo_data(
            step, time_step, temperature, positions, velocities, forces, 
            current_time, step_duration, **kwargs
        )
        
        # Store data
        for col in self.columns:
            if col in thermo_data:
                self.data_history[col].append(thermo_data[col])
        
        # Print if needed
        if step % self.thermo_period == 0:
            if not self.header_printed:
                self.print_header()
            self._print_thermo_line(thermo_data)
            
    def _calculate_thermo_data(self, 
                              step: int,
                              time_step: float, 
                              temperature: float,
                              positions: Optional[ArrayLike],
                              velocities: Optional[ArrayLike],
                              forces: Optional[ArrayLike],
                              current_time: float,
                              step_duration: float,
                              **kwargs) -> Dict[str, Any]:
        """Calculate thermodynamic quantities."""
        data = {
            'Step': step,
            'Time': step * time_step,
            'Temp': temperature,
            'TotTime': current_time - self.start_time if self.start_time else 0.0
        }
        
        # Add performance metrics
        self._add_performance_data(data, step, step_duration)
        
        # Add physical quantities if available
        if positions is not None:
            self._add_position_data(data, positions)
        if velocities is not None:
            self._add_velocity_data(data, velocities)
        if forces is not None:
            self._add_force_data(data, forces)
            
        # Add any additional quantities from kwargs
        for key, value in kwargs.items():
            if key in self.columns:
                data[key] = value
                
        return data
    
    def _add_performance_data(self, data: Dict[str, Any], step: int, step_duration: float):
        """Add performance metrics to thermodynamic data."""
        if step > 0 and len(self.step_times) > 1:
            recent_times = self.step_times[-min(100, len(self.step_times)):]
            if len(recent_times) > 1:
                avg_step_time = (recent_times[-1] - recent_times[0]) / (len(recent_times) - 1)
                data['CPU'] = avg_step_time
                data['StepsSec'] = 1.0 / avg_step_time if avg_step_time > 0 else 0.0
            else:
                data['CPU'] = step_duration
                data['StepsSec'] = 1.0 / step_duration if step_duration > 0 else 0.0
        else:
            data['CPU'] = 0.0
            data['StepsSec'] = 0.0
    
    def _add_position_data(self, data: Dict[str, Any], positions: ArrayLike):
        """Add position-based quantities to thermodynamic data."""
        if hasattr(positions, 'shape'):
            pos_array = np.array(positions) if not isinstance(positions, np.ndarray) else positions
            data['Natoms'] = pos_array.shape[0]
            
            # Center of mass
            if pos_array.ndim == 2 and pos_array.shape[1] >= 3:
                com = np.mean(pos_array[:, :3], axis=0)
                data['Xcm'] = com[0]
                data['Ycm'] = com[1] 
                data['Zcm'] = com[2]
    
    def _add_velocity_data(self, data: Dict[str, Any], velocities: ArrayLike):
        """Add velocity-based quantities to thermodynamic data."""
        if hasattr(velocities, 'shape'):
            vel_array = np.array(velocities) if not isinstance(velocities, np.ndarray) else velocities
            
            if vel_array.ndim == 2 and vel_array.shape[1] >= 3:
                # Kinetic energy (assuming unit mass)
                ke_linear = 0.5 * np.sum(vel_array[:, :3]**2)
                data['KE'] = ke_linear
                
                # Angular kinetic energy if available
                if vel_array.shape[1] >= 6:
                    ke_angular = 0.5 * np.sum(vel_array[:, 3:6]**2)
                    data['KErot'] = ke_angular
                    data['KEtot'] = ke_linear + ke_angular
                else:
                    data['KEtot'] = ke_linear
    
    def _add_force_data(self, data: Dict[str, Any], forces: ArrayLike):
        """Add force-based quantities to thermodynamic data."""
        if hasattr(forces, 'shape'):
            force_array = np.array(forces) if not isinstance(forces, np.ndarray) else forces
            
            if force_array.ndim == 2 and force_array.shape[1] >= 3:
                force_mags = np.sqrt(np.sum(force_array[:, :3]**2, axis=1))
                data['Fmax'] = np.max(force_mags)
                data['Fave'] = np.mean(force_mags)
        
    def _print_thermo_line(self, data: Dict[str, Any]):
        """Print a line of thermodynamic data."""
        line = ""
        for i, col in enumerate(self.columns):
            if i > 0:
                line += " "
                
            if col in data:
                value = data[col]
                line += self._format_value(col, value)
            else:
                line += f"{'N/A':>{self.width}}"
                
        print(line)
        sys.stdout.flush()
    
    def _format_value(self, column: str, value: Any) -> str:
        """Format a value based on its column type."""
        if column == 'Step':
            return f"{int(value):>{self.width}}"
        elif column in ['Time', 'TotTime', 'CPU']:
            return f"{value:>{self.width}.3f}"
        elif column in ['Temp', 'KE', 'KErot', 'KEtot']:
            return f"{value:>{self.width}.6f}"
        elif column in ['StepsSec']:
            return f"{value:>{self.width}.2f}"
        elif column in ['Fmax', 'Fave']:
            return f"{value:>{self.width}.3e}"
        elif column in ['Xcm', 'Ycm', 'Zcm']:
            return f"{value:>{self.width}.4f}"
        elif column == 'Natoms':
            return f"{int(value):>{self.width}}"
        else:
            # Generic formatting
            if isinstance(value, int):
                return f"{value:>{self.width}}"
            elif isinstance(value, float):
                return f"{value:>{self.width}.4f}"
            else:
                return f"{str(value):>{self.width}}"
        
    def finalize(self):
        """Print final statistics and clean up."""
        if self.start_time and len(self.step_times) > 1:
            total_time = self.step_times[-1] - self.start_time
            total_steps = len(self.step_times) - 1
            avg_performance = total_steps / total_time if total_time > 0 else 0.0
            
            print(f"\nPerformance: {avg_performance:.2f} steps/second")
            print(f"Total time: {total_time:.3f} seconds")
            print(f"Total steps: {total_steps}")
            
    def set_columns(self, columns: List[str]):
        """Set new column layout."""
        self.columns = columns
        self.data_history = {col: [] for col in self.columns}
        self.header_printed = False
        
    def get_data_history(self) -> Dict[str, List]:
        """Get the history of all thermodynamic data."""
        return self.data_history.copy()


# =============================================================================
# LAMMPS Data File I/O Classes
# =============================================================================

class LAMMPSDataReader:
    """Class to read and parse LAMMPS data files."""
    
    def __init__(self, filename: str):
        """Initialize a LAMMPS data file reader."""
        self.filename = filename
        self.data = {}
        self.sections = ["Masses", "Atoms", "Velocities"]
        self.header_params = {}
        
        logger.debug(f"Initializing LAMMPSDataReader with file: {filename}")
        self.read_file()
        logger.info(f"Reading LAMMPS data file: {filename}")
        logger.debug(f"Found sections: {list(self.data.keys())}")

    def read_file(self):
        """Read and parse a LAMMPS data file."""
        logger.debug(f"Starting to read file: {self.filename}")
        
        if not os.path.exists(self.filename):
            logger.error(f"Data file not found: {self.filename}")
            raise FileNotFoundError(f"Data file not found: {self.filename}")
            
        try:
            with open(self.filename, 'r') as file:
                section = None
                for line in file:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue

                    if any(line.startswith(sec) for sec in self.sections):
                        section = line.split()[0]
                        self.data[section] = []
                        continue

                    if section:
                        self.data[section].append(line)
                    else:
                        self.parse_header(line)
                        
        except PermissionError:
            logger.error(f"No permission to read file: {self.filename}")
            raise PermissionError(f"No permission to read file: {self.filename}")
        except UnicodeDecodeError:
            logger.error(f"File {self.filename} is not a valid text file")
            raise ValueError(f"File {self.filename} is not a valid text file")
            
        self._validate_data()
        logger.debug(f"Finished reading file. Found {len(self.data)} sections")

    def _validate_data(self):
        """Validate the consistency of the read data."""
        # Check atom count consistency
        if "atoms" in self.header_params:
            expected_atoms = self.header_params["atoms"]
            self._validate_atom_count(expected_atoms)
            
        # Validate box dimensions
        self._validate_box_dimensions()

    def _validate_atom_count(self, expected_atoms: int):
        """Validate atom count across sections."""
        if "Atoms" in self.data:
            actual_atoms = len(self.data["Atoms"])
            if actual_atoms != expected_atoms:
                raise ValueError(f"Mismatch in atom count. Header: {expected_atoms}, Found: {actual_atoms}")
                
        if "Velocities" in self.data:
            actual_velocities = len(self.data["Velocities"])
            if actual_velocities != expected_atoms:
                raise ValueError(f"Mismatch in velocities count. Expected: {expected_atoms}, Found: {actual_velocities}")

    def _validate_box_dimensions(self):
        """Validate box dimension parameters."""
        box_params = ['xlo', 'xhi', 'ylo', 'yhi', 'zlo', 'zhi']
        if not all(param in self.header_params for param in box_params):
            raise ValueError("Missing box dimension parameters in header")
            
        for dim in [('xlo', 'xhi'), ('ylo', 'yhi'), ('zlo', 'zhi')]:
            if self.header_params[dim[0]] >= self.header_params[dim[1]]:
                raise ValueError(f"Invalid box dimensions: {dim[0]} >= {dim[1]}")

    def parse_header(self, line: str):
        """Parse header lines and store parameters."""
        logger.debug(f"Parsing header line: {line}")
        try:
            if "xlo xhi" in line or "ylo yhi" in line or "zlo zhi" in line:
                self._parse_box_bounds(line)
            else:
                self._parse_count_line(line)
        except Exception as e:
            logger.error(f"Error parsing header line: {line}, Error: {e}")

    def _parse_box_bounds(self, line: str):
        """Parse box boundary lines."""
        try:
            bounds = list(map(float, line.split()[:2]))
            self.header_params[line.split()[2]] = bounds[0]
            self.header_params[line.split()[3]] = bounds[1]
        except (ValueError, IndexError) as e:
            logger.error(f"Invalid boundary values in line: {line}, Error: {e}")

    def _parse_count_line(self, line: str):
        """Parse count lines (e.g., '4000 atoms')."""
        match = re.match(r"(\d+)\s+(.+)", line)
        if match:
            value, key = match.groups()
            try:
                self.header_params[key] = int(value)
            except ValueError as e:
                logger.error(f"Failed to parse integer value in line: {line}, Error: {e}")
    
    def extract_atoms(self) -> Optional[List[Dict[str, Any]]]:
        """Extract atom information from LAMMPS data file."""
        logger.debug("Starting atom extraction")
        if "Atoms" not in self.data:
            logger.warning("No 'Atoms' section found in the data file. Returning None.")
            return None

        atoms = []
        for line_num, line in enumerate(self.data["Atoms"], 1):
            try:
                parts = line.strip().split()
                if len(parts) != 7:
                    logger.error(f"Invalid atom data at line {line_num}: {line}")
                    continue
                    
                atom_id, atom_type = map(int, parts[:2])
                diameter, density = map(float, parts[2:4])
                x, y, z = map(float, parts[4:7])

                atom_info = {
                    "id": atom_id, "type": atom_type,
                    "diameter": diameter, "density": density,
                    "x": x, "y": y, "z": z,
                }
                atoms.append(atom_info)
            except (ValueError, IndexError) as e:
                logger.error(f"Error parsing atom at line {line_num}: {e}")
                continue

        self._check_atom_count(atoms)
        logger.debug(f"Extracted {len(atoms) if atoms else 0} atoms")                
        return atoms

    def _check_atom_count(self, atoms: List[Dict[str, Any]]):
        """Check if extracted atom count matches header."""
        if "atoms" in self.header_params:
            if len(atoms) != self.header_params["atoms"]:
                logger.warning(f"Expected {self.header_params['atoms']} atoms, found {len(atoms)}.")
        else:
            logger.warning("No 'atoms' key found in header parameters, but extracted atom data.")

    def extract_velocities(self) -> Optional[List[Dict[str, Any]]]:
        """Extract velocity information from LAMMPS data file."""
        logger.debug("Starting velocity extraction")
        if "Velocities" not in self.data:
            logger.warning("No 'Velocities' section found in the data file. Returning None.")
            return None

        velocities = []
        for line_num, line in enumerate(self.data["Velocities"], 1):
            parts = line.strip().split()
            if len(parts) != 4:
                logger.error(f"Invalid velocity data at line {line_num}: {line}")
                continue
            try:
                atom_id = int(parts[0])
                vx, vy, vz = map(float, parts[1:4])
            except ValueError as e:
                logger.error(f"Error parsing velocity at line {line_num}: {e}")
                continue

            velocity_info = {
                "id": atom_id,
                "vx": vx, "vy": vy, "vz": vz,
            }
            velocities.append(velocity_info)

        if "atoms" in self.header_params:
            if len(velocities) != self.header_params["atoms"]:
                logger.warning(f"Expected {self.header_params['atoms']} velocities, found {len(velocities)}.")
        
        logger.debug(f"Extracted {len(velocities)} velocities")
        return velocities

    def extract_masses(self) -> Optional[List[Dict[str, Any]]]:
        """Extract mass information from LAMMPS data file."""
        logger.debug("Starting mass extraction")
        if "Masses" not in self.data:
            logger.warning("No 'Masses' section found in the data file. Returning None.")
            return None

        masses = []
        for line in self.data["Masses"]:
            parts = line.strip().split()
            if len(parts) != 2:
                raise ValueError(f"Invalid mass data: {line}")
                
            atom_type = int(parts[0])
            mass = float(parts[1])

            mass_info = {
                "type": atom_type,
                "mass": mass,
            }
            masses.append(mass_info)
        
        logger.debug(f"Extracted {len(masses) if masses else 0} masses")                
        return masses

    def summarize(self):
        """Print a summary of extracted data."""
        logger.info("Summarizing extracted data")
        if "Atoms" in self.data:
            logger.info(f"Extracted {len(self.data['Atoms'])} atoms.")
        if "Velocities" in self.data:
            logger.info(f"Extracted {len(self.data['Velocities'])} velocities.")


class LAMMPSDataWriter:
    """Class to write LAMMPS data files."""
    
    def __init__(self, filename: str):
        """Initialize a LAMMPS data file writer."""
        self.filename = filename
        logger.info(f"Initialized LAMMPSDataWriter for file: {filename}")
        logger.debug(f"Initialized LAMMPSDataWriter for file: {filename}")

    def write_header(self, header_params: Dict[str, Any]):
        """Write header section of LAMMPS data file."""
        logger.debug(f"Writing header with parameters: {header_params}")
        with open(self.filename, 'w') as f:
            f.write("# LAMMPS data file generated by LAMMPSDataWriter\n\n")
            
            # Write counts
            self._write_counts(f, header_params)
            f.write("\n")
            
            # Write types
            self._write_types(f, header_params)
            f.write("\n")
            
            # Write box dimensions
            self._write_box_dimensions(f, header_params)

    def _write_counts(self, f, header_params: Dict[str, Any]):
        """Write count parameters to file."""
        for key in ['atoms', 'bonds', 'angles', 'dihedrals', 'impropers']:
            if key in header_params:
                f.write(f"{header_params[key]} {key}\n")

    def _write_types(self, f, header_params: Dict[str, Any]):
        """Write type parameters to file."""
        for key in ['atom types', 'bond types', 'angle types', 'dihedral types', 'improper types']:
            if key in header_params:
                f.write(f"{header_params[key]} {key}\n")

    def _write_box_dimensions(self, f, header_params: Dict[str, Any]):
        """Write box dimension parameters to file."""
        for dim in [('xlo', 'xhi'), ('ylo', 'yhi'), ('zlo', 'zhi')]:
            if all(d in header_params for d in dim):
                f.write(f"{header_params[dim[0]]} {header_params[dim[1]]} {dim[0]} {dim[1]}\n")

    def write_atoms(self, atoms: List[Dict[str, Any]]):
        """Write Atoms section to LAMMPS data file."""
        logger.debug(f"Writing {len(atoms) if atoms else 0} atoms")
        if not atoms:
            logger.warning("No atom data to write.")
            return

        with open(self.filename, 'a') as f:
            f.write("\nAtoms # atom-ID atom-type diameter density x y z\n\n")
            for atom in atoms:
                self._validate_atom_data(atom)
                f.write(f"{atom['id']} {atom['type']} {atom['diameter']:.6f} {atom['density']:.6f} "
                       f"{atom['x']:.6f} {atom['y']:.6f} {atom['z']:.6f}\n")

    def _validate_atom_data(self, atom: Dict[str, Any]):
        """Validate atom data has required fields."""
        required_fields = ['id', 'type', 'diameter', 'density', 'x', 'y', 'z']
        if not all(field in atom for field in required_fields):
            logger.error(f"Missing required fields in atom data: {atom}")
            raise ValueError(f"Missing required fields in atom data: {atom}")

    def write_velocities(self, velocities: List[Dict[str, Any]]):
        """Write Velocities section to LAMMPS data file."""
        logger.debug(f"Writing {len(velocities) if velocities else 0} velocities")
        if not velocities:
            logger.warning("No velocity data to write.")
            return

        with open(self.filename, 'a') as f:
            f.write("\nVelocities # atom-ID vx vy vz\n\n")
            for vel in velocities:
                self._validate_velocity_data(vel)
                f.write(f"{vel['id']} {vel['vx']:.6f} {vel['vy']:.6f} {vel['vz']:.6f}\n")

    def _validate_velocity_data(self, vel: Dict[str, Any]):
        """Validate velocity data has required fields."""
        required_fields = ['id', 'vx', 'vy', 'vz']
        if not all(field in vel for field in required_fields):
            logger.error(f"Missing required fields in velocity data: {vel}")
            raise ValueError(f"Missing required fields in velocity data: {vel}")

    def write_masses(self, masses: List[Dict[str, Any]]):
        """Write Masses section to LAMMPS data file."""
        logger.debug(f"Writing {len(masses) if masses else 0} masses")
        if not masses:
            logger.warning("No mass data to write.")
            return

        with open(self.filename, 'a') as f:
            f.write("\nMasses # atom-type mass\n\n")
            for mass in masses:
                self._validate_mass_data(mass)
                f.write(f"{mass['type']} {mass['mass']:.6f}\n")

    def _validate_mass_data(self, mass: Dict[str, Any]):
        """Validate mass data has required fields."""
        required_fields = ['type', 'mass']
        if not all(field in mass for field in required_fields):
            logger.error(f"Missing required fields in mass data: {mass}")
            raise ValueError(f"Missing required fields in mass data: {mass}")

    def write_data(self, 
                   header_params: Optional[Dict[str, Any]] = None, 
                   atoms: Optional[List[Dict[str, Any]]] = None, 
                   velocities: Optional[List[Dict[str, Any]]] = None, 
                   masses: Optional[List[Dict[str, Any]]] = None):
        """Write complete LAMMPS data file."""
        self._validate_write_data_params(header_params, atoms, velocities, masses)
        
        logger.debug("Starting full data write")
        self.write_header(header_params)
        if masses is not None:
            self.write_masses(masses)
        if atoms is not None:
            self.write_atoms(atoms)
        if velocities is not None:
            self.write_velocities(velocities)
        logger.info(f"Successfully wrote LAMMPS data file to {self.filename}")
        logger.debug("Completed full data write")

    def _validate_write_data_params(self, header_params, atoms, velocities, masses):
        """Validate parameters for write_data method."""
        if header_params is None:
            raise ValueError("Header parameters are required")
            
        # Check required header parameters
        required_header = ['atoms', 'xlo', 'xhi', 'ylo', 'yhi', 'zlo', 'zhi']
        missing = [param for param in required_header if param not in header_params]
        if missing:
            raise ValueError(f"Missing required header parameters: {missing}")

        # Check data consistency
        if atoms and len(atoms) != header_params['atoms']:
            raise ValueError(f"Number of atoms ({len(atoms)}) does not match header ({header_params['atoms']})")

        if velocities and len(velocities) != header_params['atoms']:
            raise ValueError(f"Number of velocities ({len(velocities)}) does not match header ({header_params['atoms']})")

        if masses and 'atom types' in header_params and len(masses) != header_params['atom types']:
            raise ValueError(f"Number of masses ({len(masses)}) does not match atom types ({header_params['atom types']})")


# =============================================================================
# Data Format Conversion Utilities
# =============================================================================

class DataConverter:
    """Utility class to convert between different data formats."""
    
    @staticmethod
    def lammps_to_numpy(atoms: List[Dict[str, Any]]) -> Dict[str, np.ndarray]:
        """Convert LAMMPS atom data to NumPy arrays."""
        if not atoms:
            logger.warning("No atom data to convert")
            return {}
            
        n_atoms = len(atoms)
        positions = np.zeros((n_atoms, 3))
        types = np.zeros(n_atoms, dtype=np.int32)
        diameters = np.zeros(n_atoms)
        densities = np.zeros(n_atoms)
        
        try:
            for i, atom in enumerate(atoms):
                types[i] = atom['type']
                diameters[i] = atom['diameter']
                densities[i] = atom['density']
                positions[i, 0] = atom['x']
                positions[i, 1] = atom['y']
                positions[i, 2] = atom['z']
                
            return {
                'positions': positions,
                'types': types,
                'diameters': diameters,
                'densities': densities
            }
        except KeyError as e:
            logger.error(f"Missing key in atom data: {e}")
            raise ValueError(f"Missing key in atom data: {e}")
    
    @staticmethod
    def numpy_to_lammps(
        positions: np.ndarray, 
        types: Optional[np.ndarray] = None, 
        diameters: Optional[np.ndarray] = None, 
        densities: Optional[np.ndarray] = None
    ) -> List[Dict[str, Any]]:
        """Convert NumPy arrays to LAMMPS atom data format."""
        n_atoms = positions.shape[0]
        
        # Set defaults if not provided
        if types is None:
            types = np.ones(n_atoms, dtype=np.int32)
        if diameters is None:
            diameters = np.ones(n_atoms)
        if densities is None:
            densities = np.ones(n_atoms)
            
        # Validate array shapes
        if any(arr.shape[0] != n_atoms for arr in [types, diameters, densities]):
            raise ValueError("All arrays must have the same length matching the number of atoms")
            
        atoms = []
        for i in range(n_atoms):
            atom = {
                'id': i + 1,
                'type': int(types[i]),
                'diameter': float(diameters[i]),
                'density': float(densities[i]),
                'x': float(positions[i, 0]),
                'y': float(positions[i, 1]),
                'z': float(positions[i, 2])
            }
            atoms.append(atom)
            
        return atoms


# =============================================================================
# Trajectory File I/O
# =============================================================================

def write_lammpstrj(
    output_file: str, 
    step: int, 
    num_particles: int, 
    box_size: ArrayLike, 
    positions: ArrayLike,
    velocities: Optional[ArrayLike] = None, 
    orientations: Optional[ArrayLike] = None, 
    atom_types: Optional[ArrayLike] = None, 
    xy_tilt: float = 0.0):
    """Write system state to a LAMMPS trajectory file."""
    try:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
    except (OSError, TypeError):
        pass  # If output_file doesn't have a directory part, just continue
    
    # Set defaults
    if atom_types is None:
        atom_types = np.ones(num_particles, dtype=int)
    
    lx, ly, lz = box_size
    
    # Convert JAX arrays to numpy if needed
    positions = np.array(positions) if hasattr(positions, 'shape') else positions
    if velocities is not None:
        velocities = np.array(velocities) if hasattr(velocities, 'shape') else velocities
    if orientations is not None:
        orientations = np.array(orientations) if hasattr(orientations, 'shape') else orientations
    
    with open(output_file, 'a') as f:
        # Write header
        f.write("ITEM: TIMESTEP\n")
        f.write(f"{step}\n")
        f.write("ITEM: NUMBER OF ATOMS\n")
        f.write(f"{num_particles}\n")
        
        # Write box bounds
        _write_box_bounds(f, lx, ly, lz, xy_tilt)
        
        # Write atoms header and data
        columns = _build_atom_columns(velocities, orientations)
        f.write(f"ITEM: ATOMS {' '.join(columns)}\n")
        
        _write_atom_data(f, num_particles, positions, velocities, orientations, atom_types)
            
    logger.debug(f"Successfully wrote trajectory frame at step {step} to {output_file}")


def _write_box_bounds(f, lx: float, ly: float, lz: float, xy_tilt: float):
    """Write box bounds to trajectory file."""
    if abs(xy_tilt) > 1e-10:
        f.write("ITEM: BOX BOUNDS xy xz yz pp pp pp\n")
        f.write(f"{-lx/2:.6f} {lx/2:.6f} {xy_tilt:.6f}\n")
        f.write(f"{-ly/2:.6f} {ly/2:.6f} 0.0\n")
        f.write(f"{-lz/2:.6f} {lz/2:.6f} 0.0\n")
    else:
        f.write("ITEM: BOX BOUNDS pp pp pp\n")
        f.write(f"{-lx/2:.6f} {lx/2:.6f}\n")
        f.write(f"{-ly/2:.6f} {ly/2:.6f}\n") 
        f.write(f"{-lz/2:.6f} {lz/2:.6f}\n")


def _build_atom_columns(velocities: Optional[ArrayLike], orientations: Optional[ArrayLike]) -> List[str]:
    """Build list of column names for atom data."""
    columns = ["id", "type", "x", "y", "z"]
    
    if velocities is not None:
        vel_dim = velocities.shape[1] if len(velocities.shape) > 1 else 3
        if vel_dim >= 3:
            columns.extend(["vx", "vy", "vz"])
        if vel_dim == 6:
            columns.extend(["wx", "wy", "wz"])
            
    if orientations is not None:
        columns.extend(["qw", "qx", "qy", "qz"])
        
    return columns


def _write_atom_data(f, num_particles: int, positions: ArrayLike, velocities: Optional[ArrayLike], 
                    orientations: Optional[ArrayLike], atom_types: ArrayLike):
    """Write atom data to trajectory file."""
    for i in range(num_particles):
        line = [f"{i+1}", f"{atom_types[i]}"]  # id and type
        line.extend([f"{x:.6f}" for x in positions[i]])  # positions
        
        if velocities is not None:
            vel_dim = velocities.shape[1] if len(velocities.shape) > 1 else 3
            if vel_dim >= 3:
                line.extend([f"{v:.6f}" for v in velocities[i, :3]])  # linear velocities
            if vel_dim == 6:
                line.extend([f"{w:.6f}" for w in velocities[i, 3:]])  # angular velocities
                
        if orientations is not None:
            line.extend([f"{q:.6f}" for q in orientations[i]])  # quaternions
            
        f.write(" ".join(line) + "\n")


class LAMMPSTrajectoryWriter:
    """Class to manage LAMMPS trajectory file writing with configurable dump frequency."""
    
    def __init__(self, 
                 output_file: str,
                 dump_every: int = 1,
                 include_velocities: bool = False,
                 include_orientations: bool = False,
                 overwrite: bool = False):
        """Initialize LAMMPS trajectory writer."""
        self.output_file = Path(output_file)
        self.dump_every = dump_every
        self.include_velocities = include_velocities
        self.include_orientations = include_orientations
        
        # Create output directory if it doesn't exist
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize file (overwrite if requested)
        if overwrite and self.output_file.exists():
            self.output_file.unlink()
        
        self.frames_written = 0
        logger.info(f"LAMMPS trajectory writer initialized: {self.output_file}")
        logger.info(f"Dump frequency: every {self.dump_every} steps")
        
    def should_dump(self, step: int) -> bool:
        """Check if we should dump at this step."""
        return (step % self.dump_every) == 0
        
    def write_frame(self, 
                   step: int,
                   positions: ArrayLike,
                   box_size: ArrayLike,
                   velocities: Optional[ArrayLike] = None,
                   orientations: Optional[ArrayLike] = None,
                   atom_types: Optional[ArrayLike] = None,
                   xy_tilt: float = 0.0):
        """Write a single trajectory frame."""
        if not self.should_dump(step):
            return
            
        num_particles = len(positions)
        
        # Only include velocities/orientations if requested and provided
        vel_to_write = velocities if self.include_velocities else None
        ori_to_write = orientations if self.include_orientations else None
        
        write_lammpstrj(
            output_file=str(self.output_file),
            step=step,
            num_particles=num_particles,
            box_size=box_size,
            positions=positions,
            velocities=vel_to_write,
            orientations=ori_to_write,
            atom_types=atom_types,
            xy_tilt=xy_tilt
        )
        
        self.frames_written += 1
        
    def get_stats(self) -> Dict[str, Any]:
        """Get writing statistics."""
        return {
            "output_file": str(self.output_file),
            "frames_written": self.frames_written,
            "dump_every": self.dump_every,
            "include_velocities": self.include_velocities,
            "include_orientations": self.include_orientations
        }


def setup_lammps_trajectory_writers(output_dir: str, 
                                   dump_every: int = 10,
                                   include_velocities: bool = True) -> Dict[str, LAMMPSTrajectoryWriter]:
    """Setup LAMMPS trajectory writer."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    writers = {
        'trajectory': LAMMPSTrajectoryWriter(
            output_file=str(output_path / "trajectory.lammpstrj"),
            dump_every=dump_every,
            include_velocities=include_velocities,
            include_orientations=False,  # TODO: implement orientations
            overwrite=True
        )
    }
    
    logger.info(f"Set up {len(writers)} LAMMPS trajectory writers in {output_dir}")
    return writers


# =============================================================================
# Module Initialization
# =============================================================================

# Initialize basic logging when module is imported (will be reconfigured in main script)
logger.remove()
logger.add(sys.stderr, level="WARNING")


# =============================================================================
# Example Usage
# =============================================================================

if __name__ == "__main__":
    reader = LAMMPSDataReader("test.data")
    reader.summarize()

    atoms = reader.extract_atoms()
    velocities = reader.extract_velocities()
    masses = reader.extract_masses()

    writer = LAMMPSDataWriter("output.data")
    writer.write_data(reader.header_params, atoms, velocities, masses)
