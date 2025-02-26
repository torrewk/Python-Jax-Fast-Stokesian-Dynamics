"""
Author: Dimos Aslanis (Utrecht University)
Date: 2025
Email: d.aslanis@uu.nl - reach out for questions or suggestions :)
"""

import re
from loguru import logger
import sys
import os
import numpy as np
from typing import Optional, Union, Tuple

def get_next_log_file(base_path):
    """
    Generate the next available log file name by appending a number.
    Rotates existing numbered files up by one.
    """
    dir_path = os.path.dirname(base_path)
    base_name = os.path.basename(base_path)
    name, ext = os.path.splitext(base_name)
    
    # Get all existing log files with numeric suffixes
    numbered_files = []
    pattern = f"{name}_\\d+{ext}"
    for f in os.listdir(dir_path):
        if re.match(pattern, f):
            try:
                num = int(f.replace(f"{name}_", "").replace(ext, ""))
                numbered_files.append((num, f))
            except ValueError:
                continue
    
    # Sort files by number in descending order
    numbered_files.sort(reverse=True)
    
    # Rotate existing numbered files
    for num, filename in numbered_files:
        old_path = os.path.join(dir_path, filename)
        new_path = os.path.join(dir_path, f"{name}_{num + 1}{ext}")
        try:
            if os.path.exists(old_path):  # Check again before renaming
                os.rename(old_path, new_path)
        except OSError as e:
            logger.error(f"Failed to rotate log file {filename}: {e}")
    
    # Handle the current file if it exists
    original_path = os.path.join(dir_path, base_name)
    if os.path.exists(original_path):
        try:
            new_path = os.path.join(dir_path, f"{name}_1{ext}")
            os.rename(original_path, new_path)
        except OSError as e:
            logger.error(f"Failed to rotate current log file: {e}")
    
    return original_path

def setup_logging(log_file="logs/simulation.log", debug_file="logs/debug.log", debug_enabled=False):
    """
    Set up logging configuration with console and file handlers.
    
    Parameters
    ----------
    log_file : str
        Path to the main log file. Default is 'logs/simulation.log'
    debug_file : str
        Path to the debug log file. Default is 'logs/debug.log'
    debug_enabled : bool
        Whether to enable debug logging. Default is False
    """
    # Create logs directory if it doesn't exist
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    # Remove default logger
    logger.remove()
    
    # Add basic console handler first so we don't miss early logs
    logger.add(sys.stderr, level="WARNING")
    
    # Handle log file rotation
    try:
        log_file = get_next_log_file(log_file)
        if debug_enabled:
            debug_file = get_next_log_file(debug_file)
    except Exception as e:
        logger.error(f"Failed to set up log rotation: {e}")
        return

    logger.info("Logging system initialized")
    
    # Console handler for general info (clean format)
    logger.add(
        sys.stdout,
        format="{message}",
        level="INFO",
        filter=lambda record: record["level"].name in ["INFO", "SUCCESS"]
    )
    
    # Console handler for warnings and errors (with more context)
    logger.add(
        sys.stderr,
        format="<level>{level:<8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:{line} | {message}",
        level="WARNING",
        filter=lambda record: record["level"].name in ["WARNING", "ERROR", "CRITICAL"]
    )
    
    # Main log file with rotation
    logger.add(
        log_file,
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level:<8} | {name}:{function}:{line} | {message}",
        level="INFO",
        # rotation="1 day",
        # retention="30 days",
        # compression="zip",
        enqueue=True,
        catch=True
    )
    
    # Debug log file with detailed information - only if debug is enabled
    if debug_enabled:
        logger.add(
            debug_file,
            format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level:<8} | {name}:{function}:{line} | {message}",
            level="DEBUG",
            # rotation="1 day",
            # retention="7 days",
            # compression="zip",
            enqueue=True,
            catch=True
        )
        logger.debug("Debug logging enabled")
        
    return

# Initialize logging when module is imported (TODO: move this into main script)
setup_logging(debug_enabled=True)  # Set default debug logging to False

def log_and_exit(message, code=1):
    """
    Log an error message and exit the script with a status code.
    
    Parameters
    ----------
    message : str
        Error message to log
    code : int, optional
        Exit status code, default is 1
    """
    logger.error(message)
    sys.exit(code)

class LAMMPSDataReader:
    """    
    Class to read and parse LAMMPS data files.
    """
    
    def __init__(self, filename):
        """
        Initialize a LAMMPS data file reader.

        This class reads and parses LAMMPS data files, storing various sections and parameters.

        Parameters
        ----------
        filename : str
            Path to the LAMMPS data file to be read.

        Attributes
        ----------
        data : dict
            Dictionary storing the parsed data from different sections of the file.
        sections : list
            List of section names to parse from the data file.
        header_params : dict 
            Dictionary storing header parameters from the data file.
            
        Usage
        --------
        >>> reader = LAMMPSDataReader("test.data")
        
        TODO
        --------
        - Add support for other sections (Bonds, Angles, etc.)
        """
        self.filename = filename
        self.data = {}
        self.sections = ["Masses", "Atoms", "Velocities"]
        self.header_params = {}
        logger.debug(f"Initializing LAMMPSDataReader with file: {filename}")
        self.read_file()
        
        logger.info(f"Reading LAMMPS data file: {filename}")
        logger.debug(f"Found sections: {list(self.data.keys())}")
        
        return
        

    def read_file(self):
        """
        Reads and parses a LAMMPS data file, storing the contents in the object's data structure.

        The method reads the file line by line, identifying different sections based on predefined 
        section headers. Header information and section data are stored separately.

        The file contents are stored in the self.data dictionary, where:
        - Keys are section names from the LAMMPS data file
        - Values are lists containing the raw text lines for each section

        Usage
        --------
        reader = LammpsReader(filename)
        reader.read_file()

        Returns
        -------
        None

        Side Effects
        ------------
        - Populates self.data dictionary with file contents
        - Updates header information through parse_header() method

        Notes
        -----
        - Empty lines and comments (starting with #) are ignored
        - Section headers must match predefined sections in self.sections
        - Lines before any section are treated as header information

        """
        logger.debug(f"Starting to read file: {self.filename}")
        
        if not os.path.exists(self.filename):
            logger.error(f"Data file not found: {self.filename}")
            raise FileNotFoundError(f"Data file not found: {self.filename}")
            
        try:
            with open(self.filename, 'r') as file:
                section = None
                for line in file:  # Process line by line instead of reading whole file
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
                        
        except PermissionError as e:
            logger.error(f"No permission to read file: {self.filename}")
            raise PermissionError(f"No permission to read file: {self.filename}")
        except UnicodeDecodeError as e:
            logger.error(f"File {self.filename} is not a valid text file")
            raise ValueError(f"File {self.filename} is not a valid text file")
            
        # Validate data consistency
        self._validate_data()
        
        logger.debug(f"Finished reading file. Found {len(self.data)} sections")

    def _validate_data(self):
        """
        Validates the consistency of the read data.
        """
        # Check if we have the expected number of atoms
        if "atoms" in self.header_params:
            expected_atoms = self.header_params["atoms"]
            if "Atoms" in self.data:
                actual_atoms = len(self.data["Atoms"])
                if actual_atoms != expected_atoms:
                    logger.error(f"Mismatch in atom count. Header: {expected_atoms}, Found: {actual_atoms}")
                    raise ValueError(f"Mismatch in atom count. Header: {expected_atoms}, Found: {actual_atoms}")
                    
            if "Velocities" in self.data:
                actual_velocities = len(self.data["Velocities"])
                if actual_velocities != expected_atoms:
                    logger.error(f"Mismatch in velocities count. Expected: {expected_atoms}, Found: {actual_velocities}")
                    raise ValueError(f"Mismatch in velocities count. Expected: {expected_atoms}, Found: {actual_velocities}")

        # Validate box dimensions
        box_params = ['xlo', 'xhi', 'ylo', 'yhi', 'zlo', 'zhi']
        if not all(param in self.header_params for param in box_params):
            logger.error("Missing box dimension parameters in header")
            raise ValueError("Missing box dimension parameters in header")
            
        for dim in [('xlo', 'xhi'), ('ylo', 'yhi'), ('zlo', 'zhi')]:
            if self.header_params[dim[0]] >= self.header_params[dim[1]]:
                logger.error(f"Invalid box dimensions: {dim[0]} >= {dim[1]}")
                raise ValueError(f"Invalid box dimensions: {dim[0]} >= {dim[1]}")

    def parse_header(self, line):
        """
        Parses the header lines of a LAMMPS data file and stores parameters in header_params dictionary.

        This method processes each line of the LAMMPS data file header, extracting key information such as:
        - Number of atoms, bonds, angles, etc. (matched as number + keyword)
        - Simulation box boundaries (xlo/xhi, ylo/yhi, zlo/zhi)

        Parameters
        ----------
        line : str
            A single line from the LAMMPS data file header section

        Returns
        -------
        None
            Values are stored in self.header_params dictionary

        Examples
        --------
        >>> parser.parse_header("4000 atoms")  # Stores {'atoms': 4000}
        >>> parser.parse_header("0.0 10.0 xlo xhi")  # Stores {'xlo': 0.0, 'xhi': 10.0}
        
        Notes
        -----
        Only lines with a number followed by a keyword are processed.
        Only supports orthorhombic simulation boxes.
        
        TODO
        ----
        - Add support for other box types (triclinic, etc.)
        """
        logger.debug(f"Parsing header line: {line}")
        try:
            if "xlo xhi" in line or "ylo yhi" in line or "zlo zhi" in line:
                try:
                    bounds = list(map(float, line.split()[:2]))
                    self.header_params[line.split()[2]] = bounds[0]
                    self.header_params[line.split()[3]] = bounds[1]
                except (ValueError, IndexError) as e:
                    logger.error(f"Invalid boundary values in line: {line}, Error: {e}")
                    return
                    
            match = re.match(r"(\d+)\s+(.+)", line)
            if match:
                value, key = match.groups()
                try:
                    self.header_params[key] = int(value)
                except ValueError as e:
                    logger.error(f"Failed to parse integer value in line: {line}, Error: {e}")
                    
        except Exception as e:
            logger.error(f"Error parsing header line: {line}, Error: {e}")
    
    def extract_atoms(self):
        """
        Extracts atom information from LAMMPS data file.
        
        Expects each line in the Atoms section to have the format:
        atom-ID atom-type diameter density x y z
        
        Returns:
            list: A list of dictionaries containing atom information with keys:
                - id (int): Atom ID
                - type (int): Atom type
                - diameter (float): Particle diameter
                - density (float): Particle density
                - x (float): X coordinate
                - y (float): Y coordinate
                - z (float): Z coordinate
                
            Returns None if no "Atoms" section is found in the data file.
        """
        logger.debug("Starting atom extraction")
        if "Atoms" not in self.data:
            # raise warning 
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

        # Sane check for atom data
        if "atoms" in self.header_params:
            if len(atoms) != self.header_params["atoms"]:
                logger.warning(f"Expected {self.header_params['atoms']} atoms, found {len(atoms)}.")
        else:
            logger.warning("No 'atoms' key found in header parameters, but extracted atom data.")
        
        logger.debug(f"Extracted {len(atoms) if atoms else 0} atoms")                
        return atoms

    def extract_velocities(self):
        """
        Extracts velocity information from LAMMPS data file.
        
        Expects each line in the Velocities section to have the format:
        atom-ID vx vy vz
        
        Returns:
            list: A list of dictionaries containing velocity information with keys:
                - id (int): Atom ID
                - vx (float): X velocity component
                - vy (float): Y velocity component
                - vz (float): Z velocity component
                
            Returns None if no "Velocities" section is found in the data file.
        """
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

    def extract_masses(self):
        """
        Extracts mass information from LAMMPS data file.
        
        Returns:
            list: A list of dictionaries containing mass information with keys:
                - type (int): Atom type
                - mass (float): Mass value
                
            Returns None if no "Masses" section is found in the data file.
        """
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
        """Prints a summary of extracted data."""
        logger.info("Summarizing extracted data")
        if "Atoms" in self.data:
            logger.info(f"Extracted {len(self.data['Atoms'])} atoms.")
        if "Velocities" in self.data:
            logger.info(f"Extracted {len(self.data['Velocities'])} velocities.")

class LAMMPSDataWriter:
    """
    Class to write LAMMPS data files.
    """
    
    def __init__(self, filename):
        """
        Initialize a LAMMPS data file writer.

        Parameters
        ----------
        filename : str
            Path where the LAMMPS data file will be written.
        """
        self.filename = filename
        logger.info(f"Initialized LAMMPSDataWriter for file: {filename}")
        logger.debug(f"Initialized LAMMPSDataWriter for file: {filename}")

    def write_header(self, header_params):
        """
        Write header section of LAMMPS data file.

        Parameters
        ----------
        header_params : dict
            Dictionary containing header parameters like number of atoms and box dimensions.
        """
        logger.debug(f"Writing header with parameters: {header_params}")
        with open(self.filename, 'w') as f:
            f.write("# LAMMPS data file generated by LAMMPSDataWriter\n\n")
            
            # Write counts
            for key in ['atoms', 'bonds', 'angles', 'dihedrals', 'impropers']:
                if key in header_params:
                    f.write(f"{header_params[key]} {key}\n")
            
            f.write("\n")
            
            for key in ['atom types', 'bond types', 'angle types', 'dihedral types', 'improper types']:
                if key in header_params:
                    f.write(f"{header_params[key]} {key}\n")
                    
            f.write("\n")
            
            # Write box dimensions
            for dim in [('xlo', 'xhi'), ('ylo', 'yhi'), ('zlo', 'zhi')]:
                if all(d in header_params for d in dim):
                    f.write(f"{header_params[dim[0]]} {header_params[dim[1]]} {dim[0]} {dim[1]}\n")

    def write_atoms(self, atoms):
        """
        Write Atoms section to LAMMPS data file.

        Parameters
        ----------
        atoms : list
            List of dictionaries containing atom data.
        """
        logger.debug(f"Writing {len(atoms) if atoms else 0} atoms")
        if not atoms:
            logger.warning("No atom data to write.")
            return

        with open(self.filename, 'a') as f:
            f.write("\nAtoms # atom-ID atom-type diameter density x y z\n\n")
            for atom in atoms:
                # Ensure all required fields are present
                required_fields = ['id', 'type', 'diameter', 'density', 'x', 'y', 'z']
                if not all(field in atom for field in required_fields):
                    logger.error(f"Missing required fields in atom data: {atom}")
                    raise ValueError(f"Missing required fields in atom data: {atom}")
                
                f.write(f"{atom['id']} {atom['type']} {atom['diameter']:.6f} {atom['density']:.6f} "
                       f"{atom['x']:.6f} {atom['y']:.6f} {atom['z']:.6f}\n")

    def write_velocities(self, velocities):
        """
        Write Velocities section to LAMMPS data file.

        Parameters
        ----------
        velocities : list
            List of dictionaries containing velocity data.
        """
        logger.debug(f"Writing {len(velocities) if velocities else 0} velocities")
        if not velocities:
            logger.warning("No velocity data to write.")
            return

        with open(self.filename, 'a') as f:
            f.write("\nVelocities # atom-ID vx vy vz\n\n")
            for vel in velocities:
                # Ensure all required fields are present
                required_fields = ['id', 'vx', 'vy', 'vz']
                if not all(field in vel for field in required_fields):
                    logger.error(f"Missing required fields in velocity data: {vel}")
                    raise ValueError(f"Missing required fields in velocity data: {vel}")
                
                f.write(f"{vel['id']} {vel['vx']:.6f} {vel['vy']:.6f} {vel['vz']:.6f}\n")

    def write_masses(self, masses):
        """
        Write Masses section to LAMMPS data file.

        Parameters
        ----------
        masses : list
            List of dictionaries containing mass data.
        """
        logger.debug(f"Writing {len(masses) if masses else 0} masses")
        if not masses:
            logger.warning("No mass data to write.")
            return

        with open(self.filename, 'a') as f:
            f.write("\nMasses # atom-type mass\n\n")
            for mass in masses:
                required_fields = ['type', 'mass']
                if not all(field in mass for field in required_fields):
                    logger.error(f"Missing required fields in mass data: {mass}")
                    raise ValueError(f"Missing required fields in mass data: {mass}")
                
                f.write(f"{mass['type']} {mass['mass']:.6f}\n")

    def write_data(self, header_params=None, atoms=None, velocities=None, masses=None):
        """
        Write complete LAMMPS data file including header, masses, atoms, and velocities.

        Parameters
        ----------
        header_params : dict
            Dictionary containing header parameters.
        atoms : list, optional
            List of dictionaries containing atom data.
        velocities : list, optional
            List of dictionaries containing velocity data.
        masses : list, optional
            List of dictionaries containing mass data.
        """
        # Sanity checks
        if header_params is None:
            logger.error("Header parameters are required")
            raise ValueError("Header parameters are required")
            
        # Check required header parameters
        required_header = ['atoms', 'xlo', 'xhi', 'ylo', 'yhi', 'zlo', 'zhi']
        missing = [param for param in required_header if param not in header_params]
        if missing:
            logger.error(f"Missing required header parameters: {missing}")
            raise ValueError(f"Missing required header parameters: {missing}")

        # Check data consistency
        if atoms and len(atoms) != header_params['atoms']:
            logger.error(f"Number of atoms ({len(atoms)}) does not match header ({header_params['atoms']})")
            raise ValueError(f"Number of atoms ({len(atoms)}) does not match header ({header_params['atoms']})")

        if velocities and len(velocities) != header_params['atoms']:
            logger.error(f"Number of velocities ({len(velocities)}) does not match header ({header_params['atoms']})")
            raise ValueError(f"Number of velocities ({len(velocities)}) does not match header ({header_params['atoms']})")

        if masses and 'atom types' in header_params and len(masses) != header_params['atom types']:
            logger.error(f"Number of masses ({len(masses)}) does not match atom types ({header_params['atom types']})")
            raise ValueError(f"Number of masses ({len(masses)}) does not match atom types ({header_params['atom types']})")
                
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

def write_lammpstrj(
    output_file: str,
    step: int,
    num_particles: int,
    box_size: Tuple[float, float, float],
    box_tilt: float,
    positions: np.ndarray,
    velocities: Optional[np.ndarray] = None,
    orientations: Optional[np.ndarray] = None,
) -> None:
    """Write system state to a LAMMPS trajectory file.
    
    Parameters
    ----------
    output_file : str
        Path to the output file
    step : int
        Current timestep
    num_particles : int
        Number of particles in the system
    box_size : tuple[float, float, float]
        Box dimensions (lx, ly, lz)
    box_tilt : float
        Box tilt factor for sheared systems
    positions : ndarray
        Particle positions, shape (num_particles, 3)
    velocities : ndarray, optional
        Particle velocities, shape (num_particles, 3) or (num_particles, 6)
    orientations : ndarray, optional
        Particle orientations, shape (num_particles, 4) for quaternions
    """
    lx, ly, lz = box_size
    with open(output_file, 'a') as f:
        # Write header
        f.write("ITEM: TIMESTEP\n")
        f.write(f"{step}\n")
        f.write("ITEM: NUMBER OF ATOMS\n")
        f.write(f"{num_particles}\n")
        
        # Write box bounds with tilt factors for sheared system
        f.write("ITEM: BOX BOUNDS xy xz yz\n")
        f.write(f"{-lx/2:.6f} {lx/2:.6f} {box_tilt*ly:.6f}\n")  # xlo xhi xy
        f.write(f"{-ly/2:.6f} {ly/2:.6f} 0.0\n")                 # ylo yhi xz
        f.write(f"{-lz/2:.6f} {lz/2:.6f} 0.0\n")                 # zlo zhi yz
        
        # Construct the header for atom data
        columns = ["id", "type", "x", "y", "z"]
        if velocities is not None:
            # Check if we have just linear velocities or both linear and angular
            vel_dim = velocities.shape[1]
            if vel_dim >= 3:
                columns.extend(["vx", "vy", "vz"])
            if vel_dim == 6:
                columns.extend(["wx", "wy", "wz"])
        if orientations is not None:
            columns.extend(["qw", "qx", "qy", "qz"])
            
        f.write(f"ITEM: ATOMS {' '.join(columns)}\n")
        
        # Write atom data
        for i in range(num_particles):
            line = [f"{i+1}", "1"]  # id and type
            line.extend([f"{x:.6f}" for x in positions[i]])  # positions
            
            if velocities is not None:
                if vel_dim >= 3:
                    line.extend([f"{v:.6f}" for v in velocities[i, :3]])  # linear velocities
                if vel_dim == 6:
                    line.extend([f"{w:.6f}" for w in velocities[i, 3:]])  # angular velocities
                    
            if orientations is not None:
                line.extend([f"{q:.6f}" for q in orientations[i]])  # quaternions
                
            f.write(" ".join(line) + "\n")

# Example Usage:
if __name__ == "__main__":
    reader = LAMMPSDataReader("test.data")
    reader.summarize()

    atoms = reader.extract_atoms()
    velocities = reader.extract_velocities()
    masses = reader.extract_masses()

    writer = LAMMPSDataWriter("output.data")
    writer.write_data(reader.header_params, atoms, velocities, masses)
