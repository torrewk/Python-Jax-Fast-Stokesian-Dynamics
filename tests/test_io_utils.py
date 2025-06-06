import os
import tempfile
import pytest
import numpy as np
import sys
from pathlib import Path

# Add the project root to the Python path
sys.path.append(str(Path(__file__).parent.parent))

from jfsd.io_utils import (
    LAMMPSDataReader, LAMMPSDataWriter, DataConverter, 
    write_lammpstrj, LAMMPSTrajectoryWriter, setup_logging,
    ThermoOutput, get_next_log_file, setup_lammps_trajectory_writers
)

# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def temp_dir():
    """Create a temporary directory for file I/O tests."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield tmp_dir


@pytest.fixture
def sample_atoms():
    """Create sample atom data for testing."""
    return [
        {'id': 1, 'type': 1, 'diameter': 1.0, 'density': 2.0, 'x': 0.0, 'y': 0.0, 'z': 0.0},
        {'id': 2, 'type': 1, 'diameter': 1.0, 'density': 2.0, 'x': 1.0, 'y': 1.0, 'z': 1.0},
        {'id': 3, 'type': 2, 'diameter': 1.5, 'density': 1.5, 'x': -1.0, 'y': -1.0, 'z': -1.0}
    ]


@pytest.fixture
def sample_velocities():
    """Create sample velocity data for testing."""
    return [
        {'id': 1, 'vx': 0.1, 'vy': 0.2, 'vz': 0.3},
        {'id': 2, 'vx': -0.1, 'vy': -0.2, 'vz': -0.3},
        {'id': 3, 'vx': 0.0, 'vy': 0.0, 'vz': 0.0}
    ]


@pytest.fixture
def sample_masses():
    """Create sample mass data for testing."""
    return [
        {'type': 1, 'mass': 1.0},
        {'type': 2, 'mass': 2.0}
    ]


@pytest.fixture
def sample_header_params():
    """Create sample header parameters for testing."""
    return {
        'atoms': 3,
        'atom types': 2,
        'xlo': -10.0, 'xhi': 10.0,
        'ylo': -10.0, 'yhi': 10.0,
        'zlo': -10.0, 'zhi': 10.0
    }


@pytest.fixture
def sample_lammps_data_file(temp_dir, sample_atoms, sample_velocities, sample_masses, sample_header_params):
    """Create a sample LAMMPS data file for testing."""
    filename = os.path.join(temp_dir, 'test.data')
    writer = LAMMPSDataWriter(filename)
    writer.write_data(sample_header_params, sample_atoms, sample_velocities, sample_masses)
    return filename


# =============================================================================
# LAMMPS Data Reader/Writer Tests
# =============================================================================

class TestLAMMPSDataReader:

    def test_read_data_file(self, sample_lammps_data_file, sample_header_params):
        """Test reading a LAMMPS data file."""
        reader = LAMMPSDataReader(sample_lammps_data_file)
        
        # Check header parameters
        assert reader.header_params['atoms'] == sample_header_params['atoms']
        assert reader.header_params['atom types'] == sample_header_params['atom types']
        assert reader.header_params['xlo'] == sample_header_params['xlo']
        
        # Check data sections
        assert 'Atoms' in reader.data
        assert 'Velocities' in reader.data
        assert 'Masses' in reader.data
        
        # Extract and verify atom data
        atoms = reader.extract_atoms()
        
        assert atoms is not None, "Expected atoms to be a list, but got None"
        assert len(atoms) == 3
        assert atoms[0]['id'] == 1
        assert atoms[1]['x'] == 1.0
        assert atoms[2]['type'] == 2
        
        # Extract and verify velocity data
        velocities = reader.extract_velocities()
        assert velocities is not None, "Expected velocities to be a list, but got None"
        assert len(velocities) == 3
        assert velocities[0]['vx'] == 0.1
        assert velocities[1]['vy'] == -0.2
        
        # Extract and verify mass data
        masses = reader.extract_masses()
        assert masses is not None, "Expected masses to be a list, but got None"
        assert len(masses) == 2
        assert masses[0]['type'] == 1
        assert masses[1]['mass'] == 2.0

    def test_file_not_found(self):
        """Test handling of non-existent files."""
        with pytest.raises(FileNotFoundError):
            LAMMPSDataReader("nonexistent_file.data")
    
    def test_invalid_data_format(self, temp_dir):
        """Test handling of invalid data formats."""
        # Create file with invalid format
        invalid_file = os.path.join(temp_dir, 'invalid.data')
        with open(invalid_file, 'w') as f:
            f.write("This is not a valid LAMMPS data file\n")
            f.write("It has no proper header or sections\n")
        
        # Read file and expect ValueError due to missing box dimensions
        with pytest.raises(ValueError, match="Missing box dimension parameters in header"):
            reader = LAMMPSDataReader(invalid_file)
            
        # If we want to test it without raising an error, we'd need to modify the 
        # LAMMPSDataReader class to be more lenient with invalid files

class TestLAMMPSDataWriter:

    def test_write_and_read_data(self, temp_dir, sample_atoms, sample_velocities, sample_masses, sample_header_params):
        """Test writing data and then reading it back."""
        # Write data
        filename = os.path.join(temp_dir, 'write_test.data')
        writer = LAMMPSDataWriter(filename)
        writer.write_data(sample_header_params, sample_atoms, sample_velocities, sample_masses)
        
        # Verify file exists
        assert os.path.exists(filename)
        
        # Read data back
        reader = LAMMPSDataReader(filename)
        read_atoms = reader.extract_atoms()
        read_velocities = reader.extract_velocities()
        read_masses = reader.extract_masses()
        
        # Verify content matches
        assert read_atoms is not None, "Expected read_atoms to be a list, but got None"
        assert len(read_atoms) == len(sample_atoms)
        assert read_atoms[0]['x'] == sample_atoms[0]['x']
        assert read_atoms[1]['y'] == sample_atoms[1]['y']
        assert read_atoms[2]['density'] == sample_atoms[2]['density']
        
        assert read_velocities is not None, "Expected read_velocities to be a list, but got None"
        assert len(read_velocities) == len(sample_velocities)
        assert read_velocities[0]['vx'] == sample_velocities[0]['vx']
        
        assert read_masses is not None, "Expected read_masses to be a list, but got None"
        assert len(read_masses) == len(sample_masses)
        assert read_masses[1]['mass'] == sample_masses[1]['mass']

    def test_invalid_parameters(self, temp_dir):
        """Test error handling with invalid parameters."""
        filename = os.path.join(temp_dir, 'invalid_params.data')
        writer = LAMMPSDataWriter(filename)
        
        # Missing required header parameters
        incomplete_header = {'atoms': 3}
        with pytest.raises(ValueError):
            writer.write_data(incomplete_header, [])
        
        # Inconsistent atom counts
        header = {
            'atoms': 2,  # Claiming 2 atoms
            'xlo': -10.0, 'xhi': 10.0, 
            'ylo': -10.0, 'yhi': 10.0, 
            'zlo': -10.0, 'zhi': 10.0
        }
        atoms = [  # But providing 3
            {'id': 1, 'type': 1, 'diameter': 1.0, 'density': 1.0, 'x': 0.0, 'y': 0.0, 'z': 0.0},
            {'id': 2, 'type': 1, 'diameter': 1.0, 'density': 1.0, 'x': 1.0, 'y': 1.0, 'z': 1.0},
            {'id': 3, 'type': 1, 'diameter': 1.0, 'density': 1.0, 'x': 2.0, 'y': 2.0, 'z': 2.0}
        ]
        with pytest.raises(ValueError):
            writer.write_data(header, atoms)


# =============================================================================
# DataConverter Tests
# =============================================================================

class TestDataConverter:

    def test_lammps_to_numpy(self, sample_atoms):
        """Test conversion from LAMMPS atom data to NumPy arrays."""
        # Convert
        result = DataConverter.lammps_to_numpy(sample_atoms)
        
        # Verify
        assert 'positions' in result
        assert 'types' in result
        assert 'diameters' in result
        assert 'densities' in result
        
        assert result['positions'].shape == (3, 3)
        assert result['types'].shape == (3,)
        assert result['diameters'].shape == (3,)
        assert result['densities'].shape == (3,)
        
        assert result['positions'][1, 0] == 1.0  # x of second atom
        assert result['types'][2] == 2  # type of third atom
        assert result['diameters'][0] == 1.0  # diameter of first atom
        assert result['densities'][0] == 2.0  # density of first atom

    def test_numpy_to_lammps(self):
        """Test conversion from NumPy arrays to LAMMPS atom data."""
        # Create NumPy arrays
        positions = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            [-1.0, -1.0, -1.0]
        ])
        types = np.array([1, 1, 2])
        diameters = np.array([1.0, 1.0, 1.5])
        densities = np.array([2.0, 2.0, 1.5])
        
        # Convert
        result = DataConverter.numpy_to_lammps(positions, types, diameters, densities)
        
        # Verify
        assert len(result) == 3
        assert result[0]['id'] == 1
        assert result[0]['type'] == 1
        assert result[1]['x'] == 1.0
        assert result[2]['density'] == 1.5

    def test_numpy_to_lammps_defaults(self):
        """Test conversion with default values."""
        positions = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0]
        ])
        
        # Convert with defaults
        result = DataConverter.numpy_to_lammps(positions)
        
        # Verify defaults were used
        assert len(result) == 2
        assert result[0]['type'] == 1
        assert result[0]['diameter'] == 1.0
        assert result[0]['density'] == 1.0

    def test_conversion_roundtrip(self, sample_atoms):
        """Test round-trip conversion: LAMMPS -> NumPy -> LAMMPS."""
        # LAMMPS -> NumPy
        numpy_data = DataConverter.lammps_to_numpy(sample_atoms)
        
        # NumPy -> LAMMPS
        lammps_data = DataConverter.numpy_to_lammps(
            numpy_data['positions'],
            numpy_data['types'],
            numpy_data['diameters'],
            numpy_data['densities']
        )
        
        # Verify data survived the round trip
        assert len(lammps_data) == len(sample_atoms)
        for i in range(len(sample_atoms)):
            # Compare key fields (not ID, which is auto-assigned)
            assert lammps_data[i]['type'] == sample_atoms[i]['type']
            assert lammps_data[i]['diameter'] == sample_atoms[i]['diameter']
            assert lammps_data[i]['density'] == sample_atoms[i]['density']
            assert lammps_data[i]['x'] == sample_atoms[i]['x']
            assert lammps_data[i]['y'] == sample_atoms[i]['y']
            assert lammps_data[i]['z'] == sample_atoms[i]['z']


# =============================================================================
# Trajectory Writing Tests
# =============================================================================

class TestTrajectoryWriting:

    def test_write_lammpstrj_single_frame(self, temp_dir):
        """Test writing a single frame to a LAMMPS trajectory file."""
        # Setup data
        output_file = os.path.join(temp_dir, 'single_frame.lammpstrj')
        step = 0
        num_particles = 2
        box_size = np.array([10.0, 10.0, 10.0])
        positions = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0]
        ])
        velocities = np.array([
            [0.1, 0.2, 0.3],
            [-0.1, -0.2, -0.3]
        ])
        
        # Write file
        write_lammpstrj(output_file, step, num_particles, box_size, positions, velocities)
        
        # Verify file exists
        assert os.path.exists(output_file)
        
        # Read file and check content
        with open(output_file, 'r') as f:
            lines = f.readlines()
            
        # Check header
        assert "ITEM: TIMESTEP" in lines[0]
        assert "0" in lines[1]  # Step
        assert "ITEM: NUMBER OF ATOMS" in lines[2]
        assert "2" in lines[3]  # Num particles
        assert "ITEM: BOX BOUNDS" in lines[4]
        
        # Check atom data
        atoms_line_index = lines.index("ITEM: ATOMS id type x y z vx vy vz\n")
        assert float(lines[atoms_line_index + 1].split()[2]) == 0.0  # x of first atom
        assert float(lines[atoms_line_index + 2].split()[4]) == 1.0  # z of second atom
        assert float(lines[atoms_line_index + 1].split()[5]) == 0.1  # vx of first atom

    def test_trajectory_writer(self, temp_dir):
        """Test the LAMMPSTrajectoryWriter class."""
        # Setup data
        output_file = os.path.join(temp_dir, 'writer_test.lammpstrj')
        positions = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0]
        ])
        box_size = np.array([10.0, 10.0, 10.0])
        velocities = np.array([
            [0.1, 0.2, 0.3],
            [-0.1, -0.2, -0.3]
        ])
        
        # Create writer and write two frames
        writer = LAMMPSTrajectoryWriter(
            output_file=output_file,
            dump_every=5,
            include_velocities=True
        )
        
        # Write frames
        writer.write_frame(0, positions, box_size, velocities)  # Should write (step 0)
        writer.write_frame(3, positions, box_size, velocities)  # Should NOT write (step 3)
        writer.write_frame(5, positions, box_size, velocities)  # Should write (step 5)
        
        # Check stats
        stats = writer.get_stats()
        assert stats["frames_written"] == 2
        assert stats["dump_every"] == 5
        
        # Read file and check content
        with open(output_file, 'r') as f:
            content = f.read()
            
        # Verify correct frames were written
        assert "ITEM: TIMESTEP\n0" in content
        assert "ITEM: TIMESTEP\n3" not in content
        assert "ITEM: TIMESTEP\n5" in content

    def test_multiple_writers(self, temp_dir):
        """Test setting up multiple trajectory writers."""
        # Setup writers
        output_dir = os.path.join(temp_dir, 'multi_writers')
        writers = setup_lammps_trajectory_writers(
            output_dir=output_dir,
            dump_every=10,
            include_velocities=True
        )
        
        # Check that writers were created
        assert 'trajectory' in writers
        assert writers['trajectory'].dump_every == 10
        assert writers['trajectory'].include_velocities is True
        
        # Check that output directory was created
        assert os.path.exists(output_dir)


# =============================================================================
# Logging Tests
# =============================================================================

class TestLogging:

    def test_log_file_creation(self, temp_dir):
        """Test that log files are created properly."""
        log_dir = os.path.join(temp_dir, 'logs')
        setup_logging(output_dir=log_dir, debug_enabled=True)
        
        # Check that log files were created
        assert os.path.exists(os.path.join(log_dir, 'simulation.log'))
        assert os.path.exists(os.path.join(log_dir, 'debug.log'))

    def test_log_rotation(self, temp_dir):
        """Test log file rotation."""
        # Create a log file
        log_file = os.path.join(temp_dir, 'rotate.log')
        with open(log_file, 'w') as f:
            f.write("Original log content\n")
        
        # Get next log file which should rotate the existing one
        next_file = get_next_log_file(log_file, max_backups=3)
        
        # Check that original file was renamed
        assert os.path.exists(os.path.join(temp_dir, 'rotate_1.log'))
        assert os.path.exists(log_file) is False  # Original is gone
        
        # Write to the new file
        with open(next_file, 'w') as f:
            f.write("New log content\n")
        
        # Check that the new file exists
        assert os.path.exists(next_file)


# =============================================================================
# ThermoOutput Tests
# =============================================================================

class TestThermoOutput:

    def test_thermo_output_creation(self):
        """Test creating a ThermoOutput object with default parameters."""
        thermo = ThermoOutput()
        assert thermo.thermo_period == 100
        assert 'Step' in thermo.columns
        assert 'Time' in thermo.columns
        assert 'Temp' in thermo.columns

    def test_thermo_data_calculation(self):
        """Test calculation of thermodynamic data."""
        thermo = ThermoOutput(thermo_period=1)
        
        # Update with some data
        positions = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0]
        ])
        velocities = np.array([
            [0.1, 0.2, 0.3],
            [-0.1, -0.2, -0.3]
        ])
        
        # Call update - this should calculate and store thermodynamic data
        thermo.update(
            step=0,
            time_step=0.01,
            temperature=1.0,
            positions=positions,
            velocities=velocities
        )
        
        # Check that data was stored
        data_history = thermo.get_data_history()
        assert 'Step' in data_history
        assert 'Time' in data_history
        assert 'Temp' in data_history
        assert data_history['Step'][0] == 0
        assert data_history['Temp'][0] == 1.0
        
        # Update again to capture more data
        thermo.update(
            step=1,
            time_step=0.01,
            temperature=1.1,
            positions=positions,
            velocities=velocities
        )
        
        # Check updated data
        data_history = thermo.get_data_history()
        assert len(data_history['Step']) == 2
        assert data_history['Step'][1] == 1
        assert data_history['Temp'][1] == 1.1


# =============================================================================
# Additional File Format Tests
# =============================================================================

class TestTrajectoryReading:
    """Tests for reading trajectory files."""
    
    def test_read_lammpstrj(self, temp_dir):
        """Test reading a LAMMPS trajectory file."""
        # First create a trajectory file
        output_file = os.path.join(temp_dir, 'test_traj.lammpstrj')
        
        # Write a few frames
        positions1 = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
        positions2 = np.array([[0.1, 0.1, 0.1], [1.1, 1.1, 1.1]])
        box_size = np.array([10.0, 10.0, 10.0])
        
        writer = LAMMPSTrajectoryWriter(output_file, dump_every=1)
        writer.write_frame(0, positions1, box_size)
        writer.write_frame(10, positions2, box_size)
        
        # Now read the trajectory using a custom function or tool
        # This would require implementing a trajectory reader in io_utils.py
        # For now, we'll just verify the file exists and has expected content
        with open(output_file, 'r') as f:
            content = f.read()
            
        assert "ITEM: TIMESTEP\n0" in content
        assert "ITEM: TIMESTEP\n10" in content
        assert "ITEM: ATOMS" in content
        
        # TODO: Implement a proper trajectory reader and test it here


class TestEdgeCases:
    """Tests for handling edge cases in I/O operations."""
    
    def test_empty_data_file(self, temp_dir):
        """Test handling of empty data files."""
        empty_file = os.path.join(temp_dir, 'empty.data')
        with open(empty_file, 'w') as f:
            pass  # Create empty file
            
        # This should raise an appropriate error
        with pytest.raises(Exception):
            LAMMPSDataReader(empty_file)
    
    def test_partial_data_file(self, temp_dir):
        """Test handling of partial data files with missing sections."""
        partial_file = os.path.join(temp_dir, 'partial.data')
        with open(partial_file, 'w') as f:
            f.write("# LAMMPS data file with header only\n\n")
            f.write("3 atoms\n")
            f.write("2 atom types\n\n")
            f.write("-10 10 xlo xhi\n")
            f.write("-10 10 ylo yhi\n")
            f.write("-10 10 zlo zhi\n")
            
        # This should read the file but return None for missing sections
        reader = LAMMPSDataReader(partial_file)
        assert reader.extract_atoms() is None
        assert reader.extract_velocities() is None
    
    def test_permission_errors(self, temp_dir):
        """Test handling of permission errors."""
        if os.name != 'nt':  # Skip on Windows as chmod works differently
            restricted_dir = os.path.join(temp_dir, 'restricted')
            os.makedirs(restricted_dir)
            os.chmod(restricted_dir, 0o000)  # Remove all permissions
            
            try:
                # This should raise a permission error
                with pytest.raises((PermissionError, OSError)):
                    setup_logging(output_dir=restricted_dir)
            finally:
                # Restore permissions so cleanup can happen
                os.chmod(restricted_dir, 0o755)
    
    def test_unicode_handling(self, temp_dir):
        """Test handling of unicode characters in filenames and content."""
        unicode_file = os.path.join(temp_dir, 'unicode_测试.data')
        
        # Create data with unicode content
        header_params = {
            'atoms': 1,
            'atom types': 1,
            'xlo': -10.0, 'xhi': 10.0,
            'ylo': -10.0, 'yhi': 10.0,
            'zlo': -10.0, 'zhi': 10.0
        }
        atoms = [{'id': 1, 'type': 1, 'diameter': 1.0, 'density': 1.0, 
                  'x': 0.0, 'y': 0.0, 'z': 0.0}]
        
        # Write and read data
        writer = LAMMPSDataWriter(unicode_file)
        writer.write_data(header_params, atoms)
        
        # Should be able to read it back
        reader = LAMMPSDataReader(unicode_file)
        read_atoms = reader.extract_atoms()
        assert len(read_atoms) == 1


class TestLargeFiles:
    """Tests for handling large files."""
    
    def test_large_data_file(self, temp_dir):
        """Test reading and writing large data files."""
        large_file = os.path.join(temp_dir, 'large.data')
        
        # Generate large number of atoms
        n_atoms = 10000
        positions = np.random.random((n_atoms, 3)) * 100
        types = np.ones(n_atoms, dtype=np.int32)
        diameters = np.ones(n_atoms)
        densities = np.ones(n_atoms)
        
        # Convert to LAMMPS format
        atoms = DataConverter.numpy_to_lammps(positions, types, diameters, densities)
        
        # Create header
        header_params = {
            'atoms': n_atoms,
            'atom types': 1,
            'xlo': -50.0, 'xhi': 50.0,
            'ylo': -50.0, 'yhi': 50.0,
            'zlo': -50.0, 'zhi': 50.0
        }
        
        # Write data
        writer = LAMMPSDataWriter(large_file)
        writer.write_data(header_params, atoms)
        
        # Read data back
        reader = LAMMPSDataReader(large_file)
        read_atoms = reader.extract_atoms()
        
        # Verify data
        assert read_atoms is not None, "Expected read_atoms to be a list, but got None"
        assert len(read_atoms) == n_atoms
        assert read_atoms[0]['x'] == pytest.approx(positions[0, 0], rel=1e-5)
        assert read_atoms[-1]['y'] == pytest.approx(positions[-1, 1], rel=1e-5)
        assert read_atoms[5000]['type'] == 1  # All types should be 1
        assert read_atoms[9999]['density'] == 1.0  # All densities should be 1.0
        assert read_atoms[9999]['diameter'] == 1.0  # All diameters should be 1.0
        assert read_atoms[9999]['id'] == 10000  # IDs should be sequential from 1 to n_atoms
        assert read_atoms[0]['id'] == 1  # First atom ID should be 1
        assert read_atoms[5000]['id'] == 5001  # ID of the 5000th atom should be 5001
        assert read_atoms[9999]['id'] == 10000  # Last atom ID should be n_atoms
        assert read_atoms[0]['z'] == pytest.approx(positions[0, 2], rel=1e-5)
        assert read_atoms[9999]['z'] == pytest.approx(positions[-1, 2], rel=1e-5)


class TestFileOverwrite:
    """Tests for file overwrite behavior."""
    
    def test_trajectory_overwrite(self, temp_dir):
        """Test overwriting existing trajectory files."""
        traj_file = os.path.join(temp_dir, 'overwrite.lammpstrj')
        
        # Write initial data
        positions1 = np.array([[0.0, 0.0, 0.0]])
        box_size = np.array([10.0, 10.0, 10.0])
        
        # Create first writer with overwrite=False (default)
        writer1 = LAMMPSTrajectoryWriter(traj_file, overwrite=False)
        writer1.write_frame(0, positions1, box_size)
        
        # Create second writer with overwrite=True
        writer2 = LAMMPSTrajectoryWriter(traj_file, overwrite=True)
        writer2.write_frame(1, positions1, box_size)
        
        # Read the file - it should only contain the second frame
        with open(traj_file, 'r') as f:
            content = f.read()
            
        assert "ITEM: TIMESTEP\n0" not in content  # First frame should be gone
        assert "ITEM: TIMESTEP\n1" in content      # Second frame should be present

# =============================================================================

if __name__ == "__main__":
    pytest.main(["-v"])
