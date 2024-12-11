Installation
============

## Pre-requisites:
- cuda (tested with version 11.8, https://developer.nvidia.com/cuda-11-8-0-download-archive)
- cuDNN (tested with 8.6 for cuda 11, https://developer.nvidia.com/rdp/cudnn-archive)
- Python >= 3.9


1 - Set up work (virtual) environment:

.. code-block:: shell

	git clone https://github.com/torrewk/Python-Jax-Fast-Stokesian-Dynamics.git


2 - Go into the directory of the project and type the command:

.. code-block:: shell

	python3 -m venv .venv


3 - Activate the environment:

.. code-block:: shell

	source .venv/bin/activate

4 - Install correct version of jaxlib

.. code-block:: shell

	pip install jaxlib==0.4.17+cuda11.cudnn86 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

5 - Install jfsd and rest of dependencies

.. code-block:: shell

	pip install ".[test]"

		
6 - [optional] Build the documentation 	
	
.. code-block:: shell		

	cd docs ; make html ; cd ..
		
7 - Reboot the environment (needed fore pytest to work):

.. code-block:: shell

	deactivate && source .venv/bin/activate


8 - Run the JFSD unit tests

.. code-block:: shell

	pytest tests/test_class.py
