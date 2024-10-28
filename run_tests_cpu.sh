#!/bin/bash

# Force JAX to use only the CPU
export JAX_PLATFORMS=cpu

# Run pytest on the jfsd package
pytest tests/test_class.py::TestClassCPU
