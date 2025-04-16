"""
Pytest configuration file for the animax test suite.
"""
import sys
import os
from pathlib import Path

# Add the src directory to the Python path to ensure we can import the animax module
# This is needed when running tests without installing the package
src_dir = str(Path(__file__).parent.parent / "src")
if src_dir not in sys.path:
    sys.path.insert(0, src_dir) 