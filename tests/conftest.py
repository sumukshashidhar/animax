"""
Pytest configuration file for the animax test suite.
"""
import sys
from pathlib import Path
import pytest
from unittest.mock import patch
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# Add the src directory to the Python path to ensure we can import the animax module
# This is needed when running tests without installing the package
src_dir = str(Path(__file__).parent.parent / "src")
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

# Define markers for different test categories
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "unit: mark a test as a unit test")
    config.addinivalue_line("markers", "integration: mark a test as an integration test")
    config.addinivalue_line("markers", "slow: mark a test as slow running")
    config.addinivalue_line("markers", "fast: mark a test as fast running")

# Custom pytest hooks for optimizing test runs
def pytest_runtest_setup(item):
    """Set up test environment for each test."""
    # Potentially perform setup operations for specific test types
    pass

def pytest_collection_modifyitems(config, items):
    """Modify collected test items to reorder them for optimization."""
    # Run fast tests first
    fast_items = []
    slow_items = []

    for item in items:
        if "fast" in item.keywords:
            fast_items.append(item)
        else:
            slow_items.append(item)

    items[:] = fast_items + slow_items

@pytest.fixture(scope="session", autouse=True)
def patch_retry_for_tests():
    """
    Global fixture to patch the retry mechanism for faster tests.
    This significantly speeds up tests that involve retries.
    """
    # Create a fast retry decorator for tests
    def fast_retry(*args, **kwargs):
        return retry(
            stop=stop_after_attempt(2),  # Reduce to just 2 attempts
            wait=wait_exponential(multiplier=0.1, min=0.1, max=0.2),  # Much faster wait times
            retry=retry_if_exception_type(Exception),
            reraise=True
        )

    # Patch tenacity.retry to use our faster version
    with patch('tenacity.retry', fast_retry):
        yield