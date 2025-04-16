import uuid
import pytest
from loguru import logger
from animax.anima import Anima


def test_anima_creation_with_default_name():
    """Test that an Anima instance is created with auto-generated name."""
    # Create an Anima with default name
    anima = Anima()
    
    # Check that the ID is properly generated (valid UUID)
    assert len(anima._id) == 36  # UUID string length
    try:
        uuid_obj = uuid.UUID(anima._id)
        assert isinstance(uuid_obj, uuid.UUID)
    except ValueError:
        pytest.fail("Anima ID is not a valid UUID")
    
    # Check that the name follows the expected pattern
    assert anima.name.startswith("anima-")
    assert anima.name[6:] == anima._id[:8]
    assert len(anima.name) == 14  # "anima-" + 8 chars from UUID


def test_anima_creation_with_custom_name():
    """Test that an Anima instance is created with a provided custom name."""
    custom_name = "test-anima"
    anima = Anima(name=custom_name)
    
    # Check that the ID is properly generated
    assert len(anima._id) == 36
    
    # Check that the custom name is used
    assert anima.name == custom_name
    assert anima.name != "anima-" + anima._id[:8]


def test_anima_unique_ids():
    """Test that multiple Anima instances have unique IDs."""
    anima1 = Anima()
    anima2 = Anima()
    
    # Check that the IDs are different
    assert anima1._id != anima2._id
    
    # Check that the auto-generated names are different
    assert anima1.name != anima2.name


def test_anima_with_none_name():
    """Test that an Anima instance with None name uses the default pattern."""
    anima = Anima(name=None)
    
    # Check that the name follows the expected pattern
    assert anima.name.startswith("anima-")
    assert anima.name[6:] == anima._id[:8] 