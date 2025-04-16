from dataclasses import dataclass, field
from loguru import logger
from typing import Any
import uuid

class Anima:
    """
    Core class for the animax package. Serves as the main entry point for anima-related operations.

    Attributes:
        name: The name of the anima instance.
        config: Optional configuration dictionary for customization.
    """
    def __init__(self, name: str = None):
        # each anima instance is unique, and has a unique id
        self._id : str = str(uuid.uuid4())
        # animas can be named. if a name is not provided, a fun name will be generated
        self.name: str = name if name is not None else "anima-" + self._id[:8]  