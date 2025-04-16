from dataclasses import dataclass, field
from loguru import logger
from typing import Any

@dataclass
class Anima:
    """
    Core class for the animax package. Serves as the main entry point for anima-related operations.

    Attributes:
        name: The name of the anima instance.
        config: Optional configuration dictionary for customization.
    """
    name: str = "default_anima"
    config: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """
        Post-initialization hook for Anima. Logs creation and configuration details.
        """
        logger.info(f"Initialized Anima instance with name: {self.name}")
        if self.config:
            logger.debug(f"Anima config: {self.config}")

    def run(self) -> None:
        """
        Example method to demonstrate extensibility. Logs and performs a placeholder action.
        """
        logger.info(f"Running Anima instance: {self.name}")
        # Placeholder for main anima logic
        logger.debug("Anima run() called. Extend this method with your logic.") 