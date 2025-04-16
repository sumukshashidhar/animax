from importlib.metadata import version as _version
from loguru import logger

def get_version() -> str:
    """
    Returns the current version of the anima package.
    """
    try:
        animax_version: str = _version("animax")
        logger.info(f"animax version: {animax_version}")
        return animax_version
    except Exception as exc:
        logger.error(f"Failed to get animax version: {exc}")
        return "unknown" 