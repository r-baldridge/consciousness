"""
Form Registry Utilities

Utility module for discovering, loading, and inspecting all 40 consciousness
form interface modules. Provides dynamic form discovery, metadata extraction,
and instantiation helpers for use in integration tests and system tooling.
"""

import importlib
import importlib.util
import logging
import os
import sys
import types
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Base directory for the consciousness project
BASE_DIR = Path(__file__).parent.parent

# Canonical mapping of all 40 form IDs to their directory names
FORM_DIRECTORY_MAP: Dict[str, str] = {
    "01-visual": "01-visual",
    "02-auditory": "02-auditory",
    "03-somatosensory": "03-somatosensory",
    "04-olfactory": "04-olfactory",
    "05-gustatory": "05-gustatory",
    "06-interoceptive": "06-interoceptive",
    "07-emotional": "07-emotional",
    "08-arousal": "08-arousal",
    "09-perceptual": "09-perceptual",
    "10-self-recognition": "10-self-recognition",
    "11-meta-consciousness": "11-meta-consciousness",
    "12-narrative-consciousness": "12-narrative-consciousness",
    "13-integrated-information": "13-integrated-information",
    "14-global-workspace": "14-global-workspace",
    "15-higher-order-thought": "15-higher-order-thought",
    "16-predictive-coding": "16-predictive-coding",
    "17-recurrent-processing": "17-recurrent-processing",
    "18-primary-consciousness": "18-primary-consciousness",
    "19-reflective-consciousness": "19-reflective-consciousness",
    "20-collective-consciousness": "20-collective-consciousness",
    "21-artificial-consciousness": "21-artificial-consciousness",
    "22-dream-consciousness": "22-dream-consciousness",
    "23-lucid-dream": "23-lucid-dream",
    "24-locked-in": "24-locked-in",
    "25-blindsight": "25-blindsight",
    "26-split-brain": "26-split-brain",
    "27-altered-state": "27-altered-state",
    "28-philosophy": "28-philosophy",
    "29-folk-wisdom": "29-folk-wisdom",
    "30-animal-cognition": "30-animal-cognition",
    "31-plant-intelligence": "31-plant-intelligence",
    "32-fungal-intelligence": "32-fungal-intelligence",
    "33-swarm-intelligence": "33-swarm-intelligence",
    "34-gaia-intelligence": "34-gaia-intelligence",
    "35-developmental-consciousness": "35-developmental-consciousness",
    "36-contemplative-states": "36-contemplative-states",
    "37-psychedelic-consciousness": "37-psychedelic-consciousness",
    "38-neurodivergent-consciousness": "38-neurodivergent-consciousness",
    "39-trauma-consciousness": "39-trauma-consciousness",
    "40-xenoconsciousness": "40-xenoconsciousness",
}

# Known interface file names for forms that have Python implementations
KNOWN_INTERFACE_FILES: Dict[str, str] = {
    "08-arousal": "arousal_consciousness_interface.py",
    "27-altered-state": "non_dual_consciousness_interface.py",
    "28-philosophy": "philosophical_consciousness_interface.py",
    "29-folk-wisdom": "folk_wisdom_interface.py",
    "30-animal-cognition": "animal_cognition_interface.py",
    "31-plant-intelligence": "plant_intelligence_interface.py",
    "32-fungal-intelligence": "fungal_intelligence_interface.py",
    "33-swarm-intelligence": "swarm_intelligence_interface.py",
    "34-gaia-intelligence": "gaia_intelligence_interface.py",
    "35-developmental-consciousness": "developmental_consciousness_interface.py",
    "36-contemplative-states": "contemplative_states_interface.py",
    "37-psychedelic-consciousness": "psychedelic_consciousness_interface.py",
    "38-neurodivergent-consciousness": "neurodivergent_consciousness_interface.py",
    "39-trauma-consciousness": "trauma_consciousness_interface.py",
    "40-xenoconsciousness": "xenoconsciousness_interface.py",
}


def get_all_form_ids() -> List[str]:
    """
    Get all 40 form IDs in canonical order.

    Returns:
        List of form ID strings like ["01-visual", "02-auditory", ...].
    """
    return sorted(FORM_DIRECTORY_MAP.keys())


def get_form_directory(form_id: str) -> Path:
    """
    Get the filesystem path for a given form ID.

    Args:
        form_id: The form identifier (e.g., "08-arousal").

    Returns:
        Path to the form's root directory.

    Raises:
        ValueError: If the form_id is not recognized.
    """
    if form_id not in FORM_DIRECTORY_MAP:
        raise ValueError(
            f"Unknown form ID: {form_id}. "
            f"Valid IDs: {', '.join(sorted(FORM_DIRECTORY_MAP.keys()))}"
        )
    return BASE_DIR / FORM_DIRECTORY_MAP[form_id]


def discover_forms(base_dir: Optional[str] = None) -> Dict[str, str]:
    """
    Discover all form interface modules by scanning the filesystem.

    Looks for form directories (numbered NN-name pattern) that contain
    an interface/ subdirectory with Python files.

    Args:
        base_dir: Base directory to scan. Defaults to the consciousness
                  project root.

    Returns:
        Dictionary mapping form_id to the path of the primary interface
        Python file. Only forms with actual Python interface files are
        included.
    """
    scan_dir = Path(base_dir) if base_dir else BASE_DIR
    discovered: Dict[str, str] = {}

    for entry in sorted(scan_dir.iterdir()):
        if not entry.is_dir():
            continue

        # Match form directory pattern: NN-name
        name = entry.name
        if len(name) < 4 or not name[:2].isdigit() or name[2] != '-':
            continue

        form_id = name
        interface_dir = entry / "interface"

        if not interface_dir.is_dir():
            continue

        # Look for Python interface files (skip __init__.py and __pycache__)
        py_files = [
            f for f in interface_dir.iterdir()
            if f.is_file()
            and f.suffix == ".py"
            and f.name != "__init__.py"
            and not f.name.startswith("__")
        ]

        if py_files:
            # Use the known mapping if available, otherwise take the first file
            if form_id in KNOWN_INTERFACE_FILES:
                target = interface_dir / KNOWN_INTERFACE_FILES[form_id]
                if target.exists():
                    discovered[form_id] = str(target)
                else:
                    discovered[form_id] = str(py_files[0])
            else:
                discovered[form_id] = str(py_files[0])

    return discovered


def load_form_interface(form_dir: str) -> Any:
    """
    Dynamically load a form's interface module from its directory path.

    Uses importlib to load the module without requiring it to be on sys.path.
    Handles the common case of relative imports by patching the module into
    sys.modules.

    Args:
        form_dir: Path to the form's primary interface Python file.

    Returns:
        The loaded module object.

    Raises:
        ImportError: If the module cannot be loaded.
        FileNotFoundError: If the file does not exist.
    """
    file_path = Path(form_dir)

    if not file_path.exists():
        raise FileNotFoundError(f"Interface file not found: {file_path}")

    module_name = file_path.stem
    parent_dir = str(file_path.parent)

    # Ensure the parent directory is on the path
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)

    # Also add the grandparent (form root) for relative imports
    grandparent_dir = str(file_path.parent.parent)
    if grandparent_dir not in sys.path:
        sys.path.insert(0, grandparent_dir)

    spec = importlib.util.spec_from_file_location(module_name, str(file_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot create module spec for: {file_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module

    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        # Remove from sys.modules on failure to allow retry
        sys.modules.pop(module_name, None)
        raise ImportError(
            f"Failed to load module {module_name} from {file_path}: {exc}"
        ) from exc

    return module


def extract_form_metadata(module: Any) -> Dict[str, Any]:
    """
    Extract FORM_ID and FORM_NAME from a loaded interface module.

    Searches for classes in the module that have FORM_ID and FORM_NAME
    class attributes.

    Args:
        module: A loaded Python module.

    Returns:
        Dictionary with keys 'form_id', 'form_name', and 'interface_class'.
        Values are None if not found.
    """
    result = {
        "form_id": None,
        "form_name": None,
        "interface_class": None,
    }

    for attr_name in dir(module):
        obj = getattr(module, attr_name, None)
        if not isinstance(obj, type):
            continue

        form_id = getattr(obj, "FORM_ID", None)
        form_name = getattr(obj, "FORM_NAME", None) or getattr(obj, "NAME", None)

        if form_id is not None:
            result["form_id"] = form_id
            result["form_name"] = form_name
            result["interface_class"] = obj
            break

    return result


def instantiate_form_interface(module: Any) -> Optional[Any]:
    """
    Attempt to instantiate the primary interface class from a module.

    Finds the class with FORM_ID and tries a no-argument construction.

    Args:
        module: A loaded Python module.

    Returns:
        An instance of the interface class, or None if instantiation fails.
    """
    metadata = extract_form_metadata(module)
    interface_class = metadata.get("interface_class")

    if interface_class is None:
        logger.warning("No interface class found in module %s", module.__name__)
        return None

    try:
        instance = interface_class()
        return instance
    except Exception as exc:
        logger.warning(
            "Failed to instantiate %s: %s", interface_class.__name__, exc
        )
        return None


def get_forms_with_interfaces() -> List[str]:
    """
    Get list of form IDs that have Python interface implementations.

    Returns:
        Sorted list of form IDs that have discoverable interface files.
    """
    discovered = discover_forms()
    return sorted(discovered.keys())


def get_forms_without_interfaces() -> List[str]:
    """
    Get list of form IDs that lack Python interface implementations.

    Returns:
        Sorted list of form IDs whose interface directories are empty.
    """
    all_ids = set(get_all_form_ids())
    with_interfaces = set(get_forms_with_interfaces())
    return sorted(all_ids - with_interfaces)


def form_directory_exists(form_id: str) -> bool:
    """
    Check whether the directory for a given form ID exists on disk.

    Args:
        form_id: The form identifier.

    Returns:
        True if the directory exists.
    """
    try:
        form_dir = get_form_directory(form_id)
        return form_dir.is_dir()
    except ValueError:
        return False


def form_has_interface_dir(form_id: str) -> bool:
    """
    Check whether a form has an interface/ subdirectory.

    Args:
        form_id: The form identifier.

    Returns:
        True if the form directory contains an interface/ subdirectory.
    """
    try:
        form_dir = get_form_directory(form_id)
        return (form_dir / "interface").is_dir()
    except ValueError:
        return False


def form_has_spec_dir(form_id: str) -> bool:
    """
    Check whether a form has a spec/ subdirectory.

    Args:
        form_id: The form identifier.

    Returns:
        True if the form directory contains a spec/ subdirectory.
    """
    try:
        form_dir = get_form_directory(form_id)
        return (form_dir / "spec").is_dir()
    except ValueError:
        return False


def get_form_subdirectories(form_id: str) -> List[str]:
    """
    Get list of subdirectory names within a form directory.

    Args:
        form_id: The form identifier.

    Returns:
        Sorted list of subdirectory names.
    """
    try:
        form_dir = get_form_directory(form_id)
        return sorted(
            entry.name
            for entry in form_dir.iterdir()
            if entry.is_dir() and not entry.name.startswith(".")
        )
    except (ValueError, OSError):
        return []
