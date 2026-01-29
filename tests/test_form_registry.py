#!/usr/bin/env python3
"""
Test Suite: Form Registry and Discovery

Verifies that all 40 consciousness forms can be discovered, that forms
with Python interfaces expose FORM_ID and FORM_NAME metadata, and that
interface classes can be instantiated.

Tests cover:
- Form directory existence for all 40 forms
- Interface directory presence
- Python interface file discovery
- FORM_ID and FORM_NAME attribute verification
- Interface class instantiation
- Registry utility correctness
"""

import os
import sys
import unittest
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tests.form_registry_utils import (
    BASE_DIR,
    FORM_DIRECTORY_MAP,
    KNOWN_INTERFACE_FILES,
    discover_forms,
    extract_form_metadata,
    form_directory_exists,
    form_has_interface_dir,
    form_has_spec_dir,
    get_all_form_ids,
    get_form_directory,
    get_form_subdirectories,
    get_forms_with_interfaces,
    get_forms_without_interfaces,
    instantiate_form_interface,
    load_form_interface,
)


# ============================================================================
# DIRECTORY STRUCTURE TESTS
# ============================================================================


class TestFormDirectoryExistence(unittest.TestCase):
    """Verify all 40 form directories exist on disk."""

    def test_all_40_form_ids_defined(self):
        """The registry should define exactly 40 form IDs."""
        all_ids = get_all_form_ids()
        self.assertEqual(
            len(all_ids),
            40,
            f"Expected 40 form IDs, got {len(all_ids)}: {all_ids}",
        )

    def test_form_ids_are_sequential(self):
        """Form IDs should be numbered 01 through 40."""
        all_ids = get_all_form_ids()
        for i, form_id in enumerate(all_ids, start=1):
            expected_prefix = f"{i:02d}-"
            self.assertTrue(
                form_id.startswith(expected_prefix),
                f"Form at position {i} should start with '{expected_prefix}', "
                f"got '{form_id}'",
            )

    def test_all_form_directories_exist(self):
        """Every registered form should have a directory on disk."""
        missing = []
        for form_id in get_all_form_ids():
            if not form_directory_exists(form_id):
                missing.append(form_id)

        self.assertEqual(
            len(missing),
            0,
            f"Missing directories for forms: {missing}",
        )

    def test_all_forms_have_interface_dir(self):
        """Every form should have an interface/ subdirectory."""
        missing = []
        for form_id in get_all_form_ids():
            if not form_has_interface_dir(form_id):
                missing.append(form_id)

        self.assertEqual(
            len(missing),
            0,
            f"Forms missing interface/ directory: {missing}",
        )

    def test_all_forms_have_spec_dir(self):
        """Every form should have a spec/ subdirectory."""
        missing = []
        for form_id in get_all_form_ids():
            if not form_has_spec_dir(form_id):
                missing.append(form_id)

        self.assertEqual(
            len(missing),
            0,
            f"Forms missing spec/ directory: {missing}",
        )

    def test_standard_subdirectories(self):
        """
        Forms should generally have standard subdirectories:
        info/, interface/, spec/, system/, tests/, validation/.
        """
        expected_subdirs = {"info", "interface", "spec", "system", "tests", "validation"}
        forms_missing_subdirs = {}

        for form_id in get_all_form_ids():
            actual_subdirs = set(get_form_subdirectories(form_id))
            missing = expected_subdirs - actual_subdirs

            if missing:
                forms_missing_subdirs[form_id] = missing

        # Report but do not fail -- not all forms may have all subdirs
        if forms_missing_subdirs:
            count = len(forms_missing_subdirs)
            # Log as informational, not a hard failure
            print(
                f"\nINFO: {count} forms missing some standard subdirectories."
            )


# ============================================================================
# INTERFACE DISCOVERY TESTS
# ============================================================================


class TestFormDiscovery(unittest.TestCase):
    """Test the form discovery mechanism."""

    def test_discover_forms_returns_dict(self):
        """discover_forms() should return a dictionary."""
        result = discover_forms()
        self.assertIsInstance(result, dict)

    def test_discovered_forms_have_valid_ids(self):
        """All discovered form IDs should be in the canonical map."""
        discovered = discover_forms()
        all_ids = set(get_all_form_ids())

        for form_id in discovered:
            self.assertIn(
                form_id,
                all_ids,
                f"Discovered form '{form_id}' is not in the canonical map.",
            )

    def test_discovered_files_exist(self):
        """All discovered interface file paths should exist on disk."""
        discovered = discover_forms()

        for form_id, file_path in discovered.items():
            self.assertTrue(
                Path(file_path).exists(),
                f"Interface file for {form_id} does not exist: {file_path}",
            )

    def test_discovered_files_are_python(self):
        """All discovered interface files should be .py files."""
        discovered = discover_forms()

        for form_id, file_path in discovered.items():
            self.assertTrue(
                file_path.endswith(".py"),
                f"Interface file for {form_id} is not a Python file: {file_path}",
            )

    def test_arousal_form_discovered(self):
        """Form 08 (arousal) must always be discoverable -- it is critical."""
        discovered = discover_forms()
        self.assertIn(
            "08-arousal",
            discovered,
            "Critical form 08-arousal was not discovered.",
        )

    def test_known_interface_files_discovered(self):
        """All forms in KNOWN_INTERFACE_FILES should be discovered."""
        discovered = discover_forms()

        for form_id in KNOWN_INTERFACE_FILES:
            self.assertIn(
                form_id,
                discovered,
                f"Known form {form_id} was not discovered.",
            )

    def test_discover_with_custom_base_dir(self):
        """discover_forms() should accept a custom base directory."""
        result = discover_forms(str(BASE_DIR))
        self.assertIsInstance(result, dict)
        # Should find the same forms as default
        default_result = discover_forms()
        self.assertEqual(set(result.keys()), set(default_result.keys()))


# ============================================================================
# METADATA VERIFICATION TESTS
# ============================================================================


class TestFormMetadata(unittest.TestCase):
    """Verify that discovered forms expose FORM_ID and FORM_NAME."""

    @classmethod
    def setUpClass(cls):
        """Load all discoverable form modules once for the class."""
        cls.discovered = discover_forms()
        cls.loaded_modules = {}
        cls.load_errors = {}

        for form_id, file_path in cls.discovered.items():
            try:
                module = load_form_interface(file_path)
                cls.loaded_modules[form_id] = module
            except Exception as exc:
                cls.load_errors[form_id] = str(exc)

    def test_all_discovered_forms_loadable(self):
        """All discovered form modules should load without error."""
        if self.load_errors:
            error_summary = "\n".join(
                f"  {fid}: {err}" for fid, err in self.load_errors.items()
            )
            self.fail(
                f"Failed to load {len(self.load_errors)} form(s):\n{error_summary}"
            )

    def test_loaded_forms_have_form_id(self):
        """Every loaded form module should contain a class with FORM_ID."""
        missing_form_id = []

        for form_id, module in self.loaded_modules.items():
            metadata = extract_form_metadata(module)
            if metadata["form_id"] is None:
                missing_form_id.append(form_id)

        self.assertEqual(
            len(missing_form_id),
            0,
            f"Forms missing FORM_ID attribute: {missing_form_id}",
        )

    def test_loaded_forms_have_form_name(self):
        """Every loaded form module should contain a class with FORM_NAME."""
        missing_form_name = []

        for form_id, module in self.loaded_modules.items():
            metadata = extract_form_metadata(module)
            if metadata["form_name"] is None:
                missing_form_name.append(form_id)

        self.assertEqual(
            len(missing_form_name),
            0,
            f"Forms missing FORM_NAME attribute: {missing_form_name}",
        )

    def test_form_id_matches_directory(self):
        """The FORM_ID attribute should match the form's directory name."""
        mismatched = []

        for form_id, module in self.loaded_modules.items():
            metadata = extract_form_metadata(module)
            actual_form_id = metadata.get("form_id")

            if actual_form_id and actual_form_id != form_id:
                mismatched.append(
                    f"{form_id}: FORM_ID='{actual_form_id}'"
                )

        self.assertEqual(
            len(mismatched),
            0,
            f"Forms with FORM_ID not matching directory:\n"
            + "\n".join(f"  {m}" for m in mismatched),
        )

    def test_form_name_is_nonempty_string(self):
        """FORM_NAME should be a non-empty string."""
        invalid = []

        for form_id, module in self.loaded_modules.items():
            metadata = extract_form_metadata(module)
            form_name = metadata.get("form_name")

            if not isinstance(form_name, str) or not form_name.strip():
                invalid.append(f"{form_id}: FORM_NAME={form_name!r}")

        self.assertEqual(
            len(invalid),
            0,
            f"Forms with invalid FORM_NAME:\n"
            + "\n".join(f"  {i}" for i in invalid),
        )

    def test_form_id_format(self):
        """FORM_ID values should follow the NN-name pattern."""
        invalid = []

        for form_id, module in self.loaded_modules.items():
            metadata = extract_form_metadata(module)
            actual_id = metadata.get("form_id", "")

            if not actual_id or len(actual_id) < 4:
                invalid.append(f"{form_id}: '{actual_id}' too short")
            elif not actual_id[:2].isdigit():
                invalid.append(f"{form_id}: '{actual_id}' doesn't start with digits")
            elif actual_id[2] != "-":
                invalid.append(f"{form_id}: '{actual_id}' missing dash at position 2")

        self.assertEqual(
            len(invalid),
            0,
            f"Forms with invalid FORM_ID format:\n"
            + "\n".join(f"  {i}" for i in invalid),
        )


# ============================================================================
# INSTANTIATION TESTS
# ============================================================================


class TestFormInstantiation(unittest.TestCase):
    """Test that form interface classes can be instantiated."""

    @classmethod
    def setUpClass(cls):
        """Load modules and attempt instantiation."""
        cls.discovered = discover_forms()
        cls.instances = {}
        cls.instantiation_errors = {}

        for form_id, file_path in cls.discovered.items():
            try:
                module = load_form_interface(file_path)
                instance = instantiate_form_interface(module)
                if instance is not None:
                    cls.instances[form_id] = instance
                else:
                    cls.instantiation_errors[form_id] = "instantiate returned None"
            except Exception as exc:
                cls.instantiation_errors[form_id] = str(exc)

    def test_at_least_some_forms_instantiable(self):
        """At least some forms should be successfully instantiated."""
        self.assertGreater(
            len(self.instances),
            0,
            "No forms could be instantiated. "
            f"Errors: {self.instantiation_errors}",
        )

    def test_critical_form_08_instantiable(self):
        """Form 08 (arousal) must be instantiable as it is critical."""
        self.assertIn(
            "08-arousal",
            self.instances,
            f"Critical form 08-arousal failed to instantiate. "
            f"Error: {self.instantiation_errors.get('08-arousal', 'unknown')}",
        )

    def test_instances_have_form_id(self):
        """All instantiated forms should have a FORM_ID attribute."""
        missing = []
        for form_id, instance in self.instances.items():
            if not hasattr(instance, "FORM_ID"):
                missing.append(form_id)

        self.assertEqual(
            len(missing),
            0,
            f"Instantiated forms missing FORM_ID: {missing}",
        )

    def test_instances_have_form_name(self):
        """All instantiated forms should have a FORM_NAME or NAME attribute."""
        missing = []
        for form_id, instance in self.instances.items():
            has_name = hasattr(instance, "FORM_NAME") or hasattr(instance, "NAME")
            if not has_name:
                missing.append(form_id)

        self.assertEqual(
            len(missing),
            0,
            f"Instantiated forms missing FORM_NAME/NAME: {missing}",
        )

    def test_arousal_instance_has_gating(self):
        """Form 08 instance should expose gating functionality."""
        instance = self.instances.get("08-arousal")
        if instance is None:
            self.skipTest("08-arousal not instantiated")

        self.assertTrue(
            hasattr(instance, "get_gating_signals"),
            "Arousal interface missing get_gating_signals method.",
        )
        self.assertTrue(
            hasattr(instance, "is_form_allowed"),
            "Arousal interface missing is_form_allowed method.",
        )
        self.assertTrue(
            hasattr(instance, "register_form"),
            "Arousal interface missing register_form method.",
        )

    def test_form_instances_are_distinct(self):
        """Each instantiated form should be a unique object."""
        seen_ids = set()
        duplicates = []

        for form_id, instance in self.instances.items():
            obj_id = id(instance)
            if obj_id in seen_ids:
                duplicates.append(form_id)
            seen_ids.add(obj_id)

        self.assertEqual(
            len(duplicates),
            0,
            f"Duplicate instances found for: {duplicates}",
        )


# ============================================================================
# REGISTRY UTILITY TESTS
# ============================================================================


class TestRegistryUtilities(unittest.TestCase):
    """Test the form_registry_utils helper functions."""

    def test_get_all_form_ids_returns_list(self):
        """get_all_form_ids() should return a list."""
        result = get_all_form_ids()
        self.assertIsInstance(result, list)

    def test_get_all_form_ids_sorted(self):
        """get_all_form_ids() should return IDs in sorted order."""
        result = get_all_form_ids()
        self.assertEqual(result, sorted(result))

    def test_get_form_directory_valid(self):
        """get_form_directory() should return a Path for valid IDs."""
        path = get_form_directory("08-arousal")
        self.assertIsInstance(path, Path)
        self.assertTrue(path.exists())

    def test_get_form_directory_invalid(self):
        """get_form_directory() should raise ValueError for invalid IDs."""
        with self.assertRaises(ValueError):
            get_form_directory("99-nonexistent")

    def test_get_forms_with_interfaces(self):
        """get_forms_with_interfaces() should return a non-empty list."""
        result = get_forms_with_interfaces()
        self.assertIsInstance(result, list)
        self.assertGreater(len(result), 0)
        self.assertIn("08-arousal", result)

    def test_get_forms_without_interfaces(self):
        """get_forms_without_interfaces() should return complement set."""
        with_ifaces = set(get_forms_with_interfaces())
        without_ifaces = set(get_forms_without_interfaces())
        all_ids = set(get_all_form_ids())

        # Union should equal all form IDs
        self.assertEqual(
            with_ifaces | without_ifaces,
            all_ids,
            "Union of with/without interfaces should cover all forms.",
        )
        # Intersection should be empty
        self.assertEqual(
            with_ifaces & without_ifaces,
            set(),
            "A form cannot be both with and without interface.",
        )

    def test_form_directory_exists_valid(self):
        """form_directory_exists() should return True for existing forms."""
        self.assertTrue(form_directory_exists("01-visual"))
        self.assertTrue(form_directory_exists("40-xenoconsciousness"))

    def test_form_directory_exists_invalid(self):
        """form_directory_exists() should return False for invalid IDs."""
        self.assertFalse(form_directory_exists("99-nonexistent"))

    def test_form_has_interface_dir_check(self):
        """form_has_interface_dir() should return True for forms with interface/."""
        self.assertTrue(form_has_interface_dir("08-arousal"))

    def test_get_form_subdirectories(self):
        """get_form_subdirectories() should return subdirectory names."""
        subdirs = get_form_subdirectories("08-arousal")
        self.assertIsInstance(subdirs, list)
        self.assertIn("interface", subdirs)

    def test_form_map_has_40_entries(self):
        """FORM_DIRECTORY_MAP should have exactly 40 entries."""
        self.assertEqual(
            len(FORM_DIRECTORY_MAP),
            40,
            f"Expected 40 entries, got {len(FORM_DIRECTORY_MAP)}",
        )


# ============================================================================
# MAIN
# ============================================================================


if __name__ == "__main__":
    unittest.main(verbosity=2)
