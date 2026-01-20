"""
Tests for the Adapter Factory module.
"""

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add module to path for imports
module_path = Path(__file__).parent.parent.parent.parent
if str(module_path) not in sys.path:
    sys.path.insert(0, str(module_path))


class TestAdapterFactory(unittest.TestCase):
    """Tests for the AdapterFactory class."""

    def setUp(self):
        """Set up test fixtures."""
        from neural_network.core.adapter_factory import AdapterFactory
        self.factory = AdapterFactory()

    def test_factory_initialization(self):
        """Test factory initializes correctly."""
        self.assertIsNotNone(self.factory)
        self.assertEqual(len(self.factory._adapter_cache), 0)
        self.assertEqual(len(self.factory._initialization_errors), 0)

    def test_get_available_forms(self):
        """Test getting all available form IDs."""
        forms = self.factory.get_available_forms()
        self.assertEqual(len(forms), 40)
        self.assertIn("01-visual", forms)
        self.assertIn("08-arousal", forms)
        self.assertIn("13-integrated-information", forms)
        self.assertIn("14-global-workspace", forms)
        self.assertIn("40-xenoconsciousness", forms)

    def test_get_forms_by_category(self):
        """Test getting forms by category."""
        sensory = self.factory.get_forms_by_category("sensory")
        self.assertEqual(len(sensory), 6)
        self.assertIn("01-visual", sensory)
        self.assertIn("02-auditory", sensory)

        cognitive = self.factory.get_forms_by_category("cognitive")
        self.assertEqual(len(cognitive), 6)
        self.assertIn("08-arousal", cognitive)

        theoretical = self.factory.get_forms_by_category("theoretical")
        self.assertEqual(len(theoretical), 5)
        self.assertIn("13-integrated-information", theoretical)

        specialized = self.factory.get_forms_by_category("specialized")
        self.assertEqual(len(specialized), 10)

        extended = self.factory.get_forms_by_category("extended")
        self.assertEqual(len(extended), 13)
        self.assertIn("28-philosophy", extended)
        self.assertIn("40-xenoconsciousness", extended)

    def test_get_critical_forms(self):
        """Test getting critical forms."""
        critical = self.factory.get_critical_forms()
        self.assertEqual(len(critical), 3)
        self.assertIn("08-arousal", critical)
        self.assertIn("13-integrated-information", critical)
        self.assertIn("14-global-workspace", critical)

    def test_is_valid_form(self):
        """Test form validation."""
        self.assertTrue(self.factory.is_valid_form("01-visual"))
        self.assertTrue(self.factory.is_valid_form("08-arousal"))
        self.assertTrue(self.factory.is_valid_form("40-xenoconsciousness"))
        self.assertFalse(self.factory.is_valid_form("99-nonexistent"))
        self.assertFalse(self.factory.is_valid_form(""))

    def test_create_adapter_visual(self):
        """Test creating a visual adapter."""
        adapter = self.factory.create_adapter("01-visual")
        self.assertIsNotNone(adapter)
        self.assertEqual(adapter.form_id, "01-visual")
        self.assertEqual(adapter.name, "Visual Consciousness")

    def test_create_adapter_arousal(self):
        """Test creating the arousal adapter."""
        adapter = self.factory.create_adapter("08-arousal")
        self.assertIsNotNone(adapter)
        self.assertEqual(adapter.form_id, "08-arousal")
        self.assertEqual(adapter.name, "Arousal Consciousness")

    def test_create_adapter_iit(self):
        """Test creating the IIT adapter."""
        adapter = self.factory.create_adapter("13-integrated-information")
        self.assertIsNotNone(adapter)
        self.assertEqual(adapter.form_id, "13-integrated-information")

    def test_create_adapter_global_workspace(self):
        """Test creating the Global Workspace adapter."""
        adapter = self.factory.create_adapter("14-global-workspace")
        self.assertIsNotNone(adapter)
        self.assertEqual(adapter.form_id, "14-global-workspace")

    def test_create_adapter_extended_forms(self):
        """Test creating extended form adapters."""
        form_ids = [
            "28-philosophy",
            "29-folk-wisdom",
            "30-animal-cognition",
            "31-plant-intelligence",
            "32-fungal-intelligence",
            "33-swarm-intelligence",
            "34-gaia-intelligence",
            "35-developmental-consciousness",
            "36-contemplative-states",
            "37-psychedelic-consciousness",
            "38-neurodivergent-consciousness",
            "39-trauma-consciousness",
            "40-xenoconsciousness",
        ]

        for form_id in form_ids:
            with self.subTest(form_id=form_id):
                adapter = self.factory.create_adapter(form_id)
                self.assertIsNotNone(adapter, f"Failed to create adapter for {form_id}")

    def test_create_adapter_invalid_form(self):
        """Test creating adapter for invalid form returns None."""
        adapter = self.factory.create_adapter("99-nonexistent")
        self.assertIsNone(adapter)

    def test_get_adapter_caching(self):
        """Test that adapters are cached."""
        adapter1 = self.factory.get_adapter("01-visual")
        adapter2 = self.factory.get_adapter("01-visual")
        self.assertIs(adapter1, adapter2)  # Same instance

    def test_get_adapter_no_create(self):
        """Test getting adapter without creating."""
        adapter = self.factory.get_adapter("01-visual", create_if_missing=False)
        self.assertIsNone(adapter)

        # Now create it
        created = self.factory.get_adapter("01-visual", create_if_missing=True)
        self.assertIsNotNone(created)

        # Now it should exist
        cached = self.factory.get_adapter("01-visual", create_if_missing=False)
        self.assertIsNotNone(cached)
        self.assertIs(cached, created)

    def test_get_adapters_by_category(self):
        """Test getting all adapters for a category."""
        sensory_adapters = self.factory.get_adapters_by_category("sensory")
        self.assertEqual(len(sensory_adapters), 6)
        self.assertIn("01-visual", sensory_adapters)
        self.assertIn("02-auditory", sensory_adapters)

    def test_create_all_adapters(self):
        """Test creating all 40 adapters."""
        adapters = self.factory.create_all_adapters()
        self.assertEqual(len(adapters), 40)

        # Verify all form IDs are present
        for form_id in self.factory.get_available_forms():
            self.assertIn(form_id, adapters)

    def test_destroy_adapter(self):
        """Test destroying a cached adapter."""
        self.factory.get_adapter("01-visual")
        self.assertIn("01-visual", self.factory._adapter_cache)

        result = self.factory.destroy_adapter("01-visual")
        self.assertTrue(result)
        self.assertNotIn("01-visual", self.factory._adapter_cache)

        # Destroying non-existent returns False
        result = self.factory.destroy_adapter("01-visual")
        self.assertFalse(result)

    def test_clear_cache(self):
        """Test clearing the adapter cache."""
        self.factory.create_all_adapters()
        self.assertEqual(len(self.factory._adapter_cache), 40)

        count = self.factory.clear_cache()
        self.assertEqual(count, 40)
        self.assertEqual(len(self.factory._adapter_cache), 0)

    def test_get_status(self):
        """Test getting factory status."""
        self.factory.get_adapter("01-visual")
        self.factory.get_adapter("08-arousal")

        status = self.factory.get_status()
        self.assertEqual(status["total_forms"], 40)
        self.assertEqual(status["cached_adapters"], 2)
        self.assertIn("01-visual", status["cached_form_ids"])
        self.assertIn("08-arousal", status["cached_form_ids"])
        self.assertIn("sensory", status["categories"])
        self.assertEqual(status["categories"]["sensory"], 6)


class TestAdapterFactoryNervousSystemIntegration(unittest.TestCase):
    """Tests for NervousSystem integration."""

    def setUp(self):
        """Set up test fixtures."""
        from neural_network.core.adapter_factory import AdapterFactory
        self.factory = AdapterFactory()

    def test_register_critical(self):
        """Test registering critical adapters with NervousSystem."""
        mock_ns = MagicMock()
        mock_ns.register_adapter = MagicMock()

        results = self.factory.register_critical(mock_ns)

        self.assertEqual(len(results), 3)
        self.assertTrue(results["08-arousal"])
        self.assertTrue(results["13-integrated-information"])
        self.assertTrue(results["14-global-workspace"])

        self.assertEqual(mock_ns.register_adapter.call_count, 3)

    def test_register_all(self):
        """Test registering all adapters with NervousSystem."""
        mock_ns = MagicMock()
        mock_ns.register_adapter = MagicMock()

        results = self.factory.register_all(mock_ns)

        self.assertEqual(len(results), 40)
        self.assertTrue(all(results.values()))
        self.assertEqual(mock_ns.register_adapter.call_count, 40)

    def test_register_by_category(self):
        """Test registering adapters by category."""
        mock_ns = MagicMock()
        mock_ns.register_adapter = MagicMock()

        results = self.factory.register_with_nervous_system(
            mock_ns,
            categories=["sensory"],
            include_critical=False
        )

        self.assertEqual(len(results), 6)
        self.assertIn("01-visual", results)
        self.assertIn("02-auditory", results)

    def test_register_specific_forms(self):
        """Test registering specific forms."""
        mock_ns = MagicMock()
        mock_ns.register_adapter = MagicMock()

        results = self.factory.register_with_nervous_system(
            mock_ns,
            form_ids=["01-visual", "28-philosophy"],
            include_critical=False
        )

        self.assertEqual(len(results), 2)
        self.assertTrue(results["01-visual"])
        self.assertTrue(results["28-philosophy"])


class TestConvenienceFunctions(unittest.TestCase):
    """Tests for module-level convenience functions."""

    def test_get_factory(self):
        """Test getting the default factory."""
        from neural_network.core.adapter_factory import get_factory

        factory1 = get_factory()
        factory2 = get_factory()
        self.assertIs(factory1, factory2)  # Same singleton instance

    def test_create_adapter(self):
        """Test convenience create_adapter function."""
        from neural_network.core.adapter_factory import create_adapter

        adapter = create_adapter("01-visual")
        self.assertIsNotNone(adapter)
        self.assertEqual(adapter.form_id, "01-visual")

    def test_get_adapter_info(self):
        """Test getting adapter info without instantiation."""
        from neural_network.core.adapter_factory import get_adapter_info

        info = get_adapter_info("08-arousal")
        self.assertIsNotNone(info)
        self.assertEqual(info["form_id"], "08-arousal")
        self.assertEqual(info["adapter_class"], "ArousalAdapter")
        self.assertEqual(info["category"], "cognitive")
        self.assertTrue(info["is_critical"])

        info = get_adapter_info("01-visual")
        self.assertEqual(info["category"], "sensory")
        self.assertFalse(info["is_critical"])

        info = get_adapter_info("99-nonexistent")
        self.assertIsNone(info)

    def test_list_all_adapters(self):
        """Test listing all adapter info."""
        from neural_network.core.adapter_factory import list_all_adapters

        adapters = list_all_adapters()
        self.assertEqual(len(adapters), 40)

        # Check structure
        for info in adapters:
            self.assertIn("form_id", info)
            self.assertIn("adapter_class", info)
            self.assertIn("category", info)
            self.assertIn("is_critical", info)


class TestFormCoverage(unittest.TestCase):
    """Tests to verify all 40 forms are properly covered."""

    def test_all_40_forms_in_registry(self):
        """Verify all 40 forms are in the registry."""
        from neural_network.core.adapter_factory import FORM_ADAPTER_REGISTRY

        self.assertEqual(len(FORM_ADAPTER_REGISTRY), 40)

        # Check form number sequence
        for i in range(1, 41):
            found = False
            for form_id in FORM_ADAPTER_REGISTRY:
                if form_id.startswith(f"{i:02d}-"):
                    found = True
                    break
            self.assertTrue(found, f"Form {i:02d} not found in registry")

    def test_category_totals(self):
        """Verify category totals add up correctly."""
        from neural_network.core.adapter_factory import FORM_CATEGORIES

        total = sum(len(forms) for cat, forms in FORM_CATEGORIES.items() if cat != "critical")
        self.assertEqual(total, 40)

    def test_no_duplicate_forms(self):
        """Verify no form appears in multiple non-critical categories."""
        from neural_network.core.adapter_factory import FORM_CATEGORIES

        seen = set()
        for cat, forms in FORM_CATEGORIES.items():
            if cat == "critical":
                continue  # Critical overlaps with other categories
            for form_id in forms:
                self.assertNotIn(form_id, seen, f"{form_id} appears in multiple categories")
                seen.add(form_id)


if __name__ == "__main__":
    unittest.main()
