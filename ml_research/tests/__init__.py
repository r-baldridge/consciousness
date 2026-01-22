"""
ML Research Integration Tests.

This package contains comprehensive integration tests for the ml_research module,
testing the full pipeline of orchestration, technique composition, and architecture
switching.

Test Organization:
- test_integration.py: Full pipeline and integration tests
- conftest.py: Shared fixtures and mock backends

Run tests:
    # Run all tests (excluding slow/GPU tests)
    pytest consciousness/ml_research/tests/ -v

    # Run with slow tests (requires GPU/large models)
    pytest consciousness/ml_research/tests/ -v --run-slow

    # Run specific test file
    pytest consciousness/ml_research/tests/test_integration.py -v
"""

__all__ = []
