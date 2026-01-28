"""
Test Suite for Agent Frameworks Library.

This package contains comprehensive tests for all components of the
agent_frameworks library, including unit tests, integration tests,
and mocked external service tests.

Test Modules:
    - test_core: Core agent abstractions and state machines
    - test_context: Repository mapping and context management
    - test_execution: Architect-editor and session management
    - test_human_loop: Human-in-the-loop approval workflows
    - test_orchestration: Multi-agent coordination
    - test_tools: Tool registry and permissions
    - test_memory: Memory management and checkpointing
    - test_auditor: Framework auditing and pattern extraction
    - test_integration: End-to-end integration tests

Usage:
    # Run all tests
    python -m pytest tests/ -v

    # Run specific test module
    python -m pytest tests/test_core.py -v

    # Run with coverage
    python -m pytest tests/ --cov=agent_frameworks --cov-report=html
"""
