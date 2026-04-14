"""
VoiceBot Test Suite

Unit and integration tests for the Voice-Controlled Local AI Agent.

Test modules:
- test_stt.py: Speech-to-Text (Whisper) tests
- test_intent.py: Intent classification (Mistral) tests
- test_tools.py: Tool execution tests (file operations, code generation, etc.)
- test_pipeline.py: End-to-end pipeline integration tests
- conftest.py: Pytest fixtures and configuration

Running tests:

1. Quick test (skip slow/integration tests):
   pytest tests/ -m "not slow and not integration"

2. Full test suite:
   pytest tests/ -v

3. Run specific test file:
   pytest tests/test_stt.py -v

4. Run tests requiring external services:
   pytest tests/ -m "requires_ollama" -v

5. Generate coverage report:
   pytest tests/ --cov=src --cov-report=html
"""

__all__ = [
    "test_stt",
    "test_intent",
    "test_tools",
    "test_pipeline",
    "conftest"
]
