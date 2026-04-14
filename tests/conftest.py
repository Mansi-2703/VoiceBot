"""
Pytest Configuration and Shared Fixtures

Provides common fixtures for all test modules.
"""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch


@pytest.fixture
def temp_output_dir():
    """Create a temporary output directory for tests"""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Patch Path("output") to use temp directory
        with patch("pathlib.Path") as mock_path:
            def path_side_effect(p):
                if p == "output":
                    return Path(tmpdir)
                return Path(p)
            
            mock_path.side_effect = path_side_effect
            yield Path(tmpdir)


@pytest.fixture
def sample_transcript():
    """Provide sample transcripts for testing"""
    return {
        "write_code": "Write a Python function to reverse a string",
        "create_file": "Create a file called notes.txt with hello world",
        "summarize_text": "Summarize this text: Machine learning is a subset of AI",
        "general_chat": "What is Python used for?",
        "compound": "Write a Python function and save it to reverse.py"
    }


@pytest.fixture
def mock_ollama_response():
    """Mock Ollama API responses"""
    def _mock_response(intent_type):
        responses = {
            "write_code": [
                {
                    "intent": "write_code",
                    "confidence": 0.95,
                    "extracted_params": {
                        "language": "Python",
                        "filename": "reverse.py",
                        "description": "function to reverse a string"
                    }
                }
            ],
            "create_file": [
                {
                    "intent": "create_file",
                    "confidence": 0.92,
                    "extracted_params": {
                        "filename": "notes.txt",
                        "content": "hello world"
                    }
                }
            ],
            "summarize_text": [
                {
                    "intent": "summarize_text",
                    "confidence": 0.88,
                    "extracted_params": {
                        "content": "Machine learning is a subset of AI"
                    }
                }
            ],
            "general_chat": [
                {
                    "intent": "general_chat",
                    "confidence": 0.80,
                    "extracted_params": {}
                }
            ],
            "compound": [
                {
                    "intent": "write_code",
                    "confidence": 0.94,
                    "extracted_params": {
                        "language": "Python",
                        "filename": "reverse.py",
                        "description": "function to reverse a string"
                    }
                },
                {
                    "intent": "create_file",
                    "confidence": 0.90,
                    "extracted_params": {
                        "filename": "reverse.py"
                    }
                }
            ]
        }
        return responses.get(intent_type, [])
    
    return _mock_response


@pytest.fixture
def mock_whisper_model():
    """Mock Whisper model for STT testing"""
    mock_model = Mock()
    
    def mock_transcribe(audio_file, language="en"):
        return {
            "text": "This is a test transcription",
            "language": language
        }
    
    mock_model.transcribe = mock_transcribe
    return mock_model


@pytest.fixture
def sample_audio_file(tmp_path):
    """Create a sample audio file for testing"""
    try:
        import soundfile as sf
        import numpy as np
        
        # Create 1 second of silence at 16kHz
        duration = 1
        sample_rate = 16000
        audio_data = np.zeros(sample_rate * duration, dtype=np.float32)
        
        audio_file = tmp_path / "test_audio.wav"
        sf.write(str(audio_file), audio_data, sample_rate)
        return str(audio_file)
    except Exception:
        pytest.skip("soundfile not available for audio generation")


@pytest.fixture
def cleanup_output_files():
    """Cleanup files created during testing"""
    created_files = []
    
    def register_file(filepath):
        created_files.append(filepath)
    
    yield register_file
    
    # Cleanup
    for filepath in created_files:
        if Path(filepath).exists():
            Path(filepath).unlink()


# Pytest configuration
def pytest_configure(config):
    """Configure pytest with custom markers"""
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests (deselect with '-m \"not integration\"')"
    )
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "requires_ollama: marks tests that require Ollama to be running"
    )
    config.addinivalue_line(
        "markers", "requires_whisper: marks tests that require Whisper model"
    )
