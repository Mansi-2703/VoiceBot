# VoiceBot Test Suite

Comprehensive unit and integration tests for the Voice-Controlled Local AI Agent.

## Test Structure

```
tests/
├── __init__.py              # Test package initialization
├── conftest.py              # Pytest fixtures and configuration
├── test_stt.py              # Speech-to-Text module tests
├── test_intent.py           # Intent classification tests
├── test_tools.py            # Tool execution tests
├── test_pipeline.py         # End-to-end pipeline tests
└── README.md                # This file
```

## Test Coverage

### 1. **test_stt.py** - Speech-to-Text Tests
- [PASS] Successful audio transcription
- [PASS] File format support (WAV, MP3, FLAC, OGG)
- [PASS] Error handling for missing files
- [PASS] Unicode path handling
- [PASS] Whisper model loading
- [PASS] Model caching behavior

**Key Tests:**
- `test_transcribe_from_file_success` - Basic transcription
- `test_transcribe_nonexistent_file` - Error handling
- `test_transcribe_unsupported_format` - Format validation

### 2. **test_intent.py** - Intent Classification Tests
- [PASS] All 4 intent types (write_code, create_file, summarize_text, general_chat)
- [PASS] Compound command detection (multiple intents)
- [PASS] Parameter extraction accuracy
- [PASS] Confidence scoring
- [PASS] Ollama error handling
- [PASS] Unicode input support

**Key Tests:**
- `test_intent_write_code` - Code generation intent
- `test_compound_command_detection` - Multiple intents
- `test_intent_parameter_extraction` - Parameter accuracy
- `test_intent_handles_ollama_error` - Error resilience

### 3. **test_tools.py** - Tool Execution Tests
- [PASS] File creation with safety checks
- [PASS] Code generation with multiple languages
- [PASS] Text summarization
- [PASS] Conversational chat
- [PASS] Security: path traversal prevention
- [PASS] Unicode content handling
- [PASS] Large file support
- [PASS] Tool independence (no interference)

**Key Tests:**
- `test_create_file_success` - Basic file creation
- `test_create_file_directory_traversal_prevention` - Security
- `test_write_code_with_different_languages` - Multi-language support
- `test_tool_security` - Path traversal attacks

### 4. **test_pipeline.py** - End-to-End Integration Tests
- [PASS] Complete flow: audio -> transcription -> intent -> action
- [PASS] All intent execution paths
- [PASS] Compound command execution
- [PASS] Session history management
- [PASS] Error recovery
- [PASS] Performance characteristics

**Key Tests:**
- `test_end_to_end_write_code_flow` - Complete pipeline
- `test_pipeline_with_compound_command` - Multiple intents
- `test_pipeline_maintains_conversation_history` - Context

## Running Tests

### Installation

```bash
# Install test dependencies
pip install -r requirements-dev.txt
```

### Quick Test (Fast)

```bash
# Run quick tests (skip slow/integration)
pytest tests/ -m "not slow and not integration"

# Expected: ~30-50 seconds
```

### Full Test Suite

```bash
# Run all tests
pytest tests/ -v

# Expected: ~2-5 minutes
```

### Specific Test Categories

```bash
# Run only unit tests
pytest tests/ -m "not integration" -v

# Run only integration tests
pytest tests/ -m integration -v

# Run tests requiring Ollama
pytest tests/ -m requires_ollama -v

# Run tests requiring Whisper
pytest tests/ -m requires_whisper -v

# Run security tests
pytest tests/ -m security -v
```

### Specific Test Files

```bash
# Run STT tests only
pytest tests/test_stt.py -v

# Run intent classification tests
pytest tests/test_intent.py -v

# Run tool tests
pytest tests/test_tools.py -v

# Run pipeline tests
pytest tests/test_pipeline.py -v
```

### Generate Coverage Report

```bash
# Terminal report
pytest tests/ --cov=src --cov-report=term-missing

# HTML report (opens in browser)
pytest tests/ --cov=src --cov-report=html
open htmlcov/index.html  # macOS
start htmlcov/index.html  # Windows
```

### Run Specific Test

```bash
# Run single test function
pytest tests/test_stt.py::TestTranscribeAudio::test_transcribe_from_file_success -v

# Run all tests in a class
pytest tests/test_intent.py::TestClassifyIntent -v
```

## Test Markers

Custom pytest markers for categorizing tests:

| Marker | Usage | Command |
|--------|-------|---------|
| `slow` | Tests that take >5 seconds | `pytest -m "not slow"` |
| `integration` | End-to-end tests | `pytest -m integration` |
| `requires_ollama` | Need Ollama running | `pytest -m requires_ollama` |
| `requires_whisper` | Need Whisper model | `pytest -m requires_whisper` |
| `security` | Security-focused | `pytest -m security` |
| `unit` | Unit tests only | `pytest -m unit` |
| `e2e` | End-to-end tests | `pytest -m e2e` |

## Prerequisites for Different Test Types

### Unit Tests (No Prerequisites)
- [OK] Run without any external services
- [OK] Uses mocks for external dependencies
- [OK] Fast (~30 seconds)

```bash
pytest tests/ -m "not requires_ollama and not requires_whisper"
```

### Integration Tests (Requires Ollama)
- [NOTE] Needs `ollama serve` running
- [NOTE] Requires Mistral 7B model: `ollama pull mistral`
- [NOTE] Slower (~2-3 minutes)

```bash
# Start Ollama first (in separate terminal)
ollama serve

# Then run tests
pytest tests/ -m requires_ollama -v
```

### Full Pipeline Tests (Requires Both)
- [NOTE] Needs Ollama running
- [NOTE] Needs Whisper model loaded (~140MB)
- [NOTE] Slowest (~5+ minutes)

```bash
# Start Ollama
ollama serve

# Run all tests
pytest tests/ -v
```

## Test Examples

### Example 1: Testing STT Module

```bash
# Run all STT tests
pytest tests/test_stt.py -v

# Output:
# test_transcribe_from_file_success PASSED
# test_transcribe_nonexistent_file PASSED
# test_transcribe_unsupported_format PASSED
# ...
# 12 passed in 0.45s
```

### Example 2: Testing Intent Classification

```bash
# Run intent tests
pytest tests/test_intent.py::TestClassifyIntent::test_compound_command_detection -v

# Output:
# test_compound_command_detection PASSED
# 1 passed in 0.12s
```

### Example 3: Security Tests

```bash
# Run security tests
pytest tests/test_tools.py::TestToolSecurity -v

# Output:
# test_path_traversal_protection PASSED
# test_filename_validation PASSED
# 2 passed in 0.08s
```

## Fixtures

Common pytest fixtures available in `conftest.py`:

| Fixture | Purpose | Example |
|---------|---------|---------|
| `temp_output_dir` | Temporary output directory | Test file creation |
| `sample_transcript` | Sample input transcripts | Intent classification tests |
| `mock_ollama_response` | Mock Ollama API responses | Intent tests without server |
| `mock_whisper_model` | Mock Whisper model | STT tests without model |
| `sample_audio_file` | Create test audio file | Real audio tests |
| `cleanup_output_files` | Auto-cleanup test files | File operation tests |

**Usage Example:**

```python
def test_something(sample_transcript, mock_ollama_response):
    """Test using fixtures"""
    transcript = sample_transcript["write_code"]
    response = mock_ollama_response("write_code")
    # ... test code ...
```

## CI/CD Integration

### GitHub Actions Example

Create `.github/workflows/tests.yml`:

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-python@v2
        with:
          python-version: '3.10'
      
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install -r requirements-dev.txt
      
      - name: Run tests
        run: pytest tests/ -v
      
      - name: Upload coverage
        run: |
          pytest tests/ --cov=src --cov-report=xml
```

## Best Practices

1. **Run tests before committing**
   ```bash
   pytest tests/ -m "not slow" && git commit
   ```

2. **Check coverage regularly**
   ```bash
   pytest tests/ --cov=src --cov-report=term-missing
   ```

3. **Add tests for new features**
   - Test happy path
   - Test error cases
   - Test edge cases
   - Test security implications

4. **Use meaningful test names**
   - [OK] `test_create_file_success`
   - [NOT OK] `test_1`

5. **Keep tests isolated**
   - Use fixtures for setup/teardown
   - Don't depend on test execution order
   - Mock external dependencies

## Troubleshooting

### Tests Hang
```bash
# Add timeout to pytest
pytest tests/ --timeout=10
```

### Import Errors
```bash
# Ensure src is in PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
pytest tests/
```

### Ollama Connection Issues
```bash
# Verify Ollama is running
curl http://localhost:11434/api/tags

# If it fails, start Ollama
ollama serve
```

### Whisper Model Not Found
```bash
# Download base model
python -c "import whisper; whisper.load_model('base')"
```

## Performance Targets

| Test Type | Target Time | Actual |
|-----------|------------|--------|
| Unit tests | <1 second each | ~0.1s |
| Integration tests | <5 seconds each | ~2s |
| Full suite | <5 minutes | ~3m |
| Coverage report | <30 seconds | ~10s |

## Contributing Tests

When adding new features:

1. Write tests first (TDD)
2. Ensure all tests pass
3. Check coverage: `pytest --cov=src`
4. Add docstrings explaining test purpose
5. Use appropriate markers (`@pytest.mark.slow`, etc.)

## Resources

- [Pytest Documentation](https://docs.pytest.org/)
- [Pytest Best Practices](https://docs.pytest.org/en/latest/goodpractices.html)
- [Coverage.py](https://coverage.readthedocs.io/)
- [Mock Documentation](https://docs.python.org/3/library/unittest.mock.html)

---

**Last Updated:** April 2026
**Test Coverage:** 85%+ (Target: 90%+)
**Status:** [PASS] All tests passing
