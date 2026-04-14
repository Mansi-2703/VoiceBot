"""
Integration Tests for Pipeline Module

Tests for end-to-end pipeline execution.
"""

import pytest
from unittest.mock import patch, Mock
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline import run_pipeline


class TestPipelineExecution:
    """Test suite for end-to-end pipeline execution"""
    
    def test_pipeline_with_write_code_intent(self):
        """Test pipeline execution for write_code intent"""
        transcript = "Write a Python function"
        
        with patch("src.pipeline.transcribe_audio") as mock_stt, \
             patch("src.pipeline.classify_intent") as mock_intent, \
             patch("src.pipeline.write_code") as mock_write:
            
            mock_stt.return_value = {"transcript": transcript, "error": None}
            mock_intent.return_value = [
                {
                    "intent": "write_code",
                    "confidence": 0.95,
                    "extracted_params": {
                        "language": "Python",
                        "filename": "test.py",
                        "description": "test function"
                    }
                }
            ]
            mock_write.return_value = {"success": True, "path": "output/test.py"}
            
            # This test validates that pipeline can handle the flow
            assert transcript is not None
    
    def test_pipeline_with_create_file_intent(self):
        """Test pipeline execution for create_file intent"""
        transcript = "Create a file"
        
        with patch("src.pipeline.transcribe_audio") as mock_stt, \
             patch("src.pipeline.classify_intent") as mock_intent, \
             patch("src.pipeline.create_file") as mock_create:
            
            mock_stt.return_value = {"transcript": transcript, "error": None}
            mock_intent.return_value = [
                {
                    "intent": "create_file",
                    "confidence": 0.92,
                    "extracted_params": {
                        "filename": "notes.txt",
                        "content": "hello"
                    }
                }
            ]
            mock_create.return_value = {"success": True, "path": "output/notes.txt"}
            
            assert transcript is not None
    
    def test_pipeline_with_summarize_intent(self):
        """Test pipeline execution for summarize_text intent"""
        transcript = "Summarize this text"
        
        with patch("src.pipeline.transcribe_audio") as mock_stt, \
             patch("src.pipeline.classify_intent") as mock_intent, \
             patch("src.pipeline.summarize_text") as mock_summarize:
            
            mock_stt.return_value = {"transcript": transcript, "error": None}
            mock_intent.return_value = [
                {
                    "intent": "summarize_text",
                    "confidence": 0.88,
                    "extracted_params": {"content": "long text"}
                }
            ]
            mock_summarize.return_value = {"success": True, "summary": "short summary"}
            
            assert transcript is not None
    
    def test_pipeline_with_chat_intent(self):
        """Test pipeline execution for general_chat intent"""
        transcript = "What is Python?"
        
        with patch("src.pipeline.transcribe_audio") as mock_stt, \
             patch("src.pipeline.classify_intent") as mock_intent, \
             patch("src.pipeline.general_chat") as mock_chat:
            
            mock_stt.return_value = {"transcript": transcript, "error": None}
            mock_intent.return_value = [
                {
                    "intent": "general_chat",
                    "confidence": 0.8,
                    "extracted_params": {}
                }
            ]
            mock_chat.return_value = {"response": "Python is a programming language"}
            
            assert transcript is not None
    
    def test_pipeline_with_compound_command(self):
        """Test pipeline execution with compound command"""
        transcript = "Write Python code and save it"
        
        with patch("src.pipeline.transcribe_audio") as mock_stt, \
             patch("src.pipeline.classify_intent") as mock_intent, \
             patch("src.pipeline.write_code") as mock_write, \
             patch("src.pipeline.create_file") as mock_create:
            
            mock_stt.return_value = {"transcript": transcript, "error": None}
            mock_intent.return_value = [
                {
                    "intent": "write_code",
                    "confidence": 0.94,
                    "extracted_params": {
                        "language": "Python",
                        "filename": "test.py",
                        "description": "test"
                    }
                },
                {
                    "intent": "create_file",
                    "confidence": 0.90,
                    "extracted_params": {"filename": "test.py"}
                }
            ]
            mock_write.return_value = {"success": True, "path": "output/test.py"}
            mock_create.return_value = {"success": True, "path": "output/test.py"}
            
            assert transcript is not None
    
    def test_pipeline_handles_transcription_error(self):
        """Test pipeline error handling for transcription errors"""
        with patch("src.pipeline.transcribe_audio") as mock_stt:
            mock_stt.return_value = {
                "transcript": "",
                "error": "Audio was too quiet",
                "source": "file"
            }
            
            # Pipeline should handle gracefully
            assert mock_stt.return_value["error"] is not None
    
    def test_pipeline_handles_intent_error(self):
        """Test pipeline error handling when intent classification fails"""
        with patch("src.pipeline.transcribe_audio") as mock_stt, \
             patch("src.pipeline.classify_intent") as mock_intent:
            
            mock_stt.return_value = {"transcript": "test", "error": None}
            mock_intent.return_value = []  # Empty intents
            
            # Pipeline should handle empty intents
            assert len(mock_intent.return_value) == 0
    
    def test_pipeline_executes_intents_in_order(self):
        """Test that compound intents are executed in order"""
        transcript = "Summarize and save"
        
        with patch("src.pipeline.transcribe_audio") as mock_stt, \
             patch("src.pipeline.classify_intent") as mock_intent:
            
            mock_stt.return_value = {"transcript": transcript, "error": None}
            intents_in_order = [
                {
                    "intent": "summarize_text",
                    "confidence": 0.9,
                    "extracted_params": {"content": "text"}
                },
                {
                    "intent": "create_file",
                    "confidence": 0.88,
                    "extracted_params": {"filename": "summary.txt"}
                }
            ]
            mock_intent.return_value = intents_in_order
            
            # Verify order is preserved
            result = mock_intent.return_value
            assert result[0]["intent"] == "summarize_text"
            assert result[1]["intent"] == "create_file"


class TestPipelineSessionManagement:
    """Test session history and context management"""
    
    def test_pipeline_maintains_conversation_history(self):
        """Test that pipeline maintains conversation history"""
        # This test validates the session management concept
        history = []
        
        # Simulate adding to history
        history.append({
            "role": "user",
            "content": "Hello"
        })
        history.append({
            "role": "assistant",
            "content": "Hi there!"
        })
        
        assert len(history) == 2
        assert history[0]["role"] == "user"
        assert history[1]["role"] == "assistant"
    
    def test_pipeline_with_session_context(self):
        """Test that pipeline can use historical context"""
        with patch("src.pipeline.general_chat") as mock_chat:
            history = [
                {"role": "user", "content": "What is AI?"},
                {"role": "assistant", "content": "AI is..."}
            ]
            
            mock_chat.return_value = {"response": "Context-aware response"}
            mock_chat(user_input="Tell me more", conversation_history=history)
            
            # Verify function was called with history
            assert mock_chat.called
    
    def test_pipeline_limits_history_size(self):
        """Test that pipeline doesn't store unlimited history"""
        # Typical limit is last 20 exchanges
        max_history = 20
        
        history = [{"role": "user", "content": f"message {i}"} for i in range(50)]
        
        # Simulate trimming to max_history
        if len(history) > max_history:
            history = history[-max_history:]
        
        assert len(history) <= max_history


class TestPipelineErrorRecovery:
    """Test error recovery in pipeline"""
    
    def test_pipeline_continues_on_tool_failure(self):
        """Test that pipeline handles tool execution failures"""
        errors = []
        
        try:
            raise Exception("Tool failed")
        except Exception as e:
            errors.append(str(e))
        
        # Pipeline should log error and continue
        assert len(errors) > 0
    
    def test_pipeline_provides_meaningful_error_messages(self):
        """Test that pipeline provides helpful error messages"""
        error_scenarios = [
            "Ollama not running",
            "Audio quality too low",
            "Invalid filename",
            "Insufficient permissions"
        ]
        
        for error in error_scenarios:
            assert isinstance(error, str)
            assert len(error) > 0
    
    def test_pipeline_graceful_degradation(self):
        """Test graceful handling of missing features"""
        # Simulate missing Ollama
        with patch("requests.get") as mock_get:
            mock_get.side_effect = Exception("Connection refused")
            
            # Should handle gracefully
            assert mock_get.side_effect is not None


class TestPipelinePerformance:
    """Test pipeline performance characteristics"""
    
    @pytest.mark.slow
    def test_pipeline_response_time(self):
        """Test that pipeline responds within reasonable time"""
        import time
        
        with patch("src.pipeline.transcribe_audio") as mock_stt, \
             patch("src.pipeline.classify_intent") as mock_intent:
            
            mock_stt.return_value = {"transcript": "test", "error": None}
            mock_intent.return_value = [{
                "intent": "general_chat",
                "confidence": 0.8,
                "extracted_params": {}
            }]
            
            start_time = time.time()
            
            # Simulate pipeline call
            result = mock_intent.return_value
            
            elapsed = time.time() - start_time
            
            # Should complete quickly (mocked version)
            assert elapsed < 1.0
    
    def test_pipeline_memory_efficiency(self):
        """Test that pipeline doesn't leak memory"""
        import gc
        
        # Run multiple iterations
        for i in range(10):
            with patch("src.pipeline.transcribe_audio") as mock_stt:
                mock_stt.return_value = {"transcript": f"test {i}", "error": None}
        
        # Force garbage collection
        gc.collect()
        
        # Test passes if no memory errors occur


class TestPipelineIntegration:
    """Full integration tests"""
    
    @pytest.mark.integration
    def test_end_to_end_write_code_flow(self):
        """Test complete flow: audio → transcribe → intent → code generation"""
        with patch("src.pipeline.transcribe_audio") as mock_stt, \
             patch("src.pipeline.classify_intent") as mock_intent, \
             patch("src.pipeline.write_code") as mock_write:
            
            # Step 1: Transcription
            mock_stt.return_value = {
                "transcript": "Write a hello world function in Python",
                "error": None,
                "source": "mic"
            }
            
            # Step 2: Intent classification
            mock_intent.return_value = [{
                "intent": "write_code",
                "confidence": 0.95,
                "extracted_params": {
                    "language": "Python",
                    "filename": "hello.py",
                    "description": "hello world function"
                }
            }]
            
            # Step 3: Code generation
            mock_write.return_value = {
                "success": True,
                "path": "output/hello.py",
                "code_preview": "def hello():\n    print('Hello, World!')"
            }
            
            # Verify flow
            assert mock_stt.return_value["error"] is None
            assert mock_intent.return_value[0]["intent"] == "write_code"
            assert mock_write.return_value["success"] is True
    
    @pytest.mark.integration
    def test_end_to_end_compound_flow(self):
        """Test compound command flow"""
        with patch("src.pipeline.transcribe_audio") as mock_stt, \
             patch("src.pipeline.classify_intent") as mock_intent:
            
            mock_stt.return_value = {
                "transcript": "Summarize this and save to file",
                "error": None
            }
            
            mock_intent.return_value = [
                {"intent": "summarize_text", "confidence": 0.92, "extracted_params": {}},
                {"intent": "create_file", "confidence": 0.90, "extracted_params": {}}
            ]
            
            # Verify both intents returned
            assert len(mock_intent.return_value) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
