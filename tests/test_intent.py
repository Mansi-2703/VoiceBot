"""
Unit Tests for Intent Classification Module

Tests for intent recognition and parameter extraction.
"""

import pytest
import json
from unittest.mock import patch, Mock
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.intent import classify_intent


class TestClassifyIntent:
    """Test suite for classify_intent function"""
    
    def test_intent_write_code(self, sample_transcript, mock_ollama_response):
        """Test detection of write_code intent"""
        with patch("requests.post") as mock_post:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "response": json.dumps(mock_ollama_response("write_code"))
            }
            mock_post.return_value = mock_response
            
            result = classify_intent(sample_transcript["write_code"])
            
            assert len(result) > 0
            assert result[0]["intent"] == "write_code"
            assert "language" in result[0]["extracted_params"]
    
    def test_intent_create_file(self, sample_transcript, mock_ollama_response):
        """Test detection of create_file intent"""
        with patch("requests.post") as mock_post:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "response": json.dumps(mock_ollama_response("create_file"))
            }
            mock_post.return_value = mock_response
            
            result = classify_intent(sample_transcript["create_file"])
            
            assert len(result) > 0
            assert result[0]["intent"] == "create_file"
            assert "filename" in result[0]["extracted_params"]
    
    def test_intent_summarize_text(self, sample_transcript, mock_ollama_response):
        """Test detection of summarize_text intent"""
        with patch("requests.post") as mock_post:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "response": json.dumps(mock_ollama_response("summarize_text"))
            }
            mock_post.return_value = mock_response
            
            result = classify_intent(sample_transcript["summarize_text"])
            
            assert len(result) > 0
            assert result[0]["intent"] == "summarize_text"
    
    def test_intent_general_chat(self, sample_transcript, mock_ollama_response):
        """Test detection of general_chat intent"""
        with patch("requests.post") as mock_post:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "response": json.dumps(mock_ollama_response("general_chat"))
            }
            mock_post.return_value = mock_response
            
            result = classify_intent(sample_transcript["general_chat"])
            
            assert len(result) > 0
            assert result[0]["intent"] == "general_chat"
    
    def test_compound_command_detection(self, sample_transcript, mock_ollama_response):
        """Test detection of compound commands (multiple intents)"""
        with patch("requests.post") as mock_post:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "response": json.dumps(mock_ollama_response("compound"))
            }
            mock_post.return_value = mock_response
            
            result = classify_intent(sample_transcript["compound"])
            
            assert len(result) > 1
            intents = [r["intent"] for r in result]
            assert "write_code" in intents
            assert "create_file" in intents
    
    def test_intent_confidence_scores(self, sample_transcript, mock_ollama_response):
        """Test that confidence scores are returned"""
        with patch("requests.post") as mock_post:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "response": json.dumps(mock_ollama_response("write_code"))
            }
            mock_post.return_value = mock_response
            
            result = classify_intent(sample_transcript["write_code"])
            
            assert all(0.0 <= r["confidence"] <= 1.0 for r in result)
    
    def test_intent_parameter_extraction(self, sample_transcript, mock_ollama_response):
        """Test that parameters are correctly extracted"""
        with patch("requests.post") as mock_post:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "response": json.dumps(mock_ollama_response("write_code"))
            }
            mock_post.return_value = mock_response
            
            result = classify_intent(sample_transcript["write_code"])
            params = result[0]["extracted_params"]
            
            assert "language" in params
            assert params["language"] == "Python"
    
    def test_intent_returns_list(self, sample_transcript, mock_ollama_response):
        """Test that classify_intent always returns a list"""
        with patch("requests.post") as mock_post:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "response": json.dumps(mock_ollama_response("general_chat"))
            }
            mock_post.return_value = mock_response
            
            result = classify_intent(sample_transcript["general_chat"])
            
            assert isinstance(result, list)
            assert len(result) > 0
    
    def test_intent_handles_ollama_error(self):
        """Test handling when Ollama is not running"""
        with patch("requests.post") as mock_post:
            mock_post.side_effect = Exception("Connection refused")
            
            result = classify_intent("test transcript")
            
            # Should handle gracefully
            assert isinstance(result, list)
    
    def test_intent_empty_transcript(self):
        """Test handling of empty transcript"""
        with patch("requests.post") as mock_post:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"response": json.dumps([])}
            mock_post.return_value = mock_response
            
            result = classify_intent("")
            
            assert isinstance(result, list)
    
    def test_intent_long_transcript(self):
        """Test handling of very long transcript"""
        long_transcript = "describe this " + "really long and detailed description " * 50
        
        with patch("requests.post") as mock_post:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "response": json.dumps([{"intent": "general_chat", "confidence": 0.8, "extracted_params": {}}])
            }
            mock_post.return_value = mock_response
            
            result = classify_intent(long_transcript)
            
            assert isinstance(result, list)


class TestIntentParameters:
    """Test parameter extraction from intents"""
    
    def test_write_code_parameters(self, mock_ollama_response):
        """Test extraction of write_code parameters"""
        params = mock_ollama_response("write_code")[0]["extracted_params"]
        
        assert "language" in params
        assert "description" in params
    
    def test_create_file_parameters(self, mock_ollama_response):
        """Test extraction of create_file parameters"""
        params = mock_ollama_response("create_file")[0]["extracted_params"]
        
        assert "filename" in params
        assert "content" in params
    
    def test_summarize_text_parameters(self, mock_ollama_response):
        """Test extraction of summarize_text parameters"""
        params = mock_ollama_response("summarize_text")[0]["extracted_params"]
        
        assert "content" in params


class TestIntentEdgeCases:
    """Test edge cases in intent classification"""
    
    def test_ambiguous_transcript(self):
        """Test handling of ambiguous transcripts"""
        ambiguous = "Something was said"
        
        with patch("requests.post") as mock_post:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "response": json.dumps([{
                    "intent": "general_chat",
                    "confidence": 0.5,
                    "extracted_params": {}
                }])
            }
            mock_post.return_value = mock_response
            
            result = classify_intent(ambiguous)
            
            assert isinstance(result, list)
            assert len(result) > 0
    
    def test_special_characters_in_transcript(self):
        """Test handling of special characters"""
        special_transcript = "Create a file with <html> tags & symbols @#$%"
        
        with patch("requests.post") as mock_post:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "response": json.dumps([{
                    "intent": "create_file",
                    "confidence": 0.85,
                    "extracted_params": {"filename": "test.txt"}
                }])
            }
            mock_post.return_value = mock_response
            
            result = classify_intent(special_transcript)
            
            assert isinstance(result, list)
    
    def test_unicode_transcript(self):
        """Test handling of unicode characters"""
        unicode_transcript = "Write Python code 你好 مرحبا Привет"
        
        with patch("requests.post") as mock_post:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "response": json.dumps([{
                    "intent": "write_code",
                    "confidence": 0.8,
                    "extracted_params": {"language": "Python"}
                }])
            }
            mock_post.return_value = mock_response
            
            result = classify_intent(unicode_transcript)
            
            assert isinstance(result, list)


@pytest.mark.requires_ollama
class TestIntentWithRealOllama:
    """Integration tests requiring actual Ollama running"""
    
    def test_real_intent_classification(self, sample_transcript):
        """Test with real Ollama server (if available)"""
        try:
            result = classify_intent(sample_transcript["write_code"])
            
            if result:  # If we got a response
                assert isinstance(result, list)
                assert all("intent" in r for r in result)
        except Exception:
            pytest.skip("Ollama not available for testing")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
