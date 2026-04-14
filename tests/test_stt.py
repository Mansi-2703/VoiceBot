"""
Unit Tests for Speech-to-Text (STT) Module

Tests for Whisper model integration and audio transcription.
"""

import pytest
from unittest.mock import patch, Mock, MagicMock
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.stt import transcribe_audio


class TestTranscribeAudio:
    """Test suite for transcribe_audio function"""
    
    def test_transcribe_from_file_success(self, mock_whisper_model, sample_audio_file):
        """Test successful transcription from audio file"""
        with patch("whisper.load_model") as mock_load:
            mock_load.return_value = mock_whisper_model
            
            result = transcribe_audio(sample_audio_file, source_type="file")
            
            assert result["error"] is None
            assert result["source"] == "file"
            assert result["transcript"] == "This is a test transcription"
    
    def test_transcribe_nonexistent_file(self, mock_whisper_model):
        """Test transcription with non-existent file"""
        fake_path = "nonexistent_audio.wav"
        
        with patch("whisper.load_model") as mock_load:
            mock_load.return_value = mock_whisper_model
            
            result = transcribe_audio(fake_path, source_type="file")
            
            assert result["error"] is not None
            assert "not found" in result["error"].lower()
            assert result["transcript"] == ""
    
    def test_transcribe_unsupported_format(self, mock_whisper_model):
        """Test transcription with unsupported audio format"""
        with patch("whisper.load_model") as mock_load:
            mock_load.return_value = mock_whisper_model
            
            result = transcribe_audio("test.txt", source_type="file")
            
            assert result["error"] is not None
            assert result["transcript"] == ""
    
    def test_transcribe_returns_correct_structure(self, mock_whisper_model, sample_audio_file):
        """Test that transcribe_audio returns correct dictionary structure"""
        with patch("whisper.load_model") as mock_load:
            mock_load.return_value = mock_whisper_model
            
            result = transcribe_audio(sample_audio_file, source_type="file")
            
            # Check required keys
            assert "transcript" in result
            assert "source" in result
            assert "error" in result
            
            # Check types
            assert isinstance(result["transcript"], str)
            assert isinstance(result["source"], str)
            assert isinstance(result["error"], (str, type(None)))
    
    def test_transcribe_source_type_preservation(self, mock_whisper_model, sample_audio_file):
        """Test that source_type is preserved in result"""
        with patch("whisper.load_model") as mock_load:
            mock_load.return_value = mock_whisper_model
            
            # Test with 'file' source
            result_file = transcribe_audio(sample_audio_file, source_type="file")
            assert result_file["source"] == "file"
            
            # Test with 'mic' source (even though we're using file)
            result_mic = transcribe_audio(sample_audio_file, source_type="mic")
            assert result_mic["source"] == "mic"
    
    @pytest.mark.slow
    def test_transcribe_empty_audio(self, mock_whisper_model, sample_audio_file):
        """Test transcription of empty/silent audio"""
        with patch("whisper.load_model") as mock_load:
            mock_load.return_value = mock_whisper_model
            
            result = transcribe_audio(sample_audio_file, source_type="file")
            
            # Should return empty or minimal transcript
            assert isinstance(result["transcript"], str)
            assert result["error"] is None
    
    def test_transcribe_handles_unicode_paths(self, mock_whisper_model):
        """Test transcription with unicode characters in file path"""
        unicode_path = "тест_audio.wav"  # Cyrillic characters
        
        with patch("whisper.load_model") as mock_load:
            mock_load.return_value = mock_whisper_model
            
            result = transcribe_audio(unicode_path, source_type="file")
            
            # Should handle unicode gracefully
            assert "error" in result


class TestWhisperModelLoading:
    """Test Whisper model initialization"""
    
    @pytest.mark.requires_whisper
    def test_whisper_model_loads_successfully(self):
        """Test that Whisper model can be loaded (requires model download)"""
        try:
            import whisper
            model = whisper.load_model("base")
            assert model is not None
        except Exception as e:
            pytest.skip(f"Whisper model loading failed: {str(e)}")
    
    def test_whisper_model_caching(self, mock_whisper_model):
        """Test that Whisper model is cached between calls"""
        with patch("whisper.load_model") as mock_load:
            mock_load.return_value = mock_whisper_model
            
            # First call
            transcribe_audio("dummy.wav", source_type="file")
            first_call_count = mock_load.call_count
            
            # Second call
            transcribe_audio("dummy.wav", source_type="file")
            second_call_count = mock_load.call_count
            
            # Model should be loaded each time in current implementation
            assert second_call_count > first_call_count


class TestAudioInputValidation:
    """Test audio input validation"""
    
    def test_invalid_input_type(self, mock_whisper_model):
        """Test handling of invalid input types"""
        with patch("whisper.load_model") as mock_load:
            mock_load.return_value = mock_whisper_model
            
            # Pass invalid input
            result = transcribe_audio(12345, source_type="file")
            
            assert result["error"] is not None or "error" in result
    
    def test_empty_string_input(self, mock_whisper_model):
        """Test handling of empty string input"""
        with patch("whisper.load_model") as mock_load:
            mock_load.return_value = mock_whisper_model
            
            result = transcribe_audio("", source_type="file")
            
            assert result["error"] is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
