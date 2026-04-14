"""
Unit Tests for Tools Module

Tests for file creation, code generation, and text processing tools.
"""

import pytest
from pathlib import Path
from unittest.mock import patch, Mock
import tempfile
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.tools import create_file, write_code, summarize_text, general_chat


class TestCreateFile:
    """Test suite for create_file function"""
    
    def test_create_file_success(self, cleanup_output_files):
        """Test successful file creation"""
        filename = "test_file.txt"
        content = "Hello, World!"
        
        result = create_file(filename, content)
        cleanup_output_files("output/" + filename)
        
        assert result["success"] is True
        assert result["action"] == "create_file"
        assert "output" in result["path"]
        assert filename in result["path"]
    
    def test_create_file_content_preserved(self, cleanup_output_files):
        """Test that file content is correctly written"""
        filename = "test_content.txt"
        content = "This is test content\nWith multiple lines"
        
        result = create_file(filename, content)
        cleanup_output_files("output/" + filename)
        
        assert result["success"] is True
        
        # Read file and verify content
        file_path = Path(result["path"])
        written_content = file_path.read_text()
        assert written_content == content
    
    def test_create_file_directory_traversal_prevention(self):
        """Test prevention of directory traversal attacks"""
        # Attempt to use ../
        result = create_file("../../../etc/passwd", "malicious")
        
        assert result["success"] is False
        assert "Error" in result["message"] or "Invalid" in result["message"]
    
    def test_create_file_prevents_absolute_paths(self):
        """Test prevention of absolute path injection"""
        result = create_file("/etc/passwd", "malicious")
        
        assert result["success"] is False
        assert "Invalid" in result["message"] or "Error" in result["message"]
    
    def test_create_file_special_characters(self, cleanup_output_files):
        """Test file creation with special characters in name"""
        filename = "test_file_2024.txt"
        content = "Special chars: @#$%^&*()"
        
        result = create_file(filename, content)
        cleanup_output_files("output/" + filename)
        
        assert result["success"] is True
    
    def test_create_file_unicode_content(self, cleanup_output_files):
        """Test file creation with unicode content"""
        filename = "unicode_test.txt"
        content = "Hello 你好 مرحبا Привет"
        
        result = create_file(filename, content)
        cleanup_output_files("output/" + filename)
        
        assert result["success"] is True
        
        # Verify unicode is preserved
        file_path = Path(result["path"])
        written_content = file_path.read_text(encoding="utf-8")
        assert "你好" in written_content
    
    def test_create_file_large_content(self, cleanup_output_files):
        """Test file creation with large content"""
        filename = "large_file.txt"
        content = "x" * (10 * 1024 * 1024)  # 10MB
        
        result = create_file(filename, content)
        cleanup_output_files("output/" + filename)
        
        assert result["success"] is True
    
    def test_create_file_empty_content(self, cleanup_output_files):
        """Test file creation with empty content"""
        filename = "empty.txt"
        content = ""
        
        result = create_file(filename, content)
        cleanup_output_files("output/" + filename)
        
        assert result["success"] is True
        
        file_path = Path(result["path"])
        written_content = file_path.read_text()
        assert written_content == ""
    
    def test_create_file_structure(self):
        """Test that create_file returns correct structure"""
        with patch("pathlib.Path.write_text"):
            result = create_file("test.txt", "content")
            
            assert "action" in result
            assert "path" in result
            assert "success" in result
            assert "message" in result


class TestWriteCode:
    """Test suite for write_code function"""
    
    def test_write_code_returns_correct_structure(self):
        """Test that write_code returns correct dictionary structure"""
        with patch("requests.post") as mock_post:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"response": "def hello():\n    return 'world'"}
            mock_post.return_value = mock_response
            
            with patch("src.tools.create_file") as mock_create:
                mock_create.return_value = {"success": True, "path": "test.py"}
                
                result = write_code("test.py", "Python", "hello world function")
                
                assert "action" in result
                assert result["action"] == "write_code"
                assert "path" in result
                assert "code_preview" in result
                assert "success" in result
    
    def test_write_code_with_different_languages(self):
        """Test write_code with multiple programming languages"""
        languages = ["Python", "JavaScript", "Java", "C++"]
        
        for lang in languages:
            with patch("requests.post") as mock_post:
                mock_response = Mock()
                mock_response.status_code = 200
                mock_response.json.return_value = {"response": f"// {lang} code"}
                mock_post.return_value = mock_response
                
                with patch("src.tools.create_file") as mock_create:
                    mock_create.return_value = {"success": True, "path": f"test.{lang.lower()}"}
                    
                    result = write_code(f"test.{lang.lower()}", lang, "test function")
                    
                    assert result["action"] == "write_code"
    
    def test_write_code_handles_ollama_error(self):
        """Test write_code handling when Ollama returns error"""
        with patch("requests.post") as mock_post:
            mock_response = Mock()
            mock_response.status_code = 500
            mock_post.return_value = mock_response
            
            result = write_code("test.py", "Python", "test")
            
            assert result["success"] is False or "error" in result["code_preview"].lower()
    
    def test_write_code_connection_error(self):
        """Test write_code handling when Ollama is not running"""
        with patch("requests.post") as mock_post:
            mock_post.side_effect = Exception("Connection refused")
            
            result = write_code("test.py", "Python", "test")
            
            assert result["success"] is False or "error" in result.get("code_preview", "").lower()


class TestSummarizeText:
    """Test suite for summarize_text function"""
    
    def test_summarize_text_returns_correct_structure(self):
        """Test that summarize_text returns correct structure"""
        with patch("requests.post") as mock_post:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"response": "This is a summary"}
            mock_post.return_value = mock_response
            
            result = summarize_text("Long text to summarize...")
            
            assert isinstance(result, dict)
            assert "action" in result
            assert result["action"] == "summarize_text"
            assert "summary" in result or "success" in result
    
    def test_summarize_short_text(self):
        """Test summarization of short text"""
        text = "Python is great"
        
        with patch("requests.post") as mock_post:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"response": "Python is good"}
            mock_post.return_value = mock_response
            
            result = summarize_text(text)
            
            assert isinstance(result, dict)
    
    def test_summarize_long_text(self):
        """Test summarization of long text"""
        text = "Machine learning is a subset of artificial intelligence. " * 50
        
        with patch("requests.post") as mock_post:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"response": "ML is a subset of AI"}
            mock_post.return_value = mock_response
            
            result = summarize_text(text)
            
            assert isinstance(result, dict)
    
    def test_summarize_empty_text(self):
        """Test summarization of empty text"""
        with patch("requests.post") as mock_post:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"response": ""}
            mock_post.return_value = mock_response
            
            result = summarize_text("")
            
            assert isinstance(result, dict)
    
    def test_summarize_text_handles_ollama_error(self):
        """Test summarize_text error handling"""
        with patch("requests.post") as mock_post:
            mock_post.side_effect = Exception("Connection error")
            
            result = summarize_text("test text")
            
            assert isinstance(result, dict)


class TestGeneralChat:
    """Test suite for general_chat function"""
    
    def test_general_chat_returns_response(self):
        """Test that general_chat returns a response"""
        with patch("requests.post") as mock_post:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"response": "Chat response"}
            mock_post.return_value = mock_response
            
            result = general_chat("What is AI?", [])
            
            assert isinstance(result, dict)
    
    def test_general_chat_with_history(self):
        """Test general_chat with conversation history"""
        history = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"}
        ]
        
        with patch("requests.post") as mock_post:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"response": "How can I help?"}
            mock_post.return_value = mock_response
            
            result = general_chat("Tell me more", history)
            
            assert isinstance(result, dict)
    
    def test_general_chat_empty_history(self):
        """Test general_chat with empty history"""
        with patch("requests.post") as mock_post:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"response": "Hello"}
            mock_post.return_value = mock_response
            
            result = general_chat("Question", [])
            
            assert isinstance(result, dict)
    
    def test_general_chat_handles_ollama_error(self):
        """Test general_chat error handling"""
        with patch("requests.post") as mock_post:
            mock_post.side_effect = Exception("Ollama error")
            
            result = general_chat("test", [])
            
            assert isinstance(result, dict)


class TestToolSecurity:
    """Test security aspects of tools"""
    
    def test_path_traversal_protection(self):
        """Test comprehensive path traversal protection"""
        malicious_paths = [
            "../../../etc/passwd",
            "../../sensitive/file.txt",
            "..\\..\\windows\\system32",
            "/etc/shadow",
            "C:\\Windows\\System32\\config",
        ]
        
        for path in malicious_paths:
            result = create_file(path, "content")
            assert result["success"] is False
    
    def test_filename_validation(self):
        """Test filename validation"""
        with patch("pathlib.Path.write_text") as mock_write:
            # These should be rejected or handled safely
            result = create_file("..", "content")
            assert result["success"] is False or result["message"] is not None


class TestToolIntegration:
    """Integration tests for tools"""
    
    def test_tools_dont_interfere(self, cleanup_output_files):
        """Test that tools don't interfere with each other"""
        # Create a file
        result1 = create_file("test1.txt", "content1")
        cleanup_output_files("output/test1.txt")
        
        # Create another file
        result2 = create_file("test2.txt", "content2")
        cleanup_output_files("output/test2.txt")
        
        assert result1["success"] is True
        assert result2["success"] is True
        
        # Verify both exist independently
        if result1["success"] and result2["success"]:
            file1 = Path(result1["path"]).read_text()
            file2 = Path(result2["path"]).read_text()
            
            assert file1 == "content1"
            assert file2 == "content2"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
