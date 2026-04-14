"""
Tools Module

Defines and executes tools/functions based on identified intents.
Includes file creation, code generation, text summarization, and conversation using local Ollama.
"""

import os
import json
from pathlib import Path
import requests


def create_file(filename: str, content: str) -> dict:
    """
    Create a file in the output/ folder.
    
    Args:
        filename: Name of the file to create
        content: Content to write to the file
    
    Returns:
        dict: {
            "action": "create_file",
            "path": str,
            "success": bool,
            "message": str
        }
    """
    result = {
        "action": "create_file",
        "path": "",
        "success": False,
        "message": ""
    }
    
    try:
        # Prevent directory traversal attacks
        if ".." in filename or filename.startswith("/"):
            result["message"] = "Invalid filename: cannot use '..' or absolute paths"
            return result
        
        # Create output directory if it doesn't exist
        output_dir = Path("output")
        output_dir.mkdir(exist_ok=True)
        
        # Build full path
        file_path = output_dir / filename
        
        # Ensure the file is within output/
        if not str(file_path.resolve()).startswith(str(output_dir.resolve())):
            result["message"] = "Error: file path must be within output/ folder"
            return result
        
        # Write file
        file_path.write_text(content, encoding="utf-8")
        
        result["path"] = str(file_path)
        result["success"] = True
        result["message"] = f"File created successfully at {file_path}"
        
    except Exception as e:
        result["message"] = f"Error creating file: {str(e)}"
    
    return result


def write_code(filename: str, language: str, description: str) -> dict:
    """
    Generate code using local Ollama Mistral model and save to output/ folder.
    
    Args:
        filename: Name of the output file
        language: Programming language
        description: What the code should do
    
    Returns:
        dict: {
            "action": "write_code",
            "path": str,
            "code_preview": str,
            "success": bool
        }
    """
    result = {
        "action": "write_code",
        "path": "",
        "code_preview": "",
        "success": False
    }
    
    try:
        ollama_url = "http://localhost:11434/api/generate"
        
        prompt = f"""Generate {language} code for: {description}

Provide ONLY the code, no explanations."""
        
        response = requests.post(
            ollama_url,
            json={
                "model": "mistral",
                "prompt": prompt,
                "stream": False,
                "temperature": 0.2
            },
            timeout=120  # Increased timeout for slower systems
        )
        
        if response.status_code != 200:
            result["code_preview"] = f"Ollama error: {response.status_code}"
            return result
        
        generated_code = response.json().get("response", "").strip()
        
        # Save to file
        file_result = create_file(filename, generated_code)
        
        if file_result["success"]:
            result["path"] = file_result["path"]
            result["code_preview"] = generated_code[:200]
            result["success"] = True
        else:
            result["code_preview"] = file_result["message"]
        
    except requests.exceptions.ConnectionError:
        result["code_preview"] = "Cannot connect to Ollama. Is 'ollama serve' running?"
    except Exception as e:
        result["code_preview"] = f"Error generating code: {str(e)}"
    
    return result


def summarize_text(text: str) -> dict:
    """
    Summarize text using local Ollama Mistral model.
    
    Args:
        text: Text to summarize
    
    Returns:
        dict: {
            "action": "summarize_text",
            "summary": str,
            "success": bool
        }
    """
    result = {
        "action": "summarize_text",
        "summary": "",
        "success": False
    }
    
    try:
        ollama_url = "http://localhost:11434/api/generate"
        
        prompt = f"""Summarize this text concisely:

{text}"""
        
        response = requests.post(
            ollama_url,
            json={
                "model": "mistral",
                "prompt": prompt,
                "stream": False,
                "temperature": 0.3
            },
            timeout=120  # Increased timeout for slower systems
        )
        
        if response.status_code != 200:
            result["summary"] = f"Ollama error: {response.status_code}"
            return result
        
        result["summary"] = response.json().get("response", "").strip()
        result["success"] = True
        
    except requests.exceptions.ConnectionError:
        result["summary"] = "Cannot connect to Ollama. Is 'ollama serve' running?"
    except Exception as e:
        result["summary"] = f"Error summarizing text: {str(e)}"
    
    return result


def general_chat(message: str, context: str = "") -> dict:
    """
    Have a conversational response using local Ollama Mistral model.
    Supports session context for continuous conversations.
    
    Args:
        message: User message or query
        context: Optional session context (previous exchanges) for continuity
    
    Returns:
        dict: {
            "action": "general_chat",
            "response": str,
            "success": bool
        }
    """
    result = {
        "action": "general_chat",
        "response": "",
        "success": False
    }
    
    try:
        ollama_url = "http://localhost:11434/api/generate"
        
        # Build prompt with context if provided
        if context:
            prompt = f"""Previous conversation:
{context}

User: {message}"""
        else:
            prompt = message
        
        response = requests.post(
            ollama_url,
            json={
                "model": "mistral",
                "prompt": prompt,
                "stream": False,
                "temperature": 0.7
            },
            timeout=120  # Increased timeout for slower systems
        )
        
        if response.status_code != 200:
            result["response"] = f"Ollama error: {response.status_code}"
            return result
        
        result["response"] = response.json().get("response", "").strip()
        result["success"] = True
        
    except requests.exceptions.ConnectionError:
        result["response"] = "Cannot connect to Ollama. Is 'ollama serve' running?"
    except Exception as e:
        result["response"] = f"Error in chat: {str(e)}"
    
    return result


if __name__ == "__main__":
    print("Testing Tools Module\n" + "="*50)
    
    # Test 1: create_file
    print("\nTest 1: create_file()")
    result1 = create_file("test_notes.txt", "This is a test file created by the tools module.")
    print(f"  Success: {result1['success']}")
    print(f"  Path: {result1['path']}")
    print(f"  Message: {result1['message']}")
    
    # Test 2: write_code
    print("\nTest 2: write_code()")
    result2 = write_code("test_function.py", "Python", "A function that returns the sum of two numbers")
    print(f"  Success: {result2['success']}")
    print(f"  Path: {result2['path']}")
    print(f"  Preview: {result2['code_preview']}")
    
    # Test 3: summarize_text
    print("\nTest 3: summarize_text()")
    result3 = summarize_text("Machine learning is a subset of artificial intelligence that focuses on training algorithms to learn patterns from data. It has applications in image recognition, natural language processing, and predictive analytics.")
    print(f"  Success: {result3['success']}")
    print(f"  Summary: {result3['summary'][:100]}...")
    
    # Test 4: general_chat
    print("\nTest 4: general_chat()")
    result4 = general_chat("What is Python used for?")
    print(f"  Success: {result4['success']}")
    print(f"  Response: {result4['response'][:100]}...")
