"""
Intent Recognition Module

Processes text input to identify user intent and extract relevant parameters.
Uses local Ollama with Mistral model for intent classification.
"""

import json
import requests


def classify_intent(transcript):
    """
    Classify a transcript into one or more intents using local Ollama Mistral model.
    Detects compound commands (e.g., "summarize this and save it to summary.txt")
    and returns multiple intents in sequence.
    
    Args:
        transcript: str - The text to classify
    
    Returns:
        list: [
            {
                "intent": "create_file" | "write_code" | "summarize_text" | "general_chat",
                "confidence": float (0.0 to 1.0),
                "extracted_params": dict
            }
        ]
    """
    intents = []
    
    try:
        # Ollama API endpoint
        ollama_url = "http://localhost:11434/api/generate"
        
        system_prompt = """You are an intent classifier for a voice-controlled AI agent.
You can classify transcripts that contain MULTIPLE intents (compound commands).

Intent types:
1. "create_file" - User wants to create or write a file
2. "write_code" - User wants to write, generate, or create code
3. "summarize_text" - User wants to summarize text
4. "general_chat" - User wants to chat or ask questions

Examples of compound commands:
- "write a Python function and save it to reverse.py" → [write_code, create_file]
- "summarize this and save to summary.txt" → [summarize_text, create_file]
- "create a file and tell me about it" → [create_file, general_chat]

For "write_code" intent, extract:
- language: The programming language (Java, Python, JavaScript, C++, etc.)
- filename: The output filename if mentioned
- description: What the code should do

For "create_file" intent, extract:
- filename: The filename to create
- content: The content to write

For "summarize_text" intent, extract:
- content: The text to summarize

For "general_chat" intent, extract nothing or a message.

Respond ONLY with a JSON array of intents (no markdown):
[
  {
    "intent": "write_code" | "create_file" | "summarize_text" | "general_chat",
    "confidence": 0.0-1.0,
    "extracted_params": {
      "language": "if write_code",
      "filename": "if mentioned",
      "description": "what to do",
      "content": "if summarize or create_file"
    }
  }
]

If only one intent is detected, still return an array with one element."""

        prompt = f"""{system_prompt}

User: {transcript}"""

        # Call Ollama API
        response = requests.post(
            ollama_url,
            json={
                "model": "mistral",
                "prompt": prompt,
                "stream": False,
                "temperature": 0.3
            },
            timeout=120
        )
        
        if response.status_code != 200:
            # Fallback to single general_chat intent on error
            return [{
                "intent": "general_chat",
                "confidence": 0.5,
                "extracted_params": {"error": f"Ollama error: {response.status_code}"}
            }]
        
        response_text = response.json().get("response", "").strip()
        
        # Extract JSON from response
        try:
            # Try to find JSON array in response
            start = response_text.find('[')
            end = response_text.rfind(']') + 1
            if start >= 0 and end > start:
                json_str = response_text[start:end]
                parsed = json.loads(json_str)
            else:
                parsed = json.loads(response_text)
            
            # Ensure parsed is a list
            if not isinstance(parsed, list):
                parsed = [parsed]
            
            # Validate and normalize each intent
            valid_intents = {"create_file", "write_code", "summarize_text", "general_chat"}
            for item in parsed:
                if isinstance(item, dict) and item.get("intent") in valid_intents:
                    intents.append({
                        "intent": item.get("intent"),
                        "confidence": float(item.get("confidence", 0.5)),
                        "extracted_params": item.get("extracted_params", {})
                    })
            
            # If no valid intents extracted, fallback to general_chat
            if not intents:
                intents = [{
                    "intent": "general_chat",
                    "confidence": 0.5,
                    "extracted_params": {"message": transcript}
                }]
        
        except json.JSONDecodeError:
            intents = [{
                "intent": "general_chat",
                "confidence": 0.3,
                "extracted_params": {"message": transcript, "error": "Could not parse model response"}
            }]
    
    except requests.exceptions.ConnectionError:
        intents = [{
            "intent": "general_chat",
            "confidence": 0.0,
            "extracted_params": {"error": "Cannot connect to Ollama. Is 'ollama serve' running on localhost:11434?"}
        }]
    except requests.exceptions.Timeout:
        intents = [{
            "intent": "general_chat",
            "confidence": 0.0,
            "extracted_params": {"error": "Ollama timeout. Model may be loading. Try again."}
        }]
    except Exception as e:
        intents = [{
            "intent": "general_chat",
            "confidence": 0.0,
            "extracted_params": {"error": f"Error: {str(e)}"}
        }]
    
    return intents


if __name__ == "__main__":
    # Test with 3 sample transcript strings
    test_transcripts = [
        "Create a file called notes.txt with the content hello world",
        "Write a Python function that calculates the fibonacci sequence",
        "Summarize this passage: The quick brown fox jumps over the lazy dog"
    ]
    
    print("Testing Intent Classification Module\n" + "="*50)
    
    for i, transcript in enumerate(test_transcripts, 1):
        print(f"\nTest {i}:")
        print(f"Transcript: {transcript}")
        result = classify_intent(transcript)
        print(f"Intent: {result['intent']}")
        print(f"Confidence: {result['confidence']:.2f}")
        print(f"Params: {json.dumps(result['extracted_params'], indent=2)}")
        print("-" * 50)
