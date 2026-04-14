"""
Pipeline Module

Orchestrates the complete workflow from audio input to tool execution and response.
Supports compound commands (multiple intents) and session history for continuous conversations.
"""

from src.stt import transcribe_audio
from src.intent import classify_intent
from src.tools import create_file, write_code, summarize_text, general_chat

# Global session history for continuous conversations
session_history = []


def run_pipeline(audio_input):
    """
    Run the complete voice agent pipeline.
    
    Accepts audio input (bytes or file path) or a text transcript for testing.
    Orchestrates transcription → intent classification → tool execution.
    Supports compound commands (multiple sequential intents).
    
    Args:
        audio_input: bytes (microphone), str file path (.wav/.mp3), 
                    or str transcript (for testing)
    
    Returns:
        dict: {
            "transcript": str,
            "intents": [list of intent strings],
            "intents_full": [list of full intent objects],
            "intent": str (first/primary intent for backward compat),
            "confidence": float (primary intent confidence),
            "actions_taken": [list of actions],
            "results": [list of tool results],
            "error": str | None
        }
    """
    result = {
        "transcript": "",
        "intents": [],
        "intents_full": [],
        "intent": "",  # For backward compatibility
        "confidence": 0.0,  # For backward compatibility
        "actions_taken": [],
        "results": [],
        "action_taken": "",  # For backward compatibility
        "result": {},  # For backward compatibility
        "error": None
    }
    
    try:
        # Step 1: Get transcript (from audio or use string directly for testing)
        print(f"DEBUG PIPELINE: Received input type: {type(audio_input).__name__}")
        stt_result = None  # Initialize for error checking later
        
        if isinstance(audio_input, bytes):
            print(f"DEBUG PIPELINE: Processing bytes input, size: {len(audio_input)}")
            # Microphone input (raw bytes)
            stt_result = transcribe_audio(audio_input, source_type="mic")
            if stt_result["error"] == "audio_unclear":
                result["error"] = "🔊 Audio was too quiet or unclear. Please speak clearly and try again."
                return result
            elif stt_result["error"]:
                result["error"] = stt_result["error"]
                return result
            transcript = stt_result["transcript"]
        
        elif isinstance(audio_input, tuple) and len(audio_input) == 2:
            # Gradio/Streamlit mic recording: (sample_rate, numpy_array)
            stt_result = transcribe_audio(audio_input, source_type="mic")
            if stt_result["error"] == "audio_unclear":
                result["error"] = "🔊 Audio was too quiet or unclear. Please speak clearly and try again."
                return result
            elif stt_result["error"]:
                result["error"] = stt_result["error"]
                return result
            transcript = stt_result["transcript"]
        
        elif isinstance(audio_input, str):
            # Check if it's a file path or a transcript string
            if audio_input.endswith(('.wav', '.mp3', '.m4a', '.flac', '.ogg')):
                # It's a file path
                stt_result = transcribe_audio(audio_input, source_type="file")
                if stt_result["error"] == "audio_unclear":
                    result["error"] = "🔊 Audio was too quiet or unclear. Please speak clearly and try again."
                    return result
                elif stt_result["error"]:
                    result["error"] = stt_result["error"]
                    return result
                transcript = stt_result["transcript"]
            else:
                # Treat as transcript string (for testing)
                transcript = audio_input
        else:
            result["error"] = "Invalid audio_input type. Must be bytes, tuple(sample_rate, audio_data), or str."
            return result
        
        result["transcript"] = transcript
        
        # Step 2: Classify intents (now returns list for compound commands)
        intents_array = classify_intent(transcript)
        
        # Store full intent objects
        result["intents_full"] = intents_array
        result["intents"] = [intent_obj["intent"] for intent_obj in intents_array]
        
        # For backward compatibility, set primary intent
        if intents_array:
            result["intent"] = intents_array[0]["intent"]
            result["confidence"] = intents_array[0]["confidence"]
        
        # Step 3: Execute each intent in sequence
        for intent_obj in intents_array:
            intent_type = intent_obj["intent"]
            params = intent_obj["extracted_params"]
            
            tool_result = None
            
            if intent_type == "create_file":
                tool_result = create_file(
                    filename=params.get("filename", "output.txt"),
                    content=params.get("content", "")
                )
                result["actions_taken"].append("create_file")
            
            elif intent_type == "write_code":
                language = params.get("language", "Python")
                
                # Auto-generate filename if not provided
                if not params.get("filename"):
                    lang_extensions = {
                        "Python": ".py",
                        "Java": ".java",
                        "JavaScript": ".js",
                        "C++": ".cpp",
                        "C#": ".cs",
                        "Ruby": ".rb",
                        "PHP": ".php",
                        "Go": ".go",
                        "Rust": ".rs",
                        "TypeScript": ".ts"
                    }
                    ext = lang_extensions.get(language, ".txt")
                    params["filename"] = f"generated{ext}"
                
                tool_result = write_code(
                    filename=params.get("filename", "generated.py"),
                    language=language,
                    description=params.get("description", "")
                )
                result["actions_taken"].append("write_code")
            
            elif intent_type == "summarize_text":
                tool_result = summarize_text(
                    text=params.get("content", "")
                )
                result["actions_taken"].append("summarize_text")
            
            elif intent_type == "general_chat":
                # Pass session history context to general_chat
                context = ""
                if session_history:
                    # Include last 3 exchanges as context
                    context = "\n".join([
                        f"User: {h['user']}\nAssistant: {h['response']}"
                        for h in session_history[-3:]
                    ])
                
                tool_result = general_chat(
                    message=params.get("message", transcript),
                    context=context
                )
                result["actions_taken"].append("general_chat")
            
            if tool_result:
                result["results"].append(tool_result)
        
        # For backward compatibility, set action_taken and result to first action/result
        if result["actions_taken"]:
            result["action_taken"] = result["actions_taken"][0]
        if result["results"]:
            result["result"] = result["results"][0]
        
        # Add to session history
        if result["intent"] and result["transcript"]:
            session_history.append({
                "user": result["transcript"],
                "response": str(result.get("result", {})),
                "intent": result["intent"]
            })
            # Keep only last 20 exchanges
            if len(session_history) > 20:
                session_history.pop(0)
    
    except Exception as e:
        result["error"] = f"Pipeline error: {str(e)}"
    
    return result


if __name__ == "__main__":
    print("Testing Pipeline Module (End-to-End with Compound Commands)\n" + "="*60)
    
    # Test 1: Write code request
    print("\nTest 1: Write Python Code")
    print("-" * 60)
    test_transcript_1 = "Write a Python function that calculates the fibonacci sequence"
    result_1 = run_pipeline(test_transcript_1)
    
    print(f"Transcript:     {result_1['transcript']}")
    print(f"Intents:        {result_1['intents']}")
    print(f"Primary Intent: {result_1['intent']}")
    print(f"Confidence:     {result_1['confidence']:.2f}")
    print(f"Actions Taken:  {result_1['actions_taken']}")
    print(f"Num Results:    {len(result_1['results'])}")
    print(f"Error:          {result_1['error']}")
    
    # Test 2: Compound command (summarize and save)
    print("\n\nTest 2: Compound Command (Summarize & Save)")
    print("-" * 60)
    test_transcript_2 = "Summarize this text and save it to summary.txt: Machine learning is a subset of artificial intelligence. It focuses on training algorithms to learn patterns from data automatically."
    result_2 = run_pipeline(test_transcript_2)
    
    print(f"Transcript:     {result_2['transcript'][:80]}...")
    print(f"Intents:        {result_2['intents']}")
    print(f"Primary Intent: {result_2['intent']}")
    print(f"Confidence:     {result_2['confidence']:.2f}")
    print(f"Actions Taken:  {result_2['actions_taken']}")
    print(f"Num Results:    {len(result_2['results'])}")
    print(f"Error:          {result_2['error']}")
    
    # Test 3: General chat with session context
    print("\n\nTest 3: General Chat (with Session Context)")
    print("-" * 60)
    test_transcript_3 = "What is machine learning?"
    result_3 = run_pipeline(test_transcript_3)
    
    print(f"Transcript:     {result_3['transcript']}")
    print(f"Intents:        {result_3['intents']}")
    print(f"Primary Intent: {result_3['intent']}")
    print(f"Confidence:     {result_3['confidence']:.2f}")
    print(f"Actions Taken:  {result_3['actions_taken']}")
    print(f"Num Results:    {len(result_3['results'])}")
    print(f"Session Hist Len: {len(session_history)}")
    print(f"Error:          {result_3['error']}")
