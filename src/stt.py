"""
Speech-to-Text (STT) Module

Handles conversion of audio input to text using OpenAI's Whisper model locally.
Supports both microphone recordings and file-based audio inputs.
"""

import os
from pathlib import Path
import whisper


def transcribe_audio(audio_input, source_type="mic"):
    """
    Transcribe audio to text using local Whisper model.
    
    Args:
        audio_input: Either:
                    - File path string (for file input)
                    - Tuple of (sample_rate, numpy_array) from Gradio mic recording
                    - Audio bytes for microphone
        source_type: "mic" for microphone bytes or "file" for file path
    
    Returns:
        dict: {
            "transcript": str,
            "source": "mic" | "file",
            "error": str | None
        }
    """
    result = {
        "transcript": "",
        "source": source_type,
        "error": None
    }
    
    try:
        # Debug: Log input type
        print(f"DEBUG: audio_input type = {type(audio_input)}")
        if isinstance(audio_input, (bytes, str)):
            print(f"DEBUG: audio_input length/path = {len(audio_input) if isinstance(audio_input, bytes) else audio_input}")
        
        # Load Whisper model (base model, ~140MB)
        print("Loading Whisper model (base)...")
        model = whisper.load_model("base")
        
        # Handle different input types
        if isinstance(audio_input, tuple) and len(audio_input) == 2:
            # Gradio mic recording: (sample_rate, numpy_array)
            import numpy as np
            import scipy.io.wavfile as wavfile
            
            sample_rate, audio_data = audio_input
            print(f"DEBUG: Received Gradio audio - sample_rate={sample_rate}, data shape={audio_data.shape}")
            
            # Convert to 16-bit PCM
            if audio_data.dtype != np.int16:
                # Normalize to [-1, 1] if needed
                if np.max(np.abs(audio_data)) <= 1.0:
                    audio_data = (audio_data * 32767).astype(np.int16)
                else:
                    audio_data = audio_data.astype(np.int16)
            
            # Save to temp WAV file
            temp_file = "temp_mic_audio.wav"
            wavfile.write(temp_file, sample_rate, audio_data)
            
            # Transcribe
            transcription = model.transcribe(temp_file, language="en")
            result["transcript"] = transcription["text"]
            
            # Clean up
            os.remove(temp_file)
            result["source"] = "mic"
            
        elif isinstance(audio_input, str):
            # File path
            file_path = Path(audio_input)
            if not file_path.exists():
                result["error"] = f"Audio file not found: {audio_input}"
                return result
            
            # Validate file extension
            valid_extensions = {".wav", ".mp3", ".m4a", ".flac", ".ogg"}
            if file_path.suffix.lower() not in valid_extensions:
                result["error"] = f"Unsupported file format: {file_path.suffix}. Supported: {valid_extensions}"
                return result
            
            print(f"DEBUG: Transcribing file: {file_path}")
            # Transcribe
            transcription = model.transcribe(str(file_path), language="en")
            result["transcript"] = transcription["text"]
            result["source"] = "file"
        
        elif isinstance(audio_input, bytes):
            # Raw audio bytes (from microphone or Streamlit)
            print(f"DEBUG: Received raw bytes, length={len(audio_input)}")
            
            # Detect format from magic bytes
            is_mp3 = (
                audio_input[:4] == b'ID3\x02' or 
                audio_input[:3] == b'ID3' or
                (len(audio_input) > 2 and audio_input[0:2] == b'\xFF\xFB') or  # MPEG sync
                (len(audio_input) > 2 and audio_input[0:2] == b'\xFF\xFA')      # MPEG sync alternative
            )
            
            if is_mp3:
                # MP3 file - decode with librosa and resample to 16kHz for Whisper
                print("DEBUG: Detected MP3 format, decoding...")
                
                try:
                    import librosa
                    import numpy as np
                    from io import BytesIO
                    
                    print("DEBUG: Attempting MP3 decode with librosa...")
                    # Load MP3 at original sample rate, keep mono
                    audio_data, sr = librosa.load(BytesIO(audio_input), sr=None, mono=True)
                    
                    print(f"DEBUG: Librosa loaded - shape: {audio_data.shape}, sr: {sr}, dtype: {audio_data.dtype}")
                    
                    # Resample to 16kHz (Whisper's standard sample rate)
                    # This prevents hallucination issues with variable sample rates
                    if sr != 16000:
                        print(f"DEBUG: Resampling from {sr}Hz to 16000Hz...")
                        audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=16000)
                        sr = 16000
                        print(f"DEBUG: Resampled - new shape: {audio_data.shape}")
                    
                    # Ensure float32 in range [-1, 1]
                    if audio_data.dtype != np.float32:
                        audio_data = audio_data.astype(np.float32)
                    
                    # Clip to ensure values are in valid range
                    audio_data = np.clip(audio_data, -1.0, 1.0)
                    
                    print(f"DEBUG: Prepared audio - 16kHz, float32, range: [{audio_data.min():.3f}, {audio_data.max():.3f}]")
                    
                    # Pass numpy array directly to Whisper
                    transcription = model.transcribe(audio_data, language="en")
                    result["transcript"] = transcription["text"].strip()
                    print(f"DEBUG: Transcription successful: '{result['transcript'][:80]}'")
                    result["source"] = "mic"
                    return result
                    
                except ImportError as e:
                    print(f"DEBUG: librosa missing: {e}")
                    result["error"] = (
                        "MP3 support requires librosa library:\n"
                        "pip install librosa\n\n"
                        "Alternatively, convert your MP3 to WAV format using an online tool or VLC media player."
                    )
                    print(f"ERROR: {result['error']}")
                    return result
                    
                except Exception as e:
                    print(f"DEBUG: MP3 decode error: {e}")
                    result["error"] = f"Could not decode MP3: {str(e)}\n\nTry converting to WAV format instead."
                    print(f"ERROR: {result['error']}")
                    return result
            
            # Handle WAV and other formats with scipy
            from io import BytesIO
            import numpy as np
            from scipy.io import wavfile
            
            try:
                print(f"DEBUG: Attempting to read as WAV format...")
                # Read WAV data from bytes
                audio_bytesio = BytesIO(audio_input)
                sample_rate, audio_data = wavfile.read(audio_bytesio)
                
                # Convert to float32 if needed (Whisper expects float32)
                if audio_data.dtype != np.float32:
                    # Normalize to [-1, 1]
                    if audio_data.dtype == np.int16:
                        audio_data = audio_data.astype(np.float32) / 32768.0
                    elif audio_data.dtype == np.int32:
                        audio_data = audio_data.astype(np.float32) / 2147483648.0
                    else:
                        audio_data = audio_data.astype(np.float32)
                
                print(f"DEBUG: Audio array shape: {audio_data.shape}, dtype: {audio_data.dtype}")
                print(f"DEBUG: Starting Whisper with numpy array...")
                
                # Pass numpy array directly to Whisper (no file needed!)
                transcription = model.transcribe(audio_data, language="en")
                result["transcript"] = transcription["text"].strip()
                print(f"DEBUG: Transcription successful: '{result['transcript'][:80]}'")
                result["source"] = "mic"
                
            except Exception as error:
                result["error"] = f"Audio processing failed: {str(error)}. Please upload a WAV, MP3, M4A, FLAC, or OGG file."
                print(f"ERROR: {result['error']}")
        
        else:
            result["error"] = f"Invalid audio_input type: {type(audio_input).__name__}. Must be bytes, str (filepath), or tuple (sample_rate, audio_data)"
            print(f"DEBUG: {result['error']}")
        
    except FileNotFoundError as e:
        result["error"] = f"File error: {str(e)}"
    except Exception as e:
        result["error"] = f"Transcription failed: {str(e)}"
        import traceback
        traceback.print_exc()
    
    # Check if transcript is too short or empty (audio unclear)
    if result["transcript"] is not None:
        transcript_clean = result["transcript"].strip()
        if not transcript_clean or len(transcript_clean) < 2:
            result["error"] = "audio_unclear"
            result["transcript"] = ""
    
    return result


if __name__ == "__main__":
    # Test with a sample audio file path
    sample_wav_path = "assets/voice1.mp3"
    
    print("Testing STT module with file input...")
    
    # Check if sample file exists
    if not Path(sample_wav_path).exists():
        print(f"[WARNING] Sample audio file not found: {sample_wav_path}")
        print("\nTo test with your own audio:")
        print(f"  1. Place a .wav or .mp3 file in the assets/ folder")
        print(f"  2. Update the sample_wav_path variable")
        print(f"  3. Run again: python -m src.stt")
        print("\nExample:")
        print(f"  result = transcribe_audio('assets/voice1.mp3', source_type='file')")
    else:
        result = transcribe_audio(sample_wav_path, source_type="file")
        print(f"Source: {result['source']}")
        print(f"Transcript: {result['transcript']}")
        print(f"Error: {result['error']}")
