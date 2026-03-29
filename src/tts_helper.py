import pyttsx3
import threading
import os

# Initialize TTS engine
engine = pyttsx3.init()

def setup_voice(rate=150, volume=1.0, voice_index=0):
    """Configure TTS voice settings."""
    engine.setProperty('rate', rate)
    engine.setProperty('volume', volume)
    voices = engine.getProperty('voices')
    if voices and voice_index < len(voices):
        engine.setProperty('voice', voices[voice_index].id)
        print(f"✅ Voice set to: {voices[voice_index].name}")

def speak(text: str, async_mode=True):
    """Speak the given text."""
    if not text.strip():
        return

    def _speak():
        try:
            engine.say(text)
            engine.runAndWait()
        except Exception as e:
            print(f"TTS Error: {e}")

    if async_mode:
        thread = threading.Thread(target=_speak, daemon=True)
        thread.start()
    else:
        _speak()

def list_voices():
    """List all available voices."""
    voices = engine.getProperty('voices')
    print(f"\n🎙️  Available voices ({len(voices)}):")
    for i, voice in enumerate(voices):
        print(f"  [{i}] {voice.name} - {voice.id}")

if __name__ == "__main__":
    list_voices()
    setup_voice(rate=150)
    print("\n🧪 Testing TTS...")
    speak("Hello! I am the Indian Sign Language translator.", async_mode=False)
    speak("Testing text to speech functionality.", async_mode=False)
    print("✅ TTS test complete!")