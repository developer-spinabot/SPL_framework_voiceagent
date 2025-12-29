# Voice Agent with SPL Framework

A local voice assistant with Subsumption pattern learning for efficient LLM usage.

## 🚀 Features

- **Local LLM** (Phi-2 or Llama 3B) for private, offline operation
- **Voice Interface** with push-to-talk functionality
- **Structured Prompt Learning** to reduce LLM usage
- **RAG** (Retrieval Augmented Generation) with Chroma DB
- **Modular Design** for easy customization

## 🛠 Tech Stack

- **LLM**: Local models via LlamaCpp
- **STT**: Whisper for speech-to-text
- **TTS**: Local text-to-speech
- **Vector Store**: Chroma DB
- **SPL**: Custom Structured Prompt Learning engine

## 🏗 SPL Framework

### Layer 0: Reactive Rules
- Numeric-only input detection
- Filler words handling (uh, um, etc.)
- Profanity filtering
- System commands (repeat, hang up, etc.)

### Layer 1: Pattern Matching
- Greetings (hi, hello)
- Common questions (hours, location, menu)
- Thanks/acknowledgments

### Layer 2: Vector Cache (Planned)
- Semantic similarity search
- Response caching

### Layer 3: LLM Fallback
- Full LLM processing when needed

## 🚀 Quick Start

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Download models:
   - Place GGUF models in `models/` directory
   - Required: `phi-2.gguf` or `llama-3b.gguf`

3. Run the voice agent:
   ```bash
   python -m app.local_voice_agent
   ```

## ⚙️ Configuration

Edit `.env` to configure:
- Model paths
- Audio settings
- SPL parameters

## 📝 Notes

- Press and hold SPACE to record
- Release SPACE to process speech
- Say "goodbye" to exit

