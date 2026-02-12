# Open-LLM-VTuber Project Overview

## Purpose
Open-LLM-VTuber is a voice-interactive AI companion with Live2D avatar support that runs completely offline. It's a cross-platform Python application supporting real-time voice conversations, visual perception, and Live2D character animations.

## Tech Stack
- **Language**: Python 3.10-3.12
- **Web Framework**: FastAPI with WebSocket support
- **Package Manager**: conda (Anaconda/Miniconda)
- **Server**: Uvicorn ASGI server
- **Linting/Formatting**: Ruff
- **Pre-commit**: Ruff hooks for linting and formatting
- **Logging**: Loguru
- **Configuration**: YAML files with Pydantic-style validation

## Key Technologies
- **LLM Support**: Claude (Anthropic), OpenAI, Ollama, Groq, and others
- **ASR (Speech Recognition)**: Sherpa-ONNX, FunASR, Faster-Whisper, Azure, Groq Whisper
- **TTS (Text-to-Speech)**: Azure TTS, Edge TTS, MeloTTS, CosyVoice, GPT-SoVITS, ElevenLabs, Cartesia
- **VAD (Voice Activity)**: Silero VAD
- **MCP Integration**: Model Context Protocol for tool execution
- **Live2D**: Character animations and expressions

## Architecture
- **Factory Pattern**: All AI engines (ASR, TTS, Agent, VAD) use factory classes for instantiation
- **Dependency Injection**: ServiceContext acts as central container for all engines
- **WebSocket Communication**: Real-time bidirectional communication with clients
- **Modular Design**: Each component (ASR, TTS, Agent, etc.) has its own directory with interface and implementations
