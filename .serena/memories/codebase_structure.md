# Codebase Structure

## Root Directory
```
Open-LLM-VTuber/
├── run_server.py          # Main entry point
├── upgrade.py             # Project upgrade script
├── conf.yaml              # User configuration (gitignored)
├── pyproject.toml         # Project dependencies and metadata
├── uv.lock                # Dependency lock file
├── requirements.txt       # Generated requirements (for non-uv users)
├── mcp_servers.json       # MCP server configuration
└── CLAUDE.md              # Claude Code instructions
```

## Source Code (`src/open_llm_vtuber/`)
```
src/open_llm_vtuber/
├── server.py              # FastAPI WebSocket server
├── routes.py              # WebSocket route definitions
├── websocket_handler.py   # WebSocket message routing
├── service_context.py     # Dependency injection container
├── live2d_model.py        # Live2D model handling
├── chat_history_manager.py # Chat persistence
├── chat_group.py          # Multi-user group support
├── message_handler.py     # Message processing
├── proxy_handler.py       # Proxy WebSocket handling
├── proxy_message_queue.py # Proxy message queue
│
├── agent/                 # LLM agent system
│   ├── agent_factory.py   # Agent instantiation
│   ├── stateless_llm_factory.py
│   ├── agents/            # Agent implementations (basic_memory, hume_ai, letta, mem0)
│   └── stateless_llm/     # Stateless LLM implementations
│
├── asr/                   # Automatic Speech Recognition
│   ├── asr_factory.py     # ASR instantiation
│   ├── asr_interface.py   # ASR interface
│   └── *_asr.py           # ASR implementations
│
├── tts/                   # Text-to-Speech
│   ├── tts_factory.py     # TTS instantiation
│   ├── tts_interface.py   # TTS interface
│   └── *_tts.py           # TTS implementations
│
├── vad/                   # Voice Activity Detection
│   └── silero_vad.py      # Silero VAD implementation
│
├── config_manager/        # Configuration system
│   ├── main.py            # Main config class
│   ├── system.py          # System config
│   ├── character.py       # Character config
│   ├── agent.py, asr.py, tts.py, vad.py  # Component configs
│   └── utils.py           # YAML reading/validation
│
├── conversations/         # Conversation handling
│   ├── conversation_handler.py
│   ├── single_conversation.py
│   ├── group_conversation.py
│   └── tts_manager.py
│
├── mcpp/                  # MCP (Model Context Protocol)
│   ├── tool_executor.py
│   └── server_registry.py
│
├── translate/             # Translation services
├── live/                  # Live streaming integration
└── utils/                 # Utility functions
```

## Other Important Directories
```
config_templates/          # Default configuration templates
├── conf.default.yaml      # English default config
└── conf.ZH.default.yaml   # Chinese default config

characters/                # Character configuration files (YAML)
live2d-models/             # Live2D model files
frontend/                  # Web frontend (Git submodule)
prompts/                   # System prompts
scripts/                   # Utility scripts
logs/                      # Log files (gitignored)
cache/                     # Cache directory (gitignored)
chat_history/              # Persisted conversations
```
