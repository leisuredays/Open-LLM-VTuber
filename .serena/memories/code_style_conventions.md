# Code Style and Conventions

## General Style
- **Python Version**: 3.10+ (target-version in ruff config)
- **Line Length**: No explicit limit set (ruff default ~88)
- **Formatting**: Ruff formatter (similar to Black)
- **Imports**: Standard import order enforced by ruff

## Naming Conventions
- **Classes**: PascalCase (e.g., `ServiceContext`, `ASRFactory`, `WebSocketServer`)
- **Functions/Methods**: snake_case (e.g., `get_asr_system`, `init_logger`)
- **Variables**: snake_case (e.g., `config_path`, `asr_engine`)
- **Constants**: UPPER_SNAKE_CASE (e.g., `HF_HOME`, `MODELSCOPE_CACHE`)
- **Private methods**: Prefixed with underscore (e.g., `_init_mcp_components`)

## Type Hints
- Type hints are used extensively
- Union types use `|` syntax (e.g., `VADInterface | None`)
- `Type[...]` for class references
- Return types specified for public methods

## Docstrings
- Triple-quoted strings for docstrings
- Not required for all methods, used selectively
- Multi-language support in user-facing messages (Chinese/English)

## Code Patterns
- **Factory Pattern**: Static methods in factory classes (e.g., `ASRFactory.get_asr_system()`)
- **Interface Classes**: Abstract base classes for engines (e.g., `ASRInterface`, `TTSInterface`)
- **Lazy Imports**: Import statements inside factory methods for conditional loading
- **Configuration Classes**: Pydantic-style config classes in `config_manager/`

## Logging
- Use `loguru` logger: `from loguru import logger`
- Log levels: DEBUG, INFO, WARNING, ERROR, CRITICAL
- Structured logging with context using `{extra}` parameter

## Error Handling
- Use `@logger.catch` decorator for entry points
- Raise `ValueError` for unknown/invalid configurations
- Graceful fallbacks for optional components (translate_engine, etc.)
