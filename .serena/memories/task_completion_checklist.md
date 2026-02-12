# Task Completion Checklist

When completing a task, ensure the following steps are performed:

## 1. Code Quality Checks
- [ ] Run `ruff check .` - Ensure no linting errors
- [ ] Run `ruff format .` - Format the code consistently
- [ ] Or run `pre-commit run --all-files` - Runs both ruff check (with --fix) and format

## 2. Type Checking
- Ensure type hints are added for new public methods
- Use `Type | None` syntax for optional types

## 3. Configuration Updates
If configuration changes were made:
- [ ] Update `config_templates/conf.default.yaml` (English)
- [ ] Update `config_templates/conf.ZH.default.yaml` (Chinese)
- [ ] Add validation in corresponding config manager class

## 4. Factory Updates
If adding a new engine implementation:
- [ ] Create interface implementation
- [ ] Add to factory class (e.g., `asr_factory.py`, `tts_factory.py`)
- [ ] Add configuration class in `config_manager/`
- [ ] Update default YAML configurations

## 5. Documentation
- Add docstrings for complex or public functions
- Update CLAUDE.md if architectural changes are significant

## 6. Pre-commit Verification
```bash
pre-commit run --all-files
```
This runs:
- `ruff` with `--fix --exit-non-zero-on-fix`
- `ruff-format`

## Notes
- No automated test suite is configured (manual testing through web interface)
- Maintain backward compatibility when possible
- Use the upgrade system for breaking changes
