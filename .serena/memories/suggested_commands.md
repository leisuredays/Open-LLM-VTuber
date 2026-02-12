# Suggested Commands for Development

## Package Management (using conda)
- `conda activate <env_name>` - Activate conda environment
- `conda install <package>` - Install a package
- `pip install <package>` - Install via pip within conda env
- `python <script>` - Run a Python script (conda env must be activated)

## Running the Server
- `python run_server.py` - Start the server (default mode)
- `python run_server.py --verbose` - Start with debug logging
- `python run_server.py --hf_mirror` - Use Hugging Face mirror (for China)

## Code Quality
- `ruff check .` - Lint the codebase
- `ruff check . --fix` - Lint and auto-fix issues
- `ruff format .` - Format the codebase
- `pre-commit run --all-files` - Run all pre-commit hooks

## Git Operations
- `git submodule update --init --recursive` - Initialize frontend submodule
- `git restore frontend` - Reset frontend submodule if modified

## Project Upgrade
- `python upgrade.py` - Run the project upgrade script

## System Utilities (Linux)
- `ls` - List directory contents
- `cd` - Change directory
- `grep` - Search text patterns
- `find` - Find files
- `git status`, `git diff`, `git log` - Git operations
