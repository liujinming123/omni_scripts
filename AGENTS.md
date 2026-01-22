# AGENTS.md

This file provides guidelines for agentic coding tools working in this repository.

## Test Commands

### Running Tests
- Run the main test: `python test_vllm_omni.py`
- Start vLLM-Omni service: `bash start_vllm_omni.sh` or `./start_vllm_omni.sh`
- Test with custom port: `PORT=8092 ./start_vllm_omni.sh`
- Test with custom GPU memory: `GPU_MEM_UTIL=0.8 ./start_vllm_omni.sh`

### No Linting/Build Configuration
This repository currently does not have automated linting or build tools configured. Use standard Python/Bash best practices.

## Code Style Guidelines

### Python Code

#### Imports
Order imports: standard library → third-party → local. Separate groups with blank lines.
```python
import os
import sys

import torch
import vllm

from local_module import helper
```

#### Formatting
- Use f-strings for string formatting: `f"Status: {status}"`
- Use double quotes for strings
- Use 4 spaces for indentation (no tabs)
- Maximum line length: 100 characters (soft limit)
- Add docstrings to all functions and modules using triple quotes

#### Naming Conventions
- Functions and variables: `snake_case`
- Constants: `SCREAMING_SNAKE_CASE`
- Classes: `PascalCase`
- Private functions: `_leading_underscore`

#### Types
- Add type hints for function signatures: `def test_environment() -> int:`
- Use built-in types: `str`, `int`, `bool`, `list`, `dict`
- For more complex types, consider `typing` module if needed

#### Error Handling
- Use try/except for operations that may fail
- Provide meaningful error messages in exceptions
- Return status codes (0 for success, non-zero for failure)
- Use boolean flags to track success/failure

#### Structure
```python
#!/usr/bin/env python3
"""Module docstring"""

import sys

def function_name(param: type) -> return_type:
    """Function docstring"""
    # Implementation
    pass

if __name__ == "__main__":
    sys.exit(function_name())
```

### Bash Scripts

#### Formatting
- Start with shebang: `#!/bin/bash`
- Use 4 spaces for indentation
- Add comments describing sections
- Use descriptive variable names

#### Environment Variables
- Use UPPERCASE for environment variables
- Provide defaults using `${VAR:-default}` syntax
- Document important environment variables in comments

#### Naming
- Variables: `lowercase_with_underscores`
- Constants: `UPPERCASE_WITH_UNDERSCORES`
- Functions: `snake_case`

#### Error Handling
- Check command success with `$?` or `set -e`
- Provide meaningful error messages
- Return appropriate exit codes

#### Structure
```bash
#!/bin/bash
# Script description

# Configuration
VAR=${VAR:-default}

# Main logic
main() {
    echo "Starting..."
}

main "$@"
```

## Project-Specific Conventions

### Model Paths
- Default model: `/vepfs/public/model-public/Qwen3-Omni-30B-A3B-Instruct`
- Alternative models may be commented out for reference

### Port Configuration
- Default port: 8091
- Override with `PORT` environment variable

### GPU Configuration
- Default GPU memory utilization: 0.9
- Override with `GPU_MEM_UTIL` environment variable

### Logging
- Use `echo` in bash scripts for informational output
- Use `print()` in Python for test output
- Format output with visual separators using `=`, `-`, or similar characters
- Use emoji sparingly for test results (✅/❌ are acceptable in tests)

### Chinese Comments
- Chinese comments are acceptable and used in this codebase
- Maintain consistency with existing code style
