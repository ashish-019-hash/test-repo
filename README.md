# Node.js Parser

A Python tool for parsing Node.js/JavaScript codebases using Tree-sitter. Extracts classes, functions, imports, exports, and their metadata.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Parse a single file

```bash
python node_parser.py /path/to/file.js
```

### Parse an entire directory

```bash
python node_parser.py /path/to/node/project
```

### Use as a module

```python
from node_parser import NodeParser

parser = NodeParser()

# Parse a single file
classes, functions, class_map = parser.parse(
    root="/path/to/project",
    file_name="app.js",
    file_path="/path/to/project/app.js"
)

# Parse a directory
classes, functions, class_map = parser.parse_directory("/path/to/project")

# Access results
print(parser.get_classes())
print(parser.get_functions())
print(parser.get_class_map())
```

## Output Structure

The parser returns three data structures:

- **classes**: List of class keys in format `root/file/ClassName`
- **functions**: List of function names found
- **class_map**: Dictionary mapping class keys to function details including:
  - `function`: Full function key path
  - `words`: Unique words found in function body
  - `decorators`: Decorator information (always `None` for JS)
  - `imports`: List of imports in the file
  - `function_content`: Full function source code
  - `function_calls`: List of function calls made within the function

## Features

- Parses ES6 classes and methods
- Extracts function declarations
- Captures arrow functions and function expressions
- Extracts ES6 imports and CommonJS require() calls
- Extracts ES6 exports and module.exports
- Skips node_modules directory automatically
- Supports .js, .mjs, and .cjs files
