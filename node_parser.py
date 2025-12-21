"""
Node.js/JavaScript Parser using Tree-sitter

A standalone Python module for parsing Node.js codebases and extracting
classes, functions, imports, exports, and their metadata.
"""

import re
import os
from tree_sitter import Language, Parser
import tree_sitter_javascript as tsjavascript


class NodeParser:
    """
    Parser for JavaScript/Node.js files using Tree-sitter.
    
    Extracts classes, functions (including arrow functions), methods,
    imports, exports, and function call relationships.
    """

    def __init__(self):
        # Initialize tree-sitter with JavaScript language
        self.__language = Language(tsjavascript.language())
        self.__parser = Parser(self.__language)

        self.__classes = []
        self.__functions = []
        self.__class_map = {}

    def __normalized_path(self, path):
        """Normalize path separators to OS-specific format."""
        return path.replace('//', os.sep)

    def __extract_imports(self, root_node):
        """Extract all JS imports (import ... from, require(), CommonJS)."""
        imports = []

        stack = [root_node]
        while stack:
            node = stack.pop()

            # ES module import
            if node.type == "import_statement":
                imports.append(node.text.decode("utf-8"))

            # CommonJS require()
            if node.type == "call_expression":
                func = node.child_by_field_name("function")
                if func and func.text.decode("utf-8") == "require":
                    imports.append(node.text.decode("utf-8"))

            stack.extend(reversed(node.children))

        return imports

    def __extract_function_calls(self, node):
        """Extract nested function calls inside a function body."""
        calls = []
        
        # Handle None node (when function has no body)
        if node is None:
            return calls
            
        stack = [node]

        while stack:
            current = stack.pop()

            if current.type == "call_expression":
                func = current.child_by_field_name("function")
                if func:
                    calls.append(func.text.decode("utf-8"))

            stack.extend(reversed(current.children))

        return calls

    def __add_function_to_class_map(self, class_key, function_key, func_body, node, words):
        """Helper to add a function entry to the class map."""
        calls = self.__extract_function_calls(func_body)
        decorators = [{"decorator": "None"}]
        
        if class_key not in self.__class_map:
            self.__class_map[class_key] = []

        self.__class_map[class_key].append({
            "function": function_key,
            "words": list(set(words)),
            "decorators": decorators,
            "imports": self.imports,
            "function_content": node.text.decode("utf-8"),
            "function_calls": calls
        })

    def __traverse_tree(self, root, file, root_node):
        """Traverse AST with proper class scoping using (node, class_context) tuples."""
        # Stack contains tuples of (node, current_class_context)
        stack = [(root_node, "Global")]

        while stack:
            node, current_class = stack.pop()

            # ---------------------
            # CLASS DEFINITIONS
            # ---------------------
            if node.type == "class_declaration":
                class_name_node = node.child_by_field_name("name")
                if class_name_node:
                    class_name = class_name_node.text.decode("utf-8")

                    class_key = f"{root}//{file}//{class_name}"
                    class_key = self.__normalized_path(class_key)

                    self.__classes.append(class_key)
                    self.__class_map[class_key] = []

                    # Push children with the new class context
                    for child in reversed(node.children):
                        stack.append((child, class_name))
                    continue  # Skip default child processing

            # ---------------------
            # FUNCTION DECLARATIONS
            # ---------------------
            elif node.type == "function_declaration":
                func_name_node = node.child_by_field_name("name")
                if func_name_node:
                    func_name = func_name_node.text.decode("utf-8")
                    func_body = node.child_by_field_name("body")
                    
                    body_text = func_body.text.decode("utf-8") if func_body else ""
                    words = re.findall(r"\b\w+\b", body_text)

                    class_key = f"{root}//{file}//{current_class}"
                    class_key = self.__normalized_path(class_key)
                    function_key = f"{class_key}//{func_name}"
                    function_key = self.__normalized_path(function_key)

                    self.__add_function_to_class_map(class_key, function_key, func_body, node, words)
                    self.__functions.append(func_name)

            # ---------------------
            # METHOD DEFINITIONS (inside class)
            # ---------------------
            elif node.type == "method_definition":
                method_name_node = node.child_by_field_name("name")
                func_body = node.child_by_field_name("body")

                if method_name_node:
                    method_name = method_name_node.text.decode("utf-8")
                    
                    body_text = func_body.text.decode("utf-8") if func_body else ""
                    words = re.findall(r"\b\w+\b", body_text)

                    class_key = f"{root}//{file}//{current_class}"
                    class_key = self.__normalized_path(class_key)
                    function_key = f"{class_key}//{method_name}"
                    function_key = self.__normalized_path(function_key)

                    self.__add_function_to_class_map(class_key, function_key, func_body, node, words)
                    self.__functions.append(method_name)

            # ---------------------
            # ARROW FUNCTIONS & FUNCTION EXPRESSIONS
            # ---------------------
            elif node.type in ("lexical_declaration", "variable_declaration"):
                for child in node.children:
                    if child.type == "variable_declarator":
                        name_node = child.child_by_field_name("name")
                        value_node = child.child_by_field_name("value")
                        
                        if name_node and value_node and value_node.type in ("arrow_function", "function"):
                            func_name = name_node.text.decode("utf-8")
                            func_body = value_node.child_by_field_name("body")
                            
                            body_text = func_body.text.decode("utf-8") if func_body else ""
                            words = re.findall(r"\b\w+\b", body_text)

                            class_key = f"{root}//{file}//{current_class}"
                            class_key = self.__normalized_path(class_key)
                            function_key = f"{class_key}//{func_name}"
                            function_key = self.__normalized_path(function_key)

                            self.__add_function_to_class_map(class_key, function_key, func_body, node, words)
                            self.__functions.append(func_name)

            # ---------------------
            # EXPORTS (module.exports, exports.x, export statements)
            # ---------------------
            elif node.type == "export_statement":
                # Handle ES6 exports - extract any function declarations inside
                for child in node.children:
                    if child.type == "function_declaration":
                        stack.append((child, current_class))

            # Push remaining children with current class context
            for child in reversed(node.children):
                stack.append((child, current_class))

    def __extract_exports(self, root_node):
        """Extract all export statements."""
        exports = []
        stack = [root_node]
        
        while stack:
            node = stack.pop()
            
            if node.type in ("export_statement", "export_default_declaration"):
                exports.append({
                    "type": node.type,
                    "text": node.text.decode("utf-8")[:200]
                })
            
            # Handle module.exports = ... and exports.x = ...
            if node.type == "assignment_expression":
                left = node.child_by_field_name("left")
                if left:
                    left_text = left.text.decode("utf-8")
                    if left_text.startswith("module.exports") or left_text.startswith("exports."):
                        exports.append({
                            "type": "commonjs_export",
                            "text": node.text.decode("utf-8")[:200]
                        })
            
            stack.extend(reversed(node.children))
        
        return exports

    # -------------------------
    # PUBLIC METHODS
    # -------------------------
    def reset(self):
        """Reset parser state for parsing new files."""
        self.__classes = []
        self.__functions = []
        self.__class_map = {}
        self.imports = []
        self.exports = []

    def parse(self, root, file_name, file_path):
        """
        Parse a JavaScript file and extract classes, functions, and their metadata.
        
        Args:
            root: Root directory path
            file_name: Name of the file being parsed
            file_path: Full path to the file
            
        Returns:
            Tuple of (classes, functions, class_map)
        """
        with open(file_path, "r", encoding="utf-8") as f:
            code = f.read()

        tree = self.__parser.parse(code.encode("utf-8"))
        root_node = tree.root_node

        # Extract imports first (global)
        self.imports = self.__extract_imports(root_node)
        
        # Extract exports
        self.exports = self.__extract_exports(root_node)

        # Traverse AST for classes & functions
        self.__traverse_tree(root, file_name, root_node)

        return self.__classes, self.__functions, self.__class_map

    def parse_directory(self, directory):
        """
        Parse all JavaScript files in a directory.
        
        Args:
            directory: Path to the directory to parse
            
        Returns:
            Tuple of (classes, functions, class_map)
        """
        self.reset()
        
        for root, dirs, files in os.walk(directory):
            # Skip node_modules
            if 'node_modules' in dirs:
                dirs.remove('node_modules')
            
            for file in files:
                if file.endswith(('.js', '.mjs', '.cjs')):
                    file_path = os.path.join(root, file)
                    try:
                        self.parse(root, file, file_path)
                    except Exception as e:
                        print(f"Error parsing {file_path}: {e}")
        
        return self.__classes, self.__functions, self.__class_map

    def get_classes(self):
        """Get list of parsed classes."""
        return self.__classes

    def get_functions(self):
        """Get list of parsed functions."""
        return self.__functions

    def get_class_map(self):
        """Get the class map with function details."""
        return self.__class_map


# Example usage
if __name__ == '__main__':
    import json
    import sys
    
    parser = NodeParser()
    
    if len(sys.argv) > 1:
        path = sys.argv[1]
        if os.path.isfile(path):
            classes, functions, class_map = parser.parse(
                os.path.dirname(path), 
                os.path.basename(path), 
                path
            )
        else:
            classes, functions, class_map = parser.parse_directory(path)
        
        print("=" * 50)
        print("PARSING RESULTS")
        print("=" * 50)
        print(f"\nClasses found: {len(classes)}")
        print(json.dumps(classes, indent=2))
        print(f"\nFunctions found: {len(functions)}")
        print(json.dumps(functions, indent=2))
        print(f"\nClass Map:")
        print(json.dumps(class_map, indent=2))
    else:
        print("Usage: python node_parser.py <file_or_directory>")
        print("\nExamples:")
        print("  python node_parser.py /path/to/file.js")
        print("  python node_parser.py /path/to/node/project")
