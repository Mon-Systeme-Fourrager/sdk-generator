import ast
from pathlib import Path
from typing import Dict, List, Set

from rope.base.ast import parse


def extract_exportable_names(file_path: Path) -> Set[str]:
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        tree = ast.parse(content)
        names = set()

        # Only process top-level nodes in the module
        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                # Include all top-level classes
                names.add(node.name)
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                # Include top-level functions that don't start with underscore
                if not node.name.startswith("_"):
                    names.add(node.name)

        return names
    except Exception as e:
        print(f"Warning: Could not parse {file_path}: {e}")
        return set()


def get_module_exports(module_path: Path) -> Dict[str, List[str]]:
    exports = {}

    if not module_path.exists() or not module_path.is_dir():
        return exports

    # Process all Python files in the module
    for py_file in module_path.glob("*.py"):
        if py_file.name == "__init__.py":
            continue

        module_name = py_file.stem
        names = extract_exportable_names(py_file)

        if names:
            exports[module_name] = sorted(list(names))

    return exports


def generate_explicit_imports(output_path: Path) -> str:
    imports = []

    # Import from api_config.py
    api_config_path = output_path / "api_config.py"
    if api_config_path.exists():
        names = extract_exportable_names(api_config_path)
        if names:
            names_str = ", ".join(sorted(names))
            imports.append(f"from .api_config import {names_str}")

    # Import from models module
    models_path = output_path / "models"
    if models_path.exists() and models_path.is_dir():
        model_exports = get_module_exports(models_path)

        if model_exports:
            for module_name, names in model_exports.items():
                if names:
                    names_str = ", ".join(names)
                    imports.append(f"from .models.{module_name} import {names_str}")

        # Also check if models/__init__.py has explicit exports
        models_init = models_path / "__init__.py"
        if models_init.exists():
            init_names = extract_exportable_names(models_init)
            if init_names:
                names_str = ", ".join(sorted(init_names))
                imports.append(f"from .models import {names_str}")

    # Import from services module
    services_path = output_path / "services"
    if services_path.exists() and services_path.is_dir():
        service_exports = get_module_exports(services_path)

        if service_exports:
            for module_name, names in service_exports.items():
                if names:
                    names_str = ", ".join(names)
                    imports.append(f"from .services.{module_name} import {names_str}")

        # Also check if services/__init__.py has explicit exports
        services_init = services_path / "__init__.py"
        if services_init.exists():
            init_names = extract_exportable_names(services_init)
            if init_names:
                names_str = ", ".join(sorted(init_names))
                imports.append(f"from .services import {names_str}")

    return ("# ruff: noqa: F401\n" if imports else "") + "\n".join(imports)


def patch_init_file(output_path: Path) -> None:
    init_file = output_path / "__init__.py"

    if not init_file.exists():
        return

    new_content = generate_explicit_imports(output_path)

    if new_content:
        new_content = new_content + "\n"

        with open(init_file, "w", encoding="utf-8") as f:
            f.write(new_content)


def apply_monkey_patch(output_path: str) -> None:
    output_dir = Path(output_path)

    if not output_dir.exists():
        return

    make_models_into_dir(output_dir)
    fix_service_imports(output_dir)
    patch_init_file(output_dir)


def _extract_components_with_rope_parsing(
    content: str,
) -> tuple[Dict[str, str], List[str], List[str]]:
    """
    Extract classes, imports, and other content using Rope's built-in AST functionality.
    This leverages Rope's AST classes and get_source_segment for clean extraction.
    """
    try:
        # Use Rope's AST parsing directly without Project
        rope_ast = parse(content)
        from rope.base.ast import ClassDef, Import, ImportFrom, get_source_segment

        imports = []
        other_content = []
        classes = {}

        # Use Rope's built-in node type checking instead of manual inspection
        for node in rope_ast.body:
            if isinstance(node, (Import, ImportFrom)):
                # Use Rope's get_source_segment for clean extraction
                import_text = get_source_segment(content, node)
                imports.append(import_text)
            elif isinstance(node, ClassDef):
                # Extract class using Rope's built-in functionality
                class_name = node.name
                class_content = get_source_segment(content, node)
                classes[class_name] = class_content
            else:
                # Everything else goes to other_content
                other_text = get_source_segment(content, node)
                other_content.append(other_text)

        return classes, imports, other_content

    except Exception:
        # Fallback to basic AST parsing if Rope parsing fails
        return _extract_components_basic_ast(content)


def _extract_components_basic_ast(
    content: str,
) -> tuple[Dict[str, str], List[str], List[str]]:
    """
    Fallback function using basic AST parsing when Rope analysis fails.
    This maintains backward compatibility with the existing functionality.
    """
    try:
        tree = ast.parse(content)
    except SyntaxError:
        return {}, [], []

    imports = []
    other_content = []
    classes = {}

    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            imports.append(ast.get_source_segment(content, node))
        elif isinstance(node, ast.ClassDef):
            class_name = node.name
            class_content = ast.get_source_segment(content, node)
            classes[class_name] = class_content
        else:
            other_content.append(ast.get_source_segment(content, node))

    return classes, imports, other_content


def make_models_into_dir(output_path: Path) -> None:
    """
    Split a models.py file into individual class files using Rope's built-in AST functionality.
    This uses Rope's excellent AST parsing and dependency analysis without the deprecated Project class.
    """
    models_file = output_path / "models.py"
    models_dir = output_path / "models"

    if not models_file.exists() or models_dir.exists():
        return

    # Read content and use Rope's AST parsing
    content = models_file.read_text(encoding="utf-8")

    # Extract classes using Rope's AST parsing
    classes, imports, other_content = _extract_components_with_rope_parsing(content)

    if not classes:
        return

    # Create models directory and process classes using Rope's capabilities
    models_dir.mkdir()
    _create_class_files_with_rope(models_dir, classes, imports, content)
    _create_utils_file(models_dir, other_content, imports)
    _create_models_init(models_dir, classes.keys(), bool(other_content))

    # Remove original file
    models_file.unlink()


def _extract_all_class_names(rope_ast) -> List[str]:
    """Extract all class names from the AST."""
    from rope.base.ast import ClassDef, walk

    class_names = []
    for node in walk(rope_ast):
        if isinstance(node, ClassDef):
            class_names.append(node.name)

    return class_names


def _create_class_files_with_rope(
    models_dir: Path, classes: Dict[str, str], imports: List[str], full_content: str
) -> None:
    """Create individual class files using Rope's built-in dependency analysis."""
    from rope.base.ast import parse

    # Parse the full content once with Rope for better analysis
    rope_ast = parse(full_content)
    all_class_names = _extract_all_class_names(rope_ast)

    import_lines = "\n".join(imports) if imports else ""

    for class_name, class_content in classes.items():
        class_file = models_dir / f"{class_name}.py"

        # Use Rope's enhanced dependency detection
        dependencies = _find_model_dependencies(class_content, all_class_names)

        content_parts = []
        if import_lines:
            content_parts.append(import_lines)

        if dependencies:
            for dep in sorted(dependencies):
                content_parts.append(f"from .{dep} import {dep}")

        content_parts.append(class_content)

        class_file.write_text("\n\n".join(content_parts) + "\n")


def _create_utils_file(
    models_dir: Path, other_content: List[str], imports: List[str]
) -> None:
    """Create utilities file if there's non-class content."""
    if not other_content:
        return

    utils_file = models_dir / "_utils.py"
    content_parts = []

    if imports:
        content_parts.append("\n".join(imports))

    content_parts.extend(other_content)

    utils_file.write_text("\n\n".join(content_parts) + "\n")


def _create_models_init(
    models_dir: Path, class_names: List[str], has_utils: bool
) -> None:
    """Create the models/__init__.py file."""
    init_file = models_dir / "__init__.py"
    content = _generate_models_init(class_names, has_utils)
    init_file.write_text(content)


def _find_model_dependencies(
    class_content: str, all_class_names: List[str]
) -> Set[str]:
    """
    Find model dependencies using Rope's AST parsing without Project.
    Excludes inheritance, focuses on composition relationships.
    """
    # Use Rope's AST for better parsing without Project to avoid deprecation warnings
    rope_ast = parse(class_content)
    return _extract_dependencies_from_rope_ast(rope_ast, class_content, all_class_names)


def _extract_dependencies_from_rope_ast(
    rope_ast, content: str, all_class_names: List[str]
) -> Set[str]:
    """Extract dependencies using Rope's built-in AST walking functionality."""
    from rope.base.ast import AnnAssign, ClassDef, Name, Subscript, walk

    dependencies = set()
    inherited_classes = set()

    # First pass: find inherited classes using Rope's walk function
    for node in walk(rope_ast):
        if isinstance(node, ClassDef):
            # Extract base classes (inheritance) using Rope's AST nodes
            for base in node.bases:
                if isinstance(base, Name) and base.id in all_class_names:
                    inherited_classes.add(base.id)

    # Second pass: find composition dependencies using Rope's AST traversal
    for node in walk(rope_ast):
        if isinstance(node, AnnAssign) and node.annotation:
            # Handle type annotations using Rope's built-in functionality
            annotation_names = _extract_names_from_rope_node(
                node.annotation, all_class_names
            )
            dependencies.update(annotation_names - inherited_classes)
        elif isinstance(node, Subscript):
            # Handle generic types like List[SomeModel] using Rope's AST
            slice_names = _extract_names_from_rope_node(node.slice, all_class_names)
            dependencies.update(slice_names - inherited_classes)

    return dependencies


def _extract_names_from_rope_node(node, all_class_names: List[str]) -> Set[str]:
    """Extract class names from a Rope AST node using built-in traversal."""
    from rope.base.ast import Name, walk

    names = set()
    for child_node in walk(node):
        if isinstance(child_node, Name) and child_node.id in all_class_names:
            names.add(child_node.id)

    return names


def _generate_models_init(class_names: List[str], has_utils: bool = False) -> str:
    """Generate the content for the models/__init__.py file."""
    imports = []
    all_exports = []

    for class_name in sorted(class_names):
        # Use exact class name for import (no snake_case conversion)
        imports.append(f"from .{class_name} import {class_name}")
        all_exports.append(class_name)

    content = "# ruff: noqa: F401\n"
    content += "\n".join(imports)

    # Import utilities if they exist (but don't export them)
    if has_utils:
        content += "\nfrom . import _utils"

    if all_exports:
        content += "\n\n__all__ = [\n"
        for class_name in sorted(all_exports):
            content += f'    "{class_name}",\n'
        content += "]\n"

    return content


def fix_service_imports(output_path: Path) -> None:
    """Fix service files to import only specific models instead of using 'from ..models import *'."""
    services_path = output_path / "services"

    if not services_path.exists() or not services_path.is_dir():
        return

    # Get all available model names from the models directory
    models_path = output_path / "models"
    if not models_path.exists():
        return

    available_models = set()
    for model_file in models_path.glob("*.py"):
        if model_file.name != "__init__.py":
            available_models.add(model_file.stem)

    # Process each service file
    for service_file in services_path.glob("*.py"):
        if service_file.name == "__init__.py":
            continue

        _fix_single_service_file(service_file, available_models)


def _fix_single_service_file(service_file: Path, available_models: Set[str]) -> None:
    """Fix imports in a single service file."""
    with open(service_file, "r", encoding="utf-8") as f:
        content = f.read()

    # Check if the file uses the wildcard import pattern
    if "from ..models import *" not in content:
        return

    # Find all model classes used in this service file
    used_models = _find_used_models_in_service(content, available_models)

    if not used_models:
        # Remove the wildcard import if no models are used
        new_content = content.replace("from ..models import *\n", "")
    else:
        # Replace wildcard import with specific imports
        specific_imports = []
        for model in sorted(used_models):
            specific_imports.append(f"from ..models.{model} import {model}")

        specific_import_text = "\n".join(specific_imports)
        new_content = content.replace("from ..models import *", specific_import_text)

    # Write back the updated content
    with open(service_file, "w", encoding="utf-8") as f:
        f.write(new_content)


def _find_used_models_in_service(content: str, available_models: Set[str]) -> Set[str]:
    """Find which model classes are actually used in a service file - simplified version."""
    used_models = set()

    # Simple but effective approach: look for model names in the content
    # This catches most usage patterns including annotations, instantiation, etc.
    for model in available_models:
        if _is_model_used_in_content(model, content):
            used_models.add(model)

    return used_models


def _is_model_used_in_content(model_name: str, content: str) -> bool:
    """Check if a model is used in content using pattern matching."""
    import re

    # Patterns that indicate model usage
    patterns = [
        rf"\b{re.escape(model_name)}\b",  # Direct usage
        rf":\s*{re.escape(model_name)}\b",  # Type annotation
        rf"List\[{re.escape(model_name)}\]",  # List type
        rf"Optional\[{re.escape(model_name)}\]",  # Optional type
        rf"-> {re.escape(model_name)}\b",  # Return type
    ]

    return any(re.search(pattern, content) for pattern in patterns)
