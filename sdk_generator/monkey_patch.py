import ast
from pathlib import Path
from typing import Dict, List, Set


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


def make_models_into_dir(output_path: Path) -> None:
    models_file = output_path / "models.py"
    models_dir = output_path / "models"

    if not models_file.exists():
        return

    if models_dir.exists():
        return

    # Read the original models.py file
    with open(models_file, "r", encoding="utf-8") as f:
        content = f.read()

    # Parse the AST to extract classes and imports
    tree = ast.parse(content)

    # Extract imports from the top of the file
    imports = []
    other_content = []
    classes = {}

    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            imports.append(ast.get_source_segment(content, node))
        elif isinstance(node, (ast.ClassDef)):
            class_name = node.name
            class_content = ast.get_source_segment(content, node)
            classes[class_name] = class_content
        else:
            # Handle other top-level content (constants, functions, etc.)
            other_content.append(ast.get_source_segment(content, node))

    if not classes:
        return

    # Create the models directory
    models_dir.mkdir()

    # Create individual files for each class
    import_lines = "\n".join(imports) if imports else ""

    for class_name, class_content in classes.items():
        # Use the exact class name as filename (no snake_case conversion)
        class_file = models_dir / f"{class_name}.py"

        # Find model dependencies for this class
        model_dependencies = _find_model_dependencies(class_content, classes.keys())

        # Create the file content with imports and the class
        file_content = ""
        if import_lines:
            file_content += import_lines + "\n\n"

        # Add model imports if there are dependencies
        if model_dependencies:
            for dep in sorted(model_dependencies):
                file_content += f"from .{dep} import {dep}\n"
            file_content += "\n"

        # Add any other top-level content if needed
        if other_content:
            file_content += "\n".join(other_content) + "\n\n"

        file_content += class_content + "\n"

        with open(class_file, "w", encoding="utf-8") as f:
            f.write(file_content)

    # Create __init__.py that imports all classes
    init_content = _generate_models_init(classes.keys())
    init_file = models_dir / "__init__.py"
    with open(init_file, "w", encoding="utf-8") as f:
        f.write(init_content)

    # Remove the original models.py file
    models_file.unlink()


def _find_model_dependencies(
    class_content: str, all_class_names: List[str]
) -> Set[str]:
    """Find which other model classes this class depends on."""
    dependencies = set()

    # Parse the class content to find type annotations that reference other models
    try:
        class_tree = ast.parse(class_content)

        for node in ast.walk(class_tree):
            if isinstance(node, ast.AnnAssign) and node.annotation:
                # Handle direct type annotations like: field: SomeModel
                annotation_str = ast.get_source_segment(class_content, node.annotation)
                if annotation_str:
                    deps = _extract_model_names_from_annotation(
                        annotation_str, all_class_names
                    )
                    dependencies.update(deps)
            elif isinstance(node, ast.Subscript) and hasattr(node, "slice"):
                # Handle generic types like List[SomeModel] or Optional[SomeModel]
                slice_str = ast.get_source_segment(class_content, node.slice)
                if slice_str:
                    deps = _extract_model_names_from_annotation(
                        slice_str, all_class_names
                    )
                    dependencies.update(deps)
    except Exception:
        # Fallback: simple string matching if AST parsing fails
        for class_name in all_class_names:
            if (
                class_name in class_content
                and class_name != _extract_class_name_from_content(class_content)
            ):
                dependencies.add(class_name)

    return dependencies


def _extract_model_names_from_annotation(
    annotation: str, all_class_names: List[str]
) -> Set[str]:
    """Extract model class names from a type annotation string."""
    dependencies = set()

    # Remove common generic wrappers
    annotation = (
        annotation.replace("List[", "").replace("Optional[", "").replace("]", "")
    )
    annotation = annotation.replace("Union[", "").replace(" | ", ",")

    # Split by common separators and check each part
    parts = [part.strip() for part in annotation.replace(",", " ").split()]

    for part in parts:
        if part in all_class_names:
            dependencies.add(part)

    return dependencies


def _extract_class_name_from_content(class_content: str) -> str:
    """Extract the class name from class content."""
    try:
        tree = ast.parse(class_content)
        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                return node.name
    except Exception:
        pass
    return ""


def _generate_models_init(class_names: List[str]) -> str:
    """Generate the content for the models/__init__.py file."""
    imports = []
    all_exports = []

    for class_name in sorted(class_names):
        # Use exact class name for import (no snake_case conversion)
        imports.append(f"from .{class_name} import {class_name}")
        all_exports.append(class_name)

    content = "# ruff: noqa: F401\n"
    content += "\n".join(imports)
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
    """Find which model classes are actually used in a service file."""
    used_models = set()

    try:
        tree = ast.parse(content)

        for node in ast.walk(tree):
            # Check function return type annotations
            if isinstance(node, ast.FunctionDef) and node.returns:
                return_type = ast.get_source_segment(content, node.returns)
                if return_type:
                    models_in_annotation = _extract_model_names_from_annotation(
                        return_type, available_models
                    )
                    used_models.update(models_in_annotation)

            # Check function parameter type annotations
            if isinstance(node, ast.FunctionDef):
                for arg in node.args.args:
                    if arg.annotation:
                        param_type = ast.get_source_segment(content, arg.annotation)
                        if param_type:
                            models_in_annotation = _extract_model_names_from_annotation(
                                param_type, available_models
                            )
                            used_models.update(models_in_annotation)

            # Check variable annotations
            if isinstance(node, ast.AnnAssign) and node.annotation:
                var_type = ast.get_source_segment(content, node.annotation)
                if var_type:
                    models_in_annotation = _extract_model_names_from_annotation(
                        var_type, available_models
                    )
                    used_models.update(models_in_annotation)

            # Check direct usage of model names (for instantiation, etc.)
            if isinstance(node, ast.Name) and node.id in available_models:
                used_models.add(node.id)

    except Exception:
        # Fallback: simple string matching if AST parsing fails
        for model in available_models:
            if model in content:
                used_models.add(model)

    return used_models
