import ast
from pathlib import Path
from typing import List, Set, Dict


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
            imports.append(f"from .api_config import {names_str}  # noqa: F401")

    # Import from models module
    models_path = output_path / "models"
    if models_path.exists() and models_path.is_dir():
        model_exports = get_module_exports(models_path)

        if model_exports:
            for module_name, names in model_exports.items():
                if names:
                    names_str = ", ".join(names)
                    imports.append(
                        f"from .models.{module_name} import {names_str}  # noqa: F401"
                    )

        # Also check if models/__init__.py has explicit exports
        models_init = models_path / "__init__.py"
        if models_init.exists():
            init_names = extract_exportable_names(models_init)
            if init_names:
                names_str = ", ".join(sorted(init_names))
                imports.append(f"from .models import {names_str}  # noqa: F401")

    # Import from services module
    services_path = output_path / "services"
    if services_path.exists() and services_path.is_dir():
        service_exports = get_module_exports(services_path)

        if service_exports:
            for module_name, names in service_exports.items():
                if names:
                    names_str = ", ".join(names)
                    imports.append(
                        f"from .services.{module_name} import {names_str}  # noqa: F401"
                    )

        # Also check if services/__init__.py has explicit exports
        services_init = services_path / "__init__.py"
        if services_init.exists():
            init_names = extract_exportable_names(services_init)
            if init_names:
                names_str = ", ".join(sorted(init_names))
                imports.append(f"from .services import {names_str}  # noqa: F401")

    return "\n".join(imports)


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

    patch_init_file(output_dir)
