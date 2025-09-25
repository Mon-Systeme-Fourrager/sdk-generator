import os
from pathlib import Path
from shutil import copy, rmtree

from datamodel_code_generator import generate as datamodel_generate
from openapi_python_generator.common import HTTPLibrary
from openapi_python_generator.generate_data import generate_data

from .monkey_patch import apply_monkey_patch


def generate_sdk(source, output, constants_template_path=None):
    output_path = Path(output)
    if output_path.exists():
        rmtree(output_path)
    base_path = Path(__file__).parent
    templates_path = base_path / Path("templates")

    if constants_template_path is None:
        constants_template_path = templates_path / "constants-default.jinja2"

    constants_path = Path(constants_template_path)
    if not constants_path.exists():
        raise FileNotFoundError(f"Constants template file not found: {constants_path}")
    if not constants_path.is_file():
        raise ValueError(
            f"Provided constants template path is not a file: {constants_path}"
        )

    copy(constants_path, templates_path / "constants.jinja2")

    try:
        generate_data(
            source,
            output,
            custom_template_path=templates_path,
            library=HTTPLibrary.requests,
        )
        rmtree(Path(output) / "models")
        datamodel_generate(
            Path(source), output=Path(output) / "models.py", field_constraints=True
        )
        apply_monkey_patch(output)
        os.system("black " + output + " --quiet")
        os.system("ruff check --fix")
    finally:
        (templates_path / "constants.jinja2").unlink()
