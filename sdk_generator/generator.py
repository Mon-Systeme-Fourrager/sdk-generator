from pathlib import Path
from shutil import copy

from openapi_python_generator.generate_data import generate_data
from openapi_python_generator.common import HTTPLibrary


def generate_sdk(source, output, constants_template_path=None):
    path = Path(__file__).parent / Path("templates")
    print(path)

    if constants_template_path is not None:
        constants_path = Path(constants_template_path)
        if not constants_path.exists():
            raise FileNotFoundError(
                f"Constants template file not found: {constants_path}"
            )
        if not constants_path.is_file():
            raise ValueError(
                f"Provided constants template path is not a file: {constants_path}"
            )

        copy(constants_path, path / "constants.jinja2")

    generate_data(
        source, output, custom_template_path=path, library=HTTPLibrary.requests
    )

    if constants_template_path is not None:
        (path / "constants.jinja2").unlink()
