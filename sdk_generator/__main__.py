import click

from .generator import generate_sdk


@click.command()
@click.argument("source")
@click.argument("output")
@click.argument(
    "constants_template_path",
    type=str,
)
def main(source: str, output: str, constants_template_path: str | None):
    generate_sdk(source, output, constants_template_path=constants_template_path)


if __name__ == "__main__":
    main()
