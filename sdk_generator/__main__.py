import click

from .generator import generate_sdk


@click.command()
@click.argument("source")
@click.argument("output")
@click.option(
    "--constants-template-path",
    type=str,
    default=None,
    help="Path to a custom constants.jinja2 template file.",
)
def main(source: str, output: str, constants_template_path: str | None = None):
    generate_sdk(source, output, constants_template_path=constants_template_path)


if __name__ == "__main__":
    main()
