import subprocess
import sys
import os
import tempfile
import shutil
from pathlib import Path
import yaml


class TestCLICommands:
    """Test the CLI commands using subprocess calls."""

    def setup_method(self):
        """Set up test environment before each test."""
        self.test_dir = tempfile.mkdtemp()
        self.original_cwd = os.getcwd()

        # Create a simple test OpenAPI spec
        self.test_spec = {
            "openapi": "3.0.0",
            "info": {"title": "Test API", "version": "1.0.0"},
            "paths": {
                "/test": {
                    "get": {
                        "operationId": "getTest",
                        "responses": {"200": {"description": "Success"}},
                    }
                }
            },
        }

        self.spec_file = os.path.join(self.test_dir, "test_spec.yaml")
        with open(self.spec_file, "w") as f:
            yaml.dump(self.test_spec, f)

        self.constants_file = Path(__file__).parent / "constants.jinja2"

    def teardown_method(self):
        """Clean up after each test."""
        os.chdir(self.original_cwd)
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_cli_help_command(self):
        """Test CLI help command."""
        result = subprocess.run(
            [sys.executable, "-m", "sdk-generator", "--help"],
            capture_output=True,
            text=True,
            cwd=self.original_cwd,
        )

        assert result.returncode == 0, "Help command should succeed"
        assert "Usage:" in result.stdout, "Help should show usage information"
        assert "SOURCE" in result.stdout, "Help should mention SOURCE parameter"
        assert "OUTPUT" in result.stdout, "Help should mention OUTPUT parameter"

    def test_cli_basic_generation(self):
        """Test basic SDK generation via CLI."""
        output_dir = os.path.join(self.test_dir, "output")

        # Run the CLI command
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "sdk-generator",
                self.spec_file,
                output_dir,
                self.constants_file,
            ],
            capture_output=True,
            text=True,
            cwd=self.original_cwd,
        )

        # Check the command succeeded
        assert result.returncode == 0, f"CLI failed with error: {result.stderr}"

        # Check that output directory was created
        assert os.path.exists(output_dir), "Output directory was not created"

        # Check that some files were generated
        generated_files = list(Path(output_dir).rglob("*"))
        assert len(generated_files) > 0, "No files were generated"

        # Clean up the output directory
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)

    def test_cli_nonexistent_spec_file(self):
        """Test CLI behavior with non-existent spec file."""
        output_dir = os.path.join(self.test_dir, "output")
        nonexistent_spec = os.path.join(self.test_dir, "nonexistent.yaml")

        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "sdk-generator",
                nonexistent_spec,
                output_dir,
                self.constants_file,
            ],
            capture_output=True,
            text=True,
            cwd=self.original_cwd,
        )

        # Should fail with non-zero exit code
        assert result.returncode != 0, "CLI should fail with non-existent spec file"
        assert (
            "No such file" in result.stderr
            or "not found" in result.stderr.lower()
            or "FileNotFoundError" in result.stderr
        )

        # Clean up any created output directory
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)

    def test_cli_invalid_spec_file(self):
        """Test CLI behavior with invalid spec file."""
        output_dir = os.path.join(self.test_dir, "output")

        # Create an invalid YAML file
        invalid_spec_file = os.path.join(self.test_dir, "invalid.yaml")
        with open(invalid_spec_file, "w") as f:
            f.write("invalid: yaml: content: [")

        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "sdk-generator",
                invalid_spec_file,
                output_dir,
                self.constants_file,
            ],
            capture_output=True,
            text=True,
            cwd=self.original_cwd,
        )

        # Should fail with non-zero exit code
        assert result.returncode != 0, "CLI should fail with invalid spec file"

        # Clean up any created output directory
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)

    def test_cli_output_directory_creation(self):
        """Test that CLI creates nested output directories."""
        nested_output = os.path.join(self.test_dir, "deep", "nested", "output")

        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "sdk-generator",
                self.spec_file,
                nested_output,
                self.constants_file,
            ],
            capture_output=True,
            text=True,
            cwd=self.original_cwd,
        )

        assert result.returncode == 0, f"CLI failed: {result.stderr}"
        assert os.path.exists(nested_output), "Nested output directory was not created"

        # Clean up the nested directories
        root_nested = os.path.join(self.test_dir, "deep")
        if os.path.exists(root_nested):
            shutil.rmtree(root_nested)


class TestCLIOutput:
    """Test the actual output generated by the CLI."""

    def setup_method(self):
        """Set up test environment before each test."""
        self.test_dir = Path("test")
        self.test_dir.mkdir(exist_ok=True)
        self.original_cwd = os.getcwd()

        # Create a more comprehensive test OpenAPI spec
        self.test_spec = {
            "openapi": "3.0.0",
            "info": {
                "title": "Test API",
                "version": "1.2.3",
                "description": "A test API for SDK generation",
            },
            "servers": [{"url": "https://api.example.com/v1"}],
            "paths": {
                "/users": {
                    "get": {
                        "operationId": "listUsers",
                        "summary": "List users",
                        "responses": {
                            "200": {
                                "description": "Success",
                                "content": {
                                    "application/json": {
                                        "schema": {
                                            "type": "array",
                                            "items": {
                                                "$ref": "#/components/schemas/User"
                                            },
                                        }
                                    }
                                },
                            }
                        },
                    },
                    "post": {
                        "operationId": "createUser",
                        "summary": "Create user",
                        "requestBody": {
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/User"}
                                }
                            }
                        },
                        "responses": {"201": {"description": "Created"}},
                    },
                }
            },
            "components": {
                "schemas": {
                    "User": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "integer"},
                            "name": {"type": "string"},
                            "email": {"type": "string", "format": "email"},
                        },
                        "required": ["name", "email"],
                    }
                }
            },
        }

        self.spec_file = self.test_dir / "test_spec.yaml"
        with open(self.spec_file, "w+") as f:
            yaml.dump(self.test_spec, f)

        self.constants_file = Path(__file__).parent / "constants.jinja2"

    def teardown_method(self):
        """Clean up after each test."""
        os.chdir(self.original_cwd)
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_generated_service_files(self):
        """Test that service files are generated for each tag/operation."""
        output_dir = self.test_dir / "output"

        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "sdk-generator",
                self.spec_file,
                output_dir,
                self.constants_file,
            ],
            capture_output=True,
            text=True,
            cwd=self.original_cwd,
        )

        assert result.returncode == 0, f"CLI failed: {result.stderr}"

        # Check that service files exist
        service_files = list((output_dir / "services").glob("*service.py"))
        assert len(service_files) > 0, "No service files were generated"

        # Check that at least one service file contains our operations
        service_content = ""
        for service_file in service_files:
            with open(service_file, "r") as f:
                service_content += f.read()

        assert (
            "listUsers" in service_content or "list_users" in service_content
        ), "listUsers operation not found in service files"
        assert (
            "createUser" in service_content or "create_user" in service_content
        ), "createUser operation not found in service files"

        # Clean up the output directory
        # if os.path.exists(output_dir):
        #     shutil.rmtree(output_dir)

    def test_generated_file_structure(self):
        """Test that the expected file structure is generated."""
        output_dir = os.path.join(self.test_dir, "output")

        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "sdk-generator",
                self.spec_file,
                output_dir,
                self.constants_file,
            ],
            capture_output=True,
            text=True,
            cwd=self.original_cwd,
        )

        assert result.returncode == 0, f"CLI failed: {result.stderr}"

        # Check that output directory exists
        assert os.path.exists(output_dir), "Output directory was not created"

        # List all generated files
        generated_files = list(Path(output_dir).rglob("*"))
        file_names = [f.name for f in generated_files if f.is_file()]

        # Should have some Python files
        python_files = [f for f in file_names if f.endswith(".py")]
        assert len(python_files) > 0, "No Python files were generated"

        # Clean up the output directory
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
