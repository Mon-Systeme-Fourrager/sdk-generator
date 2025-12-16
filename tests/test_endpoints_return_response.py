import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml


class TestEndpointsReturnResponse:
    def setup_method(self):
        self.test_dir = tempfile.mkdtemp()
        self.original_cwd = os.getcwd()

        self.test_spec = {
            "openapi": "3.0.0",
            "info": {"title": "Test API", "version": "1.0.0"},
            "paths": {
                "/users": {
                    "get": {
                        "operationId": "listUsers",
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
                },
                "/download": {
                    "get": {
                        "operationId": "downloadFile",
                        "responses": {
                            "200": {
                                "description": "File download",
                                "content": {
                                    "application/octet-stream": {
                                        "schema": {"type": "string", "format": "binary"}
                                    }
                                },
                            }
                        },
                    }
                },
                "/status": {
                    "get": {
                        "operationId": "getStatus",
                        "responses": {"200": {"description": "Success"}},
                    }
                },
            },
            "components": {
                "schemas": {
                    "User": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "integer"},
                            "name": {"type": "string"},
                        },
                    },
                }
            },
        }

        self.spec_file = os.path.join(self.test_dir, "test_spec.yaml")
        with open(self.spec_file, "w") as f:
            yaml.dump(self.test_spec, f)

    def teardown_method(self):
        os.chdir(self.original_cwd)
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def _create_constants_file(self, endpoints_return_response: list[str]) -> str:
        constants_content = f"""
{{% set api_config_init_fields = {{"username": "str", "password": "str"}} %}}

{{% macro get_api_config_init_fields() %}}
    {{{{ api_config_init_fields }}}}
{{% endmacro %}}

{{% set api_config_additional_headers = {{}} %}}
{{% macro get_api_config_additional_headers() %}}
    {{{{ api_config_additional_headers }}}}
{{% endmacro %}}

{{% set endpoints_return_response = {endpoints_return_response} %}}
"""
        constants_file = os.path.join(self.test_dir, "constants.jinja2")
        with open(constants_file, "w") as f:
            f.write(constants_content)
        return constants_file

    def _generate_sdk(
        self, constants_file: str
    ) -> tuple[str, subprocess.CompletedProcess]:
        output_dir = os.path.join(self.test_dir, "output")
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "sdk_generator",
                self.spec_file,
                output_dir,
                "--constants-template-path",
                constants_file,
            ],
            capture_output=True,
            text=True,
            cwd=self.original_cwd,
        )
        return output_dir, result

    def _get_service_content(self, output_dir: str) -> str:
        service_files = list(Path(output_dir).rglob("*service.py"))
        content = ""
        for service_file in service_files:
            with open(service_file, "r") as f:
                content += f.read()
        return content

    def test_empty_endpoints_return_response_returns_typed(self):
        constants_file = self._create_constants_file([])
        output_dir, result = self._generate_sdk(constants_file)

        assert result.returncode == 0, f"CLI failed: {result.stderr}"

        service_content = self._get_service_content(output_dir)

        assert "def listUsers(" in service_content
        assert "-> List[User]" in service_content or "-> list[User]" in service_content
        assert "def downloadFile(" in service_content
        assert "def getStatus(" in service_content
        assert "-> None" in service_content
        assert (
            "return response" not in service_content
            or "return response.json()" in service_content
        )

    def test_single_endpoint_returns_response(self):
        constants_file = self._create_constants_file(["downloadFile"])
        output_dir, result = self._generate_sdk(constants_file)

        assert result.returncode == 0, f"CLI failed: {result.stderr}"

        service_content = self._get_service_content(output_dir)

        assert "def downloadFile(" in service_content
        assert (
            "-> Response" in service_content
            or "-> requests.Response" in service_content
        )

        lines = service_content.split("\n")
        in_download_file = False
        found_return_response = False
        for line in lines:
            if "def downloadFile(" in line:
                in_download_file = True
            elif in_download_file and line.startswith("def "):
                break
            elif (
                in_download_file
                and "return response" in line
                and "return response.json()" not in line
            ):
                found_return_response = True
                break

        assert found_return_response, "downloadFile should return response directly"

    def test_multiple_endpoints_return_response(self):
        constants_file = self._create_constants_file(["downloadFile", "getStatus"])
        output_dir, result = self._generate_sdk(constants_file)

        assert result.returncode == 0, f"CLI failed: {result.stderr}"

        service_content = self._get_service_content(output_dir)

        assert "def downloadFile(" in service_content
        assert "def getStatus(" in service_content

        lines = service_content.split("\n")
        download_returns_response = False
        status_returns_response = False

        for line in lines:
            if "def downloadFile(" in line:
                if "-> Response" in line or "-> requests.Response" in line:
                    download_returns_response = True
            elif "def getStatus(" in line:
                if "-> Response" in line or "-> requests.Response" in line:
                    status_returns_response = True

        assert (
            download_returns_response
        ), "downloadFile should have Response return type"
        assert status_returns_response, "getStatus should have Response return type"

    def test_non_matching_endpoint_unchanged(self):
        constants_file = self._create_constants_file(["nonExistentEndpoint"])
        output_dir, result = self._generate_sdk(constants_file)

        assert result.returncode == 0, f"CLI failed: {result.stderr}"

        service_content = self._get_service_content(output_dir)

        assert "def listUsers(" in service_content
        assert "-> List[User]" in service_content or "-> list[User]" in service_content

        lines = service_content.split("\n")
        for line in lines:
            if "def listUsers(" in line:
                assert "-> Response" not in line

    def test_mixed_endpoints_some_return_response(self):
        constants_file = self._create_constants_file(["downloadFile"])
        output_dir, result = self._generate_sdk(constants_file)

        assert result.returncode == 0, f"CLI failed: {result.stderr}"

        service_content = self._get_service_content(output_dir)

        lines = service_content.split("\n")
        list_users_return_type = None
        download_file_return_type = None

        for line in lines:
            if "def listUsers(" in line:
                if "-> Response" in line or "-> requests.Response" in line:
                    list_users_return_type = "Response"
                elif "-> List[User]" in line or "-> list[User]" in line:
                    list_users_return_type = "List[User]"
            elif "def downloadFile(" in line:
                if "-> Response" in line or "-> requests.Response" in line:
                    download_file_return_type = "Response"

        assert (
            list_users_return_type == "List[User]"
        ), "listUsers should return List[User]"
        assert (
            download_file_return_type == "Response"
        ), "downloadFile should return Response"


class TestDefaultConstantsEndpointsReturnResponse:
    def setup_method(self):
        self.test_dir = tempfile.mkdtemp()
        self.original_cwd = os.getcwd()

        self.test_spec = {
            "openapi": "3.0.0",
            "info": {"title": "Test API", "version": "1.0.0"},
            "paths": {
                "/test": {
                    "get": {
                        "operationId": "getTest",
                        "responses": {
                            "200": {
                                "description": "Success",
                                "content": {
                                    "application/json": {"schema": {"type": "string"}}
                                },
                            }
                        },
                    }
                }
            },
        }

        self.spec_file = os.path.join(self.test_dir, "test_spec.yaml")
        with open(self.spec_file, "w") as f:
            yaml.dump(self.test_spec, f)

        self.default_constants = (
            Path(__file__).parent.parent
            / "sdk_generator"
            / "templates"
            / "constants-default.jinja2"
        )

    def teardown_method(self):
        os.chdir(self.original_cwd)
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_default_constants_has_empty_endpoints_return_response(self):
        with open(self.default_constants, "r") as f:
            content = f.read()

        assert "endpoints_return_response" in content
        assert "{% set endpoints_return_response = [] %}" in content

    def test_generation_with_default_constants_works(self):
        output_dir = os.path.join(self.test_dir, "output")

        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "sdk_generator",
                self.spec_file,
                output_dir,
                "--constants-template-path",
                str(self.default_constants),
            ],
            capture_output=True,
            text=True,
            cwd=self.original_cwd,
        )

        assert result.returncode == 0, f"CLI failed: {result.stderr}"

        service_files = list(Path(output_dir).rglob("*service.py"))
        assert len(service_files) > 0

        service_content = ""
        for service_file in service_files:
            with open(service_file, "r") as f:
                service_content += f.read()

        assert "def getTest(" in service_content
        assert (
            "-> Response" not in service_content
            or "-> Response"
            not in service_content.split("def getTest(")[1].split("\n")[0]
        )
