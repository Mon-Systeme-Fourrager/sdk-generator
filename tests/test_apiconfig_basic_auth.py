import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml


class TestAPIConfigBasicAuth:
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
                        "responses": {"200": {"description": "Success"}},
                    }
                }
            },
        }

        self.spec_file = os.path.join(self.test_dir, "test_spec.yaml")
        with open(self.spec_file, "w") as f:
            yaml.dump(self.test_spec, f)

    def teardown_method(self):
        os.chdir(self.original_cwd)
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def _create_constants_file(
        self,
        api_config_init_fields: dict,
        api_config_additional_headers: dict | None = None,
    ) -> str:
        if api_config_additional_headers is None:
            api_config_additional_headers = {}

        constants_content = f"""
{{% set api_config_init_fields = {api_config_init_fields} %}}

{{% macro get_api_config_init_fields() %}}
    {{{{ api_config_init_fields }}}}
{{% endmacro %}}

{{% set api_config_additional_headers = {api_config_additional_headers} %}}
{{% macro get_api_config_additional_headers() %}}
    {{{{ api_config_additional_headers }}}}
{{% endmacro %}}

{{% set endpoints_return_response = [] %}}
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

    def _get_api_config_content(self, output_dir: str) -> str:
        api_config_file = Path(output_dir) / "api_config.py"
        if api_config_file.exists():
            with open(api_config_file, "r") as f:
                return f.read()
        return ""

    def test_basic_auth_present_with_username_and_password(self):
        constants_file = self._create_constants_file(
            {"username": "str", "password": "str"}
        )
        output_dir, result = self._generate_sdk(constants_file)

        assert result.returncode == 0, f"CLI failed: {result.stderr}"

        api_config_content = self._get_api_config_content(output_dir)

        assert "def get_basic_auth(self)" in api_config_content
        assert "Authorization" in api_config_content
        assert "self.get_basic_auth()" in api_config_content
        assert "b64encode" in api_config_content

    def test_basic_auth_absent_without_username_and_password(self):
        constants_file = self._create_constants_file(
            {"api_key": "str", "tenant_id": "str"}
        )
        output_dir, result = self._generate_sdk(constants_file)

        assert result.returncode == 0, f"CLI failed: {result.stderr}"

        api_config_content = self._get_api_config_content(output_dir)

        assert "def get_basic_auth(self)" not in api_config_content
        assert "Authorization" not in api_config_content

    def test_basic_auth_absent_with_only_username(self):
        constants_file = self._create_constants_file(
            {"username": "str", "api_key": "str"}
        )
        output_dir, result = self._generate_sdk(constants_file)

        assert result.returncode == 0, f"CLI failed: {result.stderr}"

        api_config_content = self._get_api_config_content(output_dir)

        assert "def get_basic_auth(self)" not in api_config_content
        assert "Authorization" not in api_config_content

    def test_basic_auth_absent_with_only_password(self):
        constants_file = self._create_constants_file(
            {"password": "str", "api_key": "str"}
        )
        output_dir, result = self._generate_sdk(constants_file)

        assert result.returncode == 0, f"CLI failed: {result.stderr}"

        api_config_content = self._get_api_config_content(output_dir)

        assert "def get_basic_auth(self)" not in api_config_content
        assert "Authorization" not in api_config_content

    def test_get_headers_present_without_basic_auth(self):
        constants_file = self._create_constants_file({"api_key": "str"})
        output_dir, result = self._generate_sdk(constants_file)

        assert result.returncode == 0, f"CLI failed: {result.stderr}"

        api_config_content = self._get_api_config_content(output_dir)

        assert "def get_headers(self)" in api_config_content
        assert '"Content-Type": "application/json"' in api_config_content
        assert '"Accept": "application/json"' in api_config_content

    def test_additional_headers_without_basic_auth(self):
        constants_file = self._create_constants_file(
            {"api_key": "str", "tenant_id": "str"},
            {"X-API-Key": "self.api_key", "X-Tenant-ID": "self.tenant_id"},
        )
        output_dir, result = self._generate_sdk(constants_file)

        assert result.returncode == 0, f"CLI failed: {result.stderr}"

        api_config_content = self._get_api_config_content(output_dir)

        assert "def get_headers(self)" in api_config_content
        assert '"X-API-Key"' in api_config_content
        assert '"X-Tenant-ID"' in api_config_content
        assert "self.api_key" in api_config_content
        assert "self.tenant_id" in api_config_content
        assert "Authorization" not in api_config_content
        assert "def get_basic_auth(self)" not in api_config_content

    def test_basic_auth_with_additional_headers(self):
        constants_file = self._create_constants_file(
            {"username": "str", "password": "str", "wscid": "str"},
            {"WSCID": "self.wscid"},
        )
        output_dir, result = self._generate_sdk(constants_file)

        assert result.returncode == 0, f"CLI failed: {result.stderr}"

        api_config_content = self._get_api_config_content(output_dir)

        assert "def get_basic_auth(self)" in api_config_content
        assert "Authorization" in api_config_content
        assert '"WSCID"' in api_config_content
        assert "self.wscid" in api_config_content

    def test_generated_api_config_is_syntactically_valid(self):
        constants_file = self._create_constants_file({"token": "str"})
        output_dir, result = self._generate_sdk(constants_file)

        assert result.returncode == 0, f"CLI failed: {result.stderr}"

        api_config_file = Path(output_dir) / "api_config.py"
        syntax_check = subprocess.run(
            [sys.executable, "-m", "py_compile", str(api_config_file)],
            capture_output=True,
            text=True,
        )

        assert (
            syntax_check.returncode == 0
        ), f"api_config.py has syntax errors: {syntax_check.stderr}"

    def test_generated_api_config_with_basic_auth_is_syntactically_valid(self):
        constants_file = self._create_constants_file(
            {"username": "str", "password": "str"}
        )
        output_dir, result = self._generate_sdk(constants_file)

        assert result.returncode == 0, f"CLI failed: {result.stderr}"

        api_config_file = Path(output_dir) / "api_config.py"
        syntax_check = subprocess.run(
            [sys.executable, "-m", "py_compile", str(api_config_file)],
            capture_output=True,
            text=True,
        )

        assert (
            syntax_check.returncode == 0
        ), f"api_config.py has syntax errors: {syntax_check.stderr}"


class TestAPIConfigFieldValidation:
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
                        "responses": {"200": {"description": "Success"}},
                    }
                }
            },
        }

        self.spec_file = os.path.join(self.test_dir, "test_spec.yaml")
        with open(self.spec_file, "w") as f:
            yaml.dump(self.test_spec, f)

    def teardown_method(self):
        os.chdir(self.original_cwd)
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def _create_constants_file(self, api_config_init_fields: dict) -> str:
        constants_content = f"""
{{% set api_config_init_fields = {api_config_init_fields} %}}

{{% macro get_api_config_init_fields() %}}
    {{{{ api_config_init_fields }}}}
{{% endmacro %}}

{{% set api_config_additional_headers = {{}} %}}
{{% macro get_api_config_additional_headers() %}}
    {{{{ api_config_additional_headers }}}}
{{% endmacro %}}

{{% set endpoints_return_response = [] %}}
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

    def _get_api_config_content(self, output_dir: str) -> str:
        api_config_file = Path(output_dir) / "api_config.py"
        if api_config_file.exists():
            with open(api_config_file, "r") as f:
                return f.read()
        return ""

    def test_custom_fields_generated(self):
        constants_file = self._create_constants_file(
            {"api_key": "str", "tenant_id": "str"}
        )
        output_dir, result = self._generate_sdk(constants_file)

        assert result.returncode == 0, f"CLI failed: {result.stderr}"

        api_config_content = self._get_api_config_content(output_dir)

        assert "api_key: str" in api_config_content
        assert "tenant_id: str" in api_config_content

    def test_model_validator_includes_custom_fields(self):
        constants_file = self._create_constants_file(
            {"api_key": "str", "tenant_id": "str"}
        )
        output_dir, result = self._generate_sdk(constants_file)

        assert result.returncode == 0, f"CLI failed: {result.stderr}"

        api_config_content = self._get_api_config_content(output_dir)

        assert "@model_validator" in api_config_content
        assert "api_key" in api_config_content
        assert "tenant_id" in api_config_content
