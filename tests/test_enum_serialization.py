import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml


class TestEnumSerialization:
    def setup_method(self):
        self.test_dir = tempfile.mkdtemp()
        self.original_cwd = os.getcwd()
        self.constants_file = Path(__file__).parent / "constants.jinja2"

    def teardown_method(self):
        os.chdir(self.original_cwd)
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def _create_spec_with_enum(
        self, enum_values: list, enum_type: str = "string"
    ) -> str:
        spec = {
            "openapi": "3.0.0",
            "info": {"title": "Test API", "version": "1.0.0"},
            "paths": {
                "/items": {
                    "get": {
                        "operationId": "listItems",
                        "parameters": [
                            {
                                "name": "status",
                                "in": "query",
                                "schema": {"$ref": "#/components/schemas/ItemStatus"},
                            }
                        ],
                        "responses": {
                            "200": {
                                "description": "Success",
                                "content": {
                                    "application/json": {
                                        "schema": {
                                            "type": "array",
                                            "items": {
                                                "$ref": "#/components/schemas/Item"
                                            },
                                        }
                                    }
                                },
                            }
                        },
                    },
                    "post": {
                        "operationId": "createItem",
                        "requestBody": {
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/Item"}
                                }
                            }
                        },
                        "responses": {"201": {"description": "Created"}},
                    },
                },
            },
            "components": {
                "schemas": {
                    "ItemStatus": {
                        "type": enum_type,
                        "enum": enum_values,
                    },
                    "Item": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "integer"},
                            "name": {"type": "string"},
                            "status": {"$ref": "#/components/schemas/ItemStatus"},
                        },
                        "required": ["name", "status"],
                    },
                }
            },
        }

        spec_file = os.path.join(self.test_dir, "test_spec.yaml")
        with open(spec_file, "w") as f:
            yaml.dump(spec, f)
        return spec_file

    def _generate_sdk(self, spec_file: str) -> tuple[str, subprocess.CompletedProcess]:
        output_dir = os.path.join(self.test_dir, "output")
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "sdk_generator",
                spec_file,
                output_dir,
                "--constants-template-path",
                str(self.constants_file),
            ],
            capture_output=True,
            text=True,
            cwd=self.original_cwd,
        )
        return output_dir, result

    def _get_models_content(self, output_dir: str) -> str:
        models_dir = Path(output_dir) / "models"
        content = ""
        if models_dir.exists():
            for model_file in models_dir.glob("*.py"):
                with open(model_file, "r") as f:
                    content += f.read() + "\n"
        return content

    def _get_service_content(self, output_dir: str) -> str:
        service_files = list(Path(output_dir).rglob("*service.py"))
        content = ""
        for service_file in service_files:
            with open(service_file, "r") as f:
                content += f.read()
        return content

    def test_string_enum_generated_in_models(self):
        spec_file = self._create_spec_with_enum(["pending", "active", "completed"])
        output_dir, result = self._generate_sdk(spec_file)

        assert result.returncode == 0, f"CLI failed: {result.stderr}"

        models_content = self._get_models_content(output_dir)

        assert "class ItemStatus" in models_content
        assert "Enum" in models_content or "enum" in models_content.lower()
        assert "pending" in models_content
        assert "active" in models_content
        assert "completed" in models_content

    def test_enum_used_in_model_property(self):
        spec_file = self._create_spec_with_enum(["draft", "published", "archived"])
        output_dir, result = self._generate_sdk(spec_file)

        assert result.returncode == 0, f"CLI failed: {result.stderr}"

        models_content = self._get_models_content(output_dir)

        assert "class Item" in models_content
        assert "status" in models_content
        assert "ItemStatus" in models_content

    def test_enum_as_query_parameter(self):
        spec_file = self._create_spec_with_enum(["new", "processing", "done"])
        output_dir, result = self._generate_sdk(spec_file)

        assert result.returncode == 0, f"CLI failed: {result.stderr}"

        service_content = self._get_service_content(output_dir)

        assert "def listItems(" in service_content
        assert "status" in service_content

    def test_integer_enum_generated(self):
        spec = {
            "openapi": "3.0.0",
            "info": {"title": "Test API", "version": "1.0.0"},
            "paths": {
                "/priorities": {
                    "get": {
                        "operationId": "getPriorities",
                        "responses": {
                            "200": {
                                "description": "Success",
                                "content": {
                                    "application/json": {
                                        "schema": {
                                            "$ref": "#/components/schemas/Priority"
                                        }
                                    }
                                },
                            }
                        },
                    }
                }
            },
            "components": {
                "schemas": {
                    "Priority": {
                        "type": "integer",
                        "enum": [1, 2, 3, 4, 5],
                    }
                }
            },
        }

        spec_file = os.path.join(self.test_dir, "test_spec.yaml")
        with open(spec_file, "w") as f:
            yaml.dump(spec, f)

        output_dir, result = self._generate_sdk(spec_file)

        assert result.returncode == 0, f"CLI failed: {result.stderr}"

        models_content = self._get_models_content(output_dir)

        assert "Priority" in models_content

    def test_enum_with_special_characters(self):
        spec_file = self._create_spec_with_enum(
            ["in-progress", "on_hold", "not_started"]
        )
        output_dir, result = self._generate_sdk(spec_file)

        assert result.returncode == 0, f"CLI failed: {result.stderr}"

        models_content = self._get_models_content(output_dir)

        assert "class ItemStatus" in models_content
        assert "in-progress" in models_content or "in_progress" in models_content

    def test_enum_in_request_body_serialization(self):
        spec_file = self._create_spec_with_enum(["low", "medium", "high"])
        output_dir, result = self._generate_sdk(spec_file)

        assert result.returncode == 0, f"CLI failed: {result.stderr}"

        service_content = self._get_service_content(output_dir)
        models_content = self._get_models_content(output_dir)

        assert "def createItem(" in service_content
        assert "class ItemStatus" in models_content
        assert "class Item" in models_content

    def test_multiple_enums_in_schema(self):
        spec = {
            "openapi": "3.0.0",
            "info": {"title": "Test API", "version": "1.0.0"},
            "paths": {
                "/tasks": {
                    "get": {
                        "operationId": "listTasks",
                        "responses": {
                            "200": {
                                "description": "Success",
                                "content": {
                                    "application/json": {
                                        "schema": {
                                            "type": "array",
                                            "items": {
                                                "$ref": "#/components/schemas/Task"
                                            },
                                        }
                                    }
                                },
                            }
                        },
                    }
                }
            },
            "components": {
                "schemas": {
                    "TaskStatus": {
                        "type": "string",
                        "enum": ["todo", "in_progress", "done"],
                    },
                    "TaskPriority": {
                        "type": "string",
                        "enum": ["low", "medium", "high", "critical"],
                    },
                    "Task": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "integer"},
                            "title": {"type": "string"},
                            "status": {"$ref": "#/components/schemas/TaskStatus"},
                            "priority": {"$ref": "#/components/schemas/TaskPriority"},
                        },
                        "required": ["title", "status", "priority"],
                    },
                }
            },
        }

        spec_file = os.path.join(self.test_dir, "test_spec.yaml")
        with open(spec_file, "w") as f:
            yaml.dump(spec, f)

        output_dir, result = self._generate_sdk(spec_file)

        assert result.returncode == 0, f"CLI failed: {result.stderr}"

        models_content = self._get_models_content(output_dir)

        assert "class TaskStatus" in models_content
        assert "class TaskPriority" in models_content
        assert "todo" in models_content
        assert "in_progress" in models_content
        assert "done" in models_content
        assert "low" in models_content
        assert "medium" in models_content
        assert "high" in models_content
        assert "critical" in models_content

    def test_optional_enum_property(self):
        spec = {
            "openapi": "3.0.0",
            "info": {"title": "Test API", "version": "1.0.0"},
            "paths": {
                "/items": {
                    "get": {
                        "operationId": "listItems",
                        "responses": {
                            "200": {
                                "description": "Success",
                                "content": {
                                    "application/json": {
                                        "schema": {
                                            "type": "array",
                                            "items": {
                                                "$ref": "#/components/schemas/Item"
                                            },
                                        }
                                    }
                                },
                            }
                        },
                    }
                }
            },
            "components": {
                "schemas": {
                    "ItemCategory": {
                        "type": "string",
                        "enum": ["electronics", "clothing", "food"],
                    },
                    "Item": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "integer"},
                            "name": {"type": "string"},
                            "category": {"$ref": "#/components/schemas/ItemCategory"},
                        },
                        "required": ["name"],
                    },
                }
            },
        }

        spec_file = os.path.join(self.test_dir, "test_spec.yaml")
        with open(spec_file, "w") as f:
            yaml.dump(spec, f)

        output_dir, result = self._generate_sdk(spec_file)

        assert result.returncode == 0, f"CLI failed: {result.stderr}"

        models_content = self._get_models_content(output_dir)

        assert "class ItemCategory" in models_content
        assert "class Item" in models_content
        assert "category" in models_content
        assert (
            "Optional" in models_content
            or "None" in models_content
            or "| None" in models_content
        )


class TestEnumImportsAndUsage:
    def setup_method(self):
        self.test_dir = tempfile.mkdtemp()
        self.original_cwd = os.getcwd()
        self.constants_file = Path(__file__).parent / "constants.jinja2"

    def teardown_method(self):
        os.chdir(self.original_cwd)
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_enum_import_in_models(self):
        spec = {
            "openapi": "3.0.0",
            "info": {"title": "Test API", "version": "1.0.0"},
            "paths": {
                "/status": {
                    "get": {
                        "operationId": "getStatus",
                        "responses": {
                            "200": {
                                "description": "Success",
                                "content": {
                                    "application/json": {
                                        "schema": {
                                            "$ref": "#/components/schemas/Status"
                                        }
                                    }
                                },
                            }
                        },
                    }
                }
            },
            "components": {
                "schemas": {
                    "Status": {
                        "type": "string",
                        "enum": ["ok", "error", "warning"],
                    }
                }
            },
        }

        spec_file = os.path.join(self.test_dir, "test_spec.yaml")
        with open(spec_file, "w") as f:
            yaml.dump(spec, f)

        output_dir = os.path.join(self.test_dir, "output")
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "sdk_generator",
                spec_file,
                output_dir,
                "--constants-template-path",
                str(self.constants_file),
            ],
            capture_output=True,
            text=True,
            cwd=self.original_cwd,
        )

        assert result.returncode == 0, f"CLI failed: {result.stderr}"

        models_dir = Path(output_dir) / "models"
        assert models_dir.exists()

        models_content = ""
        for model_file in models_dir.glob("*.py"):
            with open(model_file, "r") as f:
                models_content += f.read()

        assert (
            "from enum import Enum" in models_content
            or "import enum" in models_content
            or "Enum" in models_content
        )

    def test_generated_sdk_is_syntactically_valid(self):
        spec = {
            "openapi": "3.0.0",
            "info": {"title": "Test API", "version": "1.0.0"},
            "paths": {
                "/items": {
                    "post": {
                        "operationId": "createItem",
                        "requestBody": {
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/Item"}
                                }
                            }
                        },
                        "responses": {
                            "201": {
                                "description": "Created",
                                "content": {
                                    "application/json": {
                                        "schema": {"$ref": "#/components/schemas/Item"}
                                    }
                                },
                            }
                        },
                    }
                }
            },
            "components": {
                "schemas": {
                    "ItemType": {
                        "type": "string",
                        "enum": ["type_a", "type_b", "type_c"],
                    },
                    "Item": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "integer"},
                            "item_type": {"$ref": "#/components/schemas/ItemType"},
                        },
                        "required": ["item_type"],
                    },
                }
            },
        }

        spec_file = os.path.join(self.test_dir, "test_spec.yaml")
        with open(spec_file, "w") as f:
            yaml.dump(spec, f)

        output_dir = os.path.join(self.test_dir, "output")
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "sdk_generator",
                spec_file,
                output_dir,
                "--constants-template-path",
                str(self.constants_file),
            ],
            capture_output=True,
            text=True,
            cwd=self.original_cwd,
        )

        assert result.returncode == 0, f"CLI failed: {result.stderr}"

        models_dir = Path(output_dir) / "models"
        model_files = list(models_dir.glob("*.py")) if models_dir.exists() else []
        for model_file in model_files:
            if model_file.name != "__init__.py":
                syntax_check = subprocess.run(
                    [sys.executable, "-m", "py_compile", str(model_file)],
                    capture_output=True,
                    text=True,
                )
                assert (
                    syntax_check.returncode == 0
                ), f"{model_file} has syntax errors: {syntax_check.stderr}"

        service_files = list(Path(output_dir).rglob("*service.py"))
        for service_file in service_files:
            syntax_check = subprocess.run(
                [sys.executable, "-m", "py_compile", str(service_file)],
                capture_output=True,
                text=True,
            )
            assert (
                syntax_check.returncode == 0
            ), f"{service_file} has syntax errors: {syntax_check.stderr}"
