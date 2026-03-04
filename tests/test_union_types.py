import tempfile
from pathlib import Path

import pytest
import yaml

from sdk_generator.generator import generate_sdk


@pytest.fixture
def test_spec_with_union():
    spec = {
        "openapi": "3.0.0",
        "info": {"title": "Test API", "version": "1.0.0"},
        "paths": {
            "/users": {
                "post": {
                    "operationId": "createUser",
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/UserTypes"}
                            }
                        },
                    },
                    "responses": {
                        "200": {
                            "description": "Success",
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/UserTypes"}
                                }
                            },
                        }
                    },
                }
            }
        },
        "components": {
            "schemas": {
                "User": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "integer"},
                        "name": {"type": "string"},
                        "email": {"type": "string"},
                    },
                    "required": ["id", "name", "email"],
                },
                "User2": {
                    "type": "object",
                    "properties": {
                        "userId": {"type": "integer"},
                        "username": {"type": "string"},
                        "active": {"type": "boolean"},
                    },
                    "required": ["userId", "username"],
                },
                "UserTypes": {
                    "anyOf": [
                        {"$ref": "#/components/schemas/User"},
                        {"$ref": "#/components/schemas/User2"},
                    ]
                },
            }
        },
    }
    return spec


def test_union_types_generation(test_spec_with_union):
    with tempfile.TemporaryDirectory() as tmpdir:
        spec_file = Path(tmpdir) / "test_spec.yaml"
        with open(spec_file, "w") as f:
            yaml.dump(test_spec_with_union, f)

        output_dir = Path(tmpdir) / "test_sdk"
        generate_sdk(str(spec_file), str(output_dir))

        models_dir = output_dir / "models"
        assert models_dir.exists()

        user_types_file = models_dir / "UserTypes.py"
        user2_file = models_dir / "User2.py"
        user_file = models_dir / "User.py"

        assert user_types_file.exists()
        assert user2_file.exists()
        assert user_file.exists()

        user_types_content = user_types_file.read_text()
        user2_content = user2_file.read_text()

        assert "Union[" in user_types_content or "class UserTypes" in user_types_content

        if "Union[" in user_types_content:
            assert "from typing import Union" in user_types_content

        assert "class User2(BaseModel):" in user2_content

        services_dir = output_dir / "services"
        assert services_dir.exists()

        service_files = list(services_dir.glob("*.py"))
        assert len(service_files) > 0

        service_content = ""
        for service_file in service_files:
            if service_file.name != "__init__.py":
                service_content += service_file.read_text()

        assert "def createUser" in service_content
        assert "UserTypes" in service_content


def test_split_union_types_generation(test_spec_with_union):
    with tempfile.TemporaryDirectory() as tmpdir:
        spec_file = Path(tmpdir) / "test_spec.yaml"
        with open(spec_file, "w") as f:
            yaml.dump(test_spec_with_union, f)

        output_dir = Path(tmpdir) / "test_sdk"
        generate_sdk(str(spec_file), str(output_dir))

        models_dir = output_dir / "models"
        assert models_dir.exists()

        user_types_file = models_dir / "UserTypes.py"
        assert user_types_file.exists()

        user_types_content = user_types_file.read_text()
        assert "Union[" in user_types_content or "class UserTypes" in user_types_content

        if "Union[" in user_types_content:
            assert "from typing import Union" in user_types_content

        user2_file = models_dir / "User2.py"
        assert user2_file.exists()

        user2_content = user2_file.read_text()
        assert "class User2(BaseModel):" in user2_content

        user_file = models_dir / "User.py"
        assert user_file.exists()

        if "Union[" in user_types_content:
            assert (
                ".User import User" in user_types_content
                or "from .User import User" in user_types_content
            )
            assert (
                ".User2 import User2" in user_types_content
                or "from .User2 import User2" in user_types_content
            )

        init_file = models_dir / "__init__.py"
        assert init_file.exists()

        init_content = init_file.read_text()
        assert "UserTypes" in init_content
        assert "User2" in init_content
        assert "User" in init_content


def test_union_type_usage_in_service(test_spec_with_union):
    with tempfile.TemporaryDirectory() as tmpdir:
        spec_file = Path(tmpdir) / "test_spec.yaml"
        with open(spec_file, "w") as f:
            yaml.dump(test_spec_with_union, f)

        output_dir = Path(tmpdir) / "test_sdk"
        generate_sdk(str(spec_file), str(output_dir))

        services_dir = output_dir / "services"
        assert services_dir.exists()

        service_content = ""
        for service_file in services_dir.glob("*.py"):
            if service_file.name != "__init__.py":
                service_content += service_file.read_text()

        assert service_content
        assert "from ..models" in service_content and "UserTypes" in service_content
        assert "def createUser" in service_content

        lines = service_content.split("\n")
        create_user_start = None
        for i, line in enumerate(lines):
            if "def createUser" in line:
                create_user_start = i
                break

        assert create_user_start is not None

        method_section = "\n".join(lines[create_user_start : create_user_start + 20])
        assert "UserTypes" in method_section


def test_union_type_runtime_validation(test_spec_with_union):
    with tempfile.TemporaryDirectory() as tmpdir:
        spec_file = Path(tmpdir) / "test_spec.yaml"
        with open(spec_file, "w") as f:
            yaml.dump(test_spec_with_union, f)

        output_dir = Path(tmpdir) / "test_sdk"
        generate_sdk(str(spec_file), str(output_dir))

        import sys

        modules_to_remove = [
            key for key in sys.modules.keys() if key.startswith("models")
        ]
        for mod in modules_to_remove:
            del sys.modules[mod]

        sys.path.insert(0, str(output_dir))

        try:
            from models import User, User2, UserTypes  # type: ignore

            user_data = {"id": 1, "name": "John Doe", "email": "john@example.com"}

            if hasattr(UserTypes, "model_validate"):
                user_instance = UserTypes.model_validate(user_data)
                assert user_instance is not None
            else:
                user_instance = User.model_validate(user_data)
                assert isinstance(user_instance, User)

            user2_data = {"userId": 2, "username": "jane_doe", "active": True}

            if hasattr(UserTypes, "model_validate"):
                user2_instance = UserTypes.model_validate(user2_data)
                assert user2_instance is not None
            else:
                user2_instance = User2.model_validate(user2_data)
                assert isinstance(user2_instance, User2)

        finally:
            sys.path.remove(str(output_dir))
            modules_to_remove = [
                key for key in sys.modules.keys() if key.startswith("models")
            ]
            for mod in modules_to_remove:
                del sys.modules[mod]
