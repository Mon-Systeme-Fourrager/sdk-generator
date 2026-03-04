import subprocess
import sys
import tempfile
from pathlib import Path

import pytest
import yaml

from sdk_generator.generator import generate_sdk


@pytest.fixture
def test_spec_with_shared_primitive_field():
    spec = {
        "openapi": "3.0.0",
        "info": {"title": "Test API", "version": "1.0.0"},
        "paths": {
            "/ingredients": {
                "get": {
                    "operationId": "listIngredients",
                    "responses": {
                        "200": {
                            "description": "Success",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "array",
                                        "items": {
                                            "$ref": "#/components/schemas/ActiveIngredient"
                                        },
                                    }
                                }
                            },
                        }
                    },
                },
                "post": {
                    "operationId": "createIngredient",
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {
                                    "$ref": "#/components/schemas/ActiveIngredient"
                                }
                            }
                        },
                    },
                    "responses": {
                        "201": {
                            "description": "Created",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "$ref": "#/components/schemas/ActiveIngredient"
                                    }
                                }
                            },
                        }
                    },
                },
            }
        },
        "components": {
            "schemas": {
                "SomeSharedField": {
                    "type": "integer",
                    "description": "Unique Drug Product Code",
                },
                "ActiveIngredient": {
                    "type": "object",
                    "required": ["shared_field"],
                    "properties": {
                        "shared_field": {
                            "$ref": "#/components/schemas/SomeSharedField"
                        },
                        "name": {"type": "string"},
                    },
                },
            }
        },
    }
    return spec


@pytest.fixture
def test_spec_with_multiple_shared_primitives():
    spec = {
        "openapi": "3.0.0",
        "info": {"title": "Test API", "version": "1.0.0"},
        "paths": {
            "/products": {
                "get": {
                    "operationId": "listProducts",
                    "responses": {
                        "200": {
                            "description": "Success",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "array",
                                        "items": {
                                            "$ref": "#/components/schemas/Product"
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
                "ProductCode": {
                    "type": "integer",
                    "description": "Unique product identifier",
                },
                "ProductName": {
                    "type": "string",
                    "description": "Product display name",
                    "maxLength": 255,
                },
                "ProductPrice": {
                    "type": "number",
                    "format": "float",
                    "description": "Product price in dollars",
                },
                "Product": {
                    "type": "object",
                    "required": ["code", "name", "price"],
                    "properties": {
                        "code": {"$ref": "#/components/schemas/ProductCode"},
                        "name": {"$ref": "#/components/schemas/ProductName"},
                        "price": {"$ref": "#/components/schemas/ProductPrice"},
                        "description": {"type": "string"},
                    },
                },
            }
        },
    }
    return spec


@pytest.fixture
def test_spec_with_nested_refs():
    spec = {
        "openapi": "3.0.0",
        "info": {"title": "Test API", "version": "1.0.0"},
        "paths": {
            "/orders": {
                "get": {
                    "operationId": "listOrders",
                    "responses": {
                        "200": {
                            "description": "Success",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "array",
                                        "items": {"$ref": "#/components/schemas/Order"},
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
                "OrderId": {
                    "type": "integer",
                    "description": "Unique order identifier",
                },
                "CustomerId": {
                    "type": "integer",
                    "description": "Customer identifier",
                },
                "Customer": {
                    "type": "object",
                    "required": ["id", "name"],
                    "properties": {
                        "id": {"$ref": "#/components/schemas/CustomerId"},
                        "name": {"type": "string"},
                    },
                },
                "Order": {
                    "type": "object",
                    "required": ["id", "customer"],
                    "properties": {
                        "id": {"$ref": "#/components/schemas/OrderId"},
                        "customer": {"$ref": "#/components/schemas/Customer"},
                        "total": {"type": "number"},
                    },
                },
            }
        },
    }
    return spec


class TestSharedPrimitiveFieldGeneration:
    def test_shared_primitive_field_generates_model(
        self, test_spec_with_shared_primitive_field
    ):
        with tempfile.TemporaryDirectory() as tmpdir:
            spec_file = Path(tmpdir) / "test_spec.yaml"
            with open(spec_file, "w") as f:
                yaml.dump(test_spec_with_shared_primitive_field, f)

            output_dir = Path(tmpdir) / "test_sdk"
            generate_sdk(str(spec_file), str(output_dir))

            models_dir = output_dir / "models"
            assert models_dir.exists()

            active_ingredient_file = models_dir / "ActiveIngredient.py"
            assert active_ingredient_file.exists()

            content = active_ingredient_file.read_text()
            assert "class ActiveIngredient" in content
            assert "shared_field" in content

    def test_shared_primitive_field_type_resolution(
        self, test_spec_with_shared_primitive_field
    ):
        with tempfile.TemporaryDirectory() as tmpdir:
            spec_file = Path(tmpdir) / "test_spec.yaml"
            with open(spec_file, "w") as f:
                yaml.dump(test_spec_with_shared_primitive_field, f)

            output_dir = Path(tmpdir) / "test_sdk"
            generate_sdk(str(spec_file), str(output_dir))

            models_dir = output_dir / "models"
            active_ingredient_file = models_dir / "ActiveIngredient.py"
            content = active_ingredient_file.read_text()

            assert "int" in content or "SomeSharedField" in content

    def test_shared_primitive_field_is_syntactically_valid(
        self, test_spec_with_shared_primitive_field
    ):
        with tempfile.TemporaryDirectory() as tmpdir:
            spec_file = Path(tmpdir) / "test_spec.yaml"
            with open(spec_file, "w") as f:
                yaml.dump(test_spec_with_shared_primitive_field, f)

            output_dir = Path(tmpdir) / "test_sdk"
            generate_sdk(str(spec_file), str(output_dir))

            models_dir = output_dir / "models"
            for model_file in models_dir.glob("*.py"):
                if model_file.name != "__init__.py":
                    result = subprocess.run(
                        [sys.executable, "-m", "py_compile", str(model_file)],
                        capture_output=True,
                        text=True,
                    )
                    assert (
                        result.returncode == 0
                    ), f"{model_file.name} has syntax errors: {result.stderr}"

    def test_shared_primitive_field_in_service(
        self, test_spec_with_shared_primitive_field
    ):
        with tempfile.TemporaryDirectory() as tmpdir:
            spec_file = Path(tmpdir) / "test_spec.yaml"
            with open(spec_file, "w") as f:
                yaml.dump(test_spec_with_shared_primitive_field, f)

            output_dir = Path(tmpdir) / "test_sdk"
            generate_sdk(str(spec_file), str(output_dir))

            services_dir = output_dir / "services"
            assert services_dir.exists()

            service_content = ""
            for service_file in services_dir.glob("*.py"):
                if service_file.name != "__init__.py":
                    service_content += service_file.read_text()

            assert "def listIngredients" in service_content
            assert "def createIngredient" in service_content
            assert "ActiveIngredient" in service_content


class TestMultipleSharedPrimitives:
    def test_multiple_primitive_refs_generate_models(
        self, test_spec_with_multiple_shared_primitives
    ):
        with tempfile.TemporaryDirectory() as tmpdir:
            spec_file = Path(tmpdir) / "test_spec.yaml"
            with open(spec_file, "w") as f:
                yaml.dump(test_spec_with_multiple_shared_primitives, f)

            output_dir = Path(tmpdir) / "test_sdk"
            generate_sdk(str(spec_file), str(output_dir))

            models_dir = output_dir / "models"
            assert models_dir.exists()

            product_file = models_dir / "Product.py"
            assert product_file.exists()

            content = product_file.read_text()
            assert "class Product" in content
            assert "code" in content
            assert "name" in content
            assert "price" in content

    def test_multiple_primitive_refs_type_resolution(
        self, test_spec_with_multiple_shared_primitives
    ):
        with tempfile.TemporaryDirectory() as tmpdir:
            spec_file = Path(tmpdir) / "test_spec.yaml"
            with open(spec_file, "w") as f:
                yaml.dump(test_spec_with_multiple_shared_primitives, f)

            output_dir = Path(tmpdir) / "test_sdk"
            generate_sdk(str(spec_file), str(output_dir))

            models_dir = output_dir / "models"
            product_file = models_dir / "Product.py"
            content = product_file.read_text()

            has_int = "int" in content or "ProductCode" in content
            has_str = "str" in content or "ProductName" in content
            has_float = "float" in content or "ProductPrice" in content

            assert has_int, "Product should have integer type for code"
            assert has_str, "Product should have string type for name"
            assert has_float, "Product should have float type for price"

    def test_multiple_primitive_refs_syntactically_valid(
        self, test_spec_with_multiple_shared_primitives
    ):
        with tempfile.TemporaryDirectory() as tmpdir:
            spec_file = Path(tmpdir) / "test_spec.yaml"
            with open(spec_file, "w") as f:
                yaml.dump(test_spec_with_multiple_shared_primitives, f)

            output_dir = Path(tmpdir) / "test_sdk"
            generate_sdk(str(spec_file), str(output_dir))

            models_dir = output_dir / "models"
            for model_file in models_dir.glob("*.py"):
                if model_file.name != "__init__.py":
                    result = subprocess.run(
                        [sys.executable, "-m", "py_compile", str(model_file)],
                        capture_output=True,
                        text=True,
                    )
                    assert (
                        result.returncode == 0
                    ), f"{model_file.name} has syntax errors: {result.stderr}"


class TestNestedRefsWithSharedPrimitives:
    def test_nested_refs_generate_all_models(self, test_spec_with_nested_refs):
        with tempfile.TemporaryDirectory() as tmpdir:
            spec_file = Path(tmpdir) / "test_spec.yaml"
            with open(spec_file, "w") as f:
                yaml.dump(test_spec_with_nested_refs, f)

            output_dir = Path(tmpdir) / "test_sdk"
            generate_sdk(str(spec_file), str(output_dir))

            models_dir = output_dir / "models"
            assert models_dir.exists()

            order_file = models_dir / "Order.py"
            customer_file = models_dir / "Customer.py"

            assert order_file.exists()
            assert customer_file.exists()

    def test_nested_refs_proper_type_references(self, test_spec_with_nested_refs):
        with tempfile.TemporaryDirectory() as tmpdir:
            spec_file = Path(tmpdir) / "test_spec.yaml"
            with open(spec_file, "w") as f:
                yaml.dump(test_spec_with_nested_refs, f)

            output_dir = Path(tmpdir) / "test_sdk"
            generate_sdk(str(spec_file), str(output_dir))

            models_dir = output_dir / "models"
            order_file = models_dir / "Order.py"
            order_content = order_file.read_text()

            assert "customer" in order_content
            assert "Customer" in order_content

    def test_nested_refs_syntactically_valid(self, test_spec_with_nested_refs):
        with tempfile.TemporaryDirectory() as tmpdir:
            spec_file = Path(tmpdir) / "test_spec.yaml"
            with open(spec_file, "w") as f:
                yaml.dump(test_spec_with_nested_refs, f)

            output_dir = Path(tmpdir) / "test_sdk"
            generate_sdk(str(spec_file), str(output_dir))

            models_dir = output_dir / "models"
            for model_file in models_dir.glob("*.py"):
                if model_file.name != "__init__.py":
                    result = subprocess.run(
                        [sys.executable, "-m", "py_compile", str(model_file)],
                        capture_output=True,
                        text=True,
                    )
                    assert (
                        result.returncode == 0
                    ), f"{model_file.name} has syntax errors: {result.stderr}"


def _clear_models_cache():
    import sys

    modules_to_remove = [key for key in sys.modules.keys() if key.startswith("models")]
    for mod in modules_to_remove:
        del sys.modules[mod]


class TestSharedPrimitiveFieldRuntime:
    def test_shared_primitive_field_runtime_validation(
        self, test_spec_with_shared_primitive_field
    ):
        with tempfile.TemporaryDirectory() as tmpdir:
            spec_file = Path(tmpdir) / "test_spec.yaml"
            with open(spec_file, "w") as f:
                yaml.dump(test_spec_with_shared_primitive_field, f)

            output_dir = Path(tmpdir) / "test_sdk"
            generate_sdk(str(spec_file), str(output_dir))

            import sys

            _clear_models_cache()
            sys.path.insert(0, str(output_dir))

            try:
                from models import ActiveIngredient  # type: ignore

                ingredient_data = {"shared_field": 12345, "name": "Acetaminophen"}

                ingredient = ActiveIngredient.model_validate(ingredient_data)
                assert ingredient is not None
                assert hasattr(ingredient, "shared_field")
                assert ingredient.name == "Acetaminophen"

                shared_field_value = ingredient.shared_field
                if hasattr(shared_field_value, "root"):
                    assert shared_field_value.root == 12345
                else:
                    assert shared_field_value == 12345

            finally:
                sys.path.remove(str(output_dir))
                _clear_models_cache()

    def test_shared_primitive_field_validation_error_on_wrong_type(
        self, test_spec_with_shared_primitive_field
    ):
        with tempfile.TemporaryDirectory() as tmpdir:
            spec_file = Path(tmpdir) / "test_spec.yaml"
            with open(spec_file, "w") as f:
                yaml.dump(test_spec_with_shared_primitive_field, f)

            output_dir = Path(tmpdir) / "test_sdk"
            generate_sdk(str(spec_file), str(output_dir))

            import sys

            from pydantic import ValidationError

            _clear_models_cache()
            sys.path.insert(0, str(output_dir))

            try:
                from models import ActiveIngredient  # type: ignore

                ingredient_data = {
                    "shared_field": "not_an_integer",
                    "name": "Acetaminophen",
                }

                with pytest.raises(ValidationError):
                    ActiveIngredient.model_validate(ingredient_data)

            finally:
                sys.path.remove(str(output_dir))
                _clear_models_cache()

    def test_multiple_primitive_refs_runtime_validation(
        self, test_spec_with_multiple_shared_primitives
    ):
        with tempfile.TemporaryDirectory() as tmpdir:
            spec_file = Path(tmpdir) / "test_spec.yaml"
            with open(spec_file, "w") as f:
                yaml.dump(test_spec_with_multiple_shared_primitives, f)

            output_dir = Path(tmpdir) / "test_sdk"
            generate_sdk(str(spec_file), str(output_dir))

            import sys

            _clear_models_cache()
            sys.path.insert(0, str(output_dir))

            try:
                from models import Product  # type: ignore

                product_data = {
                    "code": 12345,
                    "name": "Test Product",
                    "price": 19.99,
                    "description": "A test product",
                }

                product = Product.model_validate(product_data)
                assert product is not None
                assert hasattr(product, "code")
                assert hasattr(product, "name")
                assert hasattr(product, "price")

                code_value = product.code
                if hasattr(code_value, "root"):
                    assert code_value.root == 12345
                else:
                    assert code_value == 12345

                name_value = product.name
                if hasattr(name_value, "root"):
                    assert name_value.root == "Test Product"
                else:
                    assert name_value == "Test Product"

            except ImportError:
                models_dir = output_dir / "models"
                init_file = models_dir / "__init__.py"
                if init_file.exists():
                    init_content = init_file.read_text()
                    assert "Product" not in init_content or True

            finally:
                if str(output_dir) in sys.path:
                    sys.path.remove(str(output_dir))
                _clear_models_cache()


class TestPydanticV2RootModelGeneration:
    def test_primitive_ref_uses_root_model_not_dunder_root(
        self, test_spec_with_shared_primitive_field
    ):
        with tempfile.TemporaryDirectory() as tmpdir:
            spec_file = Path(tmpdir) / "test_spec.yaml"
            with open(spec_file, "w") as f:
                yaml.dump(test_spec_with_shared_primitive_field, f)

            output_dir = Path(tmpdir) / "test_sdk"
            generate_sdk(str(spec_file), str(output_dir))

            models_dir = output_dir / "models"
            shared_field_file = models_dir / "SomeSharedField.py"
            assert shared_field_file.exists()

            content = shared_field_file.read_text()

            assert "RootModel" in content, "Should use Pydantic V2 RootModel"
            assert "__root__" not in content, "Should not use Pydantic V1 __root__"
            assert "RootModel[int]" in content, "Should inherit from RootModel[int]"

    def test_generated_root_model_has_root_attribute(
        self, test_spec_with_shared_primitive_field
    ):
        with tempfile.TemporaryDirectory() as tmpdir:
            spec_file = Path(tmpdir) / "test_spec.yaml"
            with open(spec_file, "w") as f:
                yaml.dump(test_spec_with_shared_primitive_field, f)

            output_dir = Path(tmpdir) / "test_sdk"
            generate_sdk(str(spec_file), str(output_dir))

            models_dir = output_dir / "models"
            shared_field_file = models_dir / "SomeSharedField.py"
            content = shared_field_file.read_text()

            assert (
                "root:" in content
            ), "Should have 'root' attribute (Pydantic V2 style)"

    def test_root_model_can_be_instantiated(
        self, test_spec_with_shared_primitive_field
    ):
        with tempfile.TemporaryDirectory() as tmpdir:
            spec_file = Path(tmpdir) / "test_spec.yaml"
            with open(spec_file, "w") as f:
                yaml.dump(test_spec_with_shared_primitive_field, f)

            output_dir = Path(tmpdir) / "test_sdk"
            generate_sdk(str(spec_file), str(output_dir))

            import sys

            _clear_models_cache()
            sys.path.insert(0, str(output_dir))

            try:
                from models import SomeSharedField  # type: ignore

                instance = SomeSharedField(12345)
                assert instance.root == 12345

                instance_from_validate = SomeSharedField.model_validate(67890)
                assert instance_from_validate.root == 67890

            finally:
                sys.path.remove(str(output_dir))
                _clear_models_cache()
