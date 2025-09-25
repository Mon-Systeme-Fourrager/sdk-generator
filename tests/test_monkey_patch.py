import pytest

from sdk_generator.monkey_patch import (
    apply_monkey_patch,
    extract_exportable_names,
    generate_explicit_imports,
    get_module_exports,
    patch_init_file,
)


class TestExtractExportableNames:
    """Test the extract_exportable_names function."""

    def test_extract_classes_and_functions(self, tmp_path):
        """Test extracting classes and public functions."""
        test_file = tmp_path / "test_module.py"
        test_file.write_text(
            """
class PublicClass:
    def method(self):
        pass

class AnotherClass:
    pass

def public_function():
    return "public"

def _private_function():
    return "private"

async def async_function():
    return "async"

async def _private_async():
    return "private async"

# This is just a variable
CONSTANT = "value"
"""
        )

        names = extract_exportable_names(test_file)

        # Should include public classes and functions
        assert "PublicClass" in names
        assert "AnotherClass" in names
        assert "public_function" in names
        assert "async_function" in names

        # Should not include private functions
        assert "_private_function" not in names
        assert "_private_async" not in names

        # Should not include variables/constants
        assert "CONSTANT" not in names

    def test_extract_only_top_level_definitions(self, tmp_path):
        """Test that only top-level definitions are extracted."""
        test_file = tmp_path / "test_nested.py"
        test_file.write_text(
            """
class OuterClass:
    class InnerClass:
        pass
    
    def inner_method(self):
        def nested_function():
            pass
        return nested_function

def outer_function():
    class LocalClass:
        pass
    
    def local_function():
        pass
    
    return LocalClass
"""
        )

        names = extract_exportable_names(test_file)

        # Should only include top-level definitions
        assert "OuterClass" in names
        assert "outer_function" in names

        # Should not include nested definitions
        assert "InnerClass" not in names
        assert "LocalClass" not in names
        assert "nested_function" not in names
        assert "local_function" not in names

    def test_extract_from_nonexistent_file(self, tmp_path):
        """Test handling of nonexistent files."""
        nonexistent_file = tmp_path / "nonexistent.py"
        names = extract_exportable_names(nonexistent_file)
        assert names == set()

    def test_extract_from_invalid_python(self, tmp_path):
        """Test handling of invalid Python syntax."""
        invalid_file = tmp_path / "invalid.py"
        invalid_file.write_text("def invalid syntax here:")

        names = extract_exportable_names(invalid_file)
        assert names == set()


class TestGetModuleExports:
    """Test the get_module_exports function."""

    def test_get_exports_from_module_directory(self, tmp_path):
        """Test getting exports from a module directory."""
        module_dir = tmp_path / "test_module"
        module_dir.mkdir()

        # Create __init__.py (should be ignored)
        (module_dir / "__init__.py").write_text("")

        # Create module files
        (module_dir / "models.py").write_text(
            """
class User:
    pass

class Product:
    pass

def get_user():
    pass
"""
        )

        (module_dir / "services.py").write_text(
            """
class UserService:
    pass

def create_user():
    pass
"""
        )

        exports = get_module_exports(module_dir)

        assert "models" in exports
        assert "services" in exports

        # Check models exports
        assert set(exports["models"]) == {"Product", "User", "get_user"}

        # Check services exports
        assert set(exports["services"]) == {"UserService", "create_user"}

    def test_get_exports_from_nonexistent_directory(self, tmp_path):
        """Test handling of nonexistent directory."""
        nonexistent_dir = tmp_path / "nonexistent"
        exports = get_module_exports(nonexistent_dir)
        assert exports == {}

    def test_get_exports_from_file_not_directory(self, tmp_path):
        """Test handling when path is a file, not directory."""
        test_file = tmp_path / "test_file.py"
        test_file.write_text("class Test: pass")

        exports = get_module_exports(test_file)
        assert exports == {}


class TestGenerateExplicitImports:
    """Test the generate_explicit_imports function."""

    def setup_test_package(self, tmp_path):
        """Set up a test package structure."""
        # Create api_config.py
        (tmp_path / "api_config.py").write_text(
            """
class APIConfig:
    pass

class HTTPException(Exception):
    pass

def configure_api():
    pass
"""
        )

        # Create models directory
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        (models_dir / "__init__.py").write_text("")
        (models_dir / "user.py").write_text(
            """
class User:
    pass

class UserProfile:
    pass
"""
        )

        # Create services directory
        services_dir = tmp_path / "services"
        services_dir.mkdir()
        (services_dir / "__init__.py").write_text("")
        (services_dir / "user_service.py").write_text(
            """
def get_user():
    pass

def create_user():
    pass
"""
        )

        return tmp_path

    def test_generate_imports_full_package(self, tmp_path):
        """Test generating imports for a complete package."""
        package_path = self.setup_test_package(tmp_path)

        imports = generate_explicit_imports(package_path)

        # Check that all expected imports are present
        assert (
            "from .api_config import APIConfig, HTTPException, configure_api" in imports
        )
        assert "from .models.user import User, UserProfile" in imports
        assert "from .services.user_service import create_user, get_user" in imports

    def test_generate_imports_api_config_only(self, tmp_path):
        """Test generating imports when only api_config exists."""
        (tmp_path / "api_config.py").write_text(
            """
class APIConfig:
    pass
"""
        )

        imports = generate_explicit_imports(tmp_path)

        assert imports == "# ruff: noqa: F401\nfrom .api_config import APIConfig"

    def test_generate_imports_empty_package(self, tmp_path):
        """Test generating imports for empty package."""
        imports = generate_explicit_imports(tmp_path)
        assert imports == ""


class TestPatchInitFile:
    """Test the patch_init_file function."""

    def test_patch_existing_init_file(self, tmp_path):
        """Test patching an existing __init__.py file."""
        # Set up package
        (tmp_path / "api_config.py").write_text(
            """
class APIConfig:
    pass
"""
        )

        # Create original __init__.py with wildcard imports
        init_file = tmp_path / "__init__.py"
        init_file.write_text(
            """from .api_config import *
from .models import *
"""
        )

        patch_init_file(tmp_path)

        # Check that file was patched
        content = init_file.read_text()
        assert "from .api_config import APIConfig" in content
        assert "from .api_config import *" not in content

    def test_patch_nonexistent_init_file(self, tmp_path):
        """Test patching when __init__.py doesn't exist."""
        # Should not raise an error
        patch_init_file(tmp_path)

        # Should not create the file
        assert not (tmp_path / "__init__.py").exists()

    def test_patch_empty_package(self, tmp_path):
        """Test patching when no exportable content exists."""
        init_file = tmp_path / "__init__.py"
        init_file.write_text("# Original content")

        patch_init_file(tmp_path)

        # Should not modify the file when no exports are found
        content = init_file.read_text()
        assert content == "# Original content"


class TestApplyMonkeyPatch:
    """Test the apply_monkey_patch function."""

    def test_apply_patch_to_existing_directory(self, tmp_path):
        """Test applying patch to existing directory."""
        # Set up package
        (tmp_path / "api_config.py").write_text(
            """
class APIConfig:
    pass
"""
        )

        init_file = tmp_path / "__init__.py"
        init_file.write_text("from .api_config import *")

        apply_monkey_patch(str(tmp_path))

        # Check that patch was applied
        content = init_file.read_text()
        assert "from .api_config import APIConfig" in content

    def test_apply_patch_to_nonexistent_directory(self, tmp_path):
        """Test applying patch to nonexistent directory."""
        nonexistent_path = tmp_path / "nonexistent"

        # Should not raise an error
        apply_monkey_patch(str(nonexistent_path))


class TestMakeModelsIntoDir:
    """Test the make_models_into_dir function."""

    def test_split_models_with_dependencies(self, tmp_path):
        """Test splitting models.py with model dependencies."""
        from sdk_generator.monkey_patch import make_models_into_dir

        # Create models.py with interdependent classes
        models_file = tmp_path / "models.py"
        models_content = """from typing import List, Optional
from pydantic import BaseModel

class User(BaseModel):
    id: int
    name: str
    email: Optional[str] = None

class Product(BaseModel):
    id: int
    name: str
    price: float
    owner: User

class Order(BaseModel):
    id: int
    user: User
    products: List[Product]
    total: float
"""
        models_file.write_text(models_content)

        # Apply the function
        make_models_into_dir(tmp_path)

        # Check that models directory was created
        models_dir = tmp_path / "models"
        assert models_dir.exists()
        assert models_dir.is_dir()

        # Check that original models.py was removed
        assert not models_file.exists()

        # Check individual model files
        user_file = models_dir / "User.py"
        product_file = models_dir / "Product.py"
        order_file = models_dir / "Order.py"
        init_file = models_dir / "__init__.py"

        assert user_file.exists()
        assert product_file.exists()
        assert order_file.exists()
        assert init_file.exists()

        # Check User.py content (no dependencies)
        user_content = user_file.read_text()
        assert "class User(BaseModel):" in user_content
        assert "from typing import List, Optional" in user_content
        assert "from pydantic import BaseModel" in user_content
        assert "from ." not in user_content  # No model imports

        # Check Product.py content (depends on User)
        product_content = product_file.read_text()
        assert "class Product(BaseModel):" in product_content
        assert "from .User import User" in product_content
        assert "owner: User" in product_content

        # Check Order.py content (depends on both User and Product)
        order_content = order_file.read_text()
        assert "class Order(BaseModel):" in order_content
        assert "from .Product import Product" in order_content
        assert "from .User import User" in order_content

        # Check __init__.py content
        init_content = init_file.read_text()
        assert "from .Order import Order" in init_content
        assert "from .Product import Product" in init_content
        assert "from .User import User" in init_content
        assert '"Order"' in init_content
        assert '"Product"' in init_content
        assert '"User"' in init_content

    def test_split_models_with_utilities(self, tmp_path):
        """Test splitting models.py with utility functions."""
        from sdk_generator.monkey_patch import make_models_into_dir

        models_file = tmp_path / "models.py"
        models_content = """from typing import List
from pydantic import BaseModel

def utility_function():
    return "utility"

CONSTANT_VALUE = "test"

class TestModel(BaseModel):
    id: int
    name: str
"""
        models_file.write_text(models_content)

        make_models_into_dir(tmp_path)

        models_dir = tmp_path / "models"
        utils_file = models_dir / "_utils.py"
        init_file = models_dir / "__init__.py"
        test_model_file = models_dir / "TestModel.py"

        # Check that utilities file was created
        assert utils_file.exists()
        utils_content = utils_file.read_text()
        assert "def utility_function():" in utils_content
        assert 'CONSTANT_VALUE = "test"' in utils_content
        assert "from typing import List" in utils_content
        assert "from pydantic import BaseModel" in utils_content

        # Check that TestModel.py doesn't contain utilities
        test_model_content = test_model_file.read_text()
        assert "def utility_function():" not in test_model_content
        assert "CONSTANT_VALUE" not in test_model_content
        assert "class TestModel(BaseModel):" in test_model_content

        # Check that __init__.py imports utilities
        init_content = init_file.read_text()
        assert "from . import _utils" in init_content
        assert "from .TestModel import TestModel" in init_content

    def test_split_models_no_classes(self, tmp_path):
        """Test that function returns early if no classes found."""
        from sdk_generator.monkey_patch import make_models_into_dir

        models_file = tmp_path / "models.py"
        models_content = """# Just imports and functions, no classes
from typing import List

def utility_function():
    return "utility"

CONSTANT = "value"
"""
        models_file.write_text(models_content)

        make_models_into_dir(tmp_path)

        # Should not create models directory if no classes
        models_dir = tmp_path / "models"
        assert not models_dir.exists()

        # Original file should still exist
        assert models_file.exists()

    def test_split_models_already_directory(self, tmp_path):
        """Test that function returns early if models directory already exists."""
        from sdk_generator.monkey_patch import make_models_into_dir

        models_file = tmp_path / "models.py"
        models_file.write_text(
            """
class TestModel:
    pass
"""
        )

        # Create models directory first
        models_dir = tmp_path / "models"
        models_dir.mkdir()

        make_models_into_dir(tmp_path)

        # Original file should still exist
        assert models_file.exists()
        # Directory should be empty (no files created)
        assert len(list(models_dir.iterdir())) == 0

    def test_split_models_no_file(self, tmp_path):
        """Test that function returns early if models.py doesn't exist."""
        from sdk_generator.monkey_patch import make_models_into_dir

        # Don't create models.py
        make_models_into_dir(tmp_path)

        # Should not create models directory
        models_dir = tmp_path / "models"
        assert not models_dir.exists()


class TestIntegration:
    """Integration tests for the monkey patch functionality."""

    def test_real_world_package_structure(self, tmp_path):
        """Test with a realistic package structure similar to generated SDKs."""
        # Create a structure similar to what the SDK generator produces
        package_path = tmp_path / "generated_sdk"
        package_path.mkdir()

        # api_config.py
        (package_path / "api_config.py").write_text(
            """
from base64 import b64encode
from pydantic import BaseModel

class APIConfig(BaseModel):
    username: str
    password: str
    
    def get_headers(self):
        return {}

class HTTPException(Exception):
    def __init__(self, status_code: int, message: str):
        self.status_code = status_code
        self.message = message
"""
        )

        # models/
        models_dir = package_path / "models"
        models_dir.mkdir()
        (models_dir / "__init__.py").write_text("")

        (models_dir / "user_models.py").write_text(
            """
from pydantic import BaseModel

class User(BaseModel):
    id: int
    name: str

class CreateUserRequest(BaseModel):
    name: str
"""
        )

        # services/
        services_dir = package_path / "services"
        services_dir.mkdir()
        (services_dir / "__init__.py").write_text("")

        (services_dir / "user_service.py").write_text(
            """
from typing import List
from ..models.user_models import User, CreateUserRequest

def get_users() -> List[User]:
    pass

def create_user(request: CreateUserRequest) -> User:
    pass

def _internal_helper():
    pass
"""
        )

        # Original __init__.py with wildcard imports
        (package_path / "__init__.py").write_text(
            """from .api_config import *
from .models import *
from .services import *
"""
        )

        # Apply monkey patch
        apply_monkey_patch(str(package_path))

        # Verify the result
        init_content = (package_path / "__init__.py").read_text()

        # Should have explicit imports
        assert "from .api_config import APIConfig, HTTPException" in init_content
        assert "from .models.user_models import CreateUserRequest, User" in init_content
        assert (
            "from .services.user_service import create_user, get_users" in init_content
        )

        # Should not have wildcard imports
        assert "from .api_config import *" not in init_content
        assert "from .models import *" not in init_content
        assert "from .services import *" not in init_content

        # Should not include private functions
        assert "_internal_helper" not in init_content

    def test_imports_are_importable(self, tmp_path):
        """Test that the generated imports are actually importable."""
        # This test verifies that the monkey patch produces valid Python
        package_path = tmp_path / "test_package"
        package_path.mkdir()

        # Create a simple package
        (package_path / "api_config.py").write_text(
            """
class TestClass:
    pass

def test_function():
    return "test"
"""
        )

        (package_path / "__init__.py").write_text("from .api_config import *")

        # Apply monkey patch
        apply_monkey_patch(str(package_path))

        # Read the generated content
        init_content = (package_path / "__init__.py").read_text()

        # Verify it's valid Python syntax
        import ast

        try:
            ast.parse(init_content)
        except SyntaxError:
            pytest.fail("Generated __init__.py contains invalid Python syntax")

        # Verify expected content
        assert "from .api_config import TestClass, test_function" in init_content

    def test_complete_models_splitting_workflow(self, tmp_path):
        """Test the complete workflow including models splitting and import fixing."""
        # Create a realistic package structure with models.py
        package_path = tmp_path / "test_sdk"
        package_path.mkdir()

        # Create models.py with interconnected models and utilities
        models_file = package_path / "models.py"
        models_content = '''from typing import List, Optional, Union
from pydantic import BaseModel
from datetime import datetime

def validate_email(email: str) -> bool:
    """Utility function to validate email."""
    return "@" in email

class BaseEntity(BaseModel):
    """Base class for all entities."""
    id: int
    created_at: datetime
    updated_at: Optional[datetime] = None

class User(BaseEntity):
    name: str
    email: str
    is_active: bool = True

class Category(BaseEntity):
    name: str
    description: Optional[str] = None

class Product(BaseEntity):
    name: str
    price: float
    category: Category
    owner: User

class OrderItem(BaseModel):
    product: Product
    quantity: int
    unit_price: float

class Order(BaseEntity):
    user: User
    items: List[OrderItem]
    total_amount: float
    status: str = "pending"
'''
        models_file.write_text(models_content)

        # Create services that use models
        services_dir = package_path / "services"
        services_dir.mkdir()
        (services_dir / "__init__.py").write_text("")

        user_service_file = services_dir / "user_service.py"
        user_service_content = '''from typing import List, Optional
from ..models import *

def get_user(user_id: int) -> Optional[User]:
    """Get user by ID."""
    pass

def create_user(name: str, email: str) -> User:
    """Create a new user."""
    pass

def get_user_orders(user_id: int) -> List[Order]:
    """Get all orders for a user."""
    pass
'''
        user_service_file.write_text(user_service_content)

        # Create __init__.py with wildcard imports
        init_file = package_path / "__init__.py"
        init_file.write_text(
            """from .models import *
from .services import *
"""
        )

        # Apply the complete monkey patch
        apply_monkey_patch(str(package_path))

        # Verify models directory was created
        models_dir = package_path / "models"
        assert models_dir.exists()
        assert not models_file.exists()  # Original should be gone

        # Check individual model files exist
        expected_files = [
            "BaseEntity.py",
            "User.py",
            "Category.py",
            "Product.py",
            "OrderItem.py",
            "Order.py",
            "_utils.py",
            "__init__.py",
        ]

        actual_files = [f.name for f in models_dir.iterdir()]
        for expected_file in expected_files:
            assert expected_file in actual_files, f"Missing {expected_file}"

        # Verify model dependencies are correctly handled
        product_content = (models_dir / "Product.py").read_text()
        assert "from .Category import Category" in product_content
        assert "from .User import User" in product_content
        assert (
            "from .BaseEntity import BaseEntity" not in product_content
        )  # Should inherit BaseEntity

        order_content = (models_dir / "Order.py").read_text()
        assert "from .OrderItem import OrderItem" in order_content
        assert "from .User import User" in order_content

        orderitem_content = (models_dir / "OrderItem.py").read_text()
        assert "from .Product import Product" in orderitem_content

        # Verify utilities are in separate file
        utils_content = (models_dir / "_utils.py").read_text()
        assert "def validate_email" in utils_content
        assert "from typing import List, Optional, Union" in utils_content

        # Verify models __init__.py imports utilities
        models_init_content = (models_dir / "__init__.py").read_text()
        assert "from . import _utils" in models_init_content

        # Verify all models are exported
        for model in [
            "BaseEntity",
            "User",
            "Category",
            "Product",
            "OrderItem",
            "Order",
        ]:
            assert f"from .{model} import {model}" in models_init_content
            assert f'"{model}"' in models_init_content

        # Verify service imports were fixed
        user_service_content_fixed = user_service_file.read_text()
        assert "from ..models import *" not in user_service_content_fixed

        # Should have specific imports for used models
        expected_imports = [
            "from ..models.User import User",
            "from ..models.Order import Order",
        ]
        for expected_import in expected_imports:
            assert expected_import in user_service_content_fixed

        # Verify main __init__.py was updated
        main_init_content = init_file.read_text()
        assert "from .models import *" not in main_init_content
        assert "from .services import *" not in main_init_content

        # Should have explicit imports
        model_imports = [
            "from .models.BaseEntity import BaseEntity",
            "from .models.Category import Category",
            "from .models.Order import Order",
            "from .models.OrderItem import OrderItem",
            "from .models.Product import Product",
            "from .models.User import User",
        ]
        for model_import in model_imports:
            assert model_import in main_init_content

        # Test that all generated code is syntactically valid
        import ast

        for py_file in models_dir.glob("*.py"):
            content = py_file.read_text()
            try:
                ast.parse(content)
            except SyntaxError as e:
                pytest.fail(f"Generated file {py_file.name} has invalid syntax: {e}")

        # Test main __init__.py syntax
        try:
            ast.parse(main_init_content)
        except SyntaxError as e:
            pytest.fail(f"Generated __init__.py has invalid syntax: {e}")

        # Test service file syntax
        try:
            ast.parse(user_service_content_fixed)
        except SyntaxError as e:
            pytest.fail(f"Fixed service file has invalid syntax: {e}")


class TestGenerateMissingModels:
    """Test the generate_missing_models functionality."""

    def test_generate_missing_models_from_services(self, tmp_path):
        """Test generating missing model classes based on service type annotations."""
        from sdk_generator.monkey_patch import generate_missing_models

        # Create package structure
        package_path = tmp_path / "test_sdk"
        package_path.mkdir()

        # Create models directory with existing Model
        models_dir = package_path / "models"
        models_dir.mkdir()
        (models_dir / "__init__.py").write_text("")
        (models_dir / "Model.py").write_text(
            """from pydantic import BaseModel
from typing import Any

class Model(BaseModel):
    __root__: Any
"""
        )

        # Create services directory with service that references missing models
        services_dir = package_path / "services"
        services_dir.mkdir()
        (services_dir / "__init__.py").write_text("")

        service_content = '''from typing import Any
from ..api_config import APIConfig, HTTPException

def get_user(user_id: int) -> UserDetail:
    """Get user by ID."""
    pass

def create_user(data: CreateUserRequest) -> UserDetail:
    """Create a new user.""" 
    pass

def get_users() -> UsersList:
    """Get all users."""
    pass
'''
        (services_dir / "user_service.py").write_text(service_content)

        # Apply generate_missing_models
        generate_missing_models(package_path)

        # Check that missing models were generated
        expected_models = ["UserDetail", "CreateUserRequest", "UsersList"]
        for model_name in expected_models:
            model_file = models_dir / f"{model_name}.py"
            assert model_file.exists(), f"Missing model file: {model_name}.py"

            content = model_file.read_text()
            assert f"class {model_name}(BaseModel):" in content
            assert '"Auto-generated model class."' in content
            assert "__root__: Any" in content

        # Check that models/__init__.py was updated
        init_content = (models_dir / "__init__.py").read_text()
        for model_name in expected_models:
            assert f"from .{model_name} import {model_name}" in init_content
            assert f'"{model_name}"' in init_content

        # Check that existing Model is still there
        assert "from .Model import Model" in init_content

    def test_generate_missing_models_filters_builtin_types(self, tmp_path):
        """Test that builtin types and keywords are not generated as models."""
        from sdk_generator.monkey_patch import generate_missing_models

        package_path = tmp_path / "test_sdk"
        package_path.mkdir()

        models_dir = package_path / "models"
        models_dir.mkdir()
        (models_dir / "__init__.py").write_text("")

        services_dir = package_path / "services"
        services_dir.mkdir()

        # Service with builtin types, keywords, and APIConfig that should be filtered out
        service_content = """from typing import Any, Optional, List, Dict
from ..api_config import APIConfig, HTTPException

def get_data() -> Dict[str, Any]:
    pass

def process_data(data: List[str], config: APIConfig) -> Optional[bool]:
    pass

def get_none() -> None:
    pass

def get_valid_model() -> ValidModel:
    pass
"""
        (services_dir / "test_service.py").write_text(service_content)

        generate_missing_models(package_path)

        # Only ValidModel should be generated
        assert (models_dir / "ValidModel.py").exists()

        # These should NOT be generated
        should_not_exist = [
            "Dict.py",
            "str.py",
            "Any.py",
            "List.py",
            "Optional.py",
            "bool.py",
            "None.py",
            "APIConfig.py",
            "HTTPException.py",
        ]
        for filename in should_not_exist:
            assert not (
                models_dir / filename
            ).exists(), f"Should not generate: {filename}"

    def test_generate_missing_models_no_services(self, tmp_path):
        """Test that function handles missing services directory gracefully."""
        from sdk_generator.monkey_patch import generate_missing_models

        package_path = tmp_path / "test_sdk"
        package_path.mkdir()

        models_dir = package_path / "models"
        models_dir.mkdir()
        (models_dir / "__init__.py").write_text("")

        # No services directory - should not crash
        generate_missing_models(package_path)

    def test_generate_missing_models_no_models_dir(self, tmp_path):
        """Test that function handles missing models directory gracefully."""
        from sdk_generator.monkey_patch import generate_missing_models

        package_path = tmp_path / "test_sdk"
        package_path.mkdir()

        services_dir = package_path / "services"
        services_dir.mkdir()

        # No models directory - should not crash
        generate_missing_models(package_path)

    def test_generate_missing_models_existing_models_not_overwritten(self, tmp_path):
        """Test that existing model files are not overwritten."""
        from sdk_generator.monkey_patch import generate_missing_models

        package_path = tmp_path / "test_sdk"
        package_path.mkdir()

        models_dir = package_path / "models"
        models_dir.mkdir()
        (models_dir / "__init__.py").write_text("")

        # Create existing model
        existing_model_content = """from pydantic import BaseModel

class ExistingModel(BaseModel):
    name: str
    value: int
"""
        (models_dir / "ExistingModel.py").write_text(existing_model_content)

        services_dir = package_path / "services"
        services_dir.mkdir()

        # Service that references existing model
        service_content = """def get_existing() -> ExistingModel:
    pass
"""
        (services_dir / "test_service.py").write_text(service_content)

        generate_missing_models(package_path)

        # Existing model should not be overwritten
        actual_content = (models_dir / "ExistingModel.py").read_text()
        assert actual_content == existing_model_content
