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
