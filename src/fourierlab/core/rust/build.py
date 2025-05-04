import os
import subprocess
import sys
from pathlib import Path

def build_rust_extension():
    """Build the Rust extension module."""
    # Get the directory containing this script
    script_dir = Path(__file__).parent
    
    # Run cargo build
    result = subprocess.run(
        ["cargo", "build", "--release"],
        cwd=script_dir,
        capture_output=True,
        text=True,
    )
    
    if result.returncode != 0:
        print("Error building Rust extension:")
        print(result.stderr)
        sys.exit(1)
    
    # Get the target directory
    target_dir = script_dir / "target" / "release"
    
    # Find the compiled library
    if sys.platform == "win32":
        lib_name = "fourierlab_core.dll"
    elif sys.platform == "darwin":
        lib_name = "libfourierlab_core.dylib"
    else:
        lib_name = "libfourierlab_core.so"
    
    lib_path = target_dir / lib_name
    if not lib_path.exists():
        print(f"Error: Could not find compiled library at {lib_path}")
        sys.exit(1)
    
    # Create the Python package directory
    package_dir = script_dir.parent / "rust"
    package_dir.mkdir(exist_ok=True)
    
    # Copy the library to the package directory
    import shutil
    shutil.copy2(lib_path, package_dir / "fourierlab_core.so")
    
    # Create __init__.py
    with open(package_dir / "__init__.py", "w") as f:
        f.write("""\"\"\"Rust extension module for FourierLab.\"\"\"
from .fourierlab_core import *
""")
    
    print("Successfully built and installed Rust extension.")

if __name__ == "__main__":
    build_rust_extension() 