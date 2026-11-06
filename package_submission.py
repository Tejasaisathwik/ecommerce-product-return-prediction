"""Script to package project into RegisterNumber_MLOps_Internal.zip."""

import sys
import zipfile
from pathlib import Path


def create_submission_zip(reg_number: str = "RegisterNumber"):
    root_dir = Path(__file__).resolve().parent
    zip_filename = f"{reg_number}_MLOps_Internal.zip"
    zip_filepath = root_dir / zip_filename

    print(f"[*] Packaging project into {zip_filename}...")

    exclude_dirs = {
        "__pycache__",
        ".pytest_cache",
        "venv",
        ".venv",
        "env",
        ".git",
        ".idea",
        ".vscode",
    }
    exclude_extensions = {".pyc", ".pyo", ".pyd", ".zip"}

    with zipfile.ZipFile(zip_filepath, "w", zipfile.ZIP_DEFLATED) as zipf:
        for file_path in root_dir.rglob("*"):
            if file_path.is_file():
                # Check if file is in excluded directory
                parts = file_path.relative_to(root_dir).parts
                if any(part in exclude_dirs for part in parts):
                    continue
                if file_path.suffix in exclude_extensions:
                    continue

                rel_path = file_path.relative_to(root_dir)
                zipf.write(file_path, arcname=rel_path)
                print(f"  + Added: {rel_path}")

    size_kb = zip_filepath.stat().st_size / 1024
    print("\n[+] Successfully created submission archive:")
    print(f"    Path: {zip_filepath}")
    print(f"    Size: {size_kb:.2f} KB")
    return zip_filepath


if __name__ == "__main__":
    reg_num = sys.argv[1] if len(sys.argv) > 1 else "RegisterNumber"
    create_submission_zip(reg_num)
