import sys
import zipfile
from pathlib import Path


def unzip_all(target_dir: Path) -> None:
    if not target_dir.is_dir():
        print(f"Error: Directory '{target_dir}' does not exist.")
        sys.exit(1)

    zip_files = sorted(target_dir.glob("*.zip"))

    if not zip_files:
        print(f"Error: No .zip files found in '{target_dir}'.")
        return

    print(f"Found {len(zip_files)} zip files. Starting extraction...")

    for zip_path in zip_files:
        folder_name = zip_path.stem  
        out_dir = target_dir / folder_name
        out_dir.mkdir(exist_ok=True)

        print(f"-> Extracting '{zip_path.name}' to '{folder_name}/'")

        try:
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(out_dir)
            print("   OK")
        except zipfile.BadZipFile:
            print(f"   Error: '{zip_path.name}' is not a valid zip file or it is corrupted.")
            error_dict.append(zip_path.name)
        except Exception as e:
            print(f"   Error during extraction of '{zip_path.name}': {e}")
            error_dict.append(zip_path.name)

    print("Done! All zip files have been extracted.")


if __name__ == "__main__":
    folder = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
    error_dict = []
    unzip_all(folder)
    print(error_dict)