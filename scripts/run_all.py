import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run(script_name: str) -> None:
    print(f"\n$ python scripts/{script_name}")
    subprocess.run([sys.executable, str(ROOT / "scripts" / script_name)], cwd=ROOT, check=True)


def main() -> None:
    for script_name in [
        "prepare_data.py",
        "train_model.py",
        "log_mlflow.py",
        "build_feature_store.py",
        "face_blur_demo.py",
        "make_report_assets.py",
    ]:
        run(script_name)
    print("\nDone. README artifacts are ready.")


if __name__ == "__main__":
    main()
