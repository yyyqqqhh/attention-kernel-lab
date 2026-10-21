"""Build a stable source archive without caches, environments or local credentials."""

import argparse
import hashlib
from pathlib import Path
import zipfile


TOP_LEVEL = {"README.md", "README.zh-CN.md", "pyproject.toml", ".gitignore", "NOTICE.md"}
DIRECTORIES = {"src", "tests", "tools", "docs", "requirements", ".github", "results"}


def source_files(root):
    files = [root / name for name in TOP_LEVEL if (root / name).is_file()]
    for name in DIRECTORIES:
        for path in (root / name).rglob("*"):
            generated = any(part == "__pycache__" or part.endswith(".egg-info") for part in path.parts)
            if path.is_file() and not path.is_symlink() and not generated:
                if path.suffix not in (".pyc", ".pyo"):
                    files.append(path)
    return sorted(files, key=lambda path: path.relative_to(root).as_posix())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = arguments.output or root.parent / "attention-kernel-lab-source.zip"
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    files = source_files(root)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            name = "attention-kernel-lab/" + path.relative_to(root).as_posix()
            info = zipfile.ZipInfo(name, date_time=(2026, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, path.read_bytes())
    with zipfile.ZipFile(output) as archive:
        failure = archive.testzip()
        if failure is not None:
            raise RuntimeError(f"archive CRC failed: {failure}")
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    checksum = output.with_suffix(output.suffix + ".sha256")
    checksum.write_text(f"{digest}  {output.name}\n", encoding="ascii")
    print(f"{len(files)} source files; {output.stat().st_size} bytes")
    print(output)
    print(f"SHA256 {digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
