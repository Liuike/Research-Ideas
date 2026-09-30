"""Tests for reconstructing source digests from an exact Git commit."""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

import pytest

from optimizer_resurrection.punn_manifold_comparison import _verify_source_equivalence


SOURCE_FILES = {
    "src/example.py": b"VALUE = 1\nprint(VALUE)\n",
    "configs/example.yaml": b"name: example\nseed: 3\n",
    "pyproject.toml": b"[project]\nname = 'example'\n",
}


def _git(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=root, text=True).strip()


def _write_and_commit(root: Path, files: dict[str, bytes]) -> str:
    for relative, contents in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(contents)
    _git(root, "add", "--", *files)
    subprocess.run(
        ["git", "-c", "user.name=Test", "-c", "user.email=test@example.invalid",
         "commit", "--quiet", "-m", "source fixture"],
        cwd=root,
        check=True,
    )
    return _git(root, "rev-parse", "HEAD")


def _digest(paths: list[str], contents: dict[str, bytes]) -> str:
    value = hashlib.sha256()
    for path in paths:
        value.update(path.encode("utf-8") + b"\0" + contents[path] + b"\0")
    return value.hexdigest()


def _manifest_for(files: dict[str, bytes], revision: str) -> dict:
    paths = sorted(files)
    normalized_digest = _digest(paths, files)

    def variant(lf_paths: list[str]) -> dict:
        raw_files = {
            path: content if path in lf_paths else content.replace(b"\n", b"\r\n")
            for path, content in files.items()
        }
        return {
            "default_line_ending": "CRLF",
            "lf_paths": lf_paths,
            "raw_digest": _digest(paths, raw_files),
        }

    return {
        "protocol": "punn-source-line-ending-equivalence-v1",
        "revision": revision,
        "file_count": len(paths),
        "normalized_lf_digest": normalized_digest,
        "variants": [variant(["src/example.py"]),
                     variant(["configs/example.yaml", "pyproject.toml"])],
    }


@pytest.fixture
def committed_source(tmp_path: Path) -> tuple[Path, dict[str, bytes], str]:
    root = tmp_path / "source"
    root.mkdir()
    _git(root, "init", "--quiet")
    _git(root, "config", "core.autocrlf", "false")
    revision = _write_and_commit(root, SOURCE_FILES)
    return root, SOURCE_FILES, revision


def test_source_equivalence_reconstructs_lf_and_crlf_hashes_from_commit(
    committed_source: tuple[Path, dict[str, bytes], str],
) -> None:
    root, files, revision = committed_source
    manifest = _manifest_for(files, revision)

    verified = _verify_source_equivalence(manifest, root)

    assert verified["verified"] is True
    assert verified["normalized_lf_digest"] == manifest["normalized_lf_digest"]
    assert len({variant["raw_digest"] for variant in manifest["variants"]}) == 2


def test_source_equivalence_rejects_substantive_committed_byte_change(
    committed_source: tuple[Path, dict[str, bytes], str],
) -> None:
    root, files, original_revision = committed_source
    manifest = _manifest_for(files, original_revision)
    changed_files = {**files, "src/example.py": b"VALUE = 2\nprint(VALUE)\n"}
    manifest["revision"] = _write_and_commit(root, changed_files)

    with pytest.raises(ValueError, match="normalized digest differs"):
        _verify_source_equivalence(manifest, root)


def test_source_equivalence_rejects_tampered_raw_digest(
    committed_source: tuple[Path, dict[str, bytes], str],
) -> None:
    root, files, revision = committed_source
    manifest = _manifest_for(files, revision)
    manifest["variants"][0]["raw_digest"] = "0" * 64

    with pytest.raises(ValueError, match="raw digest does not reconstruct"):
        _verify_source_equivalence(manifest, root)


def test_source_equivalence_rejects_lf_path_outside_committed_source(
    committed_source: tuple[Path, dict[str, bytes], str],
) -> None:
    root, files, revision = committed_source
    manifest = _manifest_for(files, revision)
    manifest["variants"][0]["lf_paths"] = ["README.md"]

    with pytest.raises(ValueError, match="invalid source-equivalence line-ending manifest"):
        _verify_source_equivalence(manifest, root)


def test_source_equivalence_rejects_wrong_file_count(
    committed_source: tuple[Path, dict[str, bytes], str],
) -> None:
    root, files, revision = committed_source
    manifest = _manifest_for(files, revision)
    manifest["file_count"] += 1

    with pytest.raises(ValueError, match="file count differs"):
        _verify_source_equivalence(manifest, root)
