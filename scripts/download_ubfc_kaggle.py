"""Download a UBFC-rPPG mirror from Kaggle (bypasses the Google Drive quota).

Google Drive throttles the official UBFC share ("Too many users have viewed or
downloaded this file recently"), which blocks ``gdown``. Kaggle hosts community
mirrors of the same UBFC-rPPG DATASET_2 (one folder per subject, each with
``vid.avi`` + ``ground_truth.txt``). This script authenticates with a Kaggle API
token (read from the environment / Colab Secrets), downloads a chosen dataset,
unzips it, and flattens whatever nesting the mirror uses so that
:func:`rppg.data.datasets.list_ubfc_subjects` finds every subject.

Set credentials first (Colab -> key icon -> add secrets, then):
    from google.colab import userdata
    import os
    os.environ["KAGGLE_USERNAME"] = userdata.get("KAGGLE_USERNAME")
    os.environ["KAGGLE_KEY"] = userdata.get("KAGGLE_KEY")

Usage:
    python scripts/download_ubfc_kaggle.py --slug ashfakyeafi/ubfc-2 --out data/UBFC
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import shutil
import subprocess
import sys

_MIN_VIDEO_BYTES = 1_000_000  # a real vid.avi is tens/hundreds of MB; tiny == bad

# Where a user-provided kaggle.json commonly lands (Colab Files upload, HOME, cwd).
_KAGGLE_JSON_CANDIDATES = (
    pathlib.Path.home() / ".kaggle" / "kaggle.json",
    pathlib.Path("/content/kaggle.json"),
    pathlib.Path("kaggle.json"),
)


def _load_creds() -> bool:
    """Ensure KAGGLE_USERNAME/KAGGLE_KEY are set, sourcing a kaggle.json if needed.

    Checks the environment first, then any common kaggle.json location. When a
    file is found it exports the two env vars and also installs the file at
    ``~/.kaggle/kaggle.json`` with 0600 perms (what the kaggle CLI expects).
    """
    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        return True
    for cand in _KAGGLE_JSON_CANDIDATES:
        if not cand.exists():
            continue
        try:
            data = json.loads(cand.read_text(encoding="utf-8"))
            os.environ["KAGGLE_USERNAME"] = str(data["username"])
            os.environ["KAGGLE_KEY"] = str(data["key"])
        except Exception:  # noqa: BLE001 - malformed file -> try the next candidate
            continue
        home_kaggle = pathlib.Path.home() / ".kaggle"
        home_kaggle.mkdir(parents=True, exist_ok=True)
        dest = home_kaggle / "kaggle.json"
        if dest.resolve() != cand.resolve():
            dest.write_text(cand.read_text(encoding="utf-8"), encoding="utf-8")
        try:
            dest.chmod(0o600)
        except OSError:
            pass
        return True
    return False


def _ensure_kaggle_installed() -> None:
    """Install the kaggle CLI into the current interpreter if it is missing."""
    try:
        import kaggle  # noqa: F401
        return
    except Exception:  # noqa: BLE001 - any import failure -> install
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "kaggle"],
                       check=True)


def _download(slug: str, raw_dir: pathlib.Path) -> None:
    """Download and unzip a Kaggle dataset into ``raw_dir`` via the kaggle CLI."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [sys.executable, "-m", "kaggle", "datasets", "download",
         "-d", slug, "-p", str(raw_dir), "--unzip"],
        check=True,
    )


def _find_subject_dirs(root: pathlib.Path) -> list[pathlib.Path]:
    """Every directory under ``root`` holding a real vid.avi + ground_truth.txt."""
    found: list[pathlib.Path] = []
    for gt in root.rglob("ground_truth.txt"):
        subject_dir = gt.parent
        video = subject_dir / "vid.avi"
        if video.exists() and video.stat().st_size > _MIN_VIDEO_BYTES:
            found.append(subject_dir)
    return sorted(found, key=lambda p: p.name)


def _flatten(raw_dir: pathlib.Path, out_dir: pathlib.Path) -> int:
    """Move each discovered subject folder to ``out_dir/<name>``; return count."""
    out_dir.mkdir(parents=True, exist_ok=True)
    subject_dirs = _find_subject_dirs(raw_dir)
    for src_dir in subject_dirs:
        dest_dir = out_dir / src_dir.name
        if dest_dir.resolve() == src_dir.resolve():
            continue
        dest_dir.mkdir(parents=True, exist_ok=True)
        for fname in ("vid.avi", "ground_truth.txt"):
            src, tgt = src_dir / fname, dest_dir / fname
            if src.exists() and not tgt.exists():
                shutil.move(str(src), str(tgt))  # move: mirrors are tens of GB
    return len(subject_dirs)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--slug", default="ashfakyeafi/ubfc-2",
                    help="Kaggle dataset slug 'owner/dataset'")
    ap.add_argument("--out", default="data/UBFC", help="flattened subject root")
    ap.add_argument("--raw", default="data/_ubfc_raw", help="raw download dir")
    args = ap.parse_args()

    if not _load_creds():
        sys.exit("No Kaggle credentials. Upload kaggle.json to /content/ (Colab "
                 "Files panel) or set KAGGLE_USERNAME and KAGGLE_KEY in the env.")

    _ensure_kaggle_installed()
    raw_dir, out_dir = pathlib.Path(args.raw), pathlib.Path(args.out)
    print(f"[kaggle] downloading {args.slug} -> {raw_dir} (this can take minutes)")
    _download(args.slug, raw_dir)
    print(f"[kaggle] flattening subject folders -> {out_dir}")
    n = _flatten(raw_dir, out_dir)
    if n == 0:
        sys.exit("Downloaded, but found no subject*/vid.avi+ground_truth.txt. "
                 f"The mirror layout may differ; inspect {raw_dir} by hand.")
    print(f"DONE: {n} UBFC subjects ready in {out_dir}")


if __name__ == "__main__":
    main()
