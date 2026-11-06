"""Download verified SAM 3.1 weights, using public ModelScope by default.

No ModelScope SDK or login is needed. Only data is downloaded; no model code
is executed. Hugging Face is an optional authenticated fallback.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import time
import urllib.request

REPO = "facebook/sam3.1"
REVISION = "616acbee0b9ed4177f1f389e3c13594a0a1f6398"
CHECKPOINT = "sam3.1_multiplex.pt"
CHECKPOINT_SHA256 = "0567debeec80ba4ac6369540c6c248025283cb3ff2b92827509e57e2b3541cb6"
CHECKPOINT_SIZE = 3502755717
# The pointer reconstructed from the size/SHA above has exactly this Git blob
# ID, independently exposed by Meta's official Hugging Face file listing.
OFFICIAL_POINTER_OID = "88589b6b0bee708b57010d3ed6f413476e4be255"
FILES = {
    CHECKPOINT: (CHECKPOINT_SIZE, CHECKPOINT_SHA256),
    "LICENSE": (7352, "bec48f70bd37bf8280a9d1ebf01642d26086f6122ba735baa08fe03c5a6e7448"),
    "README.md": (2003, "4835b42b1a317f5bbfb664f17c421d0a081ad91ddf9288c76c9c0d205cbc5179"),
    "config.json": (25843, "4616385e4b21f2e5e22c875b65679185cbccfa95de42542b9166f7dc3d57160f"),
}

def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_file(opener, url, path, size, expected_sha, retries=5, chunk_size=64 * 1024 * 1024):
    """Resume only a partial file; verify size/hash before publishing final name."""
    path = Path(path)
    if path.exists():
        if path.stat().st_size == size and sha256(path) == expected_sha:
            print("Already verified:", path.name, flush=True)
            return
        raise ValueError(f"Existing {path} has a different checksum; move it aside before retrying")
    part = path.with_name(path.name + ".part")
    path.parent.mkdir(parents=True, exist_ok=True)
    failures = 0
    while not part.exists() or part.stat().st_size < size:
        start = part.stat().st_size if part.exists() else 0
        end = min(size - 1, start + chunk_size - 1)
        request = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}", "Accept-Encoding": "identity"})
        try:
            with opener.open(request, timeout=60) as response:
                if response.status == 206:
                    match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", response.headers.get("Content-Range", ""))
                    if not match or tuple(map(int, match.groups())) != (start, end, size):
                        raise ValueError("Server returned an unexpected byte range; partial file was not appended")
                    expected_end = end + 1
                elif response.status == 200 and start == 0:
                    expected_end = size
                else:
                    raise ValueError("Server did not honor the resume request; partial file was not appended")
                with part.open("ab") as stream:
                    while stream.tell() < expected_end:
                        block = response.read(min(8 * 1024 * 1024, expected_end - stream.tell()))
                        if not block:
                            raise ConnectionError("Download ended early")
                        stream.write(block)
                failures = 0
                print(f"{path.name}: {expected_end}/{size} bytes ({100 * expected_end / size:.1f}%)", flush=True)
        except (OSError, ConnectionError) as exc:
            failures += 1
            if failures > retries:
                raise RuntimeError(f"Download interrupted ({type(exc).__name__}); rerun the same command to resume") from None
            print(f"Retry {failures}/{retries}: {type(exc).__name__}", flush=True)
            time.sleep(min(2 * failures, 10))
    if part.stat().st_size != size or sha256(part) != expected_sha:
        raise ValueError(f"Checksum mismatch: {part}. Move the partial file aside and retry")
    part.replace(path)
    print("Verified:", path.name, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("assets/sam3.1"))
    parser.add_argument("--source", choices=("modelscope", "huggingface"), default="modelscope")
    parser.add_argument("--use-proxy", action="store_true", help="Honor system/environment proxy settings (default: direct download)")
    args = parser.parse_args()
    pointer = f"version https://git-lfs.github.com/spec/v1\noid sha256:{CHECKPOINT_SHA256}\nsize {CHECKPOINT_SIZE}\n".encode()
    assert hashlib.sha1(b"blob " + str(len(pointer)).encode() + b"\0" + pointer).hexdigest() == OFFICIAL_POINTER_OID
    args.output.mkdir(parents=True, exist_ok=True)
    revision = REVISION
    if args.source == "modelscope":
        opener = urllib.request.build_opener() if args.use_proxy else urllib.request.build_opener(urllib.request.ProxyHandler({}))
        for name, (size, digest) in FILES.items():
            url = f"https://modelscope.cn/api/v1/models/{REPO}/repo?Revision={REVISION}&FilePath={name}"
            download_file(opener, url, args.output / name, size, digest)
    else:
        os.environ.pop("HF_HUB_OFFLINE", None)
        os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
        if not args.use_proxy:
            for name in ("http_proxy", "https_proxy", "all_proxy", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"):
                os.environ.pop(name, None)
        from huggingface_hub import HfApi, snapshot_download
        try:
            revision = HfApi().model_info(REPO, token=True).sha
            snapshot_download(REPO, revision=revision, local_dir=args.output, token=True,
                              allow_patterns=list(FILES), max_workers=1)
        except Exception as exc:
            raise SystemExit(f"Hugging Face download failed ({type(exc).__name__}). Accept model access and run hf auth login, or use --source modelscope.") from None
        for name, (size, digest) in FILES.items():
            path = args.output / name
            if not path.is_file() or path.stat().st_size != size or sha256(path) != digest:
                raise SystemExit(f"Unexpected official file version/checksum: {name}")
    manifest = {"repo_id": REPO, "source": args.source, "revision": revision,
                "checkpoint": CHECKPOINT, "bytes": CHECKPOINT_SIZE, "sha256": CHECKPOINT_SHA256,
                "official_hf_pointer_oid": OFFICIAL_POINTER_OID}
    (args.output / "SAM_MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    print("Ready. If downloading locally, upload this entire directory to the shared server storage.")

if __name__ == "__main__":
    main()
