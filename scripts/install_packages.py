"""Install with bounded retries for download failures, retaining pip's hash checks."""
import argparse
from pathlib import Path
import subprocess
import sys
import time


DOWNLOAD_ERRORS = (
    "do not match the hashes", "hash mismatch", "incomplete-download",
    "readtimeout", "timed out", "connection broken", "connection reset",
    "connection aborted", "connectionerror", "remote end closed",
    "sslerror", "ssl_read", "unexpected eof", "proxyerror",
    "max retries exceeded", "temporary failure in name resolution",
    "failed to establish a new connection", "network is unreachable",
)


def install(requirements, index_url, log_path):
    """Stream diagnostics; retry downloads up to three times, never bypass hashes."""
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        for attempt in range(1, 4):
            heading = f"Install attempt {attempt}/3 (log: {log_path})"
            print(heading, flush=True)
            log.write(heading + "\n")
            command = [sys.executable, "-m", "pip", "install", "--no-input",
                       "--disable-pip-version-check", "--progress-bar", "off",
                       "--retries", "5", "--timeout", "60", "--index-url", index_url]
            if attempt > 1:
                command.append("--no-cache-dir")
            retryable = False
            with subprocess.Popen(command + list(requirements), stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT, text=True, errors="replace") as process:
                for line in process.stdout:
                    print(line, end="", flush=True)
                    log.write(line)
                    log.flush()
                    retryable |= any(error in line.lower() for error in DOWNLOAD_ERRORS)
                status = process.wait()
            if status == 0:
                return 0
            if not retryable or attempt == 3:
                print(f"Installation stopped. Full output: {log_path}", file=sys.stderr)
                return status
            print("Download failed; retrying without pip's cache. Hash checks remain enabled.",
                  flush=True)
            time.sleep(2)
    return 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("torch", "packages"))
    parser.add_argument("--index-url", required=True)
    parser.add_argument("--constraint")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    requirements = (["torch==2.10.0", "torchvision==0.25.0"] if args.stage == "torch" else
                    ["-r", str(root / "requirements.txt"),
                     "-r", str(root / "requirements-model.txt"), "pytest>=8"])
    if args.constraint:
        requirements.extend(["--constraint", args.constraint])
    return install(requirements, args.index_url, root / ".bootstrap-linux" / f"pip-{args.stage}.log")


if __name__ == "__main__":
    sys.exit(main())
