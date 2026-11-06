"""Resumable downloads must not publish incomplete or mismatched checkpoints."""
import hashlib
import io
import re
import pytest
from scripts.download_sam import download_file


class RangeServer:
    def __init__(self, payload, bad_range=False):
        self.payload, self.requests, self.bad_range = payload, [], bad_range

    def open(self, request, timeout):
        start, end = map(int, re.fullmatch(r"bytes=(\d+)-(\d+)", request.headers["Range"]).groups())
        self.requests.append((start, end))
        response = io.BytesIO(self.payload[start:end+1])
        response.status = 206
        response.headers = {"Content-Range": f"bytes {start+int(self.bad_range)}-{end}/{len(self.payload)}"}
        return response


def test_resume_verified_download_and_skip_completed(tmp_path):
    payload = b"checkpoint-tensor-content" * 9
    path = tmp_path / "weights.pt"
    path.with_suffix(".pt.part").write_bytes(payload[:13])
    server = RangeServer(payload)
    digest = hashlib.sha256(payload).hexdigest()
    download_file(server, "https://example.test/weights", path, len(payload), digest, chunk_size=32)
    assert server.requests[0][0] == 13
    assert path.read_bytes() == payload and not path.with_suffix(".pt.part").exists()
    count = len(server.requests)
    download_file(server, "https://example.test/weights", path, len(payload), digest)
    assert len(server.requests) == count


def test_corrupt_or_wrong_range_never_becomes_final_checkpoint(tmp_path):
    path = tmp_path / "weights.pt"
    with pytest.raises(ValueError, match="unexpected byte range"):
        download_file(RangeServer(b"abcd", True), "https://example.test/weights", path, 4, "0" * 64)
    assert not path.exists() and not path.with_suffix(".pt.part").exists()
    with pytest.raises(ValueError, match="Checksum mismatch"):
        download_file(RangeServer(b"abcd"), "https://example.test/weights", path, 4, "0" * 64)
    assert not path.exists()
    assert path.with_suffix(".pt.part").read_bytes() == b"abcd"
