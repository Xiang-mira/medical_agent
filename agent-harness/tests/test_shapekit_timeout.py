from __future__ import annotations

import subprocess


def test_shapekit_timeout_accepts_byte_streams(tmp_path, monkeypatch):
    from cli_anything.medai.core import shapekit_runner

    root = tmp_path / "ShapeKit-main"
    root.mkdir()
    (root / "main.py").write_text("")
    case = tmp_path / "input" / "case_001" / "segmentations"
    case.mkdir(parents=True)
    (case / "liver.nii.gz").touch()

    monkeypatch.setattr(
        shapekit_runner,
        "_prepare_config",
        lambda *args, **kwargs: {"status": "success"},
    )

    class FakePopen:
        pid = 999999
        returncode = None

        def __init__(self, *args, **kwargs):
            self.calls = 0

        def communicate(self, timeout=None):
            self.calls += 1
            if self.calls == 1:
                raise subprocess.TimeoutExpired(
                    cmd=["python", "main.py"],
                    timeout=1,
                    output=b"partial stdout",
                    stderr=b"partial stderr",
                )
            self.returncode = 124
            return b"partial stdout", b"partial stderr"

    monkeypatch.setattr(subprocess, "Popen", FakePopen)
    monkeypatch.setattr(shapekit_runner.os, "killpg", lambda *args, **kwargs: None)
    result = shapekit_runner.run_shapekit(
        root,
        tmp_path / "input",
        tmp_path / "output",
        tmp_path / "logs",
        auto_config=False,
        timeout_sec=1,
    )

    assert result["status"] == "failed"
    assert result["return_code"] == 124
    assert "partial stderr" in result["stderr_tail"]
    assert "Timeout after 1 seconds" in result["stderr_tail"]
