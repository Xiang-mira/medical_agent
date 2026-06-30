from cli_anything.medai.core.subprocess_utils import subprocess_text


def test_subprocess_text_normalizes_timeout_stream_types():
    assert subprocess_text(b"byte output") == "byte output"
    assert subprocess_text("text output") == "text output"
    assert subprocess_text(None) == ""
