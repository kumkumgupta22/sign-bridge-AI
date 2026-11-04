import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from backend.app.conversation import SQLiteStore  # noqa: E402


def test_store_roundtrip():
    s = SQLiteStore(":memory:")
    s.add("sign", "I need water.", "WATER", 0.9)
    s.add("speech", "Do you need help?")
    h = s.history()
    assert [m["speaker"] for m in h] == ["sign", "speech"]
    assert h[0]["confidence"] == 0.9
    s.clear()
    assert s.history() == []
