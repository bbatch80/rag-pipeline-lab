"""The parse cache reproduces a parse exactly and never serves a stale one."""
from pathlib import Path

from raglab import parsecache
from raglab.parsing.base import Element


class _Backend:
    name = "fake-parser"
    def __init__(self): self.calls = 0
    def parse(self, path):
        self.calls += 1
        return [Element(text="Title", category="title", page=1), Element(text="| a | b |\n| --- | --- |\n| 1 | 2 |", category="table", page=2)]


def test_cache_returns_the_same_elements_and_skips_the_parser(tmp_path, monkeypatch):
    monkeypatch.setenv("RAGLAB_PARSE_CACHE", "on")  # conftest turns it off for every other test
    monkeypatch.setattr(parsecache, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(parsecache, "_library_versions", lambda name: "v=1")
    pdf = tmp_path / "doc.pdf"; pdf.write_bytes(b"%PDF fake bytes")
    backend = _Backend()
    first, cached1 = parsecache.parse(backend, pdf)
    second, cached2 = parsecache.parse(backend, pdf)
    assert (cached1, cached2) == (False, True) and backend.calls == 1
    assert first == second, "the cached parse is byte-for-byte the parser's output"


def test_cache_key_changes_with_bytes_parser_and_library_version(tmp_path, monkeypatch):
    monkeypatch.setenv("RAGLAB_PARSE_CACHE", "on")  # conftest turns it off for every other test
    monkeypatch.setattr(parsecache, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(parsecache, "_library_versions", lambda name: "v=1")
    pdf = tmp_path / "doc.pdf"; pdf.write_bytes(b"%PDF fake bytes")
    backend = _Backend()
    parsecache.parse(backend, pdf)
    pdf.write_bytes(b"%PDF changed bytes")
    assert parsecache.parse(backend, pdf)[1] is False, "new bytes -> fresh parse"
    other = _Backend(); other.name = "other-parser"
    assert parsecache.parse(other, pdf)[1] is False, "another parser -> fresh parse"
    monkeypatch.setattr(parsecache, "_library_versions", lambda name: "v=2")
    assert parsecache.parse(backend, pdf)[1] is False, "a library upgrade -> fresh parse"


def test_cache_can_be_forced_off(tmp_path, monkeypatch):
    monkeypatch.setenv("RAGLAB_PARSE_CACHE", "on")  # conftest turns it off for every other test
    monkeypatch.setattr(parsecache, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setenv("RAGLAB_PARSE_CACHE", "off")
    pdf = tmp_path / "doc.pdf"; pdf.write_bytes(b"%PDF")
    backend = _Backend()
    parsecache.parse(backend, pdf); parsecache.parse(backend, pdf)
    assert backend.calls == 2 and not (tmp_path / "cache").exists()
