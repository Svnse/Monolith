from __future__ import annotations

import zipfile
from pathlib import Path

import core.skill_runtime as skill_runtime
from core.file_readers import open_file


def _ctx(tmp_path: Path) -> skill_runtime.ToolExecutionContext:
    return skill_runtime.ToolExecutionContext(archive_dir=tmp_path)


def test_open_file_reads_text_file(tmp_path: Path) -> None:
    sample = tmp_path / "notes.md"
    sample.write_text("# Title\nbody", encoding="utf-8")

    result = skill_runtime.execute_open_file({"path": str(sample), "max_chars": 1000}, _ctx(tmp_path))

    assert "[open_file:" in result
    assert "kind=text" in result
    assert "# Title" in result
    assert "body" in result


def test_open_file_lists_zip_and_previews_text_member(tmp_path: Path) -> None:
    archive = tmp_path / "bundle.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("docs/readme.md", "hello from zip")
        zf.writestr("bin/blob.bin", b"\x00\x01")

    listed = open_file(archive, max_members=10)
    assert "kind=zip" in listed
    assert "docs/readme.md" in listed
    assert "open_file(path=" in listed

    preview = open_file(archive, member="docs/readme.md", max_chars=1000)
    assert "kind=zip-member" in preview
    assert "hello from zip" in preview


def test_open_file_reads_minimal_docx(tmp_path: Path) -> None:
    docx_path = tmp_path / "note.docx"
    content_types_xml = """<?xml version="1.0" encoding="UTF-8"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
</Types>
"""
    rels_xml = """<?xml version="1.0" encoding="UTF-8"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
</Relationships>
"""
    document_xml = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:body>
    <w:p><w:r><w:t>Hello docx</w:t></w:r></w:p>
    <w:p><w:r><w:t>Second line</w:t></w:r></w:p>
  </w:body>
</w:document>
"""
    with zipfile.ZipFile(docx_path, "w") as zf:
        zf.writestr("[Content_Types].xml", content_types_xml)
        zf.writestr("_rels/.rels", rels_xml)
        zf.writestr("word/document.xml", document_xml)

    result = open_file(docx_path)

    assert "kind=docx" in result
    assert "Hello docx" in result
    assert "Second line" in result


def test_open_file_reports_image_ocr_backend_status(tmp_path: Path) -> None:
    image_path = tmp_path / "shot.png"
    image_path.write_bytes(b"\x89PNG\r\n\x1a\n")

    result = open_file(image_path)

    assert "kind=image" in result
    assert "OCR backend" in result


def test_open_file_unknown_binary_returns_metadata(tmp_path: Path) -> None:
    binary = tmp_path / "data.bin"
    binary.write_bytes(b"\x00\x01\x02\x03")

    result = open_file(binary)

    assert "kind=binary" in result
    assert "No text extractor" in result
