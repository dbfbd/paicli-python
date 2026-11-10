"""本地 Word 文档文本解析，按正文顺序提取段落和表格。"""

from __future__ import annotations

from pathlib import Path
from zipfile import BadZipFile, ZipFile

from docx import Document
from docx.table import Table
from docx.text.paragraph import Paragraph

from paicli.tools.file_ops import FileOpResult, resolve_path


def read_docx(
    cwd: str,
    path: str,
    *,
    offset: int = 1,
    limit: int = 100,
    max_chars: int = 24_000,
    path_guard_enabled: bool = True,
) -> FileOpResult:
    """直接解析文档中的段落和表格，标记未解析对象，全程不打开或修改 Word。"""
    resolved = resolve_path(cwd, path, path_guard_enabled)
    if resolved.suffix.lower() != ".docx":
        return FileOpResult("read_document currently supports .docx files only.", is_error=True)
    if not resolved.is_file():
        return FileOpResult(f"Not a file: {resolved}", is_error=True)
    if resolved.stat().st_size > 20_000_000:
        return FileOpResult("Document exceeds the 20 MB extraction limit.", is_error=True)
    try:
        with ZipFile(resolved) as archive:
            if sum(item.file_size for item in archive.infolist()) > 100_000_000:
                return FileOpResult("Expanded document exceeds 100 MB.", is_error=True)
        document = Document(resolved)
    except (OSError, BadZipFile, KeyError, ValueError) as exc:
        return FileOpResult(f"Unable to parse .docx: {type(exc).__name__}", is_error=True)

    blocks: list[str] = []
    for item in document.iter_inner_content():
        if isinstance(item, Paragraph):
            text = item.text.strip()
            image_count = len(item._p.xpath(".//w:drawing | .//w:pict"))
            equation_count = len(item._p.xpath(".//m:oMath"))
            if image_count:
                text += f" [Embedded image objects: {image_count}; content not extracted]"
            if equation_count:
                text += f" [Equation objects: {equation_count}; formula not interpreted]"
            if text:
                blocks.append(text.strip())
        elif isinstance(item, Table):
            rows = [" | ".join(cell.text.strip() for cell in row.cells) for row in item.rows]
            blocks.append("[Table]\n" + "\n".join(rows))
    images = len(document.element.xpath(".//w:drawing | .//w:pict"))
    equations = len(document.element.xpath(".//m:oMath"))
    offset = max(1, offset)
    limit = max(1, min(limit, 500))
    max_chars = max(500, min(max_chars, 50_000))
    selected = blocks[offset - 1 : offset - 1 + limit]
    returned_range = f"{offset}-{offset + len(selected) - 1}" if selected else "none"
    more_blocks = offset - 1 + len(selected) < len(blocks)
    body = "\n\n".join(f"[{index}] {text}" for index, text in enumerate(selected, offset))
    if len(body) > max_chars:
        body = body[:max_chars] + "\n[Text truncated; request a smaller block range.]"
    header = (
        f"Document: {Path(resolved).name}\n"
        f"Body blocks: {len(blocks)}; requested offset={offset}, limit={limit}\n"
        f"Returned blocks: {returned_range}; more blocks: {more_blocks}\n"
        f"Embedded image objects: {images}; equation objects: {equations}\n"
        "Extracted body text and tables only; image content, equation layout, and page layout "
        "are not interpreted. Block numbers are not page numbers.\n\n"
    )
    if not body:
        body = "[No body text in this range.]"
    return FileOpResult(header + body, display_summary=f"Read Word document {resolved.name}")
