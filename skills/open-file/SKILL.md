---
name: open-file
description: Open/read a local file by path. First choice for PDFs, DOCX, XLSX, archives, images, code, logs, JSON/CSV, and unknown file types.
---

Use when the user asks to open, inspect, read, parse, preview, summarize, or
extract a local file and the format may not be plain text. Use `read_file` only
when the file is already known plain text.

Supported now:
- Text/code/data: .txt .md .py .json .jsonl .yaml .yml .csv .tsv .log .xml .html .css .js .ts .toml .ini .cfg .sql and common code files
- Archives: .zip .tar .tgz .gz; list entries, then pass `member` to preview one file
- Documents: .pdf text layers via PyMuPDF/pypdf, .docx, .xlsx/.xlsm with optional `sheet`
- Images: .png .jpg .jpeg .webp .bmp .tif .tiff return dimensions and OCR-needed status
- Long output: follow the returned `offset` continuation hint

{"tool":"open_file","path":"C:/file.pdf","max_chars":8000}
{"tool":"open_file","path":"C:/archive.zip","max_members":80}
{"tool":"open_file","path":"C:/archive.zip","member":"docs/readme.md","max_chars":8000}
{"tool":"open_file","path":"C:/sheet.xlsx","sheet":"Sheet1","max_rows":80}
