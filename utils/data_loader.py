import os

def read_file(path):
    ext = path.split(".")[-1].lower()
    if ext == "txt":
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    elif ext == "docx":
        from docx import Document
        doc = Document(path)
        return "\n".join([p.text for p in doc.paragraphs])
    elif ext == "pdf":
        import PyPDF2
        text = ""
        with open(path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                text += page.extract_text() or ""
        return text
    return ""
