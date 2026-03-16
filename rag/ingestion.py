# rag/ingestion.py
import fitz
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config.constants import CHUNK_SIZE, CHUNK_OVERLAP


def extract_text_from_pdf(file_path: str) -> list[dict]:
    doc = fitz.open(file_path)
    pages = []
    num_pages = len(doc)
    for page_num, page in enumerate(doc):
        text = page.get_text()
        if text.strip():
            pages.append({"text": text, "page": page_num + 1})
    doc.close()
    print(f"[ingestion] PDF: {num_pages} pages extracted")
    return pages


def extract_text_from_txt(file_path: str) -> list[dict]:
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()
    # treat entire txt as one "page"
    return [{"text": text, "page": 1}]


def extract_text_from_docx(file_path: str) -> list[dict]:
    try:
        import docx
    except ImportError:
        raise ImportError("python-docx not installed. Run: pip install python-docx")

    doc = docx.Document(file_path)
    pages = []
    current_text = ""
    page_num = 1

    for para in doc.paragraphs:
        current_text += para.text + "\n"
        # simulate page break every 3000 chars
        if len(current_text) >= 3000:
            pages.append({"text": current_text.strip(), "page": page_num})
            current_text = ""
            page_num += 1

    if current_text.strip():
        pages.append({"text": current_text.strip(), "page": page_num})

    print(f"[ingestion] DOCX: {len(pages)} sections extracted")
    return pages


def chunk_text_with_metadata(pages: list[dict]) -> tuple[list[str], list[dict]]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    all_chunks = []
    all_metadatas = []

    for page_data in pages:
        page_chunks = splitter.split_text(page_data["text"])
        for chunk in page_chunks:
            if chunk.strip():
                all_chunks.append(chunk.strip())
                all_metadatas.append({"page": page_data["page"]})

    print(f"[ingestion] Created {len(all_chunks)} chunks")
    return all_chunks, all_metadatas


def load_and_chunk_file(file_path: str, file_name: str) -> tuple[list[str], list[dict]]:
    """
    Auto-detects file type and extracts text accordingly.
    Supports: .pdf, .txt, .docx
    """
    ext = file_name.lower().split(".")[-1]

    if ext == "pdf":
        pages = extract_text_from_pdf(file_path)
    elif ext == "txt":
        pages = extract_text_from_txt(file_path)
    elif ext == "docx":
        pages = extract_text_from_docx(file_path)
    else:
        raise ValueError(f"Unsupported file type: .{ext}. Use PDF, TXT or DOCX.")

    if not pages:
        raise ValueError(f"No text found in file: {file_name}")

    chunks, metadatas = chunk_text_with_metadata(pages)
    return chunks, metadatas