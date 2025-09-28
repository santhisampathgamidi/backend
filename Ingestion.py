import os
import re
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import SupabaseVectorStore
from supabase import create_client
from langchain.schema import Document

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase_client = create_client(SUPABASE_URL, SUPABASE_KEY)

# --- Load PDF ---
loader = PyPDFLoader("Copp Lease Agreement 2025.pdf")
docs = loader.load()  # each doc has {"page": N}

# --- Regex for section headers ---
SECTION_HEADER_PATTERN = re.compile(
    r'^\s*(\d{1,2}\.\s+[A-Z][A-Z\s]+|ADDENDUM[:\s]|GUARANTY|PARKING|ANIMALS|DEFAULT|ATTORNEY\S* FEES)'
)

def split_into_sections(docs):
    """Split full text into sections based on detected headers"""
    sections = []
    buffer = []
    current_section = None
    current_page = None

    for doc in docs:
        page = doc.metadata.get("page", None)
        for line in doc.page_content.split("\n"):
            if SECTION_HEADER_PATTERN.match(line.strip()):
                # Save previous section
                if buffer:
                    sections.append({
                        "section": current_section,
                        "page": current_page,
                        "text": "\n".join(buffer).strip()
                    })
                    buffer = []
                current_section = line.strip().title()
                current_page = page
            buffer.append(line)

    # Save last section
    if buffer:
        sections.append({
            "section": current_section,
            "page": current_page,
            "text": "\n".join(buffer).strip()
        })

    return sections

# --- Step 1: Extract sections ---
sections = split_into_sections(docs)

# --- Step 2: Only split if section is very long (>2000 chars) ---
chunks = []
for sec in sections:
    text = sec["text"]
    if len(text) > 2000:
        # Split very long section into ~1000-char subchunks
        sub_texts = [text[i:i+1000] for i in range(0, len(text), 1000)]
    else:
        sub_texts = [text]

    for sub in sub_texts:
        header = f"[Section: {sec['section'] or 'Unknown'} | Page: {sec['page'] + 1 if sec['page'] is not None else 'Unknown'}]\n"
        footer = f"\n[End of Section: {sec['section'] or 'Unknown'} | Page: {sec['page'] + 1 if sec['page'] is not None else 'Unknown'}]"
        chunks.append(Document(
            page_content=header + sub + footer,
            metadata={
                "section": sec["section"] or "Unknown",
                "page": sec["page"] + 1 if sec["page"] is not None else None,
                "source": "Copp Lease Agreement 2025"
            }
        ))

# --- Step 3: Embeddings ---
embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-large-en-v1.5")

# --- Step 4: Store in Supabase ---
vectorstore = SupabaseVectorStore.from_documents(
    chunks, embedding_model, client=supabase_client, table_name="lease_rag"
)

print(f"✅ Lease ingested into Supabase with {len(chunks)} section-level chunks (long sections safely split).")
