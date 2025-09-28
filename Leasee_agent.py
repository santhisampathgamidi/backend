import os
import re
from dotenv import load_dotenv
from groq import Groq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import SupabaseVectorStore
from supabase import create_client
from langgraph.graph import StateGraph
from langchain.schema import Document

# --- Load environment ---
load_dotenv()
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase_client = create_client(SUPABASE_URL, SUPABASE_KEY)

# --- Embeddings ---
embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-large-en-v1.5")

# --- VectorStore ---
vectorstore = SupabaseVectorStore(
    client=supabase_client,
    embedding=embedding_model,
    table_name="lease_rag"
)
vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# --- LLM (Groq SDK) ---
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# --- Utility: Clean query for full-text search ---
def clean_query_for_ft(query: str) -> str:
    """
    Simplify user question into safe keywords for PostgreSQL full-text search.
    """
    cleaned = re.sub(r"[^\w\s]", " ", query)  # remove punctuation
    cleaned = cleaned.lower()
    stopwords = {"can", "i", "the", "a", "an", "is", "to", "of", "in", "on", "for", "with", "when", "will"}
    keywords = [w for w in cleaned.split() if w not in stopwords]
    return " | ".join(keywords)  # Postgres expects "word1 | word2"

def hybrid_retrieve(query: str, k: int = 5):
    """Combine vector search + keyword search, merge chunks by section."""

    # --- Step 1: Vector search ---
    vector_docs = vector_retriever.invoke(query)

    # --- Step 2: Keyword search ---
    keyword_docs = []
    tsquery = clean_query_for_ft(query)
    if tsquery.strip():
        response = supabase_client.table("lease_rag") \
            .select("content, metadata") \
            .text_search("content", tsquery) \
            .execute()

        if response.data:
            for row in response.data[:k]:
                keyword_docs.append(
                    Document(
                        page_content=row["content"],
                        metadata=row["metadata"]
                    )
                )

    # --- Step 3: Merge keyword + vector results ---
    all_docs = vector_docs + keyword_docs

    # --- Step 4: Merge chunks by section ---
    merged_by_section = {}
    for d in all_docs:
        section = d.metadata.get("section", "Unknown")
        page = d.metadata.get("page", "Unknown")
        key = (section, page)
        if key not in merged_by_section:
            merged_by_section[key] = []
        merged_by_section[key].append(d.page_content)

    merged_docs = []
    for (section, page), contents in merged_by_section.items():
        merged_text = "\n\n".join(contents)
        merged_docs.append(
            Document(
                page_content=merged_text,
                metadata={"section": section, "page": page}
            )
        )

    # --- Step 5: Return top-k merged sections ---
    return merged_docs[:k]


# --- LangGraph State ---
class AgentState(dict):
    question: str
    context: str
    answer: str


def retrieve(state: AgentState):
    docs = hybrid_retrieve(state["question"], k=5)

    print("\n🔎 DEBUG: Retrieved Documents")
    for i, d in enumerate(docs, 1):
        print(f"\n--- Document {i} ---")
        print("Page:", d.metadata.get("page"))
        print("Section:", d.metadata.get("section"))
        print("Preview:", d.page_content[:300], "...")

    state["context"] = "\n\n".join([d.page_content for d in docs])
    state["citations"] = [
        f"Section: {d.metadata.get('section', 'Unknown')} | Page: {d.metadata.get('page', 'Unknown')}"
        for d in docs
    ]
    return state


def answer(state: AgentState):
    prompt = f"""
    You are an AI Lease Assistant. 
    Answer the tenant's question using ONLY the lease context below.

    - If multiple sections mention the same topic, 
      prefer the section that specifies **exact amounts, percentages, or deadlines** 
      over sections that only make general references.
    - Provide a clear, concise answer in plain English.
    - Always cite the section and page in your answer.

    Lease Context:
    {state['context']}

    Question: {state['question']}
    """

    # --- Call Groq SDK ---
    completion = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_completion_tokens=512,
        top_p=1,
        stream=False,
    )

    resp_text = completion.choices[0].message.content

    # Format citations nicely
    citations = "\n".join([f"- {c}" for c in state.get("citations", [])])
    state["answer"] = f"**Answer:** {resp_text.strip()}\n\n📑 **Sources:**\n{citations}"
    return state


# --- Graph ---
workflow = StateGraph(AgentState)
workflow.add_node("retrieve", retrieve)
workflow.add_node("answer", answer)
workflow.add_edge("retrieve", "answer")
workflow.set_entry_point("retrieve")
workflow.set_finish_point("answer")

leasee_agent = workflow.compile()

# --- Run test ---
# if __name__ == "__main__":
#     q = "What happens if I break the lease or fail to comply with its terms?"
#     result = leasee_agent.invoke({"question": q})
#     print("Q:", q)
#     print("A:", result["answer"])
