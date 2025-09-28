import os
from dotenv import load_dotenv
from groq import Groq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import SupabaseVectorStore
from supabase import create_client
from langgraph.graph import StateGraph
from langchain.schema import Document

load_dotenv()

# --- Supabase setup ---
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

# --- Groq Client (Streaming) ---
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# ======================================================
# Hybrid Retriever (Lease Context)
# ======================================================
def hybrid_retrieve(query: str, k: int = 5):
    vector_docs = vector_retriever.invoke(query)

    response = supabase_client.table("lease_rag") \
        .select("content, metadata") \
        .text_search("content", query, {"type": "plain"}) \
        .execute()

    keyword_docs = []
    if response.data:
        for row in response.data[:k]:
            keyword_docs.append(
                Document(
                    page_content=row["content"],
                    metadata=row["metadata"]
                )
            )

    seen = set()
    merged = []
    for d in vector_docs + keyword_docs:
        key = (d.page_content[:50], d.metadata.get("page"))
        if key not in seen:
            merged.append(d)
            seen.add(key)

    return merged[:k]

# ======================================================
# --- State for Graph ---
# ======================================================
class AgentState(dict):
    question: str
    context: str
    answer: str

# ======================================================
# --- Data Helpers ---
# ======================================================
def get_occupancy_stats():
    """Compute occupancy % and vacancy breakdown"""
    response = supabase_client.table("apartment_ops").select("*").execute()
    if not response.data:
        return "No occupancy data available."

    total_units = len(response.data)
    occupied_units = sum(1 for row in response.data if row["occupancy_status"] == "occupied")
    vacant_units = total_units - occupied_units
    occupancy_rate = round((occupied_units / total_units) * 100, 2)

    return f"Building Occupancy: {occupancy_rate}% ({occupied_units} occupied / {total_units} total, {vacant_units} vacant)"

def get_avg_rent():
    """Compute average rent of building"""
    response = supabase_client.table("apartment_ops").select("rate,occupancy_status").execute()
    if not response.data:
        return "No rent data available."

    rates = [row["rate"] for row in response.data if row["rate"] is not None]
    avg_rent = round(sum(rates) / len(rates), 2) if rates else 0
    return f"Building Average Rent: ${avg_rent}"

def get_vacant_units(limit=10):
    """List some vacant units"""
    response = supabase_client.table("apartment_ops").select("unit_id,rate").eq("occupancy_status", "vacant").execute()
    if not response.data:
        return "No vacant units available."

    units = [f"Unit {row['unit_id']} (${row['rate']})" for row in response.data]
    preview = ", ".join(units[:limit])
    more = f" (+{len(units)-limit} more)" if len(units) > limit else ""
    return f"Vacant Units: {preview}{more}"

def get_benchmark_stats():
    """Fetch competitor benchmarks"""
    response = supabase_client.table("market_benchmark").select("*").execute()
    if not response.data:
        return "No benchmark data available."

    summary = "Competitor Benchmarks:\n"
    for row in response.data:
        summary += f"- {row['competitor']}: Avg Rate ${row['avg_rate']}, Occupancy {row['occupancy']}%\n"
    return summary

def compare_to_competitor(name: str):
    """Compare building rent & occupancy to a specific competitor"""
    # Building stats
    occ = get_occupancy_stats()
    rent = get_avg_rent()

    # Competitor
    response = supabase_client.table("market_benchmark").select("*").eq("competitor", name).execute()
    if not response.data:
        return f"No data available for competitor '{name}'."

    comp = response.data[0]
    return f"""
    {occ}
    {rent}

    Competitor {comp['competitor']}:
    - Avg Rent: ${comp['avg_rate']}
    - Occupancy: {comp['occupancy']}%
    """

# ======================================================
# --- Nodes ---
# ======================================================
def retrieve_contract(state: AgentState):
    docs = hybrid_retrieve(state["question"], k=5)

    state["context"] = "\n\n".join([d.page_content for d in docs])
    state["citations"] = [
        f"Section: {d.metadata.get('section', 'Unknown')} | Page: {d.metadata.get('page', 'Unknown')}"
        for d in docs
    ]
    return state

def retrieve_operations(state: AgentState):
    ops_context = get_occupancy_stats()
    rent_context = get_avg_rent()
    vacant_context = get_vacant_units()
    state["context"] += f"\n\n[Operational Data]\n{ops_context}\n{rent_context}\n{vacant_context}"
    return state

def retrieve_benchmark(state: AgentState):
    bench_context = get_benchmark_stats()
    state["context"] += f"\n\n[Benchmark Data]\n{bench_context}"
    return state

def answer(state: AgentState):
    prompt = f"""
    You are an AI **Lessor Assistant**.
    You can answer 3 categories of questions:

    1. Lease-related → use lease context and cite section/page.
    2. Operational → occupancy %, average rent, list vacant units.
    3. Benchmarking → competitor averages, comparisons to specific competitors.

    If a competitor name (e.g., Riverstone Lofts) is in the question, compare building stats vs that competitor.
    Otherwise, summarize building vs all competitors.

    Context:
    {state['context']}

    Question: {state['question']}
    """
    # Competitor check
    comp_answer = None
    competitors = ["Greenview Apartments", "Maple Heights", "Riverstone Lofts", "Oakwood Residences", "Lakeside Towers"]
    for c in competitors:
        if c.lower() in state["question"].lower():
            comp_answer = compare_to_competitor(c)

    if comp_answer:
        state["answer"] = comp_answer + "\n\n📑 Sources: Operational/Benchmark Data"
        return state

    # --- Default: Streaming LLM answer ---
    completion = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=1,
        max_completion_tokens=512,
        top_p=1,
        stream=True
    )

    answer_text = ""
    for chunk in completion:
        if chunk.choices[0].delta.content:
            answer_text += chunk.choices[0].delta.content

    citations = "; ".join(state.get("citations", []))
    state["answer"] = f"{answer_text.strip()}\n\n📑 Sources: {citations if citations else 'Operational/Benchmark Data'}"
    return state

# ======================================================
# --- Graph ---
# ======================================================
workflow = StateGraph(AgentState)
workflow.add_node("retrieve_contract", retrieve_contract)
workflow.add_node("retrieve_operations", retrieve_operations)
workflow.add_node("retrieve_benchmark", retrieve_benchmark)
workflow.add_node("answer", answer)

workflow.add_edge("retrieve_contract", "retrieve_operations")
workflow.add_edge("retrieve_operations", "retrieve_benchmark")
workflow.add_edge("retrieve_benchmark", "answer")

workflow.set_entry_point("retrieve_contract")
workflow.set_finish_point("answer")

lessor_agent = workflow.compile()

# ======================================================
# --- Test Run ---
# ======================================================

# if __name__ == "__main__":
#     q = "How does my building compare to Maple Heights?"
#     result = lessor_agent.invoke({"question": q})
#     print("Q:", q)
#     print("A:", result["answer"])
