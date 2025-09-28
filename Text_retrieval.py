import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import SupabaseVectorStore
from supabase import create_client

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase_client = create_client(SUPABASE_URL, SUPABASE_KEY)

embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-large-en-v1.5")

vectorstore = SupabaseVectorStore(
    client=supabase_client,
    embedding=embedding_model,
    table_name="lease_rag"
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

query = "Can I have a dog in the apartment?"
docs = retriever.get_relevant_documents(query)
# docs = retriever.invoke(state["question"])

print(" Query:", query)
for i, doc in enumerate(docs):
    print(f"\nResult {i+1}:")
    print("Text:", doc.page_content[:200], "...")
    print("Metadata:", doc.metadata)
