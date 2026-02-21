import os
from dotenv import load_dotenv

from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from sentence_transformers import CrossEncoder  # Standard for re-ranking

# 1. Setup Environment
load_dotenv()
working_dir = os.path.dirname(os.path.abspath(__file__))

# 2. Initialize Models
# Local Embedding model for cost-efficiency
embedding = HuggingFaceEmbeddings()

# Groq LLM for high-speed reasoning
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

# Local Re-ranker (MS Marco MiniLM) to improve precision
rerank_model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')


def process_document_to_chroma_db(file_name):
    loader = UnstructuredPDFLoader(f"{working_dir}/{file_name}")
    documents = loader.load()

    # Advanced Tip: Smaller chunks (700-800 chars) work better with re-rankers
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=120
    )
    texts = text_splitter.split_documents(documents)

    vectordb = Chroma.from_documents(
        documents=texts,
        embedding=embedding,
        persist_directory=f"{working_dir}/doc_vectorstore"
    )
    return 0


def answer_question(user_question):
    vectordb = Chroma(
        persist_directory=f"{working_dir}/doc_vectorstore",
        embedding_function=embedding
    )

    # --- ADVANCED STEP 1: MULTI-QUERY EXPANSION ---
    # Improves recall by searching for variations of the user's question
    multi_query_prompt = (
        f"Generate 3 different search queries to help find the answer to: '{user_question}'. "
        f"Output only the queries, one per line."
    )
    response = llm.invoke(multi_query_prompt).content
    all_queries = [user_question] + response.strip().split("\n")

    # --- ADVANCED STEP 2: AGGREGATED RETRIEVAL ---
    all_retrieved_docs = []
    for query in all_queries:
        docs = vectordb.similarity_search(query, k=5)
        all_retrieved_docs.extend(docs)

    # Remove duplicates
    unique_docs = {doc.page_content: doc for doc in all_retrieved_docs}.values()

    # --- ADVANCED STEP 3: CROSS-ENCODER RE-RANKING ---
    # Re-scores chunks against the original question for maximum precision
    pairs = [[user_question, doc.page_content] for doc in unique_docs]
    scores = rerank_model.predict(pairs)

    # Sort docs by their new relevance scores
    scored_docs = sorted(zip(scores, unique_docs), key=lambda x: x[0], reverse=True)

    # Take the top 5 highest quality chunks
    top_docs = [doc for score, doc in scored_docs[:5]]
    context = "\n\n".join([doc.page_content for doc in top_docs])

    # --- FINAL GENERATION ---
    final_prompt = f"""
    Answer the question using ONLY the provided context. 
    Context: {context}
    Question: {user_question}

    Assistant: Let's think step-by-step:
    """
    return llm.invoke(final_prompt).content