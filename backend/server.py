from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import pandas as pd
import json
import os
from operator import itemgetter

from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from sentence_transformers import CrossEncoder

# Global variables
rag_chain = None
llm = None
vectorstore_global = None
reranker_model = None 

# Defined Test Set of 15 questions.
# These questions are distinct, natural customer queries (paraphrased and phrased differently
# from the questions in data-clean.csv) to rigorously evaluate semantic retrieval generalization.
EVALUATION_TEST_SET = [
    {
        "id": 1,
        "question": "I cannot log in because I forgot my password. Where can I request a password change link?",
        "reference_dataset_question": "How do I reset my password?",
        "expected_ids": [0],
        "expected_intents": ["Password Recovery"],
        "category": "Account"
    },
    {
        "id": 2,
        "question": "Do you support paying via PayPal, wire transfer, or corporate credit cards?",
        "reference_dataset_question": "What payment methods do you accept?",
        "expected_ids": [1],
        "expected_intents": ["Payment Information"],
        "category": "Billing"
    },
    {
        "id": 3,
        "question": "Is it possible to connect our CRM system in Salesforce with your platform?",
        "reference_dataset_question": "Can I integrate your service with Salesforce?",
        "expected_ids": [2],
        "expected_intents": ["Integration Setup"],
        "category": "Technical"
    },
    {
        "id": 4,
        "question": "We just hired a new colleague. How do I invite them and grant them access to our organization?",
        "reference_dataset_question": "How do I add a new user to my account?",
        "expected_ids": [3],
        "expected_intents": ["Add User"],
        "category": "Account"
    },
    {
        "id": 5,
        "question": "What specialized features, SLAs, and support come with your top-tier Enterprise package?",
        "reference_dataset_question": "What's included in the Enterprise plan?",
        "expected_ids": [4],
        "expected_intents": ["Plan Information"],
        "category": "General Inquiry"
    },
    {
        "id": 6,
        "question": "What encryption standards and compliance certifications do you have to safeguard customer data?",
        "reference_dataset_question": "How secure is my data on your platform?",
        "expected_ids": [5],
        "expected_intents": ["Data Security Information"],
        "category": "Technical"
    },
    {
        "id": 7,
        "question": "There is an unfamiliar charge on my latest invoice that does not match our agreed pricing. How can I challenge this transaction?",
        "reference_dataset_question": "I'm being charged incorrectly. How do I dispute this?",
        "expected_ids": [6],
        "expected_intents": ["Billing Dispute"],
        "category": "Billing"
    },
    {
        "id": 8,
        "question": "How can I download a full backup copy of our workspace records and files to CSV or JSON?",
        "reference_dataset_question": "How do I export my data?",
        "expected_ids": [7],
        "expected_intents": ["Data Export"],
        "category": "Technical"
    },
    {
        "id": 9,
        "question": "What is your promised service availability percentage, and do you offer financial credits for outages?",
        "reference_dataset_question": "What's your uptime guarantee?",
        "expected_ids": [8],
        "expected_intents": ["SLA Information"],
        "category": "General Inquiry"
    },
    {
        "id": 10,
        "question": "We have decided not to renew our plan. What are the steps to terminate our active subscription?",
        "reference_dataset_question": "How do I cancel my subscription?",
        "expected_ids": [9],
        "expected_intents": ["Cancel Subscription"],
        "category": "Billing"
    },
    {
        "id": 11,
        "question": "How can we switch our billing tier from Standard to Professional mid-cycle?",
        "reference_dataset_question": "Can I change my plan?",
        "expected_ids": [10],
        "expected_intents": ["Change Plan"],
        "category": "Billing"
    },
    {
        "id": 12,
        "question": "Where in settings can I configure multi-factor authentication (MFA) using an authenticator app?",
        "reference_dataset_question": "How do I set up two-factor authentication?",
        "expected_ids": [11],
        "expected_intents": ["Enable 2FA"],
        "category": "Account"
    },
    {
        "id": 13,
        "question": "Do you provide RESTful or GraphQL endpoints and developer documentation for automated data ingestion?",
        "reference_dataset_question": "What APIs do you offer?",
        "expected_ids": [12],
        "expected_intents": ["API Information"],
        "category": "Technical"
    },
    {
        "id": 14,
        "question": "I cannot recall which email or handle I used to register my account. How can I look it up?",
        "reference_dataset_question": "I forgot my username. How can I recover it?",
        "expected_ids": [14],
        "expected_intents": ["Username Recovery"],
        "category": "Account"
    },
    {
        "id": 15,
        "question": "How long is customer data stored on your servers after we deactivate our account?",
        "reference_dataset_question": "What's your data retention policy?",
        "expected_ids": [17],
        "expected_intents": ["Data Retention Information"],
        "category": "General Inquiry"
    }
]

@asynccontextmanager
async def lifespan(app: FastAPI):
    global rag_chain, llm, vectorstore_global, reranker_model
    print("Initializing RAG Pipeline with Metadata & Re-ranking...")

    # Load dataset (prefer local copy if present, fallback to remote URL)
    dataset_path = "data-clean.csv"
    if os.path.exists(dataset_path):
        print(f"Loading dataset from local file: {dataset_path}")
        df = pd.read_csv(dataset_path)
    else:
        url = "https://raw.githubusercontent.com/ranga4all1/support-queue-assistant-rag/main/data/data-clean.csv"
        print(f"Loading dataset from URL: {url}")
        df = pd.read_csv(url)

    documents = []
    for idx, row in df.iterrows():
        content = (
            f"Customer Question: {row['question']}\n"
            f"Recommended Response: {row['response']}\n"
            f"Category: {row['category']} - {row['subcategory']}\n"
            f"Tags: {row['tags']}\n"
            f"Related Articles: {row['related_articles']}"
        )
        doc_id = int(row['id']) if 'id' in row and pd.notna(row['id']) else idx
        # Store metadata including id and category for filtering and evaluation
        documents.append(
            Document(
                page_content=content, 
                metadata={
                    "id": doc_id,
                    "intent": row['intent'],
                    "category": str(row['category']).strip()
                }
            )
        )

    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    llm = OllamaLLM(model="llama3.2")

    # Build FAISS vector store
    vectorstore_global = FAISS.from_documents(documents, embeddings)

    # Load a lightweight, highly efficient cross-encoder for local re-ranking
    print("Loading local Cross-Encoder re-ranker...")
    reranker_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    print("RAG Pipeline ready with Advanced Retrieval!")
    yield 
    print("Shutting down RAG Pipeline...")

app = FastAPI(title="Advanced Support RAG API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Message(BaseModel):
    role: str
    content: str

class QueryRequest(BaseModel):
    question: str
    history: list[Message] = []
    category_filter: Optional[str] = None  # Optional category filter from frontend

class FeedbackRequest(BaseModel):
    rating: str
    comment: Optional[str] = None
    question: Optional[str] = None
    answer: Optional[str] = None

class EvaluationRequest(BaseModel):
    k: int = 3
    rerank: bool = True

def compute_retrieval_metrics(k: int = 3, rerank: bool = True) -> Dict[str, Any]:
    """
    Computes Precision@k and Recall@k across the defined test set of 15 questions.
    
    Metric Definitions:
    - Hits@k: Number of retrieved documents in the top-k results that match the expected relevant document(s).
    - Precision@k: Hits@k / k (The proportion of retrieved documents in top-k that are relevant).
    - Recall@k: Hits@k / Total_Relevant_Documents (The proportion of relevant documents that were successfully retrieved in top-k).
    - Mean Precision@k: Macro-average of Precision@k across all 15 evaluation queries.
    - Mean Recall@k: Macro-average of Recall@k across all 15 evaluation queries.
    """
    global vectorstore_global, reranker_model
    if vectorstore_global is None:
        raise HTTPException(status_code=503, detail="Vector store is not initialized yet.")
    
    if k <= 0:
        raise HTTPException(status_code=400, detail="k must be a positive integer greater than 0.")

    results = []
    total_precision = 0.0
    total_recall = 0.0

    for item in EVALUATION_TEST_SET:
        query = item["question"]
        expected_ids = set(item["expected_ids"])
        expected_intents = set(item["expected_intents"])
        
        # Retrieve candidate pool from FAISS
        candidate_k = max(10, k) if (rerank and reranker_model) else k
        retriever = vectorstore_global.as_retriever(search_kwargs={"k": candidate_k})
        retrieved_candidates = retriever.invoke(query)
        
        # Apply Cross-Encoder re-ranking if enabled
        if rerank and reranker_model and retrieved_candidates:
            pairs = [[query, doc.page_content] for doc in retrieved_candidates]
            scores = reranker_model.predict(pairs)
            scored_docs = sorted(zip(retrieved_candidates, scores), key=lambda x: x[1], reverse=True)
            top_docs = [doc for doc, score in scored_docs[:k]]
        else:
            top_docs = retrieved_candidates[:k]
        
        # Evaluate relevance
        hits = []
        retrieved_details = []
        for rank, doc in enumerate(top_docs, start=1):
            doc_id = doc.metadata.get("id")
            doc_intent = doc.metadata.get("intent")
            is_relevant = (
                (doc_id is not None and doc_id in expected_ids) or
                (doc_intent is not None and doc_intent in expected_intents)
            )
            if is_relevant:
                hits.append({"rank": rank, "id": doc_id, "intent": doc_intent})
            retrieved_details.append({
                "rank": rank,
                "id": doc_id,
                "intent": doc_intent,
                "is_relevant": is_relevant
            })
            
        num_hits = len(hits)
        precision_at_k = round(num_hits / k, 4)
        total_relevant = len(expected_ids) if expected_ids else len(expected_intents)
        recall_at_k = round(min(1.0, num_hits / total_relevant), 4) if total_relevant > 0 else 0.0
        
        total_precision += precision_at_k
        total_recall += recall_at_k
        
        results.append({
            "test_id": item["id"],
            "question": query,
            "reference_dataset_question": item.get("reference_dataset_question"),
            "category": item["category"],
            "expected_ids": item["expected_ids"],
            "expected_intents": item["expected_intents"],
            "retrieved_documents": retrieved_details,
            "hits_count": num_hits,
            "precision_at_k": precision_at_k,
            "recall_at_k": recall_at_k
        })

    num_queries = len(EVALUATION_TEST_SET)
    mean_precision = round(total_precision / num_queries, 4) if num_queries > 0 else 0.0
    mean_recall = round(total_recall / num_queries, 4) if num_queries > 0 else 0.0

    return {
        "k": k,
        "reranking_applied": bool(rerank and reranker_model),
        "total_test_questions": num_queries,
        "mean_precision_at_k": mean_precision,
        "mean_recall_at_k": mean_recall,
        "metric_definitions": {
            "precision_at_k": "Precision@k is the fraction of retrieved documents in the top-k results that are relevant to the query: Hits@k / k.",
            "recall_at_k": "Recall@k is the fraction of relevant documents that are successfully retrieved in the top-k results: Hits@k / Total_Relevant_Documents.",
            "mean_precision_at_k": "Macro-average of Precision@k across all 15 questions in the defined evaluation test set.",
            "mean_recall_at_k": "Macro-average of Recall@k across all 15 questions in the defined evaluation test set."
        },
        "test_results": results
    }

@app.get("/api/metrics")
def get_metrics_endpoint(k: int = Query(default=7, ge=1, description="Top-k documents to evaluate"), 
                         rerank: bool = Query(default=True, description="Whether to apply Cross-Encoder re-ranking")):
    """
    Computes Recall@k and Precision@k metrics across the defined test set of 15 questions.
    """
    return compute_retrieval_metrics(k=k, rerank=rerank)

@app.post("/api/metrics")
def post_metrics_endpoint(request: Optional[EvaluationRequest] = None):
    """
    POST endpoint to compute Recall@k and Precision@k metrics for the defined test set of 15 questions.
    """
    k = request.k if request else 3
    rerank = request.rerank if request else True
    return compute_retrieval_metrics(k=k, rerank=rerank)

@app.get("/api/test-set")
def get_test_set_endpoint():
    """
    Returns the defined 15-question test set with ground-truth document IDs and intents.
    """
    return {
        "total_questions": len(EVALUATION_TEST_SET),
        "questions": EVALUATION_TEST_SET
    }

@app.post("/api/chat")
def chat_endpoint(request: QueryRequest):
    global llm, vectorstore_global, reranker_model
    
    question_clean = request.question.strip().lower()
    greetings = {"hi", "hello", "hey", "good morning", "good evening", "sup", "greetings", "thanks", "thank you"}
    
    # 1. Intent Routing / Greeting bypass
    if question_clean in greetings:
        direct_prompt = f"You are a friendly customer support assistant. Give a short, polite response to this user greeting: '{request.question}'"
        answer = llm.invoke(direct_prompt)
        return {"answer": answer, "sources": []}

    # 2. Metadata Filtering setup
    search_kwargs = {"k": 10} # Retrieve a larger candidate pool (k=10) for re-ranking
    if request.category_filter:
        # FAISS filter expression using document metadata
        search_kwargs["filter"] = lambda metadata: metadata.get("category") == request.category_filter

    # Retrieve initial documents from FAISS
    retriever = vectorstore_global.as_retriever(search_kwargs=search_kwargs)
    candidate_docs = retriever.invoke(request.question)

    # 3. Re-ranking Step (Thomson Reuters approach)
    if candidate_docs:
        # Prepare pairs of [Query, Document Content] for the cross-encoder
        pairs = [[request.question, doc.page_content] for doc in candidate_docs]
        scores = reranker_model.predict(pairs)
        
        # Attach scores and sort documents by relevance score descending
        scored_docs = sorted(zip(candidate_docs, scores), key=lambda x: x[1], reverse=True)
        
        # Keep only the top 3 best-ranked documents after cross-encoder evaluation
        final_docs = [doc for doc, score in scored_docs[:3]]
    else:
        final_docs = []

    # 4. Format context and history for generation
    context_str = "\n\n".join([doc.page_content for doc in final_docs])
    chat_history_str = "\n".join([f"{m.role.capitalize()}: {m.content}" for m in request.history])

    # System prompt strictly restricting responses to the customer support subject
    template = """You are a professional customer support assistant for our software service and platform.
Use the following context and conversation history to answer the customer's question.

IMPORTANT RULES:
1. Support Scope Only: You must ONLY answer questions that are directly related to our customer support subject, which includes Account management, Billing & subscriptions, Technical support, APIs & integrations, and General product inquiries.
2. Refuse Unrelated Questions: If the customer asks a question that is NOT related to our customer support subject (such as general knowledge, history, geography, recipes, entertainment, sports, creative writing, personal opinions, or general coding unrelated to our platform), you MUST politely refuse to answer. Clearly state that you can only assist with customer support inquiries related to our software platform.
3. Grounding in Context: Base your answers on the provided Context. If the context does not contain sufficient information to answer a valid support question, inform the customer that you do not have that specific information and recommend contacting our human support team. Do not hallucinate or invent answers.

Conversation History:
{chat_history}

Context:
{context}

Question: {input}
Helpful Answer:"""

    prompt = PromptTemplate.from_template(template)
    formatted_prompt = prompt.format(
        chat_history=chat_history_str,
        context=context_str,
        input=request.question
    )

    # 5. Generate final response with Ollama
    answer = llm.invoke(formatted_prompt)

    sources = [
        {"intent": doc.metadata.get("intent", ""), "content": doc.page_content}
        for doc in final_docs
    ]

    return {
        "answer": answer,
        "sources": sources
    }

@app.post("/api/feedback")
def feedback_endpoint(feedback: FeedbackRequest):
    feedback_file = "feedback.json"
    feedback_data = feedback.dict()
    try:
        with open(feedback_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(feedback_data, ensure_ascii=False) + "\n")
        return {"status": "success", "message": "Feedback saved successfully."}
    except Exception as e:
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
