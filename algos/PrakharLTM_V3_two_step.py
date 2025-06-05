from sentence_transformers import SentenceTransformer, util
import json
from utils import callgpt, deindent

# Load once
retrieval_model = SentenceTransformer('all-MiniLM-L6-v2')

def process_haystack(haystack_sessions: list[list[dict]], haystack_dates: list[str]):
    """
    Step 1: Retrieve relevant messages (with haystack_dates) from the haystack.
    Step 2: Summarize into structured facts using a chain-of-thought prompt.
    Step 3: Answer the question using only those extracted facts.
    """
    # 1: Flatten all messages with their date and role
    all_turns = []
    for session, date in zip(haystack_sessions, haystack_dates):
        for turn in session:
            if 'role' in turn and 'content' in turn:
                all_turns.append(f"[{date}] {turn['role']}: {turn['content']}")
    # Encode the corpus and the question
    corpus_embeddings = retrieval_model.encode(all_turns, convert_to_tensor=True)
    return corpus_embeddings, all_turns

def process_question(memstruct, question: str, question_date: str) -> str:
    corpus_embeddings, all_turns = memstruct
    query_embedding = retrieval_model.encode(question, convert_to_tensor=True)
    # Retrieve top-k relevant messages using semantic search
    hits = util.semantic_search(query_embedding, corpus_embeddings, top_k=10)[0]
    retrieved = [all_turns[hit['corpus_id']] for hit in hits]
    # 2: Use chain-of-thought summarization to extract structured facts
    summary_prompt = f"""
    You are a memory summarization assistant. Extract relevant facts to answer the question. Follow this chain-of-thought:
    1. Identify key events, dates, quantities, or named entities.
    2. Extract only information relevant to the question.
    3. Write the facts in structured bullet points.

    Question: {question}

    Messages:
    {json.dumps(retrieved, indent=2)}

    Now extract the structured facts:
    -
    """
    summary_prompt = deindent(summary_prompt)
    facts = callgpt([{"role": "system", "content": summary_prompt}], model="gpt-4o", max_tokens=1024)
    # 3: Answer the question using only the extracted facts
    answer_prompt = f"""
    Based strictly on the following extracted facts, answer the question as accurately and concisely as possible.

    Facts:
    {facts}

    Question: {question}
    Question date: {question_date}
    Answer:
    """
    answer_prompt = deindent(answer_prompt)
    answer = callgpt([{"role": "system", "content": answer_prompt}], model="gpt-4o", max_tokens=512)
    return answer.strip()
