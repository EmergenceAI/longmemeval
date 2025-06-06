from sentence_transformers import SentenceTransformer, util
import json
from utils import callgpt
from textwrap import dedent


# Load once
retrieval_model = SentenceTransformer('all-MiniLM-L6-v2')

def process_haystack(haystack_sessions: list[list[dict]], haystack_dates: list[str]):
    """Preprocess and encode haystack turns with dates."""
    all_turns = []
    for session, date in zip(haystack_sessions, haystack_dates):
        for turn in session:
            if 'role' in turn and 'content' in turn:
                all_turns.append(f"[{date}] {turn['role']}: {turn['content']}")
    corpus_embeddings = retrieval_model.encode(all_turns, convert_to_tensor=True)
    return corpus_embeddings, all_turns

def process_question(haystack_sessions: list[list[dict]], question: str, question_date: str, haystack_dates: list[str]) -> str:
    memstruct = process_haystack(haystack_sessions, haystack_dates)
    corpus_embeddings, all_turns = memstruct
    query_embedding = retrieval_model.encode(question, convert_to_tensor=True)

    # Step 1: Retrieve top-k relevant turns
    hits = util.semantic_search(query_embedding, corpus_embeddings, top_k=11)[0]
    retrieved_turns = [all_turns[hit['corpus_id']] for hit in hits]

    # Step 2: Extract structured facts from retrieved turns (fact-level)
    summary_prompt = f"""
    You are a memory summarization assistant. Extract relevant facts to answer the question. Follow this chain-of-thought:
    1. Identify key events, dates, quantities, or named entities.
    2. Extract only information relevant to the question.
    3. Write the facts in structured bullet points.

    Question: {question}

    Messages:
    {json.dumps(retrieved_turns, indent=2)}

    Now extract the structured facts:
    -
    """
    summary_prompt = dedent(summary_prompt)
    facts = callgpt([{"role": "system", "content": summary_prompt}], model="gpt-4o", max_tokens=1024)

    # Step 3: Combine both facts and raw turns to answer
    answer_prompt = f"""
    You are a helpful assistant. Using both the extracted facts and the original conversation turns below,
    answer the question as accurately and concisely as possible.

    Extracted Facts:
    {facts}

    Retrieved Conversation Turns:
    {json.dumps(retrieved_turns, indent=2)}

    Question: {question}
    Question Date: {question_date}
    Answer:
    """
    answer_prompt = dedent(answer_prompt)
    answer = callgpt([{"role": "system", "content": answer_prompt}], model="gpt-4o", max_tokens=512)
    return answer.strip()
