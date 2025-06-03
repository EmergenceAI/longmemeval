from sentence_transformers import SentenceTransformer, util
import json
from utils import callgpt

# Load once
retrieval_model = SentenceTransformer('all-MiniLM-L6-v2')

def process_question(haystack_sessions: list[list[dict]], question: str, question_date: str, haystack_dates: list[str]) -> str:
    """
    Step 1: Retrieve relevant messages from the haystack.
    Step 2: Summarize into structured facts using a chain-of-thought.
    Step 3: Answer the question based only on those facts.
    """

    # Flatten all turns with role/context
    all_turns = [f"{turn['role']}: {turn['content']}" for session in haystack_sessions for turn in session]

    # Step 1: Embed and retrieve top-k relevant messages
    corpus_embeddings = retrieval_model.encode(all_turns, convert_to_tensor=True)
    query_embedding = retrieval_model.encode(question, convert_to_tensor=True)
    hits = util.semantic_search(query_embedding, corpus_embeddings, top_k=10)[0]
    retrieved = [all_turns[hit['corpus_id']] for hit in hits]

    # Step 2: Structured summarization with CoT prompting
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

    facts = callgpt([{"role": "user", "content": summary_prompt}], model="gpt-4o", max_tokens=1024)

    # Step 3: Answer the question based on extracted facts
    answer_prompt = f"""
You are a helpful assistant. Based strictly on the following extracted facts, answer the question as accurately and concisely as possible.

Facts:
{facts}

Question: {question}
Answer:
"""
    answer = callgpt([{"role": "user", "content": answer_prompt}], model="gpt-4o", max_tokens=512)

    return answer.strip()
