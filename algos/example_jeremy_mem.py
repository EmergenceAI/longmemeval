from embeddings import embed_message, search
import numpy as np
import faiss
import pickle
from utils import callgpt

def hello_world_process_question(haystack_sessions: list[list[dict]], question: str, question_date:str, haystack_dates: list[str]) -> str:
    # haystack_sessions is a list of sessions, each session is a list of turns, each turn is a dict with role and content
    # question_date : '2023/08/20 (Sun) 23:59'
    # return hypothesis: str

    # embed the user messages
    msg_texts, msg_embeddings = [], []
    for idx, session in enumerate(haystack_sessions):
        for message in session:
            # if 'content' in message and message['role'] == 'user':  # for just user messages, but should be all messages probably
            if 'content' in message and 'role' in message:
                role = message['role']
                content = message['content']
                timestamp = haystack_dates[idx]
                formatted_msg = f"({timestamp}) {role}: {content}"
                msg_texts.append(formatted_msg)
                embedding = embed_message(message['content'])
                embedding = np.squeeze(embedding) # remove the extra dimension
                msg_embeddings.append(embedding)

    # convert to numpy array
    msg_embeddings_matrix = np.array(msg_embeddings)

    # build faiss index
    dimension = msg_embeddings_matrix.shape
    print(dimension)
    index = faiss.IndexFlatL2(dimension[1])
    index.add(msg_embeddings_matrix)


    # Save index and messages
    faiss.write_index(index, "messages_embedding.index")
    with open("messages.pkl", "wb") as f:
        pickle.dump(msg_texts, f)


    top_k_messages = search(question, k=3)

    result = callgpt(
        model="gpt-4o",
        max_tokens=250,
        messages=[
            {"role": "system", "content": "Please answer the question based on the context provided. Here are some past messages that might be relevant:\n" + "\n".join(top_k_messages)},
            {"role": "user", "content": question}
            ]
            )
    print(f"\nQuestion: {question}")
    print(f"Result: {result}")

    return result
