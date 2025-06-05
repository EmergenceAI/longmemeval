from sentence_transformers import SentenceTransformer
from utils import callgpt
import faiss, diskcache, os
import numpy as np
import pickle

CACHE_DIR = './cache/'
cache_embed = diskcache.Cache(os.path.join(CACHE_DIR, 'embed'), eviction_policy='none')
cache_embed.reset('cull_limit', 0)

# Load model
model = SentenceTransformer("msmarco-distilbert-base-v4")

def embed_lesson(lesson: dict) -> np.ndarray:
    """
    Embed a lesson using the pre-trained model.

    Args:
        lesson (dict): The lesson to embed.

    Returns:
        np.ndarray: The embedded lesson.
    """
    combined_text = list(lesson.values())
    embedding = model.encode(combined_text)
    return np.mean(embedding, axis=0)

@cache_embed.memoize()
def embed_message(message: str) -> np.ndarray:
    """
    Embed a query using the pre-trained model.

    Args:
        query (str): The query to embed.

    Returns:
        np.ndarray: The embedded query.
    """
    return model.encode([message])

def embed_episode(episode: list[dict]) -> np.ndarray:
    """
    Embed an episode using the pre-trained model.

    Args:
        episode (dict): The episode to embed.

    Returns:
        np.ndarray: The embedded episode.
    """
    episode_string = "\n".join(msg["role"] + ": " + msg["content"] for msg in episode)
    embedding = model.encode(episode_string)
    return embedding

def search(query, k=5):
    """
    Search for the top k most similar messages to the query.

    Args:
        query (str): The query to search for.
        k (int): The number of top results to return.

    Returns:
        list: The top k most similar messages.
    """
    # Load the index and message
    index = faiss.read_index("messages_embedding.index")
    with open("messages.pkl", "rb") as f:
        messages = pickle.load(f)
        # get the dimension of the index
        dimension = index.d
        print(f"Index dimension: {dimension}")

    # Embed the query
    query_embedding = embed_message(query)

    # Search for the top k most similar messages
    D, I = index.search(query_embedding, k)

    # Get the top k messages
    top_k_messages = [messages[i] for i in I[0]]

    return top_k_messages


def process_question(haystack_sessions: list[list[dict]], question: str, question_date:str, haystack_dates: list[str]) -> str:
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
