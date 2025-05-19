from sentence_transformers import SentenceTransformer
import numpy as np
import pickle
import faiss

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