for question in tqdm(data):
    for i, messages in enumerate(question['haystack_sessions']):
        # Skip if messages is empty or has only one element
        if len(messages) <= 1:
            try:
                client.add(
                    messages,
                    user_id=f"longmemeval-test",
                    metadata={
                        "question_id": question['question_id'],
                        "question_date": question['question_date'],
                        "haystack_date": question['haystack_dates'][i],
                        "haystack_session_id": question['haystack_session_ids'][i]
                    }
                )
            except APIError as e:
                print(f"Error adding memory for question {question['question_id']}, session {i}: {e}")
            continue

        # Break messages into chunks of 2
        for j in range(0, len(messages), 2):
            chunk = messages[j:j+2]
            try:
                client.add(
                    chunk,
                    user_id=f"longmemeval-test", 
                    metadata={
                        "question_id": question['question_id'],
                        "question_date": question['question_date'],
                        "haystack_date": question['haystack_dates'][i],
                        "haystack_session_id": question['haystack_session_ids'][i]
                    }
                )
            except APIError as e:
                print(f"Error adding memory chunk for question {question['question_id']}, session {i}, chunk {j//2}: {e}")

################################################################

import os
from mem0 import MemoryClient
from openai import OpenAI


client = MemoryClient()
openai_client = OpenAI()


def chat_with_memories(message: str, user_id: str = "default_user") -> str:
    # Retrieve relevant memories using Mem0 API
    # 👇 CRUCIAL FIX 1: Specify the working output format 'v1.1'
    relevant_memories = client.search(
        query=message,
        user_id=user_id,
        limit=3,  # Limit the number of retrieved memories
        output_format='v1.1'
    )

    # print(relevant_memories) # Optional: for debugging

    # Format the retrieved memories into a string
    # Note: Accessing ["results"] key which contains the list of memories in v1.1
    memories_str = "\n".join(f"- {entry['memory']}" for entry in relevant_memories.get("results", [])) # Use .get for safety

    # Prepare prompts for GPT-4o
    system_prompt = f"You are a helpful AI. Answer the question based on query and memories."

    # 👇 MODIFICATION: Explicitly label memories and provide context
    memory_prompt = "Here's some of the memories that we share that might be relevant to the question (yet they may not be relevant to the question). Even if they're relevant you might still need to piece them together to be able to answer the question:\n" + memories_str

    print("System prompt: ", system_prompt)
    print("Memories prompt: ", memory_prompt)
    print("User prompt: ", message)

    # 👇 MODIFICATION: Inject memories as a separate user message before the actual query
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": memory_prompt}, # Memory context provided first
        {"role": "user", "content": message}        # The actual user question/message
    ]

    # print(messages) # Optional: for debugging

    # Call the OpenAI API
    response = openai_client.chat.completions.create(model="gpt-4o", messages=messages)
    assistant_response = response.choices[0].message.content

    # Store the new user message and assistant response back into Mem0
    # (Assuming you have a separate step/call for this after getting the response)
    # e.g., client.add(data=[{"role": "user", "content": message}, {"role": "assistant", "content": assistant_response}], user_id=user_id)

    return assistant_response
