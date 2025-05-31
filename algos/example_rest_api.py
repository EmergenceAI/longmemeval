#!/usr/bin/env python3
import json, requests

def json_rest_process_question(haystack_sessions: list[list[dict]], question: str, question_date:str, haystack_dates: list[str]):
    big_json = {
        'haystack_sessions': haystack_sessions,
        'question': question,
        'question_date': question_date,
        'haystack_dates': haystack_dates
    }
    return json.dumps(big_json)

def call_rest_endpoint(haystack_sessions: list[list[dict]], question: str, question_date:str, haystack_dates: list[str]):
    # Call REST endpoint
    url = "http://localhost:5000/process_question"
    headers = {'Content-Type': 'application/json'}
    big_json = json_rest_process_question(haystack_sessions, question, question_date, haystack_dates)
    response = requests.post(url, headers=headers, data=big_json)
    return response.json()['answer']

# Example of a REST endpoint that would receive the JSON data
def rest_endpoint(big_json_str:str):
    """
    The format of big_json_str is:
    {
        'haystack_sessions': List[List[Dict[str, str]]], # List of sessions
            [
                [ # First session
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Hi there!"},
                    ...
                ],
                [ # Second session
                    {"role": "user", "content": "What is your name?"},
                    {"role": "assistant", "content": "I am a chatbot."},
                    ...
                ],
                ... # More sessions
            ],
        'question': str
            "What is the capital of France?",
        'question_date': str
            "2023/08/20 (Sun) 23:59",
        'haystack_dates': List[str]
            [
                '2023/05/20 (Sat) 02:57', # Timestamp of the first session
                '2023/05/20 (Sat) 03:51', # Timestamp of the second session
                ... # More timestamps
            ]
    }
    """
    # This will be on the receiving end of the REST endpoint
    big_json = json.loads(big_json_str)
    haystack_sessions = big_json['haystack_sessions']
    question = big_json['question']
    question_date = big_json['question_date']
    haystack_dates = big_json['haystack_dates']
    # Do something with the data
    answer = "I don't know the answer to that!"
    return answer

####
from flask import Flask, request, jsonify
app = Flask(__name__)
@app.route('/process_question', methods=['POST'])
def process_question():
    # Get the JSON data from the request
    data = request.get_json()
    # Process the data
    answer = rest_endpoint(json.dumps(data))
    # Return a JSON response
    return jsonify({'answer': answer})

def test():
    haystack_sessions = [[{"role": "user", "content": "Hello"}]]
    question = "What is the capital of France?"
    question_date = "2023/08/20 (Sun) 23:59"
    haystack_dates = ["2023/08/20 (Sun)"]
    response = call_rest_endpoint(haystack_sessions, question, question_date, haystack_dates)
    print(response)

def main(): app.run(debug=True, port=5000)

if __name__ == '__main__': main()
