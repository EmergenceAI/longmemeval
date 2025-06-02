import sys, os

sys.path.insert(0, os.path.expanduser('~/agent001'))  # ← Add parent of agent001
from memodemo import memerate

# from src.utils import pprint

def hello_world_process_question(haystack_sessions: list[list[dict]], question: str, question_date:str, haystack_dates: list[str]) -> str:
    # haystack_sessions is a list of sessions, each session is a list of turns, each turn is a dict with role and content
    # question_date : '2023/08/20 (Sun) 23:59'
    # return hypothesis: str
    return f"{question}?  I don't know the answer to that!"


#!/usr/bin/env python3
# import os
from src.utils import pprint, tokenator, remove_notes, UI_Session, MessageItem, printdebug, to_dict, threads_wait, thread_start, run_shell, mkdir
from src.cachedutils import func_description, callgpt
from src.prompts import PLANNER_SYS_MESSAGE
from src.sandbox import Sandbox
from src.coretools import reply, shell, process_commands, get_new_tools
from src.memory import Memory
from src.extract_lessons import extract_all_lessons, extract_lessons, extract_conversation, add_lessons_to_memory, print_lessons
from convlog import read_convlogs, write_convlog

VAULT = 'vault'
MEMDIR = 'memory'
USE_MEMORY = True
DEBUG = True

HAAAAAACK = True
HACK_MEMS = []

################################################################ The actual work.
def memerate(memory: Memory, model, force_memory: bool=False):
    conversation, _memories, tools = memory.getwm() # Retrieve from memory.
    # Total hack!
    if force_memory:
        memtools = []
        if USE_MEMORY: memtools += [tool for tool in tools if tool['function']['name'].startswith('retrieve')]
        tools = memtools if memtools else tools
    system_message = MessageItem(role='system', content=PLANNER_SYS_MESSAGE.strip())
    # Retrieve relevant lessons, gotchas, and tool tips.
    # The system should be aware of what it retrieved from LTM.  Where to put this?
    # if memories: system_message['content'] += f'\nHere are some memories that might be relevant:\n\n {"\n\n".join(memories)}'
    # TODO: We can also grab episodic memories from those pointed to by the retrieved lessons.
    # if episodic: system_message['content'] += f'\nHere are some snippets from your previous conversations that might be relevant:\n\n {"\n\n".join(episodic)}'
    if HAAAAAACK:
        response = to_dict(callgpt([system_message] + HACK_MEMS + conversation, model, tools=tools)[1])
    else:
        response = to_dict(callgpt([system_message] + conversation, model, tools=tools)[1])
    return MessageItem(role='assistant', tool_calls=response)

# Use tools to process the user's message.
def taketurn(message: str, memory, model, toold, sandbox, max_turns=10, max_tool_response_tokens=1024, async_mode=True):
    threads_wait() # Wait for async memory.gc to finish.
    memory.append(MessageItem(role='user', content=message))
    final_response, turns_left = None, max_turns
    # force_memory = True
    force_memory = False
    if DEBUG: print('force_memory:', force_memory)
    while final_response is None and turns_left > 0:
        response = memerate(memory, model, force_memory)
        force_memory = False
        commandlist = response['tool_calls']
        call_results, notes, final_response = process_commands(commandlist, toold, sandbox, max_tool_response_tokens)
        remove_notes(response)
        dnotes = [MessageItem(role='assistant', content=f'Notes:\n{"\n\n".join(notes)}')] if notes else []
        # This loop should be "atomic" (i.e., no memory reorgs in the middle), so that we always match tool calls with their responses.
        for msg in dnotes + [response] + call_results: memory.append(msg)
        if DEBUG: printdebug(max_turns, turns_left, response, call_results, notes, tokenator.count(str(memory.getwm())))
        turns_left -= 1
    if final_response is None: final_response = 'There is as yet insufficient data for a meaningful answer.  Can you provide more guidance?'
    memory.append(MessageItem(role='assistant', content=final_response))
    if DEBUG: pprint(final_response, 'assistant')
    # Flush if we're overflowing.  Yah, this should be done in memory.py eventually.
    wm_tokens = sum([memory.rawturns[t]['metadata']['tokens'] for t in memory.turns])
    if wm_tokens > memory.max_episodic_tokens:
        print('Memory overflow!  Flushing...')
        mbuffer = 1000 # Total hack: we want to leave a buffer.
        memory._flush_episodic(mbuffer + wm_tokens - memory.max_episodic_tokens)
    thread_start(memory.gc, async_mode) # Start async memory.gc.
    return final_response

################################################################ The best GUI ever written!
def update_tools(memory, model, toold):
    newtools = get_new_tools()
    if set(toold) & set(newtools): print(f"Warning: Duplicate tools found: {set(toold) & set(newtools)}")
    toold.update(newtools)
    for func in newtools.values(): memory.add_tool(func_description(func, note=True, model=model))

def main():
    ################ Setup
    vaultdir = mkdir(os.getcwd(), VAULT, addgitignore=True)
    memdir = mkdir(vaultdir, MEMDIR)
    max_tool_response_tokens = 1444
    max_turns = 20
    # models = {'openai': 'gpt-4o', 'claude': 'claude-3-5-sonnet@20240620', 'gemini': 'gemini-2.0-flash'}
    models = {'openai': 'gpt-4o', 'claude': 'claude-3-5-sonnet@20240620', 'gemini': 'gemini-2.0-flash'}
    model = models['openai']
    tools = [reply, shell]
    sandbox = Sandbox(mkdir(vaultdir, 'sandbox'))
    memory = Memory(tokens=2**14, model=model)
    tools += sandbox.list_tools()
    if USE_MEMORY: tools.append(memory.retrieve)
    # if USE_MEMORY: tools.append(memory.get_turn)
    toold = {func.__name__: func for func in tools}
    for func in toold.values(): memory.add_tool(func_description(func, note=True, model=model))
    update_tools(memory, model, toold)
    if USE_MEMORY:
        convs = read_convlogs(memdir)
        templessons, tempconvs = extract_all_lessons(convs, model, debug=DEBUG)
        add_lessons_to_memory(memory, templessons, tempconvs)
    # Hack to get the current turn.
    start_turn = len(memory.rawturns)

    ################################################################ HAAAAAACK!
    if HAAAAAACK: # FIXME:
        # Hacka hack lessons
        # hack_lessons = [x[2] for x in memory._user_manual]
        temp = memory.retrieve('foo', 100)
        # dedup this list, but keep the order.
        newlis = []
        for item in temp:
            if item not in newlis: newlis.append(item)
        # mems = [{'role': 'memory', 'content': x} for x in newlis]
        mems = [{'role': 'system', 'content': x} for x in newlis]
        HACK_MEMS.extend(mems)
        if toold['retrieve'] in tools: tools.remove(toold['retrieve'])
        print('\n\nThese are the lessons I have learned:')
        for i, x in enumerate(HACK_MEMS): print(f"{i}. {x['content']}")
    ################################################################ END HAAAAAACK!

    ################ GUI loop
    ui_prompt, message = UI_Session(), None
    while True:
        print()
        message = ui_prompt()
        if message.startswith('!'):
            ret = run_shell(message[1:])
            pprint(ret, 'toolcall')
            pprint('Note: the above is not visible to the assistant.', 'gui')
        elif message in ['exit', 'bye', 'quit', 'wq']: break
        elif message == 'q!':
            pprint('Exiting.  Conversation will not be saved.\n', 'gui')
            exit(0)
        else:
            update_tools(memory, model, toold)
            #TODO: Add a verification loop here.  (See verify.py)
            assistant_message = taketurn(message, memory, model, toold, sandbox, max_turns, max_tool_response_tokens)
            if not DEBUG: pprint(assistant_message, 'assistant')

    ################ Save conversation and extract lessons
    this_conv = {idx: memory.rawturns[idx] for idx in memory.rawturns if idx >= start_turn}
    print('\nSaving conversation and extracting lessons...')
    write_convlog(memdir, this_conv, model)
    threads_wait()
    # TODO: Do this here so we can digest the lessons in the background.
    if USE_MEMORY:
        # Hack to precompute lessons from this conversation during "downtime".
        conversation, _newconv = extract_conversation(this_conv)
        # newlessons = extract_lessons(conversation, claude_model)
        newlessons = extract_lessons(conversation, model)
        print(f'Learned {len(newlessons)} new lessons.')
        if DEBUG: print_lessons(newlessons)



sys.path.pop(0)  # ← Remove parent of agent001

