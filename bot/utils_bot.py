import os
from typing import Sequence
from typing_extensions import Annotated, TypedDict
import time

from langchain.chat_models import init_chat_model
from langchain_mistralai import ChatMistralAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage, trim_messages
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools import tool

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import create_react_agent


# model = init_chat_model("mistral-large-latest", model_provider="mistralai")
model = ChatMistralAI(model="mistral-large-latest")

@tool
def get_files(dir):
    """Return list of files in specified directory"""
    time.sleep(10)
    return os.listdir(dir)

tools = [get_files]
memory = MemorySaver()
agent = create_react_agent(model, tools, checkpointer=memory)
# model = model.bind_tools([get_files])


class State(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    language: str

workflow = StateGraph(state_schema=State)

trimmer = trim_messages(
    max_tokens=65,
    strategy="last",
    token_counter=model,
    include_system=True,
    allow_partial=False,
    start_on="human",
)

prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a helpful OCR assistant.
            Introduce yourself to user.
            Do your best to answer all questions in {language}
            """,
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

def call_model(state):
    messages = trimmer.invoke(state['messages']) # Cut message history to fit in the context window
    prompt = prompt_template.invoke(
        {'messages': messages, 'language': state['language']}
    )
    response = agent.invoke(prompt)
    breakpoint()
    return {'messages': response['messages']}

workflow.add_node('model', call_model)
workflow.add_edge(START, 'model')

memory = MemorySaver()
app = workflow.compile(checkpointer=memory)


config = {'configurable': {'thread_id': 'aaa111'}}

# Invoke the app
msg_history = [
    # SystemMessage(content='You are a helpful OCR assistant'),
]

language = 'English'
user_msg = None
while user_msg != 'quit':

    user_msg = HumanMessage(content=input('You: '))
    user_msg.pretty_print()
    msg_history.append(user_msg)

    output = app.invoke(
        {'messages': msg_history, 'language': language},
        config,
    )
    msg_history = output['messages']
    ai_msg = msg_history[-1]
    ai_msg.pretty_print()
