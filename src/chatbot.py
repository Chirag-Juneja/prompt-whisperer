import io
from PIL import Image
from typing import Annotated
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages

from langchain.chat_models import init_chat_model
from langchain_community.tools import DuckDuckGoSearchResults
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver

# Globals
MODEL = "qwen3:4b"
PROVIDER = "ollama"

# Memory
# Use SqliteSaver or PostgresSaver in prod
memory = MemorySaver()

# Tools
search = DuckDuckGoSearchResults(
    region="in-en", max_results=3, output_format="json")

tools = [search]


# LLM
llm = init_chat_model(MODEL, model_provider=PROVIDER)

llm_with_tools = llm.bind_tools(tools)


# State


class State(TypedDict):
    messages: Annotated[list, add_messages]

# Display Graph


def show_graph(graph):
    Image.open(io.BytesIO(graph.get_graph().draw_mermaid_png())).show()

# Nodes


def chatbot(state: State):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}


tool_node = ToolNode(tools=tools)


# Graph
graph_builder = StateGraph(State)

graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("tools", tool_node)
graph_builder.add_conditional_edges(
    "chatbot", tools_condition)
graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge(START, "chatbot")
graph = graph_builder.compile(checkpointer=memory)

# show_graph(graph)

config = {"configurable": {"thread_id": "1"}}


def stream_graph_updates(user_input: str):
    events = graph.stream(
        {"messages": [{"role": "user", "content": user_input}]},
        config,
        stream_mode="values"
    )
    for event in events:
        event["messages"][-1].pretty_print()


# user_input = "Hi, My name is Bob"
# stream_graph_updates(user_input)

if __name__ == "__main__":
    while True:
        try:
            user_input = input("User: ")
            if user_input.lower() in ["quit", "exit", "q"]:
                print("Goodbye")
                break
            stream_graph_updates(user_input)
        except Exception as e:
            print(e)
            break
