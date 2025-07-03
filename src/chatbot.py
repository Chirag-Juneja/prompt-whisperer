import io
import json
from PIL import Image
from typing import Annotated
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

from langchain.chat_models import init_chat_model
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_core.messages import ToolMessage
from langgraph.prebuilt import ToolNode, tools_condition

# Globals
MODEL = "llama3.2:3b"
PROVIDER = "ollama"

# Tools
search = DuckDuckGoSearchResults(
    region="in-en", max_output=3, output_format="json")

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


class BasicToolNode:
    def __init__(self, tools):
        self.tools_by_name = {tool.name: tool for tool in tools}

    def __call__(self, inputs):
        if messages := inputs.get("messages", []):
            message = messages[-1]
        else:
            raise ValueError("No message found in input")
        outputs = []
        for tool_call in message.tool_calls:
            tool_result = self.tools_by_name[tool_call["name"]].invoke(
                tool_call["args"]
            )
            outputs.append(
                ToolMessage(
                    content=json.dumps(tool_result),
                    name=tool_call["name"],
                    tool_call_id=tool_call["id"]
                )
            )
        return {"messages": outputs}


tool_node = ToolNode(tools=tools)

# Routing


def route_tools(state: State):
    if isinstance(state, list):
        ai_message = state[-1]
    elif messages := state.get("messages", []):
        ai_message = messages[-1]
    if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
        return "tools"
    return END


# Graph
graph_builder = StateGraph(State)

graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("tools", tool_node)
graph_builder.add_conditional_edges(
    "chatbot", tools_condition)
graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge(START, "chatbot")
graph = graph_builder.compile()

# show_graph(graph)


def stream_graph_updates(user_input: str):
    for event in graph.stream({"messages": [{"role": "user", "content": user_input}]}):
        for value in event.values():
            # print("Assistant:", value["messages"][-1].content)
            value["messages"][-1].pretty_print()


# user_input = "Who is the PM of India?"
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
