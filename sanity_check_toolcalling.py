# sanity_check_toolcalling.py — THROWAWAY, gitignored, do not commit

from typing_extensions import TypedDict, Annotated
import operator

from langchain.tools import tool
from langchain.chat_models import init_chat_model
from langchain.messages import SystemMessage, HumanMessage, AnyMessage
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from dotenv import load_dotenv
import sys

load_dotenv()

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# --- Step 1: Define tools and model ---
@tool
def dummy_weather_lookup(city: str) -> str:
    """Look up the current weather for a given city.

    Args:
        city: The city to check weather for
    """
    return f"It's sunny in {city}."


tools = [dummy_weather_lookup]

model = init_chat_model("claude-sonnet-5")
model_with_tools = model.bind_tools(tools)


# --- Step 2: Define state ---
class MessagesState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    llm_calls: int


# --- Step 3: Define model node ---
def llm_call(state: MessagesState):
    """LLM decides whether to call a tool or not"""
    return {
        "messages": [
            model_with_tools.invoke(
                [SystemMessage(content="You are a helpful assistant.")]
                + state["messages"]
            )
        ],
        "llm_calls": state.get("llm_calls", 0) + 1,
    }


# --- Step 4: Define tool node (prebuilt — locked design choice) ---
tool_node = ToolNode(tools)


# --- Step 5/6: Build and compile, using tools_condition for routing ---
agent_builder = StateGraph(MessagesState)
agent_builder.add_node("llm_call", llm_call)
agent_builder.add_node("tool_node", tool_node)
agent_builder.add_edge(START, "llm_call")
agent_builder.add_conditional_edges(
    "llm_call",
    tools_condition,
    {"tools": "tool_node", END: END},
)
agent_builder.add_edge("tool_node", "llm_call")
agent = agent_builder.compile()


# --- Sanity checks ---
if __name__ == "__main__":
    print("=== Query that SHOULD trigger the tool ===")
    result1 = model_with_tools.invoke([HumanMessage(content="What's the weather in Fresno?")])
    print("tool_calls:", result1.tool_calls)

    print("\n=== Query that should NOT trigger the tool ===")
    result2 = model_with_tools.invoke([HumanMessage(content="What is 12 * 7?")])
    print("tool_calls:", result2.tool_calls)

    print("\n=== Full graph run ===")
    messages = [HumanMessage(content="What's the weather in Fresno?")]
    final_state = agent.invoke({"messages": messages})
    for m in final_state["messages"]:
        m.pretty_print()