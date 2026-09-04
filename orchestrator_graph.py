"""
Stage B.3 — Orchestrator Graph (StateGraph)

Wires the B.2 retrieval tool into a LangGraph StateGraph with scope-routing
logic folded into the orchestrator's own system prompt (no upstream classifier
node, per the locked architecture decision).

Assumes B.2 already defines a tool-decorated function that wraps
rag_pipeline.retrieve() and returns raw {id, text, source, distance} chunks.
Update the import below to match your actual B.2 module/filename.
"""

from typing import Annotated
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import SystemMessage
from langchain_anthropic import ChatAnthropic

# --- B.2 tool import -----------------------------------------------------
# From agent_wraps_retrieve.py (renamed from agent-wraps-retrieve.py —
# hyphens aren't valid in Python module names).
from agent_wraps_retrieve import retrieve_tool  # noqa: E402


# --- Verbatim caveat openers (do not paraphrase) --------------------------
# Used both in the system prompt and to detect caveat usage for logging.
CAVEAT_JURISDICTION = (
    "This system's corpus only covers Jurisdiction D (Noridian, "
    "California), so I can't confirm the criteria for that jurisdiction, "
    "but here's what I can tell you generally:"
)
CAVEAT_NON_DME = (
    "This system's corpus is limited to Durable Medical Equipment (DME) "
    "coverage, so this falls outside it, but here's what I can tell you "
    "generally:"
)
CAVEAT_NON_TRADITIONAL_PREFIX = (
    "This system's corpus covers traditional Medicare coverage policy "
    "only, not"
)  # full sentence fills in [plan type] and ends "...but here's what I can tell you generally:"

ALL_CAVEAT_PREFIXES = (CAVEAT_JURISDICTION, CAVEAT_NON_DME, CAVEAT_NON_TRADITIONAL_PREFIX)


SYSTEM_PROMPT = SystemMessage(content="""\
You have one tool: retrieve_tool. It searches DME (Durable
Medical Equipment) LCDs (Local Coverage Determinations) for
Jurisdiction D (Noridian), California, under traditional Medicare only.

- In scope -> call the tool and answer from its results.
- Different MAC jurisdiction -> open your reply with exactly:
  "This system's corpus only covers Jurisdiction D (Noridian,
  California), so I can't confirm the criteria for that jurisdiction,
  but here's what I can tell you generally:"
- Non-DME equipment -> open with exactly:
  "This system's corpus is limited to Durable Medical Equipment (DME)
  coverage, so this falls outside it, but here's what I can tell you
  generally:"
- Non-traditional-Medicare plan (Medicare Advantage, Medicaid, private
  insurance) -> open with exactly:
  "This system's corpus covers traditional Medicare coverage policy
  only, not [plan type], but here's what I can tell you generally:"
- Fully unrelated to Medicare/DME -> answer directly, no disclaimer.

Do not call the tool for any of the three caveat cases above.

Your last sentence must be your last substantive fact, citation, or
recommendation. Add nothing after it — no closing remark that offers,
invites, hints at, or references further help, comparison, or
follow-up in any phrasing. This includes (not limited to) sentences
containing: "if you'd like", "let me know if...", "I can also...",
"I'd be happy to...", "would you like...", "I can address...". This
system does not retain conversation state between turns, so any such
offer is unactionable to the user.

Wrong (do not do this):
"...Standard fee schedule limits apply. If you'd like, I can also
walk through the accessory coverage rules."

Right:
"...Standard fee schedule limits apply."
""")


def extract_text(content) -> str:
    """Pull the plain-text answer out of a message's content.

    With extended thinking on, ChatAnthropic returns content as a list of
    blocks (thinking + text, sometimes redacted_thinking) rather than a
    plain string. Concatenate only the 'text' blocks.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            block.get("text", "")
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        )
    return str(content)


class State(TypedDict):
    messages: Annotated[list, add_messages]


model = ChatAnthropic(model="claude-sonnet-5").bind_tools([retrieve_tool])


def orchestrator(state: State):
    messages = state["messages"]
    # Prepend the system prompt only if it isn't already the first message.
    if not messages or not isinstance(messages[0], SystemMessage):
        messages = [SYSTEM_PROMPT] + messages

    response = model.invoke(messages)

    # --- B.3 observability logging (point 33) ---
    tool_calls = getattr(response, "tool_calls", None) or []
    called = bool(tool_calls)
    queries = [tc["args"].get("query") for tc in tool_calls] if called else []
    final_text = extract_text(response.content)
    caveat_used = any(final_text.startswith(p) for p in ALL_CAVEAT_PREFIXES)

    print(
        f"[orchestrator] tool_called={called} num_tool_calls={len(tool_calls)} "
        f"queries={queries!r} caveat_used={caveat_used}"
    )

    return {"messages": [response]}


graph_builder = StateGraph(State)
graph_builder.add_node("orchestrator", orchestrator)
graph_builder.add_node("tools", ToolNode([retrieve_tool]))
graph_builder.add_edge(START, "orchestrator")
graph_builder.add_conditional_edges("orchestrator", tools_condition)  # -> "tools" or END
graph_builder.add_edge("tools", "orchestrator")
graph = graph_builder.compile()


if __name__ == "__main__":
    # B.4 test set — 15 queries, mixed across buckets per the locked B.4 design:
    # in-scope (tool should be called), each of the three caveat buckets,
    # fully unrelated, and a couple of genuinely ambiguous/borderline ones.
    test_queries = [
        # --- in-scope (tool should be called) ---
        "What are the coverage criteria for a Group II oxygen concentrator?",
        "What documentation is required for a power wheelchair under the DME LCD?",
        "What are the qualifying blood gas values for home oxygen therapy?",
        "Is a hospital bed covered, and what functional criteria does the patient need to meet?",

        # --- wrong jurisdiction (caveat 1) ---
        "What does Jurisdiction F require for a hospital bed?",
        "Under Jurisdiction K, what's the coverage policy for a CPAP device?",

        # --- non-DME (caveat 2) ---
        "Is a knee replacement surgery covered under Medicare?",
        "Does Medicare cover physical therapy visits after a stroke?",
        "What are the coverage criteria for home health nursing visits?",

        # --- non-traditional plan (caveat 3) ---
        "Does my Medicare Advantage plan cover a wheelchair?",
        "Will Medicaid cover a hospital bed for home use?",

        # --- fully unrelated ---
        "What's the capital of France?",
        "Can you recommend a good recipe for banana bread?",

        # --- ambiguous / borderline ---
        # DME-adjacent but vague on device/jurisdiction/plan.
        "My doctor mentioned I might need equipment for sleep apnea — what's covered?",
        # DME-adjacent, but a wound vac plausibly falls under non-DME/supply territory.
        "I'm getting a wound vac after surgery — is that covered?",
    ]
    for q in test_queries:
        print(f"\n=== Query: {q} ===")
        result = graph.invoke({"messages": [{"role": "user", "content": q}]})
        print(extract_text(result["messages"][-1].content))