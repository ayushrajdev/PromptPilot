"""
Requirement Agent (LangGraph + Gemini)
======================================
Flow:
    START -> analyze -> (incomplete) -> ask_user -> analyze -> ...   (loops until complete)
                     -> (complete)   -> confirm  -> (approved) -> handoff -> END
                                                 -> (changes)  -> analyze  (loop again)

Run:
    python requirement_agent.py
"""

import os
import uuid
from typing import Annotated, List, Optional, Type, TypeVar, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import Command, interrupt
from pydantic import BaseModel, Field

load_dotenv()  # reads GOOGLE_API_KEY from the .env file

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
MAX_QUESTION_ROUNDS = int(os.getenv("REQ_AGENT_MAX_ROUNDS", "8"))
MAX_QUESTIONS_PER_ROUND = 3

if not os.getenv("GOOGLE_API_KEY"):
    raise RuntimeError("GOOGLE_API_KEY is missing. Copy .env.example to .env and add your key.")

llm = ChatGoogleGenerativeAI(model=MODEL_NAME, temperature=0)


# --------------------------------------------------------------------------- #
# Schemas
# (No default values on purpose: Gemini's structured output works most reliably
#  when every field is required. Empty string / empty list = still unknown.)
# --------------------------------------------------------------------------- #
class Requirements(BaseModel):
    project_name: str = Field(description="Name of the product/project, empty if unknown")
    summary: str = Field(description="One or two sentence description of the product")
    problem_statement: str = Field(description="What problem it solves and why")
    target_users: List[str] = Field(description="Who will use it")
    core_features: List[str] = Field(description="Must-have features")
    nice_to_have_features: List[str] = Field(description="Optional features")
    user_flows: List[str] = Field(description="Key end-to-end user journeys")
    platforms: List[str] = Field(description="Web, mobile, CLI, API, etc.")
    tech_stack_preferences: List[str] = Field(description="Languages, frameworks, databases requested")
    integrations: List[str] = Field(description="Third-party services / APIs")
    data_and_storage: str = Field(description="Data entities, volume, storage needs")
    authentication_and_roles: str = Field(description="Auth method, user roles, permissions")
    non_functional: List[str] = Field(description="Performance, scalability, security, compliance, uptime")
    constraints: List[str] = Field(description="Budget, timeline, team, legal")
    success_criteria: List[str] = Field(description="How success is measured")
    out_of_scope: List[str] = Field(description="Explicitly excluded items")
    assumptions: List[str] = Field(description="Assumptions made when the user could not answer")


def empty_requirements() -> dict:
    return {
        name: ("" if field.annotation is str else [])
        for name, field in Requirements.model_fields.items()
    }


class AnalysisResult(BaseModel):
    requirements: Requirements = Field(description="Full, merged requirements so far")
    is_complete: bool = Field(
        description="True only when there is enough detail to design and build the product"
    )
    missing_areas: List[str] = Field(description="Topics that are still unclear")
    next_questions: List[str] = Field(
        description=f"At most {MAX_QUESTIONS_PER_ROUND} specific, non-repeated follow-up questions. "
        "Empty list if complete."
    )


class ApprovalDecision(BaseModel):
    approved: bool = Field(description="True only if the user clearly accepted the summary as-is")
    feedback: str = Field(description="Requested changes if not approved, else empty string")


T = TypeVar("T", bound=BaseModel)


def structured_call(schema: Type[T], messages: List[BaseMessage], retries: int = 2) -> T:
    """Call Gemini with structured output; retry if it returns nothing parseable."""
    chain = llm.with_structured_output(schema)
    last_error: Optional[Exception] = None
    for _ in range(retries + 1):
        try:
            result = chain.invoke(messages)
            if result is not None:
                return result
        except Exception as exc:  # noqa: BLE001
            last_error = exc
    raise RuntimeError(f"Gemini did not return valid {schema.__name__}: {last_error}")


# --------------------------------------------------------------------------- #
# State
# --------------------------------------------------------------------------- #
class RequirementState(TypedDict, total=False):
    messages: Annotated[List[BaseMessage], add_messages]
    requirements: dict
    is_complete: bool
    missing_areas: List[str]
    pending_questions: List[str]
    round_count: int
    approved: bool
    status: str  # gathering | awaiting_approval | approved


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
ANALYST_SYSTEM_PROMPT = f"""You are a senior product analyst / business analyst gathering requirements
for a software product before any design or coding begins.

Your job on every turn:
1. Read the full conversation and the requirements collected so far.
2. MERGE any new information from the user into the requirements object (never drop known info).
3. Decide if the requirements are complete enough to design and build the product.
   Cover at minimum: purpose/problem, target users, core features, key user flows, platforms,
   auth/roles, data, integrations, non-functional needs, constraints, success criteria, scope limits.
4. If NOT complete, write up to {MAX_QUESTIONS_PER_ROUND} sharp follow-up questions:
   - Ask about the most important gaps first.
   - Be specific and concrete; offer options/examples where helpful.
   - NEVER repeat a question the user already answered.
   - If the user says "not sure" / "you decide", record a sensible default under `assumptions`
     and move on.
5. If the user has requested changes to a summary you showed, apply them to the requirements.

Only set is_complete=true when a development team could start architecture from this document."""


def _text(content) -> str:
    """Gemini can return content as a plain string or a list of parts. Normalize to str."""
    if isinstance(content, str):
        return content
    parts = []
    for part in content:
        if isinstance(part, str):
            parts.append(part)
        elif isinstance(part, dict) and part.get("type") == "text":
            parts.append(part.get("text", ""))
    return "".join(parts)


def _transcript(messages: List[BaseMessage]) -> str:
    lines = []
    for m in messages:
        role = "User" if isinstance(m, HumanMessage) else "Agent"
        lines.append(f"{role}: {_text(m.content)}")
    return "\n".join(lines)


def _render_requirements(req: dict) -> str:
    labels = [
        ("project_name", "Project Name"),
        ("summary", "Summary"),
        ("problem_statement", "Problem Statement"),
        ("target_users", "Target Users"),
        ("core_features", "Core Features"),
        ("nice_to_have_features", "Nice-to-have Features"),
        ("user_flows", "Key User Flows"),
        ("platforms", "Platforms"),
        ("tech_stack_preferences", "Tech Stack Preferences"),
        ("integrations", "Integrations"),
        ("data_and_storage", "Data & Storage"),
        ("authentication_and_roles", "Authentication & Roles"),
        ("non_functional", "Non-functional Requirements"),
        ("constraints", "Constraints"),
        ("success_criteria", "Success Criteria"),
        ("out_of_scope", "Out of Scope"),
        ("assumptions", "Assumptions Made"),
    ]
    out = ["# Requirements Summary", ""]
    for key, label in labels:
        value = req.get(key)
        if not value:
            continue
        out.append(f"## {label}")
        if isinstance(value, list):
            out.extend(f"- {item}" for item in value)
        else:
            out.append(str(value))
        out.append("")
    return "\n".join(out).strip()


APPROVE_WORDS = {
    "yes", "y", "yep", "yeah", "ok", "okay", "approve", "approved", "lgtm",
    "looks good", "good to go", "go ahead", "proceed", "confirmed", "confirm",
    "done", "perfect", "correct",
}


# --------------------------------------------------------------------------- #
# Nodes
# --------------------------------------------------------------------------- #
def analyze_node(state: RequirementState) -> dict:
    """Merge latest user info into requirements, then decide: ask more or finish."""
    current = state.get("requirements") or empty_requirements()
    round_count = state.get("round_count", 0)

    prompt = [
        SystemMessage(content=ANALYST_SYSTEM_PROMPT),
        HumanMessage(
            content=(
                "REQUIREMENTS COLLECTED SO FAR (JSON):\n"
                f"{Requirements(**current).model_dump_json(indent=2)}\n\n"
                "CONVERSATION SO FAR:\n"
                f"{_transcript(state['messages'])}\n\n"
                "Update the requirements and decide whether more questions are needed."
            )
        ),
    ]

    result = structured_call(AnalysisResult, prompt)
    updated = result.requirements.model_dump()

    # Safety valve: stop endless questioning, record what is still unknown as assumptions
    hit_limit = round_count >= MAX_QUESTION_ROUNDS
    if hit_limit and not result.is_complete:
        for area in result.missing_areas:
            note = f"Not fully specified by user: {area} (to be decided during design)"
            if note not in updated["assumptions"]:
                updated["assumptions"].append(note)

    complete = result.is_complete or hit_limit or not result.next_questions

    return {
        "requirements": updated,
        "is_complete": complete,
        "missing_areas": result.missing_areas,
        "pending_questions": [] if complete else result.next_questions[:MAX_QUESTIONS_PER_ROUND],
        "round_count": round_count + 1,
        "status": "awaiting_approval" if complete else "gathering",
    }


def route_after_analyze(state: RequirementState) -> str:
    return "confirm" if state.get("is_complete") else "ask_user"


def ask_user_node(state: RequirementState) -> dict:
    """Ask the user the cross-questions and pause until they reply."""
    questions = state["pending_questions"]
    numbered = "\n".join(f"{i}. {q}" for i, q in enumerate(questions, 1))
    agent_text = f"To make sure I understand your product correctly, I have a few questions:\n\n{numbered}"

    answer = interrupt({"type": "questions", "message": agent_text, "questions": questions})

    return {
        "messages": [AIMessage(content=agent_text), HumanMessage(content=str(answer))],
        "pending_questions": [],
    }


def confirm_node(state: RequirementState) -> dict:
    """Show the final requirements and wait for approve / change request."""
    summary = _render_requirements(state["requirements"])
    prompt_text = (
        f"{summary}\n\n---\nAre these requirements good to go? "
        "Reply 'yes' to approve, or tell me what to change."
    )

    reply = str(interrupt({"type": "approval", "message": prompt_text, "summary": summary})).strip()

    # Fast path for obvious approvals, Gemini for anything nuanced
    if reply.lower().strip(" .!") in APPROVE_WORDS:
        decision = ApprovalDecision(approved=True, feedback="")
    else:
        decision = structured_call(
            ApprovalDecision,
            [
                SystemMessage(
                    content="Decide if the user approved the requirements summary. "
                    "approved=true only if they clearly accept it as-is. "
                    "If they ask for any change or addition, approved=false and put it in feedback."
                ),
                HumanMessage(content=f"Summary shown:\n{summary}\n\nUser reply:\n{reply}"),
            ],
        )

    if decision.approved:
        return {
            "approved": True,
            "status": "approved",
            "messages": [AIMessage(content=prompt_text), HumanMessage(content=reply)],
        }

    return {
        "approved": False,
        "status": "gathering",
        "round_count": 0,  # fresh question budget for the revision
        "messages": [
            AIMessage(content=prompt_text),
            HumanMessage(content=f"[Requested changes to requirements] {decision.feedback or reply}"),
        ],
    }


def route_after_confirm(state: RequirementState) -> str:
    return "handoff" if state.get("approved") else "analyze"


def handoff_node(state: RequirementState) -> dict:
    """
    Requirements are approved. Plug the next agent in here
    (Architecture Agent -> Architecture Validator -> Engineering Preferences ...).
    `state["requirements"]` is the approved, structured requirements object.
    """
    return {
        "status": "approved",
        "messages": [AIMessage(content="Requirements approved. Proceeding to the next stage.")],
    }


# --------------------------------------------------------------------------- #
# Graph
# --------------------------------------------------------------------------- #
def build_graph():
    g = StateGraph(RequirementState)

    g.add_node("analyze", analyze_node)
    g.add_node("ask_user", ask_user_node)
    g.add_node("confirm", confirm_node)
    g.add_node("handoff", handoff_node)

    g.add_edge(START, "analyze")
    g.add_conditional_edges("analyze", route_after_analyze, {"ask_user": "ask_user", "confirm": "confirm"})
    g.add_edge("ask_user", "analyze")
    g.add_conditional_edges("confirm", route_after_confirm, {"handoff": "handoff", "analyze": "analyze"})
    g.add_edge("handoff", END)

    # interrupt() requires a checkpointer. Swap MemorySaver for SqliteSaver/PostgresSaver in prod.
    return g.compile(checkpointer=MemorySaver())


graph = build_graph()


# --------------------------------------------------------------------------- #
# CLI runner
# --------------------------------------------------------------------------- #
def run_cli(initial_query: Optional[str] = None) -> dict:
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}

    if not initial_query:
        initial_query = input("What do you want to build?\n> ").strip()

    payload = {
        "messages": [HumanMessage(content=initial_query)],
        "requirements": empty_requirements(),
        "round_count": 0,
        "status": "gathering",
    }

    result = graph.invoke(payload, config)

    # Each interrupt = the agent is waiting on the user. Resume until the graph finishes.
    while "__interrupt__" in result:
        request = result["__interrupt__"][0].value
        print(f"\n🤖 Agent:\n{request['message']}\n")
        user_reply = input("You > ").strip()
        result = graph.invoke(Command(resume=user_reply), config)

    print("\n✅ Requirements approved. Final structured output:\n")
    print(Requirements(**result["requirements"]).model_dump_json(indent=2))
    return result


if __name__ == "__main__":
    run_cli()
