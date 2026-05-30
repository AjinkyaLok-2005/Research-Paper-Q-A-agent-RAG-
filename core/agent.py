"""
core/agent.py
─────────────
LangGraph-based Query Decomposition Agent.
"""

import json
import re
from typing import List, TypedDict, Optional

from groq import Groq
from langgraph.graph import StateGraph, END

from config.settings import GROQ_API_KEY, GROQ_MODEL, MAX_SUB_QUESTIONS


class AgentState(TypedDict):
    question:      str
    sub_questions: List[str]
    is_complex:    bool


_groq_client: Optional[Groq] = None

def _get_groq_client() -> Groq:
    global _groq_client
    if _groq_client is None:
        _groq_client = Groq(api_key=GROQ_API_KEY)
    return _groq_client


ANALYZE_PROMPT = """You are an expert at analyzing research questions.

Your task: decide if the following question is SIMPLE or COMPLEX.

SIMPLE  = asks about ONE specific concept, method, result, or fact.
          Examples:
          - "What dataset did the authors use?"
          - "What is the main contribution of the paper?"

COMPLEX = asks about MULTIPLE distinct topics, requires comparison,
          or spans several independent concepts that need separate lookups.
          Examples:
          - "How does the attention mechanism work and how does it compare to RNNs?"
          - "What are the datasets used, the evaluation metrics, and the final results?"

Question: {question}

Respond with exactly one word — either SIMPLE or COMPLEX. Nothing else."""


DECOMPOSE_PROMPT = """You are an expert at breaking down complex research questions.

Original question: {question}

Break this into {max_sub} or fewer focused sub-questions. Each sub-question must:
- Cover exactly ONE distinct topic or concept
- Be self-contained and answerable independently
- Be specific enough to retrieve relevant text from a research paper

Return your answer as a JSON array of strings. Example format:
["Sub-question 1?", "Sub-question 2?", "Sub-question 3?"]

Return ONLY the JSON array. No explanation, no preamble, no markdown."""


def analyze_node(state: AgentState) -> AgentState:
    client = _get_groq_client()
    prompt = ANALYZE_PROMPT.format(question=state["question"])

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=10,
    )

    answer     = response.choices[0].message.content.strip().upper()
    is_complex = answer.startswith("COMPLEX")
    print(f"  [Agent] Complexity: {'COMPLEX' if is_complex else 'SIMPLE'}")
    return {**state, "is_complex": is_complex}


def decompose_node(state: AgentState) -> AgentState:
    client = _get_groq_client()
    prompt = DECOMPOSE_PROMPT.format(
        question=state["question"],
        max_sub=MAX_SUB_QUESTIONS,
    )

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=300,
    )

    raw           = response.choices[0].message.content.strip()
    sub_questions = _parse_sub_questions(raw, fallback=state["question"])

    print(f"  [Agent] Decomposed into {len(sub_questions)} sub-questions:")
    for i, sq in enumerate(sub_questions, 1):
        print(f"    {i}. {sq}")

    return {**state, "sub_questions": sub_questions}


def passthrough_node(state: AgentState) -> AgentState:
    print("  [Agent] Simple question — no decomposition needed.")
    return {**state, "sub_questions": [state["question"]]}


def route_by_complexity(state: AgentState) -> str:
    return "decompose" if state["is_complex"] else "passthrough"


def _parse_sub_questions(raw: str, fallback: str) -> List[str]:
    match = re.search(r'\[.*?\]', raw, re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group())
            if isinstance(parsed, list) and all(isinstance(q, str) for q in parsed):
                cleaned = [q.strip() for q in parsed if q.strip()]
                return cleaned[:MAX_SUB_QUESTIONS] if cleaned else [fallback]
        except json.JSONDecodeError:
            pass
    print("  [Agent] Warning: could not parse sub-questions JSON. Using original question.")
    return [fallback]


def _build_graph() -> StateGraph:
    graph = StateGraph(AgentState)

    graph.add_node("analyze",     analyze_node)
    graph.add_node("decompose",   decompose_node)
    graph.add_node("passthrough", passthrough_node)

    graph.set_entry_point("analyze")

    graph.add_conditional_edges(
        "analyze",
        route_by_complexity,
        {
            "decompose":   "decompose",
            "passthrough": "passthrough",
        }
    )

    graph.add_edge("decompose",   END)
    graph.add_edge("passthrough", END)

    return graph.compile()


_graph = _build_graph()


def decompose_question(question: str) -> List[str]:
    if not question.strip():
        return [question]

    initial_state: AgentState = {
        "question":      question.strip(),
        "sub_questions": [],
        "is_complex":    False,
    }

    try:
        result = _graph.invoke(initial_state)
        return result["sub_questions"]
    except Exception as e:
        print(f"  [Agent] Error: {e}. Falling back to original question.")
        return [question]