"""
core/synthesis.py
─────────────────
Generates one final grounded answer with citations from all retrieved chunks.
"""

import re
from typing import List, Dict, Any, Optional

from groq import Groq

from config.settings import GROQ_API_KEY, GROQ_MODEL


_groq_client: Optional[Groq] = None

def _get_groq_client() -> Groq:
    global _groq_client
    if _groq_client is None:
        _groq_client = Groq(api_key=GROQ_API_KEY)
    return _groq_client


SYNTHESIS_PROMPT = """You are a precise research assistant. Your job is to answer
the user's question using ONLY the context chunks provided below.

STRICT RULES:
1. Answer exclusively from the provided context. Do NOT use any outside knowledge.
2. Every factual claim must end with a citation in this exact format: [paper_name, p.PAGE]
3. If a claim is supported by multiple chunks, cite all of them.
4. If the context does not contain enough information to answer fully, explicitly
   state: "The provided documents do not contain sufficient information about [topic]."
5. Do NOT speculate, infer beyond what is stated, or fill gaps from your training data.
6. Write in clear, flowing prose. Do not use bullet points unless listing distinct items.

─────────────────────────────────────────────────────
ORIGINAL QUESTION:
{original_question}

─────────────────────────────────────────────────────
SUB-QUESTIONS AND RETRIEVED CONTEXT:

{context_block}

─────────────────────────────────────────────────────
Now write a single, coherent answer to the original question using only
the context above. Include inline citations [paper_name, p.PAGE] throughout."""


def _format_context_block(retrieval_results: List[Dict[str, Any]]) -> str:
    sections = []

    for i, result in enumerate(retrieval_results, 1):
        sub_q  = result["sub_question"]
        chunks = result["chunks"]
        lines  = [f"## Sub-question {i}: {sub_q}"]

        if not chunks:
            lines.append("  No relevant chunks found for this sub-question.")
        else:
            for chunk in chunks:
                source_tag = (
                    f"[Source: {chunk['paper_name']} | Page: {chunk['page_number']}]"
                )
                lines.append(f"\n{source_tag}")
                lines.append(chunk["text"])
                lines.append("---")

        sections.append("\n".join(lines))

    return "\n\n".join(sections)


def _extract_sources(
    retrieval_results: List[Dict[str, Any]],
    answer: str,
) -> List[Dict[str, Any]]:
    cited = set()

    citation_pattern = re.compile(r'\[([^\]]+),\s*p\.(\d+)\]')
    matches = citation_pattern.findall(answer)
    for paper_name, page_str in matches:
        cited.add((paper_name.strip(), int(page_str)))

    if not cited:
        for result in retrieval_results:
            for chunk in result["chunks"]:
                cited.add((chunk["paper_name"], chunk["page_number"]))

    return [
        {"paper_name": name, "page_number": page}
        for name, page in sorted(cited)
    ]


def synthesize_answer(
    original_question: str,
    retrieval_results: List[Dict[str, Any]],
) -> Dict[str, Any]:
    total_chunks = sum(len(r["chunks"]) for r in retrieval_results)

    if total_chunks == 0:
        return {
            "answer": (
                "No relevant content was found in the indexed documents "
                "to answer your question. Please ensure the relevant PDFs "
                "have been uploaded and indexed."
            ),
            "sources": [],
        }

    client        = _get_groq_client()
    context_block = _format_context_block(retrieval_results)

    prompt = SYNTHESIS_PROMPT.format(
        original_question=original_question,
        context_block=context_block,
    )

    print(f"  [Synthesis] Sending {total_chunks} chunks to LLM ...")

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=1500,
    )

    answer  = response.choices[0].message.content.strip()
    sources = _extract_sources(retrieval_results, answer)

    print(f"  [Synthesis] Done. Sources cited: {len(sources)}")

    return {
        "answer":  answer,
        "sources": sources,
    }