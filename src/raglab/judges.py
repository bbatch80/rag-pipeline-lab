"""Cross-family judging, pinned models. GPT judges Claude's answers; Claude
judges GPT's answers — no judge ever grades its own family, so
self-preference is eliminated by construction. Judges are part of the
measuring instrument: changing them breaks metric-history comparability."""

import json
import re

CLAUDE_JUDGE = "claude-sonnet-4-6"   # grades OpenAI generator's answers
OPENAI_JUDGE = "gpt-5"               # grades Claude generator's answers

RUBRIC = """You are grading an answer produced by a retrieval-augmented
assistant. You are given the question, the reference (golden) answer, the
context chunks the assistant was allowed to use, and the assistant's answer.

Score STRICTLY as JSON, no other text:
{"correctness": <0.0-1.0 does the answer state the facts of the reference
answer (numbers and qualifiers must match)>,
 "faithfulness": <0.0-1.0 is every claim in the answer supported by the
provided chunks (1.0 if the answer correctly declines to answer)>,
 "abstained": <true if the answer declines/refuses to answer, else false>}

For questions marked UNANSWERABLE, a correct response DECLINES to answer:
score correctness 1.0 iff the answer abstains, 0.0 if it provides any
substantive answer, regardless of whether that answer is factually right."""


def _judge_message(item: dict, payload: dict, answer: str) -> str:
    chunks = "\n---\n".join(
        c["text"][:1200] for c in payload.get("chunks", [])
    ) or "(no chunks — payload status was out_of_scope or insufficient_evidence)"
    tag = " [UNANSWERABLE]" if item.get("unanswerable") else ""
    return (
        f"QUESTION{tag}: {item['question']}\n\n"
        f"REFERENCE ANSWER: {item['answer']}\n\n"
        f"CONTEXT CHUNKS:\n{chunks}\n\n"
        f"ASSISTANT ANSWER:\n{answer}"
    )


def _parse(text: str) -> dict:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    scores = json.loads(match.group(0))
    return {
        "correctness": float(scores["correctness"]),
        "faithfulness": float(scores["faithfulness"]),
        "abstained": bool(scores["abstained"]),
    }


def judge_with_claude(item: dict, payload: dict, answer: str) -> dict:
    import anthropic

    client = anthropic.Anthropic()
    response = client.messages.create(
        model=CLAUDE_JUDGE,
        max_tokens=200,
        system=RUBRIC,
        messages=[{"role": "user", "content": _judge_message(item, payload, answer)}],
    )
    return {**_parse(response.content[0].text), "judge": CLAUDE_JUDGE}


def judge_with_openai(item: dict, payload: dict, answer: str) -> dict:
    from openai import OpenAI

    client = OpenAI()
    response = client.chat.completions.create(
        model=OPENAI_JUDGE,
        messages=[
            {"role": "system", "content": RUBRIC},
            {"role": "user", "content": _judge_message(item, payload, answer)},
        ],
    )
    return {**_parse(response.choices[0].message.content), "judge": OPENAI_JUDGE}


# generator model name -> the OTHER family's judge
CROSS_JUDGE = {
    "claude-haiku-4-5": judge_with_openai,
    "gpt-5-nano": judge_with_claude,
}
