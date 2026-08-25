"""Dual generators behind one interface (the parser-abstraction pattern).
Both receive the IDENTICAL payload; the claim under test is that the payload
does the work. Deliberately cheap models — a weak generator answering
correctly credits the context layer, not the model."""

import json

CLAUDE_MODEL = "claude-haiku-4-5"
OPENAI_MODEL = "gpt-5-nano"

SYSTEM = """You are an internal GEHA benefits assistant. Answer the user's
question using ONLY the context chunks in the payload JSON.

Rules:
- If payload status is "out_of_scope", reply with the boundary_response.
- If payload status is "insufficient_evidence", reply exactly that you
  cannot answer from the available corpus, and do not guess.
- Never use outside knowledge, even when you are confident you know the
  answer. If the chunks do not contain it, say you cannot answer.
- Cite the source title and page(s) for every fact you state.
- Be concise: 1-4 sentences."""


def _user_message(payload: dict) -> str:
    return f"PAYLOAD:\n{json.dumps(payload, indent=1)}\n\nQUESTION: {payload['query']}"


class ClaudeGenerator:
    name = CLAUDE_MODEL

    def generate(self, payload: dict) -> str:
        import anthropic

        client = anthropic.Anthropic()
        response = client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=400,
            system=SYSTEM,
            messages=[{"role": "user", "content": _user_message(payload)}],
        )
        return response.content[0].text


class OpenAIGenerator:
    name = OPENAI_MODEL

    def generate(self, payload: dict) -> str:
        from openai import OpenAI

        client = OpenAI()
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": _user_message(payload)},
            ],
        )
        return response.choices[0].message.content


GENERATORS = (ClaudeGenerator(), OpenAIGenerator())
