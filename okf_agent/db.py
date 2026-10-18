"""The qa_log table: one row per question the agent answers.

LangGraph's checkpointer stores the chat state in its own tables (checkpoints, checkpoint_blobs,
checkpoint_writes, ...). qa_log is ours: a flat, readable row per turn for pgAdmin and analysis.
It stores the cited ids, not the 18,000-character context.
"""

from __future__ import annotations

import json

import psycopg

SCHEMA = """
CREATE TABLE IF NOT EXISTS qa_log (
    id             BIGSERIAL PRIMARY KEY,
    created_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    thread_id      TEXT,
    agent          TEXT,
    question       TEXT,
    search_query   TEXT,
    followup       BOOLEAN,
    answer         TEXT,
    refused        BOOLEAN,
    cited          JSONB,      -- concept ids the answer cites
    bad_citations  JSONB,      -- cited ids that were NOT in the passages
    passages       JSONB,      -- ids of the passages the model saw, in order
    search_ms      REAL,
    generation_ms  REAL,
    total_ms       REAL,
    model          TEXT,
    prompt_tokens  INTEGER,
    answer_tokens  INTEGER
);
"""


def ensure_qa_log(conn: psycopg.Connection) -> None:
    conn.execute(SCHEMA)


def insert_qa_log(conn: psycopg.Connection, thread_id: str, agent: str, state: dict) -> int:
    t, llm = state.get("timings", {}), state.get("llm", {})
    row = conn.execute(
        """INSERT INTO qa_log (thread_id, agent, question, search_query, followup, answer, refused, cited,
                               bad_citations, passages, search_ms, generation_ms, total_ms, model,
                               prompt_tokens, answer_tokens)
           VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) RETURNING id""",
        (thread_id, agent, state.get("question"), state.get("search_query"), state.get("followup"),
         state.get("answer"), state.get("refused"),
         json.dumps([s["id"] for s in state.get("sources", [])]),
         json.dumps(state.get("bad_citations", [])),
         json.dumps([p["source"]["id"] for p in state.get("passages", [])]),
         t.get("search_ms"), t.get("generation_ms"), t.get("total_ms"),
         llm.get("model"), llm.get("prompt_tokens"), llm.get("answer_tokens")),
    ).fetchone()
    return row[0]
