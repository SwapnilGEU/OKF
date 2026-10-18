"""Where the OKF agent finds its pieces: the MCP server, Postgres, and the answering model.

Read from .env (same file as the okf package):
  OKF_MCP_URL        default http://127.0.0.1:8000/mcp   (the server from 10_mcp_server)
  OKF_POSTGRES_URL   full address, e.g. postgresql://postgres:pw@localhost:5432/OKF
     or PG_HOST, PG_PORT, PG_DBNAME, PG_USER, PG_PASSWORD (used when OKF_POSTGRES_URL isn't set)
The model, num_ctx and the rest come from okf.Settings, so the agent answers with exactly
the evaluated setup (and shares Ollama's loaded model with the MCP server).
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from urllib.parse import quote

from okf import Settings


def postgres_url() -> str:
    if os.getenv("OKF_POSTGRES_URL"):
        return os.environ["OKF_POSTGRES_URL"]
    host, port = os.getenv("PG_HOST", "localhost"), os.getenv("PG_PORT", "5432")
    db, user = os.getenv("PG_DBNAME", "OKF"), os.getenv("PG_USER", "postgres")
    password = quote(os.getenv("PG_PASSWORD", ""), safe="")      # escape @ : / etc. in the password
    return f"postgresql://{user}:{password}@{host}:{port}/{db}"


@dataclass(frozen=True)
class AgentConfig:
    okf: Settings                 # model, num_ctx, context size ... (from the okf package)
    mcp_url: str
    postgres_url: str

    @classmethod
    def load(cls, **okf_overrides) -> "AgentConfig":
        s = Settings.load(**okf_overrides)          # also reads .env
        return cls(okf=s, mcp_url=os.getenv("OKF_MCP_URL", "http://127.0.0.1:8000/mcp"),
                   postgres_url=postgres_url())

    def postgres_url_safe(self) -> str:
        """The address with the password hidden, for printing."""
        return re.sub(r"://([^:/@]+):[^@]*@", r"://\1:***@", self.postgres_url)
