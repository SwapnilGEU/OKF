"""Where the OKF agent finds its pieces: the MCP server, Postgres, and the answering model.

Read from .env (same file as the okf package):
  OKF_MCP_URL        default http://127.0.0.1:8000/mcp   (the server from 10_mcp_server)
  OKF_ARXIV_MCP_URL  default http://127.0.0.1:8001/mcp   (the third-party arXiv server, step 13)
  OKF_REPORTS_DIR    default <project>/reports            (the ONLY folder the filesystem server may touch, step 15)
  OKF_FS_MCP_COMMAND default: npx -y @modelcontextprotocol/server-filesystem@2026.8.31
                     (set it to e.g. mcp-server-filesystem after `npm install -g` to skip npx)
  OKF_POSTGRES_URL   full address, e.g. postgresql://postgres:pw@localhost:5432/OKF
     or PG_HOST, PG_PORT, PG_DBNAME, PG_USER, PG_PASSWORD (used when OKF_POSTGRES_URL isn't set)
The model, num_ctx and the rest come from okf.Settings, so the agent answers with exactly
the evaluated setup (and shares Ollama's loaded model with the MCP server).
"""

from __future__ import annotations

import os
import re
import shutil
from dataclasses import dataclass
from urllib.parse import quote

from okf import Settings
from okf.config import find_project_root

FS_MCP_PACKAGE = "@modelcontextprotocol/server-filesystem@2026.8.31"   # pinned: the official reference server


def postgres_url() -> str:
    if os.getenv("OKF_POSTGRES_URL"):
        return os.environ["OKF_POSTGRES_URL"]
    host, port = os.getenv("PG_HOST", "localhost"), os.getenv("PG_PORT", "5432")
    db, user = os.getenv("PG_DBNAME", "OKF"), os.getenv("PG_USER", "postgres")
    password = quote(os.getenv("PG_PASSWORD", ""), safe="")      # escape @ : / etc. in the password
    return f"postgresql://{user}:{password}@{host}:{port}/{db}"


def find_npx() -> str | None:
    """Full path of npx. On the PATH if possible; otherwise the usual Windows install folders, because a
    terminal, VS Code or server window opened BEFORE Node.js was installed still has the old PATH."""
    found = shutil.which("npx")
    if found:
        return found
    if os.name == "nt":
        for folder in (os.environ.get("ProgramFiles", r"C:\Program Files"), os.environ.get("ProgramFiles(x86)", ""),
                       os.environ.get("LOCALAPPDATA", "") + r"\Programs", os.environ.get("APPDATA", "")):
            for sub in ("nodejs", "npm"):
                candidate = os.path.join(folder, sub, "npx.cmd")
                if folder and os.path.isfile(candidate):
                    return candidate
    return None


@dataclass(frozen=True)
class AgentConfig:
    okf: Settings                 # model, num_ctx, context size ... (from the okf package)
    mcp_url: str
    postgres_url: str
    arxiv_mcp_url: str = "http://127.0.0.1:8001/mcp"
    reports_dir: str = "reports"

    @classmethod
    def load(cls, **okf_overrides) -> "AgentConfig":
        s = Settings.load(**okf_overrides)          # also reads .env
        return cls(okf=s, mcp_url=os.getenv("OKF_MCP_URL", "http://127.0.0.1:8000/mcp"),
                   postgres_url=postgres_url(),
                   arxiv_mcp_url=os.getenv("OKF_ARXIV_MCP_URL", "http://127.0.0.1:8001/mcp"),
                   reports_dir=os.getenv("OKF_REPORTS_DIR") or str(find_project_root() / "reports"))

    def fs_mcp_connection(self) -> dict:
        """How to start the filesystem MCP server: a child process over stdio, limited to reports_dir."""
        command = os.getenv("OKF_FS_MCP_COMMAND", "").split()
        if not command:
            command = [find_npx() or "npx", "-y", FS_MCP_PACKAGE]
        return {"command": command[0], "args": [*command[1:], self.reports_dir], "transport": "stdio"}

    def postgres_url_safe(self) -> str:
        """The address with the password hidden, for printing."""
        return re.sub(r"://([^:/@]+):[^@]*@", r"://\1:***@", self.postgres_url)
