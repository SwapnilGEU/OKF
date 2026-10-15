"""OKF: question answering over the OKF bundle (hybrid retrieval + a local or API model).

    from okf import OKF
    okf = OKF.load()
    print(okf.ask("What score did GPT-4 get on the bar exam?").answer)
"""

from .config import Settings
from .pipeline import OKF, cited_ids, is_refusal
from .schemas import Answer, Passage, Retrieval, SectionHit, Source

__all__ = ["OKF", "Settings", "Answer", "Passage", "Retrieval", "SectionHit", "Source", "cited_ids", "is_refusal"]
