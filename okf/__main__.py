"""Command line:  python -m okf "How does HELM measure calibration?"
                python -m okf --status
                python -m okf --search "bar exam"        (retrieval only, no LLM)
                python -m okf --no-vectors "..."         (keyword search + links only)
"""

import argparse
import sys

from . import OKF


def main(argv=None):
    p = argparse.ArgumentParser(prog="python -m okf", description="Ask the OKF bundle a question.")
    p.add_argument("question", nargs="?")
    p.add_argument("--status", action="store_true", help="show what's loaded and connected")
    p.add_argument("--search", action="store_true", help="retrieval only: print the context blocks' sources")
    p.add_argument("--no-vectors", action="store_true", help="don't use Qdrant")
    args = p.parse_args(argv)

    okf = OKF.load(use_vectors=not args.no_vectors)
    if args.status or not args.question:
        for k, v in okf.status().items():
            print(f"{k:15} {v}")
        if not args.question:
            return 0
    if args.search:
        for passage in okf.search(args.question):
            print(f"- [{passage.source.id}] {passage.source.label} ({len(passage.text)} chars)")
        return 0
    a = okf.ask(args.question)
    print(a.answer)
    print("\nsources:")
    for s in a.sources or []:
        print(f"  [{s.id}] {s.label}")
    if not a.sources and not a.refused:
        print("  (none cited - check the answer)")
    print(f"\nretrieval {a.retrieval_ms:.0f} ms | generation {a.generation_ms:.0f} ms | "
          f"tokens {a.prompt_tokens}/{a.answer_tokens} | {a.model} | vectors {'on' if a.retrieval.vectors_used else 'off'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
