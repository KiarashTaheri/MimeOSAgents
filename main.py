"""CLI for the CGM agent.

    uv run python main.py                      # interactive chat
    uv run python main.py "When am I in range?"  # one-shot
    uv run python main.py -v ...               # also show tool calls and latency
"""

import argparse
import asyncio
import sys

from agents import SQLiteSession

from mimoe_agent.agent import AgentError, AgentResult, CGMAssistant
from mimoe_agent.observability import setup_observability


def print_result(result: AgentResult, verbose: bool) -> None:
    if verbose:
        for call in result.tool_calls:
            print(f"  ↳ {call.name}({call.args}){'' if call.ok else '  [error]'}")
        print(f"  ({result.model}, {result.latency_ms} ms)")
        if result.blocked:
            print(f"  [guardrail: {result.blocked}]")
    print(f"\n{result.answer}\n")


async def main() -> int:
    parser = argparse.ArgumentParser(description="Ask questions about your CGM data, answered by a local model on mimOE.")
    parser.add_argument("question", nargs="*", help="Ask one question and exit.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Show tool calls and latency.")
    args = parser.parse_args()

    setup_observability()
    assistant = CGMAssistant()
    session = SQLiteSession("cli")

    try:
        agent = await assistant.agent()
    except AgentError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    if args.question:
        try:
            print_result(await assistant.ask(" ".join(args.question), session), args.verbose)
        except AgentError as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1
        return 0

    print(f"CGM assistant on mimOE ({agent.model.model}, {len(assistant.context.readings)} readings). Ctrl-D to quit.")
    while True:
        try:
            question = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not question:
            continue
        try:
            print_result(await assistant.ask(question, session), args.verbose)
        except AgentError as e:
            print(f"Error: {e}\n", file=sys.stderr)


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
