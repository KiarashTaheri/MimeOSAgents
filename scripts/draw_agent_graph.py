"""Render the agent's structure (agent -> tools) with the Agents SDK visualizer to docs/agent_graph.png.

Requires the graphviz binary (`brew install graphviz` / `apt install graphviz`). No mimOE connection is
needed: the model name is only a label here.
"""

from pathlib import Path

from agents.extensions.visualization import draw_graph

from mimoe_agent.agent import build_agent, create_client
from mimoe_agent.config import get_settings

OUT = Path(__file__).resolve().parent.parent / "docs" / "agent_graph"


def main() -> None:
    settings = get_settings()
    agent = build_agent(create_client(settings), settings.model or "local-model", settings)
    draw_graph(agent, filename=str(OUT))  # writes OUT.png
    print(f"Wrote {OUT}.png")


if __name__ == "__main__":
    main()
