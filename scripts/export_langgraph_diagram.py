"""Export the native LangGraph diagram for the PicRetrieve graph."""

from __future__ import annotations

from pathlib import Path

from mrag.graph import build_picretrieve_graph

OUTPUT_DIR = Path("docs/langgraph")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    graph = build_picretrieve_graph().get_graph()

    mermaid_path = OUTPUT_DIR / "picretrieve_graph.mmd"
    png_path = OUTPUT_DIR / "picretrieve_graph.png"

    mermaid_path.write_text(graph.draw_mermaid(), encoding="utf-8")
    png_path.write_bytes(graph.draw_mermaid_png())

    print(f"Exported {mermaid_path}")
    print(f"Exported {png_path}")


if __name__ == "__main__":
    main()
