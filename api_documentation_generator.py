"""API Documentation Generator for HOPEFX Trading Bot.

Generates a plain-text or Markdown summary of all registered FastAPI
routes.  Useful for quick documentation and README generation.
"""

from typing import List, Dict, Any, Optional


class APIDocumentationGenerator:
    """Extracts route information from a FastAPI app and renders it.

    Usage::

        from fastapi import FastAPI
        app = FastAPI()

        generator = APIDocumentationGenerator(app)
        print(generator.to_markdown())
    """

    def __init__(self, app: Optional[Any] = None) -> None:
        self.app = app

    # ------------------------------------------------------------------
    # Route extraction
    # ------------------------------------------------------------------

    def get_routes(self) -> List[Dict[str, Any]]:
        """Return a list of route descriptors from the FastAPI application.

        Returns:
            A list of dicts with keys ``path``, ``methods``, ``name`` and
            ``summary``.  Returns an empty list when no app is set.
        """
        if self.app is None:
            return []

        routes = []
        for route in getattr(self.app, "routes", []):
            methods = getattr(route, "methods", None)
            if methods is None:
                continue
            routes.append(
                {
                    "path": route.path,
                    "methods": sorted(methods),
                    "name": getattr(route, "name", ""),
                    "summary": getattr(route, "summary", "") or "",
                    "tags": getattr(route, "tags", []),
                }
            )
        return routes

    # ------------------------------------------------------------------
    # Renderers
    # ------------------------------------------------------------------

    def to_markdown(self) -> str:
        """Render API routes as a Markdown table.

        Returns:
            A Markdown string with one row per route.
        """
        routes = self.get_routes()
        if not routes:
            return "_No routes found._\n"

        lines = [
            "| Method | Path | Name | Summary |",
            "|--------|------|------|---------|",
        ]
        for r in routes:
            methods = ", ".join(r["methods"])
            lines.append(
                f"| {methods} | `{r['path']}` | {r['name']} | {r['summary']} |"
            )
        return "\n".join(lines) + "\n"

    def to_text(self) -> str:
        """Render API routes as plain text.

        Returns:
            A plain-text string listing each route.
        """
        routes = self.get_routes()
        if not routes:
            return "No routes found.\n"

        lines = []
        for r in routes:
            methods = " | ".join(r["methods"])
            lines.append(f"[{methods}] {r['path']}  –  {r['summary']}")
        return "\n".join(lines) + "\n"

    def save_markdown(self, path: str) -> None:
        """Write the Markdown documentation to *path*.

        Args:
            path: File path where the Markdown file will be written.
        """
        with open(path, "w", encoding="utf-8") as fh:
            fh.write("# HOPEFX API Reference\n\n")
            fh.write(self.to_markdown())

