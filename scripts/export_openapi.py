"""Export the OpenAPI spec to docs/openapi.json

Generates the spec from the FastAPI app, normalizes the title/version to
match the Aegra project defaults, and removes any custom-route endpoints
that are project-specific rather than part of the standard API.

NOTE: Environment variables must be set before importing the app, so the
imports below are intentionally placed after os.environ calls.
"""

import json
import logging
import os
from pathlib import Path

# Suppress all logging before imports — env vars must be set first
logging.disable(logging.CRITICAL)
os.environ["AEGRA__APP__LOG_LEVEL"] = "CRITICAL"
os.environ.setdefault("AEGRA__APP__PROJECT_NAME", "Aegra")
os.environ.setdefault("AEGRA__APP__VERSION", "0.5.0")
os.environ.setdefault("AEGRA__APP__DEBUG", "false")
os.environ.setdefault("DATABASE_URL", "postgresql+asyncpg://user:pass@localhost/db")

import structlog  # noqa: E402

structlog.configure(
    wrapper_class=structlog.make_filtering_bound_logger(logging.CRITICAL),
)

from aegra_api.main import app  # noqa: E402
from aegra_api.settings import settings  # noqa: E402

schema = app.openapi()

# Normalize title and description to standard Aegra values
schema["info"]["title"] = "Aegra"
schema["info"]["version"] = settings.app.VERSION
schema["info"]["description"] = "Production-ready Agent Protocol server"

# Remove custom-route endpoints and untagged paths (root, custom routes)
CORE_TAGS = {"Assistants", "Threads", "Runs", "Store", "Health"}
paths_to_remove: list[str] = []
for path, methods in schema.get("paths", {}).items():
    for _method, info in methods.items():
        tags = set(info.get("tags", []))
        if not tags or not tags & CORE_TAGS:
            paths_to_remove.append(path)
            break

for path in paths_to_remove:
    del schema["paths"][path]

# Keep only core tags
schema["tags"] = [t for t in schema.get("tags", []) if t["name"] in CORE_TAGS]

output_path = Path(__file__).parent.parent / "docs" / "openapi.json"
output_path.write_text(json.dumps(schema, indent=2) + "\n")
print(f"Exported {len(schema['paths'])} endpoints to {output_path}")
