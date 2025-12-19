#!/usr/bin/env python3
"""
Export OpenAPI specification from Raspberry Pi control FastAPI app.

Used by pre-commit hooks to keep docs/api/raspi_openapi.json up to date.

Usage:
    uv run python scripts/integration/export_raspi_openapi.py
    uv run python scripts/integration/export_raspi_openapi.py --output custom/path.json
    uv run python scripts/integration/export_raspi_openapi.py --check  # Exit 1 if out of date
"""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_OUTPUT = PROJECT_ROOT / "docs" / "api" / "raspi_openapi.json"


def get_openapi_schema() -> dict:
    """Get OpenAPI schema from Raspberry Pi control app."""
    from raspberry_pi.control_api import create_app

    app = create_app()
    return app.openapi()


def export_openapi(output_path: Path) -> None:
    """Export OpenAPI schema to file."""
    schema = get_openapi_schema()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(schema, f, indent=2, ensure_ascii=False)
        f.write("\n")
    print(f"OpenAPI spec exported to: {output_path}")


def check_openapi(output_path: Path) -> bool:
    """Check if OpenAPI spec is up to date."""
    if not output_path.exists():
        print(f"OpenAPI spec not found: {output_path}")
        return False

    current_schema = get_openapi_schema()
    with open(output_path, encoding="utf-8") as f:
        existing_schema = json.load(f)

    if current_schema == existing_schema:
        print("OpenAPI spec is up to date")
        return True

    print(f"OpenAPI spec is OUT OF DATE: {output_path}")
    print("Run: uv run python scripts/integration/export_raspi_openapi.py")
    return False


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export OpenAPI spec from Raspberry Pi control app",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  uv run python scripts/integration/export_raspi_openapi.py              # Export to default location
  uv run python scripts/integration/export_raspi_openapi.py --check      # Check if up to date
  uv run python scripts/integration/export_raspi_openapi.py -o api.json  # Export to custom path
        """,
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output file path (default: {DEFAULT_OUTPUT.relative_to(PROJECT_ROOT)})",
    )
    parser.add_argument(
        "--check",
        "-c",
        action="store_true",
        help="Check if spec is up to date (exit 1 if not)",
    )

    args = parser.parse_args()

    if args.check:
        if not check_openapi(args.output):
            sys.exit(1)
    else:
        export_openapi(args.output)


if __name__ == "__main__":
    main()
