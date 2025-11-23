#!/usr/bin/env python3
"""
HUTUBS Head Size Classification

Reads actual HUTUBS anthropometric data and classifies 96 subjects by head circumference.
Selects representative subjects for each size category (XS/S/M/L/XL).

Data Source: HUTUBS - Head-related Transfer Function Database
    https://depositonce.tu-berlin.de/items/dc2a3076-a291-417e-97f0-7697e332c960

License: CC BY-SA 4.0
Attribution:
    HUTUBS - Head-related Transfer Function Database of the
    Technical University of Berlin
    F. Brinkmann et al., TU Berlin, 2019

Usage:
    python scripts/fetch_hutubs.py
    python scripts/fetch_hutubs.py --output-dir data/crossfeed
"""

import argparse
import csv
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional


# Default path to HUTUBS anthropometric data (from extracted ZIP)
DEFAULT_ANTHROPOMETRIC_CSV = (
    Path(__file__).parent.parent
    / "data"
    / "crossfeed"
    / "raw"
    / "Antrhopometric measures"
    / "AntrhopometricMeasures.csv"
)

# Head circumference classification bins (in cm)
# Based on typical adult head size distribution
HEAD_SIZE_BINS = {
    "XS": (0, 53),  # Extra Small: < 53cm
    "S": (53, 55),  # Small: 53-55cm
    "M": (55, 57),  # Medium: 55-57cm (default)
    "L": (57, 59),  # Large: 57-59cm
    "XL": (59, 100),  # Extra Large: >= 59cm
}

# CIPIC anthropometric column mappings
# x16 = head circumference (in cm)
# x1 = head width, x2 = head height, x3 = head depth
COLUMN_HEAD_CIRCUMFERENCE = "x16"
COLUMN_HEAD_WIDTH = "x1"
COLUMN_HEAD_HEIGHT = "x2"
COLUMN_HEAD_DEPTH = "x3"


@dataclass
class SubjectInfo:
    """Information about a HUTUBS subject."""

    id: str  # e.g., "pp1"
    head_circumference: Optional[float]  # in cm (from x16)
    head_width: Optional[float]  # in cm (from x1)
    head_height: Optional[float]  # in cm (from x2)
    head_depth: Optional[float]  # in cm (from x3)
    size_category: str  # XS/S/M/L/XL
    sofa_file: Optional[str]  # path to SOFA file


@dataclass
class HUTUBSMetadata:
    """Metadata structure for HUTUBS classification results."""

    subjects: list[SubjectInfo]
    representative_subjects: dict[str, str]  # size -> subject_id
    is_simulated: bool = False  # This uses real HUTUBS data
    license: str = "CC BY-SA 4.0"
    source: str = (
        "https://depositonce.tu-berlin.de/items/dc2a3076-a291-417e-97f0-7697e332c960"
    )
    attribution: str = "HUTUBS - Head-related Transfer Function Database, TU Berlin"


def classify_head_size(circumference: Optional[float]) -> str:
    """Classify head circumference into size category."""
    if circumference is None:
        return "M"  # Default to medium if unknown

    for size, (low, high) in HEAD_SIZE_BINS.items():
        if low <= circumference < high:
            return size
    return "M"  # Fallback


def parse_float(value: str) -> Optional[float]:
    """Parse float value, returning None for NaN or empty values."""
    if not value or value.lower() == "nan":
        return None
    try:
        result = float(value)
        if math.isnan(result):
            return None
        return result
    except ValueError:
        return None


def load_hutubs_anthropometric_data(csv_path: Path) -> dict[str, dict]:
    """
    Load ACTUAL HUTUBS anthropometric data from CSV file.

    This reads the official HUTUBS AntrhopometricMeasures.csv file.
    Data is used as-is without modification (CC BY-SA 4.0 compliance).

    Args:
        csv_path: Path to AntrhopometricMeasures.csv

    Returns:
        Dict mapping subject_id (pp1, pp2, ...) -> anthropometric measurements
    """
    if not csv_path.exists():
        raise FileNotFoundError(
            f"HUTUBS anthropometric data not found at {csv_path}.\n"
            "Please download from:\n"
            "  https://depositonce.tu-berlin.de/items/dc2a3076-a291-417e-97f0-7697e332c960\n"
            "Extract 'Antrhopometric measures.zip' to data/crossfeed/raw/"
        )

    anthropometric = {}

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            subject_num = row.get("SubjectID", "")
            if not subject_num:
                continue

            subject_id = f"pp{subject_num}"

            # Extract measurements (use actual HUTUBS column names)
            head_circumference = parse_float(row.get(COLUMN_HEAD_CIRCUMFERENCE, ""))
            head_width = parse_float(row.get(COLUMN_HEAD_WIDTH, ""))
            head_height = parse_float(row.get(COLUMN_HEAD_HEIGHT, ""))
            head_depth = parse_float(row.get(COLUMN_HEAD_DEPTH, ""))

            # Include all subjects, even with missing head circumference
            # (missing data will default to size_category="M")
            anthropometric[subject_id] = {
                "head_circumference": head_circumference,
                "head_width": head_width,
                "head_height": head_height,
                "head_depth": head_depth,
            }

    return anthropometric


def find_representative_subjects(subjects: list[SubjectInfo]) -> dict[str, str]:
    """
    Find the most representative subject for each size category.

    Selects the subject closest to the median head circumference within each group.
    """
    from statistics import median

    representatives = {}

    for size in HEAD_SIZE_BINS.keys():
        # Get all subjects in this size category
        size_subjects = [s for s in subjects if s.size_category == size]

        if not size_subjects:
            continue

        # Find median circumference for this group
        circumferences = [
            s.head_circumference
            for s in size_subjects
            if s.head_circumference is not None
        ]

        if not circumferences:
            representatives[size] = size_subjects[0].id
            continue

        median_circ = median(circumferences)

        # Find subject closest to median
        closest = min(
            [s for s in size_subjects if s.head_circumference is not None],
            key=lambda s: abs(s.head_circumference - median_circ),
        )
        representatives[size] = closest.id

    return representatives


def create_output_directories(output_dir: Path) -> None:
    """Create necessary output directories."""
    output_dir.mkdir(parents=True, exist_ok=True)


def generate_metadata(
    output_dir: Path, csv_path: Path = DEFAULT_ANTHROPOMETRIC_CSV
) -> HUTUBSMetadata:
    """
    Generate metadata from ACTUAL HUTUBS anthropometric data.

    Args:
        output_dir: Directory to store output files
        csv_path: Path to HUTUBS AntrhopometricMeasures.csv
    """
    create_output_directories(output_dir)

    # Load actual HUTUBS data
    anthropometric = load_hutubs_anthropometric_data(csv_path)

    # Create subject info list
    subjects = []
    for subject_id, anthro in sorted(
        anthropometric.items(), key=lambda x: int(x[0][2:])
    ):
        circumference = anthro.get("head_circumference")
        size_category = classify_head_size(circumference)

        subject = SubjectInfo(
            id=subject_id,
            head_circumference=circumference,
            head_width=anthro.get("head_width"),
            head_height=anthro.get("head_height"),
            head_depth=anthro.get("head_depth"),
            size_category=size_category,
            sofa_file=None,  # SOFA files not included (too large)
        )
        subjects.append(subject)

    # Find representative subjects
    representatives = find_representative_subjects(subjects)

    return HUTUBSMetadata(
        subjects=subjects,
        representative_subjects=representatives,
        is_simulated=False,  # Real HUTUBS data!
    )


def save_metadata(metadata: HUTUBSMetadata, output_path: Path) -> None:
    """Save metadata to JSON file."""
    data = {
        "description": "HUTUBS head size classification using actual anthropometric data",
        "is_simulated": metadata.is_simulated,
        "license": metadata.license,
        "source": metadata.source,
        "attribution": metadata.attribution,
        "representative_subjects": metadata.representative_subjects,
        "head_size_bins": HEAD_SIZE_BINS,
        "subjects": [asdict(s) for s in metadata.subjects],
    }

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"Saved metadata to {output_path}")


def print_summary(metadata: HUTUBSMetadata) -> None:
    """Print summary of the classification."""
    print("\n" + "=" * 60)
    print("HUTUBS Head Size Classification Summary")
    print("=" * 60)
    print(f"\nData source: {metadata.source}")
    print(f"License: {metadata.license}")
    print(f"Total subjects with valid data: {len(metadata.subjects)}")

    # Count subjects per category
    counts = {}
    for size in HEAD_SIZE_BINS.keys():
        counts[size] = sum(1 for s in metadata.subjects if s.size_category == size)

    print("\nSubjects per size category:")
    for size, (low, high) in HEAD_SIZE_BINS.items():
        count = counts.get(size, 0)
        rep = metadata.representative_subjects.get(size, "N/A")
        print(
            f"  {size:2s} ({low:2.0f}-{high:2.0f}cm): {count:2d} subjects, "
            f"representative: {rep}"
        )

    print("\nRepresentative subjects (actual HUTUBS data):")
    for size, subject_id in metadata.representative_subjects.items():
        subject = next((s for s in metadata.subjects if s.id == subject_id), None)
        if subject:
            print(
                f"  {size}: {subject_id} "
                f"(head circumference: {subject.head_circumference}cm)"
            )

    print("\n" + "=" * 60)
    print("Attribution (CC BY-SA 4.0):")
    print(f"  {metadata.attribution}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Classify HUTUBS subjects by head size using actual anthropometric data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This script uses ACTUAL HUTUBS anthropometric data (CC BY-SA 4.0).

Prerequisites:
  1. Download 'Antrhopometric measures.zip' from:
     https://depositonce.tu-berlin.de/items/dc2a3076-a291-417e-97f0-7697e332c960
  2. Extract to data/crossfeed/raw/

Examples:
  # Generate classification from HUTUBS data
  python scripts/fetch_hutubs.py

  # Specify output directory
  python scripts/fetch_hutubs.py --output-dir data/crossfeed

License: CC BY-SA 4.0
Attribution: HUTUBS - Head-related Transfer Function Database, TU Berlin
        """,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/crossfeed"),
        help="Output directory for metadata",
    )
    parser.add_argument(
        "--csv-path",
        type=Path,
        default=DEFAULT_ANTHROPOMETRIC_CSV,
        help="Path to HUTUBS AntrhopometricMeasures.csv",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output",
    )

    args = parser.parse_args()

    if not args.quiet:
        print("HUTUBS Head Size Classification")
        print("Using actual HUTUBS anthropometric data (CC BY-SA 4.0)")
        print(f"CSV path: {args.csv_path}")
        print(f"Output directory: {args.output_dir}")
        print()

    try:
        # Generate metadata from actual HUTUBS data
        metadata = generate_metadata(output_dir=args.output_dir, csv_path=args.csv_path)

        # Save metadata
        metadata_path = args.output_dir / "hutubs_subjects.json"
        save_metadata(metadata, metadata_path)

        # Print summary
        if not args.quiet:
            print_summary(metadata)

        return 0

    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
