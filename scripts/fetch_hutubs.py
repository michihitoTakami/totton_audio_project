#!/usr/bin/env python3
"""
HUTUBS Head Size Classification Scaffold

Generates SIMULATED placeholder data for head size classification.
This is a SCAFFOLD for future HUTUBS HRTF integration.

⚠️ WARNING: This script generates SIMULATED data, NOT actual HUTUBS measurements.
The anthropometric values are randomly generated based on typical adult head size
distributions and do NOT reflect real HUTUBS subject data.

For actual HUTUBS data, download from:
    https://depositonce.tu-berlin.de/items/b92e5c75-3b68-4461-8a41-aecc6bcaed5a

HUTUBS Database License: CC BY-SA 4.0
Attribution:
    HUTUBS - Head-related Transfer Function Database of the
    Technical University of Berlin
    F. Brinkmann et al., TU Berlin, 2019
"""

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

# Subject IDs range from pp1 to pp96 in actual HUTUBS database
HUTUBS_SUBJECT_COUNT = 96

# Head circumference classification bins (in cm)
# Based on typical adult head size distribution
HEAD_SIZE_BINS = {
    "XS": (0, 53),  # Extra Small: < 53cm
    "S": (53, 55),  # Small: 53-55cm
    "M": (55, 57),  # Medium: 55-57cm (default)
    "L": (57, 59),  # Large: 57-59cm
    "XL": (59, 100),  # Extra Large: >= 59cm
}


@dataclass
class SubjectInfo:
    """Information about a subject (simulated)."""

    id: str  # e.g., "pp1"
    head_circumference: Optional[float]  # in cm (SIMULATED)
    head_width: Optional[float]  # in cm (SIMULATED)
    head_height: Optional[float]  # in cm (SIMULATED)
    head_depth: Optional[float]  # in cm (SIMULATED)
    size_category: str  # XS/S/M/L/XL
    sofa_file: Optional[str]  # path to SOFA file (placeholder)


@dataclass
class HUTUBSMetadata:
    """Metadata structure (with simulated subject data)."""

    subjects: list[SubjectInfo]
    representative_subjects: dict[str, str]  # size -> subject_id
    is_simulated: bool = True  # Flag indicating data is simulated
    license: str = "CC BY-SA 4.0"
    source: str = (
        "https://depositonce.tu-berlin.de/items/b92e5c75-3b68-4461-8a41-aecc6bcaed5a"
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


def generate_simulated_anthropometric_data() -> dict[str, dict]:
    """
    Generate SIMULATED anthropometric data for placeholder subjects.

    ⚠️ WARNING: This generates SIMULATED data based on typical adult head size
    distributions. It does NOT use actual HUTUBS measurement data.

    The simulated distribution uses:
    - Head circumference: Normal distribution, mean=56cm, std=2cm
    - Other measurements derived from circumference using approximate ratios

    Returns:
        Dict mapping subject_id -> anthropometric measurements (all simulated)
    """
    import numpy as np

    np.random.seed(42)  # For reproducibility

    # Generate realistic head circumference distribution
    # Adult head circumference: mean ~56cm, std ~2cm
    head_circumferences = np.random.normal(56.0, 2.0, HUTUBS_SUBJECT_COUNT)

    anthropometric = {}
    for i, circ in enumerate(head_circumferences, 1):
        subject_id = f"pp{i}"
        # Clamp to realistic range
        circ = max(50.0, min(62.0, circ))
        anthropometric[subject_id] = {
            "head_circumference": round(circ, 1),
            "head_width": round(circ * 0.27, 1),  # Approximate ratio
            "head_height": round(circ * 0.40, 1),  # Approximate ratio
            "head_depth": round(circ * 0.35, 1),  # Approximate ratio
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


def generate_simulated_metadata(output_dir: Path) -> HUTUBSMetadata:
    """
    Generate SIMULATED metadata with placeholder subject data.

    ⚠️ This generates SIMULATED data, not actual HUTUBS measurements.

    Args:
        output_dir: Directory to store output files
    """
    create_output_directories(output_dir)

    # Generate simulated anthropometric data
    anthropometric = generate_simulated_anthropometric_data()

    # Create subject info list
    subjects = []
    for i in range(1, HUTUBS_SUBJECT_COUNT + 1):
        subject_id = f"pp{i}"
        anthro = anthropometric.get(subject_id, {})

        circumference = anthro.get("head_circumference")
        size_category = classify_head_size(circumference)

        subject = SubjectInfo(
            id=subject_id,
            head_circumference=circumference,
            head_width=anthro.get("head_width"),
            head_height=anthro.get("head_height"),
            head_depth=anthro.get("head_depth"),
            size_category=size_category,
            sofa_file=None,  # No actual SOFA files
        )
        subjects.append(subject)

    # Find representative subjects
    representatives = find_representative_subjects(subjects)

    return HUTUBSMetadata(
        subjects=subjects,
        representative_subjects=representatives,
        is_simulated=True,
    )


def save_metadata(metadata: HUTUBSMetadata, output_path: Path) -> None:
    """Save metadata to JSON file."""
    data = {
        "WARNING": "This file contains SIMULATED data, NOT actual HUTUBS measurements",
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
    print("Head Size Classification Summary (SIMULATED DATA)")
    print("=" * 60)
    print("\n⚠️  WARNING: All data below is SIMULATED, not from HUTUBS")

    # Count subjects per category
    counts = {}
    for size in HEAD_SIZE_BINS.keys():
        counts[size] = sum(1 for s in metadata.subjects if s.size_category == size)

    print("\nSubjects per size category:")
    for size, (low, high) in HEAD_SIZE_BINS.items():
        count = counts.get(size, 0)
        rep = metadata.representative_subjects.get(size, "N/A")
        print(
            f"  {size:2s} ({low:2.0f}-{high:2.0f}cm): {count:2d} subjects, representative: {rep}"
        )

    print("\nRepresentative subjects (simulated):")
    for size, subject_id in metadata.representative_subjects.items():
        subject = next((s for s in metadata.subjects if s.id == subject_id), None)
        if subject:
            print(
                f"  {size}: {subject_id} (head circumference: {subject.head_circumference}cm)"
            )

    print("\n" + "=" * 60)
    print("For actual HUTUBS data, download from:")
    print(f"  {metadata.source}")
    print(f"\nLicense: {metadata.license}")
    print(f"Attribution: {metadata.attribution}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Generate simulated head size classification data (scaffold for HUTUBS)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
⚠️  WARNING: This script generates SIMULATED data, NOT actual HUTUBS measurements.

Examples:
  # Generate simulated metadata
  python scripts/fetch_hutubs.py

  # Specify output directory
  python scripts/fetch_hutubs.py --output-dir data/crossfeed

For actual HUTUBS HRTF data, download from:
  https://depositonce.tu-berlin.de/items/b92e5c75-3b68-4461-8a41-aecc6bcaed5a

License: CC BY-SA 4.0
        """,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/crossfeed"),
        help="Output directory for metadata",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output",
    )

    args = parser.parse_args()

    if not args.quiet:
        print("HUTUBS Head Size Classification Scaffold")
        print("⚠️  Generating SIMULATED data (not actual HUTUBS measurements)")
        print(f"Output directory: {args.output_dir}")
        print()

    # Generate simulated metadata
    metadata = generate_simulated_metadata(output_dir=args.output_dir)

    # Save metadata
    metadata_path = args.output_dir / "hutubs_subjects.json"
    save_metadata(metadata, metadata_path)

    # Print summary
    if not args.quiet:
        print_summary(metadata)

    return 0


if __name__ == "__main__":
    sys.exit(main())
