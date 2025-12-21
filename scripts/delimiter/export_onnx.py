from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from scripts.delimiter.model import DelimiterWeights, build_delimiter_model


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    root = project_root()
    default_weights = root / "data/delimiter/weights/jeonchangbin49-de-limiter/44100"
    p = argparse.ArgumentParser(
        description="Export De-limiter PyTorch weights to ONNX (Issue #1098)"
    )
    p.add_argument(
        "--weights-dir",
        type=Path,
        default=default_weights,
        help="Directory containing all.pth/all.json",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=default_weights / "delimiter.onnx",
        help="Output ONNX path",
    )
    p.add_argument(
        "--dummy-seconds",
        type=float,
        default=1.0,
        help="Length of dummy input used for export (dynamic axes are set for time dim)",
    )
    p.add_argument(
        "--opset",
        type=int,
        default=18,
        help="ONNX opset version (default: 18 for better operator coverage)",
    )
    p.add_argument(
        "--use-dynamo",
        action="store_true",
        help="Use torch.export-based exporter (default: legacy exporter via dynamo=False)",
    )
    return p.parse_args()


def _sample_rate_from_config(cfg: dict[str, Any]) -> int:
    args = cfg.get("args", {})
    data_params = args.get("data_params", {})
    return int(data_params.get("sample_rate", 44100))


def main() -> int:
    args = parse_args()

    weights = DelimiterWeights(
        config_json=args.weights_dir / "all.json",
        state_dict_pth=args.weights_dir / "all.pth",
    )

    try:
        import torch
    except Exception as e:  # pragma: no cover - environment dependent
        print(
            "PyTorch is required. Install delimiter extras: uv sync --extra delimiter",
            file=sys.stderr,
        )
        print(e, file=sys.stderr)
        return 1

    device = torch.device("cpu")
    model, cfg = build_delimiter_model(weights, device)
    model = model.to(device)
    model.eval()

    sample_rate = _sample_rate_from_config(cfg)
    frames = max(1, int(args.dummy_seconds * sample_rate))
    dummy = torch.zeros((1, 2, frames), dtype=torch.float32, device=device)

    args.output.parent.mkdir(parents=True, exist_ok=True)

    dynamic_axes = {
        "input": {0: "batch", 2: "frames"},
        "enhanced": {0: "batch", 2: "frames"},
    }

    export_kwargs = {
        "input_names": ["input"],
        "output_names": ["enhanced"],
        "dynamic_axes": dynamic_axes,
        "opset_version": args.opset,
        "training": torch.onnx.TrainingMode.EVAL,
    }
    if not args.use_dynamo:
        export_kwargs["dynamo"] = False

    with torch.no_grad():
        torch.onnx.export(model, dummy, args.output, **export_kwargs)

    print(
        f"Exported ONNX to {args.output} (sample_rate={sample_rate}, frames={frames})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
