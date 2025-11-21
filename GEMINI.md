# GEMINI.md

## Language
Think in English and asnwer in Japanese.

## Project Overview

This is a Python project named `gpu-audio-upsampler`. Its purpose is to generate high-precision FIR filter coefficients for GPU-accelerated audio upsampling.

The project uses the following main technologies:
*   **Python:** The core language for scripting and filter design.
*   **SciPy:** For signal processing and filter design functionalities.
*   **NumPy:** For numerical operations.
*   **Matplotlib:** For generating plots to analyze the filter's characteristics.

The main goal of the project is to create a 131,072-tap minimum-phase FIR filter, which is then exported for use in a C++ application.

## Building and Running

### 1. Setup

The project requires Python 3.11 or higher. To install the necessary dependencies, run:

```bash
pip install -r requirements.txt
```
*(Note: A `requirements.txt` file does not exist, so this is a placeholder. Dependencies are listed in `pyproject.toml`)*

### 2. Generating the Filter Coefficients

To generate the filter coefficients, run the `generate_filter.py` script:

```bash
python scripts/generate_filter.py
```

This script will:
*   Design a linear-phase FIR filter.
*   Convert it to a minimum-phase filter.
*   Run validation tests to ensure it meets the specifications.
*   Generate analysis plots in the `plots/analysis/` directory.
*   Export the filter coefficients to `data/coefficients/` in several formats:
    *   `filter_131k_min_phase.bin`: A binary file with float32 coefficients.
    *   `filter_coefficients.h`: A C++ header file.
    *   `metadata.json`: A JSON file with metadata about the filter.

### 3. Inspecting the Filter

To perform a more detailed analysis of the generated impulse response, run the `inspect_impulse.py` script:

```bash
python scripts/inspect_impulse.py
```

This will generate a detailed plot of the impulse response in `plots/analysis/impulse_detail.png`.

## Development Conventions

*   **Directory Structure:** The project is organized with a clear separation of concerns:
    *   `scripts/`: Contains the Python scripts for generating and analyzing the filter.
    *   `data/`: Stores the generated filter coefficients.
    *   `plots/`: Contains plots generated during the analysis.
    *   `docs/`: Includes detailed documentation and technical explanations.
*   **Code Style:** The Python code is well-documented with comments explaining the purpose of different parts of the code.
*   **Validation:** The project emphasizes validation and analysis, with scripts dedicated to verifying the filter's properties and generating detailed plots. The `docs/minimum_phase_analysis.md` file shows a commitment to documenting and justifying the technical decisions made.
