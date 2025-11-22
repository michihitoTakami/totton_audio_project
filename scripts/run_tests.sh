#!/bin/bash
# Diff-based test runner for pre-push hook
# Only runs tests relevant to changed files

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}=== Diff-based Test Runner ===${NC}"

# Get changed files compared to origin/main
# For pre-push, we compare against what we're pushing to
CHANGED_FILES=$(git diff --name-only origin/main...HEAD 2>/dev/null || git diff --name-only HEAD~1)

if [ -z "$CHANGED_FILES" ]; then
    echo -e "${GREEN}No changes detected. Skipping tests.${NC}"
    exit 0
fi

echo "Changed files:"
echo "$CHANGED_FILES" | sed 's/^/  /'
echo ""

# Flags for which tests to run
RUN_PYTHON=false
RUN_CPP=false
RUN_GPU=false

# Analyze changed files
for file in $CHANGED_FILES; do
    case "$file" in
        scripts/*.py|tests/python/*|pyproject.toml)
            RUN_PYTHON=true
            ;;
        src/*.cpp|src/*.cu|include/*.h|tests/cpp/*)
            RUN_CPP=true
            RUN_GPU=true
            ;;
        tests/gpu/*|data/coefficients/*)
            RUN_GPU=true
            ;;
        CMakeLists.txt)
            RUN_CPP=true
            RUN_GPU=true
            ;;
    esac
done

# Track overall success
TESTS_PASSED=true

# Run Python tests
if $RUN_PYTHON; then
    echo -e "${YELLOW}=== Running Python tests ===${NC}"
    if uv run pytest tests/python/ -v --tb=short; then
        echo -e "${GREEN}Python tests passed!${NC}"
    else
        echo -e "${RED}Python tests failed!${NC}"
        TESTS_PASSED=false
    fi
    echo ""
fi

# Run C++ CPU tests
if $RUN_CPP; then
    echo -e "${YELLOW}=== Running C++ CPU tests ===${NC}"

    # Check if build exists, if not build
    if [ ! -f "build/cpu_tests" ]; then
        echo "Building cpu_tests..."
        cmake -B build -DCMAKE_BUILD_TYPE=Release > /dev/null 2>&1
        cmake --build build --target cpu_tests -j8 > /dev/null 2>&1
    fi

    if ./build/cpu_tests; then
        echo -e "${GREEN}CPU tests passed!${NC}"
    else
        echo -e "${RED}CPU tests failed!${NC}"
        TESTS_PASSED=false
    fi
    echo ""
fi

# Run GPU tests
if $RUN_GPU; then
    echo -e "${YELLOW}=== Running GPU tests ===${NC}"

    # Check if build exists, if not build
    if [ ! -f "build/gpu_tests" ]; then
        echo "Building gpu_tests..."
        cmake -B build -DCMAKE_BUILD_TYPE=Release > /dev/null 2>&1
        cmake --build build --target gpu_tests -j8 > /dev/null 2>&1
    fi

    if ./build/gpu_tests; then
        echo -e "${GREEN}GPU tests passed!${NC}"
    else
        echo -e "${RED}GPU tests failed!${NC}"
        TESTS_PASSED=false
    fi
    echo ""
fi

# Summary
if ! $RUN_PYTHON && ! $RUN_CPP && ! $RUN_GPU; then
    echo -e "${GREEN}=== No tests required (docs/config only changes) ===${NC}"
    exit 0
fi

if $TESTS_PASSED; then
    echo -e "${GREEN}=== All tests passed! ===${NC}"
    exit 0
else
    echo -e "${RED}=== Some tests failed! Push blocked. ===${NC}"
    exit 1
fi
