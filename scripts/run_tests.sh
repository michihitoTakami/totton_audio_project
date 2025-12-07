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

# Prefer system cmake (avoid python shim in ~/.local/bin without module)
CMAKE_BIN=${CMAKE_BIN:-cmake}
if ! "$CMAKE_BIN" --version >/dev/null 2>&1 && [ -x /usr/bin/cmake ]; then
    CMAKE_BIN=/usr/bin/cmake
fi

CTEST_BIN=${CTEST_BIN:-ctest}
if ! "$CTEST_BIN" --version >/dev/null 2>&1 && [ -x /usr/bin/ctest ]; then
    CTEST_BIN=/usr/bin/ctest
fi

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
RUN_RPI=false
RUN_JETSON_PCM=false

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
        raspberry_pi/*)
            RUN_RPI=true
            ;;
        jetson_pcm_receiver/*)
            RUN_JETSON_PCM=true
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

    # Check if build exists or needs rebuild (check for stale binary)
    NEEDS_BUILD=false
    if [ ! -f "build/cpu_tests" ]; then
        NEEDS_BUILD=true
    else
        # Check if any source files are newer than the binary
        for src in src/*.cpp include/*.h; do
            if [ -f "$src" ] && [ "$src" -nt "build/cpu_tests" ]; then
                NEEDS_BUILD=true
                break
            fi
        done
    fi

    if $NEEDS_BUILD; then
        echo "Building cpu_tests..."
        "$CMAKE_BIN" -B build -DCMAKE_BUILD_TYPE=Release 2>&1 | tail -5
        "$CMAKE_BIN" --build build --target cpu_tests -j8 2>&1 | tail -20
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

    # Check if build exists or needs rebuild (check for stale binary)
    NEEDS_BUILD=false
    if [ ! -f "build/gpu_tests" ]; then
        NEEDS_BUILD=true
    else
        # Check if any source files are newer than the binary
        for src in src/*.cpp src/*.cu src/gpu/*.cu include/*.h data/coefficients/*.h; do
            if [ -f "$src" ] && [ "$src" -nt "build/gpu_tests" ]; then
                NEEDS_BUILD=true
                break
            fi
        done
    fi

    if $NEEDS_BUILD; then
        echo "Building gpu_tests..."
        "$CMAKE_BIN" -B build -DCMAKE_BUILD_TYPE=Release 2>&1 | tail -5
        "$CMAKE_BIN" --build build --target gpu_tests -j8 2>&1 | tail -20
    fi

    if ./build/gpu_tests; then
        echo -e "${GREEN}GPU tests passed!${NC}"
    else
        echo -e "${RED}GPU tests failed!${NC}"
        TESTS_PASSED=false
    fi
    echo ""
fi

# Raspberry Pi capture app tests
if $RUN_RPI; then
    echo -e "${YELLOW}=== Running Raspberry Pi capture tests ===${NC}"
    "$CMAKE_BIN" -S raspberry_pi -B raspberry_pi/build -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON 2>&1 | tail -20
    "$CMAKE_BIN" --build raspberry_pi/build --target rpi_capture_tests -j8 2>&1 | tail -20
    if "$CTEST_BIN" --test-dir raspberry_pi/build --output-on-failure; then
        echo -e "${GREEN}Raspberry Pi capture tests passed!${NC}"
    else
        echo -e "${RED}Raspberry Pi capture tests failed!${NC}"
        TESTS_PASSED=false
    fi
    echo ""
fi

# Run jetson_pcm_receiver tests (standalone CMake project)
if $RUN_JETSON_PCM; then
    echo -e "${YELLOW}=== Running jetson_pcm_receiver tests ===${NC}"
    JETSON_BUILD_DIR="jetson_pcm_receiver/build"
    "$CMAKE_BIN" -S jetson_pcm_receiver -B "$JETSON_BUILD_DIR" -DCMAKE_BUILD_TYPE=Release 2>&1 | tail -5
    "$CMAKE_BIN" --build "$JETSON_BUILD_DIR" --target jetson_pcm_receiver_tests -j8 2>&1 | tail -20
    if "$CTEST_BIN" --test-dir "$JETSON_BUILD_DIR" --output-on-failure; then
        echo -e "${GREEN}jetson_pcm_receiver tests passed!${NC}"
    else
        echo -e "${RED}jetson_pcm_receiver tests failed!${NC}"
        TESTS_PASSED=false
    fi
    echo ""
fi

# Summary
if ! $RUN_PYTHON && ! $RUN_CPP && ! $RUN_GPU && ! $RUN_RPI && ! $RUN_JETSON_PCM; then
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
