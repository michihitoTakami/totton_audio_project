#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  check_repo_hygiene.sh [--staged]

Checks:
  - No nested git directories/files are staged/tracked (e.g. external/**/.git)
  - No submodule metadata is staged/tracked (.gitmodules)
  - No common generated artifacts are tracked
EOF
}

mode="tracked"
if [[ "${1:-}" == "--staged" ]]; then
  mode="staged"
  shift
elif [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
elif [[ -n "${1:-}" ]]; then
  echo "Unknown argument: ${1}" >&2
  usage >&2
  exit 2
fi

if [[ "${mode}" == "staged" ]]; then
  mapfile -t files < <(git diff --cached --name-only --diff-filter=ACMR)
else
  mapfile -t files < <(git ls-files)
fi

fail=0

check_pattern() {
  local title="$1"
  local pattern="$2"
  local hit=0
  for f in "${files[@]}"; do
    if [[ "${f}" =~ ${pattern} ]]; then
      if [[ $hit -eq 0 ]]; then
        echo "NG: ${title}" >&2
        hit=1
      fi
      echo "  - ${f}" >&2
      fail=1
    fi
  done
}

check_pattern "ネストされた .git が含まれています（ネストgit禁止）" '(^|/)\.git($|/)'
check_pattern ".gitmodules が含まれています（このリポジトリでは submodule を使わない方針）" '(^|/)\.gitmodules$'

check_pattern "生成物っぽいファイルが追跡されています（.gitignore を見直してください）" \
  '(^|/)(build|build-[^/]+|dist|CMakeFiles|__pycache__|\.pytest_cache|\.mypy_cache|\.ruff_cache|\.venv|venv)/'
check_pattern "CMake 生成物が追跡されています" '(^|/)(CMakeCache\.txt|cmake_install\.cmake|Makefile)$'

if [[ $fail -ne 0 ]]; then
  echo "" >&2
  echo "対処:" >&2
  echo "  - 誤って追加した場合は: git restore --staged <path> で取り消し" >&2
  echo "  - 生成物は .gitignore に追加し、追跡済みなら: git rm -r --cached <path>" >&2
  exit 1
fi

echo "OK: repo hygiene (${mode})"

