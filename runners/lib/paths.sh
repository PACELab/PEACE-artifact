#!/usr/bin/env bash
# Repository path initialization for shell runners.

if [[ -z "${REPO_ROOT:-}" ]]; then
    PATHS_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
    REPO_ROOT="$(cd -- "$PATHS_DIR/../.." && pwd)"
fi

export REPO_ROOT
