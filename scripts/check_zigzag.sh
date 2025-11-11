#!/usr/bin/env bash
set -euo pipefail
BUILD_DIR="${BUILD_DIR:-build}"
BIN="${BUILD_DIR}/bin/protoacc-opt"
TEST="mlir/tests/protoacc/zigzag_llvm.mlir"
if [[ ! -x "${BIN}" ]]; then
  echo "protoacc-opt not found at ${BIN}. Build first."; exit 1
fi
"${BIN}" "${TEST}" --convert-protoacc-to-llvm | mlir-opt | FileCheck "${TEST}"
echo "[OK] zigzag lowering passed."
