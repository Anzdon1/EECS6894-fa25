#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
SRC_DIR="${SRC_DIR:-${ROOT}/benchmarks}"
OUT_DIR="${OUT_DIR:-${ROOT}/mlir}"
BUILD_DIR="${ROOT}/build}"
WORKLOADS="${WORKLOADS:-addressbook mapreduce telemetry}"
CFLAGS="${CFLAGS:--O2 -fno-discard-value-names -Xclang -disable-O0-optnone}"

mkdir -p "${OUT_DIR}" "${BUILD_DIR}"

echo "===> Source dir: ${SRC_DIR}"
echo "===> Workloads : ${WORKLOADS}"
echo "===> Output dir: ${OUT_DIR}"

for W in ${WORKLOADS}; do
  SRC_CXX=$(find "${SRC_DIR}" -type f \( -name "*.cc" -o -name "*.cpp" -o -name "*.c" \) \
            | grep -i "/${W}" | head -n1 || true)
  if [[ -z "${SRC_CXX}" ]]; then
    echo "No source found for '${W}'. Try setting SRC_DIR or specify files."
    echo "Hint: run 'find ${SRC_DIR} -iname \"*${W}*.c*\" | head'"
    exit 1
  fi

  echo "[${W}] source: ${SRC_CXX}"
  clang ${CFLAGS} -S -emit-llvm "${SRC_CXX}" -o "${BUILD_DIR}/${W}.ll"

  echo "[${W}] LLVM IR -> MLIR (LLVM dialect)"
  mlir-translate --import-llvm "${BUILD_DIR}/${W}.ll" > "${OUT_DIR}/${W}_llvm.mlir"
  echo "OK  -> ${OUT_DIR}/${W}_llvm.mlir"
done

echo "===> Done: $(ls ${OUT_DIR}/*_llvm.mlir | wc -l) MLIR files"
