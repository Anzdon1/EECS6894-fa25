cmake_minimum_required(VERSION 3.20)
project(protoacc-min LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Locate MLIR/LLVM (expect MLIR_DIR/LLVM_DIR to be configured by the user)
find_package(MLIR REQUIRED CONFIG)
find_package(LLVM REQUIRED CONFIG)
list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")

# Core TableGen helpers (some environments ship reduced CMake packages)
include(AddLLVM)
include(TableGen)
include(AddMLIR)

# Include directories for both C++ sources and TableGen invocations
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/protoacc-dialect/include)
include_directories(${CMAKE_CURRENT_BINARY_DIR})
include_directories(${MLIR_INCLUDE_DIRS} ${LLVM_INCLUDE_DIRS})
include_directories(/usr/local/include)   # Ensure OpBase.td is discoverable

# ------- TableGen: explicitly declare inputs and additional include paths -------
set(LLVM_TARGET_DEFINITIONS
  ${CMAKE_CURRENT_SOURCE_DIR}/protoacc-dialect/include/protoacc/IR/ProtoAccOps.td)

# Declarations (.h.inc)
mlir_tablegen(ProtoAccOps.h.inc -gen-op-decls
  -I/usr/local/include -I${MLIR_INCLUDE_DIRS} -I${LLVM_INCLUDE_DIRS})
# Definitions (.cpp.inc)
mlir_tablegen(ProtoAccOps.cpp.inc -gen-op-defs
  -I/usr/local/include -I${MLIR_INCLUDE_DIRS} -I${LLVM_INCLUDE_DIRS})

# Handle LLVM builds that omit add_public_tablegen_target
add_custom_target(ProtoAccOpsIncGen
  DEPENDS ProtoAccOps.h.inc ProtoAccOps.cpp.inc)

# ------- IR library -------
add_library(ProtoAccIR
  protoacc-dialect/lib/IR/ProtoAccDialect.cpp
  protoacc-dialect/lib/IR/ProtoAccOps.cpp
)
add_dependencies(ProtoAccIR ProtoAccOpsIncGen)
target_link_libraries(ProtoAccIR PUBLIC MLIRIR)

# ------- Lowering/Conversion library -------
add_library(ProtoAccConversion
  protoacc-dialect/lib/Conversion/ProtoAccToLLVM/ProtoAccToLLVM.cpp
)
target_link_libraries(ProtoAccConversion PUBLIC
  ProtoAccIR
  MLIRLLVMDialect
  MLIRIR
  MLIRPass
  MLIRTransforms
)

# ------- protoacc-opt tool -------
add_executable(protoacc-opt
  protoacc-dialect/tools/protoacc-opt/protoacc-opt.cpp
)
target_link_libraries(protoacc-opt PRIVATE
  ProtoAccIR
  ProtoAccConversion
  MLIRIR
  MLIRPass
  MLIRSupport
  MLIRLLVMDialect
)

install(TARGETS protoacc-opt RUNTIME DESTINATION bin)
