cmake_minimum_required(VERSION 3.20)
project(ProtoAcc LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Release)

set(LLVM_ENABLE_RTTI ON CACHE BOOL "" FORCE)

find_package(LLVM REQUIRED CONFIG)
find_package(MLIR REQUIRED CONFIG)

message(STATUS "Using LLVM at ${LLVM_DIR}")
message(STATUS "Using MLIR at ${MLIR_DIR}")

list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
include(AddLLVM)
include(TableGen)
include(AddMLIR)
include(HandleLLVMOptions)

include_directories(
  ${LLVM_INCLUDE_DIRS}
  ${MLIR_INCLUDE_DIRS}
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)
add_definitions(${LLVM_DEFINITIONS})
mlir_tablegen(ProtoAccDialect.h.inc -gen-dialect-decls -dialect=protoacc)
mlir_tablegen(ProtoAccDialect.cpp.inc -gen-dialect-defs -dialect=protoacc)
mlir_tablegen(ProtoAccOps.h.inc -gen-op-decls)
mlir_tablegen(ProtoAccOps.cpp.inc -gen-op-defs)
add_public_tablegen_target(ProtoAccTableGen)

add_mlir_library(ProtoAcc
  lib/ProtoAcc/ProtoAccDialect.cpp
  lib/ProtoAcc/ProtoAccOps.cpp
  DEPENDS ProtoAccTableGen
  LINK_LIBS PUBLIC MLIRIR
)

add_mlir_library(ProtoAccTransforms
  lib/Transforms/LowerToProtoAcc.cpp
  LINK_LIBS PUBLIC
    ProtoAcc
    MLIRIR
    MLIRLLVMDialect
    MLIRRewrite
    MLIRTransforms
)
