cmake_minimum_required(VERSION 3.20)
project(ProtoAcc LANGUAGES C CXX)

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
  ${CMAKE_CURRENT_BINARY_DIR}/include
)
add_definitions(${LLVM_DEFINITIONS})
file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/include/ProtoAcc)
set(PROTOACC_TABLEGEN_INCLUDES
  -I${CMAKE_CURRENT_SOURCE_DIR}/include
  -I${CMAKE_CURRENT_SOURCE_DIR}/include/ProtoAcc)

set(LLVM_TARGET_DEFINITIONS include/ProtoAcc/ProtoAccDialect.td)
mlir_tablegen(include/ProtoAcc/ProtoAccDialect.h.inc -gen-dialect-decls -dialect=protoacc ${PROTOACC_TABLEGEN_INCLUDES})
mlir_tablegen(include/ProtoAcc/ProtoAccDialect.cpp.inc -gen-dialect-defs -dialect=protoacc ${PROTOACC_TABLEGEN_INCLUDES})

set(LLVM_TARGET_DEFINITIONS include/ProtoAcc/ProtoAccOps.td)
mlir_tablegen(include/ProtoAcc/ProtoAccOps.h.inc -gen-op-decls ${PROTOACC_TABLEGEN_INCLUDES})
mlir_tablegen(include/ProtoAcc/ProtoAccOps.cpp.inc -gen-op-defs ${PROTOACC_TABLEGEN_INCLUDES})
add_public_tablegen_target(ProtoAccTableGen)

add_mlir_library(ProtoAcc
  lib/ProtoAcc/ProtoAccDialect.cpp
  lib/ProtoAcc/ProtoAccOps.cpp
  DEPENDS ProtoAccTableGen
  LINK_LIBS PUBLIC MLIRIR
)

add_mlir_library(ProtoAccTransforms
  lib/Transforms/LowerToProtoAcc.cpp
  DEPENDS ProtoAccTableGen
  LINK_LIBS PUBLIC
    ProtoAcc
    MLIRIR
    MLIRLLVMDialect
    MLIRPass
    MLIRRewrite
    MLIRTransforms
)

add_executable(protoacc-opt
  tools/protoacc-opt.cpp
)

set_target_properties(protoacc-opt PROPERTIES OUTPUT_NAME "protoacc-opt")

target_link_libraries(protoacc-opt
  PRIVATE
    ProtoAccTransforms
    ProtoAcc
    MLIRDLTIDialect
    MLIRIR
    MLIRParser
    MLIROptLib
    MLIRPass
    MLIRSupport
    MLIRTransforms
    MLIRLLVMDialect
)
