add_mlir_dialect(ProtoAcc protoacc)


set(LLVM_TARGET_DEFINITIONS ProtoAccPasses.td)
mlir_tablegen(ProtoAccPasses.h.inc -gen-pass-decls)
add_public_tablegen_target(MLIRProtoAccTransformsIncGen)

add_custom_target(header DEPENDS MLIRProtoAccIncGen MLIRProtoAccTransformsIncGen)