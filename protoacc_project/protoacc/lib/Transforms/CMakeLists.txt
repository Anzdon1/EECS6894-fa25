add_mlir_library(
  ProtoAccTransforms
  # ConvertProtoAccToArith.cpp
  # DCE.cpp
  BitPack.cpp
  ConvertToProtoAcc.cpp
  LowerProtoAccToVector.cpp

  DEPENDS MLIRProtoAccTransformsIncGen

  LINK_LIBS MLIRArithDialect
  MLIRArithToLLVM
  MLIRLLVMDialect
  MLIRDLTIDialect
  MLIRVectorDialect
)
