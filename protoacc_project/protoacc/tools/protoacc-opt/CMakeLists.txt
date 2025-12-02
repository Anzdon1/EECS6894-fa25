add_mlir_tool(protoacc-opt protoacc-opt.cpp)
target_link_libraries(protoacc-opt
  PRIVATE ProtoAcc
  MLIRIR MLIRParser MLIRSupport
  MLIROptLib        # 对应 #include "mlir/Tools/mlir-opt/MlirOptMain.h"
  MLIRFuncDialect   # 对应 #include "mlir/Dialect/Func/IR/FuncOps.h"
  MLIRTransforms    # 对应 #include "mlir/Transforms/Passes.h"
  ProtoAccTransforms

  MLIRArithDialect
  MLIRVectorDialect
  MLIRLLVMDialect
  
  # MLIRLLVMToLLVMIRTranslation         # 对应 registerLLVMDialectTranslation
  # MLIRInitAllDialects                 # 对应 registerAllDialects
  # MLIRInitAllTranslations             # 对应 registerAllTranslations
  
)