include "mlir/IR/DialectBase.td"

def ProtoAcc_Dialect : Dialect {
  let name = "protoacc";
  let cppNamespace = "protoacc";
}
