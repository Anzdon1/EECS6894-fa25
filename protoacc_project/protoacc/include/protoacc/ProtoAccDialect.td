#ifndef PROTOACC_DIALECT_TD
#define PROTOACC_DIALECT_TD

include "mlir/IR/OpBase.td"

def ProtoAccDialect : Dialect {
  let name = "protoacc";
  let cppNamespace = "::protoacc";
  let summary = "ProtoAcc Dialect";
}

#endif
