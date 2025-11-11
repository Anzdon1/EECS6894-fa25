//===- ProtoAccDialect.td ---------------------------------------*- tablegen -*-===//
include "mlir/IR/DialectSpecification.td"

def ProtoAcc_Dialect : Dialect {
  let name = "protoacc";
  let cppNamespace = "::protoacc";
  let summary = "ProtoAcc dialect for protobuf serialization";
  let description = [{
    Minimal subset for protobuf patterns (zigzag only for this week).
  }];
}
