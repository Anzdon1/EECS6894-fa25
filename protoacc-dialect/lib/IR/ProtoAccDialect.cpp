//===- ProtoAccDialect.cpp -----------------------------------------------===//
#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"

using namespace mlir;

namespace protoacc {
class ProtoAccDialect : public Dialect {
public:
  explicit ProtoAccDialect(MLIRContext *ctx)
      : Dialect("protoacc", ctx, TypeID::get<ProtoAccDialect>()) {
    initialize();
  }
  static StringRef getDialectNamespace() { return "protoacc"; }
  void initialize();
};
} // namespace protoacc

#include "protoacc/IR/ProtoAccOps.h.inc"

void protoacc::ProtoAccDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "protoacc/IR/ProtoAccOps.cpp.inc"
  >();
}
