#include "mlir/IR/Dialect.h"
#include "mlir/IR/Builders.h"

using namespace mlir;

namespace protoacc {
class ProtoAccDialect : public Dialect {
public:
  explicit ProtoAccDialect(MLIRContext *ctx)
      : Dialect(getDialectNamespace(), ctx, TypeID::get<ProtoAccDialect>()) {
    addOperations<
#define GET_OP_LIST
#include "ProtoAcc/ProtoAccOps.cpp.inc"
        >();
  }
  static StringRef getDialectNamespace() { return "protoacc"; }
};
} // namespace protoacc

#include "ProtoAcc/ProtoAccDialect.cpp.inc"
