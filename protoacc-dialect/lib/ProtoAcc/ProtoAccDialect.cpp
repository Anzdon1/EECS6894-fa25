#include "ProtoAcc/ProtoAccDialect.h"
#include "ProtoAcc/ProtoAccOps.h"

#include "mlir/IR/DialectImplementation.h"

using namespace mlir;
using namespace protoacc;

void ProtoAccDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "ProtoAcc/ProtoAccOps.cpp.inc"
      >();
}

#include "ProtoAcc/ProtoAccDialect.cpp.inc"
