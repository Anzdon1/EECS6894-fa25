#ifndef PROTOACC_PROTOACCOPS_H
#define PROTOACC_PROTOACCOPS_H

#include "ProtoAcc/ProtoAccDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"

#define GET_OP_CLASSES
#include "ProtoAcc/ProtoAccOps.h.inc"

#endif // PROTOACC_PROTOACCOPS_H
