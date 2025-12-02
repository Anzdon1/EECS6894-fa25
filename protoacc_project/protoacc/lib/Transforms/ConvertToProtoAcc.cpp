//===- ConvertToProtoAcc.cpp - Convert LLVM Protobuf Ops to ProtoAcc ------===//
//
// This pass matches protobuf decoding calls in LLVM dialect and replaces them
// with ProtoAcc dialect ops (e.g., decode_varint).
//
//===----------------------------------------------------------------------===//


#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#define GEN_PASS_DEF_CONVERTTOPROTOACC
#include "protoacc/ProtoAccDialect.h"
#include "protoacc/ProtoAccOps.h"
#include "protoacc/ProtoAccPasses.h"

using namespace mlir;
using namespace mlir::LLVM;
using namespace protoacc;

/// --------------------------------------------------------------------------
/// Pattern: Match protobuf varint decode calls
/// e.g. llvm.call @_ZN6google8protobuf8internal17VarintParseSlow64EPKcj(...)
/// --------------------------------------------------------------------------
namespace {

struct MatchVarintPattern : public OpRewritePattern<LLVM::CallOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(LLVM::CallOp op,
                                PatternRewriter &rewriter) const override {
    // Callee must be a symbol reference
    Attribute attr = op.getCalleeAttr();
    if (!attr)
      return failure();

    auto sym = dyn_cast<FlatSymbolRefAttr>(attr);
    if (!sym)
      return failure();

    StringRef callee = sym.getValue();

    // We match protobuf varint parsing functions
    bool isVarintCall =
        callee.contains("VarintParse") ||
        callee.contains("ReadVarint") ||
        callee.contains("ReadTagFallback");

    if (!isVarintCall)
      return failure();

    // Require at least 2 operands: ptr, ctx
    if (op.getNumOperands() < 2)
      return failure();

    Location loc = op.getLoc();
    Value ptr = op.getOperand(0);
    Value ctx = op.getOperand(1);

    // Create ProtoAcc dialect op
    auto newOp = rewriter.create<protoacc::DecodeVarintOp>(loc, ptr, ctx);

    // Replace original call
    op.replaceAllUsesWith(newOp.getResults());
    rewriter.eraseOp(op);

    return success();
  }
};

} // namespace

/// --------------------------------------------------------------------------
/// Pass: ConvertToProtoAcc
/// --------------------------------------------------------------------------

namespace {
struct ConvertToProtoAccPass
    : protoacc::impl::ConvertToProtoAccBase<ConvertToProtoAccPass> {

  void runOnOperation() override {
    MLIRContext *ctx = &getContext();

    // Register pattern
    RewritePatternSet patterns(ctx);
    patterns.add<MatchVarintPattern>(ctx);

    // Apply patterns greedily
    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                            std::move(patterns))))
      signalPassFailure();
  }
};
} // namespace

/// Register pass
std::unique_ptr<mlir::Pass> protoacc::createConvertToProtoAccPass() {
  return std::make_unique<ConvertToProtoAccPass>();
}
