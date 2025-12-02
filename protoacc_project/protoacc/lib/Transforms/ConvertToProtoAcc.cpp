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

    // Expect a single struct result (ptr, int) so we can substitute
    // protoacc.decode_varint's two results.
    if (op->getNumResults() != 1)
      return failure();

    auto structTy =
        dyn_cast<LLVM::LLVMStructType>(op.getResult().getType());
    if (!structTy || structTy.getBody().size() != 2)
      return failure();

    auto ptrTy = dyn_cast<LLVM::LLVMPointerType>(structTy.getBody()[0]);
    auto intTy = dyn_cast<IntegerType>(structTy.getBody()[1]);
    if (!ptrTy || !intTy)
      return failure();

    // The first operand is the input pointer. Use a pointer operand as ctx if
    // available; otherwise, fall back to the pointer itself (ctx is unused in
    // lowering today).
    if (op.getNumOperands() < 1)
      return failure();
    Value ptr = op.getOperand(0);
    Value ctx = ptr;
    for (Value candidate : op.getOperands().drop_front()) {
      if (isa<LLVM::LLVMPointerType>(candidate.getType())) {
        ctx = candidate;
        break;
      }
    }

    // Only rewrite if the call result is decomposed via extractvalue ops.
    SmallVector<LLVM::ExtractValueOp> extracts;
    for (Operation *user : op.getResult().getUsers())
    {
        if (auto ev = dyn_cast<LLVM::ExtractValueOp>(user))
        {
            auto pos = ev.getPosition();
            if (pos.size() == 1 && (pos[0] == 0 || pos[0] == 1))
            {
                extracts.push_back(ev);
                continue;
            }
        }
        return failure();
    }

    Location loc = op.getLoc();

    // Create ProtoAcc dialect op
    auto newOp = rewriter.create<protoacc::DecodeVarintOp>(loc, ptr, ctx);

    // Replace extractvalue uses with the appropriate results.
    for (auto ev : extracts) {
      rewriter.setInsertionPoint(ev);
      auto pos = ev.getPosition();
      Value replacement;
      if (pos[0] == 0)
        replacement = newOp.getNewPtr();
      else
        replacement = newOp.getValue();

      // Narrow to the expected integer size if needed.
      if (replacement.getType() != ev.getType()) {
        replacement = rewriter.create<LLVM::TruncOp>(
            ev.getLoc(), ev.getType(), replacement);
      }

      rewriter.replaceOp(ev, replacement);
    }

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
