#include "ProtoAcc/ProtoAccOps.h"
#include "ProtoAcc/Transforms/Passes.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "llvm/ADT/APInt.h"

using namespace mlir;

namespace {
struct ZigZagPattern : OpRewritePattern<LLVM::XOrOp> {
  using OpRewritePattern<LLVM::XOrOp>::OpRewritePattern;

  LogicalResult rewriteForPair(LLVM::ShlOp shl, LLVM::AShrOp ashr,
                               LLVM::XOrOp xorOp,
                               PatternRewriter &rewriter) const {
    if (!shl || !ashr)
      return failure();

    if (shl.getLhs() != ashr.getOperand(0))
      return failure();

    auto type = dyn_cast<IntegerType>(shl.getResult().getType());
    if (!type || !type.isSignless())
      return failure();

    if (ashr.getResult().getType() != shl.getResult().getType() ||
        xorOp.getResult().getType() != shl.getResult().getType())
      return failure();

    APInt shlAmount;
    if (!matchPattern(shl.getRhs(), m_ConstantInt(&shlAmount)))
      return failure();

    if (shlAmount != 1)
      return failure();

    APInt ashrAmount;
    if (!matchPattern(ashr.getRhs(), m_ConstantInt(&ashrAmount)))
      return failure();

    if (ashrAmount != type.getWidth() - 1)
      return failure();

    auto zigzag = rewriter.create<protoacc::ZigZagEncodeOp>(
        xorOp.getLoc(), xorOp.getResult().getType(), shl.getLhs());
    rewriter.replaceOp(xorOp, zigzag.getResult());
    if (shl->use_empty())
      rewriter.eraseOp(shl);
    if (ashr->use_empty())
      rewriter.eraseOp(ashr);
    return success();
  }

  LogicalResult matchAndRewrite(LLVM::XOrOp xorOp,
                                PatternRewriter &rewriter) const override {
    auto lhsShl = xorOp.getLhs().getDefiningOp<LLVM::ShlOp>();
    auto rhsAshr = xorOp.getRhs().getDefiningOp<LLVM::AShrOp>();
    if (succeeded(rewriteForPair(lhsShl, rhsAshr, xorOp, rewriter)))
      return success();

    auto lhsAshr = xorOp.getLhs().getDefiningOp<LLVM::AShrOp>();
    auto rhsShl = xorOp.getRhs().getDefiningOp<LLVM::ShlOp>();
    if (succeeded(rewriteForPair(rhsShl, lhsAshr, xorOp, rewriter)))
      return success();

    return failure();
  }
};

struct LowerToProtoAccPass
    : public PassWrapper<LowerToProtoAccPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerToProtoAccPass)

  StringRef getArgument() const final { return "lower-to-protoacc"; }
  StringRef getDescription() const final {
    return "Recognize zigzag (x<<1)^(x>>31) in LLVM dialect and lower to protoacc.zigzag.encode";
  }

  void runOnOperation() override {
    getContext().loadDialect<protoacc::ProtoAccDialect>();
    RewritePatternSet patterns(&getContext());
    patterns.add<ZigZagPattern>(&getContext());
    if (failed(
            applyPatternsAndFoldGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};
} // namespace

namespace protoacc {
std::unique_ptr<Pass> createLowerToProtoAccPass() {
  return std::make_unique<LowerToProtoAccPass>();
}

void registerProtoAccPasses() {
  static PassRegistration<LowerToProtoAccPass> pass;
  (void)pass;
}
} // namespace protoacc
