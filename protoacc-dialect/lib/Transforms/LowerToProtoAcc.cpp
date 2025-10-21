#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;

namespace {
struct ZigZagPattern : OpRewritePattern<LLVM::XOrOp> {
  using OpRewritePattern<LLVM::XOrOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(LLVM::XOrOp xorOp,
                                PatternRewriter &rewriter) const override {
    auto shl  = xorOp.getLhs().getDefiningOp<LLVM::ShlOp>();
    auto ashr = xorOp.getRhs().getDefiningOp<LLVM::AShrOp>();
    if (!shl || !ashr) return failure();
    if (shl.getLhs() != ashr.getLhs()) return failure();

    auto cstShl  = dyn_cast_or_null<LLVM::ConstantOp>(shl.getRhs().getDefiningOp());
    auto cstAshr = dyn_cast_or_null<LLVM::ConstantOp>(ashr.getRhs().getDefiningOp());
    if (!cstShl || !cstAshr) return failure();

    auto shlAttr  = dyn_cast_or_null<IntegerAttr>(cstShl.getValue());
    auto ashrAttr = dyn_cast_or_null<IntegerAttr>(cstAshr.getValue());
    if (!shlAttr || !ashrAttr) return failure();

    if (shlAttr.getInt() != 1 || ashrAttr.getInt() != 31) return failure();

    auto loc = xorOp.getLoc();
    OperationState st(loc, "protoacc.zigzag.encode");
    st.addOperands(shl.getLhs());
    st.addTypes(rewriter.getI32Type());
    Operation *newOp = rewriter.create(st);

    rewriter.replaceOp(xorOp, newOp->getResult(0));
    return success();
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
    RewritePatternSet patterns(&getContext());
    patterns.add<ZigZagPattern>(&getContext());
    if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};
} // namespace

static PassRegistration<LowerToProtoAccPass> pass;
