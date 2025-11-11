//===- ProtoAccToLLVM.cpp --------------------------------------------------===//
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

using namespace mlir;

namespace {
struct ZigZagLowering : public OpRewritePattern<Operation> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    if (op->getName().getStringRef() != "protoacc.zigzag_encode")
      return failure();

    Value x = op->getOperand(0);
    auto it = dyn_cast<IntegerType>(x.getType());
    if (!it || it.getWidth() != 32) return failure();

    Location loc = op->getLoc();
    auto i32Ty = IntegerType::get(op->getContext(), 32);
    Value c1  = rewriter.create<LLVM::ConstantOp>(loc, i32Ty, rewriter.getI32IntegerAttr(1));
    Value c31 = rewriter.create<LLVM::ConstantOp>(loc, i32Ty, rewriter.getI32IntegerAttr(31));

    Value shl  = rewriter.create<LLVM::ShlOp>(loc, i32Ty, x, c1);
    Value ashr = rewriter.create<LLVM::AShrOp>(loc, i32Ty, x, c31);
    Value out  = rewriter.create<LLVM::XOrOp>(loc, i32Ty, shl, ashr);

    rewriter.replaceOp(op, out);
    return success();
  }
};

struct ConvertProtoAccToLLVMPass
    : public PassWrapper<ConvertProtoAccToLLVMPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ConvertProtoAccToLLVMPass)
  void runOnOperation() override {
    MLIRContext *ctx = &getContext();
    ModuleOp m = getOperation();

    RewritePatternSet patterns(ctx);
    ConversionTarget target(*ctx);

    target.addLegalDialect<LLVM::LLVMDialect>();
    target.markUnknownOpDynamicallyLegal([](Operation *) { return true; });

    patterns.add<ZigZagLowering>(ctx);

    if (failed(applyPartialConversion(m, target, std::move(patterns))))
      signalPassFailure();
  }
};
} // namespace

std::unique_ptr<Pass> createConvertProtoAccToLLVMPass() {
  return std::make_unique<ConvertProtoAccToLLVMPass>();
}
