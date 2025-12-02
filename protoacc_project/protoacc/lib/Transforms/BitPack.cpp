#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"
#include "protoacc/ProtoAccDialect.h"
#include "protoacc/ProtoAccOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
// #include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
// #include "mlir/Conversion/LLVMCommon/TypeConverter.h"
// #include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#define GEN_PASS_DEF_BITPACK
#include "protoacc/ProtoAccPasses.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;
using namespace protoacc;
using namespace mlir::LLVM;

/// Match pattern:
///   %b0 = llvm.load %p       : i8
///   ...
///   %b1 = llvm.load (gep %p, 1) : i8
///
/// Rewrite to:
///   %pair    = llvm.load %p : i16
///   %b0_new  = trunc(pair & 255)
///   %b1_new  = trunc((pair >> 8) & 255)
///
// Updated CombineI8LoadsPattern with safer insertion points and dominance handling
struct CombineI8LoadsPattern : public RewritePattern {
  CombineI8LoadsPattern(MLIRContext *ctx)
      : RewritePattern(LLVM::LoadOp::getOperationName(), 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    auto load0 = dyn_cast<LLVM::LoadOp>(op);
    if (!load0)
      return failure();

    auto i8Ty = IntegerType::get(load0.getContext(), 8);
    if (load0.getResult().getType() != i8Ty)
      return failure();

    Value ptr0 = load0.getAddr();
    Block *block0 = load0->getBlock();

    // === Find gep(ptr0, 1) only inside same block ===
    LLVM::GEPOp gep1 = nullptr;
    for (Operation *user : ptr0.getUsers()) {
      if (auto gep = dyn_cast<LLVM::GEPOp>(user)) {
        if (gep->getBlock() != block0)
          continue;
        auto indices = gep.getRawConstantIndices();
        if (indices.size() == 1 && indices[0] == 1)
          gep1 = gep;
      }
    }
    if (!gep1)
      return failure();

    // === Find load(gep) only in same block ===
    LLVM::LoadOp load1 = nullptr;
    for (Operation *user : gep1.getResult().getUsers()) {
      if (auto ld = dyn_cast<LLVM::LoadOp>(user)) {
        if (ld.getResult().getType() == i8Ty && ld->getBlock() == block0)
          load1 = ld;
      }
    }
    if (!load1)
      return failure();

    // === Dominance check ===
    auto func = load0->getParentOfType<FunctionOpInterface>();
    DominanceInfo dom(func);
    if (!dom.dominates(load0.getOperation(), load1.getOperation()))
      return failure();

    // === Insert code at correct point: at load0 ===
    Location loc = load0.getLoc();
    auto ctx = load0.getContext();
    auto i16Ty = IntegerType::get(ctx, 16);

    rewriter.setInsertionPoint(load0);
    auto pair = rewriter.create<LLVM::LoadOp>(loc, i16Ty, ptr0, 0, false, false);
    auto c255 = rewriter.create<LLVM::ConstantOp>(loc, i16Ty, IntegerAttr::get(i16Ty, 255));
    auto c8 = rewriter.create<LLVM::ConstantOp>(loc, i16Ty, IntegerAttr::get(i16Ty, 8));

    auto lowMasked = rewriter.create<LLVM::AndOp>(loc, pair, c255);
    auto b0 = rewriter.create<LLVM::TruncOp>(loc, i8Ty, lowMasked);

    auto shifted = rewriter.create<LLVM::LShrOp>(loc, pair, c8);
    auto highMasked = rewriter.create<LLVM::AndOp>(loc, shifted, c255);
    auto b1 = rewriter.create<LLVM::TruncOp>(loc, i8Ty, highMasked);

    // === Lifetime-safe replacement ===
    auto isLifetime = [&](Operation *u) {
      return isa<LLVM::LifetimeStartOp>(u) || isa<LLVM::LifetimeEndOp>(u);
    };

    for (Operation *user : llvm::make_early_inc_range(load0.getResult().getUsers()))
      if (!isLifetime(user))
        user->replaceUsesOfWith(load0.getResult(), b0);

    for (Operation *user : llvm::make_early_inc_range(load1.getResult().getUsers()))
      if (!isLifetime(user))
        user->replaceUsesOfWith(load1.getResult(), b1);

    // Do NOT erase load0/load1 (needed by lifetime intrinsics)
    return success();
  }
};


struct BitPackPass : protoacc::impl::BitPackBase<BitPackPass>
{
  void getDependentDialects(DialectRegistry &registry) const final
  {
    registry.insert<LLVM::LLVMDialect>();
  }

  void runOnOperation() final {
      Operation *root = getOperation();

      // SAFELY remove all lifetime intrinsics
      SmallVector<Operation*, 32> toErase;
      root->walk([&](Operation *op) {
          if (isa<LLVM::LifetimeStartOp>(op) ||
              isa<LLVM::LifetimeEndOp>(op)) {
              toErase.push_back(op);
          }
      });
      for (auto *op : toErase)
          op->erase();

      // Now run your patterns
      RewritePatternSet patterns(&getContext());
      patterns.add<CombineI8LoadsPattern>(&getContext());

      if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns))))
        signalPassFailure();
  }

};

std::unique_ptr<mlir::Pass> protoacc::createBitPackPass()
{
  return std::make_unique<BitPackPass>();
}
