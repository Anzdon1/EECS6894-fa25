#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/PatternMatch.h"
#define GEN_PASS_DEF_LOWERPROTOACCTOVECTOR
#include "protoacc/ProtoAccDialect.h"
#include "protoacc/ProtoAccOps.h"
#include "protoacc/ProtoAccPasses.h"

using namespace protoacc;
using namespace mlir;

namespace {

struct DecodeVarintOpLowering : public OpRewritePattern<DecodeVarintOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(DecodeVarintOp op,
                                PatternRewriter &rewriter) const override {

    Location loc = op.getLoc();
    Value ptr = op.getPtr();
    Value ctx = op.getCtx();

    auto i8 = rewriter.getIntegerType(8);
    auto i32 = rewriter.getIntegerType(32);

    VectorType vec16Ty = VectorType::get({16}, i8);
    VectorType vec5Ty = VectorType::get({5}, i8);
    VectorType vec5i32Ty = VectorType::get({5}, i32);

    Value c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);

    // VEC LOAD
    Value vec = rewriter.create<vector::LoadOp>(loc, vec16Ty, ptr, ValueRange{c0});

    // MASK >= 128
    auto th = rewriter.create<arith::ConstantIntOp>(loc, 127, i8);
    Value mask = rewriter.create<vector::CmpIOp>(
        loc, vector::CmpIPredicate::ugt, vec,
        rewriter.create<vector::BroadcastOp>(loc, vec16Ty, th));

    // first_non_msb
    Value len = rewriter.create<vector::FirstFalseOp>(loc, mask);

    // extract first 5 bytes
    Value slice = rewriter.create<vector::ExtractStridedSliceOp>(
        loc, vec5Ty, vec, ArrayRef<int64_t>{0}, ArrayRef<int64_t>{5}, ArrayRef<int64_t>{1});

    // cast to i32
    Value slice32 = rewriter.create<arith::ExtUIOp>(loc, vec5i32Ty, slice);

    // mask 0x7f
    Value mask7f = rewriter.create<arith::ConstantOp>(
        loc, DenseElementsAttr::get(vec5i32Ty, {127,127,127,127,127}));
    Value masked = rewriter.create<arith::AndIOp>(loc, slice32, mask7f);

    // shifts
    Value shifts = rewriter.create<arith::ConstantOp>(
        loc, DenseElementsAttr::get(vec5i32Ty, {0,7,14,21,28}));
    Value shifted = rewriter.create<arith::ShLIOp>(loc, masked, shifts);

    // reduction OR
    Value result = rewriter.create<vector::ReductionOp>(
        loc, i32, rewriter.getStringAttr("or"), shifted);

    // new ptr = ptr + len
    auto i64Ty = rewriter.getIntegerType(64);
    Value len64 = rewriter.create<arith::IndexCastOp>(loc, i64Ty, len);
    Value newPtr =
        rewriter.create<LLVM::GEPOp>(loc, ptr.getType(), ptr, ValueRange{len64});

    rewriter.replaceOp(op, {result, newPtr});
    return success();
  }
};

struct LowerProtoAccToVectorPass
  : protoacc::impl::LowerProtoAccToVectorBase<LowerProtoAccToVectorPass> {

  void runOnOperation() override {
    MLIRContext *ctx = &getContext();
    RewritePatternSet patterns(ctx);

    patterns.add<DecodeVarintOpLowering>(ctx);

    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
  }
};

} // namespace

std::unique_ptr<mlir::Pass> protoacc::createLowerProtoAccToVectorPass() {
  return std::make_unique<LowerProtoAccToVectorPass>();
}
