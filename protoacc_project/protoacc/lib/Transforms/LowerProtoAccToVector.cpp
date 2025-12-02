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

/// --------------------------------------------------------------------------
/// Pattern: Lower protoacc.decode_varint → vector ops
/// --------------------------------------------------------------------------
struct DecodeVarintOpLowering : public OpRewritePattern<protoacc::DecodeVarintOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(protoacc::DecodeVarintOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value ptr = op.getPtr();
    Value ctx = op.getCtx();

    constexpr int kWidth = 16;

    // vector<16xi8>
    auto vecTy = VectorType::get({kWidth}, rewriter.getIntegerType(8));

    // Load 16 bytes: %vec = vector.load %ptr
    Value vec = rewriter.create<vector::LoadOp>(loc, vecTy, ptr);

    // Step1: broadcast(0x80)
    auto c80 = rewriter.create<arith::ConstantOp>(loc,
                    rewriter.getI8IntegerAttr(0x80));
    Value splat80 =
        rewriter.create<vector::BroadcastOp>(loc, vecTy, c80);

    // Step2: compare >= 0x80  → continuation mask
    Value mask = rewriter.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::uge, vec, splat80);

    // Step3: find first false in mask (manual implementation)
    // 3.1 cast i1 vector -> i32 vector
    auto vecI32Ty = VectorType::get({kWidth}, rewriter.getI32Type());
    Value maskExt = rewriter.create<arith::ExtUIOp>(loc, vecI32Ty, mask);

    // 3.2 build index vector [0..kWidth-1]
    SmallVector<int32_t> seq;
    for (int i = 0; i < kWidth; i++)
      seq.push_back(i);

    auto seqAttr =
        DenseIntElementsAttr::get(vecI32Ty, llvm::ArrayRef(seq));
    Value indexVec = rewriter.create<arith::ConstantOp>(loc, seqAttr);

    // create 0-splat
    Value zero = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getI32IntegerAttr(0));
    Value zeroVec = rewriter.create<vector::BroadcastOp>(loc, vecI32Ty, zero);

    // isFalse = (maskExt == 0)
    Value isFalse = rewriter.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::eq, maskExt, zeroVec);

    // select false-index or huge
    auto huge = rewriter.getI32IntegerAttr(999999);
    Value hugeC = rewriter.create<arith::ConstantOp>(loc, huge);
    Value hugeVec =
        rewriter.create<vector::BroadcastOp>(loc, vecI32Ty, hugeC);

    Value candidate = rewriter.create<arith::SelectOp>(
        loc, isFalse, indexVec, hugeVec);

    // reduce(min) to get first false index
    Value firstFalse = rewriter.create<vector::ReductionOp>(
        loc, arith::AtomicRMWKind::minu, candidate);

    // Step4: varint value = extract byte0
    Value b0 = rewriter.create<vector::ExtractOp>(loc, vec, 0);
    Value val = rewriter.create<arith::ExtUIOp>(
        loc, rewriter.getI32Type(), b0);

    // Step5: newPtr = ptr + (firstFalse+1)
    auto one = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getI32IntegerAttr(1));
    Value byteCount = rewriter.create<arith::AddIOp>(loc, firstFalse, one);

    Value byteCount64 = rewriter.create<arith::ExtUIOp>(
        loc, rewriter.getI64Type(), byteCount);

    Value newPtr = rewriter.create<arith::AddIOp>(
        loc, ptr, byteCount64);

    // replace op
    rewriter.replaceOp(op, {val, newPtr});
    return success();
  }
};


/// --------------------------------------------------------------------------
/// Pass
/// --------------------------------------------------------------------------
struct LowerProtoAccToVectorPass
    : protoacc::impl::LowerProtoAccToVectorBase<LowerProtoAccToVectorPass> {

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<mlir::arith::ArithDialect>();
    registry.insert<mlir::vector::VectorDialect>();
    registry.insert<protoacc::ProtoAccDialect>();
  }

  void runOnOperation() override {
    MLIRContext *ctx = &getContext();
    RewritePatternSet patterns(ctx);

    patterns.add<DecodeVarintOpLowering>(ctx);

    if (failed(mlir::applyPatternsAndFoldGreedily(
            getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

/// Pass factory
std::unique_ptr<mlir::Pass> protoacc::createLowerProtoAccToVectorPass() {
  return std::make_unique<LowerProtoAccToVectorPass>();
}