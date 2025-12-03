#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define GEN_PASS_DEF_LOWERPROTOACCTOVECTOR
#include "protoacc/ProtoAccDialect.h"
#include "protoacc/ProtoAccOps.h"
#include "protoacc/ProtoAccPasses.h"

using namespace mlir;
using namespace mlir::LLVM;
using namespace protoacc;

namespace {

/// Lower protoacc.decode_varint to plain LLVM integer ops.
///
/// Signature:
///   %val, %next = protoacc.decode_varint %ptr : (!llvm.ptr<i8>) -> (i64, !llvm.ptr<i8>)
///
/// Strategy:
///   - Cast ptr (i8*) to i64*
///   - Load 8 bytes as one i64
///   - Unroll 8 bytes, accumulate 7-bit payloads until MSB == 0
///   - Compute number of consumed bytes and new pointer
///   - Return (value, newPtr)
struct DecodeVarintOpLowering : public OpRewritePattern<protoacc::DecodeVarintOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(protoacc::DecodeVarintOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    MLIRContext *ctx = rewriter.getContext();

    // Use builtin integer types (compatible with your LLVM dialect version).
    Type llvmI64Ty = rewriter.getI64Type();
    Type llvmI8Ty  = rewriter.getI8Type();

    // First operand is expected to be !llvm.ptr<i8>
    if (op->getNumOperands() < 1)
      return failure();

    Value ptr = op->getOperand(0);
    Type ptrTy = ptr.getType();

    auto i8PtrTy = dyn_cast<LLVMPointerType>(ptrTy);
    if (!i8PtrTy)
      return failure();

    // Pointer to i64 (i64*)
    auto i64PtrTy = LLVMPointerType::get(ctx);

    // 1) Cast i8* to i64* and load 8 bytes at once.
    //    Note: we assume alignment is sufficient for this cast.
    Value castPtr = rewriter.create<LLVM::BitcastOp>(loc, i64PtrTy, ptr);
    Value chunk   = rewriter.create<LLVM::LoadOp>(loc, llvmI64Ty, castPtr);

    auto makeConst = [&](uint64_t v) -> Value {
      return rewriter.create<LLVM::ConstantOp>(
          loc, llvmI64Ty, rewriter.getIntegerAttr(llvmI64Ty, v));
    };

    Value zero   = makeConst(0);
    Value maskFF = makeConst(0xff);
    Value mask7F = makeConst(0x7f);
    Value mask80 = makeConst(0x80);

    // Accumulated result and current shift amount (in bits).
    Value result = zero;
    Value shift  = zero;

    // Number of bytes consumed (in range [1, 8]), 0 means "not set yet".
    Value consumedBytes = nullptr;

    // 2) Unroll up to 8 bytes.
    for (int i = 0; i < 8; ++i) {
      // Extract byte i: (chunk >> (8*i)) & 0xff
      Value shiftAmt8 = makeConst(8ull * static_cast<uint64_t>(i));
      Value bShifted  = rewriter.create<LLVM::LShrOp>(loc, chunk, shiftAmt8);
      Value byteVal   = rewriter.create<LLVM::AndOp>(loc, bShifted, maskFF);

      // payload = byteVal & 0x7f
      Value payload = rewriter.create<LLVM::AndOp>(loc, byteVal, mask7F);

      // payload << shift
      Value payloadShifted =
          rewriter.create<LLVM::ShlOp>(loc, payload, shift);

      // result |= payloadShifted
      result =
          rewriter.create<LLVM::OrOp>(loc, result, payloadShifted);

      // continuation bit: byteVal & 0x80
      Value contBit = rewriter.create<LLVM::AndOp>(loc, byteVal, mask80);
      Value isZero  = rewriter.create<LLVM::ICmpOp>(
          loc, LLVM::ICmpPredicate::eq, contBit, zero);

      // thisConsumed = i + 1
      Value thisConsumed = makeConst(static_cast<uint64_t>(i + 1));

      // If this is the first byte where MSB == 0, record i+1.
      if (!consumedBytes) {
        consumedBytes = rewriter.create<LLVM::SelectOp>(
            loc, isZero, thisConsumed, zero);
      } else {
        // Keep previous non-zero value; otherwise overwrite with thisConsumed.
        Value isPrevZero = rewriter.create<LLVM::ICmpOp>(
            loc, LLVM::ICmpPredicate::eq, consumedBytes, zero);
        Value merged = rewriter.create<LLVM::SelectOp>(
            loc, isPrevZero, thisConsumed, consumedBytes);
        consumedBytes = merged;
      }

      // shift += 7 for next loop
      Value seven = makeConst(7);
      shift = rewriter.create<LLVM::AddOp>(loc, shift, seven);
    }

    // Fallback if we never saw a terminating byte: assume 8 bytes consumed.
    Value eight = makeConst(8);
    Value isConsZero = rewriter.create<LLVM::ICmpOp>(
        loc, LLVM::ICmpPredicate::eq, consumedBytes, zero);
    consumedBytes = rewriter.create<LLVM::SelectOp>(
        loc, isConsZero, eight, consumedBytes);

    // 3) Compute new pointer: ptr + consumedBytes (i8*).
    //    GEPOp in this MLIR version needs resultType + elementType.
    Value newPtr = rewriter.create<LLVM::GEPOp>(
        loc,
        i8PtrTy,   // result pointer type (!llvm.ptr<i8>)
        llvmI8Ty,  // element type (i8)
        ptr,
        ValueRange{consumedBytes});

    // 4) Replace protoacc.decode_varint with (result, newPtr).
    rewriter.replaceOp(op, ValueRange{result, newPtr});
    return success();
  }
};

/// Pass: remove lifetime intrinsics and apply DecodeVarintOpLowering.
struct LowerProtoAccToVectorPass
    : protoacc::impl::LowerProtoAccToVectorBase<LowerProtoAccToVectorPass> {

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<LLVMDialect>();
    registry.insert<protoacc::ProtoAccDialect>();
  }

  void runOnOperation() override {
    Operation *root = getOperation();
    MLIRContext *ctx = &getContext();
    RewritePatternSet patterns(ctx);

    // ------------------------------------------------------------------
    // 0) Remove all llvm.lifetime.start / llvm.lifetime.end intrinsics.
    //    These can cause the LLVM verifier to fail after lowering.
    // ------------------------------------------------------------------
    SmallVector<Operation *> toErase;
    root->walk([&](LLVM::LifetimeStartOp op) {
      toErase.push_back(op.getOperation());
    });
    root->walk([&](LLVM::LifetimeEndOp op) {
      toErase.push_back(op.getOperation());
    });
    for (Operation *op : toErase)
      op->erase();

    // ------------------------------------------------------------------
    // 1) Apply DecodeVarintOpLowering patterns.
    // ------------------------------------------------------------------
    patterns.add<DecodeVarintOpLowering>(ctx);

    if (failed(applyPatternsAndFoldGreedily(root, std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

/// Pass factory
std::unique_ptr<mlir::Pass> protoacc::createLowerProtoAccToVectorPass() {
  return std::make_unique<LowerProtoAccToVectorPass>();
}
