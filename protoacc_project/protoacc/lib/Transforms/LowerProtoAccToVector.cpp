#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#define GEN_PASS_DEF_LOWERPROTOACCTOVECTOR
#include "protoacc/ProtoAccDialect.h"
#include "protoacc/ProtoAccOps.h"
#include "protoacc/ProtoAccPasses.h"

using namespace protoacc;
using namespace mlir;

namespace
{

    /// --------------------------------------------------------------------------
    /// Pattern: Lower protoacc.decode_varint → vector ops
    /// --------------------------------------------------------------------------
    struct DecodeVarintOpLowering : public OpRewritePattern<protoacc::DecodeVarintOp>
    {
        using OpRewritePattern::OpRewritePattern;

        LogicalResult matchAndRewrite(protoacc::DecodeVarintOp op,
                                      PatternRewriter &rewriter) const override
        {
            Location loc = op.getLoc();

            Value ptr = op.getPtr();
            Value ctx = op.getCtx();

            LLVM::LLVMPointerType i8PtrTy =
                LLVM::LLVMPointerType::get(rewriter.getContext());
            LLVM::LLVMPointerType i64PtrTy =
                LLVM::LLVMPointerType::get(rewriter.getContext());


            // ============================================================
            // Step 1: 将指针转为 i8*，用于逐字节加载
            // ============================================================
            Value basePtr = ptr;

            // vector<16xi8> 类型
            auto vecTy = VectorType::get({16}, rewriter.getI8Type());

            // ============================================================
            // Step 2: 生成 16 字节的 load
            // %vec = llvm.load <vector<16xi8>>
            // ============================================================
            Value vec = rewriter.create<LLVM::LoadOp>(loc, vecTy, basePtr);

            // ============================================================
            // Step 3: 比较每个字节 & 0x80
            // mask = (vec & 0x80)
            // ============================================================
            Value const80 = rewriter.create<LLVM::ConstantOp>(
                loc, rewriter.getI8Type(), rewriter.getIntegerAttr(rewriter.getI8Type(), 0x80));
            auto splat80 = rewriter.create<vector::BroadcastOp>(loc, vecTy, const80);

            Value masked = rewriter.create<LLVM::AndOp>(loc, vec, splat80);

            // ============================================================
            // Step 4: 查找第一个 masked == 0 的位置
            // 我们使用 (mask != 0 ? huge : index)
            // 然后取最小值作为 first_zero_pos
            // ============================================================
            auto hugeVal = rewriter.create<LLVM::ConstantOp>(
                loc, rewriter.getI64Type(), rewriter.getIntegerAttr(rewriter.getI64Type(), 999));

            // 构造 index = vector<16xi64> = [0..15]
            SmallVector<Attribute> indexAttrs;
            for (int i = 0; i < 16; i++)
                indexAttrs.push_back(rewriter.getI64IntegerAttr(i));

            auto vec64Ty = VectorType::get({16}, rewriter.getI64Type());
            auto indexVec = rewriter.create<LLVM::ConstantOp>(
                loc,
                vec64Ty,
                DenseElementsAttr::get(vec64Ty, indexAttrs));

            // 将 masked : vector<16xi8> 转为 vector<16xi64>
            // extend to i64
            auto mask64 = rewriter.create<LLVM::ZExtOp>(
                loc,
                VectorType::get({16}, rewriter.getI64Type()),
                masked);

            // cmp = (mask64 != 0)
            Value zero64 = rewriter.create<LLVM::ConstantOp>(
                loc, rewriter.getI64Type(), rewriter.getIntegerAttr(rewriter.getI64Type(), 0));
            auto zeroVec = rewriter.create<vector::BroadcastOp>(
                loc, VectorType::get({16}, rewriter.getI64Type()), zero64);

            auto cmp = rewriter.create<LLVM::ICmpOp>(
                loc,
                LLVM::ICmpPredicate::ne,
                mask64,
                zeroVec);

            // candidate = select(cmp, hugeVal, index)
            auto hugeVec = rewriter.create<vector::BroadcastOp>(
                loc, VectorType::get({16}, rewriter.getI64Type()), hugeVal);

            Value candidate =
                rewriter.create<LLVM::SelectOp>(loc, cmp, hugeVec, indexVec);

            // ============================================================
            // Step 5: 使用 vector.reduction minu candidate 得到最小 index
            // ReductionOp 构造函数：
            //   destTy, CombiningKindAttr, vector, acc?, fastmath?
            // ============================================================

            // 构造 CombiningKindAttr(MINUI)
            auto kindAttr = vector::CombiningKindAttr::get(
                rewriter.getContext(),
                vector::CombiningKind::MINUI // unsigned min
            );

            // reduction result 类型为 i64
            auto i64Ty = rewriter.getI64Type();

            Value reduced = rewriter.create<vector::ReductionOp>(
                loc,
                i64Ty,                     // result scalar type
                kindAttr,                  // combining kind
                candidate,                 // vector operand
                Value(),                   // no accumulator
                arith::FastMathFlagsAttr() // no fastmath
            );

            // ============================================================
            // Step 6: Real varint decode using vector ops
            // ============================================================

            // 6.1 常量 0x7f 用于去掉 continuation bit
            Value const7F = rewriter.create<LLVM::ConstantOp>(
                loc, rewriter.getI8Type(), rewriter.getIntegerAttr(rewriter.getI8Type(), 0x7f));
            auto splat7F = rewriter.create<vector::BroadcastOp>(
                loc, vecTy, const7F);

            // vec_low = vec & 0x7f
            Value vecLow = rewriter.create<LLVM::AndOp>(loc, vec, splat7F);

            // 6.2 zero extend to i64
            Value vecLow64 = rewriter.create<LLVM::ZExtOp>(loc, vec64Ty, vecLow);

            // 6.3 构造 shift offset [0,7,14,21,...,105]
            SmallVector<Attribute> shiftAttrs;
            for (int i = 0; i < 16; i++)
                shiftAttrs.push_back(
                    rewriter.getI64IntegerAttr(i * 7));

            Value shiftVec = rewriter.create<LLVM::ConstantOp>(
                loc, vec64Ty,
                DenseElementsAttr::get(vec64Ty, shiftAttrs));

            // 6.4 左移 (vecLow64 << shiftVec)
            Value shifted = rewriter.create<LLVM::ShlOp>(
                loc, vecLow64, shiftVec);

            // 6.5 mask 掉超过 first_zero_pos 的 lane：
            //    keep = (indexVec <= first_zero_pos ? 1 : 0)
            auto reducedVec = rewriter.create<vector::BroadcastOp>(
                loc, vec64Ty, reduced);
            auto cmpIndex = rewriter.create<LLVM::ICmpOp>(
                loc, LLVM::ICmpPredicate::ule, indexVec, reducedVec);

            // keep 作为选择器（i1 vector）
            // select(keep, shifted, 0)
            auto zeroI64 = rewriter.create<LLVM::ConstantOp>(
                loc, rewriter.getI64Type(), rewriter.getIntegerAttr(rewriter.getI64Type(), 0));
            auto zeroVec64 = rewriter.create<vector::BroadcastOp>(
                loc, vec64Ty, zeroI64);

            Value maskedShifted =
                rewriter.create<LLVM::SelectOp>(loc, cmpIndex, shifted, zeroVec64);

            // 6.6 vector.reduction add 得到 varint 值
            auto addKind = vector::CombiningKindAttr::get(
                rewriter.getContext(), vector::CombiningKind::ADD);

            Value varint =
                rewriter.create<vector::ReductionOp>(
                    loc,
                    rewriter.getI64Type(), // scalar result
                    addKind,               // ADD
                    maskedShifted,
                    Value(),               // no accumulator
                    arith::FastMathFlagsAttr());

            // 6.7 计算 newPtr = ptr + first_zero_pos + 1
            Value one64 = rewriter.create<LLVM::ConstantOp>(
                loc, rewriter.getI64Type(), rewriter.getIntegerAttr(rewriter.getI64Type(), 1));
            Value newPtrOff = rewriter.create<LLVM::AddOp>(loc, reduced, one64);

            SmallVector<LLVM::GEPArg> gepArgs{newPtrOff};
            Value newPtr = rewriter.create<LLVM::GEPOp>(
                loc, ptr.getType(), ptr.getType(), ptr, gepArgs);

            // 最终替换 op
            rewriter.replaceOp(op, {varint, newPtr});
            return success();

        }
    };

    /// --------------------------------------------------------------------------
    /// Pass
    /// --------------------------------------------------------------------------
    struct LowerProtoAccToVectorPass
        : protoacc::impl::LowerProtoAccToVectorBase<LowerProtoAccToVectorPass>
    {

        void getDependentDialects(DialectRegistry &registry) const override
        {
            registry.insert<mlir::arith::ArithDialect>();
            registry.insert<mlir::vector::VectorDialect>();
            registry.insert<LLVM::LLVMDialect>();
            registry.insert<protoacc::ProtoAccDialect>();
        }

        void runOnOperation() override
        {
            MLIRContext *ctx = &getContext();
            RewritePatternSet patterns(ctx);

            patterns.add<DecodeVarintOpLowering>(ctx);

            if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                                    std::move(patterns))))
                signalPassFailure();
        }
    };

} // namespace

/// Pass factory
std::unique_ptr<mlir::Pass> protoacc::createLowerProtoAccToVectorPass()
{
    return std::make_unique<LowerProtoAccToVectorPass>();
}
