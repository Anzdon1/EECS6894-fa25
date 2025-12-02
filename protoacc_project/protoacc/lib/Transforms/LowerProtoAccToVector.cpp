#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
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
                LLVM::LLVMPointerType::get(IntegerType::get(rewriter.getContext(), 8));
            LLVM::LLVMPointerType i64PtrTy =
                LLVM::LLVMPointerType::get(IntegerType::get(rewriter.getContext(), 64));

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
            Value const80 = rewriter.create<arith::ConstantIntOp>(loc, 0x80, 8);
            auto splat80 = rewriter.create<vector::BroadcastOp>(loc, vecTy, const80);

            Value masked = rewriter.create<arith::AndIOp>(loc, vec, splat80);

            // ============================================================
            // Step 4: 查找第一个 masked == 0 的位置
            // 我们使用 (mask != 0 ? huge : index)
            // 然后取最小值作为 first_zero_pos
            // ============================================================
            auto hugeVal = rewriter.create<arith::ConstantIntOp>(loc, 999, 64);

            // 构造 index = vector<16xi64> = [0..15]
            SmallVector<Attribute> indexAttrs;
            for (int i = 0; i < 16; i++)
                indexAttrs.push_back(rewriter.getI64IntegerAttr(i));

            auto indexVec = rewriter.create<arith::ConstantOp>(
                loc,
                VectorType::get({16}, rewriter.getI64Type()),
                DenseElementsAttr::get(
                    VectorType::get({16}, rewriter.getI64Type()),
                    indexAttrs));

            // 将 masked : vector<16xi8> 转为 vector<16xi64>
            // extend to i64
            auto mask64 = rewriter.create<arith::ExtUIOp>(
                loc,
                VectorType::get({16}, rewriter.getI64Type()),
                masked);

            // cmp = (mask64 != 0)
            Value zero64 = rewriter.create<arith::ConstantIntOp>(loc, 0, 64);
            auto zeroVec = rewriter.create<vector::BroadcastOp>(
                loc, VectorType::get({16}, rewriter.getI64Type()), zero64);

            auto cmp = rewriter.create<arith::CmpIOp>(
                loc,
                arith::CmpIPredicate::ne,
                mask64,
                zeroVec);

            // candidate = select(cmp, hugeVal, index)
            auto hugeVec = rewriter.create<vector::BroadcastOp>(
                loc, VectorType::get({16}, rewriter.getI64Type()), hugeVal);

            Value candidate =
                rewriter.create<arith::SelectOp>(loc, cmp, hugeVec, indexVec);

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
            // Step 6: reconstruct varint using mask, vec
            // 这里只做结构演示：真实构造需继续拼 varint 逻辑
            // ============================================================

            // ptr + (reduced + 1)
            Value one = rewriter.create<arith::ConstantIntOp>(loc, 1, 64);
            Value newPtrOffset = rewriter.create<arith::AddIOp>(loc, reduced, one);
            Value newPtr = rewriter.create<LLVM::GEPOp>(
                loc, ptr.getType(), ptr, newPtrOffset);

            // 伪造 varint = reduced（只是 placeholder）
            Value varint = reduced;

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