; ModuleID = 'kernels/varint_decode.c'
source_filename = "kernels/varint_decode.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-linux-gnu"

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite) uwtable
define dso_local noundef i32 @varint_decode(ptr nocapture noundef readonly %0, ptr nocapture noundef writeonly %1) local_unnamed_addr #0 {
  %3 = load i8, ptr %0, align 1, !tbaa !5
  %4 = and i8 %3, 127
  %5 = zext nneg i8 %4 to i32
  %6 = icmp slt i8 %3, 0
  br i1 %6, label %10, label %7

7:                                                ; preds = %34, %26, %18, %10, %2
  %8 = phi i32 [ 1, %2 ], [ 2, %10 ], [ 3, %18 ], [ 4, %26 ], [ 5, %34 ]
  %9 = phi i32 [ %5, %2 ], [ %16, %10 ], [ %24, %18 ], [ %32, %26 ], [ %39, %34 ]
  store i32 %9, ptr %1, align 4, !tbaa !8
  br label %41

10:                                               ; preds = %2
  %11 = getelementptr inbounds i8, ptr %0, i64 1
  %12 = load i8, ptr %11, align 1, !tbaa !5
  %13 = and i8 %12, 127
  %14 = zext nneg i8 %13 to i32
  %15 = shl nuw nsw i32 %14, 7
  %16 = or disjoint i32 %15, %5
  %17 = icmp slt i8 %12, 0
  br i1 %17, label %18, label %7

18:                                               ; preds = %10
  %19 = getelementptr inbounds i8, ptr %0, i64 2
  %20 = load i8, ptr %19, align 1, !tbaa !5
  %21 = and i8 %20, 127
  %22 = zext nneg i8 %21 to i32
  %23 = shl nuw nsw i32 %22, 14
  %24 = or disjoint i32 %23, %16
  %25 = icmp slt i8 %20, 0
  br i1 %25, label %26, label %7

26:                                               ; preds = %18
  %27 = getelementptr inbounds i8, ptr %0, i64 3
  %28 = load i8, ptr %27, align 1, !tbaa !5
  %29 = and i8 %28, 127
  %30 = zext nneg i8 %29 to i32
  %31 = shl nuw nsw i32 %30, 21
  %32 = or disjoint i32 %31, %24
  %33 = icmp slt i8 %28, 0
  br i1 %33, label %34, label %7

34:                                               ; preds = %26
  %35 = getelementptr inbounds i8, ptr %0, i64 4
  %36 = load i8, ptr %35, align 1, !tbaa !5
  %37 = zext i8 %36 to i32
  %38 = shl i32 %37, 28
  %39 = or disjoint i32 %38, %32
  %40 = icmp slt i8 %36, 0
  br i1 %40, label %41, label %7

41:                                               ; preds = %34, %7
  %42 = phi i32 [ %8, %7 ], [ -1, %34 ]
  ret i32 %42
}

attributes #0 = { mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite) uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }

!llvm.module.flags = !{!0, !1, !2, !3}
!llvm.ident = !{!4}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 8, !"PIC Level", i32 2}
!2 = !{i32 7, !"PIE Level", i32 2}
!3 = !{i32 7, !"uwtable", i32 2}
!4 = !{!"Ubuntu clang version 18.1.8 (++20240731024944+3b5b5c1ec4a3-1~exp1~20240731145000.144)"}
!5 = !{!6, !6, i64 0}
!6 = !{!"omnipotent char", !7, i64 0}
!7 = !{!"Simple C/C++ TBAA"}
!8 = !{!9, !9, i64 0}
!9 = !{!"int", !6, i64 0}
