#tbaa_root = #llvm.tbaa_root<id = "Simple C/C++ TBAA">
#tbaa_type_desc = #llvm.tbaa_type_desc<id = "omnipotent char", members = {<#tbaa_root, 0>}>
#tbaa_tag = #llvm.tbaa_tag<base_type = #tbaa_type_desc, access_type = #tbaa_type_desc, offset = 0>
#tbaa_type_desc1 = #llvm.tbaa_type_desc<id = "int", members = {<#tbaa_type_desc, 0>}>
#tbaa_tag1 = #llvm.tbaa_tag<base_type = #tbaa_type_desc1, access_type = #tbaa_type_desc1, offset = 0>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f16, dense<16> : vector<2xi64>>, #dlti.dl_entry<f64, dense<64> : vector<2xi64>>, #dlti.dl_entry<f128, dense<128> : vector<2xi64>>, #dlti.dl_entry<i128, dense<128> : vector<2xi64>>, #dlti.dl_entry<f80, dense<128> : vector<2xi64>>, #dlti.dl_entry<i64, dense<64> : vector<2xi64>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi64>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi64>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi64>>, #dlti.dl_entry<i1, dense<8> : vector<2xi64>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi64>>, #dlti.dl_entry<i8, dense<8> : vector<2xi64>>, #dlti.dl_entry<i16, dense<16> : vector<2xi64>>, #dlti.dl_entry<i32, dense<32> : vector<2xi64>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i64>, #dlti.dl_entry<"dlti.endianness", "little">>} {
  llvm.func local_unnamed_addr @varint_decode(%arg0: !llvm.ptr {llvm.nocapture, llvm.noundef, llvm.readonly}, %arg1: !llvm.ptr {llvm.nocapture, llvm.noundef, llvm.writeonly}) -> (i32 {llvm.noundef}) attributes {memory = #llvm.memory_effects<other = none, argMem = readwrite, inaccessibleMem = none>, passthrough = ["mustprogress", "nofree", "norecurse", "nosync", "nounwind", "willreturn", ["uwtable", "2"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], ["tune-cpu", "generic"]], target_cpu = "x86-64", target_features = #llvm.target_features<["+cmov", "+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>} {
    %0 = llvm.mlir.constant(127 : i8) : i8
    %1 = llvm.mlir.constant(0 : i8) : i8
    %2 = llvm.mlir.constant(1 : i32) : i32
    %3 = llvm.mlir.constant(1 : i64) : i64
    %4 = llvm.mlir.constant(7 : i32) : i32
    %5 = llvm.mlir.constant(2 : i32) : i32
    %6 = llvm.mlir.constant(2 : i64) : i64
    %7 = llvm.mlir.constant(14 : i32) : i32
    %8 = llvm.mlir.constant(3 : i32) : i32
    %9 = llvm.mlir.constant(3 : i64) : i64
    %10 = llvm.mlir.constant(21 : i32) : i32
    %11 = llvm.mlir.constant(4 : i32) : i32
    %12 = llvm.mlir.constant(4 : i64) : i64
    %13 = llvm.mlir.constant(28 : i32) : i32
    %14 = llvm.mlir.constant(-1 : i32) : i32
    %15 = llvm.mlir.constant(5 : i32) : i32
    %16 = llvm.load %arg0 {alignment = 1 : i64, tbaa = [#tbaa_tag]} : !llvm.ptr -> i8
    %17 = llvm.and %16, %0  : i8
    %18 = llvm.zext %17 : i8 to i32
    %19 = llvm.icmp "slt" %16, %1 : i8
    llvm.cond_br %19, ^bb2, ^bb1(%2, %18 : i32, i32)
  ^bb1(%20: i32, %21: i32):  // 5 preds: ^bb0, ^bb2, ^bb3, ^bb4, ^bb5
    llvm.store %21, %arg1 {alignment = 4 : i64, tbaa = [#tbaa_tag1]} : i32, !llvm.ptr
    llvm.br ^bb6(%20 : i32)
  ^bb2:  // pred: ^bb0
    %22 = llvm.getelementptr inbounds %arg0[%3] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %23 = llvm.load %22 {alignment = 1 : i64, tbaa = [#tbaa_tag]} : !llvm.ptr -> i8
    %24 = llvm.and %23, %0  : i8
    %25 = llvm.zext %24 : i8 to i32
    %26 = llvm.shl %25, %4 overflow<nsw, nuw>  : i32
    %27 = llvm.or %26, %18  : i32
    %28 = llvm.icmp "slt" %23, %1 : i8
    llvm.cond_br %28, ^bb3, ^bb1(%5, %27 : i32, i32)
  ^bb3:  // pred: ^bb2
    %29 = llvm.getelementptr inbounds %arg0[%6] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %30 = llvm.load %29 {alignment = 1 : i64, tbaa = [#tbaa_tag]} : !llvm.ptr -> i8
    %31 = llvm.and %30, %0  : i8
    %32 = llvm.zext %31 : i8 to i32
    %33 = llvm.shl %32, %7 overflow<nsw, nuw>  : i32
    %34 = llvm.or %33, %27  : i32
    %35 = llvm.icmp "slt" %30, %1 : i8
    llvm.cond_br %35, ^bb4, ^bb1(%8, %34 : i32, i32)
  ^bb4:  // pred: ^bb3
    %36 = llvm.getelementptr inbounds %arg0[%9] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %37 = llvm.load %36 {alignment = 1 : i64, tbaa = [#tbaa_tag]} : !llvm.ptr -> i8
    %38 = llvm.and %37, %0  : i8
    %39 = llvm.zext %38 : i8 to i32
    %40 = llvm.shl %39, %10 overflow<nsw, nuw>  : i32
    %41 = llvm.or %40, %34  : i32
    %42 = llvm.icmp "slt" %37, %1 : i8
    llvm.cond_br %42, ^bb5, ^bb1(%11, %41 : i32, i32)
  ^bb5:  // pred: ^bb4
    %43 = llvm.getelementptr inbounds %arg0[%12] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %44 = llvm.load %43 {alignment = 1 : i64, tbaa = [#tbaa_tag]} : !llvm.ptr -> i8
    %45 = llvm.zext %44 : i8 to i32
    %46 = llvm.shl %45, %13  : i32
    %47 = llvm.or %46, %41  : i32
    %48 = llvm.icmp "slt" %44, %1 : i8
    llvm.cond_br %48, ^bb6(%14 : i32), ^bb1(%15, %47 : i32, i32)
  ^bb6(%49: i32):  // 2 preds: ^bb1, ^bb5
    llvm.return %49 : i32
  }
}
