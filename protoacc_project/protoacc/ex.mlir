module {
  llvm.func @test(%ptr : !llvm.ptr<i8>, %shift : i32, %mask : i32) -> i32 {
    %0 = llvm.load %ptr : !llvm.ptr<i8>
    %1 = llvm.zext %0 : i8 to i32
    %2 = llvm.lshr %1, %shift : i32
    %3 = llvm.and %2, %mask : i32
    llvm.return %3 : i32
  }
}
