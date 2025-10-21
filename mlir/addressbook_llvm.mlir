#loop_annotation = #llvm.loop_annotation<mustProgress = true>
#tbaa_root = #llvm.tbaa_root<id = "Simple C++ TBAA">
#tbaa_root1 = #llvm.tbaa_root<id = "_ZTSN6google8protobuf16RepeatedPtrFieldIN8tutorial18Person_PhoneNumberEEE">
#tbaa_root2 = #llvm.tbaa_root<id = "_ZTSSt6atomicIiE">
#tbaa_root3 = #llvm.tbaa_root<id = "_ZTSSt6atomicImE">
#tbaa_type_desc = #llvm.tbaa_type_desc<id = "omnipotent char", members = {<#tbaa_root, 0>}>
#tbaa_type_desc1 = #llvm.tbaa_type_desc<id = "vtable pointer", members = {<#tbaa_root, 0>}>
#tbaa_type_desc2 = #llvm.tbaa_type_desc<id = "_ZTSN6google8protobuf8internal10CachedSizeE", members = {<#tbaa_root2, 0>}>
#tbaa_tag = #llvm.tbaa_tag<base_type = #tbaa_type_desc1, access_type = #tbaa_type_desc1, offset = 0>
#tbaa_tag1 = #llvm.tbaa_tag<base_type = #tbaa_type_desc, access_type = #tbaa_type_desc, offset = 0>
#tbaa_type_desc3 = #llvm.tbaa_type_desc<id = "any pointer", members = {<#tbaa_type_desc, 0>}>
#tbaa_type_desc4 = #llvm.tbaa_type_desc<id = "int", members = {<#tbaa_type_desc, 0>}>
#tbaa_type_desc5 = #llvm.tbaa_type_desc<id = "long", members = {<#tbaa_type_desc, 0>}>
#tbaa_type_desc6 = #llvm.tbaa_type_desc<id = "bool", members = {<#tbaa_type_desc, 0>}>
#tbaa_tag2 = #llvm.tbaa_tag<base_type = #tbaa_type_desc3, access_type = #tbaa_type_desc3, offset = 0>
#tbaa_tag3 = #llvm.tbaa_tag<base_type = #tbaa_type_desc4, access_type = #tbaa_type_desc4, offset = 0>
#tbaa_tag4 = #llvm.tbaa_tag<base_type = #tbaa_type_desc5, access_type = #tbaa_type_desc5, offset = 0>
#tbaa_type_desc7 = #llvm.tbaa_type_desc<id = "_ZTSN6google8protobuf8internal16InternalMetadataE", members = {<#tbaa_type_desc3, 0>}>
#tbaa_type_desc8 = #llvm.tbaa_type_desc<id = "_ZTSSt13__atomic_baseIiE", members = {<#tbaa_type_desc4, 0>}>
#tbaa_type_desc9 = #llvm.tbaa_type_desc<id = "_ZTSN6google8protobuf8internal20RepeatedPtrFieldBaseE", members = {<#tbaa_type_desc3, 0>, <#tbaa_type_desc4, 8>, <#tbaa_type_desc4, 12>, <#tbaa_type_desc3, 16>}>
#tbaa_type_desc10 = #llvm.tbaa_type_desc<id = "_ZTSN6google8protobuf8internal14ArenaStringPtrE", members = {<#tbaa_type_desc3, 0>}>
#tbaa_type_desc11 = #llvm.tbaa_type_desc<id = "_ZTSNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE12_Alloc_hiderE", members = {<#tbaa_type_desc3, 0>}>
#tbaa_type_desc12 = #llvm.tbaa_type_desc<id = "_ZTSN6google8protobuf8internal16InternalMetadata13ContainerBaseE", members = {<#tbaa_type_desc3, 0>}>
#tbaa_type_desc13 = #llvm.tbaa_type_desc<id = "_ZTSNSt12_Vector_baseIN6google8protobuf12UnknownFieldESaIS2_EE17_Vector_impl_dataE", members = {<#tbaa_type_desc3, 0>, <#tbaa_type_desc3, 8>, <#tbaa_type_desc3, 16>}>
#tbaa_type_desc14 = #llvm.tbaa_type_desc<id = "_ZTSN6google8protobuf8internal12ParseContext4DataE", members = {<#tbaa_type_desc3, 0>, <#tbaa_type_desc3, 8>}>
#tbaa_type_desc15 = #llvm.tbaa_type_desc<id = "_ZTSN6google8protobuf8internal18EpsCopyInputStreamE", members = {<#tbaa_type_desc3, 0>, <#tbaa_type_desc3, 8>, <#tbaa_type_desc3, 16>, <#tbaa_type_desc4, 24>, <#tbaa_type_desc4, 28>, <#tbaa_type_desc3, 32>, <#tbaa_type_desc, 40>, <#tbaa_type_desc5, 72>, <#tbaa_type_desc4, 80>, <#tbaa_type_desc4, 84>}>
#tbaa_type_desc16 = #llvm.tbaa_type_desc<id = "_ZTSN6google8protobuf2io19EpsCopyOutputStreamE", members = {<#tbaa_type_desc3, 0>, <#tbaa_type_desc3, 8>, <#tbaa_type_desc, 16>, <#tbaa_type_desc3, 48>, <#tbaa_type_desc6, 56>, <#tbaa_type_desc6, 57>, <#tbaa_type_desc6, 58>}>
#tbaa_type_desc17 = #llvm.tbaa_type_desc<id = "_ZTSN6google8protobuf8internal15DescriptorTableE", members = {<#tbaa_type_desc6, 0>, <#tbaa_type_desc6, 1>, <#tbaa_type_desc3, 8>, <#tbaa_type_desc3, 16>, <#tbaa_type_desc4, 24>, <#tbaa_type_desc3, 32>, <#tbaa_type_desc3, 40>, <#tbaa_type_desc3, 48>, <#tbaa_type_desc4, 56>, <#tbaa_type_desc4, 60>, <#tbaa_type_desc3, 64>, <#tbaa_type_desc3, 72>, <#tbaa_type_desc3, 80>, <#tbaa_type_desc3, 88>, <#tbaa_type_desc4, 96>, <#tbaa_type_desc3, 104>, <#tbaa_type_desc3, 112>}>
#tbaa_type_desc18 = #llvm.tbaa_type_desc<id = "_ZTSN6google8protobuf8internal20RepeatedPtrFieldBase3RepE", members = {<#tbaa_type_desc4, 0>, <#tbaa_type_desc, 8>}>
#tbaa_type_desc19 = #llvm.tbaa_type_desc<id = "_ZTSSt13__atomic_baseIPN6google8protobuf8internal9ArenaImpl11SerialArenaEE", members = {<#tbaa_type_desc3, 0>}>
#tbaa_type_desc20 = #llvm.tbaa_type_desc<id = "_ZTSN6google8protobuf8internal9ArenaImpl7OptionsE", members = {<#tbaa_type_desc5, 0>, <#tbaa_type_desc5, 8>, <#tbaa_type_desc3, 16>, <#tbaa_type_desc5, 24>, <#tbaa_type_desc3, 32>, <#tbaa_type_desc3, 40>}>
#tbaa_tag5 = #llvm.tbaa_tag<base_type = #tbaa_type_desc7, access_type = #tbaa_type_desc3, offset = 0>
#tbaa_tag6 = #llvm.tbaa_tag<base_type = #tbaa_type_desc8, access_type = #tbaa_type_desc4, offset = 0>
#tbaa_tag7 = #llvm.tbaa_tag<base_type = #tbaa_type_desc9, access_type = #tbaa_type_desc3, offset = 0>
#tbaa_tag8 = #llvm.tbaa_tag<base_type = #tbaa_type_desc10, access_type = #tbaa_type_desc3, offset = 0>
#tbaa_tag9 = #llvm.tbaa_tag<base_type = #tbaa_type_desc12, access_type = #tbaa_type_desc3, offset = 0>
#tbaa_tag10 = #llvm.tbaa_tag<base_type = #tbaa_type_desc13, access_type = #tbaa_type_desc3, offset = 0>
#tbaa_tag11 = #llvm.tbaa_tag<base_type = #tbaa_type_desc15, access_type = #tbaa_type_desc4, offset = 80>
#tbaa_tag12 = #llvm.tbaa_tag<base_type = #tbaa_type_desc16, access_type = #tbaa_type_desc3, offset = 0>
#tbaa_tag13 = #llvm.tbaa_tag<base_type = #tbaa_type_desc13, access_type = #tbaa_type_desc3, offset = 16>
#tbaa_tag14 = #llvm.tbaa_tag<base_type = #tbaa_type_desc17, access_type = #tbaa_type_desc3, offset = 88>
#tbaa_tag15 = #llvm.tbaa_tag<base_type = #tbaa_type_desc9, access_type = #tbaa_type_desc3, offset = 16>
#tbaa_tag16 = #llvm.tbaa_tag<base_type = #tbaa_type_desc9, access_type = #tbaa_type_desc4, offset = 12>
#tbaa_tag17 = #llvm.tbaa_tag<base_type = #tbaa_type_desc9, access_type = #tbaa_type_desc4, offset = 8>
#tbaa_tag18 = #llvm.tbaa_tag<base_type = #tbaa_type_desc18, access_type = #tbaa_type_desc4, offset = 0>
#tbaa_tag19 = #llvm.tbaa_tag<base_type = #tbaa_type_desc15, access_type = #tbaa_type_desc4, offset = 28>
#tbaa_tag20 = #llvm.tbaa_tag<base_type = #tbaa_type_desc15, access_type = #tbaa_type_desc3, offset = 8>
#tbaa_tag21 = #llvm.tbaa_tag<base_type = #tbaa_type_desc15, access_type = #tbaa_type_desc3, offset = 0>
#tbaa_tag22 = #llvm.tbaa_tag<base_type = #tbaa_type_desc11, access_type = #tbaa_type_desc3, offset = 0>
#tbaa_type_desc21 = #llvm.tbaa_type_desc<id = "_ZTSN8tutorial6PersonE", members = {<#tbaa_root1, 16>, <#tbaa_type_desc10, 40>, <#tbaa_type_desc10, 48>, <#tbaa_type_desc3, 56>, <#tbaa_type_desc4, 64>, <#tbaa_type_desc2, 68>}>
#tbaa_type_desc22 = #llvm.tbaa_type_desc<id = "_ZTSN8tutorial18Person_PhoneNumberE", members = {<#tbaa_type_desc10, 16>, <#tbaa_type_desc4, 24>, <#tbaa_type_desc2, 28>}>
#tbaa_type_desc23 = #llvm.tbaa_type_desc<id = "_ZTSNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE", members = {<#tbaa_type_desc11, 0>, <#tbaa_type_desc5, 8>, <#tbaa_type_desc, 16>}>
#tbaa_type_desc24 = #llvm.tbaa_type_desc<id = "_ZTSN6google8protobuf8internal12ParseContextE", members = {<#tbaa_type_desc4, 88>, <#tbaa_type_desc4, 92>, <#tbaa_type_desc14, 96>}>
#tbaa_type_desc25 = #llvm.tbaa_type_desc<id = "_ZTSSt6atomicIPN6google8protobuf8internal9ArenaImpl11SerialArenaEE", members = {<#tbaa_type_desc19, 0>}>
#tbaa_tag23 = #llvm.tbaa_tag<base_type = #tbaa_type_desc21, access_type = #tbaa_type_desc3, offset = 56>
#tbaa_tag24 = #llvm.tbaa_tag<base_type = #tbaa_type_desc22, access_type = #tbaa_type_desc4, offset = 24>
#tbaa_tag25 = #llvm.tbaa_tag<base_type = #tbaa_type_desc23, access_type = #tbaa_type_desc5, offset = 8>
#tbaa_tag26 = #llvm.tbaa_tag<base_type = #tbaa_type_desc23, access_type = #tbaa_type_desc3, offset = 0>
#tbaa_tag27 = #llvm.tbaa_tag<base_type = #tbaa_type_desc24, access_type = #tbaa_type_desc4, offset = 92>
#tbaa_tag28 = #llvm.tbaa_tag<base_type = #tbaa_type_desc21, access_type = #tbaa_type_desc4, offset = 64>
#tbaa_tag29 = #llvm.tbaa_tag<base_type = #tbaa_type_desc24, access_type = #tbaa_type_desc4, offset = 88>
#tbaa_type_desc26 = #llvm.tbaa_type_desc<id = "_ZTSN6google8protobuf8internal9ArenaImplE", members = {<#tbaa_type_desc25, 0>, <#tbaa_type_desc25, 8>, <#tbaa_root3, 16>, <#tbaa_type_desc3, 24>, <#tbaa_type_desc5, 32>, <#tbaa_type_desc20, 40>}>
#tbaa_type_desc27 = #llvm.tbaa_type_desc<id = "_ZTSN6google8protobuf5ArenaE", members = {<#tbaa_type_desc26, 0>, <#tbaa_type_desc3, 88>, <#tbaa_type_desc3, 96>, <#tbaa_type_desc3, 104>, <#tbaa_type_desc3, 112>}>
#tbaa_tag30 = #llvm.tbaa_tag<base_type = #tbaa_type_desc27, access_type = #tbaa_type_desc3, offset = 112>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i32, dense<32> : vector<2xi64>>, #dlti.dl_entry<f128, dense<128> : vector<2xi64>>, #dlti.dl_entry<f64, dense<64> : vector<2xi64>>, #dlti.dl_entry<f16, dense<16> : vector<2xi64>>, #dlti.dl_entry<i1, dense<8> : vector<2xi64>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi64>>, #dlti.dl_entry<i16, dense<16> : vector<2xi64>>, #dlti.dl_entry<i8, dense<8> : vector<2xi64>>, #dlti.dl_entry<f80, dense<128> : vector<2xi64>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi64>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi64>>, #dlti.dl_entry<i64, dense<64> : vector<2xi64>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi64>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i64>>} {
  llvm.comdat @__llvm_global_comdat {
    llvm.comdat_selector @_ZTSNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE any
    llvm.comdat_selector @_ZTINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE any
    llvm.comdat_selector @_ZTSN6google8protobuf8internal16InternalMetadata9ContainerINS0_15UnknownFieldSetEEE any
    llvm.comdat_selector @_ZTSN6google8protobuf8internal16InternalMetadata13ContainerBaseE any
    llvm.comdat_selector @_ZTIN6google8protobuf8internal16InternalMetadata13ContainerBaseE any
    llvm.comdat_selector @_ZTIN6google8protobuf8internal16InternalMetadata9ContainerINS0_15UnknownFieldSetEEE any
    llvm.comdat_selector @_ZN8tutorial18Person_PhoneNumber10SharedDtorEv any
    llvm.comdat_selector @_ZN6google8protobuf8internal16InternalMetadata6DeleteINS0_15UnknownFieldSetEEEvv any
    llvm.comdat_selector @__clang_call_terminate any
    llvm.comdat_selector @_ZN6google8protobuf16RepeatedPtrFieldIN8tutorial18Person_PhoneNumberEED2Ev any
    llvm.comdat_selector @_ZN8tutorial6Person10SharedDtorEv any
    llvm.comdat_selector @_ZN6google8protobuf16RepeatedPtrFieldIN8tutorial6PersonEED2Ev any
    llvm.comdat_selector @_ZNK8tutorial18Person_PhoneNumber3NewEv any
    llvm.comdat_selector @_ZNK8tutorial18Person_PhoneNumber3NewEPN6google8protobuf5ArenaE any
    llvm.comdat_selector @_ZNK8tutorial18Person_PhoneNumber13GetCachedSizeEv any
    llvm.comdat_selector @_ZNK6google8protobuf11MessageLite16InternalGetTableEv any
    llvm.comdat_selector @_ZNK8tutorial6Person3NewEv any
    llvm.comdat_selector @_ZNK8tutorial6Person3NewEPN6google8protobuf5ArenaE any
    llvm.comdat_selector @_ZNK8tutorial6Person13GetCachedSizeEv any
    llvm.comdat_selector @_ZNK8tutorial11AddressBook3NewEv any
    llvm.comdat_selector @_ZNK8tutorial11AddressBook3NewEPN6google8protobuf5ArenaE any
    llvm.comdat_selector @_ZNK8tutorial11AddressBook13GetCachedSizeEv any
    llvm.comdat_selector @_ZN6google8protobuf8internal14ArenaStringPtr14CreateInstanceEPNS0_5ArenaEPKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE any
    llvm.comdat_selector @_ZN6google8protobuf8internal21arena_destruct_objectINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEEEvPv any
    llvm.comdat_selector @_ZN6google8protobuf8internal18EpsCopyInputStream13DoneWithCheckEPPKci any
    llvm.comdat_selector @_ZNK6google8protobuf8internal20RepeatedPtrFieldBase3GetINS0_16RepeatedPtrFieldIN8tutorial18Person_PhoneNumberEE11TypeHandlerEEERKNT_4TypeEi any
    llvm.comdat_selector @_ZNK6google8protobuf8internal20RepeatedPtrFieldBase3GetINS0_16RepeatedPtrFieldIN8tutorial6PersonEE11TypeHandlerEEERKNT_4TypeEi any
    llvm.comdat_selector @_ZN6google8protobuf8internal20RepeatedPtrFieldBase5ClearINS0_16RepeatedPtrFieldIN8tutorial18Person_PhoneNumberEE11TypeHandlerEEEvv any
    llvm.comdat_selector @_ZN6google8protobuf8internal20RepeatedPtrFieldBase5ClearINS0_16RepeatedPtrFieldIN8tutorial6PersonEE11TypeHandlerEEEvv any
    llvm.comdat_selector @_ZNSt6vectorIN6google8protobuf12UnknownFieldESaIS2_EED2Ev any
    llvm.comdat_selector @_ZN6google8protobuf8internal16InternalMetadata27mutable_unknown_fields_slowINS0_15UnknownFieldSetEEEPT_v any
    llvm.comdat_selector @_ZN6google8protobuf8internal21arena_destruct_objectINS1_16InternalMetadata9ContainerINS0_15UnknownFieldSetEEEEEvPv any
    llvm.comdat_selector @_ZN6google8protobuf8internal20RepeatedPtrFieldBase7DestroyINS0_16RepeatedPtrFieldIN8tutorial18Person_PhoneNumberEE11TypeHandlerEEEvv any
    llvm.comdat_selector @_ZN6google8protobuf8internal20RepeatedPtrFieldBaseD2Ev any
    llvm.comdat_selector @_ZN6google8protobuf8internal18EpsCopyInputStream9PushLimitEPKci any
    llvm.comdat_selector @_ZN6google8protobuf8internal20RepeatedPtrFieldBase9MergeFromINS0_16RepeatedPtrFieldIN8tutorial18Person_PhoneNumberEE11TypeHandlerEEEvRKS2_ any
    llvm.comdat_selector @_ZN6google8protobuf8internal18GenericTypeHandlerIN8tutorial18Person_PhoneNumberEE5MergeERKS4_PS4_ any
    llvm.comdat_selector @_ZN6google8protobuf8internal20RepeatedPtrFieldBase12InternalSwapEPS2_ any
    llvm.comdat_selector @_ZN6google8protobuf8internal20RepeatedPtrFieldBase9MergeFromINS0_16RepeatedPtrFieldIN8tutorial6PersonEE11TypeHandlerEEEvRKS2_ any
    llvm.comdat_selector @_ZN6google8protobuf8internal18GenericTypeHandlerIN8tutorial6PersonEE5MergeERKS4_PS4_ any
    llvm.comdat_selector @_ZN6google8protobuf5Arena14InternalHelperIN8tutorial6PersonEE9ConstructIJPS1_EEEPS4_PvDpOT_ any
  }
  llvm.mlir.global internal @_ZStL8__ioinit() {addr_space = 0 : i32, alignment = 1 : i64, dso_local} : !llvm.struct<"class.std::ios_base::Init", (i8)> {
    %0 = llvm.mlir.constant(0 : i8) : i8
    %1 = llvm.mlir.undef : !llvm.struct<"class.std::ios_base::Init", (i8)>
    %2 = llvm.insertvalue %0, %1[0] : !llvm.struct<"class.std::ios_base::Init", (i8)> 
    llvm.return %2 : !llvm.struct<"class.std::ios_base::Init", (i8)>
  }
  llvm.mlir.global external hidden @__dso_handle() {addr_space = 0 : i32, dso_local} : i8
  llvm.mlir.global external @_ZN8tutorial37_Person_PhoneNumber_default_instance_E() {addr_space = 0 : i32, alignment = 8 : i64, dso_local} : !llvm.struct<"class.tutorial::Person_PhoneNumberDefaultTypeInternal", (struct<"class.google::protobuf::internal::ExplicitlyConstructed", (struct<"union.google::protobuf::internal::ExplicitlyConstructed<tutorial::Person_PhoneNumber>::AlignedUnion", (i64, array<24 x i8>)>)>)> {
    %0 = llvm.mlir.constant(0 : i8) : i8
    %1 = llvm.mlir.constant(dense<0> : tensor<24xi8>) : !llvm.array<24 x i8>
    %2 = llvm.mlir.constant(0 : i64) : i64
    %3 = llvm.mlir.undef : !llvm.struct<"union.google::protobuf::internal::ExplicitlyConstructed<tutorial::Person_PhoneNumber>::AlignedUnion", (i64, array<24 x i8>)>
    %4 = llvm.insertvalue %2, %3[0] : !llvm.struct<"union.google::protobuf::internal::ExplicitlyConstructed<tutorial::Person_PhoneNumber>::AlignedUnion", (i64, array<24 x i8>)> 
    %5 = llvm.insertvalue %1, %4[1] : !llvm.struct<"union.google::protobuf::internal::ExplicitlyConstructed<tutorial::Person_PhoneNumber>::AlignedUnion", (i64, array<24 x i8>)> 
    %6 = llvm.mlir.undef : !llvm.struct<"class.google::protobuf::internal::ExplicitlyConstructed", (struct<"union.google::protobuf::internal::ExplicitlyConstructed<tutorial::Person_PhoneNumber>::AlignedUnion", (i64, array<24 x i8>)>)>
    %7 = llvm.insertvalue %5, %6[0] : !llvm.struct<"class.google::protobuf::internal::ExplicitlyConstructed", (struct<"union.google::protobuf::internal::ExplicitlyConstructed<tutorial::Person_PhoneNumber>::AlignedUnion", (i64, array<24 x i8>)>)> 
    %8 = llvm.mlir.undef : !llvm.struct<"class.tutorial::Person_PhoneNumberDefaultTypeInternal", (struct<"class.google::protobuf::internal::ExplicitlyConstructed", (struct<"union.google::protobuf::internal::ExplicitlyConstructed<tutorial::Person_PhoneNumber>::AlignedUnion", (i64, array<24 x i8>)>)>)>
    %9 = llvm.insertvalue %7, %8[0] : !llvm.struct<"class.tutorial::Person_PhoneNumberDefaultTypeInternal", (struct<"class.google::protobuf::internal::ExplicitlyConstructed", (struct<"union.google::protobuf::internal::ExplicitlyConstructed<tutorial::Person_PhoneNumber>::AlignedUnion", (i64, array<24 x i8>)>)>)> 
    llvm.return %9 : !llvm.struct<"class.tutorial::Person_PhoneNumberDefaultTypeInternal", (struct<"class.google::protobuf::internal::ExplicitlyConstructed", (struct<"union.google::protobuf::internal::ExplicitlyConstructed<tutorial::Person_PhoneNumber>::AlignedUnion", (i64, array<24 x i8>)>)>)>
  }
  llvm.mlir.global external @_ZN8tutorial25_Person_default_instance_E() {addr_space = 0 : i32, alignment = 8 : i64, dso_local} : !llvm.struct<"class.tutorial::PersonDefaultTypeInternal", (struct<"class.google::protobuf::internal::ExplicitlyConstructed.0", (struct<"union.google::protobuf::internal::ExplicitlyConstructed<tutorial::Person>::AlignedUnion", (i64, array<64 x i8>)>)>)> {
    %0 = llvm.mlir.constant(0 : i8) : i8
    %1 = llvm.mlir.constant(dense<0> : tensor<64xi8>) : !llvm.array<64 x i8>
    %2 = llvm.mlir.constant(0 : i64) : i64
    %3 = llvm.mlir.undef : !llvm.struct<"union.google::protobuf::internal::ExplicitlyConstructed<tutorial::Person>::AlignedUnion", (i64, array<64 x i8>)>
    %4 = llvm.insertvalue %2, %3[0] : !llvm.struct<"union.google::protobuf::internal::ExplicitlyConstructed<tutorial::Person>::AlignedUnion", (i64, array<64 x i8>)> 
    %5 = llvm.insertvalue %1, %4[1] : !llvm.struct<"union.google::protobuf::internal::ExplicitlyConstructed<tutorial::Person>::AlignedUnion", (i64, array<64 x i8>)> 
    %6 = llvm.mlir.undef : !llvm.struct<"class.google::protobuf::internal::ExplicitlyConstructed.0", (struct<"union.google::protobuf::internal::ExplicitlyConstructed<tutorial::Person>::AlignedUnion", (i64, array<64 x i8>)>)>
    %7 = llvm.insertvalue %5, %6[0] : !llvm.struct<"class.google::protobuf::internal::ExplicitlyConstructed.0", (struct<"union.google::protobuf::internal::ExplicitlyConstructed<tutorial::Person>::AlignedUnion", (i64, array<64 x i8>)>)> 
    %8 = llvm.mlir.undef : !llvm.struct<"class.tutorial::PersonDefaultTypeInternal", (struct<"class.google::protobuf::internal::ExplicitlyConstructed.0", (struct<"union.google::protobuf::internal::ExplicitlyConstructed<tutorial::Person>::AlignedUnion", (i64, array<64 x i8>)>)>)>
    %9 = llvm.insertvalue %7, %8[0] : !llvm.struct<"class.tutorial::PersonDefaultTypeInternal", (struct<"class.google::protobuf::internal::ExplicitlyConstructed.0", (struct<"union.google::protobuf::internal::ExplicitlyConstructed<tutorial::Person>::AlignedUnion", (i64, array<64 x i8>)>)>)> 
    llvm.return %9 : !llvm.struct<"class.tutorial::PersonDefaultTypeInternal", (struct<"class.google::protobuf::internal::ExplicitlyConstructed.0", (struct<"union.google::protobuf::internal::ExplicitlyConstructed<tutorial::Person>::AlignedUnion", (i64, array<64 x i8>)>)>)>
  }
  llvm.mlir.global external @_ZN8tutorial30_AddressBook_default_instance_E() {addr_space = 0 : i32, alignment = 8 : i64, dso_local} : !llvm.struct<"class.tutorial::AddressBookDefaultTypeInternal", (struct<"class.google::protobuf::internal::ExplicitlyConstructed.1", (struct<"union.google::protobuf::internal::ExplicitlyConstructed<tutorial::AddressBook>::AlignedUnion", (i64, array<40 x i8>)>)>)> {
    %0 = llvm.mlir.constant(0 : i8) : i8
    %1 = llvm.mlir.constant(dense<0> : tensor<40xi8>) : !llvm.array<40 x i8>
    %2 = llvm.mlir.constant(0 : i64) : i64
    %3 = llvm.mlir.undef : !llvm.struct<"union.google::protobuf::internal::ExplicitlyConstructed<tutorial::AddressBook>::AlignedUnion", (i64, array<40 x i8>)>
    %4 = llvm.insertvalue %2, %3[0] : !llvm.struct<"union.google::protobuf::internal::ExplicitlyConstructed<tutorial::AddressBook>::AlignedUnion", (i64, array<40 x i8>)> 
    %5 = llvm.insertvalue %1, %4[1] : !llvm.struct<"union.google::protobuf::internal::ExplicitlyConstructed<tutorial::AddressBook>::AlignedUnion", (i64, array<40 x i8>)> 
    %6 = llvm.mlir.undef : !llvm.struct<"class.google::protobuf::internal::ExplicitlyConstructed.1", (struct<"union.google::protobuf::internal::ExplicitlyConstructed<tutorial::AddressBook>::AlignedUnion", (i64, array<40 x i8>)>)>
    %7 = llvm.insertvalue %5, %6[0] : !llvm.struct<"class.google::protobuf::internal::ExplicitlyConstructed.1", (struct<"union.google::protobuf::internal::ExplicitlyConstructed<tutorial::AddressBook>::AlignedUnion", (i64, array<40 x i8>)>)> 
    %8 = llvm.mlir.undef : !llvm.struct<"class.tutorial::AddressBookDefaultTypeInternal", (struct<"class.google::protobuf::internal::ExplicitlyConstructed.1", (struct<"union.google::protobuf::internal::ExplicitlyConstructed<tutorial::AddressBook>::AlignedUnion", (i64, array<40 x i8>)>)>)>
    %9 = llvm.insertvalue %7, %8[0] : !llvm.struct<"class.tutorial::AddressBookDefaultTypeInternal", (struct<"class.google::protobuf::internal::ExplicitlyConstructed.1", (struct<"union.google::protobuf::internal::ExplicitlyConstructed<tutorial::AddressBook>::AlignedUnion", (i64, array<40 x i8>)>)>)> 
    llvm.return %9 : !llvm.struct<"class.tutorial::AddressBookDefaultTypeInternal", (struct<"class.google::protobuf::internal::ExplicitlyConstructed.1", (struct<"union.google::protobuf::internal::ExplicitlyConstructed<tutorial::AddressBook>::AlignedUnion", (i64, array<40 x i8>)>)>)>
  }
  llvm.mlir.global external @scc_info_AddressBook_addressbook_2eproto() {addr_space = 0 : i32, alignment = 8 : i64, dso_local} : !llvm.struct<(struct<(struct<(i32)>, i32, i32, ptr)>, array<1 x ptr>)> {
    %0 = llvm.mlir.addressof @scc_info_Timestamp_google_2fprotobuf_2ftimestamp_2eproto : !llvm.ptr
    %1 = llvm.mlir.undef : !llvm.array<0 x ptr>
    %2 = llvm.mlir.zero : !llvm.ptr
    %3 = llvm.mlir.addressof @__gxx_personality_v0 : !llvm.ptr
    %4 = llvm.mlir.addressof @_ZL59InitDefaultsscc_info_Person_PhoneNumber_addressbook_2eprotov : !llvm.ptr
    %5 = llvm.mlir.constant(0 : i32) : i32
    %6 = llvm.mlir.constant(-1 : i32) : i32
    %7 = llvm.mlir.undef : !llvm.struct<(i32)>
    %8 = llvm.insertvalue %6, %7[0] : !llvm.struct<(i32)> 
    %9 = llvm.mlir.undef : !llvm.struct<(struct<(i32)>, i32, i32, ptr)>
    %10 = llvm.insertvalue %8, %9[0] : !llvm.struct<(struct<(i32)>, i32, i32, ptr)> 
    %11 = llvm.insertvalue %5, %10[1] : !llvm.struct<(struct<(i32)>, i32, i32, ptr)> 
    %12 = llvm.insertvalue %5, %11[2] : !llvm.struct<(struct<(i32)>, i32, i32, ptr)> 
    %13 = llvm.insertvalue %4, %12[3] : !llvm.struct<(struct<(i32)>, i32, i32, ptr)> 
    %14 = llvm.mlir.undef : !llvm.struct<(struct<(struct<(i32)>, i32, i32, ptr)>, array<0 x ptr>)>
    %15 = llvm.insertvalue %13, %14[0] : !llvm.struct<(struct<(struct<(i32)>, i32, i32, ptr)>, array<0 x ptr>)> 
    %16 = llvm.insertvalue %1, %15[1] : !llvm.struct<(struct<(struct<(i32)>, i32, i32, ptr)>, array<0 x ptr>)> 
    %17 = llvm.mlir.addressof @scc_info_Person_PhoneNumber_addressbook_2eproto : !llvm.ptr
    %18 = llvm.mlir.undef : !llvm.array<2 x ptr>
    %19 = llvm.insertvalue %17, %18[0] : !llvm.array<2 x ptr> 
    %20 = llvm.insertvalue %0, %19[1] : !llvm.array<2 x ptr> 
    %21 = llvm.mlir.addressof @_ZL47InitDefaultsscc_info_Person_addressbook_2eprotov : !llvm.ptr
    %22 = llvm.mlir.constant(2 : i32) : i32
    %23 = llvm.mlir.undef : !llvm.struct<(struct<(i32)>, i32, i32, ptr)>
    %24 = llvm.insertvalue %8, %23[0] : !llvm.struct<(struct<(i32)>, i32, i32, ptr)> 
    %25 = llvm.insertvalue %22, %24[1] : !llvm.struct<(struct<(i32)>, i32, i32, ptr)> 
    %26 = llvm.insertvalue %5, %25[2] : !llvm.struct<(struct<(i32)>, i32, i32, ptr)> 
    %27 = llvm.insertvalue %21, %26[3] : !llvm.struct<(struct<(i32)>, i32, i32, ptr)> 
    %28 = llvm.mlir.undef : !llvm.struct<(struct<(struct<(i32)>, i32, i32, ptr)>, array<2 x ptr>)>
    %29 = llvm.insertvalue %27, %28[0] : !llvm.struct<(struct<(struct<(i32)>, i32, i32, ptr)>, array<2 x ptr>)> 
    %30 = llvm.insertvalue %20, %29[1] : !llvm.struct<(struct<(struct<(i32)>, i32, i32, ptr)>, array<2 x ptr>)> 
    %31 = llvm.mlir.addressof @scc_info_Person_addressbook_2eproto : !llvm.ptr
    %32 = llvm.mlir.undef : !llvm.array<1 x ptr>
    %33 = llvm.insertvalue %31, %32[0] : !llvm.array<1 x ptr> 
    %34 = llvm.mlir.addressof @_ZL52InitDefaultsscc_info_AddressBook_addressbook_2eprotov : !llvm.ptr
    %35 = llvm.mlir.constant(1 : i32) : i32
    %36 = llvm.mlir.undef : !llvm.struct<(struct<(i32)>, i32, i32, ptr)>
    %37 = llvm.insertvalue %8, %36[0] : !llvm.struct<(struct<(i32)>, i32, i32, ptr)> 
    %38 = llvm.insertvalue %35, %37[1] : !llvm.struct<(struct<(i32)>, i32, i32, ptr)> 
    %39 = llvm.insertvalue %5, %38[2] : !llvm.struct<(struct<(i32)>, i32, i32, ptr)> 
    %40 = llvm.insertvalue %34, %39[3] : !llvm.struct<(struct<(i32)>, i32, i32, ptr)> 
    %41 = llvm.mlir.undef : !llvm.struct<(struct<(struct<(i32)>, i32, i32, ptr)>, array<1 x ptr>)>
    %42 = llvm.insertvalue %40, %41[0] : !llvm.struct<(struct<(struct<(i32)>, i32, i32, ptr)>, array<1 x ptr>)> 
    %43 = llvm.insertvalue %33, %42[1] : !llvm.struct<(struct<(struct<(i32)>, i32, i32, ptr)>, array<1 x ptr>)> 
    llvm.return %43 : !llvm.struct<(struct<(struct<(i32)>, i32, i32, ptr)>, array<1 x ptr>)>
  }
  llvm.mlir.global external @scc_info_Timestamp_google_2fprotobuf_2ftimestamp_2eproto() {addr_space = 0 : i32, alignment = 8 : i64} : !llvm.struct<"struct.google::protobuf::internal::SCCInfo.3", (struct<"struct.google::protobuf::internal::SCCInfoBase", (struct<"struct.std::atomic", (struct<"struct.std::__atomic_base", (i32)>)>, i32, i32, ptr)>, array<0 x ptr>)>
  llvm.mlir.global external @scc_info_Person_addressbook_2eproto() {addr_space = 0 : i32, alignment = 8 : i64, dso_local} : !llvm.struct<(struct<(struct<(i32)>, i32, i32, ptr)>, array<2 x ptr>)> {
    %0 = llvm.mlir.addressof @scc_info_Timestamp_google_2fprotobuf_2ftimestamp_2eproto : !llvm.ptr
    %1 = llvm.mlir.undef : !llvm.array<0 x ptr>
    %2 = llvm.mlir.zero : !llvm.ptr
    %3 = llvm.mlir.addressof @__gxx_personality_v0 : !llvm.ptr
    %4 = llvm.mlir.addressof @_ZL59InitDefaultsscc_info_Person_PhoneNumber_addressbook_2eprotov : !llvm.ptr
    %5 = llvm.mlir.constant(0 : i32) : i32
    %6 = llvm.mlir.constant(-1 : i32) : i32
    %7 = llvm.mlir.undef : !llvm.struct<(i32)>
    %8 = llvm.insertvalue %6, %7[0] : !llvm.struct<(i32)> 
    %9 = llvm.mlir.undef : !llvm.struct<(struct<(i32)>, i32, i32, ptr)>
    %10 = llvm.insertvalue %8, %9[0] : !llvm.struct<(struct<(i32)>, i32, i32, ptr)> 
    %11 = llvm.insertvalue %5, %10[1] : !llvm.struct<(struct<(i32)>, i32, i32, ptr)> 
    %12 = llvm.insertvalue %5, %11[2] : !llvm.struct<(struct<(i32)>, i32, i32, ptr)> 
    %13 = llvm.insertvalue %4, %12[3] : !llvm.struct<(struct<(i32)>, i32, i32, ptr)> 
    %14 = llvm.mlir.undef : !llvm.struct<(struct<(struct<(i32)>, i32, i32, ptr)>, array<0 x ptr>)>
    %15 = llvm.insertvalue %13, %14[0] : !llvm.struct<(struct<(struct<(i32)>, i32, i32, ptr)>, array<0 x ptr>)> 
    %16 = llvm.insertvalue %1, %15[1] : !llvm.struct<(struct<(struct<(i32)>, i32, i32, ptr)>, array<0 x ptr>)> 
    %17 = llvm.mlir.addressof @scc_info_Person_PhoneNumber_addressbook_2eproto : !llvm.ptr
    %18 = llvm.mlir.undef : !llvm.array<2 x ptr>
    %19 = llvm.insertvalue %17, %18[0] : !llvm.array<2 x ptr> 
    %20 = llvm.insertvalue %0, %19[1] : !llvm.array<2 x ptr> 
    %21 = llvm.mlir.addressof @_ZL47InitDefaultsscc_info_Person_addressbook_2eprotov : !llvm.ptr
    %22 = llvm.mlir.constant(2 : i32) : i32
    %23 = llvm.mlir.undef : !llvm.struct<(struct<(i32)>, i32, i32, ptr)>
    %24 = llvm.insertvalue %8, %23[0] : !llvm.struct<(struct<(i32)>, i32, i32, ptr)> 
    %25 = llvm.insertvalue %22, %24[1] : !llvm.struct<(struct<(i32)>, i32, i32, ptr)> 
    %26 = llvm.insertvalue %5, %25[2] : !llvm.struct<(struct<(i32)>, i32, i32, ptr)> 
    %27 = llvm.insertvalue %21, %26[3] : !llvm.struct<(struct<(i32)>, i32, i32, ptr)> 
    %28 = llvm.mlir.undef : !llvm.struct<(struct<(struct<(i32)>, i32, i32, ptr)>, array<2 x ptr>)>
    %29 = llvm.insertvalue %27, %28[0] : !llvm.struct<(struct<(struct<(i32)>, i32, i32, ptr)>, array<2 x ptr>)> 
    %30 = llvm.insertvalue %20, %29[1] : !llvm.struct<(struct<(struct<(i32)>, i32, i32, ptr)>, array<2 x ptr>)> 
    llvm.return %30 : !llvm.struct<(struct<(struct<(i32)>, i32, i32, ptr)>, array<2 x ptr>)>
  }
  llvm.mlir.global external @scc_info_Person_PhoneNumber_addressbook_2eproto() {addr_space = 0 : i32, alignment = 8 : i64, dso_local} : !llvm.struct<(struct<(struct<(i32)>, i32, i32, ptr)>, array<0 x ptr>)> {
    %0 = llvm.mlir.undef : !llvm.array<0 x ptr>
    %1 = llvm.mlir.zero : !llvm.ptr
    %2 = llvm.mlir.addressof @__gxx_personality_v0 : !llvm.ptr
    %3 = llvm.mlir.addressof @_ZL59InitDefaultsscc_info_Person_PhoneNumber_addressbook_2eprotov : !llvm.ptr
    %4 = llvm.mlir.constant(0 : i32) : i32
    %5 = llvm.mlir.constant(-1 : i32) : i32
    %6 = llvm.mlir.undef : !llvm.struct<(i32)>
    %7 = llvm.insertvalue %5, %6[0] : !llvm.struct<(i32)> 
    %8 = llvm.mlir.undef : !llvm.struct<(struct<(i32)>, i32, i32, ptr)>
    %9 = llvm.insertvalue %7, %8[0] : !llvm.struct<(struct<(i32)>, i32, i32, ptr)> 
    %10 = llvm.insertvalue %4, %9[1] : !llvm.struct<(struct<(i32)>, i32, i32, ptr)> 
    %11 = llvm.insertvalue %4, %10[2] : !llvm.struct<(struct<(i32)>, i32, i32, ptr)> 
    %12 = llvm.insertvalue %3, %11[3] : !llvm.struct<(struct<(i32)>, i32, i32, ptr)> 
    %13 = llvm.mlir.undef : !llvm.struct<(struct<(struct<(i32)>, i32, i32, ptr)>, array<0 x ptr>)>
    %14 = llvm.insertvalue %12, %13[0] : !llvm.struct<(struct<(struct<(i32)>, i32, i32, ptr)>, array<0 x ptr>)> 
    %15 = llvm.insertvalue %0, %14[1] : !llvm.struct<(struct<(struct<(i32)>, i32, i32, ptr)>, array<0 x ptr>)> 
    llvm.return %15 : !llvm.struct<(struct<(struct<(i32)>, i32, i32, ptr)>, array<0 x ptr>)>
  }
  llvm.mlir.global external constant @_ZN31TableStruct_addressbook_2eproto7offsetsE(dense<[-1, 8, -1, -1, -1, 16, 24, -1, 8, -1, -1, -1, 40, 64, 48, 16, 56, -1, 8, -1, -1, -1, 16]> : tensor<23xi32>) {addr_space = 0 : i32, alignment = 16 : i64, dso_local} : !llvm.array<23 x i32>
  llvm.mlir.global internal constant @_ZL45descriptor_table_protodef_addressbook_2eproto("\0A\11addressbook.proto\12\08tutorial\1A\1Fgoogle/protobuf/timestamp.proto\22\87\02\0A\06Person\12\0C\0A\04name\18\01 \01(\09\12\0A\0A\02id\18\02 \01(\05\12\0D\0A\05email\18\03 \01(\09\12,\0A\06phones\18\04 \03(\0B2\1C.tutorial.Person.PhoneNumber\120\0A\0Clast_updated\18\05 \01(\0B2\1A.google.protobuf.Timestamp\1AG\0A\0BPhoneNumber\12\0E\0A\06number\18\01 \01(\09\12(\0A\04type\18\02 \01(\0E2\1A.tutorial.Person.PhoneType\22+\0A\09PhoneType\12\0A\0A\06MOBILE\10\00\12\08\0A\04HOME\10\01\12\08\0A\04WORK\10\02\22/\0A\0BAddressBook\12 \0A\06people\18\01 \03(\0B2\10.tutorial.PersonB\95\01\0A\1Bcom.example.tutorial.protosB\11AddressBookProtosP\01Z:github.com/protocolbuffers/protobuf/examples/go/tutorialpb\AA\02$Google.Protobuf.Examples.AddressBookb\06proto3\00") {addr_space = 0 : i32, alignment = 16 : i64, dso_local}
  llvm.mlir.global private unnamed_addr constant @".str"("addressbook.proto\00") {addr_space = 0 : i32, alignment = 1 : i64, dso_local}
  llvm.mlir.global internal @_ZL41descriptor_table_addressbook_2eproto_once() {addr_space = 0 : i32, alignment = 4 : i64, dso_local} : !llvm.struct<"struct.std::once_flag", (i32)> {
    %0 = llvm.mlir.constant(0 : i32) : i32
    %1 = llvm.mlir.undef : !llvm.struct<"struct.std::once_flag", (i32)>
    %2 = llvm.insertvalue %0, %1[0] : !llvm.struct<"struct.std::once_flag", (i32)> 
    llvm.return %2 : !llvm.struct<"struct.std::once_flag", (i32)>
  }
  llvm.mlir.global internal constant @_ZL41descriptor_table_addressbook_2eproto_sccs() {addr_space = 0 : i32, alignment = 16 : i64, dso_local} : !llvm.array<3 x ptr> {
    %0 = llvm.mlir.undef : !llvm.array<0 x ptr>
    %1 = llvm.mlir.zero : !llvm.ptr
    %2 = llvm.mlir.addressof @__gxx_personality_v0 : !llvm.ptr
    %3 = llvm.mlir.addressof @_ZL59InitDefaultsscc_info_Person_PhoneNumber_addressbook_2eprotov : !llvm.ptr
    %4 = llvm.mlir.constant(0 : i32) : i32
    %5 = llvm.mlir.constant(-1 : i32) : i32
    %6 = llvm.mlir.undef : !llvm.struct<(i32)>
    %7 = llvm.insertvalue %5, %6[0] : !llvm.struct<(i32)> 
    %8 = llvm.mlir.undef : !llvm.struct<(struct<(i32)>, i32, i32, ptr)>
    %9 = llvm.insertvalue %7, %8[0] : !llvm.struct<(struct<(i32)>, i32, i32, ptr)> 
    %10 = llvm.insertvalue %4, %9[1] : !llvm.struct<(struct<(i32)>, i32, i32, ptr)> 
    %11 = llvm.insertvalue %4, %10[2] : !llvm.struct<(struct<(i32)>, i32, i32, ptr)> 
    %12 = llvm.insertvalue %3, %11[3] : !llvm.struct<(struct<(i32)>, i32, i32, ptr)> 
    %13 = llvm.mlir.undef : !llvm.struct<(struct<(struct<(i32)>, i32, i32, ptr)>, array<0 x ptr>)>
    %14 = llvm.insertvalue %12, %13[0] : !llvm.struct<(struct<(struct<(i32)>, i32, i32, ptr)>, array<0 x ptr>)> 
    %15 = llvm.insertvalue %0, %14[1] : !llvm.struct<(struct<(struct<(i32)>, i32, i32, ptr)>, array<0 x ptr>)> 
    %16 = llvm.mlir.addressof @scc_info_Person_PhoneNumber_addressbook_2eproto : !llvm.ptr
    %17 = llvm.mlir.addressof @scc_info_Timestamp_google_2fprotobuf_2ftimestamp_2eproto : !llvm.ptr
    %18 = llvm.mlir.undef : !llvm.array<2 x ptr>
    %19 = llvm.insertvalue %16, %18[0] : !llvm.array<2 x ptr> 
    %20 = llvm.insertvalue %17, %19[1] : !llvm.array<2 x ptr> 
    %21 = llvm.mlir.addressof @_ZL47InitDefaultsscc_info_Person_addressbook_2eprotov : !llvm.ptr
    %22 = llvm.mlir.constant(2 : i32) : i32
    %23 = llvm.mlir.undef : !llvm.struct<(struct<(i32)>, i32, i32, ptr)>
    %24 = llvm.insertvalue %7, %23[0] : !llvm.struct<(struct<(i32)>, i32, i32, ptr)> 
    %25 = llvm.insertvalue %22, %24[1] : !llvm.struct<(struct<(i32)>, i32, i32, ptr)> 
    %26 = llvm.insertvalue %4, %25[2] : !llvm.struct<(struct<(i32)>, i32, i32, ptr)> 
    %27 = llvm.insertvalue %21, %26[3] : !llvm.struct<(struct<(i32)>, i32, i32, ptr)> 
    %28 = llvm.mlir.undef : !llvm.struct<(struct<(struct<(i32)>, i32, i32, ptr)>, array<2 x ptr>)>
    %29 = llvm.insertvalue %27, %28[0] : !llvm.struct<(struct<(struct<(i32)>, i32, i32, ptr)>, array<2 x ptr>)> 
    %30 = llvm.insertvalue %20, %29[1] : !llvm.struct<(struct<(struct<(i32)>, i32, i32, ptr)>, array<2 x ptr>)> 
    %31 = llvm.mlir.addressof @scc_info_Person_addressbook_2eproto : !llvm.ptr
    %32 = llvm.mlir.undef : !llvm.array<1 x ptr>
    %33 = llvm.insertvalue %31, %32[0] : !llvm.array<1 x ptr> 
    %34 = llvm.mlir.addressof @_ZL52InitDefaultsscc_info_AddressBook_addressbook_2eprotov : !llvm.ptr
    %35 = llvm.mlir.constant(1 : i32) : i32
    %36 = llvm.mlir.undef : !llvm.struct<(struct<(i32)>, i32, i32, ptr)>
    %37 = llvm.insertvalue %7, %36[0] : !llvm.struct<(struct<(i32)>, i32, i32, ptr)> 
    %38 = llvm.insertvalue %35, %37[1] : !llvm.struct<(struct<(i32)>, i32, i32, ptr)> 
    %39 = llvm.insertvalue %4, %38[2] : !llvm.struct<(struct<(i32)>, i32, i32, ptr)> 
    %40 = llvm.insertvalue %34, %39[3] : !llvm.struct<(struct<(i32)>, i32, i32, ptr)> 
    %41 = llvm.mlir.undef : !llvm.struct<(struct<(struct<(i32)>, i32, i32, ptr)>, array<1 x ptr>)>
    %42 = llvm.insertvalue %40, %41[0] : !llvm.struct<(struct<(struct<(i32)>, i32, i32, ptr)>, array<1 x ptr>)> 
    %43 = llvm.insertvalue %33, %42[1] : !llvm.struct<(struct<(struct<(i32)>, i32, i32, ptr)>, array<1 x ptr>)> 
    %44 = llvm.mlir.addressof @scc_info_AddressBook_addressbook_2eproto : !llvm.ptr
    %45 = llvm.mlir.undef : !llvm.array<3 x ptr>
    %46 = llvm.insertvalue %44, %45[0] : !llvm.array<3 x ptr> 
    %47 = llvm.insertvalue %31, %46[1] : !llvm.array<3 x ptr> 
    %48 = llvm.insertvalue %16, %47[2] : !llvm.array<3 x ptr> 
    llvm.return %48 : !llvm.array<3 x ptr>
  }
  llvm.mlir.global internal constant @_ZL41descriptor_table_addressbook_2eproto_deps() {addr_space = 0 : i32, alignment = 8 : i64, dso_local} : !llvm.array<1 x ptr> {
    %0 = llvm.mlir.addressof @descriptor_table_google_2fprotobuf_2ftimestamp_2eproto : !llvm.ptr
    %1 = llvm.mlir.undef : !llvm.array<1 x ptr>
    %2 = llvm.insertvalue %0, %1[0] : !llvm.array<1 x ptr> 
    llvm.return %2 : !llvm.array<1 x ptr>
  }
  llvm.mlir.global internal constant @_ZL7schemas() {addr_space = 0 : i32, alignment = 16 : i64, dso_local} : !llvm.array<3 x struct<"struct.google::protobuf::internal::MigrationSchema", (i32, i32, i32)>> {
    %0 = llvm.mlir.constant(48 : i32) : i32
    %1 = llvm.mlir.constant(-1 : i32) : i32
    %2 = llvm.mlir.constant(17 : i32) : i32
    %3 = llvm.mlir.undef : !llvm.struct<"struct.google::protobuf::internal::MigrationSchema", (i32, i32, i32)>
    %4 = llvm.insertvalue %2, %3[0] : !llvm.struct<"struct.google::protobuf::internal::MigrationSchema", (i32, i32, i32)> 
    %5 = llvm.insertvalue %1, %4[1] : !llvm.struct<"struct.google::protobuf::internal::MigrationSchema", (i32, i32, i32)> 
    %6 = llvm.insertvalue %0, %5[2] : !llvm.struct<"struct.google::protobuf::internal::MigrationSchema", (i32, i32, i32)> 
    %7 = llvm.mlir.constant(72 : i32) : i32
    %8 = llvm.mlir.constant(7 : i32) : i32
    %9 = llvm.mlir.undef : !llvm.struct<"struct.google::protobuf::internal::MigrationSchema", (i32, i32, i32)>
    %10 = llvm.insertvalue %8, %9[0] : !llvm.struct<"struct.google::protobuf::internal::MigrationSchema", (i32, i32, i32)> 
    %11 = llvm.insertvalue %1, %10[1] : !llvm.struct<"struct.google::protobuf::internal::MigrationSchema", (i32, i32, i32)> 
    %12 = llvm.insertvalue %7, %11[2] : !llvm.struct<"struct.google::protobuf::internal::MigrationSchema", (i32, i32, i32)> 
    %13 = llvm.mlir.constant(32 : i32) : i32
    %14 = llvm.mlir.constant(0 : i32) : i32
    %15 = llvm.mlir.undef : !llvm.struct<"struct.google::protobuf::internal::MigrationSchema", (i32, i32, i32)>
    %16 = llvm.insertvalue %14, %15[0] : !llvm.struct<"struct.google::protobuf::internal::MigrationSchema", (i32, i32, i32)> 
    %17 = llvm.insertvalue %1, %16[1] : !llvm.struct<"struct.google::protobuf::internal::MigrationSchema", (i32, i32, i32)> 
    %18 = llvm.insertvalue %13, %17[2] : !llvm.struct<"struct.google::protobuf::internal::MigrationSchema", (i32, i32, i32)> 
    %19 = llvm.mlir.undef : !llvm.array<3 x struct<"struct.google::protobuf::internal::MigrationSchema", (i32, i32, i32)>>
    %20 = llvm.insertvalue %18, %19[0] : !llvm.array<3 x struct<"struct.google::protobuf::internal::MigrationSchema", (i32, i32, i32)>> 
    %21 = llvm.insertvalue %12, %20[1] : !llvm.array<3 x struct<"struct.google::protobuf::internal::MigrationSchema", (i32, i32, i32)>> 
    %22 = llvm.insertvalue %6, %21[2] : !llvm.array<3 x struct<"struct.google::protobuf::internal::MigrationSchema", (i32, i32, i32)>> 
    llvm.return %22 : !llvm.array<3 x struct<"struct.google::protobuf::internal::MigrationSchema", (i32, i32, i32)>>
  }
  llvm.mlir.global internal constant @_ZL22file_default_instances() {addr_space = 0 : i32, alignment = 16 : i64, dso_local} : !llvm.array<3 x ptr> {
    %0 = llvm.mlir.constant(0 : i8) : i8
    %1 = llvm.mlir.constant(dense<0> : tensor<40xi8>) : !llvm.array<40 x i8>
    %2 = llvm.mlir.constant(0 : i64) : i64
    %3 = llvm.mlir.undef : !llvm.struct<"union.google::protobuf::internal::ExplicitlyConstructed<tutorial::AddressBook>::AlignedUnion", (i64, array<40 x i8>)>
    %4 = llvm.insertvalue %2, %3[0] : !llvm.struct<"union.google::protobuf::internal::ExplicitlyConstructed<tutorial::AddressBook>::AlignedUnion", (i64, array<40 x i8>)> 
    %5 = llvm.insertvalue %1, %4[1] : !llvm.struct<"union.google::protobuf::internal::ExplicitlyConstructed<tutorial::AddressBook>::AlignedUnion", (i64, array<40 x i8>)> 
    %6 = llvm.mlir.undef : !llvm.struct<"class.google::protobuf::internal::ExplicitlyConstructed.1", (struct<"union.google::protobuf::internal::ExplicitlyConstructed<tutorial::AddressBook>::AlignedUnion", (i64, array<40 x i8>)>)>
    %7 = llvm.insertvalue %5, %6[0] : !llvm.struct<"class.google::protobuf::internal::ExplicitlyConstructed.1", (struct<"union.google::protobuf::internal::ExplicitlyConstructed<tutorial::AddressBook>::AlignedUnion", (i64, array<40 x i8>)>)> 
    %8 = llvm.mlir.undef : !llvm.struct<"class.tutorial::AddressBookDefaultTypeInternal", (struct<"class.google::protobuf::internal::ExplicitlyConstructed.1", (struct<"union.google::protobuf::internal::ExplicitlyConstructed<tutorial::AddressBook>::AlignedUnion", (i64, array<40 x i8>)>)>)>
    %9 = llvm.insertvalue %7, %8[0] : !llvm.struct<"class.tutorial::AddressBookDefaultTypeInternal", (struct<"class.google::protobuf::internal::ExplicitlyConstructed.1", (struct<"union.google::protobuf::internal::ExplicitlyConstructed<tutorial::AddressBook>::AlignedUnion", (i64, array<40 x i8>)>)>)> 
    %10 = llvm.mlir.addressof @_ZN8tutorial30_AddressBook_default_instance_E : !llvm.ptr
    %11 = llvm.mlir.constant(dense<0> : tensor<64xi8>) : !llvm.array<64 x i8>
    %12 = llvm.mlir.undef : !llvm.struct<"union.google::protobuf::internal::ExplicitlyConstructed<tutorial::Person>::AlignedUnion", (i64, array<64 x i8>)>
    %13 = llvm.insertvalue %2, %12[0] : !llvm.struct<"union.google::protobuf::internal::ExplicitlyConstructed<tutorial::Person>::AlignedUnion", (i64, array<64 x i8>)> 
    %14 = llvm.insertvalue %11, %13[1] : !llvm.struct<"union.google::protobuf::internal::ExplicitlyConstructed<tutorial::Person>::AlignedUnion", (i64, array<64 x i8>)> 
    %15 = llvm.mlir.undef : !llvm.struct<"class.google::protobuf::internal::ExplicitlyConstructed.0", (struct<"union.google::protobuf::internal::ExplicitlyConstructed<tutorial::Person>::AlignedUnion", (i64, array<64 x i8>)>)>
    %16 = llvm.insertvalue %14, %15[0] : !llvm.struct<"class.google::protobuf::internal::ExplicitlyConstructed.0", (struct<"union.google::protobuf::internal::ExplicitlyConstructed<tutorial::Person>::AlignedUnion", (i64, array<64 x i8>)>)> 
    %17 = llvm.mlir.undef : !llvm.struct<"class.tutorial::PersonDefaultTypeInternal", (struct<"class.google::protobuf::internal::ExplicitlyConstructed.0", (struct<"union.google::protobuf::internal::ExplicitlyConstructed<tutorial::Person>::AlignedUnion", (i64, array<64 x i8>)>)>)>
    %18 = llvm.insertvalue %16, %17[0] : !llvm.struct<"class.tutorial::PersonDefaultTypeInternal", (struct<"class.google::protobuf::internal::ExplicitlyConstructed.0", (struct<"union.google::protobuf::internal::ExplicitlyConstructed<tutorial::Person>::AlignedUnion", (i64, array<64 x i8>)>)>)> 
    %19 = llvm.mlir.addressof @_ZN8tutorial25_Person_default_instance_E : !llvm.ptr
    %20 = llvm.mlir.constant(dense<0> : tensor<24xi8>) : !llvm.array<24 x i8>
    %21 = llvm.mlir.undef : !llvm.struct<"union.google::protobuf::internal::ExplicitlyConstructed<tutorial::Person_PhoneNumber>::AlignedUnion", (i64, array<24 x i8>)>
    %22 = llvm.insertvalue %2, %21[0] : !llvm.struct<"union.google::protobuf::internal::ExplicitlyConstructed<tutorial::Person_PhoneNumber>::AlignedUnion", (i64, array<24 x i8>)> 
    %23 = llvm.insertvalue %20, %22[1] : !llvm.struct<"union.google::protobuf::internal::ExplicitlyConstructed<tutorial::Person_PhoneNumber>::AlignedUnion", (i64, array<24 x i8>)> 
    %24 = llvm.mlir.undef : !llvm.struct<"class.google::protobuf::internal::ExplicitlyConstructed", (struct<"union.google::protobuf::internal::ExplicitlyConstructed<tutorial::Person_PhoneNumber>::AlignedUnion", (i64, array<24 x i8>)>)>
    %25 = llvm.insertvalue %23, %24[0] : !llvm.struct<"class.google::protobuf::internal::ExplicitlyConstructed", (struct<"union.google::protobuf::internal::ExplicitlyConstructed<tutorial::Person_PhoneNumber>::AlignedUnion", (i64, array<24 x i8>)>)> 
    %26 = llvm.mlir.undef : !llvm.struct<"class.tutorial::Person_PhoneNumberDefaultTypeInternal", (struct<"class.google::protobuf::internal::ExplicitlyConstructed", (struct<"union.google::protobuf::internal::ExplicitlyConstructed<tutorial::Person_PhoneNumber>::AlignedUnion", (i64, array<24 x i8>)>)>)>
    %27 = llvm.insertvalue %25, %26[0] : !llvm.struct<"class.tutorial::Person_PhoneNumberDefaultTypeInternal", (struct<"class.google::protobuf::internal::ExplicitlyConstructed", (struct<"union.google::protobuf::internal::ExplicitlyConstructed<tutorial::Person_PhoneNumber>::AlignedUnion", (i64, array<24 x i8>)>)>)> 
    %28 = llvm.mlir.addressof @_ZN8tutorial37_Person_PhoneNumber_default_instance_E : !llvm.ptr
    %29 = llvm.mlir.undef : !llvm.array<3 x ptr>
    %30 = llvm.insertvalue %28, %29[0] : !llvm.array<3 x ptr> 
    %31 = llvm.insertvalue %19, %30[1] : !llvm.array<3 x ptr> 
    %32 = llvm.insertvalue %10, %31[2] : !llvm.array<3 x ptr> 
    llvm.return %32 : !llvm.array<3 x ptr>
  }
  llvm.mlir.global internal @_ZL39file_level_metadata_addressbook_2eproto() {addr_space = 0 : i32, alignment = 16 : i64, dso_local} : !llvm.array<3 x struct<"struct.google::protobuf::Metadata", (ptr, ptr)>> {
    %0 = llvm.mlir.zero : !llvm.ptr
    %1 = llvm.mlir.undef : !llvm.struct<"struct.google::protobuf::Metadata", (ptr, ptr)>
    %2 = llvm.insertvalue %0, %1[0] : !llvm.struct<"struct.google::protobuf::Metadata", (ptr, ptr)> 
    %3 = llvm.insertvalue %0, %2[1] : !llvm.struct<"struct.google::protobuf::Metadata", (ptr, ptr)> 
    %4 = llvm.mlir.undef : !llvm.array<3 x struct<"struct.google::protobuf::Metadata", (ptr, ptr)>>
    %5 = llvm.insertvalue %3, %4[0] : !llvm.array<3 x struct<"struct.google::protobuf::Metadata", (ptr, ptr)>> 
    %6 = llvm.insertvalue %3, %5[1] : !llvm.array<3 x struct<"struct.google::protobuf::Metadata", (ptr, ptr)>> 
    %7 = llvm.insertvalue %3, %6[2] : !llvm.array<3 x struct<"struct.google::protobuf::Metadata", (ptr, ptr)>> 
    llvm.return %7 : !llvm.array<3 x struct<"struct.google::protobuf::Metadata", (ptr, ptr)>>
  }
  llvm.mlir.global internal @_ZL47file_level_enum_descriptors_addressbook_2eproto() {addr_space = 0 : i32, alignment = 8 : i64, dso_local} : !llvm.array<1 x ptr> {
    %0 = llvm.mlir.zero : !llvm.ptr
    %1 = llvm.mlir.undef : !llvm.array<1 x ptr>
    %2 = llvm.insertvalue %0, %1[0] : !llvm.array<1 x ptr> 
    llvm.return %2 : !llvm.array<1 x ptr>
  }
  llvm.mlir.global external @descriptor_table_addressbook_2eproto() {addr_space = 0 : i32, alignment = 8 : i64, dso_local} : !llvm.struct<"struct.google::protobuf::internal::DescriptorTable", (i8, i8, ptr, ptr, i32, ptr, ptr, ptr, i32, i32, ptr, ptr, ptr, ptr, i32, ptr, ptr)> {
    %0 = llvm.mlir.zero : !llvm.ptr
    %1 = llvm.mlir.undef : !llvm.array<1 x ptr>
    %2 = llvm.insertvalue %0, %1[0] : !llvm.array<1 x ptr> 
    %3 = llvm.mlir.addressof @_ZL47file_level_enum_descriptors_addressbook_2eproto : !llvm.ptr
    %4 = llvm.mlir.constant(3 : i32) : i32
    %5 = llvm.mlir.undef : !llvm.struct<"struct.google::protobuf::Metadata", (ptr, ptr)>
    %6 = llvm.insertvalue %0, %5[0] : !llvm.struct<"struct.google::protobuf::Metadata", (ptr, ptr)> 
    %7 = llvm.insertvalue %0, %6[1] : !llvm.struct<"struct.google::protobuf::Metadata", (ptr, ptr)> 
    %8 = llvm.mlir.undef : !llvm.array<3 x struct<"struct.google::protobuf::Metadata", (ptr, ptr)>>
    %9 = llvm.insertvalue %7, %8[0] : !llvm.array<3 x struct<"struct.google::protobuf::Metadata", (ptr, ptr)>> 
    %10 = llvm.insertvalue %7, %9[1] : !llvm.array<3 x struct<"struct.google::protobuf::Metadata", (ptr, ptr)>> 
    %11 = llvm.insertvalue %7, %10[2] : !llvm.array<3 x struct<"struct.google::protobuf::Metadata", (ptr, ptr)>> 
    %12 = llvm.mlir.addressof @_ZL39file_level_metadata_addressbook_2eproto : !llvm.ptr
    %13 = llvm.mlir.constant(dense<[-1, 8, -1, -1, -1, 16, 24, -1, 8, -1, -1, -1, 40, 64, 48, 16, 56, -1, 8, -1, -1, -1, 16]> : tensor<23xi32>) : !llvm.array<23 x i32>
    %14 = llvm.mlir.addressof @_ZN31TableStruct_addressbook_2eproto7offsetsE : !llvm.ptr
    %15 = llvm.mlir.constant(0 : i8) : i8
    %16 = llvm.mlir.constant(dense<0> : tensor<40xi8>) : !llvm.array<40 x i8>
    %17 = llvm.mlir.constant(0 : i64) : i64
    %18 = llvm.mlir.undef : !llvm.struct<"union.google::protobuf::internal::ExplicitlyConstructed<tutorial::AddressBook>::AlignedUnion", (i64, array<40 x i8>)>
    %19 = llvm.insertvalue %17, %18[0] : !llvm.struct<"union.google::protobuf::internal::ExplicitlyConstructed<tutorial::AddressBook>::AlignedUnion", (i64, array<40 x i8>)> 
    %20 = llvm.insertvalue %16, %19[1] : !llvm.struct<"union.google::protobuf::internal::ExplicitlyConstructed<tutorial::AddressBook>::AlignedUnion", (i64, array<40 x i8>)> 
    %21 = llvm.mlir.undef : !llvm.struct<"class.google::protobuf::internal::ExplicitlyConstructed.1", (struct<"union.google::protobuf::internal::ExplicitlyConstructed<tutorial::AddressBook>::AlignedUnion", (i64, array<40 x i8>)>)>
    %22 = llvm.insertvalue %20, %21[0] : !llvm.struct<"class.google::protobuf::internal::ExplicitlyConstructed.1", (struct<"union.google::protobuf::internal::ExplicitlyConstructed<tutorial::AddressBook>::AlignedUnion", (i64, array<40 x i8>)>)> 
    %23 = llvm.mlir.undef : !llvm.struct<"class.tutorial::AddressBookDefaultTypeInternal", (struct<"class.google::protobuf::internal::ExplicitlyConstructed.1", (struct<"union.google::protobuf::internal::ExplicitlyConstructed<tutorial::AddressBook>::AlignedUnion", (i64, array<40 x i8>)>)>)>
    %24 = llvm.insertvalue %22, %23[0] : !llvm.struct<"class.tutorial::AddressBookDefaultTypeInternal", (struct<"class.google::protobuf::internal::ExplicitlyConstructed.1", (struct<"union.google::protobuf::internal::ExplicitlyConstructed<tutorial::AddressBook>::AlignedUnion", (i64, array<40 x i8>)>)>)> 
    %25 = llvm.mlir.addressof @_ZN8tutorial30_AddressBook_default_instance_E : !llvm.ptr
    %26 = llvm.mlir.constant(dense<0> : tensor<64xi8>) : !llvm.array<64 x i8>
    %27 = llvm.mlir.undef : !llvm.struct<"union.google::protobuf::internal::ExplicitlyConstructed<tutorial::Person>::AlignedUnion", (i64, array<64 x i8>)>
    %28 = llvm.insertvalue %17, %27[0] : !llvm.struct<"union.google::protobuf::internal::ExplicitlyConstructed<tutorial::Person>::AlignedUnion", (i64, array<64 x i8>)> 
    %29 = llvm.insertvalue %26, %28[1] : !llvm.struct<"union.google::protobuf::internal::ExplicitlyConstructed<tutorial::Person>::AlignedUnion", (i64, array<64 x i8>)> 
    %30 = llvm.mlir.undef : !llvm.struct<"class.google::protobuf::internal::ExplicitlyConstructed.0", (struct<"union.google::protobuf::internal::ExplicitlyConstructed<tutorial::Person>::AlignedUnion", (i64, array<64 x i8>)>)>
    %31 = llvm.insertvalue %29, %30[0] : !llvm.struct<"class.google::protobuf::internal::ExplicitlyConstructed.0", (struct<"union.google::protobuf::internal::ExplicitlyConstructed<tutorial::Person>::AlignedUnion", (i64, array<64 x i8>)>)> 
    %32 = llvm.mlir.undef : !llvm.struct<"class.tutorial::PersonDefaultTypeInternal", (struct<"class.google::protobuf::internal::ExplicitlyConstructed.0", (struct<"union.google::protobuf::internal::ExplicitlyConstructed<tutorial::Person>::AlignedUnion", (i64, array<64 x i8>)>)>)>
    %33 = llvm.insertvalue %31, %32[0] : !llvm.struct<"class.tutorial::PersonDefaultTypeInternal", (struct<"class.google::protobuf::internal::ExplicitlyConstructed.0", (struct<"union.google::protobuf::internal::ExplicitlyConstructed<tutorial::Person>::AlignedUnion", (i64, array<64 x i8>)>)>)> 
    %34 = llvm.mlir.addressof @_ZN8tutorial25_Person_default_instance_E : !llvm.ptr
    %35 = llvm.mlir.constant(dense<0> : tensor<24xi8>) : !llvm.array<24 x i8>
    %36 = llvm.mlir.undef : !llvm.struct<"union.google::protobuf::internal::ExplicitlyConstructed<tutorial::Person_PhoneNumber>::AlignedUnion", (i64, array<24 x i8>)>
    %37 = llvm.insertvalue %17, %36[0] : !llvm.struct<"union.google::protobuf::internal::ExplicitlyConstructed<tutorial::Person_PhoneNumber>::AlignedUnion", (i64, array<24 x i8>)> 
    %38 = llvm.insertvalue %35, %37[1] : !llvm.struct<"union.google::protobuf::internal::ExplicitlyConstructed<tutorial::Person_PhoneNumber>::AlignedUnion", (i64, array<24 x i8>)> 
    %39 = llvm.mlir.undef : !llvm.struct<"class.google::protobuf::internal::ExplicitlyConstructed", (struct<"union.google::protobuf::internal::ExplicitlyConstructed<tutorial::Person_PhoneNumber>::AlignedUnion", (i64, array<24 x i8>)>)>
    %40 = llvm.insertvalue %38, %39[0] : !llvm.struct<"class.google::protobuf::internal::ExplicitlyConstructed", (struct<"union.google::protobuf::internal::ExplicitlyConstructed<tutorial::Person_PhoneNumber>::AlignedUnion", (i64, array<24 x i8>)>)> 
    %41 = llvm.mlir.undef : !llvm.struct<"class.tutorial::Person_PhoneNumberDefaultTypeInternal", (struct<"class.google::protobuf::internal::ExplicitlyConstructed", (struct<"union.google::protobuf::internal::ExplicitlyConstructed<tutorial::Person_PhoneNumber>::AlignedUnion", (i64, array<24 x i8>)>)>)>
    %42 = llvm.insertvalue %40, %41[0] : !llvm.struct<"class.tutorial::Person_PhoneNumberDefaultTypeInternal", (struct<"class.google::protobuf::internal::ExplicitlyConstructed", (struct<"union.google::protobuf::internal::ExplicitlyConstructed<tutorial::Person_PhoneNumber>::AlignedUnion", (i64, array<24 x i8>)>)>)> 
    %43 = llvm.mlir.addressof @_ZN8tutorial37_Person_PhoneNumber_default_instance_E : !llvm.ptr
    %44 = llvm.mlir.undef : !llvm.array<3 x ptr>
    %45 = llvm.insertvalue %43, %44[0] : !llvm.array<3 x ptr> 
    %46 = llvm.insertvalue %34, %45[1] : !llvm.array<3 x ptr> 
    %47 = llvm.insertvalue %25, %46[2] : !llvm.array<3 x ptr> 
    %48 = llvm.mlir.addressof @_ZL22file_default_instances : !llvm.ptr
    %49 = llvm.mlir.constant(48 : i32) : i32
    %50 = llvm.mlir.constant(-1 : i32) : i32
    %51 = llvm.mlir.constant(17 : i32) : i32
    %52 = llvm.mlir.undef : !llvm.struct<"struct.google::protobuf::internal::MigrationSchema", (i32, i32, i32)>
    %53 = llvm.insertvalue %51, %52[0] : !llvm.struct<"struct.google::protobuf::internal::MigrationSchema", (i32, i32, i32)> 
    %54 = llvm.insertvalue %50, %53[1] : !llvm.struct<"struct.google::protobuf::internal::MigrationSchema", (i32, i32, i32)> 
    %55 = llvm.insertvalue %49, %54[2] : !llvm.struct<"struct.google::protobuf::internal::MigrationSchema", (i32, i32, i32)> 
    %56 = llvm.mlir.constant(72 : i32) : i32
    %57 = llvm.mlir.constant(7 : i32) : i32
    %58 = llvm.mlir.undef : !llvm.struct<"struct.google::protobuf::internal::MigrationSchema", (i32, i32, i32)>
    %59 = llvm.insertvalue %57, %58[0] : !llvm.struct<"struct.google::protobuf::internal::MigrationSchema", (i32, i32, i32)> 
    %60 = llvm.insertvalue %50, %59[1] : !llvm.struct<"struct.google::protobuf::internal::MigrationSchema", (i32, i32, i32)> 
    %61 = llvm.insertvalue %56, %60[2] : !llvm.struct<"struct.google::protobuf::internal::MigrationSchema", (i32, i32, i32)> 
    %62 = llvm.mlir.constant(32 : i32) : i32
    %63 = llvm.mlir.constant(0 : i32) : i32
    %64 = llvm.mlir.undef : !llvm.struct<"struct.google::protobuf::internal::MigrationSchema", (i32, i32, i32)>
    %65 = llvm.insertvalue %63, %64[0] : !llvm.struct<"struct.google::protobuf::internal::MigrationSchema", (i32, i32, i32)> 
    %66 = llvm.insertvalue %50, %65[1] : !llvm.struct<"struct.google::protobuf::internal::MigrationSchema", (i32, i32, i32)> 
    %67 = llvm.insertvalue %62, %66[2] : !llvm.struct<"struct.google::protobuf::internal::MigrationSchema", (i32, i32, i32)> 
    %68 = llvm.mlir.undef : !llvm.array<3 x struct<"struct.google::protobuf::internal::MigrationSchema", (i32, i32, i32)>>
    %69 = llvm.insertvalue %67, %68[0] : !llvm.array<3 x struct<"struct.google::protobuf::internal::MigrationSchema", (i32, i32, i32)>> 
    %70 = llvm.insertvalue %61, %69[1] : !llvm.array<3 x struct<"struct.google::protobuf::internal::MigrationSchema", (i32, i32, i32)>> 
    %71 = llvm.insertvalue %55, %70[2] : !llvm.array<3 x struct<"struct.google::protobuf::internal::MigrationSchema", (i32, i32, i32)>> 
    %72 = llvm.mlir.addressof @_ZL7schemas : !llvm.ptr
    %73 = llvm.mlir.constant(1 : i32) : i32
    %74 = llvm.mlir.addressof @descriptor_table_google_2fprotobuf_2ftimestamp_2eproto : !llvm.ptr
    %75 = llvm.mlir.undef : !llvm.array<1 x ptr>
    %76 = llvm.insertvalue %74, %75[0] : !llvm.array<1 x ptr> 
    %77 = llvm.mlir.addressof @_ZL41descriptor_table_addressbook_2eproto_deps : !llvm.ptr
    %78 = llvm.mlir.undef : !llvm.array<0 x ptr>
    %79 = llvm.mlir.addressof @__gxx_personality_v0 : !llvm.ptr
    %80 = llvm.mlir.addressof @_ZL59InitDefaultsscc_info_Person_PhoneNumber_addressbook_2eprotov : !llvm.ptr
    %81 = llvm.mlir.undef : !llvm.struct<(i32)>
    %82 = llvm.insertvalue %50, %81[0] : !llvm.struct<(i32)> 
    %83 = llvm.mlir.undef : !llvm.struct<(struct<(i32)>, i32, i32, ptr)>
    %84 = llvm.insertvalue %82, %83[0] : !llvm.struct<(struct<(i32)>, i32, i32, ptr)> 
    %85 = llvm.insertvalue %63, %84[1] : !llvm.struct<(struct<(i32)>, i32, i32, ptr)> 
    %86 = llvm.insertvalue %63, %85[2] : !llvm.struct<(struct<(i32)>, i32, i32, ptr)> 
    %87 = llvm.insertvalue %80, %86[3] : !llvm.struct<(struct<(i32)>, i32, i32, ptr)> 
    %88 = llvm.mlir.undef : !llvm.struct<(struct<(struct<(i32)>, i32, i32, ptr)>, array<0 x ptr>)>
    %89 = llvm.insertvalue %87, %88[0] : !llvm.struct<(struct<(struct<(i32)>, i32, i32, ptr)>, array<0 x ptr>)> 
    %90 = llvm.insertvalue %78, %89[1] : !llvm.struct<(struct<(struct<(i32)>, i32, i32, ptr)>, array<0 x ptr>)> 
    %91 = llvm.mlir.addressof @scc_info_Person_PhoneNumber_addressbook_2eproto : !llvm.ptr
    %92 = llvm.mlir.addressof @scc_info_Timestamp_google_2fprotobuf_2ftimestamp_2eproto : !llvm.ptr
    %93 = llvm.mlir.undef : !llvm.array<2 x ptr>
    %94 = llvm.insertvalue %91, %93[0] : !llvm.array<2 x ptr> 
    %95 = llvm.insertvalue %92, %94[1] : !llvm.array<2 x ptr> 
    %96 = llvm.mlir.addressof @_ZL47InitDefaultsscc_info_Person_addressbook_2eprotov : !llvm.ptr
    %97 = llvm.mlir.constant(2 : i32) : i32
    %98 = llvm.mlir.undef : !llvm.struct<(struct<(i32)>, i32, i32, ptr)>
    %99 = llvm.insertvalue %82, %98[0] : !llvm.struct<(struct<(i32)>, i32, i32, ptr)> 
    %100 = llvm.insertvalue %97, %99[1] : !llvm.struct<(struct<(i32)>, i32, i32, ptr)> 
    %101 = llvm.insertvalue %63, %100[2] : !llvm.struct<(struct<(i32)>, i32, i32, ptr)> 
    %102 = llvm.insertvalue %96, %101[3] : !llvm.struct<(struct<(i32)>, i32, i32, ptr)> 
    %103 = llvm.mlir.undef : !llvm.struct<(struct<(struct<(i32)>, i32, i32, ptr)>, array<2 x ptr>)>
    %104 = llvm.insertvalue %102, %103[0] : !llvm.struct<(struct<(struct<(i32)>, i32, i32, ptr)>, array<2 x ptr>)> 
    %105 = llvm.insertvalue %95, %104[1] : !llvm.struct<(struct<(struct<(i32)>, i32, i32, ptr)>, array<2 x ptr>)> 
    %106 = llvm.mlir.addressof @scc_info_Person_addressbook_2eproto : !llvm.ptr
    %107 = llvm.mlir.undef : !llvm.array<1 x ptr>
    %108 = llvm.insertvalue %106, %107[0] : !llvm.array<1 x ptr> 
    %109 = llvm.mlir.addressof @_ZL52InitDefaultsscc_info_AddressBook_addressbook_2eprotov : !llvm.ptr
    %110 = llvm.mlir.undef : !llvm.struct<(struct<(i32)>, i32, i32, ptr)>
    %111 = llvm.insertvalue %82, %110[0] : !llvm.struct<(struct<(i32)>, i32, i32, ptr)> 
    %112 = llvm.insertvalue %73, %111[1] : !llvm.struct<(struct<(i32)>, i32, i32, ptr)> 
    %113 = llvm.insertvalue %63, %112[2] : !llvm.struct<(struct<(i32)>, i32, i32, ptr)> 
    %114 = llvm.insertvalue %109, %113[3] : !llvm.struct<(struct<(i32)>, i32, i32, ptr)> 
    %115 = llvm.mlir.undef : !llvm.struct<(struct<(struct<(i32)>, i32, i32, ptr)>, array<1 x ptr>)>
    %116 = llvm.insertvalue %114, %115[0] : !llvm.struct<(struct<(struct<(i32)>, i32, i32, ptr)>, array<1 x ptr>)> 
    %117 = llvm.insertvalue %108, %116[1] : !llvm.struct<(struct<(struct<(i32)>, i32, i32, ptr)>, array<1 x ptr>)> 
    %118 = llvm.mlir.addressof @scc_info_AddressBook_addressbook_2eproto : !llvm.ptr
    %119 = llvm.mlir.undef : !llvm.array<3 x ptr>
    %120 = llvm.insertvalue %118, %119[0] : !llvm.array<3 x ptr> 
    %121 = llvm.insertvalue %106, %120[1] : !llvm.array<3 x ptr> 
    %122 = llvm.insertvalue %91, %121[2] : !llvm.array<3 x ptr> 
    %123 = llvm.mlir.addressof @_ZL41descriptor_table_addressbook_2eproto_sccs : !llvm.ptr
    %124 = llvm.mlir.undef : !llvm.struct<"struct.std::once_flag", (i32)>
    %125 = llvm.insertvalue %63, %124[0] : !llvm.struct<"struct.std::once_flag", (i32)> 
    %126 = llvm.mlir.addressof @_ZL41descriptor_table_addressbook_2eproto_once : !llvm.ptr
    %127 = llvm.mlir.constant(537 : i32) : i32
    %128 = llvm.mlir.constant("addressbook.proto\00") : !llvm.array<18 x i8>
    %129 = llvm.mlir.addressof @".str" : !llvm.ptr
    %130 = llvm.mlir.constant("\0A\11addressbook.proto\12\08tutorial\1A\1Fgoogle/protobuf/timestamp.proto\22\87\02\0A\06Person\12\0C\0A\04name\18\01 \01(\09\12\0A\0A\02id\18\02 \01(\05\12\0D\0A\05email\18\03 \01(\09\12,\0A\06phones\18\04 \03(\0B2\1C.tutorial.Person.PhoneNumber\120\0A\0Clast_updated\18\05 \01(\0B2\1A.google.protobuf.Timestamp\1AG\0A\0BPhoneNumber\12\0E\0A\06number\18\01 \01(\09\12(\0A\04type\18\02 \01(\0E2\1A.tutorial.Person.PhoneType\22+\0A\09PhoneType\12\0A\0A\06MOBILE\10\00\12\08\0A\04HOME\10\01\12\08\0A\04WORK\10\02\22/\0A\0BAddressBook\12 \0A\06people\18\01 \03(\0B2\10.tutorial.PersonB\95\01\0A\1Bcom.example.tutorial.protosB\11AddressBookProtosP\01Z:github.com/protocolbuffers/protobuf/examples/go/tutorialpb\AA\02$Google.Protobuf.Examples.AddressBookb\06proto3\00") : !llvm.array<538 x i8>
    %131 = llvm.mlir.addressof @_ZL45descriptor_table_protodef_addressbook_2eproto : !llvm.ptr
    %132 = llvm.mlir.undef : !llvm.struct<"struct.google::protobuf::internal::DescriptorTable", (i8, i8, ptr, ptr, i32, ptr, ptr, ptr, i32, i32, ptr, ptr, ptr, ptr, i32, ptr, ptr)>
    %133 = llvm.insertvalue %15, %132[0] : !llvm.struct<"struct.google::protobuf::internal::DescriptorTable", (i8, i8, ptr, ptr, i32, ptr, ptr, ptr, i32, i32, ptr, ptr, ptr, ptr, i32, ptr, ptr)> 
    %134 = llvm.insertvalue %15, %133[1] : !llvm.struct<"struct.google::protobuf::internal::DescriptorTable", (i8, i8, ptr, ptr, i32, ptr, ptr, ptr, i32, i32, ptr, ptr, ptr, ptr, i32, ptr, ptr)> 
    %135 = llvm.insertvalue %131, %134[2] : !llvm.struct<"struct.google::protobuf::internal::DescriptorTable", (i8, i8, ptr, ptr, i32, ptr, ptr, ptr, i32, i32, ptr, ptr, ptr, ptr, i32, ptr, ptr)> 
    %136 = llvm.insertvalue %129, %135[3] : !llvm.struct<"struct.google::protobuf::internal::DescriptorTable", (i8, i8, ptr, ptr, i32, ptr, ptr, ptr, i32, i32, ptr, ptr, ptr, ptr, i32, ptr, ptr)> 
    %137 = llvm.insertvalue %127, %136[4] : !llvm.struct<"struct.google::protobuf::internal::DescriptorTable", (i8, i8, ptr, ptr, i32, ptr, ptr, ptr, i32, i32, ptr, ptr, ptr, ptr, i32, ptr, ptr)> 
    %138 = llvm.insertvalue %126, %137[5] : !llvm.struct<"struct.google::protobuf::internal::DescriptorTable", (i8, i8, ptr, ptr, i32, ptr, ptr, ptr, i32, i32, ptr, ptr, ptr, ptr, i32, ptr, ptr)> 
    %139 = llvm.insertvalue %123, %138[6] : !llvm.struct<"struct.google::protobuf::internal::DescriptorTable", (i8, i8, ptr, ptr, i32, ptr, ptr, ptr, i32, i32, ptr, ptr, ptr, ptr, i32, ptr, ptr)> 
    %140 = llvm.insertvalue %77, %139[7] : !llvm.struct<"struct.google::protobuf::internal::DescriptorTable", (i8, i8, ptr, ptr, i32, ptr, ptr, ptr, i32, i32, ptr, ptr, ptr, ptr, i32, ptr, ptr)> 
    %141 = llvm.insertvalue %4, %140[8] : !llvm.struct<"struct.google::protobuf::internal::DescriptorTable", (i8, i8, ptr, ptr, i32, ptr, ptr, ptr, i32, i32, ptr, ptr, ptr, ptr, i32, ptr, ptr)> 
    %142 = llvm.insertvalue %73, %141[9] : !llvm.struct<"struct.google::protobuf::internal::DescriptorTable", (i8, i8, ptr, ptr, i32, ptr, ptr, ptr, i32, i32, ptr, ptr, ptr, ptr, i32, ptr, ptr)> 
    %143 = llvm.insertvalue %72, %142[10] : !llvm.struct<"struct.google::protobuf::internal::DescriptorTable", (i8, i8, ptr, ptr, i32, ptr, ptr, ptr, i32, i32, ptr, ptr, ptr, ptr, i32, ptr, ptr)> 
    %144 = llvm.insertvalue %48, %143[11] : !llvm.struct<"struct.google::protobuf::internal::DescriptorTable", (i8, i8, ptr, ptr, i32, ptr, ptr, ptr, i32, i32, ptr, ptr, ptr, ptr, i32, ptr, ptr)> 
    %145 = llvm.insertvalue %14, %144[12] : !llvm.struct<"struct.google::protobuf::internal::DescriptorTable", (i8, i8, ptr, ptr, i32, ptr, ptr, ptr, i32, i32, ptr, ptr, ptr, ptr, i32, ptr, ptr)> 
    %146 = llvm.insertvalue %12, %145[13] : !llvm.struct<"struct.google::protobuf::internal::DescriptorTable", (i8, i8, ptr, ptr, i32, ptr, ptr, ptr, i32, i32, ptr, ptr, ptr, ptr, i32, ptr, ptr)> 
    %147 = llvm.insertvalue %4, %146[14] : !llvm.struct<"struct.google::protobuf::internal::DescriptorTable", (i8, i8, ptr, ptr, i32, ptr, ptr, ptr, i32, i32, ptr, ptr, ptr, ptr, i32, ptr, ptr)> 
    %148 = llvm.insertvalue %3, %147[15] : !llvm.struct<"struct.google::protobuf::internal::DescriptorTable", (i8, i8, ptr, ptr, i32, ptr, ptr, ptr, i32, i32, ptr, ptr, ptr, ptr, i32, ptr, ptr)> 
    %149 = llvm.insertvalue %0, %148[16] : !llvm.struct<"struct.google::protobuf::internal::DescriptorTable", (i8, i8, ptr, ptr, i32, ptr, ptr, ptr, i32, i32, ptr, ptr, ptr, ptr, i32, ptr, ptr)> 
    llvm.return %149 : !llvm.struct<"struct.google::protobuf::internal::DescriptorTable", (i8, i8, ptr, ptr, i32, ptr, ptr, ptr, i32, i32, ptr, ptr, ptr, ptr, i32, ptr, ptr)>
  }
  llvm.mlir.global external unnamed_addr constant @_ZTVN8tutorial18Person_PhoneNumberE() {addr_space = 0 : i32, alignment = 8 : i64, dso_local} : !llvm.struct<(array<22 x ptr>)> {
    %0 = llvm.mlir.addressof @_ZNK8tutorial18Person_PhoneNumber11GetMetadataEv : !llvm.ptr
    %1 = llvm.mlir.zero : !llvm.ptr
    %2 = llvm.mlir.addressof @__gxx_personality_v0 : !llvm.ptr
    %3 = llvm.mlir.addressof @_ZNK8tutorial18Person_PhoneNumber13SetCachedSizeEi : !llvm.ptr
    %4 = llvm.mlir.addressof @_ZNK6google8protobuf7Message13SpaceUsedLongEv : !llvm.ptr
    %5 = llvm.mlir.addressof @_ZN6google8protobuf7Message20DiscardUnknownFieldsEv : !llvm.ptr
    %6 = llvm.mlir.addressof @_ZN8tutorial18Person_PhoneNumber9MergeFromERKN6google8protobuf7MessageE : !llvm.ptr
    %7 = llvm.mlir.addressof @_ZN8tutorial18Person_PhoneNumber8CopyFromERKN6google8protobuf7MessageE : !llvm.ptr
    %8 = llvm.mlir.addressof @_ZNK6google8protobuf11MessageLite16InternalGetTableEv : !llvm.ptr
    %9 = llvm.mlir.addressof @_ZNK8tutorial18Person_PhoneNumber18_InternalSerializeEPhPN6google8protobuf2io19EpsCopyOutputStreamE : !llvm.ptr
    %10 = llvm.mlir.addressof @_ZN8tutorial18Person_PhoneNumber14_InternalParseEPKcPN6google8protobuf8internal12ParseContextE : !llvm.ptr
    %11 = llvm.mlir.addressof @_ZNK8tutorial18Person_PhoneNumber13GetCachedSizeEv : !llvm.ptr
    %12 = llvm.mlir.addressof @_ZNK8tutorial18Person_PhoneNumber12ByteSizeLongEv : !llvm.ptr
    %13 = llvm.mlir.addressof @_ZN6google8protobuf7Message21CheckTypeAndMergeFromERKNS0_11MessageLiteE : !llvm.ptr
    %14 = llvm.mlir.addressof @_ZNK6google8protobuf7Message25InitializationErrorStringB5cxx11Ev : !llvm.ptr
    %15 = llvm.mlir.addressof @_ZNK8tutorial18Person_PhoneNumber13IsInitializedEv : !llvm.ptr
    %16 = llvm.mlir.addressof @_ZN8tutorial18Person_PhoneNumber5ClearEv : !llvm.ptr
    %17 = llvm.mlir.addressof @_ZNK8tutorial18Person_PhoneNumber3NewEPN6google8protobuf5ArenaE : !llvm.ptr
    %18 = llvm.mlir.addressof @_ZNK8tutorial18Person_PhoneNumber3NewEv : !llvm.ptr
    %19 = llvm.mlir.addressof @_ZNK6google8protobuf7Message11GetTypeNameB5cxx11Ev : !llvm.ptr
    %20 = llvm.mlir.addressof @_ZN8tutorial18Person_PhoneNumberD0Ev : !llvm.ptr
    %21 = llvm.mlir.addressof @_ZN8tutorial18Person_PhoneNumberD2Ev : !llvm.ptr
    %22 = llvm.mlir.addressof @_ZTIN6google8protobuf7MessageE : !llvm.ptr
    %23 = llvm.mlir.constant("N8tutorial18Person_PhoneNumberE\00") : !llvm.array<32 x i8>
    %24 = llvm.mlir.addressof @_ZTSN8tutorial18Person_PhoneNumberE : !llvm.ptr
    %25 = llvm.mlir.constant(2 : i64) : i64
    %26 = llvm.mlir.addressof @_ZTVN10__cxxabiv120__si_class_type_infoE : !llvm.ptr
    %27 = llvm.getelementptr inbounds %26[%25] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.ptr
    %28 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, ptr)>
    %29 = llvm.insertvalue %27, %28[0] : !llvm.struct<(ptr, ptr, ptr)> 
    %30 = llvm.insertvalue %24, %29[1] : !llvm.struct<(ptr, ptr, ptr)> 
    %31 = llvm.insertvalue %22, %30[2] : !llvm.struct<(ptr, ptr, ptr)> 
    %32 = llvm.mlir.addressof @_ZTIN8tutorial18Person_PhoneNumberE : !llvm.ptr
    %33 = llvm.mlir.undef : !llvm.array<22 x ptr>
    %34 = llvm.insertvalue %1, %33[0] : !llvm.array<22 x ptr> 
    %35 = llvm.insertvalue %32, %34[1] : !llvm.array<22 x ptr> 
    %36 = llvm.insertvalue %21, %35[2] : !llvm.array<22 x ptr> 
    %37 = llvm.insertvalue %20, %36[3] : !llvm.array<22 x ptr> 
    %38 = llvm.insertvalue %19, %37[4] : !llvm.array<22 x ptr> 
    %39 = llvm.insertvalue %18, %38[5] : !llvm.array<22 x ptr> 
    %40 = llvm.insertvalue %17, %39[6] : !llvm.array<22 x ptr> 
    %41 = llvm.insertvalue %16, %40[7] : !llvm.array<22 x ptr> 
    %42 = llvm.insertvalue %15, %41[8] : !llvm.array<22 x ptr> 
    %43 = llvm.insertvalue %14, %42[9] : !llvm.array<22 x ptr> 
    %44 = llvm.insertvalue %13, %43[10] : !llvm.array<22 x ptr> 
    %45 = llvm.insertvalue %12, %44[11] : !llvm.array<22 x ptr> 
    %46 = llvm.insertvalue %11, %45[12] : !llvm.array<22 x ptr> 
    %47 = llvm.insertvalue %10, %46[13] : !llvm.array<22 x ptr> 
    %48 = llvm.insertvalue %9, %47[14] : !llvm.array<22 x ptr> 
    %49 = llvm.insertvalue %8, %48[15] : !llvm.array<22 x ptr> 
    %50 = llvm.insertvalue %7, %49[16] : !llvm.array<22 x ptr> 
    %51 = llvm.insertvalue %6, %50[17] : !llvm.array<22 x ptr> 
    %52 = llvm.insertvalue %5, %51[18] : !llvm.array<22 x ptr> 
    %53 = llvm.insertvalue %4, %52[19] : !llvm.array<22 x ptr> 
    %54 = llvm.insertvalue %3, %53[20] : !llvm.array<22 x ptr> 
    %55 = llvm.insertvalue %0, %54[21] : !llvm.array<22 x ptr> 
    %56 = llvm.mlir.undef : !llvm.struct<(array<22 x ptr>)>
    %57 = llvm.insertvalue %55, %56[0] : !llvm.struct<(array<22 x ptr>)> 
    llvm.return %57 : !llvm.struct<(array<22 x ptr>)>
  }
  llvm.mlir.global private unnamed_addr constant @".str.4"("tutorial.Person.PhoneNumber.number\00") {addr_space = 0 : i32, alignment = 1 : i64, dso_local}
  llvm.mlir.global private unnamed_addr constant @".str.5"("build/addressbook.pb.cc\00") {addr_space = 0 : i32, alignment = 1 : i64, dso_local}
  llvm.mlir.global private unnamed_addr constant @".str.6"("CHECK failed: (&from) != (this): \00") {addr_space = 0 : i32, alignment = 1 : i64, dso_local}
  llvm.mlir.global external unnamed_addr constant @_ZTVN8tutorial6PersonE() {addr_space = 0 : i32, alignment = 8 : i64, dso_local} : !llvm.struct<(array<22 x ptr>)> {
    %0 = llvm.mlir.addressof @_ZNK8tutorial6Person11GetMetadataEv : !llvm.ptr
    %1 = llvm.mlir.zero : !llvm.ptr
    %2 = llvm.mlir.addressof @__gxx_personality_v0 : !llvm.ptr
    %3 = llvm.mlir.addressof @_ZNK8tutorial6Person13SetCachedSizeEi : !llvm.ptr
    %4 = llvm.mlir.addressof @_ZNK6google8protobuf7Message13SpaceUsedLongEv : !llvm.ptr
    %5 = llvm.mlir.addressof @_ZN6google8protobuf7Message20DiscardUnknownFieldsEv : !llvm.ptr
    %6 = llvm.mlir.addressof @_ZN8tutorial6Person9MergeFromERKN6google8protobuf7MessageE : !llvm.ptr
    %7 = llvm.mlir.addressof @_ZN8tutorial6Person8CopyFromERKN6google8protobuf7MessageE : !llvm.ptr
    %8 = llvm.mlir.addressof @_ZNK6google8protobuf11MessageLite16InternalGetTableEv : !llvm.ptr
    %9 = llvm.mlir.addressof @_ZNK8tutorial6Person18_InternalSerializeEPhPN6google8protobuf2io19EpsCopyOutputStreamE : !llvm.ptr
    %10 = llvm.mlir.addressof @_ZN8tutorial6Person14_InternalParseEPKcPN6google8protobuf8internal12ParseContextE : !llvm.ptr
    %11 = llvm.mlir.addressof @_ZNK8tutorial6Person13GetCachedSizeEv : !llvm.ptr
    %12 = llvm.mlir.addressof @_ZNK8tutorial6Person12ByteSizeLongEv : !llvm.ptr
    %13 = llvm.mlir.addressof @_ZN6google8protobuf7Message21CheckTypeAndMergeFromERKNS0_11MessageLiteE : !llvm.ptr
    %14 = llvm.mlir.addressof @_ZNK6google8protobuf7Message25InitializationErrorStringB5cxx11Ev : !llvm.ptr
    %15 = llvm.mlir.addressof @_ZNK8tutorial6Person13IsInitializedEv : !llvm.ptr
    %16 = llvm.mlir.addressof @_ZN8tutorial6Person5ClearEv : !llvm.ptr
    %17 = llvm.mlir.addressof @_ZNK8tutorial6Person3NewEPN6google8protobuf5ArenaE : !llvm.ptr
    %18 = llvm.mlir.addressof @_ZNK8tutorial6Person3NewEv : !llvm.ptr
    %19 = llvm.mlir.addressof @_ZNK6google8protobuf7Message11GetTypeNameB5cxx11Ev : !llvm.ptr
    %20 = llvm.mlir.addressof @_ZN8tutorial6PersonD0Ev : !llvm.ptr
    %21 = llvm.mlir.addressof @_ZN8tutorial6PersonD2Ev : !llvm.ptr
    %22 = llvm.mlir.addressof @_ZTIN6google8protobuf7MessageE : !llvm.ptr
    %23 = llvm.mlir.constant("N8tutorial6PersonE\00") : !llvm.array<19 x i8>
    %24 = llvm.mlir.addressof @_ZTSN8tutorial6PersonE : !llvm.ptr
    %25 = llvm.mlir.constant(2 : i64) : i64
    %26 = llvm.mlir.addressof @_ZTVN10__cxxabiv120__si_class_type_infoE : !llvm.ptr
    %27 = llvm.getelementptr inbounds %26[%25] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.ptr
    %28 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, ptr)>
    %29 = llvm.insertvalue %27, %28[0] : !llvm.struct<(ptr, ptr, ptr)> 
    %30 = llvm.insertvalue %24, %29[1] : !llvm.struct<(ptr, ptr, ptr)> 
    %31 = llvm.insertvalue %22, %30[2] : !llvm.struct<(ptr, ptr, ptr)> 
    %32 = llvm.mlir.addressof @_ZTIN8tutorial6PersonE : !llvm.ptr
    %33 = llvm.mlir.undef : !llvm.array<22 x ptr>
    %34 = llvm.insertvalue %1, %33[0] : !llvm.array<22 x ptr> 
    %35 = llvm.insertvalue %32, %34[1] : !llvm.array<22 x ptr> 
    %36 = llvm.insertvalue %21, %35[2] : !llvm.array<22 x ptr> 
    %37 = llvm.insertvalue %20, %36[3] : !llvm.array<22 x ptr> 
    %38 = llvm.insertvalue %19, %37[4] : !llvm.array<22 x ptr> 
    %39 = llvm.insertvalue %18, %38[5] : !llvm.array<22 x ptr> 
    %40 = llvm.insertvalue %17, %39[6] : !llvm.array<22 x ptr> 
    %41 = llvm.insertvalue %16, %40[7] : !llvm.array<22 x ptr> 
    %42 = llvm.insertvalue %15, %41[8] : !llvm.array<22 x ptr> 
    %43 = llvm.insertvalue %14, %42[9] : !llvm.array<22 x ptr> 
    %44 = llvm.insertvalue %13, %43[10] : !llvm.array<22 x ptr> 
    %45 = llvm.insertvalue %12, %44[11] : !llvm.array<22 x ptr> 
    %46 = llvm.insertvalue %11, %45[12] : !llvm.array<22 x ptr> 
    %47 = llvm.insertvalue %10, %46[13] : !llvm.array<22 x ptr> 
    %48 = llvm.insertvalue %9, %47[14] : !llvm.array<22 x ptr> 
    %49 = llvm.insertvalue %8, %48[15] : !llvm.array<22 x ptr> 
    %50 = llvm.insertvalue %7, %49[16] : !llvm.array<22 x ptr> 
    %51 = llvm.insertvalue %6, %50[17] : !llvm.array<22 x ptr> 
    %52 = llvm.insertvalue %5, %51[18] : !llvm.array<22 x ptr> 
    %53 = llvm.insertvalue %4, %52[19] : !llvm.array<22 x ptr> 
    %54 = llvm.insertvalue %3, %53[20] : !llvm.array<22 x ptr> 
    %55 = llvm.insertvalue %0, %54[21] : !llvm.array<22 x ptr> 
    %56 = llvm.mlir.undef : !llvm.struct<(array<22 x ptr>)>
    %57 = llvm.insertvalue %55, %56[0] : !llvm.struct<(array<22 x ptr>)> 
    llvm.return %57 : !llvm.struct<(array<22 x ptr>)>
  }
  llvm.mlir.global private unnamed_addr constant @".str.7"("tutorial.Person.name\00") {addr_space = 0 : i32, alignment = 1 : i64, dso_local}
  llvm.mlir.global private unnamed_addr constant @".str.8"("tutorial.Person.email\00") {addr_space = 0 : i32, alignment = 1 : i64, dso_local}
  llvm.mlir.global external unnamed_addr constant @_ZTVN8tutorial11AddressBookE() {addr_space = 0 : i32, alignment = 8 : i64, dso_local} : !llvm.struct<(array<22 x ptr>)> {
    %0 = llvm.mlir.addressof @_ZNK8tutorial11AddressBook11GetMetadataEv : !llvm.ptr
    %1 = llvm.mlir.zero : !llvm.ptr
    %2 = llvm.mlir.addressof @__gxx_personality_v0 : !llvm.ptr
    %3 = llvm.mlir.addressof @_ZNK8tutorial11AddressBook13SetCachedSizeEi : !llvm.ptr
    %4 = llvm.mlir.addressof @_ZNK6google8protobuf7Message13SpaceUsedLongEv : !llvm.ptr
    %5 = llvm.mlir.addressof @_ZN6google8protobuf7Message20DiscardUnknownFieldsEv : !llvm.ptr
    %6 = llvm.mlir.addressof @_ZN8tutorial11AddressBook9MergeFromERKN6google8protobuf7MessageE : !llvm.ptr
    %7 = llvm.mlir.addressof @_ZN8tutorial11AddressBook8CopyFromERKN6google8protobuf7MessageE : !llvm.ptr
    %8 = llvm.mlir.addressof @_ZNK6google8protobuf11MessageLite16InternalGetTableEv : !llvm.ptr
    %9 = llvm.mlir.addressof @_ZNK8tutorial11AddressBook18_InternalSerializeEPhPN6google8protobuf2io19EpsCopyOutputStreamE : !llvm.ptr
    %10 = llvm.mlir.addressof @_ZN8tutorial11AddressBook14_InternalParseEPKcPN6google8protobuf8internal12ParseContextE : !llvm.ptr
    %11 = llvm.mlir.addressof @_ZNK8tutorial11AddressBook13GetCachedSizeEv : !llvm.ptr
    %12 = llvm.mlir.addressof @_ZNK8tutorial11AddressBook12ByteSizeLongEv : !llvm.ptr
    %13 = llvm.mlir.addressof @_ZN6google8protobuf7Message21CheckTypeAndMergeFromERKNS0_11MessageLiteE : !llvm.ptr
    %14 = llvm.mlir.addressof @_ZNK6google8protobuf7Message25InitializationErrorStringB5cxx11Ev : !llvm.ptr
    %15 = llvm.mlir.addressof @_ZNK8tutorial11AddressBook13IsInitializedEv : !llvm.ptr
    %16 = llvm.mlir.addressof @_ZN8tutorial11AddressBook5ClearEv : !llvm.ptr
    %17 = llvm.mlir.addressof @_ZNK8tutorial11AddressBook3NewEPN6google8protobuf5ArenaE : !llvm.ptr
    %18 = llvm.mlir.addressof @_ZNK8tutorial11AddressBook3NewEv : !llvm.ptr
    %19 = llvm.mlir.addressof @_ZNK6google8protobuf7Message11GetTypeNameB5cxx11Ev : !llvm.ptr
    %20 = llvm.mlir.addressof @_ZN8tutorial11AddressBookD0Ev : !llvm.ptr
    %21 = llvm.mlir.addressof @_ZN8tutorial11AddressBookD2Ev : !llvm.ptr
    %22 = llvm.mlir.addressof @_ZTIN6google8protobuf7MessageE : !llvm.ptr
    %23 = llvm.mlir.constant("N8tutorial11AddressBookE\00") : !llvm.array<25 x i8>
    %24 = llvm.mlir.addressof @_ZTSN8tutorial11AddressBookE : !llvm.ptr
    %25 = llvm.mlir.constant(2 : i64) : i64
    %26 = llvm.mlir.addressof @_ZTVN10__cxxabiv120__si_class_type_infoE : !llvm.ptr
    %27 = llvm.getelementptr inbounds %26[%25] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.ptr
    %28 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, ptr)>
    %29 = llvm.insertvalue %27, %28[0] : !llvm.struct<(ptr, ptr, ptr)> 
    %30 = llvm.insertvalue %24, %29[1] : !llvm.struct<(ptr, ptr, ptr)> 
    %31 = llvm.insertvalue %22, %30[2] : !llvm.struct<(ptr, ptr, ptr)> 
    %32 = llvm.mlir.addressof @_ZTIN8tutorial11AddressBookE : !llvm.ptr
    %33 = llvm.mlir.undef : !llvm.array<22 x ptr>
    %34 = llvm.insertvalue %1, %33[0] : !llvm.array<22 x ptr> 
    %35 = llvm.insertvalue %32, %34[1] : !llvm.array<22 x ptr> 
    %36 = llvm.insertvalue %21, %35[2] : !llvm.array<22 x ptr> 
    %37 = llvm.insertvalue %20, %36[3] : !llvm.array<22 x ptr> 
    %38 = llvm.insertvalue %19, %37[4] : !llvm.array<22 x ptr> 
    %39 = llvm.insertvalue %18, %38[5] : !llvm.array<22 x ptr> 
    %40 = llvm.insertvalue %17, %39[6] : !llvm.array<22 x ptr> 
    %41 = llvm.insertvalue %16, %40[7] : !llvm.array<22 x ptr> 
    %42 = llvm.insertvalue %15, %41[8] : !llvm.array<22 x ptr> 
    %43 = llvm.insertvalue %14, %42[9] : !llvm.array<22 x ptr> 
    %44 = llvm.insertvalue %13, %43[10] : !llvm.array<22 x ptr> 
    %45 = llvm.insertvalue %12, %44[11] : !llvm.array<22 x ptr> 
    %46 = llvm.insertvalue %11, %45[12] : !llvm.array<22 x ptr> 
    %47 = llvm.insertvalue %10, %46[13] : !llvm.array<22 x ptr> 
    %48 = llvm.insertvalue %9, %47[14] : !llvm.array<22 x ptr> 
    %49 = llvm.insertvalue %8, %48[15] : !llvm.array<22 x ptr> 
    %50 = llvm.insertvalue %7, %49[16] : !llvm.array<22 x ptr> 
    %51 = llvm.insertvalue %6, %50[17] : !llvm.array<22 x ptr> 
    %52 = llvm.insertvalue %5, %51[18] : !llvm.array<22 x ptr> 
    %53 = llvm.insertvalue %4, %52[19] : !llvm.array<22 x ptr> 
    %54 = llvm.insertvalue %3, %53[20] : !llvm.array<22 x ptr> 
    %55 = llvm.insertvalue %0, %54[21] : !llvm.array<22 x ptr> 
    %56 = llvm.mlir.undef : !llvm.struct<(array<22 x ptr>)>
    %57 = llvm.insertvalue %55, %56[0] : !llvm.struct<(array<22 x ptr>)> 
    llvm.return %57 : !llvm.struct<(array<22 x ptr>)>
  }
  llvm.mlir.global external @_ZTVN10__cxxabiv120__si_class_type_infoE() {addr_space = 0 : i32} : !llvm.ptr
  llvm.mlir.global external constant @_ZTSN8tutorial18Person_PhoneNumberE("N8tutorial18Person_PhoneNumberE\00") {addr_space = 0 : i32, alignment = 1 : i64, dso_local}
  llvm.mlir.global external constant @_ZTIN6google8protobuf7MessageE() {addr_space = 0 : i32} : !llvm.ptr
  llvm.mlir.global external constant @_ZTIN8tutorial18Person_PhoneNumberE() {addr_space = 0 : i32, alignment = 8 : i64, dso_local} : !llvm.struct<(ptr, ptr, ptr)> {
    %0 = llvm.mlir.addressof @_ZTIN6google8protobuf7MessageE : !llvm.ptr
    %1 = llvm.mlir.constant("N8tutorial18Person_PhoneNumberE\00") : !llvm.array<32 x i8>
    %2 = llvm.mlir.addressof @_ZTSN8tutorial18Person_PhoneNumberE : !llvm.ptr
    %3 = llvm.mlir.constant(2 : i64) : i64
    %4 = llvm.mlir.addressof @_ZTVN10__cxxabiv120__si_class_type_infoE : !llvm.ptr
    %5 = llvm.getelementptr inbounds %4[%3] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.ptr
    %6 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, ptr)>
    %7 = llvm.insertvalue %5, %6[0] : !llvm.struct<(ptr, ptr, ptr)> 
    %8 = llvm.insertvalue %2, %7[1] : !llvm.struct<(ptr, ptr, ptr)> 
    %9 = llvm.insertvalue %0, %8[2] : !llvm.struct<(ptr, ptr, ptr)> 
    llvm.return %9 : !llvm.struct<(ptr, ptr, ptr)>
  }
  llvm.mlir.global external constant @_ZTSN8tutorial6PersonE("N8tutorial6PersonE\00") {addr_space = 0 : i32, alignment = 1 : i64, dso_local}
  llvm.mlir.global external constant @_ZTIN8tutorial6PersonE() {addr_space = 0 : i32, alignment = 8 : i64, dso_local} : !llvm.struct<(ptr, ptr, ptr)> {
    %0 = llvm.mlir.addressof @_ZTIN6google8protobuf7MessageE : !llvm.ptr
    %1 = llvm.mlir.constant("N8tutorial6PersonE\00") : !llvm.array<19 x i8>
    %2 = llvm.mlir.addressof @_ZTSN8tutorial6PersonE : !llvm.ptr
    %3 = llvm.mlir.constant(2 : i64) : i64
    %4 = llvm.mlir.addressof @_ZTVN10__cxxabiv120__si_class_type_infoE : !llvm.ptr
    %5 = llvm.getelementptr inbounds %4[%3] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.ptr
    %6 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, ptr)>
    %7 = llvm.insertvalue %5, %6[0] : !llvm.struct<(ptr, ptr, ptr)> 
    %8 = llvm.insertvalue %2, %7[1] : !llvm.struct<(ptr, ptr, ptr)> 
    %9 = llvm.insertvalue %0, %8[2] : !llvm.struct<(ptr, ptr, ptr)> 
    llvm.return %9 : !llvm.struct<(ptr, ptr, ptr)>
  }
  llvm.mlir.global external constant @_ZTSN8tutorial11AddressBookE("N8tutorial11AddressBookE\00") {addr_space = 0 : i32, alignment = 1 : i64, dso_local}
  llvm.mlir.global external constant @_ZTIN8tutorial11AddressBookE() {addr_space = 0 : i32, alignment = 8 : i64, dso_local} : !llvm.struct<(ptr, ptr, ptr)> {
    %0 = llvm.mlir.addressof @_ZTIN6google8protobuf7MessageE : !llvm.ptr
    %1 = llvm.mlir.constant("N8tutorial11AddressBookE\00") : !llvm.array<25 x i8>
    %2 = llvm.mlir.addressof @_ZTSN8tutorial11AddressBookE : !llvm.ptr
    %3 = llvm.mlir.constant(2 : i64) : i64
    %4 = llvm.mlir.addressof @_ZTVN10__cxxabiv120__si_class_type_infoE : !llvm.ptr
    %5 = llvm.getelementptr inbounds %4[%3] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.ptr
    %6 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, ptr)>
    %7 = llvm.insertvalue %5, %6[0] : !llvm.struct<(ptr, ptr, ptr)> 
    %8 = llvm.insertvalue %2, %7[1] : !llvm.struct<(ptr, ptr, ptr)> 
    %9 = llvm.insertvalue %0, %8[2] : !llvm.struct<(ptr, ptr, ptr)> 
    llvm.return %9 : !llvm.struct<(ptr, ptr, ptr)>
  }
  llvm.mlir.global external @descriptor_table_google_2fprotobuf_2ftimestamp_2eproto() {addr_space = 0 : i32, alignment = 8 : i64} : !llvm.struct<"struct.google::protobuf::internal::DescriptorTable", (i8, i8, ptr, ptr, i32, ptr, ptr, ptr, i32, i32, ptr, ptr, ptr, ptr, i32, ptr, ptr)>
  llvm.mlir.global external @_ZN6google8protobuf8internal26fixed_address_empty_stringB5cxx11E() {addr_space = 0 : i32, alignment = 8 : i64} : !llvm.struct<"class.google::protobuf::internal::ExplicitlyConstructed.20", (struct<"union.google::protobuf::internal::ExplicitlyConstructed<std::__cxx11::basic_string<char>>::AlignedUnion", (i64, array<24 x i8>)>)>
  llvm.mlir.global private unnamed_addr constant @".str.9"("/usr/include/google/protobuf/arenastring.h\00") {addr_space = 0 : i32, alignment = 1 : i64, dso_local}
  llvm.mlir.global private unnamed_addr constant @".str.10"("CHECK failed: initial_value != NULL: \00") {addr_space = 0 : i32, alignment = 1 : i64, dso_local}
  llvm.mlir.global external @_ZTVN10__cxxabiv117__class_type_infoE() {addr_space = 0 : i32} : !llvm.ptr
  llvm.mlir.global linkonce_odr constant @_ZTSNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE("NSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE\00") comdat(@__llvm_global_comdat::@_ZTSNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE) {addr_space = 0 : i32, alignment = 1 : i64, dso_local}
  llvm.mlir.global linkonce_odr constant @_ZTINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE() comdat(@__llvm_global_comdat::@_ZTINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE) {addr_space = 0 : i32, alignment = 8 : i64, dso_local} : !llvm.struct<(ptr, ptr)> {
    %0 = llvm.mlir.constant("NSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE\00") : !llvm.array<53 x i8>
    %1 = llvm.mlir.addressof @_ZTSNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE : !llvm.ptr
    %2 = llvm.mlir.constant(2 : i64) : i64
    %3 = llvm.mlir.addressof @_ZTVN10__cxxabiv117__class_type_infoE : !llvm.ptr
    %4 = llvm.getelementptr inbounds %3[%2] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.ptr
    %5 = llvm.mlir.undef : !llvm.struct<(ptr, ptr)>
    %6 = llvm.insertvalue %4, %5[0] : !llvm.struct<(ptr, ptr)> 
    %7 = llvm.insertvalue %1, %6[1] : !llvm.struct<(ptr, ptr)> 
    llvm.return %7 : !llvm.struct<(ptr, ptr)>
  }
  llvm.mlir.global private unnamed_addr constant @".str.12"("CHECK failed: GetArena() == nullptr: \00") {addr_space = 0 : i32, alignment = 1 : i64, dso_local}
  llvm.mlir.global private unnamed_addr constant @".str.13"("/usr/include/google/protobuf/parse_context.h\00") {addr_space = 0 : i32, alignment = 1 : i64, dso_local}
  llvm.mlir.global private unnamed_addr constant @".str.14"("CHECK failed: *ptr: \00") {addr_space = 0 : i32, alignment = 1 : i64, dso_local}
  llvm.mlir.global private unnamed_addr constant @".str.15"("size_t to int conversion\00") {addr_space = 0 : i32, alignment = 1 : i64, dso_local}
  llvm.mlir.global external @_ZN6google8protobuf28_Timestamp_default_instance_E() {addr_space = 0 : i32, alignment = 1 : i64} : !llvm.struct<"class.google::protobuf::TimestampDefaultTypeInternal", opaque>
  llvm.mlir.global private unnamed_addr constant @".str.16"("/usr/include/google/protobuf/repeated_field.h\00") {addr_space = 0 : i32, alignment = 1 : i64, dso_local}
  llvm.mlir.global private unnamed_addr constant @".str.17"("CHECK failed: (index) >= (0): \00") {addr_space = 0 : i32, alignment = 1 : i64, dso_local}
  llvm.mlir.global private unnamed_addr constant @".str.18"("CHECK failed: (index) < (current_size_): \00") {addr_space = 0 : i32, alignment = 1 : i64, dso_local}
  llvm.mlir.global private unnamed_addr constant @".str.19"("CHECK failed: (n) >= (0): \00") {addr_space = 0 : i32, alignment = 1 : i64, dso_local}
  llvm.mlir.global linkonce_odr constant @_ZTSN6google8protobuf8internal16InternalMetadata9ContainerINS0_15UnknownFieldSetEEE("N6google8protobuf8internal16InternalMetadata9ContainerINS0_15UnknownFieldSetEEE\00") comdat(@__llvm_global_comdat::@_ZTSN6google8protobuf8internal16InternalMetadata9ContainerINS0_15UnknownFieldSetEEE) {addr_space = 0 : i32, alignment = 1 : i64, dso_local}
  llvm.mlir.global linkonce_odr constant @_ZTSN6google8protobuf8internal16InternalMetadata13ContainerBaseE("N6google8protobuf8internal16InternalMetadata13ContainerBaseE\00") comdat(@__llvm_global_comdat::@_ZTSN6google8protobuf8internal16InternalMetadata13ContainerBaseE) {addr_space = 0 : i32, alignment = 1 : i64, dso_local}
  llvm.mlir.global linkonce_odr constant @_ZTIN6google8protobuf8internal16InternalMetadata13ContainerBaseE() comdat(@__llvm_global_comdat::@_ZTIN6google8protobuf8internal16InternalMetadata13ContainerBaseE) {addr_space = 0 : i32, alignment = 8 : i64, dso_local} : !llvm.struct<(ptr, ptr)> {
    %0 = llvm.mlir.constant("N6google8protobuf8internal16InternalMetadata13ContainerBaseE\00") : !llvm.array<61 x i8>
    %1 = llvm.mlir.addressof @_ZTSN6google8protobuf8internal16InternalMetadata13ContainerBaseE : !llvm.ptr
    %2 = llvm.mlir.constant(2 : i64) : i64
    %3 = llvm.mlir.addressof @_ZTVN10__cxxabiv117__class_type_infoE : !llvm.ptr
    %4 = llvm.getelementptr inbounds %3[%2] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.ptr
    %5 = llvm.mlir.undef : !llvm.struct<(ptr, ptr)>
    %6 = llvm.insertvalue %4, %5[0] : !llvm.struct<(ptr, ptr)> 
    %7 = llvm.insertvalue %1, %6[1] : !llvm.struct<(ptr, ptr)> 
    llvm.return %7 : !llvm.struct<(ptr, ptr)>
  }
  llvm.mlir.global linkonce_odr constant @_ZTIN6google8protobuf8internal16InternalMetadata9ContainerINS0_15UnknownFieldSetEEE() comdat(@__llvm_global_comdat::@_ZTIN6google8protobuf8internal16InternalMetadata9ContainerINS0_15UnknownFieldSetEEE) {addr_space = 0 : i32, alignment = 8 : i64, dso_local} : !llvm.struct<(ptr, ptr, ptr)> {
    %0 = llvm.mlir.constant("N6google8protobuf8internal16InternalMetadata13ContainerBaseE\00") : !llvm.array<61 x i8>
    %1 = llvm.mlir.addressof @_ZTSN6google8protobuf8internal16InternalMetadata13ContainerBaseE : !llvm.ptr
    %2 = llvm.mlir.constant(2 : i64) : i64
    %3 = llvm.mlir.addressof @_ZTVN10__cxxabiv117__class_type_infoE : !llvm.ptr
    %4 = llvm.getelementptr inbounds %3[%2] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.ptr
    %5 = llvm.mlir.undef : !llvm.struct<(ptr, ptr)>
    %6 = llvm.insertvalue %4, %5[0] : !llvm.struct<(ptr, ptr)> 
    %7 = llvm.insertvalue %1, %6[1] : !llvm.struct<(ptr, ptr)> 
    %8 = llvm.mlir.addressof @_ZTIN6google8protobuf8internal16InternalMetadata13ContainerBaseE : !llvm.ptr
    %9 = llvm.mlir.constant("N6google8protobuf8internal16InternalMetadata9ContainerINS0_15UnknownFieldSetEEE\00") : !llvm.array<80 x i8>
    %10 = llvm.mlir.addressof @_ZTSN6google8protobuf8internal16InternalMetadata9ContainerINS0_15UnknownFieldSetEEE : !llvm.ptr
    %11 = llvm.mlir.addressof @_ZTVN10__cxxabiv120__si_class_type_infoE : !llvm.ptr
    %12 = llvm.getelementptr inbounds %11[%2] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.ptr
    %13 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, ptr)>
    %14 = llvm.insertvalue %12, %13[0] : !llvm.struct<(ptr, ptr, ptr)> 
    %15 = llvm.insertvalue %10, %14[1] : !llvm.struct<(ptr, ptr, ptr)> 
    %16 = llvm.insertvalue %8, %15[2] : !llvm.struct<(ptr, ptr, ptr)> 
    llvm.return %16 : !llvm.struct<(ptr, ptr, ptr)>
  }
  llvm.mlir.global private unnamed_addr constant @".str.20"("CHECK failed: limit >= 0 && limit <= INT_MAX - kSlopBytes: \00") {addr_space = 0 : i32, alignment = 1 : i64, dso_local}
  llvm.mlir.global private unnamed_addr constant @".str.21"("CHECK failed: (&other) != (this): \00") {addr_space = 0 : i32, alignment = 1 : i64, dso_local}
  llvm.mlir.global private unnamed_addr constant @".str.22"("CHECK failed: this != other: \00") {addr_space = 0 : i32, alignment = 1 : i64, dso_local}
  llvm.mlir.global private unnamed_addr constant @".str.23"("CHECK failed: GetArena() == other->GetArena(): \00") {addr_space = 0 : i32, alignment = 1 : i64, dso_local}
  llvm.mlir.global_ctors {ctors = [@_GLOBAL__sub_I_addressbook.pb.cc], priorities = [65535 : i32]}
  llvm.func unnamed_addr @_ZNSt8ios_base4InitC1Ev(!llvm.ptr {llvm.align = 1 : i64, llvm.dereferenceable = 1 : i64, llvm.nonnull, llvm.noundef}) attributes {frame_pointer = #llvm.framePointerKind<none>, passthrough = [["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], ["tune-cpu", "generic"]], target_cpu = "x86-64", target_features = #llvm.target_features<["+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>}
  llvm.func unnamed_addr @_ZNSt8ios_base4InitD1Ev(!llvm.ptr {llvm.align = 1 : i64, llvm.dereferenceable = 1 : i64, llvm.nonnull, llvm.noundef}) attributes {frame_pointer = #llvm.framePointerKind<none>, passthrough = ["nounwind", ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], ["tune-cpu", "generic"]], target_cpu = "x86-64", target_features = #llvm.target_features<["+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>}
  llvm.func local_unnamed_addr @__cxa_atexit(!llvm.ptr, !llvm.ptr, !llvm.ptr) -> i32 attributes {passthrough = ["nofree", "nounwind"]}
  llvm.func internal @_ZL52InitDefaultsscc_info_AddressBook_addressbook_2eprotov() attributes {dso_local, frame_pointer = #llvm.framePointerKind<none>, passthrough = [["uwtable", "2"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], ["tune-cpu", "generic"]], personality = @__gxx_personality_v0, target_cpu = "x86-64", target_features = #llvm.target_features<["+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>} {
    %0 = llvm.mlir.constant(3012004 : i32) : i32
    %1 = llvm.mlir.constant(3012000 : i32) : i32
    %2 = llvm.mlir.constant("build/addressbook.pb.cc\00") : !llvm.array<24 x i8>
    %3 = llvm.mlir.addressof @".str.5" : !llvm.ptr
    %4 = llvm.mlir.zero : !llvm.ptr
    %5 = llvm.mlir.constant(0 : i64) : i64
    %6 = llvm.mlir.constant(1 : i32) : i32
    %7 = llvm.mlir.constant(0 : i32) : i32
    %8 = llvm.mlir.constant(0 : i8) : i8
    %9 = llvm.mlir.constant(dense<0> : tensor<40xi8>) : !llvm.array<40 x i8>
    %10 = llvm.mlir.undef : !llvm.struct<"union.google::protobuf::internal::ExplicitlyConstructed<tutorial::AddressBook>::AlignedUnion", (i64, array<40 x i8>)>
    %11 = llvm.insertvalue %5, %10[0] : !llvm.struct<"union.google::protobuf::internal::ExplicitlyConstructed<tutorial::AddressBook>::AlignedUnion", (i64, array<40 x i8>)> 
    %12 = llvm.insertvalue %9, %11[1] : !llvm.struct<"union.google::protobuf::internal::ExplicitlyConstructed<tutorial::AddressBook>::AlignedUnion", (i64, array<40 x i8>)> 
    %13 = llvm.mlir.undef : !llvm.struct<"class.google::protobuf::internal::ExplicitlyConstructed.1", (struct<"union.google::protobuf::internal::ExplicitlyConstructed<tutorial::AddressBook>::AlignedUnion", (i64, array<40 x i8>)>)>
    %14 = llvm.insertvalue %12, %13[0] : !llvm.struct<"class.google::protobuf::internal::ExplicitlyConstructed.1", (struct<"union.google::protobuf::internal::ExplicitlyConstructed<tutorial::AddressBook>::AlignedUnion", (i64, array<40 x i8>)>)> 
    %15 = llvm.mlir.undef : !llvm.struct<"class.tutorial::AddressBookDefaultTypeInternal", (struct<"class.google::protobuf::internal::ExplicitlyConstructed.1", (struct<"union.google::protobuf::internal::ExplicitlyConstructed<tutorial::AddressBook>::AlignedUnion", (i64, array<40 x i8>)>)>)>
    %16 = llvm.insertvalue %14, %15[0] : !llvm.struct<"class.tutorial::AddressBookDefaultTypeInternal", (struct<"class.google::protobuf::internal::ExplicitlyConstructed.1", (struct<"union.google::protobuf::internal::ExplicitlyConstructed<tutorial::AddressBook>::AlignedUnion", (i64, array<40 x i8>)>)>)> 
    %17 = llvm.mlir.addressof @_ZN8tutorial30_AddressBook_default_instance_E : !llvm.ptr
    %18 = llvm.getelementptr inbounds %17[%5, 0, 0, 1, %5] : (!llvm.ptr, i64, i64) -> !llvm.ptr, !llvm.struct<"class.tutorial::AddressBookDefaultTypeInternal", (struct<"class.google::protobuf::internal::ExplicitlyConstructed.1", (struct<"union.google::protobuf::internal::ExplicitlyConstructed<tutorial::AddressBook>::AlignedUnion", (i64, array<40 x i8>)>)>)>
    %19 = llvm.mlir.constant(2 : i64) : i64
    %20 = llvm.mlir.addressof @_ZNK8tutorial11AddressBook11GetMetadataEv : !llvm.ptr
    %21 = llvm.mlir.addressof @__gxx_personality_v0 : !llvm.ptr
    %22 = llvm.mlir.addressof @_ZNK8tutorial11AddressBook13SetCachedSizeEi : !llvm.ptr
    %23 = llvm.mlir.addressof @_ZNK6google8protobuf7Message13SpaceUsedLongEv : !llvm.ptr
    %24 = llvm.mlir.addressof @_ZN6google8protobuf7Message20DiscardUnknownFieldsEv : !llvm.ptr
    %25 = llvm.mlir.addressof @_ZN8tutorial11AddressBook9MergeFromERKN6google8protobuf7MessageE : !llvm.ptr
    %26 = llvm.mlir.addressof @_ZN8tutorial11AddressBook8CopyFromERKN6google8protobuf7MessageE : !llvm.ptr
    %27 = llvm.mlir.addressof @_ZNK6google8protobuf11MessageLite16InternalGetTableEv : !llvm.ptr
    %28 = llvm.mlir.addressof @_ZNK8tutorial11AddressBook18_InternalSerializeEPhPN6google8protobuf2io19EpsCopyOutputStreamE : !llvm.ptr
    %29 = llvm.mlir.addressof @_ZN8tutorial11AddressBook14_InternalParseEPKcPN6google8protobuf8internal12ParseContextE : !llvm.ptr
    %30 = llvm.mlir.addressof @_ZNK8tutorial11AddressBook13GetCachedSizeEv : !llvm.ptr
    %31 = llvm.mlir.addressof @_ZNK8tutorial11AddressBook12ByteSizeLongEv : !llvm.ptr
    %32 = llvm.mlir.addressof @_ZN6google8protobuf7Message21CheckTypeAndMergeFromERKNS0_11MessageLiteE : !llvm.ptr
    %33 = llvm.mlir.addressof @_ZNK6google8protobuf7Message25InitializationErrorStringB5cxx11Ev : !llvm.ptr
    %34 = llvm.mlir.addressof @_ZNK8tutorial11AddressBook13IsInitializedEv : !llvm.ptr
    %35 = llvm.mlir.addressof @_ZN8tutorial11AddressBook5ClearEv : !llvm.ptr
    %36 = llvm.mlir.addressof @_ZNK8tutorial11AddressBook3NewEPN6google8protobuf5ArenaE : !llvm.ptr
    %37 = llvm.mlir.addressof @_ZNK8tutorial11AddressBook3NewEv : !llvm.ptr
    %38 = llvm.mlir.addressof @_ZNK6google8protobuf7Message11GetTypeNameB5cxx11Ev : !llvm.ptr
    %39 = llvm.mlir.addressof @_ZN8tutorial11AddressBookD0Ev : !llvm.ptr
    %40 = llvm.mlir.addressof @_ZN8tutorial11AddressBookD2Ev : !llvm.ptr
    %41 = llvm.mlir.addressof @_ZTIN6google8protobuf7MessageE : !llvm.ptr
    %42 = llvm.mlir.constant("N8tutorial11AddressBookE\00") : !llvm.array<25 x i8>
    %43 = llvm.mlir.addressof @_ZTSN8tutorial11AddressBookE : !llvm.ptr
    %44 = llvm.mlir.addressof @_ZTVN10__cxxabiv120__si_class_type_infoE : !llvm.ptr
    %45 = llvm.getelementptr inbounds %44[%19] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.ptr
    %46 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, ptr)>
    %47 = llvm.insertvalue %45, %46[0] : !llvm.struct<(ptr, ptr, ptr)> 
    %48 = llvm.insertvalue %43, %47[1] : !llvm.struct<(ptr, ptr, ptr)> 
    %49 = llvm.insertvalue %41, %48[2] : !llvm.struct<(ptr, ptr, ptr)> 
    %50 = llvm.mlir.addressof @_ZTIN8tutorial11AddressBookE : !llvm.ptr
    %51 = llvm.mlir.undef : !llvm.array<22 x ptr>
    %52 = llvm.insertvalue %4, %51[0] : !llvm.array<22 x ptr> 
    %53 = llvm.insertvalue %50, %52[1] : !llvm.array<22 x ptr> 
    %54 = llvm.insertvalue %40, %53[2] : !llvm.array<22 x ptr> 
    %55 = llvm.insertvalue %39, %54[3] : !llvm.array<22 x ptr> 
    %56 = llvm.insertvalue %38, %55[4] : !llvm.array<22 x ptr> 
    %57 = llvm.insertvalue %37, %56[5] : !llvm.array<22 x ptr> 
    %58 = llvm.insertvalue %36, %57[6] : !llvm.array<22 x ptr> 
    %59 = llvm.insertvalue %35, %58[7] : !llvm.array<22 x ptr> 
    %60 = llvm.insertvalue %34, %59[8] : !llvm.array<22 x ptr> 
    %61 = llvm.insertvalue %33, %60[9] : !llvm.array<22 x ptr> 
    %62 = llvm.insertvalue %32, %61[10] : !llvm.array<22 x ptr> 
    %63 = llvm.insertvalue %31, %62[11] : !llvm.array<22 x ptr> 
    %64 = llvm.insertvalue %30, %63[12] : !llvm.array<22 x ptr> 
    %65 = llvm.insertvalue %29, %64[13] : !llvm.array<22 x ptr> 
    %66 = llvm.insertvalue %28, %65[14] : !llvm.array<22 x ptr> 
    %67 = llvm.insertvalue %27, %66[15] : !llvm.array<22 x ptr> 
    %68 = llvm.insertvalue %26, %67[16] : !llvm.array<22 x ptr> 
    %69 = llvm.insertvalue %25, %68[17] : !llvm.array<22 x ptr> 
    %70 = llvm.insertvalue %24, %69[18] : !llvm.array<22 x ptr> 
    %71 = llvm.insertvalue %23, %70[19] : !llvm.array<22 x ptr> 
    %72 = llvm.insertvalue %22, %71[20] : !llvm.array<22 x ptr> 
    %73 = llvm.insertvalue %20, %72[21] : !llvm.array<22 x ptr> 
    %74 = llvm.mlir.undef : !llvm.struct<(array<22 x ptr>)>
    %75 = llvm.insertvalue %73, %74[0] : !llvm.struct<(array<22 x ptr>)> 
    %76 = llvm.mlir.addressof @_ZTVN8tutorial11AddressBookE : !llvm.ptr
    %77 = llvm.getelementptr inbounds %76[%5, 0, %19] : (!llvm.ptr, i64, i64) -> !llvm.ptr, !llvm.struct<(array<22 x ptr>)>
    %78 = llvm.mlir.constant(8 : i64) : i64
    %79 = llvm.getelementptr inbounds %17[%5, 0, 0, 1, %78] : (!llvm.ptr, i64, i64) -> !llvm.ptr, !llvm.struct<"class.tutorial::AddressBookDefaultTypeInternal", (struct<"class.google::protobuf::internal::ExplicitlyConstructed.1", (struct<"union.google::protobuf::internal::ExplicitlyConstructed<tutorial::AddressBook>::AlignedUnion", (i64, array<40 x i8>)>)>)>
    %80 = llvm.mlir.constant(28 : i64) : i64
    %81 = llvm.mlir.addressof @scc_info_Timestamp_google_2fprotobuf_2ftimestamp_2eproto : !llvm.ptr
    %82 = llvm.mlir.undef : !llvm.array<0 x ptr>
    %83 = llvm.mlir.addressof @_ZL59InitDefaultsscc_info_Person_PhoneNumber_addressbook_2eprotov : !llvm.ptr
    %84 = llvm.mlir.constant(-1 : i32) : i32
    %85 = llvm.mlir.undef : !llvm.struct<(i32)>
    %86 = llvm.insertvalue %84, %85[0] : !llvm.struct<(i32)> 
    %87 = llvm.mlir.undef : !llvm.struct<(struct<(i32)>, i32, i32, ptr)>
    %88 = llvm.insertvalue %86, %87[0] : !llvm.struct<(struct<(i32)>, i32, i32, ptr)> 
    %89 = llvm.insertvalue %7, %88[1] : !llvm.struct<(struct<(i32)>, i32, i32, ptr)> 
    %90 = llvm.insertvalue %7, %89[2] : !llvm.struct<(struct<(i32)>, i32, i32, ptr)> 
    %91 = llvm.insertvalue %83, %90[3] : !llvm.struct<(struct<(i32)>, i32, i32, ptr)> 
    %92 = llvm.mlir.undef : !llvm.struct<(struct<(struct<(i32)>, i32, i32, ptr)>, array<0 x ptr>)>
    %93 = llvm.insertvalue %91, %92[0] : !llvm.struct<(struct<(struct<(i32)>, i32, i32, ptr)>, array<0 x ptr>)> 
    %94 = llvm.insertvalue %82, %93[1] : !llvm.struct<(struct<(struct<(i32)>, i32, i32, ptr)>, array<0 x ptr>)> 
    %95 = llvm.mlir.addressof @scc_info_Person_PhoneNumber_addressbook_2eproto : !llvm.ptr
    %96 = llvm.mlir.undef : !llvm.array<2 x ptr>
    %97 = llvm.insertvalue %95, %96[0] : !llvm.array<2 x ptr> 
    %98 = llvm.insertvalue %81, %97[1] : !llvm.array<2 x ptr> 
    %99 = llvm.mlir.addressof @_ZL47InitDefaultsscc_info_Person_addressbook_2eprotov : !llvm.ptr
    %100 = llvm.mlir.constant(2 : i32) : i32
    %101 = llvm.mlir.undef : !llvm.struct<(struct<(i32)>, i32, i32, ptr)>
    %102 = llvm.insertvalue %86, %101[0] : !llvm.struct<(struct<(i32)>, i32, i32, ptr)> 
    %103 = llvm.insertvalue %100, %102[1] : !llvm.struct<(struct<(i32)>, i32, i32, ptr)> 
    %104 = llvm.insertvalue %7, %103[2] : !llvm.struct<(struct<(i32)>, i32, i32, ptr)> 
    %105 = llvm.insertvalue %99, %104[3] : !llvm.struct<(struct<(i32)>, i32, i32, ptr)> 
    %106 = llvm.mlir.undef : !llvm.struct<(struct<(struct<(i32)>, i32, i32, ptr)>, array<2 x ptr>)>
    %107 = llvm.insertvalue %105, %106[0] : !llvm.struct<(struct<(struct<(i32)>, i32, i32, ptr)>, array<2 x ptr>)> 
    %108 = llvm.insertvalue %98, %107[1] : !llvm.struct<(struct<(struct<(i32)>, i32, i32, ptr)>, array<2 x ptr>)> 
    %109 = llvm.mlir.addressof @scc_info_Person_addressbook_2eproto : !llvm.ptr
    %110 = llvm.mlir.undef : !llvm.array<1 x ptr>
    %111 = llvm.insertvalue %109, %110[0] : !llvm.array<1 x ptr> 
    %112 = llvm.mlir.addressof @_ZL52InitDefaultsscc_info_AddressBook_addressbook_2eprotov : !llvm.ptr
    %113 = llvm.mlir.undef : !llvm.struct<(struct<(i32)>, i32, i32, ptr)>
    %114 = llvm.insertvalue %86, %113[0] : !llvm.struct<(struct<(i32)>, i32, i32, ptr)> 
    %115 = llvm.insertvalue %6, %114[1] : !llvm.struct<(struct<(i32)>, i32, i32, ptr)> 
    %116 = llvm.insertvalue %7, %115[2] : !llvm.struct<(struct<(i32)>, i32, i32, ptr)> 
    %117 = llvm.insertvalue %112, %116[3] : !llvm.struct<(struct<(i32)>, i32, i32, ptr)> 
    %118 = llvm.mlir.undef : !llvm.struct<(struct<(struct<(i32)>, i32, i32, ptr)>, array<1 x ptr>)>
    %119 = llvm.insertvalue %117, %118[0] : !llvm.struct<(struct<(struct<(i32)>, i32, i32, ptr)>, array<1 x ptr>)> 
    %120 = llvm.insertvalue %111, %119[1] : !llvm.struct<(struct<(struct<(i32)>, i32, i32, ptr)>, array<1 x ptr>)> 
    %121 = llvm.mlir.addressof @scc_info_AddressBook_addressbook_2eproto : !llvm.ptr
    %122 = llvm.mlir.addressof @_ZN6google8protobuf8internal14DestroyMessageEPKv : !llvm.ptr
    llvm.call @_ZN6google8protobuf8internal13VerifyVersionEiiPKc(%0, %1, %3) : (i32, i32, !llvm.ptr) -> ()
    llvm.store %4, %18 {alignment = 8 : i64, tbaa = [#tbaa_tag5]} : !llvm.ptr, !llvm.ptr
    llvm.store %77, %17 {alignment = 8 : i64, tbaa = [#tbaa_tag]} : !llvm.ptr, !llvm.ptr
    "llvm.intr.memset"(%79, %8, %80) <{isVolatile = false}> : (!llvm.ptr, i8, i64) -> ()
    %123 = llvm.load %121 atomic acquire {alignment = 8 : i64} : !llvm.ptr -> i32
    %124 = llvm.icmp "eq" %123, %7 : i32
    llvm.cond_br %124 weights([2000, 1]), ^bb3, ^bb1
  ^bb1:  // pred: ^bb0
    llvm.invoke @_ZN6google8protobuf8internal11InitSCCImplEPNS1_11SCCInfoBaseE(%121) to ^bb3 unwind ^bb2 : (!llvm.ptr) -> ()
  ^bb2:  // pred: ^bb1
    %125 = llvm.landingpad cleanup : !llvm.struct<(ptr, i32)>
    llvm.call @_ZN6google8protobuf16RepeatedPtrFieldIN8tutorial6PersonEED2Ev(%79) : (!llvm.ptr) -> ()
    llvm.resume %125 : !llvm.struct<(ptr, i32)>
  ^bb3:  // 2 preds: ^bb0, ^bb1
    llvm.call @_ZN6google8protobuf8internal13OnShutdownRunEPFvPKvES3_(%122, %17) : (!llvm.ptr, !llvm.ptr) -> ()
    llvm.return
  }
  llvm.func internal @_ZL47InitDefaultsscc_info_Person_addressbook_2eprotov() attributes {dso_local, frame_pointer = #llvm.framePointerKind<none>, passthrough = [["uwtable", "2"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], ["tune-cpu", "generic"]], personality = @__gxx_personality_v0, target_cpu = "x86-64", target_features = #llvm.target_features<["+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>} {
    %0 = llvm.mlir.constant(3012004 : i32) : i32
    %1 = llvm.mlir.constant(3012000 : i32) : i32
    %2 = llvm.mlir.constant("build/addressbook.pb.cc\00") : !llvm.array<24 x i8>
    %3 = llvm.mlir.addressof @".str.5" : !llvm.ptr
    %4 = llvm.mlir.zero : !llvm.ptr
    %5 = llvm.mlir.constant(0 : i64) : i64
    %6 = llvm.mlir.constant(1 : i32) : i32
    %7 = llvm.mlir.constant(0 : i32) : i32
    %8 = llvm.mlir.constant(0 : i8) : i8
    %9 = llvm.mlir.constant(dense<0> : tensor<64xi8>) : !llvm.array<64 x i8>
    %10 = llvm.mlir.undef : !llvm.struct<"union.google::protobuf::internal::ExplicitlyConstructed<tutorial::Person>::AlignedUnion", (i64, array<64 x i8>)>
    %11 = llvm.insertvalue %5, %10[0] : !llvm.struct<"union.google::protobuf::internal::ExplicitlyConstructed<tutorial::Person>::AlignedUnion", (i64, array<64 x i8>)> 
    %12 = llvm.insertvalue %9, %11[1] : !llvm.struct<"union.google::protobuf::internal::ExplicitlyConstructed<tutorial::Person>::AlignedUnion", (i64, array<64 x i8>)> 
    %13 = llvm.mlir.undef : !llvm.struct<"class.google::protobuf::internal::ExplicitlyConstructed.0", (struct<"union.google::protobuf::internal::ExplicitlyConstructed<tutorial::Person>::AlignedUnion", (i64, array<64 x i8>)>)>
    %14 = llvm.insertvalue %12, %13[0] : !llvm.struct<"class.google::protobuf::internal::ExplicitlyConstructed.0", (struct<"union.google::protobuf::internal::ExplicitlyConstructed<tutorial::Person>::AlignedUnion", (i64, array<64 x i8>)>)> 
    %15 = llvm.mlir.undef : !llvm.struct<"class.tutorial::PersonDefaultTypeInternal", (struct<"class.google::protobuf::internal::ExplicitlyConstructed.0", (struct<"union.google::protobuf::internal::ExplicitlyConstructed<tutorial::Person>::AlignedUnion", (i64, array<64 x i8>)>)>)>
    %16 = llvm.insertvalue %14, %15[0] : !llvm.struct<"class.tutorial::PersonDefaultTypeInternal", (struct<"class.google::protobuf::internal::ExplicitlyConstructed.0", (struct<"union.google::protobuf::internal::ExplicitlyConstructed<tutorial::Person>::AlignedUnion", (i64, array<64 x i8>)>)>)> 
    %17 = llvm.mlir.addressof @_ZN8tutorial25_Person_default_instance_E : !llvm.ptr
    %18 = llvm.getelementptr inbounds %17[%5, 0, 0, 1, %5] : (!llvm.ptr, i64, i64) -> !llvm.ptr, !llvm.struct<"class.tutorial::PersonDefaultTypeInternal", (struct<"class.google::protobuf::internal::ExplicitlyConstructed.0", (struct<"union.google::protobuf::internal::ExplicitlyConstructed<tutorial::Person>::AlignedUnion", (i64, array<64 x i8>)>)>)>
    %19 = llvm.mlir.constant(2 : i64) : i64
    %20 = llvm.mlir.addressof @_ZNK8tutorial6Person11GetMetadataEv : !llvm.ptr
    %21 = llvm.mlir.addressof @__gxx_personality_v0 : !llvm.ptr
    %22 = llvm.mlir.addressof @_ZNK8tutorial6Person13SetCachedSizeEi : !llvm.ptr
    %23 = llvm.mlir.addressof @_ZNK6google8protobuf7Message13SpaceUsedLongEv : !llvm.ptr
    %24 = llvm.mlir.addressof @_ZN6google8protobuf7Message20DiscardUnknownFieldsEv : !llvm.ptr
    %25 = llvm.mlir.addressof @_ZN8tutorial6Person9MergeFromERKN6google8protobuf7MessageE : !llvm.ptr
    %26 = llvm.mlir.addressof @_ZN8tutorial6Person8CopyFromERKN6google8protobuf7MessageE : !llvm.ptr
    %27 = llvm.mlir.addressof @_ZNK6google8protobuf11MessageLite16InternalGetTableEv : !llvm.ptr
    %28 = llvm.mlir.addressof @_ZNK8tutorial6Person18_InternalSerializeEPhPN6google8protobuf2io19EpsCopyOutputStreamE : !llvm.ptr
    %29 = llvm.mlir.addressof @_ZN8tutorial6Person14_InternalParseEPKcPN6google8protobuf8internal12ParseContextE : !llvm.ptr
    %30 = llvm.mlir.addressof @_ZNK8tutorial6Person13GetCachedSizeEv : !llvm.ptr
    %31 = llvm.mlir.addressof @_ZNK8tutorial6Person12ByteSizeLongEv : !llvm.ptr
    %32 = llvm.mlir.addressof @_ZN6google8protobuf7Message21CheckTypeAndMergeFromERKNS0_11MessageLiteE : !llvm.ptr
    %33 = llvm.mlir.addressof @_ZNK6google8protobuf7Message25InitializationErrorStringB5cxx11Ev : !llvm.ptr
    %34 = llvm.mlir.addressof @_ZNK8tutorial6Person13IsInitializedEv : !llvm.ptr
    %35 = llvm.mlir.addressof @_ZN8tutorial6Person5ClearEv : !llvm.ptr
    %36 = llvm.mlir.addressof @_ZNK8tutorial6Person3NewEPN6google8protobuf5ArenaE : !llvm.ptr
    %37 = llvm.mlir.addressof @_ZNK8tutorial6Person3NewEv : !llvm.ptr
    %38 = llvm.mlir.addressof @_ZNK6google8protobuf7Message11GetTypeNameB5cxx11Ev : !llvm.ptr
    %39 = llvm.mlir.addressof @_ZN8tutorial6PersonD0Ev : !llvm.ptr
    %40 = llvm.mlir.addressof @_ZN8tutorial6PersonD2Ev : !llvm.ptr
    %41 = llvm.mlir.addressof @_ZTIN6google8protobuf7MessageE : !llvm.ptr
    %42 = llvm.mlir.constant("N8tutorial6PersonE\00") : !llvm.array<19 x i8>
    %43 = llvm.mlir.addressof @_ZTSN8tutorial6PersonE : !llvm.ptr
    %44 = llvm.mlir.addressof @_ZTVN10__cxxabiv120__si_class_type_infoE : !llvm.ptr
    %45 = llvm.getelementptr inbounds %44[%19] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.ptr
    %46 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, ptr)>
    %47 = llvm.insertvalue %45, %46[0] : !llvm.struct<(ptr, ptr, ptr)> 
    %48 = llvm.insertvalue %43, %47[1] : !llvm.struct<(ptr, ptr, ptr)> 
    %49 = llvm.insertvalue %41, %48[2] : !llvm.struct<(ptr, ptr, ptr)> 
    %50 = llvm.mlir.addressof @_ZTIN8tutorial6PersonE : !llvm.ptr
    %51 = llvm.mlir.undef : !llvm.array<22 x ptr>
    %52 = llvm.insertvalue %4, %51[0] : !llvm.array<22 x ptr> 
    %53 = llvm.insertvalue %50, %52[1] : !llvm.array<22 x ptr> 
    %54 = llvm.insertvalue %40, %53[2] : !llvm.array<22 x ptr> 
    %55 = llvm.insertvalue %39, %54[3] : !llvm.array<22 x ptr> 
    %56 = llvm.insertvalue %38, %55[4] : !llvm.array<22 x ptr> 
    %57 = llvm.insertvalue %37, %56[5] : !llvm.array<22 x ptr> 
    %58 = llvm.insertvalue %36, %57[6] : !llvm.array<22 x ptr> 
    %59 = llvm.insertvalue %35, %58[7] : !llvm.array<22 x ptr> 
    %60 = llvm.insertvalue %34, %59[8] : !llvm.array<22 x ptr> 
    %61 = llvm.insertvalue %33, %60[9] : !llvm.array<22 x ptr> 
    %62 = llvm.insertvalue %32, %61[10] : !llvm.array<22 x ptr> 
    %63 = llvm.insertvalue %31, %62[11] : !llvm.array<22 x ptr> 
    %64 = llvm.insertvalue %30, %63[12] : !llvm.array<22 x ptr> 
    %65 = llvm.insertvalue %29, %64[13] : !llvm.array<22 x ptr> 
    %66 = llvm.insertvalue %28, %65[14] : !llvm.array<22 x ptr> 
    %67 = llvm.insertvalue %27, %66[15] : !llvm.array<22 x ptr> 
    %68 = llvm.insertvalue %26, %67[16] : !llvm.array<22 x ptr> 
    %69 = llvm.insertvalue %25, %68[17] : !llvm.array<22 x ptr> 
    %70 = llvm.insertvalue %24, %69[18] : !llvm.array<22 x ptr> 
    %71 = llvm.insertvalue %23, %70[19] : !llvm.array<22 x ptr> 
    %72 = llvm.insertvalue %22, %71[20] : !llvm.array<22 x ptr> 
    %73 = llvm.insertvalue %20, %72[21] : !llvm.array<22 x ptr> 
    %74 = llvm.mlir.undef : !llvm.struct<(array<22 x ptr>)>
    %75 = llvm.insertvalue %73, %74[0] : !llvm.struct<(array<22 x ptr>)> 
    %76 = llvm.mlir.addressof @_ZTVN8tutorial6PersonE : !llvm.ptr
    %77 = llvm.getelementptr inbounds %76[%5, 0, %19] : (!llvm.ptr, i64, i64) -> !llvm.ptr, !llvm.struct<(array<22 x ptr>)>
    %78 = llvm.mlir.constant(60 : i64) : i64
    %79 = llvm.getelementptr inbounds %17[%5, 0, 0, 1, %78] : (!llvm.ptr, i64, i64) -> !llvm.ptr, !llvm.struct<"class.tutorial::PersonDefaultTypeInternal", (struct<"class.google::protobuf::internal::ExplicitlyConstructed.0", (struct<"union.google::protobuf::internal::ExplicitlyConstructed<tutorial::Person>::AlignedUnion", (i64, array<64 x i8>)>)>)>
    %80 = llvm.mlir.constant(8 : i64) : i64
    %81 = llvm.getelementptr inbounds %17[%5, 0, 0, 1, %80] : (!llvm.ptr, i64, i64) -> !llvm.ptr, !llvm.struct<"class.tutorial::PersonDefaultTypeInternal", (struct<"class.google::protobuf::internal::ExplicitlyConstructed.0", (struct<"union.google::protobuf::internal::ExplicitlyConstructed<tutorial::Person>::AlignedUnion", (i64, array<64 x i8>)>)>)>
    %82 = llvm.mlir.constant(24 : i64) : i64
    %83 = llvm.mlir.addressof @scc_info_Timestamp_google_2fprotobuf_2ftimestamp_2eproto : !llvm.ptr
    %84 = llvm.mlir.undef : !llvm.array<0 x ptr>
    %85 = llvm.mlir.addressof @_ZL59InitDefaultsscc_info_Person_PhoneNumber_addressbook_2eprotov : !llvm.ptr
    %86 = llvm.mlir.constant(-1 : i32) : i32
    %87 = llvm.mlir.undef : !llvm.struct<(i32)>
    %88 = llvm.insertvalue %86, %87[0] : !llvm.struct<(i32)> 
    %89 = llvm.mlir.undef : !llvm.struct<(struct<(i32)>, i32, i32, ptr)>
    %90 = llvm.insertvalue %88, %89[0] : !llvm.struct<(struct<(i32)>, i32, i32, ptr)> 
    %91 = llvm.insertvalue %7, %90[1] : !llvm.struct<(struct<(i32)>, i32, i32, ptr)> 
    %92 = llvm.insertvalue %7, %91[2] : !llvm.struct<(struct<(i32)>, i32, i32, ptr)> 
    %93 = llvm.insertvalue %85, %92[3] : !llvm.struct<(struct<(i32)>, i32, i32, ptr)> 
    %94 = llvm.mlir.undef : !llvm.struct<(struct<(struct<(i32)>, i32, i32, ptr)>, array<0 x ptr>)>
    %95 = llvm.insertvalue %93, %94[0] : !llvm.struct<(struct<(struct<(i32)>, i32, i32, ptr)>, array<0 x ptr>)> 
    %96 = llvm.insertvalue %84, %95[1] : !llvm.struct<(struct<(struct<(i32)>, i32, i32, ptr)>, array<0 x ptr>)> 
    %97 = llvm.mlir.addressof @scc_info_Person_PhoneNumber_addressbook_2eproto : !llvm.ptr
    %98 = llvm.mlir.undef : !llvm.array<2 x ptr>
    %99 = llvm.insertvalue %97, %98[0] : !llvm.array<2 x ptr> 
    %100 = llvm.insertvalue %83, %99[1] : !llvm.array<2 x ptr> 
    %101 = llvm.mlir.addressof @_ZL47InitDefaultsscc_info_Person_addressbook_2eprotov : !llvm.ptr
    %102 = llvm.mlir.constant(2 : i32) : i32
    %103 = llvm.mlir.undef : !llvm.struct<(struct<(i32)>, i32, i32, ptr)>
    %104 = llvm.insertvalue %88, %103[0] : !llvm.struct<(struct<(i32)>, i32, i32, ptr)> 
    %105 = llvm.insertvalue %102, %104[1] : !llvm.struct<(struct<(i32)>, i32, i32, ptr)> 
    %106 = llvm.insertvalue %7, %105[2] : !llvm.struct<(struct<(i32)>, i32, i32, ptr)> 
    %107 = llvm.insertvalue %101, %106[3] : !llvm.struct<(struct<(i32)>, i32, i32, ptr)> 
    %108 = llvm.mlir.undef : !llvm.struct<(struct<(struct<(i32)>, i32, i32, ptr)>, array<2 x ptr>)>
    %109 = llvm.insertvalue %107, %108[0] : !llvm.struct<(struct<(struct<(i32)>, i32, i32, ptr)>, array<2 x ptr>)> 
    %110 = llvm.insertvalue %100, %109[1] : !llvm.struct<(struct<(struct<(i32)>, i32, i32, ptr)>, array<2 x ptr>)> 
    %111 = llvm.mlir.addressof @scc_info_Person_addressbook_2eproto : !llvm.ptr
    %112 = llvm.mlir.addressof @_ZN6google8protobuf8internal26fixed_address_empty_stringB5cxx11E : !llvm.ptr
    %113 = llvm.mlir.constant(32 : i64) : i64
    %114 = llvm.getelementptr inbounds %17[%5, 0, 0, 1, %113] : (!llvm.ptr, i64, i64) -> !llvm.ptr, !llvm.struct<"class.tutorial::PersonDefaultTypeInternal", (struct<"class.google::protobuf::internal::ExplicitlyConstructed.0", (struct<"union.google::protobuf::internal::ExplicitlyConstructed<tutorial::Person>::AlignedUnion", (i64, array<64 x i8>)>)>)>
    %115 = llvm.mlir.constant(40 : i64) : i64
    %116 = llvm.getelementptr inbounds %17[%5, 0, 0, 1, %115] : (!llvm.ptr, i64, i64) -> !llvm.ptr, !llvm.struct<"class.tutorial::PersonDefaultTypeInternal", (struct<"class.google::protobuf::internal::ExplicitlyConstructed.0", (struct<"union.google::protobuf::internal::ExplicitlyConstructed<tutorial::Person>::AlignedUnion", (i64, array<64 x i8>)>)>)>
    %117 = llvm.mlir.constant(48 : i64) : i64
    %118 = llvm.getelementptr inbounds %17[%5, 0, 0, 1, %117] : (!llvm.ptr, i64, i64) -> !llvm.ptr, !llvm.struct<"class.tutorial::PersonDefaultTypeInternal", (struct<"class.google::protobuf::internal::ExplicitlyConstructed.0", (struct<"union.google::protobuf::internal::ExplicitlyConstructed<tutorial::Person>::AlignedUnion", (i64, array<64 x i8>)>)>)>
    %119 = llvm.mlir.constant(12 : i64) : i64
    %120 = llvm.mlir.addressof @_ZN6google8protobuf8internal14DestroyMessageEPKv : !llvm.ptr
    %121 = llvm.mlir.addressof @_ZN6google8protobuf28_Timestamp_default_instance_E : !llvm.ptr
    llvm.call @_ZN6google8protobuf8internal13VerifyVersionEiiPKc(%0, %1, %3) : (i32, i32, !llvm.ptr) -> ()
    llvm.store %4, %18 {alignment = 8 : i64, tbaa = [#tbaa_tag5]} : !llvm.ptr, !llvm.ptr
    llvm.store %77, %17 {alignment = 8 : i64, tbaa = [#tbaa_tag]} : !llvm.ptr, !llvm.ptr
    llvm.store %7, %79 {alignment = 4 : i64, tbaa = [#tbaa_tag6]} : i32, !llvm.ptr
    "llvm.intr.memset"(%81, %8, %82) <{isVolatile = false}> : (!llvm.ptr, i8, i64) -> ()
    %122 = llvm.load %111 atomic acquire {alignment = 8 : i64} : !llvm.ptr -> i32
    %123 = llvm.icmp "eq" %122, %7 : i32
    llvm.cond_br %123 weights([2000, 1]), ^bb8, ^bb1
  ^bb1:  // pred: ^bb0
    llvm.invoke @_ZN6google8protobuf8internal11InitSCCImplEPNS1_11SCCInfoBaseE(%111) to ^bb8 unwind ^bb2 : (!llvm.ptr) -> ()
  ^bb2:  // pred: ^bb1
    %124 = llvm.landingpad cleanup : !llvm.struct<(ptr, i32)>
    llvm.invoke @_ZN6google8protobuf8internal20RepeatedPtrFieldBase7DestroyINS0_16RepeatedPtrFieldIN8tutorial18Person_PhoneNumberEE11TypeHandlerEEEvv(%81) to ^bb3 unwind ^bb6 : (!llvm.ptr) -> ()
  ^bb3:  // pred: ^bb2
    %125 = llvm.load %81 {alignment = 8 : i64, tbaa = [#tbaa_tag7]} : !llvm.ptr -> !llvm.ptr
    %126 = llvm.icmp "eq" %125, %4 : !llvm.ptr
    llvm.cond_br %126, ^bb7, ^bb4
  ^bb4:  // pred: ^bb3
    %127 = llvm.getelementptr inbounds %125[%5, 0] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.google::protobuf::Arena", (struct<"class.google::protobuf::internal::ArenaImpl", (struct<"struct.std::atomic.10", (struct<"struct.std::__atomic_base.11", (ptr)>)>, struct<"struct.std::atomic.10", (struct<"struct.std::__atomic_base.11", (ptr)>)>, struct<"struct.std::atomic.12", (struct<"struct.std::__atomic_base.13", (i64)>)>, ptr, i64, struct<"struct.google::protobuf::internal::ArenaImpl::Options", (i64, i64, ptr, i64, ptr, ptr)>)>, ptr, ptr, ptr, ptr)>
    %128 = llvm.invoke @_ZNK6google8protobuf8internal9ArenaImpl14SpaceAllocatedEv(%127) to ^bb7 unwind ^bb5 : (!llvm.ptr) -> i64
  ^bb5:  // pred: ^bb4
    %129 = llvm.landingpad (catch %4 : !llvm.ptr) : !llvm.struct<(ptr, i32)>
    %130 = llvm.extractvalue %129[0] : !llvm.struct<(ptr, i32)> 
    llvm.call @__clang_call_terminate(%130) : (!llvm.ptr) -> ()
    llvm.unreachable
  ^bb6:  // pred: ^bb2
    %131 = llvm.landingpad (catch %4 : !llvm.ptr) : !llvm.struct<(ptr, i32)>
    %132 = llvm.extractvalue %131[0] : !llvm.struct<(ptr, i32)> 
    llvm.call @_ZN6google8protobuf8internal20RepeatedPtrFieldBaseD2Ev(%81) : (!llvm.ptr) -> ()
    llvm.call @__clang_call_terminate(%132) : (!llvm.ptr) -> ()
    llvm.unreachable
  ^bb7:  // 2 preds: ^bb3, ^bb4
    llvm.resume %124 : !llvm.struct<(ptr, i32)>
  ^bb8:  // 2 preds: ^bb0, ^bb1
    llvm.store %112, %114 {alignment = 8 : i64, tbaa = [#tbaa_tag8]} : !llvm.ptr, !llvm.ptr
    llvm.store %112, %116 {alignment = 8 : i64, tbaa = [#tbaa_tag8]} : !llvm.ptr, !llvm.ptr
    "llvm.intr.memset"(%118, %8, %119) <{isVolatile = false}> : (!llvm.ptr, i8, i64) -> ()
    llvm.call @_ZN6google8protobuf8internal13OnShutdownRunEPFvPKvES3_(%120, %17) : (!llvm.ptr, !llvm.ptr) -> ()
    llvm.store %121, %118 {alignment = 8 : i64, tbaa = [#tbaa_tag23]} : !llvm.ptr, !llvm.ptr
    llvm.return
  }
  llvm.func internal @_ZL59InitDefaultsscc_info_Person_PhoneNumber_addressbook_2eprotov() attributes {dso_local, frame_pointer = #llvm.framePointerKind<none>, passthrough = [["uwtable", "2"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], ["tune-cpu", "generic"]], personality = @__gxx_personality_v0, target_cpu = "x86-64", target_features = #llvm.target_features<["+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>} {
    %0 = llvm.mlir.constant(3012004 : i32) : i32
    %1 = llvm.mlir.constant(3012000 : i32) : i32
    %2 = llvm.mlir.constant("build/addressbook.pb.cc\00") : !llvm.array<24 x i8>
    %3 = llvm.mlir.addressof @".str.5" : !llvm.ptr
    %4 = llvm.mlir.zero : !llvm.ptr
    %5 = llvm.mlir.constant(0 : i64) : i64
    %6 = llvm.mlir.constant(1 : i32) : i32
    %7 = llvm.mlir.constant(0 : i32) : i32
    %8 = llvm.mlir.constant(0 : i8) : i8
    %9 = llvm.mlir.constant(dense<0> : tensor<24xi8>) : !llvm.array<24 x i8>
    %10 = llvm.mlir.undef : !llvm.struct<"union.google::protobuf::internal::ExplicitlyConstructed<tutorial::Person_PhoneNumber>::AlignedUnion", (i64, array<24 x i8>)>
    %11 = llvm.insertvalue %5, %10[0] : !llvm.struct<"union.google::protobuf::internal::ExplicitlyConstructed<tutorial::Person_PhoneNumber>::AlignedUnion", (i64, array<24 x i8>)> 
    %12 = llvm.insertvalue %9, %11[1] : !llvm.struct<"union.google::protobuf::internal::ExplicitlyConstructed<tutorial::Person_PhoneNumber>::AlignedUnion", (i64, array<24 x i8>)> 
    %13 = llvm.mlir.undef : !llvm.struct<"class.google::protobuf::internal::ExplicitlyConstructed", (struct<"union.google::protobuf::internal::ExplicitlyConstructed<tutorial::Person_PhoneNumber>::AlignedUnion", (i64, array<24 x i8>)>)>
    %14 = llvm.insertvalue %12, %13[0] : !llvm.struct<"class.google::protobuf::internal::ExplicitlyConstructed", (struct<"union.google::protobuf::internal::ExplicitlyConstructed<tutorial::Person_PhoneNumber>::AlignedUnion", (i64, array<24 x i8>)>)> 
    %15 = llvm.mlir.undef : !llvm.struct<"class.tutorial::Person_PhoneNumberDefaultTypeInternal", (struct<"class.google::protobuf::internal::ExplicitlyConstructed", (struct<"union.google::protobuf::internal::ExplicitlyConstructed<tutorial::Person_PhoneNumber>::AlignedUnion", (i64, array<24 x i8>)>)>)>
    %16 = llvm.insertvalue %14, %15[0] : !llvm.struct<"class.tutorial::Person_PhoneNumberDefaultTypeInternal", (struct<"class.google::protobuf::internal::ExplicitlyConstructed", (struct<"union.google::protobuf::internal::ExplicitlyConstructed<tutorial::Person_PhoneNumber>::AlignedUnion", (i64, array<24 x i8>)>)>)> 
    %17 = llvm.mlir.addressof @_ZN8tutorial37_Person_PhoneNumber_default_instance_E : !llvm.ptr
    %18 = llvm.getelementptr inbounds %17[%5, 0, 0, 1, %5] : (!llvm.ptr, i64, i64) -> !llvm.ptr, !llvm.struct<"class.tutorial::Person_PhoneNumberDefaultTypeInternal", (struct<"class.google::protobuf::internal::ExplicitlyConstructed", (struct<"union.google::protobuf::internal::ExplicitlyConstructed<tutorial::Person_PhoneNumber>::AlignedUnion", (i64, array<24 x i8>)>)>)>
    %19 = llvm.mlir.constant(2 : i64) : i64
    %20 = llvm.mlir.addressof @_ZNK8tutorial18Person_PhoneNumber11GetMetadataEv : !llvm.ptr
    %21 = llvm.mlir.addressof @__gxx_personality_v0 : !llvm.ptr
    %22 = llvm.mlir.addressof @_ZNK8tutorial18Person_PhoneNumber13SetCachedSizeEi : !llvm.ptr
    %23 = llvm.mlir.addressof @_ZNK6google8protobuf7Message13SpaceUsedLongEv : !llvm.ptr
    %24 = llvm.mlir.addressof @_ZN6google8protobuf7Message20DiscardUnknownFieldsEv : !llvm.ptr
    %25 = llvm.mlir.addressof @_ZN8tutorial18Person_PhoneNumber9MergeFromERKN6google8protobuf7MessageE : !llvm.ptr
    %26 = llvm.mlir.addressof @_ZN8tutorial18Person_PhoneNumber8CopyFromERKN6google8protobuf7MessageE : !llvm.ptr
    %27 = llvm.mlir.addressof @_ZNK6google8protobuf11MessageLite16InternalGetTableEv : !llvm.ptr
    %28 = llvm.mlir.addressof @_ZNK8tutorial18Person_PhoneNumber18_InternalSerializeEPhPN6google8protobuf2io19EpsCopyOutputStreamE : !llvm.ptr
    %29 = llvm.mlir.addressof @_ZN8tutorial18Person_PhoneNumber14_InternalParseEPKcPN6google8protobuf8internal12ParseContextE : !llvm.ptr
    %30 = llvm.mlir.addressof @_ZNK8tutorial18Person_PhoneNumber13GetCachedSizeEv : !llvm.ptr
    %31 = llvm.mlir.addressof @_ZNK8tutorial18Person_PhoneNumber12ByteSizeLongEv : !llvm.ptr
    %32 = llvm.mlir.addressof @_ZN6google8protobuf7Message21CheckTypeAndMergeFromERKNS0_11MessageLiteE : !llvm.ptr
    %33 = llvm.mlir.addressof @_ZNK6google8protobuf7Message25InitializationErrorStringB5cxx11Ev : !llvm.ptr
    %34 = llvm.mlir.addressof @_ZNK8tutorial18Person_PhoneNumber13IsInitializedEv : !llvm.ptr
    %35 = llvm.mlir.addressof @_ZN8tutorial18Person_PhoneNumber5ClearEv : !llvm.ptr
    %36 = llvm.mlir.addressof @_ZNK8tutorial18Person_PhoneNumber3NewEPN6google8protobuf5ArenaE : !llvm.ptr
    %37 = llvm.mlir.addressof @_ZNK8tutorial18Person_PhoneNumber3NewEv : !llvm.ptr
    %38 = llvm.mlir.addressof @_ZNK6google8protobuf7Message11GetTypeNameB5cxx11Ev : !llvm.ptr
    %39 = llvm.mlir.addressof @_ZN8tutorial18Person_PhoneNumberD0Ev : !llvm.ptr
    %40 = llvm.mlir.addressof @_ZN8tutorial18Person_PhoneNumberD2Ev : !llvm.ptr
    %41 = llvm.mlir.addressof @_ZTIN6google8protobuf7MessageE : !llvm.ptr
    %42 = llvm.mlir.constant("N8tutorial18Person_PhoneNumberE\00") : !llvm.array<32 x i8>
    %43 = llvm.mlir.addressof @_ZTSN8tutorial18Person_PhoneNumberE : !llvm.ptr
    %44 = llvm.mlir.addressof @_ZTVN10__cxxabiv120__si_class_type_infoE : !llvm.ptr
    %45 = llvm.getelementptr inbounds %44[%19] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.ptr
    %46 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, ptr)>
    %47 = llvm.insertvalue %45, %46[0] : !llvm.struct<(ptr, ptr, ptr)> 
    %48 = llvm.insertvalue %43, %47[1] : !llvm.struct<(ptr, ptr, ptr)> 
    %49 = llvm.insertvalue %41, %48[2] : !llvm.struct<(ptr, ptr, ptr)> 
    %50 = llvm.mlir.addressof @_ZTIN8tutorial18Person_PhoneNumberE : !llvm.ptr
    %51 = llvm.mlir.undef : !llvm.array<22 x ptr>
    %52 = llvm.insertvalue %4, %51[0] : !llvm.array<22 x ptr> 
    %53 = llvm.insertvalue %50, %52[1] : !llvm.array<22 x ptr> 
    %54 = llvm.insertvalue %40, %53[2] : !llvm.array<22 x ptr> 
    %55 = llvm.insertvalue %39, %54[3] : !llvm.array<22 x ptr> 
    %56 = llvm.insertvalue %38, %55[4] : !llvm.array<22 x ptr> 
    %57 = llvm.insertvalue %37, %56[5] : !llvm.array<22 x ptr> 
    %58 = llvm.insertvalue %36, %57[6] : !llvm.array<22 x ptr> 
    %59 = llvm.insertvalue %35, %58[7] : !llvm.array<22 x ptr> 
    %60 = llvm.insertvalue %34, %59[8] : !llvm.array<22 x ptr> 
    %61 = llvm.insertvalue %33, %60[9] : !llvm.array<22 x ptr> 
    %62 = llvm.insertvalue %32, %61[10] : !llvm.array<22 x ptr> 
    %63 = llvm.insertvalue %31, %62[11] : !llvm.array<22 x ptr> 
    %64 = llvm.insertvalue %30, %63[12] : !llvm.array<22 x ptr> 
    %65 = llvm.insertvalue %29, %64[13] : !llvm.array<22 x ptr> 
    %66 = llvm.insertvalue %28, %65[14] : !llvm.array<22 x ptr> 
    %67 = llvm.insertvalue %27, %66[15] : !llvm.array<22 x ptr> 
    %68 = llvm.insertvalue %26, %67[16] : !llvm.array<22 x ptr> 
    %69 = llvm.insertvalue %25, %68[17] : !llvm.array<22 x ptr> 
    %70 = llvm.insertvalue %24, %69[18] : !llvm.array<22 x ptr> 
    %71 = llvm.insertvalue %23, %70[19] : !llvm.array<22 x ptr> 
    %72 = llvm.insertvalue %22, %71[20] : !llvm.array<22 x ptr> 
    %73 = llvm.insertvalue %20, %72[21] : !llvm.array<22 x ptr> 
    %74 = llvm.mlir.undef : !llvm.struct<(array<22 x ptr>)>
    %75 = llvm.insertvalue %73, %74[0] : !llvm.struct<(array<22 x ptr>)> 
    %76 = llvm.mlir.addressof @_ZTVN8tutorial18Person_PhoneNumberE : !llvm.ptr
    %77 = llvm.getelementptr inbounds %76[%5, 0, %19] : (!llvm.ptr, i64, i64) -> !llvm.ptr, !llvm.struct<(array<22 x ptr>)>
    %78 = llvm.mlir.constant(20 : i64) : i64
    %79 = llvm.getelementptr inbounds %17[%5, 0, 0, 1, %78] : (!llvm.ptr, i64, i64) -> !llvm.ptr, !llvm.struct<"class.tutorial::Person_PhoneNumberDefaultTypeInternal", (struct<"class.google::protobuf::internal::ExplicitlyConstructed", (struct<"union.google::protobuf::internal::ExplicitlyConstructed<tutorial::Person_PhoneNumber>::AlignedUnion", (i64, array<24 x i8>)>)>)>
    %80 = llvm.mlir.undef : !llvm.array<0 x ptr>
    %81 = llvm.mlir.addressof @_ZL59InitDefaultsscc_info_Person_PhoneNumber_addressbook_2eprotov : !llvm.ptr
    %82 = llvm.mlir.constant(-1 : i32) : i32
    %83 = llvm.mlir.undef : !llvm.struct<(i32)>
    %84 = llvm.insertvalue %82, %83[0] : !llvm.struct<(i32)> 
    %85 = llvm.mlir.undef : !llvm.struct<(struct<(i32)>, i32, i32, ptr)>
    %86 = llvm.insertvalue %84, %85[0] : !llvm.struct<(struct<(i32)>, i32, i32, ptr)> 
    %87 = llvm.insertvalue %7, %86[1] : !llvm.struct<(struct<(i32)>, i32, i32, ptr)> 
    %88 = llvm.insertvalue %7, %87[2] : !llvm.struct<(struct<(i32)>, i32, i32, ptr)> 
    %89 = llvm.insertvalue %81, %88[3] : !llvm.struct<(struct<(i32)>, i32, i32, ptr)> 
    %90 = llvm.mlir.undef : !llvm.struct<(struct<(struct<(i32)>, i32, i32, ptr)>, array<0 x ptr>)>
    %91 = llvm.insertvalue %89, %90[0] : !llvm.struct<(struct<(struct<(i32)>, i32, i32, ptr)>, array<0 x ptr>)> 
    %92 = llvm.insertvalue %80, %91[1] : !llvm.struct<(struct<(struct<(i32)>, i32, i32, ptr)>, array<0 x ptr>)> 
    %93 = llvm.mlir.addressof @scc_info_Person_PhoneNumber_addressbook_2eproto : !llvm.ptr
    %94 = llvm.mlir.addressof @_ZN6google8protobuf8internal26fixed_address_empty_stringB5cxx11E : !llvm.ptr
    %95 = llvm.mlir.constant(8 : i64) : i64
    %96 = llvm.getelementptr inbounds %17[%5, 0, 0, 1, %95] : (!llvm.ptr, i64, i64) -> !llvm.ptr, !llvm.struct<"class.tutorial::Person_PhoneNumberDefaultTypeInternal", (struct<"class.google::protobuf::internal::ExplicitlyConstructed", (struct<"union.google::protobuf::internal::ExplicitlyConstructed<tutorial::Person_PhoneNumber>::AlignedUnion", (i64, array<24 x i8>)>)>)>
    %97 = llvm.mlir.constant(16 : i64) : i64
    %98 = llvm.getelementptr inbounds %17[%5, 0, 0, 1, %97] : (!llvm.ptr, i64, i64) -> !llvm.ptr, !llvm.struct<"class.tutorial::Person_PhoneNumberDefaultTypeInternal", (struct<"class.google::protobuf::internal::ExplicitlyConstructed", (struct<"union.google::protobuf::internal::ExplicitlyConstructed<tutorial::Person_PhoneNumber>::AlignedUnion", (i64, array<24 x i8>)>)>)>
    %99 = llvm.mlir.addressof @_ZN6google8protobuf8internal14DestroyMessageEPKv : !llvm.ptr
    llvm.call @_ZN6google8protobuf8internal13VerifyVersionEiiPKc(%0, %1, %3) : (i32, i32, !llvm.ptr) -> ()
    llvm.store %4, %18 {alignment = 8 : i64, tbaa = [#tbaa_tag5]} : !llvm.ptr, !llvm.ptr
    llvm.store %77, %17 {alignment = 8 : i64, tbaa = [#tbaa_tag]} : !llvm.ptr, !llvm.ptr
    llvm.store %7, %79 {alignment = 4 : i64, tbaa = [#tbaa_tag6]} : i32, !llvm.ptr
    %100 = llvm.load %93 atomic acquire {alignment = 8 : i64} : !llvm.ptr -> i32
    %101 = llvm.icmp "eq" %100, %7 : i32
    llvm.cond_br %101 weights([2000, 1]), ^bb2, ^bb1
  ^bb1:  // pred: ^bb0
    llvm.call @_ZN6google8protobuf8internal11InitSCCImplEPNS1_11SCCInfoBaseE(%93) : (!llvm.ptr) -> ()
    llvm.br ^bb2
  ^bb2:  // 2 preds: ^bb0, ^bb1
    llvm.store %94, %96 {alignment = 8 : i64, tbaa = [#tbaa_tag8]} : !llvm.ptr, !llvm.ptr
    llvm.store %7, %98 {alignment = 8 : i64, tbaa = [#tbaa_tag24]} : i32, !llvm.ptr
    llvm.call @_ZN6google8protobuf8internal13OnShutdownRunEPFvPKvES3_(%99, %17) : (!llvm.ptr, !llvm.ptr) -> ()
    llvm.return
  }
  llvm.func local_unnamed_addr @_ZN6google8protobuf8internal14AddDescriptorsEPKNS1_15DescriptorTableE(!llvm.ptr {llvm.noundef}) attributes {frame_pointer = #llvm.framePointerKind<none>, passthrough = [["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], ["tune-cpu", "generic"]], target_cpu = "x86-64", target_features = #llvm.target_features<["+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>}
  llvm.func local_unnamed_addr @_ZN8tutorial27Person_PhoneType_descriptorEv() -> (!llvm.ptr {llvm.noundef}) attributes {frame_pointer = #llvm.framePointerKind<none>, passthrough = ["mustprogress", ["uwtable", "2"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], ["tune-cpu", "generic"]], target_cpu = "x86-64", target_features = #llvm.target_features<["+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>} {
    %0 = llvm.mlir.zero : !llvm.ptr
    %1 = llvm.mlir.undef : !llvm.array<1 x ptr>
    %2 = llvm.insertvalue %0, %1[0] : !llvm.array<1 x ptr> 
    %3 = llvm.mlir.addressof @_ZL47file_level_enum_descriptors_addressbook_2eproto : !llvm.ptr
    %4 = llvm.mlir.constant(3 : i32) : i32
    %5 = llvm.mlir.undef : !llvm.struct<"struct.google::protobuf::Metadata", (ptr, ptr)>
    %6 = llvm.insertvalue %0, %5[0] : !llvm.struct<"struct.google::protobuf::Metadata", (ptr, ptr)> 
    %7 = llvm.insertvalue %0, %6[1] : !llvm.struct<"struct.google::protobuf::Metadata", (ptr, ptr)> 
    %8 = llvm.mlir.undef : !llvm.array<3 x struct<"struct.google::protobuf::Metadata", (ptr, ptr)>>
    %9 = llvm.insertvalue %7, %8[0] : !llvm.array<3 x struct<"struct.google::protobuf::Metadata", (ptr, ptr)>> 
    %10 = llvm.insertvalue %7, %9[1] : !llvm.array<3 x struct<"struct.google::protobuf::Metadata", (ptr, ptr)>> 
    %11 = llvm.insertvalue %7, %10[2] : !llvm.array<3 x struct<"struct.google::protobuf::Metadata", (ptr, ptr)>> 
    %12 = llvm.mlir.addressof @_ZL39file_level_metadata_addressbook_2eproto : !llvm.ptr
    %13 = llvm.mlir.constant(dense<[-1, 8, -1, -1, -1, 16, 24, -1, 8, -1, -1, -1, 40, 64, 48, 16, 56, -1, 8, -1, -1, -1, 16]> : tensor<23xi32>) : !llvm.array<23 x i32>
    %14 = llvm.mlir.addressof @_ZN31TableStruct_addressbook_2eproto7offsetsE : !llvm.ptr
    %15 = llvm.mlir.constant(0 : i8) : i8
    %16 = llvm.mlir.constant(dense<0> : tensor<40xi8>) : !llvm.array<40 x i8>
    %17 = llvm.mlir.constant(0 : i64) : i64
    %18 = llvm.mlir.undef : !llvm.struct<"union.google::protobuf::internal::ExplicitlyConstructed<tutorial::AddressBook>::AlignedUnion", (i64, array<40 x i8>)>
    %19 = llvm.insertvalue %17, %18[0] : !llvm.struct<"union.google::protobuf::internal::ExplicitlyConstructed<tutorial::AddressBook>::AlignedUnion", (i64, array<40 x i8>)> 
    %20 = llvm.insertvalue %16, %19[1] : !llvm.struct<"union.google::protobuf::internal::ExplicitlyConstructed<tutorial::AddressBook>::AlignedUnion", (i64, array<40 x i8>)> 
    %21 = llvm.mlir.undef : !llvm.struct<"class.google::protobuf::internal::ExplicitlyConstructed.1", (struct<"union.google::protobuf::internal::ExplicitlyConstructed<tutorial::AddressBook>::AlignedUnion", (i64, array<40 x i8>)>)>
    %22 = llvm.insertvalue %20, %21[0] : !llvm.struct<"class.google::protobuf::internal::ExplicitlyConstructed.1", (struct<"union.google::protobuf::internal::ExplicitlyConstructed<tutorial::AddressBook>::AlignedUnion", (i64, array<40 x i8>)>)> 
    %23 = llvm.mlir.undef : !llvm.struct<"class.tutorial::AddressBookDefaultTypeInternal", (struct<"class.google::protobuf::internal::ExplicitlyConstructed.1", (struct<"union.google::protobuf::internal::ExplicitlyConstructed<tutorial::AddressBook>::AlignedUnion", (i64, array<40 x i8>)>)>)>
    %24 = llvm.insertvalue %22, %23[0] : !llvm.struct<"class.tutorial::AddressBookDefaultTypeInternal", (struct<"class.google::protobuf::internal::ExplicitlyConstructed.1", (struct<"union.google::protobuf::internal::ExplicitlyConstructed<tutorial::AddressBook>::AlignedUnion", (i64, array<40 x i8>)>)>)> 
    %25 = llvm.mlir.addressof @_ZN8tutorial30_AddressBook_default_instance_E : !llvm.ptr
    %26 = llvm.mlir.constant(dense<0> : tensor<64xi8>) : !llvm.array<64 x i8>
    %27 = llvm.mlir.undef : !llvm.struct<"union.google::protobuf::internal::ExplicitlyConstructed<tutorial::Person>::AlignedUnion", (i64, array<64 x i8>)>
    %28 = llvm.insertvalue %17, %27[0] : !llvm.struct<"union.google::protobuf::internal::ExplicitlyConstructed<tutorial::Person>::AlignedUnion", (i64, array<64 x i8>)> 
    %29 = llvm.insertvalue %26, %28[1] : !llvm.struct<"union.google::protobuf::internal::ExplicitlyConstructed<tutorial::Person>::AlignedUnion", (i64, array<64 x i8>)> 
    %30 = llvm.mlir.undef : !llvm.struct<"class.google::protobuf::internal::ExplicitlyConstructed.0", (struct<"union.google::protobuf::internal::ExplicitlyConstructed<tutorial::Person>::AlignedUnion", (i64, array<64 x i8>)>)>
    %31 = llvm.insertvalue %29, %30[0] : !llvm.struct<"class.google::protobuf::internal::ExplicitlyConstructed.0", (struct<"union.google::protobuf::internal::ExplicitlyConstructed<tutorial::Person>::AlignedUnion", (i64, array<64 x i8>)>)> 
    %32 = llvm.mlir.undef : !llvm.struct<"class.tutorial::PersonDefaultTypeInternal", (struct<"class.google::protobuf::internal::ExplicitlyConstructed.0", (struct<"union.google::protobuf::internal::ExplicitlyConstructed<tutorial::Person>::AlignedUnion", (i64, array<64 x i8>)>)>)>
    %33 = llvm.insertvalue %31, %32[0] : !llvm.struct<"class.tutorial::PersonDefaultTypeInternal", (struct<"class.google::protobuf::internal::ExplicitlyConstructed.0", (struct<"union.google::protobuf::internal::ExplicitlyConstructed<tutorial::Person>::AlignedUnion", (i64, array<64 x i8>)>)>)> 
    %34 = llvm.mlir.addressof @_ZN8tutorial25_Person_default_instance_E : !llvm.ptr
    %35 = llvm.mlir.constant(dense<0> : tensor<24xi8>) : !llvm.array<24 x i8>
    %36 = llvm.mlir.undef : !llvm.struct<"union.google::protobuf::internal::ExplicitlyConstructed<tutorial::Person_PhoneNumber>::AlignedUnion", (i64, array<24 x i8>)>
    %37 = llvm.insertvalue %17, %36[0] : !llvm.struct<"union.google::protobuf::internal::ExplicitlyConstructed<tutorial::Person_PhoneNumber>::AlignedUnion", (i64, array<24 x i8>)> 
    %38 = llvm.insertvalue %35, %37[1] : !llvm.struct<"union.google::protobuf::internal::ExplicitlyConstructed<tutorial::Person_PhoneNumber>::AlignedUnion", (i64, array<24 x i8>)> 
    %39 = llvm.mlir.undef : !llvm.struct<"class.google::protobuf::internal::ExplicitlyConstructed", (struct<"union.google::protobuf::internal::ExplicitlyConstructed<tutorial::Person_PhoneNumber>::AlignedUnion", (i64, array<24 x i8>)>)>
    %40 = llvm.insertvalue %38, %39[0] : !llvm.struct<"class.google::protobuf::internal::ExplicitlyConstructed", (struct<"union.google::protobuf::internal::ExplicitlyConstructed<tutorial::Person_PhoneNumber>::AlignedUnion", (i64, array<24 x i8>)>)> 
    %41 = llvm.mlir.undef : !llvm.struct<"class.tutorial::Person_PhoneNumberDefaultTypeInternal", (struct<"class.google::protobuf::internal::ExplicitlyConstructed", (struct<"union.google::protobuf::internal::ExplicitlyConstructed<tutorial::Person_PhoneNumber>::AlignedUnion", (i64, array<24 x i8>)>)>)>
    %42 = llvm.insertvalue %40, %41[0] : !llvm.struct<"class.tutorial::Person_PhoneNumberDefaultTypeInternal", (struct<"class.google::protobuf::internal::ExplicitlyConstructed", (struct<"union.google::protobuf::internal::ExplicitlyConstructed<tutorial::Person_PhoneNumber>::AlignedUnion", (i64, array<24 x i8>)>)>)> 
    %43 = llvm.mlir.addressof @_ZN8tutorial37_Person_PhoneNumber_default_instance_E : !llvm.ptr
    %44 = llvm.mlir.undef : !llvm.array<3 x ptr>
    %45 = llvm.insertvalue %43, %44[0] : !llvm.array<3 x ptr> 
    %46 = llvm.insertvalue %34, %45[1] : !llvm.array<3 x ptr> 
    %47 = llvm.insertvalue %25, %46[2] : !llvm.array<3 x ptr> 
    %48 = llvm.mlir.addressof @_ZL22file_default_instances : !llvm.ptr
    %49 = llvm.mlir.constant(48 : i32) : i32
    %50 = llvm.mlir.constant(-1 : i32) : i32
    %51 = llvm.mlir.constant(17 : i32) : i32
    %52 = llvm.mlir.undef : !llvm.struct<"struct.google::protobuf::internal::MigrationSchema", (i32, i32, i32)>
    %53 = llvm.insertvalue %51, %52[0] : !llvm.struct<"struct.google::protobuf::internal::MigrationSchema", (i32, i32, i32)> 
    %54 = llvm.insertvalue %50, %53[1] : !llvm.struct<"struct.google::protobuf::internal::MigrationSchema", (i32, i32, i32)> 
    %55 = llvm.insertvalue %49, %54[2] : !llvm.struct<"struct.google::protobuf::internal::MigrationSchema", (i32, i32, i32)> 
    %56 = llvm.mlir.constant(72 : i32) : i32
    %57 = llvm.mlir.constant(7 : i32) : i32
    %58 = llvm.mlir.undef : !llvm.struct<"struct.google::protobuf::internal::MigrationSchema", (i32, i32, i32)>
    %59 = llvm.insertvalue %57, %58[0] : !llvm.struct<"struct.google::protobuf::internal::MigrationSchema", (i32, i32, i32)> 
    %60 = llvm.insertvalue %50, %59[1] : !llvm.struct<"struct.google::protobuf::internal::MigrationSchema", (i32, i32, i32)> 
    %61 = llvm.insertvalue %56, %60[2] : !llvm.struct<"struct.google::protobuf::internal::MigrationSchema", (i32, i32, i32)> 
    %62 = llvm.mlir.constant(32 : i32) : i32
    %63 = llvm.mlir.constant(0 : i32) : i32
    %64 = llvm.mlir.undef : !llvm.struct<"struct.google::protobuf::internal::MigrationSchema", (i32, i32, i32)>
    %65 = llvm.insertvalue %63, %64[0] : !llvm.struct<"struct.google::protobuf::internal::MigrationSchema", (i32, i32, i32)> 
    %66 = llvm.insertvalue %50, %65[1] : !llvm.struct<"struct.google::protobuf::internal::MigrationSchema", (i32, i32, i32)> 
    %67 = llvm.insertvalue %62, %66[2] : !llvm.struct<"struct.google::protobuf::internal::MigrationSchema", (i32, i32, i32)> 
    %68 = llvm.mlir.undef : !llvm.array<3 x struct<"struct.google::protobuf::internal::MigrationSchema", (i32, i32, i32)>>
    %69 = llvm.insertvalue %67, %68[0] : !llvm.array<3 x struct<"struct.google::protobuf::internal::MigrationSchema", (i32, i32, i32)>> 
    %70 = llvm.insertvalue %61, %69[1] : !llvm.array<3 x struct<"struct.google::protobuf::internal::MigrationSchema", (i32, i32, i32)>> 
    %71 = llvm.insertvalue %55, %70[2] : !llvm.array<3 x struct<"struct.google::protobuf::internal::MigrationSchema", (i32, i32, i32)>> 
    %72 = llvm.mlir.addressof @_ZL7schemas : !llvm.ptr
    %73 = llvm.mlir.constant(1 : i32) : i32
    %74 = llvm.mlir.addressof @descriptor_table_google_2fprotobuf_2ftimestamp_2eproto : !llvm.ptr
    %75 = llvm.mlir.undef : !llvm.array<1 x ptr>
    %76 = llvm.insertvalue %74, %75[0] : !llvm.array<1 x ptr> 
    %77 = llvm.mlir.addressof @_ZL41descriptor_table_addressbook_2eproto_deps : !llvm.ptr
    %78 = llvm.mlir.undef : !llvm.array<0 x ptr>
    %79 = llvm.mlir.addressof @__gxx_personality_v0 : !llvm.ptr
    %80 = llvm.mlir.addressof @_ZL59InitDefaultsscc_info_Person_PhoneNumber_addressbook_2eprotov : !llvm.ptr
    %81 = llvm.mlir.undef : !llvm.struct<(i32)>
    %82 = llvm.insertvalue %50, %81[0] : !llvm.struct<(i32)> 
    %83 = llvm.mlir.undef : !llvm.struct<(struct<(i32)>, i32, i32, ptr)>
    %84 = llvm.insertvalue %82, %83[0] : !llvm.struct<(struct<(i32)>, i32, i32, ptr)> 
    %85 = llvm.insertvalue %63, %84[1] : !llvm.struct<(struct<(i32)>, i32, i32, ptr)> 
    %86 = llvm.insertvalue %63, %85[2] : !llvm.struct<(struct<(i32)>, i32, i32, ptr)> 
    %87 = llvm.insertvalue %80, %86[3] : !llvm.struct<(struct<(i32)>, i32, i32, ptr)> 
    %88 = llvm.mlir.undef : !llvm.struct<(struct<(struct<(i32)>, i32, i32, ptr)>, array<0 x ptr>)>
    %89 = llvm.insertvalue %87, %88[0] : !llvm.struct<(struct<(struct<(i32)>, i32, i32, ptr)>, array<0 x ptr>)> 
    %90 = llvm.insertvalue %78, %89[1] : !llvm.struct<(struct<(struct<(i32)>, i32, i32, ptr)>, array<0 x ptr>)> 
    %91 = llvm.mlir.addressof @scc_info_Person_PhoneNumber_addressbook_2eproto : !llvm.ptr
    %92 = llvm.mlir.addressof @scc_info_Timestamp_google_2fprotobuf_2ftimestamp_2eproto : !llvm.ptr
    %93 = llvm.mlir.undef : !llvm.array<2 x ptr>
    %94 = llvm.insertvalue %91, %93[0] : !llvm.array<2 x ptr> 
    %95 = llvm.insertvalue %92, %94[1] : !llvm.array<2 x ptr> 
    %96 = llvm.mlir.addressof @_ZL47InitDefaultsscc_info_Person_addressbook_2eprotov : !llvm.ptr
    %97 = llvm.mlir.constant(2 : i32) : i32
    %98 = llvm.mlir.undef : !llvm.struct<(struct<(i32)>, i32, i32, ptr)>
    %99 = llvm.insertvalue %82, %98[0] : !llvm.struct<(struct<(i32)>, i32, i32, ptr)> 
    %100 = llvm.insertvalue %97, %99[1] : !llvm.struct<(struct<(i32)>, i32, i32, ptr)> 
    %101 = llvm.insertvalue %63, %100[2] : !llvm.struct<(struct<(i32)>, i32, i32, ptr)> 
    %102 = llvm.insertvalue %96, %101[3] : !llvm.struct<(struct<(i32)>, i32, i32, ptr)> 
    %103 = llvm.mlir.undef : !llvm.struct<(struct<(struct<(i32)>, i32, i32, ptr)>, array<2 x ptr>)>
    %104 = llvm.insertvalue %102, %103[0] : !llvm.struct<(struct<(struct<(i32)>, i32, i32, ptr)>, array<2 x ptr>)> 
    %105 = llvm.insertvalue %95, %104[1] : !llvm.struct<(struct<(struct<(i32)>, i32, i32, ptr)>, array<2 x ptr>)> 
    %106 = llvm.mlir.addressof @scc_info_Person_addressbook_2eproto : !llvm.ptr
    %107 = llvm.mlir.undef : !llvm.array<1 x ptr>
    %108 = llvm.insertvalue %106, %107[0] : !llvm.array<1 x ptr> 
    %109 = llvm.mlir.addressof @_ZL52InitDefaultsscc_info_AddressBook_addressbook_2eprotov : !llvm.ptr
    %110 = llvm.mlir.undef : !llvm.struct<(struct<(i32)>, i32, i32, ptr)>
    %111 = llvm.insertvalue %82, %110[0] : !llvm.struct<(struct<(i32)>, i32, i32, ptr)> 
    %112 = llvm.insertvalue %73, %111[1] : !llvm.struct<(struct<(i32)>, i32, i32, ptr)> 
    %113 = llvm.insertvalue %63, %112[2] : !llvm.struct<(struct<(i32)>, i32, i32, ptr)> 
    %114 = llvm.insertvalue %109, %113[3] : !llvm.struct<(struct<(i32)>, i32, i32, ptr)> 
    %115 = llvm.mlir.undef : !llvm.struct<(struct<(struct<(i32)>, i32, i32, ptr)>, array<1 x ptr>)>
    %116 = llvm.insertvalue %114, %115[0] : !llvm.struct<(struct<(struct<(i32)>, i32, i32, ptr)>, array<1 x ptr>)> 
    %117 = llvm.insertvalue %108, %116[1] : !llvm.struct<(struct<(struct<(i32)>, i32, i32, ptr)>, array<1 x ptr>)> 
    %118 = llvm.mlir.addressof @scc_info_AddressBook_addressbook_2eproto : !llvm.ptr
    %119 = llvm.mlir.undef : !llvm.array<3 x ptr>
    %120 = llvm.insertvalue %118, %119[0] : !llvm.array<3 x ptr> 
    %121 = llvm.insertvalue %106, %120[1] : !llvm.array<3 x ptr> 
    %122 = llvm.insertvalue %91, %121[2] : !llvm.array<3 x ptr> 
    %123 = llvm.mlir.addressof @_ZL41descriptor_table_addressbook_2eproto_sccs : !llvm.ptr
    %124 = llvm.mlir.undef : !llvm.struct<"struct.std::once_flag", (i32)>
    %125 = llvm.insertvalue %63, %124[0] : !llvm.struct<"struct.std::once_flag", (i32)> 
    %126 = llvm.mlir.addressof @_ZL41descriptor_table_addressbook_2eproto_once : !llvm.ptr
    %127 = llvm.mlir.constant(537 : i32) : i32
    %128 = llvm.mlir.constant("addressbook.proto\00") : !llvm.array<18 x i8>
    %129 = llvm.mlir.addressof @".str" : !llvm.ptr
    %130 = llvm.mlir.constant("\0A\11addressbook.proto\12\08tutorial\1A\1Fgoogle/protobuf/timestamp.proto\22\87\02\0A\06Person\12\0C\0A\04name\18\01 \01(\09\12\0A\0A\02id\18\02 \01(\05\12\0D\0A\05email\18\03 \01(\09\12,\0A\06phones\18\04 \03(\0B2\1C.tutorial.Person.PhoneNumber\120\0A\0Clast_updated\18\05 \01(\0B2\1A.google.protobuf.Timestamp\1AG\0A\0BPhoneNumber\12\0E\0A\06number\18\01 \01(\09\12(\0A\04type\18\02 \01(\0E2\1A.tutorial.Person.PhoneType\22+\0A\09PhoneType\12\0A\0A\06MOBILE\10\00\12\08\0A\04HOME\10\01\12\08\0A\04WORK\10\02\22/\0A\0BAddressBook\12 \0A\06people\18\01 \03(\0B2\10.tutorial.PersonB\95\01\0A\1Bcom.example.tutorial.protosB\11AddressBookProtosP\01Z:github.com/protocolbuffers/protobuf/examples/go/tutorialpb\AA\02$Google.Protobuf.Examples.AddressBookb\06proto3\00") : !llvm.array<538 x i8>
    %131 = llvm.mlir.addressof @_ZL45descriptor_table_protodef_addressbook_2eproto : !llvm.ptr
    %132 = llvm.mlir.undef : !llvm.struct<"struct.google::protobuf::internal::DescriptorTable", (i8, i8, ptr, ptr, i32, ptr, ptr, ptr, i32, i32, ptr, ptr, ptr, ptr, i32, ptr, ptr)>
    %133 = llvm.insertvalue %15, %132[0] : !llvm.struct<"struct.google::protobuf::internal::DescriptorTable", (i8, i8, ptr, ptr, i32, ptr, ptr, ptr, i32, i32, ptr, ptr, ptr, ptr, i32, ptr, ptr)> 
    %134 = llvm.insertvalue %15, %133[1] : !llvm.struct<"struct.google::protobuf::internal::DescriptorTable", (i8, i8, ptr, ptr, i32, ptr, ptr, ptr, i32, i32, ptr, ptr, ptr, ptr, i32, ptr, ptr)> 
    %135 = llvm.insertvalue %131, %134[2] : !llvm.struct<"struct.google::protobuf::internal::DescriptorTable", (i8, i8, ptr, ptr, i32, ptr, ptr, ptr, i32, i32, ptr, ptr, ptr, ptr, i32, ptr, ptr)> 
    %136 = llvm.insertvalue %129, %135[3] : !llvm.struct<"struct.google::protobuf::internal::DescriptorTable", (i8, i8, ptr, ptr, i32, ptr, ptr, ptr, i32, i32, ptr, ptr, ptr, ptr, i32, ptr, ptr)> 
    %137 = llvm.insertvalue %127, %136[4] : !llvm.struct<"struct.google::protobuf::internal::DescriptorTable", (i8, i8, ptr, ptr, i32, ptr, ptr, ptr, i32, i32, ptr, ptr, ptr, ptr, i32, ptr, ptr)> 
    %138 = llvm.insertvalue %126, %137[5] : !llvm.struct<"struct.google::protobuf::internal::DescriptorTable", (i8, i8, ptr, ptr, i32, ptr, ptr, ptr, i32, i32, ptr, ptr, ptr, ptr, i32, ptr, ptr)> 
    %139 = llvm.insertvalue %123, %138[6] : !llvm.struct<"struct.google::protobuf::internal::DescriptorTable", (i8, i8, ptr, ptr, i32, ptr, ptr, ptr, i32, i32, ptr, ptr, ptr, ptr, i32, ptr, ptr)> 
    %140 = llvm.insertvalue %77, %139[7] : !llvm.struct<"struct.google::protobuf::internal::DescriptorTable", (i8, i8, ptr, ptr, i32, ptr, ptr, ptr, i32, i32, ptr, ptr, ptr, ptr, i32, ptr, ptr)> 
    %141 = llvm.insertvalue %4, %140[8] : !llvm.struct<"struct.google::protobuf::internal::DescriptorTable", (i8, i8, ptr, ptr, i32, ptr, ptr, ptr, i32, i32, ptr, ptr, ptr, ptr, i32, ptr, ptr)> 
    %142 = llvm.insertvalue %73, %141[9] : !llvm.struct<"struct.google::protobuf::internal::DescriptorTable", (i8, i8, ptr, ptr, i32, ptr, ptr, ptr, i32, i32, ptr, ptr, ptr, ptr, i32, ptr, ptr)> 
    %143 = llvm.insertvalue %72, %142[10] : !llvm.struct<"struct.google::protobuf::internal::DescriptorTable", (i8, i8, ptr, ptr, i32, ptr, ptr, ptr, i32, i32, ptr, ptr, ptr, ptr, i32, ptr, ptr)> 
    %144 = llvm.insertvalue %48, %143[11] : !llvm.struct<"struct.google::protobuf::internal::DescriptorTable", (i8, i8, ptr, ptr, i32, ptr, ptr, ptr, i32, i32, ptr, ptr, ptr, ptr, i32, ptr, ptr)> 
    %145 = llvm.insertvalue %14, %144[12] : !llvm.struct<"struct.google::protobuf::internal::DescriptorTable", (i8, i8, ptr, ptr, i32, ptr, ptr, ptr, i32, i32, ptr, ptr, ptr, ptr, i32, ptr, ptr)> 
    %146 = llvm.insertvalue %12, %145[13] : !llvm.struct<"struct.google::protobuf::internal::DescriptorTable", (i8, i8, ptr, ptr, i32, ptr, ptr, ptr, i32, i32, ptr, ptr, ptr, ptr, i32, ptr, ptr)> 
    %147 = llvm.insertvalue %4, %146[14] : !llvm.struct<"struct.google::protobuf::internal::DescriptorTable", (i8, i8, ptr, ptr, i32, ptr, ptr, ptr, i32, i32, ptr, ptr, ptr, ptr, i32, ptr, ptr)> 
    %148 = llvm.insertvalue %3, %147[15] : !llvm.struct<"struct.google::protobuf::internal::DescriptorTable", (i8, i8, ptr, ptr, i32, ptr, ptr, ptr, i32, i32, ptr, ptr, ptr, ptr, i32, ptr, ptr)> 
    %149 = llvm.insertvalue %0, %148[16] : !llvm.struct<"struct.google::protobuf::internal::DescriptorTable", (i8, i8, ptr, ptr, i32, ptr, ptr, ptr, i32, i32, ptr, ptr, ptr, ptr, i32, ptr, ptr)> 
    %150 = llvm.mlir.addressof @descriptor_table_addressbook_2eproto : !llvm.ptr
    %151 = llvm.mlir.constant(false) : i1
    llvm.call @_ZN6google8protobuf8internal17AssignDescriptorsEPKNS1_15DescriptorTableEb(%150, %151) : (!llvm.ptr, i1) -> ()
    %152 = llvm.load %3 {alignment = 8 : i64, tbaa = [#tbaa_tag2]} : !llvm.ptr -> !llvm.ptr
    llvm.return %152 : !llvm.ptr
  }
  llvm.func local_unnamed_addr @_ZN6google8protobuf8internal17AssignDescriptorsEPKNS1_15DescriptorTableEb(!llvm.ptr {llvm.noundef}, i1 {llvm.noundef, llvm.zeroext}) attributes {frame_pointer = #llvm.framePointerKind<none>, passthrough = [["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], ["tune-cpu", "generic"]], target_cpu = "x86-64", target_features = #llvm.target_features<["+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>}
  llvm.func local_unnamed_addr @_ZN8tutorial24Person_PhoneType_IsValidEi(%arg0: i32 {llvm.noundef}) -> (i1 {llvm.noundef, llvm.zeroext}) attributes {frame_pointer = #llvm.framePointerKind<none>, memory = #llvm.memory_effects<other = none, argMem = none, inaccessibleMem = none>, passthrough = ["mustprogress", "nofree", "norecurse", "nosync", "nounwind", "willreturn", ["uwtable", "2"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], ["tune-cpu", "generic"]], target_cpu = "x86-64", target_features = #llvm.target_features<["+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>} {
    %0 = llvm.mlir.constant(3 : i32) : i32
    %1 = llvm.icmp "ult" %arg0, %0 : i32
    llvm.return %1 : i1
  }
  llvm.func local_unnamed_addr @_ZN8tutorial18Person_PhoneNumber21InitAsDefaultInstanceEv() attributes {alignment = 2 : i64, frame_pointer = #llvm.framePointerKind<none>, memory = #llvm.memory_effects<other = none, argMem = none, inaccessibleMem = none>, passthrough = ["mustprogress", "nofree", "norecurse", "nosync", "nounwind", "willreturn", ["uwtable", "2"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], ["tune-cpu", "generic"]], target_cpu = "x86-64", target_features = #llvm.target_features<["+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>} {
    llvm.return
  }
  llvm.func unnamed_addr @_ZN8tutorial18Person_PhoneNumberC2EPN6google8protobuf5ArenaE(%arg0: !llvm.ptr {llvm.align = 8 : i64, llvm.dereferenceable = 32 : i64, llvm.nocapture, llvm.nonnull, llvm.noundef, llvm.writeonly}, %arg1: !llvm.ptr {llvm.noundef}) attributes {alignment = 2 : i64, frame_pointer = #llvm.framePointerKind<none>, passthrough = [["uwtable", "2"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], ["tune-cpu", "generic"]], personality = @__gxx_personality_v0, target_cpu = "x86-64", target_features = #llvm.target_features<["+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>} {
    %0 = llvm.mlir.constant(0 : i64) : i64
    %1 = llvm.mlir.constant(0 : i32) : i32
    %2 = llvm.mlir.constant(1 : i32) : i32
    %3 = llvm.mlir.constant(2 : i64) : i64
    %4 = llvm.mlir.addressof @_ZNK8tutorial18Person_PhoneNumber11GetMetadataEv : !llvm.ptr
    %5 = llvm.mlir.zero : !llvm.ptr
    %6 = llvm.mlir.addressof @__gxx_personality_v0 : !llvm.ptr
    %7 = llvm.mlir.addressof @_ZNK8tutorial18Person_PhoneNumber13SetCachedSizeEi : !llvm.ptr
    %8 = llvm.mlir.addressof @_ZNK6google8protobuf7Message13SpaceUsedLongEv : !llvm.ptr
    %9 = llvm.mlir.addressof @_ZN6google8protobuf7Message20DiscardUnknownFieldsEv : !llvm.ptr
    %10 = llvm.mlir.addressof @_ZN8tutorial18Person_PhoneNumber9MergeFromERKN6google8protobuf7MessageE : !llvm.ptr
    %11 = llvm.mlir.addressof @_ZN8tutorial18Person_PhoneNumber8CopyFromERKN6google8protobuf7MessageE : !llvm.ptr
    %12 = llvm.mlir.addressof @_ZNK6google8protobuf11MessageLite16InternalGetTableEv : !llvm.ptr
    %13 = llvm.mlir.addressof @_ZNK8tutorial18Person_PhoneNumber18_InternalSerializeEPhPN6google8protobuf2io19EpsCopyOutputStreamE : !llvm.ptr
    %14 = llvm.mlir.addressof @_ZN8tutorial18Person_PhoneNumber14_InternalParseEPKcPN6google8protobuf8internal12ParseContextE : !llvm.ptr
    %15 = llvm.mlir.addressof @_ZNK8tutorial18Person_PhoneNumber13GetCachedSizeEv : !llvm.ptr
    %16 = llvm.mlir.addressof @_ZNK8tutorial18Person_PhoneNumber12ByteSizeLongEv : !llvm.ptr
    %17 = llvm.mlir.addressof @_ZN6google8protobuf7Message21CheckTypeAndMergeFromERKNS0_11MessageLiteE : !llvm.ptr
    %18 = llvm.mlir.addressof @_ZNK6google8protobuf7Message25InitializationErrorStringB5cxx11Ev : !llvm.ptr
    %19 = llvm.mlir.addressof @_ZNK8tutorial18Person_PhoneNumber13IsInitializedEv : !llvm.ptr
    %20 = llvm.mlir.addressof @_ZN8tutorial18Person_PhoneNumber5ClearEv : !llvm.ptr
    %21 = llvm.mlir.addressof @_ZNK8tutorial18Person_PhoneNumber3NewEPN6google8protobuf5ArenaE : !llvm.ptr
    %22 = llvm.mlir.addressof @_ZNK8tutorial18Person_PhoneNumber3NewEv : !llvm.ptr
    %23 = llvm.mlir.addressof @_ZNK6google8protobuf7Message11GetTypeNameB5cxx11Ev : !llvm.ptr
    %24 = llvm.mlir.addressof @_ZN8tutorial18Person_PhoneNumberD0Ev : !llvm.ptr
    %25 = llvm.mlir.addressof @_ZN8tutorial18Person_PhoneNumberD2Ev : !llvm.ptr
    %26 = llvm.mlir.addressof @_ZTIN6google8protobuf7MessageE : !llvm.ptr
    %27 = llvm.mlir.constant("N8tutorial18Person_PhoneNumberE\00") : !llvm.array<32 x i8>
    %28 = llvm.mlir.addressof @_ZTSN8tutorial18Person_PhoneNumberE : !llvm.ptr
    %29 = llvm.mlir.addressof @_ZTVN10__cxxabiv120__si_class_type_infoE : !llvm.ptr
    %30 = llvm.getelementptr inbounds %29[%3] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.ptr
    %31 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, ptr)>
    %32 = llvm.insertvalue %30, %31[0] : !llvm.struct<(ptr, ptr, ptr)> 
    %33 = llvm.insertvalue %28, %32[1] : !llvm.struct<(ptr, ptr, ptr)> 
    %34 = llvm.insertvalue %26, %33[2] : !llvm.struct<(ptr, ptr, ptr)> 
    %35 = llvm.mlir.addressof @_ZTIN8tutorial18Person_PhoneNumberE : !llvm.ptr
    %36 = llvm.mlir.undef : !llvm.array<22 x ptr>
    %37 = llvm.insertvalue %5, %36[0] : !llvm.array<22 x ptr> 
    %38 = llvm.insertvalue %35, %37[1] : !llvm.array<22 x ptr> 
    %39 = llvm.insertvalue %25, %38[2] : !llvm.array<22 x ptr> 
    %40 = llvm.insertvalue %24, %39[3] : !llvm.array<22 x ptr> 
    %41 = llvm.insertvalue %23, %40[4] : !llvm.array<22 x ptr> 
    %42 = llvm.insertvalue %22, %41[5] : !llvm.array<22 x ptr> 
    %43 = llvm.insertvalue %21, %42[6] : !llvm.array<22 x ptr> 
    %44 = llvm.insertvalue %20, %43[7] : !llvm.array<22 x ptr> 
    %45 = llvm.insertvalue %19, %44[8] : !llvm.array<22 x ptr> 
    %46 = llvm.insertvalue %18, %45[9] : !llvm.array<22 x ptr> 
    %47 = llvm.insertvalue %17, %46[10] : !llvm.array<22 x ptr> 
    %48 = llvm.insertvalue %16, %47[11] : !llvm.array<22 x ptr> 
    %49 = llvm.insertvalue %15, %48[12] : !llvm.array<22 x ptr> 
    %50 = llvm.insertvalue %14, %49[13] : !llvm.array<22 x ptr> 
    %51 = llvm.insertvalue %13, %50[14] : !llvm.array<22 x ptr> 
    %52 = llvm.insertvalue %12, %51[15] : !llvm.array<22 x ptr> 
    %53 = llvm.insertvalue %11, %52[16] : !llvm.array<22 x ptr> 
    %54 = llvm.insertvalue %10, %53[17] : !llvm.array<22 x ptr> 
    %55 = llvm.insertvalue %9, %54[18] : !llvm.array<22 x ptr> 
    %56 = llvm.insertvalue %8, %55[19] : !llvm.array<22 x ptr> 
    %57 = llvm.insertvalue %7, %56[20] : !llvm.array<22 x ptr> 
    %58 = llvm.insertvalue %4, %57[21] : !llvm.array<22 x ptr> 
    %59 = llvm.mlir.undef : !llvm.struct<(array<22 x ptr>)>
    %60 = llvm.insertvalue %58, %59[0] : !llvm.struct<(array<22 x ptr>)> 
    %61 = llvm.mlir.addressof @_ZTVN8tutorial18Person_PhoneNumberE : !llvm.ptr
    %62 = llvm.getelementptr inbounds %61[%0, 0, %3] : (!llvm.ptr, i64, i64) -> !llvm.ptr, !llvm.struct<(array<22 x ptr>)>
    %63 = llvm.mlir.constant(3 : i32) : i32
    %64 = llvm.mlir.undef : !llvm.array<0 x ptr>
    %65 = llvm.mlir.addressof @_ZL59InitDefaultsscc_info_Person_PhoneNumber_addressbook_2eprotov : !llvm.ptr
    %66 = llvm.mlir.constant(-1 : i32) : i32
    %67 = llvm.mlir.undef : !llvm.struct<(i32)>
    %68 = llvm.insertvalue %66, %67[0] : !llvm.struct<(i32)> 
    %69 = llvm.mlir.undef : !llvm.struct<(struct<(i32)>, i32, i32, ptr)>
    %70 = llvm.insertvalue %68, %69[0] : !llvm.struct<(struct<(i32)>, i32, i32, ptr)> 
    %71 = llvm.insertvalue %1, %70[1] : !llvm.struct<(struct<(i32)>, i32, i32, ptr)> 
    %72 = llvm.insertvalue %1, %71[2] : !llvm.struct<(struct<(i32)>, i32, i32, ptr)> 
    %73 = llvm.insertvalue %65, %72[3] : !llvm.struct<(struct<(i32)>, i32, i32, ptr)> 
    %74 = llvm.mlir.undef : !llvm.struct<(struct<(struct<(i32)>, i32, i32, ptr)>, array<0 x ptr>)>
    %75 = llvm.insertvalue %73, %74[0] : !llvm.struct<(struct<(struct<(i32)>, i32, i32, ptr)>, array<0 x ptr>)> 
    %76 = llvm.insertvalue %64, %75[1] : !llvm.struct<(struct<(struct<(i32)>, i32, i32, ptr)>, array<0 x ptr>)> 
    %77 = llvm.mlir.addressof @scc_info_Person_PhoneNumber_addressbook_2eproto : !llvm.ptr
    %78 = llvm.mlir.addressof @_ZN6google8protobuf8internal26fixed_address_empty_stringB5cxx11E : !llvm.ptr
    %79 = llvm.mlir.constant(2 : i32) : i32
    %80 = llvm.getelementptr inbounds %arg0[%0, 0, 0, 1] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.tutorial::Person_PhoneNumber", (struct<"class.google::protobuf::Message", (struct<"class.google::protobuf::MessageLite", (ptr, struct<"class.google::protobuf::internal::InternalMetadata", (ptr)>)>)>, struct<"struct.google::protobuf::internal::ArenaStringPtr", (ptr)>, i32, struct<"class.google::protobuf::internal::CachedSize", (struct<"struct.std::atomic", (struct<"struct.std::__atomic_base", (i32)>)>)>)>
    %81 = llvm.bitcast %80 : !llvm.ptr to !llvm.ptr
    llvm.store %arg1, %81 {alignment = 8 : i64, tbaa = [#tbaa_tag5]} : !llvm.ptr, !llvm.ptr
    %82 = llvm.getelementptr inbounds %arg0[%0, 0, 0, 0] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.tutorial::Person_PhoneNumber", (struct<"class.google::protobuf::Message", (struct<"class.google::protobuf::MessageLite", (ptr, struct<"class.google::protobuf::internal::InternalMetadata", (ptr)>)>)>, struct<"struct.google::protobuf::internal::ArenaStringPtr", (ptr)>, i32, struct<"class.google::protobuf::internal::CachedSize", (struct<"struct.std::atomic", (struct<"struct.std::__atomic_base", (i32)>)>)>)>
    llvm.store %62, %82 {alignment = 8 : i64, tbaa = [#tbaa_tag]} : !llvm.ptr, !llvm.ptr
    %83 = llvm.getelementptr inbounds %arg0[%0, 3, 0, 0, 0] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.tutorial::Person_PhoneNumber", (struct<"class.google::protobuf::Message", (struct<"class.google::protobuf::MessageLite", (ptr, struct<"class.google::protobuf::internal::InternalMetadata", (ptr)>)>)>, struct<"struct.google::protobuf::internal::ArenaStringPtr", (ptr)>, i32, struct<"class.google::protobuf::internal::CachedSize", (struct<"struct.std::atomic", (struct<"struct.std::__atomic_base", (i32)>)>)>)>
    llvm.store %1, %83 {alignment = 4 : i64, tbaa = [#tbaa_tag6]} : i32, !llvm.ptr
    %84 = llvm.load %77 atomic acquire {alignment = 8 : i64} : !llvm.ptr -> i32
    %85 = llvm.icmp "eq" %84, %1 : i32
    llvm.cond_br %85 weights([2000, 1]), ^bb2, ^bb1
  ^bb1:  // pred: ^bb0
    llvm.call @_ZN6google8protobuf8internal11InitSCCImplEPNS1_11SCCInfoBaseE(%77) : (!llvm.ptr) -> ()
    llvm.br ^bb2
  ^bb2:  // 2 preds: ^bb0, ^bb1
    %86 = llvm.getelementptr inbounds %arg0[%0, 1, 0] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.tutorial::Person_PhoneNumber", (struct<"class.google::protobuf::Message", (struct<"class.google::protobuf::MessageLite", (ptr, struct<"class.google::protobuf::internal::InternalMetadata", (ptr)>)>)>, struct<"struct.google::protobuf::internal::ArenaStringPtr", (ptr)>, i32, struct<"class.google::protobuf::internal::CachedSize", (struct<"struct.std::atomic", (struct<"struct.std::__atomic_base", (i32)>)>)>)>
    llvm.store %78, %86 {alignment = 8 : i64, tbaa = [#tbaa_tag8]} : !llvm.ptr, !llvm.ptr
    %87 = llvm.getelementptr inbounds %arg0[%0, 2] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.tutorial::Person_PhoneNumber", (struct<"class.google::protobuf::Message", (struct<"class.google::protobuf::MessageLite", (ptr, struct<"class.google::protobuf::internal::InternalMetadata", (ptr)>)>)>, struct<"struct.google::protobuf::internal::ArenaStringPtr", (ptr)>, i32, struct<"class.google::protobuf::internal::CachedSize", (struct<"struct.std::atomic", (struct<"struct.std::__atomic_base", (i32)>)>)>)>
    llvm.store %1, %87 {alignment = 8 : i64, tbaa = [#tbaa_tag24]} : i32, !llvm.ptr
    llvm.return
  }
  llvm.func @__gxx_personality_v0(...) -> i32
  llvm.func unnamed_addr @_ZN8tutorial18Person_PhoneNumberC2ERKS0_(%arg0: !llvm.ptr {llvm.align = 8 : i64, llvm.dereferenceable = 32 : i64, llvm.nonnull, llvm.noundef}, %arg1: !llvm.ptr {llvm.align = 8 : i64, llvm.dereferenceable = 32 : i64, llvm.nocapture, llvm.nonnull, llvm.noundef, llvm.readonly}) attributes {alignment = 2 : i64, frame_pointer = #llvm.framePointerKind<none>, passthrough = [["uwtable", "2"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], ["tune-cpu", "generic"]], personality = @__gxx_personality_v0, target_cpu = "x86-64", target_features = #llvm.target_features<["+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>} {
    %0 = llvm.mlir.constant(0 : i64) : i64
    %1 = llvm.mlir.constant(0 : i32) : i32
    %2 = llvm.mlir.constant(1 : i32) : i32
    %3 = llvm.mlir.zero : !llvm.ptr
    %4 = llvm.mlir.constant(2 : i64) : i64
    %5 = llvm.mlir.addressof @_ZNK8tutorial18Person_PhoneNumber11GetMetadataEv : !llvm.ptr
    %6 = llvm.mlir.addressof @__gxx_personality_v0 : !llvm.ptr
    %7 = llvm.mlir.addressof @_ZNK8tutorial18Person_PhoneNumber13SetCachedSizeEi : !llvm.ptr
    %8 = llvm.mlir.addressof @_ZNK6google8protobuf7Message13SpaceUsedLongEv : !llvm.ptr
    %9 = llvm.mlir.addressof @_ZN6google8protobuf7Message20DiscardUnknownFieldsEv : !llvm.ptr
    %10 = llvm.mlir.addressof @_ZN8tutorial18Person_PhoneNumber9MergeFromERKN6google8protobuf7MessageE : !llvm.ptr
    %11 = llvm.mlir.addressof @_ZN8tutorial18Person_PhoneNumber8CopyFromERKN6google8protobuf7MessageE : !llvm.ptr
    %12 = llvm.mlir.addressof @_ZNK6google8protobuf11MessageLite16InternalGetTableEv : !llvm.ptr
    %13 = llvm.mlir.addressof @_ZNK8tutorial18Person_PhoneNumber18_InternalSerializeEPhPN6google8protobuf2io19EpsCopyOutputStreamE : !llvm.ptr
    %14 = llvm.mlir.addressof @_ZN8tutorial18Person_PhoneNumber14_InternalParseEPKcPN6google8protobuf8internal12ParseContextE : !llvm.ptr
    %15 = llvm.mlir.addressof @_ZNK8tutorial18Person_PhoneNumber13GetCachedSizeEv : !llvm.ptr
    %16 = llvm.mlir.addressof @_ZNK8tutorial18Person_PhoneNumber12ByteSizeLongEv : !llvm.ptr
    %17 = llvm.mlir.addressof @_ZN6google8protobuf7Message21CheckTypeAndMergeFromERKNS0_11MessageLiteE : !llvm.ptr
    %18 = llvm.mlir.addressof @_ZNK6google8protobuf7Message25InitializationErrorStringB5cxx11Ev : !llvm.ptr
    %19 = llvm.mlir.addressof @_ZNK8tutorial18Person_PhoneNumber13IsInitializedEv : !llvm.ptr
    %20 = llvm.mlir.addressof @_ZN8tutorial18Person_PhoneNumber5ClearEv : !llvm.ptr
    %21 = llvm.mlir.addressof @_ZNK8tutorial18Person_PhoneNumber3NewEPN6google8protobuf5ArenaE : !llvm.ptr
    %22 = llvm.mlir.addressof @_ZNK8tutorial18Person_PhoneNumber3NewEv : !llvm.ptr
    %23 = llvm.mlir.addressof @_ZNK6google8protobuf7Message11GetTypeNameB5cxx11Ev : !llvm.ptr
    %24 = llvm.mlir.addressof @_ZN8tutorial18Person_PhoneNumberD0Ev : !llvm.ptr
    %25 = llvm.mlir.addressof @_ZN8tutorial18Person_PhoneNumberD2Ev : !llvm.ptr
    %26 = llvm.mlir.addressof @_ZTIN6google8protobuf7MessageE : !llvm.ptr
    %27 = llvm.mlir.constant("N8tutorial18Person_PhoneNumberE\00") : !llvm.array<32 x i8>
    %28 = llvm.mlir.addressof @_ZTSN8tutorial18Person_PhoneNumberE : !llvm.ptr
    %29 = llvm.mlir.addressof @_ZTVN10__cxxabiv120__si_class_type_infoE : !llvm.ptr
    %30 = llvm.getelementptr inbounds %29[%4] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.ptr
    %31 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, ptr)>
    %32 = llvm.insertvalue %30, %31[0] : !llvm.struct<(ptr, ptr, ptr)> 
    %33 = llvm.insertvalue %28, %32[1] : !llvm.struct<(ptr, ptr, ptr)> 
    %34 = llvm.insertvalue %26, %33[2] : !llvm.struct<(ptr, ptr, ptr)> 
    %35 = llvm.mlir.addressof @_ZTIN8tutorial18Person_PhoneNumberE : !llvm.ptr
    %36 = llvm.mlir.undef : !llvm.array<22 x ptr>
    %37 = llvm.insertvalue %3, %36[0] : !llvm.array<22 x ptr> 
    %38 = llvm.insertvalue %35, %37[1] : !llvm.array<22 x ptr> 
    %39 = llvm.insertvalue %25, %38[2] : !llvm.array<22 x ptr> 
    %40 = llvm.insertvalue %24, %39[3] : !llvm.array<22 x ptr> 
    %41 = llvm.insertvalue %23, %40[4] : !llvm.array<22 x ptr> 
    %42 = llvm.insertvalue %22, %41[5] : !llvm.array<22 x ptr> 
    %43 = llvm.insertvalue %21, %42[6] : !llvm.array<22 x ptr> 
    %44 = llvm.insertvalue %20, %43[7] : !llvm.array<22 x ptr> 
    %45 = llvm.insertvalue %19, %44[8] : !llvm.array<22 x ptr> 
    %46 = llvm.insertvalue %18, %45[9] : !llvm.array<22 x ptr> 
    %47 = llvm.insertvalue %17, %46[10] : !llvm.array<22 x ptr> 
    %48 = llvm.insertvalue %16, %47[11] : !llvm.array<22 x ptr> 
    %49 = llvm.insertvalue %15, %48[12] : !llvm.array<22 x ptr> 
    %50 = llvm.insertvalue %14, %49[13] : !llvm.array<22 x ptr> 
    %51 = llvm.insertvalue %13, %50[14] : !llvm.array<22 x ptr> 
    %52 = llvm.insertvalue %12, %51[15] : !llvm.array<22 x ptr> 
    %53 = llvm.insertvalue %11, %52[16] : !llvm.array<22 x ptr> 
    %54 = llvm.insertvalue %10, %53[17] : !llvm.array<22 x ptr> 
    %55 = llvm.insertvalue %9, %54[18] : !llvm.array<22 x ptr> 
    %56 = llvm.insertvalue %8, %55[19] : !llvm.array<22 x ptr> 
    %57 = llvm.insertvalue %7, %56[20] : !llvm.array<22 x ptr> 
    %58 = llvm.insertvalue %5, %57[21] : !llvm.array<22 x ptr> 
    %59 = llvm.mlir.undef : !llvm.struct<(array<22 x ptr>)>
    %60 = llvm.insertvalue %58, %59[0] : !llvm.struct<(array<22 x ptr>)> 
    %61 = llvm.mlir.addressof @_ZTVN8tutorial18Person_PhoneNumberE : !llvm.ptr
    %62 = llvm.getelementptr inbounds %61[%0, 0, %4] : (!llvm.ptr, i64, i64) -> !llvm.ptr, !llvm.struct<(array<22 x ptr>)>
    %63 = llvm.mlir.constant(3 : i32) : i32
    %64 = llvm.mlir.constant(1 : i64) : i64
    %65 = llvm.mlir.constant(-2 : i64) : i64
    %66 = llvm.mlir.addressof @_ZN6google8protobuf8internal26fixed_address_empty_stringB5cxx11E : !llvm.ptr
    %67 = llvm.mlir.constant(2 : i32) : i32
    %68 = llvm.getelementptr inbounds %arg0[%0, 0, 0, 1, 0] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.tutorial::Person_PhoneNumber", (struct<"class.google::protobuf::Message", (struct<"class.google::protobuf::MessageLite", (ptr, struct<"class.google::protobuf::internal::InternalMetadata", (ptr)>)>)>, struct<"struct.google::protobuf::internal::ArenaStringPtr", (ptr)>, i32, struct<"class.google::protobuf::internal::CachedSize", (struct<"struct.std::atomic", (struct<"struct.std::__atomic_base", (i32)>)>)>)>
    llvm.store %3, %68 {alignment = 8 : i64, tbaa = [#tbaa_tag5]} : !llvm.ptr, !llvm.ptr
    %69 = llvm.getelementptr inbounds %arg0[%0, 0, 0, 0] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.tutorial::Person_PhoneNumber", (struct<"class.google::protobuf::Message", (struct<"class.google::protobuf::MessageLite", (ptr, struct<"class.google::protobuf::internal::InternalMetadata", (ptr)>)>)>, struct<"struct.google::protobuf::internal::ArenaStringPtr", (ptr)>, i32, struct<"class.google::protobuf::internal::CachedSize", (struct<"struct.std::atomic", (struct<"struct.std::__atomic_base", (i32)>)>)>)>
    llvm.store %62, %69 {alignment = 8 : i64, tbaa = [#tbaa_tag]} : !llvm.ptr, !llvm.ptr
    %70 = llvm.getelementptr inbounds %arg0[%0, 3, 0, 0, 0] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.tutorial::Person_PhoneNumber", (struct<"class.google::protobuf::Message", (struct<"class.google::protobuf::MessageLite", (ptr, struct<"class.google::protobuf::internal::InternalMetadata", (ptr)>)>)>, struct<"struct.google::protobuf::internal::ArenaStringPtr", (ptr)>, i32, struct<"class.google::protobuf::internal::CachedSize", (struct<"struct.std::atomic", (struct<"struct.std::__atomic_base", (i32)>)>)>)>
    llvm.store %1, %70 {alignment = 4 : i64, tbaa = [#tbaa_tag6]} : i32, !llvm.ptr
    %71 = llvm.getelementptr inbounds %arg1[%0, 0, 0, 1, 0] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.tutorial::Person_PhoneNumber", (struct<"class.google::protobuf::Message", (struct<"class.google::protobuf::MessageLite", (ptr, struct<"class.google::protobuf::internal::InternalMetadata", (ptr)>)>)>, struct<"struct.google::protobuf::internal::ArenaStringPtr", (ptr)>, i32, struct<"class.google::protobuf::internal::CachedSize", (struct<"struct.std::atomic", (struct<"struct.std::__atomic_base", (i32)>)>)>)>
    %72 = llvm.load %71 {alignment = 8 : i64, tbaa = [#tbaa_tag5]} : !llvm.ptr -> !llvm.ptr
    %73 = llvm.ptrtoint %72 : !llvm.ptr to i64
    %74 = llvm.and %73, %64  : i64
    %75 = llvm.icmp "eq" %74, %0 : i64
    llvm.cond_br %75, ^bb2, ^bb1
  ^bb1:  // pred: ^bb0
    %76 = llvm.getelementptr inbounds %arg0[%0, 0, 0, 1] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.tutorial::Person_PhoneNumber", (struct<"class.google::protobuf::Message", (struct<"class.google::protobuf::MessageLite", (ptr, struct<"class.google::protobuf::internal::InternalMetadata", (ptr)>)>)>, struct<"struct.google::protobuf::internal::ArenaStringPtr", (ptr)>, i32, struct<"class.google::protobuf::internal::CachedSize", (struct<"struct.std::atomic", (struct<"struct.std::__atomic_base", (i32)>)>)>)>
    %77 = llvm.and %73, %65  : i64
    %78 = llvm.inttoptr %77 : i64 to !llvm.ptr
    %79 = llvm.getelementptr inbounds %78[%0, 1] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.google::protobuf::internal::InternalMetadata::Container", (struct<"struct.google::protobuf::internal::InternalMetadata::ContainerBase", (ptr)>, struct<"class.google::protobuf::UnknownFieldSet", (struct<"class.std::vector", (struct<"struct.std::_Vector_base", (struct<"struct.std::_Vector_base<google::protobuf::UnknownField, std::allocator<google::protobuf::UnknownField>>::_Vector_impl", (struct<"struct.std::_Vector_base<google::protobuf::UnknownField, std::allocator<google::protobuf::UnknownField>>::_Vector_impl_data", (ptr, ptr, ptr)>)>)>)>)>)>
    %80 = llvm.call @_ZN6google8protobuf8internal16InternalMetadata27mutable_unknown_fields_slowINS0_15UnknownFieldSetEEEPT_v(%76) : (!llvm.ptr) -> !llvm.ptr
    llvm.call @_ZN6google8protobuf15UnknownFieldSet9MergeFromERKS1_(%80, %79) : (!llvm.ptr, !llvm.ptr) -> ()
    llvm.br ^bb2
  ^bb2:  // 2 preds: ^bb0, ^bb1
    %81 = llvm.getelementptr inbounds %arg0[%0, 1] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.tutorial::Person_PhoneNumber", (struct<"class.google::protobuf::Message", (struct<"class.google::protobuf::MessageLite", (ptr, struct<"class.google::protobuf::internal::InternalMetadata", (ptr)>)>)>, struct<"struct.google::protobuf::internal::ArenaStringPtr", (ptr)>, i32, struct<"class.google::protobuf::internal::CachedSize", (struct<"struct.std::atomic", (struct<"struct.std::__atomic_base", (i32)>)>)>)>
    %82 = llvm.getelementptr inbounds %81[%0, 0] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.google::protobuf::internal::ArenaStringPtr", (ptr)>
    llvm.store %66, %82 {alignment = 8 : i64, tbaa = [#tbaa_tag8]} : !llvm.ptr, !llvm.ptr
    %83 = llvm.getelementptr inbounds %arg1[%0, 1, 0] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.tutorial::Person_PhoneNumber", (struct<"class.google::protobuf::Message", (struct<"class.google::protobuf::MessageLite", (ptr, struct<"class.google::protobuf::internal::InternalMetadata", (ptr)>)>)>, struct<"struct.google::protobuf::internal::ArenaStringPtr", (ptr)>, i32, struct<"class.google::protobuf::internal::CachedSize", (struct<"struct.std::atomic", (struct<"struct.std::__atomic_base", (i32)>)>)>)>
    %84 = llvm.load %83 {alignment = 8 : i64, tbaa = [#tbaa_tag8]} : !llvm.ptr -> !llvm.ptr
    %85 = llvm.getelementptr inbounds %84[%0, 1] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.std::__cxx11::basic_string", (struct<"struct.std::__cxx11::basic_string<char>::_Alloc_hider", (ptr)>, i64, struct<"union.anon", (i64, array<8 x i8>)>)>
    %86 = llvm.load %85 {alignment = 8 : i64, tbaa = [#tbaa_tag25]} : !llvm.ptr -> i64
    %87 = llvm.icmp "eq" %86, %0 : i64
    llvm.cond_br %87, ^bb7, ^bb3
  ^bb3:  // pred: ^bb2
    %88 = llvm.load %68 {alignment = 8 : i64, tbaa = [#tbaa_tag5]} : !llvm.ptr -> !llvm.ptr
    %89 = llvm.ptrtoint %88 : !llvm.ptr to i64
    %90 = llvm.and %89, %64  : i64
    %91 = llvm.icmp "eq" %90, %0 : i64
    %92 = llvm.and %89, %65  : i64
    llvm.cond_br %91 weights([2000, 1]), ^bb5, ^bb4
  ^bb4:  // pred: ^bb3
    %93 = llvm.inttoptr %92 : i64 to !llvm.ptr
    %94 = llvm.getelementptr inbounds %93[%0, 0] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.google::protobuf::internal::InternalMetadata::ContainerBase", (ptr)>
    %95 = llvm.load %94 {alignment = 8 : i64, tbaa = [#tbaa_tag9]} : !llvm.ptr -> !llvm.ptr
    llvm.br ^bb6(%95 : !llvm.ptr)
  ^bb5:  // pred: ^bb3
    %96 = llvm.inttoptr %92 : i64 to !llvm.ptr
    llvm.br ^bb6(%96 : !llvm.ptr)
  ^bb6(%97: !llvm.ptr):  // 2 preds: ^bb4, ^bb5
    llvm.call @_ZN6google8protobuf8internal14ArenaStringPtr14CreateInstanceEPNS0_5ArenaEPKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE(%81, %97, %84) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
    llvm.br ^bb7
  ^bb7:  // 2 preds: ^bb2, ^bb6
    %98 = llvm.getelementptr inbounds %arg1[%0, 2] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.tutorial::Person_PhoneNumber", (struct<"class.google::protobuf::Message", (struct<"class.google::protobuf::MessageLite", (ptr, struct<"class.google::protobuf::internal::InternalMetadata", (ptr)>)>)>, struct<"struct.google::protobuf::internal::ArenaStringPtr", (ptr)>, i32, struct<"class.google::protobuf::internal::CachedSize", (struct<"struct.std::atomic", (struct<"struct.std::__atomic_base", (i32)>)>)>)>
    %99 = llvm.load %98 {alignment = 8 : i64, tbaa = [#tbaa_tag24]} : !llvm.ptr -> i32
    %100 = llvm.getelementptr inbounds %arg0[%0, 2] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.tutorial::Person_PhoneNumber", (struct<"class.google::protobuf::Message", (struct<"class.google::protobuf::MessageLite", (ptr, struct<"class.google::protobuf::internal::InternalMetadata", (ptr)>)>)>, struct<"struct.google::protobuf::internal::ArenaStringPtr", (ptr)>, i32, struct<"class.google::protobuf::internal::CachedSize", (struct<"struct.std::atomic", (struct<"struct.std::__atomic_base", (i32)>)>)>)>
    llvm.store %99, %100 {alignment = 8 : i64, tbaa = [#tbaa_tag24]} : i32, !llvm.ptr
    llvm.return
  }
  llvm.func unnamed_addr @_ZN8tutorial18Person_PhoneNumberD2Ev(%arg0: !llvm.ptr {llvm.align = 8 : i64, llvm.dereferenceable = 32 : i64, llvm.nonnull, llvm.noundef}) attributes {alignment = 2 : i64, frame_pointer = #llvm.framePointerKind<none>, passthrough = ["nounwind", ["uwtable", "2"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], ["tune-cpu", "generic"]], personality = @__gxx_personality_v0, target_cpu = "x86-64", target_features = #llvm.target_features<["+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>} {
    %0 = llvm.mlir.constant(0 : i64) : i64
    %1 = llvm.mlir.constant(0 : i32) : i32
    %2 = llvm.mlir.constant(1 : i32) : i32
    %3 = llvm.mlir.zero : !llvm.ptr
    llvm.invoke @_ZN8tutorial18Person_PhoneNumber10SharedDtorEv(%arg0) to ^bb1 unwind ^bb3 : (!llvm.ptr) -> ()
  ^bb1:  // pred: ^bb0
    %4 = llvm.getelementptr inbounds %arg0[%0, 0, 0, 1] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.tutorial::Person_PhoneNumber", (struct<"class.google::protobuf::Message", (struct<"class.google::protobuf::MessageLite", (ptr, struct<"class.google::protobuf::internal::InternalMetadata", (ptr)>)>)>, struct<"struct.google::protobuf::internal::ArenaStringPtr", (ptr)>, i32, struct<"class.google::protobuf::internal::CachedSize", (struct<"struct.std::atomic", (struct<"struct.std::__atomic_base", (i32)>)>)>)>
    llvm.invoke @_ZN6google8protobuf8internal16InternalMetadata6DeleteINS0_15UnknownFieldSetEEEvv(%4) to ^bb2 unwind ^bb3 : (!llvm.ptr) -> ()
  ^bb2:  // pred: ^bb1
    llvm.return
  ^bb3:  // 2 preds: ^bb0, ^bb1
    %5 = llvm.landingpad (catch %3 : !llvm.ptr) : !llvm.struct<(ptr, i32)>
    %6 = llvm.extractvalue %5[0] : !llvm.struct<(ptr, i32)> 
    llvm.call @__clang_call_terminate(%6) : (!llvm.ptr) -> ()
    llvm.unreachable
  }
  llvm.func linkonce_odr local_unnamed_addr @_ZN8tutorial18Person_PhoneNumber10SharedDtorEv(%arg0: !llvm.ptr {llvm.align = 8 : i64, llvm.dereferenceable = 32 : i64, llvm.nonnull, llvm.noundef}) comdat(@__llvm_global_comdat::@_ZN8tutorial18Person_PhoneNumber10SharedDtorEv) attributes {alignment = 2 : i64, frame_pointer = #llvm.framePointerKind<none>, passthrough = ["inlinehint", ["uwtable", "2"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], ["tune-cpu", "generic"]], personality = @__gxx_personality_v0, target_cpu = "x86-64", target_features = #llvm.target_features<["+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>} {
    %0 = llvm.mlir.constant(1 : i32) : i32
    %1 = llvm.mlir.constant(0 : i64) : i64
    %2 = llvm.mlir.constant(0 : i32) : i32
    %3 = llvm.mlir.constant(1 : i64) : i64
    %4 = llvm.mlir.constant(-2 : i64) : i64
    %5 = llvm.mlir.zero : !llvm.ptr
    %6 = llvm.mlir.constant(3 : i32) : i32
    %7 = llvm.mlir.constant("build/addressbook.pb.cc\00") : !llvm.array<24 x i8>
    %8 = llvm.mlir.addressof @".str.5" : !llvm.ptr
    %9 = llvm.mlir.constant(218 : i32) : i32
    %10 = llvm.mlir.constant("CHECK failed: GetArena() == nullptr: \00") : !llvm.array<38 x i8>
    %11 = llvm.mlir.addressof @".str.12" : !llvm.ptr
    %12 = llvm.mlir.addressof @_ZN6google8protobuf8internal26fixed_address_empty_stringB5cxx11E : !llvm.ptr
    %13 = llvm.mlir.constant(2 : i32) : i32
    %14 = llvm.alloca %0 x !llvm.struct<"class.google::protobuf::internal::LogMessage", (i32, ptr, i32, struct<"class.std::__cxx11::basic_string", (struct<"struct.std::__cxx11::basic_string<char>::_Alloc_hider", (ptr)>, i64, struct<"union.anon", (i64, array<8 x i8>)>)>)> {alignment = 8 : i64} : (i32) -> !llvm.ptr
    %15 = llvm.alloca %0 x !llvm.struct<"class.google::protobuf::internal::LogFinisher", (i8)> {alignment = 1 : i64} : (i32) -> !llvm.ptr
    %16 = llvm.getelementptr inbounds %arg0[%1, 0, 0, 1, 0] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.tutorial::Person_PhoneNumber", (struct<"class.google::protobuf::Message", (struct<"class.google::protobuf::MessageLite", (ptr, struct<"class.google::protobuf::internal::InternalMetadata", (ptr)>)>)>, struct<"struct.google::protobuf::internal::ArenaStringPtr", (ptr)>, i32, struct<"class.google::protobuf::internal::CachedSize", (struct<"struct.std::atomic", (struct<"struct.std::__atomic_base", (i32)>)>)>)>
    %17 = llvm.load %16 {alignment = 8 : i64, tbaa = [#tbaa_tag5]} : !llvm.ptr -> !llvm.ptr
    %18 = llvm.ptrtoint %17 : !llvm.ptr to i64
    %19 = llvm.and %18, %3  : i64
    %20 = llvm.icmp "eq" %19, %1 : i64
    %21 = llvm.and %18, %4  : i64
    llvm.cond_br %20 weights([2000, 1]), ^bb2, ^bb1
  ^bb1:  // pred: ^bb0
    %22 = llvm.inttoptr %21 : i64 to !llvm.ptr
    %23 = llvm.getelementptr inbounds %22[%1, 0] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.google::protobuf::internal::InternalMetadata::ContainerBase", (ptr)>
    %24 = llvm.load %23 {alignment = 8 : i64, tbaa = [#tbaa_tag9]} : !llvm.ptr -> !llvm.ptr
    llvm.br ^bb3(%24 : !llvm.ptr)
  ^bb2:  // pred: ^bb0
    %25 = llvm.inttoptr %21 : i64 to !llvm.ptr
    llvm.br ^bb3(%25 : !llvm.ptr)
  ^bb3(%26: !llvm.ptr):  // 2 preds: ^bb1, ^bb2
    %27 = llvm.icmp "eq" %26, %5 : !llvm.ptr
    %28 = llvm.getelementptr inbounds %15[%1, 0] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.google::protobuf::internal::LogFinisher", (i8)>
    llvm.intr.lifetime.start 1, %28 : !llvm.ptr
    llvm.cond_br %27, ^bb6, ^bb4
  ^bb4:  // pred: ^bb3
    %29 = llvm.bitcast %14 : !llvm.ptr to !llvm.ptr
    llvm.intr.lifetime.start 56, %29 : !llvm.ptr
    llvm.call @_ZN6google8protobuf8internal10LogMessageC1ENS0_8LogLevelEPKci(%14, %6, %8, %9) : (!llvm.ptr, i32, !llvm.ptr, i32) -> ()
    %30 = llvm.invoke @_ZN6google8protobuf8internal10LogMessagelsEPKc(%14, %11) to ^bb5 unwind ^bb13 : (!llvm.ptr, !llvm.ptr) -> !llvm.ptr
  ^bb5:  // pred: ^bb4
    llvm.invoke @_ZN6google8protobuf8internal11LogFinisheraSERNS1_10LogMessageE(%15, %30) to ^bb7 unwind ^bb14 : (!llvm.ptr, !llvm.ptr) -> ()
  ^bb6:  // pred: ^bb3
    llvm.intr.lifetime.end 1, %28 : !llvm.ptr
    llvm.br ^bb8
  ^bb7:  // pred: ^bb5
    llvm.intr.lifetime.end 1, %28 : !llvm.ptr
    llvm.call @_ZN6google8protobuf8internal10LogMessageD1Ev(%14) : (!llvm.ptr) -> ()
    llvm.intr.lifetime.end 56, %29 : !llvm.ptr
    llvm.br ^bb8
  ^bb8:  // 2 preds: ^bb6, ^bb7
    %31 = llvm.getelementptr inbounds %arg0[%1, 1, 0] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.tutorial::Person_PhoneNumber", (struct<"class.google::protobuf::Message", (struct<"class.google::protobuf::MessageLite", (ptr, struct<"class.google::protobuf::internal::InternalMetadata", (ptr)>)>)>, struct<"struct.google::protobuf::internal::ArenaStringPtr", (ptr)>, i32, struct<"class.google::protobuf::internal::CachedSize", (struct<"struct.std::atomic", (struct<"struct.std::__atomic_base", (i32)>)>)>)>
    %32 = llvm.load %31 {alignment = 8 : i64, tbaa = [#tbaa_tag8]} : !llvm.ptr -> !llvm.ptr
    %33 = llvm.icmp "eq" %32, %12 : !llvm.ptr
    %34 = llvm.icmp "eq" %32, %5 : !llvm.ptr
    %35 = llvm.or %33, %34  : i1
    llvm.cond_br %35, ^bb12, ^bb9
  ^bb9:  // pred: ^bb8
    %36 = llvm.getelementptr inbounds %32[%1, 0, 0] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.std::__cxx11::basic_string", (struct<"struct.std::__cxx11::basic_string<char>::_Alloc_hider", (ptr)>, i64, struct<"union.anon", (i64, array<8 x i8>)>)>
    %37 = llvm.load %36 {alignment = 8 : i64, tbaa = [#tbaa_tag26]} : !llvm.ptr -> !llvm.ptr
    %38 = llvm.getelementptr inbounds %32[%1, 2] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.std::__cxx11::basic_string", (struct<"struct.std::__cxx11::basic_string<char>::_Alloc_hider", (ptr)>, i64, struct<"union.anon", (i64, array<8 x i8>)>)>
    %39 = llvm.bitcast %38 : !llvm.ptr to !llvm.ptr
    %40 = llvm.icmp "eq" %37, %39 : !llvm.ptr
    llvm.cond_br %40, ^bb11, ^bb10
  ^bb10:  // pred: ^bb9
    llvm.call @_ZdlPv(%37) : (!llvm.ptr) -> ()
    llvm.br ^bb11
  ^bb11:  // 2 preds: ^bb9, ^bb10
    %41 = llvm.bitcast %32 : !llvm.ptr to !llvm.ptr
    llvm.call @_ZdlPv(%41) : (!llvm.ptr) -> ()
    llvm.br ^bb12
  ^bb12:  // 2 preds: ^bb8, ^bb11
    llvm.return
  ^bb13:  // pred: ^bb4
    %42 = llvm.landingpad cleanup : !llvm.struct<(ptr, i32)>
    llvm.br ^bb15(%42 : !llvm.struct<(ptr, i32)>)
  ^bb14:  // pred: ^bb5
    %43 = llvm.landingpad cleanup : !llvm.struct<(ptr, i32)>
    llvm.intr.lifetime.end 1, %28 : !llvm.ptr
    llvm.br ^bb15(%43 : !llvm.struct<(ptr, i32)>)
  ^bb15(%44: !llvm.struct<(ptr, i32)>):  // 2 preds: ^bb13, ^bb14
    llvm.call @_ZN6google8protobuf8internal10LogMessageD1Ev(%14) : (!llvm.ptr) -> ()
    llvm.intr.lifetime.end 56, %29 : !llvm.ptr
    llvm.resume %44 : !llvm.struct<(ptr, i32)>
  }
  llvm.func linkonce_odr local_unnamed_addr @_ZN6google8protobuf8internal16InternalMetadata6DeleteINS0_15UnknownFieldSetEEEvv(%arg0: !llvm.ptr {llvm.align = 8 : i64, llvm.dereferenceable = 8 : i64, llvm.nonnull, llvm.noundef}) comdat(@__llvm_global_comdat::@_ZN6google8protobuf8internal16InternalMetadata6DeleteINS0_15UnknownFieldSetEEEvv) attributes {alignment = 2 : i64, frame_pointer = #llvm.framePointerKind<none>, passthrough = [["uwtable", "2"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], ["tune-cpu", "generic"]], personality = @__gxx_personality_v0, target_cpu = "x86-64", target_features = #llvm.target_features<["+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>} {
    %0 = llvm.mlir.constant(0 : i64) : i64
    %1 = llvm.mlir.constant(0 : i32) : i32
    %2 = llvm.mlir.constant(1 : i64) : i64
    %3 = llvm.mlir.constant(-2 : i64) : i64
    %4 = llvm.mlir.zero : !llvm.ptr
    %5 = llvm.mlir.constant(1 : i32) : i32
    %6 = llvm.getelementptr inbounds %arg0[%0, 0] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.google::protobuf::internal::InternalMetadata", (ptr)>
    %7 = llvm.load %6 {alignment = 8 : i64, tbaa = [#tbaa_tag5]} : !llvm.ptr -> !llvm.ptr
    %8 = llvm.ptrtoint %7 : !llvm.ptr to i64
    %9 = llvm.and %8, %2  : i64
    %10 = llvm.icmp "eq" %9, %0 : i64
    llvm.cond_br %10, ^bb10, ^bb1
  ^bb1:  // pred: ^bb0
    %11 = llvm.and %8, %3  : i64
    %12 = llvm.inttoptr %11 : i64 to !llvm.ptr
    %13 = llvm.getelementptr inbounds %12[%0, 0] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.google::protobuf::internal::InternalMetadata::ContainerBase", (ptr)>
    %14 = llvm.load %13 {alignment = 8 : i64, tbaa = [#tbaa_tag9]} : !llvm.ptr -> !llvm.ptr
    %15 = llvm.icmp "eq" %14, %4 : !llvm.ptr
    llvm.cond_br %15, ^bb2, ^bb10
  ^bb2:  // pred: ^bb1
    %16 = llvm.inttoptr %11 : i64 to !llvm.ptr
    %17 = llvm.icmp "eq" %11, %0 : i64
    llvm.cond_br %17, ^bb10, ^bb3
  ^bb3:  // pred: ^bb2
    %18 = llvm.getelementptr inbounds %16[%0, 1] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.google::protobuf::internal::InternalMetadata::Container", (struct<"struct.google::protobuf::internal::InternalMetadata::ContainerBase", (ptr)>, struct<"class.google::protobuf::UnknownFieldSet", (struct<"class.std::vector", (struct<"struct.std::_Vector_base", (struct<"struct.std::_Vector_base<google::protobuf::UnknownField, std::allocator<google::protobuf::UnknownField>>::_Vector_impl", (struct<"struct.std::_Vector_base<google::protobuf::UnknownField, std::allocator<google::protobuf::UnknownField>>::_Vector_impl_data", (ptr, ptr, ptr)>)>)>)>)>)>
    %19 = llvm.getelementptr inbounds %18[%0, 0, 0, 0, 0, 0] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.google::protobuf::UnknownFieldSet", (struct<"class.std::vector", (struct<"struct.std::_Vector_base", (struct<"struct.std::_Vector_base<google::protobuf::UnknownField, std::allocator<google::protobuf::UnknownField>>::_Vector_impl", (struct<"struct.std::_Vector_base<google::protobuf::UnknownField, std::allocator<google::protobuf::UnknownField>>::_Vector_impl_data", (ptr, ptr, ptr)>)>)>)>)>
    %20 = llvm.load %19 {alignment = 8 : i64, tbaa = [#tbaa_tag2]} : !llvm.ptr -> !llvm.ptr
    %21 = llvm.getelementptr inbounds %16[%0, 1, 0, 0, 0, 0, 1] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.google::protobuf::internal::InternalMetadata::Container", (struct<"struct.google::protobuf::internal::InternalMetadata::ContainerBase", (ptr)>, struct<"class.google::protobuf::UnknownFieldSet", (struct<"class.std::vector", (struct<"struct.std::_Vector_base", (struct<"struct.std::_Vector_base<google::protobuf::UnknownField, std::allocator<google::protobuf::UnknownField>>::_Vector_impl", (struct<"struct.std::_Vector_base<google::protobuf::UnknownField, std::allocator<google::protobuf::UnknownField>>::_Vector_impl_data", (ptr, ptr, ptr)>)>)>)>)>)>
    %22 = llvm.load %21 {alignment = 8 : i64, tbaa = [#tbaa_tag2]} : !llvm.ptr -> !llvm.ptr
    %23 = llvm.icmp "eq" %20, %22 : !llvm.ptr
    llvm.cond_br %23, ^bb6(%20 : !llvm.ptr), ^bb4
  ^bb4:  // pred: ^bb3
    llvm.invoke @_ZN6google8protobuf15UnknownFieldSet13ClearFallbackEv(%18) to ^bb5 unwind ^bb8 : (!llvm.ptr) -> ()
  ^bb5:  // pred: ^bb4
    %24 = llvm.load %19 {alignment = 8 : i64, tbaa = [#tbaa_tag10]} : !llvm.ptr -> !llvm.ptr
    llvm.br ^bb6(%24 : !llvm.ptr)
  ^bb6(%25: !llvm.ptr):  // 2 preds: ^bb3, ^bb5
    %26 = llvm.icmp "eq" %25, %4 : !llvm.ptr
    llvm.cond_br %26, ^bb9, ^bb7
  ^bb7:  // pred: ^bb6
    %27 = llvm.bitcast %25 : !llvm.ptr to !llvm.ptr
    llvm.call @_ZdlPv(%27) : (!llvm.ptr) -> ()
    llvm.br ^bb9
  ^bb8:  // pred: ^bb4
    %28 = llvm.landingpad (catch %4 : !llvm.ptr) : !llvm.struct<(ptr, i32)>
    %29 = llvm.extractvalue %28[0] : !llvm.struct<(ptr, i32)> 
    %30 = llvm.getelementptr inbounds %18[%0, 0] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.google::protobuf::UnknownFieldSet", (struct<"class.std::vector", (struct<"struct.std::_Vector_base", (struct<"struct.std::_Vector_base<google::protobuf::UnknownField, std::allocator<google::protobuf::UnknownField>>::_Vector_impl", (struct<"struct.std::_Vector_base<google::protobuf::UnknownField, std::allocator<google::protobuf::UnknownField>>::_Vector_impl_data", (ptr, ptr, ptr)>)>)>)>)>
    llvm.call @_ZNSt6vectorIN6google8protobuf12UnknownFieldESaIS2_EED2Ev(%30) : (!llvm.ptr) -> ()
    llvm.call @__clang_call_terminate(%29) : (!llvm.ptr) -> ()
    llvm.unreachable
  ^bb9:  // 2 preds: ^bb6, ^bb7
    %31 = llvm.inttoptr %11 : i64 to !llvm.ptr
    llvm.call @_ZdlPv(%31) : (!llvm.ptr) -> ()
    llvm.br ^bb10
  ^bb10:  // 4 preds: ^bb0, ^bb1, ^bb2, ^bb9
    llvm.return
  }
  llvm.func linkonce_odr hidden local_unnamed_addr @__clang_call_terminate(%arg0: !llvm.ptr) comdat(@__llvm_global_comdat::@__clang_call_terminate) attributes {passthrough = ["noinline", "noreturn", "nounwind"]} {
    %0 = llvm.call @__cxa_begin_catch(%arg0) : (!llvm.ptr) -> !llvm.ptr
    llvm.call @_ZSt9terminatev() : () -> ()
    llvm.unreachable
  }
  llvm.func local_unnamed_addr @__cxa_begin_catch(!llvm.ptr) -> !llvm.ptr
  llvm.func local_unnamed_addr @_ZSt9terminatev()
  llvm.func unnamed_addr @_ZN8tutorial18Person_PhoneNumberD0Ev(%arg0: !llvm.ptr {llvm.align = 8 : i64, llvm.dereferenceable = 32 : i64, llvm.nonnull, llvm.noundef}) attributes {alignment = 2 : i64, frame_pointer = #llvm.framePointerKind<none>, passthrough = ["nounwind", ["uwtable", "2"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], ["tune-cpu", "generic"]], personality = @__gxx_personality_v0, target_cpu = "x86-64", target_features = #llvm.target_features<["+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>} {
    %0 = llvm.mlir.constant(0 : i64) : i64
    %1 = llvm.mlir.constant(0 : i32) : i32
    %2 = llvm.mlir.constant(1 : i32) : i32
    %3 = llvm.mlir.zero : !llvm.ptr
    llvm.invoke @_ZN8tutorial18Person_PhoneNumber10SharedDtorEv(%arg0) to ^bb1 unwind ^bb2 : (!llvm.ptr) -> ()
  ^bb1:  // pred: ^bb0
    %4 = llvm.getelementptr inbounds %arg0[%0, 0, 0, 1] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.tutorial::Person_PhoneNumber", (struct<"class.google::protobuf::Message", (struct<"class.google::protobuf::MessageLite", (ptr, struct<"class.google::protobuf::internal::InternalMetadata", (ptr)>)>)>, struct<"struct.google::protobuf::internal::ArenaStringPtr", (ptr)>, i32, struct<"class.google::protobuf::internal::CachedSize", (struct<"struct.std::atomic", (struct<"struct.std::__atomic_base", (i32)>)>)>)>
    llvm.invoke @_ZN6google8protobuf8internal16InternalMetadata6DeleteINS0_15UnknownFieldSetEEEvv(%4) to ^bb3 unwind ^bb2 : (!llvm.ptr) -> ()
  ^bb2:  // 2 preds: ^bb0, ^bb1
    %5 = llvm.landingpad (catch %3 : !llvm.ptr) : !llvm.struct<(ptr, i32)>
    %6 = llvm.extractvalue %5[0] : !llvm.struct<(ptr, i32)> 
    llvm.call @__clang_call_terminate(%6) : (!llvm.ptr) -> ()
    llvm.unreachable
  ^bb3:  // pred: ^bb1
    %7 = llvm.bitcast %arg0 : !llvm.ptr to !llvm.ptr
    llvm.call @_ZdlPv(%7) : (!llvm.ptr) -> ()
    llvm.return
  }
  llvm.func local_unnamed_addr @_ZdlPv(!llvm.ptr {llvm.noundef}) attributes {frame_pointer = #llvm.framePointerKind<none>, passthrough = ["nobuiltin", "nounwind", ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], ["tune-cpu", "generic"]], target_cpu = "x86-64", target_features = #llvm.target_features<["+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>}
  llvm.func local_unnamed_addr @_ZN8tutorial18Person_PhoneNumber9ArenaDtorEPv(%arg0: !llvm.ptr {llvm.nocapture, llvm.noundef}) attributes {alignment = 2 : i64, frame_pointer = #llvm.framePointerKind<none>, memory = #llvm.memory_effects<other = none, argMem = none, inaccessibleMem = none>, passthrough = ["mustprogress", "nofree", "norecurse", "nosync", "nounwind", "willreturn", ["uwtable", "2"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], ["tune-cpu", "generic"]], target_cpu = "x86-64", target_features = #llvm.target_features<["+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>} {
    llvm.return
  }
  llvm.func unnamed_addr @_ZNK8tutorial18Person_PhoneNumber13SetCachedSizeEi(%arg0: !llvm.ptr {llvm.align = 8 : i64, llvm.dereferenceable = 32 : i64, llvm.nocapture, llvm.nonnull, llvm.noundef, llvm.writeonly}, %arg1: i32 {llvm.noundef}) attributes {alignment = 2 : i64, frame_pointer = #llvm.framePointerKind<none>, passthrough = ["mustprogress", "nofree", "norecurse", "nounwind", "willreturn", ["uwtable", "2"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], ["tune-cpu", "generic"]], personality = @__gxx_personality_v0, target_cpu = "x86-64", target_features = #llvm.target_features<["+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>} {
    %0 = llvm.mlir.constant(0 : i64) : i64
    %1 = llvm.mlir.constant(3 : i32) : i32
    %2 = llvm.mlir.constant(0 : i32) : i32
    %3 = llvm.getelementptr inbounds %arg0[%0, 3, 0, 0, 0] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.tutorial::Person_PhoneNumber", (struct<"class.google::protobuf::Message", (struct<"class.google::protobuf::MessageLite", (ptr, struct<"class.google::protobuf::internal::InternalMetadata", (ptr)>)>)>, struct<"struct.google::protobuf::internal::ArenaStringPtr", (ptr)>, i32, struct<"class.google::protobuf::internal::CachedSize", (struct<"struct.std::atomic", (struct<"struct.std::__atomic_base", (i32)>)>)>)>
    llvm.store %arg1, %3 atomic monotonic {alignment = 4 : i64} : i32, !llvm.ptr
    llvm.return
  }
  llvm.func local_unnamed_addr @_ZN8tutorial18Person_PhoneNumber16default_instanceEv() -> (!llvm.ptr {llvm.align = 8 : i64, llvm.dereferenceable = 32 : i64, llvm.nonnull, llvm.noundef}) attributes {alignment = 2 : i64, frame_pointer = #llvm.framePointerKind<none>, passthrough = ["mustprogress", ["uwtable", "2"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], ["tune-cpu", "generic"]], target_cpu = "x86-64", target_features = #llvm.target_features<["+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>} {
    %0 = llvm.mlir.undef : !llvm.array<0 x ptr>
    %1 = llvm.mlir.zero : !llvm.ptr
    %2 = llvm.mlir.addressof @__gxx_personality_v0 : !llvm.ptr
    %3 = llvm.mlir.addressof @_ZL59InitDefaultsscc_info_Person_PhoneNumber_addressbook_2eprotov : !llvm.ptr
    %4 = llvm.mlir.constant(0 : i32) : i32
    %5 = llvm.mlir.constant(-1 : i32) : i32
    %6 = llvm.mlir.undef : !llvm.struct<(i32)>
    %7 = llvm.insertvalue %5, %6[0] : !llvm.struct<(i32)> 
    %8 = llvm.mlir.undef : !llvm.struct<(struct<(i32)>, i32, i32, ptr)>
    %9 = llvm.insertvalue %7, %8[0] : !llvm.struct<(struct<(i32)>, i32, i32, ptr)> 
    %10 = llvm.insertvalue %4, %9[1] : !llvm.struct<(struct<(i32)>, i32, i32, ptr)> 
    %11 = llvm.insertvalue %4, %10[2] : !llvm.struct<(struct<(i32)>, i32, i32, ptr)> 
    %12 = llvm.insertvalue %3, %11[3] : !llvm.struct<(struct<(i32)>, i32, i32, ptr)> 
    %13 = llvm.mlir.undef : !llvm.struct<(struct<(struct<(i32)>, i32, i32, ptr)>, array<0 x ptr>)>
    %14 = llvm.insertvalue %12, %13[0] : !llvm.struct<(struct<(struct<(i32)>, i32, i32, ptr)>, array<0 x ptr>)> 
    %15 = llvm.insertvalue %0, %14[1] : !llvm.struct<(struct<(struct<(i32)>, i32, i32, ptr)>, array<0 x ptr>)> 
    %16 = llvm.mlir.addressof @scc_info_Person_PhoneNumber_addressbook_2eproto : !llvm.ptr
    %17 = llvm.mlir.constant(0 : i8) : i8
    %18 = llvm.mlir.constant(dense<0> : tensor<24xi8>) : !llvm.array<24 x i8>
    %19 = llvm.mlir.constant(0 : i64) : i64
    %20 = llvm.mlir.undef : !llvm.struct<"union.google::protobuf::internal::ExplicitlyConstructed<tutorial::Person_PhoneNumber>::AlignedUnion", (i64, array<24 x i8>)>
    %21 = llvm.insertvalue %19, %20[0] : !llvm.struct<"union.google::protobuf::internal::ExplicitlyConstructed<tutorial::Person_PhoneNumber>::AlignedUnion", (i64, array<24 x i8>)> 
    %22 = llvm.insertvalue %18, %21[1] : !llvm.struct<"union.google::protobuf::internal::ExplicitlyConstructed<tutorial::Person_PhoneNumber>::AlignedUnion", (i64, array<24 x i8>)> 
    %23 = llvm.mlir.undef : !llvm.struct<"class.google::protobuf::internal::ExplicitlyConstructed", (struct<"union.google::protobuf::internal::ExplicitlyConstructed<tutorial::Person_PhoneNumber>::AlignedUnion", (i64, array<24 x i8>)>)>
    %24 = llvm.insertvalue %22, %23[0] : !llvm.struct<"class.google::protobuf::internal::ExplicitlyConstructed", (struct<"union.google::protobuf::internal::ExplicitlyConstructed<tutorial::Person_PhoneNumber>::AlignedUnion", (i64, array<24 x i8>)>)> 
    %25 = llvm.mlir.undef : !llvm.struct<"class.tutorial::Person_PhoneNumberDefaultTypeInternal", (struct<"class.google::protobuf::internal::ExplicitlyConstructed", (struct<"union.google::protobuf::internal::ExplicitlyConstructed<tutorial::Person_PhoneNumber>::AlignedUnion", (i64, array<24 x i8>)>)>)>
    %26 = llvm.insertvalue %24, %25[0] : !llvm.struct<"class.tutorial::Person_PhoneNumberDefaultTypeInternal", (struct<"class.google::protobuf::internal::ExplicitlyConstructed", (struct<"union.google::protobuf::internal::ExplicitlyConstructed<tutorial::Person_PhoneNumber>::AlignedUnion", (i64, array<24 x i8>)>)>)> 
    %27 = llvm.mlir.addressof @_ZN8tutorial37_Person_PhoneNumber_default_instance_E : !llvm.ptr
    %28 = llvm.load %16 atomic acquire {alignment = 8 : i64} : !llvm.ptr -> i32
    %29 = llvm.icmp "eq" %28, %4 : i32
    llvm.cond_br %29 weights([2000, 1]), ^bb2, ^bb1
  ^bb1:  // pred: ^bb0
    llvm.call @_ZN6google8protobuf8internal11InitSCCImplEPNS1_11SCCInfoBaseE(%16) : (!llvm.ptr) -> ()
    llvm.br ^bb2
  ^bb2:  // 2 preds: ^bb0, ^bb1
    llvm.return %27 : !llvm.ptr
  }
  llvm.func unnamed_addr @_ZN8tutorial18Person_PhoneNumber5ClearEv(%arg0: !llvm.ptr {llvm.align = 8 : i64, llvm.dereferenceable = 32 : i64, llvm.nocapture, llvm.nonnull, llvm.noundef}) attributes {alignment = 2 : i64, frame_pointer = #llvm.framePointerKind<none>, passthrough = [["uwtable", "2"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], ["tune-cpu", "generic"]], personality = @__gxx_personality_v0, target_cpu = "x86-64", target_features = #llvm.target_features<["+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>} {
    %0 = llvm.mlir.constant(0 : i64) : i64
    %1 = llvm.mlir.constant(1 : i32) : i32
    %2 = llvm.mlir.constant(0 : i32) : i32
    %3 = llvm.mlir.addressof @_ZN6google8protobuf8internal26fixed_address_empty_stringB5cxx11E : !llvm.ptr
    %4 = llvm.mlir.constant(0 : i8) : i8
    %5 = llvm.mlir.constant(2 : i32) : i32
    %6 = llvm.mlir.constant(1 : i64) : i64
    %7 = llvm.mlir.constant(-2 : i64) : i64
    %8 = llvm.getelementptr inbounds %arg0[%0, 1, 0] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.tutorial::Person_PhoneNumber", (struct<"class.google::protobuf::Message", (struct<"class.google::protobuf::MessageLite", (ptr, struct<"class.google::protobuf::internal::InternalMetadata", (ptr)>)>)>, struct<"struct.google::protobuf::internal::ArenaStringPtr", (ptr)>, i32, struct<"class.google::protobuf::internal::CachedSize", (struct<"struct.std::atomic", (struct<"struct.std::__atomic_base", (i32)>)>)>)>
    %9 = llvm.load %8 {alignment = 8 : i64, tbaa = [#tbaa_tag8]} : !llvm.ptr -> !llvm.ptr
    %10 = llvm.icmp "eq" %9, %3 : !llvm.ptr
    llvm.cond_br %10, ^bb2, ^bb1
  ^bb1:  // pred: ^bb0
    %11 = llvm.getelementptr inbounds %9[%0, 1] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.std::__cxx11::basic_string", (struct<"struct.std::__cxx11::basic_string<char>::_Alloc_hider", (ptr)>, i64, struct<"union.anon", (i64, array<8 x i8>)>)>
    llvm.store %0, %11 {alignment = 8 : i64, tbaa = [#tbaa_tag25]} : i64, !llvm.ptr
    %12 = llvm.getelementptr inbounds %9[%0, 0, 0] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.std::__cxx11::basic_string", (struct<"struct.std::__cxx11::basic_string<char>::_Alloc_hider", (ptr)>, i64, struct<"union.anon", (i64, array<8 x i8>)>)>
    %13 = llvm.load %12 {alignment = 8 : i64, tbaa = [#tbaa_tag26]} : !llvm.ptr -> !llvm.ptr
    llvm.store %4, %13 {alignment = 1 : i64, tbaa = [#tbaa_tag1]} : i8, !llvm.ptr
    llvm.br ^bb2
  ^bb2:  // 2 preds: ^bb0, ^bb1
    %14 = llvm.getelementptr inbounds %arg0[%0, 2] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.tutorial::Person_PhoneNumber", (struct<"class.google::protobuf::Message", (struct<"class.google::protobuf::MessageLite", (ptr, struct<"class.google::protobuf::internal::InternalMetadata", (ptr)>)>)>, struct<"struct.google::protobuf::internal::ArenaStringPtr", (ptr)>, i32, struct<"class.google::protobuf::internal::CachedSize", (struct<"struct.std::atomic", (struct<"struct.std::__atomic_base", (i32)>)>)>)>
    llvm.store %2, %14 {alignment = 8 : i64, tbaa = [#tbaa_tag24]} : i32, !llvm.ptr
    %15 = llvm.getelementptr inbounds %arg0[%0, 0, 0, 1, 0] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.tutorial::Person_PhoneNumber", (struct<"class.google::protobuf::Message", (struct<"class.google::protobuf::MessageLite", (ptr, struct<"class.google::protobuf::internal::InternalMetadata", (ptr)>)>)>, struct<"struct.google::protobuf::internal::ArenaStringPtr", (ptr)>, i32, struct<"class.google::protobuf::internal::CachedSize", (struct<"struct.std::atomic", (struct<"struct.std::__atomic_base", (i32)>)>)>)>
    %16 = llvm.load %15 {alignment = 8 : i64, tbaa = [#tbaa_tag5]} : !llvm.ptr -> !llvm.ptr
    %17 = llvm.ptrtoint %16 : !llvm.ptr to i64
    %18 = llvm.and %17, %6  : i64
    %19 = llvm.icmp "eq" %18, %0 : i64
    llvm.cond_br %19, ^bb5, ^bb3
  ^bb3:  // pred: ^bb2
    %20 = llvm.and %17, %7  : i64
    %21 = llvm.inttoptr %20 : i64 to !llvm.ptr
    %22 = llvm.getelementptr inbounds %21[%0, 1] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.google::protobuf::internal::InternalMetadata::Container", (struct<"struct.google::protobuf::internal::InternalMetadata::ContainerBase", (ptr)>, struct<"class.google::protobuf::UnknownFieldSet", (struct<"class.std::vector", (struct<"struct.std::_Vector_base", (struct<"struct.std::_Vector_base<google::protobuf::UnknownField, std::allocator<google::protobuf::UnknownField>>::_Vector_impl", (struct<"struct.std::_Vector_base<google::protobuf::UnknownField, std::allocator<google::protobuf::UnknownField>>::_Vector_impl_data", (ptr, ptr, ptr)>)>)>)>)>)>
    %23 = llvm.getelementptr inbounds %22[%0, 0, 0, 0, 0, 0] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.google::protobuf::UnknownFieldSet", (struct<"class.std::vector", (struct<"struct.std::_Vector_base", (struct<"struct.std::_Vector_base<google::protobuf::UnknownField, std::allocator<google::protobuf::UnknownField>>::_Vector_impl", (struct<"struct.std::_Vector_base<google::protobuf::UnknownField, std::allocator<google::protobuf::UnknownField>>::_Vector_impl_data", (ptr, ptr, ptr)>)>)>)>)>
    %24 = llvm.load %23 {alignment = 8 : i64, tbaa = [#tbaa_tag2]} : !llvm.ptr -> !llvm.ptr
    %25 = llvm.getelementptr inbounds %21[%0, 1, 0, 0, 0, 0, 1] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.google::protobuf::internal::InternalMetadata::Container", (struct<"struct.google::protobuf::internal::InternalMetadata::ContainerBase", (ptr)>, struct<"class.google::protobuf::UnknownFieldSet", (struct<"class.std::vector", (struct<"struct.std::_Vector_base", (struct<"struct.std::_Vector_base<google::protobuf::UnknownField, std::allocator<google::protobuf::UnknownField>>::_Vector_impl", (struct<"struct.std::_Vector_base<google::protobuf::UnknownField, std::allocator<google::protobuf::UnknownField>>::_Vector_impl_data", (ptr, ptr, ptr)>)>)>)>)>)>
    %26 = llvm.load %25 {alignment = 8 : i64, tbaa = [#tbaa_tag2]} : !llvm.ptr -> !llvm.ptr
    %27 = llvm.icmp "eq" %24, %26 : !llvm.ptr
    llvm.cond_br %27, ^bb5, ^bb4
  ^bb4:  // pred: ^bb3
    llvm.call @_ZN6google8protobuf15UnknownFieldSet13ClearFallbackEv(%22) : (!llvm.ptr) -> ()
    llvm.br ^bb5
  ^bb5:  // 3 preds: ^bb2, ^bb3, ^bb4
    llvm.return
  }
  llvm.func unnamed_addr @_ZN8tutorial18Person_PhoneNumber14_InternalParseEPKcPN6google8protobuf8internal12ParseContextE(%arg0: !llvm.ptr {llvm.align = 8 : i64, llvm.dereferenceable = 32 : i64, llvm.nonnull, llvm.noundef}, %arg1: !llvm.ptr {llvm.noundef}, %arg2: !llvm.ptr {llvm.noundef}) -> (!llvm.ptr {llvm.noundef}) attributes {alignment = 2 : i64, frame_pointer = #llvm.framePointerKind<none>, passthrough = [["uwtable", "2"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], ["tune-cpu", "generic"]], target_cpu = "x86-64", target_features = #llvm.target_features<["+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>} {
    %0 = llvm.mlir.constant(1 : i32) : i32
    %1 = llvm.mlir.constant(0 : i64) : i64
    %2 = llvm.mlir.constant(0 : i32) : i32
    %3 = llvm.mlir.constant(2 : i32) : i32
    %4 = llvm.mlir.constant(-1 : i8) : i8
    %5 = llvm.mlir.constant(1 : i64) : i64
    %6 = llvm.mlir.constant(7 : i32) : i32
    %7 = llvm.mlir.constant(-128 : i32) : i32
    %8 = llvm.mlir.zero : !llvm.ptr
    %9 = llvm.mlir.constant(2 : i64) : i64
    %10 = llvm.mlir.constant(3 : i32) : i32
    %11 = llvm.mlir.constant(255 : i32) : i32
    %12 = llvm.mlir.constant(16 : i32) : i32
    %13 = llvm.mlir.constant(128 : i32) : i32
    %14 = llvm.mlir.constant(10 : i32) : i32
    %15 = llvm.mlir.constant(-2 : i64) : i64
    %16 = llvm.mlir.addressof @_ZN6google8protobuf8internal26fixed_address_empty_stringB5cxx11E : !llvm.ptr
    %17 = llvm.mlir.constant("size_t to int conversion\00") : !llvm.array<25 x i8>
    %18 = llvm.mlir.addressof @".str.15" : !llvm.ptr
    %19 = llvm.mlir.constant("tutorial.Person.PhoneNumber.number\00") : !llvm.array<35 x i8>
    %20 = llvm.mlir.addressof @".str.4" : !llvm.ptr
    %21 = llvm.mlir.constant(4 : i32) : i32
    %22 = llvm.mlir.constant(-1 : i32) : i32
    %23 = llvm.mlir.constant(8 : i32) : i32
    %24 = llvm.alloca %0 x !llvm.ptr {alignment = 8 : i64} : (i32) -> !llvm.ptr
    llvm.store %arg1, %24 {alignment = 8 : i64, tbaa = [#tbaa_tag2]} : !llvm.ptr, !llvm.ptr
    %25 = llvm.getelementptr inbounds %arg2[%1, 0] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.google::protobuf::internal::ParseContext", (struct<"class.google::protobuf::internal::EpsCopyInputStream", (ptr, ptr, ptr, i32, i32, ptr, array<32 x i8>, i64, i32, i32)>, i32, i32, struct<"struct.google::protobuf::internal::ParseContext::Data", (ptr, ptr)>)>
    %26 = llvm.getelementptr inbounds %arg2[%1, 2] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.google::protobuf::internal::ParseContext", (struct<"class.google::protobuf::internal::EpsCopyInputStream", (ptr, ptr, ptr, i32, i32, ptr, array<32 x i8>, i64, i32, i32)>, i32, i32, struct<"struct.google::protobuf::internal::ParseContext::Data", (ptr, ptr)>)>
    %27 = llvm.getelementptr inbounds %arg0[%1, 2] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.tutorial::Person_PhoneNumber", (struct<"class.google::protobuf::Message", (struct<"class.google::protobuf::MessageLite", (ptr, struct<"class.google::protobuf::internal::InternalMetadata", (ptr)>)>)>, struct<"struct.google::protobuf::internal::ArenaStringPtr", (ptr)>, i32, struct<"class.google::protobuf::internal::CachedSize", (struct<"struct.std::atomic", (struct<"struct.std::__atomic_base", (i32)>)>)>)>
    %28 = llvm.getelementptr inbounds %arg0[%1, 1] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.tutorial::Person_PhoneNumber", (struct<"class.google::protobuf::Message", (struct<"class.google::protobuf::MessageLite", (ptr, struct<"class.google::protobuf::internal::InternalMetadata", (ptr)>)>)>, struct<"struct.google::protobuf::internal::ArenaStringPtr", (ptr)>, i32, struct<"class.google::protobuf::internal::CachedSize", (struct<"struct.std::atomic", (struct<"struct.std::__atomic_base", (i32)>)>)>)>
    %29 = llvm.getelementptr inbounds %arg0[%1, 0, 0, 1, 0] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.tutorial::Person_PhoneNumber", (struct<"class.google::protobuf::Message", (struct<"class.google::protobuf::MessageLite", (ptr, struct<"class.google::protobuf::internal::InternalMetadata", (ptr)>)>)>, struct<"struct.google::protobuf::internal::ArenaStringPtr", (ptr)>, i32, struct<"class.google::protobuf::internal::CachedSize", (struct<"struct.std::atomic", (struct<"struct.std::__atomic_base", (i32)>)>)>)>
    %30 = llvm.getelementptr inbounds %28[%1, 0] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.google::protobuf::internal::ArenaStringPtr", (ptr)>
    %31 = llvm.getelementptr inbounds %arg0[%1, 0, 0, 1] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.tutorial::Person_PhoneNumber", (struct<"class.google::protobuf::Message", (struct<"class.google::protobuf::MessageLite", (ptr, struct<"class.google::protobuf::internal::InternalMetadata", (ptr)>)>)>, struct<"struct.google::protobuf::internal::ArenaStringPtr", (ptr)>, i32, struct<"class.google::protobuf::internal::CachedSize", (struct<"struct.std::atomic", (struct<"struct.std::__atomic_base", (i32)>)>)>)>
    %32 = llvm.getelementptr inbounds %31[%1, 0] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.google::protobuf::internal::InternalMetadata", (ptr)>
    llvm.br ^bb1
  ^bb1:  // 2 preds: ^bb0, ^bb31
    %33 = llvm.load %26 {alignment = 4 : i64, tbaa = [#tbaa_tag27]} : !llvm.ptr -> i32
    %34 = llvm.call @_ZN6google8protobuf8internal18EpsCopyInputStream13DoneWithCheckEPPKci(%25, %24, %33) : (!llvm.ptr, !llvm.ptr, i32) -> i1
    %35 = llvm.load %24 {alignment = 8 : i64, tbaa = [#tbaa_tag2]} : !llvm.ptr -> !llvm.ptr
    llvm.cond_br %34, ^bb33(%35 : !llvm.ptr), ^bb2
  ^bb2:  // pred: ^bb1
    %36 = llvm.load %35 {alignment = 1 : i64, tbaa = [#tbaa_tag1]} : !llvm.ptr -> i8
    %37 = llvm.zext %36 : i8 to i32
    %38 = llvm.icmp "sgt" %36, %4 : i8
    %39 = llvm.getelementptr inbounds %35[%5] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    llvm.cond_br %38, ^bb5(%37, %39 : i32, !llvm.ptr), ^bb3
  ^bb3:  // pred: ^bb2
    %40 = llvm.load %39 {alignment = 1 : i64, tbaa = [#tbaa_tag1]} : !llvm.ptr -> i8
    %41 = llvm.zext %40 : i8 to i32
    %42 = llvm.shl %41, %6 overflow<nsw, nuw>  : i32
    %43 = llvm.add %37, %7 overflow<nsw>  : i32
    %44 = llvm.add %43, %42 overflow<nsw>  : i32
    %45 = llvm.icmp "sgt" %40, %4 : i8
    llvm.cond_br %45, ^bb4, ^bb6
  ^bb4:  // pred: ^bb3
    %46 = llvm.getelementptr inbounds %35[%9] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    llvm.br ^bb5(%44, %46 : i32, !llvm.ptr)
  ^bb5(%47: i32, %48: !llvm.ptr):  // 2 preds: ^bb2, ^bb4
    llvm.store %48, %24 {alignment = 8 : i64, tbaa = [#tbaa_tag2]} : !llvm.ptr, !llvm.ptr
    llvm.br ^bb7(%48, %47 : !llvm.ptr, i32)
  ^bb6:  // pred: ^bb3
    %49 = llvm.call @_ZN6google8protobuf8internal15ReadTagFallbackEPKcj(%35, %44) : (!llvm.ptr, i32) -> !llvm.struct<(ptr, i32)>
    %50 = llvm.extractvalue %49[0] : !llvm.struct<(ptr, i32)> 
    %51 = llvm.extractvalue %49[1] : !llvm.struct<(ptr, i32)> 
    llvm.store %50, %24 {alignment = 8 : i64, tbaa = [#tbaa_tag2]} : !llvm.ptr, !llvm.ptr
    %52 = llvm.icmp "eq" %50, %8 : !llvm.ptr
    llvm.cond_br %52 weights([1, 2000]), ^bb32, ^bb7(%50, %51 : !llvm.ptr, i32)
  ^bb7(%53: !llvm.ptr, %54: i32):  // 2 preds: ^bb5, ^bb6
    %55 = llvm.lshr %54, %10  : i32
    llvm.switch %55 : i32, ^bb25 [
      1: ^bb8,
      2: ^bb17
    ]
  ^bb8:  // pred: ^bb7
    %56 = llvm.and %54, %11  : i32
    %57 = llvm.icmp "eq" %56, %14 : i32
    llvm.cond_br %57 weights([2000, 1]), ^bb9, ^bb25
  ^bb9:  // pred: ^bb8
    %58 = llvm.load %29 {alignment = 8 : i64, tbaa = [#tbaa_tag5]} : !llvm.ptr -> !llvm.ptr
    %59 = llvm.ptrtoint %58 : !llvm.ptr to i64
    %60 = llvm.and %59, %5  : i64
    %61 = llvm.icmp "eq" %60, %1 : i64
    %62 = llvm.and %59, %15  : i64
    llvm.cond_br %61 weights([2000, 1]), ^bb11, ^bb10
  ^bb10:  // pred: ^bb9
    %63 = llvm.inttoptr %62 : i64 to !llvm.ptr
    %64 = llvm.getelementptr inbounds %63[%1, 0] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.google::protobuf::internal::InternalMetadata::ContainerBase", (ptr)>
    %65 = llvm.load %64 {alignment = 8 : i64, tbaa = [#tbaa_tag9]} : !llvm.ptr -> !llvm.ptr
    llvm.br ^bb12(%65 : !llvm.ptr)
  ^bb11:  // pred: ^bb9
    %66 = llvm.inttoptr %62 : i64 to !llvm.ptr
    llvm.br ^bb12(%66 : !llvm.ptr)
  ^bb12(%67: !llvm.ptr):  // 2 preds: ^bb10, ^bb11
    %68 = llvm.load %30 {alignment = 8 : i64, tbaa = [#tbaa_tag8]} : !llvm.ptr -> !llvm.ptr
    %69 = llvm.icmp "eq" %68, %16 : !llvm.ptr
    llvm.cond_br %69, ^bb13, ^bb14(%53, %68 : !llvm.ptr, !llvm.ptr)
  ^bb13:  // pred: ^bb12
    llvm.call @_ZN6google8protobuf8internal14ArenaStringPtr14CreateInstanceEPNS0_5ArenaEPKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE(%28, %67, %16) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
    %70 = llvm.load %30 {alignment = 8 : i64, tbaa = [#tbaa_tag8]} : !llvm.ptr -> !llvm.ptr
    %71 = llvm.load %24 {alignment = 8 : i64, tbaa = [#tbaa_tag2]} : !llvm.ptr -> !llvm.ptr
    llvm.br ^bb14(%71, %70 : !llvm.ptr, !llvm.ptr)
  ^bb14(%72: !llvm.ptr, %73: !llvm.ptr):  // 2 preds: ^bb12, ^bb13
    %74 = llvm.call @_ZN6google8protobuf8internal24InlineGreedyStringParserEPNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEPKcPNS1_12ParseContextE(%73, %72, %arg2) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> !llvm.ptr
    llvm.store %74, %24 {alignment = 8 : i64, tbaa = [#tbaa_tag2]} : !llvm.ptr, !llvm.ptr
    %75 = llvm.getelementptr inbounds %73[%1, 0, 0] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.std::__cxx11::basic_string", (struct<"struct.std::__cxx11::basic_string<char>::_Alloc_hider", (ptr)>, i64, struct<"union.anon", (i64, array<8 x i8>)>)>
    %76 = llvm.load %75 {alignment = 8 : i64, tbaa = [#tbaa_tag26]} : !llvm.ptr -> !llvm.ptr
    %77 = llvm.getelementptr inbounds %73[%1, 1] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.std::__cxx11::basic_string", (struct<"struct.std::__cxx11::basic_string<char>::_Alloc_hider", (ptr)>, i64, struct<"union.anon", (i64, array<8 x i8>)>)>
    %78 = llvm.load %77 {alignment = 8 : i64, tbaa = [#tbaa_tag25]} : !llvm.ptr -> i64
    %79 = llvm.icmp "slt" %78, %1 : i64
    llvm.cond_br %79, ^bb15, ^bb16
  ^bb15:  // pred: ^bb14
    llvm.call @_ZN6google8protobuf11StringPiece18LogFatalSizeTooBigEmPKc(%78, %18) : (i64, !llvm.ptr) -> ()
    llvm.br ^bb16
  ^bb16:  // 2 preds: ^bb14, ^bb15
    %80 = llvm.call @_ZN6google8protobuf8internal10VerifyUTF8ENS0_11StringPieceEPKc(%76, %78, %20) : (!llvm.ptr, i64, !llvm.ptr) -> i1
    %81 = llvm.load %24 {alignment = 8 : i64} : !llvm.ptr -> !llvm.ptr
    %82 = llvm.icmp "eq" %81, %8 : !llvm.ptr
    %83 = llvm.select %82, %21, %3 : i1, i32
    llvm.cond_br %80, ^bb31(%83 : i32), ^bb32
  ^bb17:  // pred: ^bb7
    %84 = llvm.and %54, %11  : i32
    %85 = llvm.icmp "eq" %84, %12 : i32
    llvm.cond_br %85 weights([2000, 1]), ^bb18, ^bb25
  ^bb18:  // pred: ^bb17
    %86 = llvm.load %53 {alignment = 1 : i64, tbaa = [#tbaa_tag1]} : !llvm.ptr -> i8
    %87 = llvm.zext %86 : i8 to i32
    %88 = llvm.and %87, %13  : i32
    %89 = llvm.icmp "eq" %88, %2 : i32
    llvm.cond_br %89, ^bb19, ^bb20
  ^bb19:  // pred: ^bb18
    %90 = llvm.zext %86 : i8 to i64
    llvm.br ^bb22(%5, %90 : i64, i64)
  ^bb20:  // pred: ^bb18
    %91 = llvm.getelementptr inbounds %53[%5] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %92 = llvm.load %91 {alignment = 1 : i64, tbaa = [#tbaa_tag1]} : !llvm.ptr -> i8
    %93 = llvm.zext %92 : i8 to i32
    %94 = llvm.shl %93, %6 overflow<nsw, nuw>  : i32
    %95 = llvm.add %87, %7 overflow<nsw>  : i32
    %96 = llvm.add %95, %94 overflow<nsw, nuw>  : i32
    %97 = llvm.and %93, %13  : i32
    %98 = llvm.icmp "eq" %97, %2 : i32
    llvm.cond_br %98, ^bb21, ^bb23
  ^bb21:  // pred: ^bb20
    %99 = llvm.zext %96 : i32 to i64
    llvm.br ^bb22(%9, %99 : i64, i64)
  ^bb22(%100: i64, %101: i64):  // 2 preds: ^bb19, ^bb21
    %102 = llvm.getelementptr inbounds %53[%100] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    llvm.store %102, %24 {alignment = 8 : i64, tbaa = [#tbaa_tag2]} : !llvm.ptr, !llvm.ptr
    llvm.br ^bb24(%101 : i64)
  ^bb23:  // pred: ^bb20
    %103 = llvm.call @_ZN6google8protobuf8internal17VarintParseSlow64EPKcj(%53, %96) : (!llvm.ptr, i32) -> !llvm.struct<(ptr, i64)>
    %104 = llvm.extractvalue %103[0] : !llvm.struct<(ptr, i64)> 
    %105 = llvm.extractvalue %103[1] : !llvm.struct<(ptr, i64)> 
    llvm.store %104, %24 {alignment = 8 : i64, tbaa = [#tbaa_tag2]} : !llvm.ptr, !llvm.ptr
    %106 = llvm.icmp "eq" %104, %8 : !llvm.ptr
    llvm.cond_br %106 weights([1, 2000]), ^bb32, ^bb24(%105 : i64)
  ^bb24(%107: i64):  // 2 preds: ^bb22, ^bb23
    %108 = llvm.trunc %107 : i64 to i32
    llvm.store %108, %27 {alignment = 8 : i64, tbaa = [#tbaa_tag24]} : i32, !llvm.ptr
    llvm.br ^bb31(%3 : i32)
  ^bb25:  // 3 preds: ^bb7, ^bb8, ^bb17
    %109 = llvm.and %54, %6  : i32
    %110 = llvm.icmp "eq" %109, %21 : i32
    %111 = llvm.icmp "eq" %54, %2 : i32
    %112 = llvm.or %111, %110  : i1
    llvm.cond_br %112, ^bb26, ^bb27
  ^bb26:  // pred: ^bb25
    %113 = llvm.add %54, %22  : i32
    %114 = llvm.getelementptr inbounds %arg2[%1, 0, 8] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.google::protobuf::internal::ParseContext", (struct<"class.google::protobuf::internal::EpsCopyInputStream", (ptr, ptr, ptr, i32, i32, ptr, array<32 x i8>, i64, i32, i32)>, i32, i32, struct<"struct.google::protobuf::internal::ParseContext::Data", (ptr, ptr)>)>
    llvm.store %113, %114 {alignment = 8 : i64, tbaa = [#tbaa_tag11]} : i32, !llvm.ptr
    llvm.br ^bb33(%53 : !llvm.ptr)
  ^bb27:  // pred: ^bb25
    %115 = llvm.zext %54 : i32 to i64
    %116 = llvm.load %32 {alignment = 8 : i64, tbaa = [#tbaa_tag5]} : !llvm.ptr -> !llvm.ptr
    %117 = llvm.ptrtoint %116 : !llvm.ptr to i64
    %118 = llvm.and %117, %5  : i64
    %119 = llvm.icmp "eq" %118, %1 : i64
    llvm.cond_br %119 weights([1, 2000]), ^bb29, ^bb28
  ^bb28:  // pred: ^bb27
    %120 = llvm.and %117, %15  : i64
    %121 = llvm.inttoptr %120 : i64 to !llvm.ptr
    %122 = llvm.getelementptr inbounds %121[%1, 1] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.google::protobuf::internal::InternalMetadata::Container", (struct<"struct.google::protobuf::internal::InternalMetadata::ContainerBase", (ptr)>, struct<"class.google::protobuf::UnknownFieldSet", (struct<"class.std::vector", (struct<"struct.std::_Vector_base", (struct<"struct.std::_Vector_base<google::protobuf::UnknownField, std::allocator<google::protobuf::UnknownField>>::_Vector_impl", (struct<"struct.std::_Vector_base<google::protobuf::UnknownField, std::allocator<google::protobuf::UnknownField>>::_Vector_impl_data", (ptr, ptr, ptr)>)>)>)>)>)>
    llvm.br ^bb30(%53, %122 : !llvm.ptr, !llvm.ptr)
  ^bb29:  // pred: ^bb27
    %123 = llvm.call @_ZN6google8protobuf8internal16InternalMetadata27mutable_unknown_fields_slowINS0_15UnknownFieldSetEEEPT_v(%31) : (!llvm.ptr) -> !llvm.ptr
    %124 = llvm.load %24 {alignment = 8 : i64, tbaa = [#tbaa_tag2]} : !llvm.ptr -> !llvm.ptr
    llvm.br ^bb30(%124, %123 : !llvm.ptr, !llvm.ptr)
  ^bb30(%125: !llvm.ptr, %126: !llvm.ptr):  // 2 preds: ^bb28, ^bb29
    %127 = llvm.call @_ZN6google8protobuf8internal17UnknownFieldParseEmPNS0_15UnknownFieldSetEPKcPNS1_12ParseContextE(%115, %126, %125, %arg2) : (i64, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> !llvm.ptr
    llvm.store %127, %24 {alignment = 8 : i64, tbaa = [#tbaa_tag2]} : !llvm.ptr, !llvm.ptr
    %128 = llvm.icmp "eq" %127, %8 : !llvm.ptr
    llvm.cond_br %128, ^bb32, ^bb31(%3 : i32)
  ^bb31(%129: i32):  // 3 preds: ^bb16, ^bb24, ^bb30
    llvm.switch %129 : i32, ^bb33(%35 : !llvm.ptr) [
      2: ^bb1,
      4: ^bb32
    ]
  ^bb32:  // 5 preds: ^bb6, ^bb16, ^bb23, ^bb30, ^bb31
    llvm.store %8, %24 {alignment = 8 : i64, tbaa = [#tbaa_tag2]} : !llvm.ptr, !llvm.ptr
    llvm.br ^bb33(%8 : !llvm.ptr)
  ^bb33(%130: !llvm.ptr):  // 4 preds: ^bb1, ^bb26, ^bb31, ^bb32
    llvm.return %130 : !llvm.ptr
  }
  llvm.func local_unnamed_addr @_ZN6google8protobuf8internal24InlineGreedyStringParserEPNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEPKcPNS1_12ParseContextE(!llvm.ptr {llvm.noundef}, !llvm.ptr {llvm.noundef}, !llvm.ptr {llvm.noundef}) -> (!llvm.ptr {llvm.noundef}) attributes {frame_pointer = #llvm.framePointerKind<none>, passthrough = [["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], ["tune-cpu", "generic"]], target_cpu = "x86-64", target_features = #llvm.target_features<["+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>}
  llvm.func local_unnamed_addr @_ZN6google8protobuf8internal17UnknownFieldParseEmPNS0_15UnknownFieldSetEPKcPNS1_12ParseContextE(i64 {llvm.noundef}, !llvm.ptr {llvm.noundef}, !llvm.ptr {llvm.noundef}, !llvm.ptr {llvm.noundef}) -> (!llvm.ptr {llvm.noundef}) attributes {frame_pointer = #llvm.framePointerKind<none>, passthrough = [["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], ["tune-cpu", "generic"]], target_cpu = "x86-64", target_features = #llvm.target_features<["+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>}
  llvm.func unnamed_addr @_ZNK8tutorial18Person_PhoneNumber18_InternalSerializeEPhPN6google8protobuf2io19EpsCopyOutputStreamE(%arg0: !llvm.ptr {llvm.align = 8 : i64, llvm.dereferenceable = 32 : i64, llvm.nocapture, llvm.nonnull, llvm.noundef, llvm.readonly}, %arg1: !llvm.ptr {llvm.noundef}, %arg2: !llvm.ptr {llvm.noundef}) -> (!llvm.ptr {llvm.noundef}) attributes {alignment = 2 : i64, frame_pointer = #llvm.framePointerKind<none>, passthrough = ["mustprogress", ["uwtable", "2"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], ["tune-cpu", "generic"]], target_cpu = "x86-64", target_features = #llvm.target_features<["+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>} {
    %0 = llvm.mlir.constant(0 : i64) : i64
    %1 = llvm.mlir.constant(1 : i32) : i32
    %2 = llvm.mlir.constant(0 : i32) : i32
    %3 = llvm.mlir.constant("tutorial.Person.PhoneNumber.number\00") : !llvm.array<35 x i8>
    %4 = llvm.mlir.addressof @".str.4" : !llvm.ptr
    %5 = llvm.mlir.constant(127 : i64) : i64
    %6 = llvm.mlir.constant(14 : i64) : i64
    %7 = llvm.mlir.constant(10 : i8) : i8
    %8 = llvm.mlir.constant(1 : i64) : i64
    %9 = llvm.mlir.constant(2 : i64) : i64
    %10 = llvm.mlir.constant(2 : i32) : i32
    %11 = llvm.mlir.constant(16 : i8) : i8
    %12 = llvm.mlir.constant(128 : i32) : i32
    %13 = llvm.mlir.constant(-128 : i8) : i8
    %14 = llvm.mlir.constant(7 : i64) : i64
    %15 = llvm.mlir.constant(16384 : i32) : i32
    %16 = llvm.mlir.constant(16383 : i64) : i64
    %17 = llvm.mlir.constant(3 : i64) : i64
    %18 = llvm.mlir.constant(-2 : i64) : i64
    %19 = llvm.getelementptr inbounds %arg0[%0, 1, 0] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.tutorial::Person_PhoneNumber", (struct<"class.google::protobuf::Message", (struct<"class.google::protobuf::MessageLite", (ptr, struct<"class.google::protobuf::internal::InternalMetadata", (ptr)>)>)>, struct<"struct.google::protobuf::internal::ArenaStringPtr", (ptr)>, i32, struct<"class.google::protobuf::internal::CachedSize", (struct<"struct.std::atomic", (struct<"struct.std::__atomic_base", (i32)>)>)>)>
    %20 = llvm.load %19 {alignment = 8 : i64, tbaa = [#tbaa_tag8]} : !llvm.ptr -> !llvm.ptr
    %21 = llvm.getelementptr inbounds %20[%0, 1] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.std::__cxx11::basic_string", (struct<"struct.std::__cxx11::basic_string<char>::_Alloc_hider", (ptr)>, i64, struct<"union.anon", (i64, array<8 x i8>)>)>
    %22 = llvm.load %21 {alignment = 8 : i64, tbaa = [#tbaa_tag25]} : !llvm.ptr -> i64
    %23 = llvm.icmp "eq" %22, %0 : i64
    llvm.cond_br %23, ^bb5(%arg1 : !llvm.ptr), ^bb1
  ^bb1:  // pred: ^bb0
    %24 = llvm.getelementptr inbounds %20[%0, 0, 0] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.std::__cxx11::basic_string", (struct<"struct.std::__cxx11::basic_string<char>::_Alloc_hider", (ptr)>, i64, struct<"union.anon", (i64, array<8 x i8>)>)>
    %25 = llvm.load %24 {alignment = 8 : i64, tbaa = [#tbaa_tag26]} : !llvm.ptr -> !llvm.ptr
    %26 = llvm.trunc %22 : i64 to i32
    %27 = llvm.call @_ZN6google8protobuf8internal14WireFormatLite16VerifyUtf8StringEPKciNS2_9OperationES4_(%25, %26, %1, %4) : (!llvm.ptr, i32, i32, !llvm.ptr) -> i1
    %28 = llvm.load %19 {alignment = 8 : i64, tbaa = [#tbaa_tag8]} : !llvm.ptr -> !llvm.ptr
    %29 = llvm.getelementptr inbounds %28[%0, 1] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.std::__cxx11::basic_string", (struct<"struct.std::__cxx11::basic_string<char>::_Alloc_hider", (ptr)>, i64, struct<"union.anon", (i64, array<8 x i8>)>)>
    %30 = llvm.load %29 {alignment = 8 : i64, tbaa = [#tbaa_tag25]} : !llvm.ptr -> i64
    %31 = llvm.icmp "sgt" %30, %5 : i64
    llvm.cond_br %31 weights([1, 2000]), ^bb3, ^bb2
  ^bb2:  // pred: ^bb1
    %32 = llvm.getelementptr inbounds %arg2[%0, 0] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.google::protobuf::io::EpsCopyOutputStream", packed (ptr, ptr, array<32 x i8>, ptr, i8, i8, i8, array<5 x i8>)>
    %33 = llvm.load %32 {alignment = 8 : i64, tbaa = [#tbaa_tag12]} : !llvm.ptr -> !llvm.ptr
    %34 = llvm.ptrtoint %33 : !llvm.ptr to i64
    %35 = llvm.ptrtoint %arg1 : !llvm.ptr to i64
    %36 = llvm.sub %6, %35  : i64
    %37 = llvm.add %36, %34  : i64
    %38 = llvm.icmp "slt" %37, %30 : i64
    llvm.cond_br %38 weights([1, 2000]), ^bb3, ^bb4
  ^bb3:  // 2 preds: ^bb1, ^bb2
    %39 = llvm.call @_ZN6google8protobuf2io19EpsCopyOutputStream30WriteStringMaybeAliasedOutlineEjRKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEPh(%arg2, %1, %28, %arg1) : (!llvm.ptr, i32, !llvm.ptr, !llvm.ptr) -> !llvm.ptr
    llvm.br ^bb5(%39 : !llvm.ptr)
  ^bb4:  // pred: ^bb2
    llvm.store %7, %arg1 {alignment = 1 : i64, tbaa = [#tbaa_tag1]} : i8, !llvm.ptr
    %40 = llvm.getelementptr inbounds %arg1[%8] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %41 = llvm.trunc %30 : i64 to i8
    %42 = llvm.getelementptr inbounds %arg1[%9] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    llvm.store %41, %40 {alignment = 1 : i64, tbaa = [#tbaa_tag1]} : i8, !llvm.ptr
    %43 = llvm.getelementptr inbounds %28[%0, 0, 0] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.std::__cxx11::basic_string", (struct<"struct.std::__cxx11::basic_string<char>::_Alloc_hider", (ptr)>, i64, struct<"union.anon", (i64, array<8 x i8>)>)>
    %44 = llvm.load %43 {alignment = 8 : i64, tbaa = [#tbaa_tag26]} : !llvm.ptr -> !llvm.ptr
    "llvm.intr.memcpy"(%42, %44, %30) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
    %45 = llvm.getelementptr inbounds %42[%30] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    llvm.br ^bb5(%45 : !llvm.ptr)
  ^bb5(%46: !llvm.ptr):  // 3 preds: ^bb0, ^bb3, ^bb4
    %47 = llvm.getelementptr inbounds %arg0[%0, 2] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.tutorial::Person_PhoneNumber", (struct<"class.google::protobuf::Message", (struct<"class.google::protobuf::MessageLite", (ptr, struct<"class.google::protobuf::internal::InternalMetadata", (ptr)>)>)>, struct<"struct.google::protobuf::internal::ArenaStringPtr", (ptr)>, i32, struct<"class.google::protobuf::internal::CachedSize", (struct<"struct.std::atomic", (struct<"struct.std::__atomic_base", (i32)>)>)>)>
    %48 = llvm.load %47 {alignment = 8 : i64, tbaa = [#tbaa_tag24]} : !llvm.ptr -> i32
    %49 = llvm.icmp "eq" %48, %2 : i32
    llvm.cond_br %49, ^bb15(%46 : !llvm.ptr), ^bb6
  ^bb6:  // pred: ^bb5
    %50 = llvm.getelementptr inbounds %arg2[%0, 0] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.google::protobuf::io::EpsCopyOutputStream", packed (ptr, ptr, array<32 x i8>, ptr, i8, i8, i8, array<5 x i8>)>
    %51 = llvm.load %50 {alignment = 8 : i64, tbaa = [#tbaa_tag12]} : !llvm.ptr -> !llvm.ptr
    %52 = llvm.icmp "ugt" %51, %46 : !llvm.ptr
    llvm.cond_br %52 weights([2000, 1]), ^bb8(%48, %46 : i32, !llvm.ptr), ^bb7
  ^bb7:  // pred: ^bb6
    %53 = llvm.call @_ZN6google8protobuf2io19EpsCopyOutputStream19EnsureSpaceFallbackEPh(%arg2, %46) : (!llvm.ptr, !llvm.ptr) -> !llvm.ptr
    %54 = llvm.load %47 {alignment = 8 : i64, tbaa = [#tbaa_tag24]} : !llvm.ptr -> i32
    llvm.br ^bb8(%54, %53 : i32, !llvm.ptr)
  ^bb8(%55: i32, %56: !llvm.ptr):  // 2 preds: ^bb6, ^bb7
    llvm.store %11, %56 {alignment = 1 : i64, tbaa = [#tbaa_tag1]} : i8, !llvm.ptr
    %57 = llvm.getelementptr inbounds %56[%8] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %58 = llvm.icmp "ult" %55, %12 : i32
    %59 = llvm.trunc %55 : i32 to i8
    llvm.cond_br %58, ^bb9, ^bb10
  ^bb9:  // pred: ^bb8
    llvm.store %59, %57 {alignment = 1 : i64, tbaa = [#tbaa_tag1]} : i8, !llvm.ptr
    %60 = llvm.getelementptr inbounds %56[%9] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    llvm.br ^bb15(%60 : !llvm.ptr)
  ^bb10:  // pred: ^bb8
    %61 = llvm.sext %55 : i32 to i64
    %62 = llvm.or %59, %13  : i8
    llvm.store %62, %57 {alignment = 1 : i64, tbaa = [#tbaa_tag1]} : i8, !llvm.ptr
    %63 = llvm.lshr %61, %14  : i64
    %64 = llvm.icmp "ult" %55, %15 : i32
    llvm.cond_br %64, ^bb11, ^bb12
  ^bb11:  // pred: ^bb10
    %65 = llvm.trunc %63 : i64 to i8
    %66 = llvm.getelementptr inbounds %56[%9] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    llvm.store %65, %66 {alignment = 1 : i64, tbaa = [#tbaa_tag1]} : i8, !llvm.ptr
    %67 = llvm.getelementptr inbounds %56[%17] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    llvm.br ^bb15(%67 : !llvm.ptr)
  ^bb12:  // pred: ^bb10
    %68 = llvm.getelementptr inbounds %56[%9] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    llvm.br ^bb13(%63, %68 : i64, !llvm.ptr)
  ^bb13(%69: i64, %70: !llvm.ptr):  // 2 preds: ^bb12, ^bb13
    %71 = llvm.trunc %69 : i64 to i8
    %72 = llvm.or %71, %13  : i8
    llvm.store %72, %70 {alignment = 1 : i64, tbaa = [#tbaa_tag1]} : i8, !llvm.ptr
    %73 = llvm.lshr %69, %14  : i64
    %74 = llvm.getelementptr inbounds %70[%8] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %75 = llvm.icmp "ugt" %69, %16 : i64
    llvm.cond_br %75 weights([1, 2000]), ^bb13(%73, %74 : i64, !llvm.ptr), ^bb14 {loop_annotation = #loop_annotation}
  ^bb14:  // pred: ^bb13
    %76 = llvm.trunc %73 : i64 to i8
    %77 = llvm.getelementptr inbounds %70[%9] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    llvm.store %76, %74 {alignment = 1 : i64, tbaa = [#tbaa_tag1]} : i8, !llvm.ptr
    llvm.br ^bb15(%77 : !llvm.ptr)
  ^bb15(%78: !llvm.ptr):  // 4 preds: ^bb5, ^bb9, ^bb11, ^bb14
    %79 = llvm.getelementptr inbounds %arg0[%0, 0, 0, 1, 0] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.tutorial::Person_PhoneNumber", (struct<"class.google::protobuf::Message", (struct<"class.google::protobuf::MessageLite", (ptr, struct<"class.google::protobuf::internal::InternalMetadata", (ptr)>)>)>, struct<"struct.google::protobuf::internal::ArenaStringPtr", (ptr)>, i32, struct<"class.google::protobuf::internal::CachedSize", (struct<"struct.std::atomic", (struct<"struct.std::__atomic_base", (i32)>)>)>)>
    %80 = llvm.load %79 {alignment = 8 : i64, tbaa = [#tbaa_tag5]} : !llvm.ptr -> !llvm.ptr
    %81 = llvm.ptrtoint %80 : !llvm.ptr to i64
    %82 = llvm.and %81, %8  : i64
    %83 = llvm.icmp "eq" %82, %0 : i64
    llvm.cond_br %83 weights([2000, 1]), ^bb17(%78 : !llvm.ptr), ^bb16
  ^bb16:  // pred: ^bb15
    %84 = llvm.and %81, %18  : i64
    %85 = llvm.inttoptr %84 : i64 to !llvm.ptr
    %86 = llvm.getelementptr inbounds %85[%0, 1] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.google::protobuf::internal::InternalMetadata::Container", (struct<"struct.google::protobuf::internal::InternalMetadata::ContainerBase", (ptr)>, struct<"class.google::protobuf::UnknownFieldSet", (struct<"class.std::vector", (struct<"struct.std::_Vector_base", (struct<"struct.std::_Vector_base<google::protobuf::UnknownField, std::allocator<google::protobuf::UnknownField>>::_Vector_impl", (struct<"struct.std::_Vector_base<google::protobuf::UnknownField, std::allocator<google::protobuf::UnknownField>>::_Vector_impl_data", (ptr, ptr, ptr)>)>)>)>)>)>
    %87 = llvm.call @_ZN6google8protobuf8internal10WireFormat37InternalSerializeUnknownFieldsToArrayERKNS0_15UnknownFieldSetEPhPNS0_2io19EpsCopyOutputStreamE(%86, %78, %arg2) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> !llvm.ptr
    llvm.br ^bb17(%87 : !llvm.ptr)
  ^bb17(%88: !llvm.ptr):  // 2 preds: ^bb15, ^bb16
    llvm.return %88 : !llvm.ptr
  }
  llvm.func local_unnamed_addr @_ZN6google8protobuf8internal14WireFormatLite16VerifyUtf8StringEPKciNS2_9OperationES4_(!llvm.ptr {llvm.noundef}, i32 {llvm.noundef}, i32 {llvm.noundef}, !llvm.ptr {llvm.noundef}) -> (i1 {llvm.noundef, llvm.zeroext}) attributes {frame_pointer = #llvm.framePointerKind<none>, passthrough = [["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], ["tune-cpu", "generic"]], target_cpu = "x86-64", target_features = #llvm.target_features<["+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>}
  llvm.func local_unnamed_addr @_ZN6google8protobuf8internal10WireFormat37InternalSerializeUnknownFieldsToArrayERKNS0_15UnknownFieldSetEPhPNS0_2io19EpsCopyOutputStreamE(!llvm.ptr {llvm.align = 8 : i64, llvm.dereferenceable = 24 : i64, llvm.nonnull, llvm.noundef}, !llvm.ptr {llvm.noundef}, !llvm.ptr {llvm.noundef}) -> (!llvm.ptr {llvm.noundef}) attributes {frame_pointer = #llvm.framePointerKind<none>, passthrough = [["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], ["tune-cpu", "generic"]], target_cpu = "x86-64", target_features = #llvm.target_features<["+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>}
  llvm.func unnamed_addr @_ZNK8tutorial18Person_PhoneNumber12ByteSizeLongEv(%arg0: !llvm.ptr {llvm.align = 8 : i64, llvm.dereferenceable = 32 : i64, llvm.nonnull, llvm.noundef}) -> (i64 {llvm.noundef}) attributes {alignment = 2 : i64, frame_pointer = #llvm.framePointerKind<none>, passthrough = ["mustprogress", ["uwtable", "2"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], ["tune-cpu", "generic"]], personality = @__gxx_personality_v0, target_cpu = "x86-64", target_features = #llvm.target_features<["+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>} {
    %0 = llvm.mlir.constant(0 : i64) : i64
    %1 = llvm.mlir.constant(1 : i32) : i32
    %2 = llvm.mlir.constant(0 : i32) : i32
    %3 = llvm.mlir.constant(31 : i32) : i32
    %4 = llvm.mlir.constant(9 : i32) : i32
    %5 = llvm.mlir.constant(73 : i32) : i32
    %6 = llvm.mlir.constant(6 : i32) : i32
    %7 = llvm.mlir.constant(1 : i64) : i64
    %8 = llvm.mlir.constant(2 : i32) : i32
    %9 = llvm.mlir.constant(11 : i64) : i64
    %10 = llvm.mlir.constant(3 : i32) : i32
    %11 = llvm.getelementptr inbounds %arg0[%0, 1, 0] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.tutorial::Person_PhoneNumber", (struct<"class.google::protobuf::Message", (struct<"class.google::protobuf::MessageLite", (ptr, struct<"class.google::protobuf::internal::InternalMetadata", (ptr)>)>)>, struct<"struct.google::protobuf::internal::ArenaStringPtr", (ptr)>, i32, struct<"class.google::protobuf::internal::CachedSize", (struct<"struct.std::atomic", (struct<"struct.std::__atomic_base", (i32)>)>)>)>
    %12 = llvm.load %11 {alignment = 8 : i64, tbaa = [#tbaa_tag8]} : !llvm.ptr -> !llvm.ptr
    %13 = llvm.getelementptr inbounds %12[%0, 1] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.std::__cxx11::basic_string", (struct<"struct.std::__cxx11::basic_string<char>::_Alloc_hider", (ptr)>, i64, struct<"union.anon", (i64, array<8 x i8>)>)>
    %14 = llvm.load %13 {alignment = 8 : i64, tbaa = [#tbaa_tag25]} : !llvm.ptr -> i64
    %15 = llvm.icmp "eq" %14, %0 : i64
    llvm.cond_br %15, ^bb2(%0 : i64), ^bb1
  ^bb1:  // pred: ^bb0
    %16 = llvm.trunc %14 : i64 to i32
    %17 = llvm.or %16, %1  : i32
    %18 = "llvm.intr.ctlz"(%17) <{is_zero_poison = true}> : (i32) -> i32
    %19 = llvm.xor %18, %3  : i32
    %20 = llvm.mul %19, %4 overflow<nsw, nuw>  : i32
    %21 = llvm.add %20, %5 overflow<nsw, nuw>  : i32
    %22 = llvm.lshr %21, %6  : i32
    %23 = llvm.zext %22 : i32 to i64
    %24 = llvm.add %14, %7  : i64
    %25 = llvm.add %24, %23  : i64
    llvm.br ^bb2(%25 : i64)
  ^bb2(%26: i64):  // 2 preds: ^bb0, ^bb1
    %27 = llvm.getelementptr inbounds %arg0[%0, 2] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.tutorial::Person_PhoneNumber", (struct<"class.google::protobuf::Message", (struct<"class.google::protobuf::MessageLite", (ptr, struct<"class.google::protobuf::internal::InternalMetadata", (ptr)>)>)>, struct<"struct.google::protobuf::internal::ArenaStringPtr", (ptr)>, i32, struct<"class.google::protobuf::internal::CachedSize", (struct<"struct.std::atomic", (struct<"struct.std::__atomic_base", (i32)>)>)>)>
    %28 = llvm.load %27 {alignment = 8 : i64, tbaa = [#tbaa_tag24]} : !llvm.ptr -> i32
    %29 = llvm.icmp "eq" %28, %2 : i32
    llvm.cond_br %29, ^bb6(%26 : i64), ^bb3
  ^bb3:  // pred: ^bb2
    %30 = llvm.icmp "slt" %28, %2 : i32
    llvm.cond_br %30, ^bb5(%9 : i64), ^bb4
  ^bb4:  // pred: ^bb3
    %31 = llvm.or %28, %1  : i32
    %32 = "llvm.intr.ctlz"(%31) <{is_zero_poison = true}> : (i32) -> i32
    %33 = llvm.xor %32, %3  : i32
    %34 = llvm.mul %33, %4 overflow<nsw, nuw>  : i32
    %35 = llvm.add %34, %5 overflow<nsw, nuw>  : i32
    %36 = llvm.lshr %35, %6  : i32
    %37 = llvm.add %36, %1 overflow<nsw, nuw>  : i32
    %38 = llvm.zext %37 : i32 to i64
    llvm.br ^bb5(%38 : i64)
  ^bb5(%39: i64):  // 2 preds: ^bb3, ^bb4
    %40 = llvm.add %39, %26  : i64
    llvm.br ^bb6(%40 : i64)
  ^bb6(%41: i64):  // 2 preds: ^bb2, ^bb5
    %42 = llvm.getelementptr inbounds %arg0[%0, 0, 0, 1] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.tutorial::Person_PhoneNumber", (struct<"class.google::protobuf::Message", (struct<"class.google::protobuf::MessageLite", (ptr, struct<"class.google::protobuf::internal::InternalMetadata", (ptr)>)>)>, struct<"struct.google::protobuf::internal::ArenaStringPtr", (ptr)>, i32, struct<"class.google::protobuf::internal::CachedSize", (struct<"struct.std::atomic", (struct<"struct.std::__atomic_base", (i32)>)>)>)>
    %43 = llvm.getelementptr inbounds %42[%0, 0] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.google::protobuf::internal::InternalMetadata", (ptr)>
    %44 = llvm.load %43 {alignment = 8 : i64, tbaa = [#tbaa_tag5]} : !llvm.ptr -> !llvm.ptr
    %45 = llvm.ptrtoint %44 : !llvm.ptr to i64
    %46 = llvm.and %45, %7  : i64
    %47 = llvm.icmp "eq" %46, %0 : i64
    llvm.cond_br %47 weights([2000, 1]), ^bb8, ^bb7
  ^bb7:  // pred: ^bb6
    %48 = llvm.getelementptr inbounds %arg0[%0, 3] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.tutorial::Person_PhoneNumber", (struct<"class.google::protobuf::Message", (struct<"class.google::protobuf::MessageLite", (ptr, struct<"class.google::protobuf::internal::InternalMetadata", (ptr)>)>)>, struct<"struct.google::protobuf::internal::ArenaStringPtr", (ptr)>, i32, struct<"class.google::protobuf::internal::CachedSize", (struct<"struct.std::atomic", (struct<"struct.std::__atomic_base", (i32)>)>)>)>
    %49 = llvm.call @_ZN6google8protobuf8internal24ComputeUnknownFieldsSizeERKNS1_16InternalMetadataEmPNS1_10CachedSizeE(%42, %41, %48) : (!llvm.ptr, i64, !llvm.ptr) -> i64
    llvm.br ^bb9(%49 : i64)
  ^bb8:  // pred: ^bb6
    %50 = llvm.trunc %41 : i64 to i32
    %51 = llvm.getelementptr inbounds %arg0[%0, 3, 0, 0, 0] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.tutorial::Person_PhoneNumber", (struct<"class.google::protobuf::Message", (struct<"class.google::protobuf::MessageLite", (ptr, struct<"class.google::protobuf::internal::InternalMetadata", (ptr)>)>)>, struct<"struct.google::protobuf::internal::ArenaStringPtr", (ptr)>, i32, struct<"class.google::protobuf::internal::CachedSize", (struct<"struct.std::atomic", (struct<"struct.std::__atomic_base", (i32)>)>)>)>
    llvm.store %50, %51 atomic monotonic {alignment = 4 : i64} : i32, !llvm.ptr
    llvm.br ^bb9(%41 : i64)
  ^bb9(%52: i64):  // 2 preds: ^bb7, ^bb8
    llvm.return %52 : i64
  }
  llvm.func local_unnamed_addr @_ZN6google8protobuf8internal24ComputeUnknownFieldsSizeERKNS1_16InternalMetadataEmPNS1_10CachedSizeE(!llvm.ptr {llvm.align = 8 : i64, llvm.dereferenceable = 8 : i64, llvm.nonnull, llvm.noundef}, i64 {llvm.noundef}, !llvm.ptr {llvm.noundef}) -> (i64 {llvm.noundef}) attributes {frame_pointer = #llvm.framePointerKind<none>, passthrough = [["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], ["tune-cpu", "generic"]], target_cpu = "x86-64", target_features = #llvm.target_features<["+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>}
  llvm.func unnamed_addr @_ZN8tutorial18Person_PhoneNumber9MergeFromERKN6google8protobuf7MessageE(%arg0: !llvm.ptr {llvm.align = 8 : i64, llvm.dereferenceable = 32 : i64, llvm.nonnull, llvm.noundef}, %arg1: !llvm.ptr {llvm.align = 8 : i64, llvm.dereferenceable = 16 : i64, llvm.nonnull, llvm.noundef}) attributes {alignment = 2 : i64, frame_pointer = #llvm.framePointerKind<none>, passthrough = ["mustprogress", ["uwtable", "2"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], ["tune-cpu", "generic"]], personality = @__gxx_personality_v0, target_cpu = "x86-64", target_features = #llvm.target_features<["+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>} {
    %0 = llvm.mlir.constant(1 : i32) : i32
    %1 = llvm.mlir.constant(0 : i64) : i64
    %2 = llvm.mlir.constant(0 : i32) : i32
    %3 = llvm.mlir.constant(3 : i32) : i32
    %4 = llvm.mlir.constant("build/addressbook.pb.cc\00") : !llvm.array<24 x i8>
    %5 = llvm.mlir.addressof @".str.5" : !llvm.ptr
    %6 = llvm.mlir.constant(358 : i32) : i32
    %7 = llvm.mlir.constant("CHECK failed: (&from) != (this): \00") : !llvm.array<34 x i8>
    %8 = llvm.mlir.addressof @".str.6" : !llvm.ptr
    %9 = llvm.mlir.addressof @_ZTIN6google8protobuf7MessageE : !llvm.ptr
    %10 = llvm.mlir.constant("N8tutorial18Person_PhoneNumberE\00") : !llvm.array<32 x i8>
    %11 = llvm.mlir.addressof @_ZTSN8tutorial18Person_PhoneNumberE : !llvm.ptr
    %12 = llvm.mlir.constant(2 : i64) : i64
    %13 = llvm.mlir.addressof @_ZTVN10__cxxabiv120__si_class_type_infoE : !llvm.ptr
    %14 = llvm.getelementptr inbounds %13[%12] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.ptr
    %15 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, ptr)>
    %16 = llvm.insertvalue %14, %15[0] : !llvm.struct<(ptr, ptr, ptr)> 
    %17 = llvm.insertvalue %11, %16[1] : !llvm.struct<(ptr, ptr, ptr)> 
    %18 = llvm.insertvalue %9, %17[2] : !llvm.struct<(ptr, ptr, ptr)> 
    %19 = llvm.mlir.addressof @_ZTIN8tutorial18Person_PhoneNumberE : !llvm.ptr
    %20 = llvm.mlir.zero : !llvm.ptr
    %21 = llvm.alloca %0 x !llvm.struct<"class.google::protobuf::internal::LogMessage", (i32, ptr, i32, struct<"class.std::__cxx11::basic_string", (struct<"struct.std::__cxx11::basic_string<char>::_Alloc_hider", (ptr)>, i64, struct<"union.anon", (i64, array<8 x i8>)>)>)> {alignment = 8 : i64} : (i32) -> !llvm.ptr
    %22 = llvm.alloca %0 x !llvm.struct<"class.google::protobuf::internal::LogFinisher", (i8)> {alignment = 1 : i64} : (i32) -> !llvm.ptr
    %23 = llvm.getelementptr inbounds %arg0[%1, 0] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.tutorial::Person_PhoneNumber", (struct<"class.google::protobuf::Message", (struct<"class.google::protobuf::MessageLite", (ptr, struct<"class.google::protobuf::internal::InternalMetadata", (ptr)>)>)>, struct<"struct.google::protobuf::internal::ArenaStringPtr", (ptr)>, i32, struct<"class.google::protobuf::internal::CachedSize", (struct<"struct.std::atomic", (struct<"struct.std::__atomic_base", (i32)>)>)>)>
    %24 = llvm.icmp "eq" %23, %arg1 : !llvm.ptr
    %25 = llvm.getelementptr inbounds %22[%1, 0] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.google::protobuf::internal::LogFinisher", (i8)>
    llvm.intr.lifetime.start 1, %25 : !llvm.ptr
    llvm.cond_br %24, ^bb1, ^bb3
  ^bb1:  // pred: ^bb0
    %26 = llvm.bitcast %21 : !llvm.ptr to !llvm.ptr
    llvm.intr.lifetime.start 56, %26 : !llvm.ptr
    llvm.call @_ZN6google8protobuf8internal10LogMessageC1ENS0_8LogLevelEPKci(%21, %3, %5, %6) : (!llvm.ptr, i32, !llvm.ptr, i32) -> ()
    %27 = llvm.invoke @_ZN6google8protobuf8internal10LogMessagelsEPKc(%21, %8) to ^bb2 unwind ^bb7 : (!llvm.ptr, !llvm.ptr) -> !llvm.ptr
  ^bb2:  // pred: ^bb1
    llvm.invoke @_ZN6google8protobuf8internal11LogFinisheraSERNS1_10LogMessageE(%22, %27) to ^bb4 unwind ^bb8 : (!llvm.ptr, !llvm.ptr) -> ()
  ^bb3:  // pred: ^bb0
    llvm.intr.lifetime.end 1, %25 : !llvm.ptr
    llvm.br ^bb5
  ^bb4:  // pred: ^bb2
    llvm.intr.lifetime.end 1, %25 : !llvm.ptr
    llvm.call @_ZN6google8protobuf8internal10LogMessageD1Ev(%21) : (!llvm.ptr) -> ()
    llvm.intr.lifetime.end 56, %26 : !llvm.ptr
    llvm.br ^bb5
  ^bb5:  // 2 preds: ^bb3, ^bb4
    %28 = llvm.bitcast %arg1 : !llvm.ptr to !llvm.ptr
    %29 = llvm.call @__dynamic_cast(%28, %9, %19, %1) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i64) -> !llvm.ptr
    %30 = llvm.icmp "eq" %29, %20 : !llvm.ptr
    llvm.cond_br %30, ^bb6, ^bb10
  ^bb6:  // pred: ^bb5
    llvm.call @_ZN6google8protobuf8internal13ReflectionOps5MergeERKNS0_7MessageEPS3_(%arg1, %23) : (!llvm.ptr, !llvm.ptr) -> ()
    llvm.br ^bb11
  ^bb7:  // pred: ^bb1
    %31 = llvm.landingpad cleanup : !llvm.struct<(ptr, i32)>
    llvm.br ^bb9(%31 : !llvm.struct<(ptr, i32)>)
  ^bb8:  // pred: ^bb2
    %32 = llvm.landingpad cleanup : !llvm.struct<(ptr, i32)>
    llvm.intr.lifetime.end 1, %25 : !llvm.ptr
    llvm.br ^bb9(%32 : !llvm.struct<(ptr, i32)>)
  ^bb9(%33: !llvm.struct<(ptr, i32)>):  // 2 preds: ^bb7, ^bb8
    llvm.call @_ZN6google8protobuf8internal10LogMessageD1Ev(%21) : (!llvm.ptr) -> ()
    llvm.intr.lifetime.end 56, %26 : !llvm.ptr
    llvm.resume %33 : !llvm.struct<(ptr, i32)>
  ^bb10:  // pred: ^bb5
    %34 = llvm.bitcast %29 : !llvm.ptr to !llvm.ptr
    llvm.call @_ZN8tutorial18Person_PhoneNumber9MergeFromERKS0_(%arg0, %34) : (!llvm.ptr, !llvm.ptr) -> ()
    llvm.br ^bb11
  ^bb11:  // 2 preds: ^bb6, ^bb10
    llvm.return
  }
  llvm.func unnamed_addr @_ZN6google8protobuf8internal10LogMessageC1ENS0_8LogLevelEPKci(!llvm.ptr {llvm.align = 8 : i64, llvm.dereferenceable = 56 : i64, llvm.nonnull, llvm.noundef}, i32 {llvm.noundef}, !llvm.ptr {llvm.noundef}, i32 {llvm.noundef}) attributes {frame_pointer = #llvm.framePointerKind<none>, passthrough = [["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], ["tune-cpu", "generic"]], target_cpu = "x86-64", target_features = #llvm.target_features<["+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>}
  llvm.func local_unnamed_addr @_ZN6google8protobuf8internal10LogMessagelsEPKc(!llvm.ptr {llvm.align = 8 : i64, llvm.dereferenceable = 56 : i64, llvm.nonnull, llvm.noundef}, !llvm.ptr {llvm.noundef}) -> (!llvm.ptr {llvm.align = 8 : i64, llvm.dereferenceable = 56 : i64, llvm.nonnull, llvm.noundef}) attributes {frame_pointer = #llvm.framePointerKind<none>, passthrough = [["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], ["tune-cpu", "generic"]], target_cpu = "x86-64", target_features = #llvm.target_features<["+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>}
  llvm.func local_unnamed_addr @_ZN6google8protobuf8internal11LogFinisheraSERNS1_10LogMessageE(!llvm.ptr {llvm.align = 1 : i64, llvm.dereferenceable = 1 : i64, llvm.nonnull, llvm.noundef}, !llvm.ptr {llvm.align = 8 : i64, llvm.dereferenceable = 56 : i64, llvm.nonnull, llvm.noundef}) attributes {frame_pointer = #llvm.framePointerKind<none>, passthrough = [["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], ["tune-cpu", "generic"]], target_cpu = "x86-64", target_features = #llvm.target_features<["+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>}
  llvm.func unnamed_addr @_ZN6google8protobuf8internal10LogMessageD1Ev(!llvm.ptr {llvm.align = 8 : i64, llvm.dereferenceable = 56 : i64, llvm.nonnull, llvm.noundef}) attributes {frame_pointer = #llvm.framePointerKind<none>, passthrough = ["nounwind", ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], ["tune-cpu", "generic"]], target_cpu = "x86-64", target_features = #llvm.target_features<["+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>}
  llvm.func local_unnamed_addr @_ZN6google8protobuf8internal13ReflectionOps5MergeERKNS0_7MessageEPS3_(!llvm.ptr {llvm.align = 8 : i64, llvm.dereferenceable = 16 : i64, llvm.nonnull, llvm.noundef}, !llvm.ptr {llvm.noundef}) attributes {frame_pointer = #llvm.framePointerKind<none>, passthrough = [["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], ["tune-cpu", "generic"]], target_cpu = "x86-64", target_features = #llvm.target_features<["+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>}
  llvm.func local_unnamed_addr @_ZN8tutorial18Person_PhoneNumber9MergeFromERKS0_(%arg0: !llvm.ptr {llvm.align = 8 : i64, llvm.dereferenceable = 32 : i64, llvm.nonnull, llvm.noundef}, %arg1: !llvm.ptr {llvm.align = 8 : i64, llvm.dereferenceable = 32 : i64, llvm.nonnull, llvm.noundef, llvm.readonly}) attributes {alignment = 2 : i64, frame_pointer = #llvm.framePointerKind<none>, passthrough = ["mustprogress", ["uwtable", "2"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], ["tune-cpu", "generic"]], personality = @__gxx_personality_v0, target_cpu = "x86-64", target_features = #llvm.target_features<["+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>} {
    %0 = llvm.mlir.constant(1 : i32) : i32
    %1 = llvm.mlir.constant(0 : i64) : i64
    %2 = llvm.mlir.constant(0 : i32) : i32
    %3 = llvm.mlir.constant(3 : i32) : i32
    %4 = llvm.mlir.constant("build/addressbook.pb.cc\00") : !llvm.array<24 x i8>
    %5 = llvm.mlir.addressof @".str.5" : !llvm.ptr
    %6 = llvm.mlir.constant(373 : i32) : i32
    %7 = llvm.mlir.constant("CHECK failed: (&from) != (this): \00") : !llvm.array<34 x i8>
    %8 = llvm.mlir.addressof @".str.6" : !llvm.ptr
    %9 = llvm.mlir.constant(1 : i64) : i64
    %10 = llvm.mlir.constant(-2 : i64) : i64
    %11 = llvm.mlir.addressof @_ZN6google8protobuf8internal26fixed_address_empty_stringB5cxx11E : !llvm.ptr
    %12 = llvm.mlir.constant(2 : i32) : i32
    %13 = llvm.alloca %0 x !llvm.struct<"class.google::protobuf::internal::LogMessage", (i32, ptr, i32, struct<"class.std::__cxx11::basic_string", (struct<"struct.std::__cxx11::basic_string<char>::_Alloc_hider", (ptr)>, i64, struct<"union.anon", (i64, array<8 x i8>)>)>)> {alignment = 8 : i64} : (i32) -> !llvm.ptr
    %14 = llvm.alloca %0 x !llvm.struct<"class.google::protobuf::internal::LogFinisher", (i8)> {alignment = 1 : i64} : (i32) -> !llvm.ptr
    %15 = llvm.icmp "eq" %arg1, %arg0 : !llvm.ptr
    %16 = llvm.getelementptr inbounds %14[%1, 0] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.google::protobuf::internal::LogFinisher", (i8)>
    llvm.intr.lifetime.start 1, %16 : !llvm.ptr
    llvm.cond_br %15, ^bb1, ^bb3
  ^bb1:  // pred: ^bb0
    %17 = llvm.bitcast %13 : !llvm.ptr to !llvm.ptr
    llvm.intr.lifetime.start 56, %17 : !llvm.ptr
    llvm.call @_ZN6google8protobuf8internal10LogMessageC1ENS0_8LogLevelEPKci(%13, %3, %5, %6) : (!llvm.ptr, i32, !llvm.ptr, i32) -> ()
    %18 = llvm.invoke @_ZN6google8protobuf8internal10LogMessagelsEPKc(%13, %8) to ^bb2 unwind ^bb17 : (!llvm.ptr, !llvm.ptr) -> !llvm.ptr
  ^bb2:  // pred: ^bb1
    llvm.invoke @_ZN6google8protobuf8internal11LogFinisheraSERNS1_10LogMessageE(%14, %18) to ^bb4 unwind ^bb18 : (!llvm.ptr, !llvm.ptr) -> ()
  ^bb3:  // pred: ^bb0
    llvm.intr.lifetime.end 1, %16 : !llvm.ptr
    llvm.br ^bb5
  ^bb4:  // pred: ^bb2
    llvm.intr.lifetime.end 1, %16 : !llvm.ptr
    llvm.call @_ZN6google8protobuf8internal10LogMessageD1Ev(%13) : (!llvm.ptr) -> ()
    llvm.intr.lifetime.end 56, %17 : !llvm.ptr
    llvm.br ^bb5
  ^bb5:  // 2 preds: ^bb3, ^bb4
    %19 = llvm.getelementptr inbounds %arg0[%1, 0, 0, 1] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.tutorial::Person_PhoneNumber", (struct<"class.google::protobuf::Message", (struct<"class.google::protobuf::MessageLite", (ptr, struct<"class.google::protobuf::internal::InternalMetadata", (ptr)>)>)>, struct<"struct.google::protobuf::internal::ArenaStringPtr", (ptr)>, i32, struct<"class.google::protobuf::internal::CachedSize", (struct<"struct.std::atomic", (struct<"struct.std::__atomic_base", (i32)>)>)>)>
    %20 = llvm.getelementptr inbounds %arg1[%1, 0, 0, 1, 0] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.tutorial::Person_PhoneNumber", (struct<"class.google::protobuf::Message", (struct<"class.google::protobuf::MessageLite", (ptr, struct<"class.google::protobuf::internal::InternalMetadata", (ptr)>)>)>, struct<"struct.google::protobuf::internal::ArenaStringPtr", (ptr)>, i32, struct<"class.google::protobuf::internal::CachedSize", (struct<"struct.std::atomic", (struct<"struct.std::__atomic_base", (i32)>)>)>)>
    %21 = llvm.load %20 {alignment = 8 : i64, tbaa = [#tbaa_tag5]} : !llvm.ptr -> !llvm.ptr
    %22 = llvm.ptrtoint %21 : !llvm.ptr to i64
    %23 = llvm.and %22, %9  : i64
    %24 = llvm.icmp "eq" %23, %1 : i64
    llvm.cond_br %24, ^bb10, ^bb6
  ^bb6:  // pred: ^bb5
    %25 = llvm.and %22, %10  : i64
    %26 = llvm.inttoptr %25 : i64 to !llvm.ptr
    %27 = llvm.getelementptr inbounds %26[%1, 1] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.google::protobuf::internal::InternalMetadata::Container", (struct<"struct.google::protobuf::internal::InternalMetadata::ContainerBase", (ptr)>, struct<"class.google::protobuf::UnknownFieldSet", (struct<"class.std::vector", (struct<"struct.std::_Vector_base", (struct<"struct.std::_Vector_base<google::protobuf::UnknownField, std::allocator<google::protobuf::UnknownField>>::_Vector_impl", (struct<"struct.std::_Vector_base<google::protobuf::UnknownField, std::allocator<google::protobuf::UnknownField>>::_Vector_impl_data", (ptr, ptr, ptr)>)>)>)>)>)>
    %28 = llvm.getelementptr inbounds %19[%1, 0] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.google::protobuf::internal::InternalMetadata", (ptr)>
    %29 = llvm.load %28 {alignment = 8 : i64, tbaa = [#tbaa_tag5]} : !llvm.ptr -> !llvm.ptr
    %30 = llvm.ptrtoint %29 : !llvm.ptr to i64
    %31 = llvm.and %30, %9  : i64
    %32 = llvm.icmp "eq" %31, %1 : i64
    llvm.cond_br %32 weights([1, 2000]), ^bb8, ^bb7
  ^bb7:  // pred: ^bb6
    %33 = llvm.and %30, %10  : i64
    %34 = llvm.inttoptr %33 : i64 to !llvm.ptr
    %35 = llvm.getelementptr inbounds %34[%1, 1] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.google::protobuf::internal::InternalMetadata::Container", (struct<"struct.google::protobuf::internal::InternalMetadata::ContainerBase", (ptr)>, struct<"class.google::protobuf::UnknownFieldSet", (struct<"class.std::vector", (struct<"struct.std::_Vector_base", (struct<"struct.std::_Vector_base<google::protobuf::UnknownField, std::allocator<google::protobuf::UnknownField>>::_Vector_impl", (struct<"struct.std::_Vector_base<google::protobuf::UnknownField, std::allocator<google::protobuf::UnknownField>>::_Vector_impl_data", (ptr, ptr, ptr)>)>)>)>)>)>
    llvm.br ^bb9(%35 : !llvm.ptr)
  ^bb8:  // pred: ^bb6
    %36 = llvm.call @_ZN6google8protobuf8internal16InternalMetadata27mutable_unknown_fields_slowINS0_15UnknownFieldSetEEEPT_v(%19) : (!llvm.ptr) -> !llvm.ptr
    llvm.br ^bb9(%36 : !llvm.ptr)
  ^bb9(%37: !llvm.ptr):  // 2 preds: ^bb7, ^bb8
    llvm.call @_ZN6google8protobuf15UnknownFieldSet9MergeFromERKS1_(%37, %27) : (!llvm.ptr, !llvm.ptr) -> ()
    llvm.br ^bb10
  ^bb10:  // 2 preds: ^bb5, ^bb9
    %38 = llvm.getelementptr inbounds %arg1[%1, 1, 0] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.tutorial::Person_PhoneNumber", (struct<"class.google::protobuf::Message", (struct<"class.google::protobuf::MessageLite", (ptr, struct<"class.google::protobuf::internal::InternalMetadata", (ptr)>)>)>, struct<"struct.google::protobuf::internal::ArenaStringPtr", (ptr)>, i32, struct<"class.google::protobuf::internal::CachedSize", (struct<"struct.std::atomic", (struct<"struct.std::__atomic_base", (i32)>)>)>)>
    %39 = llvm.load %38 {alignment = 8 : i64, tbaa = [#tbaa_tag8]} : !llvm.ptr -> !llvm.ptr
    %40 = llvm.getelementptr inbounds %39[%1, 1] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.std::__cxx11::basic_string", (struct<"struct.std::__cxx11::basic_string<char>::_Alloc_hider", (ptr)>, i64, struct<"union.anon", (i64, array<8 x i8>)>)>
    %41 = llvm.load %40 {alignment = 8 : i64, tbaa = [#tbaa_tag25]} : !llvm.ptr -> i64
    %42 = llvm.icmp "eq" %41, %1 : i64
    llvm.cond_br %42, ^bb20, ^bb11
  ^bb11:  // pred: ^bb10
    %43 = llvm.getelementptr inbounds %arg0[%1, 1] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.tutorial::Person_PhoneNumber", (struct<"class.google::protobuf::Message", (struct<"class.google::protobuf::MessageLite", (ptr, struct<"class.google::protobuf::internal::InternalMetadata", (ptr)>)>)>, struct<"struct.google::protobuf::internal::ArenaStringPtr", (ptr)>, i32, struct<"class.google::protobuf::internal::CachedSize", (struct<"struct.std::atomic", (struct<"struct.std::__atomic_base", (i32)>)>)>)>
    %44 = llvm.getelementptr inbounds %arg0[%1, 0, 0, 1, 0] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.tutorial::Person_PhoneNumber", (struct<"class.google::protobuf::Message", (struct<"class.google::protobuf::MessageLite", (ptr, struct<"class.google::protobuf::internal::InternalMetadata", (ptr)>)>)>, struct<"struct.google::protobuf::internal::ArenaStringPtr", (ptr)>, i32, struct<"class.google::protobuf::internal::CachedSize", (struct<"struct.std::atomic", (struct<"struct.std::__atomic_base", (i32)>)>)>)>
    %45 = llvm.load %44 {alignment = 8 : i64, tbaa = [#tbaa_tag5]} : !llvm.ptr -> !llvm.ptr
    %46 = llvm.ptrtoint %45 : !llvm.ptr to i64
    %47 = llvm.and %46, %9  : i64
    %48 = llvm.icmp "eq" %47, %1 : i64
    %49 = llvm.and %46, %10  : i64
    llvm.cond_br %48 weights([2000, 1]), ^bb13, ^bb12
  ^bb12:  // pred: ^bb11
    %50 = llvm.inttoptr %49 : i64 to !llvm.ptr
    %51 = llvm.getelementptr inbounds %50[%1, 0] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.google::protobuf::internal::InternalMetadata::ContainerBase", (ptr)>
    %52 = llvm.load %51 {alignment = 8 : i64, tbaa = [#tbaa_tag9]} : !llvm.ptr -> !llvm.ptr
    llvm.br ^bb14(%52 : !llvm.ptr)
  ^bb13:  // pred: ^bb11
    %53 = llvm.inttoptr %49 : i64 to !llvm.ptr
    llvm.br ^bb14(%53 : !llvm.ptr)
  ^bb14(%54: !llvm.ptr):  // 2 preds: ^bb12, ^bb13
    %55 = llvm.getelementptr inbounds %43[%1, 0] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.google::protobuf::internal::ArenaStringPtr", (ptr)>
    %56 = llvm.load %55 {alignment = 8 : i64, tbaa = [#tbaa_tag8]} : !llvm.ptr -> !llvm.ptr
    %57 = llvm.icmp "eq" %56, %11 : !llvm.ptr
    llvm.cond_br %57, ^bb15, ^bb16
  ^bb15:  // pred: ^bb14
    llvm.call @_ZN6google8protobuf8internal14ArenaStringPtr14CreateInstanceEPNS0_5ArenaEPKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE(%43, %54, %39) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
    llvm.br ^bb20
  ^bb16:  // pred: ^bb14
    llvm.call @_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE9_M_assignERKS4_(%56, %39) : (!llvm.ptr, !llvm.ptr) -> ()
    llvm.br ^bb20
  ^bb17:  // pred: ^bb1
    %58 = llvm.landingpad cleanup : !llvm.struct<(ptr, i32)>
    llvm.br ^bb19(%58 : !llvm.struct<(ptr, i32)>)
  ^bb18:  // pred: ^bb2
    %59 = llvm.landingpad cleanup : !llvm.struct<(ptr, i32)>
    llvm.intr.lifetime.end 1, %16 : !llvm.ptr
    llvm.br ^bb19(%59 : !llvm.struct<(ptr, i32)>)
  ^bb19(%60: !llvm.struct<(ptr, i32)>):  // 2 preds: ^bb17, ^bb18
    llvm.call @_ZN6google8protobuf8internal10LogMessageD1Ev(%13) : (!llvm.ptr) -> ()
    llvm.intr.lifetime.end 56, %17 : !llvm.ptr
    llvm.resume %60 : !llvm.struct<(ptr, i32)>
  ^bb20:  // 3 preds: ^bb10, ^bb15, ^bb16
    %61 = llvm.getelementptr inbounds %arg1[%1, 2] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.tutorial::Person_PhoneNumber", (struct<"class.google::protobuf::Message", (struct<"class.google::protobuf::MessageLite", (ptr, struct<"class.google::protobuf::internal::InternalMetadata", (ptr)>)>)>, struct<"struct.google::protobuf::internal::ArenaStringPtr", (ptr)>, i32, struct<"class.google::protobuf::internal::CachedSize", (struct<"struct.std::atomic", (struct<"struct.std::__atomic_base", (i32)>)>)>)>
    %62 = llvm.load %61 {alignment = 8 : i64, tbaa = [#tbaa_tag24]} : !llvm.ptr -> i32
    %63 = llvm.icmp "eq" %62, %2 : i32
    llvm.cond_br %63, ^bb22, ^bb21
  ^bb21:  // pred: ^bb20
    %64 = llvm.getelementptr inbounds %arg0[%1, 2] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.tutorial::Person_PhoneNumber", (struct<"class.google::protobuf::Message", (struct<"class.google::protobuf::MessageLite", (ptr, struct<"class.google::protobuf::internal::InternalMetadata", (ptr)>)>)>, struct<"struct.google::protobuf::internal::ArenaStringPtr", (ptr)>, i32, struct<"class.google::protobuf::internal::CachedSize", (struct<"struct.std::atomic", (struct<"struct.std::__atomic_base", (i32)>)>)>)>
    llvm.store %62, %64 {alignment = 8 : i64, tbaa = [#tbaa_tag24]} : i32, !llvm.ptr
    llvm.br ^bb22
  ^bb22:  // 2 preds: ^bb20, ^bb21
    llvm.return
  }
  llvm.func unnamed_addr @_ZN8tutorial18Person_PhoneNumber8CopyFromERKN6google8protobuf7MessageE(%arg0: !llvm.ptr {llvm.align = 8 : i64, llvm.dereferenceable = 32 : i64, llvm.nonnull, llvm.noundef}, %arg1: !llvm.ptr {llvm.align = 8 : i64, llvm.dereferenceable = 16 : i64, llvm.nonnull, llvm.noundef}) attributes {alignment = 2 : i64, frame_pointer = #llvm.framePointerKind<none>, passthrough = [["uwtable", "2"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], ["tune-cpu", "generic"]], personality = @__gxx_personality_v0, target_cpu = "x86-64", target_features = #llvm.target_features<["+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>} {
    %0 = llvm.mlir.constant(0 : i64) : i64
    %1 = llvm.mlir.constant(0 : i32) : i32
    %2 = llvm.mlir.constant(1 : i32) : i32
    %3 = llvm.mlir.addressof @_ZN6google8protobuf8internal26fixed_address_empty_stringB5cxx11E : !llvm.ptr
    %4 = llvm.mlir.constant(0 : i8) : i8
    %5 = llvm.mlir.constant(2 : i32) : i32
    %6 = llvm.mlir.constant(1 : i64) : i64
    %7 = llvm.mlir.constant(-2 : i64) : i64
    %8 = llvm.getelementptr inbounds %arg0[%0, 0] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.tutorial::Person_PhoneNumber", (struct<"class.google::protobuf::Message", (struct<"class.google::protobuf::MessageLite", (ptr, struct<"class.google::protobuf::internal::InternalMetadata", (ptr)>)>)>, struct<"struct.google::protobuf::internal::ArenaStringPtr", (ptr)>, i32, struct<"class.google::protobuf::internal::CachedSize", (struct<"struct.std::atomic", (struct<"struct.std::__atomic_base", (i32)>)>)>)>
    %9 = llvm.icmp "eq" %8, %arg1 : !llvm.ptr
    llvm.cond_br %9, ^bb7, ^bb1
  ^bb1:  // pred: ^bb0
    %10 = llvm.getelementptr inbounds %arg0[%0, 1, 0] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.tutorial::Person_PhoneNumber", (struct<"class.google::protobuf::Message", (struct<"class.google::protobuf::MessageLite", (ptr, struct<"class.google::protobuf::internal::InternalMetadata", (ptr)>)>)>, struct<"struct.google::protobuf::internal::ArenaStringPtr", (ptr)>, i32, struct<"class.google::protobuf::internal::CachedSize", (struct<"struct.std::atomic", (struct<"struct.std::__atomic_base", (i32)>)>)>)>
    %11 = llvm.load %10 {alignment = 8 : i64, tbaa = [#tbaa_tag8]} : !llvm.ptr -> !llvm.ptr
    %12 = llvm.icmp "eq" %11, %3 : !llvm.ptr
    llvm.cond_br %12, ^bb3, ^bb2
  ^bb2:  // pred: ^bb1
    %13 = llvm.getelementptr inbounds %11[%0, 1] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.std::__cxx11::basic_string", (struct<"struct.std::__cxx11::basic_string<char>::_Alloc_hider", (ptr)>, i64, struct<"union.anon", (i64, array<8 x i8>)>)>
    llvm.store %0, %13 {alignment = 8 : i64, tbaa = [#tbaa_tag25]} : i64, !llvm.ptr
    %14 = llvm.getelementptr inbounds %11[%0, 0, 0] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.std::__cxx11::basic_string", (struct<"struct.std::__cxx11::basic_string<char>::_Alloc_hider", (ptr)>, i64, struct<"union.anon", (i64, array<8 x i8>)>)>
    %15 = llvm.load %14 {alignment = 8 : i64, tbaa = [#tbaa_tag26]} : !llvm.ptr -> !llvm.ptr
    llvm.store %4, %15 {alignment = 1 : i64, tbaa = [#tbaa_tag1]} : i8, !llvm.ptr
    llvm.br ^bb3
  ^bb3:  // 2 preds: ^bb1, ^bb2
    %16 = llvm.getelementptr inbounds %arg0[%0, 2] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.tutorial::Person_PhoneNumber", (struct<"class.google::protobuf::Message", (struct<"class.google::protobuf::MessageLite", (ptr, struct<"class.google::protobuf::internal::InternalMetadata", (ptr)>)>)>, struct<"struct.google::protobuf::internal::ArenaStringPtr", (ptr)>, i32, struct<"class.google::protobuf::internal::CachedSize", (struct<"struct.std::atomic", (struct<"struct.std::__atomic_base", (i32)>)>)>)>
    llvm.store %1, %16 {alignment = 8 : i64, tbaa = [#tbaa_tag24]} : i32, !llvm.ptr
    %17 = llvm.getelementptr inbounds %arg0[%0, 0, 0, 1, 0] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.tutorial::Person_PhoneNumber", (struct<"class.google::protobuf::Message", (struct<"class.google::protobuf::MessageLite", (ptr, struct<"class.google::protobuf::internal::InternalMetadata", (ptr)>)>)>, struct<"struct.google::protobuf::internal::ArenaStringPtr", (ptr)>, i32, struct<"class.google::protobuf::internal::CachedSize", (struct<"struct.std::atomic", (struct<"struct.std::__atomic_base", (i32)>)>)>)>
    %18 = llvm.load %17 {alignment = 8 : i64, tbaa = [#tbaa_tag5]} : !llvm.ptr -> !llvm.ptr
    %19 = llvm.ptrtoint %18 : !llvm.ptr to i64
    %20 = llvm.and %19, %6  : i64
    %21 = llvm.icmp "eq" %20, %0 : i64
    llvm.cond_br %21, ^bb6, ^bb4
  ^bb4:  // pred: ^bb3
    %22 = llvm.and %19, %7  : i64
    %23 = llvm.inttoptr %22 : i64 to !llvm.ptr
    %24 = llvm.getelementptr inbounds %23[%0, 1] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.google::protobuf::internal::InternalMetadata::Container", (struct<"struct.google::protobuf::internal::InternalMetadata::ContainerBase", (ptr)>, struct<"class.google::protobuf::UnknownFieldSet", (struct<"class.std::vector", (struct<"struct.std::_Vector_base", (struct<"struct.std::_Vector_base<google::protobuf::UnknownField, std::allocator<google::protobuf::UnknownField>>::_Vector_impl", (struct<"struct.std::_Vector_base<google::protobuf::UnknownField, std::allocator<google::protobuf::UnknownField>>::_Vector_impl_data", (ptr, ptr, ptr)>)>)>)>)>)>
    %25 = llvm.getelementptr inbounds %24[%0, 0, 0, 0, 0, 0] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.google::protobuf::UnknownFieldSet", (struct<"class.std::vector", (struct<"struct.std::_Vector_base", (struct<"struct.std::_Vector_base<google::protobuf::UnknownField, std::allocator<google::protobuf::UnknownField>>::_Vector_impl", (struct<"struct.std::_Vector_base<google::protobuf::UnknownField, std::allocator<google::protobuf::UnknownField>>::_Vector_impl_data", (ptr, ptr, ptr)>)>)>)>)>
    %26 = llvm.load %25 {alignment = 8 : i64, tbaa = [#tbaa_tag2]} : !llvm.ptr -> !llvm.ptr
    %27 = llvm.getelementptr inbounds %23[%0, 1, 0, 0, 0, 0, 1] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.google::protobuf::internal::InternalMetadata::Container", (struct<"struct.google::protobuf::internal::InternalMetadata::ContainerBase", (ptr)>, struct<"class.google::protobuf::UnknownFieldSet", (struct<"class.std::vector", (struct<"struct.std::_Vector_base", (struct<"struct.std::_Vector_base<google::protobuf::UnknownField, std::allocator<google::protobuf::UnknownField>>::_Vector_impl", (struct<"struct.std::_Vector_base<google::protobuf::UnknownField, std::allocator<google::protobuf::UnknownField>>::_Vector_impl_data", (ptr, ptr, ptr)>)>)>)>)>)>
    %28 = llvm.load %27 {alignment = 8 : i64, tbaa = [#tbaa_tag2]} : !llvm.ptr -> !llvm.ptr
    %29 = llvm.icmp "eq" %26, %28 : !llvm.ptr
    llvm.cond_br %29, ^bb6, ^bb5
  ^bb5:  // pred: ^bb4
    llvm.call @_ZN6google8protobuf15UnknownFieldSet13ClearFallbackEv(%24) : (!llvm.ptr) -> ()
    llvm.br ^bb6
  ^bb6:  // 3 preds: ^bb3, ^bb4, ^bb5
    llvm.call @_ZN8tutorial18Person_PhoneNumber9MergeFromERKN6google8protobuf7MessageE(%arg0, %arg1) : (!llvm.ptr, !llvm.ptr) -> ()
    llvm.br ^bb7
  ^bb7:  // 2 preds: ^bb0, ^bb6
    llvm.return
  }
  llvm.func local_unnamed_addr @_ZN8tutorial18Person_PhoneNumber8CopyFromERKS0_(%arg0: !llvm.ptr {llvm.align = 8 : i64, llvm.dereferenceable = 32 : i64, llvm.nonnull, llvm.noundef}, %arg1: !llvm.ptr {llvm.align = 8 : i64, llvm.dereferenceable = 32 : i64, llvm.nonnull, llvm.noundef}) attributes {alignment = 2 : i64, frame_pointer = #llvm.framePointerKind<none>, passthrough = [["uwtable", "2"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], ["tune-cpu", "generic"]], personality = @__gxx_personality_v0, target_cpu = "x86-64", target_features = #llvm.target_features<["+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>} {
    %0 = llvm.mlir.constant(0 : i64) : i64
    %1 = llvm.mlir.constant(1 : i32) : i32
    %2 = llvm.mlir.constant(0 : i32) : i32
    %3 = llvm.mlir.addressof @_ZN6google8protobuf8internal26fixed_address_empty_stringB5cxx11E : !llvm.ptr
    %4 = llvm.mlir.constant(0 : i8) : i8
    %5 = llvm.mlir.constant(2 : i32) : i32
    %6 = llvm.mlir.constant(1 : i64) : i64
    %7 = llvm.mlir.constant(-2 : i64) : i64
    %8 = llvm.icmp "eq" %arg1, %arg0 : !llvm.ptr
    llvm.cond_br %8, ^bb7, ^bb1
  ^bb1:  // pred: ^bb0
    %9 = llvm.getelementptr inbounds %arg0[%0, 1, 0] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.tutorial::Person_PhoneNumber", (struct<"class.google::protobuf::Message", (struct<"class.google::protobuf::MessageLite", (ptr, struct<"class.google::protobuf::internal::InternalMetadata", (ptr)>)>)>, struct<"struct.google::protobuf::internal::ArenaStringPtr", (ptr)>, i32, struct<"class.google::protobuf::internal::CachedSize", (struct<"struct.std::atomic", (struct<"struct.std::__atomic_base", (i32)>)>)>)>
    %10 = llvm.load %9 {alignment = 8 : i64, tbaa = [#tbaa_tag8]} : !llvm.ptr -> !llvm.ptr
    %11 = llvm.icmp "eq" %10, %3 : !llvm.ptr
    llvm.cond_br %11, ^bb3, ^bb2
  ^bb2:  // pred: ^bb1
    %12 = llvm.getelementptr inbounds %10[%0, 1] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.std::__cxx11::basic_string", (struct<"struct.std::__cxx11::basic_string<char>::_Alloc_hider", (ptr)>, i64, struct<"union.anon", (i64, array<8 x i8>)>)>
    llvm.store %0, %12 {alignment = 8 : i64, tbaa = [#tbaa_tag25]} : i64, !llvm.ptr
    %13 = llvm.getelementptr inbounds %10[%0, 0, 0] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.std::__cxx11::basic_string", (struct<"struct.std::__cxx11::basic_string<char>::_Alloc_hider", (ptr)>, i64, struct<"union.anon", (i64, array<8 x i8>)>)>
    %14 = llvm.load %13 {alignment = 8 : i64, tbaa = [#tbaa_tag26]} : !llvm.ptr -> !llvm.ptr
    llvm.store %4, %14 {alignment = 1 : i64, tbaa = [#tbaa_tag1]} : i8, !llvm.ptr
    llvm.br ^bb3
  ^bb3:  // 2 preds: ^bb1, ^bb2
    %15 = llvm.getelementptr inbounds %arg0[%0, 2] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.tutorial::Person_PhoneNumber", (struct<"class.google::protobuf::Message", (struct<"class.google::protobuf::MessageLite", (ptr, struct<"class.google::protobuf::internal::InternalMetadata", (ptr)>)>)>, struct<"struct.google::protobuf::internal::ArenaStringPtr", (ptr)>, i32, struct<"class.google::protobuf::internal::CachedSize", (struct<"struct.std::atomic", (struct<"struct.std::__atomic_base", (i32)>)>)>)>
    llvm.store %2, %15 {alignment = 8 : i64, tbaa = [#tbaa_tag24]} : i32, !llvm.ptr
    %16 = llvm.getelementptr inbounds %arg0[%0, 0, 0, 1, 0] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.tutorial::Person_PhoneNumber", (struct<"class.google::protobuf::Message", (struct<"class.google::protobuf::MessageLite", (ptr, struct<"class.google::protobuf::internal::InternalMetadata", (ptr)>)>)>, struct<"struct.google::protobuf::internal::ArenaStringPtr", (ptr)>, i32, struct<"class.google::protobuf::internal::CachedSize", (struct<"struct.std::atomic", (struct<"struct.std::__atomic_base", (i32)>)>)>)>
    %17 = llvm.load %16 {alignment = 8 : i64, tbaa = [#tbaa_tag5]} : !llvm.ptr -> !llvm.ptr
    %18 = llvm.ptrtoint %17 : !llvm.ptr to i64
    %19 = llvm.and %18, %6  : i64
    %20 = llvm.icmp "eq" %19, %0 : i64
    llvm.cond_br %20, ^bb6, ^bb4
  ^bb4:  // pred: ^bb3
    %21 = llvm.and %18, %7  : i64
    %22 = llvm.inttoptr %21 : i64 to !llvm.ptr
    %23 = llvm.getelementptr inbounds %22[%0, 1] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.google::protobuf::internal::InternalMetadata::Container", (struct<"struct.google::protobuf::internal::InternalMetadata::ContainerBase", (ptr)>, struct<"class.google::protobuf::UnknownFieldSet", (struct<"class.std::vector", (struct<"struct.std::_Vector_base", (struct<"struct.std::_Vector_base<google::protobuf::UnknownField, std::allocator<google::protobuf::UnknownField>>::_Vector_impl", (struct<"struct.std::_Vector_base<google::protobuf::UnknownField, std::allocator<google::protobuf::UnknownField>>::_Vector_impl_data", (ptr, ptr, ptr)>)>)>)>)>)>
    %24 = llvm.getelementptr inbounds %23[%0, 0, 0, 0, 0, 0] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.google::protobuf::UnknownFieldSet", (struct<"class.std::vector", (struct<"struct.std::_Vector_base", (struct<"struct.std::_Vector_base<google::protobuf::UnknownField, std::allocator<google::protobuf::UnknownField>>::_Vector_impl", (struct<"struct.std::_Vector_base<google::protobuf::UnknownField, std::allocator<google::protobuf::UnknownField>>::_Vector_impl_data", (ptr, ptr, ptr)>)>)>)>)>
    %25 = llvm.load %24 {alignment = 8 : i64, tbaa = [#tbaa_tag2]} : !llvm.ptr -> !llvm.ptr
    %26 = llvm.getelementptr inbounds %22[%0, 1, 0, 0, 0, 0, 1] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.google::protobuf::internal::InternalMetadata::Container", (struct<"struct.google::protobuf::internal::InternalMetadata::ContainerBase", (ptr)>, struct<"class.google::protobuf::UnknownFieldSet", (struct<"class.std::vector", (struct<"struct.std::_Vector_base", (struct<"struct.std::_Vector_base<google::protobuf::UnknownField, std::allocator<google::protobuf::UnknownField>>::_Vector_impl", (struct<"struct.std::_Vector_base<google::protobuf::UnknownField, std::allocator<google::protobuf::UnknownField>>::_Vector_impl_data", (ptr, ptr, ptr)>)>)>)>)>)>
    %27 = llvm.load %26 {alignment = 8 : i64, tbaa = [#tbaa_tag2]} : !llvm.ptr -> !llvm.ptr
    %28 = llvm.icmp "eq" %25, %27 : !llvm.ptr
    llvm.cond_br %28, ^bb6, ^bb5
  ^bb5:  // pred: ^bb4
    llvm.call @_ZN6google8protobuf15UnknownFieldSet13ClearFallbackEv(%23) : (!llvm.ptr) -> ()
    llvm.br ^bb6
  ^bb6:  // 3 preds: ^bb3, ^bb4, ^bb5
    llvm.call @_ZN8tutorial18Person_PhoneNumber9MergeFromERKS0_(%arg0, %arg1) : (!llvm.ptr, !llvm.ptr) -> ()
    llvm.br ^bb7
  ^bb7:  // 2 preds: ^bb0, ^bb6
    llvm.return
  }
  llvm.func unnamed_addr @_ZNK8tutorial18Person_PhoneNumber13IsInitializedEv(%arg0: !llvm.ptr {llvm.align = 8 : i64, llvm.nocapture, llvm.nonnull, llvm.readnone}) -> (i1 {llvm.noundef, llvm.zeroext}) attributes {alignment = 2 : i64, frame_pointer = #llvm.framePointerKind<none>, memory = #llvm.memory_effects<other = none, argMem = none, inaccessibleMem = none>, passthrough = ["mustprogress", "nofree", "norecurse", "nosync", "nounwind", "willreturn", ["uwtable", "2"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], ["tune-cpu", "generic"]], target_cpu = "x86-64", target_features = #llvm.target_features<["+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>} {
    %0 = llvm.mlir.constant(true) : i1
    llvm.return %0 : i1
  }
  llvm.func local_unnamed_addr @_ZN8tutorial18Person_PhoneNumber12InternalSwapEPS0_(%arg0: !llvm.ptr {llvm.align = 8 : i64, llvm.dereferenceable = 32 : i64, llvm.nonnull, llvm.noundef}, %arg1: !llvm.ptr {llvm.noundef}) attributes {alignment = 2 : i64, frame_pointer = #llvm.framePointerKind<none>, passthrough = [["uwtable", "2"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], ["tune-cpu", "generic"]], personality = @__gxx_personality_v0, target_cpu = "x86-64", target_features = #llvm.target_features<["+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>} {
    %0 = llvm.mlir.constant(0 : i64) : i64
    %1 = llvm.mlir.constant(0 : i32) : i32
    %2 = llvm.mlir.constant(1 : i32) : i32
    %3 = llvm.mlir.constant(1 : i64) : i64
    %4 = llvm.mlir.constant(false) : i1
    %5 = llvm.mlir.constant(-2 : i64) : i64
    %6 = llvm.mlir.constant(2 : i32) : i32
    %7 = llvm.mlir.addressof @_ZN6google8protobuf8internal26fixed_address_empty_stringB5cxx11E : !llvm.ptr
    %8 = llvm.getelementptr inbounds %arg0[%0, 0, 0, 1] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.tutorial::Person_PhoneNumber", (struct<"class.google::protobuf::Message", (struct<"class.google::protobuf::MessageLite", (ptr, struct<"class.google::protobuf::internal::InternalMetadata", (ptr)>)>)>, struct<"struct.google::protobuf::internal::ArenaStringPtr", (ptr)>, i32, struct<"class.google::protobuf::internal::CachedSize", (struct<"struct.std::atomic", (struct<"struct.std::__atomic_base", (i32)>)>)>)>
    %9 = llvm.getelementptr inbounds %arg1[%0, 0, 0, 1] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.tutorial::Person_PhoneNumber", (struct<"class.google::protobuf::Message", (struct<"class.google::protobuf::MessageLite", (ptr, struct<"class.google::protobuf::internal::InternalMetadata", (ptr)>)>)>, struct<"struct.google::protobuf::internal::ArenaStringPtr", (ptr)>, i32, struct<"class.google::protobuf::internal::CachedSize", (struct<"struct.std::atomic", (struct<"struct.std::__atomic_base", (i32)>)>)>)>
    %10 = llvm.getelementptr inbounds %8[%0, 0] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.google::protobuf::internal::InternalMetadata", (ptr)>
    %11 = llvm.load %10 {alignment = 8 : i64, tbaa = [#tbaa_tag5]} : !llvm.ptr -> !llvm.ptr
    %12 = llvm.ptrtoint %11 : !llvm.ptr to i64
    %13 = llvm.and %12, %3  : i64
    %14 = llvm.icmp "eq" %13, %0 : i64
    %15 = llvm.getelementptr inbounds %9[%0, 0] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.google::protobuf::internal::InternalMetadata", (ptr)>
    %16 = llvm.load %15 {alignment = 8 : i64, tbaa = [#tbaa_tag5]} : !llvm.ptr -> !llvm.ptr
    %17 = llvm.ptrtoint %16 : !llvm.ptr to i64
    %18 = llvm.and %17, %3  : i64
    %19 = llvm.icmp "eq" %18, %0 : i64
    %20 = llvm.select %14, %19, %4 : i1, i1
    llvm.cond_br %20, ^bb8(%12 : i64), ^bb1
  ^bb1:  // pred: ^bb0
    llvm.cond_br %19 weights([1, 2000]), ^bb3, ^bb2
  ^bb2:  // pred: ^bb1
    %21 = llvm.and %17, %5  : i64
    %22 = llvm.inttoptr %21 : i64 to !llvm.ptr
    %23 = llvm.getelementptr inbounds %22[%0, 1] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.google::protobuf::internal::InternalMetadata::Container", (struct<"struct.google::protobuf::internal::InternalMetadata::ContainerBase", (ptr)>, struct<"class.google::protobuf::UnknownFieldSet", (struct<"class.std::vector", (struct<"struct.std::_Vector_base", (struct<"struct.std::_Vector_base<google::protobuf::UnknownField, std::allocator<google::protobuf::UnknownField>>::_Vector_impl", (struct<"struct.std::_Vector_base<google::protobuf::UnknownField, std::allocator<google::protobuf::UnknownField>>::_Vector_impl_data", (ptr, ptr, ptr)>)>)>)>)>)>
    llvm.br ^bb4(%11, %12, %23 : !llvm.ptr, i64, !llvm.ptr)
  ^bb3:  // pred: ^bb1
    %24 = llvm.call @_ZN6google8protobuf8internal16InternalMetadata27mutable_unknown_fields_slowINS0_15UnknownFieldSetEEEPT_v(%9) : (!llvm.ptr) -> !llvm.ptr
    %25 = llvm.load %10 {alignment = 8 : i64, tbaa = [#tbaa_tag5]} : !llvm.ptr -> !llvm.ptr
    %26 = llvm.ptrtoint %25 : !llvm.ptr to i64
    llvm.br ^bb4(%25, %26, %24 : !llvm.ptr, i64, !llvm.ptr)
  ^bb4(%27: !llvm.ptr, %28: i64, %29: !llvm.ptr):  // 2 preds: ^bb2, ^bb3
    %30 = llvm.and %28, %3  : i64
    %31 = llvm.icmp "eq" %30, %0 : i64
    llvm.cond_br %31 weights([1, 2000]), ^bb6, ^bb5
  ^bb5:  // pred: ^bb4
    %32 = llvm.and %28, %5  : i64
    %33 = llvm.inttoptr %32 : i64 to !llvm.ptr
    %34 = llvm.getelementptr inbounds %33[%0, 1] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.google::protobuf::internal::InternalMetadata::Container", (struct<"struct.google::protobuf::internal::InternalMetadata::ContainerBase", (ptr)>, struct<"class.google::protobuf::UnknownFieldSet", (struct<"class.std::vector", (struct<"struct.std::_Vector_base", (struct<"struct.std::_Vector_base<google::protobuf::UnknownField, std::allocator<google::protobuf::UnknownField>>::_Vector_impl", (struct<"struct.std::_Vector_base<google::protobuf::UnknownField, std::allocator<google::protobuf::UnknownField>>::_Vector_impl_data", (ptr, ptr, ptr)>)>)>)>)>)>
    llvm.br ^bb7(%27, %34 : !llvm.ptr, !llvm.ptr)
  ^bb6:  // pred: ^bb4
    %35 = llvm.call @_ZN6google8protobuf8internal16InternalMetadata27mutable_unknown_fields_slowINS0_15UnknownFieldSetEEEPT_v(%8) : (!llvm.ptr) -> !llvm.ptr
    %36 = llvm.getelementptr inbounds %arg0[%0, 0, 0, 1, 0] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.tutorial::Person_PhoneNumber", (struct<"class.google::protobuf::Message", (struct<"class.google::protobuf::MessageLite", (ptr, struct<"class.google::protobuf::internal::InternalMetadata", (ptr)>)>)>, struct<"struct.google::protobuf::internal::ArenaStringPtr", (ptr)>, i32, struct<"class.google::protobuf::internal::CachedSize", (struct<"struct.std::atomic", (struct<"struct.std::__atomic_base", (i32)>)>)>)>
    %37 = llvm.load %36 {alignment = 8 : i64, tbaa = [#tbaa_tag5]} : !llvm.ptr -> !llvm.ptr
    llvm.br ^bb7(%37, %35 : !llvm.ptr, !llvm.ptr)
  ^bb7(%38: !llvm.ptr, %39: !llvm.ptr):  // 2 preds: ^bb5, ^bb6
    %40 = llvm.bitcast %39 : !llvm.ptr to !llvm.ptr
    %41 = llvm.load %40 {alignment = 8 : i64, tbaa = [#tbaa_tag2]} : !llvm.ptr -> !llvm.vec<2 x ptr>
    %42 = llvm.getelementptr inbounds %39[%0, 0, 0, 0, 0, 2] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.google::protobuf::UnknownFieldSet", (struct<"class.std::vector", (struct<"struct.std::_Vector_base", (struct<"struct.std::_Vector_base<google::protobuf::UnknownField, std::allocator<google::protobuf::UnknownField>>::_Vector_impl", (struct<"struct.std::_Vector_base<google::protobuf::UnknownField, std::allocator<google::protobuf::UnknownField>>::_Vector_impl_data", (ptr, ptr, ptr)>)>)>)>)>
    %43 = llvm.load %42 {alignment = 8 : i64, tbaa = [#tbaa_tag13]} : !llvm.ptr -> !llvm.ptr
    %44 = llvm.bitcast %29 : !llvm.ptr to !llvm.ptr
    %45 = llvm.load %44 {alignment = 8 : i64, tbaa = [#tbaa_tag2]} : !llvm.ptr -> !llvm.vec<2 x ptr>
    %46 = llvm.bitcast %39 : !llvm.ptr to !llvm.ptr
    llvm.store %45, %46 {alignment = 8 : i64, tbaa = [#tbaa_tag2]} : !llvm.vec<2 x ptr>, !llvm.ptr
    %47 = llvm.getelementptr inbounds %29[%0, 0, 0, 0, 0, 2] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.google::protobuf::UnknownFieldSet", (struct<"class.std::vector", (struct<"struct.std::_Vector_base", (struct<"struct.std::_Vector_base<google::protobuf::UnknownField, std::allocator<google::protobuf::UnknownField>>::_Vector_impl", (struct<"struct.std::_Vector_base<google::protobuf::UnknownField, std::allocator<google::protobuf::UnknownField>>::_Vector_impl_data", (ptr, ptr, ptr)>)>)>)>)>
    %48 = llvm.load %47 {alignment = 8 : i64, tbaa = [#tbaa_tag13]} : !llvm.ptr -> !llvm.ptr
    llvm.store %48, %42 {alignment = 8 : i64, tbaa = [#tbaa_tag13]} : !llvm.ptr, !llvm.ptr
    %49 = llvm.bitcast %29 : !llvm.ptr to !llvm.ptr
    llvm.store %41, %49 {alignment = 8 : i64, tbaa = [#tbaa_tag2]} : !llvm.vec<2 x ptr>, !llvm.ptr
    llvm.store %43, %47 {alignment = 8 : i64, tbaa = [#tbaa_tag13]} : !llvm.ptr, !llvm.ptr
    %50 = llvm.ptrtoint %38 : !llvm.ptr to i64
    llvm.br ^bb8(%50 : i64)
  ^bb8(%51: i64):  // 2 preds: ^bb0, ^bb7
    %52 = llvm.getelementptr inbounds %arg0[%0, 1] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.tutorial::Person_PhoneNumber", (struct<"class.google::protobuf::Message", (struct<"class.google::protobuf::MessageLite", (ptr, struct<"class.google::protobuf::internal::InternalMetadata", (ptr)>)>)>, struct<"struct.google::protobuf::internal::ArenaStringPtr", (ptr)>, i32, struct<"class.google::protobuf::internal::CachedSize", (struct<"struct.std::atomic", (struct<"struct.std::__atomic_base", (i32)>)>)>)>
    %53 = llvm.getelementptr inbounds %arg1[%0, 1] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.tutorial::Person_PhoneNumber", (struct<"class.google::protobuf::Message", (struct<"class.google::protobuf::MessageLite", (ptr, struct<"class.google::protobuf::internal::InternalMetadata", (ptr)>)>)>, struct<"struct.google::protobuf::internal::ArenaStringPtr", (ptr)>, i32, struct<"class.google::protobuf::internal::CachedSize", (struct<"struct.std::atomic", (struct<"struct.std::__atomic_base", (i32)>)>)>)>
    %54 = llvm.and %51, %3  : i64
    %55 = llvm.icmp "eq" %54, %0 : i64
    %56 = llvm.and %51, %5  : i64
    llvm.cond_br %55 weights([2000, 1]), ^bb10, ^bb9
  ^bb9:  // pred: ^bb8
    %57 = llvm.inttoptr %56 : i64 to !llvm.ptr
    %58 = llvm.getelementptr inbounds %57[%0, 0] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.google::protobuf::internal::InternalMetadata::ContainerBase", (ptr)>
    %59 = llvm.load %58 {alignment = 8 : i64, tbaa = [#tbaa_tag9]} : !llvm.ptr -> !llvm.ptr
    llvm.br ^bb11(%59 : !llvm.ptr)
  ^bb10:  // pred: ^bb8
    %60 = llvm.inttoptr %56 : i64 to !llvm.ptr
    llvm.br ^bb11(%60 : !llvm.ptr)
  ^bb11(%61: !llvm.ptr):  // 2 preds: ^bb9, ^bb10
    %62 = llvm.getelementptr inbounds %52[%0, 0] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.google::protobuf::internal::ArenaStringPtr", (ptr)>
    %63 = llvm.load %62 {alignment = 8 : i64, tbaa = [#tbaa_tag8]} : !llvm.ptr -> !llvm.ptr
    %64 = llvm.icmp "eq" %63, %7 : !llvm.ptr
    llvm.cond_br %64, ^bb12, ^bb14(%63 : !llvm.ptr)
  ^bb12:  // pred: ^bb11
    %65 = llvm.getelementptr inbounds %53[%0, 0] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.google::protobuf::internal::ArenaStringPtr", (ptr)>
    %66 = llvm.load %65 {alignment = 8 : i64, tbaa = [#tbaa_tag8]} : !llvm.ptr -> !llvm.ptr
    %67 = llvm.icmp "eq" %66, %7 : !llvm.ptr
    llvm.cond_br %67, ^bb17, ^bb13
  ^bb13:  // pred: ^bb12
    llvm.call @_ZN6google8protobuf8internal14ArenaStringPtr14CreateInstanceEPNS0_5ArenaEPKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE(%52, %61, %7) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
    %68 = llvm.load %62 {alignment = 8 : i64, tbaa = [#tbaa_tag8]} : !llvm.ptr -> !llvm.ptr
    llvm.br ^bb14(%68 : !llvm.ptr)
  ^bb14(%69: !llvm.ptr):  // 2 preds: ^bb11, ^bb13
    %70 = llvm.getelementptr inbounds %53[%0, 0] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.google::protobuf::internal::ArenaStringPtr", (ptr)>
    %71 = llvm.load %70 {alignment = 8 : i64, tbaa = [#tbaa_tag8]} : !llvm.ptr -> !llvm.ptr
    %72 = llvm.icmp "eq" %71, %7 : !llvm.ptr
    llvm.cond_br %72, ^bb15, ^bb16(%71 : !llvm.ptr)
  ^bb15:  // pred: ^bb14
    llvm.call @_ZN6google8protobuf8internal14ArenaStringPtr14CreateInstanceEPNS0_5ArenaEPKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE(%53, %61, %7) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
    %73 = llvm.load %70 {alignment = 8 : i64, tbaa = [#tbaa_tag8]} : !llvm.ptr -> !llvm.ptr
    llvm.br ^bb16(%73 : !llvm.ptr)
  ^bb16(%74: !llvm.ptr):  // 2 preds: ^bb14, ^bb15
    llvm.call @_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE4swapERS4_(%69, %74) : (!llvm.ptr, !llvm.ptr) -> ()
    llvm.br ^bb17
  ^bb17:  // 2 preds: ^bb12, ^bb16
    %75 = llvm.getelementptr inbounds %arg0[%0, 2] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.tutorial::Person_PhoneNumber", (struct<"class.google::protobuf::Message", (struct<"class.google::protobuf::MessageLite", (ptr, struct<"class.google::protobuf::internal::InternalMetadata", (ptr)>)>)>, struct<"struct.google::protobuf::internal::ArenaStringPtr", (ptr)>, i32, struct<"class.google::protobuf::internal::CachedSize", (struct<"struct.std::atomic", (struct<"struct.std::__atomic_base", (i32)>)>)>)>
    %76 = llvm.getelementptr inbounds %arg1[%0, 2] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.tutorial::Person_PhoneNumber", (struct<"class.google::protobuf::Message", (struct<"class.google::protobuf::MessageLite", (ptr, struct<"class.google::protobuf::internal::InternalMetadata", (ptr)>)>)>, struct<"struct.google::protobuf::internal::ArenaStringPtr", (ptr)>, i32, struct<"class.google::protobuf::internal::CachedSize", (struct<"struct.std::atomic", (struct<"struct.std::__atomic_base", (i32)>)>)>)>
    %77 = llvm.load %75 {alignment = 8 : i64, tbaa = [#tbaa_tag3]} : !llvm.ptr -> i32
    %78 = llvm.load %76 {alignment = 4 : i64, tbaa = [#tbaa_tag3]} : !llvm.ptr -> i32
    llvm.store %78, %75 {alignment = 8 : i64, tbaa = [#tbaa_tag3]} : i32, !llvm.ptr
    llvm.store %77, %76 {alignment = 4 : i64, tbaa = [#tbaa_tag3]} : i32, !llvm.ptr
    llvm.return
  }
  llvm.func unnamed_addr @_ZNK8tutorial18Person_PhoneNumber11GetMetadataEv(%arg0: !llvm.ptr {llvm.align = 8 : i64, llvm.nocapture, llvm.nonnull, llvm.readnone}) -> !llvm.struct<(ptr, ptr)> attributes {alignment = 2 : i64, frame_pointer = #llvm.framePointerKind<none>, passthrough = ["mustprogress", ["uwtable", "2"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], ["tune-cpu", "generic"]], target_cpu = "x86-64", target_features = #llvm.target_features<["+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>} {
    %0 = llvm.mlir.zero : !llvm.ptr
    %1 = llvm.mlir.undef : !llvm.array<1 x ptr>
    %2 = llvm.insertvalue %0, %1[0] : !llvm.array<1 x ptr> 
    %3 = llvm.mlir.addressof @_ZL47file_level_enum_descriptors_addressbook_2eproto : !llvm.ptr
    %4 = llvm.mlir.constant(3 : i32) : i32
    %5 = llvm.mlir.undef : !llvm.struct<"struct.google::protobuf::Metadata", (ptr, ptr)>
    %6 = llvm.insertvalue %0, %5[0] : !llvm.struct<"struct.google::protobuf::Metadata", (ptr, ptr)> 
    %7 = llvm.insertvalue %0, %6[1] : !llvm.struct<"struct.google::protobuf::Metadata", (ptr, ptr)> 
    %8 = llvm.mlir.undef : !llvm.array<3 x struct<"struct.google::protobuf::Metadata", (ptr, ptr)>>
    %9 = llvm.insertvalue %7, %8[0] : !llvm.array<3 x struct<"struct.google::protobuf::Metadata", (ptr, ptr)>> 
    %10 = llvm.insertvalue %7, %9[1] : !llvm.array<3 x struct<"struct.google::protobuf::Metadata", (ptr, ptr)>> 
    %11 = llvm.insertvalue %7, %10[2] : !llvm.array<3 x struct<"struct.google::protobuf::Metadata", (ptr, ptr)>> 
    %12 = llvm.mlir.addressof @_ZL39file_level_metadata_addressbook_2eproto : !llvm.ptr
    %13 = llvm.mlir.constant(dense<[-1, 8, -1, -1, -1, 16, 24, -1, 8, -1, -1, -1, 40, 64, 48, 16, 56, -1, 8, -1, -1, -1, 16]> : tensor<23xi32>) : !llvm.array<23 x i32>
    %14 = llvm.mlir.addressof @_ZN31TableStruct_addressbook_2eproto7offsetsE : !llvm.ptr
    %15 = llvm.mlir.constant(0 : i8) : i8
    %16 = llvm.mlir.constant(dense<0> : tensor<40xi8>) : !llvm.array<40 x i8>
    %17 = llvm.mlir.constant(0 : i64) : i64
    %18 = llvm.mlir.undef : !llvm.struct<"union.google::protobuf::internal::ExplicitlyConstructed<tutorial::AddressBook>::AlignedUnion", (i64, array<40 x i8>)>
    %19 = llvm.insertvalue %17, %18[0] : !llvm.struct<"union.google::protobuf::internal::ExplicitlyConstructed<tutorial::AddressBook>::AlignedUnion", (i64, array<40 x i8>)> 
    %20 = llvm.insertvalue %16, %19[1] : !llvm.struct<"union.google::protobuf::internal::ExplicitlyConstructed<tutorial::AddressBook>::AlignedUnion", (i64, array<40 x i8>)> 
    %21 = llvm.mlir.undef : !llvm.struct<"class.google::protobuf::internal::ExplicitlyConstructed.1", (struct<"union.google::protobuf::internal::ExplicitlyConstructed<tutorial::AddressBook>::AlignedUnion", (i64, array<40 x i8>)>)>
    %22 = llvm.insertvalue %20, %21[0] : !llvm.struct<"class.google::protobuf::internal::ExplicitlyConstructed.1", (struct<"union.google::protobuf::internal::ExplicitlyConstructed<tutorial::AddressBook>::AlignedUnion", (i64, array<40 x i8>)>)> 
    %23 = llvm.mlir.undef : !llvm.struct<"class.tutorial::AddressBookDefaultTypeInternal", (struct<"class.google::protobuf::internal::ExplicitlyConstructed.1", (struct<"union.google::protobuf::internal::ExplicitlyConstructed<tutorial::AddressBook>::AlignedUnion", (i64, array<40 x i8>)>)>)>
    %24 = llvm.insertvalue %22, %23[0] : !llvm.struct<"class.tutorial::AddressBookDefaultTypeInternal", (struct<"class.google::protobuf::internal::ExplicitlyConstructed.1", (struct<"union.google::protobuf::internal::ExplicitlyConstructed<tutorial::AddressBook>::AlignedUnion", (i64, array<40 x i8>)>)>)> 
    %25 = llvm.mlir.addressof @_ZN8tutorial30_AddressBook_default_instance_E : !llvm.ptr
    %26 = llvm.mlir.constant(dense<0> : tensor<64xi8>) : !llvm.array<64 x i8>
    %27 = llvm.mlir.undef : !llvm.struct<"union.google::protobuf::internal::ExplicitlyConstructed<tutorial::Person>::AlignedUnion", (i64, array<64 x i8>)>
    %28 = llvm.insertvalue %17, %27[0] : !llvm.struct<"union.google::protobuf::internal::ExplicitlyConstructed<tutorial::Person>::AlignedUnion", (i64, array<64 x i8>)> 
    %29 = llvm.insertvalue %26, %28[1] : !llvm.struct<"union.google::protobuf::internal::ExplicitlyConstructed<tutorial::Person>::AlignedUnion", (i64, array<64 x i8>)> 
    %30 = llvm.mlir.undef : !llvm.struct<"class.google::protobuf::internal::ExplicitlyConstructed.0", (struct<"union.google::protobuf::internal::ExplicitlyConstructed<tutorial::Person>::AlignedUnion", (i64, array<64 x i8>)>)>
    %31 = llvm.insertvalue %29, %30[0] : !llvm.struct<"class.google::protobuf::internal::ExplicitlyConstructed.0", (struct<"union.google::protobuf::internal::ExplicitlyConstructed<tutorial::Person>::AlignedUnion", (i64, array<64 x i8>)>)> 
    %32 = llvm.mlir.undef : !llvm.struct<"class.tutorial::PersonDefaultTypeInternal", (struct<"class.google::protobuf::internal::ExplicitlyConstructed.0", (struct<"union.google::protobuf::internal::ExplicitlyConstructed<tutorial::Person>::AlignedUnion", (i64, array<64 x i8>)>)>)>
    %33 = llvm.insertvalue %31, %32[0] : !llvm.struct<"class.tutorial::PersonDefaultTypeInternal", (struct<"class.google::protobuf::internal::ExplicitlyConstructed.0", (struct<"union.google::protobuf::internal::ExplicitlyConstructed<tutorial::Person>::AlignedUnion", (i64, array<64 x i8>)>)>)> 
    %34 = llvm.mlir.addressof @_ZN8tutorial25_Person_default_instance_E : !llvm.ptr
    %35 = llvm.mlir.constant(dense<0> : tensor<24xi8>) : !llvm.array<24 x i8>
    %36 = llvm.mlir.undef : !llvm.struct<"union.google::protobuf::internal::ExplicitlyConstructed<tutorial::Person_PhoneNumber>::AlignedUnion", (i64, array<24 x i8>)>
    %37 = llvm.insertvalue %17, %36[0] : !llvm.struct<"union.google::protobuf::internal::ExplicitlyConstructed<tutorial::Person_PhoneNumber>::AlignedUnion", (i64, array<24 x i8>)> 
    %38 = llvm.insertvalue %35, %37[1] : !llvm.struct<"union.google::protobuf::internal::ExplicitlyConstructed<tutorial::Person_PhoneNumber>::AlignedUnion", (i64, array<24 x i8>)> 
    %39 = llvm.mlir.undef : !llvm.struct<"class.google::protobuf::internal::ExplicitlyConstructed", (struct<"union.google::protobuf::internal::ExplicitlyConstructed<tutorial::Person_PhoneNumber>::AlignedUnion", (i64, array<24 x i8>)>)>
    %40 = llvm.insertvalue %38, %39[0] : !llvm.struct<"class.google::protobuf::internal::ExplicitlyConstructed", (struct<"union.google::protobuf::internal::ExplicitlyConstructed<tutorial::Person_PhoneNumber>::AlignedUnion", (i64, array<24 x i8>)>)> 
    %41 = llvm.mlir.undef : !llvm.struct<"class.tutorial::Person_PhoneNumberDefaultTypeInternal", (struct<"class.google::protobuf::internal::ExplicitlyConstructed", (struct<"union.google::protobuf::internal::ExplicitlyConstructed<tutorial::Person_PhoneNumber>::AlignedUnion", (i64, array<24 x i8>)>)>)>
    %42 = llvm.insertvalue %40, %41[0] : !llvm.struct<"class.tutorial::Person_PhoneNumberDefaultTypeInternal", (struct<"class.google::protobuf::internal::ExplicitlyConstructed", (struct<"union.google::protobuf::internal::ExplicitlyConstructed<tutorial::Person_PhoneNumber>::AlignedUnion", (i64, array<24 x i8>)>)>)> 
    %43 = llvm.mlir.addressof @_ZN8tutorial37_Person_PhoneNumber_default_instance_E : !llvm.ptr
    %44 = llvm.mlir.undef : !llvm.array<3 x ptr>
    %45 = llvm.insertvalue %43, %44[0] : !llvm.array<3 x ptr> 
    %46 = llvm.insertvalue %34, %45[1] : !llvm.array<3 x ptr> 
    %47 = llvm.insertvalue %25, %46[2] : !llvm.array<3 x ptr> 
    %48 = llvm.mlir.addressof @_ZL22file_default_instances : !llvm.ptr
    %49 = llvm.mlir.constant(48 : i32) : i32
    %50 = llvm.mlir.constant(-1 : i32) : i32
    %51 = llvm.mlir.constant(17 : i32) : i32
    %52 = llvm.mlir.undef : !llvm.struct<"struct.google::protobuf::internal::MigrationSchema", (i32, i32, i32)>
    %53 = llvm.insertvalue %51, %52[0] : !llvm.struct<"struct.google::protobuf::internal::MigrationSchema", (i32, i32, i32)> 
    %54 = llvm.insertvalue %50, %53[1] : !llvm.struct<"struct.google::protobuf::internal::MigrationSchema", (i32, i32, i32)> 
    %55 = llvm.insertvalue %49, %54[2] : !llvm.struct<"struct.google::protobuf::internal::MigrationSchema", (i32, i32, i32)> 
    %56 = llvm.mlir.constant(72 : i32) : i32
    %57 = llvm.mlir.constant(7 : i32) : i32
    %58 = llvm.mlir.undef : !llvm.struct<"struct.google::protobuf::internal::MigrationSchema", (i32, i32, i32)>
    %59 = llvm.insertvalue %57, %58[0] : !llvm.struct<"struct.google::protobuf::internal::MigrationSchema", (i32, i32, i32)> 
    %60 = llvm.insertvalue %50, %59[1] : !llvm.struct<"struct.google::protobuf::internal::MigrationSchema", (i32, i32, i32)> 
    %61 = llvm.insertvalue %56, %60[2] : !llvm.struct<"struct.google::protobuf::internal::MigrationSchema", (i32, i32, i32)> 
    %62 = llvm.mlir.constant(32 : i32) : i32
    %63 = llvm.mlir.constant(0 : i32) : i32
    %64 = llvm.mlir.undef : !llvm.struct<"struct.google::protobuf::internal::MigrationSchema", (i32, i32, i32)>
    %65 = llvm.insertvalue %63, %64[0] : !llvm.struct<"struct.google::protobuf::internal::MigrationSchema", (i32, i32, i32)> 
    %66 = llvm.insertvalue %50, %65[1] : !llvm.struct<"struct.google::protobuf::internal::MigrationSchema", (i32, i32, i32)> 
    %67 = llvm.insertvalue %62, %66[2] : !llvm.struct<"struct.google::protobuf::internal::MigrationSchema", (i32, i32, i32)> 
    %68 = llvm.mlir.undef : !llvm.array<3 x struct<"struct.google::protobuf::internal::MigrationSchema", (i32, i32, i32)>>
    %69 = llvm.insertvalue %67, %68[0] : !llvm.array<3 x struct<"struct.google::protobuf::internal::MigrationSchema", (i32, i32, i32)>> 
    %70 = llvm.insertvalue %61, %69[1] : !llvm.array<3 x struct<"struct.google::protobuf::internal::MigrationSchema", (i32, i32, i32)>> 
    %71 = llvm.insertvalue %55, %70[2] : !llvm.array<3 x struct<"struct.google::protobuf::internal::MigrationSchema", (i32, i32, i32)>> 
    %72 = llvm.mlir.addressof @_ZL7schemas : !llvm.ptr
    %73 = llvm.mlir.constant(1 : i32) : i32
    %74 = llvm.mlir.addressof @descriptor_table_google_2fprotobuf_2ftimestamp_2eproto : !llvm.ptr
    %75 = llvm.mlir.undef : !llvm.array<1 x ptr>
    %76 = llvm.insertvalue %74, %75[0] : !llvm.array<1 x ptr> 
    %77 = llvm.mlir.addressof @_ZL41descriptor_table_addressbook_2eproto_deps : !llvm.ptr
    %78 = llvm.mlir.undef : !llvm.array<0 x ptr>
    %79 = llvm.mlir.addressof @__gxx_personality_v0 : !llvm.ptr
    %80 = llvm.mlir.addressof @_ZL59InitDefaultsscc_info_Person_PhoneNumber_addressbook_2eprotov : !llvm.ptr
    %81 = llvm.mlir.undef : !llvm.struct<(i32)>
    %82 = llvm.insertvalue %50, %81[0] : !llvm.struct<(i32)> 
    %83 = llvm.mlir.undef : !llvm.struct<(struct<(i32)>, i32, i32, ptr)>
    %84 = llvm.insertvalue %82, %83[0] : !llvm.struct<(struct<(i32)>, i32, i32, ptr)> 
    %85 = llvm.insertvalue %63, %84[1] : !llvm.struct<(struct<(i32)>, i32, i32, ptr)> 
    %86 = llvm.insertvalue %63, %85[2] : !llvm.struct<(struct<(i32)>, i32, i32, ptr)> 
    %87 = llvm.insertvalue %80, %86[3] : !llvm.struct<(struct<(i32)>, i32, i32, ptr)> 
    %88 = llvm.mlir.undef : !llvm.struct<(struct<(struct<(i32)>, i32, i32, ptr)>, array<0 x ptr>)>
    %89 = llvm.insertvalue %87, %88[0] : !llvm.struct<(struct<(struct<(i32)>, i32, i32, ptr)>, array<0 x ptr>)> 
    %90 = llvm.insertvalue %78, %89[1] : !llvm.struct<(struct<(struct<(i32)>, i32, i32, ptr)>, array<0 x ptr>)> 
    %91 = llvm.mlir.addressof @scc_info_Person_PhoneNumber_addressbook_2eproto : !llvm.ptr
    %92 = llvm.mlir.addressof @scc_info_Timestamp_google_2fprotobuf_2ftimestamp_2eproto : !llvm.ptr
    %93 = llvm.mlir.undef : !llvm.array<2 x ptr>
    %94 = llvm.insertvalue %91, %93[0] : !llvm.array<2 x ptr> 
    %95 = llvm.insertvalue %92, %94[1] : !llvm.array<2 x ptr> 
    %96 = llvm.mlir.addressof @_ZL47InitDefaultsscc_info_Person_addressbook_2eprotov : !llvm.ptr
    %97 = llvm.mlir.constant(2 : i32) : i32
    %98 = llvm.mlir.undef : !llvm.struct<(struct<(i32)>, i32, i32, ptr)>
    %99 = llvm.insertvalue %82, %98[0] : !llvm.struct<(struct<(i32)>, i32, i32, ptr)> 
    %100 = llvm.insertvalue %97, %99[1] : !llvm.struct<(struct<(i32)>, i32, i32, ptr)> 
    %101 = llvm.insertvalue %63, %100[2] : !llvm.struct<(struct<(i32)>, i32, i32, ptr)> 
    %102 = llvm.insertvalue %96, %101[3] : !llvm.struct<(struct<(i32)>, i32, i32, ptr)> 
    %103 = llvm.mlir.undef : !llvm.struct<(struct<(struct<(i32)>, i32, i32, ptr)>, array<2 x ptr>)>
    %104 = llvm.insertvalue %102, %103[0] : !llvm.struct<(struct<(struct<(i32)>, i32, i32, ptr)>, array<2 x ptr>)> 
    %105 = llvm.insertvalue %95, %104[1] : !llvm.struct<(struct<(struct<(i32)>, i32, i32, ptr)>, array<2 x ptr>)> 
    %106 = llvm.mlir.addressof @scc_info_Person_addressbook_2eproto : !llvm.ptr
    %107 = llvm.mlir.undef : !llvm.array<1 x ptr>
    %108 = llvm.insertvalue %106, %107[0] : !llvm.array<1 x ptr> 
    %109 = llvm.mlir.addressof @_ZL52InitDefaultsscc_info_AddressBook_addressbook_2eprotov : !llvm.ptr
    %110 = llvm.mlir.undef : !llvm.struct<(struct<(i32)>, i32, i32, ptr)>
    %111 = llvm.insertvalue %82, %110[0] : !llvm.struct<(struct<(i32)>, i32, i32, ptr)> 
    %112 = llvm.insertvalue %73, %111[1] : !llvm.struct<(struct<(i32)>, i32, i32, ptr)> 
    %113 = llvm.insertvalue %63, %112[2] : !llvm.struct<(struct<(i32)>, i32, i32, ptr)> 
    %114 = llvm.insertvalue %109, %113[3] : !llvm.struct<(struct<(i32)>, i32, i32, ptr)> 
    %115 = llvm.mlir.undef : !llvm.struct<(struct<(struct<(i32)>, i32, i32, ptr)>, array<1 x ptr>)>
    %116 = llvm.insertvalue %114, %115[0] : !llvm.struct<(struct<(struct<(i32)>, i32, i32, ptr)>, array<1 x ptr>)> 
    %117 = llvm.insertvalue %108, %116[1] : !llvm.struct<(struct<(struct<(i32)>, i32, i32, ptr)>, array<1 x ptr>)> 
    %118 = llvm.mlir.addressof @scc_info_AddressBook_addressbook_2eproto : !llvm.ptr
    %119 = llvm.mlir.undef : !llvm.array<3 x ptr>
    %120 = llvm.insertvalue %118, %119[0] : !llvm.array<3 x ptr> 
    %121 = llvm.insertvalue %106, %120[1] : !llvm.array<3 x ptr> 
    %122 = llvm.insertvalue %91, %121[2] : !llvm.array<3 x ptr> 
    %123 = llvm.mlir.addressof @_ZL41descriptor_table_addressbook_2eproto_sccs : !llvm.ptr
    %124 = llvm.mlir.undef : !llvm.struct<"struct.std::once_flag", (i32)>
    %125 = llvm.insertvalue %63, %124[0] : !llvm.struct<"struct.std::once_flag", (i32)> 
    %126 = llvm.mlir.addressof @_ZL41descriptor_table_addressbook_2eproto_once : !llvm.ptr
    %127 = llvm.mlir.constant(537 : i32) : i32
    %128 = llvm.mlir.constant("addressbook.proto\00") : !llvm.array<18 x i8>
    %129 = llvm.mlir.addressof @".str" : !llvm.ptr
    %130 = llvm.mlir.constant("\0A\11addressbook.proto\12\08tutorial\1A\1Fgoogle/protobuf/timestamp.proto\22\87\02\0A\06Person\12\0C\0A\04name\18\01 \01(\09\12\0A\0A\02id\18\02 \01(\05\12\0D\0A\05email\18\03 \01(\09\12,\0A\06phones\18\04 \03(\0B2\1C.tutorial.Person.PhoneNumber\120\0A\0Clast_updated\18\05 \01(\0B2\1A.google.protobuf.Timestamp\1AG\0A\0BPhoneNumber\12\0E\0A\06number\18\01 \01(\09\12(\0A\04type\18\02 \01(\0E2\1A.tutorial.Person.PhoneType\22+\0A\09PhoneType\12\0A\0A\06MOBILE\10\00\12\08\0A\04HOME\10\01\12\08\0A\04WORK\10\02\22/\0A\0BAddressBook\12 \0A\06people\18\01 \03(\0B2\10.tutorial.PersonB\95\01\0A\1Bcom.example.tutorial.protosB\11AddressBookProtosP\01Z:github.com/protocolbuffers/protobuf/examples/go/tutorialpb\AA\02$Google.Protobuf.Examples.AddressBookb\06proto3\00") : !llvm.array<538 x i8>
    %131 = llvm.mlir.addressof @_ZL45descriptor_table_protodef_addressbook_2eproto : !llvm.ptr
    %132 = llvm.mlir.undef : !llvm.struct<"struct.google::protobuf::internal::DescriptorTable", (i8, i8, ptr, ptr, i32, ptr, ptr, ptr, i32, i32, ptr, ptr, ptr, ptr, i32, ptr, ptr)>
    %133 = llvm.insertvalue %15, %132[0] : !llvm.struct<"struct.google::protobuf::internal::DescriptorTable", (i8, i8, ptr, ptr, i32, ptr, ptr, ptr, i32, i32, ptr, ptr, ptr, ptr, i32, ptr, ptr)> 
    %134 = llvm.insertvalue %15, %133[1] : !llvm.struct<"struct.google::protobuf::internal::DescriptorTable", (i8, i8, ptr, ptr, i32, ptr, ptr, ptr, i32, i32, ptr, ptr, ptr, ptr, i32, ptr, ptr)> 
    %135 = llvm.insertvalue %131, %134[2] : !llvm.struct<"struct.google::protobuf::internal::DescriptorTable", (i8, i8, ptr, ptr, i32, ptr, ptr, ptr, i32, i32, ptr, ptr, ptr, ptr, i32, ptr, ptr)> 
    %136 = llvm.insertvalue %129, %135[3] : !llvm.struct<"struct.google::protobuf::internal::DescriptorTable", (i8, i8, ptr, ptr, i32, ptr, ptr, ptr, i32, i32, ptr, ptr, ptr, ptr, i32, ptr, ptr)> 
    %137 = llvm.insertvalue %127, %136[4] : !llvm.struct<"struct.google::protobuf::internal::DescriptorTable", (i8, i8, ptr, ptr, i32, ptr, ptr, ptr, i32, i32, ptr, ptr, ptr, ptr, i32, ptr, ptr)> 
    %138 = llvm.insertvalue %126, %137[5] : !llvm.struct<"struct.google::protobuf::internal::DescriptorTable", (i8, i8, ptr, ptr, i32, ptr, ptr, ptr, i32, i32, ptr, ptr, ptr, ptr, i32, ptr, ptr)> 
    %139 = llvm.insertvalue %123, %138[6] : !llvm.struct<"struct.google::protobuf::internal::DescriptorTable", (i8, i8, ptr, ptr, i32, ptr, ptr, ptr, i32, i32, ptr, ptr, ptr, ptr, i32, ptr, ptr)> 
    %140 = llvm.insertvalue %77, %139[7] : !llvm.struct<"struct.google::protobuf::internal::DescriptorTable", (i8, i8, ptr, ptr, i32, ptr, ptr, ptr, i32, i32, ptr, ptr, ptr, ptr, i32, ptr, ptr)> 
    %141 = llvm.insertvalue %4, %140[8] : !llvm.struct<"struct.google::protobuf::internal::DescriptorTable", (i8, i8, ptr, ptr, i32, ptr, ptr, ptr, i32, i32, ptr, ptr, ptr, ptr, i32, ptr, ptr)> 
    %142 = llvm.insertvalue %73, %141[9] : !llvm.struct<"struct.google::protobuf::internal::DescriptorTable", (i8, i8, ptr, ptr, i32, ptr, ptr, ptr, i32, i32, ptr, ptr, ptr, ptr, i32, ptr, ptr)> 
    %143 = llvm.insertvalue %72, %142[10] : !llvm.struct<"struct.google::protobuf::internal::DescriptorTable", (i8, i8, ptr, ptr, i32, ptr, ptr, ptr, i32, i32, ptr, ptr, ptr, ptr, i32, ptr, ptr)> 
    %144 = llvm.insertvalue %48, %143[11] : !llvm.struct<"struct.google::protobuf::internal::DescriptorTable", (i8, i8, ptr, ptr, i32, ptr, ptr, ptr, i32, i32, ptr, ptr, ptr, ptr, i32, ptr, ptr)> 
    %145 = llvm.insertvalue %14, %144[12] : !llvm.struct<"struct.google::protobuf::internal::DescriptorTable", (i8, i8, ptr, ptr, i32, ptr, ptr, ptr, i32, i32, ptr, ptr, ptr, ptr, i32, ptr, ptr)> 
    %146 = llvm.insertvalue %12, %145[13] : !llvm.struct<"struct.google::protobuf::internal::DescriptorTable", (i8, i8, ptr, ptr, i32, ptr, ptr, ptr, i32, i32, ptr, ptr, ptr, ptr, i32, ptr, ptr)> 
    %147 = llvm.insertvalue %4, %146[14] : !llvm.struct<"struct.google::protobuf::internal::DescriptorTable", (i8, i8, ptr, ptr, i32, ptr, ptr, ptr, i32, i32, ptr, ptr, ptr, ptr, i32, ptr, ptr)> 
    %148 = llvm.insertvalue %3, %147[15] : !llvm.struct<"struct.google::protobuf::internal::DescriptorTable", (i8, i8, ptr, ptr, i32, ptr, ptr, ptr, i32, i32, ptr, ptr, ptr, ptr, i32, ptr, ptr)> 
    %149 = llvm.insertvalue %0, %148[16] : !llvm.struct<"struct.google::protobuf::internal::DescriptorTable", (i8, i8, ptr, ptr, i32, ptr, ptr, ptr, i32, i32, ptr, ptr, ptr, ptr, i32, ptr, ptr)> 
    %150 = llvm.mlir.addressof @descriptor_table_addressbook_2eproto : !llvm.ptr
    %151 = llvm.mlir.constant(false) : i1
    %152 = llvm.mlir.constant(13 : i32) : i32
    %153 = llvm.getelementptr inbounds %150[%17, 13] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.google::protobuf::internal::DescriptorTable", (i8, i8, ptr, ptr, i32, ptr, ptr, ptr, i32, i32, ptr, ptr, ptr, ptr, i32, ptr, ptr)>
    %154 = llvm.mlir.poison : !llvm.struct<(ptr, ptr)>
    llvm.call @_ZN6google8protobuf8internal17AssignDescriptorsEPKNS1_15DescriptorTableEb(%150, %151) : (!llvm.ptr, i1) -> ()
    %155 = llvm.load %153 {alignment = 8 : i64, tbaa = [#tbaa_tag14]} : !llvm.ptr -> !llvm.ptr
    %156 = llvm.getelementptr inbounds %155[%17, 0] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.google::protobuf::Metadata", (ptr, ptr)>
    %157 = llvm.load %156 {alignment = 8 : i64} : !llvm.ptr -> !llvm.ptr
    %158 = llvm.getelementptr inbounds %155[%17, 1] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.google::protobuf::Metadata", (ptr, ptr)>
    %159 = llvm.load %158 {alignment = 8 : i64} : !llvm.ptr -> !llvm.ptr
    %160 = llvm.insertvalue %157, %154[0] : !llvm.struct<(ptr, ptr)> 
    %161 = llvm.insertvalue %159, %160[1] : !llvm.struct<(ptr, ptr)> 
    llvm.return %161 : !llvm.struct<(ptr, ptr)>
  }
  llvm.func local_unnamed_addr @_ZN8tutorial6Person21InitAsDefaultInstanceEv() attributes {alignment = 2 : i64, frame_pointer = #llvm.framePointerKind<none>, memory = #llvm.memory_effects<other = write, argMem = write, inaccessibleMem = write>, passthrough = ["mustprogress", "nofree", "norecurse", "nosync", "nounwind", "willreturn", ["uwtable", "2"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], ["tune-cpu", "generic"]], target_cpu = "x86-64", target_features = #llvm.target_features<["+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>} {
    %0 = llvm.mlir.addressof @_ZN6google8protobuf28_Timestamp_default_instance_E : !llvm.ptr
    %1 = llvm.mlir.constant(48 : i64) : i64
    %2 = llvm.mlir.constant(1 : i32) : i32
    %3 = llvm.mlir.constant(0 : i32) : i32
    %4 = llvm.mlir.constant(0 : i64) : i64
    %5 = llvm.mlir.constant(0 : i8) : i8
    %6 = llvm.mlir.constant(dense<0> : tensor<64xi8>) : !llvm.array<64 x i8>
    %7 = llvm.mlir.undef : !llvm.struct<"union.google::protobuf::internal::ExplicitlyConstructed<tutorial::Person>::AlignedUnion", (i64, array<64 x i8>)>
    %8 = llvm.insertvalue %4, %7[0] : !llvm.struct<"union.google::protobuf::internal::ExplicitlyConstructed<tutorial::Person>::AlignedUnion", (i64, array<64 x i8>)> 
    %9 = llvm.insertvalue %6, %8[1] : !llvm.struct<"union.google::protobuf::internal::ExplicitlyConstructed<tutorial::Person>::AlignedUnion", (i64, array<64 x i8>)> 
    %10 = llvm.mlir.undef : !llvm.struct<"class.google::protobuf::internal::ExplicitlyConstructed.0", (struct<"union.google::protobuf::internal::ExplicitlyConstructed<tutorial::Person>::AlignedUnion", (i64, array<64 x i8>)>)>
    %11 = llvm.insertvalue %9, %10[0] : !llvm.struct<"class.google::protobuf::internal::ExplicitlyConstructed.0", (struct<"union.google::protobuf::internal::ExplicitlyConstructed<tutorial::Person>::AlignedUnion", (i64, array<64 x i8>)>)> 
    %12 = llvm.mlir.undef : !llvm.struct<"class.tutorial::PersonDefaultTypeInternal", (struct<"class.google::protobuf::internal::ExplicitlyConstructed.0", (struct<"union.google::protobuf::internal::ExplicitlyConstructed<tutorial::Person>::AlignedUnion", (i64, array<64 x i8>)>)>)>
    %13 = llvm.insertvalue %11, %12[0] : !llvm.struct<"class.tutorial::PersonDefaultTypeInternal", (struct<"class.google::protobuf::internal::ExplicitlyConstructed.0", (struct<"union.google::protobuf::internal::ExplicitlyConstructed<tutorial::Person>::AlignedUnion", (i64, array<64 x i8>)>)>)> 
    %14 = llvm.mlir.addressof @_ZN8tutorial25_Person_default_instance_E : !llvm.ptr
    %15 = llvm.getelementptr inbounds %14[%4, 0, 0, 1, %1] : (!llvm.ptr, i64, i64) -> !llvm.ptr, !llvm.struct<"class.tutorial::PersonDefaultTypeInternal", (struct<"class.google::protobuf::internal::ExplicitlyConstructed.0", (struct<"union.google::protobuf::internal::ExplicitlyConstructed<tutorial::Person>::AlignedUnion", (i64, array<64 x i8>)>)>)>
    llvm.store %0, %15 {alignment = 8 : i64, tbaa = [#tbaa_tag23]} : !llvm.ptr, !llvm.ptr
    llvm.return
  }
  llvm.func local_unnamed_addr @_ZN8tutorial6Person9_Internal12last_updatedEPKS0_(%arg0: !llvm.ptr {llvm.nocapture, llvm.noundef, llvm.readonly}) -> (!llvm.ptr {llvm.align = 8 : i64, llvm.dereferenceable = 32 : i64, llvm.nonnull, llvm.noundef}) attributes {alignment = 2 : i64, frame_pointer = #llvm.framePointerKind<none>, memory = #llvm.memory_effects<other = read, argMem = read, inaccessibleMem = read>, passthrough = ["mustprogress", "nofree", "norecurse", "nosync", "nounwind", "willreturn", ["uwtable", "2"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], ["tune-cpu", "generic"]], target_cpu = "x86-64", target_features = #llvm.target_features<["+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>} {
    %0 = llvm.mlir.constant(0 : i64) : i64
    %1 = llvm.mlir.constant(4 : i32) : i32
    %2 = llvm.getelementptr inbounds %arg0[%0, 4] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.tutorial::Person", (struct<"class.google::protobuf::Message", (struct<"class.google::protobuf::MessageLite", (ptr, struct<"class.google::protobuf::internal::InternalMetadata", (ptr)>)>)>, struct<"class.google::protobuf::RepeatedPtrField", (struct<"class.google::protobuf::internal::RepeatedPtrFieldBase", (ptr, i32, i32, ptr)>)>, struct<"struct.google::protobuf::internal::ArenaStringPtr", (ptr)>, struct<"struct.google::protobuf::internal::ArenaStringPtr", (ptr)>, ptr, i32, struct<"class.google::protobuf::internal::CachedSize", (struct<"struct.std::atomic", (struct<"struct.std::__atomic_base", (i32)>)>)>)>
    %3 = llvm.load %2 {alignment = 8 : i64, tbaa = [#tbaa_tag23]} : !llvm.ptr -> !llvm.ptr
    llvm.return %3 : !llvm.ptr
  }
  llvm.func local_unnamed_addr @_ZN8tutorial6Person18clear_last_updatedEv(%arg0: !llvm.ptr {llvm.align = 8 : i64, llvm.dereferenceable = 72 : i64, llvm.nocapture, llvm.nonnull, llvm.noundef}) attributes {alignment = 2 : i64, frame_pointer = #llvm.framePointerKind<none>, passthrough = ["mustprogress", "nounwind", ["uwtable", "2"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], ["tune-cpu", "generic"]], target_cpu = "x86-64", target_features = #llvm.target_features<["+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>} {
    %0 = llvm.mlir.constant(0 : i64) : i64
    %1 = llvm.mlir.constant(0 : i32) : i32
    %2 = llvm.mlir.constant(1 : i32) : i32
    %3 = llvm.mlir.constant(1 : i64) : i64
    %4 = llvm.mlir.constant(-2 : i64) : i64
    %5 = llvm.mlir.zero : !llvm.ptr
    %6 = llvm.mlir.constant(4 : i32) : i32
    %7 = llvm.getelementptr inbounds %arg0[%0, 0, 0, 1, 0] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.tutorial::Person", (struct<"class.google::protobuf::Message", (struct<"class.google::protobuf::MessageLite", (ptr, struct<"class.google::protobuf::internal::InternalMetadata", (ptr)>)>)>, struct<"class.google::protobuf::RepeatedPtrField", (struct<"class.google::protobuf::internal::RepeatedPtrFieldBase", (ptr, i32, i32, ptr)>)>, struct<"struct.google::protobuf::internal::ArenaStringPtr", (ptr)>, struct<"struct.google::protobuf::internal::ArenaStringPtr", (ptr)>, ptr, i32, struct<"class.google::protobuf::internal::CachedSize", (struct<"struct.std::atomic", (struct<"struct.std::__atomic_base", (i32)>)>)>)>
    %8 = llvm.load %7 {alignment = 8 : i64, tbaa = [#tbaa_tag5]} : !llvm.ptr -> !llvm.ptr
    %9 = llvm.ptrtoint %8 : !llvm.ptr to i64
    %10 = llvm.and %9, %3  : i64
    %11 = llvm.icmp "eq" %10, %0 : i64
    %12 = llvm.and %9, %4  : i64
    llvm.cond_br %11 weights([2000, 1]), ^bb2, ^bb1
  ^bb1:  // pred: ^bb0
    %13 = llvm.inttoptr %12 : i64 to !llvm.ptr
    %14 = llvm.getelementptr inbounds %13[%0, 0] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.google::protobuf::internal::InternalMetadata::ContainerBase", (ptr)>
    %15 = llvm.load %14 {alignment = 8 : i64, tbaa = [#tbaa_tag9]} : !llvm.ptr -> !llvm.ptr
    llvm.br ^bb3(%15 : !llvm.ptr)
  ^bb2:  // pred: ^bb0
    %16 = llvm.inttoptr %12 : i64 to !llvm.ptr
    llvm.br ^bb3(%16 : !llvm.ptr)
  ^bb3(%17: !llvm.ptr):  // 2 preds: ^bb1, ^bb2
    %18 = llvm.icmp "eq" %17, %5 : !llvm.ptr
    llvm.cond_br %18, ^bb4, ^bb6
  ^bb4:  // pred: ^bb3
    %19 = llvm.getelementptr inbounds %arg0[%0, 4] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.tutorial::Person", (struct<"class.google::protobuf::Message", (struct<"class.google::protobuf::MessageLite", (ptr, struct<"class.google::protobuf::internal::InternalMetadata", (ptr)>)>)>, struct<"class.google::protobuf::RepeatedPtrField", (struct<"class.google::protobuf::internal::RepeatedPtrFieldBase", (ptr, i32, i32, ptr)>)>, struct<"struct.google::protobuf::internal::ArenaStringPtr", (ptr)>, struct<"struct.google::protobuf::internal::ArenaStringPtr", (ptr)>, ptr, i32, struct<"class.google::protobuf::internal::CachedSize", (struct<"struct.std::atomic", (struct<"struct.std::__atomic_base", (i32)>)>)>)>
    %20 = llvm.load %19 {alignment = 8 : i64, tbaa = [#tbaa_tag23]} : !llvm.ptr -> !llvm.ptr
    %21 = llvm.icmp "eq" %20, %5 : !llvm.ptr
    llvm.cond_br %21, ^bb6, ^bb5
  ^bb5:  // pred: ^bb4
    llvm.call @_ZN6google8protobuf9TimestampD1Ev(%20) : (!llvm.ptr) -> ()
    %22 = llvm.bitcast %20 : !llvm.ptr to !llvm.ptr
    llvm.call @_ZdlPv(%22) : (!llvm.ptr) -> ()
    llvm.br ^bb6
  ^bb6:  // 3 preds: ^bb3, ^bb4, ^bb5
    %23 = llvm.getelementptr inbounds %arg0[%0, 4] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.tutorial::Person", (struct<"class.google::protobuf::Message", (struct<"class.google::protobuf::MessageLite", (ptr, struct<"class.google::protobuf::internal::InternalMetadata", (ptr)>)>)>, struct<"class.google::protobuf::RepeatedPtrField", (struct<"class.google::protobuf::internal::RepeatedPtrFieldBase", (ptr, i32, i32, ptr)>)>, struct<"struct.google::protobuf::internal::ArenaStringPtr", (ptr)>, struct<"struct.google::protobuf::internal::ArenaStringPtr", (ptr)>, ptr, i32, struct<"class.google::protobuf::internal::CachedSize", (struct<"struct.std::atomic", (struct<"struct.std::__atomic_base", (i32)>)>)>)>
    llvm.store %5, %23 {alignment = 8 : i64, tbaa = [#tbaa_tag23]} : !llvm.ptr, !llvm.ptr
    llvm.return
  }
  llvm.func unnamed_addr @_ZN6google8protobuf9TimestampD1Ev(!llvm.ptr {llvm.align = 8 : i64, llvm.dereferenceable = 32 : i64, llvm.nonnull, llvm.noundef}) attributes {frame_pointer = #llvm.framePointerKind<none>, passthrough = ["nounwind", ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], ["tune-cpu", "generic"]], target_cpu = "x86-64", target_features = #llvm.target_features<["+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>}
  llvm.func unnamed_addr @_ZN8tutorial6PersonC2EPN6google8protobuf5ArenaE(%arg0: !llvm.ptr {llvm.align = 8 : i64, llvm.dereferenceable = 72 : i64, llvm.nonnull, llvm.noundef}, %arg1: !llvm.ptr {llvm.noundef}) attributes {alignment = 2 : i64, frame_pointer = #llvm.framePointerKind<none>, passthrough = [["uwtable", "2"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], ["tune-cpu", "generic"]], personality = @__gxx_personality_v0, target_cpu = "x86-64", target_features = #llvm.target_features<["+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>} {
    %0 = llvm.mlir.constant(0 : i64) : i64
    %1 = llvm.mlir.constant(0 : i32) : i32
    %2 = llvm.mlir.constant(1 : i32) : i32
    %3 = llvm.mlir.constant(2 : i64) : i64
    %4 = llvm.mlir.addressof @_ZNK8tutorial6Person11GetMetadataEv : !llvm.ptr
    %5 = llvm.mlir.zero : !llvm.ptr
    %6 = llvm.mlir.addressof @__gxx_personality_v0 : !llvm.ptr
    %7 = llvm.mlir.addressof @_ZNK8tutorial6Person13SetCachedSizeEi : !llvm.ptr
    %8 = llvm.mlir.addressof @_ZNK6google8protobuf7Message13SpaceUsedLongEv : !llvm.ptr
    %9 = llvm.mlir.addressof @_ZN6google8protobuf7Message20DiscardUnknownFieldsEv : !llvm.ptr
    %10 = llvm.mlir.addressof @_ZN8tutorial6Person9MergeFromERKN6google8protobuf7MessageE : !llvm.ptr
    %11 = llvm.mlir.addressof @_ZN8tutorial6Person8CopyFromERKN6google8protobuf7MessageE : !llvm.ptr
    %12 = llvm.mlir.addressof @_ZNK6google8protobuf11MessageLite16InternalGetTableEv : !llvm.ptr
    %13 = llvm.mlir.addressof @_ZNK8tutorial6Person18_InternalSerializeEPhPN6google8protobuf2io19EpsCopyOutputStreamE : !llvm.ptr
    %14 = llvm.mlir.addressof @_ZN8tutorial6Person14_InternalParseEPKcPN6google8protobuf8internal12ParseContextE : !llvm.ptr
    %15 = llvm.mlir.addressof @_ZNK8tutorial6Person13GetCachedSizeEv : !llvm.ptr
    %16 = llvm.mlir.addressof @_ZNK8tutorial6Person12ByteSizeLongEv : !llvm.ptr
    %17 = llvm.mlir.addressof @_ZN6google8protobuf7Message21CheckTypeAndMergeFromERKNS0_11MessageLiteE : !llvm.ptr
    %18 = llvm.mlir.addressof @_ZNK6google8protobuf7Message25InitializationErrorStringB5cxx11Ev : !llvm.ptr
    %19 = llvm.mlir.addressof @_ZNK8tutorial6Person13IsInitializedEv : !llvm.ptr
    %20 = llvm.mlir.addressof @_ZN8tutorial6Person5ClearEv : !llvm.ptr
    %21 = llvm.mlir.addressof @_ZNK8tutorial6Person3NewEPN6google8protobuf5ArenaE : !llvm.ptr
    %22 = llvm.mlir.addressof @_ZNK8tutorial6Person3NewEv : !llvm.ptr
    %23 = llvm.mlir.addressof @_ZNK6google8protobuf7Message11GetTypeNameB5cxx11Ev : !llvm.ptr
    %24 = llvm.mlir.addressof @_ZN8tutorial6PersonD0Ev : !llvm.ptr
    %25 = llvm.mlir.addressof @_ZN8tutorial6PersonD2Ev : !llvm.ptr
    %26 = llvm.mlir.addressof @_ZTIN6google8protobuf7MessageE : !llvm.ptr
    %27 = llvm.mlir.constant("N8tutorial6PersonE\00") : !llvm.array<19 x i8>
    %28 = llvm.mlir.addressof @_ZTSN8tutorial6PersonE : !llvm.ptr
    %29 = llvm.mlir.addressof @_ZTVN10__cxxabiv120__si_class_type_infoE : !llvm.ptr
    %30 = llvm.getelementptr inbounds %29[%3] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.ptr
    %31 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, ptr)>
    %32 = llvm.insertvalue %30, %31[0] : !llvm.struct<(ptr, ptr, ptr)> 
    %33 = llvm.insertvalue %28, %32[1] : !llvm.struct<(ptr, ptr, ptr)> 
    %34 = llvm.insertvalue %26, %33[2] : !llvm.struct<(ptr, ptr, ptr)> 
    %35 = llvm.mlir.addressof @_ZTIN8tutorial6PersonE : !llvm.ptr
    %36 = llvm.mlir.undef : !llvm.array<22 x ptr>
    %37 = llvm.insertvalue %5, %36[0] : !llvm.array<22 x ptr> 
    %38 = llvm.insertvalue %35, %37[1] : !llvm.array<22 x ptr> 
    %39 = llvm.insertvalue %25, %38[2] : !llvm.array<22 x ptr> 
    %40 = llvm.insertvalue %24, %39[3] : !llvm.array<22 x ptr> 
    %41 = llvm.insertvalue %23, %40[4] : !llvm.array<22 x ptr> 
    %42 = llvm.insertvalue %22, %41[5] : !llvm.array<22 x ptr> 
    %43 = llvm.insertvalue %21, %42[6] : !llvm.array<22 x ptr> 
    %44 = llvm.insertvalue %20, %43[7] : !llvm.array<22 x ptr> 
    %45 = llvm.insertvalue %19, %44[8] : !llvm.array<22 x ptr> 
    %46 = llvm.insertvalue %18, %45[9] : !llvm.array<22 x ptr> 
    %47 = llvm.insertvalue %17, %46[10] : !llvm.array<22 x ptr> 
    %48 = llvm.insertvalue %16, %47[11] : !llvm.array<22 x ptr> 
    %49 = llvm.insertvalue %15, %48[12] : !llvm.array<22 x ptr> 
    %50 = llvm.insertvalue %14, %49[13] : !llvm.array<22 x ptr> 
    %51 = llvm.insertvalue %13, %50[14] : !llvm.array<22 x ptr> 
    %52 = llvm.insertvalue %12, %51[15] : !llvm.array<22 x ptr> 
    %53 = llvm.insertvalue %11, %52[16] : !llvm.array<22 x ptr> 
    %54 = llvm.insertvalue %10, %53[17] : !llvm.array<22 x ptr> 
    %55 = llvm.insertvalue %9, %54[18] : !llvm.array<22 x ptr> 
    %56 = llvm.insertvalue %8, %55[19] : !llvm.array<22 x ptr> 
    %57 = llvm.insertvalue %7, %56[20] : !llvm.array<22 x ptr> 
    %58 = llvm.insertvalue %4, %57[21] : !llvm.array<22 x ptr> 
    %59 = llvm.mlir.undef : !llvm.struct<(array<22 x ptr>)>
    %60 = llvm.insertvalue %58, %59[0] : !llvm.struct<(array<22 x ptr>)> 
    %61 = llvm.mlir.addressof @_ZTVN8tutorial6PersonE : !llvm.ptr
    %62 = llvm.getelementptr inbounds %61[%0, 0, %3] : (!llvm.ptr, i64, i64) -> !llvm.ptr, !llvm.struct<(array<22 x ptr>)>
    %63 = llvm.mlir.constant(0 : i8) : i8
    %64 = llvm.mlir.constant(16 : i64) : i64
    %65 = llvm.mlir.constant(6 : i32) : i32
    %66 = llvm.mlir.addressof @scc_info_Timestamp_google_2fprotobuf_2ftimestamp_2eproto : !llvm.ptr
    %67 = llvm.mlir.undef : !llvm.array<0 x ptr>
    %68 = llvm.mlir.addressof @_ZL59InitDefaultsscc_info_Person_PhoneNumber_addressbook_2eprotov : !llvm.ptr
    %69 = llvm.mlir.constant(-1 : i32) : i32
    %70 = llvm.mlir.undef : !llvm.struct<(i32)>
    %71 = llvm.insertvalue %69, %70[0] : !llvm.struct<(i32)> 
    %72 = llvm.mlir.undef : !llvm.struct<(struct<(i32)>, i32, i32, ptr)>
    %73 = llvm.insertvalue %71, %72[0] : !llvm.struct<(struct<(i32)>, i32, i32, ptr)> 
    %74 = llvm.insertvalue %1, %73[1] : !llvm.struct<(struct<(i32)>, i32, i32, ptr)> 
    %75 = llvm.insertvalue %1, %74[2] : !llvm.struct<(struct<(i32)>, i32, i32, ptr)> 
    %76 = llvm.insertvalue %68, %75[3] : !llvm.struct<(struct<(i32)>, i32, i32, ptr)> 
    %77 = llvm.mlir.undef : !llvm.struct<(struct<(struct<(i32)>, i32, i32, ptr)>, array<0 x ptr>)>
    %78 = llvm.insertvalue %76, %77[0] : !llvm.struct<(struct<(struct<(i32)>, i32, i32, ptr)>, array<0 x ptr>)> 
    %79 = llvm.insertvalue %67, %78[1] : !llvm.struct<(struct<(struct<(i32)>, i32, i32, ptr)>, array<0 x ptr>)> 
    %80 = llvm.mlir.addressof @scc_info_Person_PhoneNumber_addressbook_2eproto : !llvm.ptr
    %81 = llvm.mlir.undef : !llvm.array<2 x ptr>
    %82 = llvm.insertvalue %80, %81[0] : !llvm.array<2 x ptr> 
    %83 = llvm.insertvalue %66, %82[1] : !llvm.array<2 x ptr> 
    %84 = llvm.mlir.addressof @_ZL47InitDefaultsscc_info_Person_addressbook_2eprotov : !llvm.ptr
    %85 = llvm.mlir.constant(2 : i32) : i32
    %86 = llvm.mlir.undef : !llvm.struct<(struct<(i32)>, i32, i32, ptr)>
    %87 = llvm.insertvalue %71, %86[0] : !llvm.struct<(struct<(i32)>, i32, i32, ptr)> 
    %88 = llvm.insertvalue %85, %87[1] : !llvm.struct<(struct<(i32)>, i32, i32, ptr)> 
    %89 = llvm.insertvalue %1, %88[2] : !llvm.struct<(struct<(i32)>, i32, i32, ptr)> 
    %90 = llvm.insertvalue %84, %89[3] : !llvm.struct<(struct<(i32)>, i32, i32, ptr)> 
    %91 = llvm.mlir.undef : !llvm.struct<(struct<(struct<(i32)>, i32, i32, ptr)>, array<2 x ptr>)>
    %92 = llvm.insertvalue %90, %91[0] : !llvm.struct<(struct<(struct<(i32)>, i32, i32, ptr)>, array<2 x ptr>)> 
    %93 = llvm.insertvalue %83, %92[1] : !llvm.struct<(struct<(struct<(i32)>, i32, i32, ptr)>, array<2 x ptr>)> 
    %94 = llvm.mlir.addressof @scc_info_Person_addressbook_2eproto : !llvm.ptr
    %95 = llvm.mlir.addressof @_ZN6google8protobuf8internal26fixed_address_empty_stringB5cxx11E : !llvm.ptr
    %96 = llvm.mlir.constant(3 : i32) : i32
    %97 = llvm.mlir.constant(4 : i32) : i32
    %98 = llvm.mlir.constant(12 : i64) : i64
    %99 = llvm.getelementptr inbounds %arg0[%0, 0, 0, 1] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.tutorial::Person", (struct<"class.google::protobuf::Message", (struct<"class.google::protobuf::MessageLite", (ptr, struct<"class.google::protobuf::internal::InternalMetadata", (ptr)>)>)>, struct<"class.google::protobuf::RepeatedPtrField", (struct<"class.google::protobuf::internal::RepeatedPtrFieldBase", (ptr, i32, i32, ptr)>)>, struct<"struct.google::protobuf::internal::ArenaStringPtr", (ptr)>, struct<"struct.google::protobuf::internal::ArenaStringPtr", (ptr)>, ptr, i32, struct<"class.google::protobuf::internal::CachedSize", (struct<"struct.std::atomic", (struct<"struct.std::__atomic_base", (i32)>)>)>)>
    %100 = llvm.bitcast %99 : !llvm.ptr to !llvm.ptr
    llvm.store %arg1, %100 {alignment = 8 : i64, tbaa = [#tbaa_tag5]} : !llvm.ptr, !llvm.ptr
    %101 = llvm.getelementptr inbounds %arg0[%0, 0, 0, 0] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.tutorial::Person", (struct<"class.google::protobuf::Message", (struct<"class.google::protobuf::MessageLite", (ptr, struct<"class.google::protobuf::internal::InternalMetadata", (ptr)>)>)>, struct<"class.google::protobuf::RepeatedPtrField", (struct<"class.google::protobuf::internal::RepeatedPtrFieldBase", (ptr, i32, i32, ptr)>)>, struct<"struct.google::protobuf::internal::ArenaStringPtr", (ptr)>, struct<"struct.google::protobuf::internal::ArenaStringPtr", (ptr)>, ptr, i32, struct<"class.google::protobuf::internal::CachedSize", (struct<"struct.std::atomic", (struct<"struct.std::__atomic_base", (i32)>)>)>)>
    llvm.store %62, %101 {alignment = 8 : i64, tbaa = [#tbaa_tag]} : !llvm.ptr, !llvm.ptr
    %102 = llvm.getelementptr inbounds %arg0[%0, 1] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.tutorial::Person", (struct<"class.google::protobuf::Message", (struct<"class.google::protobuf::MessageLite", (ptr, struct<"class.google::protobuf::internal::InternalMetadata", (ptr)>)>)>, struct<"class.google::protobuf::RepeatedPtrField", (struct<"class.google::protobuf::internal::RepeatedPtrFieldBase", (ptr, i32, i32, ptr)>)>, struct<"struct.google::protobuf::internal::ArenaStringPtr", (ptr)>, struct<"struct.google::protobuf::internal::ArenaStringPtr", (ptr)>, ptr, i32, struct<"class.google::protobuf::internal::CachedSize", (struct<"struct.std::atomic", (struct<"struct.std::__atomic_base", (i32)>)>)>)>
    %103 = llvm.getelementptr inbounds %102[%0, 0, 0] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.google::protobuf::RepeatedPtrField", (struct<"class.google::protobuf::internal::RepeatedPtrFieldBase", (ptr, i32, i32, ptr)>)>
    llvm.store %arg1, %103 {alignment = 8 : i64, tbaa = [#tbaa_tag7]} : !llvm.ptr, !llvm.ptr
    %104 = llvm.getelementptr inbounds %arg0[%0, 1, 0, 1] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.tutorial::Person", (struct<"class.google::protobuf::Message", (struct<"class.google::protobuf::MessageLite", (ptr, struct<"class.google::protobuf::internal::InternalMetadata", (ptr)>)>)>, struct<"class.google::protobuf::RepeatedPtrField", (struct<"class.google::protobuf::internal::RepeatedPtrFieldBase", (ptr, i32, i32, ptr)>)>, struct<"struct.google::protobuf::internal::ArenaStringPtr", (ptr)>, struct<"struct.google::protobuf::internal::ArenaStringPtr", (ptr)>, ptr, i32, struct<"class.google::protobuf::internal::CachedSize", (struct<"struct.std::atomic", (struct<"struct.std::__atomic_base", (i32)>)>)>)>
    %105 = llvm.bitcast %104 : !llvm.ptr to !llvm.ptr
    "llvm.intr.memset"(%105, %63, %64) <{isVolatile = false}> : (!llvm.ptr, i8, i64) -> ()
    %106 = llvm.getelementptr inbounds %arg0[%0, 6, 0, 0, 0] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.tutorial::Person", (struct<"class.google::protobuf::Message", (struct<"class.google::protobuf::MessageLite", (ptr, struct<"class.google::protobuf::internal::InternalMetadata", (ptr)>)>)>, struct<"class.google::protobuf::RepeatedPtrField", (struct<"class.google::protobuf::internal::RepeatedPtrFieldBase", (ptr, i32, i32, ptr)>)>, struct<"struct.google::protobuf::internal::ArenaStringPtr", (ptr)>, struct<"struct.google::protobuf::internal::ArenaStringPtr", (ptr)>, ptr, i32, struct<"class.google::protobuf::internal::CachedSize", (struct<"struct.std::atomic", (struct<"struct.std::__atomic_base", (i32)>)>)>)>
    llvm.store %1, %106 {alignment = 4 : i64, tbaa = [#tbaa_tag6]} : i32, !llvm.ptr
    %107 = llvm.load %94 atomic acquire {alignment = 8 : i64} : !llvm.ptr -> i32
    %108 = llvm.icmp "eq" %107, %1 : i32
    llvm.cond_br %108 weights([2000, 1]), ^bb2, ^bb1
  ^bb1:  // pred: ^bb0
    llvm.invoke @_ZN6google8protobuf8internal11InitSCCImplEPNS1_11SCCInfoBaseE(%94) to ^bb2 unwind ^bb3 : (!llvm.ptr) -> ()
  ^bb2:  // 2 preds: ^bb0, ^bb1
    %109 = llvm.getelementptr inbounds %arg0[%0, 2, 0] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.tutorial::Person", (struct<"class.google::protobuf::Message", (struct<"class.google::protobuf::MessageLite", (ptr, struct<"class.google::protobuf::internal::InternalMetadata", (ptr)>)>)>, struct<"class.google::protobuf::RepeatedPtrField", (struct<"class.google::protobuf::internal::RepeatedPtrFieldBase", (ptr, i32, i32, ptr)>)>, struct<"struct.google::protobuf::internal::ArenaStringPtr", (ptr)>, struct<"struct.google::protobuf::internal::ArenaStringPtr", (ptr)>, ptr, i32, struct<"class.google::protobuf::internal::CachedSize", (struct<"struct.std::atomic", (struct<"struct.std::__atomic_base", (i32)>)>)>)>
    llvm.store %95, %109 {alignment = 8 : i64, tbaa = [#tbaa_tag8]} : !llvm.ptr, !llvm.ptr
    %110 = llvm.getelementptr inbounds %arg0[%0, 3, 0] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.tutorial::Person", (struct<"class.google::protobuf::Message", (struct<"class.google::protobuf::MessageLite", (ptr, struct<"class.google::protobuf::internal::InternalMetadata", (ptr)>)>)>, struct<"class.google::protobuf::RepeatedPtrField", (struct<"class.google::protobuf::internal::RepeatedPtrFieldBase", (ptr, i32, i32, ptr)>)>, struct<"struct.google::protobuf::internal::ArenaStringPtr", (ptr)>, struct<"struct.google::protobuf::internal::ArenaStringPtr", (ptr)>, ptr, i32, struct<"class.google::protobuf::internal::CachedSize", (struct<"struct.std::atomic", (struct<"struct.std::__atomic_base", (i32)>)>)>)>
    llvm.store %95, %110 {alignment = 8 : i64, tbaa = [#tbaa_tag8]} : !llvm.ptr, !llvm.ptr
    %111 = llvm.getelementptr inbounds %arg0[%0, 4] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.tutorial::Person", (struct<"class.google::protobuf::Message", (struct<"class.google::protobuf::MessageLite", (ptr, struct<"class.google::protobuf::internal::InternalMetadata", (ptr)>)>)>, struct<"class.google::protobuf::RepeatedPtrField", (struct<"class.google::protobuf::internal::RepeatedPtrFieldBase", (ptr, i32, i32, ptr)>)>, struct<"struct.google::protobuf::internal::ArenaStringPtr", (ptr)>, struct<"struct.google::protobuf::internal::ArenaStringPtr", (ptr)>, ptr, i32, struct<"class.google::protobuf::internal::CachedSize", (struct<"struct.std::atomic", (struct<"struct.std::__atomic_base", (i32)>)>)>)>
    %112 = llvm.bitcast %111 : !llvm.ptr to !llvm.ptr
    "llvm.intr.memset"(%112, %63, %98) <{isVolatile = false}> : (!llvm.ptr, i8, i64) -> ()
    llvm.return
  ^bb3:  // pred: ^bb1
    %113 = llvm.landingpad cleanup : !llvm.struct<(ptr, i32)>
    llvm.call @_ZN6google8protobuf16RepeatedPtrFieldIN8tutorial18Person_PhoneNumberEED2Ev(%102) : (!llvm.ptr) -> ()
    llvm.resume %113 : !llvm.struct<(ptr, i32)>
  }
  llvm.func linkonce_odr unnamed_addr @_ZN6google8protobuf16RepeatedPtrFieldIN8tutorial18Person_PhoneNumberEED2Ev(%arg0: !llvm.ptr {llvm.align = 8 : i64, llvm.dereferenceable = 24 : i64, llvm.nonnull, llvm.noundef}) comdat(@__llvm_global_comdat::@_ZN6google8protobuf16RepeatedPtrFieldIN8tutorial18Person_PhoneNumberEED2Ev) attributes {alignment = 2 : i64, frame_pointer = #llvm.framePointerKind<none>, passthrough = ["nounwind", ["uwtable", "2"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], ["tune-cpu", "generic"]], personality = @__gxx_personality_v0, target_cpu = "x86-64", target_features = #llvm.target_features<["+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>} {
    %0 = llvm.mlir.constant(0 : i64) : i64
    %1 = llvm.mlir.constant(0 : i32) : i32
    %2 = llvm.mlir.zero : !llvm.ptr
    %3 = llvm.getelementptr inbounds %arg0[%0, 0] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.google::protobuf::RepeatedPtrField", (struct<"class.google::protobuf::internal::RepeatedPtrFieldBase", (ptr, i32, i32, ptr)>)>
    llvm.invoke @_ZN6google8protobuf8internal20RepeatedPtrFieldBase7DestroyINS0_16RepeatedPtrFieldIN8tutorial18Person_PhoneNumberEE11TypeHandlerEEEvv(%3) to ^bb1 unwind ^bb5 : (!llvm.ptr) -> ()
  ^bb1:  // pred: ^bb0
    %4 = llvm.getelementptr inbounds %arg0[%0, 0, 0] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.google::protobuf::RepeatedPtrField", (struct<"class.google::protobuf::internal::RepeatedPtrFieldBase", (ptr, i32, i32, ptr)>)>
    %5 = llvm.load %4 {alignment = 8 : i64, tbaa = [#tbaa_tag7]} : !llvm.ptr -> !llvm.ptr
    %6 = llvm.icmp "eq" %5, %2 : !llvm.ptr
    llvm.cond_br %6, ^bb4, ^bb2
  ^bb2:  // pred: ^bb1
    %7 = llvm.getelementptr inbounds %5[%0, 0] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.google::protobuf::Arena", (struct<"class.google::protobuf::internal::ArenaImpl", (struct<"struct.std::atomic.10", (struct<"struct.std::__atomic_base.11", (ptr)>)>, struct<"struct.std::atomic.10", (struct<"struct.std::__atomic_base.11", (ptr)>)>, struct<"struct.std::atomic.12", (struct<"struct.std::__atomic_base.13", (i64)>)>, ptr, i64, struct<"struct.google::protobuf::internal::ArenaImpl::Options", (i64, i64, ptr, i64, ptr, ptr)>)>, ptr, ptr, ptr, ptr)>
    %8 = llvm.invoke @_ZNK6google8protobuf8internal9ArenaImpl14SpaceAllocatedEv(%7) to ^bb4 unwind ^bb3 : (!llvm.ptr) -> i64
  ^bb3:  // pred: ^bb2
    %9 = llvm.landingpad (catch %2 : !llvm.ptr) : !llvm.struct<(ptr, i32)>
    %10 = llvm.extractvalue %9[0] : !llvm.struct<(ptr, i32)> 
    llvm.call @__clang_call_terminate(%10) : (!llvm.ptr) -> ()
    llvm.unreachable
  ^bb4:  // 2 preds: ^bb1, ^bb2
    llvm.return
  ^bb5:  // pred: ^bb0
    %11 = llvm.landingpad (catch %2 : !llvm.ptr) : !llvm.struct<(ptr, i32)>
    %12 = llvm.extractvalue %11[0] : !llvm.struct<(ptr, i32)> 
    llvm.call @_ZN6google8protobuf8internal20RepeatedPtrFieldBaseD2Ev(%3) : (!llvm.ptr) -> ()
    llvm.call @__clang_call_terminate(%12) : (!llvm.ptr) -> ()
    llvm.unreachable
  }
  llvm.func unnamed_addr @_ZN8tutorial6PersonC2ERKS0_(%arg0: !llvm.ptr {llvm.align = 8 : i64, llvm.dereferenceable = 72 : i64, llvm.nonnull, llvm.noundef}, %arg1: !llvm.ptr {llvm.align = 8 : i64, llvm.dereferenceable = 72 : i64, llvm.nonnull, llvm.noundef}) attributes {alignment = 2 : i64, frame_pointer = #llvm.framePointerKind<none>, passthrough = [["uwtable", "2"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], ["tune-cpu", "generic"]], personality = @__gxx_personality_v0, target_cpu = "x86-64", target_features = #llvm.target_features<["+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>} {
    %0 = llvm.mlir.constant(0 : i64) : i64
    %1 = llvm.mlir.constant(0 : i32) : i32
    %2 = llvm.mlir.constant(1 : i32) : i32
    %3 = llvm.mlir.zero : !llvm.ptr
    %4 = llvm.mlir.constant(2 : i64) : i64
    %5 = llvm.mlir.addressof @_ZNK8tutorial6Person11GetMetadataEv : !llvm.ptr
    %6 = llvm.mlir.addressof @__gxx_personality_v0 : !llvm.ptr
    %7 = llvm.mlir.addressof @_ZNK8tutorial6Person13SetCachedSizeEi : !llvm.ptr
    %8 = llvm.mlir.addressof @_ZNK6google8protobuf7Message13SpaceUsedLongEv : !llvm.ptr
    %9 = llvm.mlir.addressof @_ZN6google8protobuf7Message20DiscardUnknownFieldsEv : !llvm.ptr
    %10 = llvm.mlir.addressof @_ZN8tutorial6Person9MergeFromERKN6google8protobuf7MessageE : !llvm.ptr
    %11 = llvm.mlir.addressof @_ZN8tutorial6Person8CopyFromERKN6google8protobuf7MessageE : !llvm.ptr
    %12 = llvm.mlir.addressof @_ZNK6google8protobuf11MessageLite16InternalGetTableEv : !llvm.ptr
    %13 = llvm.mlir.addressof @_ZNK8tutorial6Person18_InternalSerializeEPhPN6google8protobuf2io19EpsCopyOutputStreamE : !llvm.ptr
    %14 = llvm.mlir.addressof @_ZN8tutorial6Person14_InternalParseEPKcPN6google8protobuf8internal12ParseContextE : !llvm.ptr
    %15 = llvm.mlir.addressof @_ZNK8tutorial6Person13GetCachedSizeEv : !llvm.ptr
    %16 = llvm.mlir.addressof @_ZNK8tutorial6Person12ByteSizeLongEv : !llvm.ptr
    %17 = llvm.mlir.addressof @_ZN6google8protobuf7Message21CheckTypeAndMergeFromERKNS0_11MessageLiteE : !llvm.ptr
    %18 = llvm.mlir.addressof @_ZNK6google8protobuf7Message25InitializationErrorStringB5cxx11Ev : !llvm.ptr
    %19 = llvm.mlir.addressof @_ZNK8tutorial6Person13IsInitializedEv : !llvm.ptr
    %20 = llvm.mlir.addressof @_ZN8tutorial6Person5ClearEv : !llvm.ptr
    %21 = llvm.mlir.addressof @_ZNK8tutorial6Person3NewEPN6google8protobuf5ArenaE : !llvm.ptr
    %22 = llvm.mlir.addressof @_ZNK8tutorial6Person3NewEv : !llvm.ptr
    %23 = llvm.mlir.addressof @_ZNK6google8protobuf7Message11GetTypeNameB5cxx11Ev : !llvm.ptr
    %24 = llvm.mlir.addressof @_ZN8tutorial6PersonD0Ev : !llvm.ptr
    %25 = llvm.mlir.addressof @_ZN8tutorial6PersonD2Ev : !llvm.ptr
    %26 = llvm.mlir.addressof @_ZTIN6google8protobuf7MessageE : !llvm.ptr
    %27 = llvm.mlir.constant("N8tutorial6PersonE\00") : !llvm.array<19 x i8>
    %28 = llvm.mlir.addressof @_ZTSN8tutorial6PersonE : !llvm.ptr
    %29 = llvm.mlir.addressof @_ZTVN10__cxxabiv120__si_class_type_infoE : !llvm.ptr
    %30 = llvm.getelementptr inbounds %29[%4] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.ptr
    %31 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, ptr)>
    %32 = llvm.insertvalue %30, %31[0] : !llvm.struct<(ptr, ptr, ptr)> 
    %33 = llvm.insertvalue %28, %32[1] : !llvm.struct<(ptr, ptr, ptr)> 
    %34 = llvm.insertvalue %26, %33[2] : !llvm.struct<(ptr, ptr, ptr)> 
    %35 = llvm.mlir.addressof @_ZTIN8tutorial6PersonE : !llvm.ptr
    %36 = llvm.mlir.undef : !llvm.array<22 x ptr>
    %37 = llvm.insertvalue %3, %36[0] : !llvm.array<22 x ptr> 
    %38 = llvm.insertvalue %35, %37[1] : !llvm.array<22 x ptr> 
    %39 = llvm.insertvalue %25, %38[2] : !llvm.array<22 x ptr> 
    %40 = llvm.insertvalue %24, %39[3] : !llvm.array<22 x ptr> 
    %41 = llvm.insertvalue %23, %40[4] : !llvm.array<22 x ptr> 
    %42 = llvm.insertvalue %22, %41[5] : !llvm.array<22 x ptr> 
    %43 = llvm.insertvalue %21, %42[6] : !llvm.array<22 x ptr> 
    %44 = llvm.insertvalue %20, %43[7] : !llvm.array<22 x ptr> 
    %45 = llvm.insertvalue %19, %44[8] : !llvm.array<22 x ptr> 
    %46 = llvm.insertvalue %18, %45[9] : !llvm.array<22 x ptr> 
    %47 = llvm.insertvalue %17, %46[10] : !llvm.array<22 x ptr> 
    %48 = llvm.insertvalue %16, %47[11] : !llvm.array<22 x ptr> 
    %49 = llvm.insertvalue %15, %48[12] : !llvm.array<22 x ptr> 
    %50 = llvm.insertvalue %14, %49[13] : !llvm.array<22 x ptr> 
    %51 = llvm.insertvalue %13, %50[14] : !llvm.array<22 x ptr> 
    %52 = llvm.insertvalue %12, %51[15] : !llvm.array<22 x ptr> 
    %53 = llvm.insertvalue %11, %52[16] : !llvm.array<22 x ptr> 
    %54 = llvm.insertvalue %10, %53[17] : !llvm.array<22 x ptr> 
    %55 = llvm.insertvalue %9, %54[18] : !llvm.array<22 x ptr> 
    %56 = llvm.insertvalue %8, %55[19] : !llvm.array<22 x ptr> 
    %57 = llvm.insertvalue %7, %56[20] : !llvm.array<22 x ptr> 
    %58 = llvm.insertvalue %5, %57[21] : !llvm.array<22 x ptr> 
    %59 = llvm.mlir.undef : !llvm.struct<(array<22 x ptr>)>
    %60 = llvm.insertvalue %58, %59[0] : !llvm.struct<(array<22 x ptr>)> 
    %61 = llvm.mlir.addressof @_ZTVN8tutorial6PersonE : !llvm.ptr
    %62 = llvm.getelementptr inbounds %61[%0, 0, %4] : (!llvm.ptr, i64, i64) -> !llvm.ptr, !llvm.struct<(array<22 x ptr>)>
    %63 = llvm.mlir.constant(0 : i8) : i8
    %64 = llvm.mlir.constant(24 : i64) : i64
    %65 = llvm.mlir.constant(6 : i32) : i32
    %66 = llvm.mlir.constant(1 : i64) : i64
    %67 = llvm.mlir.constant(-2 : i64) : i64
    %68 = llvm.mlir.constant(2 : i32) : i32
    %69 = llvm.mlir.addressof @_ZN6google8protobuf8internal26fixed_address_empty_stringB5cxx11E : !llvm.ptr
    %70 = llvm.mlir.constant(3 : i32) : i32
    %71 = llvm.mlir.constant(dense<0> : tensor<64xi8>) : !llvm.array<64 x i8>
    %72 = llvm.mlir.undef : !llvm.struct<"union.google::protobuf::internal::ExplicitlyConstructed<tutorial::Person>::AlignedUnion", (i64, array<64 x i8>)>
    %73 = llvm.insertvalue %0, %72[0] : !llvm.struct<"union.google::protobuf::internal::ExplicitlyConstructed<tutorial::Person>::AlignedUnion", (i64, array<64 x i8>)> 
    %74 = llvm.insertvalue %71, %73[1] : !llvm.struct<"union.google::protobuf::internal::ExplicitlyConstructed<tutorial::Person>::AlignedUnion", (i64, array<64 x i8>)> 
    %75 = llvm.mlir.undef : !llvm.struct<"class.google::protobuf::internal::ExplicitlyConstructed.0", (struct<"union.google::protobuf::internal::ExplicitlyConstructed<tutorial::Person>::AlignedUnion", (i64, array<64 x i8>)>)>
    %76 = llvm.insertvalue %74, %75[0] : !llvm.struct<"class.google::protobuf::internal::ExplicitlyConstructed.0", (struct<"union.google::protobuf::internal::ExplicitlyConstructed<tutorial::Person>::AlignedUnion", (i64, array<64 x i8>)>)> 
    %77 = llvm.mlir.undef : !llvm.struct<"class.tutorial::PersonDefaultTypeInternal", (struct<"class.google::protobuf::internal::ExplicitlyConstructed.0", (struct<"union.google::protobuf::internal::ExplicitlyConstructed<tutorial::Person>::AlignedUnion", (i64, array<64 x i8>)>)>)>
    %78 = llvm.insertvalue %76, %77[0] : !llvm.struct<"class.tutorial::PersonDefaultTypeInternal", (struct<"class.google::protobuf::internal::ExplicitlyConstructed.0", (struct<"union.google::protobuf::internal::ExplicitlyConstructed<tutorial::Person>::AlignedUnion", (i64, array<64 x i8>)>)>)> 
    %79 = llvm.mlir.addressof @_ZN8tutorial25_Person_default_instance_E : !llvm.ptr
    %80 = llvm.mlir.constant(4 : i32) : i32
    %81 = llvm.mlir.constant(false) : i1
    %82 = llvm.mlir.constant(32 : i64) : i64
    %83 = llvm.mlir.constant(5 : i32) : i32
    %84 = llvm.getelementptr inbounds %arg0[%0, 0, 0, 1, 0] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.tutorial::Person", (struct<"class.google::protobuf::Message", (struct<"class.google::protobuf::MessageLite", (ptr, struct<"class.google::protobuf::internal::InternalMetadata", (ptr)>)>)>, struct<"class.google::protobuf::RepeatedPtrField", (struct<"class.google::protobuf::internal::RepeatedPtrFieldBase", (ptr, i32, i32, ptr)>)>, struct<"struct.google::protobuf::internal::ArenaStringPtr", (ptr)>, struct<"struct.google::protobuf::internal::ArenaStringPtr", (ptr)>, ptr, i32, struct<"class.google::protobuf::internal::CachedSize", (struct<"struct.std::atomic", (struct<"struct.std::__atomic_base", (i32)>)>)>)>
    llvm.store %3, %84 {alignment = 8 : i64, tbaa = [#tbaa_tag5]} : !llvm.ptr, !llvm.ptr
    %85 = llvm.getelementptr inbounds %arg0[%0, 0, 0, 0] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.tutorial::Person", (struct<"class.google::protobuf::Message", (struct<"class.google::protobuf::MessageLite", (ptr, struct<"class.google::protobuf::internal::InternalMetadata", (ptr)>)>)>, struct<"class.google::protobuf::RepeatedPtrField", (struct<"class.google::protobuf::internal::RepeatedPtrFieldBase", (ptr, i32, i32, ptr)>)>, struct<"struct.google::protobuf::internal::ArenaStringPtr", (ptr)>, struct<"struct.google::protobuf::internal::ArenaStringPtr", (ptr)>, ptr, i32, struct<"class.google::protobuf::internal::CachedSize", (struct<"struct.std::atomic", (struct<"struct.std::__atomic_base", (i32)>)>)>)>
    llvm.store %62, %85 {alignment = 8 : i64, tbaa = [#tbaa_tag]} : !llvm.ptr, !llvm.ptr
    %86 = llvm.getelementptr inbounds %arg0[%0, 1] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.tutorial::Person", (struct<"class.google::protobuf::Message", (struct<"class.google::protobuf::MessageLite", (ptr, struct<"class.google::protobuf::internal::InternalMetadata", (ptr)>)>)>, struct<"class.google::protobuf::RepeatedPtrField", (struct<"class.google::protobuf::internal::RepeatedPtrFieldBase", (ptr, i32, i32, ptr)>)>, struct<"struct.google::protobuf::internal::ArenaStringPtr", (ptr)>, struct<"struct.google::protobuf::internal::ArenaStringPtr", (ptr)>, ptr, i32, struct<"class.google::protobuf::internal::CachedSize", (struct<"struct.std::atomic", (struct<"struct.std::__atomic_base", (i32)>)>)>)>
    %87 = llvm.getelementptr inbounds %86[%0, 0] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.google::protobuf::RepeatedPtrField", (struct<"class.google::protobuf::internal::RepeatedPtrFieldBase", (ptr, i32, i32, ptr)>)>
    %88 = llvm.bitcast %86 : !llvm.ptr to !llvm.ptr
    "llvm.intr.memset"(%88, %63, %64) <{isVolatile = false}> : (!llvm.ptr, i8, i64) -> ()
    %89 = llvm.getelementptr inbounds %arg1[%0, 1, 0] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.tutorial::Person", (struct<"class.google::protobuf::Message", (struct<"class.google::protobuf::MessageLite", (ptr, struct<"class.google::protobuf::internal::InternalMetadata", (ptr)>)>)>, struct<"class.google::protobuf::RepeatedPtrField", (struct<"class.google::protobuf::internal::RepeatedPtrFieldBase", (ptr, i32, i32, ptr)>)>, struct<"struct.google::protobuf::internal::ArenaStringPtr", (ptr)>, struct<"struct.google::protobuf::internal::ArenaStringPtr", (ptr)>, ptr, i32, struct<"class.google::protobuf::internal::CachedSize", (struct<"struct.std::atomic", (struct<"struct.std::__atomic_base", (i32)>)>)>)>
    llvm.invoke @_ZN6google8protobuf8internal20RepeatedPtrFieldBase9MergeFromINS0_16RepeatedPtrFieldIN8tutorial18Person_PhoneNumberEE11TypeHandlerEEEvRKS2_(%87, %89) to ^bb2 unwind ^bb1 : (!llvm.ptr, !llvm.ptr) -> ()
  ^bb1:  // pred: ^bb0
    %90 = llvm.landingpad cleanup : !llvm.struct<(ptr, i32)>
    llvm.call @_ZN6google8protobuf8internal20RepeatedPtrFieldBaseD2Ev(%87) : (!llvm.ptr) -> ()
    llvm.br ^bb27(%90 : !llvm.struct<(ptr, i32)>)
  ^bb2:  // pred: ^bb0
    %91 = llvm.getelementptr inbounds %arg0[%0, 6, 0, 0, 0] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.tutorial::Person", (struct<"class.google::protobuf::Message", (struct<"class.google::protobuf::MessageLite", (ptr, struct<"class.google::protobuf::internal::InternalMetadata", (ptr)>)>)>, struct<"class.google::protobuf::RepeatedPtrField", (struct<"class.google::protobuf::internal::RepeatedPtrFieldBase", (ptr, i32, i32, ptr)>)>, struct<"struct.google::protobuf::internal::ArenaStringPtr", (ptr)>, struct<"struct.google::protobuf::internal::ArenaStringPtr", (ptr)>, ptr, i32, struct<"class.google::protobuf::internal::CachedSize", (struct<"struct.std::atomic", (struct<"struct.std::__atomic_base", (i32)>)>)>)>
    llvm.store %1, %91 {alignment = 4 : i64, tbaa = [#tbaa_tag6]} : i32, !llvm.ptr
    %92 = llvm.getelementptr inbounds %arg0[%0, 0, 0, 1] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.tutorial::Person", (struct<"class.google::protobuf::Message", (struct<"class.google::protobuf::MessageLite", (ptr, struct<"class.google::protobuf::internal::InternalMetadata", (ptr)>)>)>, struct<"class.google::protobuf::RepeatedPtrField", (struct<"class.google::protobuf::internal::RepeatedPtrFieldBase", (ptr, i32, i32, ptr)>)>, struct<"struct.google::protobuf::internal::ArenaStringPtr", (ptr)>, struct<"struct.google::protobuf::internal::ArenaStringPtr", (ptr)>, ptr, i32, struct<"class.google::protobuf::internal::CachedSize", (struct<"struct.std::atomic", (struct<"struct.std::__atomic_base", (i32)>)>)>)>
    %93 = llvm.getelementptr inbounds %arg1[%0, 0, 0, 1, 0] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.tutorial::Person", (struct<"class.google::protobuf::Message", (struct<"class.google::protobuf::MessageLite", (ptr, struct<"class.google::protobuf::internal::InternalMetadata", (ptr)>)>)>, struct<"class.google::protobuf::RepeatedPtrField", (struct<"class.google::protobuf::internal::RepeatedPtrFieldBase", (ptr, i32, i32, ptr)>)>, struct<"struct.google::protobuf::internal::ArenaStringPtr", (ptr)>, struct<"struct.google::protobuf::internal::ArenaStringPtr", (ptr)>, ptr, i32, struct<"class.google::protobuf::internal::CachedSize", (struct<"struct.std::atomic", (struct<"struct.std::__atomic_base", (i32)>)>)>)>
    %94 = llvm.load %93 {alignment = 8 : i64, tbaa = [#tbaa_tag5]} : !llvm.ptr -> !llvm.ptr
    %95 = llvm.ptrtoint %94 : !llvm.ptr to i64
    %96 = llvm.and %95, %66  : i64
    %97 = llvm.icmp "eq" %96, %0 : i64
    llvm.cond_br %97, ^bb8, ^bb3
  ^bb3:  // pred: ^bb2
    %98 = llvm.and %95, %67  : i64
    %99 = llvm.inttoptr %98 : i64 to !llvm.ptr
    %100 = llvm.getelementptr inbounds %99[%0, 1] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.google::protobuf::internal::InternalMetadata::Container", (struct<"struct.google::protobuf::internal::InternalMetadata::ContainerBase", (ptr)>, struct<"class.google::protobuf::UnknownFieldSet", (struct<"class.std::vector", (struct<"struct.std::_Vector_base", (struct<"struct.std::_Vector_base<google::protobuf::UnknownField, std::allocator<google::protobuf::UnknownField>>::_Vector_impl", (struct<"struct.std::_Vector_base<google::protobuf::UnknownField, std::allocator<google::protobuf::UnknownField>>::_Vector_impl_data", (ptr, ptr, ptr)>)>)>)>)>)>
    %101 = llvm.getelementptr inbounds %92[%0, 0] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.google::protobuf::internal::InternalMetadata", (ptr)>
    %102 = llvm.load %101 {alignment = 8 : i64, tbaa = [#tbaa_tag5]} : !llvm.ptr -> !llvm.ptr
    %103 = llvm.ptrtoint %102 : !llvm.ptr to i64
    %104 = llvm.and %103, %66  : i64
    %105 = llvm.icmp "eq" %104, %0 : i64
    llvm.cond_br %105 weights([1, 2000]), ^bb5, ^bb4
  ^bb4:  // pred: ^bb3
    %106 = llvm.and %103, %67  : i64
    %107 = llvm.inttoptr %106 : i64 to !llvm.ptr
    %108 = llvm.getelementptr inbounds %107[%0, 1] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.google::protobuf::internal::InternalMetadata::Container", (struct<"struct.google::protobuf::internal::InternalMetadata::ContainerBase", (ptr)>, struct<"class.google::protobuf::UnknownFieldSet", (struct<"class.std::vector", (struct<"struct.std::_Vector_base", (struct<"struct.std::_Vector_base<google::protobuf::UnknownField, std::allocator<google::protobuf::UnknownField>>::_Vector_impl", (struct<"struct.std::_Vector_base<google::protobuf::UnknownField, std::allocator<google::protobuf::UnknownField>>::_Vector_impl_data", (ptr, ptr, ptr)>)>)>)>)>)>
    llvm.br ^bb7(%108 : !llvm.ptr)
  ^bb5:  // pred: ^bb3
    %109 = llvm.invoke @_ZN6google8protobuf8internal16InternalMetadata27mutable_unknown_fields_slowINS0_15UnknownFieldSetEEEPT_v(%92) to ^bb6 unwind ^bb13 : (!llvm.ptr) -> !llvm.ptr
  ^bb6:  // pred: ^bb5
    llvm.br ^bb7(%109 : !llvm.ptr)
  ^bb7(%110: !llvm.ptr):  // 2 preds: ^bb4, ^bb6
    llvm.invoke @_ZN6google8protobuf15UnknownFieldSet9MergeFromERKS1_(%110, %100) to ^bb8 unwind ^bb13 : (!llvm.ptr, !llvm.ptr) -> ()
  ^bb8:  // 2 preds: ^bb2, ^bb7
    %111 = llvm.getelementptr inbounds %arg0[%0, 2] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.tutorial::Person", (struct<"class.google::protobuf::Message", (struct<"class.google::protobuf::MessageLite", (ptr, struct<"class.google::protobuf::internal::InternalMetadata", (ptr)>)>)>, struct<"class.google::protobuf::RepeatedPtrField", (struct<"class.google::protobuf::internal::RepeatedPtrFieldBase", (ptr, i32, i32, ptr)>)>, struct<"struct.google::protobuf::internal::ArenaStringPtr", (ptr)>, struct<"struct.google::protobuf::internal::ArenaStringPtr", (ptr)>, ptr, i32, struct<"class.google::protobuf::internal::CachedSize", (struct<"struct.std::atomic", (struct<"struct.std::__atomic_base", (i32)>)>)>)>
    %112 = llvm.getelementptr inbounds %111[%0, 0] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.google::protobuf::internal::ArenaStringPtr", (ptr)>
    llvm.store %69, %112 {alignment = 8 : i64, tbaa = [#tbaa_tag8]} : !llvm.ptr, !llvm.ptr
    %113 = llvm.getelementptr inbounds %arg1[%0, 2, 0] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.tutorial::Person", (struct<"class.google::protobuf::Message", (struct<"class.google::protobuf::MessageLite", (ptr, struct<"class.google::protobuf::internal::InternalMetadata", (ptr)>)>)>, struct<"class.google::protobuf::RepeatedPtrField", (struct<"class.google::protobuf::internal::RepeatedPtrFieldBase", (ptr, i32, i32, ptr)>)>, struct<"struct.google::protobuf::internal::ArenaStringPtr", (ptr)>, struct<"struct.google::protobuf::internal::ArenaStringPtr", (ptr)>, ptr, i32, struct<"class.google::protobuf::internal::CachedSize", (struct<"struct.std::atomic", (struct<"struct.std::__atomic_base", (i32)>)>)>)>
    %114 = llvm.load %113 {alignment = 8 : i64, tbaa = [#tbaa_tag8]} : !llvm.ptr -> !llvm.ptr
    %115 = llvm.getelementptr inbounds %114[%0, 1] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.std::__cxx11::basic_string", (struct<"struct.std::__cxx11::basic_string<char>::_Alloc_hider", (ptr)>, i64, struct<"union.anon", (i64, array<8 x i8>)>)>
    %116 = llvm.load %115 {alignment = 8 : i64, tbaa = [#tbaa_tag25]} : !llvm.ptr -> i64
    %117 = llvm.icmp "eq" %116, %0 : i64
    llvm.cond_br %117, ^bb14, ^bb9
  ^bb9:  // pred: ^bb8
    %118 = llvm.load %84 {alignment = 8 : i64, tbaa = [#tbaa_tag5]} : !llvm.ptr -> !llvm.ptr
    %119 = llvm.ptrtoint %118 : !llvm.ptr to i64
    %120 = llvm.and %119, %66  : i64
    %121 = llvm.icmp "eq" %120, %0 : i64
    %122 = llvm.and %119, %67  : i64
    llvm.cond_br %121 weights([2000, 1]), ^bb11, ^bb10
  ^bb10:  // pred: ^bb9
    %123 = llvm.inttoptr %122 : i64 to !llvm.ptr
    %124 = llvm.getelementptr inbounds %123[%0, 0] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.google::protobuf::internal::InternalMetadata::ContainerBase", (ptr)>
    %125 = llvm.load %124 {alignment = 8 : i64, tbaa = [#tbaa_tag9]} : !llvm.ptr -> !llvm.ptr
    llvm.br ^bb12(%125 : !llvm.ptr)
  ^bb11:  // pred: ^bb9
    %126 = llvm.inttoptr %122 : i64 to !llvm.ptr
    llvm.br ^bb12(%126 : !llvm.ptr)
  ^bb12(%127: !llvm.ptr):  // 2 preds: ^bb10, ^bb11
    llvm.invoke @_ZN6google8protobuf8internal14ArenaStringPtr14CreateInstanceEPNS0_5ArenaEPKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE(%111, %127, %114) to ^bb14 unwind ^bb13 : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
  ^bb13:  // 5 preds: ^bb5, ^bb7, ^bb12, ^bb18, ^bb20
    %128 = llvm.landingpad cleanup : !llvm.struct<(ptr, i32)>
    llvm.br ^bb26(%128 : !llvm.struct<(ptr, i32)>)
  ^bb14:  // 2 preds: ^bb8, ^bb12
    %129 = llvm.getelementptr inbounds %arg0[%0, 3] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.tutorial::Person", (struct<"class.google::protobuf::Message", (struct<"class.google::protobuf::MessageLite", (ptr, struct<"class.google::protobuf::internal::InternalMetadata", (ptr)>)>)>, struct<"class.google::protobuf::RepeatedPtrField", (struct<"class.google::protobuf::internal::RepeatedPtrFieldBase", (ptr, i32, i32, ptr)>)>, struct<"struct.google::protobuf::internal::ArenaStringPtr", (ptr)>, struct<"struct.google::protobuf::internal::ArenaStringPtr", (ptr)>, ptr, i32, struct<"class.google::protobuf::internal::CachedSize", (struct<"struct.std::atomic", (struct<"struct.std::__atomic_base", (i32)>)>)>)>
    %130 = llvm.getelementptr inbounds %129[%0, 0] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.google::protobuf::internal::ArenaStringPtr", (ptr)>
    llvm.store %69, %130 {alignment = 8 : i64, tbaa = [#tbaa_tag8]} : !llvm.ptr, !llvm.ptr
    %131 = llvm.getelementptr inbounds %arg1[%0, 3, 0] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.tutorial::Person", (struct<"class.google::protobuf::Message", (struct<"class.google::protobuf::MessageLite", (ptr, struct<"class.google::protobuf::internal::InternalMetadata", (ptr)>)>)>, struct<"class.google::protobuf::RepeatedPtrField", (struct<"class.google::protobuf::internal::RepeatedPtrFieldBase", (ptr, i32, i32, ptr)>)>, struct<"struct.google::protobuf::internal::ArenaStringPtr", (ptr)>, struct<"struct.google::protobuf::internal::ArenaStringPtr", (ptr)>, ptr, i32, struct<"class.google::protobuf::internal::CachedSize", (struct<"struct.std::atomic", (struct<"struct.std::__atomic_base", (i32)>)>)>)>
    %132 = llvm.load %131 {alignment = 8 : i64, tbaa = [#tbaa_tag8]} : !llvm.ptr -> !llvm.ptr
    %133 = llvm.getelementptr inbounds %132[%0, 1] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.std::__cxx11::basic_string", (struct<"struct.std::__cxx11::basic_string<char>::_Alloc_hider", (ptr)>, i64, struct<"union.anon", (i64, array<8 x i8>)>)>
    %134 = llvm.load %133 {alignment = 8 : i64, tbaa = [#tbaa_tag25]} : !llvm.ptr -> i64
    %135 = llvm.icmp "eq" %134, %0 : i64
    llvm.cond_br %135, ^bb19, ^bb15
  ^bb15:  // pred: ^bb14
    %136 = llvm.load %84 {alignment = 8 : i64, tbaa = [#tbaa_tag5]} : !llvm.ptr -> !llvm.ptr
    %137 = llvm.ptrtoint %136 : !llvm.ptr to i64
    %138 = llvm.and %137, %66  : i64
    %139 = llvm.icmp "eq" %138, %0 : i64
    %140 = llvm.and %137, %67  : i64
    llvm.cond_br %139 weights([2000, 1]), ^bb17, ^bb16
  ^bb16:  // pred: ^bb15
    %141 = llvm.inttoptr %140 : i64 to !llvm.ptr
    %142 = llvm.getelementptr inbounds %141[%0, 0] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.google::protobuf::internal::InternalMetadata::ContainerBase", (ptr)>
    %143 = llvm.load %142 {alignment = 8 : i64, tbaa = [#tbaa_tag9]} : !llvm.ptr -> !llvm.ptr
    llvm.br ^bb18(%143 : !llvm.ptr)
  ^bb17:  // pred: ^bb15
    %144 = llvm.inttoptr %140 : i64 to !llvm.ptr
    llvm.br ^bb18(%144 : !llvm.ptr)
  ^bb18(%145: !llvm.ptr):  // 2 preds: ^bb16, ^bb17
    llvm.invoke @_ZN6google8protobuf8internal14ArenaStringPtr14CreateInstanceEPNS0_5ArenaEPKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE(%129, %145, %132) to ^bb19 unwind ^bb13 : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
  ^bb19:  // 2 preds: ^bb14, ^bb18
    %146 = llvm.icmp "ne" %arg1, %79 : !llvm.ptr
    %147 = llvm.getelementptr inbounds %arg1[%0, 4] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.tutorial::Person", (struct<"class.google::protobuf::Message", (struct<"class.google::protobuf::MessageLite", (ptr, struct<"class.google::protobuf::internal::InternalMetadata", (ptr)>)>)>, struct<"class.google::protobuf::RepeatedPtrField", (struct<"class.google::protobuf::internal::RepeatedPtrFieldBase", (ptr, i32, i32, ptr)>)>, struct<"struct.google::protobuf::internal::ArenaStringPtr", (ptr)>, struct<"struct.google::protobuf::internal::ArenaStringPtr", (ptr)>, ptr, i32, struct<"class.google::protobuf::internal::CachedSize", (struct<"struct.std::atomic", (struct<"struct.std::__atomic_base", (i32)>)>)>)>
    %148 = llvm.load %147 {alignment = 8 : i64} : !llvm.ptr -> !llvm.ptr
    %149 = llvm.icmp "ne" %148, %3 : !llvm.ptr
    %150 = llvm.select %146, %149, %81 : i1, i1
    llvm.cond_br %150, ^bb20, ^bb24
  ^bb20:  // pred: ^bb19
    %151 = llvm.invoke @_Znwm(%82) to ^bb21 unwind ^bb13 : (i64) -> !llvm.ptr
  ^bb21:  // pred: ^bb20
    %152 = llvm.bitcast %151 : !llvm.ptr to !llvm.ptr
    llvm.invoke @_ZN6google8protobuf9TimestampC1ERKS1_(%152, %148) to ^bb22 unwind ^bb23 : (!llvm.ptr, !llvm.ptr) -> ()
  ^bb22:  // pred: ^bb21
    %153 = llvm.getelementptr inbounds %arg0[%0, 4] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.tutorial::Person", (struct<"class.google::protobuf::Message", (struct<"class.google::protobuf::MessageLite", (ptr, struct<"class.google::protobuf::internal::InternalMetadata", (ptr)>)>)>, struct<"class.google::protobuf::RepeatedPtrField", (struct<"class.google::protobuf::internal::RepeatedPtrFieldBase", (ptr, i32, i32, ptr)>)>, struct<"struct.google::protobuf::internal::ArenaStringPtr", (ptr)>, struct<"struct.google::protobuf::internal::ArenaStringPtr", (ptr)>, ptr, i32, struct<"class.google::protobuf::internal::CachedSize", (struct<"struct.std::atomic", (struct<"struct.std::__atomic_base", (i32)>)>)>)>
    %154 = llvm.bitcast %153 : !llvm.ptr to !llvm.ptr
    llvm.store %151, %154 {alignment = 8 : i64, tbaa = [#tbaa_tag23]} : !llvm.ptr, !llvm.ptr
    llvm.br ^bb25
  ^bb23:  // pred: ^bb21
    %155 = llvm.landingpad cleanup : !llvm.struct<(ptr, i32)>
    llvm.call @_ZdlPv(%151) : (!llvm.ptr) -> ()
    llvm.br ^bb26(%155 : !llvm.struct<(ptr, i32)>)
  ^bb24:  // pred: ^bb19
    %156 = llvm.getelementptr inbounds %arg0[%0, 4] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.tutorial::Person", (struct<"class.google::protobuf::Message", (struct<"class.google::protobuf::MessageLite", (ptr, struct<"class.google::protobuf::internal::InternalMetadata", (ptr)>)>)>, struct<"class.google::protobuf::RepeatedPtrField", (struct<"class.google::protobuf::internal::RepeatedPtrFieldBase", (ptr, i32, i32, ptr)>)>, struct<"struct.google::protobuf::internal::ArenaStringPtr", (ptr)>, struct<"struct.google::protobuf::internal::ArenaStringPtr", (ptr)>, ptr, i32, struct<"class.google::protobuf::internal::CachedSize", (struct<"struct.std::atomic", (struct<"struct.std::__atomic_base", (i32)>)>)>)>
    llvm.store %3, %156 {alignment = 8 : i64, tbaa = [#tbaa_tag23]} : !llvm.ptr, !llvm.ptr
    llvm.br ^bb25
  ^bb25:  // 2 preds: ^bb22, ^bb24
    %157 = llvm.getelementptr inbounds %arg1[%0, 5] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.tutorial::Person", (struct<"class.google::protobuf::Message", (struct<"class.google::protobuf::MessageLite", (ptr, struct<"class.google::protobuf::internal::InternalMetadata", (ptr)>)>)>, struct<"class.google::protobuf::RepeatedPtrField", (struct<"class.google::protobuf::internal::RepeatedPtrFieldBase", (ptr, i32, i32, ptr)>)>, struct<"struct.google::protobuf::internal::ArenaStringPtr", (ptr)>, struct<"struct.google::protobuf::internal::ArenaStringPtr", (ptr)>, ptr, i32, struct<"class.google::protobuf::internal::CachedSize", (struct<"struct.std::atomic", (struct<"struct.std::__atomic_base", (i32)>)>)>)>
    %158 = llvm.load %157 {alignment = 8 : i64, tbaa = [#tbaa_tag28]} : !llvm.ptr -> i32
    %159 = llvm.getelementptr inbounds %arg0[%0, 5] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.tutorial::Person", (struct<"class.google::protobuf::Message", (struct<"class.google::protobuf::MessageLite", (ptr, struct<"class.google::protobuf::internal::InternalMetadata", (ptr)>)>)>, struct<"class.google::protobuf::RepeatedPtrField", (struct<"class.google::protobuf::internal::RepeatedPtrFieldBase", (ptr, i32, i32, ptr)>)>, struct<"struct.google::protobuf::internal::ArenaStringPtr", (ptr)>, struct<"struct.google::protobuf::internal::ArenaStringPtr", (ptr)>, ptr, i32, struct<"class.google::protobuf::internal::CachedSize", (struct<"struct.std::atomic", (struct<"struct.std::__atomic_base", (i32)>)>)>)>
    llvm.store %158, %159 {alignment = 8 : i64, tbaa = [#tbaa_tag28]} : i32, !llvm.ptr
    llvm.return
  ^bb26(%160: !llvm.struct<(ptr, i32)>):  // 2 preds: ^bb13, ^bb23
    llvm.call @_ZN6google8protobuf16RepeatedPtrFieldIN8tutorial18Person_PhoneNumberEED2Ev(%86) : (!llvm.ptr) -> ()
    llvm.br ^bb27(%160 : !llvm.struct<(ptr, i32)>)
  ^bb27(%161: !llvm.struct<(ptr, i32)>):  // 2 preds: ^bb1, ^bb26
    llvm.resume %161 : !llvm.struct<(ptr, i32)>
  }
  llvm.func local_unnamed_addr @_Znwm(i64 {llvm.noundef}) -> (!llvm.ptr {llvm.nonnull, llvm.noundef}) attributes {frame_pointer = #llvm.framePointerKind<none>, passthrough = ["nobuiltin", ["allocsize", "4294967295"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], ["tune-cpu", "generic"]], target_cpu = "x86-64", target_features = #llvm.target_features<["+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>}
  llvm.func unnamed_addr @_ZN6google8protobuf9TimestampC1ERKS1_(!llvm.ptr {llvm.align = 8 : i64, llvm.dereferenceable = 32 : i64, llvm.nonnull, llvm.noundef}, !llvm.ptr {llvm.align = 8 : i64, llvm.dereferenceable = 32 : i64, llvm.nonnull, llvm.noundef}) attributes {frame_pointer = #llvm.framePointerKind<none>, passthrough = [["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], ["tune-cpu", "generic"]], target_cpu = "x86-64", target_features = #llvm.target_features<["+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>}
  llvm.func unnamed_addr @_ZN8tutorial6PersonD2Ev(%arg0: !llvm.ptr {llvm.align = 8 : i64, llvm.dereferenceable = 72 : i64, llvm.nonnull, llvm.noundef}) attributes {alignment = 2 : i64, frame_pointer = #llvm.framePointerKind<none>, passthrough = ["nounwind", ["uwtable", "2"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], ["tune-cpu", "generic"]], personality = @__gxx_personality_v0, target_cpu = "x86-64", target_features = #llvm.target_features<["+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>} {
    %0 = llvm.mlir.constant(0 : i64) : i64
    %1 = llvm.mlir.constant(0 : i32) : i32
    %2 = llvm.mlir.constant(1 : i32) : i32
    %3 = llvm.mlir.zero : !llvm.ptr
    llvm.invoke @_ZN8tutorial6Person10SharedDtorEv(%arg0) to ^bb1 unwind ^bb8 : (!llvm.ptr) -> ()
  ^bb1:  // pred: ^bb0
    %4 = llvm.getelementptr inbounds %arg0[%0, 0, 0, 1] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.tutorial::Person", (struct<"class.google::protobuf::Message", (struct<"class.google::protobuf::MessageLite", (ptr, struct<"class.google::protobuf::internal::InternalMetadata", (ptr)>)>)>, struct<"class.google::protobuf::RepeatedPtrField", (struct<"class.google::protobuf::internal::RepeatedPtrFieldBase", (ptr, i32, i32, ptr)>)>, struct<"struct.google::protobuf::internal::ArenaStringPtr", (ptr)>, struct<"struct.google::protobuf::internal::ArenaStringPtr", (ptr)>, ptr, i32, struct<"class.google::protobuf::internal::CachedSize", (struct<"struct.std::atomic", (struct<"struct.std::__atomic_base", (i32)>)>)>)>
    llvm.invoke @_ZN6google8protobuf8internal16InternalMetadata6DeleteINS0_15UnknownFieldSetEEEvv(%4) to ^bb2 unwind ^bb8 : (!llvm.ptr) -> ()
  ^bb2:  // pred: ^bb1
    %5 = llvm.getelementptr inbounds %arg0[%0, 1] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.tutorial::Person", (struct<"class.google::protobuf::Message", (struct<"class.google::protobuf::MessageLite", (ptr, struct<"class.google::protobuf::internal::InternalMetadata", (ptr)>)>)>, struct<"class.google::protobuf::RepeatedPtrField", (struct<"class.google::protobuf::internal::RepeatedPtrFieldBase", (ptr, i32, i32, ptr)>)>, struct<"struct.google::protobuf::internal::ArenaStringPtr", (ptr)>, struct<"struct.google::protobuf::internal::ArenaStringPtr", (ptr)>, ptr, i32, struct<"class.google::protobuf::internal::CachedSize", (struct<"struct.std::atomic", (struct<"struct.std::__atomic_base", (i32)>)>)>)>
    %6 = llvm.getelementptr inbounds %5[%0, 0] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.google::protobuf::RepeatedPtrField", (struct<"class.google::protobuf::internal::RepeatedPtrFieldBase", (ptr, i32, i32, ptr)>)>
    llvm.invoke @_ZN6google8protobuf8internal20RepeatedPtrFieldBase7DestroyINS0_16RepeatedPtrFieldIN8tutorial18Person_PhoneNumberEE11TypeHandlerEEEvv(%6) to ^bb3 unwind ^bb6 : (!llvm.ptr) -> ()
  ^bb3:  // pred: ^bb2
    %7 = llvm.getelementptr inbounds %5[%0, 0, 0] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.google::protobuf::RepeatedPtrField", (struct<"class.google::protobuf::internal::RepeatedPtrFieldBase", (ptr, i32, i32, ptr)>)>
    %8 = llvm.load %7 {alignment = 8 : i64, tbaa = [#tbaa_tag7]} : !llvm.ptr -> !llvm.ptr
    %9 = llvm.icmp "eq" %8, %3 : !llvm.ptr
    llvm.cond_br %9, ^bb7, ^bb4
  ^bb4:  // pred: ^bb3
    %10 = llvm.getelementptr inbounds %8[%0, 0] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.google::protobuf::Arena", (struct<"class.google::protobuf::internal::ArenaImpl", (struct<"struct.std::atomic.10", (struct<"struct.std::__atomic_base.11", (ptr)>)>, struct<"struct.std::atomic.10", (struct<"struct.std::__atomic_base.11", (ptr)>)>, struct<"struct.std::atomic.12", (struct<"struct.std::__atomic_base.13", (i64)>)>, ptr, i64, struct<"struct.google::protobuf::internal::ArenaImpl::Options", (i64, i64, ptr, i64, ptr, ptr)>)>, ptr, ptr, ptr, ptr)>
    %11 = llvm.invoke @_ZNK6google8protobuf8internal9ArenaImpl14SpaceAllocatedEv(%10) to ^bb7 unwind ^bb5 : (!llvm.ptr) -> i64
  ^bb5:  // pred: ^bb4
    %12 = llvm.landingpad (catch %3 : !llvm.ptr) : !llvm.struct<(ptr, i32)>
    %13 = llvm.extractvalue %12[0] : !llvm.struct<(ptr, i32)> 
    llvm.call @__clang_call_terminate(%13) : (!llvm.ptr) -> ()
    llvm.unreachable
  ^bb6:  // pred: ^bb2
    %14 = llvm.landingpad (catch %3 : !llvm.ptr) : !llvm.struct<(ptr, i32)>
    %15 = llvm.extractvalue %14[0] : !llvm.struct<(ptr, i32)> 
    llvm.call @_ZN6google8protobuf8internal20RepeatedPtrFieldBaseD2Ev(%6) : (!llvm.ptr) -> ()
    llvm.call @__clang_call_terminate(%15) : (!llvm.ptr) -> ()
    llvm.unreachable
  ^bb7:  // 2 preds: ^bb3, ^bb4
    llvm.return
  ^bb8:  // 2 preds: ^bb0, ^bb1
    %16 = llvm.landingpad (catch %3 : !llvm.ptr) : !llvm.struct<(ptr, i32)>
    %17 = llvm.extractvalue %16[0] : !llvm.struct<(ptr, i32)> 
    %18 = llvm.getelementptr inbounds %arg0[%0, 1] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.tutorial::Person", (struct<"class.google::protobuf::Message", (struct<"class.google::protobuf::MessageLite", (ptr, struct<"class.google::protobuf::internal::InternalMetadata", (ptr)>)>)>, struct<"class.google::protobuf::RepeatedPtrField", (struct<"class.google::protobuf::internal::RepeatedPtrFieldBase", (ptr, i32, i32, ptr)>)>, struct<"struct.google::protobuf::internal::ArenaStringPtr", (ptr)>, struct<"struct.google::protobuf::internal::ArenaStringPtr", (ptr)>, ptr, i32, struct<"class.google::protobuf::internal::CachedSize", (struct<"struct.std::atomic", (struct<"struct.std::__atomic_base", (i32)>)>)>)>
    llvm.call @_ZN6google8protobuf16RepeatedPtrFieldIN8tutorial18Person_PhoneNumberEED2Ev(%18) : (!llvm.ptr) -> ()
    llvm.call @__clang_call_terminate(%17) : (!llvm.ptr) -> ()
    llvm.unreachable
  }
  llvm.func linkonce_odr local_unnamed_addr @_ZN8tutorial6Person10SharedDtorEv(%arg0: !llvm.ptr {llvm.align = 8 : i64, llvm.dereferenceable = 72 : i64, llvm.nonnull, llvm.noundef}) comdat(@__llvm_global_comdat::@_ZN8tutorial6Person10SharedDtorEv) attributes {alignment = 2 : i64, frame_pointer = #llvm.framePointerKind<none>, passthrough = ["inlinehint", ["uwtable", "2"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], ["tune-cpu", "generic"]], personality = @__gxx_personality_v0, target_cpu = "x86-64", target_features = #llvm.target_features<["+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>} {
    %0 = llvm.mlir.constant(1 : i32) : i32
    %1 = llvm.mlir.constant(0 : i64) : i64
    %2 = llvm.mlir.constant(0 : i32) : i32
    %3 = llvm.mlir.constant(1 : i64) : i64
    %4 = llvm.mlir.constant(-2 : i64) : i64
    %5 = llvm.mlir.zero : !llvm.ptr
    %6 = llvm.mlir.constant(3 : i32) : i32
    %7 = llvm.mlir.constant("build/addressbook.pb.cc\00") : !llvm.array<24 x i8>
    %8 = llvm.mlir.addressof @".str.5" : !llvm.ptr
    %9 = llvm.mlir.constant(483 : i32) : i32
    %10 = llvm.mlir.constant("CHECK failed: GetArena() == nullptr: \00") : !llvm.array<38 x i8>
    %11 = llvm.mlir.addressof @".str.12" : !llvm.ptr
    %12 = llvm.mlir.constant(2 : i32) : i32
    %13 = llvm.mlir.addressof @_ZN6google8protobuf8internal26fixed_address_empty_stringB5cxx11E : !llvm.ptr
    %14 = llvm.mlir.constant(0 : i8) : i8
    %15 = llvm.mlir.constant(dense<0> : tensor<64xi8>) : !llvm.array<64 x i8>
    %16 = llvm.mlir.undef : !llvm.struct<"union.google::protobuf::internal::ExplicitlyConstructed<tutorial::Person>::AlignedUnion", (i64, array<64 x i8>)>
    %17 = llvm.insertvalue %1, %16[0] : !llvm.struct<"union.google::protobuf::internal::ExplicitlyConstructed<tutorial::Person>::AlignedUnion", (i64, array<64 x i8>)> 
    %18 = llvm.insertvalue %15, %17[1] : !llvm.struct<"union.google::protobuf::internal::ExplicitlyConstructed<tutorial::Person>::AlignedUnion", (i64, array<64 x i8>)> 
    %19 = llvm.mlir.undef : !llvm.struct<"class.google::protobuf::internal::ExplicitlyConstructed.0", (struct<"union.google::protobuf::internal::ExplicitlyConstructed<tutorial::Person>::AlignedUnion", (i64, array<64 x i8>)>)>
    %20 = llvm.insertvalue %18, %19[0] : !llvm.struct<"class.google::protobuf::internal::ExplicitlyConstructed.0", (struct<"union.google::protobuf::internal::ExplicitlyConstructed<tutorial::Person>::AlignedUnion", (i64, array<64 x i8>)>)> 
    %21 = llvm.mlir.undef : !llvm.struct<"class.tutorial::PersonDefaultTypeInternal", (struct<"class.google::protobuf::internal::ExplicitlyConstructed.0", (struct<"union.google::protobuf::internal::ExplicitlyConstructed<tutorial::Person>::AlignedUnion", (i64, array<64 x i8>)>)>)>
    %22 = llvm.insertvalue %20, %21[0] : !llvm.struct<"class.tutorial::PersonDefaultTypeInternal", (struct<"class.google::protobuf::internal::ExplicitlyConstructed.0", (struct<"union.google::protobuf::internal::ExplicitlyConstructed<tutorial::Person>::AlignedUnion", (i64, array<64 x i8>)>)>)> 
    %23 = llvm.mlir.addressof @_ZN8tutorial25_Person_default_instance_E : !llvm.ptr
    %24 = llvm.mlir.constant(4 : i32) : i32
    %25 = llvm.alloca %0 x !llvm.struct<"class.google::protobuf::internal::LogMessage", (i32, ptr, i32, struct<"class.std::__cxx11::basic_string", (struct<"struct.std::__cxx11::basic_string<char>::_Alloc_hider", (ptr)>, i64, struct<"union.anon", (i64, array<8 x i8>)>)>)> {alignment = 8 : i64} : (i32) -> !llvm.ptr
    %26 = llvm.alloca %0 x !llvm.struct<"class.google::protobuf::internal::LogFinisher", (i8)> {alignment = 1 : i64} : (i32) -> !llvm.ptr
    %27 = llvm.getelementptr inbounds %arg0[%1, 0, 0, 1, 0] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.tutorial::Person", (struct<"class.google::protobuf::Message", (struct<"class.google::protobuf::MessageLite", (ptr, struct<"class.google::protobuf::internal::InternalMetadata", (ptr)>)>)>, struct<"class.google::protobuf::RepeatedPtrField", (struct<"class.google::protobuf::internal::RepeatedPtrFieldBase", (ptr, i32, i32, ptr)>)>, struct<"struct.google::protobuf::internal::ArenaStringPtr", (ptr)>, struct<"struct.google::protobuf::internal::ArenaStringPtr", (ptr)>, ptr, i32, struct<"class.google::protobuf::internal::CachedSize", (struct<"struct.std::atomic", (struct<"struct.std::__atomic_base", (i32)>)>)>)>
    %28 = llvm.load %27 {alignment = 8 : i64, tbaa = [#tbaa_tag5]} : !llvm.ptr -> !llvm.ptr
    %29 = llvm.ptrtoint %28 : !llvm.ptr to i64
    %30 = llvm.and %29, %3  : i64
    %31 = llvm.icmp "eq" %30, %1 : i64
    %32 = llvm.and %29, %4  : i64
    llvm.cond_br %31 weights([2000, 1]), ^bb2, ^bb1
  ^bb1:  // pred: ^bb0
    %33 = llvm.inttoptr %32 : i64 to !llvm.ptr
    %34 = llvm.getelementptr inbounds %33[%1, 0] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.google::protobuf::internal::InternalMetadata::ContainerBase", (ptr)>
    %35 = llvm.load %34 {alignment = 8 : i64, tbaa = [#tbaa_tag9]} : !llvm.ptr -> !llvm.ptr
    llvm.br ^bb3(%35 : !llvm.ptr)
  ^bb2:  // pred: ^bb0
    %36 = llvm.inttoptr %32 : i64 to !llvm.ptr
    llvm.br ^bb3(%36 : !llvm.ptr)
  ^bb3(%37: !llvm.ptr):  // 2 preds: ^bb1, ^bb2
    %38 = llvm.icmp "eq" %37, %5 : !llvm.ptr
    %39 = llvm.getelementptr inbounds %26[%1, 0] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.google::protobuf::internal::LogFinisher", (i8)>
    llvm.intr.lifetime.start 1, %39 : !llvm.ptr
    llvm.cond_br %38, ^bb6, ^bb4
  ^bb4:  // pred: ^bb3
    %40 = llvm.bitcast %25 : !llvm.ptr to !llvm.ptr
    llvm.intr.lifetime.start 56, %40 : !llvm.ptr
    llvm.call @_ZN6google8protobuf8internal10LogMessageC1ENS0_8LogLevelEPKci(%25, %6, %8, %9) : (!llvm.ptr, i32, !llvm.ptr, i32) -> ()
    %41 = llvm.invoke @_ZN6google8protobuf8internal10LogMessagelsEPKc(%25, %11) to ^bb5 unwind ^bb19 : (!llvm.ptr, !llvm.ptr) -> !llvm.ptr
  ^bb5:  // pred: ^bb4
    llvm.invoke @_ZN6google8protobuf8internal11LogFinisheraSERNS1_10LogMessageE(%26, %41) to ^bb7 unwind ^bb20 : (!llvm.ptr, !llvm.ptr) -> ()
  ^bb6:  // pred: ^bb3
    llvm.intr.lifetime.end 1, %39 : !llvm.ptr
    llvm.br ^bb8
  ^bb7:  // pred: ^bb5
    llvm.intr.lifetime.end 1, %39 : !llvm.ptr
    llvm.call @_ZN6google8protobuf8internal10LogMessageD1Ev(%25) : (!llvm.ptr) -> ()
    llvm.intr.lifetime.end 56, %40 : !llvm.ptr
    llvm.br ^bb8
  ^bb8:  // 2 preds: ^bb6, ^bb7
    %42 = llvm.getelementptr inbounds %arg0[%1, 2, 0] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.tutorial::Person", (struct<"class.google::protobuf::Message", (struct<"class.google::protobuf::MessageLite", (ptr, struct<"class.google::protobuf::internal::InternalMetadata", (ptr)>)>)>, struct<"class.google::protobuf::RepeatedPtrField", (struct<"class.google::protobuf::internal::RepeatedPtrFieldBase", (ptr, i32, i32, ptr)>)>, struct<"struct.google::protobuf::internal::ArenaStringPtr", (ptr)>, struct<"struct.google::protobuf::internal::ArenaStringPtr", (ptr)>, ptr, i32, struct<"class.google::protobuf::internal::CachedSize", (struct<"struct.std::atomic", (struct<"struct.std::__atomic_base", (i32)>)>)>)>
    %43 = llvm.load %42 {alignment = 8 : i64, tbaa = [#tbaa_tag8]} : !llvm.ptr -> !llvm.ptr
    %44 = llvm.icmp "eq" %43, %13 : !llvm.ptr
    %45 = llvm.icmp "eq" %43, %5 : !llvm.ptr
    %46 = llvm.or %44, %45  : i1
    llvm.cond_br %46, ^bb12, ^bb9
  ^bb9:  // pred: ^bb8
    %47 = llvm.getelementptr inbounds %43[%1, 0, 0] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.std::__cxx11::basic_string", (struct<"struct.std::__cxx11::basic_string<char>::_Alloc_hider", (ptr)>, i64, struct<"union.anon", (i64, array<8 x i8>)>)>
    %48 = llvm.load %47 {alignment = 8 : i64, tbaa = [#tbaa_tag26]} : !llvm.ptr -> !llvm.ptr
    %49 = llvm.getelementptr inbounds %43[%1, 2] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.std::__cxx11::basic_string", (struct<"struct.std::__cxx11::basic_string<char>::_Alloc_hider", (ptr)>, i64, struct<"union.anon", (i64, array<8 x i8>)>)>
    %50 = llvm.bitcast %49 : !llvm.ptr to !llvm.ptr
    %51 = llvm.icmp "eq" %48, %50 : !llvm.ptr
    llvm.cond_br %51, ^bb11, ^bb10
  ^bb10:  // pred: ^bb9
    llvm.call @_ZdlPv(%48) : (!llvm.ptr) -> ()
    llvm.br ^bb11
  ^bb11:  // 2 preds: ^bb9, ^bb10
    %52 = llvm.bitcast %43 : !llvm.ptr to !llvm.ptr
    llvm.call @_ZdlPv(%52) : (!llvm.ptr) -> ()
    llvm.br ^bb12
  ^bb12:  // 2 preds: ^bb8, ^bb11
    %53 = llvm.getelementptr inbounds %arg0[%1, 3, 0] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.tutorial::Person", (struct<"class.google::protobuf::Message", (struct<"class.google::protobuf::MessageLite", (ptr, struct<"class.google::protobuf::internal::InternalMetadata", (ptr)>)>)>, struct<"class.google::protobuf::RepeatedPtrField", (struct<"class.google::protobuf::internal::RepeatedPtrFieldBase", (ptr, i32, i32, ptr)>)>, struct<"struct.google::protobuf::internal::ArenaStringPtr", (ptr)>, struct<"struct.google::protobuf::internal::ArenaStringPtr", (ptr)>, ptr, i32, struct<"class.google::protobuf::internal::CachedSize", (struct<"struct.std::atomic", (struct<"struct.std::__atomic_base", (i32)>)>)>)>
    %54 = llvm.load %53 {alignment = 8 : i64, tbaa = [#tbaa_tag8]} : !llvm.ptr -> !llvm.ptr
    %55 = llvm.icmp "eq" %54, %13 : !llvm.ptr
    %56 = llvm.icmp "eq" %54, %5 : !llvm.ptr
    %57 = llvm.or %55, %56  : i1
    llvm.cond_br %57, ^bb16, ^bb13
  ^bb13:  // pred: ^bb12
    %58 = llvm.getelementptr inbounds %54[%1, 0, 0] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.std::__cxx11::basic_string", (struct<"struct.std::__cxx11::basic_string<char>::_Alloc_hider", (ptr)>, i64, struct<"union.anon", (i64, array<8 x i8>)>)>
    %59 = llvm.load %58 {alignment = 8 : i64, tbaa = [#tbaa_tag26]} : !llvm.ptr -> !llvm.ptr
    %60 = llvm.getelementptr inbounds %54[%1, 2] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.std::__cxx11::basic_string", (struct<"struct.std::__cxx11::basic_string<char>::_Alloc_hider", (ptr)>, i64, struct<"union.anon", (i64, array<8 x i8>)>)>
    %61 = llvm.bitcast %60 : !llvm.ptr to !llvm.ptr
    %62 = llvm.icmp "eq" %59, %61 : !llvm.ptr
    llvm.cond_br %62, ^bb15, ^bb14
  ^bb14:  // pred: ^bb13
    llvm.call @_ZdlPv(%59) : (!llvm.ptr) -> ()
    llvm.br ^bb15
  ^bb15:  // 2 preds: ^bb13, ^bb14
    %63 = llvm.bitcast %54 : !llvm.ptr to !llvm.ptr
    llvm.call @_ZdlPv(%63) : (!llvm.ptr) -> ()
    llvm.br ^bb16
  ^bb16:  // 2 preds: ^bb12, ^bb15
    %64 = llvm.icmp "eq" %arg0, %23 : !llvm.ptr
    llvm.cond_br %64, ^bb22, ^bb17
  ^bb17:  // pred: ^bb16
    %65 = llvm.getelementptr inbounds %arg0[%1, 4] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.tutorial::Person", (struct<"class.google::protobuf::Message", (struct<"class.google::protobuf::MessageLite", (ptr, struct<"class.google::protobuf::internal::InternalMetadata", (ptr)>)>)>, struct<"class.google::protobuf::RepeatedPtrField", (struct<"class.google::protobuf::internal::RepeatedPtrFieldBase", (ptr, i32, i32, ptr)>)>, struct<"struct.google::protobuf::internal::ArenaStringPtr", (ptr)>, struct<"struct.google::protobuf::internal::ArenaStringPtr", (ptr)>, ptr, i32, struct<"class.google::protobuf::internal::CachedSize", (struct<"struct.std::atomic", (struct<"struct.std::__atomic_base", (i32)>)>)>)>
    %66 = llvm.load %65 {alignment = 8 : i64, tbaa = [#tbaa_tag23]} : !llvm.ptr -> !llvm.ptr
    %67 = llvm.icmp "eq" %66, %5 : !llvm.ptr
    llvm.cond_br %67, ^bb22, ^bb18
  ^bb18:  // pred: ^bb17
    llvm.call @_ZN6google8protobuf9TimestampD1Ev(%66) : (!llvm.ptr) -> ()
    %68 = llvm.bitcast %66 : !llvm.ptr to !llvm.ptr
    llvm.call @_ZdlPv(%68) : (!llvm.ptr) -> ()
    llvm.br ^bb22
  ^bb19:  // pred: ^bb4
    %69 = llvm.landingpad cleanup : !llvm.struct<(ptr, i32)>
    llvm.br ^bb21(%69 : !llvm.struct<(ptr, i32)>)
  ^bb20:  // pred: ^bb5
    %70 = llvm.landingpad cleanup : !llvm.struct<(ptr, i32)>
    llvm.intr.lifetime.end 1, %39 : !llvm.ptr
    llvm.br ^bb21(%70 : !llvm.struct<(ptr, i32)>)
  ^bb21(%71: !llvm.struct<(ptr, i32)>):  // 2 preds: ^bb19, ^bb20
    llvm.call @_ZN6google8protobuf8internal10LogMessageD1Ev(%25) : (!llvm.ptr) -> ()
    llvm.intr.lifetime.end 56, %40 : !llvm.ptr
    llvm.resume %71 : !llvm.struct<(ptr, i32)>
  ^bb22:  // 3 preds: ^bb16, ^bb17, ^bb18
    llvm.return
  }
  llvm.func unnamed_addr @_ZN8tutorial6PersonD0Ev(%arg0: !llvm.ptr {llvm.align = 8 : i64, llvm.dereferenceable = 72 : i64, llvm.nonnull, llvm.noundef}) attributes {alignment = 2 : i64, frame_pointer = #llvm.framePointerKind<none>, passthrough = ["nounwind", ["uwtable", "2"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], ["tune-cpu", "generic"]], target_cpu = "x86-64", target_features = #llvm.target_features<["+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>} {
    llvm.call @_ZN8tutorial6PersonD2Ev(%arg0) : (!llvm.ptr) -> ()
    %0 = llvm.bitcast %arg0 : !llvm.ptr to !llvm.ptr
    llvm.call @_ZdlPv(%0) : (!llvm.ptr) -> ()
    llvm.return
  }
  llvm.func local_unnamed_addr @_ZN8tutorial6Person9ArenaDtorEPv(%arg0: !llvm.ptr {llvm.nocapture, llvm.noundef}) attributes {alignment = 2 : i64, frame_pointer = #llvm.framePointerKind<none>, memory = #llvm.memory_effects<other = none, argMem = none, inaccessibleMem = none>, passthrough = ["mustprogress", "nofree", "norecurse", "nosync", "nounwind", "willreturn", ["uwtable", "2"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], ["tune-cpu", "generic"]], target_cpu = "x86-64", target_features = #llvm.target_features<["+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>} {
    llvm.return
  }
  llvm.func unnamed_addr @_ZNK8tutorial6Person13SetCachedSizeEi(%arg0: !llvm.ptr {llvm.align = 8 : i64, llvm.dereferenceable = 72 : i64, llvm.nocapture, llvm.nonnull, llvm.noundef, llvm.writeonly}, %arg1: i32 {llvm.noundef}) attributes {alignment = 2 : i64, frame_pointer = #llvm.framePointerKind<none>, passthrough = ["mustprogress", "nofree", "norecurse", "nounwind", "willreturn", ["uwtable", "2"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], ["tune-cpu", "generic"]], personality = @__gxx_personality_v0, target_cpu = "x86-64", target_features = #llvm.target_features<["+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>} {
    %0 = llvm.mlir.constant(0 : i64) : i64
    %1 = llvm.mlir.constant(6 : i32) : i32
    %2 = llvm.mlir.constant(0 : i32) : i32
    %3 = llvm.getelementptr inbounds %arg0[%0, 6, 0, 0, 0] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.tutorial::Person", (struct<"class.google::protobuf::Message", (struct<"class.google::protobuf::MessageLite", (ptr, struct<"class.google::protobuf::internal::InternalMetadata", (ptr)>)>)>, struct<"class.google::protobuf::RepeatedPtrField", (struct<"class.google::protobuf::internal::RepeatedPtrFieldBase", (ptr, i32, i32, ptr)>)>, struct<"struct.google::protobuf::internal::ArenaStringPtr", (ptr)>, struct<"struct.google::protobuf::internal::ArenaStringPtr", (ptr)>, ptr, i32, struct<"class.google::protobuf::internal::CachedSize", (struct<"struct.std::atomic", (struct<"struct.std::__atomic_base", (i32)>)>)>)>
    llvm.store %arg1, %3 atomic monotonic {alignment = 4 : i64} : i32, !llvm.ptr
    llvm.return
  }
  llvm.func local_unnamed_addr @_ZN8tutorial6Person16default_instanceEv() -> (!llvm.ptr {llvm.align = 8 : i64, llvm.dereferenceable = 72 : i64, llvm.nonnull, llvm.noundef}) attributes {alignment = 2 : i64, frame_pointer = #llvm.framePointerKind<none>, passthrough = ["mustprogress", ["uwtable", "2"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], ["tune-cpu", "generic"]], target_cpu = "x86-64", target_features = #llvm.target_features<["+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>} {
    %0 = llvm.mlir.addressof @scc_info_Timestamp_google_2fprotobuf_2ftimestamp_2eproto : !llvm.ptr
    %1 = llvm.mlir.undef : !llvm.array<0 x ptr>
    %2 = llvm.mlir.zero : !llvm.ptr
    %3 = llvm.mlir.addressof @__gxx_personality_v0 : !llvm.ptr
    %4 = llvm.mlir.addressof @_ZL59InitDefaultsscc_info_Person_PhoneNumber_addressbook_2eprotov : !llvm.ptr
    %5 = llvm.mlir.constant(0 : i32) : i32
    %6 = llvm.mlir.constant(-1 : i32) : i32
    %7 = llvm.mlir.undef : !llvm.struct<(i32)>
    %8 = llvm.insertvalue %6, %7[0] : !llvm.struct<(i32)> 
    %9 = llvm.mlir.undef : !llvm.struct<(struct<(i32)>, i32, i32, ptr)>
    %10 = llvm.insertvalue %8, %9[0] : !llvm.struct<(struct<(i32)>, i32, i32, ptr)> 
    %11 = llvm.insertvalue %5, %10[1] : !llvm.struct<(struct<(i32)>, i32, i32, ptr)> 
    %12 = llvm.insertvalue %5, %11[2] : !llvm.struct<(struct<(i32)>, i32, i32, ptr)> 
    %13 = llvm.insertvalue %4, %12[3] : !llvm.struct<(struct<(i32)>, i32, i32, ptr)> 
    %14 = llvm.mlir.undef : !llvm.struct<(struct<(struct<(i32)>, i32, i32, ptr)>, array<0 x ptr>)>
    %15 = llvm.insertvalue %13, %14[0] : !llvm.struct<(struct<(struct<(i32)>, i32, i32, ptr)>, array<0 x ptr>)> 
    %16 = llvm.insertvalue %1, %15[1] : !llvm.struct<(struct<(struct<(i32)>, i32, i32, ptr)>, array<0 x ptr>)> 
    %17 = llvm.mlir.addressof @scc_info_Person_PhoneNumber_addressbook_2eproto : !llvm.ptr
    %18 = llvm.mlir.undef : !llvm.array<2 x ptr>
    %19 = llvm.insertvalue %17, %18[0] : !llvm.array<2 x ptr> 
    %20 = llvm.insertvalue %0, %19[1] : !llvm.array<2 x ptr> 
    %21 = llvm.mlir.addressof @_ZL47InitDefaultsscc_info_Person_addressbook_2eprotov : !llvm.ptr
    %22 = llvm.mlir.constant(2 : i32) : i32
    %23 = llvm.mlir.undef : !llvm.struct<(struct<(i32)>, i32, i32, ptr)>
    %24 = llvm.insertvalue %8, %23[0] : !llvm.struct<(struct<(i32)>, i32, i32, ptr)> 
    %25 = llvm.insertvalue %22, %24[1] : !llvm.struct<(struct<(i32)>, i32, i32, ptr)> 
    %26 = llvm.insertvalue %5, %25[2] : !llvm.struct<(struct<(i32)>, i32, i32, ptr)> 
    %27 = llvm.insertvalue %21, %26[3] : !llvm.struct<(struct<(i32)>, i32, i32, ptr)> 
    %28 = llvm.mlir.undef : !llvm.struct<(struct<(struct<(i32)>, i32, i32, ptr)>, array<2 x ptr>)>
    %29 = llvm.insertvalue %27, %28[0] : !llvm.struct<(struct<(struct<(i32)>, i32, i32, ptr)>, array<2 x ptr>)> 
    %30 = llvm.insertvalue %20, %29[1] : !llvm.struct<(struct<(struct<(i32)>, i32, i32, ptr)>, array<2 x ptr>)> 
    %31 = llvm.mlir.addressof @scc_info_Person_addressbook_2eproto : !llvm.ptr
    %32 = llvm.mlir.constant(0 : i8) : i8
    %33 = llvm.mlir.constant(dense<0> : tensor<64xi8>) : !llvm.array<64 x i8>
    %34 = llvm.mlir.constant(0 : i64) : i64
    %35 = llvm.mlir.undef : !llvm.struct<"union.google::protobuf::internal::ExplicitlyConstructed<tutorial::Person>::AlignedUnion", (i64, array<64 x i8>)>
    %36 = llvm.insertvalue %34, %35[0] : !llvm.struct<"union.google::protobuf::internal::ExplicitlyConstructed<tutorial::Person>::AlignedUnion", (i64, array<64 x i8>)> 
    %37 = llvm.insertvalue %33, %36[1] : !llvm.struct<"union.google::protobuf::internal::ExplicitlyConstructed<tutorial::Person>::AlignedUnion", (i64, array<64 x i8>)> 
    %38 = llvm.mlir.undef : !llvm.struct<"class.google::protobuf::internal::ExplicitlyConstructed.0", (struct<"union.google::protobuf::internal::ExplicitlyConstructed<tutorial::Person>::AlignedUnion", (i64, array<64 x i8>)>)>
    %39 = llvm.insertvalue %37, %38[0] : !llvm.struct<"class.google::protobuf::internal::ExplicitlyConstructed.0", (struct<"union.google::protobuf::internal::ExplicitlyConstructed<tutorial::Person>::AlignedUnion", (i64, array<64 x i8>)>)> 
    %40 = llvm.mlir.undef : !llvm.struct<"class.tutorial::PersonDefaultTypeInternal", (struct<"class.google::protobuf::internal::ExplicitlyConstructed.0", (struct<"union.google::protobuf::internal::ExplicitlyConstructed<tutorial::Person>::AlignedUnion", (i64, array<64 x i8>)>)>)>
    %41 = llvm.insertvalue %39, %40[0] : !llvm.struct<"class.tutorial::PersonDefaultTypeInternal", (struct<"class.google::protobuf::internal::ExplicitlyConstructed.0", (struct<"union.google::protobuf::internal::ExplicitlyConstructed<tutorial::Person>::AlignedUnion", (i64, array<64 x i8>)>)>)> 
    %42 = llvm.mlir.addressof @_ZN8tutorial25_Person_default_instance_E : !llvm.ptr
    %43 = llvm.load %31 atomic acquire {alignment = 8 : i64} : !llvm.ptr -> i32
    %44 = llvm.icmp "eq" %43, %5 : i32
    llvm.cond_br %44 weights([2000, 1]), ^bb2, ^bb1
  ^bb1:  // pred: ^bb0
    llvm.call @_ZN6google8protobuf8internal11InitSCCImplEPNS1_11SCCInfoBaseE(%31) : (!llvm.ptr) -> ()
    llvm.br ^bb2
  ^bb2:  // 2 preds: ^bb0, ^bb1
    llvm.return %42 : !llvm.ptr
  }
  llvm.func unnamed_addr @_ZN8tutorial6Person5ClearEv(%arg0: !llvm.ptr {llvm.align = 8 : i64, llvm.dereferenceable = 72 : i64, llvm.nonnull, llvm.noundef}) attributes {alignment = 2 : i64, frame_pointer = #llvm.framePointerKind<none>, passthrough = [["uwtable", "2"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], ["tune-cpu", "generic"]], personality = @__gxx_personality_v0, target_cpu = "x86-64", target_features = #llvm.target_features<["+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>} {
    %0 = llvm.mlir.constant(0 : i64) : i64
    %1 = llvm.mlir.constant(1 : i32) : i32
    %2 = llvm.mlir.constant(0 : i32) : i32
    %3 = llvm.mlir.constant(2 : i32) : i32
    %4 = llvm.mlir.addressof @_ZN6google8protobuf8internal26fixed_address_empty_stringB5cxx11E : !llvm.ptr
    %5 = llvm.mlir.constant(0 : i8) : i8
    %6 = llvm.mlir.constant(3 : i32) : i32
    %7 = llvm.mlir.constant(1 : i64) : i64
    %8 = llvm.mlir.constant(-2 : i64) : i64
    %9 = llvm.mlir.zero : !llvm.ptr
    %10 = llvm.mlir.constant(4 : i32) : i32
    %11 = llvm.mlir.constant(5 : i32) : i32
    %12 = llvm.getelementptr inbounds %arg0[%0, 1, 0] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.tutorial::Person", (struct<"class.google::protobuf::Message", (struct<"class.google::protobuf::MessageLite", (ptr, struct<"class.google::protobuf::internal::InternalMetadata", (ptr)>)>)>, struct<"class.google::protobuf::RepeatedPtrField", (struct<"class.google::protobuf::internal::RepeatedPtrFieldBase", (ptr, i32, i32, ptr)>)>, struct<"struct.google::protobuf::internal::ArenaStringPtr", (ptr)>, struct<"struct.google::protobuf::internal::ArenaStringPtr", (ptr)>, ptr, i32, struct<"class.google::protobuf::internal::CachedSize", (struct<"struct.std::atomic", (struct<"struct.std::__atomic_base", (i32)>)>)>)>
    llvm.call @_ZN6google8protobuf8internal20RepeatedPtrFieldBase5ClearINS0_16RepeatedPtrFieldIN8tutorial18Person_PhoneNumberEE11TypeHandlerEEEvv(%12) : (!llvm.ptr) -> ()
    %13 = llvm.getelementptr inbounds %arg0[%0, 0, 0, 1, 0] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.tutorial::Person", (struct<"class.google::protobuf::Message", (struct<"class.google::protobuf::MessageLite", (ptr, struct<"class.google::protobuf::internal::InternalMetadata", (ptr)>)>)>, struct<"class.google::protobuf::RepeatedPtrField", (struct<"class.google::protobuf::internal::RepeatedPtrFieldBase", (ptr, i32, i32, ptr)>)>, struct<"struct.google::protobuf::internal::ArenaStringPtr", (ptr)>, struct<"struct.google::protobuf::internal::ArenaStringPtr", (ptr)>, ptr, i32, struct<"class.google::protobuf::internal::CachedSize", (struct<"struct.std::atomic", (struct<"struct.std::__atomic_base", (i32)>)>)>)>
    %14 = llvm.getelementptr inbounds %arg0[%0, 2, 0] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.tutorial::Person", (struct<"class.google::protobuf::Message", (struct<"class.google::protobuf::MessageLite", (ptr, struct<"class.google::protobuf::internal::InternalMetadata", (ptr)>)>)>, struct<"class.google::protobuf::RepeatedPtrField", (struct<"class.google::protobuf::internal::RepeatedPtrFieldBase", (ptr, i32, i32, ptr)>)>, struct<"struct.google::protobuf::internal::ArenaStringPtr", (ptr)>, struct<"struct.google::protobuf::internal::ArenaStringPtr", (ptr)>, ptr, i32, struct<"class.google::protobuf::internal::CachedSize", (struct<"struct.std::atomic", (struct<"struct.std::__atomic_base", (i32)>)>)>)>
    %15 = llvm.load %14 {alignment = 8 : i64, tbaa = [#tbaa_tag8]} : !llvm.ptr -> !llvm.ptr
    %16 = llvm.icmp "eq" %15, %4 : !llvm.ptr
    llvm.cond_br %16, ^bb2, ^bb1
  ^bb1:  // pred: ^bb0
    %17 = llvm.getelementptr inbounds %15[%0, 1] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.std::__cxx11::basic_string", (struct<"struct.std::__cxx11::basic_string<char>::_Alloc_hider", (ptr)>, i64, struct<"union.anon", (i64, array<8 x i8>)>)>
    llvm.store %0, %17 {alignment = 8 : i64, tbaa = [#tbaa_tag25]} : i64, !llvm.ptr
    %18 = llvm.getelementptr inbounds %15[%0, 0, 0] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.std::__cxx11::basic_string", (struct<"struct.std::__cxx11::basic_string<char>::_Alloc_hider", (ptr)>, i64, struct<"union.anon", (i64, array<8 x i8>)>)>
    %19 = llvm.load %18 {alignment = 8 : i64, tbaa = [#tbaa_tag26]} : !llvm.ptr -> !llvm.ptr
    llvm.store %5, %19 {alignment = 1 : i64, tbaa = [#tbaa_tag1]} : i8, !llvm.ptr
    llvm.br ^bb2
  ^bb2:  // 2 preds: ^bb0, ^bb1
    %20 = llvm.getelementptr inbounds %arg0[%0, 3, 0] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.tutorial::Person", (struct<"class.google::protobuf::Message", (struct<"class.google::protobuf::MessageLite", (ptr, struct<"class.google::protobuf::internal::InternalMetadata", (ptr)>)>)>, struct<"class.google::protobuf::RepeatedPtrField", (struct<"class.google::protobuf::internal::RepeatedPtrFieldBase", (ptr, i32, i32, ptr)>)>, struct<"struct.google::protobuf::internal::ArenaStringPtr", (ptr)>, struct<"struct.google::protobuf::internal::ArenaStringPtr", (ptr)>, ptr, i32, struct<"class.google::protobuf::internal::CachedSize", (struct<"struct.std::atomic", (struct<"struct.std::__atomic_base", (i32)>)>)>)>
    %21 = llvm.load %20 {alignment = 8 : i64, tbaa = [#tbaa_tag8]} : !llvm.ptr -> !llvm.ptr
    %22 = llvm.icmp "eq" %21, %4 : !llvm.ptr
    llvm.cond_br %22, ^bb4, ^bb3
  ^bb3:  // pred: ^bb2
    %23 = llvm.getelementptr inbounds %21[%0, 1] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.std::__cxx11::basic_string", (struct<"struct.std::__cxx11::basic_string<char>::_Alloc_hider", (ptr)>, i64, struct<"union.anon", (i64, array<8 x i8>)>)>
    llvm.store %0, %23 {alignment = 8 : i64, tbaa = [#tbaa_tag25]} : i64, !llvm.ptr
    %24 = llvm.getelementptr inbounds %21[%0, 0, 0] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.std::__cxx11::basic_string", (struct<"struct.std::__cxx11::basic_string<char>::_Alloc_hider", (ptr)>, i64, struct<"union.anon", (i64, array<8 x i8>)>)>
    %25 = llvm.load %24 {alignment = 8 : i64, tbaa = [#tbaa_tag26]} : !llvm.ptr -> !llvm.ptr
    llvm.store %5, %25 {alignment = 1 : i64, tbaa = [#tbaa_tag1]} : i8, !llvm.ptr
    llvm.br ^bb4
  ^bb4:  // 2 preds: ^bb2, ^bb3
    %26 = llvm.load %13 {alignment = 8 : i64, tbaa = [#tbaa_tag5]} : !llvm.ptr -> !llvm.ptr
    %27 = llvm.ptrtoint %26 : !llvm.ptr to i64
    %28 = llvm.and %27, %7  : i64
    %29 = llvm.icmp "eq" %28, %0 : i64
    %30 = llvm.and %27, %8  : i64
    llvm.cond_br %29 weights([2000, 1]), ^bb6, ^bb5
  ^bb5:  // pred: ^bb4
    %31 = llvm.inttoptr %30 : i64 to !llvm.ptr
    %32 = llvm.getelementptr inbounds %31[%0, 0] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.google::protobuf::internal::InternalMetadata::ContainerBase", (ptr)>
    %33 = llvm.load %32 {alignment = 8 : i64, tbaa = [#tbaa_tag9]} : !llvm.ptr -> !llvm.ptr
    llvm.br ^bb7(%33 : !llvm.ptr)
  ^bb6:  // pred: ^bb4
    %34 = llvm.inttoptr %30 : i64 to !llvm.ptr
    llvm.br ^bb7(%34 : !llvm.ptr)
  ^bb7(%35: !llvm.ptr):  // 2 preds: ^bb5, ^bb6
    %36 = llvm.icmp "eq" %35, %9 : !llvm.ptr
    llvm.cond_br %36, ^bb8, ^bb10(%28, %27 : i64, i64)
  ^bb8:  // pred: ^bb7
    %37 = llvm.getelementptr inbounds %arg0[%0, 4] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.tutorial::Person", (struct<"class.google::protobuf::Message", (struct<"class.google::protobuf::MessageLite", (ptr, struct<"class.google::protobuf::internal::InternalMetadata", (ptr)>)>)>, struct<"class.google::protobuf::RepeatedPtrField", (struct<"class.google::protobuf::internal::RepeatedPtrFieldBase", (ptr, i32, i32, ptr)>)>, struct<"struct.google::protobuf::internal::ArenaStringPtr", (ptr)>, struct<"struct.google::protobuf::internal::ArenaStringPtr", (ptr)>, ptr, i32, struct<"class.google::protobuf::internal::CachedSize", (struct<"struct.std::atomic", (struct<"struct.std::__atomic_base", (i32)>)>)>)>
    %38 = llvm.load %37 {alignment = 8 : i64, tbaa = [#tbaa_tag23]} : !llvm.ptr -> !llvm.ptr
    %39 = llvm.icmp "eq" %38, %9 : !llvm.ptr
    llvm.cond_br %39, ^bb10(%28, %27 : i64, i64), ^bb9
  ^bb9:  // pred: ^bb8
    llvm.call @_ZN6google8protobuf9TimestampD1Ev(%38) : (!llvm.ptr) -> ()
    %40 = llvm.bitcast %38 : !llvm.ptr to !llvm.ptr
    llvm.call @_ZdlPv(%40) : (!llvm.ptr) -> ()
    %41 = llvm.load %13 {alignment = 8 : i64, tbaa = [#tbaa_tag5]} : !llvm.ptr -> !llvm.ptr
    %42 = llvm.ptrtoint %41 : !llvm.ptr to i64
    %43 = llvm.and %42, %7  : i64
    llvm.br ^bb10(%43, %42 : i64, i64)
  ^bb10(%44: i64, %45: i64):  // 3 preds: ^bb7, ^bb8, ^bb9
    %46 = llvm.getelementptr inbounds %arg0[%0, 4] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.tutorial::Person", (struct<"class.google::protobuf::Message", (struct<"class.google::protobuf::MessageLite", (ptr, struct<"class.google::protobuf::internal::InternalMetadata", (ptr)>)>)>, struct<"class.google::protobuf::RepeatedPtrField", (struct<"class.google::protobuf::internal::RepeatedPtrFieldBase", (ptr, i32, i32, ptr)>)>, struct<"struct.google::protobuf::internal::ArenaStringPtr", (ptr)>, struct<"struct.google::protobuf::internal::ArenaStringPtr", (ptr)>, ptr, i32, struct<"class.google::protobuf::internal::CachedSize", (struct<"struct.std::atomic", (struct<"struct.std::__atomic_base", (i32)>)>)>)>
    llvm.store %9, %46 {alignment = 8 : i64, tbaa = [#tbaa_tag23]} : !llvm.ptr, !llvm.ptr
    %47 = llvm.getelementptr inbounds %arg0[%0, 5] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.tutorial::Person", (struct<"class.google::protobuf::Message", (struct<"class.google::protobuf::MessageLite", (ptr, struct<"class.google::protobuf::internal::InternalMetadata", (ptr)>)>)>, struct<"class.google::protobuf::RepeatedPtrField", (struct<"class.google::protobuf::internal::RepeatedPtrFieldBase", (ptr, i32, i32, ptr)>)>, struct<"struct.google::protobuf::internal::ArenaStringPtr", (ptr)>, struct<"struct.google::protobuf::internal::ArenaStringPtr", (ptr)>, ptr, i32, struct<"class.google::protobuf::internal::CachedSize", (struct<"struct.std::atomic", (struct<"struct.std::__atomic_base", (i32)>)>)>)>
    llvm.store %2, %47 {alignment = 8 : i64, tbaa = [#tbaa_tag28]} : i32, !llvm.ptr
    %48 = llvm.icmp "eq" %44, %0 : i64
    llvm.cond_br %48, ^bb13, ^bb11
  ^bb11:  // pred: ^bb10
    %49 = llvm.and %45, %8  : i64
    %50 = llvm.inttoptr %49 : i64 to !llvm.ptr
    %51 = llvm.getelementptr inbounds %50[%0, 1] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.google::protobuf::internal::InternalMetadata::Container", (struct<"struct.google::protobuf::internal::InternalMetadata::ContainerBase", (ptr)>, struct<"class.google::protobuf::UnknownFieldSet", (struct<"class.std::vector", (struct<"struct.std::_Vector_base", (struct<"struct.std::_Vector_base<google::protobuf::UnknownField, std::allocator<google::protobuf::UnknownField>>::_Vector_impl", (struct<"struct.std::_Vector_base<google::protobuf::UnknownField, std::allocator<google::protobuf::UnknownField>>::_Vector_impl_data", (ptr, ptr, ptr)>)>)>)>)>)>
    %52 = llvm.getelementptr inbounds %51[%0, 0, 0, 0, 0, 0] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.google::protobuf::UnknownFieldSet", (struct<"class.std::vector", (struct<"struct.std::_Vector_base", (struct<"struct.std::_Vector_base<google::protobuf::UnknownField, std::allocator<google::protobuf::UnknownField>>::_Vector_impl", (struct<"struct.std::_Vector_base<google::protobuf::UnknownField, std::allocator<google::protobuf::UnknownField>>::_Vector_impl_data", (ptr, ptr, ptr)>)>)>)>)>
    %53 = llvm.load %52 {alignment = 8 : i64, tbaa = [#tbaa_tag2]} : !llvm.ptr -> !llvm.ptr
    %54 = llvm.getelementptr inbounds %50[%0, 1, 0, 0, 0, 0, 1] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.google::protobuf::internal::InternalMetadata::Container", (struct<"struct.google::protobuf::internal::InternalMetadata::ContainerBase", (ptr)>, struct<"class.google::protobuf::UnknownFieldSet", (struct<"class.std::vector", (struct<"struct.std::_Vector_base", (struct<"struct.std::_Vector_base<google::protobuf::UnknownField, std::allocator<google::protobuf::UnknownField>>::_Vector_impl", (struct<"struct.std::_Vector_base<google::protobuf::UnknownField, std::allocator<google::protobuf::UnknownField>>::_Vector_impl_data", (ptr, ptr, ptr)>)>)>)>)>)>
    %55 = llvm.load %54 {alignment = 8 : i64, tbaa = [#tbaa_tag2]} : !llvm.ptr -> !llvm.ptr
    %56 = llvm.icmp "eq" %53, %55 : !llvm.ptr
    llvm.cond_br %56, ^bb13, ^bb12
  ^bb12:  // pred: ^bb11
    llvm.call @_ZN6google8protobuf15UnknownFieldSet13ClearFallbackEv(%51) : (!llvm.ptr) -> ()
    llvm.br ^bb13
  ^bb13:  // 3 preds: ^bb10, ^bb11, ^bb12
    llvm.return
  }
  llvm.func unnamed_addr @_ZN8tutorial6Person14_InternalParseEPKcPN6google8protobuf8internal12ParseContextE(%arg0: !llvm.ptr {llvm.align = 8 : i64, llvm.dereferenceable = 72 : i64, llvm.nonnull, llvm.noundef}, %arg1: !llvm.ptr {llvm.noundef}, %arg2: !llvm.ptr {llvm.noundef}) -> (!llvm.ptr {llvm.noundef}) attributes {alignment = 2 : i64, frame_pointer = #llvm.framePointerKind<none>, passthrough = [["uwtable", "2"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], ["tune-cpu", "generic"]], target_cpu = "x86-64", target_features = #llvm.target_features<["+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>} {
    %0 = llvm.mlir.constant(1 : i32) : i32
    %1 = llvm.mlir.constant(0 : i64) : i64
    %2 = llvm.mlir.constant(0 : i32) : i32
    %3 = llvm.mlir.constant(2 : i32) : i32
    %4 = llvm.mlir.constant(4 : i32) : i32
    %5 = llvm.mlir.constant(8 : i32) : i32
    %6 = llvm.mlir.constant(3 : i32) : i32
    %7 = llvm.mlir.constant(5 : i32) : i32
    %8 = llvm.mlir.constant(-1 : i8) : i8
    %9 = llvm.mlir.constant(1 : i64) : i64
    %10 = llvm.mlir.constant(7 : i32) : i32
    %11 = llvm.mlir.constant(-128 : i32) : i32
    %12 = llvm.mlir.zero : !llvm.ptr
    %13 = llvm.mlir.constant(2 : i64) : i64
    %14 = llvm.mlir.constant(255 : i32) : i32
    %15 = llvm.mlir.constant(42 : i32) : i32
    %16 = llvm.mlir.constant(-2 : i64) : i64
    %17 = llvm.mlir.constant(-1 : i32) : i32
    %18 = llvm.mlir.constant(34 : i32) : i32
    %19 = llvm.mlir.constant(-1 : i64) : i64
    %20 = llvm.mlir.constant(34 : i8) : i8
    %21 = llvm.mlir.constant(26 : i32) : i32
    %22 = llvm.mlir.addressof @_ZN6google8protobuf8internal26fixed_address_empty_stringB5cxx11E : !llvm.ptr
    %23 = llvm.mlir.constant("size_t to int conversion\00") : !llvm.array<25 x i8>
    %24 = llvm.mlir.addressof @".str.15" : !llvm.ptr
    %25 = llvm.mlir.constant("tutorial.Person.email\00") : !llvm.array<22 x i8>
    %26 = llvm.mlir.addressof @".str.8" : !llvm.ptr
    %27 = llvm.mlir.constant(true) : i1
    %28 = llvm.mlir.constant(16 : i32) : i32
    %29 = llvm.mlir.constant(128 : i32) : i32
    %30 = llvm.mlir.constant(10 : i32) : i32
    %31 = llvm.mlir.constant("tutorial.Person.name\00") : !llvm.array<21 x i8>
    %32 = llvm.mlir.addressof @".str.7" : !llvm.ptr
    %33 = llvm.alloca %0 x !llvm.ptr {alignment = 8 : i64} : (i32) -> !llvm.ptr
    llvm.store %arg1, %33 {alignment = 8 : i64, tbaa = [#tbaa_tag2]} : !llvm.ptr, !llvm.ptr
    %34 = llvm.getelementptr inbounds %arg2[%1, 0] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.google::protobuf::internal::ParseContext", (struct<"class.google::protobuf::internal::EpsCopyInputStream", (ptr, ptr, ptr, i32, i32, ptr, array<32 x i8>, i64, i32, i32)>, i32, i32, struct<"struct.google::protobuf::internal::ParseContext::Data", (ptr, ptr)>)>
    %35 = llvm.getelementptr inbounds %arg2[%1, 2] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.google::protobuf::internal::ParseContext", (struct<"class.google::protobuf::internal::EpsCopyInputStream", (ptr, ptr, ptr, i32, i32, ptr, array<32 x i8>, i64, i32, i32)>, i32, i32, struct<"struct.google::protobuf::internal::ParseContext::Data", (ptr, ptr)>)>
    %36 = llvm.getelementptr inbounds %arg0[%1, 4] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.tutorial::Person", (struct<"class.google::protobuf::Message", (struct<"class.google::protobuf::MessageLite", (ptr, struct<"class.google::protobuf::internal::InternalMetadata", (ptr)>)>)>, struct<"class.google::protobuf::RepeatedPtrField", (struct<"class.google::protobuf::internal::RepeatedPtrFieldBase", (ptr, i32, i32, ptr)>)>, struct<"struct.google::protobuf::internal::ArenaStringPtr", (ptr)>, struct<"struct.google::protobuf::internal::ArenaStringPtr", (ptr)>, ptr, i32, struct<"class.google::protobuf::internal::CachedSize", (struct<"struct.std::atomic", (struct<"struct.std::__atomic_base", (i32)>)>)>)>
    %37 = llvm.getelementptr inbounds %arg0[%1, 0, 0, 1, 0] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.tutorial::Person", (struct<"class.google::protobuf::Message", (struct<"class.google::protobuf::MessageLite", (ptr, struct<"class.google::protobuf::internal::InternalMetadata", (ptr)>)>)>, struct<"class.google::protobuf::RepeatedPtrField", (struct<"class.google::protobuf::internal::RepeatedPtrFieldBase", (ptr, i32, i32, ptr)>)>, struct<"struct.google::protobuf::internal::ArenaStringPtr", (ptr)>, struct<"struct.google::protobuf::internal::ArenaStringPtr", (ptr)>, ptr, i32, struct<"class.google::protobuf::internal::CachedSize", (struct<"struct.std::atomic", (struct<"struct.std::__atomic_base", (i32)>)>)>)>
    %38 = llvm.getelementptr inbounds %arg2[%1, 1] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.google::protobuf::internal::ParseContext", (struct<"class.google::protobuf::internal::EpsCopyInputStream", (ptr, ptr, ptr, i32, i32, ptr, array<32 x i8>, i64, i32, i32)>, i32, i32, struct<"struct.google::protobuf::internal::ParseContext::Data", (ptr, ptr)>)>
    %39 = llvm.getelementptr inbounds %arg2[%1, 0, 8] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.google::protobuf::internal::ParseContext", (struct<"class.google::protobuf::internal::EpsCopyInputStream", (ptr, ptr, ptr, i32, i32, ptr, array<32 x i8>, i64, i32, i32)>, i32, i32, struct<"struct.google::protobuf::internal::ParseContext::Data", (ptr, ptr)>)>
    %40 = llvm.getelementptr inbounds %arg2[%1, 0, 4] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.google::protobuf::internal::ParseContext", (struct<"class.google::protobuf::internal::EpsCopyInputStream", (ptr, ptr, ptr, i32, i32, ptr, array<32 x i8>, i64, i32, i32)>, i32, i32, struct<"struct.google::protobuf::internal::ParseContext::Data", (ptr, ptr)>)>
    %41 = llvm.getelementptr inbounds %arg2[%1, 0, 1] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.google::protobuf::internal::ParseContext", (struct<"class.google::protobuf::internal::EpsCopyInputStream", (ptr, ptr, ptr, i32, i32, ptr, array<32 x i8>, i64, i32, i32)>, i32, i32, struct<"struct.google::protobuf::internal::ParseContext::Data", (ptr, ptr)>)>
    %42 = llvm.getelementptr inbounds %arg2[%1, 0, 0] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.google::protobuf::internal::ParseContext", (struct<"class.google::protobuf::internal::EpsCopyInputStream", (ptr, ptr, ptr, i32, i32, ptr, array<32 x i8>, i64, i32, i32)>, i32, i32, struct<"struct.google::protobuf::internal::ParseContext::Data", (ptr, ptr)>)>
    %43 = llvm.getelementptr inbounds %arg0[%1, 1] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.tutorial::Person", (struct<"class.google::protobuf::Message", (struct<"class.google::protobuf::MessageLite", (ptr, struct<"class.google::protobuf::internal::InternalMetadata", (ptr)>)>)>, struct<"class.google::protobuf::RepeatedPtrField", (struct<"class.google::protobuf::internal::RepeatedPtrFieldBase", (ptr, i32, i32, ptr)>)>, struct<"struct.google::protobuf::internal::ArenaStringPtr", (ptr)>, struct<"struct.google::protobuf::internal::ArenaStringPtr", (ptr)>, ptr, i32, struct<"class.google::protobuf::internal::CachedSize", (struct<"struct.std::atomic", (struct<"struct.std::__atomic_base", (i32)>)>)>)>
    %44 = llvm.getelementptr inbounds %43[%1, 0] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.google::protobuf::RepeatedPtrField", (struct<"class.google::protobuf::internal::RepeatedPtrFieldBase", (ptr, i32, i32, ptr)>)>
    %45 = llvm.getelementptr inbounds %arg0[%1, 1, 0, 3] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.tutorial::Person", (struct<"class.google::protobuf::Message", (struct<"class.google::protobuf::MessageLite", (ptr, struct<"class.google::protobuf::internal::InternalMetadata", (ptr)>)>)>, struct<"class.google::protobuf::RepeatedPtrField", (struct<"class.google::protobuf::internal::RepeatedPtrFieldBase", (ptr, i32, i32, ptr)>)>, struct<"struct.google::protobuf::internal::ArenaStringPtr", (ptr)>, struct<"struct.google::protobuf::internal::ArenaStringPtr", (ptr)>, ptr, i32, struct<"class.google::protobuf::internal::CachedSize", (struct<"struct.std::atomic", (struct<"struct.std::__atomic_base", (i32)>)>)>)>
    %46 = llvm.getelementptr inbounds %arg0[%1, 1, 0, 1] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.tutorial::Person", (struct<"class.google::protobuf::Message", (struct<"class.google::protobuf::MessageLite", (ptr, struct<"class.google::protobuf::internal::InternalMetadata", (ptr)>)>)>, struct<"class.google::protobuf::RepeatedPtrField", (struct<"class.google::protobuf::internal::RepeatedPtrFieldBase", (ptr, i32, i32, ptr)>)>, struct<"struct.google::protobuf::internal::ArenaStringPtr", (ptr)>, struct<"struct.google::protobuf::internal::ArenaStringPtr", (ptr)>, ptr, i32, struct<"class.google::protobuf::internal::CachedSize", (struct<"struct.std::atomic", (struct<"struct.std::__atomic_base", (i32)>)>)>)>
    %47 = llvm.getelementptr inbounds %arg0[%1, 1, 0, 2] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.tutorial::Person", (struct<"class.google::protobuf::Message", (struct<"class.google::protobuf::MessageLite", (ptr, struct<"class.google::protobuf::internal::InternalMetadata", (ptr)>)>)>, struct<"class.google::protobuf::RepeatedPtrField", (struct<"class.google::protobuf::internal::RepeatedPtrFieldBase", (ptr, i32, i32, ptr)>)>, struct<"struct.google::protobuf::internal::ArenaStringPtr", (ptr)>, struct<"struct.google::protobuf::internal::ArenaStringPtr", (ptr)>, ptr, i32, struct<"class.google::protobuf::internal::CachedSize", (struct<"struct.std::atomic", (struct<"struct.std::__atomic_base", (i32)>)>)>)>
    %48 = llvm.getelementptr inbounds %43[%1, 0, 0] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.google::protobuf::RepeatedPtrField", (struct<"class.google::protobuf::internal::RepeatedPtrFieldBase", (ptr, i32, i32, ptr)>)>
    %49 = llvm.getelementptr inbounds %arg0[%1, 3] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.tutorial::Person", (struct<"class.google::protobuf::Message", (struct<"class.google::protobuf::MessageLite", (ptr, struct<"class.google::protobuf::internal::InternalMetadata", (ptr)>)>)>, struct<"class.google::protobuf::RepeatedPtrField", (struct<"class.google::protobuf::internal::RepeatedPtrFieldBase", (ptr, i32, i32, ptr)>)>, struct<"struct.google::protobuf::internal::ArenaStringPtr", (ptr)>, struct<"struct.google::protobuf::internal::ArenaStringPtr", (ptr)>, ptr, i32, struct<"class.google::protobuf::internal::CachedSize", (struct<"struct.std::atomic", (struct<"struct.std::__atomic_base", (i32)>)>)>)>
    %50 = llvm.getelementptr inbounds %49[%1, 0] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.google::protobuf::internal::ArenaStringPtr", (ptr)>
    %51 = llvm.getelementptr inbounds %arg0[%1, 5] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.tutorial::Person", (struct<"class.google::protobuf::Message", (struct<"class.google::protobuf::MessageLite", (ptr, struct<"class.google::protobuf::internal::InternalMetadata", (ptr)>)>)>, struct<"class.google::protobuf::RepeatedPtrField", (struct<"class.google::protobuf::internal::RepeatedPtrFieldBase", (ptr, i32, i32, ptr)>)>, struct<"struct.google::protobuf::internal::ArenaStringPtr", (ptr)>, struct<"struct.google::protobuf::internal::ArenaStringPtr", (ptr)>, ptr, i32, struct<"class.google::protobuf::internal::CachedSize", (struct<"struct.std::atomic", (struct<"struct.std::__atomic_base", (i32)>)>)>)>
    %52 = llvm.getelementptr inbounds %arg0[%1, 2] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.tutorial::Person", (struct<"class.google::protobuf::Message", (struct<"class.google::protobuf::MessageLite", (ptr, struct<"class.google::protobuf::internal::InternalMetadata", (ptr)>)>)>, struct<"class.google::protobuf::RepeatedPtrField", (struct<"class.google::protobuf::internal::RepeatedPtrFieldBase", (ptr, i32, i32, ptr)>)>, struct<"struct.google::protobuf::internal::ArenaStringPtr", (ptr)>, struct<"struct.google::protobuf::internal::ArenaStringPtr", (ptr)>, ptr, i32, struct<"class.google::protobuf::internal::CachedSize", (struct<"struct.std::atomic", (struct<"struct.std::__atomic_base", (i32)>)>)>)>
    %53 = llvm.getelementptr inbounds %52[%1, 0] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.google::protobuf::internal::ArenaStringPtr", (ptr)>
    %54 = llvm.getelementptr inbounds %arg0[%1, 0, 0, 1] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.tutorial::Person", (struct<"class.google::protobuf::Message", (struct<"class.google::protobuf::MessageLite", (ptr, struct<"class.google::protobuf::internal::InternalMetadata", (ptr)>)>)>, struct<"class.google::protobuf::RepeatedPtrField", (struct<"class.google::protobuf::internal::RepeatedPtrFieldBase", (ptr, i32, i32, ptr)>)>, struct<"struct.google::protobuf::internal::ArenaStringPtr", (ptr)>, struct<"struct.google::protobuf::internal::ArenaStringPtr", (ptr)>, ptr, i32, struct<"class.google::protobuf::internal::CachedSize", (struct<"struct.std::atomic", (struct<"struct.std::__atomic_base", (i32)>)>)>)>
    %55 = llvm.getelementptr inbounds %54[%1, 0] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.google::protobuf::internal::InternalMetadata", (ptr)>
    %56 = llvm.load %35 {alignment = 4 : i64, tbaa = [#tbaa_tag27]} : !llvm.ptr -> i32
    %57 = llvm.call @_ZN6google8protobuf8internal18EpsCopyInputStream13DoneWithCheckEPPKci(%34, %33, %56) : (!llvm.ptr, !llvm.ptr, i32) -> i1
    %58 = llvm.load %33 {alignment = 8 : i64, tbaa = [#tbaa_tag2]} : !llvm.ptr -> !llvm.ptr
    llvm.cond_br %57, ^bb69(%58 : !llvm.ptr), ^bb1(%58 : !llvm.ptr)
  ^bb1(%59: !llvm.ptr):  // 2 preds: ^bb0, ^bb67
    %60 = llvm.load %59 {alignment = 1 : i64, tbaa = [#tbaa_tag1]} : !llvm.ptr -> i8
    %61 = llvm.zext %60 : i8 to i32
    %62 = llvm.icmp "sgt" %60, %8 : i8
    %63 = llvm.getelementptr inbounds %59[%9] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    llvm.cond_br %62, ^bb4(%61, %63 : i32, !llvm.ptr), ^bb2
  ^bb2:  // pred: ^bb1
    %64 = llvm.load %63 {alignment = 1 : i64, tbaa = [#tbaa_tag1]} : !llvm.ptr -> i8
    %65 = llvm.zext %64 : i8 to i32
    %66 = llvm.shl %65, %10 overflow<nsw, nuw>  : i32
    %67 = llvm.add %61, %11 overflow<nsw>  : i32
    %68 = llvm.add %67, %66 overflow<nsw>  : i32
    %69 = llvm.icmp "sgt" %64, %8 : i8
    llvm.cond_br %69, ^bb3, ^bb5
  ^bb3:  // pred: ^bb2
    %70 = llvm.getelementptr inbounds %59[%13] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    llvm.br ^bb4(%68, %70 : i32, !llvm.ptr)
  ^bb4(%71: i32, %72: !llvm.ptr):  // 2 preds: ^bb1, ^bb3
    llvm.store %72, %33 {alignment = 8 : i64, tbaa = [#tbaa_tag2]} : !llvm.ptr, !llvm.ptr
    llvm.br ^bb6(%72, %71 : !llvm.ptr, i32)
  ^bb5:  // pred: ^bb2
    %73 = llvm.call @_ZN6google8protobuf8internal15ReadTagFallbackEPKcj(%59, %68) : (!llvm.ptr, i32) -> !llvm.struct<(ptr, i32)>
    %74 = llvm.extractvalue %73[0] : !llvm.struct<(ptr, i32)> 
    %75 = llvm.extractvalue %73[1] : !llvm.struct<(ptr, i32)> 
    llvm.store %74, %33 {alignment = 8 : i64, tbaa = [#tbaa_tag2]} : !llvm.ptr, !llvm.ptr
    %76 = llvm.icmp "eq" %74, %12 : !llvm.ptr
    llvm.cond_br %76 weights([1, 2000]), ^bb68, ^bb6(%74, %75 : !llvm.ptr, i32)
  ^bb6(%77: !llvm.ptr, %78: i32):  // 2 preds: ^bb4, ^bb5
    %79 = llvm.lshr %78, %6  : i32
    llvm.switch %79 : i32, ^bb61 [
      1: ^bb7,
      2: ^bb16,
      3: ^bb22,
      4: ^bb31,
      5: ^bb48
    ]
  ^bb7:  // pred: ^bb6
    %80 = llvm.and %78, %14  : i32
    %81 = llvm.icmp "eq" %80, %30 : i32
    llvm.cond_br %81 weights([2000, 1]), ^bb8, ^bb61
  ^bb8:  // pred: ^bb7
    %82 = llvm.load %37 {alignment = 8 : i64, tbaa = [#tbaa_tag5]} : !llvm.ptr -> !llvm.ptr
    %83 = llvm.ptrtoint %82 : !llvm.ptr to i64
    %84 = llvm.and %83, %9  : i64
    %85 = llvm.icmp "eq" %84, %1 : i64
    %86 = llvm.and %83, %16  : i64
    llvm.cond_br %85 weights([2000, 1]), ^bb10, ^bb9
  ^bb9:  // pred: ^bb8
    %87 = llvm.inttoptr %86 : i64 to !llvm.ptr
    %88 = llvm.getelementptr inbounds %87[%1, 0] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.google::protobuf::internal::InternalMetadata::ContainerBase", (ptr)>
    %89 = llvm.load %88 {alignment = 8 : i64, tbaa = [#tbaa_tag9]} : !llvm.ptr -> !llvm.ptr
    llvm.br ^bb11(%89 : !llvm.ptr)
  ^bb10:  // pred: ^bb8
    %90 = llvm.inttoptr %86 : i64 to !llvm.ptr
    llvm.br ^bb11(%90 : !llvm.ptr)
  ^bb11(%91: !llvm.ptr):  // 2 preds: ^bb9, ^bb10
    %92 = llvm.load %53 {alignment = 8 : i64, tbaa = [#tbaa_tag8]} : !llvm.ptr -> !llvm.ptr
    %93 = llvm.icmp "eq" %92, %22 : !llvm.ptr
    llvm.cond_br %93, ^bb12, ^bb13(%77, %92 : !llvm.ptr, !llvm.ptr)
  ^bb12:  // pred: ^bb11
    llvm.call @_ZN6google8protobuf8internal14ArenaStringPtr14CreateInstanceEPNS0_5ArenaEPKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE(%52, %91, %22) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
    %94 = llvm.load %53 {alignment = 8 : i64, tbaa = [#tbaa_tag8]} : !llvm.ptr -> !llvm.ptr
    %95 = llvm.load %33 {alignment = 8 : i64, tbaa = [#tbaa_tag2]} : !llvm.ptr -> !llvm.ptr
    llvm.br ^bb13(%95, %94 : !llvm.ptr, !llvm.ptr)
  ^bb13(%96: !llvm.ptr, %97: !llvm.ptr):  // 2 preds: ^bb11, ^bb12
    %98 = llvm.call @_ZN6google8protobuf8internal24InlineGreedyStringParserEPNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEPKcPNS1_12ParseContextE(%97, %96, %arg2) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> !llvm.ptr
    llvm.store %98, %33 {alignment = 8 : i64, tbaa = [#tbaa_tag2]} : !llvm.ptr, !llvm.ptr
    %99 = llvm.getelementptr inbounds %97[%1, 0, 0] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.std::__cxx11::basic_string", (struct<"struct.std::__cxx11::basic_string<char>::_Alloc_hider", (ptr)>, i64, struct<"union.anon", (i64, array<8 x i8>)>)>
    %100 = llvm.load %99 {alignment = 8 : i64, tbaa = [#tbaa_tag26]} : !llvm.ptr -> !llvm.ptr
    %101 = llvm.getelementptr inbounds %97[%1, 1] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.std::__cxx11::basic_string", (struct<"struct.std::__cxx11::basic_string<char>::_Alloc_hider", (ptr)>, i64, struct<"union.anon", (i64, array<8 x i8>)>)>
    %102 = llvm.load %101 {alignment = 8 : i64, tbaa = [#tbaa_tag25]} : !llvm.ptr -> i64
    %103 = llvm.icmp "slt" %102, %1 : i64
    llvm.cond_br %103, ^bb14, ^bb15
  ^bb14:  // pred: ^bb13
    llvm.call @_ZN6google8protobuf11StringPiece18LogFatalSizeTooBigEmPKc(%102, %24) : (i64, !llvm.ptr) -> ()
    llvm.br ^bb15
  ^bb15:  // 2 preds: ^bb13, ^bb14
    %104 = llvm.call @_ZN6google8protobuf8internal10VerifyUTF8ENS0_11StringPieceEPKc(%100, %102, %32) : (!llvm.ptr, i64, !llvm.ptr) -> i1
    %105 = llvm.xor %104, %27  : i1
    %106 = llvm.load %33 {alignment = 8 : i64} : !llvm.ptr -> !llvm.ptr
    %107 = llvm.icmp "eq" %106, %12 : !llvm.ptr
    %108 = llvm.select %105, %27, %107 : i1, i1
    llvm.cond_br %108 weights([2002, 2000]), ^bb68, ^bb67
  ^bb16:  // pred: ^bb6
    %109 = llvm.and %78, %14  : i32
    %110 = llvm.icmp "eq" %109, %28 : i32
    llvm.cond_br %110 weights([2000, 1]), ^bb17, ^bb61
  ^bb17:  // pred: ^bb16
    %111 = llvm.load %77 {alignment = 1 : i64, tbaa = [#tbaa_tag1]} : !llvm.ptr -> i8
    %112 = llvm.zext %111 : i8 to i32
    %113 = llvm.and %112, %29  : i32
    %114 = llvm.icmp "eq" %113, %2 : i32
    %115 = llvm.getelementptr inbounds %77[%9] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    llvm.cond_br %114, ^bb20(%112, %115 : i32, !llvm.ptr), ^bb18
  ^bb18:  // pred: ^bb17
    %116 = llvm.load %115 {alignment = 1 : i64, tbaa = [#tbaa_tag1]} : !llvm.ptr -> i8
    %117 = llvm.zext %116 : i8 to i32
    %118 = llvm.shl %117, %10 overflow<nsw, nuw>  : i32
    %119 = llvm.add %112, %11 overflow<nsw>  : i32
    %120 = llvm.add %119, %118 overflow<nsw, nuw>  : i32
    %121 = llvm.and %117, %29  : i32
    %122 = llvm.icmp "eq" %121, %2 : i32
    llvm.cond_br %122, ^bb19, ^bb21
  ^bb19:  // pred: ^bb18
    %123 = llvm.getelementptr inbounds %77[%13] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    llvm.br ^bb20(%120, %123 : i32, !llvm.ptr)
  ^bb20(%124: i32, %125: !llvm.ptr):  // 2 preds: ^bb17, ^bb19
    llvm.store %125, %33 {alignment = 8 : i64, tbaa = [#tbaa_tag2]} : !llvm.ptr, !llvm.ptr
    llvm.store %124, %51 {alignment = 8 : i64, tbaa = [#tbaa_tag28]} : i32, !llvm.ptr
    llvm.br ^bb67
  ^bb21:  // pred: ^bb18
    %126 = llvm.call @_ZN6google8protobuf8internal17VarintParseSlow64EPKcj(%77, %120) : (!llvm.ptr, i32) -> !llvm.struct<(ptr, i64)>
    %127 = llvm.extractvalue %126[0] : !llvm.struct<(ptr, i64)> 
    %128 = llvm.extractvalue %126[1] : !llvm.struct<(ptr, i64)> 
    llvm.store %127, %33 {alignment = 8 : i64, tbaa = [#tbaa_tag2]} : !llvm.ptr, !llvm.ptr
    %129 = llvm.trunc %128 : i64 to i32
    llvm.store %129, %51 {alignment = 8 : i64, tbaa = [#tbaa_tag28]} : i32, !llvm.ptr
    %130 = llvm.icmp "eq" %127, %12 : !llvm.ptr
    llvm.cond_br %130, ^bb68, ^bb67
  ^bb22:  // pred: ^bb6
    %131 = llvm.and %78, %14  : i32
    %132 = llvm.icmp "eq" %131, %21 : i32
    llvm.cond_br %132 weights([2000, 1]), ^bb23, ^bb61
  ^bb23:  // pred: ^bb22
    %133 = llvm.load %37 {alignment = 8 : i64, tbaa = [#tbaa_tag5]} : !llvm.ptr -> !llvm.ptr
    %134 = llvm.ptrtoint %133 : !llvm.ptr to i64
    %135 = llvm.and %134, %9  : i64
    %136 = llvm.icmp "eq" %135, %1 : i64
    %137 = llvm.and %134, %16  : i64
    llvm.cond_br %136 weights([2000, 1]), ^bb25, ^bb24
  ^bb24:  // pred: ^bb23
    %138 = llvm.inttoptr %137 : i64 to !llvm.ptr
    %139 = llvm.getelementptr inbounds %138[%1, 0] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.google::protobuf::internal::InternalMetadata::ContainerBase", (ptr)>
    %140 = llvm.load %139 {alignment = 8 : i64, tbaa = [#tbaa_tag9]} : !llvm.ptr -> !llvm.ptr
    llvm.br ^bb26(%140 : !llvm.ptr)
  ^bb25:  // pred: ^bb23
    %141 = llvm.inttoptr %137 : i64 to !llvm.ptr
    llvm.br ^bb26(%141 : !llvm.ptr)
  ^bb26(%142: !llvm.ptr):  // 2 preds: ^bb24, ^bb25
    %143 = llvm.load %50 {alignment = 8 : i64, tbaa = [#tbaa_tag8]} : !llvm.ptr -> !llvm.ptr
    %144 = llvm.icmp "eq" %143, %22 : !llvm.ptr
    llvm.cond_br %144, ^bb27, ^bb28(%77, %143 : !llvm.ptr, !llvm.ptr)
  ^bb27:  // pred: ^bb26
    llvm.call @_ZN6google8protobuf8internal14ArenaStringPtr14CreateInstanceEPNS0_5ArenaEPKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE(%49, %142, %22) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
    %145 = llvm.load %50 {alignment = 8 : i64, tbaa = [#tbaa_tag8]} : !llvm.ptr -> !llvm.ptr
    %146 = llvm.load %33 {alignment = 8 : i64, tbaa = [#tbaa_tag2]} : !llvm.ptr -> !llvm.ptr
    llvm.br ^bb28(%146, %145 : !llvm.ptr, !llvm.ptr)
  ^bb28(%147: !llvm.ptr, %148: !llvm.ptr):  // 2 preds: ^bb26, ^bb27
    %149 = llvm.call @_ZN6google8protobuf8internal24InlineGreedyStringParserEPNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEPKcPNS1_12ParseContextE(%148, %147, %arg2) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> !llvm.ptr
    llvm.store %149, %33 {alignment = 8 : i64, tbaa = [#tbaa_tag2]} : !llvm.ptr, !llvm.ptr
    %150 = llvm.getelementptr inbounds %148[%1, 0, 0] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.std::__cxx11::basic_string", (struct<"struct.std::__cxx11::basic_string<char>::_Alloc_hider", (ptr)>, i64, struct<"union.anon", (i64, array<8 x i8>)>)>
    %151 = llvm.load %150 {alignment = 8 : i64, tbaa = [#tbaa_tag26]} : !llvm.ptr -> !llvm.ptr
    %152 = llvm.getelementptr inbounds %148[%1, 1] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.std::__cxx11::basic_string", (struct<"struct.std::__cxx11::basic_string<char>::_Alloc_hider", (ptr)>, i64, struct<"union.anon", (i64, array<8 x i8>)>)>
    %153 = llvm.load %152 {alignment = 8 : i64, tbaa = [#tbaa_tag25]} : !llvm.ptr -> i64
    %154 = llvm.icmp "slt" %153, %1 : i64
    llvm.cond_br %154, ^bb29, ^bb30
  ^bb29:  // pred: ^bb28
    llvm.call @_ZN6google8protobuf11StringPiece18LogFatalSizeTooBigEmPKc(%153, %24) : (i64, !llvm.ptr) -> ()
    llvm.br ^bb30
  ^bb30:  // 2 preds: ^bb28, ^bb29
    %155 = llvm.call @_ZN6google8protobuf8internal10VerifyUTF8ENS0_11StringPieceEPKc(%151, %153, %26) : (!llvm.ptr, i64, !llvm.ptr) -> i1
    %156 = llvm.xor %155, %27  : i1
    %157 = llvm.load %33 {alignment = 8 : i64} : !llvm.ptr -> !llvm.ptr
    %158 = llvm.icmp "eq" %157, %12 : !llvm.ptr
    %159 = llvm.select %156, %27, %158 : i1, i1
    llvm.cond_br %159 weights([2002, 2000]), ^bb68, ^bb67
  ^bb31:  // pred: ^bb6
    %160 = llvm.and %78, %14  : i32
    %161 = llvm.icmp "eq" %160, %18 : i32
    llvm.cond_br %161 weights([2000, 1]), ^bb32, ^bb61
  ^bb32:  // pred: ^bb31
    %162 = llvm.getelementptr inbounds %77[%19] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    llvm.br ^bb33(%162 : !llvm.ptr)
  ^bb33(%163: !llvm.ptr):  // 2 preds: ^bb32, ^bb47
    %164 = llvm.getelementptr inbounds %163[%9] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    llvm.store %164, %33 {alignment = 8 : i64, tbaa = [#tbaa_tag2]} : !llvm.ptr, !llvm.ptr
    %165 = llvm.load %45 {alignment = 8 : i64, tbaa = [#tbaa_tag15]} : !llvm.ptr -> !llvm.ptr
    %166 = llvm.icmp "eq" %165, %12 : !llvm.ptr
    llvm.cond_br %166, ^bb34, ^bb35
  ^bb34:  // pred: ^bb33
    %167 = llvm.load %47 {alignment = 4 : i64, tbaa = [#tbaa_tag16]} : !llvm.ptr -> i32
    llvm.br ^bb38(%167 : i32)
  ^bb35:  // pred: ^bb33
    %168 = llvm.load %46 {alignment = 8 : i64, tbaa = [#tbaa_tag17]} : !llvm.ptr -> i32
    %169 = llvm.getelementptr inbounds %165[%1, 0] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.google::protobuf::internal::RepeatedPtrFieldBase::Rep", (i32, array<1 x ptr>)>
    %170 = llvm.load %169 {alignment = 8 : i64, tbaa = [#tbaa_tag18]} : !llvm.ptr -> i32
    %171 = llvm.icmp "slt" %168, %170 : i32
    llvm.cond_br %171, ^bb36, ^bb37
  ^bb36:  // pred: ^bb35
    %172 = llvm.add %168, %0 overflow<nsw>  : i32
    llvm.store %172, %46 {alignment = 8 : i64, tbaa = [#tbaa_tag17]} : i32, !llvm.ptr
    %173 = llvm.sext %168 : i32 to i64
    %174 = llvm.getelementptr inbounds %165[%1, 1, %173] : (!llvm.ptr, i64, i64) -> !llvm.ptr, !llvm.struct<"struct.google::protobuf::internal::RepeatedPtrFieldBase::Rep", (i32, array<1 x ptr>)>
    %175 = llvm.bitcast %174 : !llvm.ptr to !llvm.ptr
    %176 = llvm.load %175 {alignment = 8 : i64, tbaa = [#tbaa_tag2]} : !llvm.ptr -> !llvm.ptr
    llvm.br ^bb40(%164, %176 : !llvm.ptr, !llvm.ptr)
  ^bb37:  // pred: ^bb35
    %177 = llvm.load %47 {alignment = 4 : i64, tbaa = [#tbaa_tag16]} : !llvm.ptr -> i32
    %178 = llvm.icmp "eq" %170, %177 : i32
    llvm.cond_br %178, ^bb38(%170 : i32), ^bb39(%170, %165 : i32, !llvm.ptr)
  ^bb38(%179: i32):  // 2 preds: ^bb34, ^bb37
    %180 = llvm.add %179, %0 overflow<nsw>  : i32
    llvm.call @_ZN6google8protobuf8internal20RepeatedPtrFieldBase7ReserveEi(%44, %180) : (!llvm.ptr, i32) -> ()
    %181 = llvm.load %45 {alignment = 8 : i64, tbaa = [#tbaa_tag15]} : !llvm.ptr -> !llvm.ptr
    %182 = llvm.getelementptr inbounds %181[%1, 0] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.google::protobuf::internal::RepeatedPtrFieldBase::Rep", (i32, array<1 x ptr>)>
    %183 = llvm.load %182 {alignment = 8 : i64, tbaa = [#tbaa_tag18]} : !llvm.ptr -> i32
    llvm.br ^bb39(%183, %181 : i32, !llvm.ptr)
  ^bb39(%184: i32, %185: !llvm.ptr):  // 2 preds: ^bb37, ^bb38
    %186 = llvm.getelementptr inbounds %185[%1, 0] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.google::protobuf::internal::RepeatedPtrFieldBase::Rep", (i32, array<1 x ptr>)>
    %187 = llvm.add %184, %0 overflow<nsw>  : i32
    llvm.store %187, %186 {alignment = 8 : i64, tbaa = [#tbaa_tag18]} : i32, !llvm.ptr
    %188 = llvm.load %48 {alignment = 8 : i64, tbaa = [#tbaa_tag7]} : !llvm.ptr -> !llvm.ptr
    %189 = llvm.call @_ZN6google8protobuf5Arena18CreateMaybeMessageIN8tutorial18Person_PhoneNumberEJEEEPT_PS1_DpOT0_(%188) : (!llvm.ptr) -> !llvm.ptr
    %190 = llvm.load %45 {alignment = 8 : i64, tbaa = [#tbaa_tag15]} : !llvm.ptr -> !llvm.ptr
    %191 = llvm.load %46 {alignment = 8 : i64, tbaa = [#tbaa_tag17]} : !llvm.ptr -> i32
    %192 = llvm.add %191, %0 overflow<nsw>  : i32
    llvm.store %192, %46 {alignment = 8 : i64, tbaa = [#tbaa_tag17]} : i32, !llvm.ptr
    %193 = llvm.sext %191 : i32 to i64
    %194 = llvm.getelementptr inbounds %190[%1, 1, %193] : (!llvm.ptr, i64, i64) -> !llvm.ptr, !llvm.struct<"struct.google::protobuf::internal::RepeatedPtrFieldBase::Rep", (i32, array<1 x ptr>)>
    %195 = llvm.bitcast %194 : !llvm.ptr to !llvm.ptr
    llvm.store %189, %195 {alignment = 8 : i64, tbaa = [#tbaa_tag2]} : !llvm.ptr, !llvm.ptr
    %196 = llvm.load %33 {alignment = 8 : i64, tbaa = [#tbaa_tag2]} : !llvm.ptr -> !llvm.ptr
    llvm.br ^bb40(%196, %189 : !llvm.ptr, !llvm.ptr)
  ^bb40(%197: !llvm.ptr, %198: !llvm.ptr):  // 2 preds: ^bb36, ^bb39
    %199 = llvm.load %197 {alignment = 1 : i64, tbaa = [#tbaa_tag1]} : !llvm.ptr -> i8
    %200 = llvm.zext %199 : i8 to i32
    %201 = llvm.icmp "sgt" %199, %8 : i8
    llvm.cond_br %201, ^bb41, ^bb42
  ^bb41:  // pred: ^bb40
    %202 = llvm.getelementptr inbounds %197[%9] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    llvm.br ^bb43(%200, %202 : i32, !llvm.ptr)
  ^bb42:  // pred: ^bb40
    %203 = llvm.call @_ZN6google8protobuf8internal16ReadSizeFallbackEPKcj(%197, %200) : (!llvm.ptr, i32) -> !llvm.struct<(ptr, i32)>
    %204 = llvm.extractvalue %203[0] : !llvm.struct<(ptr, i32)> 
    %205 = llvm.extractvalue %203[1] : !llvm.struct<(ptr, i32)> 
    %206 = llvm.icmp "eq" %204, %12 : !llvm.ptr
    llvm.cond_br %206, ^bb68, ^bb43(%205, %204 : i32, !llvm.ptr)
  ^bb43(%207: i32, %208: !llvm.ptr):  // 2 preds: ^bb41, ^bb42
    %209 = llvm.call @_ZN6google8protobuf8internal18EpsCopyInputStream9PushLimitEPKci(%34, %208, %207) : (!llvm.ptr, !llvm.ptr, i32) -> i32
    %210 = llvm.load %38 {alignment = 8 : i64, tbaa = [#tbaa_tag29]} : !llvm.ptr -> i32
    %211 = llvm.add %210, %17 overflow<nsw>  : i32
    llvm.store %211, %38 {alignment = 8 : i64, tbaa = [#tbaa_tag29]} : i32, !llvm.ptr
    %212 = llvm.icmp "slt" %210, %0 : i32
    llvm.cond_br %212, ^bb68, ^bb44
  ^bb44:  // pred: ^bb43
    %213 = llvm.call @_ZN8tutorial18Person_PhoneNumber14_InternalParseEPKcPN6google8protobuf8internal12ParseContextE(%198, %208, %arg2) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> !llvm.ptr
    %214 = llvm.icmp "eq" %213, %12 : !llvm.ptr
    llvm.cond_br %214 weights([1, 2000]), ^bb68, ^bb45
  ^bb45:  // pred: ^bb44
    %215 = llvm.load %38 {alignment = 8 : i64, tbaa = [#tbaa_tag29]} : !llvm.ptr -> i32
    %216 = llvm.add %215, %0 overflow<nsw>  : i32
    llvm.store %216, %38 {alignment = 8 : i64, tbaa = [#tbaa_tag29]} : i32, !llvm.ptr
    %217 = llvm.load %39 {alignment = 8 : i64, tbaa = [#tbaa_tag11]} : !llvm.ptr -> i32
    %218 = llvm.icmp "eq" %217, %2 : i32
    llvm.cond_br %218 weights([2000, 1]), ^bb46, ^bb68
  ^bb46:  // pred: ^bb45
    %219 = llvm.load %40 {alignment = 4 : i64, tbaa = [#tbaa_tag19]} : !llvm.ptr -> i32
    %220 = llvm.add %219, %209 overflow<nsw>  : i32
    llvm.store %220, %40 {alignment = 4 : i64, tbaa = [#tbaa_tag19]} : i32, !llvm.ptr
    %221 = llvm.load %41 {alignment = 8 : i64, tbaa = [#tbaa_tag20]} : !llvm.ptr -> !llvm.ptr
    %222 = llvm.icmp "slt" %220, %2 : i32
    %223 = llvm.select %222, %220, %2 : i1, i32
    %224 = llvm.sext %223 : i32 to i64
    %225 = llvm.getelementptr inbounds %221[%224] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    llvm.store %225, %42 {alignment = 8 : i64, tbaa = [#tbaa_tag21]} : !llvm.ptr, !llvm.ptr
    llvm.store %213, %33 {alignment = 8 : i64, tbaa = [#tbaa_tag2]} : !llvm.ptr, !llvm.ptr
    %226 = llvm.icmp "ugt" %225, %213 : !llvm.ptr
    llvm.cond_br %226, ^bb47, ^bb67 {loop_annotation = #loop_annotation}
  ^bb47:  // pred: ^bb46
    %227 = llvm.load %213 {alignment = 1 : i64, tbaa = [#tbaa_tag1]} : !llvm.ptr -> i8
    %228 = llvm.icmp "eq" %227, %20 : i8
    llvm.cond_br %228, ^bb33(%213 : !llvm.ptr), ^bb67 {loop_annotation = #loop_annotation}
  ^bb48:  // pred: ^bb6
    %229 = llvm.and %78, %14  : i32
    %230 = llvm.icmp "eq" %229, %15 : i32
    llvm.cond_br %230 weights([2000, 1]), ^bb49, ^bb61
  ^bb49:  // pred: ^bb48
    %231 = llvm.load %36 {alignment = 8 : i64, tbaa = [#tbaa_tag23]} : !llvm.ptr -> !llvm.ptr
    %232 = llvm.icmp "eq" %231, %12 : !llvm.ptr
    llvm.cond_br %232, ^bb50, ^bb54(%77, %231 : !llvm.ptr, !llvm.ptr)
  ^bb50:  // pred: ^bb49
    %233 = llvm.load %37 {alignment = 8 : i64, tbaa = [#tbaa_tag5]} : !llvm.ptr -> !llvm.ptr
    %234 = llvm.ptrtoint %233 : !llvm.ptr to i64
    %235 = llvm.and %234, %9  : i64
    %236 = llvm.icmp "eq" %235, %1 : i64
    %237 = llvm.and %234, %16  : i64
    llvm.cond_br %236 weights([2000, 1]), ^bb52, ^bb51
  ^bb51:  // pred: ^bb50
    %238 = llvm.inttoptr %237 : i64 to !llvm.ptr
    %239 = llvm.getelementptr inbounds %238[%1, 0] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.google::protobuf::internal::InternalMetadata::ContainerBase", (ptr)>
    %240 = llvm.load %239 {alignment = 8 : i64, tbaa = [#tbaa_tag9]} : !llvm.ptr -> !llvm.ptr
    llvm.br ^bb53(%240 : !llvm.ptr)
  ^bb52:  // pred: ^bb50
    %241 = llvm.inttoptr %237 : i64 to !llvm.ptr
    llvm.br ^bb53(%241 : !llvm.ptr)
  ^bb53(%242: !llvm.ptr):  // 2 preds: ^bb51, ^bb52
    %243 = llvm.call @_ZN6google8protobuf5Arena18CreateMaybeMessageINS0_9TimestampEJEEEPT_PS1_DpOT0_(%242) : (!llvm.ptr) -> !llvm.ptr
    llvm.store %243, %36 {alignment = 8 : i64, tbaa = [#tbaa_tag23]} : !llvm.ptr, !llvm.ptr
    %244 = llvm.load %33 {alignment = 8 : i64, tbaa = [#tbaa_tag2]} : !llvm.ptr -> !llvm.ptr
    llvm.br ^bb54(%244, %243 : !llvm.ptr, !llvm.ptr)
  ^bb54(%245: !llvm.ptr, %246: !llvm.ptr):  // 2 preds: ^bb49, ^bb53
    %247 = llvm.load %245 {alignment = 1 : i64, tbaa = [#tbaa_tag1]} : !llvm.ptr -> i8
    %248 = llvm.zext %247 : i8 to i32
    %249 = llvm.icmp "sgt" %247, %8 : i8
    llvm.cond_br %249, ^bb55, ^bb56
  ^bb55:  // pred: ^bb54
    %250 = llvm.getelementptr inbounds %245[%9] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    llvm.br ^bb57(%248, %250 : i32, !llvm.ptr)
  ^bb56:  // pred: ^bb54
    %251 = llvm.call @_ZN6google8protobuf8internal16ReadSizeFallbackEPKcj(%245, %248) : (!llvm.ptr, i32) -> !llvm.struct<(ptr, i32)>
    %252 = llvm.extractvalue %251[0] : !llvm.struct<(ptr, i32)> 
    %253 = llvm.extractvalue %251[1] : !llvm.struct<(ptr, i32)> 
    %254 = llvm.icmp "eq" %252, %12 : !llvm.ptr
    llvm.cond_br %254, ^bb68, ^bb57(%253, %252 : i32, !llvm.ptr)
  ^bb57(%255: i32, %256: !llvm.ptr):  // 2 preds: ^bb55, ^bb56
    %257 = llvm.call @_ZN6google8protobuf8internal18EpsCopyInputStream9PushLimitEPKci(%34, %256, %255) : (!llvm.ptr, !llvm.ptr, i32) -> i32
    %258 = llvm.load %38 {alignment = 8 : i64, tbaa = [#tbaa_tag29]} : !llvm.ptr -> i32
    %259 = llvm.add %258, %17 overflow<nsw>  : i32
    llvm.store %259, %38 {alignment = 8 : i64, tbaa = [#tbaa_tag29]} : i32, !llvm.ptr
    %260 = llvm.icmp "slt" %258, %0 : i32
    llvm.cond_br %260, ^bb68, ^bb58
  ^bb58:  // pred: ^bb57
    %261 = llvm.call @_ZN6google8protobuf9Timestamp14_InternalParseEPKcPNS0_8internal12ParseContextE(%246, %256, %arg2) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> !llvm.ptr
    %262 = llvm.icmp "eq" %261, %12 : !llvm.ptr
    llvm.cond_br %262 weights([1, 2000]), ^bb68, ^bb59
  ^bb59:  // pred: ^bb58
    %263 = llvm.load %38 {alignment = 8 : i64, tbaa = [#tbaa_tag29]} : !llvm.ptr -> i32
    %264 = llvm.add %263, %0 overflow<nsw>  : i32
    llvm.store %264, %38 {alignment = 8 : i64, tbaa = [#tbaa_tag29]} : i32, !llvm.ptr
    %265 = llvm.load %39 {alignment = 8 : i64, tbaa = [#tbaa_tag11]} : !llvm.ptr -> i32
    %266 = llvm.icmp "eq" %265, %2 : i32
    llvm.cond_br %266 weights([2000, 1]), ^bb60, ^bb68
  ^bb60:  // pred: ^bb59
    %267 = llvm.load %40 {alignment = 4 : i64, tbaa = [#tbaa_tag19]} : !llvm.ptr -> i32
    %268 = llvm.add %267, %257 overflow<nsw>  : i32
    llvm.store %268, %40 {alignment = 4 : i64, tbaa = [#tbaa_tag19]} : i32, !llvm.ptr
    %269 = llvm.load %41 {alignment = 8 : i64, tbaa = [#tbaa_tag20]} : !llvm.ptr -> !llvm.ptr
    %270 = llvm.icmp "slt" %268, %2 : i32
    %271 = llvm.select %270, %268, %2 : i1, i32
    %272 = llvm.sext %271 : i32 to i64
    %273 = llvm.getelementptr inbounds %269[%272] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    llvm.store %273, %42 {alignment = 8 : i64, tbaa = [#tbaa_tag21]} : !llvm.ptr, !llvm.ptr
    llvm.store %261, %33 {alignment = 8 : i64, tbaa = [#tbaa_tag2]} : !llvm.ptr, !llvm.ptr
    llvm.br ^bb67
  ^bb61:  // 6 preds: ^bb6, ^bb7, ^bb16, ^bb22, ^bb31, ^bb48
    %274 = llvm.and %78, %10  : i32
    %275 = llvm.icmp "eq" %274, %4 : i32
    %276 = llvm.icmp "eq" %78, %2 : i32
    %277 = llvm.or %276, %275  : i1
    llvm.cond_br %277, ^bb62, ^bb63
  ^bb62:  // pred: ^bb61
    %278 = llvm.add %78, %17  : i32
    llvm.store %278, %39 {alignment = 8 : i64, tbaa = [#tbaa_tag11]} : i32, !llvm.ptr
    llvm.br ^bb69(%77 : !llvm.ptr)
  ^bb63:  // pred: ^bb61
    %279 = llvm.zext %78 : i32 to i64
    %280 = llvm.load %55 {alignment = 8 : i64, tbaa = [#tbaa_tag5]} : !llvm.ptr -> !llvm.ptr
    %281 = llvm.ptrtoint %280 : !llvm.ptr to i64
    %282 = llvm.and %281, %9  : i64
    %283 = llvm.icmp "eq" %282, %1 : i64
    llvm.cond_br %283 weights([1, 2000]), ^bb65, ^bb64
  ^bb64:  // pred: ^bb63
    %284 = llvm.and %281, %16  : i64
    %285 = llvm.inttoptr %284 : i64 to !llvm.ptr
    %286 = llvm.getelementptr inbounds %285[%1, 1] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.google::protobuf::internal::InternalMetadata::Container", (struct<"struct.google::protobuf::internal::InternalMetadata::ContainerBase", (ptr)>, struct<"class.google::protobuf::UnknownFieldSet", (struct<"class.std::vector", (struct<"struct.std::_Vector_base", (struct<"struct.std::_Vector_base<google::protobuf::UnknownField, std::allocator<google::protobuf::UnknownField>>::_Vector_impl", (struct<"struct.std::_Vector_base<google::protobuf::UnknownField, std::allocator<google::protobuf::UnknownField>>::_Vector_impl_data", (ptr, ptr, ptr)>)>)>)>)>)>
    llvm.br ^bb66(%77, %286 : !llvm.ptr, !llvm.ptr)
  ^bb65:  // pred: ^bb63
    %287 = llvm.call @_ZN6google8protobuf8internal16InternalMetadata27mutable_unknown_fields_slowINS0_15UnknownFieldSetEEEPT_v(%54) : (!llvm.ptr) -> !llvm.ptr
    %288 = llvm.load %33 {alignment = 8 : i64, tbaa = [#tbaa_tag2]} : !llvm.ptr -> !llvm.ptr
    llvm.br ^bb66(%288, %287 : !llvm.ptr, !llvm.ptr)
  ^bb66(%289: !llvm.ptr, %290: !llvm.ptr):  // 2 preds: ^bb64, ^bb65
    %291 = llvm.call @_ZN6google8protobuf8internal17UnknownFieldParseEmPNS0_15UnknownFieldSetEPKcPNS1_12ParseContextE(%279, %290, %289, %arg2) : (i64, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> !llvm.ptr
    llvm.store %291, %33 {alignment = 8 : i64, tbaa = [#tbaa_tag2]} : !llvm.ptr, !llvm.ptr
    %292 = llvm.icmp "eq" %291, %12 : !llvm.ptr
    llvm.cond_br %292, ^bb68, ^bb67
  ^bb67:  // 8 preds: ^bb15, ^bb20, ^bb21, ^bb30, ^bb46, ^bb47, ^bb60, ^bb66
    %293 = llvm.load %35 {alignment = 4 : i64, tbaa = [#tbaa_tag27]} : !llvm.ptr -> i32
    %294 = llvm.call @_ZN6google8protobuf8internal18EpsCopyInputStream13DoneWithCheckEPPKci(%34, %33, %293) : (!llvm.ptr, !llvm.ptr, i32) -> i1
    %295 = llvm.load %33 {alignment = 8 : i64, tbaa = [#tbaa_tag2]} : !llvm.ptr -> !llvm.ptr
    llvm.cond_br %294, ^bb69(%295 : !llvm.ptr), ^bb1(%295 : !llvm.ptr)
  ^bb68:  // 13 preds: ^bb5, ^bb15, ^bb21, ^bb30, ^bb42, ^bb43, ^bb44, ^bb45, ^bb56, ^bb57, ^bb58, ^bb59, ^bb66
    llvm.store %12, %33 {alignment = 8 : i64, tbaa = [#tbaa_tag2]} : !llvm.ptr, !llvm.ptr
    llvm.br ^bb69(%12 : !llvm.ptr)
  ^bb69(%296: !llvm.ptr):  // 4 preds: ^bb0, ^bb62, ^bb67, ^bb68
    llvm.return %296 : !llvm.ptr
  }
  llvm.func unnamed_addr @_ZNK8tutorial6Person18_InternalSerializeEPhPN6google8protobuf2io19EpsCopyOutputStreamE(%arg0: !llvm.ptr {llvm.align = 8 : i64, llvm.dereferenceable = 72 : i64, llvm.nonnull, llvm.noundef}, %arg1: !llvm.ptr {llvm.noundef}, %arg2: !llvm.ptr {llvm.noundef}) -> (!llvm.ptr {llvm.noundef}) attributes {alignment = 2 : i64, frame_pointer = #llvm.framePointerKind<none>, passthrough = ["mustprogress", ["uwtable", "2"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], ["tune-cpu", "generic"]], target_cpu = "x86-64", target_features = #llvm.target_features<["+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>} {
    %0 = llvm.mlir.constant(0 : i64) : i64
    %1 = llvm.mlir.constant(2 : i32) : i32
    %2 = llvm.mlir.constant(0 : i32) : i32
    %3 = llvm.mlir.constant(1 : i32) : i32
    %4 = llvm.mlir.constant("tutorial.Person.name\00") : !llvm.array<21 x i8>
    %5 = llvm.mlir.addressof @".str.7" : !llvm.ptr
    %6 = llvm.mlir.constant(127 : i64) : i64
    %7 = llvm.mlir.constant(14 : i64) : i64
    %8 = llvm.mlir.constant(10 : i8) : i8
    %9 = llvm.mlir.constant(1 : i64) : i64
    %10 = llvm.mlir.constant(2 : i64) : i64
    %11 = llvm.mlir.constant(5 : i32) : i32
    %12 = llvm.mlir.constant(16 : i8) : i8
    %13 = llvm.mlir.constant(128 : i32) : i32
    %14 = llvm.mlir.constant(-128 : i8) : i8
    %15 = llvm.mlir.constant(7 : i64) : i64
    %16 = llvm.mlir.constant(16384 : i32) : i32
    %17 = llvm.mlir.constant(16383 : i64) : i64
    %18 = llvm.mlir.constant(3 : i64) : i64
    %19 = llvm.mlir.constant(3 : i32) : i32
    %20 = llvm.mlir.constant("tutorial.Person.email\00") : !llvm.array<22 x i8>
    %21 = llvm.mlir.addressof @".str.8" : !llvm.ptr
    %22 = llvm.mlir.constant(26 : i8) : i8
    %23 = llvm.mlir.constant(34 : i8) : i8
    %24 = llvm.mlir.constant(7 : i32) : i32
    %25 = llvm.mlir.constant(16383 : i32) : i32
    %26 = llvm.mlir.constant(0 : i8) : i8
    %27 = llvm.mlir.constant(dense<0> : tensor<64xi8>) : !llvm.array<64 x i8>
    %28 = llvm.mlir.undef : !llvm.struct<"union.google::protobuf::internal::ExplicitlyConstructed<tutorial::Person>::AlignedUnion", (i64, array<64 x i8>)>
    %29 = llvm.insertvalue %0, %28[0] : !llvm.struct<"union.google::protobuf::internal::ExplicitlyConstructed<tutorial::Person>::AlignedUnion", (i64, array<64 x i8>)> 
    %30 = llvm.insertvalue %27, %29[1] : !llvm.struct<"union.google::protobuf::internal::ExplicitlyConstructed<tutorial::Person>::AlignedUnion", (i64, array<64 x i8>)> 
    %31 = llvm.mlir.undef : !llvm.struct<"class.google::protobuf::internal::ExplicitlyConstructed.0", (struct<"union.google::protobuf::internal::ExplicitlyConstructed<tutorial::Person>::AlignedUnion", (i64, array<64 x i8>)>)>
    %32 = llvm.insertvalue %30, %31[0] : !llvm.struct<"class.google::protobuf::internal::ExplicitlyConstructed.0", (struct<"union.google::protobuf::internal::ExplicitlyConstructed<tutorial::Person>::AlignedUnion", (i64, array<64 x i8>)>)> 
    %33 = llvm.mlir.undef : !llvm.struct<"class.tutorial::PersonDefaultTypeInternal", (struct<"class.google::protobuf::internal::ExplicitlyConstructed.0", (struct<"union.google::protobuf::internal::ExplicitlyConstructed<tutorial::Person>::AlignedUnion", (i64, array<64 x i8>)>)>)>
    %34 = llvm.insertvalue %32, %33[0] : !llvm.struct<"class.tutorial::PersonDefaultTypeInternal", (struct<"class.google::protobuf::internal::ExplicitlyConstructed.0", (struct<"union.google::protobuf::internal::ExplicitlyConstructed<tutorial::Person>::AlignedUnion", (i64, array<64 x i8>)>)>)> 
    %35 = llvm.mlir.addressof @_ZN8tutorial25_Person_default_instance_E : !llvm.ptr
    %36 = llvm.mlir.constant(4 : i32) : i32
    %37 = llvm.mlir.zero : !llvm.ptr
    %38 = llvm.mlir.constant(false) : i1
    %39 = llvm.mlir.constant(42 : i8) : i8
    %40 = llvm.mlir.constant(-2 : i64) : i64
    %41 = llvm.getelementptr inbounds %arg0[%0, 2, 0] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.tutorial::Person", (struct<"class.google::protobuf::Message", (struct<"class.google::protobuf::MessageLite", (ptr, struct<"class.google::protobuf::internal::InternalMetadata", (ptr)>)>)>, struct<"class.google::protobuf::RepeatedPtrField", (struct<"class.google::protobuf::internal::RepeatedPtrFieldBase", (ptr, i32, i32, ptr)>)>, struct<"struct.google::protobuf::internal::ArenaStringPtr", (ptr)>, struct<"struct.google::protobuf::internal::ArenaStringPtr", (ptr)>, ptr, i32, struct<"class.google::protobuf::internal::CachedSize", (struct<"struct.std::atomic", (struct<"struct.std::__atomic_base", (i32)>)>)>)>
    %42 = llvm.load %41 {alignment = 8 : i64, tbaa = [#tbaa_tag8]} : !llvm.ptr -> !llvm.ptr
    %43 = llvm.getelementptr inbounds %42[%0, 1] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.std::__cxx11::basic_string", (struct<"struct.std::__cxx11::basic_string<char>::_Alloc_hider", (ptr)>, i64, struct<"union.anon", (i64, array<8 x i8>)>)>
    %44 = llvm.load %43 {alignment = 8 : i64, tbaa = [#tbaa_tag25]} : !llvm.ptr -> i64
    %45 = llvm.icmp "eq" %44, %0 : i64
    llvm.cond_br %45, ^bb5(%arg1 : !llvm.ptr), ^bb1
  ^bb1:  // pred: ^bb0
    %46 = llvm.getelementptr inbounds %42[%0, 0, 0] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.std::__cxx11::basic_string", (struct<"struct.std::__cxx11::basic_string<char>::_Alloc_hider", (ptr)>, i64, struct<"union.anon", (i64, array<8 x i8>)>)>
    %47 = llvm.load %46 {alignment = 8 : i64, tbaa = [#tbaa_tag26]} : !llvm.ptr -> !llvm.ptr
    %48 = llvm.trunc %44 : i64 to i32
    %49 = llvm.call @_ZN6google8protobuf8internal14WireFormatLite16VerifyUtf8StringEPKciNS2_9OperationES4_(%47, %48, %3, %5) : (!llvm.ptr, i32, i32, !llvm.ptr) -> i1
    %50 = llvm.load %41 {alignment = 8 : i64, tbaa = [#tbaa_tag8]} : !llvm.ptr -> !llvm.ptr
    %51 = llvm.getelementptr inbounds %50[%0, 1] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.std::__cxx11::basic_string", (struct<"struct.std::__cxx11::basic_string<char>::_Alloc_hider", (ptr)>, i64, struct<"union.anon", (i64, array<8 x i8>)>)>
    %52 = llvm.load %51 {alignment = 8 : i64, tbaa = [#tbaa_tag25]} : !llvm.ptr -> i64
    %53 = llvm.icmp "sgt" %52, %6 : i64
    llvm.cond_br %53 weights([1, 2000]), ^bb3, ^bb2
  ^bb2:  // pred: ^bb1
    %54 = llvm.getelementptr inbounds %arg2[%0, 0] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.google::protobuf::io::EpsCopyOutputStream", packed (ptr, ptr, array<32 x i8>, ptr, i8, i8, i8, array<5 x i8>)>
    %55 = llvm.load %54 {alignment = 8 : i64, tbaa = [#tbaa_tag12]} : !llvm.ptr -> !llvm.ptr
    %56 = llvm.ptrtoint %55 : !llvm.ptr to i64
    %57 = llvm.ptrtoint %arg1 : !llvm.ptr to i64
    %58 = llvm.sub %7, %57  : i64
    %59 = llvm.add %58, %56  : i64
    %60 = llvm.icmp "slt" %59, %52 : i64
    llvm.cond_br %60 weights([1, 2000]), ^bb3, ^bb4
  ^bb3:  // 2 preds: ^bb1, ^bb2
    %61 = llvm.call @_ZN6google8protobuf2io19EpsCopyOutputStream30WriteStringMaybeAliasedOutlineEjRKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEPh(%arg2, %3, %50, %arg1) : (!llvm.ptr, i32, !llvm.ptr, !llvm.ptr) -> !llvm.ptr
    llvm.br ^bb5(%61 : !llvm.ptr)
  ^bb4:  // pred: ^bb2
    llvm.store %8, %arg1 {alignment = 1 : i64, tbaa = [#tbaa_tag1]} : i8, !llvm.ptr
    %62 = llvm.getelementptr inbounds %arg1[%9] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %63 = llvm.trunc %52 : i64 to i8
    %64 = llvm.getelementptr inbounds %arg1[%10] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    llvm.store %63, %62 {alignment = 1 : i64, tbaa = [#tbaa_tag1]} : i8, !llvm.ptr
    %65 = llvm.getelementptr inbounds %50[%0, 0, 0] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.std::__cxx11::basic_string", (struct<"struct.std::__cxx11::basic_string<char>::_Alloc_hider", (ptr)>, i64, struct<"union.anon", (i64, array<8 x i8>)>)>
    %66 = llvm.load %65 {alignment = 8 : i64, tbaa = [#tbaa_tag26]} : !llvm.ptr -> !llvm.ptr
    "llvm.intr.memcpy"(%64, %66, %52) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
    %67 = llvm.getelementptr inbounds %64[%52] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    llvm.br ^bb5(%67 : !llvm.ptr)
  ^bb5(%68: !llvm.ptr):  // 3 preds: ^bb0, ^bb3, ^bb4
    %69 = llvm.getelementptr inbounds %arg0[%0, 5] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.tutorial::Person", (struct<"class.google::protobuf::Message", (struct<"class.google::protobuf::MessageLite", (ptr, struct<"class.google::protobuf::internal::InternalMetadata", (ptr)>)>)>, struct<"class.google::protobuf::RepeatedPtrField", (struct<"class.google::protobuf::internal::RepeatedPtrFieldBase", (ptr, i32, i32, ptr)>)>, struct<"struct.google::protobuf::internal::ArenaStringPtr", (ptr)>, struct<"struct.google::protobuf::internal::ArenaStringPtr", (ptr)>, ptr, i32, struct<"class.google::protobuf::internal::CachedSize", (struct<"struct.std::atomic", (struct<"struct.std::__atomic_base", (i32)>)>)>)>
    %70 = llvm.load %69 {alignment = 8 : i64, tbaa = [#tbaa_tag28]} : !llvm.ptr -> i32
    %71 = llvm.icmp "eq" %70, %2 : i32
    llvm.cond_br %71, ^bb15(%68 : !llvm.ptr), ^bb6
  ^bb6:  // pred: ^bb5
    %72 = llvm.getelementptr inbounds %arg2[%0, 0] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.google::protobuf::io::EpsCopyOutputStream", packed (ptr, ptr, array<32 x i8>, ptr, i8, i8, i8, array<5 x i8>)>
    %73 = llvm.load %72 {alignment = 8 : i64, tbaa = [#tbaa_tag12]} : !llvm.ptr -> !llvm.ptr
    %74 = llvm.icmp "ugt" %73, %68 : !llvm.ptr
    llvm.cond_br %74 weights([2000, 1]), ^bb8(%70, %68 : i32, !llvm.ptr), ^bb7
  ^bb7:  // pred: ^bb6
    %75 = llvm.call @_ZN6google8protobuf2io19EpsCopyOutputStream19EnsureSpaceFallbackEPh(%arg2, %68) : (!llvm.ptr, !llvm.ptr) -> !llvm.ptr
    %76 = llvm.load %69 {alignment = 8 : i64, tbaa = [#tbaa_tag28]} : !llvm.ptr -> i32
    llvm.br ^bb8(%76, %75 : i32, !llvm.ptr)
  ^bb8(%77: i32, %78: !llvm.ptr):  // 2 preds: ^bb6, ^bb7
    llvm.store %12, %78 {alignment = 1 : i64, tbaa = [#tbaa_tag1]} : i8, !llvm.ptr
    %79 = llvm.getelementptr inbounds %78[%9] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %80 = llvm.icmp "ult" %77, %13 : i32
    %81 = llvm.trunc %77 : i32 to i8
    llvm.cond_br %80, ^bb9, ^bb10
  ^bb9:  // pred: ^bb8
    llvm.store %81, %79 {alignment = 1 : i64, tbaa = [#tbaa_tag1]} : i8, !llvm.ptr
    %82 = llvm.getelementptr inbounds %78[%10] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    llvm.br ^bb15(%82 : !llvm.ptr)
  ^bb10:  // pred: ^bb8
    %83 = llvm.sext %77 : i32 to i64
    %84 = llvm.or %81, %14  : i8
    llvm.store %84, %79 {alignment = 1 : i64, tbaa = [#tbaa_tag1]} : i8, !llvm.ptr
    %85 = llvm.lshr %83, %15  : i64
    %86 = llvm.icmp "ult" %77, %16 : i32
    llvm.cond_br %86, ^bb11, ^bb12
  ^bb11:  // pred: ^bb10
    %87 = llvm.trunc %85 : i64 to i8
    %88 = llvm.getelementptr inbounds %78[%10] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    llvm.store %87, %88 {alignment = 1 : i64, tbaa = [#tbaa_tag1]} : i8, !llvm.ptr
    %89 = llvm.getelementptr inbounds %78[%18] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    llvm.br ^bb15(%89 : !llvm.ptr)
  ^bb12:  // pred: ^bb10
    %90 = llvm.getelementptr inbounds %78[%10] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    llvm.br ^bb13(%85, %90 : i64, !llvm.ptr)
  ^bb13(%91: i64, %92: !llvm.ptr):  // 2 preds: ^bb12, ^bb13
    %93 = llvm.trunc %91 : i64 to i8
    %94 = llvm.or %93, %14  : i8
    llvm.store %94, %92 {alignment = 1 : i64, tbaa = [#tbaa_tag1]} : i8, !llvm.ptr
    %95 = llvm.lshr %91, %15  : i64
    %96 = llvm.getelementptr inbounds %92[%9] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %97 = llvm.icmp "ugt" %91, %17 : i64
    llvm.cond_br %97 weights([1, 2000]), ^bb13(%95, %96 : i64, !llvm.ptr), ^bb14 {loop_annotation = #loop_annotation}
  ^bb14:  // pred: ^bb13
    %98 = llvm.trunc %95 : i64 to i8
    %99 = llvm.getelementptr inbounds %92[%10] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    llvm.store %98, %96 {alignment = 1 : i64, tbaa = [#tbaa_tag1]} : i8, !llvm.ptr
    llvm.br ^bb15(%99 : !llvm.ptr)
  ^bb15(%100: !llvm.ptr):  // 4 preds: ^bb5, ^bb9, ^bb11, ^bb14
    %101 = llvm.getelementptr inbounds %arg0[%0, 3, 0] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.tutorial::Person", (struct<"class.google::protobuf::Message", (struct<"class.google::protobuf::MessageLite", (ptr, struct<"class.google::protobuf::internal::InternalMetadata", (ptr)>)>)>, struct<"class.google::protobuf::RepeatedPtrField", (struct<"class.google::protobuf::internal::RepeatedPtrFieldBase", (ptr, i32, i32, ptr)>)>, struct<"struct.google::protobuf::internal::ArenaStringPtr", (ptr)>, struct<"struct.google::protobuf::internal::ArenaStringPtr", (ptr)>, ptr, i32, struct<"class.google::protobuf::internal::CachedSize", (struct<"struct.std::atomic", (struct<"struct.std::__atomic_base", (i32)>)>)>)>
    %102 = llvm.load %101 {alignment = 8 : i64, tbaa = [#tbaa_tag8]} : !llvm.ptr -> !llvm.ptr
    %103 = llvm.getelementptr inbounds %102[%0, 1] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.std::__cxx11::basic_string", (struct<"struct.std::__cxx11::basic_string<char>::_Alloc_hider", (ptr)>, i64, struct<"union.anon", (i64, array<8 x i8>)>)>
    %104 = llvm.load %103 {alignment = 8 : i64, tbaa = [#tbaa_tag25]} : !llvm.ptr -> i64
    %105 = llvm.icmp "eq" %104, %0 : i64
    llvm.cond_br %105, ^bb20(%100 : !llvm.ptr), ^bb16
  ^bb16:  // pred: ^bb15
    %106 = llvm.getelementptr inbounds %102[%0, 0, 0] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.std::__cxx11::basic_string", (struct<"struct.std::__cxx11::basic_string<char>::_Alloc_hider", (ptr)>, i64, struct<"union.anon", (i64, array<8 x i8>)>)>
    %107 = llvm.load %106 {alignment = 8 : i64, tbaa = [#tbaa_tag26]} : !llvm.ptr -> !llvm.ptr
    %108 = llvm.trunc %104 : i64 to i32
    %109 = llvm.call @_ZN6google8protobuf8internal14WireFormatLite16VerifyUtf8StringEPKciNS2_9OperationES4_(%107, %108, %3, %21) : (!llvm.ptr, i32, i32, !llvm.ptr) -> i1
    %110 = llvm.load %101 {alignment = 8 : i64, tbaa = [#tbaa_tag8]} : !llvm.ptr -> !llvm.ptr
    %111 = llvm.getelementptr inbounds %110[%0, 1] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.std::__cxx11::basic_string", (struct<"struct.std::__cxx11::basic_string<char>::_Alloc_hider", (ptr)>, i64, struct<"union.anon", (i64, array<8 x i8>)>)>
    %112 = llvm.load %111 {alignment = 8 : i64, tbaa = [#tbaa_tag25]} : !llvm.ptr -> i64
    %113 = llvm.icmp "sgt" %112, %6 : i64
    llvm.cond_br %113 weights([1, 2000]), ^bb18, ^bb17
  ^bb17:  // pred: ^bb16
    %114 = llvm.getelementptr inbounds %arg2[%0, 0] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.google::protobuf::io::EpsCopyOutputStream", packed (ptr, ptr, array<32 x i8>, ptr, i8, i8, i8, array<5 x i8>)>
    %115 = llvm.load %114 {alignment = 8 : i64, tbaa = [#tbaa_tag12]} : !llvm.ptr -> !llvm.ptr
    %116 = llvm.ptrtoint %115 : !llvm.ptr to i64
    %117 = llvm.ptrtoint %100 : !llvm.ptr to i64
    %118 = llvm.sub %7, %117  : i64
    %119 = llvm.add %118, %116  : i64
    %120 = llvm.icmp "slt" %119, %112 : i64
    llvm.cond_br %120 weights([1, 2000]), ^bb18, ^bb19
  ^bb18:  // 2 preds: ^bb16, ^bb17
    %121 = llvm.call @_ZN6google8protobuf2io19EpsCopyOutputStream30WriteStringMaybeAliasedOutlineEjRKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEPh(%arg2, %19, %110, %100) : (!llvm.ptr, i32, !llvm.ptr, !llvm.ptr) -> !llvm.ptr
    llvm.br ^bb20(%121 : !llvm.ptr)
  ^bb19:  // pred: ^bb17
    llvm.store %22, %100 {alignment = 1 : i64, tbaa = [#tbaa_tag1]} : i8, !llvm.ptr
    %122 = llvm.getelementptr inbounds %100[%9] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %123 = llvm.trunc %112 : i64 to i8
    %124 = llvm.getelementptr inbounds %100[%10] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    llvm.store %123, %122 {alignment = 1 : i64, tbaa = [#tbaa_tag1]} : i8, !llvm.ptr
    %125 = llvm.getelementptr inbounds %110[%0, 0, 0] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.std::__cxx11::basic_string", (struct<"struct.std::__cxx11::basic_string<char>::_Alloc_hider", (ptr)>, i64, struct<"union.anon", (i64, array<8 x i8>)>)>
    %126 = llvm.load %125 {alignment = 8 : i64, tbaa = [#tbaa_tag26]} : !llvm.ptr -> !llvm.ptr
    "llvm.intr.memcpy"(%124, %126, %112) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
    %127 = llvm.getelementptr inbounds %124[%112] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    llvm.br ^bb20(%127 : !llvm.ptr)
  ^bb20(%128: !llvm.ptr):  // 3 preds: ^bb15, ^bb18, ^bb19
    %129 = llvm.getelementptr inbounds %arg0[%0, 1, 0, 1] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.tutorial::Person", (struct<"class.google::protobuf::Message", (struct<"class.google::protobuf::MessageLite", (ptr, struct<"class.google::protobuf::internal::InternalMetadata", (ptr)>)>)>, struct<"class.google::protobuf::RepeatedPtrField", (struct<"class.google::protobuf::internal::RepeatedPtrFieldBase", (ptr, i32, i32, ptr)>)>, struct<"struct.google::protobuf::internal::ArenaStringPtr", (ptr)>, struct<"struct.google::protobuf::internal::ArenaStringPtr", (ptr)>, ptr, i32, struct<"class.google::protobuf::internal::CachedSize", (struct<"struct.std::atomic", (struct<"struct.std::__atomic_base", (i32)>)>)>)>
    %130 = llvm.load %129 {alignment = 8 : i64, tbaa = [#tbaa_tag17]} : !llvm.ptr -> i32
    %131 = llvm.icmp "eq" %130, %2 : i32
    llvm.cond_br %131, ^bb22(%128 : !llvm.ptr), ^bb21
  ^bb21:  // pred: ^bb20
    %132 = llvm.getelementptr inbounds %arg2[%0, 0] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.google::protobuf::io::EpsCopyOutputStream", packed (ptr, ptr, array<32 x i8>, ptr, i8, i8, i8, array<5 x i8>)>
    %133 = llvm.getelementptr inbounds %arg0[%0, 1, 0] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.tutorial::Person", (struct<"class.google::protobuf::Message", (struct<"class.google::protobuf::MessageLite", (ptr, struct<"class.google::protobuf::internal::InternalMetadata", (ptr)>)>)>, struct<"class.google::protobuf::RepeatedPtrField", (struct<"class.google::protobuf::internal::RepeatedPtrFieldBase", (ptr, i32, i32, ptr)>)>, struct<"struct.google::protobuf::internal::ArenaStringPtr", (ptr)>, struct<"struct.google::protobuf::internal::ArenaStringPtr", (ptr)>, ptr, i32, struct<"class.google::protobuf::internal::CachedSize", (struct<"struct.std::atomic", (struct<"struct.std::__atomic_base", (i32)>)>)>)>
    llvm.br ^bb23(%128, %2 : !llvm.ptr, i32)
  ^bb22(%134: !llvm.ptr):  // 2 preds: ^bb20, ^bb32
    %135 = llvm.icmp "ne" %arg0, %35 : !llvm.ptr
    %136 = llvm.getelementptr inbounds %arg0[%0, 4] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.tutorial::Person", (struct<"class.google::protobuf::Message", (struct<"class.google::protobuf::MessageLite", (ptr, struct<"class.google::protobuf::internal::InternalMetadata", (ptr)>)>)>, struct<"class.google::protobuf::RepeatedPtrField", (struct<"class.google::protobuf::internal::RepeatedPtrFieldBase", (ptr, i32, i32, ptr)>)>, struct<"struct.google::protobuf::internal::ArenaStringPtr", (ptr)>, struct<"struct.google::protobuf::internal::ArenaStringPtr", (ptr)>, ptr, i32, struct<"class.google::protobuf::internal::CachedSize", (struct<"struct.std::atomic", (struct<"struct.std::__atomic_base", (i32)>)>)>)>
    %137 = llvm.load %136 {alignment = 8 : i64} : !llvm.ptr -> !llvm.ptr
    %138 = llvm.icmp "ne" %137, %37 : !llvm.ptr
    %139 = llvm.select %135, %138, %38 : i1, i1
    llvm.cond_br %139, ^bb33, ^bb43(%134 : !llvm.ptr)
  ^bb23(%140: !llvm.ptr, %141: i32):  // 2 preds: ^bb21, ^bb32
    %142 = llvm.load %132 {alignment = 8 : i64, tbaa = [#tbaa_tag12]} : !llvm.ptr -> !llvm.ptr
    %143 = llvm.icmp "ugt" %142, %140 : !llvm.ptr
    llvm.cond_br %143 weights([2000, 1]), ^bb25(%140 : !llvm.ptr), ^bb24
  ^bb24:  // pred: ^bb23
    %144 = llvm.call @_ZN6google8protobuf2io19EpsCopyOutputStream19EnsureSpaceFallbackEPh(%arg2, %140) : (!llvm.ptr, !llvm.ptr) -> !llvm.ptr
    llvm.br ^bb25(%144 : !llvm.ptr)
  ^bb25(%145: !llvm.ptr):  // 2 preds: ^bb23, ^bb24
    %146 = llvm.call @_ZNK6google8protobuf8internal20RepeatedPtrFieldBase3GetINS0_16RepeatedPtrFieldIN8tutorial18Person_PhoneNumberEE11TypeHandlerEEERKNT_4TypeEi(%133, %141) : (!llvm.ptr, i32) -> !llvm.ptr
    llvm.store %23, %145 {alignment = 1 : i64, tbaa = [#tbaa_tag1]} : i8, !llvm.ptr
    %147 = llvm.getelementptr inbounds %145[%9] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %148 = llvm.getelementptr inbounds %146[%0, 3, 0, 0, 0] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.tutorial::Person_PhoneNumber", (struct<"class.google::protobuf::Message", (struct<"class.google::protobuf::MessageLite", (ptr, struct<"class.google::protobuf::internal::InternalMetadata", (ptr)>)>)>, struct<"struct.google::protobuf::internal::ArenaStringPtr", (ptr)>, i32, struct<"class.google::protobuf::internal::CachedSize", (struct<"struct.std::atomic", (struct<"struct.std::__atomic_base", (i32)>)>)>)>
    %149 = llvm.load %148 atomic monotonic {alignment = 4 : i64} : !llvm.ptr -> i32
    %150 = llvm.icmp "ult" %149, %13 : i32
    %151 = llvm.trunc %149 : i32 to i8
    llvm.cond_br %150, ^bb26, ^bb27
  ^bb26:  // pred: ^bb25
    llvm.store %151, %147 {alignment = 1 : i64, tbaa = [#tbaa_tag1]} : i8, !llvm.ptr
    %152 = llvm.getelementptr inbounds %145[%10] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    llvm.br ^bb32(%152 : !llvm.ptr)
  ^bb27:  // pred: ^bb25
    %153 = llvm.or %151, %14  : i8
    llvm.store %153, %147 {alignment = 1 : i64, tbaa = [#tbaa_tag1]} : i8, !llvm.ptr
    %154 = llvm.lshr %149, %24  : i32
    %155 = llvm.icmp "ult" %149, %16 : i32
    llvm.cond_br %155, ^bb28, ^bb29
  ^bb28:  // pred: ^bb27
    %156 = llvm.trunc %154 : i32 to i8
    %157 = llvm.getelementptr inbounds %145[%10] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    llvm.store %156, %157 {alignment = 1 : i64, tbaa = [#tbaa_tag1]} : i8, !llvm.ptr
    %158 = llvm.getelementptr inbounds %145[%18] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    llvm.br ^bb32(%158 : !llvm.ptr)
  ^bb29:  // pred: ^bb27
    %159 = llvm.getelementptr inbounds %145[%10] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    llvm.br ^bb30(%154, %159 : i32, !llvm.ptr)
  ^bb30(%160: i32, %161: !llvm.ptr):  // 2 preds: ^bb29, ^bb30
    %162 = llvm.trunc %160 : i32 to i8
    %163 = llvm.or %162, %14  : i8
    llvm.store %163, %161 {alignment = 1 : i64, tbaa = [#tbaa_tag1]} : i8, !llvm.ptr
    %164 = llvm.lshr %160, %24  : i32
    %165 = llvm.getelementptr inbounds %161[%9] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %166 = llvm.icmp "ugt" %160, %25 : i32
    llvm.cond_br %166 weights([1, 2000]), ^bb30(%164, %165 : i32, !llvm.ptr), ^bb31 {loop_annotation = #loop_annotation}
  ^bb31:  // pred: ^bb30
    %167 = llvm.trunc %164 : i32 to i8
    %168 = llvm.getelementptr inbounds %161[%10] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    llvm.store %167, %165 {alignment = 1 : i64, tbaa = [#tbaa_tag1]} : i8, !llvm.ptr
    llvm.br ^bb32(%168 : !llvm.ptr)
  ^bb32(%169: !llvm.ptr):  // 3 preds: ^bb26, ^bb28, ^bb31
    %170 = llvm.call @_ZNK8tutorial18Person_PhoneNumber18_InternalSerializeEPhPN6google8protobuf2io19EpsCopyOutputStreamE(%146, %169, %arg2) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> !llvm.ptr
    %171 = llvm.add %141, %3 overflow<nuw>  : i32
    %172 = llvm.icmp "eq" %171, %130 : i32
    llvm.cond_br %172, ^bb22(%170 : !llvm.ptr), ^bb23(%170, %171 : !llvm.ptr, i32) {loop_annotation = #loop_annotation}
  ^bb33:  // pred: ^bb22
    %173 = llvm.getelementptr inbounds %arg2[%0, 0] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.google::protobuf::io::EpsCopyOutputStream", packed (ptr, ptr, array<32 x i8>, ptr, i8, i8, i8, array<5 x i8>)>
    %174 = llvm.load %173 {alignment = 8 : i64, tbaa = [#tbaa_tag12]} : !llvm.ptr -> !llvm.ptr
    %175 = llvm.icmp "ugt" %174, %134 : !llvm.ptr
    llvm.cond_br %175 weights([2000, 1]), ^bb35(%137, %134 : !llvm.ptr, !llvm.ptr), ^bb34
  ^bb34:  // pred: ^bb33
    %176 = llvm.call @_ZN6google8protobuf2io19EpsCopyOutputStream19EnsureSpaceFallbackEPh(%arg2, %134) : (!llvm.ptr, !llvm.ptr) -> !llvm.ptr
    %177 = llvm.load %136 {alignment = 8 : i64, tbaa = [#tbaa_tag23]} : !llvm.ptr -> !llvm.ptr
    llvm.br ^bb35(%177, %176 : !llvm.ptr, !llvm.ptr)
  ^bb35(%178: !llvm.ptr, %179: !llvm.ptr):  // 2 preds: ^bb33, ^bb34
    llvm.store %39, %179 {alignment = 1 : i64, tbaa = [#tbaa_tag1]} : i8, !llvm.ptr
    %180 = llvm.getelementptr inbounds %179[%9] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %181 = llvm.getelementptr inbounds %178[%0, 3, 0, 0, 0] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.google::protobuf::Timestamp", (struct<"class.google::protobuf::Message", (struct<"class.google::protobuf::MessageLite", (ptr, struct<"class.google::protobuf::internal::InternalMetadata", (ptr)>)>)>, i64, i32, struct<"class.google::protobuf::internal::CachedSize", (struct<"struct.std::atomic", (struct<"struct.std::__atomic_base", (i32)>)>)>)>
    %182 = llvm.load %181 atomic monotonic {alignment = 4 : i64} : !llvm.ptr -> i32
    %183 = llvm.icmp "ult" %182, %13 : i32
    %184 = llvm.trunc %182 : i32 to i8
    llvm.cond_br %183, ^bb36, ^bb37
  ^bb36:  // pred: ^bb35
    llvm.store %184, %180 {alignment = 1 : i64, tbaa = [#tbaa_tag1]} : i8, !llvm.ptr
    %185 = llvm.getelementptr inbounds %179[%10] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    llvm.br ^bb42(%185 : !llvm.ptr)
  ^bb37:  // pred: ^bb35
    %186 = llvm.or %184, %14  : i8
    llvm.store %186, %180 {alignment = 1 : i64, tbaa = [#tbaa_tag1]} : i8, !llvm.ptr
    %187 = llvm.lshr %182, %24  : i32
    %188 = llvm.icmp "ult" %182, %16 : i32
    llvm.cond_br %188, ^bb38, ^bb39
  ^bb38:  // pred: ^bb37
    %189 = llvm.trunc %187 : i32 to i8
    %190 = llvm.getelementptr inbounds %179[%10] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    llvm.store %189, %190 {alignment = 1 : i64, tbaa = [#tbaa_tag1]} : i8, !llvm.ptr
    %191 = llvm.getelementptr inbounds %179[%18] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    llvm.br ^bb42(%191 : !llvm.ptr)
  ^bb39:  // pred: ^bb37
    %192 = llvm.getelementptr inbounds %179[%10] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    llvm.br ^bb40(%187, %192 : i32, !llvm.ptr)
  ^bb40(%193: i32, %194: !llvm.ptr):  // 2 preds: ^bb39, ^bb40
    %195 = llvm.trunc %193 : i32 to i8
    %196 = llvm.or %195, %14  : i8
    llvm.store %196, %194 {alignment = 1 : i64, tbaa = [#tbaa_tag1]} : i8, !llvm.ptr
    %197 = llvm.lshr %193, %24  : i32
    %198 = llvm.getelementptr inbounds %194[%9] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %199 = llvm.icmp "ugt" %193, %25 : i32
    llvm.cond_br %199 weights([1, 2000]), ^bb40(%197, %198 : i32, !llvm.ptr), ^bb41 {loop_annotation = #loop_annotation}
  ^bb41:  // pred: ^bb40
    %200 = llvm.trunc %197 : i32 to i8
    %201 = llvm.getelementptr inbounds %194[%10] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    llvm.store %200, %198 {alignment = 1 : i64, tbaa = [#tbaa_tag1]} : i8, !llvm.ptr
    llvm.br ^bb42(%201 : !llvm.ptr)
  ^bb42(%202: !llvm.ptr):  // 3 preds: ^bb36, ^bb38, ^bb41
    %203 = llvm.call @_ZNK6google8protobuf9Timestamp18_InternalSerializeEPhPNS0_2io19EpsCopyOutputStreamE(%178, %202, %arg2) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> !llvm.ptr
    llvm.br ^bb43(%203 : !llvm.ptr)
  ^bb43(%204: !llvm.ptr):  // 2 preds: ^bb22, ^bb42
    %205 = llvm.getelementptr inbounds %arg0[%0, 0, 0, 1, 0] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.tutorial::Person", (struct<"class.google::protobuf::Message", (struct<"class.google::protobuf::MessageLite", (ptr, struct<"class.google::protobuf::internal::InternalMetadata", (ptr)>)>)>, struct<"class.google::protobuf::RepeatedPtrField", (struct<"class.google::protobuf::internal::RepeatedPtrFieldBase", (ptr, i32, i32, ptr)>)>, struct<"struct.google::protobuf::internal::ArenaStringPtr", (ptr)>, struct<"struct.google::protobuf::internal::ArenaStringPtr", (ptr)>, ptr, i32, struct<"class.google::protobuf::internal::CachedSize", (struct<"struct.std::atomic", (struct<"struct.std::__atomic_base", (i32)>)>)>)>
    %206 = llvm.load %205 {alignment = 8 : i64, tbaa = [#tbaa_tag5]} : !llvm.ptr -> !llvm.ptr
    %207 = llvm.ptrtoint %206 : !llvm.ptr to i64
    %208 = llvm.and %207, %9  : i64
    %209 = llvm.icmp "eq" %208, %0 : i64
    llvm.cond_br %209 weights([2000, 1]), ^bb45(%204 : !llvm.ptr), ^bb44
  ^bb44:  // pred: ^bb43
    %210 = llvm.and %207, %40  : i64
    %211 = llvm.inttoptr %210 : i64 to !llvm.ptr
    %212 = llvm.getelementptr inbounds %211[%0, 1] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.google::protobuf::internal::InternalMetadata::Container", (struct<"struct.google::protobuf::internal::InternalMetadata::ContainerBase", (ptr)>, struct<"class.google::protobuf::UnknownFieldSet", (struct<"class.std::vector", (struct<"struct.std::_Vector_base", (struct<"struct.std::_Vector_base<google::protobuf::UnknownField, std::allocator<google::protobuf::UnknownField>>::_Vector_impl", (struct<"struct.std::_Vector_base<google::protobuf::UnknownField, std::allocator<google::protobuf::UnknownField>>::_Vector_impl_data", (ptr, ptr, ptr)>)>)>)>)>)>
    %213 = llvm.call @_ZN6google8protobuf8internal10WireFormat37InternalSerializeUnknownFieldsToArrayERKNS0_15UnknownFieldSetEPhPNS0_2io19EpsCopyOutputStreamE(%212, %204, %arg2) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> !llvm.ptr
    llvm.br ^bb45(%213 : !llvm.ptr)
  ^bb45(%214: !llvm.ptr):  // 2 preds: ^bb43, ^bb44
    llvm.return %214 : !llvm.ptr
  }
  llvm.func unnamed_addr @_ZNK8tutorial6Person12ByteSizeLongEv(%arg0: !llvm.ptr {llvm.align = 8 : i64, llvm.dereferenceable = 72 : i64, llvm.nonnull, llvm.noundef}) -> (i64 {llvm.noundef}) attributes {alignment = 2 : i64, frame_pointer = #llvm.framePointerKind<none>, passthrough = [["uwtable", "2"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], ["tune-cpu", "generic"]], personality = @__gxx_personality_v0, target_cpu = "x86-64", target_features = #llvm.target_features<["+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>} {
    %0 = llvm.mlir.constant(0 : i64) : i64
    %1 = llvm.mlir.constant(1 : i32) : i32
    %2 = llvm.mlir.constant(0 : i32) : i32
    %3 = llvm.mlir.constant(3 : i32) : i32
    %4 = llvm.mlir.zero : !llvm.ptr
    %5 = llvm.mlir.constant(31 : i32) : i32
    %6 = llvm.mlir.constant(9 : i32) : i32
    %7 = llvm.mlir.constant(73 : i32) : i32
    %8 = llvm.mlir.constant(6 : i32) : i32
    %9 = llvm.mlir.constant(1 : i64) : i64
    %10 = llvm.mlir.constant(2 : i32) : i32
    %11 = llvm.mlir.constant(11 : i64) : i64
    %12 = llvm.mlir.constant(0 : i8) : i8
    %13 = llvm.mlir.constant(dense<0> : tensor<64xi8>) : !llvm.array<64 x i8>
    %14 = llvm.mlir.undef : !llvm.struct<"union.google::protobuf::internal::ExplicitlyConstructed<tutorial::Person>::AlignedUnion", (i64, array<64 x i8>)>
    %15 = llvm.insertvalue %0, %14[0] : !llvm.struct<"union.google::protobuf::internal::ExplicitlyConstructed<tutorial::Person>::AlignedUnion", (i64, array<64 x i8>)> 
    %16 = llvm.insertvalue %13, %15[1] : !llvm.struct<"union.google::protobuf::internal::ExplicitlyConstructed<tutorial::Person>::AlignedUnion", (i64, array<64 x i8>)> 
    %17 = llvm.mlir.undef : !llvm.struct<"class.google::protobuf::internal::ExplicitlyConstructed.0", (struct<"union.google::protobuf::internal::ExplicitlyConstructed<tutorial::Person>::AlignedUnion", (i64, array<64 x i8>)>)>
    %18 = llvm.insertvalue %16, %17[0] : !llvm.struct<"class.google::protobuf::internal::ExplicitlyConstructed.0", (struct<"union.google::protobuf::internal::ExplicitlyConstructed<tutorial::Person>::AlignedUnion", (i64, array<64 x i8>)>)> 
    %19 = llvm.mlir.undef : !llvm.struct<"class.tutorial::PersonDefaultTypeInternal", (struct<"class.google::protobuf::internal::ExplicitlyConstructed.0", (struct<"union.google::protobuf::internal::ExplicitlyConstructed<tutorial::Person>::AlignedUnion", (i64, array<64 x i8>)>)>)>
    %20 = llvm.insertvalue %18, %19[0] : !llvm.struct<"class.tutorial::PersonDefaultTypeInternal", (struct<"class.google::protobuf::internal::ExplicitlyConstructed.0", (struct<"union.google::protobuf::internal::ExplicitlyConstructed<tutorial::Person>::AlignedUnion", (i64, array<64 x i8>)>)>)> 
    %21 = llvm.mlir.addressof @_ZN8tutorial25_Person_default_instance_E : !llvm.ptr
    %22 = llvm.mlir.constant(4 : i32) : i32
    %23 = llvm.mlir.constant(false) : i1
    %24 = llvm.mlir.constant(5 : i32) : i32
    %25 = llvm.getelementptr inbounds %arg0[%0, 1, 0, 1] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.tutorial::Person", (struct<"class.google::protobuf::Message", (struct<"class.google::protobuf::MessageLite", (ptr, struct<"class.google::protobuf::internal::InternalMetadata", (ptr)>)>)>, struct<"class.google::protobuf::RepeatedPtrField", (struct<"class.google::protobuf::internal::RepeatedPtrFieldBase", (ptr, i32, i32, ptr)>)>, struct<"struct.google::protobuf::internal::ArenaStringPtr", (ptr)>, struct<"struct.google::protobuf::internal::ArenaStringPtr", (ptr)>, ptr, i32, struct<"class.google::protobuf::internal::CachedSize", (struct<"struct.std::atomic", (struct<"struct.std::__atomic_base", (i32)>)>)>)>
    %26 = llvm.load %25 {alignment = 8 : i64, tbaa = [#tbaa_tag17]} : !llvm.ptr -> i32
    %27 = llvm.sext %26 : i32 to i64
    %28 = llvm.getelementptr inbounds %arg0[%0, 1, 0, 3] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.tutorial::Person", (struct<"class.google::protobuf::Message", (struct<"class.google::protobuf::MessageLite", (ptr, struct<"class.google::protobuf::internal::InternalMetadata", (ptr)>)>)>, struct<"class.google::protobuf::RepeatedPtrField", (struct<"class.google::protobuf::internal::RepeatedPtrFieldBase", (ptr, i32, i32, ptr)>)>, struct<"struct.google::protobuf::internal::ArenaStringPtr", (ptr)>, struct<"struct.google::protobuf::internal::ArenaStringPtr", (ptr)>, ptr, i32, struct<"class.google::protobuf::internal::CachedSize", (struct<"struct.std::atomic", (struct<"struct.std::__atomic_base", (i32)>)>)>)>
    %29 = llvm.load %28 {alignment = 8 : i64, tbaa = [#tbaa_tag15]} : !llvm.ptr -> !llvm.ptr
    %30 = llvm.icmp "eq" %29, %4 : !llvm.ptr
    %31 = llvm.getelementptr inbounds %29[%0, 1, %0] : (!llvm.ptr, i64, i64) -> !llvm.ptr, !llvm.struct<"struct.google::protobuf::internal::RepeatedPtrFieldBase::Rep", (i32, array<1 x ptr>)>
    %32 = llvm.select %30, %4, %31 : i1, !llvm.ptr
    %33 = llvm.getelementptr inbounds %32[%27] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.ptr
    %34 = llvm.icmp "eq" %26, %2 : i32
    llvm.cond_br %34, ^bb1(%0 : i64), ^bb2(%27, %32 : i64, !llvm.ptr)
  ^bb1(%35: i64):  // 2 preds: ^bb0, ^bb11
    %36 = llvm.getelementptr inbounds %arg0[%0, 2, 0] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.tutorial::Person", (struct<"class.google::protobuf::Message", (struct<"class.google::protobuf::MessageLite", (ptr, struct<"class.google::protobuf::internal::InternalMetadata", (ptr)>)>)>, struct<"class.google::protobuf::RepeatedPtrField", (struct<"class.google::protobuf::internal::RepeatedPtrFieldBase", (ptr, i32, i32, ptr)>)>, struct<"struct.google::protobuf::internal::ArenaStringPtr", (ptr)>, struct<"struct.google::protobuf::internal::ArenaStringPtr", (ptr)>, ptr, i32, struct<"class.google::protobuf::internal::CachedSize", (struct<"struct.std::atomic", (struct<"struct.std::__atomic_base", (i32)>)>)>)>
    %37 = llvm.load %36 {alignment = 8 : i64, tbaa = [#tbaa_tag8]} : !llvm.ptr -> !llvm.ptr
    %38 = llvm.getelementptr inbounds %37[%0, 1] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.std::__cxx11::basic_string", (struct<"struct.std::__cxx11::basic_string<char>::_Alloc_hider", (ptr)>, i64, struct<"union.anon", (i64, array<8 x i8>)>)>
    %39 = llvm.load %38 {alignment = 8 : i64, tbaa = [#tbaa_tag25]} : !llvm.ptr -> i64
    %40 = llvm.icmp "eq" %39, %0 : i64
    llvm.cond_br %40, ^bb13(%35 : i64), ^bb12
  ^bb2(%41: i64, %42: !llvm.ptr):  // 2 preds: ^bb0, ^bb11
    %43 = llvm.bitcast %42 : !llvm.ptr to !llvm.ptr
    %44 = llvm.load %43 {alignment = 8 : i64, tbaa = [#tbaa_tag2]} : !llvm.ptr -> !llvm.ptr
    %45 = llvm.getelementptr inbounds %44[%0, 1, 0] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.tutorial::Person_PhoneNumber", (struct<"class.google::protobuf::Message", (struct<"class.google::protobuf::MessageLite", (ptr, struct<"class.google::protobuf::internal::InternalMetadata", (ptr)>)>)>, struct<"struct.google::protobuf::internal::ArenaStringPtr", (ptr)>, i32, struct<"class.google::protobuf::internal::CachedSize", (struct<"struct.std::atomic", (struct<"struct.std::__atomic_base", (i32)>)>)>)>
    %46 = llvm.load %45 {alignment = 8 : i64, tbaa = [#tbaa_tag8]} : !llvm.ptr -> !llvm.ptr
    %47 = llvm.getelementptr inbounds %46[%0, 1] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.std::__cxx11::basic_string", (struct<"struct.std::__cxx11::basic_string<char>::_Alloc_hider", (ptr)>, i64, struct<"union.anon", (i64, array<8 x i8>)>)>
    %48 = llvm.load %47 {alignment = 8 : i64, tbaa = [#tbaa_tag25]} : !llvm.ptr -> i64
    %49 = llvm.icmp "eq" %48, %0 : i64
    llvm.cond_br %49, ^bb4(%0 : i64), ^bb3
  ^bb3:  // pred: ^bb2
    %50 = llvm.trunc %48 : i64 to i32
    %51 = llvm.or %50, %1  : i32
    %52 = "llvm.intr.ctlz"(%51) <{is_zero_poison = true}> : (i32) -> i32
    %53 = llvm.xor %52, %5  : i32
    %54 = llvm.mul %53, %6 overflow<nsw, nuw>  : i32
    %55 = llvm.add %54, %7 overflow<nsw, nuw>  : i32
    %56 = llvm.lshr %55, %8  : i32
    %57 = llvm.zext %56 : i32 to i64
    %58 = llvm.add %48, %9  : i64
    %59 = llvm.add %58, %57  : i64
    llvm.br ^bb4(%59 : i64)
  ^bb4(%60: i64):  // 2 preds: ^bb2, ^bb3
    %61 = llvm.getelementptr inbounds %44[%0, 2] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.tutorial::Person_PhoneNumber", (struct<"class.google::protobuf::Message", (struct<"class.google::protobuf::MessageLite", (ptr, struct<"class.google::protobuf::internal::InternalMetadata", (ptr)>)>)>, struct<"struct.google::protobuf::internal::ArenaStringPtr", (ptr)>, i32, struct<"class.google::protobuf::internal::CachedSize", (struct<"struct.std::atomic", (struct<"struct.std::__atomic_base", (i32)>)>)>)>
    %62 = llvm.load %61 {alignment = 8 : i64, tbaa = [#tbaa_tag24]} : !llvm.ptr -> i32
    %63 = llvm.icmp "eq" %62, %2 : i32
    llvm.cond_br %63, ^bb8(%60 : i64), ^bb5
  ^bb5:  // pred: ^bb4
    %64 = llvm.icmp "slt" %62, %2 : i32
    llvm.cond_br %64, ^bb7(%11 : i64), ^bb6
  ^bb6:  // pred: ^bb5
    %65 = llvm.or %62, %1  : i32
    %66 = "llvm.intr.ctlz"(%65) <{is_zero_poison = true}> : (i32) -> i32
    %67 = llvm.xor %66, %5  : i32
    %68 = llvm.mul %67, %6 overflow<nsw, nuw>  : i32
    %69 = llvm.add %68, %7 overflow<nsw, nuw>  : i32
    %70 = llvm.lshr %69, %8  : i32
    %71 = llvm.add %70, %1 overflow<nsw, nuw>  : i32
    %72 = llvm.zext %71 : i32 to i64
    llvm.br ^bb7(%72 : i64)
  ^bb7(%73: i64):  // 2 preds: ^bb5, ^bb6
    %74 = llvm.add %73, %60  : i64
    llvm.br ^bb8(%74 : i64)
  ^bb8(%75: i64):  // 2 preds: ^bb4, ^bb7
    %76 = llvm.getelementptr inbounds %44[%0, 0, 0, 1] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.tutorial::Person_PhoneNumber", (struct<"class.google::protobuf::Message", (struct<"class.google::protobuf::MessageLite", (ptr, struct<"class.google::protobuf::internal::InternalMetadata", (ptr)>)>)>, struct<"struct.google::protobuf::internal::ArenaStringPtr", (ptr)>, i32, struct<"class.google::protobuf::internal::CachedSize", (struct<"struct.std::atomic", (struct<"struct.std::__atomic_base", (i32)>)>)>)>
    %77 = llvm.getelementptr inbounds %76[%0, 0] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.google::protobuf::internal::InternalMetadata", (ptr)>
    %78 = llvm.load %77 {alignment = 8 : i64, tbaa = [#tbaa_tag5]} : !llvm.ptr -> !llvm.ptr
    %79 = llvm.ptrtoint %78 : !llvm.ptr to i64
    %80 = llvm.and %79, %9  : i64
    %81 = llvm.icmp "eq" %80, %0 : i64
    llvm.cond_br %81 weights([2000, 1]), ^bb10, ^bb9
  ^bb9:  // pred: ^bb8
    %82 = llvm.getelementptr inbounds %44[%0, 3] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.tutorial::Person_PhoneNumber", (struct<"class.google::protobuf::Message", (struct<"class.google::protobuf::MessageLite", (ptr, struct<"class.google::protobuf::internal::InternalMetadata", (ptr)>)>)>, struct<"struct.google::protobuf::internal::ArenaStringPtr", (ptr)>, i32, struct<"class.google::protobuf::internal::CachedSize", (struct<"struct.std::atomic", (struct<"struct.std::__atomic_base", (i32)>)>)>)>
    %83 = llvm.call @_ZN6google8protobuf8internal24ComputeUnknownFieldsSizeERKNS1_16InternalMetadataEmPNS1_10CachedSizeE(%76, %75, %82) : (!llvm.ptr, i64, !llvm.ptr) -> i64
    %84 = llvm.trunc %83 : i64 to i32
    llvm.br ^bb11(%84, %83 : i32, i64)
  ^bb10:  // pred: ^bb8
    %85 = llvm.trunc %75 : i64 to i32
    %86 = llvm.getelementptr inbounds %44[%0, 3, 0, 0, 0] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.tutorial::Person_PhoneNumber", (struct<"class.google::protobuf::Message", (struct<"class.google::protobuf::MessageLite", (ptr, struct<"class.google::protobuf::internal::InternalMetadata", (ptr)>)>)>, struct<"struct.google::protobuf::internal::ArenaStringPtr", (ptr)>, i32, struct<"class.google::protobuf::internal::CachedSize", (struct<"struct.std::atomic", (struct<"struct.std::__atomic_base", (i32)>)>)>)>
    llvm.store %85, %86 atomic monotonic {alignment = 4 : i64} : i32, !llvm.ptr
    llvm.br ^bb11(%85, %75 : i32, i64)
  ^bb11(%87: i32, %88: i64):  // 2 preds: ^bb9, ^bb10
    %89 = llvm.or %87, %1  : i32
    %90 = "llvm.intr.ctlz"(%89) <{is_zero_poison = true}> : (i32) -> i32
    %91 = llvm.xor %90, %5  : i32
    %92 = llvm.mul %91, %6 overflow<nsw, nuw>  : i32
    %93 = llvm.add %92, %7 overflow<nsw, nuw>  : i32
    %94 = llvm.lshr %93, %8  : i32
    %95 = llvm.zext %94 : i32 to i64
    %96 = llvm.add %88, %41  : i64
    %97 = llvm.add %96, %95  : i64
    %98 = llvm.getelementptr inbounds %42[%9] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.ptr
    %99 = llvm.icmp "eq" %98, %33 : !llvm.ptr
    llvm.cond_br %99, ^bb1(%97 : i64), ^bb2(%97, %98 : i64, !llvm.ptr)
  ^bb12:  // pred: ^bb1
    %100 = llvm.trunc %39 : i64 to i32
    %101 = llvm.or %100, %1  : i32
    %102 = "llvm.intr.ctlz"(%101) <{is_zero_poison = true}> : (i32) -> i32
    %103 = llvm.xor %102, %5  : i32
    %104 = llvm.mul %103, %6 overflow<nsw, nuw>  : i32
    %105 = llvm.add %104, %7 overflow<nsw, nuw>  : i32
    %106 = llvm.lshr %105, %8  : i32
    %107 = llvm.zext %106 : i32 to i64
    %108 = llvm.add %35, %9  : i64
    %109 = llvm.add %108, %39  : i64
    %110 = llvm.add %109, %107  : i64
    llvm.br ^bb13(%110 : i64)
  ^bb13(%111: i64):  // 2 preds: ^bb1, ^bb12
    %112 = llvm.getelementptr inbounds %arg0[%0, 3, 0] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.tutorial::Person", (struct<"class.google::protobuf::Message", (struct<"class.google::protobuf::MessageLite", (ptr, struct<"class.google::protobuf::internal::InternalMetadata", (ptr)>)>)>, struct<"class.google::protobuf::RepeatedPtrField", (struct<"class.google::protobuf::internal::RepeatedPtrFieldBase", (ptr, i32, i32, ptr)>)>, struct<"struct.google::protobuf::internal::ArenaStringPtr", (ptr)>, struct<"struct.google::protobuf::internal::ArenaStringPtr", (ptr)>, ptr, i32, struct<"class.google::protobuf::internal::CachedSize", (struct<"struct.std::atomic", (struct<"struct.std::__atomic_base", (i32)>)>)>)>
    %113 = llvm.load %112 {alignment = 8 : i64, tbaa = [#tbaa_tag8]} : !llvm.ptr -> !llvm.ptr
    %114 = llvm.getelementptr inbounds %113[%0, 1] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.std::__cxx11::basic_string", (struct<"struct.std::__cxx11::basic_string<char>::_Alloc_hider", (ptr)>, i64, struct<"union.anon", (i64, array<8 x i8>)>)>
    %115 = llvm.load %114 {alignment = 8 : i64, tbaa = [#tbaa_tag25]} : !llvm.ptr -> i64
    %116 = llvm.icmp "eq" %115, %0 : i64
    llvm.cond_br %116, ^bb15(%111 : i64), ^bb14
  ^bb14:  // pred: ^bb13
    %117 = llvm.trunc %115 : i64 to i32
    %118 = llvm.or %117, %1  : i32
    %119 = "llvm.intr.ctlz"(%118) <{is_zero_poison = true}> : (i32) -> i32
    %120 = llvm.xor %119, %5  : i32
    %121 = llvm.mul %120, %6 overflow<nsw, nuw>  : i32
    %122 = llvm.add %121, %7 overflow<nsw, nuw>  : i32
    %123 = llvm.lshr %122, %8  : i32
    %124 = llvm.zext %123 : i32 to i64
    %125 = llvm.add %111, %9  : i64
    %126 = llvm.add %125, %115  : i64
    %127 = llvm.add %126, %124  : i64
    llvm.br ^bb15(%127 : i64)
  ^bb15(%128: i64):  // 2 preds: ^bb13, ^bb14
    %129 = llvm.icmp "ne" %arg0, %21 : !llvm.ptr
    %130 = llvm.getelementptr inbounds %arg0[%0, 4] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.tutorial::Person", (struct<"class.google::protobuf::Message", (struct<"class.google::protobuf::MessageLite", (ptr, struct<"class.google::protobuf::internal::InternalMetadata", (ptr)>)>)>, struct<"class.google::protobuf::RepeatedPtrField", (struct<"class.google::protobuf::internal::RepeatedPtrFieldBase", (ptr, i32, i32, ptr)>)>, struct<"struct.google::protobuf::internal::ArenaStringPtr", (ptr)>, struct<"struct.google::protobuf::internal::ArenaStringPtr", (ptr)>, ptr, i32, struct<"class.google::protobuf::internal::CachedSize", (struct<"struct.std::atomic", (struct<"struct.std::__atomic_base", (i32)>)>)>)>
    %131 = llvm.load %130 {alignment = 8 : i64} : !llvm.ptr -> !llvm.ptr
    %132 = llvm.icmp "ne" %131, %4 : !llvm.ptr
    %133 = llvm.select %129, %132, %23 : i1, i1
    llvm.cond_br %133, ^bb16, ^bb17(%128 : i64)
  ^bb16:  // pred: ^bb15
    %134 = llvm.call @_ZNK6google8protobuf9Timestamp12ByteSizeLongEv(%131) : (!llvm.ptr) -> i64
    %135 = llvm.trunc %134 : i64 to i32
    %136 = llvm.or %135, %1  : i32
    %137 = "llvm.intr.ctlz"(%136) <{is_zero_poison = true}> : (i32) -> i32
    %138 = llvm.xor %137, %5  : i32
    %139 = llvm.mul %138, %6 overflow<nsw, nuw>  : i32
    %140 = llvm.add %139, %7 overflow<nsw, nuw>  : i32
    %141 = llvm.lshr %140, %8  : i32
    %142 = llvm.zext %141 : i32 to i64
    %143 = llvm.add %128, %9  : i64
    %144 = llvm.add %143, %134  : i64
    %145 = llvm.add %144, %142  : i64
    llvm.br ^bb17(%145 : i64)
  ^bb17(%146: i64):  // 2 preds: ^bb15, ^bb16
    %147 = llvm.getelementptr inbounds %arg0[%0, 5] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.tutorial::Person", (struct<"class.google::protobuf::Message", (struct<"class.google::protobuf::MessageLite", (ptr, struct<"class.google::protobuf::internal::InternalMetadata", (ptr)>)>)>, struct<"class.google::protobuf::RepeatedPtrField", (struct<"class.google::protobuf::internal::RepeatedPtrFieldBase", (ptr, i32, i32, ptr)>)>, struct<"struct.google::protobuf::internal::ArenaStringPtr", (ptr)>, struct<"struct.google::protobuf::internal::ArenaStringPtr", (ptr)>, ptr, i32, struct<"class.google::protobuf::internal::CachedSize", (struct<"struct.std::atomic", (struct<"struct.std::__atomic_base", (i32)>)>)>)>
    %148 = llvm.load %147 {alignment = 8 : i64, tbaa = [#tbaa_tag28]} : !llvm.ptr -> i32
    %149 = llvm.icmp "eq" %148, %2 : i32
    llvm.cond_br %149, ^bb21(%146 : i64), ^bb18
  ^bb18:  // pred: ^bb17
    %150 = llvm.icmp "slt" %148, %2 : i32
    llvm.cond_br %150, ^bb20(%11 : i64), ^bb19
  ^bb19:  // pred: ^bb18
    %151 = llvm.or %148, %1  : i32
    %152 = "llvm.intr.ctlz"(%151) <{is_zero_poison = true}> : (i32) -> i32
    %153 = llvm.xor %152, %5  : i32
    %154 = llvm.mul %153, %6 overflow<nsw, nuw>  : i32
    %155 = llvm.add %154, %7 overflow<nsw, nuw>  : i32
    %156 = llvm.lshr %155, %8  : i32
    %157 = llvm.add %156, %1 overflow<nsw, nuw>  : i32
    %158 = llvm.zext %157 : i32 to i64
    llvm.br ^bb20(%158 : i64)
  ^bb20(%159: i64):  // 2 preds: ^bb18, ^bb19
    %160 = llvm.add %159, %146  : i64
    llvm.br ^bb21(%160 : i64)
  ^bb21(%161: i64):  // 2 preds: ^bb17, ^bb20
    %162 = llvm.getelementptr inbounds %arg0[%0, 0, 0, 1] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.tutorial::Person", (struct<"class.google::protobuf::Message", (struct<"class.google::protobuf::MessageLite", (ptr, struct<"class.google::protobuf::internal::InternalMetadata", (ptr)>)>)>, struct<"class.google::protobuf::RepeatedPtrField", (struct<"class.google::protobuf::internal::RepeatedPtrFieldBase", (ptr, i32, i32, ptr)>)>, struct<"struct.google::protobuf::internal::ArenaStringPtr", (ptr)>, struct<"struct.google::protobuf::internal::ArenaStringPtr", (ptr)>, ptr, i32, struct<"class.google::protobuf::internal::CachedSize", (struct<"struct.std::atomic", (struct<"struct.std::__atomic_base", (i32)>)>)>)>
    %163 = llvm.getelementptr inbounds %162[%0, 0] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.google::protobuf::internal::InternalMetadata", (ptr)>
    %164 = llvm.load %163 {alignment = 8 : i64, tbaa = [#tbaa_tag5]} : !llvm.ptr -> !llvm.ptr
    %165 = llvm.ptrtoint %164 : !llvm.ptr to i64
    %166 = llvm.and %165, %9  : i64
    %167 = llvm.icmp "eq" %166, %0 : i64
    llvm.cond_br %167 weights([2000, 1]), ^bb23, ^bb22
  ^bb22:  // pred: ^bb21
    %168 = llvm.getelementptr inbounds %arg0[%0, 6] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.tutorial::Person", (struct<"class.google::protobuf::Message", (struct<"class.google::protobuf::MessageLite", (ptr, struct<"class.google::protobuf::internal::InternalMetadata", (ptr)>)>)>, struct<"class.google::protobuf::RepeatedPtrField", (struct<"class.google::protobuf::internal::RepeatedPtrFieldBase", (ptr, i32, i32, ptr)>)>, struct<"struct.google::protobuf::internal::ArenaStringPtr", (ptr)>, struct<"struct.google::protobuf::internal::ArenaStringPtr", (ptr)>, ptr, i32, struct<"class.google::protobuf::internal::CachedSize", (struct<"struct.std::atomic", (struct<"struct.std::__atomic_base", (i32)>)>)>)>
    %169 = llvm.call @_ZN6google8protobuf8internal24ComputeUnknownFieldsSizeERKNS1_16InternalMetadataEmPNS1_10CachedSizeE(%162, %161, %168) : (!llvm.ptr, i64, !llvm.ptr) -> i64
    llvm.br ^bb24(%169 : i64)
  ^bb23:  // pred: ^bb21
    %170 = llvm.trunc %161 : i64 to i32
    %171 = llvm.getelementptr inbounds %arg0[%0, 6, 0, 0, 0] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.tutorial::Person", (struct<"class.google::protobuf::Message", (struct<"class.google::protobuf::MessageLite", (ptr, struct<"class.google::protobuf::internal::InternalMetadata", (ptr)>)>)>, struct<"class.google::protobuf::RepeatedPtrField", (struct<"class.google::protobuf::internal::RepeatedPtrFieldBase", (ptr, i32, i32, ptr)>)>, struct<"struct.google::protobuf::internal::ArenaStringPtr", (ptr)>, struct<"struct.google::protobuf::internal::ArenaStringPtr", (ptr)>, ptr, i32, struct<"class.google::protobuf::internal::CachedSize", (struct<"struct.std::atomic", (struct<"struct.std::__atomic_base", (i32)>)>)>)>
    llvm.store %170, %171 atomic monotonic {alignment = 4 : i64} : i32, !llvm.ptr
    llvm.br ^bb24(%161 : i64)
  ^bb24(%172: i64):  // 2 preds: ^bb22, ^bb23
    llvm.return %172 : i64
  }
  llvm.func unnamed_addr @_ZN8tutorial6Person9MergeFromERKN6google8protobuf7MessageE(%arg0: !llvm.ptr {llvm.align = 8 : i64, llvm.dereferenceable = 72 : i64, llvm.nonnull, llvm.noundef}, %arg1: !llvm.ptr {llvm.align = 8 : i64, llvm.dereferenceable = 16 : i64, llvm.nonnull, llvm.noundef}) attributes {alignment = 2 : i64, frame_pointer = #llvm.framePointerKind<none>, passthrough = ["mustprogress", ["uwtable", "2"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], ["tune-cpu", "generic"]], personality = @__gxx_personality_v0, target_cpu = "x86-64", target_features = #llvm.target_features<["+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>} {
    %0 = llvm.mlir.constant(1 : i32) : i32
    %1 = llvm.mlir.constant(0 : i64) : i64
    %2 = llvm.mlir.constant(0 : i32) : i32
    %3 = llvm.mlir.constant(3 : i32) : i32
    %4 = llvm.mlir.constant("build/addressbook.pb.cc\00") : !llvm.array<24 x i8>
    %5 = llvm.mlir.addressof @".str.5" : !llvm.ptr
    %6 = llvm.mlir.constant(705 : i32) : i32
    %7 = llvm.mlir.constant("CHECK failed: (&from) != (this): \00") : !llvm.array<34 x i8>
    %8 = llvm.mlir.addressof @".str.6" : !llvm.ptr
    %9 = llvm.mlir.addressof @_ZTIN6google8protobuf7MessageE : !llvm.ptr
    %10 = llvm.mlir.constant("N8tutorial6PersonE\00") : !llvm.array<19 x i8>
    %11 = llvm.mlir.addressof @_ZTSN8tutorial6PersonE : !llvm.ptr
    %12 = llvm.mlir.constant(2 : i64) : i64
    %13 = llvm.mlir.addressof @_ZTVN10__cxxabiv120__si_class_type_infoE : !llvm.ptr
    %14 = llvm.getelementptr inbounds %13[%12] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.ptr
    %15 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, ptr)>
    %16 = llvm.insertvalue %14, %15[0] : !llvm.struct<(ptr, ptr, ptr)> 
    %17 = llvm.insertvalue %11, %16[1] : !llvm.struct<(ptr, ptr, ptr)> 
    %18 = llvm.insertvalue %9, %17[2] : !llvm.struct<(ptr, ptr, ptr)> 
    %19 = llvm.mlir.addressof @_ZTIN8tutorial6PersonE : !llvm.ptr
    %20 = llvm.mlir.zero : !llvm.ptr
    %21 = llvm.alloca %0 x !llvm.struct<"class.google::protobuf::internal::LogMessage", (i32, ptr, i32, struct<"class.std::__cxx11::basic_string", (struct<"struct.std::__cxx11::basic_string<char>::_Alloc_hider", (ptr)>, i64, struct<"union.anon", (i64, array<8 x i8>)>)>)> {alignment = 8 : i64} : (i32) -> !llvm.ptr
    %22 = llvm.alloca %0 x !llvm.struct<"class.google::protobuf::internal::LogFinisher", (i8)> {alignment = 1 : i64} : (i32) -> !llvm.ptr
    %23 = llvm.getelementptr inbounds %arg0[%1, 0] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.tutorial::Person", (struct<"class.google::protobuf::Message", (struct<"class.google::protobuf::MessageLite", (ptr, struct<"class.google::protobuf::internal::InternalMetadata", (ptr)>)>)>, struct<"class.google::protobuf::RepeatedPtrField", (struct<"class.google::protobuf::internal::RepeatedPtrFieldBase", (ptr, i32, i32, ptr)>)>, struct<"struct.google::protobuf::internal::ArenaStringPtr", (ptr)>, struct<"struct.google::protobuf::internal::ArenaStringPtr", (ptr)>, ptr, i32, struct<"class.google::protobuf::internal::CachedSize", (struct<"struct.std::atomic", (struct<"struct.std::__atomic_base", (i32)>)>)>)>
    %24 = llvm.icmp "eq" %23, %arg1 : !llvm.ptr
    %25 = llvm.getelementptr inbounds %22[%1, 0] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.google::protobuf::internal::LogFinisher", (i8)>
    llvm.intr.lifetime.start 1, %25 : !llvm.ptr
    llvm.cond_br %24, ^bb1, ^bb3
  ^bb1:  // pred: ^bb0
    %26 = llvm.bitcast %21 : !llvm.ptr to !llvm.ptr
    llvm.intr.lifetime.start 56, %26 : !llvm.ptr
    llvm.call @_ZN6google8protobuf8internal10LogMessageC1ENS0_8LogLevelEPKci(%21, %3, %5, %6) : (!llvm.ptr, i32, !llvm.ptr, i32) -> ()
    %27 = llvm.invoke @_ZN6google8protobuf8internal10LogMessagelsEPKc(%21, %8) to ^bb2 unwind ^bb7 : (!llvm.ptr, !llvm.ptr) -> !llvm.ptr
  ^bb2:  // pred: ^bb1
    llvm.invoke @_ZN6google8protobuf8internal11LogFinisheraSERNS1_10LogMessageE(%22, %27) to ^bb4 unwind ^bb8 : (!llvm.ptr, !llvm.ptr) -> ()
  ^bb3:  // pred: ^bb0
    llvm.intr.lifetime.end 1, %25 : !llvm.ptr
    llvm.br ^bb5
  ^bb4:  // pred: ^bb2
    llvm.intr.lifetime.end 1, %25 : !llvm.ptr
    llvm.call @_ZN6google8protobuf8internal10LogMessageD1Ev(%21) : (!llvm.ptr) -> ()
    llvm.intr.lifetime.end 56, %26 : !llvm.ptr
    llvm.br ^bb5
  ^bb5:  // 2 preds: ^bb3, ^bb4
    %28 = llvm.bitcast %arg1 : !llvm.ptr to !llvm.ptr
    %29 = llvm.call @__dynamic_cast(%28, %9, %19, %1) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i64) -> !llvm.ptr
    %30 = llvm.icmp "eq" %29, %20 : !llvm.ptr
    llvm.cond_br %30, ^bb6, ^bb10
  ^bb6:  // pred: ^bb5
    llvm.call @_ZN6google8protobuf8internal13ReflectionOps5MergeERKNS0_7MessageEPS3_(%arg1, %23) : (!llvm.ptr, !llvm.ptr) -> ()
    llvm.br ^bb11
  ^bb7:  // pred: ^bb1
    %31 = llvm.landingpad cleanup : !llvm.struct<(ptr, i32)>
    llvm.br ^bb9(%31 : !llvm.struct<(ptr, i32)>)
  ^bb8:  // pred: ^bb2
    %32 = llvm.landingpad cleanup : !llvm.struct<(ptr, i32)>
    llvm.intr.lifetime.end 1, %25 : !llvm.ptr
    llvm.br ^bb9(%32 : !llvm.struct<(ptr, i32)>)
  ^bb9(%33: !llvm.struct<(ptr, i32)>):  // 2 preds: ^bb7, ^bb8
    llvm.call @_ZN6google8protobuf8internal10LogMessageD1Ev(%21) : (!llvm.ptr) -> ()
    llvm.intr.lifetime.end 56, %26 : !llvm.ptr
    llvm.resume %33 : !llvm.struct<(ptr, i32)>
  ^bb10:  // pred: ^bb5
    %34 = llvm.bitcast %29 : !llvm.ptr to !llvm.ptr
    llvm.call @_ZN8tutorial6Person9MergeFromERKS0_(%arg0, %34) : (!llvm.ptr, !llvm.ptr) -> ()
    llvm.br ^bb11
  ^bb11:  // 2 preds: ^bb6, ^bb10
    llvm.return
  }
  llvm.func local_unnamed_addr @_ZN8tutorial6Person9MergeFromERKS0_(%arg0: !llvm.ptr {llvm.align = 8 : i64, llvm.dereferenceable = 72 : i64, llvm.nonnull, llvm.noundef}, %arg1: !llvm.ptr {llvm.align = 8 : i64, llvm.dereferenceable = 72 : i64, llvm.nonnull, llvm.noundef}) attributes {alignment = 2 : i64, frame_pointer = #llvm.framePointerKind<none>, passthrough = ["mustprogress", ["uwtable", "2"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], ["tune-cpu", "generic"]], personality = @__gxx_personality_v0, target_cpu = "x86-64", target_features = #llvm.target_features<["+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>} {
    %0 = llvm.mlir.constant(1 : i32) : i32
    %1 = llvm.mlir.constant(0 : i64) : i64
    %2 = llvm.mlir.constant(0 : i32) : i32
    %3 = llvm.mlir.constant(3 : i32) : i32
    %4 = llvm.mlir.constant("build/addressbook.pb.cc\00") : !llvm.array<24 x i8>
    %5 = llvm.mlir.addressof @".str.5" : !llvm.ptr
    %6 = llvm.mlir.constant(720 : i32) : i32
    %7 = llvm.mlir.constant("CHECK failed: (&from) != (this): \00") : !llvm.array<34 x i8>
    %8 = llvm.mlir.addressof @".str.6" : !llvm.ptr
    %9 = llvm.mlir.constant(1 : i64) : i64
    %10 = llvm.mlir.constant(-2 : i64) : i64
    %11 = llvm.mlir.constant(2 : i32) : i32
    %12 = llvm.mlir.addressof @_ZN6google8protobuf8internal26fixed_address_empty_stringB5cxx11E : !llvm.ptr
    %13 = llvm.mlir.constant(0 : i8) : i8
    %14 = llvm.mlir.constant(dense<0> : tensor<64xi8>) : !llvm.array<64 x i8>
    %15 = llvm.mlir.undef : !llvm.struct<"union.google::protobuf::internal::ExplicitlyConstructed<tutorial::Person>::AlignedUnion", (i64, array<64 x i8>)>
    %16 = llvm.insertvalue %1, %15[0] : !llvm.struct<"union.google::protobuf::internal::ExplicitlyConstructed<tutorial::Person>::AlignedUnion", (i64, array<64 x i8>)> 
    %17 = llvm.insertvalue %14, %16[1] : !llvm.struct<"union.google::protobuf::internal::ExplicitlyConstructed<tutorial::Person>::AlignedUnion", (i64, array<64 x i8>)> 
    %18 = llvm.mlir.undef : !llvm.struct<"class.google::protobuf::internal::ExplicitlyConstructed.0", (struct<"union.google::protobuf::internal::ExplicitlyConstructed<tutorial::Person>::AlignedUnion", (i64, array<64 x i8>)>)>
    %19 = llvm.insertvalue %17, %18[0] : !llvm.struct<"class.google::protobuf::internal::ExplicitlyConstructed.0", (struct<"union.google::protobuf::internal::ExplicitlyConstructed<tutorial::Person>::AlignedUnion", (i64, array<64 x i8>)>)> 
    %20 = llvm.mlir.undef : !llvm.struct<"class.tutorial::PersonDefaultTypeInternal", (struct<"class.google::protobuf::internal::ExplicitlyConstructed.0", (struct<"union.google::protobuf::internal::ExplicitlyConstructed<tutorial::Person>::AlignedUnion", (i64, array<64 x i8>)>)>)>
    %21 = llvm.insertvalue %19, %20[0] : !llvm.struct<"class.tutorial::PersonDefaultTypeInternal", (struct<"class.google::protobuf::internal::ExplicitlyConstructed.0", (struct<"union.google::protobuf::internal::ExplicitlyConstructed<tutorial::Person>::AlignedUnion", (i64, array<64 x i8>)>)>)> 
    %22 = llvm.mlir.addressof @_ZN8tutorial25_Person_default_instance_E : !llvm.ptr
    %23 = llvm.mlir.constant(4 : i32) : i32
    %24 = llvm.mlir.zero : !llvm.ptr
    %25 = llvm.mlir.constant(false) : i1
    %26 = llvm.mlir.addressof @_ZN6google8protobuf28_Timestamp_default_instance_E : !llvm.ptr
    %27 = llvm.mlir.constant(5 : i32) : i32
    %28 = llvm.alloca %0 x !llvm.struct<"class.google::protobuf::internal::LogMessage", (i32, ptr, i32, struct<"class.std::__cxx11::basic_string", (struct<"struct.std::__cxx11::basic_string<char>::_Alloc_hider", (ptr)>, i64, struct<"union.anon", (i64, array<8 x i8>)>)>)> {alignment = 8 : i64} : (i32) -> !llvm.ptr
    %29 = llvm.alloca %0 x !llvm.struct<"class.google::protobuf::internal::LogFinisher", (i8)> {alignment = 1 : i64} : (i32) -> !llvm.ptr
    %30 = llvm.icmp "eq" %arg1, %arg0 : !llvm.ptr
    %31 = llvm.getelementptr inbounds %29[%1, 0] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.google::protobuf::internal::LogFinisher", (i8)>
    llvm.intr.lifetime.start 1, %31 : !llvm.ptr
    llvm.cond_br %30, ^bb1, ^bb3
  ^bb1:  // pred: ^bb0
    %32 = llvm.bitcast %28 : !llvm.ptr to !llvm.ptr
    llvm.intr.lifetime.start 56, %32 : !llvm.ptr
    llvm.call @_ZN6google8protobuf8internal10LogMessageC1ENS0_8LogLevelEPKci(%28, %3, %5, %6) : (!llvm.ptr, i32, !llvm.ptr, i32) -> ()
    %33 = llvm.invoke @_ZN6google8protobuf8internal10LogMessagelsEPKc(%28, %8) to ^bb2 unwind ^bb17 : (!llvm.ptr, !llvm.ptr) -> !llvm.ptr
  ^bb2:  // pred: ^bb1
    llvm.invoke @_ZN6google8protobuf8internal11LogFinisheraSERNS1_10LogMessageE(%29, %33) to ^bb4 unwind ^bb18 : (!llvm.ptr, !llvm.ptr) -> ()
  ^bb3:  // pred: ^bb0
    llvm.intr.lifetime.end 1, %31 : !llvm.ptr
    llvm.br ^bb5
  ^bb4:  // pred: ^bb2
    llvm.intr.lifetime.end 1, %31 : !llvm.ptr
    llvm.call @_ZN6google8protobuf8internal10LogMessageD1Ev(%28) : (!llvm.ptr) -> ()
    llvm.intr.lifetime.end 56, %32 : !llvm.ptr
    llvm.br ^bb5
  ^bb5:  // 2 preds: ^bb3, ^bb4
    %34 = llvm.getelementptr inbounds %arg0[%1, 0, 0, 1] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.tutorial::Person", (struct<"class.google::protobuf::Message", (struct<"class.google::protobuf::MessageLite", (ptr, struct<"class.google::protobuf::internal::InternalMetadata", (ptr)>)>)>, struct<"class.google::protobuf::RepeatedPtrField", (struct<"class.google::protobuf::internal::RepeatedPtrFieldBase", (ptr, i32, i32, ptr)>)>, struct<"struct.google::protobuf::internal::ArenaStringPtr", (ptr)>, struct<"struct.google::protobuf::internal::ArenaStringPtr", (ptr)>, ptr, i32, struct<"class.google::protobuf::internal::CachedSize", (struct<"struct.std::atomic", (struct<"struct.std::__atomic_base", (i32)>)>)>)>
    %35 = llvm.getelementptr inbounds %arg1[%1, 0, 0, 1, 0] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.tutorial::Person", (struct<"class.google::protobuf::Message", (struct<"class.google::protobuf::MessageLite", (ptr, struct<"class.google::protobuf::internal::InternalMetadata", (ptr)>)>)>, struct<"class.google::protobuf::RepeatedPtrField", (struct<"class.google::protobuf::internal::RepeatedPtrFieldBase", (ptr, i32, i32, ptr)>)>, struct<"struct.google::protobuf::internal::ArenaStringPtr", (ptr)>, struct<"struct.google::protobuf::internal::ArenaStringPtr", (ptr)>, ptr, i32, struct<"class.google::protobuf::internal::CachedSize", (struct<"struct.std::atomic", (struct<"struct.std::__atomic_base", (i32)>)>)>)>
    %36 = llvm.load %35 {alignment = 8 : i64, tbaa = [#tbaa_tag5]} : !llvm.ptr -> !llvm.ptr
    %37 = llvm.ptrtoint %36 : !llvm.ptr to i64
    %38 = llvm.and %37, %9  : i64
    %39 = llvm.icmp "eq" %38, %1 : i64
    llvm.cond_br %39, ^bb10, ^bb6
  ^bb6:  // pred: ^bb5
    %40 = llvm.and %37, %10  : i64
    %41 = llvm.inttoptr %40 : i64 to !llvm.ptr
    %42 = llvm.getelementptr inbounds %41[%1, 1] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.google::protobuf::internal::InternalMetadata::Container", (struct<"struct.google::protobuf::internal::InternalMetadata::ContainerBase", (ptr)>, struct<"class.google::protobuf::UnknownFieldSet", (struct<"class.std::vector", (struct<"struct.std::_Vector_base", (struct<"struct.std::_Vector_base<google::protobuf::UnknownField, std::allocator<google::protobuf::UnknownField>>::_Vector_impl", (struct<"struct.std::_Vector_base<google::protobuf::UnknownField, std::allocator<google::protobuf::UnknownField>>::_Vector_impl_data", (ptr, ptr, ptr)>)>)>)>)>)>
    %43 = llvm.getelementptr inbounds %34[%1, 0] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.google::protobuf::internal::InternalMetadata", (ptr)>
    %44 = llvm.load %43 {alignment = 8 : i64, tbaa = [#tbaa_tag5]} : !llvm.ptr -> !llvm.ptr
    %45 = llvm.ptrtoint %44 : !llvm.ptr to i64
    %46 = llvm.and %45, %9  : i64
    %47 = llvm.icmp "eq" %46, %1 : i64
    llvm.cond_br %47 weights([1, 2000]), ^bb8, ^bb7
  ^bb7:  // pred: ^bb6
    %48 = llvm.and %45, %10  : i64
    %49 = llvm.inttoptr %48 : i64 to !llvm.ptr
    %50 = llvm.getelementptr inbounds %49[%1, 1] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.google::protobuf::internal::InternalMetadata::Container", (struct<"struct.google::protobuf::internal::InternalMetadata::ContainerBase", (ptr)>, struct<"class.google::protobuf::UnknownFieldSet", (struct<"class.std::vector", (struct<"struct.std::_Vector_base", (struct<"struct.std::_Vector_base<google::protobuf::UnknownField, std::allocator<google::protobuf::UnknownField>>::_Vector_impl", (struct<"struct.std::_Vector_base<google::protobuf::UnknownField, std::allocator<google::protobuf::UnknownField>>::_Vector_impl_data", (ptr, ptr, ptr)>)>)>)>)>)>
    llvm.br ^bb9(%50 : !llvm.ptr)
  ^bb8:  // pred: ^bb6
    %51 = llvm.call @_ZN6google8protobuf8internal16InternalMetadata27mutable_unknown_fields_slowINS0_15UnknownFieldSetEEEPT_v(%34) : (!llvm.ptr) -> !llvm.ptr
    llvm.br ^bb9(%51 : !llvm.ptr)
  ^bb9(%52: !llvm.ptr):  // 2 preds: ^bb7, ^bb8
    llvm.call @_ZN6google8protobuf15UnknownFieldSet9MergeFromERKS1_(%52, %42) : (!llvm.ptr, !llvm.ptr) -> ()
    llvm.br ^bb10
  ^bb10:  // 2 preds: ^bb5, ^bb9
    %53 = llvm.getelementptr inbounds %arg0[%1, 1, 0] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.tutorial::Person", (struct<"class.google::protobuf::Message", (struct<"class.google::protobuf::MessageLite", (ptr, struct<"class.google::protobuf::internal::InternalMetadata", (ptr)>)>)>, struct<"class.google::protobuf::RepeatedPtrField", (struct<"class.google::protobuf::internal::RepeatedPtrFieldBase", (ptr, i32, i32, ptr)>)>, struct<"struct.google::protobuf::internal::ArenaStringPtr", (ptr)>, struct<"struct.google::protobuf::internal::ArenaStringPtr", (ptr)>, ptr, i32, struct<"class.google::protobuf::internal::CachedSize", (struct<"struct.std::atomic", (struct<"struct.std::__atomic_base", (i32)>)>)>)>
    %54 = llvm.getelementptr inbounds %arg1[%1, 1, 0] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.tutorial::Person", (struct<"class.google::protobuf::Message", (struct<"class.google::protobuf::MessageLite", (ptr, struct<"class.google::protobuf::internal::InternalMetadata", (ptr)>)>)>, struct<"class.google::protobuf::RepeatedPtrField", (struct<"class.google::protobuf::internal::RepeatedPtrFieldBase", (ptr, i32, i32, ptr)>)>, struct<"struct.google::protobuf::internal::ArenaStringPtr", (ptr)>, struct<"struct.google::protobuf::internal::ArenaStringPtr", (ptr)>, ptr, i32, struct<"class.google::protobuf::internal::CachedSize", (struct<"struct.std::atomic", (struct<"struct.std::__atomic_base", (i32)>)>)>)>
    llvm.call @_ZN6google8protobuf8internal20RepeatedPtrFieldBase9MergeFromINS0_16RepeatedPtrFieldIN8tutorial18Person_PhoneNumberEE11TypeHandlerEEEvRKS2_(%53, %54) : (!llvm.ptr, !llvm.ptr) -> ()
    %55 = llvm.getelementptr inbounds %arg1[%1, 2, 0] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.tutorial::Person", (struct<"class.google::protobuf::Message", (struct<"class.google::protobuf::MessageLite", (ptr, struct<"class.google::protobuf::internal::InternalMetadata", (ptr)>)>)>, struct<"class.google::protobuf::RepeatedPtrField", (struct<"class.google::protobuf::internal::RepeatedPtrFieldBase", (ptr, i32, i32, ptr)>)>, struct<"struct.google::protobuf::internal::ArenaStringPtr", (ptr)>, struct<"struct.google::protobuf::internal::ArenaStringPtr", (ptr)>, ptr, i32, struct<"class.google::protobuf::internal::CachedSize", (struct<"struct.std::atomic", (struct<"struct.std::__atomic_base", (i32)>)>)>)>
    %56 = llvm.load %55 {alignment = 8 : i64, tbaa = [#tbaa_tag8]} : !llvm.ptr -> !llvm.ptr
    %57 = llvm.getelementptr inbounds %56[%1, 1] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.std::__cxx11::basic_string", (struct<"struct.std::__cxx11::basic_string<char>::_Alloc_hider", (ptr)>, i64, struct<"union.anon", (i64, array<8 x i8>)>)>
    %58 = llvm.load %57 {alignment = 8 : i64, tbaa = [#tbaa_tag25]} : !llvm.ptr -> i64
    %59 = llvm.icmp "eq" %58, %1 : i64
    llvm.cond_br %59, ^bb20, ^bb11
  ^bb11:  // pred: ^bb10
    %60 = llvm.getelementptr inbounds %arg0[%1, 2] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.tutorial::Person", (struct<"class.google::protobuf::Message", (struct<"class.google::protobuf::MessageLite", (ptr, struct<"class.google::protobuf::internal::InternalMetadata", (ptr)>)>)>, struct<"class.google::protobuf::RepeatedPtrField", (struct<"class.google::protobuf::internal::RepeatedPtrFieldBase", (ptr, i32, i32, ptr)>)>, struct<"struct.google::protobuf::internal::ArenaStringPtr", (ptr)>, struct<"struct.google::protobuf::internal::ArenaStringPtr", (ptr)>, ptr, i32, struct<"class.google::protobuf::internal::CachedSize", (struct<"struct.std::atomic", (struct<"struct.std::__atomic_base", (i32)>)>)>)>
    %61 = llvm.getelementptr inbounds %arg0[%1, 0, 0, 1, 0] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.tutorial::Person", (struct<"class.google::protobuf::Message", (struct<"class.google::protobuf::MessageLite", (ptr, struct<"class.google::protobuf::internal::InternalMetadata", (ptr)>)>)>, struct<"class.google::protobuf::RepeatedPtrField", (struct<"class.google::protobuf::internal::RepeatedPtrFieldBase", (ptr, i32, i32, ptr)>)>, struct<"struct.google::protobuf::internal::ArenaStringPtr", (ptr)>, struct<"struct.google::protobuf::internal::ArenaStringPtr", (ptr)>, ptr, i32, struct<"class.google::protobuf::internal::CachedSize", (struct<"struct.std::atomic", (struct<"struct.std::__atomic_base", (i32)>)>)>)>
    %62 = llvm.load %61 {alignment = 8 : i64, tbaa = [#tbaa_tag5]} : !llvm.ptr -> !llvm.ptr
    %63 = llvm.ptrtoint %62 : !llvm.ptr to i64
    %64 = llvm.and %63, %9  : i64
    %65 = llvm.icmp "eq" %64, %1 : i64
    %66 = llvm.and %63, %10  : i64
    llvm.cond_br %65 weights([2000, 1]), ^bb13, ^bb12
  ^bb12:  // pred: ^bb11
    %67 = llvm.inttoptr %66 : i64 to !llvm.ptr
    %68 = llvm.getelementptr inbounds %67[%1, 0] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.google::protobuf::internal::InternalMetadata::ContainerBase", (ptr)>
    %69 = llvm.load %68 {alignment = 8 : i64, tbaa = [#tbaa_tag9]} : !llvm.ptr -> !llvm.ptr
    llvm.br ^bb14(%69 : !llvm.ptr)
  ^bb13:  // pred: ^bb11
    %70 = llvm.inttoptr %66 : i64 to !llvm.ptr
    llvm.br ^bb14(%70 : !llvm.ptr)
  ^bb14(%71: !llvm.ptr):  // 2 preds: ^bb12, ^bb13
    %72 = llvm.getelementptr inbounds %60[%1, 0] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.google::protobuf::internal::ArenaStringPtr", (ptr)>
    %73 = llvm.load %72 {alignment = 8 : i64, tbaa = [#tbaa_tag8]} : !llvm.ptr -> !llvm.ptr
    %74 = llvm.icmp "eq" %73, %12 : !llvm.ptr
    llvm.cond_br %74, ^bb15, ^bb16
  ^bb15:  // pred: ^bb14
    llvm.call @_ZN6google8protobuf8internal14ArenaStringPtr14CreateInstanceEPNS0_5ArenaEPKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE(%60, %71, %56) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
    llvm.br ^bb20
  ^bb16:  // pred: ^bb14
    llvm.call @_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE9_M_assignERKS4_(%73, %56) : (!llvm.ptr, !llvm.ptr) -> ()
    llvm.br ^bb20
  ^bb17:  // pred: ^bb1
    %75 = llvm.landingpad cleanup : !llvm.struct<(ptr, i32)>
    llvm.br ^bb19(%75 : !llvm.struct<(ptr, i32)>)
  ^bb18:  // pred: ^bb2
    %76 = llvm.landingpad cleanup : !llvm.struct<(ptr, i32)>
    llvm.intr.lifetime.end 1, %31 : !llvm.ptr
    llvm.br ^bb19(%76 : !llvm.struct<(ptr, i32)>)
  ^bb19(%77: !llvm.struct<(ptr, i32)>):  // 2 preds: ^bb17, ^bb18
    llvm.call @_ZN6google8protobuf8internal10LogMessageD1Ev(%28) : (!llvm.ptr) -> ()
    llvm.intr.lifetime.end 56, %32 : !llvm.ptr
    llvm.resume %77 : !llvm.struct<(ptr, i32)>
  ^bb20:  // 3 preds: ^bb10, ^bb15, ^bb16
    %78 = llvm.getelementptr inbounds %arg1[%1, 3, 0] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.tutorial::Person", (struct<"class.google::protobuf::Message", (struct<"class.google::protobuf::MessageLite", (ptr, struct<"class.google::protobuf::internal::InternalMetadata", (ptr)>)>)>, struct<"class.google::protobuf::RepeatedPtrField", (struct<"class.google::protobuf::internal::RepeatedPtrFieldBase", (ptr, i32, i32, ptr)>)>, struct<"struct.google::protobuf::internal::ArenaStringPtr", (ptr)>, struct<"struct.google::protobuf::internal::ArenaStringPtr", (ptr)>, ptr, i32, struct<"class.google::protobuf::internal::CachedSize", (struct<"struct.std::atomic", (struct<"struct.std::__atomic_base", (i32)>)>)>)>
    %79 = llvm.load %78 {alignment = 8 : i64, tbaa = [#tbaa_tag8]} : !llvm.ptr -> !llvm.ptr
    %80 = llvm.getelementptr inbounds %79[%1, 1] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.std::__cxx11::basic_string", (struct<"struct.std::__cxx11::basic_string<char>::_Alloc_hider", (ptr)>, i64, struct<"union.anon", (i64, array<8 x i8>)>)>
    %81 = llvm.load %80 {alignment = 8 : i64, tbaa = [#tbaa_tag25]} : !llvm.ptr -> i64
    %82 = llvm.icmp "eq" %81, %1 : i64
    llvm.cond_br %82, ^bb27, ^bb21
  ^bb21:  // pred: ^bb20
    %83 = llvm.getelementptr inbounds %arg0[%1, 3] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.tutorial::Person", (struct<"class.google::protobuf::Message", (struct<"class.google::protobuf::MessageLite", (ptr, struct<"class.google::protobuf::internal::InternalMetadata", (ptr)>)>)>, struct<"class.google::protobuf::RepeatedPtrField", (struct<"class.google::protobuf::internal::RepeatedPtrFieldBase", (ptr, i32, i32, ptr)>)>, struct<"struct.google::protobuf::internal::ArenaStringPtr", (ptr)>, struct<"struct.google::protobuf::internal::ArenaStringPtr", (ptr)>, ptr, i32, struct<"class.google::protobuf::internal::CachedSize", (struct<"struct.std::atomic", (struct<"struct.std::__atomic_base", (i32)>)>)>)>
    %84 = llvm.getelementptr inbounds %arg0[%1, 0, 0, 1, 0] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.tutorial::Person", (struct<"class.google::protobuf::Message", (struct<"class.google::protobuf::MessageLite", (ptr, struct<"class.google::protobuf::internal::InternalMetadata", (ptr)>)>)>, struct<"class.google::protobuf::RepeatedPtrField", (struct<"class.google::protobuf::internal::RepeatedPtrFieldBase", (ptr, i32, i32, ptr)>)>, struct<"struct.google::protobuf::internal::ArenaStringPtr", (ptr)>, struct<"struct.google::protobuf::internal::ArenaStringPtr", (ptr)>, ptr, i32, struct<"class.google::protobuf::internal::CachedSize", (struct<"struct.std::atomic", (struct<"struct.std::__atomic_base", (i32)>)>)>)>
    %85 = llvm.load %84 {alignment = 8 : i64, tbaa = [#tbaa_tag5]} : !llvm.ptr -> !llvm.ptr
    %86 = llvm.ptrtoint %85 : !llvm.ptr to i64
    %87 = llvm.and %86, %9  : i64
    %88 = llvm.icmp "eq" %87, %1 : i64
    %89 = llvm.and %86, %10  : i64
    llvm.cond_br %88 weights([2000, 1]), ^bb23, ^bb22
  ^bb22:  // pred: ^bb21
    %90 = llvm.inttoptr %89 : i64 to !llvm.ptr
    %91 = llvm.getelementptr inbounds %90[%1, 0] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.google::protobuf::internal::InternalMetadata::ContainerBase", (ptr)>
    %92 = llvm.load %91 {alignment = 8 : i64, tbaa = [#tbaa_tag9]} : !llvm.ptr -> !llvm.ptr
    llvm.br ^bb24(%92 : !llvm.ptr)
  ^bb23:  // pred: ^bb21
    %93 = llvm.inttoptr %89 : i64 to !llvm.ptr
    llvm.br ^bb24(%93 : !llvm.ptr)
  ^bb24(%94: !llvm.ptr):  // 2 preds: ^bb22, ^bb23
    %95 = llvm.getelementptr inbounds %83[%1, 0] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.google::protobuf::internal::ArenaStringPtr", (ptr)>
    %96 = llvm.load %95 {alignment = 8 : i64, tbaa = [#tbaa_tag8]} : !llvm.ptr -> !llvm.ptr
    %97 = llvm.icmp "eq" %96, %12 : !llvm.ptr
    llvm.cond_br %97, ^bb25, ^bb26
  ^bb25:  // pred: ^bb24
    llvm.call @_ZN6google8protobuf8internal14ArenaStringPtr14CreateInstanceEPNS0_5ArenaEPKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE(%83, %94, %79) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
    llvm.br ^bb27
  ^bb26:  // pred: ^bb24
    llvm.call @_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE9_M_assignERKS4_(%96, %79) : (!llvm.ptr, !llvm.ptr) -> ()
    llvm.br ^bb27
  ^bb27:  // 3 preds: ^bb20, ^bb25, ^bb26
    %98 = llvm.icmp "ne" %arg1, %22 : !llvm.ptr
    %99 = llvm.getelementptr inbounds %arg1[%1, 4] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.tutorial::Person", (struct<"class.google::protobuf::Message", (struct<"class.google::protobuf::MessageLite", (ptr, struct<"class.google::protobuf::internal::InternalMetadata", (ptr)>)>)>, struct<"class.google::protobuf::RepeatedPtrField", (struct<"class.google::protobuf::internal::RepeatedPtrFieldBase", (ptr, i32, i32, ptr)>)>, struct<"struct.google::protobuf::internal::ArenaStringPtr", (ptr)>, struct<"struct.google::protobuf::internal::ArenaStringPtr", (ptr)>, ptr, i32, struct<"class.google::protobuf::internal::CachedSize", (struct<"struct.std::atomic", (struct<"struct.std::__atomic_base", (i32)>)>)>)>
    %100 = llvm.load %99 {alignment = 8 : i64} : !llvm.ptr -> !llvm.ptr
    %101 = llvm.icmp "ne" %100, %24 : !llvm.ptr
    %102 = llvm.select %98, %101, %25 : i1, i1
    llvm.cond_br %102, ^bb28, ^bb34
  ^bb28:  // pred: ^bb27
    %103 = llvm.getelementptr inbounds %arg0[%1, 4] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.tutorial::Person", (struct<"class.google::protobuf::Message", (struct<"class.google::protobuf::MessageLite", (ptr, struct<"class.google::protobuf::internal::InternalMetadata", (ptr)>)>)>, struct<"class.google::protobuf::RepeatedPtrField", (struct<"class.google::protobuf::internal::RepeatedPtrFieldBase", (ptr, i32, i32, ptr)>)>, struct<"struct.google::protobuf::internal::ArenaStringPtr", (ptr)>, struct<"struct.google::protobuf::internal::ArenaStringPtr", (ptr)>, ptr, i32, struct<"class.google::protobuf::internal::CachedSize", (struct<"struct.std::atomic", (struct<"struct.std::__atomic_base", (i32)>)>)>)>
    %104 = llvm.load %103 {alignment = 8 : i64, tbaa = [#tbaa_tag23]} : !llvm.ptr -> !llvm.ptr
    %105 = llvm.icmp "eq" %104, %24 : !llvm.ptr
    llvm.cond_br %105, ^bb29, ^bb33(%100, %104 : !llvm.ptr, !llvm.ptr)
  ^bb29:  // pred: ^bb28
    %106 = llvm.getelementptr inbounds %arg0[%1, 0, 0, 1, 0] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.tutorial::Person", (struct<"class.google::protobuf::Message", (struct<"class.google::protobuf::MessageLite", (ptr, struct<"class.google::protobuf::internal::InternalMetadata", (ptr)>)>)>, struct<"class.google::protobuf::RepeatedPtrField", (struct<"class.google::protobuf::internal::RepeatedPtrFieldBase", (ptr, i32, i32, ptr)>)>, struct<"struct.google::protobuf::internal::ArenaStringPtr", (ptr)>, struct<"struct.google::protobuf::internal::ArenaStringPtr", (ptr)>, ptr, i32, struct<"class.google::protobuf::internal::CachedSize", (struct<"struct.std::atomic", (struct<"struct.std::__atomic_base", (i32)>)>)>)>
    %107 = llvm.load %106 {alignment = 8 : i64, tbaa = [#tbaa_tag5]} : !llvm.ptr -> !llvm.ptr
    %108 = llvm.ptrtoint %107 : !llvm.ptr to i64
    %109 = llvm.and %108, %9  : i64
    %110 = llvm.icmp "eq" %109, %1 : i64
    %111 = llvm.and %108, %10  : i64
    llvm.cond_br %110 weights([2000, 1]), ^bb31, ^bb30
  ^bb30:  // pred: ^bb29
    %112 = llvm.inttoptr %111 : i64 to !llvm.ptr
    %113 = llvm.getelementptr inbounds %112[%1, 0] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.google::protobuf::internal::InternalMetadata::ContainerBase", (ptr)>
    %114 = llvm.load %113 {alignment = 8 : i64, tbaa = [#tbaa_tag9]} : !llvm.ptr -> !llvm.ptr
    llvm.br ^bb32(%114 : !llvm.ptr)
  ^bb31:  // pred: ^bb29
    %115 = llvm.inttoptr %111 : i64 to !llvm.ptr
    llvm.br ^bb32(%115 : !llvm.ptr)
  ^bb32(%116: !llvm.ptr):  // 2 preds: ^bb30, ^bb31
    %117 = llvm.call @_ZN6google8protobuf5Arena18CreateMaybeMessageINS0_9TimestampEJEEEPT_PS1_DpOT0_(%116) : (!llvm.ptr) -> !llvm.ptr
    llvm.store %117, %103 {alignment = 8 : i64, tbaa = [#tbaa_tag23]} : !llvm.ptr, !llvm.ptr
    %118 = llvm.load %99 {alignment = 8 : i64, tbaa = [#tbaa_tag23]} : !llvm.ptr -> !llvm.ptr
    llvm.br ^bb33(%118, %117 : !llvm.ptr, !llvm.ptr)
  ^bb33(%119: !llvm.ptr, %120: !llvm.ptr):  // 2 preds: ^bb28, ^bb32
    %121 = llvm.icmp "eq" %119, %24 : !llvm.ptr
    %122 = llvm.select %121, %26, %119 : i1, !llvm.ptr
    llvm.call @_ZN6google8protobuf9Timestamp9MergeFromERKS1_(%120, %122) : (!llvm.ptr, !llvm.ptr) -> ()
    llvm.br ^bb34
  ^bb34:  // 2 preds: ^bb27, ^bb33
    %123 = llvm.getelementptr inbounds %arg1[%1, 5] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.tutorial::Person", (struct<"class.google::protobuf::Message", (struct<"class.google::protobuf::MessageLite", (ptr, struct<"class.google::protobuf::internal::InternalMetadata", (ptr)>)>)>, struct<"class.google::protobuf::RepeatedPtrField", (struct<"class.google::protobuf::internal::RepeatedPtrFieldBase", (ptr, i32, i32, ptr)>)>, struct<"struct.google::protobuf::internal::ArenaStringPtr", (ptr)>, struct<"struct.google::protobuf::internal::ArenaStringPtr", (ptr)>, ptr, i32, struct<"class.google::protobuf::internal::CachedSize", (struct<"struct.std::atomic", (struct<"struct.std::__atomic_base", (i32)>)>)>)>
    %124 = llvm.load %123 {alignment = 8 : i64, tbaa = [#tbaa_tag28]} : !llvm.ptr -> i32
    %125 = llvm.icmp "eq" %124, %2 : i32
    llvm.cond_br %125, ^bb36, ^bb35
  ^bb35:  // pred: ^bb34
    %126 = llvm.getelementptr inbounds %arg0[%1, 5] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.tutorial::Person", (struct<"class.google::protobuf::Message", (struct<"class.google::protobuf::MessageLite", (ptr, struct<"class.google::protobuf::internal::InternalMetadata", (ptr)>)>)>, struct<"class.google::protobuf::RepeatedPtrField", (struct<"class.google::protobuf::internal::RepeatedPtrFieldBase", (ptr, i32, i32, ptr)>)>, struct<"struct.google::protobuf::internal::ArenaStringPtr", (ptr)>, struct<"struct.google::protobuf::internal::ArenaStringPtr", (ptr)>, ptr, i32, struct<"class.google::protobuf::internal::CachedSize", (struct<"struct.std::atomic", (struct<"struct.std::__atomic_base", (i32)>)>)>)>
    llvm.store %124, %126 {alignment = 8 : i64, tbaa = [#tbaa_tag28]} : i32, !llvm.ptr
    llvm.br ^bb36
  ^bb36:  // 2 preds: ^bb34, ^bb35
    llvm.return
  }
  llvm.func local_unnamed_addr @_ZN6google8protobuf9Timestamp9MergeFromERKS1_(!llvm.ptr {llvm.align = 8 : i64, llvm.dereferenceable = 32 : i64, llvm.nonnull, llvm.noundef}, !llvm.ptr {llvm.align = 8 : i64, llvm.dereferenceable = 32 : i64, llvm.nonnull, llvm.noundef}) attributes {frame_pointer = #llvm.framePointerKind<none>, passthrough = [["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], ["tune-cpu", "generic"]], target_cpu = "x86-64", target_features = #llvm.target_features<["+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>}
  llvm.func unnamed_addr @_ZN8tutorial6Person8CopyFromERKN6google8protobuf7MessageE(%arg0: !llvm.ptr {llvm.align = 8 : i64, llvm.dereferenceable = 72 : i64, llvm.nonnull, llvm.noundef}, %arg1: !llvm.ptr {llvm.align = 8 : i64, llvm.dereferenceable = 16 : i64, llvm.nonnull, llvm.noundef}) attributes {alignment = 2 : i64, frame_pointer = #llvm.framePointerKind<none>, passthrough = ["mustprogress", ["uwtable", "2"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], ["tune-cpu", "generic"]], target_cpu = "x86-64", target_features = #llvm.target_features<["+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>} {
    %0 = llvm.mlir.constant(0 : i64) : i64
    %1 = llvm.mlir.constant(0 : i32) : i32
    %2 = llvm.getelementptr inbounds %arg0[%0, 0] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.tutorial::Person", (struct<"class.google::protobuf::Message", (struct<"class.google::protobuf::MessageLite", (ptr, struct<"class.google::protobuf::internal::InternalMetadata", (ptr)>)>)>, struct<"class.google::protobuf::RepeatedPtrField", (struct<"class.google::protobuf::internal::RepeatedPtrFieldBase", (ptr, i32, i32, ptr)>)>, struct<"struct.google::protobuf::internal::ArenaStringPtr", (ptr)>, struct<"struct.google::protobuf::internal::ArenaStringPtr", (ptr)>, ptr, i32, struct<"class.google::protobuf::internal::CachedSize", (struct<"struct.std::atomic", (struct<"struct.std::__atomic_base", (i32)>)>)>)>
    %3 = llvm.icmp "eq" %2, %arg1 : !llvm.ptr
    llvm.cond_br %3, ^bb2, ^bb1
  ^bb1:  // pred: ^bb0
    llvm.call @_ZN8tutorial6Person5ClearEv(%arg0) : (!llvm.ptr) -> ()
    llvm.call @_ZN8tutorial6Person9MergeFromERKN6google8protobuf7MessageE(%arg0, %arg1) : (!llvm.ptr, !llvm.ptr) -> ()
    llvm.br ^bb2
  ^bb2:  // 2 preds: ^bb0, ^bb1
    llvm.return
  }
  llvm.func local_unnamed_addr @_ZN8tutorial6Person8CopyFromERKS0_(%arg0: !llvm.ptr {llvm.align = 8 : i64, llvm.dereferenceable = 72 : i64, llvm.nonnull, llvm.noundef}, %arg1: !llvm.ptr {llvm.align = 8 : i64, llvm.dereferenceable = 72 : i64, llvm.nonnull, llvm.noundef}) attributes {alignment = 2 : i64, frame_pointer = #llvm.framePointerKind<none>, passthrough = ["mustprogress", ["uwtable", "2"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], ["tune-cpu", "generic"]], target_cpu = "x86-64", target_features = #llvm.target_features<["+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>} {
    %0 = llvm.icmp "eq" %arg1, %arg0 : !llvm.ptr
    llvm.cond_br %0, ^bb2, ^bb1
  ^bb1:  // pred: ^bb0
    llvm.call @_ZN8tutorial6Person5ClearEv(%arg0) : (!llvm.ptr) -> ()
    llvm.call @_ZN8tutorial6Person9MergeFromERKS0_(%arg0, %arg1) : (!llvm.ptr, !llvm.ptr) -> ()
    llvm.br ^bb2
  ^bb2:  // 2 preds: ^bb0, ^bb1
    llvm.return
  }
  llvm.func unnamed_addr @_ZNK8tutorial6Person13IsInitializedEv(%arg0: !llvm.ptr {llvm.align = 8 : i64, llvm.nocapture, llvm.nonnull, llvm.readnone}) -> (i1 {llvm.noundef, llvm.zeroext}) attributes {alignment = 2 : i64, frame_pointer = #llvm.framePointerKind<none>, memory = #llvm.memory_effects<other = none, argMem = none, inaccessibleMem = none>, passthrough = ["mustprogress", "nofree", "norecurse", "nosync", "nounwind", "willreturn", ["uwtable", "2"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], ["tune-cpu", "generic"]], target_cpu = "x86-64", target_features = #llvm.target_features<["+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>} {
    %0 = llvm.mlir.constant(true) : i1
    llvm.return %0 : i1
  }
  llvm.func local_unnamed_addr @_ZN8tutorial6Person12InternalSwapEPS0_(%arg0: !llvm.ptr {llvm.align = 8 : i64, llvm.dereferenceable = 72 : i64, llvm.nonnull, llvm.noundef}, %arg1: !llvm.ptr {llvm.noundef}) attributes {alignment = 2 : i64, frame_pointer = #llvm.framePointerKind<none>, passthrough = [["uwtable", "2"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], ["tune-cpu", "generic"]], personality = @__gxx_personality_v0, target_cpu = "x86-64", target_features = #llvm.target_features<["+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>} {
    %0 = llvm.mlir.constant(0 : i64) : i64
    %1 = llvm.mlir.constant(0 : i32) : i32
    %2 = llvm.mlir.constant(1 : i32) : i32
    %3 = llvm.mlir.constant(1 : i64) : i64
    %4 = llvm.mlir.constant(false) : i1
    %5 = llvm.mlir.constant(-2 : i64) : i64
    %6 = llvm.mlir.constant(2 : i32) : i32
    %7 = llvm.mlir.addressof @_ZN6google8protobuf8internal26fixed_address_empty_stringB5cxx11E : !llvm.ptr
    %8 = llvm.mlir.constant(3 : i32) : i32
    %9 = llvm.mlir.constant(4 : i32) : i32
    %10 = llvm.getelementptr inbounds %arg0[%0, 0, 0, 1] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.tutorial::Person", (struct<"class.google::protobuf::Message", (struct<"class.google::protobuf::MessageLite", (ptr, struct<"class.google::protobuf::internal::InternalMetadata", (ptr)>)>)>, struct<"class.google::protobuf::RepeatedPtrField", (struct<"class.google::protobuf::internal::RepeatedPtrFieldBase", (ptr, i32, i32, ptr)>)>, struct<"struct.google::protobuf::internal::ArenaStringPtr", (ptr)>, struct<"struct.google::protobuf::internal::ArenaStringPtr", (ptr)>, ptr, i32, struct<"class.google::protobuf::internal::CachedSize", (struct<"struct.std::atomic", (struct<"struct.std::__atomic_base", (i32)>)>)>)>
    %11 = llvm.getelementptr inbounds %arg1[%0, 0, 0, 1] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.tutorial::Person", (struct<"class.google::protobuf::Message", (struct<"class.google::protobuf::MessageLite", (ptr, struct<"class.google::protobuf::internal::InternalMetadata", (ptr)>)>)>, struct<"class.google::protobuf::RepeatedPtrField", (struct<"class.google::protobuf::internal::RepeatedPtrFieldBase", (ptr, i32, i32, ptr)>)>, struct<"struct.google::protobuf::internal::ArenaStringPtr", (ptr)>, struct<"struct.google::protobuf::internal::ArenaStringPtr", (ptr)>, ptr, i32, struct<"class.google::protobuf::internal::CachedSize", (struct<"struct.std::atomic", (struct<"struct.std::__atomic_base", (i32)>)>)>)>
    %12 = llvm.getelementptr inbounds %10[%0, 0] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.google::protobuf::internal::InternalMetadata", (ptr)>
    %13 = llvm.load %12 {alignment = 8 : i64, tbaa = [#tbaa_tag5]} : !llvm.ptr -> !llvm.ptr
    %14 = llvm.ptrtoint %13 : !llvm.ptr to i64
    %15 = llvm.and %14, %3  : i64
    %16 = llvm.icmp "eq" %15, %0 : i64
    %17 = llvm.getelementptr inbounds %11[%0, 0] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.google::protobuf::internal::InternalMetadata", (ptr)>
    %18 = llvm.load %17 {alignment = 8 : i64, tbaa = [#tbaa_tag5]} : !llvm.ptr -> !llvm.ptr
    %19 = llvm.ptrtoint %18 : !llvm.ptr to i64
    %20 = llvm.and %19, %3  : i64
    %21 = llvm.icmp "eq" %20, %0 : i64
    %22 = llvm.select %16, %21, %4 : i1, i1
    llvm.cond_br %22, ^bb8, ^bb1
  ^bb1:  // pred: ^bb0
    llvm.cond_br %21 weights([1, 2000]), ^bb3, ^bb2
  ^bb2:  // pred: ^bb1
    %23 = llvm.and %19, %5  : i64
    %24 = llvm.inttoptr %23 : i64 to !llvm.ptr
    %25 = llvm.getelementptr inbounds %24[%0, 1] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.google::protobuf::internal::InternalMetadata::Container", (struct<"struct.google::protobuf::internal::InternalMetadata::ContainerBase", (ptr)>, struct<"class.google::protobuf::UnknownFieldSet", (struct<"class.std::vector", (struct<"struct.std::_Vector_base", (struct<"struct.std::_Vector_base<google::protobuf::UnknownField, std::allocator<google::protobuf::UnknownField>>::_Vector_impl", (struct<"struct.std::_Vector_base<google::protobuf::UnknownField, std::allocator<google::protobuf::UnknownField>>::_Vector_impl_data", (ptr, ptr, ptr)>)>)>)>)>)>
    llvm.br ^bb4(%14, %25 : i64, !llvm.ptr)
  ^bb3:  // pred: ^bb1
    %26 = llvm.call @_ZN6google8protobuf8internal16InternalMetadata27mutable_unknown_fields_slowINS0_15UnknownFieldSetEEEPT_v(%11) : (!llvm.ptr) -> !llvm.ptr
    %27 = llvm.load %12 {alignment = 8 : i64, tbaa = [#tbaa_tag5]} : !llvm.ptr -> !llvm.ptr
    %28 = llvm.ptrtoint %27 : !llvm.ptr to i64
    llvm.br ^bb4(%28, %26 : i64, !llvm.ptr)
  ^bb4(%29: i64, %30: !llvm.ptr):  // 2 preds: ^bb2, ^bb3
    %31 = llvm.and %29, %3  : i64
    %32 = llvm.icmp "eq" %31, %0 : i64
    llvm.cond_br %32 weights([1, 2000]), ^bb6, ^bb5
  ^bb5:  // pred: ^bb4
    %33 = llvm.and %29, %5  : i64
    %34 = llvm.inttoptr %33 : i64 to !llvm.ptr
    %35 = llvm.getelementptr inbounds %34[%0, 1] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.google::protobuf::internal::InternalMetadata::Container", (struct<"struct.google::protobuf::internal::InternalMetadata::ContainerBase", (ptr)>, struct<"class.google::protobuf::UnknownFieldSet", (struct<"class.std::vector", (struct<"struct.std::_Vector_base", (struct<"struct.std::_Vector_base<google::protobuf::UnknownField, std::allocator<google::protobuf::UnknownField>>::_Vector_impl", (struct<"struct.std::_Vector_base<google::protobuf::UnknownField, std::allocator<google::protobuf::UnknownField>>::_Vector_impl_data", (ptr, ptr, ptr)>)>)>)>)>)>
    llvm.br ^bb7(%35 : !llvm.ptr)
  ^bb6:  // pred: ^bb4
    %36 = llvm.call @_ZN6google8protobuf8internal16InternalMetadata27mutable_unknown_fields_slowINS0_15UnknownFieldSetEEEPT_v(%10) : (!llvm.ptr) -> !llvm.ptr
    llvm.br ^bb7(%36 : !llvm.ptr)
  ^bb7(%37: !llvm.ptr):  // 2 preds: ^bb5, ^bb6
    %38 = llvm.bitcast %37 : !llvm.ptr to !llvm.ptr
    %39 = llvm.load %38 {alignment = 8 : i64, tbaa = [#tbaa_tag2]} : !llvm.ptr -> !llvm.vec<2 x ptr>
    %40 = llvm.getelementptr inbounds %37[%0, 0, 0, 0, 0, 2] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.google::protobuf::UnknownFieldSet", (struct<"class.std::vector", (struct<"struct.std::_Vector_base", (struct<"struct.std::_Vector_base<google::protobuf::UnknownField, std::allocator<google::protobuf::UnknownField>>::_Vector_impl", (struct<"struct.std::_Vector_base<google::protobuf::UnknownField, std::allocator<google::protobuf::UnknownField>>::_Vector_impl_data", (ptr, ptr, ptr)>)>)>)>)>
    %41 = llvm.load %40 {alignment = 8 : i64, tbaa = [#tbaa_tag13]} : !llvm.ptr -> !llvm.ptr
    %42 = llvm.bitcast %30 : !llvm.ptr to !llvm.ptr
    %43 = llvm.load %42 {alignment = 8 : i64, tbaa = [#tbaa_tag2]} : !llvm.ptr -> !llvm.vec<2 x ptr>
    %44 = llvm.bitcast %37 : !llvm.ptr to !llvm.ptr
    llvm.store %43, %44 {alignment = 8 : i64, tbaa = [#tbaa_tag2]} : !llvm.vec<2 x ptr>, !llvm.ptr
    %45 = llvm.getelementptr inbounds %30[%0, 0, 0, 0, 0, 2] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.google::protobuf::UnknownFieldSet", (struct<"class.std::vector", (struct<"struct.std::_Vector_base", (struct<"struct.std::_Vector_base<google::protobuf::UnknownField, std::allocator<google::protobuf::UnknownField>>::_Vector_impl", (struct<"struct.std::_Vector_base<google::protobuf::UnknownField, std::allocator<google::protobuf::UnknownField>>::_Vector_impl_data", (ptr, ptr, ptr)>)>)>)>)>
    %46 = llvm.load %45 {alignment = 8 : i64, tbaa = [#tbaa_tag13]} : !llvm.ptr -> !llvm.ptr
    llvm.store %46, %40 {alignment = 8 : i64, tbaa = [#tbaa_tag13]} : !llvm.ptr, !llvm.ptr
    %47 = llvm.bitcast %30 : !llvm.ptr to !llvm.ptr
    llvm.store %39, %47 {alignment = 8 : i64, tbaa = [#tbaa_tag2]} : !llvm.vec<2 x ptr>, !llvm.ptr
    llvm.store %41, %45 {alignment = 8 : i64, tbaa = [#tbaa_tag13]} : !llvm.ptr, !llvm.ptr
    llvm.br ^bb8
  ^bb8:  // 2 preds: ^bb0, ^bb7
    %48 = llvm.getelementptr inbounds %arg0[%0, 1, 0] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.tutorial::Person", (struct<"class.google::protobuf::Message", (struct<"class.google::protobuf::MessageLite", (ptr, struct<"class.google::protobuf::internal::InternalMetadata", (ptr)>)>)>, struct<"class.google::protobuf::RepeatedPtrField", (struct<"class.google::protobuf::internal::RepeatedPtrFieldBase", (ptr, i32, i32, ptr)>)>, struct<"struct.google::protobuf::internal::ArenaStringPtr", (ptr)>, struct<"struct.google::protobuf::internal::ArenaStringPtr", (ptr)>, ptr, i32, struct<"class.google::protobuf::internal::CachedSize", (struct<"struct.std::atomic", (struct<"struct.std::__atomic_base", (i32)>)>)>)>
    %49 = llvm.getelementptr inbounds %arg1[%0, 1, 0] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.tutorial::Person", (struct<"class.google::protobuf::Message", (struct<"class.google::protobuf::MessageLite", (ptr, struct<"class.google::protobuf::internal::InternalMetadata", (ptr)>)>)>, struct<"class.google::protobuf::RepeatedPtrField", (struct<"class.google::protobuf::internal::RepeatedPtrFieldBase", (ptr, i32, i32, ptr)>)>, struct<"struct.google::protobuf::internal::ArenaStringPtr", (ptr)>, struct<"struct.google::protobuf::internal::ArenaStringPtr", (ptr)>, ptr, i32, struct<"class.google::protobuf::internal::CachedSize", (struct<"struct.std::atomic", (struct<"struct.std::__atomic_base", (i32)>)>)>)>
    llvm.call @_ZN6google8protobuf8internal20RepeatedPtrFieldBase12InternalSwapEPS2_(%48, %49) : (!llvm.ptr, !llvm.ptr) -> ()
    %50 = llvm.getelementptr inbounds %arg0[%0, 2] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.tutorial::Person", (struct<"class.google::protobuf::Message", (struct<"class.google::protobuf::MessageLite", (ptr, struct<"class.google::protobuf::internal::InternalMetadata", (ptr)>)>)>, struct<"class.google::protobuf::RepeatedPtrField", (struct<"class.google::protobuf::internal::RepeatedPtrFieldBase", (ptr, i32, i32, ptr)>)>, struct<"struct.google::protobuf::internal::ArenaStringPtr", (ptr)>, struct<"struct.google::protobuf::internal::ArenaStringPtr", (ptr)>, ptr, i32, struct<"class.google::protobuf::internal::CachedSize", (struct<"struct.std::atomic", (struct<"struct.std::__atomic_base", (i32)>)>)>)>
    %51 = llvm.getelementptr inbounds %arg1[%0, 2] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.tutorial::Person", (struct<"class.google::protobuf::Message", (struct<"class.google::protobuf::MessageLite", (ptr, struct<"class.google::protobuf::internal::InternalMetadata", (ptr)>)>)>, struct<"class.google::protobuf::RepeatedPtrField", (struct<"class.google::protobuf::internal::RepeatedPtrFieldBase", (ptr, i32, i32, ptr)>)>, struct<"struct.google::protobuf::internal::ArenaStringPtr", (ptr)>, struct<"struct.google::protobuf::internal::ArenaStringPtr", (ptr)>, ptr, i32, struct<"class.google::protobuf::internal::CachedSize", (struct<"struct.std::atomic", (struct<"struct.std::__atomic_base", (i32)>)>)>)>
    %52 = llvm.getelementptr inbounds %arg0[%0, 0, 0, 1, 0] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.tutorial::Person", (struct<"class.google::protobuf::Message", (struct<"class.google::protobuf::MessageLite", (ptr, struct<"class.google::protobuf::internal::InternalMetadata", (ptr)>)>)>, struct<"class.google::protobuf::RepeatedPtrField", (struct<"class.google::protobuf::internal::RepeatedPtrFieldBase", (ptr, i32, i32, ptr)>)>, struct<"struct.google::protobuf::internal::ArenaStringPtr", (ptr)>, struct<"struct.google::protobuf::internal::ArenaStringPtr", (ptr)>, ptr, i32, struct<"class.google::protobuf::internal::CachedSize", (struct<"struct.std::atomic", (struct<"struct.std::__atomic_base", (i32)>)>)>)>
    %53 = llvm.load %52 {alignment = 8 : i64, tbaa = [#tbaa_tag5]} : !llvm.ptr -> !llvm.ptr
    %54 = llvm.ptrtoint %53 : !llvm.ptr to i64
    %55 = llvm.and %54, %3  : i64
    %56 = llvm.icmp "eq" %55, %0 : i64
    %57 = llvm.and %54, %5  : i64
    llvm.cond_br %56 weights([2000, 1]), ^bb10, ^bb9
  ^bb9:  // pred: ^bb8
    %58 = llvm.inttoptr %57 : i64 to !llvm.ptr
    %59 = llvm.getelementptr inbounds %58[%0, 0] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.google::protobuf::internal::InternalMetadata::ContainerBase", (ptr)>
    %60 = llvm.load %59 {alignment = 8 : i64, tbaa = [#tbaa_tag9]} : !llvm.ptr -> !llvm.ptr
    llvm.br ^bb11(%60 : !llvm.ptr)
  ^bb10:  // pred: ^bb8
    %61 = llvm.inttoptr %57 : i64 to !llvm.ptr
    llvm.br ^bb11(%61 : !llvm.ptr)
  ^bb11(%62: !llvm.ptr):  // 2 preds: ^bb9, ^bb10
    %63 = llvm.getelementptr inbounds %50[%0, 0] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.google::protobuf::internal::ArenaStringPtr", (ptr)>
    %64 = llvm.load %63 {alignment = 8 : i64, tbaa = [#tbaa_tag8]} : !llvm.ptr -> !llvm.ptr
    %65 = llvm.icmp "eq" %64, %7 : !llvm.ptr
    llvm.cond_br %65, ^bb12, ^bb14(%64 : !llvm.ptr)
  ^bb12:  // pred: ^bb11
    %66 = llvm.getelementptr inbounds %51[%0, 0] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.google::protobuf::internal::ArenaStringPtr", (ptr)>
    %67 = llvm.load %66 {alignment = 8 : i64, tbaa = [#tbaa_tag8]} : !llvm.ptr -> !llvm.ptr
    %68 = llvm.icmp "eq" %67, %7 : !llvm.ptr
    llvm.cond_br %68, ^bb17(%57, %54 : i64, i64), ^bb13
  ^bb13:  // pred: ^bb12
    llvm.call @_ZN6google8protobuf8internal14ArenaStringPtr14CreateInstanceEPNS0_5ArenaEPKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE(%50, %62, %7) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
    %69 = llvm.load %63 {alignment = 8 : i64, tbaa = [#tbaa_tag8]} : !llvm.ptr -> !llvm.ptr
    llvm.br ^bb14(%69 : !llvm.ptr)
  ^bb14(%70: !llvm.ptr):  // 2 preds: ^bb11, ^bb13
    %71 = llvm.getelementptr inbounds %51[%0, 0] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.google::protobuf::internal::ArenaStringPtr", (ptr)>
    %72 = llvm.load %71 {alignment = 8 : i64, tbaa = [#tbaa_tag8]} : !llvm.ptr -> !llvm.ptr
    %73 = llvm.icmp "eq" %72, %7 : !llvm.ptr
    llvm.cond_br %73, ^bb15, ^bb16(%72 : !llvm.ptr)
  ^bb15:  // pred: ^bb14
    llvm.call @_ZN6google8protobuf8internal14ArenaStringPtr14CreateInstanceEPNS0_5ArenaEPKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE(%51, %62, %7) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
    %74 = llvm.load %71 {alignment = 8 : i64, tbaa = [#tbaa_tag8]} : !llvm.ptr -> !llvm.ptr
    llvm.br ^bb16(%74 : !llvm.ptr)
  ^bb16(%75: !llvm.ptr):  // 2 preds: ^bb14, ^bb15
    llvm.call @_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE4swapERS4_(%70, %75) : (!llvm.ptr, !llvm.ptr) -> ()
    %76 = llvm.load %52 {alignment = 8 : i64, tbaa = [#tbaa_tag5]} : !llvm.ptr -> !llvm.ptr
    %77 = llvm.ptrtoint %76 : !llvm.ptr to i64
    %78 = llvm.and %77, %5  : i64
    llvm.br ^bb17(%78, %77 : i64, i64)
  ^bb17(%79: i64, %80: i64):  // 2 preds: ^bb12, ^bb16
    %81 = llvm.getelementptr inbounds %arg0[%0, 3] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.tutorial::Person", (struct<"class.google::protobuf::Message", (struct<"class.google::protobuf::MessageLite", (ptr, struct<"class.google::protobuf::internal::InternalMetadata", (ptr)>)>)>, struct<"class.google::protobuf::RepeatedPtrField", (struct<"class.google::protobuf::internal::RepeatedPtrFieldBase", (ptr, i32, i32, ptr)>)>, struct<"struct.google::protobuf::internal::ArenaStringPtr", (ptr)>, struct<"struct.google::protobuf::internal::ArenaStringPtr", (ptr)>, ptr, i32, struct<"class.google::protobuf::internal::CachedSize", (struct<"struct.std::atomic", (struct<"struct.std::__atomic_base", (i32)>)>)>)>
    %82 = llvm.getelementptr inbounds %arg1[%0, 3] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.tutorial::Person", (struct<"class.google::protobuf::Message", (struct<"class.google::protobuf::MessageLite", (ptr, struct<"class.google::protobuf::internal::InternalMetadata", (ptr)>)>)>, struct<"class.google::protobuf::RepeatedPtrField", (struct<"class.google::protobuf::internal::RepeatedPtrFieldBase", (ptr, i32, i32, ptr)>)>, struct<"struct.google::protobuf::internal::ArenaStringPtr", (ptr)>, struct<"struct.google::protobuf::internal::ArenaStringPtr", (ptr)>, ptr, i32, struct<"class.google::protobuf::internal::CachedSize", (struct<"struct.std::atomic", (struct<"struct.std::__atomic_base", (i32)>)>)>)>
    %83 = llvm.and %80, %3  : i64
    %84 = llvm.icmp "eq" %83, %0 : i64
    llvm.cond_br %84 weights([2000, 1]), ^bb19, ^bb18
  ^bb18:  // pred: ^bb17
    %85 = llvm.inttoptr %79 : i64 to !llvm.ptr
    %86 = llvm.getelementptr inbounds %85[%0, 0] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.google::protobuf::internal::InternalMetadata::ContainerBase", (ptr)>
    %87 = llvm.load %86 {alignment = 8 : i64, tbaa = [#tbaa_tag9]} : !llvm.ptr -> !llvm.ptr
    llvm.br ^bb20(%87 : !llvm.ptr)
  ^bb19:  // pred: ^bb17
    %88 = llvm.inttoptr %79 : i64 to !llvm.ptr
    llvm.br ^bb20(%88 : !llvm.ptr)
  ^bb20(%89: !llvm.ptr):  // 2 preds: ^bb18, ^bb19
    %90 = llvm.getelementptr inbounds %81[%0, 0] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.google::protobuf::internal::ArenaStringPtr", (ptr)>
    %91 = llvm.load %90 {alignment = 8 : i64, tbaa = [#tbaa_tag8]} : !llvm.ptr -> !llvm.ptr
    %92 = llvm.icmp "eq" %91, %7 : !llvm.ptr
    llvm.cond_br %92, ^bb21, ^bb23(%91 : !llvm.ptr)
  ^bb21:  // pred: ^bb20
    %93 = llvm.getelementptr inbounds %82[%0, 0] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.google::protobuf::internal::ArenaStringPtr", (ptr)>
    %94 = llvm.load %93 {alignment = 8 : i64, tbaa = [#tbaa_tag8]} : !llvm.ptr -> !llvm.ptr
    %95 = llvm.icmp "eq" %94, %7 : !llvm.ptr
    llvm.cond_br %95, ^bb26, ^bb22
  ^bb22:  // pred: ^bb21
    llvm.call @_ZN6google8protobuf8internal14ArenaStringPtr14CreateInstanceEPNS0_5ArenaEPKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE(%81, %89, %7) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
    %96 = llvm.load %90 {alignment = 8 : i64, tbaa = [#tbaa_tag8]} : !llvm.ptr -> !llvm.ptr
    llvm.br ^bb23(%96 : !llvm.ptr)
  ^bb23(%97: !llvm.ptr):  // 2 preds: ^bb20, ^bb22
    %98 = llvm.getelementptr inbounds %82[%0, 0] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.google::protobuf::internal::ArenaStringPtr", (ptr)>
    %99 = llvm.load %98 {alignment = 8 : i64, tbaa = [#tbaa_tag8]} : !llvm.ptr -> !llvm.ptr
    %100 = llvm.icmp "eq" %99, %7 : !llvm.ptr
    llvm.cond_br %100, ^bb24, ^bb25(%99 : !llvm.ptr)
  ^bb24:  // pred: ^bb23
    llvm.call @_ZN6google8protobuf8internal14ArenaStringPtr14CreateInstanceEPNS0_5ArenaEPKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE(%82, %89, %7) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
    %101 = llvm.load %98 {alignment = 8 : i64, tbaa = [#tbaa_tag8]} : !llvm.ptr -> !llvm.ptr
    llvm.br ^bb25(%101 : !llvm.ptr)
  ^bb25(%102: !llvm.ptr):  // 2 preds: ^bb23, ^bb24
    llvm.call @_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE4swapERS4_(%97, %102) : (!llvm.ptr, !llvm.ptr) -> ()
    llvm.br ^bb26
  ^bb26:  // 2 preds: ^bb21, ^bb25
    %103 = llvm.getelementptr inbounds %arg0[%0, 4] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.tutorial::Person", (struct<"class.google::protobuf::Message", (struct<"class.google::protobuf::MessageLite", (ptr, struct<"class.google::protobuf::internal::InternalMetadata", (ptr)>)>)>, struct<"class.google::protobuf::RepeatedPtrField", (struct<"class.google::protobuf::internal::RepeatedPtrFieldBase", (ptr, i32, i32, ptr)>)>, struct<"struct.google::protobuf::internal::ArenaStringPtr", (ptr)>, struct<"struct.google::protobuf::internal::ArenaStringPtr", (ptr)>, ptr, i32, struct<"class.google::protobuf::internal::CachedSize", (struct<"struct.std::atomic", (struct<"struct.std::__atomic_base", (i32)>)>)>)>
    %104 = llvm.getelementptr inbounds %arg1[%0, 4] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.tutorial::Person", (struct<"class.google::protobuf::Message", (struct<"class.google::protobuf::MessageLite", (ptr, struct<"class.google::protobuf::internal::InternalMetadata", (ptr)>)>)>, struct<"class.google::protobuf::RepeatedPtrField", (struct<"class.google::protobuf::internal::RepeatedPtrFieldBase", (ptr, i32, i32, ptr)>)>, struct<"struct.google::protobuf::internal::ArenaStringPtr", (ptr)>, struct<"struct.google::protobuf::internal::ArenaStringPtr", (ptr)>, ptr, i32, struct<"class.google::protobuf::internal::CachedSize", (struct<"struct.std::atomic", (struct<"struct.std::__atomic_base", (i32)>)>)>)>
    %105 = llvm.bitcast %103 : !llvm.ptr to !llvm.ptr
    %106 = llvm.load %105 {alignment = 8 : i64} : !llvm.ptr -> i64
    %107 = llvm.bitcast %104 : !llvm.ptr to !llvm.ptr
    %108 = llvm.load %107 {alignment = 1 : i64} : !llvm.ptr -> i64
    llvm.store %108, %105 {alignment = 8 : i64} : i64, !llvm.ptr
    llvm.store %106, %107 {alignment = 1 : i64} : i64, !llvm.ptr
    %109 = llvm.getelementptr inbounds %103[%3] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.ptr
    %110 = llvm.getelementptr inbounds %104[%3] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.ptr
    %111 = llvm.bitcast %109 : !llvm.ptr to !llvm.ptr
    %112 = llvm.load %111 {alignment = 8 : i64} : !llvm.ptr -> i32
    %113 = llvm.bitcast %110 : !llvm.ptr to !llvm.ptr
    %114 = llvm.load %113 {alignment = 1 : i64} : !llvm.ptr -> i32
    llvm.store %114, %111 {alignment = 8 : i64} : i32, !llvm.ptr
    llvm.store %112, %113 {alignment = 1 : i64} : i32, !llvm.ptr
    llvm.return
  }
  llvm.func unnamed_addr @_ZNK8tutorial6Person11GetMetadataEv(%arg0: !llvm.ptr {llvm.align = 8 : i64, llvm.nocapture, llvm.nonnull, llvm.readnone}) -> !llvm.struct<(ptr, ptr)> attributes {alignment = 2 : i64, frame_pointer = #llvm.framePointerKind<none>, passthrough = ["mustprogress", ["uwtable", "2"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], ["tune-cpu", "generic"]], target_cpu = "x86-64", target_features = #llvm.target_features<["+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>} {
    %0 = llvm.mlir.zero : !llvm.ptr
    %1 = llvm.mlir.undef : !llvm.array<1 x ptr>
    %2 = llvm.insertvalue %0, %1[0] : !llvm.array<1 x ptr> 
    %3 = llvm.mlir.addressof @_ZL47file_level_enum_descriptors_addressbook_2eproto : !llvm.ptr
    %4 = llvm.mlir.constant(3 : i32) : i32
    %5 = llvm.mlir.undef : !llvm.struct<"struct.google::protobuf::Metadata", (ptr, ptr)>
    %6 = llvm.insertvalue %0, %5[0] : !llvm.struct<"struct.google::protobuf::Metadata", (ptr, ptr)> 
    %7 = llvm.insertvalue %0, %6[1] : !llvm.struct<"struct.google::protobuf::Metadata", (ptr, ptr)> 
    %8 = llvm.mlir.undef : !llvm.array<3 x struct<"struct.google::protobuf::Metadata", (ptr, ptr)>>
    %9 = llvm.insertvalue %7, %8[0] : !llvm.array<3 x struct<"struct.google::protobuf::Metadata", (ptr, ptr)>> 
    %10 = llvm.insertvalue %7, %9[1] : !llvm.array<3 x struct<"struct.google::protobuf::Metadata", (ptr, ptr)>> 
    %11 = llvm.insertvalue %7, %10[2] : !llvm.array<3 x struct<"struct.google::protobuf::Metadata", (ptr, ptr)>> 
    %12 = llvm.mlir.addressof @_ZL39file_level_metadata_addressbook_2eproto : !llvm.ptr
    %13 = llvm.mlir.constant(dense<[-1, 8, -1, -1, -1, 16, 24, -1, 8, -1, -1, -1, 40, 64, 48, 16, 56, -1, 8, -1, -1, -1, 16]> : tensor<23xi32>) : !llvm.array<23 x i32>
    %14 = llvm.mlir.addressof @_ZN31TableStruct_addressbook_2eproto7offsetsE : !llvm.ptr
    %15 = llvm.mlir.constant(0 : i8) : i8
    %16 = llvm.mlir.constant(dense<0> : tensor<40xi8>) : !llvm.array<40 x i8>
    %17 = llvm.mlir.constant(0 : i64) : i64
    %18 = llvm.mlir.undef : !llvm.struct<"union.google::protobuf::internal::ExplicitlyConstructed<tutorial::AddressBook>::AlignedUnion", (i64, array<40 x i8>)>
    %19 = llvm.insertvalue %17, %18[0] : !llvm.struct<"union.google::protobuf::internal::ExplicitlyConstructed<tutorial::AddressBook>::AlignedUnion", (i64, array<40 x i8>)> 
    %20 = llvm.insertvalue %16, %19[1] : !llvm.struct<"union.google::protobuf::internal::ExplicitlyConstructed<tutorial::AddressBook>::AlignedUnion", (i64, array<40 x i8>)> 
    %21 = llvm.mlir.undef : !llvm.struct<"class.google::protobuf::internal::ExplicitlyConstructed.1", (struct<"union.google::protobuf::internal::ExplicitlyConstructed<tutorial::AddressBook>::AlignedUnion", (i64, array<40 x i8>)>)>
    %22 = llvm.insertvalue %20, %21[0] : !llvm.struct<"class.google::protobuf::internal::ExplicitlyConstructed.1", (struct<"union.google::protobuf::internal::ExplicitlyConstructed<tutorial::AddressBook>::AlignedUnion", (i64, array<40 x i8>)>)> 
    %23 = llvm.mlir.undef : !llvm.struct<"class.tutorial::AddressBookDefaultTypeInternal", (struct<"class.google::protobuf::internal::ExplicitlyConstructed.1", (struct<"union.google::protobuf::internal::ExplicitlyConstructed<tutorial::AddressBook>::AlignedUnion", (i64, array<40 x i8>)>)>)>
    %24 = llvm.insertvalue %22, %23[0] : !llvm.struct<"class.tutorial::AddressBookDefaultTypeInternal", (struct<"class.google::protobuf::internal::ExplicitlyConstructed.1", (struct<"union.google::protobuf::internal::ExplicitlyConstructed<tutorial::AddressBook>::AlignedUnion", (i64, array<40 x i8>)>)>)> 
    %25 = llvm.mlir.addressof @_ZN8tutorial30_AddressBook_default_instance_E : !llvm.ptr
    %26 = llvm.mlir.constant(dense<0> : tensor<64xi8>) : !llvm.array<64 x i8>
    %27 = llvm.mlir.undef : !llvm.struct<"union.google::protobuf::internal::ExplicitlyConstructed<tutorial::Person>::AlignedUnion", (i64, array<64 x i8>)>
    %28 = llvm.insertvalue %17, %27[0] : !llvm.struct<"union.google::protobuf::internal::ExplicitlyConstructed<tutorial::Person>::AlignedUnion", (i64, array<64 x i8>)> 
    %29 = llvm.insertvalue %26, %28[1] : !llvm.struct<"union.google::protobuf::internal::ExplicitlyConstructed<tutorial::Person>::AlignedUnion", (i64, array<64 x i8>)> 
    %30 = llvm.mlir.undef : !llvm.struct<"class.google::protobuf::internal::ExplicitlyConstructed.0", (struct<"union.google::protobuf::internal::ExplicitlyConstructed<tutorial::Person>::AlignedUnion", (i64, array<64 x i8>)>)>
    %31 = llvm.insertvalue %29, %30[0] : !llvm.struct<"class.google::protobuf::internal::ExplicitlyConstructed.0", (struct<"union.google::protobuf::internal::ExplicitlyConstructed<tutorial::Person>::AlignedUnion", (i64, array<64 x i8>)>)> 
    %32 = llvm.mlir.undef : !llvm.struct<"class.tutorial::PersonDefaultTypeInternal", (struct<"class.google::protobuf::internal::ExplicitlyConstructed.0", (struct<"union.google::protobuf::internal::ExplicitlyConstructed<tutorial::Person>::AlignedUnion", (i64, array<64 x i8>)>)>)>
    %33 = llvm.insertvalue %31, %32[0] : !llvm.struct<"class.tutorial::PersonDefaultTypeInternal", (struct<"class.google::protobuf::internal::ExplicitlyConstructed.0", (struct<"union.google::protobuf::internal::ExplicitlyConstructed<tutorial::Person>::AlignedUnion", (i64, array<64 x i8>)>)>)> 
    %34 = llvm.mlir.addressof @_ZN8tutorial25_Person_default_instance_E : !llvm.ptr
    %35 = llvm.mlir.constant(dense<0> : tensor<24xi8>) : !llvm.array<24 x i8>
    %36 = llvm.mlir.undef : !llvm.struct<"union.google::protobuf::internal::ExplicitlyConstructed<tutorial::Person_PhoneNumber>::AlignedUnion", (i64, array<24 x i8>)>
    %37 = llvm.insertvalue %17, %36[0] : !llvm.struct<"union.google::protobuf::internal::ExplicitlyConstructed<tutorial::Person_PhoneNumber>::AlignedUnion", (i64, array<24 x i8>)> 
    %38 = llvm.insertvalue %35, %37[1] : !llvm.struct<"union.google::protobuf::internal::ExplicitlyConstructed<tutorial::Person_PhoneNumber>::AlignedUnion", (i64, array<24 x i8>)> 
    %39 = llvm.mlir.undef : !llvm.struct<"class.google::protobuf::internal::ExplicitlyConstructed", (struct<"union.google::protobuf::internal::ExplicitlyConstructed<tutorial::Person_PhoneNumber>::AlignedUnion", (i64, array<24 x i8>)>)>
    %40 = llvm.insertvalue %38, %39[0] : !llvm.struct<"class.google::protobuf::internal::ExplicitlyConstructed", (struct<"union.google::protobuf::internal::ExplicitlyConstructed<tutorial::Person_PhoneNumber>::AlignedUnion", (i64, array<24 x i8>)>)> 
    %41 = llvm.mlir.undef : !llvm.struct<"class.tutorial::Person_PhoneNumberDefaultTypeInternal", (struct<"class.google::protobuf::internal::ExplicitlyConstructed", (struct<"union.google::protobuf::internal::ExplicitlyConstructed<tutorial::Person_PhoneNumber>::AlignedUnion", (i64, array<24 x i8>)>)>)>
    %42 = llvm.insertvalue %40, %41[0] : !llvm.struct<"class.tutorial::Person_PhoneNumberDefaultTypeInternal", (struct<"class.google::protobuf::internal::ExplicitlyConstructed", (struct<"union.google::protobuf::internal::ExplicitlyConstructed<tutorial::Person_PhoneNumber>::AlignedUnion", (i64, array<24 x i8>)>)>)> 
    %43 = llvm.mlir.addressof @_ZN8tutorial37_Person_PhoneNumber_default_instance_E : !llvm.ptr
    %44 = llvm.mlir.undef : !llvm.array<3 x ptr>
    %45 = llvm.insertvalue %43, %44[0] : !llvm.array<3 x ptr> 
    %46 = llvm.insertvalue %34, %45[1] : !llvm.array<3 x ptr> 
    %47 = llvm.insertvalue %25, %46[2] : !llvm.array<3 x ptr> 
    %48 = llvm.mlir.addressof @_ZL22file_default_instances : !llvm.ptr
    %49 = llvm.mlir.constant(48 : i32) : i32
    %50 = llvm.mlir.constant(-1 : i32) : i32
    %51 = llvm.mlir.constant(17 : i32) : i32
    %52 = llvm.mlir.undef : !llvm.struct<"struct.google::protobuf::internal::MigrationSchema", (i32, i32, i32)>
    %53 = llvm.insertvalue %51, %52[0] : !llvm.struct<"struct.google::protobuf::internal::MigrationSchema", (i32, i32, i32)> 
    %54 = llvm.insertvalue %50, %53[1] : !llvm.struct<"struct.google::protobuf::internal::MigrationSchema", (i32, i32, i32)> 
    %55 = llvm.insertvalue %49, %54[2] : !llvm.struct<"struct.google::protobuf::internal::MigrationSchema", (i32, i32, i32)> 
    %56 = llvm.mlir.constant(72 : i32) : i32
    %57 = llvm.mlir.constant(7 : i32) : i32
    %58 = llvm.mlir.undef : !llvm.struct<"struct.google::protobuf::internal::MigrationSchema", (i32, i32, i32)>
    %59 = llvm.insertvalue %57, %58[0] : !llvm.struct<"struct.google::protobuf::internal::MigrationSchema", (i32, i32, i32)> 
    %60 = llvm.insertvalue %50, %59[1] : !llvm.struct<"struct.google::protobuf::internal::MigrationSchema", (i32, i32, i32)> 
    %61 = llvm.insertvalue %56, %60[2] : !llvm.struct<"struct.google::protobuf::internal::MigrationSchema", (i32, i32, i32)> 
    %62 = llvm.mlir.constant(32 : i32) : i32
    %63 = llvm.mlir.constant(0 : i32) : i32
    %64 = llvm.mlir.undef : !llvm.struct<"struct.google::protobuf::internal::MigrationSchema", (i32, i32, i32)>
    %65 = llvm.insertvalue %63, %64[0] : !llvm.struct<"struct.google::protobuf::internal::MigrationSchema", (i32, i32, i32)> 
    %66 = llvm.insertvalue %50, %65[1] : !llvm.struct<"struct.google::protobuf::internal::MigrationSchema", (i32, i32, i32)> 
    %67 = llvm.insertvalue %62, %66[2] : !llvm.struct<"struct.google::protobuf::internal::MigrationSchema", (i32, i32, i32)> 
    %68 = llvm.mlir.undef : !llvm.array<3 x struct<"struct.google::protobuf::internal::MigrationSchema", (i32, i32, i32)>>
    %69 = llvm.insertvalue %67, %68[0] : !llvm.array<3 x struct<"struct.google::protobuf::internal::MigrationSchema", (i32, i32, i32)>> 
    %70 = llvm.insertvalue %61, %69[1] : !llvm.array<3 x struct<"struct.google::protobuf::internal::MigrationSchema", (i32, i32, i32)>> 
    %71 = llvm.insertvalue %55, %70[2] : !llvm.array<3 x struct<"struct.google::protobuf::internal::MigrationSchema", (i32, i32, i32)>> 
    %72 = llvm.mlir.addressof @_ZL7schemas : !llvm.ptr
    %73 = llvm.mlir.constant(1 : i32) : i32
    %74 = llvm.mlir.addressof @descriptor_table_google_2fprotobuf_2ftimestamp_2eproto : !llvm.ptr
    %75 = llvm.mlir.undef : !llvm.array<1 x ptr>
    %76 = llvm.insertvalue %74, %75[0] : !llvm.array<1 x ptr> 
    %77 = llvm.mlir.addressof @_ZL41descriptor_table_addressbook_2eproto_deps : !llvm.ptr
    %78 = llvm.mlir.undef : !llvm.array<0 x ptr>
    %79 = llvm.mlir.addressof @__gxx_personality_v0 : !llvm.ptr
    %80 = llvm.mlir.addressof @_ZL59InitDefaultsscc_info_Person_PhoneNumber_addressbook_2eprotov : !llvm.ptr
    %81 = llvm.mlir.undef : !llvm.struct<(i32)>
    %82 = llvm.insertvalue %50, %81[0] : !llvm.struct<(i32)> 
    %83 = llvm.mlir.undef : !llvm.struct<(struct<(i32)>, i32, i32, ptr)>
    %84 = llvm.insertvalue %82, %83[0] : !llvm.struct<(struct<(i32)>, i32, i32, ptr)> 
    %85 = llvm.insertvalue %63, %84[1] : !llvm.struct<(struct<(i32)>, i32, i32, ptr)> 
    %86 = llvm.insertvalue %63, %85[2] : !llvm.struct<(struct<(i32)>, i32, i32, ptr)> 
    %87 = llvm.insertvalue %80, %86[3] : !llvm.struct<(struct<(i32)>, i32, i32, ptr)> 
    %88 = llvm.mlir.undef : !llvm.struct<(struct<(struct<(i32)>, i32, i32, ptr)>, array<0 x ptr>)>
    %89 = llvm.insertvalue %87, %88[0] : !llvm.struct<(struct<(struct<(i32)>, i32, i32, ptr)>, array<0 x ptr>)> 
    %90 = llvm.insertvalue %78, %89[1] : !llvm.struct<(struct<(struct<(i32)>, i32, i32, ptr)>, array<0 x ptr>)> 
    %91 = llvm.mlir.addressof @scc_info_Person_PhoneNumber_addressbook_2eproto : !llvm.ptr
    %92 = llvm.mlir.addressof @scc_info_Timestamp_google_2fprotobuf_2ftimestamp_2eproto : !llvm.ptr
    %93 = llvm.mlir.undef : !llvm.array<2 x ptr>
    %94 = llvm.insertvalue %91, %93[0] : !llvm.array<2 x ptr> 
    %95 = llvm.insertvalue %92, %94[1] : !llvm.array<2 x ptr> 
    %96 = llvm.mlir.addressof @_ZL47InitDefaultsscc_info_Person_addressbook_2eprotov : !llvm.ptr
    %97 = llvm.mlir.constant(2 : i32) : i32
    %98 = llvm.mlir.undef : !llvm.struct<(struct<(i32)>, i32, i32, ptr)>
    %99 = llvm.insertvalue %82, %98[0] : !llvm.struct<(struct<(i32)>, i32, i32, ptr)> 
    %100 = llvm.insertvalue %97, %99[1] : !llvm.struct<(struct<(i32)>, i32, i32, ptr)> 
    %101 = llvm.insertvalue %63, %100[2] : !llvm.struct<(struct<(i32)>, i32, i32, ptr)> 
    %102 = llvm.insertvalue %96, %101[3] : !llvm.struct<(struct<(i32)>, i32, i32, ptr)> 
    %103 = llvm.mlir.undef : !llvm.struct<(struct<(struct<(i32)>, i32, i32, ptr)>, array<2 x ptr>)>
    %104 = llvm.insertvalue %102, %103[0] : !llvm.struct<(struct<(struct<(i32)>, i32, i32, ptr)>, array<2 x ptr>)> 
    %105 = llvm.insertvalue %95, %104[1] : !llvm.struct<(struct<(struct<(i32)>, i32, i32, ptr)>, array<2 x ptr>)> 
    %106 = llvm.mlir.addressof @scc_info_Person_addressbook_2eproto : !llvm.ptr
    %107 = llvm.mlir.undef : !llvm.array<1 x ptr>
    %108 = llvm.insertvalue %106, %107[0] : !llvm.array<1 x ptr> 
    %109 = llvm.mlir.addressof @_ZL52InitDefaultsscc_info_AddressBook_addressbook_2eprotov : !llvm.ptr
    %110 = llvm.mlir.undef : !llvm.struct<(struct<(i32)>, i32, i32, ptr)>
    %111 = llvm.insertvalue %82, %110[0] : !llvm.struct<(struct<(i32)>, i32, i32, ptr)> 
    %112 = llvm.insertvalue %73, %111[1] : !llvm.struct<(struct<(i32)>, i32, i32, ptr)> 
    %113 = llvm.insertvalue %63, %112[2] : !llvm.struct<(struct<(i32)>, i32, i32, ptr)> 
    %114 = llvm.insertvalue %109, %113[3] : !llvm.struct<(struct<(i32)>, i32, i32, ptr)> 
    %115 = llvm.mlir.undef : !llvm.struct<(struct<(struct<(i32)>, i32, i32, ptr)>, array<1 x ptr>)>
    %116 = llvm.insertvalue %114, %115[0] : !llvm.struct<(struct<(struct<(i32)>, i32, i32, ptr)>, array<1 x ptr>)> 
    %117 = llvm.insertvalue %108, %116[1] : !llvm.struct<(struct<(struct<(i32)>, i32, i32, ptr)>, array<1 x ptr>)> 
    %118 = llvm.mlir.addressof @scc_info_AddressBook_addressbook_2eproto : !llvm.ptr
    %119 = llvm.mlir.undef : !llvm.array<3 x ptr>
    %120 = llvm.insertvalue %118, %119[0] : !llvm.array<3 x ptr> 
    %121 = llvm.insertvalue %106, %120[1] : !llvm.array<3 x ptr> 
    %122 = llvm.insertvalue %91, %121[2] : !llvm.array<3 x ptr> 
    %123 = llvm.mlir.addressof @_ZL41descriptor_table_addressbook_2eproto_sccs : !llvm.ptr
    %124 = llvm.mlir.undef : !llvm.struct<"struct.std::once_flag", (i32)>
    %125 = llvm.insertvalue %63, %124[0] : !llvm.struct<"struct.std::once_flag", (i32)> 
    %126 = llvm.mlir.addressof @_ZL41descriptor_table_addressbook_2eproto_once : !llvm.ptr
    %127 = llvm.mlir.constant(537 : i32) : i32
    %128 = llvm.mlir.constant("addressbook.proto\00") : !llvm.array<18 x i8>
    %129 = llvm.mlir.addressof @".str" : !llvm.ptr
    %130 = llvm.mlir.constant("\0A\11addressbook.proto\12\08tutorial\1A\1Fgoogle/protobuf/timestamp.proto\22\87\02\0A\06Person\12\0C\0A\04name\18\01 \01(\09\12\0A\0A\02id\18\02 \01(\05\12\0D\0A\05email\18\03 \01(\09\12,\0A\06phones\18\04 \03(\0B2\1C.tutorial.Person.PhoneNumber\120\0A\0Clast_updated\18\05 \01(\0B2\1A.google.protobuf.Timestamp\1AG\0A\0BPhoneNumber\12\0E\0A\06number\18\01 \01(\09\12(\0A\04type\18\02 \01(\0E2\1A.tutorial.Person.PhoneType\22+\0A\09PhoneType\12\0A\0A\06MOBILE\10\00\12\08\0A\04HOME\10\01\12\08\0A\04WORK\10\02\22/\0A\0BAddressBook\12 \0A\06people\18\01 \03(\0B2\10.tutorial.PersonB\95\01\0A\1Bcom.example.tutorial.protosB\11AddressBookProtosP\01Z:github.com/protocolbuffers/protobuf/examples/go/tutorialpb\AA\02$Google.Protobuf.Examples.AddressBookb\06proto3\00") : !llvm.array<538 x i8>
    %131 = llvm.mlir.addressof @_ZL45descriptor_table_protodef_addressbook_2eproto : !llvm.ptr
    %132 = llvm.mlir.undef : !llvm.struct<"struct.google::protobuf::internal::DescriptorTable", (i8, i8, ptr, ptr, i32, ptr, ptr, ptr, i32, i32, ptr, ptr, ptr, ptr, i32, ptr, ptr)>
    %133 = llvm.insertvalue %15, %132[0] : !llvm.struct<"struct.google::protobuf::internal::DescriptorTable", (i8, i8, ptr, ptr, i32, ptr, ptr, ptr, i32, i32, ptr, ptr, ptr, ptr, i32, ptr, ptr)> 
    %134 = llvm.insertvalue %15, %133[1] : !llvm.struct<"struct.google::protobuf::internal::DescriptorTable", (i8, i8, ptr, ptr, i32, ptr, ptr, ptr, i32, i32, ptr, ptr, ptr, ptr, i32, ptr, ptr)> 
    %135 = llvm.insertvalue %131, %134[2] : !llvm.struct<"struct.google::protobuf::internal::DescriptorTable", (i8, i8, ptr, ptr, i32, ptr, ptr, ptr, i32, i32, ptr, ptr, ptr, ptr, i32, ptr, ptr)> 
    %136 = llvm.insertvalue %129, %135[3] : !llvm.struct<"struct.google::protobuf::internal::DescriptorTable", (i8, i8, ptr, ptr, i32, ptr, ptr, ptr, i32, i32, ptr, ptr, ptr, ptr, i32, ptr, ptr)> 
    %137 = llvm.insertvalue %127, %136[4] : !llvm.struct<"struct.google::protobuf::internal::DescriptorTable", (i8, i8, ptr, ptr, i32, ptr, ptr, ptr, i32, i32, ptr, ptr, ptr, ptr, i32, ptr, ptr)> 
    %138 = llvm.insertvalue %126, %137[5] : !llvm.struct<"struct.google::protobuf::internal::DescriptorTable", (i8, i8, ptr, ptr, i32, ptr, ptr, ptr, i32, i32, ptr, ptr, ptr, ptr, i32, ptr, ptr)> 
    %139 = llvm.insertvalue %123, %138[6] : !llvm.struct<"struct.google::protobuf::internal::DescriptorTable", (i8, i8, ptr, ptr, i32, ptr, ptr, ptr, i32, i32, ptr, ptr, ptr, ptr, i32, ptr, ptr)> 
    %140 = llvm.insertvalue %77, %139[7] : !llvm.struct<"struct.google::protobuf::internal::DescriptorTable", (i8, i8, ptr, ptr, i32, ptr, ptr, ptr, i32, i32, ptr, ptr, ptr, ptr, i32, ptr, ptr)> 
    %141 = llvm.insertvalue %4, %140[8] : !llvm.struct<"struct.google::protobuf::internal::DescriptorTable", (i8, i8, ptr, ptr, i32, ptr, ptr, ptr, i32, i32, ptr, ptr, ptr, ptr, i32, ptr, ptr)> 
    %142 = llvm.insertvalue %73, %141[9] : !llvm.struct<"struct.google::protobuf::internal::DescriptorTable", (i8, i8, ptr, ptr, i32, ptr, ptr, ptr, i32, i32, ptr, ptr, ptr, ptr, i32, ptr, ptr)> 
    %143 = llvm.insertvalue %72, %142[10] : !llvm.struct<"struct.google::protobuf::internal::DescriptorTable", (i8, i8, ptr, ptr, i32, ptr, ptr, ptr, i32, i32, ptr, ptr, ptr, ptr, i32, ptr, ptr)> 
    %144 = llvm.insertvalue %48, %143[11] : !llvm.struct<"struct.google::protobuf::internal::DescriptorTable", (i8, i8, ptr, ptr, i32, ptr, ptr, ptr, i32, i32, ptr, ptr, ptr, ptr, i32, ptr, ptr)> 
    %145 = llvm.insertvalue %14, %144[12] : !llvm.struct<"struct.google::protobuf::internal::DescriptorTable", (i8, i8, ptr, ptr, i32, ptr, ptr, ptr, i32, i32, ptr, ptr, ptr, ptr, i32, ptr, ptr)> 
    %146 = llvm.insertvalue %12, %145[13] : !llvm.struct<"struct.google::protobuf::internal::DescriptorTable", (i8, i8, ptr, ptr, i32, ptr, ptr, ptr, i32, i32, ptr, ptr, ptr, ptr, i32, ptr, ptr)> 
    %147 = llvm.insertvalue %4, %146[14] : !llvm.struct<"struct.google::protobuf::internal::DescriptorTable", (i8, i8, ptr, ptr, i32, ptr, ptr, ptr, i32, i32, ptr, ptr, ptr, ptr, i32, ptr, ptr)> 
    %148 = llvm.insertvalue %3, %147[15] : !llvm.struct<"struct.google::protobuf::internal::DescriptorTable", (i8, i8, ptr, ptr, i32, ptr, ptr, ptr, i32, i32, ptr, ptr, ptr, ptr, i32, ptr, ptr)> 
    %149 = llvm.insertvalue %0, %148[16] : !llvm.struct<"struct.google::protobuf::internal::DescriptorTable", (i8, i8, ptr, ptr, i32, ptr, ptr, ptr, i32, i32, ptr, ptr, ptr, ptr, i32, ptr, ptr)> 
    %150 = llvm.mlir.addressof @descriptor_table_addressbook_2eproto : !llvm.ptr
    %151 = llvm.mlir.constant(false) : i1
    %152 = llvm.mlir.constant(13 : i32) : i32
    %153 = llvm.getelementptr inbounds %150[%17, 13] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.google::protobuf::internal::DescriptorTable", (i8, i8, ptr, ptr, i32, ptr, ptr, ptr, i32, i32, ptr, ptr, ptr, ptr, i32, ptr, ptr)>
    %154 = llvm.mlir.constant(1 : i64) : i64
    %155 = llvm.mlir.poison : !llvm.struct<(ptr, ptr)>
    llvm.call @_ZN6google8protobuf8internal17AssignDescriptorsEPKNS1_15DescriptorTableEb(%150, %151) : (!llvm.ptr, i1) -> ()
    %156 = llvm.load %153 {alignment = 8 : i64, tbaa = [#tbaa_tag14]} : !llvm.ptr -> !llvm.ptr
    %157 = llvm.getelementptr inbounds %156[%154, 0] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.google::protobuf::Metadata", (ptr, ptr)>
    %158 = llvm.load %157 {alignment = 8 : i64} : !llvm.ptr -> !llvm.ptr
    %159 = llvm.getelementptr inbounds %156[%154, 1] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.google::protobuf::Metadata", (ptr, ptr)>
    %160 = llvm.load %159 {alignment = 8 : i64} : !llvm.ptr -> !llvm.ptr
    %161 = llvm.insertvalue %158, %155[0] : !llvm.struct<(ptr, ptr)> 
    %162 = llvm.insertvalue %160, %161[1] : !llvm.struct<(ptr, ptr)> 
    llvm.return %162 : !llvm.struct<(ptr, ptr)>
  }
  llvm.func local_unnamed_addr @_ZN8tutorial11AddressBook21InitAsDefaultInstanceEv() attributes {alignment = 2 : i64, frame_pointer = #llvm.framePointerKind<none>, memory = #llvm.memory_effects<other = none, argMem = none, inaccessibleMem = none>, passthrough = ["mustprogress", "nofree", "norecurse", "nosync", "nounwind", "willreturn", ["uwtable", "2"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], ["tune-cpu", "generic"]], target_cpu = "x86-64", target_features = #llvm.target_features<["+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>} {
    llvm.return
  }
  llvm.func unnamed_addr @_ZN8tutorial11AddressBookC2EPN6google8protobuf5ArenaE(%arg0: !llvm.ptr {llvm.align = 8 : i64, llvm.dereferenceable = 48 : i64, llvm.nonnull, llvm.noundef}, %arg1: !llvm.ptr {llvm.noundef}) attributes {alignment = 2 : i64, frame_pointer = #llvm.framePointerKind<none>, passthrough = [["uwtable", "2"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], ["tune-cpu", "generic"]], personality = @__gxx_personality_v0, target_cpu = "x86-64", target_features = #llvm.target_features<["+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>} {
    %0 = llvm.mlir.constant(0 : i64) : i64
    %1 = llvm.mlir.constant(0 : i32) : i32
    %2 = llvm.mlir.constant(1 : i32) : i32
    %3 = llvm.mlir.constant(2 : i64) : i64
    %4 = llvm.mlir.addressof @_ZNK8tutorial11AddressBook11GetMetadataEv : !llvm.ptr
    %5 = llvm.mlir.zero : !llvm.ptr
    %6 = llvm.mlir.addressof @__gxx_personality_v0 : !llvm.ptr
    %7 = llvm.mlir.addressof @_ZNK8tutorial11AddressBook13SetCachedSizeEi : !llvm.ptr
    %8 = llvm.mlir.addressof @_ZNK6google8protobuf7Message13SpaceUsedLongEv : !llvm.ptr
    %9 = llvm.mlir.addressof @_ZN6google8protobuf7Message20DiscardUnknownFieldsEv : !llvm.ptr
    %10 = llvm.mlir.addressof @_ZN8tutorial11AddressBook9MergeFromERKN6google8protobuf7MessageE : !llvm.ptr
    %11 = llvm.mlir.addressof @_ZN8tutorial11AddressBook8CopyFromERKN6google8protobuf7MessageE : !llvm.ptr
    %12 = llvm.mlir.addressof @_ZNK6google8protobuf11MessageLite16InternalGetTableEv : !llvm.ptr
    %13 = llvm.mlir.addressof @_ZNK8tutorial11AddressBook18_InternalSerializeEPhPN6google8protobuf2io19EpsCopyOutputStreamE : !llvm.ptr
    %14 = llvm.mlir.addressof @_ZN8tutorial11AddressBook14_InternalParseEPKcPN6google8protobuf8internal12ParseContextE : !llvm.ptr
    %15 = llvm.mlir.addressof @_ZNK8tutorial11AddressBook13GetCachedSizeEv : !llvm.ptr
    %16 = llvm.mlir.addressof @_ZNK8tutorial11AddressBook12ByteSizeLongEv : !llvm.ptr
    %17 = llvm.mlir.addressof @_ZN6google8protobuf7Message21CheckTypeAndMergeFromERKNS0_11MessageLiteE : !llvm.ptr
    %18 = llvm.mlir.addressof @_ZNK6google8protobuf7Message25InitializationErrorStringB5cxx11Ev : !llvm.ptr
    %19 = llvm.mlir.addressof @_ZNK8tutorial11AddressBook13IsInitializedEv : !llvm.ptr
    %20 = llvm.mlir.addressof @_ZN8tutorial11AddressBook5ClearEv : !llvm.ptr
    %21 = llvm.mlir.addressof @_ZNK8tutorial11AddressBook3NewEPN6google8protobuf5ArenaE : !llvm.ptr
    %22 = llvm.mlir.addressof @_ZNK8tutorial11AddressBook3NewEv : !llvm.ptr
    %23 = llvm.mlir.addressof @_ZNK6google8protobuf7Message11GetTypeNameB5cxx11Ev : !llvm.ptr
    %24 = llvm.mlir.addressof @_ZN8tutorial11AddressBookD0Ev : !llvm.ptr
    %25 = llvm.mlir.addressof @_ZN8tutorial11AddressBookD2Ev : !llvm.ptr
    %26 = llvm.mlir.addressof @_ZTIN6google8protobuf7MessageE : !llvm.ptr
    %27 = llvm.mlir.constant("N8tutorial11AddressBookE\00") : !llvm.array<25 x i8>
    %28 = llvm.mlir.addressof @_ZTSN8tutorial11AddressBookE : !llvm.ptr
    %29 = llvm.mlir.addressof @_ZTVN10__cxxabiv120__si_class_type_infoE : !llvm.ptr
    %30 = llvm.getelementptr inbounds %29[%3] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.ptr
    %31 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, ptr)>
    %32 = llvm.insertvalue %30, %31[0] : !llvm.struct<(ptr, ptr, ptr)> 
    %33 = llvm.insertvalue %28, %32[1] : !llvm.struct<(ptr, ptr, ptr)> 
    %34 = llvm.insertvalue %26, %33[2] : !llvm.struct<(ptr, ptr, ptr)> 
    %35 = llvm.mlir.addressof @_ZTIN8tutorial11AddressBookE : !llvm.ptr
    %36 = llvm.mlir.undef : !llvm.array<22 x ptr>
    %37 = llvm.insertvalue %5, %36[0] : !llvm.array<22 x ptr> 
    %38 = llvm.insertvalue %35, %37[1] : !llvm.array<22 x ptr> 
    %39 = llvm.insertvalue %25, %38[2] : !llvm.array<22 x ptr> 
    %40 = llvm.insertvalue %24, %39[3] : !llvm.array<22 x ptr> 
    %41 = llvm.insertvalue %23, %40[4] : !llvm.array<22 x ptr> 
    %42 = llvm.insertvalue %22, %41[5] : !llvm.array<22 x ptr> 
    %43 = llvm.insertvalue %21, %42[6] : !llvm.array<22 x ptr> 
    %44 = llvm.insertvalue %20, %43[7] : !llvm.array<22 x ptr> 
    %45 = llvm.insertvalue %19, %44[8] : !llvm.array<22 x ptr> 
    %46 = llvm.insertvalue %18, %45[9] : !llvm.array<22 x ptr> 
    %47 = llvm.insertvalue %17, %46[10] : !llvm.array<22 x ptr> 
    %48 = llvm.insertvalue %16, %47[11] : !llvm.array<22 x ptr> 
    %49 = llvm.insertvalue %15, %48[12] : !llvm.array<22 x ptr> 
    %50 = llvm.insertvalue %14, %49[13] : !llvm.array<22 x ptr> 
    %51 = llvm.insertvalue %13, %50[14] : !llvm.array<22 x ptr> 
    %52 = llvm.insertvalue %12, %51[15] : !llvm.array<22 x ptr> 
    %53 = llvm.insertvalue %11, %52[16] : !llvm.array<22 x ptr> 
    %54 = llvm.insertvalue %10, %53[17] : !llvm.array<22 x ptr> 
    %55 = llvm.insertvalue %9, %54[18] : !llvm.array<22 x ptr> 
    %56 = llvm.insertvalue %8, %55[19] : !llvm.array<22 x ptr> 
    %57 = llvm.insertvalue %7, %56[20] : !llvm.array<22 x ptr> 
    %58 = llvm.insertvalue %4, %57[21] : !llvm.array<22 x ptr> 
    %59 = llvm.mlir.undef : !llvm.struct<(array<22 x ptr>)>
    %60 = llvm.insertvalue %58, %59[0] : !llvm.struct<(array<22 x ptr>)> 
    %61 = llvm.mlir.addressof @_ZTVN8tutorial11AddressBookE : !llvm.ptr
    %62 = llvm.getelementptr inbounds %61[%0, 0, %3] : (!llvm.ptr, i64, i64) -> !llvm.ptr, !llvm.struct<(array<22 x ptr>)>
    %63 = llvm.mlir.constant(0 : i8) : i8
    %64 = llvm.mlir.constant(20 : i64) : i64
    %65 = llvm.mlir.addressof @scc_info_Timestamp_google_2fprotobuf_2ftimestamp_2eproto : !llvm.ptr
    %66 = llvm.mlir.undef : !llvm.array<0 x ptr>
    %67 = llvm.mlir.addressof @_ZL59InitDefaultsscc_info_Person_PhoneNumber_addressbook_2eprotov : !llvm.ptr
    %68 = llvm.mlir.constant(-1 : i32) : i32
    %69 = llvm.mlir.undef : !llvm.struct<(i32)>
    %70 = llvm.insertvalue %68, %69[0] : !llvm.struct<(i32)> 
    %71 = llvm.mlir.undef : !llvm.struct<(struct<(i32)>, i32, i32, ptr)>
    %72 = llvm.insertvalue %70, %71[0] : !llvm.struct<(struct<(i32)>, i32, i32, ptr)> 
    %73 = llvm.insertvalue %1, %72[1] : !llvm.struct<(struct<(i32)>, i32, i32, ptr)> 
    %74 = llvm.insertvalue %1, %73[2] : !llvm.struct<(struct<(i32)>, i32, i32, ptr)> 
    %75 = llvm.insertvalue %67, %74[3] : !llvm.struct<(struct<(i32)>, i32, i32, ptr)> 
    %76 = llvm.mlir.undef : !llvm.struct<(struct<(struct<(i32)>, i32, i32, ptr)>, array<0 x ptr>)>
    %77 = llvm.insertvalue %75, %76[0] : !llvm.struct<(struct<(struct<(i32)>, i32, i32, ptr)>, array<0 x ptr>)> 
    %78 = llvm.insertvalue %66, %77[1] : !llvm.struct<(struct<(struct<(i32)>, i32, i32, ptr)>, array<0 x ptr>)> 
    %79 = llvm.mlir.addressof @scc_info_Person_PhoneNumber_addressbook_2eproto : !llvm.ptr
    %80 = llvm.mlir.undef : !llvm.array<2 x ptr>
    %81 = llvm.insertvalue %79, %80[0] : !llvm.array<2 x ptr> 
    %82 = llvm.insertvalue %65, %81[1] : !llvm.array<2 x ptr> 
    %83 = llvm.mlir.addressof @_ZL47InitDefaultsscc_info_Person_addressbook_2eprotov : !llvm.ptr
    %84 = llvm.mlir.constant(2 : i32) : i32
    %85 = llvm.mlir.undef : !llvm.struct<(struct<(i32)>, i32, i32, ptr)>
    %86 = llvm.insertvalue %70, %85[0] : !llvm.struct<(struct<(i32)>, i32, i32, ptr)> 
    %87 = llvm.insertvalue %84, %86[1] : !llvm.struct<(struct<(i32)>, i32, i32, ptr)> 
    %88 = llvm.insertvalue %1, %87[2] : !llvm.struct<(struct<(i32)>, i32, i32, ptr)> 
    %89 = llvm.insertvalue %83, %88[3] : !llvm.struct<(struct<(i32)>, i32, i32, ptr)> 
    %90 = llvm.mlir.undef : !llvm.struct<(struct<(struct<(i32)>, i32, i32, ptr)>, array<2 x ptr>)>
    %91 = llvm.insertvalue %89, %90[0] : !llvm.struct<(struct<(struct<(i32)>, i32, i32, ptr)>, array<2 x ptr>)> 
    %92 = llvm.insertvalue %82, %91[1] : !llvm.struct<(struct<(struct<(i32)>, i32, i32, ptr)>, array<2 x ptr>)> 
    %93 = llvm.mlir.addressof @scc_info_Person_addressbook_2eproto : !llvm.ptr
    %94 = llvm.mlir.undef : !llvm.array<1 x ptr>
    %95 = llvm.insertvalue %93, %94[0] : !llvm.array<1 x ptr> 
    %96 = llvm.mlir.addressof @_ZL52InitDefaultsscc_info_AddressBook_addressbook_2eprotov : !llvm.ptr
    %97 = llvm.mlir.undef : !llvm.struct<(struct<(i32)>, i32, i32, ptr)>
    %98 = llvm.insertvalue %70, %97[0] : !llvm.struct<(struct<(i32)>, i32, i32, ptr)> 
    %99 = llvm.insertvalue %2, %98[1] : !llvm.struct<(struct<(i32)>, i32, i32, ptr)> 
    %100 = llvm.insertvalue %1, %99[2] : !llvm.struct<(struct<(i32)>, i32, i32, ptr)> 
    %101 = llvm.insertvalue %96, %100[3] : !llvm.struct<(struct<(i32)>, i32, i32, ptr)> 
    %102 = llvm.mlir.undef : !llvm.struct<(struct<(struct<(i32)>, i32, i32, ptr)>, array<1 x ptr>)>
    %103 = llvm.insertvalue %101, %102[0] : !llvm.struct<(struct<(struct<(i32)>, i32, i32, ptr)>, array<1 x ptr>)> 
    %104 = llvm.insertvalue %95, %103[1] : !llvm.struct<(struct<(struct<(i32)>, i32, i32, ptr)>, array<1 x ptr>)> 
    %105 = llvm.mlir.addressof @scc_info_AddressBook_addressbook_2eproto : !llvm.ptr
    %106 = llvm.getelementptr inbounds %arg0[%0, 0, 0, 1] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.tutorial::AddressBook", packed (struct<"class.google::protobuf::Message", (struct<"class.google::protobuf::MessageLite", (ptr, struct<"class.google::protobuf::internal::InternalMetadata", (ptr)>)>)>, struct<"class.google::protobuf::RepeatedPtrField.18", (struct<"class.google::protobuf::internal::RepeatedPtrFieldBase", (ptr, i32, i32, ptr)>)>, struct<"class.google::protobuf::internal::CachedSize", (struct<"struct.std::atomic", (struct<"struct.std::__atomic_base", (i32)>)>)>, array<4 x i8>)>
    %107 = llvm.bitcast %106 : !llvm.ptr to !llvm.ptr
    llvm.store %arg1, %107 {alignment = 8 : i64, tbaa = [#tbaa_tag5]} : !llvm.ptr, !llvm.ptr
    %108 = llvm.getelementptr inbounds %arg0[%0, 0, 0, 0] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.tutorial::AddressBook", packed (struct<"class.google::protobuf::Message", (struct<"class.google::protobuf::MessageLite", (ptr, struct<"class.google::protobuf::internal::InternalMetadata", (ptr)>)>)>, struct<"class.google::protobuf::RepeatedPtrField.18", (struct<"class.google::protobuf::internal::RepeatedPtrFieldBase", (ptr, i32, i32, ptr)>)>, struct<"class.google::protobuf::internal::CachedSize", (struct<"struct.std::atomic", (struct<"struct.std::__atomic_base", (i32)>)>)>, array<4 x i8>)>
    llvm.store %62, %108 {alignment = 8 : i64, tbaa = [#tbaa_tag]} : !llvm.ptr, !llvm.ptr
    %109 = llvm.getelementptr inbounds %arg0[%0, 1] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.tutorial::AddressBook", packed (struct<"class.google::protobuf::Message", (struct<"class.google::protobuf::MessageLite", (ptr, struct<"class.google::protobuf::internal::InternalMetadata", (ptr)>)>)>, struct<"class.google::protobuf::RepeatedPtrField.18", (struct<"class.google::protobuf::internal::RepeatedPtrFieldBase", (ptr, i32, i32, ptr)>)>, struct<"class.google::protobuf::internal::CachedSize", (struct<"struct.std::atomic", (struct<"struct.std::__atomic_base", (i32)>)>)>, array<4 x i8>)>
    %110 = llvm.getelementptr inbounds %109[%0, 0, 0] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.google::protobuf::RepeatedPtrField.18", (struct<"class.google::protobuf::internal::RepeatedPtrFieldBase", (ptr, i32, i32, ptr)>)>
    llvm.store %arg1, %110 {alignment = 8 : i64, tbaa = [#tbaa_tag7]} : !llvm.ptr, !llvm.ptr
    %111 = llvm.getelementptr inbounds %arg0[%0, 1, 0, 1] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.tutorial::AddressBook", packed (struct<"class.google::protobuf::Message", (struct<"class.google::protobuf::MessageLite", (ptr, struct<"class.google::protobuf::internal::InternalMetadata", (ptr)>)>)>, struct<"class.google::protobuf::RepeatedPtrField.18", (struct<"class.google::protobuf::internal::RepeatedPtrFieldBase", (ptr, i32, i32, ptr)>)>, struct<"class.google::protobuf::internal::CachedSize", (struct<"struct.std::atomic", (struct<"struct.std::__atomic_base", (i32)>)>)>, array<4 x i8>)>
    %112 = llvm.bitcast %111 : !llvm.ptr to !llvm.ptr
    "llvm.intr.memset"(%112, %63, %64) <{isVolatile = false}> : (!llvm.ptr, i8, i64) -> ()
    %113 = llvm.load %105 atomic acquire {alignment = 8 : i64} : !llvm.ptr -> i32
    %114 = llvm.icmp "eq" %113, %1 : i32
    llvm.cond_br %114 weights([2000, 1]), ^bb2, ^bb1
  ^bb1:  // pred: ^bb0
    llvm.invoke @_ZN6google8protobuf8internal11InitSCCImplEPNS1_11SCCInfoBaseE(%105) to ^bb2 unwind ^bb3 : (!llvm.ptr) -> ()
  ^bb2:  // 2 preds: ^bb0, ^bb1
    llvm.return
  ^bb3:  // pred: ^bb1
    %115 = llvm.landingpad cleanup : !llvm.struct<(ptr, i32)>
    llvm.call @_ZN6google8protobuf16RepeatedPtrFieldIN8tutorial6PersonEED2Ev(%109) : (!llvm.ptr) -> ()
    llvm.resume %115 : !llvm.struct<(ptr, i32)>
  }
  llvm.func linkonce_odr unnamed_addr @_ZN6google8protobuf16RepeatedPtrFieldIN8tutorial6PersonEED2Ev(%arg0: !llvm.ptr {llvm.align = 8 : i64, llvm.dereferenceable = 24 : i64, llvm.nonnull, llvm.noundef}) comdat(@__llvm_global_comdat::@_ZN6google8protobuf16RepeatedPtrFieldIN8tutorial6PersonEED2Ev) attributes {alignment = 2 : i64, frame_pointer = #llvm.framePointerKind<none>, passthrough = ["nounwind", ["uwtable", "2"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], ["tune-cpu", "generic"]], personality = @__gxx_personality_v0, target_cpu = "x86-64", target_features = #llvm.target_features<["+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>} {
    %0 = llvm.mlir.constant(0 : i64) : i64
    %1 = llvm.mlir.constant(0 : i32) : i32
    %2 = llvm.mlir.constant(3 : i32) : i32
    %3 = llvm.mlir.zero : !llvm.ptr
    %4 = llvm.mlir.constant(false) : i1
    %5 = llvm.mlir.constant(1 : i32) : i32
    %6 = llvm.mlir.constant(1 : i64) : i64
    %7 = llvm.getelementptr inbounds %arg0[%0, 0, 3] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.google::protobuf::RepeatedPtrField.18", (struct<"class.google::protobuf::internal::RepeatedPtrFieldBase", (ptr, i32, i32, ptr)>)>
    %8 = llvm.load %7 {alignment = 8 : i64, tbaa = [#tbaa_tag15]} : !llvm.ptr -> !llvm.ptr
    %9 = llvm.icmp "ne" %8, %3 : !llvm.ptr
    %10 = llvm.getelementptr inbounds %arg0[%0, 0, 0] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.google::protobuf::RepeatedPtrField.18", (struct<"class.google::protobuf::internal::RepeatedPtrFieldBase", (ptr, i32, i32, ptr)>)>
    %11 = llvm.load %10 {alignment = 8 : i64} : !llvm.ptr -> !llvm.ptr
    %12 = llvm.icmp "eq" %11, %3 : !llvm.ptr
    %13 = llvm.select %9, %12, %4 : i1, i1
    llvm.cond_br %13, ^bb1, ^bb8(%11 : !llvm.ptr)
  ^bb1:  // pred: ^bb0
    %14 = llvm.bitcast %8 : !llvm.ptr to !llvm.ptr
    %15 = llvm.getelementptr inbounds %8[%0, 0] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.google::protobuf::internal::RepeatedPtrFieldBase::Rep", (i32, array<1 x ptr>)>
    %16 = llvm.load %15 {alignment = 8 : i64, tbaa = [#tbaa_tag18]} : !llvm.ptr -> i32
    %17 = llvm.icmp "sgt" %16, %1 : i32
    llvm.cond_br %17, ^bb2, ^bb4(%14 : !llvm.ptr)
  ^bb2:  // pred: ^bb1
    %18 = llvm.zext %16 : i32 to i64
    llvm.br ^bb5(%0 : i64)
  ^bb3:  // pred: ^bb7
    %19 = llvm.bitcast %7 : !llvm.ptr to !llvm.ptr
    %20 = llvm.load %19 {alignment = 8 : i64, tbaa = [#tbaa_tag15]} : !llvm.ptr -> !llvm.ptr
    llvm.br ^bb4(%20 : !llvm.ptr)
  ^bb4(%21: !llvm.ptr):  // 2 preds: ^bb1, ^bb3
    llvm.call @_ZdlPv(%21) : (!llvm.ptr) -> ()
    %22 = llvm.load %10 {alignment = 8 : i64, tbaa = [#tbaa_tag7]} : !llvm.ptr -> !llvm.ptr
    llvm.br ^bb8(%22 : !llvm.ptr)
  ^bb5(%23: i64):  // 2 preds: ^bb2, ^bb7
    %24 = llvm.getelementptr inbounds %8[%0, 1, %23] : (!llvm.ptr, i64, i64) -> !llvm.ptr, !llvm.struct<"struct.google::protobuf::internal::RepeatedPtrFieldBase::Rep", (i32, array<1 x ptr>)>
    %25 = llvm.load %24 {alignment = 8 : i64, tbaa = [#tbaa_tag2]} : !llvm.ptr -> !llvm.ptr
    %26 = llvm.icmp "eq" %25, %3 : !llvm.ptr
    llvm.cond_br %26, ^bb7, ^bb6
  ^bb6:  // pred: ^bb5
    %27 = llvm.bitcast %25 : !llvm.ptr to !llvm.ptr
    llvm.call @_ZN8tutorial6PersonD2Ev(%27) : (!llvm.ptr) -> ()
    llvm.call @_ZdlPv(%25) : (!llvm.ptr) -> ()
    llvm.br ^bb7
  ^bb7:  // 2 preds: ^bb5, ^bb6
    %28 = llvm.add %23, %6 overflow<nsw, nuw>  : i64
    %29 = llvm.icmp "eq" %28, %18 : i64
    llvm.cond_br %29, ^bb3, ^bb5(%28 : i64) {loop_annotation = #loop_annotation}
  ^bb8(%30: !llvm.ptr):  // 2 preds: ^bb0, ^bb4
    llvm.store %3, %7 {alignment = 8 : i64, tbaa = [#tbaa_tag15]} : !llvm.ptr, !llvm.ptr
    %31 = llvm.icmp "eq" %30, %3 : !llvm.ptr
    llvm.cond_br %31, ^bb11, ^bb9
  ^bb9:  // pred: ^bb8
    %32 = llvm.getelementptr inbounds %30[%0, 0] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.google::protobuf::Arena", (struct<"class.google::protobuf::internal::ArenaImpl", (struct<"struct.std::atomic.10", (struct<"struct.std::__atomic_base.11", (ptr)>)>, struct<"struct.std::atomic.10", (struct<"struct.std::__atomic_base.11", (ptr)>)>, struct<"struct.std::atomic.12", (struct<"struct.std::__atomic_base.13", (i64)>)>, ptr, i64, struct<"struct.google::protobuf::internal::ArenaImpl::Options", (i64, i64, ptr, i64, ptr, ptr)>)>, ptr, ptr, ptr, ptr)>
    %33 = llvm.invoke @_ZNK6google8protobuf8internal9ArenaImpl14SpaceAllocatedEv(%32) to ^bb11 unwind ^bb10 : (!llvm.ptr) -> i64
  ^bb10:  // pred: ^bb9
    %34 = llvm.landingpad (catch %3 : !llvm.ptr) : !llvm.struct<(ptr, i32)>
    %35 = llvm.extractvalue %34[0] : !llvm.struct<(ptr, i32)> 
    llvm.call @__clang_call_terminate(%35) : (!llvm.ptr) -> ()
    llvm.unreachable
  ^bb11:  // 2 preds: ^bb8, ^bb9
    llvm.return
  }
  llvm.func unnamed_addr @_ZN8tutorial11AddressBookC2ERKS0_(%arg0: !llvm.ptr {llvm.align = 8 : i64, llvm.dereferenceable = 48 : i64, llvm.nonnull, llvm.noundef}, %arg1: !llvm.ptr {llvm.align = 8 : i64, llvm.dereferenceable = 48 : i64, llvm.nonnull, llvm.noundef}) attributes {alignment = 2 : i64, frame_pointer = #llvm.framePointerKind<none>, passthrough = [["uwtable", "2"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], ["tune-cpu", "generic"]], personality = @__gxx_personality_v0, target_cpu = "x86-64", target_features = #llvm.target_features<["+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>} {
    %0 = llvm.mlir.constant(0 : i64) : i64
    %1 = llvm.mlir.constant(0 : i32) : i32
    %2 = llvm.mlir.constant(1 : i32) : i32
    %3 = llvm.mlir.zero : !llvm.ptr
    %4 = llvm.mlir.constant(2 : i64) : i64
    %5 = llvm.mlir.addressof @_ZNK8tutorial11AddressBook11GetMetadataEv : !llvm.ptr
    %6 = llvm.mlir.addressof @__gxx_personality_v0 : !llvm.ptr
    %7 = llvm.mlir.addressof @_ZNK8tutorial11AddressBook13SetCachedSizeEi : !llvm.ptr
    %8 = llvm.mlir.addressof @_ZNK6google8protobuf7Message13SpaceUsedLongEv : !llvm.ptr
    %9 = llvm.mlir.addressof @_ZN6google8protobuf7Message20DiscardUnknownFieldsEv : !llvm.ptr
    %10 = llvm.mlir.addressof @_ZN8tutorial11AddressBook9MergeFromERKN6google8protobuf7MessageE : !llvm.ptr
    %11 = llvm.mlir.addressof @_ZN8tutorial11AddressBook8CopyFromERKN6google8protobuf7MessageE : !llvm.ptr
    %12 = llvm.mlir.addressof @_ZNK6google8protobuf11MessageLite16InternalGetTableEv : !llvm.ptr
    %13 = llvm.mlir.addressof @_ZNK8tutorial11AddressBook18_InternalSerializeEPhPN6google8protobuf2io19EpsCopyOutputStreamE : !llvm.ptr
    %14 = llvm.mlir.addressof @_ZN8tutorial11AddressBook14_InternalParseEPKcPN6google8protobuf8internal12ParseContextE : !llvm.ptr
    %15 = llvm.mlir.addressof @_ZNK8tutorial11AddressBook13GetCachedSizeEv : !llvm.ptr
    %16 = llvm.mlir.addressof @_ZNK8tutorial11AddressBook12ByteSizeLongEv : !llvm.ptr
    %17 = llvm.mlir.addressof @_ZN6google8protobuf7Message21CheckTypeAndMergeFromERKNS0_11MessageLiteE : !llvm.ptr
    %18 = llvm.mlir.addressof @_ZNK6google8protobuf7Message25InitializationErrorStringB5cxx11Ev : !llvm.ptr
    %19 = llvm.mlir.addressof @_ZNK8tutorial11AddressBook13IsInitializedEv : !llvm.ptr
    %20 = llvm.mlir.addressof @_ZN8tutorial11AddressBook5ClearEv : !llvm.ptr
    %21 = llvm.mlir.addressof @_ZNK8tutorial11AddressBook3NewEPN6google8protobuf5ArenaE : !llvm.ptr
    %22 = llvm.mlir.addressof @_ZNK8tutorial11AddressBook3NewEv : !llvm.ptr
    %23 = llvm.mlir.addressof @_ZNK6google8protobuf7Message11GetTypeNameB5cxx11Ev : !llvm.ptr
    %24 = llvm.mlir.addressof @_ZN8tutorial11AddressBookD0Ev : !llvm.ptr
    %25 = llvm.mlir.addressof @_ZN8tutorial11AddressBookD2Ev : !llvm.ptr
    %26 = llvm.mlir.addressof @_ZTIN6google8protobuf7MessageE : !llvm.ptr
    %27 = llvm.mlir.constant("N8tutorial11AddressBookE\00") : !llvm.array<25 x i8>
    %28 = llvm.mlir.addressof @_ZTSN8tutorial11AddressBookE : !llvm.ptr
    %29 = llvm.mlir.addressof @_ZTVN10__cxxabiv120__si_class_type_infoE : !llvm.ptr
    %30 = llvm.getelementptr inbounds %29[%4] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.ptr
    %31 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, ptr)>
    %32 = llvm.insertvalue %30, %31[0] : !llvm.struct<(ptr, ptr, ptr)> 
    %33 = llvm.insertvalue %28, %32[1] : !llvm.struct<(ptr, ptr, ptr)> 
    %34 = llvm.insertvalue %26, %33[2] : !llvm.struct<(ptr, ptr, ptr)> 
    %35 = llvm.mlir.addressof @_ZTIN8tutorial11AddressBookE : !llvm.ptr
    %36 = llvm.mlir.undef : !llvm.array<22 x ptr>
    %37 = llvm.insertvalue %3, %36[0] : !llvm.array<22 x ptr> 
    %38 = llvm.insertvalue %35, %37[1] : !llvm.array<22 x ptr> 
    %39 = llvm.insertvalue %25, %38[2] : !llvm.array<22 x ptr> 
    %40 = llvm.insertvalue %24, %39[3] : !llvm.array<22 x ptr> 
    %41 = llvm.insertvalue %23, %40[4] : !llvm.array<22 x ptr> 
    %42 = llvm.insertvalue %22, %41[5] : !llvm.array<22 x ptr> 
    %43 = llvm.insertvalue %21, %42[6] : !llvm.array<22 x ptr> 
    %44 = llvm.insertvalue %20, %43[7] : !llvm.array<22 x ptr> 
    %45 = llvm.insertvalue %19, %44[8] : !llvm.array<22 x ptr> 
    %46 = llvm.insertvalue %18, %45[9] : !llvm.array<22 x ptr> 
    %47 = llvm.insertvalue %17, %46[10] : !llvm.array<22 x ptr> 
    %48 = llvm.insertvalue %16, %47[11] : !llvm.array<22 x ptr> 
    %49 = llvm.insertvalue %15, %48[12] : !llvm.array<22 x ptr> 
    %50 = llvm.insertvalue %14, %49[13] : !llvm.array<22 x ptr> 
    %51 = llvm.insertvalue %13, %50[14] : !llvm.array<22 x ptr> 
    %52 = llvm.insertvalue %12, %51[15] : !llvm.array<22 x ptr> 
    %53 = llvm.insertvalue %11, %52[16] : !llvm.array<22 x ptr> 
    %54 = llvm.insertvalue %10, %53[17] : !llvm.array<22 x ptr> 
    %55 = llvm.insertvalue %9, %54[18] : !llvm.array<22 x ptr> 
    %56 = llvm.insertvalue %8, %55[19] : !llvm.array<22 x ptr> 
    %57 = llvm.insertvalue %7, %56[20] : !llvm.array<22 x ptr> 
    %58 = llvm.insertvalue %5, %57[21] : !llvm.array<22 x ptr> 
    %59 = llvm.mlir.undef : !llvm.struct<(array<22 x ptr>)>
    %60 = llvm.insertvalue %58, %59[0] : !llvm.struct<(array<22 x ptr>)> 
    %61 = llvm.mlir.addressof @_ZTVN8tutorial11AddressBookE : !llvm.ptr
    %62 = llvm.getelementptr inbounds %61[%0, 0, %4] : (!llvm.ptr, i64, i64) -> !llvm.ptr, !llvm.struct<(array<22 x ptr>)>
    %63 = llvm.mlir.constant(0 : i8) : i8
    %64 = llvm.mlir.constant(24 : i64) : i64
    %65 = llvm.mlir.constant(2 : i32) : i32
    %66 = llvm.mlir.constant(1 : i64) : i64
    %67 = llvm.mlir.constant(-2 : i64) : i64
    %68 = llvm.getelementptr inbounds %arg0[%0, 0, 0, 1, 0] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.tutorial::AddressBook", packed (struct<"class.google::protobuf::Message", (struct<"class.google::protobuf::MessageLite", (ptr, struct<"class.google::protobuf::internal::InternalMetadata", (ptr)>)>)>, struct<"class.google::protobuf::RepeatedPtrField.18", (struct<"class.google::protobuf::internal::RepeatedPtrFieldBase", (ptr, i32, i32, ptr)>)>, struct<"class.google::protobuf::internal::CachedSize", (struct<"struct.std::atomic", (struct<"struct.std::__atomic_base", (i32)>)>)>, array<4 x i8>)>
    llvm.store %3, %68 {alignment = 8 : i64, tbaa = [#tbaa_tag5]} : !llvm.ptr, !llvm.ptr
    %69 = llvm.getelementptr inbounds %arg0[%0, 0, 0, 0] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.tutorial::AddressBook", packed (struct<"class.google::protobuf::Message", (struct<"class.google::protobuf::MessageLite", (ptr, struct<"class.google::protobuf::internal::InternalMetadata", (ptr)>)>)>, struct<"class.google::protobuf::RepeatedPtrField.18", (struct<"class.google::protobuf::internal::RepeatedPtrFieldBase", (ptr, i32, i32, ptr)>)>, struct<"class.google::protobuf::internal::CachedSize", (struct<"struct.std::atomic", (struct<"struct.std::__atomic_base", (i32)>)>)>, array<4 x i8>)>
    llvm.store %62, %69 {alignment = 8 : i64, tbaa = [#tbaa_tag]} : !llvm.ptr, !llvm.ptr
    %70 = llvm.getelementptr inbounds %arg0[%0, 1] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.tutorial::AddressBook", packed (struct<"class.google::protobuf::Message", (struct<"class.google::protobuf::MessageLite", (ptr, struct<"class.google::protobuf::internal::InternalMetadata", (ptr)>)>)>, struct<"class.google::protobuf::RepeatedPtrField.18", (struct<"class.google::protobuf::internal::RepeatedPtrFieldBase", (ptr, i32, i32, ptr)>)>, struct<"class.google::protobuf::internal::CachedSize", (struct<"struct.std::atomic", (struct<"struct.std::__atomic_base", (i32)>)>)>, array<4 x i8>)>
    %71 = llvm.getelementptr inbounds %70[%0, 0] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.google::protobuf::RepeatedPtrField.18", (struct<"class.google::protobuf::internal::RepeatedPtrFieldBase", (ptr, i32, i32, ptr)>)>
    %72 = llvm.bitcast %70 : !llvm.ptr to !llvm.ptr
    "llvm.intr.memset"(%72, %63, %64) <{isVolatile = false}> : (!llvm.ptr, i8, i64) -> ()
    %73 = llvm.getelementptr inbounds %arg1[%0, 1, 0] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.tutorial::AddressBook", packed (struct<"class.google::protobuf::Message", (struct<"class.google::protobuf::MessageLite", (ptr, struct<"class.google::protobuf::internal::InternalMetadata", (ptr)>)>)>, struct<"class.google::protobuf::RepeatedPtrField.18", (struct<"class.google::protobuf::internal::RepeatedPtrFieldBase", (ptr, i32, i32, ptr)>)>, struct<"class.google::protobuf::internal::CachedSize", (struct<"struct.std::atomic", (struct<"struct.std::__atomic_base", (i32)>)>)>, array<4 x i8>)>
    llvm.invoke @_ZN6google8protobuf8internal20RepeatedPtrFieldBase9MergeFromINS0_16RepeatedPtrFieldIN8tutorial6PersonEE11TypeHandlerEEEvRKS2_(%71, %73) to ^bb2 unwind ^bb1 : (!llvm.ptr, !llvm.ptr) -> ()
  ^bb1:  // pred: ^bb0
    %74 = llvm.landingpad cleanup : !llvm.struct<(ptr, i32)>
    llvm.call @_ZN6google8protobuf8internal20RepeatedPtrFieldBaseD2Ev(%71) : (!llvm.ptr) -> ()
    llvm.br ^bb10(%74 : !llvm.struct<(ptr, i32)>)
  ^bb2:  // pred: ^bb0
    %75 = llvm.getelementptr inbounds %arg0[%0, 2, 0, 0, 0] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.tutorial::AddressBook", packed (struct<"class.google::protobuf::Message", (struct<"class.google::protobuf::MessageLite", (ptr, struct<"class.google::protobuf::internal::InternalMetadata", (ptr)>)>)>, struct<"class.google::protobuf::RepeatedPtrField.18", (struct<"class.google::protobuf::internal::RepeatedPtrFieldBase", (ptr, i32, i32, ptr)>)>, struct<"class.google::protobuf::internal::CachedSize", (struct<"struct.std::atomic", (struct<"struct.std::__atomic_base", (i32)>)>)>, array<4 x i8>)>
    llvm.store %1, %75 {alignment = 8 : i64, tbaa = [#tbaa_tag6]} : i32, !llvm.ptr
    %76 = llvm.getelementptr inbounds %arg0[%0, 0, 0, 1] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.tutorial::AddressBook", packed (struct<"class.google::protobuf::Message", (struct<"class.google::protobuf::MessageLite", (ptr, struct<"class.google::protobuf::internal::InternalMetadata", (ptr)>)>)>, struct<"class.google::protobuf::RepeatedPtrField.18", (struct<"class.google::protobuf::internal::RepeatedPtrFieldBase", (ptr, i32, i32, ptr)>)>, struct<"class.google::protobuf::internal::CachedSize", (struct<"struct.std::atomic", (struct<"struct.std::__atomic_base", (i32)>)>)>, array<4 x i8>)>
    %77 = llvm.getelementptr inbounds %arg1[%0, 0, 0, 1, 0] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.tutorial::AddressBook", packed (struct<"class.google::protobuf::Message", (struct<"class.google::protobuf::MessageLite", (ptr, struct<"class.google::protobuf::internal::InternalMetadata", (ptr)>)>)>, struct<"class.google::protobuf::RepeatedPtrField.18", (struct<"class.google::protobuf::internal::RepeatedPtrFieldBase", (ptr, i32, i32, ptr)>)>, struct<"class.google::protobuf::internal::CachedSize", (struct<"struct.std::atomic", (struct<"struct.std::__atomic_base", (i32)>)>)>, array<4 x i8>)>
    %78 = llvm.load %77 {alignment = 8 : i64, tbaa = [#tbaa_tag5]} : !llvm.ptr -> !llvm.ptr
    %79 = llvm.ptrtoint %78 : !llvm.ptr to i64
    %80 = llvm.and %79, %66  : i64
    %81 = llvm.icmp "eq" %80, %0 : i64
    llvm.cond_br %81, ^bb8, ^bb3
  ^bb3:  // pred: ^bb2
    %82 = llvm.and %79, %67  : i64
    %83 = llvm.inttoptr %82 : i64 to !llvm.ptr
    %84 = llvm.getelementptr inbounds %83[%0, 1] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.google::protobuf::internal::InternalMetadata::Container", (struct<"struct.google::protobuf::internal::InternalMetadata::ContainerBase", (ptr)>, struct<"class.google::protobuf::UnknownFieldSet", (struct<"class.std::vector", (struct<"struct.std::_Vector_base", (struct<"struct.std::_Vector_base<google::protobuf::UnknownField, std::allocator<google::protobuf::UnknownField>>::_Vector_impl", (struct<"struct.std::_Vector_base<google::protobuf::UnknownField, std::allocator<google::protobuf::UnknownField>>::_Vector_impl_data", (ptr, ptr, ptr)>)>)>)>)>)>
    %85 = llvm.getelementptr inbounds %76[%0, 0] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.google::protobuf::internal::InternalMetadata", (ptr)>
    %86 = llvm.load %85 {alignment = 8 : i64, tbaa = [#tbaa_tag5]} : !llvm.ptr -> !llvm.ptr
    %87 = llvm.ptrtoint %86 : !llvm.ptr to i64
    %88 = llvm.and %87, %66  : i64
    %89 = llvm.icmp "eq" %88, %0 : i64
    llvm.cond_br %89 weights([1, 2000]), ^bb5, ^bb4
  ^bb4:  // pred: ^bb3
    %90 = llvm.and %87, %67  : i64
    %91 = llvm.inttoptr %90 : i64 to !llvm.ptr
    %92 = llvm.getelementptr inbounds %91[%0, 1] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.google::protobuf::internal::InternalMetadata::Container", (struct<"struct.google::protobuf::internal::InternalMetadata::ContainerBase", (ptr)>, struct<"class.google::protobuf::UnknownFieldSet", (struct<"class.std::vector", (struct<"struct.std::_Vector_base", (struct<"struct.std::_Vector_base<google::protobuf::UnknownField, std::allocator<google::protobuf::UnknownField>>::_Vector_impl", (struct<"struct.std::_Vector_base<google::protobuf::UnknownField, std::allocator<google::protobuf::UnknownField>>::_Vector_impl_data", (ptr, ptr, ptr)>)>)>)>)>)>
    llvm.br ^bb7(%92 : !llvm.ptr)
  ^bb5:  // pred: ^bb3
    %93 = llvm.invoke @_ZN6google8protobuf8internal16InternalMetadata27mutable_unknown_fields_slowINS0_15UnknownFieldSetEEEPT_v(%76) to ^bb6 unwind ^bb9 : (!llvm.ptr) -> !llvm.ptr
  ^bb6:  // pred: ^bb5
    llvm.br ^bb7(%93 : !llvm.ptr)
  ^bb7(%94: !llvm.ptr):  // 2 preds: ^bb4, ^bb6
    llvm.invoke @_ZN6google8protobuf15UnknownFieldSet9MergeFromERKS1_(%94, %84) to ^bb8 unwind ^bb9 : (!llvm.ptr, !llvm.ptr) -> ()
  ^bb8:  // 2 preds: ^bb2, ^bb7
    llvm.return
  ^bb9:  // 2 preds: ^bb5, ^bb7
    %95 = llvm.landingpad cleanup : !llvm.struct<(ptr, i32)>
    llvm.call @_ZN6google8protobuf16RepeatedPtrFieldIN8tutorial6PersonEED2Ev(%70) : (!llvm.ptr) -> ()
    llvm.br ^bb10(%95 : !llvm.struct<(ptr, i32)>)
  ^bb10(%96: !llvm.struct<(ptr, i32)>):  // 2 preds: ^bb1, ^bb9
    llvm.resume %96 : !llvm.struct<(ptr, i32)>
  }
  llvm.func unnamed_addr @_ZN8tutorial11AddressBookD2Ev(%arg0: !llvm.ptr {llvm.align = 8 : i64, llvm.dereferenceable = 48 : i64, llvm.nonnull, llvm.noundef}) attributes {alignment = 2 : i64, frame_pointer = #llvm.framePointerKind<none>, passthrough = ["nounwind", ["uwtable", "2"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], ["tune-cpu", "generic"]], personality = @__gxx_personality_v0, target_cpu = "x86-64", target_features = #llvm.target_features<["+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>} {
    %0 = llvm.mlir.constant(1 : i32) : i32
    %1 = llvm.mlir.constant(0 : i64) : i64
    %2 = llvm.mlir.constant(0 : i32) : i32
    %3 = llvm.mlir.constant(1 : i64) : i64
    %4 = llvm.mlir.constant(-2 : i64) : i64
    %5 = llvm.mlir.zero : !llvm.ptr
    %6 = llvm.mlir.constant(3 : i32) : i32
    %7 = llvm.mlir.constant("build/addressbook.pb.cc\00") : !llvm.array<24 x i8>
    %8 = llvm.mlir.addressof @".str.5" : !llvm.ptr
    %9 = llvm.mlir.constant(810 : i32) : i32
    %10 = llvm.mlir.constant("CHECK failed: GetArena() == nullptr: \00") : !llvm.array<38 x i8>
    %11 = llvm.mlir.addressof @".str.12" : !llvm.ptr
    %12 = llvm.alloca %0 x !llvm.struct<"class.google::protobuf::internal::LogMessage", (i32, ptr, i32, struct<"class.std::__cxx11::basic_string", (struct<"struct.std::__cxx11::basic_string<char>::_Alloc_hider", (ptr)>, i64, struct<"union.anon", (i64, array<8 x i8>)>)>)> {alignment = 8 : i64} : (i32) -> !llvm.ptr
    %13 = llvm.alloca %0 x !llvm.struct<"class.google::protobuf::internal::LogFinisher", (i8)> {alignment = 1 : i64} : (i32) -> !llvm.ptr
    %14 = llvm.getelementptr inbounds %arg0[%1, 0, 0, 1, 0] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.tutorial::AddressBook", packed (struct<"class.google::protobuf::Message", (struct<"class.google::protobuf::MessageLite", (ptr, struct<"class.google::protobuf::internal::InternalMetadata", (ptr)>)>)>, struct<"class.google::protobuf::RepeatedPtrField.18", (struct<"class.google::protobuf::internal::RepeatedPtrFieldBase", (ptr, i32, i32, ptr)>)>, struct<"class.google::protobuf::internal::CachedSize", (struct<"struct.std::atomic", (struct<"struct.std::__atomic_base", (i32)>)>)>, array<4 x i8>)>
    %15 = llvm.load %14 {alignment = 8 : i64, tbaa = [#tbaa_tag5]} : !llvm.ptr -> !llvm.ptr
    %16 = llvm.ptrtoint %15 : !llvm.ptr to i64
    %17 = llvm.and %16, %3  : i64
    %18 = llvm.icmp "eq" %17, %1 : i64
    %19 = llvm.and %16, %4  : i64
    llvm.cond_br %18 weights([2000, 1]), ^bb2, ^bb1
  ^bb1:  // pred: ^bb0
    %20 = llvm.inttoptr %19 : i64 to !llvm.ptr
    %21 = llvm.getelementptr inbounds %20[%1, 0] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.google::protobuf::internal::InternalMetadata::ContainerBase", (ptr)>
    %22 = llvm.load %21 {alignment = 8 : i64, tbaa = [#tbaa_tag9]} : !llvm.ptr -> !llvm.ptr
    llvm.br ^bb3(%22 : !llvm.ptr)
  ^bb2:  // pred: ^bb0
    %23 = llvm.inttoptr %19 : i64 to !llvm.ptr
    llvm.br ^bb3(%23 : !llvm.ptr)
  ^bb3(%24: !llvm.ptr):  // 2 preds: ^bb1, ^bb2
    %25 = llvm.icmp "eq" %24, %5 : !llvm.ptr
    %26 = llvm.getelementptr inbounds %13[%1, 0] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.google::protobuf::internal::LogFinisher", (i8)>
    llvm.intr.lifetime.start 1, %26 : !llvm.ptr
    llvm.cond_br %25, ^bb7, ^bb4
  ^bb4:  // pred: ^bb3
    %27 = llvm.bitcast %12 : !llvm.ptr to !llvm.ptr
    llvm.intr.lifetime.start 56, %27 : !llvm.ptr
    llvm.invoke @_ZN6google8protobuf8internal10LogMessageC1ENS0_8LogLevelEPKci(%12, %6, %8, %9) to ^bb5 unwind ^bb14 : (!llvm.ptr, i32, !llvm.ptr, i32) -> ()
  ^bb5:  // pred: ^bb4
    %28 = llvm.invoke @_ZN6google8protobuf8internal10LogMessagelsEPKc(%12, %11) to ^bb6 unwind ^bb9 : (!llvm.ptr, !llvm.ptr) -> !llvm.ptr
  ^bb6:  // pred: ^bb5
    llvm.invoke @_ZN6google8protobuf8internal11LogFinisheraSERNS1_10LogMessageE(%13, %28) to ^bb8 unwind ^bb10 : (!llvm.ptr, !llvm.ptr) -> ()
  ^bb7:  // pred: ^bb3
    llvm.intr.lifetime.end 1, %26 : !llvm.ptr
    llvm.br ^bb12
  ^bb8:  // pred: ^bb6
    llvm.intr.lifetime.end 1, %26 : !llvm.ptr
    llvm.call @_ZN6google8protobuf8internal10LogMessageD1Ev(%12) : (!llvm.ptr) -> ()
    llvm.intr.lifetime.end 56, %27 : !llvm.ptr
    llvm.br ^bb12
  ^bb9:  // pred: ^bb5
    %29 = llvm.landingpad (catch %5 : !llvm.ptr) : !llvm.struct<(ptr, i32)>
    llvm.br ^bb11(%29 : !llvm.struct<(ptr, i32)>)
  ^bb10:  // pred: ^bb6
    %30 = llvm.landingpad (catch %5 : !llvm.ptr) : !llvm.struct<(ptr, i32)>
    llvm.intr.lifetime.end 1, %26 : !llvm.ptr
    llvm.br ^bb11(%30 : !llvm.struct<(ptr, i32)>)
  ^bb11(%31: !llvm.struct<(ptr, i32)>):  // 2 preds: ^bb9, ^bb10
    llvm.call @_ZN6google8protobuf8internal10LogMessageD1Ev(%12) : (!llvm.ptr) -> ()
    llvm.intr.lifetime.end 56, %27 : !llvm.ptr
    llvm.br ^bb15(%31 : !llvm.struct<(ptr, i32)>)
  ^bb12:  // 2 preds: ^bb7, ^bb8
    %32 = llvm.getelementptr inbounds %arg0[%1, 0, 0, 1] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.tutorial::AddressBook", packed (struct<"class.google::protobuf::Message", (struct<"class.google::protobuf::MessageLite", (ptr, struct<"class.google::protobuf::internal::InternalMetadata", (ptr)>)>)>, struct<"class.google::protobuf::RepeatedPtrField.18", (struct<"class.google::protobuf::internal::RepeatedPtrFieldBase", (ptr, i32, i32, ptr)>)>, struct<"class.google::protobuf::internal::CachedSize", (struct<"struct.std::atomic", (struct<"struct.std::__atomic_base", (i32)>)>)>, array<4 x i8>)>
    llvm.invoke @_ZN6google8protobuf8internal16InternalMetadata6DeleteINS0_15UnknownFieldSetEEEvv(%32) to ^bb13 unwind ^bb14 : (!llvm.ptr) -> ()
  ^bb13:  // pred: ^bb12
    %33 = llvm.getelementptr inbounds %arg0[%1, 1] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.tutorial::AddressBook", packed (struct<"class.google::protobuf::Message", (struct<"class.google::protobuf::MessageLite", (ptr, struct<"class.google::protobuf::internal::InternalMetadata", (ptr)>)>)>, struct<"class.google::protobuf::RepeatedPtrField.18", (struct<"class.google::protobuf::internal::RepeatedPtrFieldBase", (ptr, i32, i32, ptr)>)>, struct<"class.google::protobuf::internal::CachedSize", (struct<"struct.std::atomic", (struct<"struct.std::__atomic_base", (i32)>)>)>, array<4 x i8>)>
    llvm.call @_ZN6google8protobuf16RepeatedPtrFieldIN8tutorial6PersonEED2Ev(%33) : (!llvm.ptr) -> ()
    llvm.return
  ^bb14:  // 2 preds: ^bb4, ^bb12
    %34 = llvm.landingpad (catch %5 : !llvm.ptr) : !llvm.struct<(ptr, i32)>
    llvm.br ^bb15(%34 : !llvm.struct<(ptr, i32)>)
  ^bb15(%35: !llvm.struct<(ptr, i32)>):  // 2 preds: ^bb11, ^bb14
    %36 = llvm.extractvalue %35[0] : !llvm.struct<(ptr, i32)> 
    %37 = llvm.getelementptr inbounds %arg0[%1, 1] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.tutorial::AddressBook", packed (struct<"class.google::protobuf::Message", (struct<"class.google::protobuf::MessageLite", (ptr, struct<"class.google::protobuf::internal::InternalMetadata", (ptr)>)>)>, struct<"class.google::protobuf::RepeatedPtrField.18", (struct<"class.google::protobuf::internal::RepeatedPtrFieldBase", (ptr, i32, i32, ptr)>)>, struct<"class.google::protobuf::internal::CachedSize", (struct<"struct.std::atomic", (struct<"struct.std::__atomic_base", (i32)>)>)>, array<4 x i8>)>
    llvm.call @_ZN6google8protobuf16RepeatedPtrFieldIN8tutorial6PersonEED2Ev(%37) : (!llvm.ptr) -> ()
    llvm.call @__clang_call_terminate(%36) : (!llvm.ptr) -> ()
    llvm.unreachable
  }
  llvm.func unnamed_addr @_ZN8tutorial11AddressBookD0Ev(%arg0: !llvm.ptr {llvm.align = 8 : i64, llvm.dereferenceable = 48 : i64, llvm.nonnull, llvm.noundef}) attributes {alignment = 2 : i64, frame_pointer = #llvm.framePointerKind<none>, passthrough = ["nounwind", ["uwtable", "2"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], ["tune-cpu", "generic"]], target_cpu = "x86-64", target_features = #llvm.target_features<["+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>} {
    llvm.call @_ZN8tutorial11AddressBookD2Ev(%arg0) : (!llvm.ptr) -> ()
    %0 = llvm.bitcast %arg0 : !llvm.ptr to !llvm.ptr
    llvm.call @_ZdlPv(%0) : (!llvm.ptr) -> ()
    llvm.return
  }
  llvm.func local_unnamed_addr @_ZN8tutorial11AddressBook9ArenaDtorEPv(%arg0: !llvm.ptr {llvm.nocapture, llvm.noundef}) attributes {alignment = 2 : i64, frame_pointer = #llvm.framePointerKind<none>, memory = #llvm.memory_effects<other = none, argMem = none, inaccessibleMem = none>, passthrough = ["mustprogress", "nofree", "norecurse", "nosync", "nounwind", "willreturn", ["uwtable", "2"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], ["tune-cpu", "generic"]], target_cpu = "x86-64", target_features = #llvm.target_features<["+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>} {
    llvm.return
  }
  llvm.func unnamed_addr @_ZNK8tutorial11AddressBook13SetCachedSizeEi(%arg0: !llvm.ptr {llvm.align = 8 : i64, llvm.dereferenceable = 48 : i64, llvm.nocapture, llvm.nonnull, llvm.noundef, llvm.writeonly}, %arg1: i32 {llvm.noundef}) attributes {alignment = 2 : i64, frame_pointer = #llvm.framePointerKind<none>, passthrough = ["mustprogress", "nofree", "norecurse", "nounwind", "willreturn", ["uwtable", "2"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], ["tune-cpu", "generic"]], personality = @__gxx_personality_v0, target_cpu = "x86-64", target_features = #llvm.target_features<["+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>} {
    %0 = llvm.mlir.constant(0 : i64) : i64
    %1 = llvm.mlir.constant(2 : i32) : i32
    %2 = llvm.mlir.constant(0 : i32) : i32
    %3 = llvm.getelementptr inbounds %arg0[%0, 2, 0, 0, 0] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.tutorial::AddressBook", packed (struct<"class.google::protobuf::Message", (struct<"class.google::protobuf::MessageLite", (ptr, struct<"class.google::protobuf::internal::InternalMetadata", (ptr)>)>)>, struct<"class.google::protobuf::RepeatedPtrField.18", (struct<"class.google::protobuf::internal::RepeatedPtrFieldBase", (ptr, i32, i32, ptr)>)>, struct<"class.google::protobuf::internal::CachedSize", (struct<"struct.std::atomic", (struct<"struct.std::__atomic_base", (i32)>)>)>, array<4 x i8>)>
    llvm.store %arg1, %3 atomic monotonic {alignment = 8 : i64} : i32, !llvm.ptr
    llvm.return
  }
  llvm.func local_unnamed_addr @_ZN8tutorial11AddressBook16default_instanceEv() -> (!llvm.ptr {llvm.align = 8 : i64, llvm.dereferenceable = 48 : i64, llvm.nonnull, llvm.noundef}) attributes {alignment = 2 : i64, frame_pointer = #llvm.framePointerKind<none>, passthrough = ["mustprogress", ["uwtable", "2"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], ["tune-cpu", "generic"]], target_cpu = "x86-64", target_features = #llvm.target_features<["+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>} {
    %0 = llvm.mlir.addressof @scc_info_Timestamp_google_2fprotobuf_2ftimestamp_2eproto : !llvm.ptr
    %1 = llvm.mlir.undef : !llvm.array<0 x ptr>
    %2 = llvm.mlir.zero : !llvm.ptr
    %3 = llvm.mlir.addressof @__gxx_personality_v0 : !llvm.ptr
    %4 = llvm.mlir.addressof @_ZL59InitDefaultsscc_info_Person_PhoneNumber_addressbook_2eprotov : !llvm.ptr
    %5 = llvm.mlir.constant(0 : i32) : i32
    %6 = llvm.mlir.constant(-1 : i32) : i32
    %7 = llvm.mlir.undef : !llvm.struct<(i32)>
    %8 = llvm.insertvalue %6, %7[0] : !llvm.struct<(i32)> 
    %9 = llvm.mlir.undef : !llvm.struct<(struct<(i32)>, i32, i32, ptr)>
    %10 = llvm.insertvalue %8, %9[0] : !llvm.struct<(struct<(i32)>, i32, i32, ptr)> 
    %11 = llvm.insertvalue %5, %10[1] : !llvm.struct<(struct<(i32)>, i32, i32, ptr)> 
    %12 = llvm.insertvalue %5, %11[2] : !llvm.struct<(struct<(i32)>, i32, i32, ptr)> 
    %13 = llvm.insertvalue %4, %12[3] : !llvm.struct<(struct<(i32)>, i32, i32, ptr)> 
    %14 = llvm.mlir.undef : !llvm.struct<(struct<(struct<(i32)>, i32, i32, ptr)>, array<0 x ptr>)>
    %15 = llvm.insertvalue %13, %14[0] : !llvm.struct<(struct<(struct<(i32)>, i32, i32, ptr)>, array<0 x ptr>)> 
    %16 = llvm.insertvalue %1, %15[1] : !llvm.struct<(struct<(struct<(i32)>, i32, i32, ptr)>, array<0 x ptr>)> 
    %17 = llvm.mlir.addressof @scc_info_Person_PhoneNumber_addressbook_2eproto : !llvm.ptr
    %18 = llvm.mlir.undef : !llvm.array<2 x ptr>
    %19 = llvm.insertvalue %17, %18[0] : !llvm.array<2 x ptr> 
    %20 = llvm.insertvalue %0, %19[1] : !llvm.array<2 x ptr> 
    %21 = llvm.mlir.addressof @_ZL47InitDefaultsscc_info_Person_addressbook_2eprotov : !llvm.ptr
    %22 = llvm.mlir.constant(2 : i32) : i32
    %23 = llvm.mlir.undef : !llvm.struct<(struct<(i32)>, i32, i32, ptr)>
    %24 = llvm.insertvalue %8, %23[0] : !llvm.struct<(struct<(i32)>, i32, i32, ptr)> 
    %25 = llvm.insertvalue %22, %24[1] : !llvm.struct<(struct<(i32)>, i32, i32, ptr)> 
    %26 = llvm.insertvalue %5, %25[2] : !llvm.struct<(struct<(i32)>, i32, i32, ptr)> 
    %27 = llvm.insertvalue %21, %26[3] : !llvm.struct<(struct<(i32)>, i32, i32, ptr)> 
    %28 = llvm.mlir.undef : !llvm.struct<(struct<(struct<(i32)>, i32, i32, ptr)>, array<2 x ptr>)>
    %29 = llvm.insertvalue %27, %28[0] : !llvm.struct<(struct<(struct<(i32)>, i32, i32, ptr)>, array<2 x ptr>)> 
    %30 = llvm.insertvalue %20, %29[1] : !llvm.struct<(struct<(struct<(i32)>, i32, i32, ptr)>, array<2 x ptr>)> 
    %31 = llvm.mlir.addressof @scc_info_Person_addressbook_2eproto : !llvm.ptr
    %32 = llvm.mlir.undef : !llvm.array<1 x ptr>
    %33 = llvm.insertvalue %31, %32[0] : !llvm.array<1 x ptr> 
    %34 = llvm.mlir.addressof @_ZL52InitDefaultsscc_info_AddressBook_addressbook_2eprotov : !llvm.ptr
    %35 = llvm.mlir.constant(1 : i32) : i32
    %36 = llvm.mlir.undef : !llvm.struct<(struct<(i32)>, i32, i32, ptr)>
    %37 = llvm.insertvalue %8, %36[0] : !llvm.struct<(struct<(i32)>, i32, i32, ptr)> 
    %38 = llvm.insertvalue %35, %37[1] : !llvm.struct<(struct<(i32)>, i32, i32, ptr)> 
    %39 = llvm.insertvalue %5, %38[2] : !llvm.struct<(struct<(i32)>, i32, i32, ptr)> 
    %40 = llvm.insertvalue %34, %39[3] : !llvm.struct<(struct<(i32)>, i32, i32, ptr)> 
    %41 = llvm.mlir.undef : !llvm.struct<(struct<(struct<(i32)>, i32, i32, ptr)>, array<1 x ptr>)>
    %42 = llvm.insertvalue %40, %41[0] : !llvm.struct<(struct<(struct<(i32)>, i32, i32, ptr)>, array<1 x ptr>)> 
    %43 = llvm.insertvalue %33, %42[1] : !llvm.struct<(struct<(struct<(i32)>, i32, i32, ptr)>, array<1 x ptr>)> 
    %44 = llvm.mlir.addressof @scc_info_AddressBook_addressbook_2eproto : !llvm.ptr
    %45 = llvm.mlir.constant(0 : i8) : i8
    %46 = llvm.mlir.constant(dense<0> : tensor<40xi8>) : !llvm.array<40 x i8>
    %47 = llvm.mlir.constant(0 : i64) : i64
    %48 = llvm.mlir.undef : !llvm.struct<"union.google::protobuf::internal::ExplicitlyConstructed<tutorial::AddressBook>::AlignedUnion", (i64, array<40 x i8>)>
    %49 = llvm.insertvalue %47, %48[0] : !llvm.struct<"union.google::protobuf::internal::ExplicitlyConstructed<tutorial::AddressBook>::AlignedUnion", (i64, array<40 x i8>)> 
    %50 = llvm.insertvalue %46, %49[1] : !llvm.struct<"union.google::protobuf::internal::ExplicitlyConstructed<tutorial::AddressBook>::AlignedUnion", (i64, array<40 x i8>)> 
    %51 = llvm.mlir.undef : !llvm.struct<"class.google::protobuf::internal::ExplicitlyConstructed.1", (struct<"union.google::protobuf::internal::ExplicitlyConstructed<tutorial::AddressBook>::AlignedUnion", (i64, array<40 x i8>)>)>
    %52 = llvm.insertvalue %50, %51[0] : !llvm.struct<"class.google::protobuf::internal::ExplicitlyConstructed.1", (struct<"union.google::protobuf::internal::ExplicitlyConstructed<tutorial::AddressBook>::AlignedUnion", (i64, array<40 x i8>)>)> 
    %53 = llvm.mlir.undef : !llvm.struct<"class.tutorial::AddressBookDefaultTypeInternal", (struct<"class.google::protobuf::internal::ExplicitlyConstructed.1", (struct<"union.google::protobuf::internal::ExplicitlyConstructed<tutorial::AddressBook>::AlignedUnion", (i64, array<40 x i8>)>)>)>
    %54 = llvm.insertvalue %52, %53[0] : !llvm.struct<"class.tutorial::AddressBookDefaultTypeInternal", (struct<"class.google::protobuf::internal::ExplicitlyConstructed.1", (struct<"union.google::protobuf::internal::ExplicitlyConstructed<tutorial::AddressBook>::AlignedUnion", (i64, array<40 x i8>)>)>)> 
    %55 = llvm.mlir.addressof @_ZN8tutorial30_AddressBook_default_instance_E : !llvm.ptr
    %56 = llvm.load %44 atomic acquire {alignment = 8 : i64} : !llvm.ptr -> i32
    %57 = llvm.icmp "eq" %56, %5 : i32
    llvm.cond_br %57 weights([2000, 1]), ^bb2, ^bb1
  ^bb1:  // pred: ^bb0
    llvm.call @_ZN6google8protobuf8internal11InitSCCImplEPNS1_11SCCInfoBaseE(%44) : (!llvm.ptr) -> ()
    llvm.br ^bb2
  ^bb2:  // 2 preds: ^bb0, ^bb1
    llvm.return %55 : !llvm.ptr
  }
  llvm.func unnamed_addr @_ZN8tutorial11AddressBook5ClearEv(%arg0: !llvm.ptr {llvm.align = 8 : i64, llvm.dereferenceable = 48 : i64, llvm.nonnull, llvm.noundef}) attributes {alignment = 2 : i64, frame_pointer = #llvm.framePointerKind<none>, passthrough = [["uwtable", "2"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], ["tune-cpu", "generic"]], target_cpu = "x86-64", target_features = #llvm.target_features<["+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>} {
    %0 = llvm.mlir.constant(0 : i64) : i64
    %1 = llvm.mlir.constant(1 : i32) : i32
    %2 = llvm.mlir.constant(0 : i32) : i32
    %3 = llvm.mlir.constant(1 : i64) : i64
    %4 = llvm.mlir.constant(-2 : i64) : i64
    %5 = llvm.getelementptr inbounds %arg0[%0, 1, 0] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.tutorial::AddressBook", packed (struct<"class.google::protobuf::Message", (struct<"class.google::protobuf::MessageLite", (ptr, struct<"class.google::protobuf::internal::InternalMetadata", (ptr)>)>)>, struct<"class.google::protobuf::RepeatedPtrField.18", (struct<"class.google::protobuf::internal::RepeatedPtrFieldBase", (ptr, i32, i32, ptr)>)>, struct<"class.google::protobuf::internal::CachedSize", (struct<"struct.std::atomic", (struct<"struct.std::__atomic_base", (i32)>)>)>, array<4 x i8>)>
    llvm.call @_ZN6google8protobuf8internal20RepeatedPtrFieldBase5ClearINS0_16RepeatedPtrFieldIN8tutorial6PersonEE11TypeHandlerEEEvv(%5) : (!llvm.ptr) -> ()
    %6 = llvm.getelementptr inbounds %arg0[%0, 0, 0, 1, 0] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.tutorial::AddressBook", packed (struct<"class.google::protobuf::Message", (struct<"class.google::protobuf::MessageLite", (ptr, struct<"class.google::protobuf::internal::InternalMetadata", (ptr)>)>)>, struct<"class.google::protobuf::RepeatedPtrField.18", (struct<"class.google::protobuf::internal::RepeatedPtrFieldBase", (ptr, i32, i32, ptr)>)>, struct<"class.google::protobuf::internal::CachedSize", (struct<"struct.std::atomic", (struct<"struct.std::__atomic_base", (i32)>)>)>, array<4 x i8>)>
    %7 = llvm.load %6 {alignment = 8 : i64, tbaa = [#tbaa_tag5]} : !llvm.ptr -> !llvm.ptr
    %8 = llvm.ptrtoint %7 : !llvm.ptr to i64
    %9 = llvm.and %8, %3  : i64
    %10 = llvm.icmp "eq" %9, %0 : i64
    llvm.cond_br %10, ^bb3, ^bb1
  ^bb1:  // pred: ^bb0
    %11 = llvm.and %8, %4  : i64
    %12 = llvm.inttoptr %11 : i64 to !llvm.ptr
    %13 = llvm.getelementptr inbounds %12[%0, 1] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.google::protobuf::internal::InternalMetadata::Container", (struct<"struct.google::protobuf::internal::InternalMetadata::ContainerBase", (ptr)>, struct<"class.google::protobuf::UnknownFieldSet", (struct<"class.std::vector", (struct<"struct.std::_Vector_base", (struct<"struct.std::_Vector_base<google::protobuf::UnknownField, std::allocator<google::protobuf::UnknownField>>::_Vector_impl", (struct<"struct.std::_Vector_base<google::protobuf::UnknownField, std::allocator<google::protobuf::UnknownField>>::_Vector_impl_data", (ptr, ptr, ptr)>)>)>)>)>)>
    %14 = llvm.getelementptr inbounds %13[%0, 0, 0, 0, 0, 0] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.google::protobuf::UnknownFieldSet", (struct<"class.std::vector", (struct<"struct.std::_Vector_base", (struct<"struct.std::_Vector_base<google::protobuf::UnknownField, std::allocator<google::protobuf::UnknownField>>::_Vector_impl", (struct<"struct.std::_Vector_base<google::protobuf::UnknownField, std::allocator<google::protobuf::UnknownField>>::_Vector_impl_data", (ptr, ptr, ptr)>)>)>)>)>
    %15 = llvm.load %14 {alignment = 8 : i64, tbaa = [#tbaa_tag2]} : !llvm.ptr -> !llvm.ptr
    %16 = llvm.getelementptr inbounds %12[%0, 1, 0, 0, 0, 0, 1] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.google::protobuf::internal::InternalMetadata::Container", (struct<"struct.google::protobuf::internal::InternalMetadata::ContainerBase", (ptr)>, struct<"class.google::protobuf::UnknownFieldSet", (struct<"class.std::vector", (struct<"struct.std::_Vector_base", (struct<"struct.std::_Vector_base<google::protobuf::UnknownField, std::allocator<google::protobuf::UnknownField>>::_Vector_impl", (struct<"struct.std::_Vector_base<google::protobuf::UnknownField, std::allocator<google::protobuf::UnknownField>>::_Vector_impl_data", (ptr, ptr, ptr)>)>)>)>)>)>
    %17 = llvm.load %16 {alignment = 8 : i64, tbaa = [#tbaa_tag2]} : !llvm.ptr -> !llvm.ptr
    %18 = llvm.icmp "eq" %15, %17 : !llvm.ptr
    llvm.cond_br %18, ^bb3, ^bb2
  ^bb2:  // pred: ^bb1
    llvm.call @_ZN6google8protobuf15UnknownFieldSet13ClearFallbackEv(%13) : (!llvm.ptr) -> ()
    llvm.br ^bb3
  ^bb3:  // 3 preds: ^bb0, ^bb1, ^bb2
    llvm.return
  }
  llvm.func unnamed_addr @_ZN8tutorial11AddressBook14_InternalParseEPKcPN6google8protobuf8internal12ParseContextE(%arg0: !llvm.ptr {llvm.align = 8 : i64, llvm.dereferenceable = 48 : i64, llvm.nonnull, llvm.noundef}, %arg1: !llvm.ptr {llvm.noundef}, %arg2: !llvm.ptr {llvm.noundef}) -> (!llvm.ptr {llvm.noundef}) attributes {alignment = 2 : i64, frame_pointer = #llvm.framePointerKind<none>, passthrough = ["mustprogress", ["uwtable", "2"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], ["tune-cpu", "generic"]], target_cpu = "x86-64", target_features = #llvm.target_features<["+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>} {
    %0 = llvm.mlir.constant(1 : i32) : i32
    %1 = llvm.mlir.constant(0 : i64) : i64
    %2 = llvm.mlir.constant(0 : i32) : i32
    %3 = llvm.mlir.constant(2 : i32) : i32
    %4 = llvm.mlir.constant(3 : i32) : i32
    %5 = llvm.mlir.constant(8 : i32) : i32
    %6 = llvm.mlir.constant(4 : i32) : i32
    %7 = llvm.mlir.constant(-1 : i8) : i8
    %8 = llvm.mlir.constant(1 : i64) : i64
    %9 = llvm.mlir.constant(7 : i32) : i32
    %10 = llvm.mlir.constant(-128 : i32) : i32
    %11 = llvm.mlir.zero : !llvm.ptr
    %12 = llvm.mlir.constant(2 : i64) : i64
    %13 = llvm.mlir.constant(10 : i32) : i32
    %14 = llvm.mlir.constant(-2 : i64) : i64
    %15 = llvm.mlir.constant(-1 : i32) : i32
    %16 = llvm.mlir.constant(-1 : i64) : i64
    %17 = llvm.mlir.constant(10 : i8) : i8
    %18 = llvm.alloca %0 x !llvm.ptr {alignment = 8 : i64} : (i32) -> !llvm.ptr
    llvm.store %arg1, %18 {alignment = 8 : i64, tbaa = [#tbaa_tag2]} : !llvm.ptr, !llvm.ptr
    %19 = llvm.getelementptr inbounds %arg2[%1, 0] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.google::protobuf::internal::ParseContext", (struct<"class.google::protobuf::internal::EpsCopyInputStream", (ptr, ptr, ptr, i32, i32, ptr, array<32 x i8>, i64, i32, i32)>, i32, i32, struct<"struct.google::protobuf::internal::ParseContext::Data", (ptr, ptr)>)>
    %20 = llvm.getelementptr inbounds %arg2[%1, 2] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.google::protobuf::internal::ParseContext", (struct<"class.google::protobuf::internal::EpsCopyInputStream", (ptr, ptr, ptr, i32, i32, ptr, array<32 x i8>, i64, i32, i32)>, i32, i32, struct<"struct.google::protobuf::internal::ParseContext::Data", (ptr, ptr)>)>
    %21 = llvm.load %20 {alignment = 4 : i64, tbaa = [#tbaa_tag27]} : !llvm.ptr -> i32
    %22 = llvm.call @_ZN6google8protobuf8internal18EpsCopyInputStream13DoneWithCheckEPPKci(%19, %18, %21) : (!llvm.ptr, !llvm.ptr, i32) -> i1
    llvm.cond_br %22, ^bb32, ^bb1
  ^bb1:  // pred: ^bb0
    %23 = llvm.getelementptr inbounds %arg0[%1, 0, 0, 1] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.tutorial::AddressBook", packed (struct<"class.google::protobuf::Message", (struct<"class.google::protobuf::MessageLite", (ptr, struct<"class.google::protobuf::internal::InternalMetadata", (ptr)>)>)>, struct<"class.google::protobuf::RepeatedPtrField.18", (struct<"class.google::protobuf::internal::RepeatedPtrFieldBase", (ptr, i32, i32, ptr)>)>, struct<"class.google::protobuf::internal::CachedSize", (struct<"struct.std::atomic", (struct<"struct.std::__atomic_base", (i32)>)>)>, array<4 x i8>)>
    %24 = llvm.getelementptr inbounds %23[%1, 0] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.google::protobuf::internal::InternalMetadata", (ptr)>
    %25 = llvm.getelementptr inbounds %arg0[%1, 1] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.tutorial::AddressBook", packed (struct<"class.google::protobuf::Message", (struct<"class.google::protobuf::MessageLite", (ptr, struct<"class.google::protobuf::internal::InternalMetadata", (ptr)>)>)>, struct<"class.google::protobuf::RepeatedPtrField.18", (struct<"class.google::protobuf::internal::RepeatedPtrFieldBase", (ptr, i32, i32, ptr)>)>, struct<"class.google::protobuf::internal::CachedSize", (struct<"struct.std::atomic", (struct<"struct.std::__atomic_base", (i32)>)>)>, array<4 x i8>)>
    %26 = llvm.getelementptr inbounds %25[%1, 0] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.google::protobuf::RepeatedPtrField.18", (struct<"class.google::protobuf::internal::RepeatedPtrFieldBase", (ptr, i32, i32, ptr)>)>
    %27 = llvm.getelementptr inbounds %arg0[%1, 1, 0, 3] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.tutorial::AddressBook", packed (struct<"class.google::protobuf::Message", (struct<"class.google::protobuf::MessageLite", (ptr, struct<"class.google::protobuf::internal::InternalMetadata", (ptr)>)>)>, struct<"class.google::protobuf::RepeatedPtrField.18", (struct<"class.google::protobuf::internal::RepeatedPtrFieldBase", (ptr, i32, i32, ptr)>)>, struct<"class.google::protobuf::internal::CachedSize", (struct<"struct.std::atomic", (struct<"struct.std::__atomic_base", (i32)>)>)>, array<4 x i8>)>
    %28 = llvm.getelementptr inbounds %arg0[%1, 1, 0, 1] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.tutorial::AddressBook", packed (struct<"class.google::protobuf::Message", (struct<"class.google::protobuf::MessageLite", (ptr, struct<"class.google::protobuf::internal::InternalMetadata", (ptr)>)>)>, struct<"class.google::protobuf::RepeatedPtrField.18", (struct<"class.google::protobuf::internal::RepeatedPtrFieldBase", (ptr, i32, i32, ptr)>)>, struct<"class.google::protobuf::internal::CachedSize", (struct<"struct.std::atomic", (struct<"struct.std::__atomic_base", (i32)>)>)>, array<4 x i8>)>
    %29 = llvm.getelementptr inbounds %arg0[%1, 1, 0, 2] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.tutorial::AddressBook", packed (struct<"class.google::protobuf::Message", (struct<"class.google::protobuf::MessageLite", (ptr, struct<"class.google::protobuf::internal::InternalMetadata", (ptr)>)>)>, struct<"class.google::protobuf::RepeatedPtrField.18", (struct<"class.google::protobuf::internal::RepeatedPtrFieldBase", (ptr, i32, i32, ptr)>)>, struct<"class.google::protobuf::internal::CachedSize", (struct<"struct.std::atomic", (struct<"struct.std::__atomic_base", (i32)>)>)>, array<4 x i8>)>
    %30 = llvm.getelementptr inbounds %25[%1, 0, 0] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.google::protobuf::RepeatedPtrField.18", (struct<"class.google::protobuf::internal::RepeatedPtrFieldBase", (ptr, i32, i32, ptr)>)>
    %31 = llvm.getelementptr inbounds %arg2[%1, 1] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.google::protobuf::internal::ParseContext", (struct<"class.google::protobuf::internal::EpsCopyInputStream", (ptr, ptr, ptr, i32, i32, ptr, array<32 x i8>, i64, i32, i32)>, i32, i32, struct<"struct.google::protobuf::internal::ParseContext::Data", (ptr, ptr)>)>
    %32 = llvm.getelementptr inbounds %arg2[%1, 0, 8] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.google::protobuf::internal::ParseContext", (struct<"class.google::protobuf::internal::EpsCopyInputStream", (ptr, ptr, ptr, i32, i32, ptr, array<32 x i8>, i64, i32, i32)>, i32, i32, struct<"struct.google::protobuf::internal::ParseContext::Data", (ptr, ptr)>)>
    %33 = llvm.getelementptr inbounds %arg2[%1, 0, 4] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.google::protobuf::internal::ParseContext", (struct<"class.google::protobuf::internal::EpsCopyInputStream", (ptr, ptr, ptr, i32, i32, ptr, array<32 x i8>, i64, i32, i32)>, i32, i32, struct<"struct.google::protobuf::internal::ParseContext::Data", (ptr, ptr)>)>
    %34 = llvm.getelementptr inbounds %arg2[%1, 0, 1] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.google::protobuf::internal::ParseContext", (struct<"class.google::protobuf::internal::EpsCopyInputStream", (ptr, ptr, ptr, i32, i32, ptr, array<32 x i8>, i64, i32, i32)>, i32, i32, struct<"struct.google::protobuf::internal::ParseContext::Data", (ptr, ptr)>)>
    %35 = llvm.getelementptr inbounds %arg2[%1, 0, 0] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.google::protobuf::internal::ParseContext", (struct<"class.google::protobuf::internal::EpsCopyInputStream", (ptr, ptr, ptr, i32, i32, ptr, array<32 x i8>, i64, i32, i32)>, i32, i32, struct<"struct.google::protobuf::internal::ParseContext::Data", (ptr, ptr)>)>
    llvm.br ^bb2
  ^bb2:  // 2 preds: ^bb1, ^bb30
    %36 = llvm.load %18 {alignment = 8 : i64, tbaa = [#tbaa_tag2]} : !llvm.ptr -> !llvm.ptr
    %37 = llvm.load %36 {alignment = 1 : i64, tbaa = [#tbaa_tag1]} : !llvm.ptr -> i8
    %38 = llvm.zext %37 : i8 to i32
    %39 = llvm.icmp "sgt" %37, %7 : i8
    %40 = llvm.getelementptr inbounds %36[%8] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    llvm.cond_br %39, ^bb5(%38, %40 : i32, !llvm.ptr), ^bb3
  ^bb3:  // pred: ^bb2
    %41 = llvm.load %40 {alignment = 1 : i64, tbaa = [#tbaa_tag1]} : !llvm.ptr -> i8
    %42 = llvm.zext %41 : i8 to i32
    %43 = llvm.shl %42, %9 overflow<nsw, nuw>  : i32
    %44 = llvm.add %38, %10 overflow<nsw>  : i32
    %45 = llvm.add %44, %43 overflow<nsw>  : i32
    %46 = llvm.icmp "sgt" %41, %7 : i8
    llvm.cond_br %46, ^bb4, ^bb6
  ^bb4:  // pred: ^bb3
    %47 = llvm.getelementptr inbounds %36[%12] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    llvm.br ^bb5(%45, %47 : i32, !llvm.ptr)
  ^bb5(%48: i32, %49: !llvm.ptr):  // 2 preds: ^bb2, ^bb4
    llvm.store %49, %18 {alignment = 8 : i64, tbaa = [#tbaa_tag2]} : !llvm.ptr, !llvm.ptr
    llvm.br ^bb7(%49, %48 : !llvm.ptr, i32)
  ^bb6:  // pred: ^bb3
    %50 = llvm.call @_ZN6google8protobuf8internal15ReadTagFallbackEPKcj(%36, %45) : (!llvm.ptr, i32) -> !llvm.struct<(ptr, i32)>
    %51 = llvm.extractvalue %50[0] : !llvm.struct<(ptr, i32)> 
    %52 = llvm.extractvalue %50[1] : !llvm.struct<(ptr, i32)> 
    llvm.store %51, %18 {alignment = 8 : i64, tbaa = [#tbaa_tag2]} : !llvm.ptr, !llvm.ptr
    %53 = llvm.icmp "eq" %51, %11 : !llvm.ptr
    llvm.cond_br %53 weights([1, 2000]), ^bb31, ^bb7(%51, %52 : !llvm.ptr, i32)
  ^bb7(%54: !llvm.ptr, %55: i32):  // 2 preds: ^bb5, ^bb6
    %56 = llvm.icmp "eq" %55, %13 : i32
    llvm.cond_br %56 weights([2000, 2002]), ^bb8, ^bb24
  ^bb8:  // pred: ^bb7
    %57 = llvm.getelementptr inbounds %54[%16] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    llvm.br ^bb9(%57 : !llvm.ptr)
  ^bb9(%58: !llvm.ptr):  // 2 preds: ^bb8, ^bb23
    %59 = llvm.getelementptr inbounds %58[%8] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    llvm.store %59, %18 {alignment = 8 : i64, tbaa = [#tbaa_tag2]} : !llvm.ptr, !llvm.ptr
    %60 = llvm.load %27 {alignment = 8 : i64, tbaa = [#tbaa_tag15]} : !llvm.ptr -> !llvm.ptr
    %61 = llvm.icmp "eq" %60, %11 : !llvm.ptr
    llvm.cond_br %61, ^bb10, ^bb11
  ^bb10:  // pred: ^bb9
    %62 = llvm.load %29 {alignment = 4 : i64, tbaa = [#tbaa_tag16]} : !llvm.ptr -> i32
    llvm.br ^bb14(%62 : i32)
  ^bb11:  // pred: ^bb9
    %63 = llvm.load %28 {alignment = 8 : i64, tbaa = [#tbaa_tag17]} : !llvm.ptr -> i32
    %64 = llvm.getelementptr inbounds %60[%1, 0] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.google::protobuf::internal::RepeatedPtrFieldBase::Rep", (i32, array<1 x ptr>)>
    %65 = llvm.load %64 {alignment = 8 : i64, tbaa = [#tbaa_tag18]} : !llvm.ptr -> i32
    %66 = llvm.icmp "slt" %63, %65 : i32
    llvm.cond_br %66, ^bb12, ^bb13
  ^bb12:  // pred: ^bb11
    %67 = llvm.add %63, %0 overflow<nsw>  : i32
    llvm.store %67, %28 {alignment = 8 : i64, tbaa = [#tbaa_tag17]} : i32, !llvm.ptr
    %68 = llvm.sext %63 : i32 to i64
    %69 = llvm.getelementptr inbounds %60[%1, 1, %68] : (!llvm.ptr, i64, i64) -> !llvm.ptr, !llvm.struct<"struct.google::protobuf::internal::RepeatedPtrFieldBase::Rep", (i32, array<1 x ptr>)>
    %70 = llvm.bitcast %69 : !llvm.ptr to !llvm.ptr
    %71 = llvm.load %70 {alignment = 8 : i64, tbaa = [#tbaa_tag2]} : !llvm.ptr -> !llvm.ptr
    llvm.br ^bb16(%59, %71 : !llvm.ptr, !llvm.ptr)
  ^bb13:  // pred: ^bb11
    %72 = llvm.load %29 {alignment = 4 : i64, tbaa = [#tbaa_tag16]} : !llvm.ptr -> i32
    %73 = llvm.icmp "eq" %65, %72 : i32
    llvm.cond_br %73, ^bb14(%65 : i32), ^bb15(%65, %60 : i32, !llvm.ptr)
  ^bb14(%74: i32):  // 2 preds: ^bb10, ^bb13
    %75 = llvm.add %74, %0 overflow<nsw>  : i32
    llvm.call @_ZN6google8protobuf8internal20RepeatedPtrFieldBase7ReserveEi(%26, %75) : (!llvm.ptr, i32) -> ()
    %76 = llvm.load %27 {alignment = 8 : i64, tbaa = [#tbaa_tag15]} : !llvm.ptr -> !llvm.ptr
    %77 = llvm.getelementptr inbounds %76[%1, 0] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.google::protobuf::internal::RepeatedPtrFieldBase::Rep", (i32, array<1 x ptr>)>
    %78 = llvm.load %77 {alignment = 8 : i64, tbaa = [#tbaa_tag18]} : !llvm.ptr -> i32
    llvm.br ^bb15(%78, %76 : i32, !llvm.ptr)
  ^bb15(%79: i32, %80: !llvm.ptr):  // 2 preds: ^bb13, ^bb14
    %81 = llvm.getelementptr inbounds %80[%1, 0] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.google::protobuf::internal::RepeatedPtrFieldBase::Rep", (i32, array<1 x ptr>)>
    %82 = llvm.add %79, %0 overflow<nsw>  : i32
    llvm.store %82, %81 {alignment = 8 : i64, tbaa = [#tbaa_tag18]} : i32, !llvm.ptr
    %83 = llvm.load %30 {alignment = 8 : i64, tbaa = [#tbaa_tag7]} : !llvm.ptr -> !llvm.ptr
    %84 = llvm.call @_ZN6google8protobuf5Arena18CreateMaybeMessageIN8tutorial6PersonEJEEEPT_PS1_DpOT0_(%83) : (!llvm.ptr) -> !llvm.ptr
    %85 = llvm.load %27 {alignment = 8 : i64, tbaa = [#tbaa_tag15]} : !llvm.ptr -> !llvm.ptr
    %86 = llvm.load %28 {alignment = 8 : i64, tbaa = [#tbaa_tag17]} : !llvm.ptr -> i32
    %87 = llvm.add %86, %0 overflow<nsw>  : i32
    llvm.store %87, %28 {alignment = 8 : i64, tbaa = [#tbaa_tag17]} : i32, !llvm.ptr
    %88 = llvm.sext %86 : i32 to i64
    %89 = llvm.getelementptr inbounds %85[%1, 1, %88] : (!llvm.ptr, i64, i64) -> !llvm.ptr, !llvm.struct<"struct.google::protobuf::internal::RepeatedPtrFieldBase::Rep", (i32, array<1 x ptr>)>
    %90 = llvm.bitcast %89 : !llvm.ptr to !llvm.ptr
    llvm.store %84, %90 {alignment = 8 : i64, tbaa = [#tbaa_tag2]} : !llvm.ptr, !llvm.ptr
    %91 = llvm.load %18 {alignment = 8 : i64, tbaa = [#tbaa_tag2]} : !llvm.ptr -> !llvm.ptr
    llvm.br ^bb16(%91, %84 : !llvm.ptr, !llvm.ptr)
  ^bb16(%92: !llvm.ptr, %93: !llvm.ptr):  // 2 preds: ^bb12, ^bb15
    %94 = llvm.load %92 {alignment = 1 : i64, tbaa = [#tbaa_tag1]} : !llvm.ptr -> i8
    %95 = llvm.zext %94 : i8 to i32
    %96 = llvm.icmp "sgt" %94, %7 : i8
    llvm.cond_br %96, ^bb17, ^bb18
  ^bb17:  // pred: ^bb16
    %97 = llvm.getelementptr inbounds %92[%8] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    llvm.br ^bb19(%95, %97 : i32, !llvm.ptr)
  ^bb18:  // pred: ^bb16
    %98 = llvm.call @_ZN6google8protobuf8internal16ReadSizeFallbackEPKcj(%92, %95) : (!llvm.ptr, i32) -> !llvm.struct<(ptr, i32)>
    %99 = llvm.extractvalue %98[0] : !llvm.struct<(ptr, i32)> 
    %100 = llvm.extractvalue %98[1] : !llvm.struct<(ptr, i32)> 
    %101 = llvm.icmp "eq" %99, %11 : !llvm.ptr
    llvm.cond_br %101, ^bb31, ^bb19(%100, %99 : i32, !llvm.ptr)
  ^bb19(%102: i32, %103: !llvm.ptr):  // 2 preds: ^bb17, ^bb18
    %104 = llvm.call @_ZN6google8protobuf8internal18EpsCopyInputStream9PushLimitEPKci(%19, %103, %102) : (!llvm.ptr, !llvm.ptr, i32) -> i32
    %105 = llvm.load %31 {alignment = 8 : i64, tbaa = [#tbaa_tag29]} : !llvm.ptr -> i32
    %106 = llvm.add %105, %15 overflow<nsw>  : i32
    llvm.store %106, %31 {alignment = 8 : i64, tbaa = [#tbaa_tag29]} : i32, !llvm.ptr
    %107 = llvm.icmp "slt" %105, %0 : i32
    llvm.cond_br %107, ^bb31, ^bb20
  ^bb20:  // pred: ^bb19
    %108 = llvm.call @_ZN8tutorial6Person14_InternalParseEPKcPN6google8protobuf8internal12ParseContextE(%93, %103, %arg2) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> !llvm.ptr
    %109 = llvm.icmp "eq" %108, %11 : !llvm.ptr
    llvm.cond_br %109 weights([1, 2000]), ^bb31, ^bb21
  ^bb21:  // pred: ^bb20
    %110 = llvm.load %31 {alignment = 8 : i64, tbaa = [#tbaa_tag29]} : !llvm.ptr -> i32
    %111 = llvm.add %110, %0 overflow<nsw>  : i32
    llvm.store %111, %31 {alignment = 8 : i64, tbaa = [#tbaa_tag29]} : i32, !llvm.ptr
    %112 = llvm.load %32 {alignment = 8 : i64, tbaa = [#tbaa_tag11]} : !llvm.ptr -> i32
    %113 = llvm.icmp "eq" %112, %2 : i32
    llvm.cond_br %113 weights([2000, 1]), ^bb22, ^bb31
  ^bb22:  // pred: ^bb21
    %114 = llvm.load %33 {alignment = 4 : i64, tbaa = [#tbaa_tag19]} : !llvm.ptr -> i32
    %115 = llvm.add %114, %104 overflow<nsw>  : i32
    llvm.store %115, %33 {alignment = 4 : i64, tbaa = [#tbaa_tag19]} : i32, !llvm.ptr
    %116 = llvm.load %34 {alignment = 8 : i64, tbaa = [#tbaa_tag20]} : !llvm.ptr -> !llvm.ptr
    %117 = llvm.icmp "slt" %115, %2 : i32
    %118 = llvm.select %117, %115, %2 : i1, i32
    %119 = llvm.sext %118 : i32 to i64
    %120 = llvm.getelementptr inbounds %116[%119] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    llvm.store %120, %35 {alignment = 8 : i64, tbaa = [#tbaa_tag21]} : !llvm.ptr, !llvm.ptr
    llvm.store %108, %18 {alignment = 8 : i64, tbaa = [#tbaa_tag2]} : !llvm.ptr, !llvm.ptr
    %121 = llvm.icmp "ugt" %120, %108 : !llvm.ptr
    llvm.cond_br %121, ^bb23, ^bb30 {loop_annotation = #loop_annotation}
  ^bb23:  // pred: ^bb22
    %122 = llvm.load %108 {alignment = 1 : i64, tbaa = [#tbaa_tag1]} : !llvm.ptr -> i8
    %123 = llvm.icmp "eq" %122, %17 : i8
    llvm.cond_br %123, ^bb9(%108 : !llvm.ptr), ^bb30 {loop_annotation = #loop_annotation}
  ^bb24:  // pred: ^bb7
    %124 = llvm.and %55, %9  : i32
    %125 = llvm.icmp "eq" %124, %6 : i32
    %126 = llvm.icmp "eq" %55, %2 : i32
    %127 = llvm.or %126, %125  : i1
    llvm.cond_br %127, ^bb25, ^bb26
  ^bb25:  // pred: ^bb24
    %128 = llvm.add %55, %15  : i32
    llvm.store %128, %32 {alignment = 8 : i64, tbaa = [#tbaa_tag11]} : i32, !llvm.ptr
    llvm.br ^bb32
  ^bb26:  // pred: ^bb24
    %129 = llvm.zext %55 : i32 to i64
    %130 = llvm.load %24 {alignment = 8 : i64, tbaa = [#tbaa_tag5]} : !llvm.ptr -> !llvm.ptr
    %131 = llvm.ptrtoint %130 : !llvm.ptr to i64
    %132 = llvm.and %131, %8  : i64
    %133 = llvm.icmp "eq" %132, %1 : i64
    llvm.cond_br %133 weights([1, 2000]), ^bb28, ^bb27
  ^bb27:  // pred: ^bb26
    %134 = llvm.and %131, %14  : i64
    %135 = llvm.inttoptr %134 : i64 to !llvm.ptr
    %136 = llvm.getelementptr inbounds %135[%1, 1] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.google::protobuf::internal::InternalMetadata::Container", (struct<"struct.google::protobuf::internal::InternalMetadata::ContainerBase", (ptr)>, struct<"class.google::protobuf::UnknownFieldSet", (struct<"class.std::vector", (struct<"struct.std::_Vector_base", (struct<"struct.std::_Vector_base<google::protobuf::UnknownField, std::allocator<google::protobuf::UnknownField>>::_Vector_impl", (struct<"struct.std::_Vector_base<google::protobuf::UnknownField, std::allocator<google::protobuf::UnknownField>>::_Vector_impl_data", (ptr, ptr, ptr)>)>)>)>)>)>
    llvm.br ^bb29(%54, %136 : !llvm.ptr, !llvm.ptr)
  ^bb28:  // pred: ^bb26
    %137 = llvm.call @_ZN6google8protobuf8internal16InternalMetadata27mutable_unknown_fields_slowINS0_15UnknownFieldSetEEEPT_v(%23) : (!llvm.ptr) -> !llvm.ptr
    %138 = llvm.load %18 {alignment = 8 : i64, tbaa = [#tbaa_tag2]} : !llvm.ptr -> !llvm.ptr
    llvm.br ^bb29(%138, %137 : !llvm.ptr, !llvm.ptr)
  ^bb29(%139: !llvm.ptr, %140: !llvm.ptr):  // 2 preds: ^bb27, ^bb28
    %141 = llvm.call @_ZN6google8protobuf8internal17UnknownFieldParseEmPNS0_15UnknownFieldSetEPKcPNS1_12ParseContextE(%129, %140, %139, %arg2) : (i64, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> !llvm.ptr
    llvm.store %141, %18 {alignment = 8 : i64, tbaa = [#tbaa_tag2]} : !llvm.ptr, !llvm.ptr
    %142 = llvm.icmp "eq" %141, %11 : !llvm.ptr
    llvm.cond_br %142, ^bb31, ^bb30
  ^bb30:  // 3 preds: ^bb22, ^bb23, ^bb29
    %143 = llvm.load %20 {alignment = 4 : i64, tbaa = [#tbaa_tag27]} : !llvm.ptr -> i32
    %144 = llvm.call @_ZN6google8protobuf8internal18EpsCopyInputStream13DoneWithCheckEPPKci(%19, %18, %143) : (!llvm.ptr, !llvm.ptr, i32) -> i1
    llvm.cond_br %144, ^bb32, ^bb2
  ^bb31:  // 6 preds: ^bb6, ^bb18, ^bb19, ^bb20, ^bb21, ^bb29
    llvm.store %11, %18 {alignment = 8 : i64, tbaa = [#tbaa_tag2]} : !llvm.ptr, !llvm.ptr
    llvm.br ^bb32
  ^bb32:  // 4 preds: ^bb0, ^bb25, ^bb30, ^bb31
    %145 = llvm.load %18 {alignment = 8 : i64, tbaa = [#tbaa_tag2]} : !llvm.ptr -> !llvm.ptr
    llvm.return %145 : !llvm.ptr
  }
  llvm.func unnamed_addr @_ZNK8tutorial11AddressBook18_InternalSerializeEPhPN6google8protobuf2io19EpsCopyOutputStreamE(%arg0: !llvm.ptr {llvm.align = 8 : i64, llvm.dereferenceable = 48 : i64, llvm.nonnull, llvm.noundef}, %arg1: !llvm.ptr {llvm.noundef}, %arg2: !llvm.ptr {llvm.noundef}) -> (!llvm.ptr {llvm.noundef}) attributes {alignment = 2 : i64, frame_pointer = #llvm.framePointerKind<none>, passthrough = ["mustprogress", ["uwtable", "2"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], ["tune-cpu", "generic"]], target_cpu = "x86-64", target_features = #llvm.target_features<["+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>} {
    %0 = llvm.mlir.constant(0 : i64) : i64
    %1 = llvm.mlir.constant(1 : i32) : i32
    %2 = llvm.mlir.constant(0 : i32) : i32
    %3 = llvm.mlir.constant(10 : i8) : i8
    %4 = llvm.mlir.constant(1 : i64) : i64
    %5 = llvm.mlir.constant(6 : i32) : i32
    %6 = llvm.mlir.constant(128 : i32) : i32
    %7 = llvm.mlir.constant(-128 : i8) : i8
    %8 = llvm.mlir.constant(7 : i32) : i32
    %9 = llvm.mlir.constant(16384 : i32) : i32
    %10 = llvm.mlir.constant(2 : i64) : i64
    %11 = llvm.mlir.constant(16383 : i32) : i32
    %12 = llvm.mlir.constant(3 : i64) : i64
    %13 = llvm.mlir.constant(-2 : i64) : i64
    %14 = llvm.getelementptr inbounds %arg0[%0, 1, 0, 1] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.tutorial::AddressBook", packed (struct<"class.google::protobuf::Message", (struct<"class.google::protobuf::MessageLite", (ptr, struct<"class.google::protobuf::internal::InternalMetadata", (ptr)>)>)>, struct<"class.google::protobuf::RepeatedPtrField.18", (struct<"class.google::protobuf::internal::RepeatedPtrFieldBase", (ptr, i32, i32, ptr)>)>, struct<"class.google::protobuf::internal::CachedSize", (struct<"struct.std::atomic", (struct<"struct.std::__atomic_base", (i32)>)>)>, array<4 x i8>)>
    %15 = llvm.load %14 {alignment = 8 : i64, tbaa = [#tbaa_tag17]} : !llvm.ptr -> i32
    %16 = llvm.icmp "eq" %15, %2 : i32
    llvm.cond_br %16, ^bb2(%arg1 : !llvm.ptr), ^bb1
  ^bb1:  // pred: ^bb0
    %17 = llvm.getelementptr inbounds %arg2[%0, 0] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.google::protobuf::io::EpsCopyOutputStream", packed (ptr, ptr, array<32 x i8>, ptr, i8, i8, i8, array<5 x i8>)>
    %18 = llvm.getelementptr inbounds %arg0[%0, 1, 0] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.tutorial::AddressBook", packed (struct<"class.google::protobuf::Message", (struct<"class.google::protobuf::MessageLite", (ptr, struct<"class.google::protobuf::internal::InternalMetadata", (ptr)>)>)>, struct<"class.google::protobuf::RepeatedPtrField.18", (struct<"class.google::protobuf::internal::RepeatedPtrFieldBase", (ptr, i32, i32, ptr)>)>, struct<"class.google::protobuf::internal::CachedSize", (struct<"struct.std::atomic", (struct<"struct.std::__atomic_base", (i32)>)>)>, array<4 x i8>)>
    llvm.br ^bb3(%arg1, %2 : !llvm.ptr, i32)
  ^bb2(%19: !llvm.ptr):  // 2 preds: ^bb0, ^bb12
    %20 = llvm.getelementptr inbounds %arg0[%0, 0, 0, 1, 0] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.tutorial::AddressBook", packed (struct<"class.google::protobuf::Message", (struct<"class.google::protobuf::MessageLite", (ptr, struct<"class.google::protobuf::internal::InternalMetadata", (ptr)>)>)>, struct<"class.google::protobuf::RepeatedPtrField.18", (struct<"class.google::protobuf::internal::RepeatedPtrFieldBase", (ptr, i32, i32, ptr)>)>, struct<"class.google::protobuf::internal::CachedSize", (struct<"struct.std::atomic", (struct<"struct.std::__atomic_base", (i32)>)>)>, array<4 x i8>)>
    %21 = llvm.load %20 {alignment = 8 : i64, tbaa = [#tbaa_tag5]} : !llvm.ptr -> !llvm.ptr
    %22 = llvm.ptrtoint %21 : !llvm.ptr to i64
    %23 = llvm.and %22, %4  : i64
    %24 = llvm.icmp "eq" %23, %0 : i64
    llvm.cond_br %24 weights([2000, 1]), ^bb14(%19 : !llvm.ptr), ^bb13
  ^bb3(%25: !llvm.ptr, %26: i32):  // 2 preds: ^bb1, ^bb12
    %27 = llvm.load %17 {alignment = 8 : i64, tbaa = [#tbaa_tag12]} : !llvm.ptr -> !llvm.ptr
    %28 = llvm.icmp "ugt" %27, %25 : !llvm.ptr
    llvm.cond_br %28 weights([2000, 1]), ^bb5(%25 : !llvm.ptr), ^bb4
  ^bb4:  // pred: ^bb3
    %29 = llvm.call @_ZN6google8protobuf2io19EpsCopyOutputStream19EnsureSpaceFallbackEPh(%arg2, %25) : (!llvm.ptr, !llvm.ptr) -> !llvm.ptr
    llvm.br ^bb5(%29 : !llvm.ptr)
  ^bb5(%30: !llvm.ptr):  // 2 preds: ^bb3, ^bb4
    %31 = llvm.call @_ZNK6google8protobuf8internal20RepeatedPtrFieldBase3GetINS0_16RepeatedPtrFieldIN8tutorial6PersonEE11TypeHandlerEEERKNT_4TypeEi(%18, %26) : (!llvm.ptr, i32) -> !llvm.ptr
    llvm.store %3, %30 {alignment = 1 : i64, tbaa = [#tbaa_tag1]} : i8, !llvm.ptr
    %32 = llvm.getelementptr inbounds %30[%4] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %33 = llvm.getelementptr inbounds %31[%0, 6, 0, 0, 0] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.tutorial::Person", (struct<"class.google::protobuf::Message", (struct<"class.google::protobuf::MessageLite", (ptr, struct<"class.google::protobuf::internal::InternalMetadata", (ptr)>)>)>, struct<"class.google::protobuf::RepeatedPtrField", (struct<"class.google::protobuf::internal::RepeatedPtrFieldBase", (ptr, i32, i32, ptr)>)>, struct<"struct.google::protobuf::internal::ArenaStringPtr", (ptr)>, struct<"struct.google::protobuf::internal::ArenaStringPtr", (ptr)>, ptr, i32, struct<"class.google::protobuf::internal::CachedSize", (struct<"struct.std::atomic", (struct<"struct.std::__atomic_base", (i32)>)>)>)>
    %34 = llvm.load %33 atomic monotonic {alignment = 4 : i64} : !llvm.ptr -> i32
    %35 = llvm.icmp "ult" %34, %6 : i32
    %36 = llvm.trunc %34 : i32 to i8
    llvm.cond_br %35, ^bb6, ^bb7
  ^bb6:  // pred: ^bb5
    llvm.store %36, %32 {alignment = 1 : i64, tbaa = [#tbaa_tag1]} : i8, !llvm.ptr
    %37 = llvm.getelementptr inbounds %30[%10] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    llvm.br ^bb12(%37 : !llvm.ptr)
  ^bb7:  // pred: ^bb5
    %38 = llvm.or %36, %7  : i8
    llvm.store %38, %32 {alignment = 1 : i64, tbaa = [#tbaa_tag1]} : i8, !llvm.ptr
    %39 = llvm.lshr %34, %8  : i32
    %40 = llvm.icmp "ult" %34, %9 : i32
    llvm.cond_br %40, ^bb8, ^bb9
  ^bb8:  // pred: ^bb7
    %41 = llvm.trunc %39 : i32 to i8
    %42 = llvm.getelementptr inbounds %30[%10] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    llvm.store %41, %42 {alignment = 1 : i64, tbaa = [#tbaa_tag1]} : i8, !llvm.ptr
    %43 = llvm.getelementptr inbounds %30[%12] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    llvm.br ^bb12(%43 : !llvm.ptr)
  ^bb9:  // pred: ^bb7
    %44 = llvm.getelementptr inbounds %30[%10] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    llvm.br ^bb10(%39, %44 : i32, !llvm.ptr)
  ^bb10(%45: i32, %46: !llvm.ptr):  // 2 preds: ^bb9, ^bb10
    %47 = llvm.trunc %45 : i32 to i8
    %48 = llvm.or %47, %7  : i8
    llvm.store %48, %46 {alignment = 1 : i64, tbaa = [#tbaa_tag1]} : i8, !llvm.ptr
    %49 = llvm.lshr %45, %8  : i32
    %50 = llvm.getelementptr inbounds %46[%4] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %51 = llvm.icmp "ugt" %45, %11 : i32
    llvm.cond_br %51 weights([1, 2000]), ^bb10(%49, %50 : i32, !llvm.ptr), ^bb11 {loop_annotation = #loop_annotation}
  ^bb11:  // pred: ^bb10
    %52 = llvm.trunc %49 : i32 to i8
    %53 = llvm.getelementptr inbounds %46[%10] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    llvm.store %52, %50 {alignment = 1 : i64, tbaa = [#tbaa_tag1]} : i8, !llvm.ptr
    llvm.br ^bb12(%53 : !llvm.ptr)
  ^bb12(%54: !llvm.ptr):  // 3 preds: ^bb6, ^bb8, ^bb11
    %55 = llvm.call @_ZNK8tutorial6Person18_InternalSerializeEPhPN6google8protobuf2io19EpsCopyOutputStreamE(%31, %54, %arg2) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> !llvm.ptr
    %56 = llvm.add %26, %1 overflow<nuw>  : i32
    %57 = llvm.icmp "eq" %56, %15 : i32
    llvm.cond_br %57, ^bb2(%55 : !llvm.ptr), ^bb3(%55, %56 : !llvm.ptr, i32) {loop_annotation = #loop_annotation}
  ^bb13:  // pred: ^bb2
    %58 = llvm.and %22, %13  : i64
    %59 = llvm.inttoptr %58 : i64 to !llvm.ptr
    %60 = llvm.getelementptr inbounds %59[%0, 1] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.google::protobuf::internal::InternalMetadata::Container", (struct<"struct.google::protobuf::internal::InternalMetadata::ContainerBase", (ptr)>, struct<"class.google::protobuf::UnknownFieldSet", (struct<"class.std::vector", (struct<"struct.std::_Vector_base", (struct<"struct.std::_Vector_base<google::protobuf::UnknownField, std::allocator<google::protobuf::UnknownField>>::_Vector_impl", (struct<"struct.std::_Vector_base<google::protobuf::UnknownField, std::allocator<google::protobuf::UnknownField>>::_Vector_impl_data", (ptr, ptr, ptr)>)>)>)>)>)>
    %61 = llvm.call @_ZN6google8protobuf8internal10WireFormat37InternalSerializeUnknownFieldsToArrayERKNS0_15UnknownFieldSetEPhPNS0_2io19EpsCopyOutputStreamE(%60, %19, %arg2) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> !llvm.ptr
    llvm.br ^bb14(%61 : !llvm.ptr)
  ^bb14(%62: !llvm.ptr):  // 2 preds: ^bb2, ^bb13
    llvm.return %62 : !llvm.ptr
  }
  llvm.func unnamed_addr @_ZNK8tutorial11AddressBook12ByteSizeLongEv(%arg0: !llvm.ptr {llvm.align = 8 : i64, llvm.dereferenceable = 48 : i64, llvm.nonnull, llvm.noundef}) -> (i64 {llvm.noundef}) attributes {alignment = 2 : i64, frame_pointer = #llvm.framePointerKind<none>, passthrough = [["uwtable", "2"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], ["tune-cpu", "generic"]], personality = @__gxx_personality_v0, target_cpu = "x86-64", target_features = #llvm.target_features<["+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>} {
    %0 = llvm.mlir.constant(0 : i64) : i64
    %1 = llvm.mlir.constant(1 : i32) : i32
    %2 = llvm.mlir.constant(0 : i32) : i32
    %3 = llvm.mlir.constant(3 : i32) : i32
    %4 = llvm.mlir.zero : !llvm.ptr
    %5 = llvm.mlir.constant(31 : i32) : i32
    %6 = llvm.mlir.constant(9 : i32) : i32
    %7 = llvm.mlir.constant(73 : i32) : i32
    %8 = llvm.mlir.constant(6 : i32) : i32
    %9 = llvm.mlir.constant(1 : i64) : i64
    %10 = llvm.mlir.constant(2 : i32) : i32
    %11 = llvm.getelementptr inbounds %arg0[%0, 1, 0, 1] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.tutorial::AddressBook", packed (struct<"class.google::protobuf::Message", (struct<"class.google::protobuf::MessageLite", (ptr, struct<"class.google::protobuf::internal::InternalMetadata", (ptr)>)>)>, struct<"class.google::protobuf::RepeatedPtrField.18", (struct<"class.google::protobuf::internal::RepeatedPtrFieldBase", (ptr, i32, i32, ptr)>)>, struct<"class.google::protobuf::internal::CachedSize", (struct<"struct.std::atomic", (struct<"struct.std::__atomic_base", (i32)>)>)>, array<4 x i8>)>
    %12 = llvm.load %11 {alignment = 8 : i64, tbaa = [#tbaa_tag17]} : !llvm.ptr -> i32
    %13 = llvm.sext %12 : i32 to i64
    %14 = llvm.getelementptr inbounds %arg0[%0, 1, 0, 3] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.tutorial::AddressBook", packed (struct<"class.google::protobuf::Message", (struct<"class.google::protobuf::MessageLite", (ptr, struct<"class.google::protobuf::internal::InternalMetadata", (ptr)>)>)>, struct<"class.google::protobuf::RepeatedPtrField.18", (struct<"class.google::protobuf::internal::RepeatedPtrFieldBase", (ptr, i32, i32, ptr)>)>, struct<"class.google::protobuf::internal::CachedSize", (struct<"struct.std::atomic", (struct<"struct.std::__atomic_base", (i32)>)>)>, array<4 x i8>)>
    %15 = llvm.load %14 {alignment = 8 : i64, tbaa = [#tbaa_tag15]} : !llvm.ptr -> !llvm.ptr
    %16 = llvm.icmp "eq" %15, %4 : !llvm.ptr
    %17 = llvm.getelementptr inbounds %15[%0, 1, %0] : (!llvm.ptr, i64, i64) -> !llvm.ptr, !llvm.struct<"struct.google::protobuf::internal::RepeatedPtrFieldBase::Rep", (i32, array<1 x ptr>)>
    %18 = llvm.select %16, %4, %17 : i1, !llvm.ptr
    %19 = llvm.getelementptr inbounds %18[%13] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.ptr
    %20 = llvm.icmp "eq" %12, %2 : i32
    llvm.cond_br %20, ^bb1(%0 : i64), ^bb2(%13, %18 : i64, !llvm.ptr)
  ^bb1(%21: i64):  // 2 preds: ^bb0, ^bb2
    %22 = llvm.getelementptr inbounds %arg0[%0, 0, 0, 1] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.tutorial::AddressBook", packed (struct<"class.google::protobuf::Message", (struct<"class.google::protobuf::MessageLite", (ptr, struct<"class.google::protobuf::internal::InternalMetadata", (ptr)>)>)>, struct<"class.google::protobuf::RepeatedPtrField.18", (struct<"class.google::protobuf::internal::RepeatedPtrFieldBase", (ptr, i32, i32, ptr)>)>, struct<"class.google::protobuf::internal::CachedSize", (struct<"struct.std::atomic", (struct<"struct.std::__atomic_base", (i32)>)>)>, array<4 x i8>)>
    %23 = llvm.getelementptr inbounds %22[%0, 0] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.google::protobuf::internal::InternalMetadata", (ptr)>
    %24 = llvm.load %23 {alignment = 8 : i64, tbaa = [#tbaa_tag5]} : !llvm.ptr -> !llvm.ptr
    %25 = llvm.ptrtoint %24 : !llvm.ptr to i64
    %26 = llvm.and %25, %9  : i64
    %27 = llvm.icmp "eq" %26, %0 : i64
    llvm.cond_br %27 weights([2000, 1]), ^bb4, ^bb3
  ^bb2(%28: i64, %29: !llvm.ptr):  // 2 preds: ^bb0, ^bb2
    %30 = llvm.bitcast %29 : !llvm.ptr to !llvm.ptr
    %31 = llvm.load %30 {alignment = 8 : i64, tbaa = [#tbaa_tag2]} : !llvm.ptr -> !llvm.ptr
    %32 = llvm.call @_ZNK8tutorial6Person12ByteSizeLongEv(%31) : (!llvm.ptr) -> i64
    %33 = llvm.trunc %32 : i64 to i32
    %34 = llvm.or %33, %1  : i32
    %35 = "llvm.intr.ctlz"(%34) <{is_zero_poison = true}> : (i32) -> i32
    %36 = llvm.xor %35, %5  : i32
    %37 = llvm.mul %36, %6 overflow<nsw, nuw>  : i32
    %38 = llvm.add %37, %7 overflow<nsw, nuw>  : i32
    %39 = llvm.lshr %38, %8  : i32
    %40 = llvm.zext %39 : i32 to i64
    %41 = llvm.add %32, %28  : i64
    %42 = llvm.add %41, %40  : i64
    %43 = llvm.getelementptr inbounds %29[%9] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.ptr
    %44 = llvm.icmp "eq" %43, %19 : !llvm.ptr
    llvm.cond_br %44, ^bb1(%42 : i64), ^bb2(%42, %43 : i64, !llvm.ptr)
  ^bb3:  // pred: ^bb1
    %45 = llvm.getelementptr inbounds %arg0[%0, 2] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.tutorial::AddressBook", packed (struct<"class.google::protobuf::Message", (struct<"class.google::protobuf::MessageLite", (ptr, struct<"class.google::protobuf::internal::InternalMetadata", (ptr)>)>)>, struct<"class.google::protobuf::RepeatedPtrField.18", (struct<"class.google::protobuf::internal::RepeatedPtrFieldBase", (ptr, i32, i32, ptr)>)>, struct<"class.google::protobuf::internal::CachedSize", (struct<"struct.std::atomic", (struct<"struct.std::__atomic_base", (i32)>)>)>, array<4 x i8>)>
    %46 = llvm.call @_ZN6google8protobuf8internal24ComputeUnknownFieldsSizeERKNS1_16InternalMetadataEmPNS1_10CachedSizeE(%22, %21, %45) : (!llvm.ptr, i64, !llvm.ptr) -> i64
    llvm.br ^bb5(%46 : i64)
  ^bb4:  // pred: ^bb1
    %47 = llvm.trunc %21 : i64 to i32
    %48 = llvm.getelementptr inbounds %arg0[%0, 2, 0, 0, 0] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.tutorial::AddressBook", packed (struct<"class.google::protobuf::Message", (struct<"class.google::protobuf::MessageLite", (ptr, struct<"class.google::protobuf::internal::InternalMetadata", (ptr)>)>)>, struct<"class.google::protobuf::RepeatedPtrField.18", (struct<"class.google::protobuf::internal::RepeatedPtrFieldBase", (ptr, i32, i32, ptr)>)>, struct<"class.google::protobuf::internal::CachedSize", (struct<"struct.std::atomic", (struct<"struct.std::__atomic_base", (i32)>)>)>, array<4 x i8>)>
    llvm.store %47, %48 atomic monotonic {alignment = 8 : i64} : i32, !llvm.ptr
    llvm.br ^bb5(%21 : i64)
  ^bb5(%49: i64):  // 2 preds: ^bb3, ^bb4
    llvm.return %49 : i64
  }
  llvm.func unnamed_addr @_ZN8tutorial11AddressBook9MergeFromERKN6google8protobuf7MessageE(%arg0: !llvm.ptr {llvm.align = 8 : i64, llvm.dereferenceable = 48 : i64, llvm.nonnull, llvm.noundef}, %arg1: !llvm.ptr {llvm.align = 8 : i64, llvm.dereferenceable = 16 : i64, llvm.nonnull, llvm.noundef}) attributes {alignment = 2 : i64, frame_pointer = #llvm.framePointerKind<none>, passthrough = ["mustprogress", ["uwtable", "2"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], ["tune-cpu", "generic"]], personality = @__gxx_personality_v0, target_cpu = "x86-64", target_features = #llvm.target_features<["+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>} {
    %0 = llvm.mlir.constant(1 : i32) : i32
    %1 = llvm.mlir.constant(0 : i64) : i64
    %2 = llvm.mlir.constant(0 : i32) : i32
    %3 = llvm.mlir.constant(3 : i32) : i32
    %4 = llvm.mlir.constant("build/addressbook.pb.cc\00") : !llvm.array<24 x i8>
    %5 = llvm.mlir.addressof @".str.5" : !llvm.ptr
    %6 = llvm.mlir.constant(928 : i32) : i32
    %7 = llvm.mlir.constant("CHECK failed: (&from) != (this): \00") : !llvm.array<34 x i8>
    %8 = llvm.mlir.addressof @".str.6" : !llvm.ptr
    %9 = llvm.mlir.addressof @_ZTIN6google8protobuf7MessageE : !llvm.ptr
    %10 = llvm.mlir.constant("N8tutorial11AddressBookE\00") : !llvm.array<25 x i8>
    %11 = llvm.mlir.addressof @_ZTSN8tutorial11AddressBookE : !llvm.ptr
    %12 = llvm.mlir.constant(2 : i64) : i64
    %13 = llvm.mlir.addressof @_ZTVN10__cxxabiv120__si_class_type_infoE : !llvm.ptr
    %14 = llvm.getelementptr inbounds %13[%12] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.ptr
    %15 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, ptr)>
    %16 = llvm.insertvalue %14, %15[0] : !llvm.struct<(ptr, ptr, ptr)> 
    %17 = llvm.insertvalue %11, %16[1] : !llvm.struct<(ptr, ptr, ptr)> 
    %18 = llvm.insertvalue %9, %17[2] : !llvm.struct<(ptr, ptr, ptr)> 
    %19 = llvm.mlir.addressof @_ZTIN8tutorial11AddressBookE : !llvm.ptr
    %20 = llvm.mlir.zero : !llvm.ptr
    %21 = llvm.alloca %0 x !llvm.struct<"class.google::protobuf::internal::LogMessage", (i32, ptr, i32, struct<"class.std::__cxx11::basic_string", (struct<"struct.std::__cxx11::basic_string<char>::_Alloc_hider", (ptr)>, i64, struct<"union.anon", (i64, array<8 x i8>)>)>)> {alignment = 8 : i64} : (i32) -> !llvm.ptr
    %22 = llvm.alloca %0 x !llvm.struct<"class.google::protobuf::internal::LogFinisher", (i8)> {alignment = 1 : i64} : (i32) -> !llvm.ptr
    %23 = llvm.getelementptr inbounds %arg0[%1, 0] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.tutorial::AddressBook", packed (struct<"class.google::protobuf::Message", (struct<"class.google::protobuf::MessageLite", (ptr, struct<"class.google::protobuf::internal::InternalMetadata", (ptr)>)>)>, struct<"class.google::protobuf::RepeatedPtrField.18", (struct<"class.google::protobuf::internal::RepeatedPtrFieldBase", (ptr, i32, i32, ptr)>)>, struct<"class.google::protobuf::internal::CachedSize", (struct<"struct.std::atomic", (struct<"struct.std::__atomic_base", (i32)>)>)>, array<4 x i8>)>
    %24 = llvm.icmp "eq" %23, %arg1 : !llvm.ptr
    %25 = llvm.getelementptr inbounds %22[%1, 0] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.google::protobuf::internal::LogFinisher", (i8)>
    llvm.intr.lifetime.start 1, %25 : !llvm.ptr
    llvm.cond_br %24, ^bb1, ^bb3
  ^bb1:  // pred: ^bb0
    %26 = llvm.bitcast %21 : !llvm.ptr to !llvm.ptr
    llvm.intr.lifetime.start 56, %26 : !llvm.ptr
    llvm.call @_ZN6google8protobuf8internal10LogMessageC1ENS0_8LogLevelEPKci(%21, %3, %5, %6) : (!llvm.ptr, i32, !llvm.ptr, i32) -> ()
    %27 = llvm.invoke @_ZN6google8protobuf8internal10LogMessagelsEPKc(%21, %8) to ^bb2 unwind ^bb7 : (!llvm.ptr, !llvm.ptr) -> !llvm.ptr
  ^bb2:  // pred: ^bb1
    llvm.invoke @_ZN6google8protobuf8internal11LogFinisheraSERNS1_10LogMessageE(%22, %27) to ^bb4 unwind ^bb8 : (!llvm.ptr, !llvm.ptr) -> ()
  ^bb3:  // pred: ^bb0
    llvm.intr.lifetime.end 1, %25 : !llvm.ptr
    llvm.br ^bb5
  ^bb4:  // pred: ^bb2
    llvm.intr.lifetime.end 1, %25 : !llvm.ptr
    llvm.call @_ZN6google8protobuf8internal10LogMessageD1Ev(%21) : (!llvm.ptr) -> ()
    llvm.intr.lifetime.end 56, %26 : !llvm.ptr
    llvm.br ^bb5
  ^bb5:  // 2 preds: ^bb3, ^bb4
    %28 = llvm.bitcast %arg1 : !llvm.ptr to !llvm.ptr
    %29 = llvm.call @__dynamic_cast(%28, %9, %19, %1) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i64) -> !llvm.ptr
    %30 = llvm.icmp "eq" %29, %20 : !llvm.ptr
    llvm.cond_br %30, ^bb6, ^bb10
  ^bb6:  // pred: ^bb5
    llvm.call @_ZN6google8protobuf8internal13ReflectionOps5MergeERKNS0_7MessageEPS3_(%arg1, %23) : (!llvm.ptr, !llvm.ptr) -> ()
    llvm.br ^bb11
  ^bb7:  // pred: ^bb1
    %31 = llvm.landingpad cleanup : !llvm.struct<(ptr, i32)>
    llvm.br ^bb9(%31 : !llvm.struct<(ptr, i32)>)
  ^bb8:  // pred: ^bb2
    %32 = llvm.landingpad cleanup : !llvm.struct<(ptr, i32)>
    llvm.intr.lifetime.end 1, %25 : !llvm.ptr
    llvm.br ^bb9(%32 : !llvm.struct<(ptr, i32)>)
  ^bb9(%33: !llvm.struct<(ptr, i32)>):  // 2 preds: ^bb7, ^bb8
    llvm.call @_ZN6google8protobuf8internal10LogMessageD1Ev(%21) : (!llvm.ptr) -> ()
    llvm.intr.lifetime.end 56, %26 : !llvm.ptr
    llvm.resume %33 : !llvm.struct<(ptr, i32)>
  ^bb10:  // pred: ^bb5
    %34 = llvm.bitcast %29 : !llvm.ptr to !llvm.ptr
    llvm.call @_ZN8tutorial11AddressBook9MergeFromERKS0_(%arg0, %34) : (!llvm.ptr, !llvm.ptr) -> ()
    llvm.br ^bb11
  ^bb11:  // 2 preds: ^bb6, ^bb10
    llvm.return
  }
  llvm.func local_unnamed_addr @_ZN8tutorial11AddressBook9MergeFromERKS0_(%arg0: !llvm.ptr {llvm.align = 8 : i64, llvm.dereferenceable = 48 : i64, llvm.nonnull, llvm.noundef}, %arg1: !llvm.ptr {llvm.align = 8 : i64, llvm.dereferenceable = 48 : i64, llvm.nonnull, llvm.noundef}) attributes {alignment = 2 : i64, frame_pointer = #llvm.framePointerKind<none>, passthrough = ["mustprogress", ["uwtable", "2"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], ["tune-cpu", "generic"]], personality = @__gxx_personality_v0, target_cpu = "x86-64", target_features = #llvm.target_features<["+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>} {
    %0 = llvm.mlir.constant(1 : i32) : i32
    %1 = llvm.mlir.constant(0 : i64) : i64
    %2 = llvm.mlir.constant(0 : i32) : i32
    %3 = llvm.mlir.constant(3 : i32) : i32
    %4 = llvm.mlir.constant("build/addressbook.pb.cc\00") : !llvm.array<24 x i8>
    %5 = llvm.mlir.addressof @".str.5" : !llvm.ptr
    %6 = llvm.mlir.constant(943 : i32) : i32
    %7 = llvm.mlir.constant("CHECK failed: (&from) != (this): \00") : !llvm.array<34 x i8>
    %8 = llvm.mlir.addressof @".str.6" : !llvm.ptr
    %9 = llvm.mlir.constant(1 : i64) : i64
    %10 = llvm.mlir.constant(-2 : i64) : i64
    %11 = llvm.alloca %0 x !llvm.struct<"class.google::protobuf::internal::LogMessage", (i32, ptr, i32, struct<"class.std::__cxx11::basic_string", (struct<"struct.std::__cxx11::basic_string<char>::_Alloc_hider", (ptr)>, i64, struct<"union.anon", (i64, array<8 x i8>)>)>)> {alignment = 8 : i64} : (i32) -> !llvm.ptr
    %12 = llvm.alloca %0 x !llvm.struct<"class.google::protobuf::internal::LogFinisher", (i8)> {alignment = 1 : i64} : (i32) -> !llvm.ptr
    %13 = llvm.icmp "eq" %arg1, %arg0 : !llvm.ptr
    %14 = llvm.getelementptr inbounds %12[%1, 0] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.google::protobuf::internal::LogFinisher", (i8)>
    llvm.intr.lifetime.start 1, %14 : !llvm.ptr
    llvm.cond_br %13, ^bb1, ^bb3
  ^bb1:  // pred: ^bb0
    %15 = llvm.bitcast %11 : !llvm.ptr to !llvm.ptr
    llvm.intr.lifetime.start 56, %15 : !llvm.ptr
    llvm.call @_ZN6google8protobuf8internal10LogMessageC1ENS0_8LogLevelEPKci(%11, %3, %5, %6) : (!llvm.ptr, i32, !llvm.ptr, i32) -> ()
    %16 = llvm.invoke @_ZN6google8protobuf8internal10LogMessagelsEPKc(%11, %8) to ^bb2 unwind ^bb11 : (!llvm.ptr, !llvm.ptr) -> !llvm.ptr
  ^bb2:  // pred: ^bb1
    llvm.invoke @_ZN6google8protobuf8internal11LogFinisheraSERNS1_10LogMessageE(%12, %16) to ^bb4 unwind ^bb12 : (!llvm.ptr, !llvm.ptr) -> ()
  ^bb3:  // pred: ^bb0
    llvm.intr.lifetime.end 1, %14 : !llvm.ptr
    llvm.br ^bb5
  ^bb4:  // pred: ^bb2
    llvm.intr.lifetime.end 1, %14 : !llvm.ptr
    llvm.call @_ZN6google8protobuf8internal10LogMessageD1Ev(%11) : (!llvm.ptr) -> ()
    llvm.intr.lifetime.end 56, %15 : !llvm.ptr
    llvm.br ^bb5
  ^bb5:  // 2 preds: ^bb3, ^bb4
    %17 = llvm.getelementptr inbounds %arg0[%1, 0, 0, 1] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.tutorial::AddressBook", packed (struct<"class.google::protobuf::Message", (struct<"class.google::protobuf::MessageLite", (ptr, struct<"class.google::protobuf::internal::InternalMetadata", (ptr)>)>)>, struct<"class.google::protobuf::RepeatedPtrField.18", (struct<"class.google::protobuf::internal::RepeatedPtrFieldBase", (ptr, i32, i32, ptr)>)>, struct<"class.google::protobuf::internal::CachedSize", (struct<"struct.std::atomic", (struct<"struct.std::__atomic_base", (i32)>)>)>, array<4 x i8>)>
    %18 = llvm.getelementptr inbounds %arg1[%1, 0, 0, 1, 0] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.tutorial::AddressBook", packed (struct<"class.google::protobuf::Message", (struct<"class.google::protobuf::MessageLite", (ptr, struct<"class.google::protobuf::internal::InternalMetadata", (ptr)>)>)>, struct<"class.google::protobuf::RepeatedPtrField.18", (struct<"class.google::protobuf::internal::RepeatedPtrFieldBase", (ptr, i32, i32, ptr)>)>, struct<"class.google::protobuf::internal::CachedSize", (struct<"struct.std::atomic", (struct<"struct.std::__atomic_base", (i32)>)>)>, array<4 x i8>)>
    %19 = llvm.load %18 {alignment = 8 : i64, tbaa = [#tbaa_tag5]} : !llvm.ptr -> !llvm.ptr
    %20 = llvm.ptrtoint %19 : !llvm.ptr to i64
    %21 = llvm.and %20, %9  : i64
    %22 = llvm.icmp "eq" %21, %1 : i64
    llvm.cond_br %22, ^bb10, ^bb6
  ^bb6:  // pred: ^bb5
    %23 = llvm.and %20, %10  : i64
    %24 = llvm.inttoptr %23 : i64 to !llvm.ptr
    %25 = llvm.getelementptr inbounds %24[%1, 1] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.google::protobuf::internal::InternalMetadata::Container", (struct<"struct.google::protobuf::internal::InternalMetadata::ContainerBase", (ptr)>, struct<"class.google::protobuf::UnknownFieldSet", (struct<"class.std::vector", (struct<"struct.std::_Vector_base", (struct<"struct.std::_Vector_base<google::protobuf::UnknownField, std::allocator<google::protobuf::UnknownField>>::_Vector_impl", (struct<"struct.std::_Vector_base<google::protobuf::UnknownField, std::allocator<google::protobuf::UnknownField>>::_Vector_impl_data", (ptr, ptr, ptr)>)>)>)>)>)>
    %26 = llvm.getelementptr inbounds %17[%1, 0] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.google::protobuf::internal::InternalMetadata", (ptr)>
    %27 = llvm.load %26 {alignment = 8 : i64, tbaa = [#tbaa_tag5]} : !llvm.ptr -> !llvm.ptr
    %28 = llvm.ptrtoint %27 : !llvm.ptr to i64
    %29 = llvm.and %28, %9  : i64
    %30 = llvm.icmp "eq" %29, %1 : i64
    llvm.cond_br %30 weights([1, 2000]), ^bb8, ^bb7
  ^bb7:  // pred: ^bb6
    %31 = llvm.and %28, %10  : i64
    %32 = llvm.inttoptr %31 : i64 to !llvm.ptr
    %33 = llvm.getelementptr inbounds %32[%1, 1] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.google::protobuf::internal::InternalMetadata::Container", (struct<"struct.google::protobuf::internal::InternalMetadata::ContainerBase", (ptr)>, struct<"class.google::protobuf::UnknownFieldSet", (struct<"class.std::vector", (struct<"struct.std::_Vector_base", (struct<"struct.std::_Vector_base<google::protobuf::UnknownField, std::allocator<google::protobuf::UnknownField>>::_Vector_impl", (struct<"struct.std::_Vector_base<google::protobuf::UnknownField, std::allocator<google::protobuf::UnknownField>>::_Vector_impl_data", (ptr, ptr, ptr)>)>)>)>)>)>
    llvm.br ^bb9(%33 : !llvm.ptr)
  ^bb8:  // pred: ^bb6
    %34 = llvm.call @_ZN6google8protobuf8internal16InternalMetadata27mutable_unknown_fields_slowINS0_15UnknownFieldSetEEEPT_v(%17) : (!llvm.ptr) -> !llvm.ptr
    llvm.br ^bb9(%34 : !llvm.ptr)
  ^bb9(%35: !llvm.ptr):  // 2 preds: ^bb7, ^bb8
    llvm.call @_ZN6google8protobuf15UnknownFieldSet9MergeFromERKS1_(%35, %25) : (!llvm.ptr, !llvm.ptr) -> ()
    llvm.br ^bb10
  ^bb10:  // 2 preds: ^bb5, ^bb9
    %36 = llvm.getelementptr inbounds %arg0[%1, 1, 0] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.tutorial::AddressBook", packed (struct<"class.google::protobuf::Message", (struct<"class.google::protobuf::MessageLite", (ptr, struct<"class.google::protobuf::internal::InternalMetadata", (ptr)>)>)>, struct<"class.google::protobuf::RepeatedPtrField.18", (struct<"class.google::protobuf::internal::RepeatedPtrFieldBase", (ptr, i32, i32, ptr)>)>, struct<"class.google::protobuf::internal::CachedSize", (struct<"struct.std::atomic", (struct<"struct.std::__atomic_base", (i32)>)>)>, array<4 x i8>)>
    %37 = llvm.getelementptr inbounds %arg1[%1, 1, 0] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.tutorial::AddressBook", packed (struct<"class.google::protobuf::Message", (struct<"class.google::protobuf::MessageLite", (ptr, struct<"class.google::protobuf::internal::InternalMetadata", (ptr)>)>)>, struct<"class.google::protobuf::RepeatedPtrField.18", (struct<"class.google::protobuf::internal::RepeatedPtrFieldBase", (ptr, i32, i32, ptr)>)>, struct<"class.google::protobuf::internal::CachedSize", (struct<"struct.std::atomic", (struct<"struct.std::__atomic_base", (i32)>)>)>, array<4 x i8>)>
    llvm.call @_ZN6google8protobuf8internal20RepeatedPtrFieldBase9MergeFromINS0_16RepeatedPtrFieldIN8tutorial6PersonEE11TypeHandlerEEEvRKS2_(%36, %37) : (!llvm.ptr, !llvm.ptr) -> ()
    llvm.return
  ^bb11:  // pred: ^bb1
    %38 = llvm.landingpad cleanup : !llvm.struct<(ptr, i32)>
    llvm.br ^bb13(%38 : !llvm.struct<(ptr, i32)>)
  ^bb12:  // pred: ^bb2
    %39 = llvm.landingpad cleanup : !llvm.struct<(ptr, i32)>
    llvm.intr.lifetime.end 1, %14 : !llvm.ptr
    llvm.br ^bb13(%39 : !llvm.struct<(ptr, i32)>)
  ^bb13(%40: !llvm.struct<(ptr, i32)>):  // 2 preds: ^bb11, ^bb12
    llvm.call @_ZN6google8protobuf8internal10LogMessageD1Ev(%11) : (!llvm.ptr) -> ()
    llvm.intr.lifetime.end 56, %15 : !llvm.ptr
    llvm.resume %40 : !llvm.struct<(ptr, i32)>
  }
  llvm.func unnamed_addr @_ZN8tutorial11AddressBook8CopyFromERKN6google8protobuf7MessageE(%arg0: !llvm.ptr {llvm.align = 8 : i64, llvm.dereferenceable = 48 : i64, llvm.nonnull, llvm.noundef}, %arg1: !llvm.ptr {llvm.align = 8 : i64, llvm.dereferenceable = 16 : i64, llvm.nonnull, llvm.noundef}) attributes {alignment = 2 : i64, frame_pointer = #llvm.framePointerKind<none>, passthrough = [["uwtable", "2"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], ["tune-cpu", "generic"]], target_cpu = "x86-64", target_features = #llvm.target_features<["+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>} {
    %0 = llvm.mlir.constant(0 : i64) : i64
    %1 = llvm.mlir.constant(0 : i32) : i32
    %2 = llvm.mlir.constant(1 : i32) : i32
    %3 = llvm.mlir.constant(1 : i64) : i64
    %4 = llvm.mlir.constant(-2 : i64) : i64
    %5 = llvm.getelementptr inbounds %arg0[%0, 0] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.tutorial::AddressBook", packed (struct<"class.google::protobuf::Message", (struct<"class.google::protobuf::MessageLite", (ptr, struct<"class.google::protobuf::internal::InternalMetadata", (ptr)>)>)>, struct<"class.google::protobuf::RepeatedPtrField.18", (struct<"class.google::protobuf::internal::RepeatedPtrFieldBase", (ptr, i32, i32, ptr)>)>, struct<"class.google::protobuf::internal::CachedSize", (struct<"struct.std::atomic", (struct<"struct.std::__atomic_base", (i32)>)>)>, array<4 x i8>)>
    %6 = llvm.icmp "eq" %5, %arg1 : !llvm.ptr
    llvm.cond_br %6, ^bb5, ^bb1
  ^bb1:  // pred: ^bb0
    %7 = llvm.getelementptr inbounds %arg0[%0, 1, 0] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.tutorial::AddressBook", packed (struct<"class.google::protobuf::Message", (struct<"class.google::protobuf::MessageLite", (ptr, struct<"class.google::protobuf::internal::InternalMetadata", (ptr)>)>)>, struct<"class.google::protobuf::RepeatedPtrField.18", (struct<"class.google::protobuf::internal::RepeatedPtrFieldBase", (ptr, i32, i32, ptr)>)>, struct<"class.google::protobuf::internal::CachedSize", (struct<"struct.std::atomic", (struct<"struct.std::__atomic_base", (i32)>)>)>, array<4 x i8>)>
    llvm.call @_ZN6google8protobuf8internal20RepeatedPtrFieldBase5ClearINS0_16RepeatedPtrFieldIN8tutorial6PersonEE11TypeHandlerEEEvv(%7) : (!llvm.ptr) -> ()
    %8 = llvm.getelementptr inbounds %arg0[%0, 0, 0, 1, 0] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.tutorial::AddressBook", packed (struct<"class.google::protobuf::Message", (struct<"class.google::protobuf::MessageLite", (ptr, struct<"class.google::protobuf::internal::InternalMetadata", (ptr)>)>)>, struct<"class.google::protobuf::RepeatedPtrField.18", (struct<"class.google::protobuf::internal::RepeatedPtrFieldBase", (ptr, i32, i32, ptr)>)>, struct<"class.google::protobuf::internal::CachedSize", (struct<"struct.std::atomic", (struct<"struct.std::__atomic_base", (i32)>)>)>, array<4 x i8>)>
    %9 = llvm.load %8 {alignment = 8 : i64, tbaa = [#tbaa_tag5]} : !llvm.ptr -> !llvm.ptr
    %10 = llvm.ptrtoint %9 : !llvm.ptr to i64
    %11 = llvm.and %10, %3  : i64
    %12 = llvm.icmp "eq" %11, %0 : i64
    llvm.cond_br %12, ^bb4, ^bb2
  ^bb2:  // pred: ^bb1
    %13 = llvm.and %10, %4  : i64
    %14 = llvm.inttoptr %13 : i64 to !llvm.ptr
    %15 = llvm.getelementptr inbounds %14[%0, 1] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.google::protobuf::internal::InternalMetadata::Container", (struct<"struct.google::protobuf::internal::InternalMetadata::ContainerBase", (ptr)>, struct<"class.google::protobuf::UnknownFieldSet", (struct<"class.std::vector", (struct<"struct.std::_Vector_base", (struct<"struct.std::_Vector_base<google::protobuf::UnknownField, std::allocator<google::protobuf::UnknownField>>::_Vector_impl", (struct<"struct.std::_Vector_base<google::protobuf::UnknownField, std::allocator<google::protobuf::UnknownField>>::_Vector_impl_data", (ptr, ptr, ptr)>)>)>)>)>)>
    %16 = llvm.getelementptr inbounds %15[%0, 0, 0, 0, 0, 0] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.google::protobuf::UnknownFieldSet", (struct<"class.std::vector", (struct<"struct.std::_Vector_base", (struct<"struct.std::_Vector_base<google::protobuf::UnknownField, std::allocator<google::protobuf::UnknownField>>::_Vector_impl", (struct<"struct.std::_Vector_base<google::protobuf::UnknownField, std::allocator<google::protobuf::UnknownField>>::_Vector_impl_data", (ptr, ptr, ptr)>)>)>)>)>
    %17 = llvm.load %16 {alignment = 8 : i64, tbaa = [#tbaa_tag2]} : !llvm.ptr -> !llvm.ptr
    %18 = llvm.getelementptr inbounds %14[%0, 1, 0, 0, 0, 0, 1] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.google::protobuf::internal::InternalMetadata::Container", (struct<"struct.google::protobuf::internal::InternalMetadata::ContainerBase", (ptr)>, struct<"class.google::protobuf::UnknownFieldSet", (struct<"class.std::vector", (struct<"struct.std::_Vector_base", (struct<"struct.std::_Vector_base<google::protobuf::UnknownField, std::allocator<google::protobuf::UnknownField>>::_Vector_impl", (struct<"struct.std::_Vector_base<google::protobuf::UnknownField, std::allocator<google::protobuf::UnknownField>>::_Vector_impl_data", (ptr, ptr, ptr)>)>)>)>)>)>
    %19 = llvm.load %18 {alignment = 8 : i64, tbaa = [#tbaa_tag2]} : !llvm.ptr -> !llvm.ptr
    %20 = llvm.icmp "eq" %17, %19 : !llvm.ptr
    llvm.cond_br %20, ^bb4, ^bb3
  ^bb3:  // pred: ^bb2
    llvm.call @_ZN6google8protobuf15UnknownFieldSet13ClearFallbackEv(%15) : (!llvm.ptr) -> ()
    llvm.br ^bb4
  ^bb4:  // 3 preds: ^bb1, ^bb2, ^bb3
    llvm.call @_ZN8tutorial11AddressBook9MergeFromERKN6google8protobuf7MessageE(%arg0, %arg1) : (!llvm.ptr, !llvm.ptr) -> ()
    llvm.br ^bb5
  ^bb5:  // 2 preds: ^bb0, ^bb4
    llvm.return
  }
  llvm.func local_unnamed_addr @_ZN8tutorial11AddressBook8CopyFromERKS0_(%arg0: !llvm.ptr {llvm.align = 8 : i64, llvm.dereferenceable = 48 : i64, llvm.nonnull, llvm.noundef}, %arg1: !llvm.ptr {llvm.align = 8 : i64, llvm.dereferenceable = 48 : i64, llvm.nonnull, llvm.noundef}) attributes {alignment = 2 : i64, frame_pointer = #llvm.framePointerKind<none>, passthrough = [["uwtable", "2"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], ["tune-cpu", "generic"]], target_cpu = "x86-64", target_features = #llvm.target_features<["+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>} {
    %0 = llvm.mlir.constant(0 : i64) : i64
    %1 = llvm.mlir.constant(1 : i32) : i32
    %2 = llvm.mlir.constant(0 : i32) : i32
    %3 = llvm.mlir.constant(1 : i64) : i64
    %4 = llvm.mlir.constant(-2 : i64) : i64
    %5 = llvm.icmp "eq" %arg1, %arg0 : !llvm.ptr
    llvm.cond_br %5, ^bb5, ^bb1
  ^bb1:  // pred: ^bb0
    %6 = llvm.getelementptr inbounds %arg0[%0, 1, 0] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.tutorial::AddressBook", packed (struct<"class.google::protobuf::Message", (struct<"class.google::protobuf::MessageLite", (ptr, struct<"class.google::protobuf::internal::InternalMetadata", (ptr)>)>)>, struct<"class.google::protobuf::RepeatedPtrField.18", (struct<"class.google::protobuf::internal::RepeatedPtrFieldBase", (ptr, i32, i32, ptr)>)>, struct<"class.google::protobuf::internal::CachedSize", (struct<"struct.std::atomic", (struct<"struct.std::__atomic_base", (i32)>)>)>, array<4 x i8>)>
    llvm.call @_ZN6google8protobuf8internal20RepeatedPtrFieldBase5ClearINS0_16RepeatedPtrFieldIN8tutorial6PersonEE11TypeHandlerEEEvv(%6) : (!llvm.ptr) -> ()
    %7 = llvm.getelementptr inbounds %arg0[%0, 0, 0, 1, 0] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.tutorial::AddressBook", packed (struct<"class.google::protobuf::Message", (struct<"class.google::protobuf::MessageLite", (ptr, struct<"class.google::protobuf::internal::InternalMetadata", (ptr)>)>)>, struct<"class.google::protobuf::RepeatedPtrField.18", (struct<"class.google::protobuf::internal::RepeatedPtrFieldBase", (ptr, i32, i32, ptr)>)>, struct<"class.google::protobuf::internal::CachedSize", (struct<"struct.std::atomic", (struct<"struct.std::__atomic_base", (i32)>)>)>, array<4 x i8>)>
    %8 = llvm.load %7 {alignment = 8 : i64, tbaa = [#tbaa_tag5]} : !llvm.ptr -> !llvm.ptr
    %9 = llvm.ptrtoint %8 : !llvm.ptr to i64
    %10 = llvm.and %9, %3  : i64
    %11 = llvm.icmp "eq" %10, %0 : i64
    llvm.cond_br %11, ^bb4, ^bb2
  ^bb2:  // pred: ^bb1
    %12 = llvm.and %9, %4  : i64
    %13 = llvm.inttoptr %12 : i64 to !llvm.ptr
    %14 = llvm.getelementptr inbounds %13[%0, 1] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.google::protobuf::internal::InternalMetadata::Container", (struct<"struct.google::protobuf::internal::InternalMetadata::ContainerBase", (ptr)>, struct<"class.google::protobuf::UnknownFieldSet", (struct<"class.std::vector", (struct<"struct.std::_Vector_base", (struct<"struct.std::_Vector_base<google::protobuf::UnknownField, std::allocator<google::protobuf::UnknownField>>::_Vector_impl", (struct<"struct.std::_Vector_base<google::protobuf::UnknownField, std::allocator<google::protobuf::UnknownField>>::_Vector_impl_data", (ptr, ptr, ptr)>)>)>)>)>)>
    %15 = llvm.getelementptr inbounds %14[%0, 0, 0, 0, 0, 0] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.google::protobuf::UnknownFieldSet", (struct<"class.std::vector", (struct<"struct.std::_Vector_base", (struct<"struct.std::_Vector_base<google::protobuf::UnknownField, std::allocator<google::protobuf::UnknownField>>::_Vector_impl", (struct<"struct.std::_Vector_base<google::protobuf::UnknownField, std::allocator<google::protobuf::UnknownField>>::_Vector_impl_data", (ptr, ptr, ptr)>)>)>)>)>
    %16 = llvm.load %15 {alignment = 8 : i64, tbaa = [#tbaa_tag2]} : !llvm.ptr -> !llvm.ptr
    %17 = llvm.getelementptr inbounds %13[%0, 1, 0, 0, 0, 0, 1] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.google::protobuf::internal::InternalMetadata::Container", (struct<"struct.google::protobuf::internal::InternalMetadata::ContainerBase", (ptr)>, struct<"class.google::protobuf::UnknownFieldSet", (struct<"class.std::vector", (struct<"struct.std::_Vector_base", (struct<"struct.std::_Vector_base<google::protobuf::UnknownField, std::allocator<google::protobuf::UnknownField>>::_Vector_impl", (struct<"struct.std::_Vector_base<google::protobuf::UnknownField, std::allocator<google::protobuf::UnknownField>>::_Vector_impl_data", (ptr, ptr, ptr)>)>)>)>)>)>
    %18 = llvm.load %17 {alignment = 8 : i64, tbaa = [#tbaa_tag2]} : !llvm.ptr -> !llvm.ptr
    %19 = llvm.icmp "eq" %16, %18 : !llvm.ptr
    llvm.cond_br %19, ^bb4, ^bb3
  ^bb3:  // pred: ^bb2
    llvm.call @_ZN6google8protobuf15UnknownFieldSet13ClearFallbackEv(%14) : (!llvm.ptr) -> ()
    llvm.br ^bb4
  ^bb4:  // 3 preds: ^bb1, ^bb2, ^bb3
    llvm.call @_ZN8tutorial11AddressBook9MergeFromERKS0_(%arg0, %arg1) : (!llvm.ptr, !llvm.ptr) -> ()
    llvm.br ^bb5
  ^bb5:  // 2 preds: ^bb0, ^bb4
    llvm.return
  }
  llvm.func unnamed_addr @_ZNK8tutorial11AddressBook13IsInitializedEv(%arg0: !llvm.ptr {llvm.align = 8 : i64, llvm.nocapture, llvm.nonnull, llvm.readnone}) -> (i1 {llvm.noundef, llvm.zeroext}) attributes {alignment = 2 : i64, frame_pointer = #llvm.framePointerKind<none>, memory = #llvm.memory_effects<other = none, argMem = none, inaccessibleMem = none>, passthrough = ["mustprogress", "nofree", "norecurse", "nosync", "nounwind", "willreturn", ["uwtable", "2"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], ["tune-cpu", "generic"]], target_cpu = "x86-64", target_features = #llvm.target_features<["+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>} {
    %0 = llvm.mlir.constant(true) : i1
    llvm.return %0 : i1
  }
  llvm.func local_unnamed_addr @_ZN8tutorial11AddressBook12InternalSwapEPS0_(%arg0: !llvm.ptr {llvm.align = 8 : i64, llvm.dereferenceable = 48 : i64, llvm.nonnull, llvm.noundef}, %arg1: !llvm.ptr {llvm.noundef}) attributes {alignment = 2 : i64, frame_pointer = #llvm.framePointerKind<none>, passthrough = [["uwtable", "2"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], ["tune-cpu", "generic"]], personality = @__gxx_personality_v0, target_cpu = "x86-64", target_features = #llvm.target_features<["+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>} {
    %0 = llvm.mlir.constant(0 : i64) : i64
    %1 = llvm.mlir.constant(0 : i32) : i32
    %2 = llvm.mlir.constant(1 : i32) : i32
    %3 = llvm.mlir.constant(1 : i64) : i64
    %4 = llvm.mlir.constant(false) : i1
    %5 = llvm.mlir.constant(-2 : i64) : i64
    %6 = llvm.mlir.constant(2 : i32) : i32
    %7 = llvm.getelementptr inbounds %arg0[%0, 0, 0, 1] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.tutorial::AddressBook", packed (struct<"class.google::protobuf::Message", (struct<"class.google::protobuf::MessageLite", (ptr, struct<"class.google::protobuf::internal::InternalMetadata", (ptr)>)>)>, struct<"class.google::protobuf::RepeatedPtrField.18", (struct<"class.google::protobuf::internal::RepeatedPtrFieldBase", (ptr, i32, i32, ptr)>)>, struct<"class.google::protobuf::internal::CachedSize", (struct<"struct.std::atomic", (struct<"struct.std::__atomic_base", (i32)>)>)>, array<4 x i8>)>
    %8 = llvm.getelementptr inbounds %arg1[%0, 0, 0, 1] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.tutorial::AddressBook", packed (struct<"class.google::protobuf::Message", (struct<"class.google::protobuf::MessageLite", (ptr, struct<"class.google::protobuf::internal::InternalMetadata", (ptr)>)>)>, struct<"class.google::protobuf::RepeatedPtrField.18", (struct<"class.google::protobuf::internal::RepeatedPtrFieldBase", (ptr, i32, i32, ptr)>)>, struct<"class.google::protobuf::internal::CachedSize", (struct<"struct.std::atomic", (struct<"struct.std::__atomic_base", (i32)>)>)>, array<4 x i8>)>
    %9 = llvm.getelementptr inbounds %7[%0, 0] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.google::protobuf::internal::InternalMetadata", (ptr)>
    %10 = llvm.load %9 {alignment = 8 : i64, tbaa = [#tbaa_tag5]} : !llvm.ptr -> !llvm.ptr
    %11 = llvm.ptrtoint %10 : !llvm.ptr to i64
    %12 = llvm.and %11, %3  : i64
    %13 = llvm.icmp "eq" %12, %0 : i64
    %14 = llvm.getelementptr inbounds %8[%0, 0] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.google::protobuf::internal::InternalMetadata", (ptr)>
    %15 = llvm.load %14 {alignment = 8 : i64, tbaa = [#tbaa_tag5]} : !llvm.ptr -> !llvm.ptr
    %16 = llvm.ptrtoint %15 : !llvm.ptr to i64
    %17 = llvm.and %16, %3  : i64
    %18 = llvm.icmp "eq" %17, %0 : i64
    %19 = llvm.select %13, %18, %4 : i1, i1
    llvm.cond_br %19, ^bb8, ^bb1
  ^bb1:  // pred: ^bb0
    llvm.cond_br %18 weights([1, 2000]), ^bb3, ^bb2
  ^bb2:  // pred: ^bb1
    %20 = llvm.and %16, %5  : i64
    %21 = llvm.inttoptr %20 : i64 to !llvm.ptr
    %22 = llvm.getelementptr inbounds %21[%0, 1] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.google::protobuf::internal::InternalMetadata::Container", (struct<"struct.google::protobuf::internal::InternalMetadata::ContainerBase", (ptr)>, struct<"class.google::protobuf::UnknownFieldSet", (struct<"class.std::vector", (struct<"struct.std::_Vector_base", (struct<"struct.std::_Vector_base<google::protobuf::UnknownField, std::allocator<google::protobuf::UnknownField>>::_Vector_impl", (struct<"struct.std::_Vector_base<google::protobuf::UnknownField, std::allocator<google::protobuf::UnknownField>>::_Vector_impl_data", (ptr, ptr, ptr)>)>)>)>)>)>
    llvm.br ^bb4(%11, %22 : i64, !llvm.ptr)
  ^bb3:  // pred: ^bb1
    %23 = llvm.call @_ZN6google8protobuf8internal16InternalMetadata27mutable_unknown_fields_slowINS0_15UnknownFieldSetEEEPT_v(%8) : (!llvm.ptr) -> !llvm.ptr
    %24 = llvm.load %9 {alignment = 8 : i64, tbaa = [#tbaa_tag5]} : !llvm.ptr -> !llvm.ptr
    %25 = llvm.ptrtoint %24 : !llvm.ptr to i64
    llvm.br ^bb4(%25, %23 : i64, !llvm.ptr)
  ^bb4(%26: i64, %27: !llvm.ptr):  // 2 preds: ^bb2, ^bb3
    %28 = llvm.and %26, %3  : i64
    %29 = llvm.icmp "eq" %28, %0 : i64
    llvm.cond_br %29 weights([1, 2000]), ^bb6, ^bb5
  ^bb5:  // pred: ^bb4
    %30 = llvm.and %26, %5  : i64
    %31 = llvm.inttoptr %30 : i64 to !llvm.ptr
    %32 = llvm.getelementptr inbounds %31[%0, 1] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.google::protobuf::internal::InternalMetadata::Container", (struct<"struct.google::protobuf::internal::InternalMetadata::ContainerBase", (ptr)>, struct<"class.google::protobuf::UnknownFieldSet", (struct<"class.std::vector", (struct<"struct.std::_Vector_base", (struct<"struct.std::_Vector_base<google::protobuf::UnknownField, std::allocator<google::protobuf::UnknownField>>::_Vector_impl", (struct<"struct.std::_Vector_base<google::protobuf::UnknownField, std::allocator<google::protobuf::UnknownField>>::_Vector_impl_data", (ptr, ptr, ptr)>)>)>)>)>)>
    llvm.br ^bb7(%32 : !llvm.ptr)
  ^bb6:  // pred: ^bb4
    %33 = llvm.call @_ZN6google8protobuf8internal16InternalMetadata27mutable_unknown_fields_slowINS0_15UnknownFieldSetEEEPT_v(%7) : (!llvm.ptr) -> !llvm.ptr
    llvm.br ^bb7(%33 : !llvm.ptr)
  ^bb7(%34: !llvm.ptr):  // 2 preds: ^bb5, ^bb6
    %35 = llvm.bitcast %34 : !llvm.ptr to !llvm.ptr
    %36 = llvm.load %35 {alignment = 8 : i64, tbaa = [#tbaa_tag2]} : !llvm.ptr -> !llvm.vec<2 x ptr>
    %37 = llvm.getelementptr inbounds %34[%0, 0, 0, 0, 0, 2] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.google::protobuf::UnknownFieldSet", (struct<"class.std::vector", (struct<"struct.std::_Vector_base", (struct<"struct.std::_Vector_base<google::protobuf::UnknownField, std::allocator<google::protobuf::UnknownField>>::_Vector_impl", (struct<"struct.std::_Vector_base<google::protobuf::UnknownField, std::allocator<google::protobuf::UnknownField>>::_Vector_impl_data", (ptr, ptr, ptr)>)>)>)>)>
    %38 = llvm.load %37 {alignment = 8 : i64, tbaa = [#tbaa_tag13]} : !llvm.ptr -> !llvm.ptr
    %39 = llvm.bitcast %27 : !llvm.ptr to !llvm.ptr
    %40 = llvm.load %39 {alignment = 8 : i64, tbaa = [#tbaa_tag2]} : !llvm.ptr -> !llvm.vec<2 x ptr>
    %41 = llvm.bitcast %34 : !llvm.ptr to !llvm.ptr
    llvm.store %40, %41 {alignment = 8 : i64, tbaa = [#tbaa_tag2]} : !llvm.vec<2 x ptr>, !llvm.ptr
    %42 = llvm.getelementptr inbounds %27[%0, 0, 0, 0, 0, 2] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.google::protobuf::UnknownFieldSet", (struct<"class.std::vector", (struct<"struct.std::_Vector_base", (struct<"struct.std::_Vector_base<google::protobuf::UnknownField, std::allocator<google::protobuf::UnknownField>>::_Vector_impl", (struct<"struct.std::_Vector_base<google::protobuf::UnknownField, std::allocator<google::protobuf::UnknownField>>::_Vector_impl_data", (ptr, ptr, ptr)>)>)>)>)>
    %43 = llvm.load %42 {alignment = 8 : i64, tbaa = [#tbaa_tag13]} : !llvm.ptr -> !llvm.ptr
    llvm.store %43, %37 {alignment = 8 : i64, tbaa = [#tbaa_tag13]} : !llvm.ptr, !llvm.ptr
    %44 = llvm.bitcast %27 : !llvm.ptr to !llvm.ptr
    llvm.store %36, %44 {alignment = 8 : i64, tbaa = [#tbaa_tag2]} : !llvm.vec<2 x ptr>, !llvm.ptr
    llvm.store %38, %42 {alignment = 8 : i64, tbaa = [#tbaa_tag13]} : !llvm.ptr, !llvm.ptr
    llvm.br ^bb8
  ^bb8:  // 2 preds: ^bb0, ^bb7
    %45 = llvm.getelementptr inbounds %arg0[%0, 1, 0] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.tutorial::AddressBook", packed (struct<"class.google::protobuf::Message", (struct<"class.google::protobuf::MessageLite", (ptr, struct<"class.google::protobuf::internal::InternalMetadata", (ptr)>)>)>, struct<"class.google::protobuf::RepeatedPtrField.18", (struct<"class.google::protobuf::internal::RepeatedPtrFieldBase", (ptr, i32, i32, ptr)>)>, struct<"class.google::protobuf::internal::CachedSize", (struct<"struct.std::atomic", (struct<"struct.std::__atomic_base", (i32)>)>)>, array<4 x i8>)>
    %46 = llvm.getelementptr inbounds %arg1[%0, 1, 0] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.tutorial::AddressBook", packed (struct<"class.google::protobuf::Message", (struct<"class.google::protobuf::MessageLite", (ptr, struct<"class.google::protobuf::internal::InternalMetadata", (ptr)>)>)>, struct<"class.google::protobuf::RepeatedPtrField.18", (struct<"class.google::protobuf::internal::RepeatedPtrFieldBase", (ptr, i32, i32, ptr)>)>, struct<"class.google::protobuf::internal::CachedSize", (struct<"struct.std::atomic", (struct<"struct.std::__atomic_base", (i32)>)>)>, array<4 x i8>)>
    llvm.call @_ZN6google8protobuf8internal20RepeatedPtrFieldBase12InternalSwapEPS2_(%45, %46) : (!llvm.ptr, !llvm.ptr) -> ()
    llvm.return
  }
  llvm.func unnamed_addr @_ZNK8tutorial11AddressBook11GetMetadataEv(%arg0: !llvm.ptr {llvm.align = 8 : i64, llvm.nocapture, llvm.nonnull, llvm.readnone}) -> !llvm.struct<(ptr, ptr)> attributes {alignment = 2 : i64, frame_pointer = #llvm.framePointerKind<none>, passthrough = ["mustprogress", ["uwtable", "2"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], ["tune-cpu", "generic"]], target_cpu = "x86-64", target_features = #llvm.target_features<["+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>} {
    %0 = llvm.mlir.zero : !llvm.ptr
    %1 = llvm.mlir.undef : !llvm.array<1 x ptr>
    %2 = llvm.insertvalue %0, %1[0] : !llvm.array<1 x ptr> 
    %3 = llvm.mlir.addressof @_ZL47file_level_enum_descriptors_addressbook_2eproto : !llvm.ptr
    %4 = llvm.mlir.constant(3 : i32) : i32
    %5 = llvm.mlir.undef : !llvm.struct<"struct.google::protobuf::Metadata", (ptr, ptr)>
    %6 = llvm.insertvalue %0, %5[0] : !llvm.struct<"struct.google::protobuf::Metadata", (ptr, ptr)> 
    %7 = llvm.insertvalue %0, %6[1] : !llvm.struct<"struct.google::protobuf::Metadata", (ptr, ptr)> 
    %8 = llvm.mlir.undef : !llvm.array<3 x struct<"struct.google::protobuf::Metadata", (ptr, ptr)>>
    %9 = llvm.insertvalue %7, %8[0] : !llvm.array<3 x struct<"struct.google::protobuf::Metadata", (ptr, ptr)>> 
    %10 = llvm.insertvalue %7, %9[1] : !llvm.array<3 x struct<"struct.google::protobuf::Metadata", (ptr, ptr)>> 
    %11 = llvm.insertvalue %7, %10[2] : !llvm.array<3 x struct<"struct.google::protobuf::Metadata", (ptr, ptr)>> 
    %12 = llvm.mlir.addressof @_ZL39file_level_metadata_addressbook_2eproto : !llvm.ptr
    %13 = llvm.mlir.constant(dense<[-1, 8, -1, -1, -1, 16, 24, -1, 8, -1, -1, -1, 40, 64, 48, 16, 56, -1, 8, -1, -1, -1, 16]> : tensor<23xi32>) : !llvm.array<23 x i32>
    %14 = llvm.mlir.addressof @_ZN31TableStruct_addressbook_2eproto7offsetsE : !llvm.ptr
    %15 = llvm.mlir.constant(0 : i8) : i8
    %16 = llvm.mlir.constant(dense<0> : tensor<40xi8>) : !llvm.array<40 x i8>
    %17 = llvm.mlir.constant(0 : i64) : i64
    %18 = llvm.mlir.undef : !llvm.struct<"union.google::protobuf::internal::ExplicitlyConstructed<tutorial::AddressBook>::AlignedUnion", (i64, array<40 x i8>)>
    %19 = llvm.insertvalue %17, %18[0] : !llvm.struct<"union.google::protobuf::internal::ExplicitlyConstructed<tutorial::AddressBook>::AlignedUnion", (i64, array<40 x i8>)> 
    %20 = llvm.insertvalue %16, %19[1] : !llvm.struct<"union.google::protobuf::internal::ExplicitlyConstructed<tutorial::AddressBook>::AlignedUnion", (i64, array<40 x i8>)> 
    %21 = llvm.mlir.undef : !llvm.struct<"class.google::protobuf::internal::ExplicitlyConstructed.1", (struct<"union.google::protobuf::internal::ExplicitlyConstructed<tutorial::AddressBook>::AlignedUnion", (i64, array<40 x i8>)>)>
    %22 = llvm.insertvalue %20, %21[0] : !llvm.struct<"class.google::protobuf::internal::ExplicitlyConstructed.1", (struct<"union.google::protobuf::internal::ExplicitlyConstructed<tutorial::AddressBook>::AlignedUnion", (i64, array<40 x i8>)>)> 
    %23 = llvm.mlir.undef : !llvm.struct<"class.tutorial::AddressBookDefaultTypeInternal", (struct<"class.google::protobuf::internal::ExplicitlyConstructed.1", (struct<"union.google::protobuf::internal::ExplicitlyConstructed<tutorial::AddressBook>::AlignedUnion", (i64, array<40 x i8>)>)>)>
    %24 = llvm.insertvalue %22, %23[0] : !llvm.struct<"class.tutorial::AddressBookDefaultTypeInternal", (struct<"class.google::protobuf::internal::ExplicitlyConstructed.1", (struct<"union.google::protobuf::internal::ExplicitlyConstructed<tutorial::AddressBook>::AlignedUnion", (i64, array<40 x i8>)>)>)> 
    %25 = llvm.mlir.addressof @_ZN8tutorial30_AddressBook_default_instance_E : !llvm.ptr
    %26 = llvm.mlir.constant(dense<0> : tensor<64xi8>) : !llvm.array<64 x i8>
    %27 = llvm.mlir.undef : !llvm.struct<"union.google::protobuf::internal::ExplicitlyConstructed<tutorial::Person>::AlignedUnion", (i64, array<64 x i8>)>
    %28 = llvm.insertvalue %17, %27[0] : !llvm.struct<"union.google::protobuf::internal::ExplicitlyConstructed<tutorial::Person>::AlignedUnion", (i64, array<64 x i8>)> 
    %29 = llvm.insertvalue %26, %28[1] : !llvm.struct<"union.google::protobuf::internal::ExplicitlyConstructed<tutorial::Person>::AlignedUnion", (i64, array<64 x i8>)> 
    %30 = llvm.mlir.undef : !llvm.struct<"class.google::protobuf::internal::ExplicitlyConstructed.0", (struct<"union.google::protobuf::internal::ExplicitlyConstructed<tutorial::Person>::AlignedUnion", (i64, array<64 x i8>)>)>
    %31 = llvm.insertvalue %29, %30[0] : !llvm.struct<"class.google::protobuf::internal::ExplicitlyConstructed.0", (struct<"union.google::protobuf::internal::ExplicitlyConstructed<tutorial::Person>::AlignedUnion", (i64, array<64 x i8>)>)> 
    %32 = llvm.mlir.undef : !llvm.struct<"class.tutorial::PersonDefaultTypeInternal", (struct<"class.google::protobuf::internal::ExplicitlyConstructed.0", (struct<"union.google::protobuf::internal::ExplicitlyConstructed<tutorial::Person>::AlignedUnion", (i64, array<64 x i8>)>)>)>
    %33 = llvm.insertvalue %31, %32[0] : !llvm.struct<"class.tutorial::PersonDefaultTypeInternal", (struct<"class.google::protobuf::internal::ExplicitlyConstructed.0", (struct<"union.google::protobuf::internal::ExplicitlyConstructed<tutorial::Person>::AlignedUnion", (i64, array<64 x i8>)>)>)> 
    %34 = llvm.mlir.addressof @_ZN8tutorial25_Person_default_instance_E : !llvm.ptr
    %35 = llvm.mlir.constant(dense<0> : tensor<24xi8>) : !llvm.array<24 x i8>
    %36 = llvm.mlir.undef : !llvm.struct<"union.google::protobuf::internal::ExplicitlyConstructed<tutorial::Person_PhoneNumber>::AlignedUnion", (i64, array<24 x i8>)>
    %37 = llvm.insertvalue %17, %36[0] : !llvm.struct<"union.google::protobuf::internal::ExplicitlyConstructed<tutorial::Person_PhoneNumber>::AlignedUnion", (i64, array<24 x i8>)> 
    %38 = llvm.insertvalue %35, %37[1] : !llvm.struct<"union.google::protobuf::internal::ExplicitlyConstructed<tutorial::Person_PhoneNumber>::AlignedUnion", (i64, array<24 x i8>)> 
    %39 = llvm.mlir.undef : !llvm.struct<"class.google::protobuf::internal::ExplicitlyConstructed", (struct<"union.google::protobuf::internal::ExplicitlyConstructed<tutorial::Person_PhoneNumber>::AlignedUnion", (i64, array<24 x i8>)>)>
    %40 = llvm.insertvalue %38, %39[0] : !llvm.struct<"class.google::protobuf::internal::ExplicitlyConstructed", (struct<"union.google::protobuf::internal::ExplicitlyConstructed<tutorial::Person_PhoneNumber>::AlignedUnion", (i64, array<24 x i8>)>)> 
    %41 = llvm.mlir.undef : !llvm.struct<"class.tutorial::Person_PhoneNumberDefaultTypeInternal", (struct<"class.google::protobuf::internal::ExplicitlyConstructed", (struct<"union.google::protobuf::internal::ExplicitlyConstructed<tutorial::Person_PhoneNumber>::AlignedUnion", (i64, array<24 x i8>)>)>)>
    %42 = llvm.insertvalue %40, %41[0] : !llvm.struct<"class.tutorial::Person_PhoneNumberDefaultTypeInternal", (struct<"class.google::protobuf::internal::ExplicitlyConstructed", (struct<"union.google::protobuf::internal::ExplicitlyConstructed<tutorial::Person_PhoneNumber>::AlignedUnion", (i64, array<24 x i8>)>)>)> 
    %43 = llvm.mlir.addressof @_ZN8tutorial37_Person_PhoneNumber_default_instance_E : !llvm.ptr
    %44 = llvm.mlir.undef : !llvm.array<3 x ptr>
    %45 = llvm.insertvalue %43, %44[0] : !llvm.array<3 x ptr> 
    %46 = llvm.insertvalue %34, %45[1] : !llvm.array<3 x ptr> 
    %47 = llvm.insertvalue %25, %46[2] : !llvm.array<3 x ptr> 
    %48 = llvm.mlir.addressof @_ZL22file_default_instances : !llvm.ptr
    %49 = llvm.mlir.constant(48 : i32) : i32
    %50 = llvm.mlir.constant(-1 : i32) : i32
    %51 = llvm.mlir.constant(17 : i32) : i32
    %52 = llvm.mlir.undef : !llvm.struct<"struct.google::protobuf::internal::MigrationSchema", (i32, i32, i32)>
    %53 = llvm.insertvalue %51, %52[0] : !llvm.struct<"struct.google::protobuf::internal::MigrationSchema", (i32, i32, i32)> 
    %54 = llvm.insertvalue %50, %53[1] : !llvm.struct<"struct.google::protobuf::internal::MigrationSchema", (i32, i32, i32)> 
    %55 = llvm.insertvalue %49, %54[2] : !llvm.struct<"struct.google::protobuf::internal::MigrationSchema", (i32, i32, i32)> 
    %56 = llvm.mlir.constant(72 : i32) : i32
    %57 = llvm.mlir.constant(7 : i32) : i32
    %58 = llvm.mlir.undef : !llvm.struct<"struct.google::protobuf::internal::MigrationSchema", (i32, i32, i32)>
    %59 = llvm.insertvalue %57, %58[0] : !llvm.struct<"struct.google::protobuf::internal::MigrationSchema", (i32, i32, i32)> 
    %60 = llvm.insertvalue %50, %59[1] : !llvm.struct<"struct.google::protobuf::internal::MigrationSchema", (i32, i32, i32)> 
    %61 = llvm.insertvalue %56, %60[2] : !llvm.struct<"struct.google::protobuf::internal::MigrationSchema", (i32, i32, i32)> 
    %62 = llvm.mlir.constant(32 : i32) : i32
    %63 = llvm.mlir.constant(0 : i32) : i32
    %64 = llvm.mlir.undef : !llvm.struct<"struct.google::protobuf::internal::MigrationSchema", (i32, i32, i32)>
    %65 = llvm.insertvalue %63, %64[0] : !llvm.struct<"struct.google::protobuf::internal::MigrationSchema", (i32, i32, i32)> 
    %66 = llvm.insertvalue %50, %65[1] : !llvm.struct<"struct.google::protobuf::internal::MigrationSchema", (i32, i32, i32)> 
    %67 = llvm.insertvalue %62, %66[2] : !llvm.struct<"struct.google::protobuf::internal::MigrationSchema", (i32, i32, i32)> 
    %68 = llvm.mlir.undef : !llvm.array<3 x struct<"struct.google::protobuf::internal::MigrationSchema", (i32, i32, i32)>>
    %69 = llvm.insertvalue %67, %68[0] : !llvm.array<3 x struct<"struct.google::protobuf::internal::MigrationSchema", (i32, i32, i32)>> 
    %70 = llvm.insertvalue %61, %69[1] : !llvm.array<3 x struct<"struct.google::protobuf::internal::MigrationSchema", (i32, i32, i32)>> 
    %71 = llvm.insertvalue %55, %70[2] : !llvm.array<3 x struct<"struct.google::protobuf::internal::MigrationSchema", (i32, i32, i32)>> 
    %72 = llvm.mlir.addressof @_ZL7schemas : !llvm.ptr
    %73 = llvm.mlir.constant(1 : i32) : i32
    %74 = llvm.mlir.addressof @descriptor_table_google_2fprotobuf_2ftimestamp_2eproto : !llvm.ptr
    %75 = llvm.mlir.undef : !llvm.array<1 x ptr>
    %76 = llvm.insertvalue %74, %75[0] : !llvm.array<1 x ptr> 
    %77 = llvm.mlir.addressof @_ZL41descriptor_table_addressbook_2eproto_deps : !llvm.ptr
    %78 = llvm.mlir.undef : !llvm.array<0 x ptr>
    %79 = llvm.mlir.addressof @__gxx_personality_v0 : !llvm.ptr
    %80 = llvm.mlir.addressof @_ZL59InitDefaultsscc_info_Person_PhoneNumber_addressbook_2eprotov : !llvm.ptr
    %81 = llvm.mlir.undef : !llvm.struct<(i32)>
    %82 = llvm.insertvalue %50, %81[0] : !llvm.struct<(i32)> 
    %83 = llvm.mlir.undef : !llvm.struct<(struct<(i32)>, i32, i32, ptr)>
    %84 = llvm.insertvalue %82, %83[0] : !llvm.struct<(struct<(i32)>, i32, i32, ptr)> 
    %85 = llvm.insertvalue %63, %84[1] : !llvm.struct<(struct<(i32)>, i32, i32, ptr)> 
    %86 = llvm.insertvalue %63, %85[2] : !llvm.struct<(struct<(i32)>, i32, i32, ptr)> 
    %87 = llvm.insertvalue %80, %86[3] : !llvm.struct<(struct<(i32)>, i32, i32, ptr)> 
    %88 = llvm.mlir.undef : !llvm.struct<(struct<(struct<(i32)>, i32, i32, ptr)>, array<0 x ptr>)>
    %89 = llvm.insertvalue %87, %88[0] : !llvm.struct<(struct<(struct<(i32)>, i32, i32, ptr)>, array<0 x ptr>)> 
    %90 = llvm.insertvalue %78, %89[1] : !llvm.struct<(struct<(struct<(i32)>, i32, i32, ptr)>, array<0 x ptr>)> 
    %91 = llvm.mlir.addressof @scc_info_Person_PhoneNumber_addressbook_2eproto : !llvm.ptr
    %92 = llvm.mlir.addressof @scc_info_Timestamp_google_2fprotobuf_2ftimestamp_2eproto : !llvm.ptr
    %93 = llvm.mlir.undef : !llvm.array<2 x ptr>
    %94 = llvm.insertvalue %91, %93[0] : !llvm.array<2 x ptr> 
    %95 = llvm.insertvalue %92, %94[1] : !llvm.array<2 x ptr> 
    %96 = llvm.mlir.addressof @_ZL47InitDefaultsscc_info_Person_addressbook_2eprotov : !llvm.ptr
    %97 = llvm.mlir.constant(2 : i32) : i32
    %98 = llvm.mlir.undef : !llvm.struct<(struct<(i32)>, i32, i32, ptr)>
    %99 = llvm.insertvalue %82, %98[0] : !llvm.struct<(struct<(i32)>, i32, i32, ptr)> 
    %100 = llvm.insertvalue %97, %99[1] : !llvm.struct<(struct<(i32)>, i32, i32, ptr)> 
    %101 = llvm.insertvalue %63, %100[2] : !llvm.struct<(struct<(i32)>, i32, i32, ptr)> 
    %102 = llvm.insertvalue %96, %101[3] : !llvm.struct<(struct<(i32)>, i32, i32, ptr)> 
    %103 = llvm.mlir.undef : !llvm.struct<(struct<(struct<(i32)>, i32, i32, ptr)>, array<2 x ptr>)>
    %104 = llvm.insertvalue %102, %103[0] : !llvm.struct<(struct<(struct<(i32)>, i32, i32, ptr)>, array<2 x ptr>)> 
    %105 = llvm.insertvalue %95, %104[1] : !llvm.struct<(struct<(struct<(i32)>, i32, i32, ptr)>, array<2 x ptr>)> 
    %106 = llvm.mlir.addressof @scc_info_Person_addressbook_2eproto : !llvm.ptr
    %107 = llvm.mlir.undef : !llvm.array<1 x ptr>
    %108 = llvm.insertvalue %106, %107[0] : !llvm.array<1 x ptr> 
    %109 = llvm.mlir.addressof @_ZL52InitDefaultsscc_info_AddressBook_addressbook_2eprotov : !llvm.ptr
    %110 = llvm.mlir.undef : !llvm.struct<(struct<(i32)>, i32, i32, ptr)>
    %111 = llvm.insertvalue %82, %110[0] : !llvm.struct<(struct<(i32)>, i32, i32, ptr)> 
    %112 = llvm.insertvalue %73, %111[1] : !llvm.struct<(struct<(i32)>, i32, i32, ptr)> 
    %113 = llvm.insertvalue %63, %112[2] : !llvm.struct<(struct<(i32)>, i32, i32, ptr)> 
    %114 = llvm.insertvalue %109, %113[3] : !llvm.struct<(struct<(i32)>, i32, i32, ptr)> 
    %115 = llvm.mlir.undef : !llvm.struct<(struct<(struct<(i32)>, i32, i32, ptr)>, array<1 x ptr>)>
    %116 = llvm.insertvalue %114, %115[0] : !llvm.struct<(struct<(struct<(i32)>, i32, i32, ptr)>, array<1 x ptr>)> 
    %117 = llvm.insertvalue %108, %116[1] : !llvm.struct<(struct<(struct<(i32)>, i32, i32, ptr)>, array<1 x ptr>)> 
    %118 = llvm.mlir.addressof @scc_info_AddressBook_addressbook_2eproto : !llvm.ptr
    %119 = llvm.mlir.undef : !llvm.array<3 x ptr>
    %120 = llvm.insertvalue %118, %119[0] : !llvm.array<3 x ptr> 
    %121 = llvm.insertvalue %106, %120[1] : !llvm.array<3 x ptr> 
    %122 = llvm.insertvalue %91, %121[2] : !llvm.array<3 x ptr> 
    %123 = llvm.mlir.addressof @_ZL41descriptor_table_addressbook_2eproto_sccs : !llvm.ptr
    %124 = llvm.mlir.undef : !llvm.struct<"struct.std::once_flag", (i32)>
    %125 = llvm.insertvalue %63, %124[0] : !llvm.struct<"struct.std::once_flag", (i32)> 
    %126 = llvm.mlir.addressof @_ZL41descriptor_table_addressbook_2eproto_once : !llvm.ptr
    %127 = llvm.mlir.constant(537 : i32) : i32
    %128 = llvm.mlir.constant("addressbook.proto\00") : !llvm.array<18 x i8>
    %129 = llvm.mlir.addressof @".str" : !llvm.ptr
    %130 = llvm.mlir.constant("\0A\11addressbook.proto\12\08tutorial\1A\1Fgoogle/protobuf/timestamp.proto\22\87\02\0A\06Person\12\0C\0A\04name\18\01 \01(\09\12\0A\0A\02id\18\02 \01(\05\12\0D\0A\05email\18\03 \01(\09\12,\0A\06phones\18\04 \03(\0B2\1C.tutorial.Person.PhoneNumber\120\0A\0Clast_updated\18\05 \01(\0B2\1A.google.protobuf.Timestamp\1AG\0A\0BPhoneNumber\12\0E\0A\06number\18\01 \01(\09\12(\0A\04type\18\02 \01(\0E2\1A.tutorial.Person.PhoneType\22+\0A\09PhoneType\12\0A\0A\06MOBILE\10\00\12\08\0A\04HOME\10\01\12\08\0A\04WORK\10\02\22/\0A\0BAddressBook\12 \0A\06people\18\01 \03(\0B2\10.tutorial.PersonB\95\01\0A\1Bcom.example.tutorial.protosB\11AddressBookProtosP\01Z:github.com/protocolbuffers/protobuf/examples/go/tutorialpb\AA\02$Google.Protobuf.Examples.AddressBookb\06proto3\00") : !llvm.array<538 x i8>
    %131 = llvm.mlir.addressof @_ZL45descriptor_table_protodef_addressbook_2eproto : !llvm.ptr
    %132 = llvm.mlir.undef : !llvm.struct<"struct.google::protobuf::internal::DescriptorTable", (i8, i8, ptr, ptr, i32, ptr, ptr, ptr, i32, i32, ptr, ptr, ptr, ptr, i32, ptr, ptr)>
    %133 = llvm.insertvalue %15, %132[0] : !llvm.struct<"struct.google::protobuf::internal::DescriptorTable", (i8, i8, ptr, ptr, i32, ptr, ptr, ptr, i32, i32, ptr, ptr, ptr, ptr, i32, ptr, ptr)> 
    %134 = llvm.insertvalue %15, %133[1] : !llvm.struct<"struct.google::protobuf::internal::DescriptorTable", (i8, i8, ptr, ptr, i32, ptr, ptr, ptr, i32, i32, ptr, ptr, ptr, ptr, i32, ptr, ptr)> 
    %135 = llvm.insertvalue %131, %134[2] : !llvm.struct<"struct.google::protobuf::internal::DescriptorTable", (i8, i8, ptr, ptr, i32, ptr, ptr, ptr, i32, i32, ptr, ptr, ptr, ptr, i32, ptr, ptr)> 
    %136 = llvm.insertvalue %129, %135[3] : !llvm.struct<"struct.google::protobuf::internal::DescriptorTable", (i8, i8, ptr, ptr, i32, ptr, ptr, ptr, i32, i32, ptr, ptr, ptr, ptr, i32, ptr, ptr)> 
    %137 = llvm.insertvalue %127, %136[4] : !llvm.struct<"struct.google::protobuf::internal::DescriptorTable", (i8, i8, ptr, ptr, i32, ptr, ptr, ptr, i32, i32, ptr, ptr, ptr, ptr, i32, ptr, ptr)> 
    %138 = llvm.insertvalue %126, %137[5] : !llvm.struct<"struct.google::protobuf::internal::DescriptorTable", (i8, i8, ptr, ptr, i32, ptr, ptr, ptr, i32, i32, ptr, ptr, ptr, ptr, i32, ptr, ptr)> 
    %139 = llvm.insertvalue %123, %138[6] : !llvm.struct<"struct.google::protobuf::internal::DescriptorTable", (i8, i8, ptr, ptr, i32, ptr, ptr, ptr, i32, i32, ptr, ptr, ptr, ptr, i32, ptr, ptr)> 
    %140 = llvm.insertvalue %77, %139[7] : !llvm.struct<"struct.google::protobuf::internal::DescriptorTable", (i8, i8, ptr, ptr, i32, ptr, ptr, ptr, i32, i32, ptr, ptr, ptr, ptr, i32, ptr, ptr)> 
    %141 = llvm.insertvalue %4, %140[8] : !llvm.struct<"struct.google::protobuf::internal::DescriptorTable", (i8, i8, ptr, ptr, i32, ptr, ptr, ptr, i32, i32, ptr, ptr, ptr, ptr, i32, ptr, ptr)> 
    %142 = llvm.insertvalue %73, %141[9] : !llvm.struct<"struct.google::protobuf::internal::DescriptorTable", (i8, i8, ptr, ptr, i32, ptr, ptr, ptr, i32, i32, ptr, ptr, ptr, ptr, i32, ptr, ptr)> 
    %143 = llvm.insertvalue %72, %142[10] : !llvm.struct<"struct.google::protobuf::internal::DescriptorTable", (i8, i8, ptr, ptr, i32, ptr, ptr, ptr, i32, i32, ptr, ptr, ptr, ptr, i32, ptr, ptr)> 
    %144 = llvm.insertvalue %48, %143[11] : !llvm.struct<"struct.google::protobuf::internal::DescriptorTable", (i8, i8, ptr, ptr, i32, ptr, ptr, ptr, i32, i32, ptr, ptr, ptr, ptr, i32, ptr, ptr)> 
    %145 = llvm.insertvalue %14, %144[12] : !llvm.struct<"struct.google::protobuf::internal::DescriptorTable", (i8, i8, ptr, ptr, i32, ptr, ptr, ptr, i32, i32, ptr, ptr, ptr, ptr, i32, ptr, ptr)> 
    %146 = llvm.insertvalue %12, %145[13] : !llvm.struct<"struct.google::protobuf::internal::DescriptorTable", (i8, i8, ptr, ptr, i32, ptr, ptr, ptr, i32, i32, ptr, ptr, ptr, ptr, i32, ptr, ptr)> 
    %147 = llvm.insertvalue %4, %146[14] : !llvm.struct<"struct.google::protobuf::internal::DescriptorTable", (i8, i8, ptr, ptr, i32, ptr, ptr, ptr, i32, i32, ptr, ptr, ptr, ptr, i32, ptr, ptr)> 
    %148 = llvm.insertvalue %3, %147[15] : !llvm.struct<"struct.google::protobuf::internal::DescriptorTable", (i8, i8, ptr, ptr, i32, ptr, ptr, ptr, i32, i32, ptr, ptr, ptr, ptr, i32, ptr, ptr)> 
    %149 = llvm.insertvalue %0, %148[16] : !llvm.struct<"struct.google::protobuf::internal::DescriptorTable", (i8, i8, ptr, ptr, i32, ptr, ptr, ptr, i32, i32, ptr, ptr, ptr, ptr, i32, ptr, ptr)> 
    %150 = llvm.mlir.addressof @descriptor_table_addressbook_2eproto : !llvm.ptr
    %151 = llvm.mlir.constant(false) : i1
    %152 = llvm.mlir.constant(13 : i32) : i32
    %153 = llvm.getelementptr inbounds %150[%17, 13] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.google::protobuf::internal::DescriptorTable", (i8, i8, ptr, ptr, i32, ptr, ptr, ptr, i32, i32, ptr, ptr, ptr, ptr, i32, ptr, ptr)>
    %154 = llvm.mlir.constant(2 : i64) : i64
    %155 = llvm.mlir.poison : !llvm.struct<(ptr, ptr)>
    llvm.call @_ZN6google8protobuf8internal17AssignDescriptorsEPKNS1_15DescriptorTableEb(%150, %151) : (!llvm.ptr, i1) -> ()
    %156 = llvm.load %153 {alignment = 8 : i64, tbaa = [#tbaa_tag14]} : !llvm.ptr -> !llvm.ptr
    %157 = llvm.getelementptr inbounds %156[%154, 0] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.google::protobuf::Metadata", (ptr, ptr)>
    %158 = llvm.load %157 {alignment = 8 : i64} : !llvm.ptr -> !llvm.ptr
    %159 = llvm.getelementptr inbounds %156[%154, 1] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.google::protobuf::Metadata", (ptr, ptr)>
    %160 = llvm.load %159 {alignment = 8 : i64} : !llvm.ptr -> !llvm.ptr
    %161 = llvm.insertvalue %158, %155[0] : !llvm.struct<(ptr, ptr)> 
    %162 = llvm.insertvalue %160, %161[1] : !llvm.struct<(ptr, ptr)> 
    llvm.return %162 : !llvm.struct<(ptr, ptr)>
  }
  llvm.func local_unnamed_addr @_ZN6google8protobuf5Arena18CreateMaybeMessageIN8tutorial18Person_PhoneNumberEJEEEPT_PS1_DpOT0_(%arg0: !llvm.ptr {llvm.noundef}) -> (!llvm.ptr {llvm.noundef}) attributes {alignment = 2 : i64, frame_pointer = #llvm.framePointerKind<none>, passthrough = ["noinline", ["uwtable", "2"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], ["tune-cpu", "generic"]], personality = @__gxx_personality_v0, target_cpu = "x86-64", target_features = #llvm.target_features<["+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>} {
    %0 = llvm.mlir.zero : !llvm.ptr
    %1 = llvm.mlir.constant(0 : i64) : i64
    %2 = llvm.mlir.constant(4 : i32) : i32
    %3 = llvm.mlir.addressof @_ZTIN6google8protobuf7MessageE : !llvm.ptr
    %4 = llvm.mlir.constant("N8tutorial18Person_PhoneNumberE\00") : !llvm.array<32 x i8>
    %5 = llvm.mlir.addressof @_ZTSN8tutorial18Person_PhoneNumberE : !llvm.ptr
    %6 = llvm.mlir.constant(2 : i64) : i64
    %7 = llvm.mlir.addressof @_ZTVN10__cxxabiv120__si_class_type_infoE : !llvm.ptr
    %8 = llvm.getelementptr inbounds %7[%6] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.ptr
    %9 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, ptr)>
    %10 = llvm.insertvalue %8, %9[0] : !llvm.struct<(ptr, ptr, ptr)> 
    %11 = llvm.insertvalue %5, %10[1] : !llvm.struct<(ptr, ptr, ptr)> 
    %12 = llvm.insertvalue %3, %11[2] : !llvm.struct<(ptr, ptr, ptr)> 
    %13 = llvm.mlir.addressof @_ZTIN8tutorial18Person_PhoneNumberE : !llvm.ptr
    %14 = llvm.mlir.constant(32 : i64) : i64
    %15 = llvm.mlir.constant(8 : i64) : i64
    %16 = llvm.mlir.constant(0 : i32) : i32
    %17 = llvm.mlir.addressof @_ZNK8tutorial18Person_PhoneNumber11GetMetadataEv : !llvm.ptr
    %18 = llvm.mlir.addressof @__gxx_personality_v0 : !llvm.ptr
    %19 = llvm.mlir.addressof @_ZNK8tutorial18Person_PhoneNumber13SetCachedSizeEi : !llvm.ptr
    %20 = llvm.mlir.addressof @_ZNK6google8protobuf7Message13SpaceUsedLongEv : !llvm.ptr
    %21 = llvm.mlir.addressof @_ZN6google8protobuf7Message20DiscardUnknownFieldsEv : !llvm.ptr
    %22 = llvm.mlir.addressof @_ZN8tutorial18Person_PhoneNumber9MergeFromERKN6google8protobuf7MessageE : !llvm.ptr
    %23 = llvm.mlir.addressof @_ZN8tutorial18Person_PhoneNumber8CopyFromERKN6google8protobuf7MessageE : !llvm.ptr
    %24 = llvm.mlir.addressof @_ZNK6google8protobuf11MessageLite16InternalGetTableEv : !llvm.ptr
    %25 = llvm.mlir.addressof @_ZNK8tutorial18Person_PhoneNumber18_InternalSerializeEPhPN6google8protobuf2io19EpsCopyOutputStreamE : !llvm.ptr
    %26 = llvm.mlir.addressof @_ZN8tutorial18Person_PhoneNumber14_InternalParseEPKcPN6google8protobuf8internal12ParseContextE : !llvm.ptr
    %27 = llvm.mlir.addressof @_ZNK8tutorial18Person_PhoneNumber13GetCachedSizeEv : !llvm.ptr
    %28 = llvm.mlir.addressof @_ZNK8tutorial18Person_PhoneNumber12ByteSizeLongEv : !llvm.ptr
    %29 = llvm.mlir.addressof @_ZN6google8protobuf7Message21CheckTypeAndMergeFromERKNS0_11MessageLiteE : !llvm.ptr
    %30 = llvm.mlir.addressof @_ZNK6google8protobuf7Message25InitializationErrorStringB5cxx11Ev : !llvm.ptr
    %31 = llvm.mlir.addressof @_ZNK8tutorial18Person_PhoneNumber13IsInitializedEv : !llvm.ptr
    %32 = llvm.mlir.addressof @_ZN8tutorial18Person_PhoneNumber5ClearEv : !llvm.ptr
    %33 = llvm.mlir.addressof @_ZNK8tutorial18Person_PhoneNumber3NewEPN6google8protobuf5ArenaE : !llvm.ptr
    %34 = llvm.mlir.addressof @_ZNK8tutorial18Person_PhoneNumber3NewEv : !llvm.ptr
    %35 = llvm.mlir.addressof @_ZNK6google8protobuf7Message11GetTypeNameB5cxx11Ev : !llvm.ptr
    %36 = llvm.mlir.addressof @_ZN8tutorial18Person_PhoneNumberD0Ev : !llvm.ptr
    %37 = llvm.mlir.addressof @_ZN8tutorial18Person_PhoneNumberD2Ev : !llvm.ptr
    %38 = llvm.mlir.undef : !llvm.array<22 x ptr>
    %39 = llvm.insertvalue %0, %38[0] : !llvm.array<22 x ptr> 
    %40 = llvm.insertvalue %13, %39[1] : !llvm.array<22 x ptr> 
    %41 = llvm.insertvalue %37, %40[2] : !llvm.array<22 x ptr> 
    %42 = llvm.insertvalue %36, %41[3] : !llvm.array<22 x ptr> 
    %43 = llvm.insertvalue %35, %42[4] : !llvm.array<22 x ptr> 
    %44 = llvm.insertvalue %34, %43[5] : !llvm.array<22 x ptr> 
    %45 = llvm.insertvalue %33, %44[6] : !llvm.array<22 x ptr> 
    %46 = llvm.insertvalue %32, %45[7] : !llvm.array<22 x ptr> 
    %47 = llvm.insertvalue %31, %46[8] : !llvm.array<22 x ptr> 
    %48 = llvm.insertvalue %30, %47[9] : !llvm.array<22 x ptr> 
    %49 = llvm.insertvalue %29, %48[10] : !llvm.array<22 x ptr> 
    %50 = llvm.insertvalue %28, %49[11] : !llvm.array<22 x ptr> 
    %51 = llvm.insertvalue %27, %50[12] : !llvm.array<22 x ptr> 
    %52 = llvm.insertvalue %26, %51[13] : !llvm.array<22 x ptr> 
    %53 = llvm.insertvalue %25, %52[14] : !llvm.array<22 x ptr> 
    %54 = llvm.insertvalue %24, %53[15] : !llvm.array<22 x ptr> 
    %55 = llvm.insertvalue %23, %54[16] : !llvm.array<22 x ptr> 
    %56 = llvm.insertvalue %22, %55[17] : !llvm.array<22 x ptr> 
    %57 = llvm.insertvalue %21, %56[18] : !llvm.array<22 x ptr> 
    %58 = llvm.insertvalue %20, %57[19] : !llvm.array<22 x ptr> 
    %59 = llvm.insertvalue %19, %58[20] : !llvm.array<22 x ptr> 
    %60 = llvm.insertvalue %17, %59[21] : !llvm.array<22 x ptr> 
    %61 = llvm.mlir.undef : !llvm.struct<(array<22 x ptr>)>
    %62 = llvm.insertvalue %60, %61[0] : !llvm.struct<(array<22 x ptr>)> 
    %63 = llvm.mlir.addressof @_ZTVN8tutorial18Person_PhoneNumberE : !llvm.ptr
    %64 = llvm.getelementptr inbounds %63[%1, 0, %6] : (!llvm.ptr, i64, i64) -> !llvm.ptr, !llvm.struct<(array<22 x ptr>)>
    %65 = llvm.mlir.constant(28 : i64) : i64
    %66 = llvm.mlir.undef : !llvm.array<0 x ptr>
    %67 = llvm.mlir.addressof @_ZL59InitDefaultsscc_info_Person_PhoneNumber_addressbook_2eprotov : !llvm.ptr
    %68 = llvm.mlir.constant(-1 : i32) : i32
    %69 = llvm.mlir.undef : !llvm.struct<(i32)>
    %70 = llvm.insertvalue %68, %69[0] : !llvm.struct<(i32)> 
    %71 = llvm.mlir.undef : !llvm.struct<(struct<(i32)>, i32, i32, ptr)>
    %72 = llvm.insertvalue %70, %71[0] : !llvm.struct<(struct<(i32)>, i32, i32, ptr)> 
    %73 = llvm.insertvalue %16, %72[1] : !llvm.struct<(struct<(i32)>, i32, i32, ptr)> 
    %74 = llvm.insertvalue %16, %73[2] : !llvm.struct<(struct<(i32)>, i32, i32, ptr)> 
    %75 = llvm.insertvalue %67, %74[3] : !llvm.struct<(struct<(i32)>, i32, i32, ptr)> 
    %76 = llvm.mlir.undef : !llvm.struct<(struct<(struct<(i32)>, i32, i32, ptr)>, array<0 x ptr>)>
    %77 = llvm.insertvalue %75, %76[0] : !llvm.struct<(struct<(struct<(i32)>, i32, i32, ptr)>, array<0 x ptr>)> 
    %78 = llvm.insertvalue %66, %77[1] : !llvm.struct<(struct<(struct<(i32)>, i32, i32, ptr)>, array<0 x ptr>)> 
    %79 = llvm.mlir.addressof @scc_info_Person_PhoneNumber_addressbook_2eproto : !llvm.ptr
    %80 = llvm.mlir.constant(16 : i64) : i64
    %81 = llvm.mlir.addressof @_ZN6google8protobuf8internal26fixed_address_empty_stringB5cxx11E : !llvm.ptr
    %82 = llvm.mlir.constant(24 : i64) : i64
    %83 = llvm.mlir.constant(1 : i32) : i32
    %84 = llvm.mlir.constant(3 : i32) : i32
    %85 = llvm.mlir.constant(2 : i32) : i32
    %86 = llvm.icmp "eq" %arg0, %0 : !llvm.ptr
    llvm.cond_br %86, ^bb1, ^bb5
  ^bb1:  // pred: ^bb0
    %87 = llvm.call @_Znwm(%14) : (i64) -> !llvm.ptr
    %88 = llvm.bitcast %87 : !llvm.ptr to !llvm.ptr
    %89 = llvm.getelementptr inbounds %88[%1, 0, 0, 1] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.tutorial::Person_PhoneNumber", (struct<"class.google::protobuf::Message", (struct<"class.google::protobuf::MessageLite", (ptr, struct<"class.google::protobuf::internal::InternalMetadata", (ptr)>)>)>, struct<"struct.google::protobuf::internal::ArenaStringPtr", (ptr)>, i32, struct<"class.google::protobuf::internal::CachedSize", (struct<"struct.std::atomic", (struct<"struct.std::__atomic_base", (i32)>)>)>)>
    %90 = llvm.bitcast %89 : !llvm.ptr to !llvm.ptr
    llvm.store %0, %90 {alignment = 8 : i64, tbaa = [#tbaa_tag5]} : !llvm.ptr, !llvm.ptr
    %91 = llvm.getelementptr inbounds %88[%1, 0, 0, 0] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.tutorial::Person_PhoneNumber", (struct<"class.google::protobuf::Message", (struct<"class.google::protobuf::MessageLite", (ptr, struct<"class.google::protobuf::internal::InternalMetadata", (ptr)>)>)>, struct<"struct.google::protobuf::internal::ArenaStringPtr", (ptr)>, i32, struct<"class.google::protobuf::internal::CachedSize", (struct<"struct.std::atomic", (struct<"struct.std::__atomic_base", (i32)>)>)>)>
    llvm.store %64, %91 {alignment = 8 : i64, tbaa = [#tbaa_tag]} : !llvm.ptr, !llvm.ptr
    %92 = llvm.getelementptr inbounds %88[%1, 3, 0, 0, 0] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.tutorial::Person_PhoneNumber", (struct<"class.google::protobuf::Message", (struct<"class.google::protobuf::MessageLite", (ptr, struct<"class.google::protobuf::internal::InternalMetadata", (ptr)>)>)>, struct<"struct.google::protobuf::internal::ArenaStringPtr", (ptr)>, i32, struct<"class.google::protobuf::internal::CachedSize", (struct<"struct.std::atomic", (struct<"struct.std::__atomic_base", (i32)>)>)>)>
    llvm.store %16, %92 {alignment = 4 : i64, tbaa = [#tbaa_tag6]} : i32, !llvm.ptr
    %93 = llvm.load %79 atomic acquire {alignment = 8 : i64} : !llvm.ptr -> i32
    %94 = llvm.icmp "eq" %93, %16 : i32
    llvm.cond_br %94 weights([2000, 1]), ^bb3, ^bb2
  ^bb2:  // pred: ^bb1
    llvm.invoke @_ZN6google8protobuf8internal11InitSCCImplEPNS1_11SCCInfoBaseE(%79) to ^bb3 unwind ^bb4 : (!llvm.ptr) -> ()
  ^bb3:  // 2 preds: ^bb1, ^bb2
    %95 = llvm.getelementptr inbounds %88[%1, 1, 0] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.tutorial::Person_PhoneNumber", (struct<"class.google::protobuf::Message", (struct<"class.google::protobuf::MessageLite", (ptr, struct<"class.google::protobuf::internal::InternalMetadata", (ptr)>)>)>, struct<"struct.google::protobuf::internal::ArenaStringPtr", (ptr)>, i32, struct<"class.google::protobuf::internal::CachedSize", (struct<"struct.std::atomic", (struct<"struct.std::__atomic_base", (i32)>)>)>)>
    llvm.store %81, %95 {alignment = 8 : i64, tbaa = [#tbaa_tag8]} : !llvm.ptr, !llvm.ptr
    %96 = llvm.getelementptr inbounds %88[%1, 2] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.tutorial::Person_PhoneNumber", (struct<"class.google::protobuf::Message", (struct<"class.google::protobuf::MessageLite", (ptr, struct<"class.google::protobuf::internal::InternalMetadata", (ptr)>)>)>, struct<"struct.google::protobuf::internal::ArenaStringPtr", (ptr)>, i32, struct<"class.google::protobuf::internal::CachedSize", (struct<"struct.std::atomic", (struct<"struct.std::__atomic_base", (i32)>)>)>)>
    llvm.br ^bb10(%96, %88 : !llvm.ptr, !llvm.ptr)
  ^bb4:  // pred: ^bb2
    %97 = llvm.landingpad cleanup : !llvm.struct<(ptr, i32)>
    llvm.call @_ZdlPv(%87) : (!llvm.ptr) -> ()
    llvm.resume %97 : !llvm.struct<(ptr, i32)>
  ^bb5:  // pred: ^bb0
    %98 = llvm.getelementptr inbounds %arg0[%1, 4] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.google::protobuf::Arena", (struct<"class.google::protobuf::internal::ArenaImpl", (struct<"struct.std::atomic.10", (struct<"struct.std::__atomic_base.11", (ptr)>)>, struct<"struct.std::atomic.10", (struct<"struct.std::__atomic_base.11", (ptr)>)>, struct<"struct.std::atomic.12", (struct<"struct.std::__atomic_base.13", (i64)>)>, ptr, i64, struct<"struct.google::protobuf::internal::ArenaImpl::Options", (i64, i64, ptr, i64, ptr, ptr)>)>, ptr, ptr, ptr, ptr)>
    %99 = llvm.load %98 {alignment = 8 : i64, tbaa = [#tbaa_tag30]} : !llvm.ptr -> !llvm.ptr
    %100 = llvm.icmp "eq" %99, %0 : !llvm.ptr
    llvm.cond_br %100 weights([2000, 1]), ^bb7, ^bb6
  ^bb6:  // pred: ^bb5
    llvm.call @_ZNK6google8protobuf5Arena17OnArenaAllocationEPKSt9type_infom(%arg0, %13, %14) : (!llvm.ptr, !llvm.ptr, i64) -> ()
    llvm.br ^bb7
  ^bb7:  // 2 preds: ^bb5, ^bb6
    %101 = llvm.call @_ZN6google8protobuf5Arena21AllocateAlignedNoHookEm(%arg0, %14) : (!llvm.ptr, i64) -> !llvm.ptr
    %102 = llvm.getelementptr inbounds %101[%15] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %103 = llvm.bitcast %102 : !llvm.ptr to !llvm.ptr
    llvm.store %arg0, %103 {alignment = 8 : i64, tbaa = [#tbaa_tag5]} : !llvm.ptr, !llvm.ptr
    %104 = llvm.bitcast %101 : !llvm.ptr to !llvm.ptr
    llvm.store %64, %104 {alignment = 8 : i64, tbaa = [#tbaa_tag]} : !llvm.ptr, !llvm.ptr
    %105 = llvm.getelementptr inbounds %101[%65] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %106 = llvm.bitcast %105 : !llvm.ptr to !llvm.ptr
    llvm.store %16, %106 {alignment = 4 : i64, tbaa = [#tbaa_tag6]} : i32, !llvm.ptr
    %107 = llvm.load %79 atomic acquire {alignment = 8 : i64} : !llvm.ptr -> i32
    %108 = llvm.icmp "eq" %107, %16 : i32
    llvm.cond_br %108 weights([2000, 1]), ^bb9, ^bb8
  ^bb8:  // pred: ^bb7
    llvm.call @_ZN6google8protobuf8internal11InitSCCImplEPNS1_11SCCInfoBaseE(%79) : (!llvm.ptr) -> ()
    llvm.br ^bb9
  ^bb9:  // 2 preds: ^bb7, ^bb8
    %109 = llvm.bitcast %101 : !llvm.ptr to !llvm.ptr
    %110 = llvm.getelementptr inbounds %101[%80] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %111 = llvm.bitcast %110 : !llvm.ptr to !llvm.ptr
    llvm.store %81, %111 {alignment = 8 : i64, tbaa = [#tbaa_tag8]} : !llvm.ptr, !llvm.ptr
    %112 = llvm.getelementptr inbounds %101[%82] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %113 = llvm.bitcast %112 : !llvm.ptr to !llvm.ptr
    llvm.br ^bb10(%113, %109 : !llvm.ptr, !llvm.ptr)
  ^bb10(%114: !llvm.ptr, %115: !llvm.ptr):  // 2 preds: ^bb3, ^bb9
    llvm.store %16, %114 {alignment = 8 : i64, tbaa = [#tbaa_tag24]} : i32, !llvm.ptr
    llvm.return %115 : !llvm.ptr
  }
  llvm.func local_unnamed_addr @_ZN6google8protobuf5Arena18CreateMaybeMessageIN8tutorial6PersonEJEEEPT_PS1_DpOT0_(%arg0: !llvm.ptr {llvm.noundef}) -> (!llvm.ptr {llvm.noundef}) attributes {alignment = 2 : i64, frame_pointer = #llvm.framePointerKind<none>, passthrough = ["noinline", ["uwtable", "2"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], ["tune-cpu", "generic"]], personality = @__gxx_personality_v0, target_cpu = "x86-64", target_features = #llvm.target_features<["+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>} {
    %0 = llvm.mlir.constant(1 : i32) : i32
    %1 = llvm.mlir.zero : !llvm.ptr
    %2 = llvm.mlir.constant(0 : i64) : i64
    %3 = llvm.mlir.constant(4 : i32) : i32
    %4 = llvm.mlir.addressof @_ZTIN6google8protobuf7MessageE : !llvm.ptr
    %5 = llvm.mlir.constant("N8tutorial6PersonE\00") : !llvm.array<19 x i8>
    %6 = llvm.mlir.addressof @_ZTSN8tutorial6PersonE : !llvm.ptr
    %7 = llvm.mlir.constant(2 : i64) : i64
    %8 = llvm.mlir.addressof @_ZTVN10__cxxabiv120__si_class_type_infoE : !llvm.ptr
    %9 = llvm.getelementptr inbounds %8[%7] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.ptr
    %10 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, ptr)>
    %11 = llvm.insertvalue %9, %10[0] : !llvm.struct<(ptr, ptr, ptr)> 
    %12 = llvm.insertvalue %6, %11[1] : !llvm.struct<(ptr, ptr, ptr)> 
    %13 = llvm.insertvalue %4, %12[2] : !llvm.struct<(ptr, ptr, ptr)> 
    %14 = llvm.mlir.addressof @_ZTIN8tutorial6PersonE : !llvm.ptr
    %15 = llvm.mlir.constant(72 : i64) : i64
    %16 = llvm.mlir.constant(0 : i32) : i32
    %17 = llvm.mlir.addressof @_ZNK8tutorial6Person11GetMetadataEv : !llvm.ptr
    %18 = llvm.mlir.addressof @__gxx_personality_v0 : !llvm.ptr
    %19 = llvm.mlir.addressof @_ZNK8tutorial6Person13SetCachedSizeEi : !llvm.ptr
    %20 = llvm.mlir.addressof @_ZNK6google8protobuf7Message13SpaceUsedLongEv : !llvm.ptr
    %21 = llvm.mlir.addressof @_ZN6google8protobuf7Message20DiscardUnknownFieldsEv : !llvm.ptr
    %22 = llvm.mlir.addressof @_ZN8tutorial6Person9MergeFromERKN6google8protobuf7MessageE : !llvm.ptr
    %23 = llvm.mlir.addressof @_ZN8tutorial6Person8CopyFromERKN6google8protobuf7MessageE : !llvm.ptr
    %24 = llvm.mlir.addressof @_ZNK6google8protobuf11MessageLite16InternalGetTableEv : !llvm.ptr
    %25 = llvm.mlir.addressof @_ZNK8tutorial6Person18_InternalSerializeEPhPN6google8protobuf2io19EpsCopyOutputStreamE : !llvm.ptr
    %26 = llvm.mlir.addressof @_ZN8tutorial6Person14_InternalParseEPKcPN6google8protobuf8internal12ParseContextE : !llvm.ptr
    %27 = llvm.mlir.addressof @_ZNK8tutorial6Person13GetCachedSizeEv : !llvm.ptr
    %28 = llvm.mlir.addressof @_ZNK8tutorial6Person12ByteSizeLongEv : !llvm.ptr
    %29 = llvm.mlir.addressof @_ZN6google8protobuf7Message21CheckTypeAndMergeFromERKNS0_11MessageLiteE : !llvm.ptr
    %30 = llvm.mlir.addressof @_ZNK6google8protobuf7Message25InitializationErrorStringB5cxx11Ev : !llvm.ptr
    %31 = llvm.mlir.addressof @_ZNK8tutorial6Person13IsInitializedEv : !llvm.ptr
    %32 = llvm.mlir.addressof @_ZN8tutorial6Person5ClearEv : !llvm.ptr
    %33 = llvm.mlir.addressof @_ZNK8tutorial6Person3NewEPN6google8protobuf5ArenaE : !llvm.ptr
    %34 = llvm.mlir.addressof @_ZNK8tutorial6Person3NewEv : !llvm.ptr
    %35 = llvm.mlir.addressof @_ZNK6google8protobuf7Message11GetTypeNameB5cxx11Ev : !llvm.ptr
    %36 = llvm.mlir.addressof @_ZN8tutorial6PersonD0Ev : !llvm.ptr
    %37 = llvm.mlir.addressof @_ZN8tutorial6PersonD2Ev : !llvm.ptr
    %38 = llvm.mlir.undef : !llvm.array<22 x ptr>
    %39 = llvm.insertvalue %1, %38[0] : !llvm.array<22 x ptr> 
    %40 = llvm.insertvalue %14, %39[1] : !llvm.array<22 x ptr> 
    %41 = llvm.insertvalue %37, %40[2] : !llvm.array<22 x ptr> 
    %42 = llvm.insertvalue %36, %41[3] : !llvm.array<22 x ptr> 
    %43 = llvm.insertvalue %35, %42[4] : !llvm.array<22 x ptr> 
    %44 = llvm.insertvalue %34, %43[5] : !llvm.array<22 x ptr> 
    %45 = llvm.insertvalue %33, %44[6] : !llvm.array<22 x ptr> 
    %46 = llvm.insertvalue %32, %45[7] : !llvm.array<22 x ptr> 
    %47 = llvm.insertvalue %31, %46[8] : !llvm.array<22 x ptr> 
    %48 = llvm.insertvalue %30, %47[9] : !llvm.array<22 x ptr> 
    %49 = llvm.insertvalue %29, %48[10] : !llvm.array<22 x ptr> 
    %50 = llvm.insertvalue %28, %49[11] : !llvm.array<22 x ptr> 
    %51 = llvm.insertvalue %27, %50[12] : !llvm.array<22 x ptr> 
    %52 = llvm.insertvalue %26, %51[13] : !llvm.array<22 x ptr> 
    %53 = llvm.insertvalue %25, %52[14] : !llvm.array<22 x ptr> 
    %54 = llvm.insertvalue %24, %53[15] : !llvm.array<22 x ptr> 
    %55 = llvm.insertvalue %23, %54[16] : !llvm.array<22 x ptr> 
    %56 = llvm.insertvalue %22, %55[17] : !llvm.array<22 x ptr> 
    %57 = llvm.insertvalue %21, %56[18] : !llvm.array<22 x ptr> 
    %58 = llvm.insertvalue %20, %57[19] : !llvm.array<22 x ptr> 
    %59 = llvm.insertvalue %19, %58[20] : !llvm.array<22 x ptr> 
    %60 = llvm.insertvalue %17, %59[21] : !llvm.array<22 x ptr> 
    %61 = llvm.mlir.undef : !llvm.struct<(array<22 x ptr>)>
    %62 = llvm.insertvalue %60, %61[0] : !llvm.struct<(array<22 x ptr>)> 
    %63 = llvm.mlir.addressof @_ZTVN8tutorial6PersonE : !llvm.ptr
    %64 = llvm.getelementptr inbounds %63[%2, 0, %7] : (!llvm.ptr, i64, i64) -> !llvm.ptr, !llvm.struct<(array<22 x ptr>)>
    %65 = llvm.mlir.constant(6 : i32) : i32
    %66 = llvm.mlir.constant(0 : i8) : i8
    %67 = llvm.mlir.constant(24 : i64) : i64
    %68 = llvm.mlir.addressof @scc_info_Timestamp_google_2fprotobuf_2ftimestamp_2eproto : !llvm.ptr
    %69 = llvm.mlir.undef : !llvm.array<0 x ptr>
    %70 = llvm.mlir.addressof @_ZL59InitDefaultsscc_info_Person_PhoneNumber_addressbook_2eprotov : !llvm.ptr
    %71 = llvm.mlir.constant(-1 : i32) : i32
    %72 = llvm.mlir.undef : !llvm.struct<(i32)>
    %73 = llvm.insertvalue %71, %72[0] : !llvm.struct<(i32)> 
    %74 = llvm.mlir.undef : !llvm.struct<(struct<(i32)>, i32, i32, ptr)>
    %75 = llvm.insertvalue %73, %74[0] : !llvm.struct<(struct<(i32)>, i32, i32, ptr)> 
    %76 = llvm.insertvalue %16, %75[1] : !llvm.struct<(struct<(i32)>, i32, i32, ptr)> 
    %77 = llvm.insertvalue %16, %76[2] : !llvm.struct<(struct<(i32)>, i32, i32, ptr)> 
    %78 = llvm.insertvalue %70, %77[3] : !llvm.struct<(struct<(i32)>, i32, i32, ptr)> 
    %79 = llvm.mlir.undef : !llvm.struct<(struct<(struct<(i32)>, i32, i32, ptr)>, array<0 x ptr>)>
    %80 = llvm.insertvalue %78, %79[0] : !llvm.struct<(struct<(struct<(i32)>, i32, i32, ptr)>, array<0 x ptr>)> 
    %81 = llvm.insertvalue %69, %80[1] : !llvm.struct<(struct<(struct<(i32)>, i32, i32, ptr)>, array<0 x ptr>)> 
    %82 = llvm.mlir.addressof @scc_info_Person_PhoneNumber_addressbook_2eproto : !llvm.ptr
    %83 = llvm.mlir.undef : !llvm.array<2 x ptr>
    %84 = llvm.insertvalue %82, %83[0] : !llvm.array<2 x ptr> 
    %85 = llvm.insertvalue %68, %84[1] : !llvm.array<2 x ptr> 
    %86 = llvm.mlir.addressof @_ZL47InitDefaultsscc_info_Person_addressbook_2eprotov : !llvm.ptr
    %87 = llvm.mlir.constant(2 : i32) : i32
    %88 = llvm.mlir.undef : !llvm.struct<(struct<(i32)>, i32, i32, ptr)>
    %89 = llvm.insertvalue %73, %88[0] : !llvm.struct<(struct<(i32)>, i32, i32, ptr)> 
    %90 = llvm.insertvalue %87, %89[1] : !llvm.struct<(struct<(i32)>, i32, i32, ptr)> 
    %91 = llvm.insertvalue %16, %90[2] : !llvm.struct<(struct<(i32)>, i32, i32, ptr)> 
    %92 = llvm.insertvalue %86, %91[3] : !llvm.struct<(struct<(i32)>, i32, i32, ptr)> 
    %93 = llvm.mlir.undef : !llvm.struct<(struct<(struct<(i32)>, i32, i32, ptr)>, array<2 x ptr>)>
    %94 = llvm.insertvalue %92, %93[0] : !llvm.struct<(struct<(struct<(i32)>, i32, i32, ptr)>, array<2 x ptr>)> 
    %95 = llvm.insertvalue %85, %94[1] : !llvm.struct<(struct<(struct<(i32)>, i32, i32, ptr)>, array<2 x ptr>)> 
    %96 = llvm.mlir.addressof @scc_info_Person_addressbook_2eproto : !llvm.ptr
    %97 = llvm.mlir.addressof @_ZN6google8protobuf8internal26fixed_address_empty_stringB5cxx11E : !llvm.ptr
    %98 = llvm.mlir.constant(3 : i32) : i32
    %99 = llvm.mlir.constant(12 : i64) : i64
    %100 = llvm.alloca %0 x !llvm.ptr {alignment = 8 : i64} : (i32) -> !llvm.ptr
    %101 = llvm.icmp "eq" %arg0, %1 : !llvm.ptr
    llvm.cond_br %101, ^bb1, ^bb10
  ^bb1:  // pred: ^bb0
    %102 = llvm.call @_Znwm(%15) : (i64) -> !llvm.ptr
    %103 = llvm.bitcast %102 : !llvm.ptr to !llvm.ptr
    %104 = llvm.getelementptr inbounds %103[%2, 0, 0, 1] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.tutorial::Person", (struct<"class.google::protobuf::Message", (struct<"class.google::protobuf::MessageLite", (ptr, struct<"class.google::protobuf::internal::InternalMetadata", (ptr)>)>)>, struct<"class.google::protobuf::RepeatedPtrField", (struct<"class.google::protobuf::internal::RepeatedPtrFieldBase", (ptr, i32, i32, ptr)>)>, struct<"struct.google::protobuf::internal::ArenaStringPtr", (ptr)>, struct<"struct.google::protobuf::internal::ArenaStringPtr", (ptr)>, ptr, i32, struct<"class.google::protobuf::internal::CachedSize", (struct<"struct.std::atomic", (struct<"struct.std::__atomic_base", (i32)>)>)>)>
    %105 = llvm.bitcast %104 : !llvm.ptr to !llvm.ptr
    llvm.store %1, %105 {alignment = 8 : i64, tbaa = [#tbaa_tag5]} : !llvm.ptr, !llvm.ptr
    %106 = llvm.getelementptr inbounds %103[%2, 0, 0, 0] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.tutorial::Person", (struct<"class.google::protobuf::Message", (struct<"class.google::protobuf::MessageLite", (ptr, struct<"class.google::protobuf::internal::InternalMetadata", (ptr)>)>)>, struct<"class.google::protobuf::RepeatedPtrField", (struct<"class.google::protobuf::internal::RepeatedPtrFieldBase", (ptr, i32, i32, ptr)>)>, struct<"struct.google::protobuf::internal::ArenaStringPtr", (ptr)>, struct<"struct.google::protobuf::internal::ArenaStringPtr", (ptr)>, ptr, i32, struct<"class.google::protobuf::internal::CachedSize", (struct<"struct.std::atomic", (struct<"struct.std::__atomic_base", (i32)>)>)>)>
    llvm.store %64, %106 {alignment = 8 : i64, tbaa = [#tbaa_tag]} : !llvm.ptr, !llvm.ptr
    %107 = llvm.getelementptr inbounds %103[%2, 1] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.tutorial::Person", (struct<"class.google::protobuf::Message", (struct<"class.google::protobuf::MessageLite", (ptr, struct<"class.google::protobuf::internal::InternalMetadata", (ptr)>)>)>, struct<"class.google::protobuf::RepeatedPtrField", (struct<"class.google::protobuf::internal::RepeatedPtrFieldBase", (ptr, i32, i32, ptr)>)>, struct<"struct.google::protobuf::internal::ArenaStringPtr", (ptr)>, struct<"struct.google::protobuf::internal::ArenaStringPtr", (ptr)>, ptr, i32, struct<"class.google::protobuf::internal::CachedSize", (struct<"struct.std::atomic", (struct<"struct.std::__atomic_base", (i32)>)>)>)>
    %108 = llvm.getelementptr inbounds %107[%2, 0, 0] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.google::protobuf::RepeatedPtrField", (struct<"class.google::protobuf::internal::RepeatedPtrFieldBase", (ptr, i32, i32, ptr)>)>
    %109 = llvm.getelementptr inbounds %103[%2, 6, 0, 0, 0] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.tutorial::Person", (struct<"class.google::protobuf::Message", (struct<"class.google::protobuf::MessageLite", (ptr, struct<"class.google::protobuf::internal::InternalMetadata", (ptr)>)>)>, struct<"class.google::protobuf::RepeatedPtrField", (struct<"class.google::protobuf::internal::RepeatedPtrFieldBase", (ptr, i32, i32, ptr)>)>, struct<"struct.google::protobuf::internal::ArenaStringPtr", (ptr)>, struct<"struct.google::protobuf::internal::ArenaStringPtr", (ptr)>, ptr, i32, struct<"class.google::protobuf::internal::CachedSize", (struct<"struct.std::atomic", (struct<"struct.std::__atomic_base", (i32)>)>)>)>
    llvm.store %16, %109 {alignment = 4 : i64, tbaa = [#tbaa_tag6]} : i32, !llvm.ptr
    %110 = llvm.bitcast %107 : !llvm.ptr to !llvm.ptr
    "llvm.intr.memset"(%110, %66, %67) <{isVolatile = false}> : (!llvm.ptr, i8, i64) -> ()
    %111 = llvm.load %96 atomic acquire {alignment = 8 : i64} : !llvm.ptr -> i32
    %112 = llvm.icmp "eq" %111, %16 : i32
    llvm.cond_br %112 weights([2000, 1]), ^bb8, ^bb2
  ^bb2:  // pred: ^bb1
    llvm.invoke @_ZN6google8protobuf8internal11InitSCCImplEPNS1_11SCCInfoBaseE(%96) to ^bb8 unwind ^bb3 : (!llvm.ptr) -> ()
  ^bb3:  // pred: ^bb2
    %113 = llvm.landingpad cleanup : !llvm.struct<(ptr, i32)>
    %114 = llvm.getelementptr inbounds %107[%2, 0] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.google::protobuf::RepeatedPtrField", (struct<"class.google::protobuf::internal::RepeatedPtrFieldBase", (ptr, i32, i32, ptr)>)>
    llvm.invoke @_ZN6google8protobuf8internal20RepeatedPtrFieldBase7DestroyINS0_16RepeatedPtrFieldIN8tutorial18Person_PhoneNumberEE11TypeHandlerEEEvv(%114) to ^bb4 unwind ^bb7 : (!llvm.ptr) -> ()
  ^bb4:  // pred: ^bb3
    %115 = llvm.load %108 {alignment = 8 : i64, tbaa = [#tbaa_tag7]} : !llvm.ptr -> !llvm.ptr
    %116 = llvm.icmp "eq" %115, %1 : !llvm.ptr
    llvm.cond_br %116, ^bb9, ^bb5
  ^bb5:  // pred: ^bb4
    %117 = llvm.getelementptr inbounds %115[%2, 0] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.google::protobuf::Arena", (struct<"class.google::protobuf::internal::ArenaImpl", (struct<"struct.std::atomic.10", (struct<"struct.std::__atomic_base.11", (ptr)>)>, struct<"struct.std::atomic.10", (struct<"struct.std::__atomic_base.11", (ptr)>)>, struct<"struct.std::atomic.12", (struct<"struct.std::__atomic_base.13", (i64)>)>, ptr, i64, struct<"struct.google::protobuf::internal::ArenaImpl::Options", (i64, i64, ptr, i64, ptr, ptr)>)>, ptr, ptr, ptr, ptr)>
    %118 = llvm.invoke @_ZNK6google8protobuf8internal9ArenaImpl14SpaceAllocatedEv(%117) to ^bb9 unwind ^bb6 : (!llvm.ptr) -> i64
  ^bb6:  // pred: ^bb5
    %119 = llvm.landingpad (catch %1 : !llvm.ptr) : !llvm.struct<(ptr, i32)>
    %120 = llvm.extractvalue %119[0] : !llvm.struct<(ptr, i32)> 
    llvm.call @__clang_call_terminate(%120) : (!llvm.ptr) -> ()
    llvm.unreachable
  ^bb7:  // pred: ^bb3
    %121 = llvm.landingpad (catch %1 : !llvm.ptr) : !llvm.struct<(ptr, i32)>
    %122 = llvm.extractvalue %121[0] : !llvm.struct<(ptr, i32)> 
    llvm.call @_ZN6google8protobuf8internal20RepeatedPtrFieldBaseD2Ev(%114) : (!llvm.ptr) -> ()
    llvm.call @__clang_call_terminate(%122) : (!llvm.ptr) -> ()
    llvm.unreachable
  ^bb8:  // 2 preds: ^bb1, ^bb2
    %123 = llvm.getelementptr inbounds %103[%2, 2, 0] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.tutorial::Person", (struct<"class.google::protobuf::Message", (struct<"class.google::protobuf::MessageLite", (ptr, struct<"class.google::protobuf::internal::InternalMetadata", (ptr)>)>)>, struct<"class.google::protobuf::RepeatedPtrField", (struct<"class.google::protobuf::internal::RepeatedPtrFieldBase", (ptr, i32, i32, ptr)>)>, struct<"struct.google::protobuf::internal::ArenaStringPtr", (ptr)>, struct<"struct.google::protobuf::internal::ArenaStringPtr", (ptr)>, ptr, i32, struct<"class.google::protobuf::internal::CachedSize", (struct<"struct.std::atomic", (struct<"struct.std::__atomic_base", (i32)>)>)>)>
    llvm.store %97, %123 {alignment = 8 : i64, tbaa = [#tbaa_tag8]} : !llvm.ptr, !llvm.ptr
    %124 = llvm.getelementptr inbounds %103[%2, 3, 0] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.tutorial::Person", (struct<"class.google::protobuf::Message", (struct<"class.google::protobuf::MessageLite", (ptr, struct<"class.google::protobuf::internal::InternalMetadata", (ptr)>)>)>, struct<"class.google::protobuf::RepeatedPtrField", (struct<"class.google::protobuf::internal::RepeatedPtrFieldBase", (ptr, i32, i32, ptr)>)>, struct<"struct.google::protobuf::internal::ArenaStringPtr", (ptr)>, struct<"struct.google::protobuf::internal::ArenaStringPtr", (ptr)>, ptr, i32, struct<"class.google::protobuf::internal::CachedSize", (struct<"struct.std::atomic", (struct<"struct.std::__atomic_base", (i32)>)>)>)>
    llvm.store %97, %124 {alignment = 8 : i64, tbaa = [#tbaa_tag8]} : !llvm.ptr, !llvm.ptr
    %125 = llvm.getelementptr inbounds %103[%2, 4] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.tutorial::Person", (struct<"class.google::protobuf::Message", (struct<"class.google::protobuf::MessageLite", (ptr, struct<"class.google::protobuf::internal::InternalMetadata", (ptr)>)>)>, struct<"class.google::protobuf::RepeatedPtrField", (struct<"class.google::protobuf::internal::RepeatedPtrFieldBase", (ptr, i32, i32, ptr)>)>, struct<"struct.google::protobuf::internal::ArenaStringPtr", (ptr)>, struct<"struct.google::protobuf::internal::ArenaStringPtr", (ptr)>, ptr, i32, struct<"class.google::protobuf::internal::CachedSize", (struct<"struct.std::atomic", (struct<"struct.std::__atomic_base", (i32)>)>)>)>
    %126 = llvm.bitcast %125 : !llvm.ptr to !llvm.ptr
    "llvm.intr.memset"(%126, %66, %99) <{isVolatile = false}> : (!llvm.ptr, i8, i64) -> ()
    llvm.br ^bb13(%103 : !llvm.ptr)
  ^bb9:  // 2 preds: ^bb4, ^bb5
    llvm.call @_ZdlPv(%102) : (!llvm.ptr) -> ()
    llvm.resume %113 : !llvm.struct<(ptr, i32)>
  ^bb10:  // pred: ^bb0
    %127 = llvm.getelementptr inbounds %arg0[%2, 4] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.google::protobuf::Arena", (struct<"class.google::protobuf::internal::ArenaImpl", (struct<"struct.std::atomic.10", (struct<"struct.std::__atomic_base.11", (ptr)>)>, struct<"struct.std::atomic.10", (struct<"struct.std::__atomic_base.11", (ptr)>)>, struct<"struct.std::atomic.12", (struct<"struct.std::__atomic_base.13", (i64)>)>, ptr, i64, struct<"struct.google::protobuf::internal::ArenaImpl::Options", (i64, i64, ptr, i64, ptr, ptr)>)>, ptr, ptr, ptr, ptr)>
    %128 = llvm.load %127 {alignment = 8 : i64, tbaa = [#tbaa_tag30]} : !llvm.ptr -> !llvm.ptr
    %129 = llvm.icmp "eq" %128, %1 : !llvm.ptr
    llvm.cond_br %129 weights([2000, 1]), ^bb12, ^bb11
  ^bb11:  // pred: ^bb10
    llvm.call @_ZNK6google8protobuf5Arena17OnArenaAllocationEPKSt9type_infom(%arg0, %14, %15) : (!llvm.ptr, !llvm.ptr, i64) -> ()
    llvm.br ^bb12
  ^bb12:  // 2 preds: ^bb10, ^bb11
    %130 = llvm.call @_ZN6google8protobuf5Arena21AllocateAlignedNoHookEm(%arg0, %15) : (!llvm.ptr, i64) -> !llvm.ptr
    %131 = llvm.bitcast %100 : !llvm.ptr to !llvm.ptr
    llvm.intr.lifetime.start 8, %131 : !llvm.ptr
    llvm.store %arg0, %100 {alignment = 8 : i64, tbaa = [#tbaa_tag2]} : !llvm.ptr, !llvm.ptr
    %132 = llvm.call @_ZN6google8protobuf5Arena14InternalHelperIN8tutorial6PersonEE9ConstructIJPS1_EEEPS4_PvDpOT_(%130, %100) : (!llvm.ptr, !llvm.ptr) -> !llvm.ptr
    llvm.intr.lifetime.end 8, %131 : !llvm.ptr
    llvm.br ^bb13(%132 : !llvm.ptr)
  ^bb13(%133: !llvm.ptr):  // 2 preds: ^bb8, ^bb12
    llvm.return %133 : !llvm.ptr
  }
  llvm.func local_unnamed_addr @_ZN6google8protobuf5Arena18CreateMaybeMessageIN8tutorial11AddressBookEJEEEPT_PS1_DpOT0_(%arg0: !llvm.ptr {llvm.noundef}) -> (!llvm.ptr {llvm.noundef}) attributes {alignment = 2 : i64, frame_pointer = #llvm.framePointerKind<none>, passthrough = ["noinline", ["uwtable", "2"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], ["tune-cpu", "generic"]], personality = @__gxx_personality_v0, target_cpu = "x86-64", target_features = #llvm.target_features<["+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>} {
    %0 = llvm.mlir.zero : !llvm.ptr
    %1 = llvm.mlir.constant(0 : i64) : i64
    %2 = llvm.mlir.constant(4 : i32) : i32
    %3 = llvm.mlir.addressof @_ZTIN6google8protobuf7MessageE : !llvm.ptr
    %4 = llvm.mlir.constant("N8tutorial11AddressBookE\00") : !llvm.array<25 x i8>
    %5 = llvm.mlir.addressof @_ZTSN8tutorial11AddressBookE : !llvm.ptr
    %6 = llvm.mlir.constant(2 : i64) : i64
    %7 = llvm.mlir.addressof @_ZTVN10__cxxabiv120__si_class_type_infoE : !llvm.ptr
    %8 = llvm.getelementptr inbounds %7[%6] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.ptr
    %9 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, ptr)>
    %10 = llvm.insertvalue %8, %9[0] : !llvm.struct<(ptr, ptr, ptr)> 
    %11 = llvm.insertvalue %5, %10[1] : !llvm.struct<(ptr, ptr, ptr)> 
    %12 = llvm.insertvalue %3, %11[2] : !llvm.struct<(ptr, ptr, ptr)> 
    %13 = llvm.mlir.addressof @_ZTIN8tutorial11AddressBookE : !llvm.ptr
    %14 = llvm.mlir.constant(48 : i64) : i64
    %15 = llvm.mlir.constant(8 : i64) : i64
    %16 = llvm.mlir.constant(0 : i32) : i32
    %17 = llvm.mlir.addressof @_ZNK8tutorial11AddressBook11GetMetadataEv : !llvm.ptr
    %18 = llvm.mlir.addressof @__gxx_personality_v0 : !llvm.ptr
    %19 = llvm.mlir.addressof @_ZNK8tutorial11AddressBook13SetCachedSizeEi : !llvm.ptr
    %20 = llvm.mlir.addressof @_ZNK6google8protobuf7Message13SpaceUsedLongEv : !llvm.ptr
    %21 = llvm.mlir.addressof @_ZN6google8protobuf7Message20DiscardUnknownFieldsEv : !llvm.ptr
    %22 = llvm.mlir.addressof @_ZN8tutorial11AddressBook9MergeFromERKN6google8protobuf7MessageE : !llvm.ptr
    %23 = llvm.mlir.addressof @_ZN8tutorial11AddressBook8CopyFromERKN6google8protobuf7MessageE : !llvm.ptr
    %24 = llvm.mlir.addressof @_ZNK6google8protobuf11MessageLite16InternalGetTableEv : !llvm.ptr
    %25 = llvm.mlir.addressof @_ZNK8tutorial11AddressBook18_InternalSerializeEPhPN6google8protobuf2io19EpsCopyOutputStreamE : !llvm.ptr
    %26 = llvm.mlir.addressof @_ZN8tutorial11AddressBook14_InternalParseEPKcPN6google8protobuf8internal12ParseContextE : !llvm.ptr
    %27 = llvm.mlir.addressof @_ZNK8tutorial11AddressBook13GetCachedSizeEv : !llvm.ptr
    %28 = llvm.mlir.addressof @_ZNK8tutorial11AddressBook12ByteSizeLongEv : !llvm.ptr
    %29 = llvm.mlir.addressof @_ZN6google8protobuf7Message21CheckTypeAndMergeFromERKNS0_11MessageLiteE : !llvm.ptr
    %30 = llvm.mlir.addressof @_ZNK6google8protobuf7Message25InitializationErrorStringB5cxx11Ev : !llvm.ptr
    %31 = llvm.mlir.addressof @_ZNK8tutorial11AddressBook13IsInitializedEv : !llvm.ptr
    %32 = llvm.mlir.addressof @_ZN8tutorial11AddressBook5ClearEv : !llvm.ptr
    %33 = llvm.mlir.addressof @_ZNK8tutorial11AddressBook3NewEPN6google8protobuf5ArenaE : !llvm.ptr
    %34 = llvm.mlir.addressof @_ZNK8tutorial11AddressBook3NewEv : !llvm.ptr
    %35 = llvm.mlir.addressof @_ZNK6google8protobuf7Message11GetTypeNameB5cxx11Ev : !llvm.ptr
    %36 = llvm.mlir.addressof @_ZN8tutorial11AddressBookD0Ev : !llvm.ptr
    %37 = llvm.mlir.addressof @_ZN8tutorial11AddressBookD2Ev : !llvm.ptr
    %38 = llvm.mlir.undef : !llvm.array<22 x ptr>
    %39 = llvm.insertvalue %0, %38[0] : !llvm.array<22 x ptr> 
    %40 = llvm.insertvalue %13, %39[1] : !llvm.array<22 x ptr> 
    %41 = llvm.insertvalue %37, %40[2] : !llvm.array<22 x ptr> 
    %42 = llvm.insertvalue %36, %41[3] : !llvm.array<22 x ptr> 
    %43 = llvm.insertvalue %35, %42[4] : !llvm.array<22 x ptr> 
    %44 = llvm.insertvalue %34, %43[5] : !llvm.array<22 x ptr> 
    %45 = llvm.insertvalue %33, %44[6] : !llvm.array<22 x ptr> 
    %46 = llvm.insertvalue %32, %45[7] : !llvm.array<22 x ptr> 
    %47 = llvm.insertvalue %31, %46[8] : !llvm.array<22 x ptr> 
    %48 = llvm.insertvalue %30, %47[9] : !llvm.array<22 x ptr> 
    %49 = llvm.insertvalue %29, %48[10] : !llvm.array<22 x ptr> 
    %50 = llvm.insertvalue %28, %49[11] : !llvm.array<22 x ptr> 
    %51 = llvm.insertvalue %27, %50[12] : !llvm.array<22 x ptr> 
    %52 = llvm.insertvalue %26, %51[13] : !llvm.array<22 x ptr> 
    %53 = llvm.insertvalue %25, %52[14] : !llvm.array<22 x ptr> 
    %54 = llvm.insertvalue %24, %53[15] : !llvm.array<22 x ptr> 
    %55 = llvm.insertvalue %23, %54[16] : !llvm.array<22 x ptr> 
    %56 = llvm.insertvalue %22, %55[17] : !llvm.array<22 x ptr> 
    %57 = llvm.insertvalue %21, %56[18] : !llvm.array<22 x ptr> 
    %58 = llvm.insertvalue %20, %57[19] : !llvm.array<22 x ptr> 
    %59 = llvm.insertvalue %19, %58[20] : !llvm.array<22 x ptr> 
    %60 = llvm.insertvalue %17, %59[21] : !llvm.array<22 x ptr> 
    %61 = llvm.mlir.undef : !llvm.struct<(array<22 x ptr>)>
    %62 = llvm.insertvalue %60, %61[0] : !llvm.struct<(array<22 x ptr>)> 
    %63 = llvm.mlir.addressof @_ZTVN8tutorial11AddressBookE : !llvm.ptr
    %64 = llvm.getelementptr inbounds %63[%1, 0, %6] : (!llvm.ptr, i64, i64) -> !llvm.ptr, !llvm.struct<(array<22 x ptr>)>
    %65 = llvm.mlir.constant(16 : i64) : i64
    %66 = llvm.mlir.constant(24 : i64) : i64
    %67 = llvm.mlir.constant(0 : i8) : i8
    %68 = llvm.mlir.constant(20 : i64) : i64
    %69 = llvm.mlir.addressof @scc_info_Timestamp_google_2fprotobuf_2ftimestamp_2eproto : !llvm.ptr
    %70 = llvm.mlir.undef : !llvm.array<0 x ptr>
    %71 = llvm.mlir.addressof @_ZL59InitDefaultsscc_info_Person_PhoneNumber_addressbook_2eprotov : !llvm.ptr
    %72 = llvm.mlir.constant(-1 : i32) : i32
    %73 = llvm.mlir.undef : !llvm.struct<(i32)>
    %74 = llvm.insertvalue %72, %73[0] : !llvm.struct<(i32)> 
    %75 = llvm.mlir.undef : !llvm.struct<(struct<(i32)>, i32, i32, ptr)>
    %76 = llvm.insertvalue %74, %75[0] : !llvm.struct<(struct<(i32)>, i32, i32, ptr)> 
    %77 = llvm.insertvalue %16, %76[1] : !llvm.struct<(struct<(i32)>, i32, i32, ptr)> 
    %78 = llvm.insertvalue %16, %77[2] : !llvm.struct<(struct<(i32)>, i32, i32, ptr)> 
    %79 = llvm.insertvalue %71, %78[3] : !llvm.struct<(struct<(i32)>, i32, i32, ptr)> 
    %80 = llvm.mlir.undef : !llvm.struct<(struct<(struct<(i32)>, i32, i32, ptr)>, array<0 x ptr>)>
    %81 = llvm.insertvalue %79, %80[0] : !llvm.struct<(struct<(struct<(i32)>, i32, i32, ptr)>, array<0 x ptr>)> 
    %82 = llvm.insertvalue %70, %81[1] : !llvm.struct<(struct<(struct<(i32)>, i32, i32, ptr)>, array<0 x ptr>)> 
    %83 = llvm.mlir.addressof @scc_info_Person_PhoneNumber_addressbook_2eproto : !llvm.ptr
    %84 = llvm.mlir.undef : !llvm.array<2 x ptr>
    %85 = llvm.insertvalue %83, %84[0] : !llvm.array<2 x ptr> 
    %86 = llvm.insertvalue %69, %85[1] : !llvm.array<2 x ptr> 
    %87 = llvm.mlir.addressof @_ZL47InitDefaultsscc_info_Person_addressbook_2eprotov : !llvm.ptr
    %88 = llvm.mlir.constant(2 : i32) : i32
    %89 = llvm.mlir.undef : !llvm.struct<(struct<(i32)>, i32, i32, ptr)>
    %90 = llvm.insertvalue %74, %89[0] : !llvm.struct<(struct<(i32)>, i32, i32, ptr)> 
    %91 = llvm.insertvalue %88, %90[1] : !llvm.struct<(struct<(i32)>, i32, i32, ptr)> 
    %92 = llvm.insertvalue %16, %91[2] : !llvm.struct<(struct<(i32)>, i32, i32, ptr)> 
    %93 = llvm.insertvalue %87, %92[3] : !llvm.struct<(struct<(i32)>, i32, i32, ptr)> 
    %94 = llvm.mlir.undef : !llvm.struct<(struct<(struct<(i32)>, i32, i32, ptr)>, array<2 x ptr>)>
    %95 = llvm.insertvalue %93, %94[0] : !llvm.struct<(struct<(struct<(i32)>, i32, i32, ptr)>, array<2 x ptr>)> 
    %96 = llvm.insertvalue %86, %95[1] : !llvm.struct<(struct<(struct<(i32)>, i32, i32, ptr)>, array<2 x ptr>)> 
    %97 = llvm.mlir.addressof @scc_info_Person_addressbook_2eproto : !llvm.ptr
    %98 = llvm.mlir.undef : !llvm.array<1 x ptr>
    %99 = llvm.insertvalue %97, %98[0] : !llvm.array<1 x ptr> 
    %100 = llvm.mlir.addressof @_ZL52InitDefaultsscc_info_AddressBook_addressbook_2eprotov : !llvm.ptr
    %101 = llvm.mlir.constant(1 : i32) : i32
    %102 = llvm.mlir.undef : !llvm.struct<(struct<(i32)>, i32, i32, ptr)>
    %103 = llvm.insertvalue %74, %102[0] : !llvm.struct<(struct<(i32)>, i32, i32, ptr)> 
    %104 = llvm.insertvalue %101, %103[1] : !llvm.struct<(struct<(i32)>, i32, i32, ptr)> 
    %105 = llvm.insertvalue %16, %104[2] : !llvm.struct<(struct<(i32)>, i32, i32, ptr)> 
    %106 = llvm.insertvalue %100, %105[3] : !llvm.struct<(struct<(i32)>, i32, i32, ptr)> 
    %107 = llvm.mlir.undef : !llvm.struct<(struct<(struct<(i32)>, i32, i32, ptr)>, array<1 x ptr>)>
    %108 = llvm.insertvalue %106, %107[0] : !llvm.struct<(struct<(struct<(i32)>, i32, i32, ptr)>, array<1 x ptr>)> 
    %109 = llvm.insertvalue %99, %108[1] : !llvm.struct<(struct<(struct<(i32)>, i32, i32, ptr)>, array<1 x ptr>)> 
    %110 = llvm.mlir.addressof @scc_info_AddressBook_addressbook_2eproto : !llvm.ptr
    %111 = llvm.mlir.constant(28 : i64) : i64
    %112 = llvm.icmp "eq" %arg0, %0 : !llvm.ptr
    llvm.cond_br %112, ^bb1, ^bb5
  ^bb1:  // pred: ^bb0
    %113 = llvm.call @_Znwm(%14) : (i64) -> !llvm.ptr
    %114 = llvm.bitcast %113 : !llvm.ptr to !llvm.ptr
    %115 = llvm.getelementptr inbounds %114[%1, 0, 0, 1] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.tutorial::AddressBook", packed (struct<"class.google::protobuf::Message", (struct<"class.google::protobuf::MessageLite", (ptr, struct<"class.google::protobuf::internal::InternalMetadata", (ptr)>)>)>, struct<"class.google::protobuf::RepeatedPtrField.18", (struct<"class.google::protobuf::internal::RepeatedPtrFieldBase", (ptr, i32, i32, ptr)>)>, struct<"class.google::protobuf::internal::CachedSize", (struct<"struct.std::atomic", (struct<"struct.std::__atomic_base", (i32)>)>)>, array<4 x i8>)>
    %116 = llvm.bitcast %115 : !llvm.ptr to !llvm.ptr
    llvm.store %0, %116 {alignment = 8 : i64, tbaa = [#tbaa_tag5]} : !llvm.ptr, !llvm.ptr
    %117 = llvm.getelementptr inbounds %114[%1, 0, 0, 0] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.tutorial::AddressBook", packed (struct<"class.google::protobuf::Message", (struct<"class.google::protobuf::MessageLite", (ptr, struct<"class.google::protobuf::internal::InternalMetadata", (ptr)>)>)>, struct<"class.google::protobuf::RepeatedPtrField.18", (struct<"class.google::protobuf::internal::RepeatedPtrFieldBase", (ptr, i32, i32, ptr)>)>, struct<"class.google::protobuf::internal::CachedSize", (struct<"struct.std::atomic", (struct<"struct.std::__atomic_base", (i32)>)>)>, array<4 x i8>)>
    llvm.store %64, %117 {alignment = 8 : i64, tbaa = [#tbaa_tag]} : !llvm.ptr, !llvm.ptr
    %118 = llvm.getelementptr inbounds %114[%1, 1] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.tutorial::AddressBook", packed (struct<"class.google::protobuf::Message", (struct<"class.google::protobuf::MessageLite", (ptr, struct<"class.google::protobuf::internal::InternalMetadata", (ptr)>)>)>, struct<"class.google::protobuf::RepeatedPtrField.18", (struct<"class.google::protobuf::internal::RepeatedPtrFieldBase", (ptr, i32, i32, ptr)>)>, struct<"class.google::protobuf::internal::CachedSize", (struct<"struct.std::atomic", (struct<"struct.std::__atomic_base", (i32)>)>)>, array<4 x i8>)>
    %119 = llvm.bitcast %118 : !llvm.ptr to !llvm.ptr
    "llvm.intr.memset"(%119, %67, %111) <{isVolatile = false}> : (!llvm.ptr, i8, i64) -> ()
    %120 = llvm.load %110 atomic acquire {alignment = 8 : i64} : !llvm.ptr -> i32
    %121 = llvm.icmp "eq" %120, %16 : i32
    llvm.cond_br %121 weights([2000, 1]), ^bb11(%114 : !llvm.ptr), ^bb2
  ^bb2:  // pred: ^bb1
    llvm.invoke @_ZN6google8protobuf8internal11InitSCCImplEPNS1_11SCCInfoBaseE(%110, %114) to ^bb11(%114 : !llvm.ptr) unwind ^bb3 : (!llvm.ptr, !llvm.ptr) -> ()
  ^bb3:  // pred: ^bb2
    %122 = llvm.landingpad cleanup : !llvm.struct<(ptr, i32)>
    llvm.call @_ZN6google8protobuf16RepeatedPtrFieldIN8tutorial6PersonEED2Ev(%118) : (!llvm.ptr) -> ()
    llvm.call @_ZdlPv(%113) : (!llvm.ptr) -> ()
    llvm.br ^bb4(%122 : !llvm.struct<(ptr, i32)>)
  ^bb4(%123: !llvm.struct<(ptr, i32)>):  // 2 preds: ^bb3, ^bb9
    llvm.resume %123 : !llvm.struct<(ptr, i32)>
  ^bb5:  // pred: ^bb0
    %124 = llvm.getelementptr inbounds %arg0[%1, 4] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.google::protobuf::Arena", (struct<"class.google::protobuf::internal::ArenaImpl", (struct<"struct.std::atomic.10", (struct<"struct.std::__atomic_base.11", (ptr)>)>, struct<"struct.std::atomic.10", (struct<"struct.std::__atomic_base.11", (ptr)>)>, struct<"struct.std::atomic.12", (struct<"struct.std::__atomic_base.13", (i64)>)>, ptr, i64, struct<"struct.google::protobuf::internal::ArenaImpl::Options", (i64, i64, ptr, i64, ptr, ptr)>)>, ptr, ptr, ptr, ptr)>
    %125 = llvm.load %124 {alignment = 8 : i64, tbaa = [#tbaa_tag30]} : !llvm.ptr -> !llvm.ptr
    %126 = llvm.icmp "eq" %125, %0 : !llvm.ptr
    llvm.cond_br %126 weights([2000, 1]), ^bb7, ^bb6
  ^bb6:  // pred: ^bb5
    llvm.call @_ZNK6google8protobuf5Arena17OnArenaAllocationEPKSt9type_infom(%arg0, %13, %14) : (!llvm.ptr, !llvm.ptr, i64) -> ()
    llvm.br ^bb7
  ^bb7:  // 2 preds: ^bb5, ^bb6
    %127 = llvm.call @_ZN6google8protobuf5Arena21AllocateAlignedNoHookEm(%arg0, %14) : (!llvm.ptr, i64) -> !llvm.ptr
    %128 = llvm.getelementptr inbounds %127[%15] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %129 = llvm.bitcast %128 : !llvm.ptr to !llvm.ptr
    llvm.store %arg0, %129 {alignment = 8 : i64, tbaa = [#tbaa_tag5]} : !llvm.ptr, !llvm.ptr
    %130 = llvm.bitcast %127 : !llvm.ptr to !llvm.ptr
    llvm.store %64, %130 {alignment = 8 : i64, tbaa = [#tbaa_tag]} : !llvm.ptr, !llvm.ptr
    %131 = llvm.getelementptr inbounds %127[%65] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %132 = llvm.bitcast %131 : !llvm.ptr to !llvm.ptr
    llvm.store %arg0, %132 {alignment = 8 : i64, tbaa = [#tbaa_tag7]} : !llvm.ptr, !llvm.ptr
    %133 = llvm.getelementptr inbounds %127[%66] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    "llvm.intr.memset"(%133, %67, %68) <{isVolatile = false}> : (!llvm.ptr, i8, i64) -> ()
    %134 = llvm.load %110 atomic acquire {alignment = 8 : i64} : !llvm.ptr -> i32
    %135 = llvm.icmp "eq" %134, %16 : i32
    llvm.cond_br %135 weights([2000, 1]), ^bb10, ^bb8
  ^bb8:  // pred: ^bb7
    llvm.invoke @_ZN6google8protobuf8internal11InitSCCImplEPNS1_11SCCInfoBaseE(%110) to ^bb10 unwind ^bb9 : (!llvm.ptr) -> ()
  ^bb9:  // pred: ^bb8
    %136 = llvm.landingpad cleanup : !llvm.struct<(ptr, i32)>
    %137 = llvm.bitcast %131 : !llvm.ptr to !llvm.ptr
    llvm.call @_ZN6google8protobuf16RepeatedPtrFieldIN8tutorial6PersonEED2Ev(%137) : (!llvm.ptr) -> ()
    llvm.br ^bb4(%136 : !llvm.struct<(ptr, i32)>)
  ^bb10:  // 2 preds: ^bb7, ^bb8
    %138 = llvm.bitcast %127 : !llvm.ptr to !llvm.ptr
    llvm.br ^bb11(%138 : !llvm.ptr)
  ^bb11(%139: !llvm.ptr):  // 3 preds: ^bb1, ^bb2, ^bb10
    llvm.return %139 : !llvm.ptr
  }
  llvm.func unnamed_addr @_ZNK6google8protobuf7Message11GetTypeNameB5cxx11Ev(!llvm.ptr {llvm.align = 8 : i64, llvm.sret = !llvm.struct<"class.std::__cxx11::basic_string", (struct<"struct.std::__cxx11::basic_string<char>::_Alloc_hider", (ptr)>, i64, struct<"union.anon", (i64, array<8 x i8>)>)>}, !llvm.ptr {llvm.align = 8 : i64, llvm.dereferenceable = 16 : i64, llvm.nonnull, llvm.noundef}) attributes {frame_pointer = #llvm.framePointerKind<none>, passthrough = [["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], ["tune-cpu", "generic"]], target_cpu = "x86-64", target_features = #llvm.target_features<["+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>}
  llvm.func linkonce_odr unnamed_addr @_ZNK8tutorial18Person_PhoneNumber3NewEv(%arg0: !llvm.ptr {llvm.align = 8 : i64, llvm.dereferenceable = 32 : i64, llvm.nonnull, llvm.noundef}) -> (!llvm.ptr {llvm.noundef}) comdat(@__llvm_global_comdat::@_ZNK8tutorial18Person_PhoneNumber3NewEv) attributes {alignment = 2 : i64, frame_pointer = #llvm.framePointerKind<none>, passthrough = ["inlinehint", "mustprogress", ["uwtable", "2"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], ["tune-cpu", "generic"]], target_cpu = "x86-64", target_features = #llvm.target_features<["+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>} {
    %0 = llvm.mlir.zero : !llvm.ptr
    %1 = llvm.call @_ZN6google8protobuf5Arena18CreateMaybeMessageIN8tutorial18Person_PhoneNumberEJEEEPT_PS1_DpOT0_(%0) : (!llvm.ptr) -> !llvm.ptr
    llvm.return %1 : !llvm.ptr
  }
  llvm.func linkonce_odr unnamed_addr @_ZNK8tutorial18Person_PhoneNumber3NewEPN6google8protobuf5ArenaE(%arg0: !llvm.ptr {llvm.align = 8 : i64, llvm.dereferenceable = 32 : i64, llvm.nonnull, llvm.noundef}, %arg1: !llvm.ptr {llvm.noundef}) -> (!llvm.ptr {llvm.noundef}) comdat(@__llvm_global_comdat::@_ZNK8tutorial18Person_PhoneNumber3NewEPN6google8protobuf5ArenaE) attributes {alignment = 2 : i64, frame_pointer = #llvm.framePointerKind<none>, passthrough = ["mustprogress", ["uwtable", "2"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], ["tune-cpu", "generic"]], target_cpu = "x86-64", target_features = #llvm.target_features<["+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>} {
    %0 = llvm.call @_ZN6google8protobuf5Arena18CreateMaybeMessageIN8tutorial18Person_PhoneNumberEJEEEPT_PS1_DpOT0_(%arg1) : (!llvm.ptr) -> !llvm.ptr
    llvm.return %0 : !llvm.ptr
  }
  llvm.func unnamed_addr @_ZNK6google8protobuf7Message25InitializationErrorStringB5cxx11Ev(!llvm.ptr {llvm.align = 8 : i64, llvm.sret = !llvm.struct<"class.std::__cxx11::basic_string", (struct<"struct.std::__cxx11::basic_string<char>::_Alloc_hider", (ptr)>, i64, struct<"union.anon", (i64, array<8 x i8>)>)>}, !llvm.ptr {llvm.align = 8 : i64, llvm.dereferenceable = 16 : i64, llvm.nonnull, llvm.noundef}) attributes {frame_pointer = #llvm.framePointerKind<none>, passthrough = [["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], ["tune-cpu", "generic"]], target_cpu = "x86-64", target_features = #llvm.target_features<["+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>}
  llvm.func unnamed_addr @_ZN6google8protobuf7Message21CheckTypeAndMergeFromERKNS0_11MessageLiteE(!llvm.ptr {llvm.align = 8 : i64, llvm.dereferenceable = 16 : i64, llvm.nonnull, llvm.noundef}, !llvm.ptr {llvm.align = 8 : i64, llvm.dereferenceable = 16 : i64, llvm.nonnull, llvm.noundef}) attributes {frame_pointer = #llvm.framePointerKind<none>, passthrough = [["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], ["tune-cpu", "generic"]], target_cpu = "x86-64", target_features = #llvm.target_features<["+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>}
  llvm.func linkonce_odr unnamed_addr @_ZNK8tutorial18Person_PhoneNumber13GetCachedSizeEv(%arg0: !llvm.ptr {llvm.align = 8 : i64, llvm.dereferenceable = 32 : i64, llvm.nonnull, llvm.noundef}) -> (i32 {llvm.noundef}) comdat(@__llvm_global_comdat::@_ZNK8tutorial18Person_PhoneNumber13GetCachedSizeEv) attributes {alignment = 2 : i64, frame_pointer = #llvm.framePointerKind<none>, passthrough = ["mustprogress", ["uwtable", "2"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], ["tune-cpu", "generic"]], target_cpu = "x86-64", target_features = #llvm.target_features<["+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>} {
    %0 = llvm.mlir.constant(0 : i64) : i64
    %1 = llvm.mlir.constant(3 : i32) : i32
    %2 = llvm.mlir.constant(0 : i32) : i32
    %3 = llvm.getelementptr inbounds %arg0[%0, 3, 0, 0, 0] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.tutorial::Person_PhoneNumber", (struct<"class.google::protobuf::Message", (struct<"class.google::protobuf::MessageLite", (ptr, struct<"class.google::protobuf::internal::InternalMetadata", (ptr)>)>)>, struct<"struct.google::protobuf::internal::ArenaStringPtr", (ptr)>, i32, struct<"class.google::protobuf::internal::CachedSize", (struct<"struct.std::atomic", (struct<"struct.std::__atomic_base", (i32)>)>)>)>
    %4 = llvm.load %3 atomic monotonic {alignment = 4 : i64} : !llvm.ptr -> i32
    llvm.return %4 : i32
  }
  llvm.func linkonce_odr unnamed_addr @_ZNK6google8protobuf11MessageLite16InternalGetTableEv(%arg0: !llvm.ptr {llvm.align = 8 : i64, llvm.dereferenceable = 16 : i64, llvm.nonnull, llvm.noundef}) -> (!llvm.ptr {llvm.noundef}) comdat(@__llvm_global_comdat::@_ZNK6google8protobuf11MessageLite16InternalGetTableEv) attributes {alignment = 2 : i64, frame_pointer = #llvm.framePointerKind<none>, passthrough = ["mustprogress", "nounwind", ["uwtable", "2"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], ["tune-cpu", "generic"]], target_cpu = "x86-64", target_features = #llvm.target_features<["+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>} {
    %0 = llvm.mlir.zero : !llvm.ptr
    llvm.return %0 : !llvm.ptr
  }
  llvm.func unnamed_addr @_ZN6google8protobuf7Message20DiscardUnknownFieldsEv(!llvm.ptr {llvm.align = 8 : i64, llvm.dereferenceable = 16 : i64, llvm.nonnull, llvm.noundef}) attributes {frame_pointer = #llvm.framePointerKind<none>, passthrough = [["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], ["tune-cpu", "generic"]], target_cpu = "x86-64", target_features = #llvm.target_features<["+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>}
  llvm.func unnamed_addr @_ZNK6google8protobuf7Message13SpaceUsedLongEv(!llvm.ptr {llvm.align = 8 : i64, llvm.dereferenceable = 16 : i64, llvm.nonnull, llvm.noundef}) -> (i64 {llvm.noundef}) attributes {frame_pointer = #llvm.framePointerKind<none>, passthrough = [["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], ["tune-cpu", "generic"]], target_cpu = "x86-64", target_features = #llvm.target_features<["+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>}
  llvm.func linkonce_odr unnamed_addr @_ZNK8tutorial6Person3NewEv(%arg0: !llvm.ptr {llvm.align = 8 : i64, llvm.dereferenceable = 72 : i64, llvm.nonnull, llvm.noundef}) -> (!llvm.ptr {llvm.noundef}) comdat(@__llvm_global_comdat::@_ZNK8tutorial6Person3NewEv) attributes {alignment = 2 : i64, frame_pointer = #llvm.framePointerKind<none>, passthrough = ["inlinehint", "mustprogress", ["uwtable", "2"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], ["tune-cpu", "generic"]], target_cpu = "x86-64", target_features = #llvm.target_features<["+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>} {
    %0 = llvm.mlir.zero : !llvm.ptr
    %1 = llvm.call @_ZN6google8protobuf5Arena18CreateMaybeMessageIN8tutorial6PersonEJEEEPT_PS1_DpOT0_(%0) : (!llvm.ptr) -> !llvm.ptr
    llvm.return %1 : !llvm.ptr
  }
  llvm.func linkonce_odr unnamed_addr @_ZNK8tutorial6Person3NewEPN6google8protobuf5ArenaE(%arg0: !llvm.ptr {llvm.align = 8 : i64, llvm.dereferenceable = 72 : i64, llvm.nonnull, llvm.noundef}, %arg1: !llvm.ptr {llvm.noundef}) -> (!llvm.ptr {llvm.noundef}) comdat(@__llvm_global_comdat::@_ZNK8tutorial6Person3NewEPN6google8protobuf5ArenaE) attributes {alignment = 2 : i64, frame_pointer = #llvm.framePointerKind<none>, passthrough = ["mustprogress", ["uwtable", "2"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], ["tune-cpu", "generic"]], target_cpu = "x86-64", target_features = #llvm.target_features<["+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>} {
    %0 = llvm.call @_ZN6google8protobuf5Arena18CreateMaybeMessageIN8tutorial6PersonEJEEEPT_PS1_DpOT0_(%arg1) : (!llvm.ptr) -> !llvm.ptr
    llvm.return %0 : !llvm.ptr
  }
  llvm.func linkonce_odr unnamed_addr @_ZNK8tutorial6Person13GetCachedSizeEv(%arg0: !llvm.ptr {llvm.align = 8 : i64, llvm.dereferenceable = 72 : i64, llvm.nonnull, llvm.noundef}) -> (i32 {llvm.noundef}) comdat(@__llvm_global_comdat::@_ZNK8tutorial6Person13GetCachedSizeEv) attributes {alignment = 2 : i64, frame_pointer = #llvm.framePointerKind<none>, passthrough = ["mustprogress", "nounwind", ["uwtable", "2"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], ["tune-cpu", "generic"]], target_cpu = "x86-64", target_features = #llvm.target_features<["+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>} {
    %0 = llvm.mlir.constant(0 : i64) : i64
    %1 = llvm.mlir.constant(6 : i32) : i32
    %2 = llvm.mlir.constant(0 : i32) : i32
    %3 = llvm.getelementptr inbounds %arg0[%0, 6, 0, 0, 0] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.tutorial::Person", (struct<"class.google::protobuf::Message", (struct<"class.google::protobuf::MessageLite", (ptr, struct<"class.google::protobuf::internal::InternalMetadata", (ptr)>)>)>, struct<"class.google::protobuf::RepeatedPtrField", (struct<"class.google::protobuf::internal::RepeatedPtrFieldBase", (ptr, i32, i32, ptr)>)>, struct<"struct.google::protobuf::internal::ArenaStringPtr", (ptr)>, struct<"struct.google::protobuf::internal::ArenaStringPtr", (ptr)>, ptr, i32, struct<"class.google::protobuf::internal::CachedSize", (struct<"struct.std::atomic", (struct<"struct.std::__atomic_base", (i32)>)>)>)>
    %4 = llvm.load %3 atomic monotonic {alignment = 4 : i64} : !llvm.ptr -> i32
    llvm.return %4 : i32
  }
  llvm.func linkonce_odr unnamed_addr @_ZNK8tutorial11AddressBook3NewEv(%arg0: !llvm.ptr {llvm.align = 8 : i64, llvm.dereferenceable = 48 : i64, llvm.nonnull, llvm.noundef}) -> (!llvm.ptr {llvm.noundef}) comdat(@__llvm_global_comdat::@_ZNK8tutorial11AddressBook3NewEv) attributes {alignment = 2 : i64, frame_pointer = #llvm.framePointerKind<none>, passthrough = ["inlinehint", "mustprogress", ["uwtable", "2"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], ["tune-cpu", "generic"]], target_cpu = "x86-64", target_features = #llvm.target_features<["+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>} {
    %0 = llvm.mlir.zero : !llvm.ptr
    %1 = llvm.call @_ZN6google8protobuf5Arena18CreateMaybeMessageIN8tutorial11AddressBookEJEEEPT_PS1_DpOT0_(%0) : (!llvm.ptr) -> !llvm.ptr
    llvm.return %1 : !llvm.ptr
  }
  llvm.func linkonce_odr unnamed_addr @_ZNK8tutorial11AddressBook3NewEPN6google8protobuf5ArenaE(%arg0: !llvm.ptr {llvm.align = 8 : i64, llvm.dereferenceable = 48 : i64, llvm.nonnull, llvm.noundef}, %arg1: !llvm.ptr {llvm.noundef}) -> (!llvm.ptr {llvm.noundef}) comdat(@__llvm_global_comdat::@_ZNK8tutorial11AddressBook3NewEPN6google8protobuf5ArenaE) attributes {alignment = 2 : i64, frame_pointer = #llvm.framePointerKind<none>, passthrough = ["mustprogress", ["uwtable", "2"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], ["tune-cpu", "generic"]], target_cpu = "x86-64", target_features = #llvm.target_features<["+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>} {
    %0 = llvm.call @_ZN6google8protobuf5Arena18CreateMaybeMessageIN8tutorial11AddressBookEJEEEPT_PS1_DpOT0_(%arg1) : (!llvm.ptr) -> !llvm.ptr
    llvm.return %0 : !llvm.ptr
  }
  llvm.func linkonce_odr unnamed_addr @_ZNK8tutorial11AddressBook13GetCachedSizeEv(%arg0: !llvm.ptr {llvm.align = 8 : i64, llvm.dereferenceable = 48 : i64, llvm.nonnull, llvm.noundef}) -> (i32 {llvm.noundef}) comdat(@__llvm_global_comdat::@_ZNK8tutorial11AddressBook13GetCachedSizeEv) attributes {alignment = 2 : i64, frame_pointer = #llvm.framePointerKind<none>, passthrough = ["mustprogress", "nounwind", ["uwtable", "2"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], ["tune-cpu", "generic"]], target_cpu = "x86-64", target_features = #llvm.target_features<["+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>} {
    %0 = llvm.mlir.constant(0 : i64) : i64
    %1 = llvm.mlir.constant(2 : i32) : i32
    %2 = llvm.mlir.constant(0 : i32) : i32
    %3 = llvm.getelementptr inbounds %arg0[%0, 2, 0, 0, 0] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.tutorial::AddressBook", packed (struct<"class.google::protobuf::Message", (struct<"class.google::protobuf::MessageLite", (ptr, struct<"class.google::protobuf::internal::InternalMetadata", (ptr)>)>)>, struct<"class.google::protobuf::RepeatedPtrField.18", (struct<"class.google::protobuf::internal::RepeatedPtrFieldBase", (ptr, i32, i32, ptr)>)>, struct<"class.google::protobuf::internal::CachedSize", (struct<"struct.std::atomic", (struct<"struct.std::__atomic_base", (i32)>)>)>, array<4 x i8>)>
    %4 = llvm.load %3 atomic monotonic {alignment = 8 : i64} : !llvm.ptr -> i32
    llvm.return %4 : i32
  }
  llvm.func local_unnamed_addr @_ZN6google8protobuf8internal13VerifyVersionEiiPKc(i32 {llvm.noundef}, i32 {llvm.noundef}, !llvm.ptr {llvm.noundef}) attributes {frame_pointer = #llvm.framePointerKind<none>, passthrough = [["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], ["tune-cpu", "generic"]], target_cpu = "x86-64", target_features = #llvm.target_features<["+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>}
  llvm.func local_unnamed_addr @_ZN6google8protobuf8internal13OnShutdownRunEPFvPKvES3_(!llvm.ptr {llvm.noundef}, !llvm.ptr {llvm.noundef}) attributes {frame_pointer = #llvm.framePointerKind<none>, passthrough = [["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], ["tune-cpu", "generic"]], target_cpu = "x86-64", target_features = #llvm.target_features<["+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>}
  llvm.func @_ZN6google8protobuf8internal14DestroyMessageEPKv(!llvm.ptr {llvm.noundef}) attributes {frame_pointer = #llvm.framePointerKind<none>, passthrough = [["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], ["tune-cpu", "generic"]], target_cpu = "x86-64", target_features = #llvm.target_features<["+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>}
  llvm.func linkonce_odr local_unnamed_addr @_ZN6google8protobuf8internal14ArenaStringPtr14CreateInstanceEPNS0_5ArenaEPKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE(%arg0: !llvm.ptr {llvm.align = 8 : i64, llvm.dereferenceable = 8 : i64, llvm.nonnull, llvm.noundef}, %arg1: !llvm.ptr {llvm.noundef}, %arg2: !llvm.ptr {llvm.noundef}) comdat(@__llvm_global_comdat::@_ZN6google8protobuf8internal14ArenaStringPtr14CreateInstanceEPNS0_5ArenaEPKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE) attributes {alignment = 2 : i64, frame_pointer = #llvm.framePointerKind<none>, passthrough = ["noinline", ["uwtable", "2"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], ["tune-cpu", "generic"]], personality = @__gxx_personality_v0, target_cpu = "x86-64", target_features = #llvm.target_features<["+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>} {
    %0 = llvm.mlir.constant(1 : i32) : i32
    %1 = llvm.mlir.zero : !llvm.ptr
    %2 = llvm.mlir.constant(0 : i64) : i64
    %3 = llvm.mlir.constant(0 : i32) : i32
    %4 = llvm.mlir.constant(3 : i32) : i32
    %5 = llvm.mlir.constant("/usr/include/google/protobuf/arenastring.h\00") : !llvm.array<43 x i8>
    %6 = llvm.mlir.addressof @".str.9" : !llvm.ptr
    %7 = llvm.mlir.constant(371 : i32) : i32
    %8 = llvm.mlir.constant("CHECK failed: initial_value != NULL: \00") : !llvm.array<38 x i8>
    %9 = llvm.mlir.addressof @".str.10" : !llvm.ptr
    %10 = llvm.mlir.constant(4 : i32) : i32
    %11 = llvm.mlir.constant("NSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE\00") : !llvm.array<53 x i8>
    %12 = llvm.mlir.addressof @_ZTSNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE : !llvm.ptr
    %13 = llvm.mlir.constant(2 : i64) : i64
    %14 = llvm.mlir.addressof @_ZTVN10__cxxabiv117__class_type_infoE : !llvm.ptr
    %15 = llvm.getelementptr inbounds %14[%13] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.ptr
    %16 = llvm.mlir.undef : !llvm.struct<(ptr, ptr)>
    %17 = llvm.insertvalue %15, %16[0] : !llvm.struct<(ptr, ptr)> 
    %18 = llvm.insertvalue %12, %17[1] : !llvm.struct<(ptr, ptr)> 
    %19 = llvm.mlir.addressof @_ZTINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE : !llvm.ptr
    %20 = llvm.mlir.constant(32 : i64) : i64
    %21 = llvm.mlir.addressof @__gxx_personality_v0 : !llvm.ptr
    %22 = llvm.mlir.addressof @_ZN6google8protobuf8internal21arena_destruct_objectINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEEEvPv : !llvm.ptr
    %23 = llvm.mlir.constant(16 : i64) : i64
    %24 = llvm.mlir.constant(15 : i64) : i64
    %25 = llvm.mlir.constant(8 : i64) : i64
    %26 = llvm.mlir.constant(0 : i8) : i8
    %27 = llvm.mlir.constant(2 : i32) : i32
    %28 = llvm.alloca %0 x i64 {alignment = 8 : i64} : (i32) -> !llvm.ptr
    %29 = llvm.alloca %0 x i64 {alignment = 8 : i64} : (i32) -> !llvm.ptr
    %30 = llvm.alloca %0 x !llvm.struct<"class.google::protobuf::internal::LogMessage", (i32, ptr, i32, struct<"class.std::__cxx11::basic_string", (struct<"struct.std::__cxx11::basic_string<char>::_Alloc_hider", (ptr)>, i64, struct<"union.anon", (i64, array<8 x i8>)>)>)> {alignment = 8 : i64} : (i32) -> !llvm.ptr
    %31 = llvm.alloca %0 x !llvm.struct<"class.google::protobuf::internal::LogFinisher", (i8)> {alignment = 1 : i64} : (i32) -> !llvm.ptr
    %32 = llvm.icmp "eq" %arg2, %1 : !llvm.ptr
    %33 = llvm.getelementptr inbounds %31[%2, 0] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.google::protobuf::internal::LogFinisher", (i8)>
    llvm.intr.lifetime.start 1, %33 : !llvm.ptr
    llvm.cond_br %32, ^bb1, ^bb3
  ^bb1:  // pred: ^bb0
    %34 = llvm.bitcast %30 : !llvm.ptr to !llvm.ptr
    llvm.intr.lifetime.start 56, %34 : !llvm.ptr
    llvm.call @_ZN6google8protobuf8internal10LogMessageC1ENS0_8LogLevelEPKci(%30, %4, %6, %7) : (!llvm.ptr, i32, !llvm.ptr, i32) -> ()
    %35 = llvm.invoke @_ZN6google8protobuf8internal10LogMessagelsEPKc(%30, %9) to ^bb2 unwind ^bb25 : (!llvm.ptr, !llvm.ptr) -> !llvm.ptr
  ^bb2:  // pred: ^bb1
    llvm.invoke @_ZN6google8protobuf8internal11LogFinisheraSERNS1_10LogMessageE(%31, %35) to ^bb4 unwind ^bb26 : (!llvm.ptr, !llvm.ptr) -> ()
  ^bb3:  // pred: ^bb0
    llvm.intr.lifetime.end 1, %33 : !llvm.ptr
    llvm.br ^bb5
  ^bb4:  // pred: ^bb2
    llvm.intr.lifetime.end 1, %33 : !llvm.ptr
    llvm.call @_ZN6google8protobuf8internal10LogMessageD1Ev(%30) : (!llvm.ptr) -> ()
    llvm.intr.lifetime.end 56, %34 : !llvm.ptr
    llvm.br ^bb5
  ^bb5:  // 2 preds: ^bb3, ^bb4
    %36 = llvm.icmp "eq" %arg1, %1 : !llvm.ptr
    llvm.cond_br %36, ^bb6, ^bb16
  ^bb6:  // pred: ^bb5
    %37 = llvm.call @_Znwm(%20) : (i64) -> !llvm.ptr
    %38 = llvm.bitcast %37 : !llvm.ptr to !llvm.ptr
    %39 = llvm.getelementptr inbounds %38[%2, 2] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.std::__cxx11::basic_string", (struct<"struct.std::__cxx11::basic_string<char>::_Alloc_hider", (ptr)>, i64, struct<"union.anon", (i64, array<8 x i8>)>)>
    %40 = llvm.bitcast %37 : !llvm.ptr to !llvm.ptr
    llvm.store %39, %40 {alignment = 8 : i64, tbaa = [#tbaa_tag22]} : !llvm.ptr, !llvm.ptr
    %41 = llvm.getelementptr inbounds %arg2[%2, 0, 0] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.std::__cxx11::basic_string", (struct<"struct.std::__cxx11::basic_string<char>::_Alloc_hider", (ptr)>, i64, struct<"union.anon", (i64, array<8 x i8>)>)>
    %42 = llvm.load %41 {alignment = 8 : i64, tbaa = [#tbaa_tag26]} : !llvm.ptr -> !llvm.ptr
    %43 = llvm.getelementptr inbounds %arg2[%2, 1] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.std::__cxx11::basic_string", (struct<"struct.std::__cxx11::basic_string<char>::_Alloc_hider", (ptr)>, i64, struct<"union.anon", (i64, array<8 x i8>)>)>
    %44 = llvm.load %43 {alignment = 8 : i64, tbaa = [#tbaa_tag25]} : !llvm.ptr -> i64
    %45 = llvm.bitcast %28 : !llvm.ptr to !llvm.ptr
    llvm.intr.lifetime.start 8, %45 : !llvm.ptr
    llvm.store %44, %28 {alignment = 8 : i64, tbaa = [#tbaa_tag4]} : i64, !llvm.ptr
    %46 = llvm.icmp "ugt" %44, %24 : i64
    llvm.cond_br %46, ^bb8, ^bb7
  ^bb7:  // pred: ^bb6
    %47 = llvm.bitcast %39 : !llvm.ptr to !llvm.ptr
    llvm.br ^bb10(%47 : !llvm.ptr)
  ^bb8:  // pred: ^bb6
    %48 = llvm.invoke @_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE9_M_createERmm(%38, %28, %2) to ^bb9 unwind ^bb15 : (!llvm.ptr, !llvm.ptr, i64) -> !llvm.ptr
  ^bb9:  // pred: ^bb8
    %49 = llvm.getelementptr inbounds %38[%2, 0, 0] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.std::__cxx11::basic_string", (struct<"struct.std::__cxx11::basic_string<char>::_Alloc_hider", (ptr)>, i64, struct<"union.anon", (i64, array<8 x i8>)>)>
    llvm.store %48, %49 {alignment = 8 : i64, tbaa = [#tbaa_tag26]} : !llvm.ptr, !llvm.ptr
    %50 = llvm.load %28 {alignment = 8 : i64, tbaa = [#tbaa_tag4]} : !llvm.ptr -> i64
    %51 = llvm.getelementptr inbounds %38[%2, 2, 0] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.std::__cxx11::basic_string", (struct<"struct.std::__cxx11::basic_string<char>::_Alloc_hider", (ptr)>, i64, struct<"union.anon", (i64, array<8 x i8>)>)>
    llvm.store %50, %51 {alignment = 8 : i64, tbaa = [#tbaa_tag1]} : i64, !llvm.ptr
    llvm.br ^bb10(%48 : !llvm.ptr)
  ^bb10(%52: !llvm.ptr):  // 2 preds: ^bb7, ^bb9
    llvm.switch %44 : i64, ^bb12 [
      1: ^bb11,
      0: ^bb13
    ]
  ^bb11:  // pred: ^bb10
    %53 = llvm.load %42 {alignment = 1 : i64, tbaa = [#tbaa_tag1]} : !llvm.ptr -> i8
    llvm.store %53, %52 {alignment = 1 : i64, tbaa = [#tbaa_tag1]} : i8, !llvm.ptr
    llvm.br ^bb13
  ^bb12:  // pred: ^bb10
    "llvm.intr.memcpy"(%52, %42, %44) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
    llvm.br ^bb13
  ^bb13:  // 3 preds: ^bb10, ^bb11, ^bb12
    %54 = llvm.getelementptr inbounds %38[%2, 0, 0] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.std::__cxx11::basic_string", (struct<"struct.std::__cxx11::basic_string<char>::_Alloc_hider", (ptr)>, i64, struct<"union.anon", (i64, array<8 x i8>)>)>
    %55 = llvm.load %28 {alignment = 8 : i64, tbaa = [#tbaa_tag4]} : !llvm.ptr -> i64
    %56 = llvm.getelementptr inbounds %38[%2, 1] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.std::__cxx11::basic_string", (struct<"struct.std::__cxx11::basic_string<char>::_Alloc_hider", (ptr)>, i64, struct<"union.anon", (i64, array<8 x i8>)>)>
    llvm.store %55, %56 {alignment = 8 : i64, tbaa = [#tbaa_tag25]} : i64, !llvm.ptr
    %57 = llvm.load %54 {alignment = 8 : i64, tbaa = [#tbaa_tag26]} : !llvm.ptr -> !llvm.ptr
    %58 = llvm.getelementptr inbounds %57[%55] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    llvm.store %26, %58 {alignment = 1 : i64, tbaa = [#tbaa_tag1]} : i8, !llvm.ptr
    llvm.intr.lifetime.end 8, %45 : !llvm.ptr
    llvm.br ^bb24(%38 : !llvm.ptr)
  ^bb14(%59: !llvm.struct<(ptr, i32)>):  // 2 preds: ^bb15, ^bb27
    llvm.resume %59 : !llvm.struct<(ptr, i32)>
  ^bb15:  // pred: ^bb8
    %60 = llvm.landingpad cleanup : !llvm.struct<(ptr, i32)>
    llvm.call @_ZdlPv(%37) : (!llvm.ptr) -> ()
    llvm.br ^bb14(%60 : !llvm.struct<(ptr, i32)>)
  ^bb16:  // pred: ^bb5
    %61 = llvm.getelementptr inbounds %arg1[%2, 4] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.google::protobuf::Arena", (struct<"class.google::protobuf::internal::ArenaImpl", (struct<"struct.std::atomic.10", (struct<"struct.std::__atomic_base.11", (ptr)>)>, struct<"struct.std::atomic.10", (struct<"struct.std::__atomic_base.11", (ptr)>)>, struct<"struct.std::atomic.12", (struct<"struct.std::__atomic_base.13", (i64)>)>, ptr, i64, struct<"struct.google::protobuf::internal::ArenaImpl::Options", (i64, i64, ptr, i64, ptr, ptr)>)>, ptr, ptr, ptr, ptr)>
    %62 = llvm.load %61 {alignment = 8 : i64, tbaa = [#tbaa_tag30]} : !llvm.ptr -> !llvm.ptr
    %63 = llvm.icmp "eq" %62, %1 : !llvm.ptr
    llvm.cond_br %63 weights([2000, 1]), ^bb18, ^bb17
  ^bb17:  // pred: ^bb16
    llvm.call @_ZNK6google8protobuf5Arena17OnArenaAllocationEPKSt9type_infom(%arg1, %19, %20) : (!llvm.ptr, !llvm.ptr, i64) -> ()
    llvm.br ^bb18
  ^bb18:  // 2 preds: ^bb16, ^bb17
    %64 = llvm.getelementptr inbounds %arg1[%2, 0] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.google::protobuf::Arena", (struct<"class.google::protobuf::internal::ArenaImpl", (struct<"struct.std::atomic.10", (struct<"struct.std::__atomic_base.11", (ptr)>)>, struct<"struct.std::atomic.10", (struct<"struct.std::__atomic_base.11", (ptr)>)>, struct<"struct.std::atomic.12", (struct<"struct.std::__atomic_base.13", (i64)>)>, ptr, i64, struct<"struct.google::protobuf::internal::ArenaImpl::Options", (i64, i64, ptr, i64, ptr, ptr)>)>, ptr, ptr, ptr, ptr)>
    %65 = llvm.call @_ZN6google8protobuf8internal9ArenaImpl28AllocateAlignedAndAddCleanupEmPFvPvE(%64, %20, %22) : (!llvm.ptr, i64, !llvm.ptr) -> !llvm.ptr
    %66 = llvm.bitcast %65 : !llvm.ptr to !llvm.ptr
    %67 = llvm.getelementptr inbounds %65[%23] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %68 = llvm.bitcast %65 : !llvm.ptr to !llvm.ptr
    llvm.store %67, %68 {alignment = 8 : i64, tbaa = [#tbaa_tag22]} : !llvm.ptr, !llvm.ptr
    %69 = llvm.getelementptr inbounds %arg2[%2, 0, 0] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.std::__cxx11::basic_string", (struct<"struct.std::__cxx11::basic_string<char>::_Alloc_hider", (ptr)>, i64, struct<"union.anon", (i64, array<8 x i8>)>)>
    %70 = llvm.load %69 {alignment = 8 : i64, tbaa = [#tbaa_tag26]} : !llvm.ptr -> !llvm.ptr
    %71 = llvm.getelementptr inbounds %arg2[%2, 1] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.std::__cxx11::basic_string", (struct<"struct.std::__cxx11::basic_string<char>::_Alloc_hider", (ptr)>, i64, struct<"union.anon", (i64, array<8 x i8>)>)>
    %72 = llvm.load %71 {alignment = 8 : i64, tbaa = [#tbaa_tag25]} : !llvm.ptr -> i64
    %73 = llvm.bitcast %29 : !llvm.ptr to !llvm.ptr
    llvm.intr.lifetime.start 8, %73 : !llvm.ptr
    llvm.store %72, %29 {alignment = 8 : i64, tbaa = [#tbaa_tag4]} : i64, !llvm.ptr
    %74 = llvm.icmp "ugt" %72, %24 : i64
    llvm.cond_br %74, ^bb19, ^bb20(%67 : !llvm.ptr)
  ^bb19:  // pred: ^bb18
    %75 = llvm.call @_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE9_M_createERmm(%66, %29, %2) : (!llvm.ptr, !llvm.ptr, i64) -> !llvm.ptr
    llvm.store %75, %68 {alignment = 8 : i64, tbaa = [#tbaa_tag26]} : !llvm.ptr, !llvm.ptr
    %76 = llvm.load %29 {alignment = 8 : i64, tbaa = [#tbaa_tag4]} : !llvm.ptr -> i64
    %77 = llvm.bitcast %67 : !llvm.ptr to !llvm.ptr
    llvm.store %76, %77 {alignment = 8 : i64, tbaa = [#tbaa_tag1]} : i64, !llvm.ptr
    llvm.br ^bb20(%75 : !llvm.ptr)
  ^bb20(%78: !llvm.ptr):  // 2 preds: ^bb18, ^bb19
    llvm.switch %72 : i64, ^bb22 [
      1: ^bb21,
      0: ^bb23
    ]
  ^bb21:  // pred: ^bb20
    %79 = llvm.load %70 {alignment = 1 : i64, tbaa = [#tbaa_tag1]} : !llvm.ptr -> i8
    llvm.store %79, %78 {alignment = 1 : i64, tbaa = [#tbaa_tag1]} : i8, !llvm.ptr
    llvm.br ^bb23
  ^bb22:  // pred: ^bb20
    "llvm.intr.memcpy"(%78, %70, %72) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
    llvm.br ^bb23
  ^bb23:  // 3 preds: ^bb20, ^bb21, ^bb22
    %80 = llvm.load %29 {alignment = 8 : i64, tbaa = [#tbaa_tag4]} : !llvm.ptr -> i64
    %81 = llvm.getelementptr inbounds %65[%25] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %82 = llvm.bitcast %81 : !llvm.ptr to !llvm.ptr
    llvm.store %80, %82 {alignment = 8 : i64, tbaa = [#tbaa_tag25]} : i64, !llvm.ptr
    %83 = llvm.load %68 {alignment = 8 : i64, tbaa = [#tbaa_tag26]} : !llvm.ptr -> !llvm.ptr
    %84 = llvm.getelementptr inbounds %83[%80] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    llvm.store %26, %84 {alignment = 1 : i64, tbaa = [#tbaa_tag1]} : i8, !llvm.ptr
    llvm.intr.lifetime.end 8, %73 : !llvm.ptr
    llvm.br ^bb24(%66 : !llvm.ptr)
  ^bb24(%85: !llvm.ptr):  // 2 preds: ^bb13, ^bb23
    %86 = llvm.getelementptr inbounds %arg0[%2, 0] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.google::protobuf::internal::ArenaStringPtr", (ptr)>
    llvm.store %85, %86 {alignment = 8 : i64, tbaa = [#tbaa_tag8]} : !llvm.ptr, !llvm.ptr
    llvm.return
  ^bb25:  // pred: ^bb1
    %87 = llvm.landingpad cleanup : !llvm.struct<(ptr, i32)>
    llvm.br ^bb27(%87 : !llvm.struct<(ptr, i32)>)
  ^bb26:  // pred: ^bb2
    %88 = llvm.landingpad cleanup : !llvm.struct<(ptr, i32)>
    llvm.intr.lifetime.end 1, %33 : !llvm.ptr
    llvm.br ^bb27(%88 : !llvm.struct<(ptr, i32)>)
  ^bb27(%89: !llvm.struct<(ptr, i32)>):  // 2 preds: ^bb25, ^bb26
    llvm.call @_ZN6google8protobuf8internal10LogMessageD1Ev(%30) : (!llvm.ptr) -> ()
    llvm.intr.lifetime.end 56, %34 : !llvm.ptr
    llvm.br ^bb14(%89 : !llvm.struct<(ptr, i32)>)
  }
  llvm.func local_unnamed_addr @_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE9_M_createERmm(!llvm.ptr {llvm.align = 8 : i64, llvm.dereferenceable = 32 : i64, llvm.nonnull, llvm.noundef}, !llvm.ptr {llvm.align = 8 : i64, llvm.dereferenceable = 8 : i64, llvm.nonnull, llvm.noundef}, i64 {llvm.noundef}) -> (!llvm.ptr {llvm.noundef}) attributes {frame_pointer = #llvm.framePointerKind<none>, passthrough = [["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], ["tune-cpu", "generic"]], target_cpu = "x86-64", target_features = #llvm.target_features<["+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>}
  llvm.func local_unnamed_addr @_ZN6google8protobuf5Arena21AllocateAlignedNoHookEm(!llvm.ptr {llvm.align = 8 : i64, llvm.dereferenceable = 120 : i64, llvm.nonnull, llvm.noundef}, i64 {llvm.noundef}) -> (!llvm.ptr {llvm.noundef}) attributes {frame_pointer = #llvm.framePointerKind<none>, passthrough = [["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], ["tune-cpu", "generic"]], target_cpu = "x86-64", target_features = #llvm.target_features<["+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>}
  llvm.func local_unnamed_addr @_ZN6google8protobuf8internal9ArenaImpl28AllocateAlignedAndAddCleanupEmPFvPvE(!llvm.ptr {llvm.align = 8 : i64, llvm.dereferenceable = 88 : i64, llvm.nonnull, llvm.noundef}, i64 {llvm.noundef}, !llvm.ptr {llvm.noundef}) -> (!llvm.ptr {llvm.noundef}) attributes {frame_pointer = #llvm.framePointerKind<none>, passthrough = [["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], ["tune-cpu", "generic"]], target_cpu = "x86-64", target_features = #llvm.target_features<["+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>}
  llvm.func linkonce_odr @_ZN6google8protobuf8internal21arena_destruct_objectINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEEEvPv(%arg0: !llvm.ptr {llvm.noundef}) comdat(@__llvm_global_comdat::@_ZN6google8protobuf8internal21arena_destruct_objectINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEEEvPv) attributes {frame_pointer = #llvm.framePointerKind<none>, passthrough = ["nounwind", ["uwtable", "2"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], ["tune-cpu", "generic"]], personality = @__gxx_personality_v0, target_cpu = "x86-64", target_features = #llvm.target_features<["+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>} {
    %0 = llvm.mlir.constant(16 : i64) : i64
    %1 = llvm.bitcast %arg0 : !llvm.ptr to !llvm.ptr
    %2 = llvm.load %1 {alignment = 8 : i64, tbaa = [#tbaa_tag26]} : !llvm.ptr -> !llvm.ptr
    %3 = llvm.getelementptr inbounds %arg0[%0] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %4 = llvm.icmp "eq" %2, %3 : !llvm.ptr
    llvm.cond_br %4, ^bb2, ^bb1
  ^bb1:  // pred: ^bb0
    llvm.call @_ZdlPv(%2) : (!llvm.ptr) -> ()
    llvm.br ^bb2
  ^bb2:  // 2 preds: ^bb0, ^bb1
    llvm.return
  }
  llvm.func local_unnamed_addr @_ZNK6google8protobuf5Arena17OnArenaAllocationEPKSt9type_infom(!llvm.ptr {llvm.align = 8 : i64, llvm.dereferenceable = 120 : i64, llvm.nonnull, llvm.noundef}, !llvm.ptr {llvm.noundef}, i64 {llvm.noundef}) attributes {frame_pointer = #llvm.framePointerKind<none>, passthrough = [["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], ["tune-cpu", "generic"]], target_cpu = "x86-64", target_features = #llvm.target_features<["+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>}
  llvm.func local_unnamed_addr @_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE9_M_assignERKS4_(!llvm.ptr {llvm.align = 8 : i64, llvm.dereferenceable = 32 : i64, llvm.nonnull, llvm.noundef}, !llvm.ptr {llvm.align = 8 : i64, llvm.dereferenceable = 32 : i64, llvm.nonnull, llvm.noundef}) attributes {frame_pointer = #llvm.framePointerKind<none>, passthrough = [["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], ["tune-cpu", "generic"]], target_cpu = "x86-64", target_features = #llvm.target_features<["+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>}
  llvm.func local_unnamed_addr @_ZN6google8protobuf8internal11InitSCCImplEPNS1_11SCCInfoBaseE(!llvm.ptr {llvm.noundef}) attributes {frame_pointer = #llvm.framePointerKind<none>, passthrough = [["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], ["tune-cpu", "generic"]], target_cpu = "x86-64", target_features = #llvm.target_features<["+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>}
  llvm.func linkonce_odr local_unnamed_addr @_ZN6google8protobuf8internal18EpsCopyInputStream13DoneWithCheckEPPKci(%arg0: !llvm.ptr {llvm.align = 8 : i64, llvm.dereferenceable = 88 : i64, llvm.nonnull, llvm.noundef}, %arg1: !llvm.ptr {llvm.noundef}, %arg2: i32 {llvm.noundef}) -> (i1 {llvm.noundef, llvm.zeroext}) comdat(@__llvm_global_comdat::@_ZN6google8protobuf8internal18EpsCopyInputStream13DoneWithCheckEPPKci) attributes {alignment = 2 : i64, frame_pointer = #llvm.framePointerKind<none>, passthrough = ["mustprogress", ["uwtable", "2"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], ["tune-cpu", "generic"]], personality = @__gxx_personality_v0, target_cpu = "x86-64", target_features = #llvm.target_features<["+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>} {
    %0 = llvm.mlir.constant(1 : i32) : i32
    %1 = llvm.mlir.zero : !llvm.ptr
    %2 = llvm.mlir.constant(0 : i64) : i64
    %3 = llvm.mlir.constant(0 : i32) : i32
    %4 = llvm.mlir.constant(3 : i32) : i32
    %5 = llvm.mlir.constant("/usr/include/google/protobuf/parse_context.h\00") : !llvm.array<45 x i8>
    %6 = llvm.mlir.addressof @".str.13" : !llvm.ptr
    %7 = llvm.mlir.constant(209 : i32) : i32
    %8 = llvm.mlir.constant("CHECK failed: *ptr: \00") : !llvm.array<21 x i8>
    %9 = llvm.mlir.addressof @".str.14" : !llvm.ptr
    %10 = llvm.mlir.constant(false) : i1
    %11 = llvm.mlir.constant(4 : i32) : i32
    %12 = llvm.mlir.constant(true) : i1
    %13 = llvm.mlir.constant(1 : i8) : i8
    %14 = llvm.mlir.constant(0 : i8) : i8
    %15 = llvm.alloca %0 x !llvm.struct<"class.google::protobuf::internal::LogMessage", (i32, ptr, i32, struct<"class.std::__cxx11::basic_string", (struct<"struct.std::__cxx11::basic_string<char>::_Alloc_hider", (ptr)>, i64, struct<"union.anon", (i64, array<8 x i8>)>)>)> {alignment = 8 : i64} : (i32) -> !llvm.ptr
    %16 = llvm.alloca %0 x !llvm.struct<"class.google::protobuf::internal::LogFinisher", (i8)> {alignment = 1 : i64} : (i32) -> !llvm.ptr
    %17 = llvm.load %arg1 {alignment = 8 : i64, tbaa = [#tbaa_tag2]} : !llvm.ptr -> !llvm.ptr
    %18 = llvm.icmp "eq" %17, %1 : !llvm.ptr
    %19 = llvm.getelementptr inbounds %16[%2, 0] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.google::protobuf::internal::LogFinisher", (i8)>
    llvm.intr.lifetime.start 1, %19 : !llvm.ptr
    llvm.cond_br %18, ^bb1, ^bb3
  ^bb1:  // pred: ^bb0
    %20 = llvm.bitcast %15 : !llvm.ptr to !llvm.ptr
    llvm.intr.lifetime.start 56, %20 : !llvm.ptr
    llvm.call @_ZN6google8protobuf8internal10LogMessageC1ENS0_8LogLevelEPKci(%15, %4, %6, %7) : (!llvm.ptr, i32, !llvm.ptr, i32) -> ()
    %21 = llvm.invoke @_ZN6google8protobuf8internal10LogMessagelsEPKc(%15, %9) to ^bb2 unwind ^bb6 : (!llvm.ptr, !llvm.ptr) -> !llvm.ptr
  ^bb2:  // pred: ^bb1
    llvm.invoke @_ZN6google8protobuf8internal11LogFinisheraSERNS1_10LogMessageE(%16, %21) to ^bb4 unwind ^bb7 : (!llvm.ptr, !llvm.ptr) -> ()
  ^bb3:  // pred: ^bb0
    llvm.intr.lifetime.end 1, %19 : !llvm.ptr
    llvm.br ^bb5(%17 : !llvm.ptr)
  ^bb4:  // pred: ^bb2
    llvm.intr.lifetime.end 1, %19 : !llvm.ptr
    llvm.call @_ZN6google8protobuf8internal10LogMessageD1Ev(%15) : (!llvm.ptr) -> ()
    llvm.intr.lifetime.end 56, %20 : !llvm.ptr
    %22 = llvm.load %arg1 {alignment = 8 : i64, tbaa = [#tbaa_tag2]} : !llvm.ptr -> !llvm.ptr
    llvm.br ^bb5(%22 : !llvm.ptr)
  ^bb5(%23: !llvm.ptr):  // 2 preds: ^bb3, ^bb4
    %24 = llvm.getelementptr inbounds %arg0[%2, 0] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.google::protobuf::internal::EpsCopyInputStream", (ptr, ptr, ptr, i32, i32, ptr, array<32 x i8>, i64, i32, i32)>
    %25 = llvm.load %24 {alignment = 8 : i64, tbaa = [#tbaa_tag21]} : !llvm.ptr -> !llvm.ptr
    %26 = llvm.icmp "ult" %23, %25 : !llvm.ptr
    llvm.cond_br %26 weights([2000, 1]), ^bb11(%10 : i1), ^bb9
  ^bb6:  // pred: ^bb1
    %27 = llvm.landingpad cleanup : !llvm.struct<(ptr, i32)>
    llvm.br ^bb8(%27 : !llvm.struct<(ptr, i32)>)
  ^bb7:  // pred: ^bb2
    %28 = llvm.landingpad cleanup : !llvm.struct<(ptr, i32)>
    llvm.intr.lifetime.end 1, %19 : !llvm.ptr
    llvm.br ^bb8(%28 : !llvm.struct<(ptr, i32)>)
  ^bb8(%29: !llvm.struct<(ptr, i32)>):  // 2 preds: ^bb6, ^bb7
    llvm.call @_ZN6google8protobuf8internal10LogMessageD1Ev(%15) : (!llvm.ptr) -> ()
    llvm.intr.lifetime.end 56, %20 : !llvm.ptr
    llvm.resume %29 : !llvm.struct<(ptr, i32)>
  ^bb9:  // pred: ^bb5
    %30 = llvm.getelementptr inbounds %arg0[%2, 1] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.google::protobuf::internal::EpsCopyInputStream", (ptr, ptr, ptr, i32, i32, ptr, array<32 x i8>, i64, i32, i32)>
    %31 = llvm.load %30 {alignment = 8 : i64, tbaa = [#tbaa_tag20]} : !llvm.ptr -> !llvm.ptr
    %32 = llvm.ptrtoint %23 : !llvm.ptr to i64
    %33 = llvm.ptrtoint %31 : !llvm.ptr to i64
    %34 = llvm.sub %32, %33  : i64
    %35 = llvm.getelementptr inbounds %arg0[%2, 4] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.google::protobuf::internal::EpsCopyInputStream", (ptr, ptr, ptr, i32, i32, ptr, array<32 x i8>, i64, i32, i32)>
    %36 = llvm.load %35 {alignment = 4 : i64, tbaa = [#tbaa_tag19]} : !llvm.ptr -> i32
    %37 = llvm.sext %36 : i32 to i64
    %38 = llvm.icmp "eq" %34, %37 : i64
    llvm.cond_br %38, ^bb11(%12 : i1), ^bb10
  ^bb10:  // pred: ^bb9
    %39 = llvm.call @_ZN6google8protobuf8internal18EpsCopyInputStream12DoneFallbackEPKci(%arg0, %23, %arg2) : (!llvm.ptr, !llvm.ptr, i32) -> !llvm.struct<(ptr, i8)>
    %40 = llvm.extractvalue %39[0] : !llvm.struct<(ptr, i8)> 
    %41 = llvm.extractvalue %39[1] : !llvm.struct<(ptr, i8)> 
    llvm.store %40, %arg1 {alignment = 8 : i64, tbaa = [#tbaa_tag2]} : !llvm.ptr, !llvm.ptr
    %42 = llvm.and %41, %13  : i8
    %43 = llvm.icmp "ne" %42, %14 : i8
    llvm.br ^bb11(%43 : i1)
  ^bb11(%44: i1):  // 3 preds: ^bb5, ^bb9, ^bb10
    llvm.return %44 : i1
  }
  llvm.func local_unnamed_addr @_ZN6google8protobuf8internal18EpsCopyInputStream12DoneFallbackEPKci(!llvm.ptr {llvm.align = 8 : i64, llvm.dereferenceable = 88 : i64, llvm.nonnull, llvm.noundef}, !llvm.ptr {llvm.noundef}, i32 {llvm.noundef}) -> !llvm.struct<(ptr, i8)> attributes {frame_pointer = #llvm.framePointerKind<none>, passthrough = [["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], ["tune-cpu", "generic"]], target_cpu = "x86-64", target_features = #llvm.target_features<["+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>}
  llvm.func local_unnamed_addr @_ZN6google8protobuf8internal15ReadTagFallbackEPKcj(!llvm.ptr {llvm.noundef}, i32 {llvm.noundef}) -> !llvm.struct<(ptr, i32)> attributes {frame_pointer = #llvm.framePointerKind<none>, passthrough = [["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], ["tune-cpu", "generic"]], target_cpu = "x86-64", target_features = #llvm.target_features<["+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>}
  llvm.func local_unnamed_addr @_ZN6google8protobuf8internal10VerifyUTF8ENS0_11StringPieceEPKc(!llvm.ptr, i64, !llvm.ptr {llvm.noundef}) -> (i1 {llvm.noundef, llvm.zeroext}) attributes {frame_pointer = #llvm.framePointerKind<none>, passthrough = [["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], ["tune-cpu", "generic"]], target_cpu = "x86-64", target_features = #llvm.target_features<["+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>}
  llvm.func local_unnamed_addr @_ZN6google8protobuf11StringPiece18LogFatalSizeTooBigEmPKc(i64 {llvm.noundef}, !llvm.ptr {llvm.noundef}) attributes {frame_pointer = #llvm.framePointerKind<none>, passthrough = [["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], ["tune-cpu", "generic"]], target_cpu = "x86-64", target_features = #llvm.target_features<["+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>}
  llvm.func local_unnamed_addr @_ZN6google8protobuf8internal17VarintParseSlow64EPKcj(!llvm.ptr {llvm.noundef}, i32 {llvm.noundef}) -> !llvm.struct<(ptr, i64)> attributes {frame_pointer = #llvm.framePointerKind<none>, passthrough = [["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], ["tune-cpu", "generic"]], target_cpu = "x86-64", target_features = #llvm.target_features<["+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>}
  llvm.func local_unnamed_addr @_ZN6google8protobuf2io19EpsCopyOutputStream30WriteStringMaybeAliasedOutlineEjRKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEPh(!llvm.ptr {llvm.align = 8 : i64, llvm.dereferenceable = 59 : i64, llvm.nonnull, llvm.noundef}, i32 {llvm.noundef}, !llvm.ptr {llvm.align = 8 : i64, llvm.dereferenceable = 32 : i64, llvm.nonnull, llvm.noundef}, !llvm.ptr {llvm.noundef}) -> (!llvm.ptr {llvm.noundef}) attributes {frame_pointer = #llvm.framePointerKind<none>, passthrough = [["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], ["tune-cpu", "generic"]], target_cpu = "x86-64", target_features = #llvm.target_features<["+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>}
  llvm.func local_unnamed_addr @_ZN6google8protobuf2io19EpsCopyOutputStream19EnsureSpaceFallbackEPh(!llvm.ptr {llvm.align = 8 : i64, llvm.dereferenceable = 59 : i64, llvm.nonnull, llvm.noundef}, !llvm.ptr {llvm.noundef}) -> (!llvm.ptr {llvm.noundef}) attributes {frame_pointer = #llvm.framePointerKind<none>, passthrough = [["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], ["tune-cpu", "generic"]], target_cpu = "x86-64", target_features = #llvm.target_features<["+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>}
  llvm.func local_unnamed_addr @_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE4swapERS4_(!llvm.ptr {llvm.align = 8 : i64, llvm.dereferenceable = 32 : i64, llvm.nonnull, llvm.noundef}, !llvm.ptr {llvm.align = 8 : i64, llvm.dereferenceable = 32 : i64, llvm.nonnull, llvm.noundef}) attributes {frame_pointer = #llvm.framePointerKind<none>, passthrough = ["nounwind", ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], ["tune-cpu", "generic"]], target_cpu = "x86-64", target_features = #llvm.target_features<["+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>}
  llvm.func local_unnamed_addr @_ZN6google8protobuf8internal20RepeatedPtrFieldBase7ReserveEi(!llvm.ptr {llvm.align = 8 : i64, llvm.dereferenceable = 24 : i64, llvm.nonnull, llvm.noundef}, i32 {llvm.noundef}) attributes {frame_pointer = #llvm.framePointerKind<none>, passthrough = [["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], ["tune-cpu", "generic"]], target_cpu = "x86-64", target_features = #llvm.target_features<["+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>}
  llvm.func local_unnamed_addr @_ZN6google8protobuf5Arena18CreateMaybeMessageINS0_9TimestampEJEEEPT_PS1_DpOT0_(!llvm.ptr {llvm.noundef}) -> (!llvm.ptr {llvm.noundef}) attributes {frame_pointer = #llvm.framePointerKind<none>, passthrough = [["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], ["tune-cpu", "generic"]], target_cpu = "x86-64", target_features = #llvm.target_features<["+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>}
  llvm.func linkonce_odr local_unnamed_addr @_ZNK6google8protobuf8internal20RepeatedPtrFieldBase3GetINS0_16RepeatedPtrFieldIN8tutorial18Person_PhoneNumberEE11TypeHandlerEEERKNT_4TypeEi(%arg0: !llvm.ptr {llvm.align = 8 : i64, llvm.dereferenceable = 24 : i64, llvm.nonnull, llvm.noundef}, %arg1: i32 {llvm.noundef}) -> (!llvm.ptr {llvm.align = 8 : i64, llvm.dereferenceable = 32 : i64, llvm.nonnull, llvm.noundef}) comdat(@__llvm_global_comdat::@_ZNK6google8protobuf8internal20RepeatedPtrFieldBase3GetINS0_16RepeatedPtrFieldIN8tutorial18Person_PhoneNumberEE11TypeHandlerEEERKNT_4TypeEi) attributes {alignment = 2 : i64, frame_pointer = #llvm.framePointerKind<none>, passthrough = ["inlinehint", "mustprogress", ["uwtable", "2"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], ["tune-cpu", "generic"]], personality = @__gxx_personality_v0, target_cpu = "x86-64", target_features = #llvm.target_features<["+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>} {
    %0 = llvm.mlir.constant(1 : i32) : i32
    %1 = llvm.mlir.constant(-1 : i32) : i32
    %2 = llvm.mlir.constant(0 : i64) : i64
    %3 = llvm.mlir.constant(0 : i32) : i32
    %4 = llvm.mlir.constant(3 : i32) : i32
    %5 = llvm.mlir.constant("/usr/include/google/protobuf/repeated_field.h\00") : !llvm.array<46 x i8>
    %6 = llvm.mlir.addressof @".str.16" : !llvm.ptr
    %7 = llvm.mlir.constant(1693 : i32) : i32
    %8 = llvm.mlir.constant("CHECK failed: (index) >= (0): \00") : !llvm.array<31 x i8>
    %9 = llvm.mlir.addressof @".str.17" : !llvm.ptr
    %10 = llvm.mlir.constant(1694 : i32) : i32
    %11 = llvm.mlir.constant("CHECK failed: (index) < (current_size_): \00") : !llvm.array<42 x i8>
    %12 = llvm.mlir.addressof @".str.18" : !llvm.ptr
    %13 = llvm.alloca %0 x !llvm.struct<"class.google::protobuf::internal::LogMessage", (i32, ptr, i32, struct<"class.std::__cxx11::basic_string", (struct<"struct.std::__cxx11::basic_string<char>::_Alloc_hider", (ptr)>, i64, struct<"union.anon", (i64, array<8 x i8>)>)>)> {alignment = 8 : i64} : (i32) -> !llvm.ptr
    %14 = llvm.alloca %0 x !llvm.struct<"class.google::protobuf::internal::LogFinisher", (i8)> {alignment = 1 : i64} : (i32) -> !llvm.ptr
    %15 = llvm.alloca %0 x !llvm.struct<"class.google::protobuf::internal::LogMessage", (i32, ptr, i32, struct<"class.std::__cxx11::basic_string", (struct<"struct.std::__cxx11::basic_string<char>::_Alloc_hider", (ptr)>, i64, struct<"union.anon", (i64, array<8 x i8>)>)>)> {alignment = 8 : i64} : (i32) -> !llvm.ptr
    %16 = llvm.alloca %0 x !llvm.struct<"class.google::protobuf::internal::LogFinisher", (i8)> {alignment = 1 : i64} : (i32) -> !llvm.ptr
    %17 = llvm.icmp "sgt" %arg1, %1 : i32
    %18 = llvm.getelementptr inbounds %14[%2, 0] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.google::protobuf::internal::LogFinisher", (i8)>
    llvm.intr.lifetime.start 1, %18 : !llvm.ptr
    llvm.cond_br %17, ^bb3, ^bb1
  ^bb1:  // pred: ^bb0
    %19 = llvm.bitcast %13 : !llvm.ptr to !llvm.ptr
    llvm.intr.lifetime.start 56, %19 : !llvm.ptr
    llvm.call @_ZN6google8protobuf8internal10LogMessageC1ENS0_8LogLevelEPKci(%13, %4, %6, %7) : (!llvm.ptr, i32, !llvm.ptr, i32) -> ()
    %20 = llvm.invoke @_ZN6google8protobuf8internal10LogMessagelsEPKc(%13, %9) to ^bb2 unwind ^bb11 : (!llvm.ptr, !llvm.ptr) -> !llvm.ptr
  ^bb2:  // pred: ^bb1
    llvm.invoke @_ZN6google8protobuf8internal11LogFinisheraSERNS1_10LogMessageE(%14, %20) to ^bb4 unwind ^bb12 : (!llvm.ptr, !llvm.ptr) -> ()
  ^bb3:  // pred: ^bb0
    llvm.intr.lifetime.end 1, %18 : !llvm.ptr
    llvm.br ^bb5
  ^bb4:  // pred: ^bb2
    llvm.intr.lifetime.end 1, %18 : !llvm.ptr
    llvm.call @_ZN6google8protobuf8internal10LogMessageD1Ev(%13) : (!llvm.ptr) -> ()
    llvm.intr.lifetime.end 56, %19 : !llvm.ptr
    llvm.br ^bb5
  ^bb5:  // 2 preds: ^bb3, ^bb4
    %21 = llvm.getelementptr inbounds %arg0[%2, 1] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.google::protobuf::internal::RepeatedPtrFieldBase", (ptr, i32, i32, ptr)>
    %22 = llvm.load %21 {alignment = 8 : i64, tbaa = [#tbaa_tag17]} : !llvm.ptr -> i32
    %23 = llvm.icmp "sgt" %22, %arg1 : i32
    %24 = llvm.getelementptr inbounds %16[%2, 0] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.google::protobuf::internal::LogFinisher", (i8)>
    llvm.intr.lifetime.start 1, %24 : !llvm.ptr
    llvm.cond_br %23, ^bb8, ^bb6
  ^bb6:  // pred: ^bb5
    %25 = llvm.bitcast %15 : !llvm.ptr to !llvm.ptr
    llvm.intr.lifetime.start 56, %25 : !llvm.ptr
    llvm.call @_ZN6google8protobuf8internal10LogMessageC1ENS0_8LogLevelEPKci(%15, %4, %6, %10) : (!llvm.ptr, i32, !llvm.ptr, i32) -> ()
    %26 = llvm.invoke @_ZN6google8protobuf8internal10LogMessagelsEPKc(%15, %12) to ^bb7 unwind ^bb14 : (!llvm.ptr, !llvm.ptr) -> !llvm.ptr
  ^bb7:  // pred: ^bb6
    llvm.invoke @_ZN6google8protobuf8internal11LogFinisheraSERNS1_10LogMessageE(%16, %26) to ^bb9 unwind ^bb15 : (!llvm.ptr, !llvm.ptr) -> ()
  ^bb8:  // pred: ^bb5
    llvm.intr.lifetime.end 1, %24 : !llvm.ptr
    llvm.br ^bb10
  ^bb9:  // pred: ^bb7
    llvm.intr.lifetime.end 1, %24 : !llvm.ptr
    llvm.call @_ZN6google8protobuf8internal10LogMessageD1Ev(%15) : (!llvm.ptr) -> ()
    llvm.intr.lifetime.end 56, %25 : !llvm.ptr
    llvm.br ^bb10
  ^bb10:  // 2 preds: ^bb8, ^bb9
    %27 = llvm.getelementptr inbounds %arg0[%2, 3] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.google::protobuf::internal::RepeatedPtrFieldBase", (ptr, i32, i32, ptr)>
    %28 = llvm.load %27 {alignment = 8 : i64, tbaa = [#tbaa_tag15]} : !llvm.ptr -> !llvm.ptr
    %29 = llvm.sext %arg1 : i32 to i64
    %30 = llvm.getelementptr inbounds %28[%2, 1, %29] : (!llvm.ptr, i64, i64) -> !llvm.ptr, !llvm.struct<"struct.google::protobuf::internal::RepeatedPtrFieldBase::Rep", (i32, array<1 x ptr>)>
    %31 = llvm.bitcast %30 : !llvm.ptr to !llvm.ptr
    %32 = llvm.load %31 {alignment = 8 : i64, tbaa = [#tbaa_tag2]} : !llvm.ptr -> !llvm.ptr
    llvm.return %32 : !llvm.ptr
  ^bb11:  // pred: ^bb1
    %33 = llvm.landingpad cleanup : !llvm.struct<(ptr, i32)>
    llvm.br ^bb13(%33 : !llvm.struct<(ptr, i32)>)
  ^bb12:  // pred: ^bb2
    %34 = llvm.landingpad cleanup : !llvm.struct<(ptr, i32)>
    llvm.intr.lifetime.end 1, %18 : !llvm.ptr
    llvm.br ^bb13(%34 : !llvm.struct<(ptr, i32)>)
  ^bb13(%35: !llvm.struct<(ptr, i32)>):  // 2 preds: ^bb11, ^bb12
    llvm.call @_ZN6google8protobuf8internal10LogMessageD1Ev(%13) : (!llvm.ptr) -> ()
    llvm.intr.lifetime.end 56, %19 : !llvm.ptr
    llvm.br ^bb17(%35 : !llvm.struct<(ptr, i32)>)
  ^bb14:  // pred: ^bb6
    %36 = llvm.landingpad cleanup : !llvm.struct<(ptr, i32)>
    llvm.br ^bb16(%36 : !llvm.struct<(ptr, i32)>)
  ^bb15:  // pred: ^bb7
    %37 = llvm.landingpad cleanup : !llvm.struct<(ptr, i32)>
    llvm.intr.lifetime.end 1, %24 : !llvm.ptr
    llvm.br ^bb16(%37 : !llvm.struct<(ptr, i32)>)
  ^bb16(%38: !llvm.struct<(ptr, i32)>):  // 2 preds: ^bb14, ^bb15
    llvm.call @_ZN6google8protobuf8internal10LogMessageD1Ev(%15) : (!llvm.ptr) -> ()
    llvm.intr.lifetime.end 56, %25 : !llvm.ptr
    llvm.br ^bb17(%38 : !llvm.struct<(ptr, i32)>)
  ^bb17(%39: !llvm.struct<(ptr, i32)>):  // 2 preds: ^bb13, ^bb16
    llvm.resume %39 : !llvm.struct<(ptr, i32)>
  }
  llvm.func linkonce_odr local_unnamed_addr @_ZNK6google8protobuf8internal20RepeatedPtrFieldBase3GetINS0_16RepeatedPtrFieldIN8tutorial6PersonEE11TypeHandlerEEERKNT_4TypeEi(%arg0: !llvm.ptr {llvm.align = 8 : i64, llvm.dereferenceable = 24 : i64, llvm.nonnull, llvm.noundef}, %arg1: i32 {llvm.noundef}) -> (!llvm.ptr {llvm.align = 8 : i64, llvm.dereferenceable = 72 : i64, llvm.nonnull, llvm.noundef}) comdat(@__llvm_global_comdat::@_ZNK6google8protobuf8internal20RepeatedPtrFieldBase3GetINS0_16RepeatedPtrFieldIN8tutorial6PersonEE11TypeHandlerEEERKNT_4TypeEi) attributes {alignment = 2 : i64, frame_pointer = #llvm.framePointerKind<none>, passthrough = ["inlinehint", "mustprogress", ["uwtable", "2"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], ["tune-cpu", "generic"]], personality = @__gxx_personality_v0, target_cpu = "x86-64", target_features = #llvm.target_features<["+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>} {
    %0 = llvm.mlir.constant(1 : i32) : i32
    %1 = llvm.mlir.constant(-1 : i32) : i32
    %2 = llvm.mlir.constant(0 : i64) : i64
    %3 = llvm.mlir.constant(0 : i32) : i32
    %4 = llvm.mlir.constant(3 : i32) : i32
    %5 = llvm.mlir.constant("/usr/include/google/protobuf/repeated_field.h\00") : !llvm.array<46 x i8>
    %6 = llvm.mlir.addressof @".str.16" : !llvm.ptr
    %7 = llvm.mlir.constant(1693 : i32) : i32
    %8 = llvm.mlir.constant("CHECK failed: (index) >= (0): \00") : !llvm.array<31 x i8>
    %9 = llvm.mlir.addressof @".str.17" : !llvm.ptr
    %10 = llvm.mlir.constant(1694 : i32) : i32
    %11 = llvm.mlir.constant("CHECK failed: (index) < (current_size_): \00") : !llvm.array<42 x i8>
    %12 = llvm.mlir.addressof @".str.18" : !llvm.ptr
    %13 = llvm.alloca %0 x !llvm.struct<"class.google::protobuf::internal::LogMessage", (i32, ptr, i32, struct<"class.std::__cxx11::basic_string", (struct<"struct.std::__cxx11::basic_string<char>::_Alloc_hider", (ptr)>, i64, struct<"union.anon", (i64, array<8 x i8>)>)>)> {alignment = 8 : i64} : (i32) -> !llvm.ptr
    %14 = llvm.alloca %0 x !llvm.struct<"class.google::protobuf::internal::LogFinisher", (i8)> {alignment = 1 : i64} : (i32) -> !llvm.ptr
    %15 = llvm.alloca %0 x !llvm.struct<"class.google::protobuf::internal::LogMessage", (i32, ptr, i32, struct<"class.std::__cxx11::basic_string", (struct<"struct.std::__cxx11::basic_string<char>::_Alloc_hider", (ptr)>, i64, struct<"union.anon", (i64, array<8 x i8>)>)>)> {alignment = 8 : i64} : (i32) -> !llvm.ptr
    %16 = llvm.alloca %0 x !llvm.struct<"class.google::protobuf::internal::LogFinisher", (i8)> {alignment = 1 : i64} : (i32) -> !llvm.ptr
    %17 = llvm.icmp "sgt" %arg1, %1 : i32
    %18 = llvm.getelementptr inbounds %14[%2, 0] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.google::protobuf::internal::LogFinisher", (i8)>
    llvm.intr.lifetime.start 1, %18 : !llvm.ptr
    llvm.cond_br %17, ^bb3, ^bb1
  ^bb1:  // pred: ^bb0
    %19 = llvm.bitcast %13 : !llvm.ptr to !llvm.ptr
    llvm.intr.lifetime.start 56, %19 : !llvm.ptr
    llvm.call @_ZN6google8protobuf8internal10LogMessageC1ENS0_8LogLevelEPKci(%13, %4, %6, %7) : (!llvm.ptr, i32, !llvm.ptr, i32) -> ()
    %20 = llvm.invoke @_ZN6google8protobuf8internal10LogMessagelsEPKc(%13, %9) to ^bb2 unwind ^bb11 : (!llvm.ptr, !llvm.ptr) -> !llvm.ptr
  ^bb2:  // pred: ^bb1
    llvm.invoke @_ZN6google8protobuf8internal11LogFinisheraSERNS1_10LogMessageE(%14, %20) to ^bb4 unwind ^bb12 : (!llvm.ptr, !llvm.ptr) -> ()
  ^bb3:  // pred: ^bb0
    llvm.intr.lifetime.end 1, %18 : !llvm.ptr
    llvm.br ^bb5
  ^bb4:  // pred: ^bb2
    llvm.intr.lifetime.end 1, %18 : !llvm.ptr
    llvm.call @_ZN6google8protobuf8internal10LogMessageD1Ev(%13) : (!llvm.ptr) -> ()
    llvm.intr.lifetime.end 56, %19 : !llvm.ptr
    llvm.br ^bb5
  ^bb5:  // 2 preds: ^bb3, ^bb4
    %21 = llvm.getelementptr inbounds %arg0[%2, 1] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.google::protobuf::internal::RepeatedPtrFieldBase", (ptr, i32, i32, ptr)>
    %22 = llvm.load %21 {alignment = 8 : i64, tbaa = [#tbaa_tag17]} : !llvm.ptr -> i32
    %23 = llvm.icmp "sgt" %22, %arg1 : i32
    %24 = llvm.getelementptr inbounds %16[%2, 0] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.google::protobuf::internal::LogFinisher", (i8)>
    llvm.intr.lifetime.start 1, %24 : !llvm.ptr
    llvm.cond_br %23, ^bb8, ^bb6
  ^bb6:  // pred: ^bb5
    %25 = llvm.bitcast %15 : !llvm.ptr to !llvm.ptr
    llvm.intr.lifetime.start 56, %25 : !llvm.ptr
    llvm.call @_ZN6google8protobuf8internal10LogMessageC1ENS0_8LogLevelEPKci(%15, %4, %6, %10) : (!llvm.ptr, i32, !llvm.ptr, i32) -> ()
    %26 = llvm.invoke @_ZN6google8protobuf8internal10LogMessagelsEPKc(%15, %12) to ^bb7 unwind ^bb14 : (!llvm.ptr, !llvm.ptr) -> !llvm.ptr
  ^bb7:  // pred: ^bb6
    llvm.invoke @_ZN6google8protobuf8internal11LogFinisheraSERNS1_10LogMessageE(%16, %26) to ^bb9 unwind ^bb15 : (!llvm.ptr, !llvm.ptr) -> ()
  ^bb8:  // pred: ^bb5
    llvm.intr.lifetime.end 1, %24 : !llvm.ptr
    llvm.br ^bb10
  ^bb9:  // pred: ^bb7
    llvm.intr.lifetime.end 1, %24 : !llvm.ptr
    llvm.call @_ZN6google8protobuf8internal10LogMessageD1Ev(%15) : (!llvm.ptr) -> ()
    llvm.intr.lifetime.end 56, %25 : !llvm.ptr
    llvm.br ^bb10
  ^bb10:  // 2 preds: ^bb8, ^bb9
    %27 = llvm.getelementptr inbounds %arg0[%2, 3] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.google::protobuf::internal::RepeatedPtrFieldBase", (ptr, i32, i32, ptr)>
    %28 = llvm.load %27 {alignment = 8 : i64, tbaa = [#tbaa_tag15]} : !llvm.ptr -> !llvm.ptr
    %29 = llvm.sext %arg1 : i32 to i64
    %30 = llvm.getelementptr inbounds %28[%2, 1, %29] : (!llvm.ptr, i64, i64) -> !llvm.ptr, !llvm.struct<"struct.google::protobuf::internal::RepeatedPtrFieldBase::Rep", (i32, array<1 x ptr>)>
    %31 = llvm.bitcast %30 : !llvm.ptr to !llvm.ptr
    %32 = llvm.load %31 {alignment = 8 : i64, tbaa = [#tbaa_tag2]} : !llvm.ptr -> !llvm.ptr
    llvm.return %32 : !llvm.ptr
  ^bb11:  // pred: ^bb1
    %33 = llvm.landingpad cleanup : !llvm.struct<(ptr, i32)>
    llvm.br ^bb13(%33 : !llvm.struct<(ptr, i32)>)
  ^bb12:  // pred: ^bb2
    %34 = llvm.landingpad cleanup : !llvm.struct<(ptr, i32)>
    llvm.intr.lifetime.end 1, %18 : !llvm.ptr
    llvm.br ^bb13(%34 : !llvm.struct<(ptr, i32)>)
  ^bb13(%35: !llvm.struct<(ptr, i32)>):  // 2 preds: ^bb11, ^bb12
    llvm.call @_ZN6google8protobuf8internal10LogMessageD1Ev(%13) : (!llvm.ptr) -> ()
    llvm.intr.lifetime.end 56, %19 : !llvm.ptr
    llvm.br ^bb17(%35 : !llvm.struct<(ptr, i32)>)
  ^bb14:  // pred: ^bb6
    %36 = llvm.landingpad cleanup : !llvm.struct<(ptr, i32)>
    llvm.br ^bb16(%36 : !llvm.struct<(ptr, i32)>)
  ^bb15:  // pred: ^bb7
    %37 = llvm.landingpad cleanup : !llvm.struct<(ptr, i32)>
    llvm.intr.lifetime.end 1, %24 : !llvm.ptr
    llvm.br ^bb16(%37 : !llvm.struct<(ptr, i32)>)
  ^bb16(%38: !llvm.struct<(ptr, i32)>):  // 2 preds: ^bb14, ^bb15
    llvm.call @_ZN6google8protobuf8internal10LogMessageD1Ev(%15) : (!llvm.ptr) -> ()
    llvm.intr.lifetime.end 56, %25 : !llvm.ptr
    llvm.br ^bb17(%38 : !llvm.struct<(ptr, i32)>)
  ^bb17(%39: !llvm.struct<(ptr, i32)>):  // 2 preds: ^bb13, ^bb16
    llvm.resume %39 : !llvm.struct<(ptr, i32)>
  }
  llvm.func linkonce_odr local_unnamed_addr @_ZN6google8protobuf8internal20RepeatedPtrFieldBase5ClearINS0_16RepeatedPtrFieldIN8tutorial18Person_PhoneNumberEE11TypeHandlerEEEvv(%arg0: !llvm.ptr {llvm.align = 8 : i64, llvm.dereferenceable = 24 : i64, llvm.nonnull, llvm.noundef}) comdat(@__llvm_global_comdat::@_ZN6google8protobuf8internal20RepeatedPtrFieldBase5ClearINS0_16RepeatedPtrFieldIN8tutorial18Person_PhoneNumberEE11TypeHandlerEEEvv) attributes {alignment = 2 : i64, frame_pointer = #llvm.framePointerKind<none>, passthrough = [["uwtable", "2"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], ["tune-cpu", "generic"]], personality = @__gxx_personality_v0, target_cpu = "x86-64", target_features = #llvm.target_features<["+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>} {
    %0 = llvm.mlir.constant(1 : i32) : i32
    %1 = llvm.mlir.constant(0 : i64) : i64
    %2 = llvm.mlir.constant(-1 : i32) : i32
    %3 = llvm.mlir.constant(0 : i32) : i32
    %4 = llvm.mlir.constant(3 : i32) : i32
    %5 = llvm.mlir.constant("/usr/include/google/protobuf/repeated_field.h\00") : !llvm.array<46 x i8>
    %6 = llvm.mlir.addressof @".str.16" : !llvm.ptr
    %7 = llvm.mlir.constant(1768 : i32) : i32
    %8 = llvm.mlir.constant("CHECK failed: (n) >= (0): \00") : !llvm.array<27 x i8>
    %9 = llvm.mlir.addressof @".str.19" : !llvm.ptr
    %10 = llvm.mlir.constant(1 : i64) : i64
    %11 = llvm.mlir.addressof @_ZN6google8protobuf8internal26fixed_address_empty_stringB5cxx11E : !llvm.ptr
    %12 = llvm.mlir.constant(0 : i8) : i8
    %13 = llvm.mlir.constant(2 : i32) : i32
    %14 = llvm.mlir.constant(-2 : i64) : i64
    %15 = llvm.alloca %0 x !llvm.struct<"class.google::protobuf::internal::LogMessage", (i32, ptr, i32, struct<"class.std::__cxx11::basic_string", (struct<"struct.std::__cxx11::basic_string<char>::_Alloc_hider", (ptr)>, i64, struct<"union.anon", (i64, array<8 x i8>)>)>)> {alignment = 8 : i64} : (i32) -> !llvm.ptr
    %16 = llvm.alloca %0 x !llvm.struct<"class.google::protobuf::internal::LogFinisher", (i8)> {alignment = 1 : i64} : (i32) -> !llvm.ptr
    %17 = llvm.getelementptr inbounds %arg0[%1, 1] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.google::protobuf::internal::RepeatedPtrFieldBase", (ptr, i32, i32, ptr)>
    %18 = llvm.load %17 {alignment = 8 : i64, tbaa = [#tbaa_tag17]} : !llvm.ptr -> i32
    %19 = llvm.icmp "sgt" %18, %2 : i32
    %20 = llvm.getelementptr inbounds %16[%1, 0] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.google::protobuf::internal::LogFinisher", (i8)>
    llvm.intr.lifetime.start 1, %20 : !llvm.ptr
    llvm.cond_br %19, ^bb4, ^bb1
  ^bb1:  // pred: ^bb0
    %21 = llvm.bitcast %15 : !llvm.ptr to !llvm.ptr
    llvm.intr.lifetime.start 56, %21 : !llvm.ptr
    llvm.call @_ZN6google8protobuf8internal10LogMessageC1ENS0_8LogLevelEPKci(%15, %4, %6, %7) : (!llvm.ptr, i32, !llvm.ptr, i32) -> ()
    %22 = llvm.invoke @_ZN6google8protobuf8internal10LogMessagelsEPKc(%15, %9) to ^bb2 unwind ^bb13 : (!llvm.ptr, !llvm.ptr) -> !llvm.ptr
  ^bb2:  // pred: ^bb1
    llvm.invoke @_ZN6google8protobuf8internal11LogFinisheraSERNS1_10LogMessageE(%16, %22) to ^bb3 unwind ^bb14 : (!llvm.ptr, !llvm.ptr) -> ()
  ^bb3:  // pred: ^bb2
    llvm.intr.lifetime.end 1, %20 : !llvm.ptr
    llvm.call @_ZN6google8protobuf8internal10LogMessageD1Ev(%15) : (!llvm.ptr) -> ()
    llvm.intr.lifetime.end 56, %21 : !llvm.ptr
    llvm.br ^bb16
  ^bb4:  // pred: ^bb0
    llvm.intr.lifetime.end 1, %20 : !llvm.ptr
    %23 = llvm.icmp "eq" %18, %3 : i32
    llvm.cond_br %23, ^bb16, ^bb5
  ^bb5:  // pred: ^bb4
    %24 = llvm.getelementptr inbounds %arg0[%1, 3] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.google::protobuf::internal::RepeatedPtrFieldBase", (ptr, i32, i32, ptr)>
    %25 = llvm.load %24 {alignment = 8 : i64, tbaa = [#tbaa_tag15]} : !llvm.ptr -> !llvm.ptr
    %26 = llvm.zext %18 : i32 to i64
    llvm.br ^bb6(%1 : i64)
  ^bb6(%27: i64):  // 2 preds: ^bb5, ^bb11
    %28 = llvm.add %27, %10 overflow<nsw, nuw>  : i64
    %29 = llvm.getelementptr inbounds %25[%1, 1, %27] : (!llvm.ptr, i64, i64) -> !llvm.ptr, !llvm.struct<"struct.google::protobuf::internal::RepeatedPtrFieldBase::Rep", (i32, array<1 x ptr>)>
    %30 = llvm.bitcast %29 : !llvm.ptr to !llvm.ptr
    %31 = llvm.load %30 {alignment = 8 : i64, tbaa = [#tbaa_tag2]} : !llvm.ptr -> !llvm.ptr
    %32 = llvm.getelementptr inbounds %31[%1, 1, 0] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.tutorial::Person_PhoneNumber", (struct<"class.google::protobuf::Message", (struct<"class.google::protobuf::MessageLite", (ptr, struct<"class.google::protobuf::internal::InternalMetadata", (ptr)>)>)>, struct<"struct.google::protobuf::internal::ArenaStringPtr", (ptr)>, i32, struct<"class.google::protobuf::internal::CachedSize", (struct<"struct.std::atomic", (struct<"struct.std::__atomic_base", (i32)>)>)>)>
    %33 = llvm.load %32 {alignment = 8 : i64, tbaa = [#tbaa_tag8]} : !llvm.ptr -> !llvm.ptr
    %34 = llvm.icmp "eq" %33, %11 : !llvm.ptr
    llvm.cond_br %34, ^bb8, ^bb7
  ^bb7:  // pred: ^bb6
    %35 = llvm.getelementptr inbounds %33[%1, 1] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.std::__cxx11::basic_string", (struct<"struct.std::__cxx11::basic_string<char>::_Alloc_hider", (ptr)>, i64, struct<"union.anon", (i64, array<8 x i8>)>)>
    llvm.store %1, %35 {alignment = 8 : i64, tbaa = [#tbaa_tag25]} : i64, !llvm.ptr
    %36 = llvm.getelementptr inbounds %33[%1, 0, 0] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.std::__cxx11::basic_string", (struct<"struct.std::__cxx11::basic_string<char>::_Alloc_hider", (ptr)>, i64, struct<"union.anon", (i64, array<8 x i8>)>)>
    %37 = llvm.load %36 {alignment = 8 : i64, tbaa = [#tbaa_tag26]} : !llvm.ptr -> !llvm.ptr
    llvm.store %12, %37 {alignment = 1 : i64, tbaa = [#tbaa_tag1]} : i8, !llvm.ptr
    llvm.br ^bb8
  ^bb8:  // 2 preds: ^bb6, ^bb7
    %38 = llvm.getelementptr inbounds %31[%1, 2] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.tutorial::Person_PhoneNumber", (struct<"class.google::protobuf::Message", (struct<"class.google::protobuf::MessageLite", (ptr, struct<"class.google::protobuf::internal::InternalMetadata", (ptr)>)>)>, struct<"struct.google::protobuf::internal::ArenaStringPtr", (ptr)>, i32, struct<"class.google::protobuf::internal::CachedSize", (struct<"struct.std::atomic", (struct<"struct.std::__atomic_base", (i32)>)>)>)>
    llvm.store %3, %38 {alignment = 8 : i64, tbaa = [#tbaa_tag24]} : i32, !llvm.ptr
    %39 = llvm.getelementptr inbounds %31[%1, 0, 0, 1, 0] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.tutorial::Person_PhoneNumber", (struct<"class.google::protobuf::Message", (struct<"class.google::protobuf::MessageLite", (ptr, struct<"class.google::protobuf::internal::InternalMetadata", (ptr)>)>)>, struct<"struct.google::protobuf::internal::ArenaStringPtr", (ptr)>, i32, struct<"class.google::protobuf::internal::CachedSize", (struct<"struct.std::atomic", (struct<"struct.std::__atomic_base", (i32)>)>)>)>
    %40 = llvm.load %39 {alignment = 8 : i64, tbaa = [#tbaa_tag5]} : !llvm.ptr -> !llvm.ptr
    %41 = llvm.ptrtoint %40 : !llvm.ptr to i64
    %42 = llvm.and %41, %10  : i64
    %43 = llvm.icmp "eq" %42, %1 : i64
    llvm.cond_br %43, ^bb11, ^bb9
  ^bb9:  // pred: ^bb8
    %44 = llvm.and %41, %14  : i64
    %45 = llvm.inttoptr %44 : i64 to !llvm.ptr
    %46 = llvm.getelementptr inbounds %45[%1, 1] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.google::protobuf::internal::InternalMetadata::Container", (struct<"struct.google::protobuf::internal::InternalMetadata::ContainerBase", (ptr)>, struct<"class.google::protobuf::UnknownFieldSet", (struct<"class.std::vector", (struct<"struct.std::_Vector_base", (struct<"struct.std::_Vector_base<google::protobuf::UnknownField, std::allocator<google::protobuf::UnknownField>>::_Vector_impl", (struct<"struct.std::_Vector_base<google::protobuf::UnknownField, std::allocator<google::protobuf::UnknownField>>::_Vector_impl_data", (ptr, ptr, ptr)>)>)>)>)>)>
    %47 = llvm.getelementptr inbounds %46[%1, 0, 0, 0, 0, 0] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.google::protobuf::UnknownFieldSet", (struct<"class.std::vector", (struct<"struct.std::_Vector_base", (struct<"struct.std::_Vector_base<google::protobuf::UnknownField, std::allocator<google::protobuf::UnknownField>>::_Vector_impl", (struct<"struct.std::_Vector_base<google::protobuf::UnknownField, std::allocator<google::protobuf::UnknownField>>::_Vector_impl_data", (ptr, ptr, ptr)>)>)>)>)>
    %48 = llvm.load %47 {alignment = 8 : i64, tbaa = [#tbaa_tag2]} : !llvm.ptr -> !llvm.ptr
    %49 = llvm.getelementptr inbounds %45[%1, 1, 0, 0, 0, 0, 1] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.google::protobuf::internal::InternalMetadata::Container", (struct<"struct.google::protobuf::internal::InternalMetadata::ContainerBase", (ptr)>, struct<"class.google::protobuf::UnknownFieldSet", (struct<"class.std::vector", (struct<"struct.std::_Vector_base", (struct<"struct.std::_Vector_base<google::protobuf::UnknownField, std::allocator<google::protobuf::UnknownField>>::_Vector_impl", (struct<"struct.std::_Vector_base<google::protobuf::UnknownField, std::allocator<google::protobuf::UnknownField>>::_Vector_impl_data", (ptr, ptr, ptr)>)>)>)>)>)>
    %50 = llvm.load %49 {alignment = 8 : i64, tbaa = [#tbaa_tag2]} : !llvm.ptr -> !llvm.ptr
    %51 = llvm.icmp "eq" %48, %50 : !llvm.ptr
    llvm.cond_br %51, ^bb11, ^bb10
  ^bb10:  // pred: ^bb9
    llvm.call @_ZN6google8protobuf15UnknownFieldSet13ClearFallbackEv(%46) : (!llvm.ptr) -> ()
    llvm.br ^bb11
  ^bb11:  // 3 preds: ^bb8, ^bb9, ^bb10
    %52 = llvm.icmp "eq" %28, %26 : i64
    llvm.cond_br %52, ^bb12, ^bb6(%28 : i64) {loop_annotation = #loop_annotation}
  ^bb12:  // pred: ^bb11
    llvm.store %3, %17 {alignment = 8 : i64, tbaa = [#tbaa_tag17]} : i32, !llvm.ptr
    llvm.br ^bb16
  ^bb13:  // pred: ^bb1
    %53 = llvm.landingpad cleanup : !llvm.struct<(ptr, i32)>
    llvm.br ^bb15(%53 : !llvm.struct<(ptr, i32)>)
  ^bb14:  // pred: ^bb2
    %54 = llvm.landingpad cleanup : !llvm.struct<(ptr, i32)>
    llvm.intr.lifetime.end 1, %20 : !llvm.ptr
    llvm.br ^bb15(%54 : !llvm.struct<(ptr, i32)>)
  ^bb15(%55: !llvm.struct<(ptr, i32)>):  // 2 preds: ^bb13, ^bb14
    llvm.call @_ZN6google8protobuf8internal10LogMessageD1Ev(%15) : (!llvm.ptr) -> ()
    llvm.intr.lifetime.end 56, %21 : !llvm.ptr
    llvm.resume %55 : !llvm.struct<(ptr, i32)>
  ^bb16:  // 3 preds: ^bb3, ^bb4, ^bb12
    llvm.return
  }
  llvm.func linkonce_odr local_unnamed_addr @_ZN6google8protobuf8internal20RepeatedPtrFieldBase5ClearINS0_16RepeatedPtrFieldIN8tutorial6PersonEE11TypeHandlerEEEvv(%arg0: !llvm.ptr {llvm.align = 8 : i64, llvm.dereferenceable = 24 : i64, llvm.nonnull, llvm.noundef}) comdat(@__llvm_global_comdat::@_ZN6google8protobuf8internal20RepeatedPtrFieldBase5ClearINS0_16RepeatedPtrFieldIN8tutorial6PersonEE11TypeHandlerEEEvv) attributes {alignment = 2 : i64, frame_pointer = #llvm.framePointerKind<none>, passthrough = ["mustprogress", ["uwtable", "2"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], ["tune-cpu", "generic"]], personality = @__gxx_personality_v0, target_cpu = "x86-64", target_features = #llvm.target_features<["+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>} {
    %0 = llvm.mlir.constant(1 : i32) : i32
    %1 = llvm.mlir.constant(0 : i64) : i64
    %2 = llvm.mlir.constant(-1 : i32) : i32
    %3 = llvm.mlir.constant(0 : i32) : i32
    %4 = llvm.mlir.constant(3 : i32) : i32
    %5 = llvm.mlir.constant("/usr/include/google/protobuf/repeated_field.h\00") : !llvm.array<46 x i8>
    %6 = llvm.mlir.addressof @".str.16" : !llvm.ptr
    %7 = llvm.mlir.constant(1768 : i32) : i32
    %8 = llvm.mlir.constant("CHECK failed: (n) >= (0): \00") : !llvm.array<27 x i8>
    %9 = llvm.mlir.addressof @".str.19" : !llvm.ptr
    %10 = llvm.mlir.constant(1 : i64) : i64
    %11 = llvm.alloca %0 x !llvm.struct<"class.google::protobuf::internal::LogMessage", (i32, ptr, i32, struct<"class.std::__cxx11::basic_string", (struct<"struct.std::__cxx11::basic_string<char>::_Alloc_hider", (ptr)>, i64, struct<"union.anon", (i64, array<8 x i8>)>)>)> {alignment = 8 : i64} : (i32) -> !llvm.ptr
    %12 = llvm.alloca %0 x !llvm.struct<"class.google::protobuf::internal::LogFinisher", (i8)> {alignment = 1 : i64} : (i32) -> !llvm.ptr
    %13 = llvm.getelementptr inbounds %arg0[%1, 1] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.google::protobuf::internal::RepeatedPtrFieldBase", (ptr, i32, i32, ptr)>
    %14 = llvm.load %13 {alignment = 8 : i64, tbaa = [#tbaa_tag17]} : !llvm.ptr -> i32
    %15 = llvm.icmp "sgt" %14, %2 : i32
    %16 = llvm.getelementptr inbounds %12[%1, 0] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.google::protobuf::internal::LogFinisher", (i8)>
    llvm.intr.lifetime.start 1, %16 : !llvm.ptr
    llvm.cond_br %15, ^bb4, ^bb1
  ^bb1:  // pred: ^bb0
    %17 = llvm.bitcast %11 : !llvm.ptr to !llvm.ptr
    llvm.intr.lifetime.start 56, %17 : !llvm.ptr
    llvm.call @_ZN6google8protobuf8internal10LogMessageC1ENS0_8LogLevelEPKci(%11, %4, %6, %7) : (!llvm.ptr, i32, !llvm.ptr, i32) -> ()
    %18 = llvm.invoke @_ZN6google8protobuf8internal10LogMessagelsEPKc(%11, %9) to ^bb2 unwind ^bb8 : (!llvm.ptr, !llvm.ptr) -> !llvm.ptr
  ^bb2:  // pred: ^bb1
    llvm.invoke @_ZN6google8protobuf8internal11LogFinisheraSERNS1_10LogMessageE(%12, %18) to ^bb3 unwind ^bb9 : (!llvm.ptr, !llvm.ptr) -> ()
  ^bb3:  // pred: ^bb2
    llvm.intr.lifetime.end 1, %16 : !llvm.ptr
    llvm.call @_ZN6google8protobuf8internal10LogMessageD1Ev(%11) : (!llvm.ptr) -> ()
    llvm.intr.lifetime.end 56, %17 : !llvm.ptr
    llvm.br ^bb11
  ^bb4:  // pred: ^bb0
    llvm.intr.lifetime.end 1, %16 : !llvm.ptr
    %19 = llvm.icmp "eq" %14, %3 : i32
    llvm.cond_br %19, ^bb11, ^bb5
  ^bb5:  // pred: ^bb4
    %20 = llvm.getelementptr inbounds %arg0[%1, 3] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.google::protobuf::internal::RepeatedPtrFieldBase", (ptr, i32, i32, ptr)>
    %21 = llvm.load %20 {alignment = 8 : i64, tbaa = [#tbaa_tag15]} : !llvm.ptr -> !llvm.ptr
    %22 = llvm.zext %14 : i32 to i64
    llvm.br ^bb6(%1 : i64)
  ^bb6(%23: i64):  // 2 preds: ^bb5, ^bb6
    %24 = llvm.add %23, %10 overflow<nsw, nuw>  : i64
    %25 = llvm.getelementptr inbounds %21[%1, 1, %23] : (!llvm.ptr, i64, i64) -> !llvm.ptr, !llvm.struct<"struct.google::protobuf::internal::RepeatedPtrFieldBase::Rep", (i32, array<1 x ptr>)>
    %26 = llvm.bitcast %25 : !llvm.ptr to !llvm.ptr
    %27 = llvm.load %26 {alignment = 8 : i64, tbaa = [#tbaa_tag2]} : !llvm.ptr -> !llvm.ptr
    llvm.call @_ZN8tutorial6Person5ClearEv(%27) : (!llvm.ptr) -> ()
    %28 = llvm.icmp "eq" %24, %22 : i64
    llvm.cond_br %28, ^bb7, ^bb6(%24 : i64) {loop_annotation = #loop_annotation}
  ^bb7:  // pred: ^bb6
    llvm.store %3, %13 {alignment = 8 : i64, tbaa = [#tbaa_tag17]} : i32, !llvm.ptr
    llvm.br ^bb11
  ^bb8:  // pred: ^bb1
    %29 = llvm.landingpad cleanup : !llvm.struct<(ptr, i32)>
    llvm.br ^bb10(%29 : !llvm.struct<(ptr, i32)>)
  ^bb9:  // pred: ^bb2
    %30 = llvm.landingpad cleanup : !llvm.struct<(ptr, i32)>
    llvm.intr.lifetime.end 1, %16 : !llvm.ptr
    llvm.br ^bb10(%30 : !llvm.struct<(ptr, i32)>)
  ^bb10(%31: !llvm.struct<(ptr, i32)>):  // 2 preds: ^bb8, ^bb9
    llvm.call @_ZN6google8protobuf8internal10LogMessageD1Ev(%11) : (!llvm.ptr) -> ()
    llvm.intr.lifetime.end 56, %17 : !llvm.ptr
    llvm.resume %31 : !llvm.struct<(ptr, i32)>
  ^bb11:  // 3 preds: ^bb3, ^bb4, ^bb7
    llvm.return
  }
  llvm.func local_unnamed_addr @_ZN6google8protobuf15UnknownFieldSet9MergeFromERKS1_(!llvm.ptr {llvm.align = 8 : i64, llvm.dereferenceable = 24 : i64, llvm.nonnull, llvm.noundef}, !llvm.ptr {llvm.align = 8 : i64, llvm.dereferenceable = 24 : i64, llvm.nonnull, llvm.noundef}) attributes {frame_pointer = #llvm.framePointerKind<none>, passthrough = [["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], ["tune-cpu", "generic"]], target_cpu = "x86-64", target_features = #llvm.target_features<["+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>}
  llvm.func linkonce_odr unnamed_addr @_ZNSt6vectorIN6google8protobuf12UnknownFieldESaIS2_EED2Ev(%arg0: !llvm.ptr {llvm.align = 8 : i64, llvm.dereferenceable = 24 : i64, llvm.nonnull, llvm.noundef}) comdat(@__llvm_global_comdat::@_ZNSt6vectorIN6google8protobuf12UnknownFieldESaIS2_EED2Ev) attributes {alignment = 2 : i64, frame_pointer = #llvm.framePointerKind<none>, passthrough = ["nounwind", ["uwtable", "2"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], ["tune-cpu", "generic"]], personality = @__gxx_personality_v0, target_cpu = "x86-64", target_features = #llvm.target_features<["+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>} {
    %0 = llvm.mlir.constant(0 : i64) : i64
    %1 = llvm.mlir.constant(0 : i32) : i32
    %2 = llvm.mlir.zero : !llvm.ptr
    %3 = llvm.getelementptr inbounds %arg0[%0, 0, 0, 0, 0] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.std::vector", (struct<"struct.std::_Vector_base", (struct<"struct.std::_Vector_base<google::protobuf::UnknownField, std::allocator<google::protobuf::UnknownField>>::_Vector_impl", (struct<"struct.std::_Vector_base<google::protobuf::UnknownField, std::allocator<google::protobuf::UnknownField>>::_Vector_impl_data", (ptr, ptr, ptr)>)>)>)>
    %4 = llvm.load %3 {alignment = 8 : i64, tbaa = [#tbaa_tag10]} : !llvm.ptr -> !llvm.ptr
    %5 = llvm.icmp "eq" %4, %2 : !llvm.ptr
    llvm.cond_br %5, ^bb2, ^bb1
  ^bb1:  // pred: ^bb0
    %6 = llvm.bitcast %4 : !llvm.ptr to !llvm.ptr
    llvm.call @_ZdlPv(%6) : (!llvm.ptr) -> ()
    llvm.br ^bb2
  ^bb2:  // 2 preds: ^bb0, ^bb1
    llvm.return
  }
  llvm.func local_unnamed_addr @_ZN6google8protobuf15UnknownFieldSet13ClearFallbackEv(!llvm.ptr {llvm.align = 8 : i64, llvm.dereferenceable = 24 : i64, llvm.nonnull, llvm.noundef}) attributes {frame_pointer = #llvm.framePointerKind<none>, passthrough = [["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], ["tune-cpu", "generic"]], target_cpu = "x86-64", target_features = #llvm.target_features<["+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>}
  llvm.func linkonce_odr local_unnamed_addr @_ZN6google8protobuf8internal16InternalMetadata27mutable_unknown_fields_slowINS0_15UnknownFieldSetEEEPT_v(%arg0: !llvm.ptr {llvm.align = 8 : i64, llvm.dereferenceable = 8 : i64, llvm.nonnull, llvm.noundef}) -> (!llvm.ptr {llvm.noundef}) comdat(@__llvm_global_comdat::@_ZN6google8protobuf8internal16InternalMetadata27mutable_unknown_fields_slowINS0_15UnknownFieldSetEEEPT_v) attributes {alignment = 2 : i64, frame_pointer = #llvm.framePointerKind<none>, passthrough = ["noinline", ["uwtable", "2"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], ["tune-cpu", "generic"]], personality = @__gxx_personality_v0, target_cpu = "x86-64", target_features = #llvm.target_features<["+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>} {
    %0 = llvm.mlir.constant(0 : i64) : i64
    %1 = llvm.mlir.constant(0 : i32) : i32
    %2 = llvm.mlir.constant(1 : i64) : i64
    %3 = llvm.mlir.constant(-2 : i64) : i64
    %4 = llvm.mlir.zero : !llvm.ptr
    %5 = llvm.mlir.constant(4 : i32) : i32
    %6 = llvm.mlir.constant("N6google8protobuf8internal16InternalMetadata13ContainerBaseE\00") : !llvm.array<61 x i8>
    %7 = llvm.mlir.addressof @_ZTSN6google8protobuf8internal16InternalMetadata13ContainerBaseE : !llvm.ptr
    %8 = llvm.mlir.constant(2 : i64) : i64
    %9 = llvm.mlir.addressof @_ZTVN10__cxxabiv117__class_type_infoE : !llvm.ptr
    %10 = llvm.getelementptr inbounds %9[%8] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.ptr
    %11 = llvm.mlir.undef : !llvm.struct<(ptr, ptr)>
    %12 = llvm.insertvalue %10, %11[0] : !llvm.struct<(ptr, ptr)> 
    %13 = llvm.insertvalue %7, %12[1] : !llvm.struct<(ptr, ptr)> 
    %14 = llvm.mlir.addressof @_ZTIN6google8protobuf8internal16InternalMetadata13ContainerBaseE : !llvm.ptr
    %15 = llvm.mlir.constant("N6google8protobuf8internal16InternalMetadata9ContainerINS0_15UnknownFieldSetEEE\00") : !llvm.array<80 x i8>
    %16 = llvm.mlir.addressof @_ZTSN6google8protobuf8internal16InternalMetadata9ContainerINS0_15UnknownFieldSetEEE : !llvm.ptr
    %17 = llvm.mlir.addressof @_ZTVN10__cxxabiv120__si_class_type_infoE : !llvm.ptr
    %18 = llvm.getelementptr inbounds %17[%8] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.ptr
    %19 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, ptr)>
    %20 = llvm.insertvalue %18, %19[0] : !llvm.struct<(ptr, ptr, ptr)> 
    %21 = llvm.insertvalue %16, %20[1] : !llvm.struct<(ptr, ptr, ptr)> 
    %22 = llvm.insertvalue %14, %21[2] : !llvm.struct<(ptr, ptr, ptr)> 
    %23 = llvm.mlir.addressof @_ZTIN6google8protobuf8internal16InternalMetadata9ContainerINS0_15UnknownFieldSetEEE : !llvm.ptr
    %24 = llvm.mlir.constant(32 : i64) : i64
    %25 = llvm.mlir.addressof @__gxx_personality_v0 : !llvm.ptr
    %26 = llvm.mlir.addressof @_ZN6google8protobuf8internal21arena_destruct_objectINS1_16InternalMetadata9ContainerINS0_15UnknownFieldSetEEEEEvPv : !llvm.ptr
    %27 = llvm.mlir.constant(0 : i8) : i8
    %28 = llvm.mlir.constant(8 : i64) : i64
    %29 = llvm.getelementptr inbounds %arg0[%0, 0] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.google::protobuf::internal::InternalMetadata", (ptr)>
    %30 = llvm.load %29 {alignment = 8 : i64, tbaa = [#tbaa_tag5]} : !llvm.ptr -> !llvm.ptr
    %31 = llvm.ptrtoint %30 : !llvm.ptr to i64
    %32 = llvm.and %31, %2  : i64
    %33 = llvm.icmp "eq" %32, %0 : i64
    %34 = llvm.and %31, %3  : i64
    llvm.cond_br %33 weights([2000, 1]), ^bb2, ^bb1
  ^bb1:  // pred: ^bb0
    %35 = llvm.inttoptr %34 : i64 to !llvm.ptr
    %36 = llvm.getelementptr inbounds %35[%0, 0] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.google::protobuf::internal::InternalMetadata::ContainerBase", (ptr)>
    %37 = llvm.load %36 {alignment = 8 : i64, tbaa = [#tbaa_tag9]} : !llvm.ptr -> !llvm.ptr
    llvm.br ^bb3(%37 : !llvm.ptr)
  ^bb2:  // pred: ^bb0
    %38 = llvm.inttoptr %34 : i64 to !llvm.ptr
    llvm.br ^bb3(%38 : !llvm.ptr)
  ^bb3(%39: !llvm.ptr):  // 2 preds: ^bb1, ^bb2
    %40 = llvm.icmp "eq" %39, %4 : !llvm.ptr
    llvm.cond_br %40, ^bb4, ^bb5
  ^bb4:  // pred: ^bb3
    %41 = llvm.call @_Znwm(%24) : (i64) -> !llvm.ptr
    "llvm.intr.memset"(%41, %27, %24) <{isVolatile = false}> : (!llvm.ptr, i8, i64) -> ()
    llvm.br ^bb8(%41 : !llvm.ptr)
  ^bb5:  // pred: ^bb3
    %42 = llvm.getelementptr inbounds %39[%0, 4] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.google::protobuf::Arena", (struct<"class.google::protobuf::internal::ArenaImpl", (struct<"struct.std::atomic.10", (struct<"struct.std::__atomic_base.11", (ptr)>)>, struct<"struct.std::atomic.10", (struct<"struct.std::__atomic_base.11", (ptr)>)>, struct<"struct.std::atomic.12", (struct<"struct.std::__atomic_base.13", (i64)>)>, ptr, i64, struct<"struct.google::protobuf::internal::ArenaImpl::Options", (i64, i64, ptr, i64, ptr, ptr)>)>, ptr, ptr, ptr, ptr)>
    %43 = llvm.load %42 {alignment = 8 : i64, tbaa = [#tbaa_tag30]} : !llvm.ptr -> !llvm.ptr
    %44 = llvm.icmp "eq" %43, %4 : !llvm.ptr
    llvm.cond_br %44 weights([2000, 1]), ^bb7, ^bb6
  ^bb6:  // pred: ^bb5
    llvm.call @_ZNK6google8protobuf5Arena17OnArenaAllocationEPKSt9type_infom(%39, %23, %24) : (!llvm.ptr, !llvm.ptr, i64) -> ()
    llvm.br ^bb7
  ^bb7:  // 2 preds: ^bb5, ^bb6
    %45 = llvm.getelementptr inbounds %39[%0, 0] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.google::protobuf::Arena", (struct<"class.google::protobuf::internal::ArenaImpl", (struct<"struct.std::atomic.10", (struct<"struct.std::__atomic_base.11", (ptr)>)>, struct<"struct.std::atomic.10", (struct<"struct.std::__atomic_base.11", (ptr)>)>, struct<"struct.std::atomic.12", (struct<"struct.std::__atomic_base.13", (i64)>)>, ptr, i64, struct<"struct.google::protobuf::internal::ArenaImpl::Options", (i64, i64, ptr, i64, ptr, ptr)>)>, ptr, ptr, ptr, ptr)>
    %46 = llvm.call @_ZN6google8protobuf8internal9ArenaImpl28AllocateAlignedAndAddCleanupEmPFvPvE(%45, %24, %26) : (!llvm.ptr, i64, !llvm.ptr) -> !llvm.ptr
    "llvm.intr.memset"(%46, %27, %24) <{isVolatile = false}> : (!llvm.ptr, i8, i64) -> ()
    llvm.br ^bb8(%46 : !llvm.ptr)
  ^bb8(%47: !llvm.ptr):  // 2 preds: ^bb4, ^bb7
    %48 = llvm.ptrtoint %47 : !llvm.ptr to i64
    %49 = llvm.or %48, %2  : i64
    %50 = llvm.inttoptr %49 : i64 to !llvm.ptr
    llvm.store %50, %29 {alignment = 8 : i64, tbaa = [#tbaa_tag5]} : !llvm.ptr, !llvm.ptr
    %51 = llvm.bitcast %47 : !llvm.ptr to !llvm.ptr
    llvm.store %39, %51 {alignment = 8 : i64, tbaa = [#tbaa_tag9]} : !llvm.ptr, !llvm.ptr
    %52 = llvm.getelementptr inbounds %47[%28] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %53 = llvm.bitcast %52 : !llvm.ptr to !llvm.ptr
    llvm.return %53 : !llvm.ptr
  }
  llvm.func linkonce_odr @_ZN6google8protobuf8internal21arena_destruct_objectINS1_16InternalMetadata9ContainerINS0_15UnknownFieldSetEEEEEvPv(%arg0: !llvm.ptr {llvm.noundef}) comdat(@__llvm_global_comdat::@_ZN6google8protobuf8internal21arena_destruct_objectINS1_16InternalMetadata9ContainerINS0_15UnknownFieldSetEEEEEvPv) attributes {frame_pointer = #llvm.framePointerKind<none>, passthrough = ["nounwind", ["uwtable", "2"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], ["tune-cpu", "generic"]], personality = @__gxx_personality_v0, target_cpu = "x86-64", target_features = #llvm.target_features<["+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>} {
    %0 = llvm.mlir.constant(8 : i64) : i64
    %1 = llvm.mlir.constant(16 : i64) : i64
    %2 = llvm.mlir.zero : !llvm.ptr
    %3 = llvm.getelementptr inbounds %arg0[%0] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %4 = llvm.bitcast %3 : !llvm.ptr to !llvm.ptr
    %5 = llvm.load %4 {alignment = 8 : i64, tbaa = [#tbaa_tag2]} : !llvm.ptr -> !llvm.ptr
    %6 = llvm.getelementptr inbounds %arg0[%1] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %7 = llvm.bitcast %6 : !llvm.ptr to !llvm.ptr
    %8 = llvm.load %7 {alignment = 8 : i64, tbaa = [#tbaa_tag2]} : !llvm.ptr -> !llvm.ptr
    %9 = llvm.icmp "eq" %5, %8 : !llvm.ptr
    llvm.cond_br %9, ^bb3(%5 : !llvm.ptr), ^bb1
  ^bb1:  // pred: ^bb0
    %10 = llvm.bitcast %3 : !llvm.ptr to !llvm.ptr
    llvm.invoke @_ZN6google8protobuf15UnknownFieldSet13ClearFallbackEv(%10) to ^bb2 unwind ^bb5 : (!llvm.ptr) -> ()
  ^bb2:  // pred: ^bb1
    %11 = llvm.load %4 {alignment = 8 : i64, tbaa = [#tbaa_tag10]} : !llvm.ptr -> !llvm.ptr
    llvm.br ^bb3(%11 : !llvm.ptr)
  ^bb3(%12: !llvm.ptr):  // 2 preds: ^bb0, ^bb2
    %13 = llvm.icmp "eq" %12, %2 : !llvm.ptr
    llvm.cond_br %13, ^bb6, ^bb4
  ^bb4:  // pred: ^bb3
    %14 = llvm.bitcast %12 : !llvm.ptr to !llvm.ptr
    llvm.call @_ZdlPv(%14) : (!llvm.ptr) -> ()
    llvm.br ^bb6
  ^bb5:  // pred: ^bb1
    %15 = llvm.landingpad (catch %2 : !llvm.ptr) : !llvm.struct<(ptr, i32)>
    %16 = llvm.extractvalue %15[0] : !llvm.struct<(ptr, i32)> 
    %17 = llvm.bitcast %3 : !llvm.ptr to !llvm.ptr
    llvm.call @_ZNSt6vectorIN6google8protobuf12UnknownFieldESaIS2_EED2Ev(%17) : (!llvm.ptr) -> ()
    llvm.call @__clang_call_terminate(%16) : (!llvm.ptr) -> ()
    llvm.unreachable
  ^bb6:  // 2 preds: ^bb3, ^bb4
    llvm.return
  }
  llvm.func local_unnamed_addr @__dynamic_cast(!llvm.ptr, !llvm.ptr, !llvm.ptr, i64) -> !llvm.ptr attributes {memory = #llvm.memory_effects<other = read, argMem = read, inaccessibleMem = read>, passthrough = ["nofree", "nounwind"]}
  llvm.func linkonce_odr local_unnamed_addr @_ZN6google8protobuf8internal20RepeatedPtrFieldBase7DestroyINS0_16RepeatedPtrFieldIN8tutorial18Person_PhoneNumberEE11TypeHandlerEEEvv(%arg0: !llvm.ptr {llvm.align = 8 : i64, llvm.dereferenceable = 24 : i64, llvm.nonnull, llvm.noundef}) comdat(@__llvm_global_comdat::@_ZN6google8protobuf8internal20RepeatedPtrFieldBase7DestroyINS0_16RepeatedPtrFieldIN8tutorial18Person_PhoneNumberEE11TypeHandlerEEEvv) attributes {alignment = 2 : i64, frame_pointer = #llvm.framePointerKind<none>, passthrough = [["uwtable", "2"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], ["tune-cpu", "generic"]], personality = @__gxx_personality_v0, target_cpu = "x86-64", target_features = #llvm.target_features<["+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>} {
    %0 = llvm.mlir.constant(0 : i64) : i64
    %1 = llvm.mlir.constant(3 : i32) : i32
    %2 = llvm.mlir.zero : !llvm.ptr
    %3 = llvm.mlir.constant(0 : i32) : i32
    %4 = llvm.mlir.constant(false) : i1
    %5 = llvm.mlir.constant(1 : i32) : i32
    %6 = llvm.mlir.constant(8 : i64) : i64
    %7 = llvm.mlir.constant(1 : i64) : i64
    %8 = llvm.getelementptr inbounds %arg0[%0, 3] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.google::protobuf::internal::RepeatedPtrFieldBase", (ptr, i32, i32, ptr)>
    %9 = llvm.load %8 {alignment = 8 : i64, tbaa = [#tbaa_tag15]} : !llvm.ptr -> !llvm.ptr
    %10 = llvm.icmp "ne" %9, %2 : !llvm.ptr
    %11 = llvm.getelementptr inbounds %arg0[%0, 0] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.google::protobuf::internal::RepeatedPtrFieldBase", (ptr, i32, i32, ptr)>
    %12 = llvm.load %11 {alignment = 8 : i64} : !llvm.ptr -> !llvm.ptr
    %13 = llvm.icmp "eq" %12, %2 : !llvm.ptr
    %14 = llvm.select %10, %13, %4 : i1, i1
    llvm.cond_br %14, ^bb1, ^bb11
  ^bb1:  // pred: ^bb0
    %15 = llvm.bitcast %9 : !llvm.ptr to !llvm.ptr
    %16 = llvm.getelementptr inbounds %9[%0, 0] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.google::protobuf::internal::RepeatedPtrFieldBase::Rep", (i32, array<1 x ptr>)>
    %17 = llvm.load %16 {alignment = 8 : i64, tbaa = [#tbaa_tag18]} : !llvm.ptr -> i32
    %18 = llvm.icmp "sgt" %17, %3 : i32
    llvm.cond_br %18, ^bb2, ^bb4(%15 : !llvm.ptr)
  ^bb2:  // pred: ^bb1
    %19 = llvm.zext %17 : i32 to i64
    llvm.br ^bb5(%0 : i64)
  ^bb3:  // pred: ^bb10
    %20 = llvm.bitcast %8 : !llvm.ptr to !llvm.ptr
    %21 = llvm.load %20 {alignment = 8 : i64, tbaa = [#tbaa_tag15]} : !llvm.ptr -> !llvm.ptr
    llvm.br ^bb4(%21 : !llvm.ptr)
  ^bb4(%22: !llvm.ptr):  // 2 preds: ^bb1, ^bb3
    llvm.call @_ZdlPv(%22) : (!llvm.ptr) -> ()
    llvm.br ^bb11
  ^bb5(%23: i64):  // 2 preds: ^bb2, ^bb10
    %24 = llvm.getelementptr inbounds %9[%0, 1, %23] : (!llvm.ptr, i64, i64) -> !llvm.ptr, !llvm.struct<"struct.google::protobuf::internal::RepeatedPtrFieldBase::Rep", (i32, array<1 x ptr>)>
    %25 = llvm.load %24 {alignment = 8 : i64, tbaa = [#tbaa_tag2]} : !llvm.ptr -> !llvm.ptr
    %26 = llvm.icmp "eq" %25, %2 : !llvm.ptr
    llvm.cond_br %26, ^bb10, ^bb6
  ^bb6:  // pred: ^bb5
    %27 = llvm.bitcast %25 : !llvm.ptr to !llvm.ptr
    llvm.invoke @_ZN8tutorial18Person_PhoneNumber10SharedDtorEv(%27) to ^bb7 unwind ^bb8 : (!llvm.ptr) -> ()
  ^bb7:  // pred: ^bb6
    %28 = llvm.getelementptr inbounds %25[%6] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %29 = llvm.bitcast %28 : !llvm.ptr to !llvm.ptr
    llvm.invoke @_ZN6google8protobuf8internal16InternalMetadata6DeleteINS0_15UnknownFieldSetEEEvv(%29) to ^bb9 unwind ^bb8 : (!llvm.ptr) -> ()
  ^bb8:  // 2 preds: ^bb6, ^bb7
    %30 = llvm.landingpad (catch %2 : !llvm.ptr) : !llvm.struct<(ptr, i32)>
    %31 = llvm.extractvalue %30[0] : !llvm.struct<(ptr, i32)> 
    llvm.call @__clang_call_terminate(%31) : (!llvm.ptr) -> ()
    llvm.unreachable
  ^bb9:  // pred: ^bb7
    llvm.call @_ZdlPv(%25) : (!llvm.ptr) -> ()
    llvm.br ^bb10
  ^bb10:  // 2 preds: ^bb5, ^bb9
    %32 = llvm.add %23, %7 overflow<nsw, nuw>  : i64
    %33 = llvm.icmp "eq" %32, %19 : i64
    llvm.cond_br %33, ^bb3, ^bb5(%32 : i64) {loop_annotation = #loop_annotation}
  ^bb11:  // 2 preds: ^bb0, ^bb4
    llvm.store %2, %8 {alignment = 8 : i64, tbaa = [#tbaa_tag15]} : !llvm.ptr, !llvm.ptr
    llvm.return
  }
  llvm.func linkonce_odr unnamed_addr @_ZN6google8protobuf8internal20RepeatedPtrFieldBaseD2Ev(%arg0: !llvm.ptr {llvm.align = 8 : i64, llvm.dereferenceable = 24 : i64, llvm.nonnull, llvm.noundef}) comdat(@__llvm_global_comdat::@_ZN6google8protobuf8internal20RepeatedPtrFieldBaseD2Ev) attributes {alignment = 2 : i64, frame_pointer = #llvm.framePointerKind<none>, passthrough = ["nounwind", ["uwtable", "2"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], ["tune-cpu", "generic"]], personality = @__gxx_personality_v0, target_cpu = "x86-64", target_features = #llvm.target_features<["+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>} {
    %0 = llvm.mlir.constant(0 : i64) : i64
    %1 = llvm.mlir.constant(0 : i32) : i32
    %2 = llvm.mlir.zero : !llvm.ptr
    %3 = llvm.getelementptr inbounds %arg0[%0, 0] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.google::protobuf::internal::RepeatedPtrFieldBase", (ptr, i32, i32, ptr)>
    %4 = llvm.load %3 {alignment = 8 : i64, tbaa = [#tbaa_tag7]} : !llvm.ptr -> !llvm.ptr
    %5 = llvm.icmp "eq" %4, %2 : !llvm.ptr
    llvm.cond_br %5, ^bb2, ^bb1
  ^bb1:  // pred: ^bb0
    %6 = llvm.getelementptr inbounds %4[%0, 0] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.google::protobuf::Arena", (struct<"class.google::protobuf::internal::ArenaImpl", (struct<"struct.std::atomic.10", (struct<"struct.std::__atomic_base.11", (ptr)>)>, struct<"struct.std::atomic.10", (struct<"struct.std::__atomic_base.11", (ptr)>)>, struct<"struct.std::atomic.12", (struct<"struct.std::__atomic_base.13", (i64)>)>, ptr, i64, struct<"struct.google::protobuf::internal::ArenaImpl::Options", (i64, i64, ptr, i64, ptr, ptr)>)>, ptr, ptr, ptr, ptr)>
    %7 = llvm.invoke @_ZNK6google8protobuf8internal9ArenaImpl14SpaceAllocatedEv(%6) to ^bb2 unwind ^bb3 : (!llvm.ptr) -> i64
  ^bb2:  // 2 preds: ^bb0, ^bb1
    llvm.return
  ^bb3:  // pred: ^bb1
    %8 = llvm.landingpad (catch %2 : !llvm.ptr) : !llvm.struct<(ptr, i32)>
    %9 = llvm.extractvalue %8[0] : !llvm.struct<(ptr, i32)> 
    llvm.call @__clang_call_terminate(%9) : (!llvm.ptr) -> ()
    llvm.unreachable
  }
  llvm.func local_unnamed_addr @_ZNK6google8protobuf8internal9ArenaImpl14SpaceAllocatedEv(!llvm.ptr {llvm.align = 8 : i64, llvm.dereferenceable = 88 : i64, llvm.nonnull, llvm.noundef}) -> (i64 {llvm.noundef}) attributes {frame_pointer = #llvm.framePointerKind<none>, passthrough = [["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], ["tune-cpu", "generic"]], target_cpu = "x86-64", target_features = #llvm.target_features<["+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>}
  llvm.func linkonce_odr local_unnamed_addr @_ZN6google8protobuf8internal18EpsCopyInputStream9PushLimitEPKci(%arg0: !llvm.ptr {llvm.align = 8 : i64, llvm.dereferenceable = 88 : i64, llvm.nonnull, llvm.noundef}, %arg1: !llvm.ptr {llvm.noundef}, %arg2: i32 {llvm.noundef}) -> (i32 {llvm.noundef}) comdat(@__llvm_global_comdat::@_ZN6google8protobuf8internal18EpsCopyInputStream9PushLimitEPKci) attributes {alignment = 2 : i64, frame_pointer = #llvm.framePointerKind<none>, passthrough = ["mustprogress", ["uwtable", "2"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], ["tune-cpu", "generic"]], personality = @__gxx_personality_v0, target_cpu = "x86-64", target_features = #llvm.target_features<["+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>} {
    %0 = llvm.mlir.constant(1 : i32) : i32
    %1 = llvm.mlir.constant(0 : i64) : i64
    %2 = llvm.mlir.constant(0 : i32) : i32
    %3 = llvm.mlir.constant(2147483632 : i32) : i32
    %4 = llvm.mlir.constant(3 : i32) : i32
    %5 = llvm.mlir.constant("/usr/include/google/protobuf/parse_context.h\00") : !llvm.array<45 x i8>
    %6 = llvm.mlir.addressof @".str.13" : !llvm.ptr
    %7 = llvm.mlir.constant(128 : i32) : i32
    %8 = llvm.mlir.constant("CHECK failed: limit >= 0 && limit <= INT_MAX - kSlopBytes: \00") : !llvm.array<60 x i8>
    %9 = llvm.mlir.addressof @".str.20" : !llvm.ptr
    %10 = llvm.mlir.constant(4 : i32) : i32
    %11 = llvm.alloca %0 x !llvm.struct<"class.google::protobuf::internal::LogMessage", (i32, ptr, i32, struct<"class.std::__cxx11::basic_string", (struct<"struct.std::__cxx11::basic_string<char>::_Alloc_hider", (ptr)>, i64, struct<"union.anon", (i64, array<8 x i8>)>)>)> {alignment = 8 : i64} : (i32) -> !llvm.ptr
    %12 = llvm.alloca %0 x !llvm.struct<"class.google::protobuf::internal::LogFinisher", (i8)> {alignment = 1 : i64} : (i32) -> !llvm.ptr
    %13 = llvm.getelementptr inbounds %12[%1, 0] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.google::protobuf::internal::LogFinisher", (i8)>
    llvm.intr.lifetime.start 1, %13 : !llvm.ptr
    %14 = llvm.icmp "ult" %arg2, %3 : i32
    llvm.cond_br %14, ^bb3, ^bb1
  ^bb1:  // pred: ^bb0
    %15 = llvm.bitcast %11 : !llvm.ptr to !llvm.ptr
    llvm.intr.lifetime.start 56, %15 : !llvm.ptr
    llvm.call @_ZN6google8protobuf8internal10LogMessageC1ENS0_8LogLevelEPKci(%11, %4, %6, %7) : (!llvm.ptr, i32, !llvm.ptr, i32) -> ()
    %16 = llvm.invoke @_ZN6google8protobuf8internal10LogMessagelsEPKc(%11, %9) to ^bb2 unwind ^bb6 : (!llvm.ptr, !llvm.ptr) -> !llvm.ptr
  ^bb2:  // pred: ^bb1
    llvm.invoke @_ZN6google8protobuf8internal11LogFinisheraSERNS1_10LogMessageE(%12, %16) to ^bb4 unwind ^bb7 : (!llvm.ptr, !llvm.ptr) -> ()
  ^bb3:  // pred: ^bb0
    llvm.intr.lifetime.end 1, %13 : !llvm.ptr
    llvm.br ^bb5
  ^bb4:  // pred: ^bb2
    llvm.intr.lifetime.end 1, %13 : !llvm.ptr
    llvm.call @_ZN6google8protobuf8internal10LogMessageD1Ev(%11) : (!llvm.ptr) -> ()
    llvm.intr.lifetime.end 56, %15 : !llvm.ptr
    llvm.br ^bb5
  ^bb5:  // 2 preds: ^bb3, ^bb4
    %17 = llvm.getelementptr inbounds %arg0[%1, 1] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.google::protobuf::internal::EpsCopyInputStream", (ptr, ptr, ptr, i32, i32, ptr, array<32 x i8>, i64, i32, i32)>
    %18 = llvm.load %17 {alignment = 8 : i64, tbaa = [#tbaa_tag20]} : !llvm.ptr -> !llvm.ptr
    %19 = llvm.ptrtoint %arg1 : !llvm.ptr to i64
    %20 = llvm.ptrtoint %18 : !llvm.ptr to i64
    %21 = llvm.sub %19, %20  : i64
    %22 = llvm.trunc %21 : i64 to i32
    %23 = llvm.add %22, %arg2 overflow<nsw>  : i32
    %24 = llvm.icmp "slt" %23, %2 : i32
    %25 = llvm.select %24, %23, %2 : i1, i32
    %26 = llvm.sext %25 : i32 to i64
    %27 = llvm.getelementptr inbounds %18[%26] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %28 = llvm.getelementptr inbounds %arg0[%1, 0] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.google::protobuf::internal::EpsCopyInputStream", (ptr, ptr, ptr, i32, i32, ptr, array<32 x i8>, i64, i32, i32)>
    llvm.store %27, %28 {alignment = 8 : i64, tbaa = [#tbaa_tag21]} : !llvm.ptr, !llvm.ptr
    %29 = llvm.getelementptr inbounds %arg0[%1, 4] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.google::protobuf::internal::EpsCopyInputStream", (ptr, ptr, ptr, i32, i32, ptr, array<32 x i8>, i64, i32, i32)>
    %30 = llvm.load %29 {alignment = 4 : i64, tbaa = [#tbaa_tag19]} : !llvm.ptr -> i32
    llvm.store %23, %29 {alignment = 4 : i64, tbaa = [#tbaa_tag19]} : i32, !llvm.ptr
    %31 = llvm.sub %30, %23 overflow<nsw>  : i32
    llvm.return %31 : i32
  ^bb6:  // pred: ^bb1
    %32 = llvm.landingpad cleanup : !llvm.struct<(ptr, i32)>
    llvm.br ^bb8(%32 : !llvm.struct<(ptr, i32)>)
  ^bb7:  // pred: ^bb2
    %33 = llvm.landingpad cleanup : !llvm.struct<(ptr, i32)>
    llvm.intr.lifetime.end 1, %13 : !llvm.ptr
    llvm.br ^bb8(%33 : !llvm.struct<(ptr, i32)>)
  ^bb8(%34: !llvm.struct<(ptr, i32)>):  // 2 preds: ^bb6, ^bb7
    llvm.call @_ZN6google8protobuf8internal10LogMessageD1Ev(%11) : (!llvm.ptr) -> ()
    llvm.intr.lifetime.end 56, %15 : !llvm.ptr
    llvm.resume %34 : !llvm.struct<(ptr, i32)>
  }
  llvm.func local_unnamed_addr @_ZN6google8protobuf8internal16ReadSizeFallbackEPKcj(!llvm.ptr {llvm.noundef}, i32 {llvm.noundef}) -> !llvm.struct<(ptr, i32)> attributes {frame_pointer = #llvm.framePointerKind<none>, passthrough = [["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], ["tune-cpu", "generic"]], target_cpu = "x86-64", target_features = #llvm.target_features<["+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>}
  llvm.func unnamed_addr @_ZN6google8protobuf9Timestamp14_InternalParseEPKcPNS0_8internal12ParseContextE(!llvm.ptr {llvm.align = 8 : i64, llvm.dereferenceable = 32 : i64, llvm.nonnull, llvm.noundef}, !llvm.ptr {llvm.noundef}, !llvm.ptr {llvm.noundef}) -> (!llvm.ptr {llvm.noundef}) attributes {frame_pointer = #llvm.framePointerKind<none>, passthrough = [["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], ["tune-cpu", "generic"]], target_cpu = "x86-64", target_features = #llvm.target_features<["+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>}
  llvm.func unnamed_addr @_ZNK6google8protobuf9Timestamp18_InternalSerializeEPhPNS0_2io19EpsCopyOutputStreamE(!llvm.ptr {llvm.align = 8 : i64, llvm.dereferenceable = 32 : i64, llvm.nonnull, llvm.noundef}, !llvm.ptr {llvm.noundef}, !llvm.ptr {llvm.noundef}) -> (!llvm.ptr {llvm.noundef}) attributes {frame_pointer = #llvm.framePointerKind<none>, passthrough = [["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], ["tune-cpu", "generic"]], target_cpu = "x86-64", target_features = #llvm.target_features<["+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>}
  llvm.func unnamed_addr @_ZNK6google8protobuf9Timestamp12ByteSizeLongEv(!llvm.ptr {llvm.align = 8 : i64, llvm.dereferenceable = 32 : i64, llvm.nonnull, llvm.noundef}) -> (i64 {llvm.noundef}) attributes {frame_pointer = #llvm.framePointerKind<none>, passthrough = [["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], ["tune-cpu", "generic"]], target_cpu = "x86-64", target_features = #llvm.target_features<["+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>}
  llvm.func linkonce_odr local_unnamed_addr @_ZN6google8protobuf8internal20RepeatedPtrFieldBase9MergeFromINS0_16RepeatedPtrFieldIN8tutorial18Person_PhoneNumberEE11TypeHandlerEEEvRKS2_(%arg0: !llvm.ptr {llvm.align = 8 : i64, llvm.dereferenceable = 24 : i64, llvm.nonnull, llvm.noundef}, %arg1: !llvm.ptr {llvm.align = 8 : i64, llvm.dereferenceable = 24 : i64, llvm.nonnull, llvm.noundef}) comdat(@__llvm_global_comdat::@_ZN6google8protobuf8internal20RepeatedPtrFieldBase9MergeFromINS0_16RepeatedPtrFieldIN8tutorial18Person_PhoneNumberEE11TypeHandlerEEEvRKS2_) attributes {alignment = 2 : i64, frame_pointer = #llvm.framePointerKind<none>, passthrough = ["inlinehint", "mustprogress", ["uwtable", "2"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], ["tune-cpu", "generic"]], personality = @__gxx_personality_v0, target_cpu = "x86-64", target_features = #llvm.target_features<["+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>} {
    %0 = llvm.mlir.constant(1 : i32) : i32
    %1 = llvm.mlir.constant(0 : i64) : i64
    %2 = llvm.mlir.constant(0 : i32) : i32
    %3 = llvm.mlir.constant(3 : i32) : i32
    %4 = llvm.mlir.constant("/usr/include/google/protobuf/repeated_field.h\00") : !llvm.array<46 x i8>
    %5 = llvm.mlir.addressof @".str.16" : !llvm.ptr
    %6 = llvm.mlir.constant(1787 : i32) : i32
    %7 = llvm.mlir.constant("CHECK failed: (&other) != (this): \00") : !llvm.array<35 x i8>
    %8 = llvm.mlir.addressof @".str.21" : !llvm.ptr
    %9 = llvm.mlir.constant(1 : i64) : i64
    %10 = llvm.alloca %0 x !llvm.struct<"class.google::protobuf::internal::LogMessage", (i32, ptr, i32, struct<"class.std::__cxx11::basic_string", (struct<"struct.std::__cxx11::basic_string<char>::_Alloc_hider", (ptr)>, i64, struct<"union.anon", (i64, array<8 x i8>)>)>)> {alignment = 8 : i64} : (i32) -> !llvm.ptr
    %11 = llvm.alloca %0 x !llvm.struct<"class.google::protobuf::internal::LogFinisher", (i8)> {alignment = 1 : i64} : (i32) -> !llvm.ptr
    %12 = llvm.icmp "eq" %arg1, %arg0 : !llvm.ptr
    %13 = llvm.getelementptr inbounds %11[%1, 0] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.google::protobuf::internal::LogFinisher", (i8)>
    llvm.intr.lifetime.start 1, %13 : !llvm.ptr
    llvm.cond_br %12, ^bb1, ^bb3
  ^bb1:  // pred: ^bb0
    %14 = llvm.bitcast %10 : !llvm.ptr to !llvm.ptr
    llvm.intr.lifetime.start 56, %14 : !llvm.ptr
    llvm.call @_ZN6google8protobuf8internal10LogMessageC1ENS0_8LogLevelEPKci(%10, %3, %5, %6) : (!llvm.ptr, i32, !llvm.ptr, i32) -> ()
    %15 = llvm.invoke @_ZN6google8protobuf8internal10LogMessagelsEPKc(%10, %8) to ^bb2 unwind ^bb6 : (!llvm.ptr, !llvm.ptr) -> !llvm.ptr
  ^bb2:  // pred: ^bb1
    llvm.invoke @_ZN6google8protobuf8internal11LogFinisheraSERNS1_10LogMessageE(%11, %15) to ^bb4 unwind ^bb7 : (!llvm.ptr, !llvm.ptr) -> ()
  ^bb3:  // pred: ^bb0
    llvm.intr.lifetime.end 1, %13 : !llvm.ptr
    llvm.br ^bb5
  ^bb4:  // pred: ^bb2
    llvm.intr.lifetime.end 1, %13 : !llvm.ptr
    llvm.call @_ZN6google8protobuf8internal10LogMessageD1Ev(%10) : (!llvm.ptr) -> ()
    llvm.intr.lifetime.end 56, %14 : !llvm.ptr
    llvm.br ^bb5
  ^bb5:  // 2 preds: ^bb3, ^bb4
    %16 = llvm.getelementptr inbounds %arg1[%1, 1] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.google::protobuf::internal::RepeatedPtrFieldBase", (ptr, i32, i32, ptr)>
    %17 = llvm.load %16 {alignment = 8 : i64, tbaa = [#tbaa_tag17]} : !llvm.ptr -> i32
    %18 = llvm.icmp "eq" %17, %2 : i32
    llvm.cond_br %18, ^bb17, ^bb9
  ^bb6:  // pred: ^bb1
    %19 = llvm.landingpad cleanup : !llvm.struct<(ptr, i32)>
    llvm.br ^bb8(%19 : !llvm.struct<(ptr, i32)>)
  ^bb7:  // pred: ^bb2
    %20 = llvm.landingpad cleanup : !llvm.struct<(ptr, i32)>
    llvm.intr.lifetime.end 1, %13 : !llvm.ptr
    llvm.br ^bb8(%20 : !llvm.struct<(ptr, i32)>)
  ^bb8(%21: !llvm.struct<(ptr, i32)>):  // 2 preds: ^bb6, ^bb7
    llvm.call @_ZN6google8protobuf8internal10LogMessageD1Ev(%10) : (!llvm.ptr) -> ()
    llvm.intr.lifetime.end 56, %14 : !llvm.ptr
    llvm.resume %21 : !llvm.struct<(ptr, i32)>
  ^bb9:  // pred: ^bb5
    %22 = llvm.getelementptr inbounds %arg1[%1, 3] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.google::protobuf::internal::RepeatedPtrFieldBase", (ptr, i32, i32, ptr)>
    %23 = llvm.load %22 {alignment = 8 : i64, tbaa = [#tbaa_tag15]} : !llvm.ptr -> !llvm.ptr
    %24 = llvm.call @_ZN6google8protobuf8internal20RepeatedPtrFieldBase14InternalExtendEi(%arg0, %17) : (!llvm.ptr, i32) -> !llvm.ptr
    %25 = llvm.getelementptr inbounds %arg0[%1, 3] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.google::protobuf::internal::RepeatedPtrFieldBase", (ptr, i32, i32, ptr)>
    %26 = llvm.load %25 {alignment = 8 : i64, tbaa = [#tbaa_tag15]} : !llvm.ptr -> !llvm.ptr
    %27 = llvm.getelementptr inbounds %26[%1, 0] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.google::protobuf::internal::RepeatedPtrFieldBase::Rep", (i32, array<1 x ptr>)>
    %28 = llvm.load %27 {alignment = 8 : i64, tbaa = [#tbaa_tag18]} : !llvm.ptr -> i32
    %29 = llvm.getelementptr inbounds %arg0[%1, 1] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.google::protobuf::internal::RepeatedPtrFieldBase", (ptr, i32, i32, ptr)>
    %30 = llvm.load %29 {alignment = 8 : i64, tbaa = [#tbaa_tag17]} : !llvm.ptr -> i32
    %31 = llvm.sub %28, %30  : i32
    %32 = llvm.icmp "sgt" %31, %2 : i32
    %33 = llvm.icmp "sgt" %17, %2 : i32
    %34 = llvm.and %32, %33  : i1
    llvm.cond_br %34, ^bb10, ^bb11
  ^bb10:  // pred: ^bb9
    %35 = llvm.zext %17 : i32 to i64
    %36 = llvm.zext %31 : i32 to i64
    llvm.br ^bb13(%1 : i64)
  ^bb11:  // 2 preds: ^bb9, ^bb13
    %37 = llvm.getelementptr inbounds %arg0[%1, 0] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.google::protobuf::internal::RepeatedPtrFieldBase", (ptr, i32, i32, ptr)>
    %38 = llvm.load %37 {alignment = 8 : i64, tbaa = [#tbaa_tag7]} : !llvm.ptr -> !llvm.ptr
    %39 = llvm.icmp "slt" %31, %17 : i32
    llvm.cond_br %39, ^bb12, ^bb15
  ^bb12:  // pred: ^bb11
    %40 = llvm.sext %31 : i32 to i64
    llvm.br ^bb14(%40 : i64)
  ^bb13(%41: i64):  // 2 preds: ^bb10, ^bb13
    %42 = llvm.getelementptr inbounds %23[%1, 1, %41] : (!llvm.ptr, i64, i64) -> !llvm.ptr, !llvm.struct<"struct.google::protobuf::internal::RepeatedPtrFieldBase::Rep", (i32, array<1 x ptr>)>
    %43 = llvm.bitcast %42 : !llvm.ptr to !llvm.ptr
    %44 = llvm.load %43 {alignment = 8 : i64, tbaa = [#tbaa_tag2]} : !llvm.ptr -> !llvm.ptr
    %45 = llvm.getelementptr inbounds %24[%41] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.ptr
    %46 = llvm.bitcast %45 : !llvm.ptr to !llvm.ptr
    %47 = llvm.load %46 {alignment = 8 : i64, tbaa = [#tbaa_tag2]} : !llvm.ptr -> !llvm.ptr
    llvm.call @_ZN6google8protobuf8internal18GenericTypeHandlerIN8tutorial18Person_PhoneNumberEE5MergeERKS4_PS4_(%44, %47) : (!llvm.ptr, !llvm.ptr) -> ()
    %48 = llvm.add %41, %9 overflow<nsw, nuw>  : i64
    %49 = llvm.icmp "ult" %48, %36 : i64
    %50 = llvm.icmp "ult" %48, %35 : i64
    %51 = llvm.and %49, %50  : i1
    llvm.cond_br %51, ^bb13(%48 : i64), ^bb11 {loop_annotation = #loop_annotation}
  ^bb14(%52: i64):  // 2 preds: ^bb12, ^bb14
    %53 = llvm.getelementptr inbounds %23[%1, 1, %52] : (!llvm.ptr, i64, i64) -> !llvm.ptr, !llvm.struct<"struct.google::protobuf::internal::RepeatedPtrFieldBase::Rep", (i32, array<1 x ptr>)>
    %54 = llvm.bitcast %53 : !llvm.ptr to !llvm.ptr
    %55 = llvm.load %54 {alignment = 8 : i64, tbaa = [#tbaa_tag2]} : !llvm.ptr -> !llvm.ptr
    %56 = llvm.call @_ZN6google8protobuf5Arena18CreateMaybeMessageIN8tutorial18Person_PhoneNumberEJEEEPT_PS1_DpOT0_(%38) : (!llvm.ptr) -> !llvm.ptr
    llvm.call @_ZN6google8protobuf8internal18GenericTypeHandlerIN8tutorial18Person_PhoneNumberEE5MergeERKS4_PS4_(%55, %56) : (!llvm.ptr, !llvm.ptr) -> ()
    %57 = llvm.getelementptr inbounds %24[%52] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.ptr
    %58 = llvm.bitcast %57 : !llvm.ptr to !llvm.ptr
    llvm.store %56, %58 {alignment = 8 : i64, tbaa = [#tbaa_tag2]} : !llvm.ptr, !llvm.ptr
    %59 = llvm.add %52, %9 overflow<nsw>  : i64
    %60 = llvm.trunc %59 : i64 to i32
    %61 = llvm.icmp "eq" %17, %60 : i32
    llvm.cond_br %61, ^bb15, ^bb14(%59 : i64) {loop_annotation = #loop_annotation}
  ^bb15:  // 2 preds: ^bb11, ^bb14
    %62 = llvm.load %29 {alignment = 8 : i64, tbaa = [#tbaa_tag17]} : !llvm.ptr -> i32
    %63 = llvm.add %62, %17 overflow<nsw>  : i32
    llvm.store %63, %29 {alignment = 8 : i64, tbaa = [#tbaa_tag17]} : i32, !llvm.ptr
    %64 = llvm.load %25 {alignment = 8 : i64, tbaa = [#tbaa_tag15]} : !llvm.ptr -> !llvm.ptr
    %65 = llvm.getelementptr inbounds %64[%1, 0] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.google::protobuf::internal::RepeatedPtrFieldBase::Rep", (i32, array<1 x ptr>)>
    %66 = llvm.load %65 {alignment = 8 : i64, tbaa = [#tbaa_tag18]} : !llvm.ptr -> i32
    %67 = llvm.icmp "slt" %66, %63 : i32
    llvm.cond_br %67, ^bb16, ^bb17
  ^bb16:  // pred: ^bb15
    llvm.store %63, %65 {alignment = 8 : i64, tbaa = [#tbaa_tag18]} : i32, !llvm.ptr
    llvm.br ^bb17
  ^bb17:  // 3 preds: ^bb5, ^bb15, ^bb16
    llvm.return
  }
  llvm.func local_unnamed_addr @_ZN6google8protobuf8internal20RepeatedPtrFieldBase14InternalExtendEi(!llvm.ptr {llvm.align = 8 : i64, llvm.dereferenceable = 24 : i64, llvm.nonnull, llvm.noundef}, i32 {llvm.noundef}) -> (!llvm.ptr {llvm.noundef}) attributes {frame_pointer = #llvm.framePointerKind<none>, passthrough = [["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], ["tune-cpu", "generic"]], target_cpu = "x86-64", target_features = #llvm.target_features<["+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>}
  llvm.func linkonce_odr local_unnamed_addr @_ZN6google8protobuf8internal18GenericTypeHandlerIN8tutorial18Person_PhoneNumberEE5MergeERKS4_PS4_(%arg0: !llvm.ptr {llvm.align = 8 : i64, llvm.dereferenceable = 32 : i64, llvm.nonnull, llvm.noundef}, %arg1: !llvm.ptr {llvm.noundef}) comdat(@__llvm_global_comdat::@_ZN6google8protobuf8internal18GenericTypeHandlerIN8tutorial18Person_PhoneNumberEE5MergeERKS4_PS4_) attributes {alignment = 2 : i64, frame_pointer = #llvm.framePointerKind<none>, passthrough = ["mustprogress", "noinline", ["uwtable", "2"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], ["tune-cpu", "generic"]], target_cpu = "x86-64", target_features = #llvm.target_features<["+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>} {
    llvm.call @_ZN8tutorial18Person_PhoneNumber9MergeFromERKS0_(%arg1, %arg0) : (!llvm.ptr, !llvm.ptr) -> ()
    llvm.return
  }
  llvm.func linkonce_odr local_unnamed_addr @_ZN6google8protobuf8internal20RepeatedPtrFieldBase12InternalSwapEPS2_(%arg0: !llvm.ptr {llvm.align = 8 : i64, llvm.dereferenceable = 24 : i64, llvm.nonnull, llvm.noundef}, %arg1: !llvm.ptr {llvm.noundef}) comdat(@__llvm_global_comdat::@_ZN6google8protobuf8internal20RepeatedPtrFieldBase12InternalSwapEPS2_) attributes {alignment = 2 : i64, frame_pointer = #llvm.framePointerKind<none>, passthrough = ["inlinehint", "mustprogress", ["uwtable", "2"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], ["tune-cpu", "generic"]], personality = @__gxx_personality_v0, target_cpu = "x86-64", target_features = #llvm.target_features<["+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>} {
    %0 = llvm.mlir.constant(1 : i32) : i32
    %1 = llvm.mlir.constant(0 : i64) : i64
    %2 = llvm.mlir.constant(0 : i32) : i32
    %3 = llvm.mlir.constant(3 : i32) : i32
    %4 = llvm.mlir.constant("/usr/include/google/protobuf/repeated_field.h\00") : !llvm.array<46 x i8>
    %5 = llvm.mlir.addressof @".str.16" : !llvm.ptr
    %6 = llvm.mlir.constant(2577 : i32) : i32
    %7 = llvm.mlir.constant("CHECK failed: this != other: \00") : !llvm.array<30 x i8>
    %8 = llvm.mlir.addressof @".str.22" : !llvm.ptr
    %9 = llvm.mlir.constant(2578 : i32) : i32
    %10 = llvm.mlir.constant("CHECK failed: GetArena() == other->GetArena(): \00") : !llvm.array<48 x i8>
    %11 = llvm.mlir.addressof @".str.23" : !llvm.ptr
    %12 = llvm.mlir.constant(16 : i64) : i64
    %13 = llvm.alloca %0 x !llvm.struct<"class.google::protobuf::internal::LogMessage", (i32, ptr, i32, struct<"class.std::__cxx11::basic_string", (struct<"struct.std::__cxx11::basic_string<char>::_Alloc_hider", (ptr)>, i64, struct<"union.anon", (i64, array<8 x i8>)>)>)> {alignment = 8 : i64} : (i32) -> !llvm.ptr
    %14 = llvm.alloca %0 x !llvm.struct<"class.google::protobuf::internal::LogFinisher", (i8)> {alignment = 1 : i64} : (i32) -> !llvm.ptr
    %15 = llvm.alloca %0 x !llvm.struct<"class.google::protobuf::internal::LogMessage", (i32, ptr, i32, struct<"class.std::__cxx11::basic_string", (struct<"struct.std::__cxx11::basic_string<char>::_Alloc_hider", (ptr)>, i64, struct<"union.anon", (i64, array<8 x i8>)>)>)> {alignment = 8 : i64} : (i32) -> !llvm.ptr
    %16 = llvm.alloca %0 x !llvm.struct<"class.google::protobuf::internal::LogFinisher", (i8)> {alignment = 1 : i64} : (i32) -> !llvm.ptr
    %17 = llvm.icmp "eq" %arg0, %arg1 : !llvm.ptr
    %18 = llvm.getelementptr inbounds %14[%1, 0] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.google::protobuf::internal::LogFinisher", (i8)>
    llvm.intr.lifetime.start 1, %18 : !llvm.ptr
    llvm.cond_br %17, ^bb1, ^bb3
  ^bb1:  // pred: ^bb0
    %19 = llvm.bitcast %13 : !llvm.ptr to !llvm.ptr
    llvm.intr.lifetime.start 56, %19 : !llvm.ptr
    llvm.call @_ZN6google8protobuf8internal10LogMessageC1ENS0_8LogLevelEPKci(%13, %3, %5, %6) : (!llvm.ptr, i32, !llvm.ptr, i32) -> ()
    %20 = llvm.invoke @_ZN6google8protobuf8internal10LogMessagelsEPKc(%13, %8) to ^bb2 unwind ^bb11 : (!llvm.ptr, !llvm.ptr) -> !llvm.ptr
  ^bb2:  // pred: ^bb1
    llvm.invoke @_ZN6google8protobuf8internal11LogFinisheraSERNS1_10LogMessageE(%14, %20) to ^bb4 unwind ^bb12 : (!llvm.ptr, !llvm.ptr) -> ()
  ^bb3:  // pred: ^bb0
    llvm.intr.lifetime.end 1, %18 : !llvm.ptr
    llvm.br ^bb5
  ^bb4:  // pred: ^bb2
    llvm.intr.lifetime.end 1, %18 : !llvm.ptr
    llvm.call @_ZN6google8protobuf8internal10LogMessageD1Ev(%13) : (!llvm.ptr) -> ()
    llvm.intr.lifetime.end 56, %19 : !llvm.ptr
    llvm.br ^bb5
  ^bb5:  // 2 preds: ^bb3, ^bb4
    %21 = llvm.getelementptr inbounds %arg0[%1, 0] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.google::protobuf::internal::RepeatedPtrFieldBase", (ptr, i32, i32, ptr)>
    %22 = llvm.load %21 {alignment = 8 : i64, tbaa = [#tbaa_tag7]} : !llvm.ptr -> !llvm.ptr
    %23 = llvm.getelementptr inbounds %arg1[%1, 0] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.google::protobuf::internal::RepeatedPtrFieldBase", (ptr, i32, i32, ptr)>
    %24 = llvm.load %23 {alignment = 8 : i64, tbaa = [#tbaa_tag7]} : !llvm.ptr -> !llvm.ptr
    %25 = llvm.icmp "eq" %22, %24 : !llvm.ptr
    %26 = llvm.getelementptr inbounds %16[%1, 0] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.google::protobuf::internal::LogFinisher", (i8)>
    llvm.intr.lifetime.start 1, %26 : !llvm.ptr
    llvm.cond_br %25, ^bb8, ^bb6
  ^bb6:  // pred: ^bb5
    %27 = llvm.bitcast %15 : !llvm.ptr to !llvm.ptr
    llvm.intr.lifetime.start 56, %27 : !llvm.ptr
    llvm.call @_ZN6google8protobuf8internal10LogMessageC1ENS0_8LogLevelEPKci(%15, %3, %5, %9) : (!llvm.ptr, i32, !llvm.ptr, i32) -> ()
    %28 = llvm.invoke @_ZN6google8protobuf8internal10LogMessagelsEPKc(%15, %11) to ^bb7 unwind ^bb14 : (!llvm.ptr, !llvm.ptr) -> !llvm.ptr
  ^bb7:  // pred: ^bb6
    llvm.invoke @_ZN6google8protobuf8internal11LogFinisheraSERNS1_10LogMessageE(%16, %28) to ^bb9 unwind ^bb15 : (!llvm.ptr, !llvm.ptr) -> ()
  ^bb8:  // pred: ^bb5
    llvm.intr.lifetime.end 1, %26 : !llvm.ptr
    llvm.br ^bb10
  ^bb9:  // pred: ^bb7
    llvm.intr.lifetime.end 1, %26 : !llvm.ptr
    llvm.call @_ZN6google8protobuf8internal10LogMessageD1Ev(%15) : (!llvm.ptr) -> ()
    llvm.intr.lifetime.end 56, %27 : !llvm.ptr
    llvm.br ^bb10
  ^bb10:  // 2 preds: ^bb8, ^bb9
    %29 = llvm.getelementptr inbounds %arg0[%1, 1] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.google::protobuf::internal::RepeatedPtrFieldBase", (ptr, i32, i32, ptr)>
    %30 = llvm.bitcast %29 : !llvm.ptr to !llvm.ptr
    %31 = llvm.getelementptr inbounds %arg1[%1, 1] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.google::protobuf::internal::RepeatedPtrFieldBase", (ptr, i32, i32, ptr)>
    %32 = llvm.bitcast %31 : !llvm.ptr to !llvm.ptr
    %33 = llvm.bitcast %29 : !llvm.ptr to !llvm.ptr
    %34 = llvm.load %33 {alignment = 8 : i64} : !llvm.ptr -> i128
    "llvm.intr.memcpy"(%30, %32, %12) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
    %35 = llvm.bitcast %31 : !llvm.ptr to !llvm.ptr
    llvm.store %34, %35 {alignment = 1 : i64} : i128, !llvm.ptr
    llvm.return
  ^bb11:  // pred: ^bb1
    %36 = llvm.landingpad cleanup : !llvm.struct<(ptr, i32)>
    llvm.br ^bb13(%36 : !llvm.struct<(ptr, i32)>)
  ^bb12:  // pred: ^bb2
    %37 = llvm.landingpad cleanup : !llvm.struct<(ptr, i32)>
    llvm.intr.lifetime.end 1, %18 : !llvm.ptr
    llvm.br ^bb13(%37 : !llvm.struct<(ptr, i32)>)
  ^bb13(%38: !llvm.struct<(ptr, i32)>):  // 2 preds: ^bb11, ^bb12
    llvm.call @_ZN6google8protobuf8internal10LogMessageD1Ev(%13) : (!llvm.ptr) -> ()
    llvm.intr.lifetime.end 56, %19 : !llvm.ptr
    llvm.br ^bb17(%38 : !llvm.struct<(ptr, i32)>)
  ^bb14:  // pred: ^bb6
    %39 = llvm.landingpad cleanup : !llvm.struct<(ptr, i32)>
    llvm.br ^bb16(%39 : !llvm.struct<(ptr, i32)>)
  ^bb15:  // pred: ^bb7
    %40 = llvm.landingpad cleanup : !llvm.struct<(ptr, i32)>
    llvm.intr.lifetime.end 1, %26 : !llvm.ptr
    llvm.br ^bb16(%40 : !llvm.struct<(ptr, i32)>)
  ^bb16(%41: !llvm.struct<(ptr, i32)>):  // 2 preds: ^bb14, ^bb15
    llvm.call @_ZN6google8protobuf8internal10LogMessageD1Ev(%15) : (!llvm.ptr) -> ()
    llvm.intr.lifetime.end 56, %27 : !llvm.ptr
    llvm.br ^bb17(%41 : !llvm.struct<(ptr, i32)>)
  ^bb17(%42: !llvm.struct<(ptr, i32)>):  // 2 preds: ^bb13, ^bb16
    llvm.resume %42 : !llvm.struct<(ptr, i32)>
  }
  llvm.func linkonce_odr local_unnamed_addr @_ZN6google8protobuf8internal20RepeatedPtrFieldBase9MergeFromINS0_16RepeatedPtrFieldIN8tutorial6PersonEE11TypeHandlerEEEvRKS2_(%arg0: !llvm.ptr {llvm.align = 8 : i64, llvm.dereferenceable = 24 : i64, llvm.nonnull, llvm.noundef}, %arg1: !llvm.ptr {llvm.align = 8 : i64, llvm.dereferenceable = 24 : i64, llvm.nonnull, llvm.noundef}) comdat(@__llvm_global_comdat::@_ZN6google8protobuf8internal20RepeatedPtrFieldBase9MergeFromINS0_16RepeatedPtrFieldIN8tutorial6PersonEE11TypeHandlerEEEvRKS2_) attributes {alignment = 2 : i64, frame_pointer = #llvm.framePointerKind<none>, passthrough = ["inlinehint", "mustprogress", ["uwtable", "2"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], ["tune-cpu", "generic"]], personality = @__gxx_personality_v0, target_cpu = "x86-64", target_features = #llvm.target_features<["+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>} {
    %0 = llvm.mlir.constant(1 : i32) : i32
    %1 = llvm.mlir.constant(0 : i64) : i64
    %2 = llvm.mlir.constant(0 : i32) : i32
    %3 = llvm.mlir.constant(3 : i32) : i32
    %4 = llvm.mlir.constant("/usr/include/google/protobuf/repeated_field.h\00") : !llvm.array<46 x i8>
    %5 = llvm.mlir.addressof @".str.16" : !llvm.ptr
    %6 = llvm.mlir.constant(1787 : i32) : i32
    %7 = llvm.mlir.constant("CHECK failed: (&other) != (this): \00") : !llvm.array<35 x i8>
    %8 = llvm.mlir.addressof @".str.21" : !llvm.ptr
    %9 = llvm.mlir.constant(-1 : i64) : i64
    %10 = llvm.mlir.constant(1 : i64) : i64
    %11 = llvm.alloca %0 x !llvm.struct<"class.google::protobuf::internal::LogMessage", (i32, ptr, i32, struct<"class.std::__cxx11::basic_string", (struct<"struct.std::__cxx11::basic_string<char>::_Alloc_hider", (ptr)>, i64, struct<"union.anon", (i64, array<8 x i8>)>)>)> {alignment = 8 : i64} : (i32) -> !llvm.ptr
    %12 = llvm.alloca %0 x !llvm.struct<"class.google::protobuf::internal::LogFinisher", (i8)> {alignment = 1 : i64} : (i32) -> !llvm.ptr
    %13 = llvm.icmp "eq" %arg1, %arg0 : !llvm.ptr
    %14 = llvm.getelementptr inbounds %12[%1, 0] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.google::protobuf::internal::LogFinisher", (i8)>
    llvm.intr.lifetime.start 1, %14 : !llvm.ptr
    llvm.cond_br %13, ^bb1, ^bb3
  ^bb1:  // pred: ^bb0
    %15 = llvm.bitcast %11 : !llvm.ptr to !llvm.ptr
    llvm.intr.lifetime.start 56, %15 : !llvm.ptr
    llvm.call @_ZN6google8protobuf8internal10LogMessageC1ENS0_8LogLevelEPKci(%11, %3, %5, %6) : (!llvm.ptr, i32, !llvm.ptr, i32) -> ()
    %16 = llvm.invoke @_ZN6google8protobuf8internal10LogMessagelsEPKc(%11, %8) to ^bb2 unwind ^bb6 : (!llvm.ptr, !llvm.ptr) -> !llvm.ptr
  ^bb2:  // pred: ^bb1
    llvm.invoke @_ZN6google8protobuf8internal11LogFinisheraSERNS1_10LogMessageE(%12, %16) to ^bb4 unwind ^bb7 : (!llvm.ptr, !llvm.ptr) -> ()
  ^bb3:  // pred: ^bb0
    llvm.intr.lifetime.end 1, %14 : !llvm.ptr
    llvm.br ^bb5
  ^bb4:  // pred: ^bb2
    llvm.intr.lifetime.end 1, %14 : !llvm.ptr
    llvm.call @_ZN6google8protobuf8internal10LogMessageD1Ev(%11) : (!llvm.ptr) -> ()
    llvm.intr.lifetime.end 56, %15 : !llvm.ptr
    llvm.br ^bb5
  ^bb5:  // 2 preds: ^bb3, ^bb4
    %17 = llvm.getelementptr inbounds %arg1[%1, 1] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.google::protobuf::internal::RepeatedPtrFieldBase", (ptr, i32, i32, ptr)>
    %18 = llvm.load %17 {alignment = 8 : i64, tbaa = [#tbaa_tag17]} : !llvm.ptr -> i32
    %19 = llvm.icmp "eq" %18, %2 : i32
    llvm.cond_br %19, ^bb17, ^bb9
  ^bb6:  // pred: ^bb1
    %20 = llvm.landingpad cleanup : !llvm.struct<(ptr, i32)>
    llvm.br ^bb8(%20 : !llvm.struct<(ptr, i32)>)
  ^bb7:  // pred: ^bb2
    %21 = llvm.landingpad cleanup : !llvm.struct<(ptr, i32)>
    llvm.intr.lifetime.end 1, %14 : !llvm.ptr
    llvm.br ^bb8(%21 : !llvm.struct<(ptr, i32)>)
  ^bb8(%22: !llvm.struct<(ptr, i32)>):  // 2 preds: ^bb6, ^bb7
    llvm.call @_ZN6google8protobuf8internal10LogMessageD1Ev(%11) : (!llvm.ptr) -> ()
    llvm.intr.lifetime.end 56, %15 : !llvm.ptr
    llvm.resume %22 : !llvm.struct<(ptr, i32)>
  ^bb9:  // pred: ^bb5
    %23 = llvm.getelementptr inbounds %arg1[%1, 3] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.google::protobuf::internal::RepeatedPtrFieldBase", (ptr, i32, i32, ptr)>
    %24 = llvm.load %23 {alignment = 8 : i64, tbaa = [#tbaa_tag15]} : !llvm.ptr -> !llvm.ptr
    %25 = llvm.call @_ZN6google8protobuf8internal20RepeatedPtrFieldBase14InternalExtendEi(%arg0, %18) : (!llvm.ptr, i32) -> !llvm.ptr
    %26 = llvm.getelementptr inbounds %arg0[%1, 3] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.google::protobuf::internal::RepeatedPtrFieldBase", (ptr, i32, i32, ptr)>
    %27 = llvm.load %26 {alignment = 8 : i64, tbaa = [#tbaa_tag15]} : !llvm.ptr -> !llvm.ptr
    %28 = llvm.getelementptr inbounds %27[%1, 0] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.google::protobuf::internal::RepeatedPtrFieldBase::Rep", (i32, array<1 x ptr>)>
    %29 = llvm.load %28 {alignment = 8 : i64, tbaa = [#tbaa_tag18]} : !llvm.ptr -> i32
    %30 = llvm.getelementptr inbounds %arg0[%1, 1] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.google::protobuf::internal::RepeatedPtrFieldBase", (ptr, i32, i32, ptr)>
    %31 = llvm.load %30 {alignment = 8 : i64, tbaa = [#tbaa_tag17]} : !llvm.ptr -> i32
    %32 = llvm.sub %29, %31  : i32
    %33 = llvm.icmp "sgt" %32, %2 : i32
    %34 = llvm.icmp "sgt" %18, %2 : i32
    %35 = llvm.and %34, %33  : i1
    llvm.cond_br %35, ^bb10, ^bb11
  ^bb10:  // pred: ^bb9
    %36 = llvm.zext %18 : i32 to i64
    %37 = llvm.zext %32 : i32 to i64
    %38 = llvm.add %36, %9 overflow<nsw>  : i64
    %39 = llvm.add %37, %9 overflow<nsw>  : i64
    %40 = llvm.intr.umin(%38, %39)  : (i64, i64) -> i64
    llvm.br ^bb13(%1 : i64)
  ^bb11:  // 2 preds: ^bb9, ^bb13
    %41 = llvm.getelementptr inbounds %arg0[%1, 0] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.google::protobuf::internal::RepeatedPtrFieldBase", (ptr, i32, i32, ptr)>
    %42 = llvm.load %41 {alignment = 8 : i64, tbaa = [#tbaa_tag7]} : !llvm.ptr -> !llvm.ptr
    %43 = llvm.icmp "slt" %32, %18 : i32
    llvm.cond_br %43, ^bb12, ^bb15
  ^bb12:  // pred: ^bb11
    %44 = llvm.sext %32 : i32 to i64
    llvm.br ^bb14(%44 : i64)
  ^bb13(%45: i64):  // 2 preds: ^bb10, ^bb13
    %46 = llvm.getelementptr inbounds %24[%1, 1, %45] : (!llvm.ptr, i64, i64) -> !llvm.ptr, !llvm.struct<"struct.google::protobuf::internal::RepeatedPtrFieldBase::Rep", (i32, array<1 x ptr>)>
    %47 = llvm.bitcast %46 : !llvm.ptr to !llvm.ptr
    %48 = llvm.load %47 {alignment = 8 : i64, tbaa = [#tbaa_tag2]} : !llvm.ptr -> !llvm.ptr
    %49 = llvm.getelementptr inbounds %25[%45] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.ptr
    %50 = llvm.bitcast %49 : !llvm.ptr to !llvm.ptr
    %51 = llvm.load %50 {alignment = 8 : i64, tbaa = [#tbaa_tag2]} : !llvm.ptr -> !llvm.ptr
    llvm.call @_ZN6google8protobuf8internal18GenericTypeHandlerIN8tutorial6PersonEE5MergeERKS4_PS4_(%48, %51) : (!llvm.ptr, !llvm.ptr) -> ()
    %52 = llvm.add %45, %10 overflow<nsw, nuw>  : i64
    %53 = llvm.icmp "eq" %45, %40 : i64
    llvm.cond_br %53, ^bb11, ^bb13(%52 : i64) {loop_annotation = #loop_annotation}
  ^bb14(%54: i64):  // 2 preds: ^bb12, ^bb14
    %55 = llvm.getelementptr inbounds %24[%1, 1, %54] : (!llvm.ptr, i64, i64) -> !llvm.ptr, !llvm.struct<"struct.google::protobuf::internal::RepeatedPtrFieldBase::Rep", (i32, array<1 x ptr>)>
    %56 = llvm.bitcast %55 : !llvm.ptr to !llvm.ptr
    %57 = llvm.load %56 {alignment = 8 : i64, tbaa = [#tbaa_tag2]} : !llvm.ptr -> !llvm.ptr
    %58 = llvm.call @_ZN6google8protobuf5Arena18CreateMaybeMessageIN8tutorial6PersonEJEEEPT_PS1_DpOT0_(%42) : (!llvm.ptr) -> !llvm.ptr
    llvm.call @_ZN6google8protobuf8internal18GenericTypeHandlerIN8tutorial6PersonEE5MergeERKS4_PS4_(%57, %58) : (!llvm.ptr, !llvm.ptr) -> ()
    %59 = llvm.getelementptr inbounds %25[%54] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.ptr
    %60 = llvm.bitcast %59 : !llvm.ptr to !llvm.ptr
    llvm.store %58, %60 {alignment = 8 : i64, tbaa = [#tbaa_tag2]} : !llvm.ptr, !llvm.ptr
    %61 = llvm.add %54, %10 overflow<nsw>  : i64
    %62 = llvm.trunc %61 : i64 to i32
    %63 = llvm.icmp "eq" %18, %62 : i32
    llvm.cond_br %63, ^bb15, ^bb14(%61 : i64) {loop_annotation = #loop_annotation}
  ^bb15:  // 2 preds: ^bb11, ^bb14
    %64 = llvm.load %30 {alignment = 8 : i64, tbaa = [#tbaa_tag17]} : !llvm.ptr -> i32
    %65 = llvm.add %64, %18 overflow<nsw>  : i32
    llvm.store %65, %30 {alignment = 8 : i64, tbaa = [#tbaa_tag17]} : i32, !llvm.ptr
    %66 = llvm.load %26 {alignment = 8 : i64, tbaa = [#tbaa_tag15]} : !llvm.ptr -> !llvm.ptr
    %67 = llvm.getelementptr inbounds %66[%1, 0] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.google::protobuf::internal::RepeatedPtrFieldBase::Rep", (i32, array<1 x ptr>)>
    %68 = llvm.load %67 {alignment = 8 : i64, tbaa = [#tbaa_tag18]} : !llvm.ptr -> i32
    %69 = llvm.icmp "slt" %68, %65 : i32
    llvm.cond_br %69, ^bb16, ^bb17
  ^bb16:  // pred: ^bb15
    llvm.store %65, %67 {alignment = 8 : i64, tbaa = [#tbaa_tag18]} : i32, !llvm.ptr
    llvm.br ^bb17
  ^bb17:  // 3 preds: ^bb5, ^bb15, ^bb16
    llvm.return
  }
  llvm.func linkonce_odr local_unnamed_addr @_ZN6google8protobuf8internal18GenericTypeHandlerIN8tutorial6PersonEE5MergeERKS4_PS4_(%arg0: !llvm.ptr {llvm.align = 8 : i64, llvm.dereferenceable = 72 : i64, llvm.nonnull, llvm.noundef}, %arg1: !llvm.ptr {llvm.noundef}) comdat(@__llvm_global_comdat::@_ZN6google8protobuf8internal18GenericTypeHandlerIN8tutorial6PersonEE5MergeERKS4_PS4_) attributes {alignment = 2 : i64, frame_pointer = #llvm.framePointerKind<none>, passthrough = ["mustprogress", "noinline", ["uwtable", "2"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], ["tune-cpu", "generic"]], target_cpu = "x86-64", target_features = #llvm.target_features<["+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>} {
    llvm.call @_ZN8tutorial6Person9MergeFromERKS0_(%arg1, %arg0) : (!llvm.ptr, !llvm.ptr) -> ()
    llvm.return
  }
  llvm.func linkonce_odr local_unnamed_addr @_ZN6google8protobuf5Arena14InternalHelperIN8tutorial6PersonEE9ConstructIJPS1_EEEPS4_PvDpOT_(%arg0: !llvm.ptr {llvm.noundef}, %arg1: !llvm.ptr {llvm.align = 8 : i64, llvm.dereferenceable = 8 : i64, llvm.nonnull, llvm.noundef}) -> (!llvm.ptr {llvm.noundef}) comdat(@__llvm_global_comdat::@_ZN6google8protobuf5Arena14InternalHelperIN8tutorial6PersonEE9ConstructIJPS1_EEEPS4_PvDpOT_) attributes {alignment = 2 : i64, frame_pointer = #llvm.framePointerKind<none>, passthrough = [["uwtable", "2"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], ["tune-cpu", "generic"]], personality = @__gxx_personality_v0, target_cpu = "x86-64", target_features = #llvm.target_features<["+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>} {
    %0 = llvm.mlir.constant(8 : i64) : i64
    %1 = llvm.mlir.constant(2 : i64) : i64
    %2 = llvm.mlir.constant(0 : i32) : i32
    %3 = llvm.mlir.constant(0 : i64) : i64
    %4 = llvm.mlir.addressof @_ZNK8tutorial6Person11GetMetadataEv : !llvm.ptr
    %5 = llvm.mlir.zero : !llvm.ptr
    %6 = llvm.mlir.addressof @__gxx_personality_v0 : !llvm.ptr
    %7 = llvm.mlir.addressof @_ZNK8tutorial6Person13SetCachedSizeEi : !llvm.ptr
    %8 = llvm.mlir.addressof @_ZNK6google8protobuf7Message13SpaceUsedLongEv : !llvm.ptr
    %9 = llvm.mlir.addressof @_ZN6google8protobuf7Message20DiscardUnknownFieldsEv : !llvm.ptr
    %10 = llvm.mlir.addressof @_ZN8tutorial6Person9MergeFromERKN6google8protobuf7MessageE : !llvm.ptr
    %11 = llvm.mlir.addressof @_ZN8tutorial6Person8CopyFromERKN6google8protobuf7MessageE : !llvm.ptr
    %12 = llvm.mlir.addressof @_ZNK6google8protobuf11MessageLite16InternalGetTableEv : !llvm.ptr
    %13 = llvm.mlir.addressof @_ZNK8tutorial6Person18_InternalSerializeEPhPN6google8protobuf2io19EpsCopyOutputStreamE : !llvm.ptr
    %14 = llvm.mlir.addressof @_ZN8tutorial6Person14_InternalParseEPKcPN6google8protobuf8internal12ParseContextE : !llvm.ptr
    %15 = llvm.mlir.addressof @_ZNK8tutorial6Person13GetCachedSizeEv : !llvm.ptr
    %16 = llvm.mlir.addressof @_ZNK8tutorial6Person12ByteSizeLongEv : !llvm.ptr
    %17 = llvm.mlir.addressof @_ZN6google8protobuf7Message21CheckTypeAndMergeFromERKNS0_11MessageLiteE : !llvm.ptr
    %18 = llvm.mlir.addressof @_ZNK6google8protobuf7Message25InitializationErrorStringB5cxx11Ev : !llvm.ptr
    %19 = llvm.mlir.addressof @_ZNK8tutorial6Person13IsInitializedEv : !llvm.ptr
    %20 = llvm.mlir.addressof @_ZN8tutorial6Person5ClearEv : !llvm.ptr
    %21 = llvm.mlir.addressof @_ZNK8tutorial6Person3NewEPN6google8protobuf5ArenaE : !llvm.ptr
    %22 = llvm.mlir.addressof @_ZNK8tutorial6Person3NewEv : !llvm.ptr
    %23 = llvm.mlir.addressof @_ZNK6google8protobuf7Message11GetTypeNameB5cxx11Ev : !llvm.ptr
    %24 = llvm.mlir.addressof @_ZN8tutorial6PersonD0Ev : !llvm.ptr
    %25 = llvm.mlir.addressof @_ZN8tutorial6PersonD2Ev : !llvm.ptr
    %26 = llvm.mlir.addressof @_ZTIN6google8protobuf7MessageE : !llvm.ptr
    %27 = llvm.mlir.constant("N8tutorial6PersonE\00") : !llvm.array<19 x i8>
    %28 = llvm.mlir.addressof @_ZTSN8tutorial6PersonE : !llvm.ptr
    %29 = llvm.mlir.addressof @_ZTVN10__cxxabiv120__si_class_type_infoE : !llvm.ptr
    %30 = llvm.getelementptr inbounds %29[%1] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.ptr
    %31 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, ptr)>
    %32 = llvm.insertvalue %30, %31[0] : !llvm.struct<(ptr, ptr, ptr)> 
    %33 = llvm.insertvalue %28, %32[1] : !llvm.struct<(ptr, ptr, ptr)> 
    %34 = llvm.insertvalue %26, %33[2] : !llvm.struct<(ptr, ptr, ptr)> 
    %35 = llvm.mlir.addressof @_ZTIN8tutorial6PersonE : !llvm.ptr
    %36 = llvm.mlir.undef : !llvm.array<22 x ptr>
    %37 = llvm.insertvalue %5, %36[0] : !llvm.array<22 x ptr> 
    %38 = llvm.insertvalue %35, %37[1] : !llvm.array<22 x ptr> 
    %39 = llvm.insertvalue %25, %38[2] : !llvm.array<22 x ptr> 
    %40 = llvm.insertvalue %24, %39[3] : !llvm.array<22 x ptr> 
    %41 = llvm.insertvalue %23, %40[4] : !llvm.array<22 x ptr> 
    %42 = llvm.insertvalue %22, %41[5] : !llvm.array<22 x ptr> 
    %43 = llvm.insertvalue %21, %42[6] : !llvm.array<22 x ptr> 
    %44 = llvm.insertvalue %20, %43[7] : !llvm.array<22 x ptr> 
    %45 = llvm.insertvalue %19, %44[8] : !llvm.array<22 x ptr> 
    %46 = llvm.insertvalue %18, %45[9] : !llvm.array<22 x ptr> 
    %47 = llvm.insertvalue %17, %46[10] : !llvm.array<22 x ptr> 
    %48 = llvm.insertvalue %16, %47[11] : !llvm.array<22 x ptr> 
    %49 = llvm.insertvalue %15, %48[12] : !llvm.array<22 x ptr> 
    %50 = llvm.insertvalue %14, %49[13] : !llvm.array<22 x ptr> 
    %51 = llvm.insertvalue %13, %50[14] : !llvm.array<22 x ptr> 
    %52 = llvm.insertvalue %12, %51[15] : !llvm.array<22 x ptr> 
    %53 = llvm.insertvalue %11, %52[16] : !llvm.array<22 x ptr> 
    %54 = llvm.insertvalue %10, %53[17] : !llvm.array<22 x ptr> 
    %55 = llvm.insertvalue %9, %54[18] : !llvm.array<22 x ptr> 
    %56 = llvm.insertvalue %8, %55[19] : !llvm.array<22 x ptr> 
    %57 = llvm.insertvalue %7, %56[20] : !llvm.array<22 x ptr> 
    %58 = llvm.insertvalue %4, %57[21] : !llvm.array<22 x ptr> 
    %59 = llvm.mlir.undef : !llvm.struct<(array<22 x ptr>)>
    %60 = llvm.insertvalue %58, %59[0] : !llvm.struct<(array<22 x ptr>)> 
    %61 = llvm.mlir.addressof @_ZTVN8tutorial6PersonE : !llvm.ptr
    %62 = llvm.getelementptr inbounds %61[%3, 0, %1] : (!llvm.ptr, i64, i64) -> !llvm.ptr, !llvm.struct<(array<22 x ptr>)>
    %63 = llvm.mlir.constant(16 : i64) : i64
    %64 = llvm.mlir.constant(24 : i64) : i64
    %65 = llvm.mlir.constant(0 : i8) : i8
    %66 = llvm.mlir.constant(68 : i64) : i64
    %67 = llvm.mlir.addressof @scc_info_Timestamp_google_2fprotobuf_2ftimestamp_2eproto : !llvm.ptr
    %68 = llvm.mlir.undef : !llvm.array<0 x ptr>
    %69 = llvm.mlir.addressof @_ZL59InitDefaultsscc_info_Person_PhoneNumber_addressbook_2eprotov : !llvm.ptr
    %70 = llvm.mlir.constant(-1 : i32) : i32
    %71 = llvm.mlir.undef : !llvm.struct<(i32)>
    %72 = llvm.insertvalue %70, %71[0] : !llvm.struct<(i32)> 
    %73 = llvm.mlir.undef : !llvm.struct<(struct<(i32)>, i32, i32, ptr)>
    %74 = llvm.insertvalue %72, %73[0] : !llvm.struct<(struct<(i32)>, i32, i32, ptr)> 
    %75 = llvm.insertvalue %2, %74[1] : !llvm.struct<(struct<(i32)>, i32, i32, ptr)> 
    %76 = llvm.insertvalue %2, %75[2] : !llvm.struct<(struct<(i32)>, i32, i32, ptr)> 
    %77 = llvm.insertvalue %69, %76[3] : !llvm.struct<(struct<(i32)>, i32, i32, ptr)> 
    %78 = llvm.mlir.undef : !llvm.struct<(struct<(struct<(i32)>, i32, i32, ptr)>, array<0 x ptr>)>
    %79 = llvm.insertvalue %77, %78[0] : !llvm.struct<(struct<(struct<(i32)>, i32, i32, ptr)>, array<0 x ptr>)> 
    %80 = llvm.insertvalue %68, %79[1] : !llvm.struct<(struct<(struct<(i32)>, i32, i32, ptr)>, array<0 x ptr>)> 
    %81 = llvm.mlir.addressof @scc_info_Person_PhoneNumber_addressbook_2eproto : !llvm.ptr
    %82 = llvm.mlir.undef : !llvm.array<2 x ptr>
    %83 = llvm.insertvalue %81, %82[0] : !llvm.array<2 x ptr> 
    %84 = llvm.insertvalue %67, %83[1] : !llvm.array<2 x ptr> 
    %85 = llvm.mlir.addressof @_ZL47InitDefaultsscc_info_Person_addressbook_2eprotov : !llvm.ptr
    %86 = llvm.mlir.constant(2 : i32) : i32
    %87 = llvm.mlir.undef : !llvm.struct<(struct<(i32)>, i32, i32, ptr)>
    %88 = llvm.insertvalue %72, %87[0] : !llvm.struct<(struct<(i32)>, i32, i32, ptr)> 
    %89 = llvm.insertvalue %86, %88[1] : !llvm.struct<(struct<(i32)>, i32, i32, ptr)> 
    %90 = llvm.insertvalue %2, %89[2] : !llvm.struct<(struct<(i32)>, i32, i32, ptr)> 
    %91 = llvm.insertvalue %85, %90[3] : !llvm.struct<(struct<(i32)>, i32, i32, ptr)> 
    %92 = llvm.mlir.undef : !llvm.struct<(struct<(struct<(i32)>, i32, i32, ptr)>, array<2 x ptr>)>
    %93 = llvm.insertvalue %91, %92[0] : !llvm.struct<(struct<(struct<(i32)>, i32, i32, ptr)>, array<2 x ptr>)> 
    %94 = llvm.insertvalue %84, %93[1] : !llvm.struct<(struct<(struct<(i32)>, i32, i32, ptr)>, array<2 x ptr>)> 
    %95 = llvm.mlir.addressof @scc_info_Person_addressbook_2eproto : !llvm.ptr
    %96 = llvm.mlir.constant(40 : i64) : i64
    %97 = llvm.mlir.addressof @_ZN6google8protobuf8internal26fixed_address_empty_stringB5cxx11E : !llvm.ptr
    %98 = llvm.mlir.constant(48 : i64) : i64
    %99 = llvm.mlir.constant(56 : i64) : i64
    %100 = llvm.mlir.constant(12 : i64) : i64
    %101 = llvm.load %arg1 {alignment = 8 : i64, tbaa = [#tbaa_tag2]} : !llvm.ptr -> !llvm.ptr
    %102 = llvm.getelementptr inbounds %arg0[%0] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %103 = llvm.bitcast %102 : !llvm.ptr to !llvm.ptr
    llvm.store %101, %103 {alignment = 8 : i64, tbaa = [#tbaa_tag5]} : !llvm.ptr, !llvm.ptr
    %104 = llvm.bitcast %arg0 : !llvm.ptr to !llvm.ptr
    llvm.store %62, %104 {alignment = 8 : i64, tbaa = [#tbaa_tag]} : !llvm.ptr, !llvm.ptr
    %105 = llvm.getelementptr inbounds %arg0[%63] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %106 = llvm.bitcast %105 : !llvm.ptr to !llvm.ptr
    llvm.store %101, %106 {alignment = 8 : i64, tbaa = [#tbaa_tag7]} : !llvm.ptr, !llvm.ptr
    %107 = llvm.getelementptr inbounds %arg0[%64] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    "llvm.intr.memset"(%107, %65, %63) <{isVolatile = false}> : (!llvm.ptr, i8, i64) -> ()
    %108 = llvm.getelementptr inbounds %arg0[%66] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %109 = llvm.bitcast %108 : !llvm.ptr to !llvm.ptr
    llvm.store %2, %109 {alignment = 4 : i64, tbaa = [#tbaa_tag6]} : i32, !llvm.ptr
    %110 = llvm.load %95 atomic acquire {alignment = 8 : i64} : !llvm.ptr -> i32
    %111 = llvm.icmp "eq" %110, %2 : i32
    llvm.cond_br %111 weights([2000, 1]), ^bb8, ^bb1
  ^bb1:  // pred: ^bb0
    llvm.invoke @_ZN6google8protobuf8internal11InitSCCImplEPNS1_11SCCInfoBaseE(%95) to ^bb8 unwind ^bb2 : (!llvm.ptr) -> ()
  ^bb2:  // pred: ^bb1
    %112 = llvm.landingpad cleanup : !llvm.struct<(ptr, i32)>
    %113 = llvm.bitcast %105 : !llvm.ptr to !llvm.ptr
    llvm.invoke @_ZN6google8protobuf8internal20RepeatedPtrFieldBase7DestroyINS0_16RepeatedPtrFieldIN8tutorial18Person_PhoneNumberEE11TypeHandlerEEEvv(%113) to ^bb3 unwind ^bb6 : (!llvm.ptr) -> ()
  ^bb3:  // pred: ^bb2
    %114 = llvm.load %106 {alignment = 8 : i64, tbaa = [#tbaa_tag7]} : !llvm.ptr -> !llvm.ptr
    %115 = llvm.icmp "eq" %114, %5 : !llvm.ptr
    llvm.cond_br %115, ^bb7, ^bb4
  ^bb4:  // pred: ^bb3
    %116 = llvm.getelementptr inbounds %114[%3, 0] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.google::protobuf::Arena", (struct<"class.google::protobuf::internal::ArenaImpl", (struct<"struct.std::atomic.10", (struct<"struct.std::__atomic_base.11", (ptr)>)>, struct<"struct.std::atomic.10", (struct<"struct.std::__atomic_base.11", (ptr)>)>, struct<"struct.std::atomic.12", (struct<"struct.std::__atomic_base.13", (i64)>)>, ptr, i64, struct<"struct.google::protobuf::internal::ArenaImpl::Options", (i64, i64, ptr, i64, ptr, ptr)>)>, ptr, ptr, ptr, ptr)>
    %117 = llvm.invoke @_ZNK6google8protobuf8internal9ArenaImpl14SpaceAllocatedEv(%116) to ^bb7 unwind ^bb5 : (!llvm.ptr) -> i64
  ^bb5:  // pred: ^bb4
    %118 = llvm.landingpad (catch %5 : !llvm.ptr) : !llvm.struct<(ptr, i32)>
    %119 = llvm.extractvalue %118[0] : !llvm.struct<(ptr, i32)> 
    llvm.call @__clang_call_terminate(%119) : (!llvm.ptr) -> ()
    llvm.unreachable
  ^bb6:  // pred: ^bb2
    %120 = llvm.landingpad (catch %5 : !llvm.ptr) : !llvm.struct<(ptr, i32)>
    %121 = llvm.extractvalue %120[0] : !llvm.struct<(ptr, i32)> 
    llvm.call @_ZN6google8protobuf8internal20RepeatedPtrFieldBaseD2Ev(%113) : (!llvm.ptr) -> ()
    llvm.call @__clang_call_terminate(%121) : (!llvm.ptr) -> ()
    llvm.unreachable
  ^bb7:  // 2 preds: ^bb3, ^bb4
    llvm.resume %112 : !llvm.struct<(ptr, i32)>
  ^bb8:  // 2 preds: ^bb0, ^bb1
    %122 = llvm.bitcast %arg0 : !llvm.ptr to !llvm.ptr
    %123 = llvm.getelementptr inbounds %arg0[%96] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %124 = llvm.bitcast %123 : !llvm.ptr to !llvm.ptr
    llvm.store %97, %124 {alignment = 8 : i64, tbaa = [#tbaa_tag8]} : !llvm.ptr, !llvm.ptr
    %125 = llvm.getelementptr inbounds %arg0[%98] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %126 = llvm.bitcast %125 : !llvm.ptr to !llvm.ptr
    llvm.store %97, %126 {alignment = 8 : i64, tbaa = [#tbaa_tag8]} : !llvm.ptr, !llvm.ptr
    %127 = llvm.getelementptr inbounds %arg0[%99] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    "llvm.intr.memset"(%127, %65, %100) <{isVolatile = false}> : (!llvm.ptr, i8, i64) -> ()
    llvm.return %122 : !llvm.ptr
  }
  llvm.func internal @_GLOBAL__sub_I_addressbook.pb.cc() attributes {dso_local, frame_pointer = #llvm.framePointerKind<none>, passthrough = [["uwtable", "2"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], ["tune-cpu", "generic"]], section = ".text.startup", target_cpu = "x86-64", target_features = #llvm.target_features<["+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>} {
    %0 = llvm.mlir.constant(0 : i8) : i8
    %1 = llvm.mlir.undef : !llvm.struct<"class.std::ios_base::Init", (i8)>
    %2 = llvm.insertvalue %0, %1[0] : !llvm.struct<"class.std::ios_base::Init", (i8)> 
    %3 = llvm.mlir.addressof @_ZStL8__ioinit : !llvm.ptr
    %4 = llvm.mlir.addressof @_ZNSt8ios_base4InitD1Ev : !llvm.ptr
    %5 = llvm.mlir.addressof @__dso_handle : !llvm.ptr
    %6 = llvm.mlir.zero : !llvm.ptr
    %7 = llvm.mlir.undef : !llvm.array<1 x ptr>
    %8 = llvm.insertvalue %6, %7[0] : !llvm.array<1 x ptr> 
    %9 = llvm.mlir.addressof @_ZL47file_level_enum_descriptors_addressbook_2eproto : !llvm.ptr
    %10 = llvm.mlir.constant(3 : i32) : i32
    %11 = llvm.mlir.undef : !llvm.struct<"struct.google::protobuf::Metadata", (ptr, ptr)>
    %12 = llvm.insertvalue %6, %11[0] : !llvm.struct<"struct.google::protobuf::Metadata", (ptr, ptr)> 
    %13 = llvm.insertvalue %6, %12[1] : !llvm.struct<"struct.google::protobuf::Metadata", (ptr, ptr)> 
    %14 = llvm.mlir.undef : !llvm.array<3 x struct<"struct.google::protobuf::Metadata", (ptr, ptr)>>
    %15 = llvm.insertvalue %13, %14[0] : !llvm.array<3 x struct<"struct.google::protobuf::Metadata", (ptr, ptr)>> 
    %16 = llvm.insertvalue %13, %15[1] : !llvm.array<3 x struct<"struct.google::protobuf::Metadata", (ptr, ptr)>> 
    %17 = llvm.insertvalue %13, %16[2] : !llvm.array<3 x struct<"struct.google::protobuf::Metadata", (ptr, ptr)>> 
    %18 = llvm.mlir.addressof @_ZL39file_level_metadata_addressbook_2eproto : !llvm.ptr
    %19 = llvm.mlir.constant(dense<[-1, 8, -1, -1, -1, 16, 24, -1, 8, -1, -1, -1, 40, 64, 48, 16, 56, -1, 8, -1, -1, -1, 16]> : tensor<23xi32>) : !llvm.array<23 x i32>
    %20 = llvm.mlir.addressof @_ZN31TableStruct_addressbook_2eproto7offsetsE : !llvm.ptr
    %21 = llvm.mlir.constant(dense<0> : tensor<40xi8>) : !llvm.array<40 x i8>
    %22 = llvm.mlir.constant(0 : i64) : i64
    %23 = llvm.mlir.undef : !llvm.struct<"union.google::protobuf::internal::ExplicitlyConstructed<tutorial::AddressBook>::AlignedUnion", (i64, array<40 x i8>)>
    %24 = llvm.insertvalue %22, %23[0] : !llvm.struct<"union.google::protobuf::internal::ExplicitlyConstructed<tutorial::AddressBook>::AlignedUnion", (i64, array<40 x i8>)> 
    %25 = llvm.insertvalue %21, %24[1] : !llvm.struct<"union.google::protobuf::internal::ExplicitlyConstructed<tutorial::AddressBook>::AlignedUnion", (i64, array<40 x i8>)> 
    %26 = llvm.mlir.undef : !llvm.struct<"class.google::protobuf::internal::ExplicitlyConstructed.1", (struct<"union.google::protobuf::internal::ExplicitlyConstructed<tutorial::AddressBook>::AlignedUnion", (i64, array<40 x i8>)>)>
    %27 = llvm.insertvalue %25, %26[0] : !llvm.struct<"class.google::protobuf::internal::ExplicitlyConstructed.1", (struct<"union.google::protobuf::internal::ExplicitlyConstructed<tutorial::AddressBook>::AlignedUnion", (i64, array<40 x i8>)>)> 
    %28 = llvm.mlir.undef : !llvm.struct<"class.tutorial::AddressBookDefaultTypeInternal", (struct<"class.google::protobuf::internal::ExplicitlyConstructed.1", (struct<"union.google::protobuf::internal::ExplicitlyConstructed<tutorial::AddressBook>::AlignedUnion", (i64, array<40 x i8>)>)>)>
    %29 = llvm.insertvalue %27, %28[0] : !llvm.struct<"class.tutorial::AddressBookDefaultTypeInternal", (struct<"class.google::protobuf::internal::ExplicitlyConstructed.1", (struct<"union.google::protobuf::internal::ExplicitlyConstructed<tutorial::AddressBook>::AlignedUnion", (i64, array<40 x i8>)>)>)> 
    %30 = llvm.mlir.addressof @_ZN8tutorial30_AddressBook_default_instance_E : !llvm.ptr
    %31 = llvm.mlir.constant(dense<0> : tensor<64xi8>) : !llvm.array<64 x i8>
    %32 = llvm.mlir.undef : !llvm.struct<"union.google::protobuf::internal::ExplicitlyConstructed<tutorial::Person>::AlignedUnion", (i64, array<64 x i8>)>
    %33 = llvm.insertvalue %22, %32[0] : !llvm.struct<"union.google::protobuf::internal::ExplicitlyConstructed<tutorial::Person>::AlignedUnion", (i64, array<64 x i8>)> 
    %34 = llvm.insertvalue %31, %33[1] : !llvm.struct<"union.google::protobuf::internal::ExplicitlyConstructed<tutorial::Person>::AlignedUnion", (i64, array<64 x i8>)> 
    %35 = llvm.mlir.undef : !llvm.struct<"class.google::protobuf::internal::ExplicitlyConstructed.0", (struct<"union.google::protobuf::internal::ExplicitlyConstructed<tutorial::Person>::AlignedUnion", (i64, array<64 x i8>)>)>
    %36 = llvm.insertvalue %34, %35[0] : !llvm.struct<"class.google::protobuf::internal::ExplicitlyConstructed.0", (struct<"union.google::protobuf::internal::ExplicitlyConstructed<tutorial::Person>::AlignedUnion", (i64, array<64 x i8>)>)> 
    %37 = llvm.mlir.undef : !llvm.struct<"class.tutorial::PersonDefaultTypeInternal", (struct<"class.google::protobuf::internal::ExplicitlyConstructed.0", (struct<"union.google::protobuf::internal::ExplicitlyConstructed<tutorial::Person>::AlignedUnion", (i64, array<64 x i8>)>)>)>
    %38 = llvm.insertvalue %36, %37[0] : !llvm.struct<"class.tutorial::PersonDefaultTypeInternal", (struct<"class.google::protobuf::internal::ExplicitlyConstructed.0", (struct<"union.google::protobuf::internal::ExplicitlyConstructed<tutorial::Person>::AlignedUnion", (i64, array<64 x i8>)>)>)> 
    %39 = llvm.mlir.addressof @_ZN8tutorial25_Person_default_instance_E : !llvm.ptr
    %40 = llvm.mlir.constant(dense<0> : tensor<24xi8>) : !llvm.array<24 x i8>
    %41 = llvm.mlir.undef : !llvm.struct<"union.google::protobuf::internal::ExplicitlyConstructed<tutorial::Person_PhoneNumber>::AlignedUnion", (i64, array<24 x i8>)>
    %42 = llvm.insertvalue %22, %41[0] : !llvm.struct<"union.google::protobuf::internal::ExplicitlyConstructed<tutorial::Person_PhoneNumber>::AlignedUnion", (i64, array<24 x i8>)> 
    %43 = llvm.insertvalue %40, %42[1] : !llvm.struct<"union.google::protobuf::internal::ExplicitlyConstructed<tutorial::Person_PhoneNumber>::AlignedUnion", (i64, array<24 x i8>)> 
    %44 = llvm.mlir.undef : !llvm.struct<"class.google::protobuf::internal::ExplicitlyConstructed", (struct<"union.google::protobuf::internal::ExplicitlyConstructed<tutorial::Person_PhoneNumber>::AlignedUnion", (i64, array<24 x i8>)>)>
    %45 = llvm.insertvalue %43, %44[0] : !llvm.struct<"class.google::protobuf::internal::ExplicitlyConstructed", (struct<"union.google::protobuf::internal::ExplicitlyConstructed<tutorial::Person_PhoneNumber>::AlignedUnion", (i64, array<24 x i8>)>)> 
    %46 = llvm.mlir.undef : !llvm.struct<"class.tutorial::Person_PhoneNumberDefaultTypeInternal", (struct<"class.google::protobuf::internal::ExplicitlyConstructed", (struct<"union.google::protobuf::internal::ExplicitlyConstructed<tutorial::Person_PhoneNumber>::AlignedUnion", (i64, array<24 x i8>)>)>)>
    %47 = llvm.insertvalue %45, %46[0] : !llvm.struct<"class.tutorial::Person_PhoneNumberDefaultTypeInternal", (struct<"class.google::protobuf::internal::ExplicitlyConstructed", (struct<"union.google::protobuf::internal::ExplicitlyConstructed<tutorial::Person_PhoneNumber>::AlignedUnion", (i64, array<24 x i8>)>)>)> 
    %48 = llvm.mlir.addressof @_ZN8tutorial37_Person_PhoneNumber_default_instance_E : !llvm.ptr
    %49 = llvm.mlir.undef : !llvm.array<3 x ptr>
    %50 = llvm.insertvalue %48, %49[0] : !llvm.array<3 x ptr> 
    %51 = llvm.insertvalue %39, %50[1] : !llvm.array<3 x ptr> 
    %52 = llvm.insertvalue %30, %51[2] : !llvm.array<3 x ptr> 
    %53 = llvm.mlir.addressof @_ZL22file_default_instances : !llvm.ptr
    %54 = llvm.mlir.constant(48 : i32) : i32
    %55 = llvm.mlir.constant(-1 : i32) : i32
    %56 = llvm.mlir.constant(17 : i32) : i32
    %57 = llvm.mlir.undef : !llvm.struct<"struct.google::protobuf::internal::MigrationSchema", (i32, i32, i32)>
    %58 = llvm.insertvalue %56, %57[0] : !llvm.struct<"struct.google::protobuf::internal::MigrationSchema", (i32, i32, i32)> 
    %59 = llvm.insertvalue %55, %58[1] : !llvm.struct<"struct.google::protobuf::internal::MigrationSchema", (i32, i32, i32)> 
    %60 = llvm.insertvalue %54, %59[2] : !llvm.struct<"struct.google::protobuf::internal::MigrationSchema", (i32, i32, i32)> 
    %61 = llvm.mlir.constant(72 : i32) : i32
    %62 = llvm.mlir.constant(7 : i32) : i32
    %63 = llvm.mlir.undef : !llvm.struct<"struct.google::protobuf::internal::MigrationSchema", (i32, i32, i32)>
    %64 = llvm.insertvalue %62, %63[0] : !llvm.struct<"struct.google::protobuf::internal::MigrationSchema", (i32, i32, i32)> 
    %65 = llvm.insertvalue %55, %64[1] : !llvm.struct<"struct.google::protobuf::internal::MigrationSchema", (i32, i32, i32)> 
    %66 = llvm.insertvalue %61, %65[2] : !llvm.struct<"struct.google::protobuf::internal::MigrationSchema", (i32, i32, i32)> 
    %67 = llvm.mlir.constant(32 : i32) : i32
    %68 = llvm.mlir.constant(0 : i32) : i32
    %69 = llvm.mlir.undef : !llvm.struct<"struct.google::protobuf::internal::MigrationSchema", (i32, i32, i32)>
    %70 = llvm.insertvalue %68, %69[0] : !llvm.struct<"struct.google::protobuf::internal::MigrationSchema", (i32, i32, i32)> 
    %71 = llvm.insertvalue %55, %70[1] : !llvm.struct<"struct.google::protobuf::internal::MigrationSchema", (i32, i32, i32)> 
    %72 = llvm.insertvalue %67, %71[2] : !llvm.struct<"struct.google::protobuf::internal::MigrationSchema", (i32, i32, i32)> 
    %73 = llvm.mlir.undef : !llvm.array<3 x struct<"struct.google::protobuf::internal::MigrationSchema", (i32, i32, i32)>>
    %74 = llvm.insertvalue %72, %73[0] : !llvm.array<3 x struct<"struct.google::protobuf::internal::MigrationSchema", (i32, i32, i32)>> 
    %75 = llvm.insertvalue %66, %74[1] : !llvm.array<3 x struct<"struct.google::protobuf::internal::MigrationSchema", (i32, i32, i32)>> 
    %76 = llvm.insertvalue %60, %75[2] : !llvm.array<3 x struct<"struct.google::protobuf::internal::MigrationSchema", (i32, i32, i32)>> 
    %77 = llvm.mlir.addressof @_ZL7schemas : !llvm.ptr
    %78 = llvm.mlir.constant(1 : i32) : i32
    %79 = llvm.mlir.addressof @descriptor_table_google_2fprotobuf_2ftimestamp_2eproto : !llvm.ptr
    %80 = llvm.mlir.undef : !llvm.array<1 x ptr>
    %81 = llvm.insertvalue %79, %80[0] : !llvm.array<1 x ptr> 
    %82 = llvm.mlir.addressof @_ZL41descriptor_table_addressbook_2eproto_deps : !llvm.ptr
    %83 = llvm.mlir.undef : !llvm.array<0 x ptr>
    %84 = llvm.mlir.addressof @__gxx_personality_v0 : !llvm.ptr
    %85 = llvm.mlir.addressof @_ZL59InitDefaultsscc_info_Person_PhoneNumber_addressbook_2eprotov : !llvm.ptr
    %86 = llvm.mlir.undef : !llvm.struct<(i32)>
    %87 = llvm.insertvalue %55, %86[0] : !llvm.struct<(i32)> 
    %88 = llvm.mlir.undef : !llvm.struct<(struct<(i32)>, i32, i32, ptr)>
    %89 = llvm.insertvalue %87, %88[0] : !llvm.struct<(struct<(i32)>, i32, i32, ptr)> 
    %90 = llvm.insertvalue %68, %89[1] : !llvm.struct<(struct<(i32)>, i32, i32, ptr)> 
    %91 = llvm.insertvalue %68, %90[2] : !llvm.struct<(struct<(i32)>, i32, i32, ptr)> 
    %92 = llvm.insertvalue %85, %91[3] : !llvm.struct<(struct<(i32)>, i32, i32, ptr)> 
    %93 = llvm.mlir.undef : !llvm.struct<(struct<(struct<(i32)>, i32, i32, ptr)>, array<0 x ptr>)>
    %94 = llvm.insertvalue %92, %93[0] : !llvm.struct<(struct<(struct<(i32)>, i32, i32, ptr)>, array<0 x ptr>)> 
    %95 = llvm.insertvalue %83, %94[1] : !llvm.struct<(struct<(struct<(i32)>, i32, i32, ptr)>, array<0 x ptr>)> 
    %96 = llvm.mlir.addressof @scc_info_Person_PhoneNumber_addressbook_2eproto : !llvm.ptr
    %97 = llvm.mlir.addressof @scc_info_Timestamp_google_2fprotobuf_2ftimestamp_2eproto : !llvm.ptr
    %98 = llvm.mlir.undef : !llvm.array<2 x ptr>
    %99 = llvm.insertvalue %96, %98[0] : !llvm.array<2 x ptr> 
    %100 = llvm.insertvalue %97, %99[1] : !llvm.array<2 x ptr> 
    %101 = llvm.mlir.addressof @_ZL47InitDefaultsscc_info_Person_addressbook_2eprotov : !llvm.ptr
    %102 = llvm.mlir.constant(2 : i32) : i32
    %103 = llvm.mlir.undef : !llvm.struct<(struct<(i32)>, i32, i32, ptr)>
    %104 = llvm.insertvalue %87, %103[0] : !llvm.struct<(struct<(i32)>, i32, i32, ptr)> 
    %105 = llvm.insertvalue %102, %104[1] : !llvm.struct<(struct<(i32)>, i32, i32, ptr)> 
    %106 = llvm.insertvalue %68, %105[2] : !llvm.struct<(struct<(i32)>, i32, i32, ptr)> 
    %107 = llvm.insertvalue %101, %106[3] : !llvm.struct<(struct<(i32)>, i32, i32, ptr)> 
    %108 = llvm.mlir.undef : !llvm.struct<(struct<(struct<(i32)>, i32, i32, ptr)>, array<2 x ptr>)>
    %109 = llvm.insertvalue %107, %108[0] : !llvm.struct<(struct<(struct<(i32)>, i32, i32, ptr)>, array<2 x ptr>)> 
    %110 = llvm.insertvalue %100, %109[1] : !llvm.struct<(struct<(struct<(i32)>, i32, i32, ptr)>, array<2 x ptr>)> 
    %111 = llvm.mlir.addressof @scc_info_Person_addressbook_2eproto : !llvm.ptr
    %112 = llvm.mlir.undef : !llvm.array<1 x ptr>
    %113 = llvm.insertvalue %111, %112[0] : !llvm.array<1 x ptr> 
    %114 = llvm.mlir.addressof @_ZL52InitDefaultsscc_info_AddressBook_addressbook_2eprotov : !llvm.ptr
    %115 = llvm.mlir.undef : !llvm.struct<(struct<(i32)>, i32, i32, ptr)>
    %116 = llvm.insertvalue %87, %115[0] : !llvm.struct<(struct<(i32)>, i32, i32, ptr)> 
    %117 = llvm.insertvalue %78, %116[1] : !llvm.struct<(struct<(i32)>, i32, i32, ptr)> 
    %118 = llvm.insertvalue %68, %117[2] : !llvm.struct<(struct<(i32)>, i32, i32, ptr)> 
    %119 = llvm.insertvalue %114, %118[3] : !llvm.struct<(struct<(i32)>, i32, i32, ptr)> 
    %120 = llvm.mlir.undef : !llvm.struct<(struct<(struct<(i32)>, i32, i32, ptr)>, array<1 x ptr>)>
    %121 = llvm.insertvalue %119, %120[0] : !llvm.struct<(struct<(struct<(i32)>, i32, i32, ptr)>, array<1 x ptr>)> 
    %122 = llvm.insertvalue %113, %121[1] : !llvm.struct<(struct<(struct<(i32)>, i32, i32, ptr)>, array<1 x ptr>)> 
    %123 = llvm.mlir.addressof @scc_info_AddressBook_addressbook_2eproto : !llvm.ptr
    %124 = llvm.mlir.undef : !llvm.array<3 x ptr>
    %125 = llvm.insertvalue %123, %124[0] : !llvm.array<3 x ptr> 
    %126 = llvm.insertvalue %111, %125[1] : !llvm.array<3 x ptr> 
    %127 = llvm.insertvalue %96, %126[2] : !llvm.array<3 x ptr> 
    %128 = llvm.mlir.addressof @_ZL41descriptor_table_addressbook_2eproto_sccs : !llvm.ptr
    %129 = llvm.mlir.undef : !llvm.struct<"struct.std::once_flag", (i32)>
    %130 = llvm.insertvalue %68, %129[0] : !llvm.struct<"struct.std::once_flag", (i32)> 
    %131 = llvm.mlir.addressof @_ZL41descriptor_table_addressbook_2eproto_once : !llvm.ptr
    %132 = llvm.mlir.constant(537 : i32) : i32
    %133 = llvm.mlir.constant("addressbook.proto\00") : !llvm.array<18 x i8>
    %134 = llvm.mlir.addressof @".str" : !llvm.ptr
    %135 = llvm.mlir.constant("\0A\11addressbook.proto\12\08tutorial\1A\1Fgoogle/protobuf/timestamp.proto\22\87\02\0A\06Person\12\0C\0A\04name\18\01 \01(\09\12\0A\0A\02id\18\02 \01(\05\12\0D\0A\05email\18\03 \01(\09\12,\0A\06phones\18\04 \03(\0B2\1C.tutorial.Person.PhoneNumber\120\0A\0Clast_updated\18\05 \01(\0B2\1A.google.protobuf.Timestamp\1AG\0A\0BPhoneNumber\12\0E\0A\06number\18\01 \01(\09\12(\0A\04type\18\02 \01(\0E2\1A.tutorial.Person.PhoneType\22+\0A\09PhoneType\12\0A\0A\06MOBILE\10\00\12\08\0A\04HOME\10\01\12\08\0A\04WORK\10\02\22/\0A\0BAddressBook\12 \0A\06people\18\01 \03(\0B2\10.tutorial.PersonB\95\01\0A\1Bcom.example.tutorial.protosB\11AddressBookProtosP\01Z:github.com/protocolbuffers/protobuf/examples/go/tutorialpb\AA\02$Google.Protobuf.Examples.AddressBookb\06proto3\00") : !llvm.array<538 x i8>
    %136 = llvm.mlir.addressof @_ZL45descriptor_table_protodef_addressbook_2eproto : !llvm.ptr
    %137 = llvm.mlir.undef : !llvm.struct<"struct.google::protobuf::internal::DescriptorTable", (i8, i8, ptr, ptr, i32, ptr, ptr, ptr, i32, i32, ptr, ptr, ptr, ptr, i32, ptr, ptr)>
    %138 = llvm.insertvalue %0, %137[0] : !llvm.struct<"struct.google::protobuf::internal::DescriptorTable", (i8, i8, ptr, ptr, i32, ptr, ptr, ptr, i32, i32, ptr, ptr, ptr, ptr, i32, ptr, ptr)> 
    %139 = llvm.insertvalue %0, %138[1] : !llvm.struct<"struct.google::protobuf::internal::DescriptorTable", (i8, i8, ptr, ptr, i32, ptr, ptr, ptr, i32, i32, ptr, ptr, ptr, ptr, i32, ptr, ptr)> 
    %140 = llvm.insertvalue %136, %139[2] : !llvm.struct<"struct.google::protobuf::internal::DescriptorTable", (i8, i8, ptr, ptr, i32, ptr, ptr, ptr, i32, i32, ptr, ptr, ptr, ptr, i32, ptr, ptr)> 
    %141 = llvm.insertvalue %134, %140[3] : !llvm.struct<"struct.google::protobuf::internal::DescriptorTable", (i8, i8, ptr, ptr, i32, ptr, ptr, ptr, i32, i32, ptr, ptr, ptr, ptr, i32, ptr, ptr)> 
    %142 = llvm.insertvalue %132, %141[4] : !llvm.struct<"struct.google::protobuf::internal::DescriptorTable", (i8, i8, ptr, ptr, i32, ptr, ptr, ptr, i32, i32, ptr, ptr, ptr, ptr, i32, ptr, ptr)> 
    %143 = llvm.insertvalue %131, %142[5] : !llvm.struct<"struct.google::protobuf::internal::DescriptorTable", (i8, i8, ptr, ptr, i32, ptr, ptr, ptr, i32, i32, ptr, ptr, ptr, ptr, i32, ptr, ptr)> 
    %144 = llvm.insertvalue %128, %143[6] : !llvm.struct<"struct.google::protobuf::internal::DescriptorTable", (i8, i8, ptr, ptr, i32, ptr, ptr, ptr, i32, i32, ptr, ptr, ptr, ptr, i32, ptr, ptr)> 
    %145 = llvm.insertvalue %82, %144[7] : !llvm.struct<"struct.google::protobuf::internal::DescriptorTable", (i8, i8, ptr, ptr, i32, ptr, ptr, ptr, i32, i32, ptr, ptr, ptr, ptr, i32, ptr, ptr)> 
    %146 = llvm.insertvalue %10, %145[8] : !llvm.struct<"struct.google::protobuf::internal::DescriptorTable", (i8, i8, ptr, ptr, i32, ptr, ptr, ptr, i32, i32, ptr, ptr, ptr, ptr, i32, ptr, ptr)> 
    %147 = llvm.insertvalue %78, %146[9] : !llvm.struct<"struct.google::protobuf::internal::DescriptorTable", (i8, i8, ptr, ptr, i32, ptr, ptr, ptr, i32, i32, ptr, ptr, ptr, ptr, i32, ptr, ptr)> 
    %148 = llvm.insertvalue %77, %147[10] : !llvm.struct<"struct.google::protobuf::internal::DescriptorTable", (i8, i8, ptr, ptr, i32, ptr, ptr, ptr, i32, i32, ptr, ptr, ptr, ptr, i32, ptr, ptr)> 
    %149 = llvm.insertvalue %53, %148[11] : !llvm.struct<"struct.google::protobuf::internal::DescriptorTable", (i8, i8, ptr, ptr, i32, ptr, ptr, ptr, i32, i32, ptr, ptr, ptr, ptr, i32, ptr, ptr)> 
    %150 = llvm.insertvalue %20, %149[12] : !llvm.struct<"struct.google::protobuf::internal::DescriptorTable", (i8, i8, ptr, ptr, i32, ptr, ptr, ptr, i32, i32, ptr, ptr, ptr, ptr, i32, ptr, ptr)> 
    %151 = llvm.insertvalue %18, %150[13] : !llvm.struct<"struct.google::protobuf::internal::DescriptorTable", (i8, i8, ptr, ptr, i32, ptr, ptr, ptr, i32, i32, ptr, ptr, ptr, ptr, i32, ptr, ptr)> 
    %152 = llvm.insertvalue %10, %151[14] : !llvm.struct<"struct.google::protobuf::internal::DescriptorTable", (i8, i8, ptr, ptr, i32, ptr, ptr, ptr, i32, i32, ptr, ptr, ptr, ptr, i32, ptr, ptr)> 
    %153 = llvm.insertvalue %9, %152[15] : !llvm.struct<"struct.google::protobuf::internal::DescriptorTable", (i8, i8, ptr, ptr, i32, ptr, ptr, ptr, i32, i32, ptr, ptr, ptr, ptr, i32, ptr, ptr)> 
    %154 = llvm.insertvalue %6, %153[16] : !llvm.struct<"struct.google::protobuf::internal::DescriptorTable", (i8, i8, ptr, ptr, i32, ptr, ptr, ptr, i32, i32, ptr, ptr, ptr, ptr, i32, ptr, ptr)> 
    %155 = llvm.mlir.addressof @descriptor_table_addressbook_2eproto : !llvm.ptr
    llvm.call @_ZNSt8ios_base4InitC1Ev(%3) : (!llvm.ptr) -> ()
    %156 = llvm.call @__cxa_atexit(%4, %3, %5) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> i32
    llvm.call @_ZN6google8protobuf8internal14AddDescriptorsEPKNS1_15DescriptorTableE(%155) : (!llvm.ptr) -> ()
    llvm.return
  }
}
