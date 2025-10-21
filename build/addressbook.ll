; ModuleID = 'build/addressbook.pb.cc'
source_filename = "build/addressbook.pb.cc"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-linux-gnu"

%"class.std::ios_base::Init" = type { i8 }
%"class.tutorial::Person_PhoneNumberDefaultTypeInternal" = type { %"class.google::protobuf::internal::ExplicitlyConstructed" }
%"class.google::protobuf::internal::ExplicitlyConstructed" = type { %"union.google::protobuf::internal::ExplicitlyConstructed<tutorial::Person_PhoneNumber>::AlignedUnion" }
%"union.google::protobuf::internal::ExplicitlyConstructed<tutorial::Person_PhoneNumber>::AlignedUnion" = type { i64, [24 x i8] }
%"class.tutorial::PersonDefaultTypeInternal" = type { %"class.google::protobuf::internal::ExplicitlyConstructed.0" }
%"class.google::protobuf::internal::ExplicitlyConstructed.0" = type { %"union.google::protobuf::internal::ExplicitlyConstructed<tutorial::Person>::AlignedUnion" }
%"union.google::protobuf::internal::ExplicitlyConstructed<tutorial::Person>::AlignedUnion" = type { i64, [64 x i8] }
%"class.tutorial::AddressBookDefaultTypeInternal" = type { %"class.google::protobuf::internal::ExplicitlyConstructed.1" }
%"class.google::protobuf::internal::ExplicitlyConstructed.1" = type { %"union.google::protobuf::internal::ExplicitlyConstructed<tutorial::AddressBook>::AlignedUnion" }
%"union.google::protobuf::internal::ExplicitlyConstructed<tutorial::AddressBook>::AlignedUnion" = type { i64, [40 x i8] }
%"struct.google::protobuf::internal::SCCInfo.3" = type { %"struct.google::protobuf::internal::SCCInfoBase", [0 x i8*] }
%"struct.google::protobuf::internal::SCCInfoBase" = type { %"struct.std::atomic", i32, i32, void ()* }
%"struct.std::atomic" = type { %"struct.std::__atomic_base" }
%"struct.std::__atomic_base" = type { i32 }
%"struct.std::once_flag" = type { i32 }
%"struct.google::protobuf::internal::DescriptorTable" = type { i8, i8, i8*, i8*, i32, %"struct.std::once_flag"*, %"struct.google::protobuf::internal::SCCInfoBase"**, %"struct.google::protobuf::internal::DescriptorTable"**, i32, i32, %"struct.google::protobuf::internal::MigrationSchema"*, %"class.google::protobuf::Message"**, i32*, %"struct.google::protobuf::Metadata"*, i32, %"class.google::protobuf::EnumDescriptor"**, %"class.google::protobuf::ServiceDescriptor"** }
%"struct.google::protobuf::internal::MigrationSchema" = type { i32, i32, i32 }
%"class.google::protobuf::Message" = type { %"class.google::protobuf::MessageLite" }
%"class.google::protobuf::MessageLite" = type { i32 (...)**, %"class.google::protobuf::internal::InternalMetadata" }
%"class.google::protobuf::internal::InternalMetadata" = type { i8* }
%"struct.google::protobuf::Metadata" = type { %"class.google::protobuf::Descriptor"*, %"class.google::protobuf::Reflection"* }
%"class.google::protobuf::Descriptor" = type <{ %"class.std::__cxx11::basic_string"*, %"class.std::__cxx11::basic_string"*, %"class.google::protobuf::FileDescriptor"*, %"class.google::protobuf::Descriptor"*, %"class.google::protobuf::MessageOptions"*, %"class.google::protobuf::FieldDescriptor"*, %"class.google::protobuf::OneofDescriptor"*, %"class.google::protobuf::Descriptor"*, %"class.google::protobuf::EnumDescriptor"*, %"struct.google::protobuf::Descriptor::ExtensionRange"*, %"class.google::protobuf::FieldDescriptor"*, %"struct.google::protobuf::Descriptor::ReservedRange"*, %"class.std::__cxx11::basic_string"**, i32, i32, i32, i32, i32, i32, i32, i32, i32, i8, i8, i8, i8 }>
%"class.std::__cxx11::basic_string" = type { %"struct.std::__cxx11::basic_string<char>::_Alloc_hider", i64, %union.anon }
%"struct.std::__cxx11::basic_string<char>::_Alloc_hider" = type { i8* }
%union.anon = type { i64, [8 x i8] }
%"class.google::protobuf::FileDescriptor" = type { %"class.std::__cxx11::basic_string"*, %"class.std::__cxx11::basic_string"*, %"class.google::protobuf::DescriptorPool"*, %"struct.std::once_flag"*, i32, i32, i32, i32, i32, i32, i32, i32, i8, i8, %"class.google::protobuf::FileDescriptor"**, %"class.std::__cxx11::basic_string"**, i32*, i32*, %"class.google::protobuf::Descriptor"*, %"class.google::protobuf::EnumDescriptor"*, %"class.google::protobuf::ServiceDescriptor"*, %"class.google::protobuf::FieldDescriptor"*, %"class.google::protobuf::FileOptions"*, %"class.google::protobuf::FileDescriptorTables"*, %"class.google::protobuf::SourceCodeInfo"* }
%"class.google::protobuf::DescriptorPool" = type { %"class.google::protobuf::internal::WrappedMutex"*, %"class.google::protobuf::DescriptorDatabase"*, %"class.google::protobuf::DescriptorPool::ErrorCollector"*, %"class.google::protobuf::DescriptorPool"*, %"class.std::unique_ptr", i8, i8, i8, i8, i8, %"class.std::map" }
%"class.google::protobuf::internal::WrappedMutex" = type { %"class.std::mutex" }
%"class.std::mutex" = type { %"class.std::__mutex_base" }
%"class.std::__mutex_base" = type { %union.pthread_mutex_t }
%union.pthread_mutex_t = type { %struct.__pthread_mutex_s }
%struct.__pthread_mutex_s = type { i32, i32, i32, i32, i32, i16, i16, %struct.__pthread_internal_list }
%struct.__pthread_internal_list = type { %struct.__pthread_internal_list*, %struct.__pthread_internal_list* }
%"class.google::protobuf::DescriptorDatabase" = type opaque
%"class.google::protobuf::DescriptorPool::ErrorCollector" = type { i32 (...)** }
%"class.std::unique_ptr" = type { %"struct.std::__uniq_ptr_data" }
%"struct.std::__uniq_ptr_data" = type { %"class.std::__uniq_ptr_impl" }
%"class.std::__uniq_ptr_impl" = type { %"class.std::tuple" }
%"class.std::tuple" = type { %"struct.std::_Tuple_impl" }
%"struct.std::_Tuple_impl" = type { %"struct.std::_Head_base.5" }
%"struct.std::_Head_base.5" = type { %"class.google::protobuf::DescriptorPool::Tables"* }
%"class.google::protobuf::DescriptorPool::Tables" = type opaque
%"class.std::map" = type { %"class.std::_Rb_tree" }
%"class.std::_Rb_tree" = type { %"struct.std::_Rb_tree<std::__cxx11::basic_string<char>, std::pair<const std::__cxx11::basic_string<char>, bool>, std::_Select1st<std::pair<const std::__cxx11::basic_string<char>, bool>>, std::less<std::__cxx11::basic_string<char>>>::_Rb_tree_impl" }
%"struct.std::_Rb_tree<std::__cxx11::basic_string<char>, std::pair<const std::__cxx11::basic_string<char>, bool>, std::_Select1st<std::pair<const std::__cxx11::basic_string<char>, bool>>, std::less<std::__cxx11::basic_string<char>>>::_Rb_tree_impl" = type { %"struct.std::_Rb_tree_key_compare", %"struct.std::_Rb_tree_header" }
%"struct.std::_Rb_tree_key_compare" = type { %"struct.std::less" }
%"struct.std::less" = type { i8 }
%"struct.std::_Rb_tree_header" = type { %"struct.std::_Rb_tree_node_base", i64 }
%"struct.std::_Rb_tree_node_base" = type { i32, %"struct.std::_Rb_tree_node_base"*, %"struct.std::_Rb_tree_node_base"*, %"struct.std::_Rb_tree_node_base"* }
%"class.google::protobuf::ServiceDescriptor" = type <{ %"class.std::__cxx11::basic_string"*, %"class.std::__cxx11::basic_string"*, %"class.google::protobuf::FileDescriptor"*, %"class.google::protobuf::ServiceOptions"*, %"class.google::protobuf::MethodDescriptor"*, i32, [4 x i8] }>
%"class.google::protobuf::ServiceOptions" = type opaque
%"class.google::protobuf::MethodDescriptor" = type <{ %"class.std::__cxx11::basic_string"*, %"class.std::__cxx11::basic_string"*, %"class.google::protobuf::ServiceDescriptor"*, %"class.google::protobuf::internal::LazyDescriptor", %"class.google::protobuf::internal::LazyDescriptor", %"class.google::protobuf::MethodOptions"*, i8, i8, [6 x i8] }>
%"class.google::protobuf::internal::LazyDescriptor" = type { %"class.google::protobuf::Descriptor"*, %"class.std::__cxx11::basic_string"*, %"struct.std::once_flag"*, %"class.google::protobuf::FileDescriptor"* }
%"class.google::protobuf::MethodOptions" = type opaque
%"class.google::protobuf::FileOptions" = type opaque
%"class.google::protobuf::FileDescriptorTables" = type opaque
%"class.google::protobuf::SourceCodeInfo" = type opaque
%"class.google::protobuf::MessageOptions" = type opaque
%"class.google::protobuf::OneofDescriptor" = type { %"class.std::__cxx11::basic_string"*, %"class.std::__cxx11::basic_string"*, %"class.google::protobuf::Descriptor"*, i32, %"class.google::protobuf::FieldDescriptor"**, %"class.google::protobuf::OneofOptions"* }
%"class.google::protobuf::OneofOptions" = type opaque
%"class.google::protobuf::EnumDescriptor" = type { %"class.std::__cxx11::basic_string"*, %"class.std::__cxx11::basic_string"*, %"class.google::protobuf::FileDescriptor"*, %"class.google::protobuf::Descriptor"*, %"class.google::protobuf::EnumOptions"*, i8, i8, i32, %"class.google::protobuf::EnumValueDescriptor"*, i32, i32, %"struct.google::protobuf::EnumDescriptor::ReservedRange"*, %"class.std::__cxx11::basic_string"** }
%"class.google::protobuf::EnumOptions" = type opaque
%"class.google::protobuf::EnumValueDescriptor" = type { %"class.std::__cxx11::basic_string"*, %"class.std::__cxx11::basic_string"*, i32, %"class.google::protobuf::EnumDescriptor"*, %"class.google::protobuf::EnumValueOptions"* }
%"class.google::protobuf::EnumValueOptions" = type opaque
%"struct.google::protobuf::EnumDescriptor::ReservedRange" = type { i32, i32 }
%"struct.google::protobuf::Descriptor::ExtensionRange" = type { i32, i32, %"class.google::protobuf::ExtensionRangeOptions"* }
%"class.google::protobuf::ExtensionRangeOptions" = type opaque
%"class.google::protobuf::FieldDescriptor" = type { %"class.std::__cxx11::basic_string"*, %"class.std::__cxx11::basic_string"*, %"class.std::__cxx11::basic_string"*, %"class.std::__cxx11::basic_string"*, %"class.std::__cxx11::basic_string"*, %"class.google::protobuf::FileDescriptor"*, %"struct.std::once_flag"*, i32, i32, i8, i8, i8, i8, i32, i32, %"class.google::protobuf::Descriptor"*, %"class.google::protobuf::OneofDescriptor"*, %"class.google::protobuf::Descriptor"*, %"class.google::protobuf::Descriptor"*, %"class.google::protobuf::EnumDescriptor"*, %"class.google::protobuf::FieldOptions"*, %"class.std::__cxx11::basic_string"*, %"class.std::__cxx11::basic_string"*, %union.anon.9 }
%"class.google::protobuf::FieldOptions" = type opaque
%union.anon.9 = type { i64 }
%"struct.google::protobuf::Descriptor::ReservedRange" = type { i32, i32 }
%"class.google::protobuf::Reflection" = type <{ %"class.google::protobuf::Descriptor"*, %"struct.google::protobuf::internal::ReflectionSchema", %"class.google::protobuf::DescriptorPool"*, %"class.google::protobuf::MessageFactory"*, i32, [4 x i8] }>
%"struct.google::protobuf::internal::ReflectionSchema" = type { %"class.google::protobuf::Message"*, i32*, i32*, i32, i32, i32, i32, i32, i32 }
%"class.google::protobuf::MessageFactory" = type { i32 (...)** }
%"class.google::protobuf::internal::ExplicitlyConstructed.20" = type { %"union.google::protobuf::internal::ExplicitlyConstructed<std::__cxx11::basic_string<char>>::AlignedUnion" }
%"union.google::protobuf::internal::ExplicitlyConstructed<std::__cxx11::basic_string<char>>::AlignedUnion" = type { i64, [24 x i8] }
%"class.google::protobuf::TimestampDefaultTypeInternal" = type opaque
%"class.tutorial::Person_PhoneNumber" = type { %"class.google::protobuf::Message", %"struct.google::protobuf::internal::ArenaStringPtr", i32, %"class.google::protobuf::internal::CachedSize" }
%"struct.google::protobuf::internal::ArenaStringPtr" = type { %"class.std::__cxx11::basic_string"* }
%"class.google::protobuf::internal::CachedSize" = type { %"struct.std::atomic" }
%"class.google::protobuf::Arena" = type { %"class.google::protobuf::internal::ArenaImpl", void (%"class.std::type_info"*, i64, i8*)*, void (%"class.google::protobuf::Arena"*, i8*, i64)*, void (%"class.google::protobuf::Arena"*, i8*, i64)*, i8* }
%"class.google::protobuf::internal::ArenaImpl" = type { %"struct.std::atomic.10", %"struct.std::atomic.10", %"struct.std::atomic.12", %"class.google::protobuf::internal::ArenaImpl::Block"*, i64, %"struct.google::protobuf::internal::ArenaImpl::Options" }
%"struct.std::atomic.10" = type { %"struct.std::__atomic_base.11" }
%"struct.std::__atomic_base.11" = type { %"class.google::protobuf::internal::ArenaImpl::SerialArena"* }
%"class.google::protobuf::internal::ArenaImpl::SerialArena" = type { %"class.google::protobuf::internal::ArenaImpl"*, i8*, %"class.google::protobuf::internal::ArenaImpl::Block"*, %"struct.google::protobuf::internal::ArenaImpl::CleanupChunk"*, %"class.google::protobuf::internal::ArenaImpl::SerialArena"*, i8*, i8*, %"struct.google::protobuf::internal::ArenaImpl::CleanupNode"*, %"struct.google::protobuf::internal::ArenaImpl::CleanupNode"* }
%"struct.google::protobuf::internal::ArenaImpl::CleanupChunk" = type { i64, %"struct.google::protobuf::internal::ArenaImpl::CleanupChunk"*, [1 x %"struct.google::protobuf::internal::ArenaImpl::CleanupNode"] }
%"struct.google::protobuf::internal::ArenaImpl::CleanupNode" = type { i8*, void (i8*)* }
%"struct.std::atomic.12" = type { %"struct.std::__atomic_base.13" }
%"struct.std::__atomic_base.13" = type { i64 }
%"class.google::protobuf::internal::ArenaImpl::Block" = type { %"class.google::protobuf::internal::ArenaImpl::Block"*, i64, i64 }
%"struct.google::protobuf::internal::ArenaImpl::Options" = type { i64, i64, i8*, i64, i8* (i64)*, void (i8*, i64)* }
%"class.std::type_info" = type { i32 (...)**, i8* }
%"class.tutorial::Person" = type { %"class.google::protobuf::Message", %"class.google::protobuf::RepeatedPtrField", %"struct.google::protobuf::internal::ArenaStringPtr", %"struct.google::protobuf::internal::ArenaStringPtr", %"class.google::protobuf::Timestamp"*, i32, %"class.google::protobuf::internal::CachedSize" }
%"class.google::protobuf::RepeatedPtrField" = type { %"class.google::protobuf::internal::RepeatedPtrFieldBase" }
%"class.google::protobuf::internal::RepeatedPtrFieldBase" = type { %"class.google::protobuf::Arena"*, i32, i32, %"struct.google::protobuf::internal::RepeatedPtrFieldBase::Rep"* }
%"struct.google::protobuf::internal::RepeatedPtrFieldBase::Rep" = type { i32, [1 x i8*] }
%"class.google::protobuf::Timestamp" = type { %"class.google::protobuf::Message", i64, i32, %"class.google::protobuf::internal::CachedSize" }
%"class.tutorial::AddressBook" = type <{ %"class.google::protobuf::Message", %"class.google::protobuf::RepeatedPtrField.18", %"class.google::protobuf::internal::CachedSize", [4 x i8] }>
%"class.google::protobuf::RepeatedPtrField.18" = type { %"class.google::protobuf::internal::RepeatedPtrFieldBase" }
%"struct.google::protobuf::internal::InternalMetadata::Container" = type { %"struct.google::protobuf::internal::InternalMetadata::ContainerBase", %"class.google::protobuf::UnknownFieldSet" }
%"struct.google::protobuf::internal::InternalMetadata::ContainerBase" = type { %"class.google::protobuf::Arena"* }
%"class.google::protobuf::UnknownFieldSet" = type { %"class.std::vector" }
%"class.std::vector" = type { %"struct.std::_Vector_base" }
%"struct.std::_Vector_base" = type { %"struct.std::_Vector_base<google::protobuf::UnknownField, std::allocator<google::protobuf::UnknownField>>::_Vector_impl" }
%"struct.std::_Vector_base<google::protobuf::UnknownField, std::allocator<google::protobuf::UnknownField>>::_Vector_impl" = type { %"struct.std::_Vector_base<google::protobuf::UnknownField, std::allocator<google::protobuf::UnknownField>>::_Vector_impl_data" }
%"struct.std::_Vector_base<google::protobuf::UnknownField, std::allocator<google::protobuf::UnknownField>>::_Vector_impl_data" = type { %"class.google::protobuf::UnknownField"*, %"class.google::protobuf::UnknownField"*, %"class.google::protobuf::UnknownField"* }
%"class.google::protobuf::UnknownField" = type { i32, i32, %union.anon.17 }
%union.anon.17 = type { i64 }
%"class.google::protobuf::internal::LogMessage" = type { i32, i8*, i32, %"class.std::__cxx11::basic_string" }
%"class.google::protobuf::internal::LogFinisher" = type { i8 }
%"class.google::protobuf::internal::ParseContext" = type { %"class.google::protobuf::internal::EpsCopyInputStream", i32, i32, %"struct.google::protobuf::internal::ParseContext::Data" }
%"class.google::protobuf::internal::EpsCopyInputStream" = type { i8*, i8*, i8*, i32, i32, %"class.google::protobuf::io::ZeroCopyInputStream"*, [32 x i8], i64, i32, i32 }
%"class.google::protobuf::io::ZeroCopyInputStream" = type { i32 (...)** }
%"struct.google::protobuf::internal::ParseContext::Data" = type { %"class.google::protobuf::DescriptorPool"*, %"class.google::protobuf::MessageFactory"* }
%"class.google::protobuf::io::EpsCopyOutputStream" = type <{ i8*, i8*, [32 x i8], %"class.google::protobuf::io::ZeroCopyOutputStream"*, i8, i8, i8, [5 x i8] }>
%"class.google::protobuf::io::ZeroCopyOutputStream" = type { i32 (...)** }

$_ZN8tutorial18Person_PhoneNumber10SharedDtorEv = comdat any

$_ZN6google8protobuf8internal16InternalMetadata6DeleteINS0_15UnknownFieldSetEEEvv = comdat any

$__clang_call_terminate = comdat any

$_ZN6google8protobuf16RepeatedPtrFieldIN8tutorial18Person_PhoneNumberEED2Ev = comdat any

$_ZN8tutorial6Person10SharedDtorEv = comdat any

$_ZN6google8protobuf16RepeatedPtrFieldIN8tutorial6PersonEED2Ev = comdat any

$_ZNK8tutorial18Person_PhoneNumber3NewEv = comdat any

$_ZNK8tutorial18Person_PhoneNumber3NewEPN6google8protobuf5ArenaE = comdat any

$_ZNK8tutorial18Person_PhoneNumber13GetCachedSizeEv = comdat any

$_ZNK6google8protobuf11MessageLite16InternalGetTableEv = comdat any

$_ZNK8tutorial6Person3NewEv = comdat any

$_ZNK8tutorial6Person3NewEPN6google8protobuf5ArenaE = comdat any

$_ZNK8tutorial6Person13GetCachedSizeEv = comdat any

$_ZNK8tutorial11AddressBook3NewEv = comdat any

$_ZNK8tutorial11AddressBook3NewEPN6google8protobuf5ArenaE = comdat any

$_ZNK8tutorial11AddressBook13GetCachedSizeEv = comdat any

$_ZN6google8protobuf8internal14ArenaStringPtr14CreateInstanceEPNS0_5ArenaEPKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE = comdat any

$_ZN6google8protobuf8internal21arena_destruct_objectINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEEEvPv = comdat any

$_ZN6google8protobuf8internal18EpsCopyInputStream13DoneWithCheckEPPKci = comdat any

$_ZNK6google8protobuf8internal20RepeatedPtrFieldBase3GetINS0_16RepeatedPtrFieldIN8tutorial18Person_PhoneNumberEE11TypeHandlerEEERKNT_4TypeEi = comdat any

$_ZNK6google8protobuf8internal20RepeatedPtrFieldBase3GetINS0_16RepeatedPtrFieldIN8tutorial6PersonEE11TypeHandlerEEERKNT_4TypeEi = comdat any

$_ZN6google8protobuf8internal20RepeatedPtrFieldBase5ClearINS0_16RepeatedPtrFieldIN8tutorial18Person_PhoneNumberEE11TypeHandlerEEEvv = comdat any

$_ZN6google8protobuf8internal20RepeatedPtrFieldBase5ClearINS0_16RepeatedPtrFieldIN8tutorial6PersonEE11TypeHandlerEEEvv = comdat any

$_ZNSt6vectorIN6google8protobuf12UnknownFieldESaIS2_EED2Ev = comdat any

$_ZN6google8protobuf8internal16InternalMetadata27mutable_unknown_fields_slowINS0_15UnknownFieldSetEEEPT_v = comdat any

$_ZN6google8protobuf8internal21arena_destruct_objectINS1_16InternalMetadata9ContainerINS0_15UnknownFieldSetEEEEEvPv = comdat any

$_ZN6google8protobuf8internal20RepeatedPtrFieldBase7DestroyINS0_16RepeatedPtrFieldIN8tutorial18Person_PhoneNumberEE11TypeHandlerEEEvv = comdat any

$_ZN6google8protobuf8internal20RepeatedPtrFieldBaseD2Ev = comdat any

$_ZN6google8protobuf8internal18EpsCopyInputStream9PushLimitEPKci = comdat any

$_ZN6google8protobuf8internal20RepeatedPtrFieldBase9MergeFromINS0_16RepeatedPtrFieldIN8tutorial18Person_PhoneNumberEE11TypeHandlerEEEvRKS2_ = comdat any

$_ZN6google8protobuf8internal18GenericTypeHandlerIN8tutorial18Person_PhoneNumberEE5MergeERKS4_PS4_ = comdat any

$_ZN6google8protobuf8internal20RepeatedPtrFieldBase12InternalSwapEPS2_ = comdat any

$_ZN6google8protobuf8internal20RepeatedPtrFieldBase9MergeFromINS0_16RepeatedPtrFieldIN8tutorial6PersonEE11TypeHandlerEEEvRKS2_ = comdat any

$_ZN6google8protobuf8internal18GenericTypeHandlerIN8tutorial6PersonEE5MergeERKS4_PS4_ = comdat any

$_ZN6google8protobuf5Arena14InternalHelperIN8tutorial6PersonEE9ConstructIJPS1_EEEPS4_PvDpOT_ = comdat any

$_ZTSNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE = comdat any

$_ZTINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE = comdat any

$_ZTSN6google8protobuf8internal16InternalMetadata9ContainerINS0_15UnknownFieldSetEEE = comdat any

$_ZTSN6google8protobuf8internal16InternalMetadata13ContainerBaseE = comdat any

$_ZTIN6google8protobuf8internal16InternalMetadata13ContainerBaseE = comdat any

$_ZTIN6google8protobuf8internal16InternalMetadata9ContainerINS0_15UnknownFieldSetEEE = comdat any

@_ZStL8__ioinit = internal global %"class.std::ios_base::Init" zeroinitializer, align 1
@__dso_handle = external hidden global i8
@_ZN8tutorial37_Person_PhoneNumber_default_instance_E = dso_local global %"class.tutorial::Person_PhoneNumberDefaultTypeInternal" zeroinitializer, align 8
@_ZN8tutorial25_Person_default_instance_E = dso_local global %"class.tutorial::PersonDefaultTypeInternal" zeroinitializer, align 8
@_ZN8tutorial30_AddressBook_default_instance_E = dso_local global %"class.tutorial::AddressBookDefaultTypeInternal" zeroinitializer, align 8
@scc_info_AddressBook_addressbook_2eproto = dso_local global { { { i32 }, i32, i32, void ()* }, [1 x i8*] } { { { i32 }, i32, i32, void ()* } { { i32 } { i32 -1 }, i32 1, i32 0, void ()* @_ZL52InitDefaultsscc_info_AddressBook_addressbook_2eprotov }, [1 x i8*] [i8* bitcast ({ { { i32 }, i32, i32, void ()* }, [2 x i8*] }* @scc_info_Person_addressbook_2eproto to i8*)] }, align 8
@scc_info_Timestamp_google_2fprotobuf_2ftimestamp_2eproto = external global %"struct.google::protobuf::internal::SCCInfo.3", align 8
@scc_info_Person_addressbook_2eproto = dso_local global { { { i32 }, i32, i32, void ()* }, [2 x i8*] } { { { i32 }, i32, i32, void ()* } { { i32 } { i32 -1 }, i32 2, i32 0, void ()* @_ZL47InitDefaultsscc_info_Person_addressbook_2eprotov }, [2 x i8*] [i8* bitcast ({ { { i32 }, i32, i32, void ()* }, [0 x i8*] }* @scc_info_Person_PhoneNumber_addressbook_2eproto to i8*), i8* bitcast (%"struct.google::protobuf::internal::SCCInfo.3"* @scc_info_Timestamp_google_2fprotobuf_2ftimestamp_2eproto to i8*)] }, align 8
@scc_info_Person_PhoneNumber_addressbook_2eproto = dso_local global { { { i32 }, i32, i32, void ()* }, [0 x i8*] } { { { i32 }, i32, i32, void ()* } { { i32 } { i32 -1 }, i32 0, i32 0, void ()* @_ZL59InitDefaultsscc_info_Person_PhoneNumber_addressbook_2eprotov }, [0 x i8*] zeroinitializer }, align 8
@_ZN31TableStruct_addressbook_2eproto7offsetsE = dso_local constant [23 x i32] [i32 -1, i32 8, i32 -1, i32 -1, i32 -1, i32 16, i32 24, i32 -1, i32 8, i32 -1, i32 -1, i32 -1, i32 40, i32 64, i32 48, i32 16, i32 56, i32 -1, i32 8, i32 -1, i32 -1, i32 -1, i32 16], align 16
@_ZL45descriptor_table_protodef_addressbook_2eproto = internal constant [538 x i8] c"\0A\11addressbook.proto\12\08tutorial\1A\1Fgoogle/protobuf/timestamp.proto\22\87\02\0A\06Person\12\0C\0A\04name\18\01 \01(\09\12\0A\0A\02id\18\02 \01(\05\12\0D\0A\05email\18\03 \01(\09\12,\0A\06phones\18\04 \03(\0B2\1C.tutorial.Person.PhoneNumber\120\0A\0Clast_updated\18\05 \01(\0B2\1A.google.protobuf.Timestamp\1AG\0A\0BPhoneNumber\12\0E\0A\06number\18\01 \01(\09\12(\0A\04type\18\02 \01(\0E2\1A.tutorial.Person.PhoneType\22+\0A\09PhoneType\12\0A\0A\06MOBILE\10\00\12\08\0A\04HOME\10\01\12\08\0A\04WORK\10\02\22/\0A\0BAddressBook\12 \0A\06people\18\01 \03(\0B2\10.tutorial.PersonB\95\01\0A\1Bcom.example.tutorial.protosB\11AddressBookProtosP\01Z:github.com/protocolbuffers/protobuf/examples/go/tutorialpb\AA\02$Google.Protobuf.Examples.AddressBookb\06proto3\00", align 16
@.str = private unnamed_addr constant [18 x i8] c"addressbook.proto\00", align 1
@_ZL41descriptor_table_addressbook_2eproto_once = internal global %"struct.std::once_flag" zeroinitializer, align 4
@_ZL41descriptor_table_addressbook_2eproto_sccs = internal constant [3 x %"struct.google::protobuf::internal::SCCInfoBase"*] [%"struct.google::protobuf::internal::SCCInfoBase"* bitcast ({ { { i32 }, i32, i32, void ()* }, [1 x i8*] }* @scc_info_AddressBook_addressbook_2eproto to %"struct.google::protobuf::internal::SCCInfoBase"*), %"struct.google::protobuf::internal::SCCInfoBase"* bitcast ({ { { i32 }, i32, i32, void ()* }, [2 x i8*] }* @scc_info_Person_addressbook_2eproto to %"struct.google::protobuf::internal::SCCInfoBase"*), %"struct.google::protobuf::internal::SCCInfoBase"* bitcast ({ { { i32 }, i32, i32, void ()* }, [0 x i8*] }* @scc_info_Person_PhoneNumber_addressbook_2eproto to %"struct.google::protobuf::internal::SCCInfoBase"*)], align 16
@_ZL41descriptor_table_addressbook_2eproto_deps = internal constant [1 x %"struct.google::protobuf::internal::DescriptorTable"*] [%"struct.google::protobuf::internal::DescriptorTable"* @descriptor_table_google_2fprotobuf_2ftimestamp_2eproto], align 8
@_ZL7schemas = internal constant [3 x %"struct.google::protobuf::internal::MigrationSchema"] [%"struct.google::protobuf::internal::MigrationSchema" { i32 0, i32 -1, i32 32 }, %"struct.google::protobuf::internal::MigrationSchema" { i32 7, i32 -1, i32 72 }, %"struct.google::protobuf::internal::MigrationSchema" { i32 17, i32 -1, i32 48 }], align 16
@_ZL22file_default_instances = internal constant [3 x %"class.google::protobuf::Message"*] [%"class.google::protobuf::Message"* bitcast (%"class.tutorial::Person_PhoneNumberDefaultTypeInternal"* @_ZN8tutorial37_Person_PhoneNumber_default_instance_E to %"class.google::protobuf::Message"*), %"class.google::protobuf::Message"* bitcast (%"class.tutorial::PersonDefaultTypeInternal"* @_ZN8tutorial25_Person_default_instance_E to %"class.google::protobuf::Message"*), %"class.google::protobuf::Message"* bitcast (%"class.tutorial::AddressBookDefaultTypeInternal"* @_ZN8tutorial30_AddressBook_default_instance_E to %"class.google::protobuf::Message"*)], align 16
@_ZL39file_level_metadata_addressbook_2eproto = internal global [3 x %"struct.google::protobuf::Metadata"] zeroinitializer, align 16
@_ZL47file_level_enum_descriptors_addressbook_2eproto = internal global [1 x %"class.google::protobuf::EnumDescriptor"*] zeroinitializer, align 8
@descriptor_table_addressbook_2eproto = dso_local global %"struct.google::protobuf::internal::DescriptorTable" { i8 0, i8 0, i8* getelementptr inbounds ([538 x i8], [538 x i8]* @_ZL45descriptor_table_protodef_addressbook_2eproto, i32 0, i32 0), i8* getelementptr inbounds ([18 x i8], [18 x i8]* @.str, i32 0, i32 0), i32 537, %"struct.std::once_flag"* @_ZL41descriptor_table_addressbook_2eproto_once, %"struct.google::protobuf::internal::SCCInfoBase"** getelementptr inbounds ([3 x %"struct.google::protobuf::internal::SCCInfoBase"*], [3 x %"struct.google::protobuf::internal::SCCInfoBase"*]* @_ZL41descriptor_table_addressbook_2eproto_sccs, i32 0, i32 0), %"struct.google::protobuf::internal::DescriptorTable"** getelementptr inbounds ([1 x %"struct.google::protobuf::internal::DescriptorTable"*], [1 x %"struct.google::protobuf::internal::DescriptorTable"*]* @_ZL41descriptor_table_addressbook_2eproto_deps, i32 0, i32 0), i32 3, i32 1, %"struct.google::protobuf::internal::MigrationSchema"* getelementptr inbounds ([3 x %"struct.google::protobuf::internal::MigrationSchema"], [3 x %"struct.google::protobuf::internal::MigrationSchema"]* @_ZL7schemas, i32 0, i32 0), %"class.google::protobuf::Message"** getelementptr inbounds ([3 x %"class.google::protobuf::Message"*], [3 x %"class.google::protobuf::Message"*]* @_ZL22file_default_instances, i32 0, i32 0), i32* getelementptr inbounds ([23 x i32], [23 x i32]* @_ZN31TableStruct_addressbook_2eproto7offsetsE, i32 0, i32 0), %"struct.google::protobuf::Metadata"* getelementptr inbounds ([3 x %"struct.google::protobuf::Metadata"], [3 x %"struct.google::protobuf::Metadata"]* @_ZL39file_level_metadata_addressbook_2eproto, i32 0, i32 0), i32 3, %"class.google::protobuf::EnumDescriptor"** getelementptr inbounds ([1 x %"class.google::protobuf::EnumDescriptor"*], [1 x %"class.google::protobuf::EnumDescriptor"*]* @_ZL47file_level_enum_descriptors_addressbook_2eproto, i32 0, i32 0), %"class.google::protobuf::ServiceDescriptor"** null }, align 8
@_ZTVN8tutorial18Person_PhoneNumberE = dso_local unnamed_addr constant { [22 x i8*] } { [22 x i8*] [i8* null, i8* bitcast ({ i8*, i8*, i8* }* @_ZTIN8tutorial18Person_PhoneNumberE to i8*), i8* bitcast (void (%"class.tutorial::Person_PhoneNumber"*)* @_ZN8tutorial18Person_PhoneNumberD2Ev to i8*), i8* bitcast (void (%"class.tutorial::Person_PhoneNumber"*)* @_ZN8tutorial18Person_PhoneNumberD0Ev to i8*), i8* bitcast (void (%"class.std::__cxx11::basic_string"*, %"class.google::protobuf::Message"*)* @_ZNK6google8protobuf7Message11GetTypeNameB5cxx11Ev to i8*), i8* bitcast (%"class.tutorial::Person_PhoneNumber"* (%"class.tutorial::Person_PhoneNumber"*)* @_ZNK8tutorial18Person_PhoneNumber3NewEv to i8*), i8* bitcast (%"class.tutorial::Person_PhoneNumber"* (%"class.tutorial::Person_PhoneNumber"*, %"class.google::protobuf::Arena"*)* @_ZNK8tutorial18Person_PhoneNumber3NewEPN6google8protobuf5ArenaE to i8*), i8* bitcast (void (%"class.tutorial::Person_PhoneNumber"*)* @_ZN8tutorial18Person_PhoneNumber5ClearEv to i8*), i8* bitcast (i1 (%"class.tutorial::Person_PhoneNumber"*)* @_ZNK8tutorial18Person_PhoneNumber13IsInitializedEv to i8*), i8* bitcast (void (%"class.std::__cxx11::basic_string"*, %"class.google::protobuf::Message"*)* @_ZNK6google8protobuf7Message25InitializationErrorStringB5cxx11Ev to i8*), i8* bitcast (void (%"class.google::protobuf::Message"*, %"class.google::protobuf::MessageLite"*)* @_ZN6google8protobuf7Message21CheckTypeAndMergeFromERKNS0_11MessageLiteE to i8*), i8* bitcast (i64 (%"class.tutorial::Person_PhoneNumber"*)* @_ZNK8tutorial18Person_PhoneNumber12ByteSizeLongEv to i8*), i8* bitcast (i32 (%"class.tutorial::Person_PhoneNumber"*)* @_ZNK8tutorial18Person_PhoneNumber13GetCachedSizeEv to i8*), i8* bitcast (i8* (%"class.tutorial::Person_PhoneNumber"*, i8*, %"class.google::protobuf::internal::ParseContext"*)* @_ZN8tutorial18Person_PhoneNumber14_InternalParseEPKcPN6google8protobuf8internal12ParseContextE to i8*), i8* bitcast (i8* (%"class.tutorial::Person_PhoneNumber"*, i8*, %"class.google::protobuf::io::EpsCopyOutputStream"*)* @_ZNK8tutorial18Person_PhoneNumber18_InternalSerializeEPhPN6google8protobuf2io19EpsCopyOutputStreamE to i8*), i8* bitcast (i8* (%"class.google::protobuf::MessageLite"*)* @_ZNK6google8protobuf11MessageLite16InternalGetTableEv to i8*), i8* bitcast (void (%"class.tutorial::Person_PhoneNumber"*, %"class.google::protobuf::Message"*)* @_ZN8tutorial18Person_PhoneNumber8CopyFromERKN6google8protobuf7MessageE to i8*), i8* bitcast (void (%"class.tutorial::Person_PhoneNumber"*, %"class.google::protobuf::Message"*)* @_ZN8tutorial18Person_PhoneNumber9MergeFromERKN6google8protobuf7MessageE to i8*), i8* bitcast (void (%"class.google::protobuf::Message"*)* @_ZN6google8protobuf7Message20DiscardUnknownFieldsEv to i8*), i8* bitcast (i64 (%"class.google::protobuf::Message"*)* @_ZNK6google8protobuf7Message13SpaceUsedLongEv to i8*), i8* bitcast (void (%"class.tutorial::Person_PhoneNumber"*, i32)* @_ZNK8tutorial18Person_PhoneNumber13SetCachedSizeEi to i8*), i8* bitcast ({ %"class.google::protobuf::Descriptor"*, %"class.google::protobuf::Reflection"* } (%"class.tutorial::Person_PhoneNumber"*)* @_ZNK8tutorial18Person_PhoneNumber11GetMetadataEv to i8*)] }, align 8
@.str.4 = private unnamed_addr constant [35 x i8] c"tutorial.Person.PhoneNumber.number\00", align 1
@.str.5 = private unnamed_addr constant [24 x i8] c"build/addressbook.pb.cc\00", align 1
@.str.6 = private unnamed_addr constant [34 x i8] c"CHECK failed: (&from) != (this): \00", align 1
@_ZTVN8tutorial6PersonE = dso_local unnamed_addr constant { [22 x i8*] } { [22 x i8*] [i8* null, i8* bitcast ({ i8*, i8*, i8* }* @_ZTIN8tutorial6PersonE to i8*), i8* bitcast (void (%"class.tutorial::Person"*)* @_ZN8tutorial6PersonD2Ev to i8*), i8* bitcast (void (%"class.tutorial::Person"*)* @_ZN8tutorial6PersonD0Ev to i8*), i8* bitcast (void (%"class.std::__cxx11::basic_string"*, %"class.google::protobuf::Message"*)* @_ZNK6google8protobuf7Message11GetTypeNameB5cxx11Ev to i8*), i8* bitcast (%"class.tutorial::Person"* (%"class.tutorial::Person"*)* @_ZNK8tutorial6Person3NewEv to i8*), i8* bitcast (%"class.tutorial::Person"* (%"class.tutorial::Person"*, %"class.google::protobuf::Arena"*)* @_ZNK8tutorial6Person3NewEPN6google8protobuf5ArenaE to i8*), i8* bitcast (void (%"class.tutorial::Person"*)* @_ZN8tutorial6Person5ClearEv to i8*), i8* bitcast (i1 (%"class.tutorial::Person"*)* @_ZNK8tutorial6Person13IsInitializedEv to i8*), i8* bitcast (void (%"class.std::__cxx11::basic_string"*, %"class.google::protobuf::Message"*)* @_ZNK6google8protobuf7Message25InitializationErrorStringB5cxx11Ev to i8*), i8* bitcast (void (%"class.google::protobuf::Message"*, %"class.google::protobuf::MessageLite"*)* @_ZN6google8protobuf7Message21CheckTypeAndMergeFromERKNS0_11MessageLiteE to i8*), i8* bitcast (i64 (%"class.tutorial::Person"*)* @_ZNK8tutorial6Person12ByteSizeLongEv to i8*), i8* bitcast (i32 (%"class.tutorial::Person"*)* @_ZNK8tutorial6Person13GetCachedSizeEv to i8*), i8* bitcast (i8* (%"class.tutorial::Person"*, i8*, %"class.google::protobuf::internal::ParseContext"*)* @_ZN8tutorial6Person14_InternalParseEPKcPN6google8protobuf8internal12ParseContextE to i8*), i8* bitcast (i8* (%"class.tutorial::Person"*, i8*, %"class.google::protobuf::io::EpsCopyOutputStream"*)* @_ZNK8tutorial6Person18_InternalSerializeEPhPN6google8protobuf2io19EpsCopyOutputStreamE to i8*), i8* bitcast (i8* (%"class.google::protobuf::MessageLite"*)* @_ZNK6google8protobuf11MessageLite16InternalGetTableEv to i8*), i8* bitcast (void (%"class.tutorial::Person"*, %"class.google::protobuf::Message"*)* @_ZN8tutorial6Person8CopyFromERKN6google8protobuf7MessageE to i8*), i8* bitcast (void (%"class.tutorial::Person"*, %"class.google::protobuf::Message"*)* @_ZN8tutorial6Person9MergeFromERKN6google8protobuf7MessageE to i8*), i8* bitcast (void (%"class.google::protobuf::Message"*)* @_ZN6google8protobuf7Message20DiscardUnknownFieldsEv to i8*), i8* bitcast (i64 (%"class.google::protobuf::Message"*)* @_ZNK6google8protobuf7Message13SpaceUsedLongEv to i8*), i8* bitcast (void (%"class.tutorial::Person"*, i32)* @_ZNK8tutorial6Person13SetCachedSizeEi to i8*), i8* bitcast ({ %"class.google::protobuf::Descriptor"*, %"class.google::protobuf::Reflection"* } (%"class.tutorial::Person"*)* @_ZNK8tutorial6Person11GetMetadataEv to i8*)] }, align 8
@.str.7 = private unnamed_addr constant [21 x i8] c"tutorial.Person.name\00", align 1
@.str.8 = private unnamed_addr constant [22 x i8] c"tutorial.Person.email\00", align 1
@_ZTVN8tutorial11AddressBookE = dso_local unnamed_addr constant { [22 x i8*] } { [22 x i8*] [i8* null, i8* bitcast ({ i8*, i8*, i8* }* @_ZTIN8tutorial11AddressBookE to i8*), i8* bitcast (void (%"class.tutorial::AddressBook"*)* @_ZN8tutorial11AddressBookD2Ev to i8*), i8* bitcast (void (%"class.tutorial::AddressBook"*)* @_ZN8tutorial11AddressBookD0Ev to i8*), i8* bitcast (void (%"class.std::__cxx11::basic_string"*, %"class.google::protobuf::Message"*)* @_ZNK6google8protobuf7Message11GetTypeNameB5cxx11Ev to i8*), i8* bitcast (%"class.tutorial::AddressBook"* (%"class.tutorial::AddressBook"*)* @_ZNK8tutorial11AddressBook3NewEv to i8*), i8* bitcast (%"class.tutorial::AddressBook"* (%"class.tutorial::AddressBook"*, %"class.google::protobuf::Arena"*)* @_ZNK8tutorial11AddressBook3NewEPN6google8protobuf5ArenaE to i8*), i8* bitcast (void (%"class.tutorial::AddressBook"*)* @_ZN8tutorial11AddressBook5ClearEv to i8*), i8* bitcast (i1 (%"class.tutorial::AddressBook"*)* @_ZNK8tutorial11AddressBook13IsInitializedEv to i8*), i8* bitcast (void (%"class.std::__cxx11::basic_string"*, %"class.google::protobuf::Message"*)* @_ZNK6google8protobuf7Message25InitializationErrorStringB5cxx11Ev to i8*), i8* bitcast (void (%"class.google::protobuf::Message"*, %"class.google::protobuf::MessageLite"*)* @_ZN6google8protobuf7Message21CheckTypeAndMergeFromERKNS0_11MessageLiteE to i8*), i8* bitcast (i64 (%"class.tutorial::AddressBook"*)* @_ZNK8tutorial11AddressBook12ByteSizeLongEv to i8*), i8* bitcast (i32 (%"class.tutorial::AddressBook"*)* @_ZNK8tutorial11AddressBook13GetCachedSizeEv to i8*), i8* bitcast (i8* (%"class.tutorial::AddressBook"*, i8*, %"class.google::protobuf::internal::ParseContext"*)* @_ZN8tutorial11AddressBook14_InternalParseEPKcPN6google8protobuf8internal12ParseContextE to i8*), i8* bitcast (i8* (%"class.tutorial::AddressBook"*, i8*, %"class.google::protobuf::io::EpsCopyOutputStream"*)* @_ZNK8tutorial11AddressBook18_InternalSerializeEPhPN6google8protobuf2io19EpsCopyOutputStreamE to i8*), i8* bitcast (i8* (%"class.google::protobuf::MessageLite"*)* @_ZNK6google8protobuf11MessageLite16InternalGetTableEv to i8*), i8* bitcast (void (%"class.tutorial::AddressBook"*, %"class.google::protobuf::Message"*)* @_ZN8tutorial11AddressBook8CopyFromERKN6google8protobuf7MessageE to i8*), i8* bitcast (void (%"class.tutorial::AddressBook"*, %"class.google::protobuf::Message"*)* @_ZN8tutorial11AddressBook9MergeFromERKN6google8protobuf7MessageE to i8*), i8* bitcast (void (%"class.google::protobuf::Message"*)* @_ZN6google8protobuf7Message20DiscardUnknownFieldsEv to i8*), i8* bitcast (i64 (%"class.google::protobuf::Message"*)* @_ZNK6google8protobuf7Message13SpaceUsedLongEv to i8*), i8* bitcast (void (%"class.tutorial::AddressBook"*, i32)* @_ZNK8tutorial11AddressBook13SetCachedSizeEi to i8*), i8* bitcast ({ %"class.google::protobuf::Descriptor"*, %"class.google::protobuf::Reflection"* } (%"class.tutorial::AddressBook"*)* @_ZNK8tutorial11AddressBook11GetMetadataEv to i8*)] }, align 8
@_ZTVN10__cxxabiv120__si_class_type_infoE = external global i8*
@_ZTSN8tutorial18Person_PhoneNumberE = dso_local constant [32 x i8] c"N8tutorial18Person_PhoneNumberE\00", align 1
@_ZTIN6google8protobuf7MessageE = external constant i8*
@_ZTIN8tutorial18Person_PhoneNumberE = dso_local constant { i8*, i8*, i8* } { i8* bitcast (i8** getelementptr inbounds (i8*, i8** @_ZTVN10__cxxabiv120__si_class_type_infoE, i64 2) to i8*), i8* getelementptr inbounds ([32 x i8], [32 x i8]* @_ZTSN8tutorial18Person_PhoneNumberE, i32 0, i32 0), i8* bitcast (i8** @_ZTIN6google8protobuf7MessageE to i8*) }, align 8
@_ZTSN8tutorial6PersonE = dso_local constant [19 x i8] c"N8tutorial6PersonE\00", align 1
@_ZTIN8tutorial6PersonE = dso_local constant { i8*, i8*, i8* } { i8* bitcast (i8** getelementptr inbounds (i8*, i8** @_ZTVN10__cxxabiv120__si_class_type_infoE, i64 2) to i8*), i8* getelementptr inbounds ([19 x i8], [19 x i8]* @_ZTSN8tutorial6PersonE, i32 0, i32 0), i8* bitcast (i8** @_ZTIN6google8protobuf7MessageE to i8*) }, align 8
@_ZTSN8tutorial11AddressBookE = dso_local constant [25 x i8] c"N8tutorial11AddressBookE\00", align 1
@_ZTIN8tutorial11AddressBookE = dso_local constant { i8*, i8*, i8* } { i8* bitcast (i8** getelementptr inbounds (i8*, i8** @_ZTVN10__cxxabiv120__si_class_type_infoE, i64 2) to i8*), i8* getelementptr inbounds ([25 x i8], [25 x i8]* @_ZTSN8tutorial11AddressBookE, i32 0, i32 0), i8* bitcast (i8** @_ZTIN6google8protobuf7MessageE to i8*) }, align 8
@descriptor_table_google_2fprotobuf_2ftimestamp_2eproto = external global %"struct.google::protobuf::internal::DescriptorTable", align 8
@_ZN6google8protobuf8internal26fixed_address_empty_stringB5cxx11E = external global %"class.google::protobuf::internal::ExplicitlyConstructed.20", align 8
@.str.9 = private unnamed_addr constant [43 x i8] c"/usr/include/google/protobuf/arenastring.h\00", align 1
@.str.10 = private unnamed_addr constant [38 x i8] c"CHECK failed: initial_value != NULL: \00", align 1
@_ZTVN10__cxxabiv117__class_type_infoE = external global i8*
@_ZTSNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE = linkonce_odr dso_local constant [53 x i8] c"NSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE\00", comdat, align 1
@_ZTINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE = linkonce_odr dso_local constant { i8*, i8* } { i8* bitcast (i8** getelementptr inbounds (i8*, i8** @_ZTVN10__cxxabiv117__class_type_infoE, i64 2) to i8*), i8* getelementptr inbounds ([53 x i8], [53 x i8]* @_ZTSNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE, i32 0, i32 0) }, comdat, align 8
@.str.12 = private unnamed_addr constant [38 x i8] c"CHECK failed: GetArena() == nullptr: \00", align 1
@.str.13 = private unnamed_addr constant [45 x i8] c"/usr/include/google/protobuf/parse_context.h\00", align 1
@.str.14 = private unnamed_addr constant [21 x i8] c"CHECK failed: *ptr: \00", align 1
@.str.15 = private unnamed_addr constant [25 x i8] c"size_t to int conversion\00", align 1
@_ZN6google8protobuf28_Timestamp_default_instance_E = external global %"class.google::protobuf::TimestampDefaultTypeInternal", align 1
@.str.16 = private unnamed_addr constant [46 x i8] c"/usr/include/google/protobuf/repeated_field.h\00", align 1
@.str.17 = private unnamed_addr constant [31 x i8] c"CHECK failed: (index) >= (0): \00", align 1
@.str.18 = private unnamed_addr constant [42 x i8] c"CHECK failed: (index) < (current_size_): \00", align 1
@.str.19 = private unnamed_addr constant [27 x i8] c"CHECK failed: (n) >= (0): \00", align 1
@_ZTSN6google8protobuf8internal16InternalMetadata9ContainerINS0_15UnknownFieldSetEEE = linkonce_odr dso_local constant [80 x i8] c"N6google8protobuf8internal16InternalMetadata9ContainerINS0_15UnknownFieldSetEEE\00", comdat, align 1
@_ZTSN6google8protobuf8internal16InternalMetadata13ContainerBaseE = linkonce_odr dso_local constant [61 x i8] c"N6google8protobuf8internal16InternalMetadata13ContainerBaseE\00", comdat, align 1
@_ZTIN6google8protobuf8internal16InternalMetadata13ContainerBaseE = linkonce_odr dso_local constant { i8*, i8* } { i8* bitcast (i8** getelementptr inbounds (i8*, i8** @_ZTVN10__cxxabiv117__class_type_infoE, i64 2) to i8*), i8* getelementptr inbounds ([61 x i8], [61 x i8]* @_ZTSN6google8protobuf8internal16InternalMetadata13ContainerBaseE, i32 0, i32 0) }, comdat, align 8
@_ZTIN6google8protobuf8internal16InternalMetadata9ContainerINS0_15UnknownFieldSetEEE = linkonce_odr dso_local constant { i8*, i8*, i8* } { i8* bitcast (i8** getelementptr inbounds (i8*, i8** @_ZTVN10__cxxabiv120__si_class_type_infoE, i64 2) to i8*), i8* getelementptr inbounds ([80 x i8], [80 x i8]* @_ZTSN6google8protobuf8internal16InternalMetadata9ContainerINS0_15UnknownFieldSetEEE, i32 0, i32 0), i8* bitcast ({ i8*, i8* }* @_ZTIN6google8protobuf8internal16InternalMetadata13ContainerBaseE to i8*) }, comdat, align 8
@.str.20 = private unnamed_addr constant [60 x i8] c"CHECK failed: limit >= 0 && limit <= INT_MAX - kSlopBytes: \00", align 1
@.str.21 = private unnamed_addr constant [35 x i8] c"CHECK failed: (&other) != (this): \00", align 1
@.str.22 = private unnamed_addr constant [30 x i8] c"CHECK failed: this != other: \00", align 1
@.str.23 = private unnamed_addr constant [48 x i8] c"CHECK failed: GetArena() == other->GetArena(): \00", align 1
@llvm.global_ctors = appending global [1 x { i32, void ()*, i8* }] [{ i32, void ()*, i8* } { i32 65535, void ()* @_GLOBAL__sub_I_addressbook.pb.cc, i8* null }]

@_ZN8tutorial18Person_PhoneNumberC1EPN6google8protobuf5ArenaE = dso_local unnamed_addr alias void (%"class.tutorial::Person_PhoneNumber"*, %"class.google::protobuf::Arena"*), void (%"class.tutorial::Person_PhoneNumber"*, %"class.google::protobuf::Arena"*)* @_ZN8tutorial18Person_PhoneNumberC2EPN6google8protobuf5ArenaE
@_ZN8tutorial18Person_PhoneNumberC1ERKS0_ = dso_local unnamed_addr alias void (%"class.tutorial::Person_PhoneNumber"*, %"class.tutorial::Person_PhoneNumber"*), void (%"class.tutorial::Person_PhoneNumber"*, %"class.tutorial::Person_PhoneNumber"*)* @_ZN8tutorial18Person_PhoneNumberC2ERKS0_
@_ZN8tutorial18Person_PhoneNumberD1Ev = dso_local unnamed_addr alias void (%"class.tutorial::Person_PhoneNumber"*), void (%"class.tutorial::Person_PhoneNumber"*)* @_ZN8tutorial18Person_PhoneNumberD2Ev
@_ZN8tutorial6PersonC1EPN6google8protobuf5ArenaE = dso_local unnamed_addr alias void (%"class.tutorial::Person"*, %"class.google::protobuf::Arena"*), void (%"class.tutorial::Person"*, %"class.google::protobuf::Arena"*)* @_ZN8tutorial6PersonC2EPN6google8protobuf5ArenaE
@_ZN8tutorial6PersonC1ERKS0_ = dso_local unnamed_addr alias void (%"class.tutorial::Person"*, %"class.tutorial::Person"*), void (%"class.tutorial::Person"*, %"class.tutorial::Person"*)* @_ZN8tutorial6PersonC2ERKS0_
@_ZN8tutorial6PersonD1Ev = dso_local unnamed_addr alias void (%"class.tutorial::Person"*), void (%"class.tutorial::Person"*)* @_ZN8tutorial6PersonD2Ev
@_ZN8tutorial11AddressBookC1EPN6google8protobuf5ArenaE = dso_local unnamed_addr alias void (%"class.tutorial::AddressBook"*, %"class.google::protobuf::Arena"*), void (%"class.tutorial::AddressBook"*, %"class.google::protobuf::Arena"*)* @_ZN8tutorial11AddressBookC2EPN6google8protobuf5ArenaE
@_ZN8tutorial11AddressBookC1ERKS0_ = dso_local unnamed_addr alias void (%"class.tutorial::AddressBook"*, %"class.tutorial::AddressBook"*), void (%"class.tutorial::AddressBook"*, %"class.tutorial::AddressBook"*)* @_ZN8tutorial11AddressBookC2ERKS0_
@_ZN8tutorial11AddressBookD1Ev = dso_local unnamed_addr alias void (%"class.tutorial::AddressBook"*), void (%"class.tutorial::AddressBook"*)* @_ZN8tutorial11AddressBookD2Ev

declare void @_ZNSt8ios_base4InitC1Ev(%"class.std::ios_base::Init"* noundef nonnull align 1 dereferenceable(1)) unnamed_addr #0

; Function Attrs: nounwind
declare void @_ZNSt8ios_base4InitD1Ev(%"class.std::ios_base::Init"* noundef nonnull align 1 dereferenceable(1)) unnamed_addr #1

; Function Attrs: nofree nounwind
declare i32 @__cxa_atexit(void (i8*)*, i8*, i8*) local_unnamed_addr #2

; Function Attrs: uwtable
define internal void @_ZL52InitDefaultsscc_info_AddressBook_addressbook_2eprotov() #3 personality i32 (...)* @__gxx_personality_v0 {
  tail call void @_ZN6google8protobuf8internal13VerifyVersionEiiPKc(i32 noundef 3012004, i32 noundef 3012000, i8* noundef getelementptr inbounds ([24 x i8], [24 x i8]* @.str.5, i64 0, i64 0))
  store %"class.google::protobuf::Arena"* null, %"class.google::protobuf::Arena"** bitcast (i8* getelementptr inbounds (%"class.tutorial::AddressBookDefaultTypeInternal", %"class.tutorial::AddressBookDefaultTypeInternal"* @_ZN8tutorial30_AddressBook_default_instance_E, i64 0, i32 0, i32 0, i32 1, i64 0) to %"class.google::protobuf::Arena"**), align 8, !tbaa !5
  store i32 (...)** bitcast (i8** getelementptr inbounds ({ [22 x i8*] }, { [22 x i8*] }* @_ZTVN8tutorial11AddressBookE, i64 0, inrange i32 0, i64 2) to i32 (...)**), i32 (...)*** bitcast (%"class.tutorial::AddressBookDefaultTypeInternal"* @_ZN8tutorial30_AddressBook_default_instance_E to i32 (...)***), align 8, !tbaa !10
  tail call void @llvm.memset.p0i8.i64(i8* noundef nonnull align 8 dereferenceable(28) getelementptr inbounds (%"class.tutorial::AddressBookDefaultTypeInternal", %"class.tutorial::AddressBookDefaultTypeInternal"* @_ZN8tutorial30_AddressBook_default_instance_E, i64 0, i32 0, i32 0, i32 1, i64 8), i8 0, i64 28, i1 false)
  %1 = load atomic i32, i32* getelementptr inbounds ({ { { i32 }, i32, i32, void ()* }, [1 x i8*] }, { { { i32 }, i32, i32, void ()* }, [1 x i8*] }* @scc_info_AddressBook_addressbook_2eproto, i64 0, i32 0, i32 0, i32 0) acquire, align 8
  %2 = icmp eq i32 %1, 0
  br i1 %2, label %6, label %3, !prof !12

3:                                                ; preds = %0
  invoke void @_ZN6google8protobuf8internal11InitSCCImplEPNS1_11SCCInfoBaseE(%"struct.google::protobuf::internal::SCCInfoBase"* noundef nonnull bitcast ({ { { i32 }, i32, i32, void ()* }, [1 x i8*] }* @scc_info_AddressBook_addressbook_2eproto to %"struct.google::protobuf::internal::SCCInfoBase"*))
          to label %6 unwind label %4

4:                                                ; preds = %3
  %5 = landingpad { i8*, i32 }
          cleanup
  tail call void @_ZN6google8protobuf16RepeatedPtrFieldIN8tutorial6PersonEED2Ev(%"class.google::protobuf::RepeatedPtrField.18"* noundef nonnull align 8 dereferenceable(24) bitcast (i8* getelementptr inbounds (%"class.tutorial::AddressBookDefaultTypeInternal", %"class.tutorial::AddressBookDefaultTypeInternal"* @_ZN8tutorial30_AddressBook_default_instance_E, i64 0, i32 0, i32 0, i32 1, i64 8) to %"class.google::protobuf::RepeatedPtrField.18"*)) #24
  resume { i8*, i32 } %5

6:                                                ; preds = %0, %3
  tail call void @_ZN6google8protobuf8internal13OnShutdownRunEPFvPKvES3_(void (i8*)* noundef nonnull @_ZN6google8protobuf8internal14DestroyMessageEPKv, i8* noundef bitcast (%"class.tutorial::AddressBookDefaultTypeInternal"* @_ZN8tutorial30_AddressBook_default_instance_E to i8*))
  ret void
}

; Function Attrs: uwtable
define internal void @_ZL47InitDefaultsscc_info_Person_addressbook_2eprotov() #3 personality i32 (...)* @__gxx_personality_v0 {
  tail call void @_ZN6google8protobuf8internal13VerifyVersionEiiPKc(i32 noundef 3012004, i32 noundef 3012000, i8* noundef getelementptr inbounds ([24 x i8], [24 x i8]* @.str.5, i64 0, i64 0))
  store %"class.google::protobuf::Arena"* null, %"class.google::protobuf::Arena"** bitcast (i8* getelementptr inbounds (%"class.tutorial::PersonDefaultTypeInternal", %"class.tutorial::PersonDefaultTypeInternal"* @_ZN8tutorial25_Person_default_instance_E, i64 0, i32 0, i32 0, i32 1, i64 0) to %"class.google::protobuf::Arena"**), align 8, !tbaa !5
  store i32 (...)** bitcast (i8** getelementptr inbounds ({ [22 x i8*] }, { [22 x i8*] }* @_ZTVN8tutorial6PersonE, i64 0, inrange i32 0, i64 2) to i32 (...)**), i32 (...)*** bitcast (%"class.tutorial::PersonDefaultTypeInternal"* @_ZN8tutorial25_Person_default_instance_E to i32 (...)***), align 8, !tbaa !10
  store i32 0, i32* bitcast (i8* getelementptr inbounds (%"class.tutorial::PersonDefaultTypeInternal", %"class.tutorial::PersonDefaultTypeInternal"* @_ZN8tutorial25_Person_default_instance_E, i64 0, i32 0, i32 0, i32 1, i64 60) to i32*), align 4, !tbaa !13
  tail call void @llvm.memset.p0i8.i64(i8* noundef nonnull align 8 dereferenceable(24) getelementptr inbounds (%"class.tutorial::PersonDefaultTypeInternal", %"class.tutorial::PersonDefaultTypeInternal"* @_ZN8tutorial25_Person_default_instance_E, i64 0, i32 0, i32 0, i32 1, i64 8), i8 0, i64 24, i1 false)
  %1 = load atomic i32, i32* getelementptr inbounds ({ { { i32 }, i32, i32, void ()* }, [2 x i8*] }, { { { i32 }, i32, i32, void ()* }, [2 x i8*] }* @scc_info_Person_addressbook_2eproto, i64 0, i32 0, i32 0, i32 0) acquire, align 8
  %2 = icmp eq i32 %1, 0
  br i1 %2, label %19, label %3, !prof !12

3:                                                ; preds = %0
  invoke void @_ZN6google8protobuf8internal11InitSCCImplEPNS1_11SCCInfoBaseE(%"struct.google::protobuf::internal::SCCInfoBase"* noundef nonnull bitcast ({ { { i32 }, i32, i32, void ()* }, [2 x i8*] }* @scc_info_Person_addressbook_2eproto to %"struct.google::protobuf::internal::SCCInfoBase"*))
          to label %19 unwind label %4

4:                                                ; preds = %3
  %5 = landingpad { i8*, i32 }
          cleanup
  invoke void @_ZN6google8protobuf8internal20RepeatedPtrFieldBase7DestroyINS0_16RepeatedPtrFieldIN8tutorial18Person_PhoneNumberEE11TypeHandlerEEEvv(%"class.google::protobuf::internal::RepeatedPtrFieldBase"* noundef nonnull align 8 dereferenceable(24) bitcast (i8* getelementptr inbounds (%"class.tutorial::PersonDefaultTypeInternal", %"class.tutorial::PersonDefaultTypeInternal"* @_ZN8tutorial25_Person_default_instance_E, i64 0, i32 0, i32 0, i32 1, i64 8) to %"class.google::protobuf::internal::RepeatedPtrFieldBase"*))
          to label %6 unwind label %15

6:                                                ; preds = %4
  %7 = load %"class.google::protobuf::Arena"*, %"class.google::protobuf::Arena"** bitcast (i8* getelementptr inbounds (%"class.tutorial::PersonDefaultTypeInternal", %"class.tutorial::PersonDefaultTypeInternal"* @_ZN8tutorial25_Person_default_instance_E, i64 0, i32 0, i32 0, i32 1, i64 8) to %"class.google::protobuf::Arena"**), align 8, !tbaa !16
  %8 = icmp eq %"class.google::protobuf::Arena"* %7, null
  br i1 %8, label %18, label %9

9:                                                ; preds = %6
  %10 = getelementptr inbounds %"class.google::protobuf::Arena", %"class.google::protobuf::Arena"* %7, i64 0, i32 0
  %11 = invoke noundef i64 @_ZNK6google8protobuf8internal9ArenaImpl14SpaceAllocatedEv(%"class.google::protobuf::internal::ArenaImpl"* noundef nonnull align 8 dereferenceable(88) %10)
          to label %18 unwind label %12

12:                                               ; preds = %9
  %13 = landingpad { i8*, i32 }
          catch i8* null
  %14 = extractvalue { i8*, i32 } %13, 0
  tail call void @__clang_call_terminate(i8* %14) #25
  unreachable

15:                                               ; preds = %4
  %16 = landingpad { i8*, i32 }
          catch i8* null
  %17 = extractvalue { i8*, i32 } %16, 0
  tail call void @_ZN6google8protobuf8internal20RepeatedPtrFieldBaseD2Ev(%"class.google::protobuf::internal::RepeatedPtrFieldBase"* noundef nonnull align 8 dereferenceable(24) bitcast (i8* getelementptr inbounds (%"class.tutorial::PersonDefaultTypeInternal", %"class.tutorial::PersonDefaultTypeInternal"* @_ZN8tutorial25_Person_default_instance_E, i64 0, i32 0, i32 0, i32 1, i64 8) to %"class.google::protobuf::internal::RepeatedPtrFieldBase"*)) #24
  tail call void @__clang_call_terminate(i8* %17) #25
  unreachable

18:                                               ; preds = %9, %6
  resume { i8*, i32 } %5

19:                                               ; preds = %0, %3
  store %"class.std::__cxx11::basic_string"* bitcast (%"class.google::protobuf::internal::ExplicitlyConstructed.20"* @_ZN6google8protobuf8internal26fixed_address_empty_stringB5cxx11E to %"class.std::__cxx11::basic_string"*), %"class.std::__cxx11::basic_string"** bitcast (i8* getelementptr inbounds (%"class.tutorial::PersonDefaultTypeInternal", %"class.tutorial::PersonDefaultTypeInternal"* @_ZN8tutorial25_Person_default_instance_E, i64 0, i32 0, i32 0, i32 1, i64 32) to %"class.std::__cxx11::basic_string"**), align 8, !tbaa !18
  store %"class.std::__cxx11::basic_string"* bitcast (%"class.google::protobuf::internal::ExplicitlyConstructed.20"* @_ZN6google8protobuf8internal26fixed_address_empty_stringB5cxx11E to %"class.std::__cxx11::basic_string"*), %"class.std::__cxx11::basic_string"** bitcast (i8* getelementptr inbounds (%"class.tutorial::PersonDefaultTypeInternal", %"class.tutorial::PersonDefaultTypeInternal"* @_ZN8tutorial25_Person_default_instance_E, i64 0, i32 0, i32 0, i32 1, i64 40) to %"class.std::__cxx11::basic_string"**), align 8, !tbaa !18
  tail call void @llvm.memset.p0i8.i64(i8* noundef nonnull align 8 dereferenceable(12) getelementptr inbounds (%"class.tutorial::PersonDefaultTypeInternal", %"class.tutorial::PersonDefaultTypeInternal"* @_ZN8tutorial25_Person_default_instance_E, i64 0, i32 0, i32 0, i32 1, i64 48), i8 0, i64 12, i1 false)
  tail call void @_ZN6google8protobuf8internal13OnShutdownRunEPFvPKvES3_(void (i8*)* noundef nonnull @_ZN6google8protobuf8internal14DestroyMessageEPKv, i8* noundef bitcast (%"class.tutorial::PersonDefaultTypeInternal"* @_ZN8tutorial25_Person_default_instance_E to i8*))
  store %"class.google::protobuf::Timestamp"* bitcast (%"class.google::protobuf::TimestampDefaultTypeInternal"* @_ZN6google8protobuf28_Timestamp_default_instance_E to %"class.google::protobuf::Timestamp"*), %"class.google::protobuf::Timestamp"** bitcast (i8* getelementptr inbounds (%"class.tutorial::PersonDefaultTypeInternal", %"class.tutorial::PersonDefaultTypeInternal"* @_ZN8tutorial25_Person_default_instance_E, i64 0, i32 0, i32 0, i32 1, i64 48) to %"class.google::protobuf::Timestamp"**), align 8, !tbaa !20
  ret void
}

; Function Attrs: uwtable
define internal void @_ZL59InitDefaultsscc_info_Person_PhoneNumber_addressbook_2eprotov() #3 personality i32 (...)* @__gxx_personality_v0 {
  tail call void @_ZN6google8protobuf8internal13VerifyVersionEiiPKc(i32 noundef 3012004, i32 noundef 3012000, i8* noundef getelementptr inbounds ([24 x i8], [24 x i8]* @.str.5, i64 0, i64 0))
  store %"class.google::protobuf::Arena"* null, %"class.google::protobuf::Arena"** bitcast (i8* getelementptr inbounds (%"class.tutorial::Person_PhoneNumberDefaultTypeInternal", %"class.tutorial::Person_PhoneNumberDefaultTypeInternal"* @_ZN8tutorial37_Person_PhoneNumber_default_instance_E, i64 0, i32 0, i32 0, i32 1, i64 0) to %"class.google::protobuf::Arena"**), align 8, !tbaa !5
  store i32 (...)** bitcast (i8** getelementptr inbounds ({ [22 x i8*] }, { [22 x i8*] }* @_ZTVN8tutorial18Person_PhoneNumberE, i64 0, inrange i32 0, i64 2) to i32 (...)**), i32 (...)*** bitcast (%"class.tutorial::Person_PhoneNumberDefaultTypeInternal"* @_ZN8tutorial37_Person_PhoneNumber_default_instance_E to i32 (...)***), align 8, !tbaa !10
  store i32 0, i32* bitcast (i8* getelementptr inbounds (%"class.tutorial::Person_PhoneNumberDefaultTypeInternal", %"class.tutorial::Person_PhoneNumberDefaultTypeInternal"* @_ZN8tutorial37_Person_PhoneNumber_default_instance_E, i64 0, i32 0, i32 0, i32 1, i64 20) to i32*), align 4, !tbaa !13
  %1 = load atomic i32, i32* getelementptr inbounds ({ { { i32 }, i32, i32, void ()* }, [0 x i8*] }, { { { i32 }, i32, i32, void ()* }, [0 x i8*] }* @scc_info_Person_PhoneNumber_addressbook_2eproto, i64 0, i32 0, i32 0, i32 0) acquire, align 8
  %2 = icmp eq i32 %1, 0
  br i1 %2, label %4, label %3, !prof !12

3:                                                ; preds = %0
  tail call void @_ZN6google8protobuf8internal11InitSCCImplEPNS1_11SCCInfoBaseE(%"struct.google::protobuf::internal::SCCInfoBase"* noundef nonnull bitcast ({ { { i32 }, i32, i32, void ()* }, [0 x i8*] }* @scc_info_Person_PhoneNumber_addressbook_2eproto to %"struct.google::protobuf::internal::SCCInfoBase"*))
  br label %4

4:                                                ; preds = %0, %3
  store %"class.std::__cxx11::basic_string"* bitcast (%"class.google::protobuf::internal::ExplicitlyConstructed.20"* @_ZN6google8protobuf8internal26fixed_address_empty_stringB5cxx11E to %"class.std::__cxx11::basic_string"*), %"class.std::__cxx11::basic_string"** bitcast (i8* getelementptr inbounds (%"class.tutorial::Person_PhoneNumberDefaultTypeInternal", %"class.tutorial::Person_PhoneNumberDefaultTypeInternal"* @_ZN8tutorial37_Person_PhoneNumber_default_instance_E, i64 0, i32 0, i32 0, i32 1, i64 8) to %"class.std::__cxx11::basic_string"**), align 8, !tbaa !18
  store i32 0, i32* bitcast (i8* getelementptr inbounds (%"class.tutorial::Person_PhoneNumberDefaultTypeInternal", %"class.tutorial::Person_PhoneNumberDefaultTypeInternal"* @_ZN8tutorial37_Person_PhoneNumber_default_instance_E, i64 0, i32 0, i32 0, i32 1, i64 16) to i32*), align 8, !tbaa !25
  tail call void @_ZN6google8protobuf8internal13OnShutdownRunEPFvPKvES3_(void (i8*)* noundef nonnull @_ZN6google8protobuf8internal14DestroyMessageEPKv, i8* noundef bitcast (%"class.tutorial::Person_PhoneNumberDefaultTypeInternal"* @_ZN8tutorial37_Person_PhoneNumber_default_instance_E to i8*))
  ret void
}

declare void @_ZN6google8protobuf8internal14AddDescriptorsEPKNS1_15DescriptorTableE(%"struct.google::protobuf::internal::DescriptorTable"* noundef) local_unnamed_addr #0

; Function Attrs: mustprogress uwtable
define dso_local noundef %"class.google::protobuf::EnumDescriptor"* @_ZN8tutorial27Person_PhoneType_descriptorEv() local_unnamed_addr #4 {
  tail call void @_ZN6google8protobuf8internal17AssignDescriptorsEPKNS1_15DescriptorTableEb(%"struct.google::protobuf::internal::DescriptorTable"* noundef nonnull @descriptor_table_addressbook_2eproto, i1 noundef zeroext false)
  %1 = load %"class.google::protobuf::EnumDescriptor"*, %"class.google::protobuf::EnumDescriptor"** getelementptr inbounds ([1 x %"class.google::protobuf::EnumDescriptor"*], [1 x %"class.google::protobuf::EnumDescriptor"*]* @_ZL47file_level_enum_descriptors_addressbook_2eproto, i64 0, i64 0), align 8, !tbaa !27
  ret %"class.google::protobuf::EnumDescriptor"* %1
}

declare void @_ZN6google8protobuf8internal17AssignDescriptorsEPKNS1_15DescriptorTableEb(%"struct.google::protobuf::internal::DescriptorTable"* noundef, i1 noundef zeroext) local_unnamed_addr #0

; Function Attrs: mustprogress nofree norecurse nosync nounwind readnone uwtable willreturn
define dso_local noundef zeroext i1 @_ZN8tutorial24Person_PhoneType_IsValidEi(i32 noundef %0) local_unnamed_addr #5 {
  %2 = icmp ult i32 %0, 3
  ret i1 %2
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind readnone uwtable willreturn
define dso_local void @_ZN8tutorial18Person_PhoneNumber21InitAsDefaultInstanceEv() local_unnamed_addr #5 align 2 {
  ret void
}

; Function Attrs: uwtable
define dso_local void @_ZN8tutorial18Person_PhoneNumberC2EPN6google8protobuf5ArenaE(%"class.tutorial::Person_PhoneNumber"* nocapture noundef nonnull writeonly align 8 dereferenceable(32) %0, %"class.google::protobuf::Arena"* noundef %1) unnamed_addr #3 align 2 personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*) {
  %3 = getelementptr inbounds %"class.tutorial::Person_PhoneNumber", %"class.tutorial::Person_PhoneNumber"* %0, i64 0, i32 0, i32 0, i32 1
  %4 = bitcast %"class.google::protobuf::internal::InternalMetadata"* %3 to %"class.google::protobuf::Arena"**
  store %"class.google::protobuf::Arena"* %1, %"class.google::protobuf::Arena"** %4, align 8, !tbaa !5
  %5 = getelementptr inbounds %"class.tutorial::Person_PhoneNumber", %"class.tutorial::Person_PhoneNumber"* %0, i64 0, i32 0, i32 0, i32 0
  store i32 (...)** bitcast (i8** getelementptr inbounds ({ [22 x i8*] }, { [22 x i8*] }* @_ZTVN8tutorial18Person_PhoneNumberE, i64 0, inrange i32 0, i64 2) to i32 (...)**), i32 (...)*** %5, align 8, !tbaa !10
  %6 = getelementptr inbounds %"class.tutorial::Person_PhoneNumber", %"class.tutorial::Person_PhoneNumber"* %0, i64 0, i32 3, i32 0, i32 0, i32 0
  store i32 0, i32* %6, align 4, !tbaa !13
  %7 = load atomic i32, i32* getelementptr inbounds ({ { { i32 }, i32, i32, void ()* }, [0 x i8*] }, { { { i32 }, i32, i32, void ()* }, [0 x i8*] }* @scc_info_Person_PhoneNumber_addressbook_2eproto, i64 0, i32 0, i32 0, i32 0) acquire, align 8
  %8 = icmp eq i32 %7, 0
  br i1 %8, label %10, label %9, !prof !12

9:                                                ; preds = %2
  tail call void @_ZN6google8protobuf8internal11InitSCCImplEPNS1_11SCCInfoBaseE(%"struct.google::protobuf::internal::SCCInfoBase"* noundef nonnull bitcast ({ { { i32 }, i32, i32, void ()* }, [0 x i8*] }* @scc_info_Person_PhoneNumber_addressbook_2eproto to %"struct.google::protobuf::internal::SCCInfoBase"*))
  br label %10

10:                                               ; preds = %9, %2
  %11 = getelementptr inbounds %"class.tutorial::Person_PhoneNumber", %"class.tutorial::Person_PhoneNumber"* %0, i64 0, i32 1, i32 0
  store %"class.std::__cxx11::basic_string"* bitcast (%"class.google::protobuf::internal::ExplicitlyConstructed.20"* @_ZN6google8protobuf8internal26fixed_address_empty_stringB5cxx11E to %"class.std::__cxx11::basic_string"*), %"class.std::__cxx11::basic_string"** %11, align 8, !tbaa !18
  %12 = getelementptr inbounds %"class.tutorial::Person_PhoneNumber", %"class.tutorial::Person_PhoneNumber"* %0, i64 0, i32 2
  store i32 0, i32* %12, align 8, !tbaa !25
  ret void
}

declare i32 @__gxx_personality_v0(...)

; Function Attrs: uwtable
define dso_local void @_ZN8tutorial18Person_PhoneNumberC2ERKS0_(%"class.tutorial::Person_PhoneNumber"* noundef nonnull align 8 dereferenceable(32) %0, %"class.tutorial::Person_PhoneNumber"* nocapture noundef nonnull readonly align 8 dereferenceable(32) %1) unnamed_addr #3 align 2 personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*) {
  %3 = getelementptr inbounds %"class.tutorial::Person_PhoneNumber", %"class.tutorial::Person_PhoneNumber"* %0, i64 0, i32 0, i32 0, i32 1, i32 0
  store i8* null, i8** %3, align 8, !tbaa !5
  %4 = getelementptr inbounds %"class.tutorial::Person_PhoneNumber", %"class.tutorial::Person_PhoneNumber"* %0, i64 0, i32 0, i32 0, i32 0
  store i32 (...)** bitcast (i8** getelementptr inbounds ({ [22 x i8*] }, { [22 x i8*] }* @_ZTVN8tutorial18Person_PhoneNumberE, i64 0, inrange i32 0, i64 2) to i32 (...)**), i32 (...)*** %4, align 8, !tbaa !10
  %5 = getelementptr inbounds %"class.tutorial::Person_PhoneNumber", %"class.tutorial::Person_PhoneNumber"* %0, i64 0, i32 3, i32 0, i32 0, i32 0
  store i32 0, i32* %5, align 4, !tbaa !13
  %6 = getelementptr inbounds %"class.tutorial::Person_PhoneNumber", %"class.tutorial::Person_PhoneNumber"* %1, i64 0, i32 0, i32 0, i32 1, i32 0
  %7 = load i8*, i8** %6, align 8, !tbaa !5
  %8 = ptrtoint i8* %7 to i64
  %9 = and i64 %8, 1
  %10 = icmp eq i64 %9, 0
  br i1 %10, label %17, label %11

11:                                               ; preds = %2
  %12 = getelementptr inbounds %"class.tutorial::Person_PhoneNumber", %"class.tutorial::Person_PhoneNumber"* %0, i64 0, i32 0, i32 0, i32 1
  %13 = and i64 %8, -2
  %14 = inttoptr i64 %13 to %"struct.google::protobuf::internal::InternalMetadata::Container"*
  %15 = getelementptr inbounds %"struct.google::protobuf::internal::InternalMetadata::Container", %"struct.google::protobuf::internal::InternalMetadata::Container"* %14, i64 0, i32 1
  %16 = tail call noundef %"class.google::protobuf::UnknownFieldSet"* @_ZN6google8protobuf8internal16InternalMetadata27mutable_unknown_fields_slowINS0_15UnknownFieldSetEEEPT_v(%"class.google::protobuf::internal::InternalMetadata"* noundef nonnull align 8 dereferenceable(8) %12)
  tail call void @_ZN6google8protobuf15UnknownFieldSet9MergeFromERKS1_(%"class.google::protobuf::UnknownFieldSet"* noundef nonnull align 8 dereferenceable(24) %16, %"class.google::protobuf::UnknownFieldSet"* noundef nonnull align 8 dereferenceable(24) %15)
  br label %17

17:                                               ; preds = %11, %2
  %18 = getelementptr inbounds %"class.tutorial::Person_PhoneNumber", %"class.tutorial::Person_PhoneNumber"* %0, i64 0, i32 1
  %19 = getelementptr inbounds %"struct.google::protobuf::internal::ArenaStringPtr", %"struct.google::protobuf::internal::ArenaStringPtr"* %18, i64 0, i32 0
  store %"class.std::__cxx11::basic_string"* bitcast (%"class.google::protobuf::internal::ExplicitlyConstructed.20"* @_ZN6google8protobuf8internal26fixed_address_empty_stringB5cxx11E to %"class.std::__cxx11::basic_string"*), %"class.std::__cxx11::basic_string"** %19, align 8, !tbaa !18
  %20 = getelementptr inbounds %"class.tutorial::Person_PhoneNumber", %"class.tutorial::Person_PhoneNumber"* %1, i64 0, i32 1, i32 0
  %21 = load %"class.std::__cxx11::basic_string"*, %"class.std::__cxx11::basic_string"** %20, align 8, !tbaa !18
  %22 = getelementptr inbounds %"class.std::__cxx11::basic_string", %"class.std::__cxx11::basic_string"* %21, i64 0, i32 1
  %23 = load i64, i64* %22, align 8, !tbaa !28
  %24 = icmp eq i64 %23, 0
  br i1 %24, label %39, label %25

25:                                               ; preds = %17
  %26 = load i8*, i8** %3, align 8, !tbaa !5
  %27 = ptrtoint i8* %26 to i64
  %28 = and i64 %27, 1
  %29 = icmp eq i64 %28, 0
  %30 = and i64 %27, -2
  br i1 %29, label %35, label %31, !prof !12

31:                                               ; preds = %25
  %32 = inttoptr i64 %30 to %"struct.google::protobuf::internal::InternalMetadata::ContainerBase"*
  %33 = getelementptr inbounds %"struct.google::protobuf::internal::InternalMetadata::ContainerBase", %"struct.google::protobuf::internal::InternalMetadata::ContainerBase"* %32, i64 0, i32 0
  %34 = load %"class.google::protobuf::Arena"*, %"class.google::protobuf::Arena"** %33, align 8, !tbaa !32
  br label %37

35:                                               ; preds = %25
  %36 = inttoptr i64 %30 to %"class.google::protobuf::Arena"*
  br label %37

37:                                               ; preds = %31, %35
  %38 = phi %"class.google::protobuf::Arena"* [ %34, %31 ], [ %36, %35 ]
  tail call void @_ZN6google8protobuf8internal14ArenaStringPtr14CreateInstanceEPNS0_5ArenaEPKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE(%"struct.google::protobuf::internal::ArenaStringPtr"* noundef nonnull align 8 dereferenceable(8) %18, %"class.google::protobuf::Arena"* noundef %38, %"class.std::__cxx11::basic_string"* noundef nonnull %21)
  br label %39

39:                                               ; preds = %37, %17
  %40 = getelementptr inbounds %"class.tutorial::Person_PhoneNumber", %"class.tutorial::Person_PhoneNumber"* %1, i64 0, i32 2
  %41 = load i32, i32* %40, align 8, !tbaa !25
  %42 = getelementptr inbounds %"class.tutorial::Person_PhoneNumber", %"class.tutorial::Person_PhoneNumber"* %0, i64 0, i32 2
  store i32 %41, i32* %42, align 8, !tbaa !25
  ret void
}

; Function Attrs: nounwind uwtable
define dso_local void @_ZN8tutorial18Person_PhoneNumberD2Ev(%"class.tutorial::Person_PhoneNumber"* noundef nonnull align 8 dereferenceable(32) %0) unnamed_addr #6 align 2 personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*) {
  invoke void @_ZN8tutorial18Person_PhoneNumber10SharedDtorEv(%"class.tutorial::Person_PhoneNumber"* noundef nonnull align 8 dereferenceable(32) %0)
          to label %2 unwind label %5

2:                                                ; preds = %1
  %3 = getelementptr inbounds %"class.tutorial::Person_PhoneNumber", %"class.tutorial::Person_PhoneNumber"* %0, i64 0, i32 0, i32 0, i32 1
  invoke void @_ZN6google8protobuf8internal16InternalMetadata6DeleteINS0_15UnknownFieldSetEEEvv(%"class.google::protobuf::internal::InternalMetadata"* noundef nonnull align 8 dereferenceable(8) %3)
          to label %4 unwind label %5

4:                                                ; preds = %2
  ret void

5:                                                ; preds = %2, %1
  %6 = landingpad { i8*, i32 }
          catch i8* null
  %7 = extractvalue { i8*, i32 } %6, 0
  tail call void @__clang_call_terminate(i8* %7) #25
  unreachable
}

; Function Attrs: inlinehint uwtable
define linkonce_odr dso_local void @_ZN8tutorial18Person_PhoneNumber10SharedDtorEv(%"class.tutorial::Person_PhoneNumber"* noundef nonnull align 8 dereferenceable(32) %0) local_unnamed_addr #7 comdat align 2 personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*) {
  %2 = alloca %"class.google::protobuf::internal::LogMessage", align 8
  %3 = alloca %"class.google::protobuf::internal::LogFinisher", align 1
  %4 = getelementptr inbounds %"class.tutorial::Person_PhoneNumber", %"class.tutorial::Person_PhoneNumber"* %0, i64 0, i32 0, i32 0, i32 1, i32 0
  %5 = load i8*, i8** %4, align 8, !tbaa !5
  %6 = ptrtoint i8* %5 to i64
  %7 = and i64 %6, 1
  %8 = icmp eq i64 %7, 0
  %9 = and i64 %6, -2
  br i1 %8, label %14, label %10, !prof !12

10:                                               ; preds = %1
  %11 = inttoptr i64 %9 to %"struct.google::protobuf::internal::InternalMetadata::ContainerBase"*
  %12 = getelementptr inbounds %"struct.google::protobuf::internal::InternalMetadata::ContainerBase", %"struct.google::protobuf::internal::InternalMetadata::ContainerBase"* %11, i64 0, i32 0
  %13 = load %"class.google::protobuf::Arena"*, %"class.google::protobuf::Arena"** %12, align 8, !tbaa !32
  br label %16

14:                                               ; preds = %1
  %15 = inttoptr i64 %9 to %"class.google::protobuf::Arena"*
  br label %16

16:                                               ; preds = %10, %14
  %17 = phi %"class.google::protobuf::Arena"* [ %13, %10 ], [ %15, %14 ]
  %18 = icmp eq %"class.google::protobuf::Arena"* %17, null
  %19 = getelementptr inbounds %"class.google::protobuf::internal::LogFinisher", %"class.google::protobuf::internal::LogFinisher"* %3, i64 0, i32 0
  call void @llvm.lifetime.start.p0i8(i64 1, i8* nonnull %19) #24
  br i1 %18, label %24, label %20

20:                                               ; preds = %16
  %21 = bitcast %"class.google::protobuf::internal::LogMessage"* %2 to i8*
  call void @llvm.lifetime.start.p0i8(i64 56, i8* nonnull %21) #24
  call void @_ZN6google8protobuf8internal10LogMessageC1ENS0_8LogLevelEPKci(%"class.google::protobuf::internal::LogMessage"* noundef nonnull align 8 dereferenceable(56) %2, i32 noundef 3, i8* noundef getelementptr inbounds ([24 x i8], [24 x i8]* @.str.5, i64 0, i64 0), i32 noundef 218)
  %22 = invoke noundef nonnull align 8 dereferenceable(56) %"class.google::protobuf::internal::LogMessage"* @_ZN6google8protobuf8internal10LogMessagelsEPKc(%"class.google::protobuf::internal::LogMessage"* noundef nonnull align 8 dereferenceable(56) %2, i8* noundef getelementptr inbounds ([38 x i8], [38 x i8]* @.str.12, i64 0, i64 0))
          to label %23 unwind label %42

23:                                               ; preds = %20
  invoke void @_ZN6google8protobuf8internal11LogFinisheraSERNS1_10LogMessageE(%"class.google::protobuf::internal::LogFinisher"* noundef nonnull align 1 dereferenceable(1) %3, %"class.google::protobuf::internal::LogMessage"* noundef nonnull align 8 dereferenceable(56) %22)
          to label %25 unwind label %44

24:                                               ; preds = %16
  call void @llvm.lifetime.end.p0i8(i64 1, i8* nonnull %19) #24
  br label %26

25:                                               ; preds = %23
  call void @llvm.lifetime.end.p0i8(i64 1, i8* nonnull %19) #24
  call void @_ZN6google8protobuf8internal10LogMessageD1Ev(%"class.google::protobuf::internal::LogMessage"* noundef nonnull align 8 dereferenceable(56) %2) #24
  call void @llvm.lifetime.end.p0i8(i64 56, i8* nonnull %21) #24
  br label %26

26:                                               ; preds = %24, %25
  %27 = getelementptr inbounds %"class.tutorial::Person_PhoneNumber", %"class.tutorial::Person_PhoneNumber"* %0, i64 0, i32 1, i32 0
  %28 = load %"class.std::__cxx11::basic_string"*, %"class.std::__cxx11::basic_string"** %27, align 8, !tbaa !18
  %29 = icmp eq %"class.std::__cxx11::basic_string"* %28, bitcast (%"class.google::protobuf::internal::ExplicitlyConstructed.20"* @_ZN6google8protobuf8internal26fixed_address_empty_stringB5cxx11E to %"class.std::__cxx11::basic_string"*)
  %30 = icmp eq %"class.std::__cxx11::basic_string"* %28, null
  %31 = or i1 %29, %30
  br i1 %31, label %41, label %32

32:                                               ; preds = %26
  %33 = getelementptr inbounds %"class.std::__cxx11::basic_string", %"class.std::__cxx11::basic_string"* %28, i64 0, i32 0, i32 0
  %34 = load i8*, i8** %33, align 8, !tbaa !34
  %35 = getelementptr inbounds %"class.std::__cxx11::basic_string", %"class.std::__cxx11::basic_string"* %28, i64 0, i32 2
  %36 = bitcast %union.anon* %35 to i8*
  %37 = icmp eq i8* %34, %36
  br i1 %37, label %39, label %38

38:                                               ; preds = %32
  call void @_ZdlPv(i8* noundef %34) #24
  br label %39

39:                                               ; preds = %38, %32
  %40 = bitcast %"class.std::__cxx11::basic_string"* %28 to i8*
  call void @_ZdlPv(i8* noundef %40) #26
  br label %41

41:                                               ; preds = %26, %39
  ret void

42:                                               ; preds = %20
  %43 = landingpad { i8*, i32 }
          cleanup
  br label %46

44:                                               ; preds = %23
  %45 = landingpad { i8*, i32 }
          cleanup
  call void @llvm.lifetime.end.p0i8(i64 1, i8* nonnull %19) #24
  br label %46

46:                                               ; preds = %42, %44
  %47 = phi { i8*, i32 } [ %45, %44 ], [ %43, %42 ]
  call void @_ZN6google8protobuf8internal10LogMessageD1Ev(%"class.google::protobuf::internal::LogMessage"* noundef nonnull align 8 dereferenceable(56) %2) #24
  call void @llvm.lifetime.end.p0i8(i64 56, i8* nonnull %21) #24
  resume { i8*, i32 } %47
}

; Function Attrs: uwtable
define linkonce_odr dso_local void @_ZN6google8protobuf8internal16InternalMetadata6DeleteINS0_15UnknownFieldSetEEEvv(%"class.google::protobuf::internal::InternalMetadata"* noundef nonnull align 8 dereferenceable(8) %0) local_unnamed_addr #3 comdat align 2 personality i32 (...)* @__gxx_personality_v0 {
  %2 = getelementptr inbounds %"class.google::protobuf::internal::InternalMetadata", %"class.google::protobuf::internal::InternalMetadata"* %0, i64 0, i32 0
  %3 = load i8*, i8** %2, align 8, !tbaa !5
  %4 = ptrtoint i8* %3 to i64
  %5 = and i64 %4, 1
  %6 = icmp eq i64 %5, 0
  br i1 %6, label %37, label %7

7:                                                ; preds = %1
  %8 = and i64 %4, -2
  %9 = inttoptr i64 %8 to %"struct.google::protobuf::internal::InternalMetadata::ContainerBase"*
  %10 = getelementptr inbounds %"struct.google::protobuf::internal::InternalMetadata::ContainerBase", %"struct.google::protobuf::internal::InternalMetadata::ContainerBase"* %9, i64 0, i32 0
  %11 = load %"class.google::protobuf::Arena"*, %"class.google::protobuf::Arena"** %10, align 8, !tbaa !32
  %12 = icmp eq %"class.google::protobuf::Arena"* %11, null
  br i1 %12, label %13, label %37

13:                                               ; preds = %7
  %14 = inttoptr i64 %8 to %"struct.google::protobuf::internal::InternalMetadata::Container"*
  %15 = icmp eq i64 %8, 0
  br i1 %15, label %37, label %16

16:                                               ; preds = %13
  %17 = getelementptr inbounds %"struct.google::protobuf::internal::InternalMetadata::Container", %"struct.google::protobuf::internal::InternalMetadata::Container"* %14, i64 0, i32 1
  %18 = getelementptr inbounds %"class.google::protobuf::UnknownFieldSet", %"class.google::protobuf::UnknownFieldSet"* %17, i64 0, i32 0, i32 0, i32 0, i32 0, i32 0
  %19 = load %"class.google::protobuf::UnknownField"*, %"class.google::protobuf::UnknownField"** %18, align 8, !tbaa !27
  %20 = getelementptr inbounds %"struct.google::protobuf::internal::InternalMetadata::Container", %"struct.google::protobuf::internal::InternalMetadata::Container"* %14, i64 0, i32 1, i32 0, i32 0, i32 0, i32 0, i32 1
  %21 = load %"class.google::protobuf::UnknownField"*, %"class.google::protobuf::UnknownField"** %20, align 8, !tbaa !27
  %22 = icmp eq %"class.google::protobuf::UnknownField"* %19, %21
  br i1 %22, label %26, label %23

23:                                               ; preds = %16
  invoke void @_ZN6google8protobuf15UnknownFieldSet13ClearFallbackEv(%"class.google::protobuf::UnknownFieldSet"* noundef nonnull align 8 dereferenceable(24) %17)
          to label %24 unwind label %31

24:                                               ; preds = %23
  %25 = load %"class.google::protobuf::UnknownField"*, %"class.google::protobuf::UnknownField"** %18, align 8, !tbaa !35
  br label %26

26:                                               ; preds = %24, %16
  %27 = phi %"class.google::protobuf::UnknownField"* [ %25, %24 ], [ %19, %16 ]
  %28 = icmp eq %"class.google::protobuf::UnknownField"* %27, null
  br i1 %28, label %35, label %29

29:                                               ; preds = %26
  %30 = bitcast %"class.google::protobuf::UnknownField"* %27 to i8*
  tail call void @_ZdlPv(i8* noundef nonnull %30) #24
  br label %35

31:                                               ; preds = %23
  %32 = landingpad { i8*, i32 }
          catch i8* null
  %33 = extractvalue { i8*, i32 } %32, 0
  %34 = getelementptr inbounds %"class.google::protobuf::UnknownFieldSet", %"class.google::protobuf::UnknownFieldSet"* %17, i64 0, i32 0
  tail call void @_ZNSt6vectorIN6google8protobuf12UnknownFieldESaIS2_EED2Ev(%"class.std::vector"* noundef nonnull align 8 dereferenceable(24) %34) #24
  tail call void @__clang_call_terminate(i8* %33) #25
  unreachable

35:                                               ; preds = %26, %29
  %36 = inttoptr i64 %8 to i8*
  tail call void @_ZdlPv(i8* noundef %36) #26
  br label %37

37:                                               ; preds = %13, %35, %7, %1
  ret void
}

; Function Attrs: noinline noreturn nounwind
define linkonce_odr hidden void @__clang_call_terminate(i8* %0) local_unnamed_addr #8 comdat {
  %2 = tail call i8* @__cxa_begin_catch(i8* %0) #24
  tail call void @_ZSt9terminatev() #25
  unreachable
}

declare i8* @__cxa_begin_catch(i8*) local_unnamed_addr

declare void @_ZSt9terminatev() local_unnamed_addr

; Function Attrs: nounwind uwtable
define dso_local void @_ZN8tutorial18Person_PhoneNumberD0Ev(%"class.tutorial::Person_PhoneNumber"* noundef nonnull align 8 dereferenceable(32) %0) unnamed_addr #6 align 2 personality i32 (...)* @__gxx_personality_v0 {
  invoke void @_ZN8tutorial18Person_PhoneNumber10SharedDtorEv(%"class.tutorial::Person_PhoneNumber"* noundef nonnull align 8 dereferenceable(32) %0)
          to label %2 unwind label %4

2:                                                ; preds = %1
  %3 = getelementptr inbounds %"class.tutorial::Person_PhoneNumber", %"class.tutorial::Person_PhoneNumber"* %0, i64 0, i32 0, i32 0, i32 1
  invoke void @_ZN6google8protobuf8internal16InternalMetadata6DeleteINS0_15UnknownFieldSetEEEvv(%"class.google::protobuf::internal::InternalMetadata"* noundef nonnull align 8 dereferenceable(8) %3)
          to label %7 unwind label %4

4:                                                ; preds = %2, %1
  %5 = landingpad { i8*, i32 }
          catch i8* null
  %6 = extractvalue { i8*, i32 } %5, 0
  tail call void @__clang_call_terminate(i8* %6) #25
  unreachable

7:                                                ; preds = %2
  %8 = bitcast %"class.tutorial::Person_PhoneNumber"* %0 to i8*
  tail call void @_ZdlPv(i8* noundef nonnull %8) #26
  ret void
}

; Function Attrs: nobuiltin nounwind
declare void @_ZdlPv(i8* noundef) local_unnamed_addr #9

; Function Attrs: mustprogress nofree norecurse nosync nounwind readnone uwtable willreturn
define dso_local void @_ZN8tutorial18Person_PhoneNumber9ArenaDtorEPv(i8* nocapture noundef %0) local_unnamed_addr #5 align 2 {
  ret void
}

; Function Attrs: argmemonly mustprogress nofree nosync nounwind willreturn
declare void @llvm.lifetime.start.p0i8(i64 immarg, i8* nocapture) #10

; Function Attrs: argmemonly mustprogress nofree nosync nounwind willreturn
declare void @llvm.lifetime.end.p0i8(i64 immarg, i8* nocapture) #10

; Function Attrs: mustprogress nofree norecurse nounwind uwtable willreturn
define dso_local void @_ZNK8tutorial18Person_PhoneNumber13SetCachedSizeEi(%"class.tutorial::Person_PhoneNumber"* nocapture noundef nonnull writeonly align 8 dereferenceable(32) %0, i32 noundef %1) unnamed_addr #11 align 2 personality i32 (...)* @__gxx_personality_v0 {
  %3 = getelementptr inbounds %"class.tutorial::Person_PhoneNumber", %"class.tutorial::Person_PhoneNumber"* %0, i64 0, i32 3, i32 0, i32 0, i32 0
  store atomic i32 %1, i32* %3 monotonic, align 4
  ret void
}

; Function Attrs: mustprogress uwtable
define dso_local noundef nonnull align 8 dereferenceable(32) %"class.tutorial::Person_PhoneNumber"* @_ZN8tutorial18Person_PhoneNumber16default_instanceEv() local_unnamed_addr #4 align 2 {
  %1 = load atomic i32, i32* getelementptr inbounds ({ { { i32 }, i32, i32, void ()* }, [0 x i8*] }, { { { i32 }, i32, i32, void ()* }, [0 x i8*] }* @scc_info_Person_PhoneNumber_addressbook_2eproto, i64 0, i32 0, i32 0, i32 0) acquire, align 8
  %2 = icmp eq i32 %1, 0
  br i1 %2, label %4, label %3, !prof !12

3:                                                ; preds = %0
  tail call void @_ZN6google8protobuf8internal11InitSCCImplEPNS1_11SCCInfoBaseE(%"struct.google::protobuf::internal::SCCInfoBase"* noundef nonnull bitcast ({ { { i32 }, i32, i32, void ()* }, [0 x i8*] }* @scc_info_Person_PhoneNumber_addressbook_2eproto to %"struct.google::protobuf::internal::SCCInfoBase"*))
  br label %4

4:                                                ; preds = %0, %3
  ret %"class.tutorial::Person_PhoneNumber"* bitcast (%"class.tutorial::Person_PhoneNumberDefaultTypeInternal"* @_ZN8tutorial37_Person_PhoneNumber_default_instance_E to %"class.tutorial::Person_PhoneNumber"*)
}

; Function Attrs: uwtable
define dso_local void @_ZN8tutorial18Person_PhoneNumber5ClearEv(%"class.tutorial::Person_PhoneNumber"* nocapture noundef nonnull align 8 dereferenceable(32) %0) unnamed_addr #3 align 2 personality i32 (...)* @__gxx_personality_v0 {
  %2 = getelementptr inbounds %"class.tutorial::Person_PhoneNumber", %"class.tutorial::Person_PhoneNumber"* %0, i64 0, i32 1, i32 0
  %3 = load %"class.std::__cxx11::basic_string"*, %"class.std::__cxx11::basic_string"** %2, align 8, !tbaa !18
  %4 = icmp eq %"class.std::__cxx11::basic_string"* %3, bitcast (%"class.google::protobuf::internal::ExplicitlyConstructed.20"* @_ZN6google8protobuf8internal26fixed_address_empty_stringB5cxx11E to %"class.std::__cxx11::basic_string"*)
  br i1 %4, label %9, label %5

5:                                                ; preds = %1
  %6 = getelementptr inbounds %"class.std::__cxx11::basic_string", %"class.std::__cxx11::basic_string"* %3, i64 0, i32 1
  store i64 0, i64* %6, align 8, !tbaa !28
  %7 = getelementptr inbounds %"class.std::__cxx11::basic_string", %"class.std::__cxx11::basic_string"* %3, i64 0, i32 0, i32 0
  %8 = load i8*, i8** %7, align 8, !tbaa !34
  store i8 0, i8* %8, align 1, !tbaa !37
  br label %9

9:                                                ; preds = %1, %5
  %10 = getelementptr inbounds %"class.tutorial::Person_PhoneNumber", %"class.tutorial::Person_PhoneNumber"* %0, i64 0, i32 2
  store i32 0, i32* %10, align 8, !tbaa !25
  %11 = getelementptr inbounds %"class.tutorial::Person_PhoneNumber", %"class.tutorial::Person_PhoneNumber"* %0, i64 0, i32 0, i32 0, i32 1, i32 0
  %12 = load i8*, i8** %11, align 8, !tbaa !5
  %13 = ptrtoint i8* %12 to i64
  %14 = and i64 %13, 1
  %15 = icmp eq i64 %14, 0
  br i1 %15, label %26, label %16

16:                                               ; preds = %9
  %17 = and i64 %13, -2
  %18 = inttoptr i64 %17 to %"struct.google::protobuf::internal::InternalMetadata::Container"*
  %19 = getelementptr inbounds %"struct.google::protobuf::internal::InternalMetadata::Container", %"struct.google::protobuf::internal::InternalMetadata::Container"* %18, i64 0, i32 1
  %20 = getelementptr inbounds %"class.google::protobuf::UnknownFieldSet", %"class.google::protobuf::UnknownFieldSet"* %19, i64 0, i32 0, i32 0, i32 0, i32 0, i32 0
  %21 = load %"class.google::protobuf::UnknownField"*, %"class.google::protobuf::UnknownField"** %20, align 8, !tbaa !27
  %22 = getelementptr inbounds %"struct.google::protobuf::internal::InternalMetadata::Container", %"struct.google::protobuf::internal::InternalMetadata::Container"* %18, i64 0, i32 1, i32 0, i32 0, i32 0, i32 0, i32 1
  %23 = load %"class.google::protobuf::UnknownField"*, %"class.google::protobuf::UnknownField"** %22, align 8, !tbaa !27
  %24 = icmp eq %"class.google::protobuf::UnknownField"* %21, %23
  br i1 %24, label %26, label %25

25:                                               ; preds = %16
  tail call void @_ZN6google8protobuf15UnknownFieldSet13ClearFallbackEv(%"class.google::protobuf::UnknownFieldSet"* noundef nonnull align 8 dereferenceable(24) %19)
  br label %26

26:                                               ; preds = %9, %16, %25
  ret void
}

; Function Attrs: uwtable
define dso_local noundef i8* @_ZN8tutorial18Person_PhoneNumber14_InternalParseEPKcPN6google8protobuf8internal12ParseContextE(%"class.tutorial::Person_PhoneNumber"* noundef nonnull align 8 dereferenceable(32) %0, i8* noundef %1, %"class.google::protobuf::internal::ParseContext"* noundef %2) unnamed_addr #3 align 2 {
  %4 = alloca i8*, align 8
  store i8* %1, i8** %4, align 8, !tbaa !27
  %5 = getelementptr inbounds %"class.google::protobuf::internal::ParseContext", %"class.google::protobuf::internal::ParseContext"* %2, i64 0, i32 0
  %6 = getelementptr inbounds %"class.google::protobuf::internal::ParseContext", %"class.google::protobuf::internal::ParseContext"* %2, i64 0, i32 2
  %7 = getelementptr inbounds %"class.tutorial::Person_PhoneNumber", %"class.tutorial::Person_PhoneNumber"* %0, i64 0, i32 2
  %8 = getelementptr inbounds %"class.tutorial::Person_PhoneNumber", %"class.tutorial::Person_PhoneNumber"* %0, i64 0, i32 1
  %9 = getelementptr inbounds %"class.tutorial::Person_PhoneNumber", %"class.tutorial::Person_PhoneNumber"* %0, i64 0, i32 0, i32 0, i32 1, i32 0
  %10 = getelementptr inbounds %"struct.google::protobuf::internal::ArenaStringPtr", %"struct.google::protobuf::internal::ArenaStringPtr"* %8, i64 0, i32 0
  %11 = getelementptr inbounds %"class.tutorial::Person_PhoneNumber", %"class.tutorial::Person_PhoneNumber"* %0, i64 0, i32 0, i32 0, i32 1
  %12 = getelementptr inbounds %"class.google::protobuf::internal::InternalMetadata", %"class.google::protobuf::internal::InternalMetadata"* %11, i64 0, i32 0
  br label %13

13:                                               ; preds = %139, %3
  %14 = load i32, i32* %6, align 4, !tbaa !38
  %15 = call noundef zeroext i1 @_ZN6google8protobuf8internal18EpsCopyInputStream13DoneWithCheckEPPKci(%"class.google::protobuf::internal::EpsCopyInputStream"* noundef nonnull align 8 dereferenceable(88) %5, i8** noundef nonnull %4, i32 noundef %14)
  %16 = load i8*, i8** %4, align 8, !tbaa !27
  br i1 %15, label %142, label %17

17:                                               ; preds = %13
  %18 = load i8, i8* %16, align 1, !tbaa !37
  %19 = zext i8 %18 to i32
  %20 = icmp sgt i8 %18, -1
  %21 = getelementptr inbounds i8, i8* %16, i64 1
  br i1 %20, label %31, label %22

22:                                               ; preds = %17
  %23 = load i8, i8* %21, align 1, !tbaa !37
  %24 = zext i8 %23 to i32
  %25 = shl nuw nsw i32 %24, 7
  %26 = add nsw i32 %19, -128
  %27 = add nsw i32 %26, %25
  %28 = icmp sgt i8 %23, -1
  br i1 %28, label %29, label %34

29:                                               ; preds = %22
  %30 = getelementptr inbounds i8, i8* %16, i64 2
  br label %31

31:                                               ; preds = %17, %29
  %32 = phi i32 [ %27, %29 ], [ %19, %17 ]
  %33 = phi i8* [ %30, %29 ], [ %21, %17 ]
  store i8* %33, i8** %4, align 8, !tbaa !27
  br label %39

34:                                               ; preds = %22
  %35 = call { i8*, i32 } @_ZN6google8protobuf8internal15ReadTagFallbackEPKcj(i8* noundef nonnull %16, i32 noundef %27)
  %36 = extractvalue { i8*, i32 } %35, 0
  %37 = extractvalue { i8*, i32 } %35, 1
  store i8* %36, i8** %4, align 8, !tbaa !27
  %38 = icmp eq i8* %36, null
  br i1 %38, label %141, label %39, !prof !41

39:                                               ; preds = %31, %34
  %40 = phi i8* [ %33, %31 ], [ %36, %34 ]
  %41 = phi i32 [ %32, %31 ], [ %37, %34 ]
  %42 = lshr i32 %41, 3
  switch i32 %42, label %113 [
    i32 1, label %43
    i32 2, label %80
  ]

43:                                               ; preds = %39
  %44 = and i32 %41, 255
  %45 = icmp eq i32 %44, 10
  br i1 %45, label %46, label %113, !prof !12

46:                                               ; preds = %43
  %47 = load i8*, i8** %9, align 8, !tbaa !5
  %48 = ptrtoint i8* %47 to i64
  %49 = and i64 %48, 1
  %50 = icmp eq i64 %49, 0
  %51 = and i64 %48, -2
  br i1 %50, label %56, label %52, !prof !12

52:                                               ; preds = %46
  %53 = inttoptr i64 %51 to %"struct.google::protobuf::internal::InternalMetadata::ContainerBase"*
  %54 = getelementptr inbounds %"struct.google::protobuf::internal::InternalMetadata::ContainerBase", %"struct.google::protobuf::internal::InternalMetadata::ContainerBase"* %53, i64 0, i32 0
  %55 = load %"class.google::protobuf::Arena"*, %"class.google::protobuf::Arena"** %54, align 8, !tbaa !32
  br label %58

56:                                               ; preds = %46
  %57 = inttoptr i64 %51 to %"class.google::protobuf::Arena"*
  br label %58

58:                                               ; preds = %56, %52
  %59 = phi %"class.google::protobuf::Arena"* [ %55, %52 ], [ %57, %56 ]
  %60 = load %"class.std::__cxx11::basic_string"*, %"class.std::__cxx11::basic_string"** %10, align 8, !tbaa !18
  %61 = icmp eq %"class.std::__cxx11::basic_string"* %60, bitcast (%"class.google::protobuf::internal::ExplicitlyConstructed.20"* @_ZN6google8protobuf8internal26fixed_address_empty_stringB5cxx11E to %"class.std::__cxx11::basic_string"*)
  br i1 %61, label %62, label %65

62:                                               ; preds = %58
  call void @_ZN6google8protobuf8internal14ArenaStringPtr14CreateInstanceEPNS0_5ArenaEPKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE(%"struct.google::protobuf::internal::ArenaStringPtr"* noundef nonnull align 8 dereferenceable(8) %8, %"class.google::protobuf::Arena"* noundef %59, %"class.std::__cxx11::basic_string"* noundef bitcast (%"class.google::protobuf::internal::ExplicitlyConstructed.20"* @_ZN6google8protobuf8internal26fixed_address_empty_stringB5cxx11E to %"class.std::__cxx11::basic_string"*))
  %63 = load %"class.std::__cxx11::basic_string"*, %"class.std::__cxx11::basic_string"** %10, align 8, !tbaa !18
  %64 = load i8*, i8** %4, align 8, !tbaa !27
  br label %65

65:                                               ; preds = %58, %62
  %66 = phi i8* [ %64, %62 ], [ %40, %58 ]
  %67 = phi %"class.std::__cxx11::basic_string"* [ %63, %62 ], [ %60, %58 ]
  %68 = call noundef i8* @_ZN6google8protobuf8internal24InlineGreedyStringParserEPNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEPKcPNS1_12ParseContextE(%"class.std::__cxx11::basic_string"* noundef %67, i8* noundef %66, %"class.google::protobuf::internal::ParseContext"* noundef nonnull %2)
  store i8* %68, i8** %4, align 8, !tbaa !27
  %69 = getelementptr inbounds %"class.std::__cxx11::basic_string", %"class.std::__cxx11::basic_string"* %67, i64 0, i32 0, i32 0
  %70 = load i8*, i8** %69, align 8, !tbaa !34
  %71 = getelementptr inbounds %"class.std::__cxx11::basic_string", %"class.std::__cxx11::basic_string"* %67, i64 0, i32 1
  %72 = load i64, i64* %71, align 8, !tbaa !28
  %73 = icmp slt i64 %72, 0
  br i1 %73, label %74, label %75

74:                                               ; preds = %65
  call void @_ZN6google8protobuf11StringPiece18LogFatalSizeTooBigEmPKc(i64 noundef %72, i8* noundef getelementptr inbounds ([25 x i8], [25 x i8]* @.str.15, i64 0, i64 0))
  br label %75

75:                                               ; preds = %65, %74
  %76 = call noundef zeroext i1 @_ZN6google8protobuf8internal10VerifyUTF8ENS0_11StringPieceEPKc(i8* %70, i64 %72, i8* noundef getelementptr inbounds ([35 x i8], [35 x i8]* @.str.4, i64 0, i64 0))
  %77 = load i8*, i8** %4, align 8
  %78 = icmp eq i8* %77, null
  %79 = select i1 %78, i32 4, i32 2
  br i1 %76, label %139, label %141

80:                                               ; preds = %39
  %81 = and i32 %41, 255
  %82 = icmp eq i32 %81, 16
  br i1 %82, label %83, label %113, !prof !12

83:                                               ; preds = %80
  %84 = load i8, i8* %40, align 1, !tbaa !37
  %85 = zext i8 %84 to i32
  %86 = and i32 %85, 128
  %87 = icmp eq i32 %86, 0
  br i1 %87, label %88, label %90

88:                                               ; preds = %83
  %89 = zext i8 %84 to i64
  br label %101

90:                                               ; preds = %83
  %91 = getelementptr inbounds i8, i8* %40, i64 1
  %92 = load i8, i8* %91, align 1, !tbaa !37
  %93 = zext i8 %92 to i32
  %94 = shl nuw nsw i32 %93, 7
  %95 = add nsw i32 %85, -128
  %96 = add nuw nsw i32 %95, %94
  %97 = and i32 %93, 128
  %98 = icmp eq i32 %97, 0
  br i1 %98, label %99, label %105

99:                                               ; preds = %90
  %100 = zext i32 %96 to i64
  br label %101

101:                                              ; preds = %88, %99
  %102 = phi i64 [ 1, %88 ], [ 2, %99 ]
  %103 = phi i64 [ %89, %88 ], [ %100, %99 ]
  %104 = getelementptr inbounds i8, i8* %40, i64 %102
  store i8* %104, i8** %4, align 8, !tbaa !27
  br label %110

105:                                              ; preds = %90
  %106 = call { i8*, i64 } @_ZN6google8protobuf8internal17VarintParseSlow64EPKcj(i8* noundef nonnull %40, i32 noundef %96)
  %107 = extractvalue { i8*, i64 } %106, 0
  %108 = extractvalue { i8*, i64 } %106, 1
  store i8* %107, i8** %4, align 8, !tbaa !27
  %109 = icmp eq i8* %107, null
  br i1 %109, label %141, label %110, !prof !41

110:                                              ; preds = %101, %105
  %111 = phi i64 [ %103, %101 ], [ %108, %105 ]
  %112 = trunc i64 %111 to i32
  store i32 %112, i32* %7, align 8, !tbaa !25
  br label %139

113:                                              ; preds = %39, %80, %43
  %114 = and i32 %41, 7
  %115 = icmp eq i32 %114, 4
  %116 = icmp eq i32 %41, 0
  %117 = or i1 %116, %115
  br i1 %117, label %118, label %121

118:                                              ; preds = %113
  %119 = add i32 %41, -1
  %120 = getelementptr inbounds %"class.google::protobuf::internal::ParseContext", %"class.google::protobuf::internal::ParseContext"* %2, i64 0, i32 0, i32 8
  store i32 %119, i32* %120, align 8, !tbaa !42
  br label %142

121:                                              ; preds = %113
  %122 = zext i32 %41 to i64
  %123 = load i8*, i8** %12, align 8, !tbaa !5
  %124 = ptrtoint i8* %123 to i64
  %125 = and i64 %124, 1
  %126 = icmp eq i64 %125, 0
  br i1 %126, label %131, label %127, !prof !41

127:                                              ; preds = %121
  %128 = and i64 %124, -2
  %129 = inttoptr i64 %128 to %"struct.google::protobuf::internal::InternalMetadata::Container"*
  %130 = getelementptr inbounds %"struct.google::protobuf::internal::InternalMetadata::Container", %"struct.google::protobuf::internal::InternalMetadata::Container"* %129, i64 0, i32 1
  br label %134

131:                                              ; preds = %121
  %132 = call noundef %"class.google::protobuf::UnknownFieldSet"* @_ZN6google8protobuf8internal16InternalMetadata27mutable_unknown_fields_slowINS0_15UnknownFieldSetEEEPT_v(%"class.google::protobuf::internal::InternalMetadata"* noundef nonnull align 8 dereferenceable(8) %11)
  %133 = load i8*, i8** %4, align 8, !tbaa !27
  br label %134

134:                                              ; preds = %127, %131
  %135 = phi i8* [ %40, %127 ], [ %133, %131 ]
  %136 = phi %"class.google::protobuf::UnknownFieldSet"* [ %130, %127 ], [ %132, %131 ]
  %137 = call noundef i8* @_ZN6google8protobuf8internal17UnknownFieldParseEmPNS0_15UnknownFieldSetEPKcPNS1_12ParseContextE(i64 noundef %122, %"class.google::protobuf::UnknownFieldSet"* noundef %136, i8* noundef %135, %"class.google::protobuf::internal::ParseContext"* noundef nonnull %2)
  store i8* %137, i8** %4, align 8, !tbaa !27
  %138 = icmp eq i8* %137, null
  br i1 %138, label %141, label %139

139:                                              ; preds = %75, %134, %110
  %140 = phi i32 [ 2, %110 ], [ 2, %134 ], [ %79, %75 ]
  switch i32 %140, label %142 [
    i32 2, label %13
    i32 4, label %141
  ]

141:                                              ; preds = %139, %134, %105, %34, %75
  store i8* null, i8** %4, align 8, !tbaa !27
  br label %142

142:                                              ; preds = %139, %13, %141, %118
  %143 = phi i8* [ %40, %118 ], [ null, %141 ], [ %16, %13 ], [ %16, %139 ]
  ret i8* %143
}

declare noundef i8* @_ZN6google8protobuf8internal24InlineGreedyStringParserEPNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEPKcPNS1_12ParseContextE(%"class.std::__cxx11::basic_string"* noundef, i8* noundef, %"class.google::protobuf::internal::ParseContext"* noundef) local_unnamed_addr #0

declare noundef i8* @_ZN6google8protobuf8internal17UnknownFieldParseEmPNS0_15UnknownFieldSetEPKcPNS1_12ParseContextE(i64 noundef, %"class.google::protobuf::UnknownFieldSet"* noundef, i8* noundef, %"class.google::protobuf::internal::ParseContext"* noundef) local_unnamed_addr #0

; Function Attrs: mustprogress uwtable
define dso_local noundef i8* @_ZNK8tutorial18Person_PhoneNumber18_InternalSerializeEPhPN6google8protobuf2io19EpsCopyOutputStreamE(%"class.tutorial::Person_PhoneNumber"* nocapture noundef nonnull readonly align 8 dereferenceable(32) %0, i8* noundef %1, %"class.google::protobuf::io::EpsCopyOutputStream"* noundef %2) unnamed_addr #4 align 2 {
  %4 = getelementptr inbounds %"class.tutorial::Person_PhoneNumber", %"class.tutorial::Person_PhoneNumber"* %0, i64 0, i32 1, i32 0
  %5 = load %"class.std::__cxx11::basic_string"*, %"class.std::__cxx11::basic_string"** %4, align 8, !tbaa !18
  %6 = getelementptr inbounds %"class.std::__cxx11::basic_string", %"class.std::__cxx11::basic_string"* %5, i64 0, i32 1
  %7 = load i64, i64* %6, align 8, !tbaa !28
  %8 = icmp eq i64 %7, 0
  br i1 %8, label %35, label %9

9:                                                ; preds = %3
  %10 = getelementptr inbounds %"class.std::__cxx11::basic_string", %"class.std::__cxx11::basic_string"* %5, i64 0, i32 0, i32 0
  %11 = load i8*, i8** %10, align 8, !tbaa !34
  %12 = trunc i64 %7 to i32
  %13 = tail call noundef zeroext i1 @_ZN6google8protobuf8internal14WireFormatLite16VerifyUtf8StringEPKciNS2_9OperationES4_(i8* noundef %11, i32 noundef %12, i32 noundef 1, i8* noundef getelementptr inbounds ([35 x i8], [35 x i8]* @.str.4, i64 0, i64 0))
  %14 = load %"class.std::__cxx11::basic_string"*, %"class.std::__cxx11::basic_string"** %4, align 8, !tbaa !18
  %15 = getelementptr inbounds %"class.std::__cxx11::basic_string", %"class.std::__cxx11::basic_string"* %14, i64 0, i32 1
  %16 = load i64, i64* %15, align 8, !tbaa !28
  %17 = icmp sgt i64 %16, 127
  br i1 %17, label %26, label %18, !prof !41

18:                                               ; preds = %9
  %19 = getelementptr inbounds %"class.google::protobuf::io::EpsCopyOutputStream", %"class.google::protobuf::io::EpsCopyOutputStream"* %2, i64 0, i32 0
  %20 = load i8*, i8** %19, align 8, !tbaa !44
  %21 = ptrtoint i8* %20 to i64
  %22 = ptrtoint i8* %1 to i64
  %23 = sub i64 14, %22
  %24 = add i64 %23, %21
  %25 = icmp slt i64 %24, %16
  br i1 %25, label %26, label %28, !prof !41

26:                                               ; preds = %18, %9
  %27 = tail call noundef i8* @_ZN6google8protobuf2io19EpsCopyOutputStream30WriteStringMaybeAliasedOutlineEjRKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEPh(%"class.google::protobuf::io::EpsCopyOutputStream"* noundef nonnull align 8 dereferenceable(59) %2, i32 noundef 1, %"class.std::__cxx11::basic_string"* noundef nonnull align 8 dereferenceable(32) %14, i8* noundef %1)
  br label %35

28:                                               ; preds = %18
  store i8 10, i8* %1, align 1, !tbaa !37
  %29 = getelementptr inbounds i8, i8* %1, i64 1
  %30 = trunc i64 %16 to i8
  %31 = getelementptr inbounds i8, i8* %1, i64 2
  store i8 %30, i8* %29, align 1, !tbaa !37
  %32 = getelementptr inbounds %"class.std::__cxx11::basic_string", %"class.std::__cxx11::basic_string"* %14, i64 0, i32 0, i32 0
  %33 = load i8*, i8** %32, align 8, !tbaa !34
  tail call void @llvm.memcpy.p0i8.p0i8.i64(i8* nonnull align 1 %31, i8* align 1 %33, i64 %16, i1 false)
  %34 = getelementptr inbounds i8, i8* %31, i64 %16
  br label %35

35:                                               ; preds = %28, %26, %3
  %36 = phi i8* [ %1, %3 ], [ %27, %26 ], [ %34, %28 ]
  %37 = getelementptr inbounds %"class.tutorial::Person_PhoneNumber", %"class.tutorial::Person_PhoneNumber"* %0, i64 0, i32 2
  %38 = load i32, i32* %37, align 8, !tbaa !25
  %39 = icmp eq i32 %38, 0
  br i1 %39, label %77, label %40

40:                                               ; preds = %35
  %41 = getelementptr inbounds %"class.google::protobuf::io::EpsCopyOutputStream", %"class.google::protobuf::io::EpsCopyOutputStream"* %2, i64 0, i32 0
  %42 = load i8*, i8** %41, align 8, !tbaa !44
  %43 = icmp ugt i8* %42, %36
  br i1 %43, label %47, label %44, !prof !12

44:                                               ; preds = %40
  %45 = tail call noundef i8* @_ZN6google8protobuf2io19EpsCopyOutputStream19EnsureSpaceFallbackEPh(%"class.google::protobuf::io::EpsCopyOutputStream"* noundef nonnull align 8 dereferenceable(59) %2, i8* noundef %36)
  %46 = load i32, i32* %37, align 8, !tbaa !25
  br label %47

47:                                               ; preds = %40, %44
  %48 = phi i32 [ %46, %44 ], [ %38, %40 ]
  %49 = phi i8* [ %45, %44 ], [ %36, %40 ]
  store i8 16, i8* %49, align 1, !tbaa !37
  %50 = getelementptr inbounds i8, i8* %49, i64 1
  %51 = icmp ult i32 %48, 128
  %52 = trunc i32 %48 to i8
  br i1 %51, label %53, label %55

53:                                               ; preds = %47
  store i8 %52, i8* %50, align 1, !tbaa !37
  %54 = getelementptr inbounds i8, i8* %49, i64 2
  br label %77

55:                                               ; preds = %47
  %56 = sext i32 %48 to i64
  %57 = or i8 %52, -128
  store i8 %57, i8* %50, align 1, !tbaa !37
  %58 = lshr i64 %56, 7
  %59 = icmp ult i32 %48, 16384
  br i1 %59, label %60, label %64

60:                                               ; preds = %55
  %61 = trunc i64 %58 to i8
  %62 = getelementptr inbounds i8, i8* %49, i64 2
  store i8 %61, i8* %62, align 1, !tbaa !37
  %63 = getelementptr inbounds i8, i8* %49, i64 3
  br label %77

64:                                               ; preds = %55
  %65 = getelementptr inbounds i8, i8* %49, i64 2
  br label %66

66:                                               ; preds = %66, %64
  %67 = phi i64 [ %58, %64 ], [ %71, %66 ]
  %68 = phi i8* [ %65, %64 ], [ %72, %66 ]
  %69 = trunc i64 %67 to i8
  %70 = or i8 %69, -128
  store i8 %70, i8* %68, align 1, !tbaa !37
  %71 = lshr i64 %67, 7
  %72 = getelementptr inbounds i8, i8* %68, i64 1
  %73 = icmp ugt i64 %67, 16383
  br i1 %73, label %66, label %74, !prof !41, !llvm.loop !47

74:                                               ; preds = %66
  %75 = trunc i64 %71 to i8
  %76 = getelementptr inbounds i8, i8* %68, i64 2
  store i8 %75, i8* %72, align 1, !tbaa !37
  br label %77

77:                                               ; preds = %74, %60, %53, %35
  %78 = phi i8* [ %36, %35 ], [ %54, %53 ], [ %63, %60 ], [ %76, %74 ]
  %79 = getelementptr inbounds %"class.tutorial::Person_PhoneNumber", %"class.tutorial::Person_PhoneNumber"* %0, i64 0, i32 0, i32 0, i32 1, i32 0
  %80 = load i8*, i8** %79, align 8, !tbaa !5
  %81 = ptrtoint i8* %80 to i64
  %82 = and i64 %81, 1
  %83 = icmp eq i64 %82, 0
  br i1 %83, label %89, label %84, !prof !12

84:                                               ; preds = %77
  %85 = and i64 %81, -2
  %86 = inttoptr i64 %85 to %"struct.google::protobuf::internal::InternalMetadata::Container"*
  %87 = getelementptr inbounds %"struct.google::protobuf::internal::InternalMetadata::Container", %"struct.google::protobuf::internal::InternalMetadata::Container"* %86, i64 0, i32 1
  %88 = tail call noundef i8* @_ZN6google8protobuf8internal10WireFormat37InternalSerializeUnknownFieldsToArrayERKNS0_15UnknownFieldSetEPhPNS0_2io19EpsCopyOutputStreamE(%"class.google::protobuf::UnknownFieldSet"* noundef nonnull align 8 dereferenceable(24) %87, i8* noundef %78, %"class.google::protobuf::io::EpsCopyOutputStream"* noundef %2)
  br label %89

89:                                               ; preds = %84, %77
  %90 = phi i8* [ %88, %84 ], [ %78, %77 ]
  ret i8* %90
}

declare noundef zeroext i1 @_ZN6google8protobuf8internal14WireFormatLite16VerifyUtf8StringEPKciNS2_9OperationES4_(i8* noundef, i32 noundef, i32 noundef, i8* noundef) local_unnamed_addr #0

declare noundef i8* @_ZN6google8protobuf8internal10WireFormat37InternalSerializeUnknownFieldsToArrayERKNS0_15UnknownFieldSetEPhPNS0_2io19EpsCopyOutputStreamE(%"class.google::protobuf::UnknownFieldSet"* noundef nonnull align 8 dereferenceable(24), i8* noundef, %"class.google::protobuf::io::EpsCopyOutputStream"* noundef) local_unnamed_addr #0

; Function Attrs: mustprogress uwtable
define dso_local noundef i64 @_ZNK8tutorial18Person_PhoneNumber12ByteSizeLongEv(%"class.tutorial::Person_PhoneNumber"* noundef nonnull align 8 dereferenceable(32) %0) unnamed_addr #4 align 2 personality i32 (...)* @__gxx_personality_v0 {
  %2 = getelementptr inbounds %"class.tutorial::Person_PhoneNumber", %"class.tutorial::Person_PhoneNumber"* %0, i64 0, i32 1, i32 0
  %3 = load %"class.std::__cxx11::basic_string"*, %"class.std::__cxx11::basic_string"** %2, align 8, !tbaa !18
  %4 = getelementptr inbounds %"class.std::__cxx11::basic_string", %"class.std::__cxx11::basic_string"* %3, i64 0, i32 1
  %5 = load i64, i64* %4, align 8, !tbaa !28
  %6 = icmp eq i64 %5, 0
  br i1 %6, label %18, label %7

7:                                                ; preds = %1
  %8 = trunc i64 %5 to i32
  %9 = or i32 %8, 1
  %10 = tail call i32 @llvm.ctlz.i32(i32 %9, i1 true) #24, !range !49
  %11 = xor i32 %10, 31
  %12 = mul nuw nsw i32 %11, 9
  %13 = add nuw nsw i32 %12, 73
  %14 = lshr i32 %13, 6
  %15 = zext i32 %14 to i64
  %16 = add i64 %5, 1
  %17 = add i64 %16, %15
  br label %18

18:                                               ; preds = %7, %1
  %19 = phi i64 [ %17, %7 ], [ 0, %1 ]
  %20 = getelementptr inbounds %"class.tutorial::Person_PhoneNumber", %"class.tutorial::Person_PhoneNumber"* %0, i64 0, i32 2
  %21 = load i32, i32* %20, align 8, !tbaa !25
  %22 = icmp eq i32 %21, 0
  br i1 %22, label %37, label %23

23:                                               ; preds = %18
  %24 = icmp slt i32 %21, 0
  br i1 %24, label %34, label %25

25:                                               ; preds = %23
  %26 = or i32 %21, 1
  %27 = tail call i32 @llvm.ctlz.i32(i32 %26, i1 true) #24, !range !49
  %28 = xor i32 %27, 31
  %29 = mul nuw nsw i32 %28, 9
  %30 = add nuw nsw i32 %29, 73
  %31 = lshr i32 %30, 6
  %32 = add nuw nsw i32 %31, 1
  %33 = zext i32 %32 to i64
  br label %34

34:                                               ; preds = %23, %25
  %35 = phi i64 [ %33, %25 ], [ 11, %23 ]
  %36 = add i64 %35, %19
  br label %37

37:                                               ; preds = %34, %18
  %38 = phi i64 [ %36, %34 ], [ %19, %18 ]
  %39 = getelementptr inbounds %"class.tutorial::Person_PhoneNumber", %"class.tutorial::Person_PhoneNumber"* %0, i64 0, i32 0, i32 0, i32 1
  %40 = getelementptr inbounds %"class.google::protobuf::internal::InternalMetadata", %"class.google::protobuf::internal::InternalMetadata"* %39, i64 0, i32 0
  %41 = load i8*, i8** %40, align 8, !tbaa !5
  %42 = ptrtoint i8* %41 to i64
  %43 = and i64 %42, 1
  %44 = icmp eq i64 %43, 0
  br i1 %44, label %48, label %45, !prof !12

45:                                               ; preds = %37
  %46 = getelementptr inbounds %"class.tutorial::Person_PhoneNumber", %"class.tutorial::Person_PhoneNumber"* %0, i64 0, i32 3
  %47 = tail call noundef i64 @_ZN6google8protobuf8internal24ComputeUnknownFieldsSizeERKNS1_16InternalMetadataEmPNS1_10CachedSizeE(%"class.google::protobuf::internal::InternalMetadata"* noundef nonnull align 8 dereferenceable(8) %39, i64 noundef %38, %"class.google::protobuf::internal::CachedSize"* noundef nonnull %46)
  br label %51

48:                                               ; preds = %37
  %49 = trunc i64 %38 to i32
  %50 = getelementptr inbounds %"class.tutorial::Person_PhoneNumber", %"class.tutorial::Person_PhoneNumber"* %0, i64 0, i32 3, i32 0, i32 0, i32 0
  store atomic i32 %49, i32* %50 monotonic, align 4
  br label %51

51:                                               ; preds = %48, %45
  %52 = phi i64 [ %47, %45 ], [ %38, %48 ]
  ret i64 %52
}

declare noundef i64 @_ZN6google8protobuf8internal24ComputeUnknownFieldsSizeERKNS1_16InternalMetadataEmPNS1_10CachedSizeE(%"class.google::protobuf::internal::InternalMetadata"* noundef nonnull align 8 dereferenceable(8), i64 noundef, %"class.google::protobuf::internal::CachedSize"* noundef) local_unnamed_addr #0

; Function Attrs: mustprogress uwtable
define dso_local void @_ZN8tutorial18Person_PhoneNumber9MergeFromERKN6google8protobuf7MessageE(%"class.tutorial::Person_PhoneNumber"* noundef nonnull align 8 dereferenceable(32) %0, %"class.google::protobuf::Message"* noundef nonnull align 8 dereferenceable(16) %1) unnamed_addr #4 align 2 personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*) {
  %3 = alloca %"class.google::protobuf::internal::LogMessage", align 8
  %4 = alloca %"class.google::protobuf::internal::LogFinisher", align 1
  %5 = getelementptr inbounds %"class.tutorial::Person_PhoneNumber", %"class.tutorial::Person_PhoneNumber"* %0, i64 0, i32 0
  %6 = icmp eq %"class.google::protobuf::Message"* %5, %1
  %7 = getelementptr inbounds %"class.google::protobuf::internal::LogFinisher", %"class.google::protobuf::internal::LogFinisher"* %4, i64 0, i32 0
  call void @llvm.lifetime.start.p0i8(i64 1, i8* nonnull %7) #24
  br i1 %6, label %8, label %12

8:                                                ; preds = %2
  %9 = bitcast %"class.google::protobuf::internal::LogMessage"* %3 to i8*
  call void @llvm.lifetime.start.p0i8(i64 56, i8* nonnull %9) #24
  call void @_ZN6google8protobuf8internal10LogMessageC1ENS0_8LogLevelEPKci(%"class.google::protobuf::internal::LogMessage"* noundef nonnull align 8 dereferenceable(56) %3, i32 noundef 3, i8* noundef getelementptr inbounds ([24 x i8], [24 x i8]* @.str.5, i64 0, i64 0), i32 noundef 358)
  %10 = invoke noundef nonnull align 8 dereferenceable(56) %"class.google::protobuf::internal::LogMessage"* @_ZN6google8protobuf8internal10LogMessagelsEPKc(%"class.google::protobuf::internal::LogMessage"* noundef nonnull align 8 dereferenceable(56) %3, i8* noundef getelementptr inbounds ([34 x i8], [34 x i8]* @.str.6, i64 0, i64 0))
          to label %11 unwind label %19

11:                                               ; preds = %8
  invoke void @_ZN6google8protobuf8internal11LogFinisheraSERNS1_10LogMessageE(%"class.google::protobuf::internal::LogFinisher"* noundef nonnull align 1 dereferenceable(1) %4, %"class.google::protobuf::internal::LogMessage"* noundef nonnull align 8 dereferenceable(56) %10)
          to label %13 unwind label %21

12:                                               ; preds = %2
  call void @llvm.lifetime.end.p0i8(i64 1, i8* nonnull %7) #24
  br label %14

13:                                               ; preds = %11
  call void @llvm.lifetime.end.p0i8(i64 1, i8* nonnull %7) #24
  call void @_ZN6google8protobuf8internal10LogMessageD1Ev(%"class.google::protobuf::internal::LogMessage"* noundef nonnull align 8 dereferenceable(56) %3) #24
  call void @llvm.lifetime.end.p0i8(i64 56, i8* nonnull %9) #24
  br label %14

14:                                               ; preds = %12, %13
  %15 = bitcast %"class.google::protobuf::Message"* %1 to i8*
  %16 = call i8* @__dynamic_cast(i8* nonnull %15, i8* bitcast (i8** @_ZTIN6google8protobuf7MessageE to i8*), i8* bitcast ({ i8*, i8*, i8* }* @_ZTIN8tutorial18Person_PhoneNumberE to i8*), i64 0) #24
  %17 = icmp eq i8* %16, null
  br i1 %17, label %18, label %25

18:                                               ; preds = %14
  call void @_ZN6google8protobuf8internal13ReflectionOps5MergeERKNS0_7MessageEPS3_(%"class.google::protobuf::Message"* noundef nonnull align 8 dereferenceable(16) %1, %"class.google::protobuf::Message"* noundef nonnull %5)
  br label %27

19:                                               ; preds = %8
  %20 = landingpad { i8*, i32 }
          cleanup
  br label %23

21:                                               ; preds = %11
  %22 = landingpad { i8*, i32 }
          cleanup
  call void @llvm.lifetime.end.p0i8(i64 1, i8* nonnull %7) #24
  br label %23

23:                                               ; preds = %19, %21
  %24 = phi { i8*, i32 } [ %22, %21 ], [ %20, %19 ]
  call void @_ZN6google8protobuf8internal10LogMessageD1Ev(%"class.google::protobuf::internal::LogMessage"* noundef nonnull align 8 dereferenceable(56) %3) #24
  call void @llvm.lifetime.end.p0i8(i64 56, i8* nonnull %9) #24
  resume { i8*, i32 } %24

25:                                               ; preds = %14
  %26 = bitcast i8* %16 to %"class.tutorial::Person_PhoneNumber"*
  call void @_ZN8tutorial18Person_PhoneNumber9MergeFromERKS0_(%"class.tutorial::Person_PhoneNumber"* noundef nonnull align 8 dereferenceable(32) %0, %"class.tutorial::Person_PhoneNumber"* noundef nonnull align 8 dereferenceable(32) %26)
  br label %27

27:                                               ; preds = %25, %18
  ret void
}

declare void @_ZN6google8protobuf8internal10LogMessageC1ENS0_8LogLevelEPKci(%"class.google::protobuf::internal::LogMessage"* noundef nonnull align 8 dereferenceable(56), i32 noundef, i8* noundef, i32 noundef) unnamed_addr #0

declare noundef nonnull align 8 dereferenceable(56) %"class.google::protobuf::internal::LogMessage"* @_ZN6google8protobuf8internal10LogMessagelsEPKc(%"class.google::protobuf::internal::LogMessage"* noundef nonnull align 8 dereferenceable(56), i8* noundef) local_unnamed_addr #0

declare void @_ZN6google8protobuf8internal11LogFinisheraSERNS1_10LogMessageE(%"class.google::protobuf::internal::LogFinisher"* noundef nonnull align 1 dereferenceable(1), %"class.google::protobuf::internal::LogMessage"* noundef nonnull align 8 dereferenceable(56)) local_unnamed_addr #0

; Function Attrs: nounwind
declare void @_ZN6google8protobuf8internal10LogMessageD1Ev(%"class.google::protobuf::internal::LogMessage"* noundef nonnull align 8 dereferenceable(56)) unnamed_addr #1

declare void @_ZN6google8protobuf8internal13ReflectionOps5MergeERKNS0_7MessageEPS3_(%"class.google::protobuf::Message"* noundef nonnull align 8 dereferenceable(16), %"class.google::protobuf::Message"* noundef) local_unnamed_addr #0

; Function Attrs: mustprogress uwtable
define dso_local void @_ZN8tutorial18Person_PhoneNumber9MergeFromERKS0_(%"class.tutorial::Person_PhoneNumber"* noundef nonnull align 8 dereferenceable(32) %0, %"class.tutorial::Person_PhoneNumber"* noundef nonnull readonly align 8 dereferenceable(32) %1) local_unnamed_addr #4 align 2 personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*) {
  %3 = alloca %"class.google::protobuf::internal::LogMessage", align 8
  %4 = alloca %"class.google::protobuf::internal::LogFinisher", align 1
  %5 = icmp eq %"class.tutorial::Person_PhoneNumber"* %1, %0
  %6 = getelementptr inbounds %"class.google::protobuf::internal::LogFinisher", %"class.google::protobuf::internal::LogFinisher"* %4, i64 0, i32 0
  call void @llvm.lifetime.start.p0i8(i64 1, i8* nonnull %6) #24
  br i1 %5, label %7, label %11

7:                                                ; preds = %2
  %8 = bitcast %"class.google::protobuf::internal::LogMessage"* %3 to i8*
  call void @llvm.lifetime.start.p0i8(i64 56, i8* nonnull %8) #24
  call void @_ZN6google8protobuf8internal10LogMessageC1ENS0_8LogLevelEPKci(%"class.google::protobuf::internal::LogMessage"* noundef nonnull align 8 dereferenceable(56) %3, i32 noundef 3, i8* noundef getelementptr inbounds ([24 x i8], [24 x i8]* @.str.5, i64 0, i64 0), i32 noundef 373)
  %9 = invoke noundef nonnull align 8 dereferenceable(56) %"class.google::protobuf::internal::LogMessage"* @_ZN6google8protobuf8internal10LogMessagelsEPKc(%"class.google::protobuf::internal::LogMessage"* noundef nonnull align 8 dereferenceable(56) %3, i8* noundef getelementptr inbounds ([34 x i8], [34 x i8]* @.str.6, i64 0, i64 0))
          to label %10 unwind label %64

10:                                               ; preds = %7
  invoke void @_ZN6google8protobuf8internal11LogFinisheraSERNS1_10LogMessageE(%"class.google::protobuf::internal::LogFinisher"* noundef nonnull align 1 dereferenceable(1) %4, %"class.google::protobuf::internal::LogMessage"* noundef nonnull align 8 dereferenceable(56) %9)
          to label %12 unwind label %66

11:                                               ; preds = %2
  call void @llvm.lifetime.end.p0i8(i64 1, i8* nonnull %6) #24
  br label %13

12:                                               ; preds = %10
  call void @llvm.lifetime.end.p0i8(i64 1, i8* nonnull %6) #24
  call void @_ZN6google8protobuf8internal10LogMessageD1Ev(%"class.google::protobuf::internal::LogMessage"* noundef nonnull align 8 dereferenceable(56) %3) #24
  call void @llvm.lifetime.end.p0i8(i64 56, i8* nonnull %8) #24
  br label %13

13:                                               ; preds = %11, %12
  %14 = getelementptr inbounds %"class.tutorial::Person_PhoneNumber", %"class.tutorial::Person_PhoneNumber"* %0, i64 0, i32 0, i32 0, i32 1
  %15 = getelementptr inbounds %"class.tutorial::Person_PhoneNumber", %"class.tutorial::Person_PhoneNumber"* %1, i64 0, i32 0, i32 0, i32 1, i32 0
  %16 = load i8*, i8** %15, align 8, !tbaa !5
  %17 = ptrtoint i8* %16 to i64
  %18 = and i64 %17, 1
  %19 = icmp eq i64 %18, 0
  br i1 %19, label %37, label %20

20:                                               ; preds = %13
  %21 = and i64 %17, -2
  %22 = inttoptr i64 %21 to %"struct.google::protobuf::internal::InternalMetadata::Container"*
  %23 = getelementptr inbounds %"struct.google::protobuf::internal::InternalMetadata::Container", %"struct.google::protobuf::internal::InternalMetadata::Container"* %22, i64 0, i32 1
  %24 = getelementptr inbounds %"class.google::protobuf::internal::InternalMetadata", %"class.google::protobuf::internal::InternalMetadata"* %14, i64 0, i32 0
  %25 = load i8*, i8** %24, align 8, !tbaa !5
  %26 = ptrtoint i8* %25 to i64
  %27 = and i64 %26, 1
  %28 = icmp eq i64 %27, 0
  br i1 %28, label %33, label %29, !prof !41

29:                                               ; preds = %20
  %30 = and i64 %26, -2
  %31 = inttoptr i64 %30 to %"struct.google::protobuf::internal::InternalMetadata::Container"*
  %32 = getelementptr inbounds %"struct.google::protobuf::internal::InternalMetadata::Container", %"struct.google::protobuf::internal::InternalMetadata::Container"* %31, i64 0, i32 1
  br label %35

33:                                               ; preds = %20
  %34 = call noundef %"class.google::protobuf::UnknownFieldSet"* @_ZN6google8protobuf8internal16InternalMetadata27mutable_unknown_fields_slowINS0_15UnknownFieldSetEEEPT_v(%"class.google::protobuf::internal::InternalMetadata"* noundef nonnull align 8 dereferenceable(8) %14)
  br label %35

35:                                               ; preds = %33, %29
  %36 = phi %"class.google::protobuf::UnknownFieldSet"* [ %32, %29 ], [ %34, %33 ]
  call void @_ZN6google8protobuf15UnknownFieldSet9MergeFromERKS1_(%"class.google::protobuf::UnknownFieldSet"* noundef nonnull align 8 dereferenceable(24) %36, %"class.google::protobuf::UnknownFieldSet"* noundef nonnull align 8 dereferenceable(24) %23)
  br label %37

37:                                               ; preds = %13, %35
  %38 = getelementptr inbounds %"class.tutorial::Person_PhoneNumber", %"class.tutorial::Person_PhoneNumber"* %1, i64 0, i32 1, i32 0
  %39 = load %"class.std::__cxx11::basic_string"*, %"class.std::__cxx11::basic_string"** %38, align 8, !tbaa !18
  %40 = getelementptr inbounds %"class.std::__cxx11::basic_string", %"class.std::__cxx11::basic_string"* %39, i64 0, i32 1
  %41 = load i64, i64* %40, align 8, !tbaa !28
  %42 = icmp eq i64 %41, 0
  br i1 %42, label %70, label %43

43:                                               ; preds = %37
  %44 = getelementptr inbounds %"class.tutorial::Person_PhoneNumber", %"class.tutorial::Person_PhoneNumber"* %0, i64 0, i32 1
  %45 = getelementptr inbounds %"class.tutorial::Person_PhoneNumber", %"class.tutorial::Person_PhoneNumber"* %0, i64 0, i32 0, i32 0, i32 1, i32 0
  %46 = load i8*, i8** %45, align 8, !tbaa !5
  %47 = ptrtoint i8* %46 to i64
  %48 = and i64 %47, 1
  %49 = icmp eq i64 %48, 0
  %50 = and i64 %47, -2
  br i1 %49, label %55, label %51, !prof !12

51:                                               ; preds = %43
  %52 = inttoptr i64 %50 to %"struct.google::protobuf::internal::InternalMetadata::ContainerBase"*
  %53 = getelementptr inbounds %"struct.google::protobuf::internal::InternalMetadata::ContainerBase", %"struct.google::protobuf::internal::InternalMetadata::ContainerBase"* %52, i64 0, i32 0
  %54 = load %"class.google::protobuf::Arena"*, %"class.google::protobuf::Arena"** %53, align 8, !tbaa !32
  br label %57

55:                                               ; preds = %43
  %56 = inttoptr i64 %50 to %"class.google::protobuf::Arena"*
  br label %57

57:                                               ; preds = %55, %51
  %58 = phi %"class.google::protobuf::Arena"* [ %54, %51 ], [ %56, %55 ]
  %59 = getelementptr inbounds %"struct.google::protobuf::internal::ArenaStringPtr", %"struct.google::protobuf::internal::ArenaStringPtr"* %44, i64 0, i32 0
  %60 = load %"class.std::__cxx11::basic_string"*, %"class.std::__cxx11::basic_string"** %59, align 8, !tbaa !18
  %61 = icmp eq %"class.std::__cxx11::basic_string"* %60, bitcast (%"class.google::protobuf::internal::ExplicitlyConstructed.20"* @_ZN6google8protobuf8internal26fixed_address_empty_stringB5cxx11E to %"class.std::__cxx11::basic_string"*)
  br i1 %61, label %62, label %63

62:                                               ; preds = %57
  call void @_ZN6google8protobuf8internal14ArenaStringPtr14CreateInstanceEPNS0_5ArenaEPKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE(%"struct.google::protobuf::internal::ArenaStringPtr"* noundef nonnull align 8 dereferenceable(8) %44, %"class.google::protobuf::Arena"* noundef %58, %"class.std::__cxx11::basic_string"* noundef nonnull %39)
  br label %70

63:                                               ; preds = %57
  call void @_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE9_M_assignERKS4_(%"class.std::__cxx11::basic_string"* noundef nonnull align 8 dereferenceable(32) %60, %"class.std::__cxx11::basic_string"* noundef nonnull align 8 dereferenceable(32) %39)
  br label %70

64:                                               ; preds = %7
  %65 = landingpad { i8*, i32 }
          cleanup
  br label %68

66:                                               ; preds = %10
  %67 = landingpad { i8*, i32 }
          cleanup
  call void @llvm.lifetime.end.p0i8(i64 1, i8* nonnull %6) #24
  br label %68

68:                                               ; preds = %64, %66
  %69 = phi { i8*, i32 } [ %67, %66 ], [ %65, %64 ]
  call void @_ZN6google8protobuf8internal10LogMessageD1Ev(%"class.google::protobuf::internal::LogMessage"* noundef nonnull align 8 dereferenceable(56) %3) #24
  call void @llvm.lifetime.end.p0i8(i64 56, i8* nonnull %8) #24
  resume { i8*, i32 } %69

70:                                               ; preds = %63, %62, %37
  %71 = getelementptr inbounds %"class.tutorial::Person_PhoneNumber", %"class.tutorial::Person_PhoneNumber"* %1, i64 0, i32 2
  %72 = load i32, i32* %71, align 8, !tbaa !25
  %73 = icmp eq i32 %72, 0
  br i1 %73, label %76, label %74

74:                                               ; preds = %70
  %75 = getelementptr inbounds %"class.tutorial::Person_PhoneNumber", %"class.tutorial::Person_PhoneNumber"* %0, i64 0, i32 2
  store i32 %72, i32* %75, align 8, !tbaa !25
  br label %76

76:                                               ; preds = %74, %70
  ret void
}

; Function Attrs: uwtable
define dso_local void @_ZN8tutorial18Person_PhoneNumber8CopyFromERKN6google8protobuf7MessageE(%"class.tutorial::Person_PhoneNumber"* noundef nonnull align 8 dereferenceable(32) %0, %"class.google::protobuf::Message"* noundef nonnull align 8 dereferenceable(16) %1) unnamed_addr #3 align 2 personality i32 (...)* @__gxx_personality_v0 {
  %3 = getelementptr inbounds %"class.tutorial::Person_PhoneNumber", %"class.tutorial::Person_PhoneNumber"* %0, i64 0, i32 0
  %4 = icmp eq %"class.google::protobuf::Message"* %3, %1
  br i1 %4, label %31, label %5

5:                                                ; preds = %2
  %6 = getelementptr inbounds %"class.tutorial::Person_PhoneNumber", %"class.tutorial::Person_PhoneNumber"* %0, i64 0, i32 1, i32 0
  %7 = load %"class.std::__cxx11::basic_string"*, %"class.std::__cxx11::basic_string"** %6, align 8, !tbaa !18
  %8 = icmp eq %"class.std::__cxx11::basic_string"* %7, bitcast (%"class.google::protobuf::internal::ExplicitlyConstructed.20"* @_ZN6google8protobuf8internal26fixed_address_empty_stringB5cxx11E to %"class.std::__cxx11::basic_string"*)
  br i1 %8, label %13, label %9

9:                                                ; preds = %5
  %10 = getelementptr inbounds %"class.std::__cxx11::basic_string", %"class.std::__cxx11::basic_string"* %7, i64 0, i32 1
  store i64 0, i64* %10, align 8, !tbaa !28
  %11 = getelementptr inbounds %"class.std::__cxx11::basic_string", %"class.std::__cxx11::basic_string"* %7, i64 0, i32 0, i32 0
  %12 = load i8*, i8** %11, align 8, !tbaa !34
  store i8 0, i8* %12, align 1, !tbaa !37
  br label %13

13:                                               ; preds = %9, %5
  %14 = getelementptr inbounds %"class.tutorial::Person_PhoneNumber", %"class.tutorial::Person_PhoneNumber"* %0, i64 0, i32 2
  store i32 0, i32* %14, align 8, !tbaa !25
  %15 = getelementptr inbounds %"class.tutorial::Person_PhoneNumber", %"class.tutorial::Person_PhoneNumber"* %0, i64 0, i32 0, i32 0, i32 1, i32 0
  %16 = load i8*, i8** %15, align 8, !tbaa !5
  %17 = ptrtoint i8* %16 to i64
  %18 = and i64 %17, 1
  %19 = icmp eq i64 %18, 0
  br i1 %19, label %30, label %20

20:                                               ; preds = %13
  %21 = and i64 %17, -2
  %22 = inttoptr i64 %21 to %"struct.google::protobuf::internal::InternalMetadata::Container"*
  %23 = getelementptr inbounds %"struct.google::protobuf::internal::InternalMetadata::Container", %"struct.google::protobuf::internal::InternalMetadata::Container"* %22, i64 0, i32 1
  %24 = getelementptr inbounds %"class.google::protobuf::UnknownFieldSet", %"class.google::protobuf::UnknownFieldSet"* %23, i64 0, i32 0, i32 0, i32 0, i32 0, i32 0
  %25 = load %"class.google::protobuf::UnknownField"*, %"class.google::protobuf::UnknownField"** %24, align 8, !tbaa !27
  %26 = getelementptr inbounds %"struct.google::protobuf::internal::InternalMetadata::Container", %"struct.google::protobuf::internal::InternalMetadata::Container"* %22, i64 0, i32 1, i32 0, i32 0, i32 0, i32 0, i32 1
  %27 = load %"class.google::protobuf::UnknownField"*, %"class.google::protobuf::UnknownField"** %26, align 8, !tbaa !27
  %28 = icmp eq %"class.google::protobuf::UnknownField"* %25, %27
  br i1 %28, label %30, label %29

29:                                               ; preds = %20
  tail call void @_ZN6google8protobuf15UnknownFieldSet13ClearFallbackEv(%"class.google::protobuf::UnknownFieldSet"* noundef nonnull align 8 dereferenceable(24) %23)
  br label %30

30:                                               ; preds = %13, %20, %29
  tail call void @_ZN8tutorial18Person_PhoneNumber9MergeFromERKN6google8protobuf7MessageE(%"class.tutorial::Person_PhoneNumber"* noundef nonnull align 8 dereferenceable(32) %0, %"class.google::protobuf::Message"* noundef nonnull align 8 dereferenceable(16) %1)
  br label %31

31:                                               ; preds = %2, %30
  ret void
}

; Function Attrs: uwtable
define dso_local void @_ZN8tutorial18Person_PhoneNumber8CopyFromERKS0_(%"class.tutorial::Person_PhoneNumber"* noundef nonnull align 8 dereferenceable(32) %0, %"class.tutorial::Person_PhoneNumber"* noundef nonnull align 8 dereferenceable(32) %1) local_unnamed_addr #3 align 2 personality i32 (...)* @__gxx_personality_v0 {
  %3 = icmp eq %"class.tutorial::Person_PhoneNumber"* %1, %0
  br i1 %3, label %30, label %4

4:                                                ; preds = %2
  %5 = getelementptr inbounds %"class.tutorial::Person_PhoneNumber", %"class.tutorial::Person_PhoneNumber"* %0, i64 0, i32 1, i32 0
  %6 = load %"class.std::__cxx11::basic_string"*, %"class.std::__cxx11::basic_string"** %5, align 8, !tbaa !18
  %7 = icmp eq %"class.std::__cxx11::basic_string"* %6, bitcast (%"class.google::protobuf::internal::ExplicitlyConstructed.20"* @_ZN6google8protobuf8internal26fixed_address_empty_stringB5cxx11E to %"class.std::__cxx11::basic_string"*)
  br i1 %7, label %12, label %8

8:                                                ; preds = %4
  %9 = getelementptr inbounds %"class.std::__cxx11::basic_string", %"class.std::__cxx11::basic_string"* %6, i64 0, i32 1
  store i64 0, i64* %9, align 8, !tbaa !28
  %10 = getelementptr inbounds %"class.std::__cxx11::basic_string", %"class.std::__cxx11::basic_string"* %6, i64 0, i32 0, i32 0
  %11 = load i8*, i8** %10, align 8, !tbaa !34
  store i8 0, i8* %11, align 1, !tbaa !37
  br label %12

12:                                               ; preds = %8, %4
  %13 = getelementptr inbounds %"class.tutorial::Person_PhoneNumber", %"class.tutorial::Person_PhoneNumber"* %0, i64 0, i32 2
  store i32 0, i32* %13, align 8, !tbaa !25
  %14 = getelementptr inbounds %"class.tutorial::Person_PhoneNumber", %"class.tutorial::Person_PhoneNumber"* %0, i64 0, i32 0, i32 0, i32 1, i32 0
  %15 = load i8*, i8** %14, align 8, !tbaa !5
  %16 = ptrtoint i8* %15 to i64
  %17 = and i64 %16, 1
  %18 = icmp eq i64 %17, 0
  br i1 %18, label %29, label %19

19:                                               ; preds = %12
  %20 = and i64 %16, -2
  %21 = inttoptr i64 %20 to %"struct.google::protobuf::internal::InternalMetadata::Container"*
  %22 = getelementptr inbounds %"struct.google::protobuf::internal::InternalMetadata::Container", %"struct.google::protobuf::internal::InternalMetadata::Container"* %21, i64 0, i32 1
  %23 = getelementptr inbounds %"class.google::protobuf::UnknownFieldSet", %"class.google::protobuf::UnknownFieldSet"* %22, i64 0, i32 0, i32 0, i32 0, i32 0, i32 0
  %24 = load %"class.google::protobuf::UnknownField"*, %"class.google::protobuf::UnknownField"** %23, align 8, !tbaa !27
  %25 = getelementptr inbounds %"struct.google::protobuf::internal::InternalMetadata::Container", %"struct.google::protobuf::internal::InternalMetadata::Container"* %21, i64 0, i32 1, i32 0, i32 0, i32 0, i32 0, i32 1
  %26 = load %"class.google::protobuf::UnknownField"*, %"class.google::protobuf::UnknownField"** %25, align 8, !tbaa !27
  %27 = icmp eq %"class.google::protobuf::UnknownField"* %24, %26
  br i1 %27, label %29, label %28

28:                                               ; preds = %19
  tail call void @_ZN6google8protobuf15UnknownFieldSet13ClearFallbackEv(%"class.google::protobuf::UnknownFieldSet"* noundef nonnull align 8 dereferenceable(24) %22)
  br label %29

29:                                               ; preds = %12, %19, %28
  tail call void @_ZN8tutorial18Person_PhoneNumber9MergeFromERKS0_(%"class.tutorial::Person_PhoneNumber"* noundef nonnull align 8 dereferenceable(32) %0, %"class.tutorial::Person_PhoneNumber"* noundef nonnull align 8 dereferenceable(32) %1)
  br label %30

30:                                               ; preds = %2, %29
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind readnone uwtable willreturn
define dso_local noundef zeroext i1 @_ZNK8tutorial18Person_PhoneNumber13IsInitializedEv(%"class.tutorial::Person_PhoneNumber"* nocapture nonnull readnone align 8 %0) unnamed_addr #5 align 2 {
  ret i1 true
}

; Function Attrs: uwtable
define dso_local void @_ZN8tutorial18Person_PhoneNumber12InternalSwapEPS0_(%"class.tutorial::Person_PhoneNumber"* noundef nonnull align 8 dereferenceable(32) %0, %"class.tutorial::Person_PhoneNumber"* noundef %1) local_unnamed_addr #3 align 2 personality i32 (...)* @__gxx_personality_v0 {
  %3 = getelementptr inbounds %"class.tutorial::Person_PhoneNumber", %"class.tutorial::Person_PhoneNumber"* %0, i64 0, i32 0, i32 0, i32 1
  %4 = getelementptr inbounds %"class.tutorial::Person_PhoneNumber", %"class.tutorial::Person_PhoneNumber"* %1, i64 0, i32 0, i32 0, i32 1
  %5 = getelementptr inbounds %"class.google::protobuf::internal::InternalMetadata", %"class.google::protobuf::internal::InternalMetadata"* %3, i64 0, i32 0
  %6 = load i8*, i8** %5, align 8, !tbaa !5
  %7 = ptrtoint i8* %6 to i64
  %8 = and i64 %7, 1
  %9 = icmp eq i64 %8, 0
  %10 = getelementptr inbounds %"class.google::protobuf::internal::InternalMetadata", %"class.google::protobuf::internal::InternalMetadata"* %4, i64 0, i32 0
  %11 = load i8*, i8** %10, align 8, !tbaa !5
  %12 = ptrtoint i8* %11 to i64
  %13 = and i64 %12, 1
  %14 = icmp eq i64 %13, 0
  %15 = select i1 %9, i1 %14, i1 false
  br i1 %15, label %53, label %16

16:                                               ; preds = %2
  br i1 %14, label %21, label %17, !prof !41

17:                                               ; preds = %16
  %18 = and i64 %12, -2
  %19 = inttoptr i64 %18 to %"struct.google::protobuf::internal::InternalMetadata::Container"*
  %20 = getelementptr inbounds %"struct.google::protobuf::internal::InternalMetadata::Container", %"struct.google::protobuf::internal::InternalMetadata::Container"* %19, i64 0, i32 1
  br label %25

21:                                               ; preds = %16
  %22 = tail call noundef %"class.google::protobuf::UnknownFieldSet"* @_ZN6google8protobuf8internal16InternalMetadata27mutable_unknown_fields_slowINS0_15UnknownFieldSetEEEPT_v(%"class.google::protobuf::internal::InternalMetadata"* noundef nonnull align 8 dereferenceable(8) %4)
  %23 = load i8*, i8** %5, align 8, !tbaa !5
  %24 = ptrtoint i8* %23 to i64
  br label %25

25:                                               ; preds = %21, %17
  %26 = phi i8* [ %6, %17 ], [ %23, %21 ]
  %27 = phi i64 [ %7, %17 ], [ %24, %21 ]
  %28 = phi %"class.google::protobuf::UnknownFieldSet"* [ %20, %17 ], [ %22, %21 ]
  %29 = and i64 %27, 1
  %30 = icmp eq i64 %29, 0
  br i1 %30, label %35, label %31, !prof !41

31:                                               ; preds = %25
  %32 = and i64 %27, -2
  %33 = inttoptr i64 %32 to %"struct.google::protobuf::internal::InternalMetadata::Container"*
  %34 = getelementptr inbounds %"struct.google::protobuf::internal::InternalMetadata::Container", %"struct.google::protobuf::internal::InternalMetadata::Container"* %33, i64 0, i32 1
  br label %39

35:                                               ; preds = %25
  %36 = tail call noundef %"class.google::protobuf::UnknownFieldSet"* @_ZN6google8protobuf8internal16InternalMetadata27mutable_unknown_fields_slowINS0_15UnknownFieldSetEEEPT_v(%"class.google::protobuf::internal::InternalMetadata"* noundef nonnull align 8 dereferenceable(8) %3)
  %37 = getelementptr inbounds %"class.tutorial::Person_PhoneNumber", %"class.tutorial::Person_PhoneNumber"* %0, i64 0, i32 0, i32 0, i32 1, i32 0
  %38 = load i8*, i8** %37, align 8, !tbaa !5
  br label %39

39:                                               ; preds = %35, %31
  %40 = phi i8* [ %26, %31 ], [ %38, %35 ]
  %41 = phi %"class.google::protobuf::UnknownFieldSet"* [ %34, %31 ], [ %36, %35 ]
  %42 = bitcast %"class.google::protobuf::UnknownFieldSet"* %41 to <2 x %"class.google::protobuf::UnknownField"*>*
  %43 = load <2 x %"class.google::protobuf::UnknownField"*>, <2 x %"class.google::protobuf::UnknownField"*>* %42, align 8, !tbaa !27
  %44 = getelementptr inbounds %"class.google::protobuf::UnknownFieldSet", %"class.google::protobuf::UnknownFieldSet"* %41, i64 0, i32 0, i32 0, i32 0, i32 0, i32 2
  %45 = load %"class.google::protobuf::UnknownField"*, %"class.google::protobuf::UnknownField"** %44, align 8, !tbaa !50
  %46 = bitcast %"class.google::protobuf::UnknownFieldSet"* %28 to <2 x %"class.google::protobuf::UnknownField"*>*
  %47 = load <2 x %"class.google::protobuf::UnknownField"*>, <2 x %"class.google::protobuf::UnknownField"*>* %46, align 8, !tbaa !27
  %48 = bitcast %"class.google::protobuf::UnknownFieldSet"* %41 to <2 x %"class.google::protobuf::UnknownField"*>*
  store <2 x %"class.google::protobuf::UnknownField"*> %47, <2 x %"class.google::protobuf::UnknownField"*>* %48, align 8, !tbaa !27
  %49 = getelementptr inbounds %"class.google::protobuf::UnknownFieldSet", %"class.google::protobuf::UnknownFieldSet"* %28, i64 0, i32 0, i32 0, i32 0, i32 0, i32 2
  %50 = load %"class.google::protobuf::UnknownField"*, %"class.google::protobuf::UnknownField"** %49, align 8, !tbaa !50
  store %"class.google::protobuf::UnknownField"* %50, %"class.google::protobuf::UnknownField"** %44, align 8, !tbaa !50
  %51 = bitcast %"class.google::protobuf::UnknownFieldSet"* %28 to <2 x %"class.google::protobuf::UnknownField"*>*
  store <2 x %"class.google::protobuf::UnknownField"*> %43, <2 x %"class.google::protobuf::UnknownField"*>* %51, align 8, !tbaa !27
  store %"class.google::protobuf::UnknownField"* %45, %"class.google::protobuf::UnknownField"** %49, align 8, !tbaa !50
  %52 = ptrtoint i8* %40 to i64
  br label %53

53:                                               ; preds = %2, %39
  %54 = phi i64 [ %7, %2 ], [ %52, %39 ]
  %55 = getelementptr inbounds %"class.tutorial::Person_PhoneNumber", %"class.tutorial::Person_PhoneNumber"* %0, i64 0, i32 1
  %56 = getelementptr inbounds %"class.tutorial::Person_PhoneNumber", %"class.tutorial::Person_PhoneNumber"* %1, i64 0, i32 1
  %57 = and i64 %54, 1
  %58 = icmp eq i64 %57, 0
  %59 = and i64 %54, -2
  br i1 %58, label %64, label %60, !prof !12

60:                                               ; preds = %53
  %61 = inttoptr i64 %59 to %"struct.google::protobuf::internal::InternalMetadata::ContainerBase"*
  %62 = getelementptr inbounds %"struct.google::protobuf::internal::InternalMetadata::ContainerBase", %"struct.google::protobuf::internal::InternalMetadata::ContainerBase"* %61, i64 0, i32 0
  %63 = load %"class.google::protobuf::Arena"*, %"class.google::protobuf::Arena"** %62, align 8, !tbaa !32
  br label %66

64:                                               ; preds = %53
  %65 = inttoptr i64 %59 to %"class.google::protobuf::Arena"*
  br label %66

66:                                               ; preds = %60, %64
  %67 = phi %"class.google::protobuf::Arena"* [ %63, %60 ], [ %65, %64 ]
  %68 = getelementptr inbounds %"struct.google::protobuf::internal::ArenaStringPtr", %"struct.google::protobuf::internal::ArenaStringPtr"* %55, i64 0, i32 0
  %69 = load %"class.std::__cxx11::basic_string"*, %"class.std::__cxx11::basic_string"** %68, align 8, !tbaa !18
  %70 = icmp eq %"class.std::__cxx11::basic_string"* %69, bitcast (%"class.google::protobuf::internal::ExplicitlyConstructed.20"* @_ZN6google8protobuf8internal26fixed_address_empty_stringB5cxx11E to %"class.std::__cxx11::basic_string"*)
  br i1 %70, label %71, label %77

71:                                               ; preds = %66
  %72 = getelementptr inbounds %"struct.google::protobuf::internal::ArenaStringPtr", %"struct.google::protobuf::internal::ArenaStringPtr"* %56, i64 0, i32 0
  %73 = load %"class.std::__cxx11::basic_string"*, %"class.std::__cxx11::basic_string"** %72, align 8, !tbaa !18
  %74 = icmp eq %"class.std::__cxx11::basic_string"* %73, bitcast (%"class.google::protobuf::internal::ExplicitlyConstructed.20"* @_ZN6google8protobuf8internal26fixed_address_empty_stringB5cxx11E to %"class.std::__cxx11::basic_string"*)
  br i1 %74, label %86, label %75

75:                                               ; preds = %71
  tail call void @_ZN6google8protobuf8internal14ArenaStringPtr14CreateInstanceEPNS0_5ArenaEPKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE(%"struct.google::protobuf::internal::ArenaStringPtr"* noundef nonnull align 8 dereferenceable(8) %55, %"class.google::protobuf::Arena"* noundef %67, %"class.std::__cxx11::basic_string"* noundef bitcast (%"class.google::protobuf::internal::ExplicitlyConstructed.20"* @_ZN6google8protobuf8internal26fixed_address_empty_stringB5cxx11E to %"class.std::__cxx11::basic_string"*))
  %76 = load %"class.std::__cxx11::basic_string"*, %"class.std::__cxx11::basic_string"** %68, align 8, !tbaa !18
  br label %77

77:                                               ; preds = %75, %66
  %78 = phi %"class.std::__cxx11::basic_string"* [ %76, %75 ], [ %69, %66 ]
  %79 = getelementptr inbounds %"struct.google::protobuf::internal::ArenaStringPtr", %"struct.google::protobuf::internal::ArenaStringPtr"* %56, i64 0, i32 0
  %80 = load %"class.std::__cxx11::basic_string"*, %"class.std::__cxx11::basic_string"** %79, align 8, !tbaa !18
  %81 = icmp eq %"class.std::__cxx11::basic_string"* %80, bitcast (%"class.google::protobuf::internal::ExplicitlyConstructed.20"* @_ZN6google8protobuf8internal26fixed_address_empty_stringB5cxx11E to %"class.std::__cxx11::basic_string"*)
  br i1 %81, label %82, label %84

82:                                               ; preds = %77
  tail call void @_ZN6google8protobuf8internal14ArenaStringPtr14CreateInstanceEPNS0_5ArenaEPKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE(%"struct.google::protobuf::internal::ArenaStringPtr"* noundef nonnull align 8 dereferenceable(8) %56, %"class.google::protobuf::Arena"* noundef %67, %"class.std::__cxx11::basic_string"* noundef bitcast (%"class.google::protobuf::internal::ExplicitlyConstructed.20"* @_ZN6google8protobuf8internal26fixed_address_empty_stringB5cxx11E to %"class.std::__cxx11::basic_string"*))
  %83 = load %"class.std::__cxx11::basic_string"*, %"class.std::__cxx11::basic_string"** %79, align 8, !tbaa !18
  br label %84

84:                                               ; preds = %82, %77
  %85 = phi %"class.std::__cxx11::basic_string"* [ %83, %82 ], [ %80, %77 ]
  tail call void @_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE4swapERS4_(%"class.std::__cxx11::basic_string"* noundef nonnull align 8 dereferenceable(32) %78, %"class.std::__cxx11::basic_string"* noundef nonnull align 8 dereferenceable(32) %85) #24
  br label %86

86:                                               ; preds = %71, %84
  %87 = getelementptr inbounds %"class.tutorial::Person_PhoneNumber", %"class.tutorial::Person_PhoneNumber"* %0, i64 0, i32 2
  %88 = getelementptr inbounds %"class.tutorial::Person_PhoneNumber", %"class.tutorial::Person_PhoneNumber"* %1, i64 0, i32 2
  %89 = load i32, i32* %87, align 8, !tbaa !51
  %90 = load i32, i32* %88, align 4, !tbaa !51
  store i32 %90, i32* %87, align 8, !tbaa !51
  store i32 %89, i32* %88, align 4, !tbaa !51
  ret void
}

; Function Attrs: mustprogress uwtable
define dso_local { %"class.google::protobuf::Descriptor"*, %"class.google::protobuf::Reflection"* } @_ZNK8tutorial18Person_PhoneNumber11GetMetadataEv(%"class.tutorial::Person_PhoneNumber"* nocapture nonnull readnone align 8 %0) unnamed_addr #4 align 2 {
  tail call void @_ZN6google8protobuf8internal17AssignDescriptorsEPKNS1_15DescriptorTableEb(%"struct.google::protobuf::internal::DescriptorTable"* noundef nonnull @descriptor_table_addressbook_2eproto, i1 noundef zeroext false)
  %2 = load %"struct.google::protobuf::Metadata"*, %"struct.google::protobuf::Metadata"** getelementptr inbounds (%"struct.google::protobuf::internal::DescriptorTable", %"struct.google::protobuf::internal::DescriptorTable"* @descriptor_table_addressbook_2eproto, i64 0, i32 13), align 8, !tbaa !52
  %3 = getelementptr inbounds %"struct.google::protobuf::Metadata", %"struct.google::protobuf::Metadata"* %2, i64 0, i32 0
  %4 = load %"class.google::protobuf::Descriptor"*, %"class.google::protobuf::Descriptor"** %3, align 8, !tbaa.struct !54
  %5 = getelementptr inbounds %"struct.google::protobuf::Metadata", %"struct.google::protobuf::Metadata"* %2, i64 0, i32 1
  %6 = load %"class.google::protobuf::Reflection"*, %"class.google::protobuf::Reflection"** %5, align 8, !tbaa.struct !55
  %7 = insertvalue { %"class.google::protobuf::Descriptor"*, %"class.google::protobuf::Reflection"* } poison, %"class.google::protobuf::Descriptor"* %4, 0
  %8 = insertvalue { %"class.google::protobuf::Descriptor"*, %"class.google::protobuf::Reflection"* } %7, %"class.google::protobuf::Reflection"* %6, 1
  ret { %"class.google::protobuf::Descriptor"*, %"class.google::protobuf::Reflection"* } %8
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind uwtable willreturn writeonly
define dso_local void @_ZN8tutorial6Person21InitAsDefaultInstanceEv() local_unnamed_addr #12 align 2 {
  store %"class.google::protobuf::Timestamp"* bitcast (%"class.google::protobuf::TimestampDefaultTypeInternal"* @_ZN6google8protobuf28_Timestamp_default_instance_E to %"class.google::protobuf::Timestamp"*), %"class.google::protobuf::Timestamp"** bitcast (i8* getelementptr inbounds (%"class.tutorial::PersonDefaultTypeInternal", %"class.tutorial::PersonDefaultTypeInternal"* @_ZN8tutorial25_Person_default_instance_E, i64 0, i32 0, i32 0, i32 1, i64 48) to %"class.google::protobuf::Timestamp"**), align 8, !tbaa !20
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind readonly uwtable willreturn
define dso_local noundef nonnull align 8 dereferenceable(32) %"class.google::protobuf::Timestamp"* @_ZN8tutorial6Person9_Internal12last_updatedEPKS0_(%"class.tutorial::Person"* nocapture noundef readonly %0) local_unnamed_addr #13 align 2 {
  %2 = getelementptr inbounds %"class.tutorial::Person", %"class.tutorial::Person"* %0, i64 0, i32 4
  %3 = load %"class.google::protobuf::Timestamp"*, %"class.google::protobuf::Timestamp"** %2, align 8, !tbaa !20
  ret %"class.google::protobuf::Timestamp"* %3
}

; Function Attrs: mustprogress nounwind uwtable
define dso_local void @_ZN8tutorial6Person18clear_last_updatedEv(%"class.tutorial::Person"* nocapture noundef nonnull align 8 dereferenceable(72) %0) local_unnamed_addr #14 align 2 {
  %2 = getelementptr inbounds %"class.tutorial::Person", %"class.tutorial::Person"* %0, i64 0, i32 0, i32 0, i32 1, i32 0
  %3 = load i8*, i8** %2, align 8, !tbaa !5
  %4 = ptrtoint i8* %3 to i64
  %5 = and i64 %4, 1
  %6 = icmp eq i64 %5, 0
  %7 = and i64 %4, -2
  br i1 %6, label %12, label %8, !prof !12

8:                                                ; preds = %1
  %9 = inttoptr i64 %7 to %"struct.google::protobuf::internal::InternalMetadata::ContainerBase"*
  %10 = getelementptr inbounds %"struct.google::protobuf::internal::InternalMetadata::ContainerBase", %"struct.google::protobuf::internal::InternalMetadata::ContainerBase"* %9, i64 0, i32 0
  %11 = load %"class.google::protobuf::Arena"*, %"class.google::protobuf::Arena"** %10, align 8, !tbaa !32
  br label %14

12:                                               ; preds = %1
  %13 = inttoptr i64 %7 to %"class.google::protobuf::Arena"*
  br label %14

14:                                               ; preds = %8, %12
  %15 = phi %"class.google::protobuf::Arena"* [ %11, %8 ], [ %13, %12 ]
  %16 = icmp eq %"class.google::protobuf::Arena"* %15, null
  br i1 %16, label %17, label %23

17:                                               ; preds = %14
  %18 = getelementptr inbounds %"class.tutorial::Person", %"class.tutorial::Person"* %0, i64 0, i32 4
  %19 = load %"class.google::protobuf::Timestamp"*, %"class.google::protobuf::Timestamp"** %18, align 8, !tbaa !20
  %20 = icmp eq %"class.google::protobuf::Timestamp"* %19, null
  br i1 %20, label %23, label %21

21:                                               ; preds = %17
  tail call void @_ZN6google8protobuf9TimestampD1Ev(%"class.google::protobuf::Timestamp"* noundef nonnull align 8 dereferenceable(32) %19) #24
  %22 = bitcast %"class.google::protobuf::Timestamp"* %19 to i8*
  tail call void @_ZdlPv(i8* noundef %22) #26
  br label %23

23:                                               ; preds = %21, %17, %14
  %24 = getelementptr inbounds %"class.tutorial::Person", %"class.tutorial::Person"* %0, i64 0, i32 4
  store %"class.google::protobuf::Timestamp"* null, %"class.google::protobuf::Timestamp"** %24, align 8, !tbaa !20
  ret void
}

; Function Attrs: nounwind
declare void @_ZN6google8protobuf9TimestampD1Ev(%"class.google::protobuf::Timestamp"* noundef nonnull align 8 dereferenceable(32)) unnamed_addr #1

; Function Attrs: uwtable
define dso_local void @_ZN8tutorial6PersonC2EPN6google8protobuf5ArenaE(%"class.tutorial::Person"* noundef nonnull align 8 dereferenceable(72) %0, %"class.google::protobuf::Arena"* noundef %1) unnamed_addr #3 align 2 personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*) {
  %3 = getelementptr inbounds %"class.tutorial::Person", %"class.tutorial::Person"* %0, i64 0, i32 0, i32 0, i32 1
  %4 = bitcast %"class.google::protobuf::internal::InternalMetadata"* %3 to %"class.google::protobuf::Arena"**
  store %"class.google::protobuf::Arena"* %1, %"class.google::protobuf::Arena"** %4, align 8, !tbaa !5
  %5 = getelementptr inbounds %"class.tutorial::Person", %"class.tutorial::Person"* %0, i64 0, i32 0, i32 0, i32 0
  store i32 (...)** bitcast (i8** getelementptr inbounds ({ [22 x i8*] }, { [22 x i8*] }* @_ZTVN8tutorial6PersonE, i64 0, inrange i32 0, i64 2) to i32 (...)**), i32 (...)*** %5, align 8, !tbaa !10
  %6 = getelementptr inbounds %"class.tutorial::Person", %"class.tutorial::Person"* %0, i64 0, i32 1
  %7 = getelementptr inbounds %"class.google::protobuf::RepeatedPtrField", %"class.google::protobuf::RepeatedPtrField"* %6, i64 0, i32 0, i32 0
  store %"class.google::protobuf::Arena"* %1, %"class.google::protobuf::Arena"** %7, align 8, !tbaa !16
  %8 = getelementptr inbounds %"class.tutorial::Person", %"class.tutorial::Person"* %0, i64 0, i32 1, i32 0, i32 1
  %9 = bitcast i32* %8 to i8*
  tail call void @llvm.memset.p0i8.i64(i8* noundef nonnull align 8 dereferenceable(16) %9, i8 0, i64 16, i1 false) #24
  %10 = getelementptr inbounds %"class.tutorial::Person", %"class.tutorial::Person"* %0, i64 0, i32 6, i32 0, i32 0, i32 0
  store i32 0, i32* %10, align 4, !tbaa !13
  %11 = load atomic i32, i32* getelementptr inbounds ({ { { i32 }, i32, i32, void ()* }, [2 x i8*] }, { { { i32 }, i32, i32, void ()* }, [2 x i8*] }* @scc_info_Person_addressbook_2eproto, i64 0, i32 0, i32 0, i32 0) acquire, align 8
  %12 = icmp eq i32 %11, 0
  br i1 %12, label %14, label %13, !prof !12

13:                                               ; preds = %2
  invoke void @_ZN6google8protobuf8internal11InitSCCImplEPNS1_11SCCInfoBaseE(%"struct.google::protobuf::internal::SCCInfoBase"* noundef nonnull bitcast ({ { { i32 }, i32, i32, void ()* }, [2 x i8*] }* @scc_info_Person_addressbook_2eproto to %"struct.google::protobuf::internal::SCCInfoBase"*))
          to label %14 unwind label %19

14:                                               ; preds = %2, %13
  %15 = getelementptr inbounds %"class.tutorial::Person", %"class.tutorial::Person"* %0, i64 0, i32 2, i32 0
  store %"class.std::__cxx11::basic_string"* bitcast (%"class.google::protobuf::internal::ExplicitlyConstructed.20"* @_ZN6google8protobuf8internal26fixed_address_empty_stringB5cxx11E to %"class.std::__cxx11::basic_string"*), %"class.std::__cxx11::basic_string"** %15, align 8, !tbaa !18
  %16 = getelementptr inbounds %"class.tutorial::Person", %"class.tutorial::Person"* %0, i64 0, i32 3, i32 0
  store %"class.std::__cxx11::basic_string"* bitcast (%"class.google::protobuf::internal::ExplicitlyConstructed.20"* @_ZN6google8protobuf8internal26fixed_address_empty_stringB5cxx11E to %"class.std::__cxx11::basic_string"*), %"class.std::__cxx11::basic_string"** %16, align 8, !tbaa !18
  %17 = getelementptr inbounds %"class.tutorial::Person", %"class.tutorial::Person"* %0, i64 0, i32 4
  %18 = bitcast %"class.google::protobuf::Timestamp"** %17 to i8*
  tail call void @llvm.memset.p0i8.i64(i8* noundef nonnull align 8 dereferenceable(12) %18, i8 0, i64 12, i1 false)
  ret void

19:                                               ; preds = %13
  %20 = landingpad { i8*, i32 }
          cleanup
  tail call void @_ZN6google8protobuf16RepeatedPtrFieldIN8tutorial18Person_PhoneNumberEED2Ev(%"class.google::protobuf::RepeatedPtrField"* noundef nonnull align 8 dereferenceable(24) %6) #24
  resume { i8*, i32 } %20
}

; Function Attrs: nounwind uwtable
define linkonce_odr dso_local void @_ZN6google8protobuf16RepeatedPtrFieldIN8tutorial18Person_PhoneNumberEED2Ev(%"class.google::protobuf::RepeatedPtrField"* noundef nonnull align 8 dereferenceable(24) %0) unnamed_addr #6 comdat align 2 personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*) {
  %2 = getelementptr inbounds %"class.google::protobuf::RepeatedPtrField", %"class.google::protobuf::RepeatedPtrField"* %0, i64 0, i32 0
  invoke void @_ZN6google8protobuf8internal20RepeatedPtrFieldBase7DestroyINS0_16RepeatedPtrFieldIN8tutorial18Person_PhoneNumberEE11TypeHandlerEEEvv(%"class.google::protobuf::internal::RepeatedPtrFieldBase"* noundef nonnull align 8 dereferenceable(24) %2)
          to label %3 unwind label %14

3:                                                ; preds = %1
  %4 = getelementptr inbounds %"class.google::protobuf::RepeatedPtrField", %"class.google::protobuf::RepeatedPtrField"* %0, i64 0, i32 0, i32 0
  %5 = load %"class.google::protobuf::Arena"*, %"class.google::protobuf::Arena"** %4, align 8, !tbaa !16
  %6 = icmp eq %"class.google::protobuf::Arena"* %5, null
  br i1 %6, label %13, label %7

7:                                                ; preds = %3
  %8 = getelementptr inbounds %"class.google::protobuf::Arena", %"class.google::protobuf::Arena"* %5, i64 0, i32 0
  %9 = invoke noundef i64 @_ZNK6google8protobuf8internal9ArenaImpl14SpaceAllocatedEv(%"class.google::protobuf::internal::ArenaImpl"* noundef nonnull align 8 dereferenceable(88) %8)
          to label %13 unwind label %10

10:                                               ; preds = %7
  %11 = landingpad { i8*, i32 }
          catch i8* null
  %12 = extractvalue { i8*, i32 } %11, 0
  tail call void @__clang_call_terminate(i8* %12) #25
  unreachable

13:                                               ; preds = %3, %7
  ret void

14:                                               ; preds = %1
  %15 = landingpad { i8*, i32 }
          catch i8* null
  %16 = extractvalue { i8*, i32 } %15, 0
  tail call void @_ZN6google8protobuf8internal20RepeatedPtrFieldBaseD2Ev(%"class.google::protobuf::internal::RepeatedPtrFieldBase"* noundef nonnull align 8 dereferenceable(24) %2) #24
  tail call void @__clang_call_terminate(i8* %16) #25
  unreachable
}

; Function Attrs: uwtable
define dso_local void @_ZN8tutorial6PersonC2ERKS0_(%"class.tutorial::Person"* noundef nonnull align 8 dereferenceable(72) %0, %"class.tutorial::Person"* noundef nonnull align 8 dereferenceable(72) %1) unnamed_addr #3 align 2 personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*) {
  %3 = getelementptr inbounds %"class.tutorial::Person", %"class.tutorial::Person"* %0, i64 0, i32 0, i32 0, i32 1, i32 0
  store i8* null, i8** %3, align 8, !tbaa !5
  %4 = getelementptr inbounds %"class.tutorial::Person", %"class.tutorial::Person"* %0, i64 0, i32 0, i32 0, i32 0
  store i32 (...)** bitcast (i8** getelementptr inbounds ({ [22 x i8*] }, { [22 x i8*] }* @_ZTVN8tutorial6PersonE, i64 0, inrange i32 0, i64 2) to i32 (...)**), i32 (...)*** %4, align 8, !tbaa !10
  %5 = getelementptr inbounds %"class.tutorial::Person", %"class.tutorial::Person"* %0, i64 0, i32 1
  %6 = getelementptr inbounds %"class.google::protobuf::RepeatedPtrField", %"class.google::protobuf::RepeatedPtrField"* %5, i64 0, i32 0
  %7 = bitcast %"class.google::protobuf::RepeatedPtrField"* %5 to i8*
  tail call void @llvm.memset.p0i8.i64(i8* noundef nonnull align 8 dereferenceable(24) %7, i8 0, i64 24, i1 false) #24
  %8 = getelementptr inbounds %"class.tutorial::Person", %"class.tutorial::Person"* %1, i64 0, i32 1, i32 0
  invoke void @_ZN6google8protobuf8internal20RepeatedPtrFieldBase9MergeFromINS0_16RepeatedPtrFieldIN8tutorial18Person_PhoneNumberEE11TypeHandlerEEEvRKS2_(%"class.google::protobuf::internal::RepeatedPtrFieldBase"* noundef nonnull align 8 dereferenceable(24) %6, %"class.google::protobuf::internal::RepeatedPtrFieldBase"* noundef nonnull align 8 dereferenceable(24) %8)
          to label %11 unwind label %9

9:                                                ; preds = %2
  %10 = landingpad { i8*, i32 }
          cleanup
  tail call void @_ZN6google8protobuf8internal20RepeatedPtrFieldBaseD2Ev(%"class.google::protobuf::internal::RepeatedPtrFieldBase"* noundef nonnull align 8 dereferenceable(24) %6) #24
  br label %105

11:                                               ; preds = %2
  %12 = getelementptr inbounds %"class.tutorial::Person", %"class.tutorial::Person"* %0, i64 0, i32 6, i32 0, i32 0, i32 0
  store i32 0, i32* %12, align 4, !tbaa !13
  %13 = getelementptr inbounds %"class.tutorial::Person", %"class.tutorial::Person"* %0, i64 0, i32 0, i32 0, i32 1
  %14 = getelementptr inbounds %"class.tutorial::Person", %"class.tutorial::Person"* %1, i64 0, i32 0, i32 0, i32 1, i32 0
  %15 = load i8*, i8** %14, align 8, !tbaa !5
  %16 = ptrtoint i8* %15 to i64
  %17 = and i64 %16, 1
  %18 = icmp eq i64 %17, 0
  br i1 %18, label %36, label %19

19:                                               ; preds = %11
  %20 = and i64 %16, -2
  %21 = inttoptr i64 %20 to %"struct.google::protobuf::internal::InternalMetadata::Container"*
  %22 = getelementptr inbounds %"struct.google::protobuf::internal::InternalMetadata::Container", %"struct.google::protobuf::internal::InternalMetadata::Container"* %21, i64 0, i32 1
  %23 = getelementptr inbounds %"class.google::protobuf::internal::InternalMetadata", %"class.google::protobuf::internal::InternalMetadata"* %13, i64 0, i32 0
  %24 = load i8*, i8** %23, align 8, !tbaa !5
  %25 = ptrtoint i8* %24 to i64
  %26 = and i64 %25, 1
  %27 = icmp eq i64 %26, 0
  br i1 %27, label %32, label %28, !prof !41

28:                                               ; preds = %19
  %29 = and i64 %25, -2
  %30 = inttoptr i64 %29 to %"struct.google::protobuf::internal::InternalMetadata::Container"*
  %31 = getelementptr inbounds %"struct.google::protobuf::internal::InternalMetadata::Container", %"struct.google::protobuf::internal::InternalMetadata::Container"* %30, i64 0, i32 1
  br label %34

32:                                               ; preds = %19
  %33 = invoke noundef %"class.google::protobuf::UnknownFieldSet"* @_ZN6google8protobuf8internal16InternalMetadata27mutable_unknown_fields_slowINS0_15UnknownFieldSetEEEPT_v(%"class.google::protobuf::internal::InternalMetadata"* noundef nonnull align 8 dereferenceable(8) %13)
          to label %34 unwind label %58

34:                                               ; preds = %32, %28
  %35 = phi %"class.google::protobuf::UnknownFieldSet"* [ %31, %28 ], [ %33, %32 ]
  invoke void @_ZN6google8protobuf15UnknownFieldSet9MergeFromERKS1_(%"class.google::protobuf::UnknownFieldSet"* noundef nonnull align 8 dereferenceable(24) %35, %"class.google::protobuf::UnknownFieldSet"* noundef nonnull align 8 dereferenceable(24) %22)
          to label %36 unwind label %58

36:                                               ; preds = %34, %11
  %37 = getelementptr inbounds %"class.tutorial::Person", %"class.tutorial::Person"* %0, i64 0, i32 2
  %38 = getelementptr inbounds %"struct.google::protobuf::internal::ArenaStringPtr", %"struct.google::protobuf::internal::ArenaStringPtr"* %37, i64 0, i32 0
  store %"class.std::__cxx11::basic_string"* bitcast (%"class.google::protobuf::internal::ExplicitlyConstructed.20"* @_ZN6google8protobuf8internal26fixed_address_empty_stringB5cxx11E to %"class.std::__cxx11::basic_string"*), %"class.std::__cxx11::basic_string"** %38, align 8, !tbaa !18
  %39 = getelementptr inbounds %"class.tutorial::Person", %"class.tutorial::Person"* %1, i64 0, i32 2, i32 0
  %40 = load %"class.std::__cxx11::basic_string"*, %"class.std::__cxx11::basic_string"** %39, align 8, !tbaa !18
  %41 = getelementptr inbounds %"class.std::__cxx11::basic_string", %"class.std::__cxx11::basic_string"* %40, i64 0, i32 1
  %42 = load i64, i64* %41, align 8, !tbaa !28
  %43 = icmp eq i64 %42, 0
  br i1 %43, label %60, label %44

44:                                               ; preds = %36
  %45 = load i8*, i8** %3, align 8, !tbaa !5
  %46 = ptrtoint i8* %45 to i64
  %47 = and i64 %46, 1
  %48 = icmp eq i64 %47, 0
  %49 = and i64 %46, -2
  br i1 %48, label %54, label %50, !prof !12

50:                                               ; preds = %44
  %51 = inttoptr i64 %49 to %"struct.google::protobuf::internal::InternalMetadata::ContainerBase"*
  %52 = getelementptr inbounds %"struct.google::protobuf::internal::InternalMetadata::ContainerBase", %"struct.google::protobuf::internal::InternalMetadata::ContainerBase"* %51, i64 0, i32 0
  %53 = load %"class.google::protobuf::Arena"*, %"class.google::protobuf::Arena"** %52, align 8, !tbaa !32
  br label %56

54:                                               ; preds = %44
  %55 = inttoptr i64 %49 to %"class.google::protobuf::Arena"*
  br label %56

56:                                               ; preds = %50, %54
  %57 = phi %"class.google::protobuf::Arena"* [ %53, %50 ], [ %55, %54 ]
  invoke void @_ZN6google8protobuf8internal14ArenaStringPtr14CreateInstanceEPNS0_5ArenaEPKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE(%"struct.google::protobuf::internal::ArenaStringPtr"* noundef nonnull align 8 dereferenceable(8) %37, %"class.google::protobuf::Arena"* noundef %57, %"class.std::__cxx11::basic_string"* noundef nonnull %40)
          to label %60 unwind label %58

58:                                               ; preds = %80, %56, %34, %32, %88
  %59 = landingpad { i8*, i32 }
          cleanup
  br label %103

60:                                               ; preds = %36, %56
  %61 = getelementptr inbounds %"class.tutorial::Person", %"class.tutorial::Person"* %0, i64 0, i32 3
  %62 = getelementptr inbounds %"struct.google::protobuf::internal::ArenaStringPtr", %"struct.google::protobuf::internal::ArenaStringPtr"* %61, i64 0, i32 0
  store %"class.std::__cxx11::basic_string"* bitcast (%"class.google::protobuf::internal::ExplicitlyConstructed.20"* @_ZN6google8protobuf8internal26fixed_address_empty_stringB5cxx11E to %"class.std::__cxx11::basic_string"*), %"class.std::__cxx11::basic_string"** %62, align 8, !tbaa !18
  %63 = getelementptr inbounds %"class.tutorial::Person", %"class.tutorial::Person"* %1, i64 0, i32 3, i32 0
  %64 = load %"class.std::__cxx11::basic_string"*, %"class.std::__cxx11::basic_string"** %63, align 8, !tbaa !18
  %65 = getelementptr inbounds %"class.std::__cxx11::basic_string", %"class.std::__cxx11::basic_string"* %64, i64 0, i32 1
  %66 = load i64, i64* %65, align 8, !tbaa !28
  %67 = icmp eq i64 %66, 0
  br i1 %67, label %82, label %68

68:                                               ; preds = %60
  %69 = load i8*, i8** %3, align 8, !tbaa !5
  %70 = ptrtoint i8* %69 to i64
  %71 = and i64 %70, 1
  %72 = icmp eq i64 %71, 0
  %73 = and i64 %70, -2
  br i1 %72, label %78, label %74, !prof !12

74:                                               ; preds = %68
  %75 = inttoptr i64 %73 to %"struct.google::protobuf::internal::InternalMetadata::ContainerBase"*
  %76 = getelementptr inbounds %"struct.google::protobuf::internal::InternalMetadata::ContainerBase", %"struct.google::protobuf::internal::InternalMetadata::ContainerBase"* %75, i64 0, i32 0
  %77 = load %"class.google::protobuf::Arena"*, %"class.google::protobuf::Arena"** %76, align 8, !tbaa !32
  br label %80

78:                                               ; preds = %68
  %79 = inttoptr i64 %73 to %"class.google::protobuf::Arena"*
  br label %80

80:                                               ; preds = %74, %78
  %81 = phi %"class.google::protobuf::Arena"* [ %77, %74 ], [ %79, %78 ]
  invoke void @_ZN6google8protobuf8internal14ArenaStringPtr14CreateInstanceEPNS0_5ArenaEPKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE(%"struct.google::protobuf::internal::ArenaStringPtr"* noundef nonnull align 8 dereferenceable(8) %61, %"class.google::protobuf::Arena"* noundef %81, %"class.std::__cxx11::basic_string"* noundef nonnull %64)
          to label %82 unwind label %58

82:                                               ; preds = %60, %80
  %83 = icmp ne %"class.tutorial::Person"* %1, bitcast (%"class.tutorial::PersonDefaultTypeInternal"* @_ZN8tutorial25_Person_default_instance_E to %"class.tutorial::Person"*)
  %84 = getelementptr inbounds %"class.tutorial::Person", %"class.tutorial::Person"* %1, i64 0, i32 4
  %85 = load %"class.google::protobuf::Timestamp"*, %"class.google::protobuf::Timestamp"** %84, align 8
  %86 = icmp ne %"class.google::protobuf::Timestamp"* %85, null
  %87 = select i1 %83, i1 %86, i1 false
  br i1 %87, label %88, label %97

88:                                               ; preds = %82
  %89 = invoke noalias noundef nonnull dereferenceable(32) i8* @_Znwm(i64 noundef 32) #27
          to label %90 unwind label %58

90:                                               ; preds = %88
  %91 = bitcast i8* %89 to %"class.google::protobuf::Timestamp"*
  invoke void @_ZN6google8protobuf9TimestampC1ERKS1_(%"class.google::protobuf::Timestamp"* noundef nonnull align 8 dereferenceable(32) %91, %"class.google::protobuf::Timestamp"* noundef nonnull align 8 dereferenceable(32) %85)
          to label %92 unwind label %95

92:                                               ; preds = %90
  %93 = getelementptr inbounds %"class.tutorial::Person", %"class.tutorial::Person"* %0, i64 0, i32 4
  %94 = bitcast %"class.google::protobuf::Timestamp"** %93 to i8**
  store i8* %89, i8** %94, align 8, !tbaa !20
  br label %99

95:                                               ; preds = %90
  %96 = landingpad { i8*, i32 }
          cleanup
  tail call void @_ZdlPv(i8* noundef nonnull %89) #26
  br label %103

97:                                               ; preds = %82
  %98 = getelementptr inbounds %"class.tutorial::Person", %"class.tutorial::Person"* %0, i64 0, i32 4
  store %"class.google::protobuf::Timestamp"* null, %"class.google::protobuf::Timestamp"** %98, align 8, !tbaa !20
  br label %99

99:                                               ; preds = %97, %92
  %100 = getelementptr inbounds %"class.tutorial::Person", %"class.tutorial::Person"* %1, i64 0, i32 5
  %101 = load i32, i32* %100, align 8, !tbaa !56
  %102 = getelementptr inbounds %"class.tutorial::Person", %"class.tutorial::Person"* %0, i64 0, i32 5
  store i32 %101, i32* %102, align 8, !tbaa !56
  ret void

103:                                              ; preds = %95, %58
  %104 = phi { i8*, i32 } [ %96, %95 ], [ %59, %58 ]
  tail call void @_ZN6google8protobuf16RepeatedPtrFieldIN8tutorial18Person_PhoneNumberEED2Ev(%"class.google::protobuf::RepeatedPtrField"* noundef nonnull align 8 dereferenceable(24) %5) #24
  br label %105

105:                                              ; preds = %9, %103
  %106 = phi { i8*, i32 } [ %104, %103 ], [ %10, %9 ]
  resume { i8*, i32 } %106
}

; Function Attrs: nobuiltin allocsize(0)
declare noundef nonnull i8* @_Znwm(i64 noundef) local_unnamed_addr #15

declare void @_ZN6google8protobuf9TimestampC1ERKS1_(%"class.google::protobuf::Timestamp"* noundef nonnull align 8 dereferenceable(32), %"class.google::protobuf::Timestamp"* noundef nonnull align 8 dereferenceable(32)) unnamed_addr #0

; Function Attrs: nounwind uwtable
define dso_local void @_ZN8tutorial6PersonD2Ev(%"class.tutorial::Person"* noundef nonnull align 8 dereferenceable(72) %0) unnamed_addr #6 align 2 personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*) {
  invoke void @_ZN8tutorial6Person10SharedDtorEv(%"class.tutorial::Person"* noundef nonnull align 8 dereferenceable(72) %0)
          to label %2 unwind label %21

2:                                                ; preds = %1
  %3 = getelementptr inbounds %"class.tutorial::Person", %"class.tutorial::Person"* %0, i64 0, i32 0, i32 0, i32 1
  invoke void @_ZN6google8protobuf8internal16InternalMetadata6DeleteINS0_15UnknownFieldSetEEEvv(%"class.google::protobuf::internal::InternalMetadata"* noundef nonnull align 8 dereferenceable(8) %3)
          to label %4 unwind label %21

4:                                                ; preds = %2
  %5 = getelementptr inbounds %"class.tutorial::Person", %"class.tutorial::Person"* %0, i64 0, i32 1
  %6 = getelementptr inbounds %"class.google::protobuf::RepeatedPtrField", %"class.google::protobuf::RepeatedPtrField"* %5, i64 0, i32 0
  invoke void @_ZN6google8protobuf8internal20RepeatedPtrFieldBase7DestroyINS0_16RepeatedPtrFieldIN8tutorial18Person_PhoneNumberEE11TypeHandlerEEEvv(%"class.google::protobuf::internal::RepeatedPtrFieldBase"* noundef nonnull align 8 dereferenceable(24) %6)
          to label %7 unwind label %17

7:                                                ; preds = %4
  %8 = getelementptr inbounds %"class.google::protobuf::RepeatedPtrField", %"class.google::protobuf::RepeatedPtrField"* %5, i64 0, i32 0, i32 0
  %9 = load %"class.google::protobuf::Arena"*, %"class.google::protobuf::Arena"** %8, align 8, !tbaa !16
  %10 = icmp eq %"class.google::protobuf::Arena"* %9, null
  br i1 %10, label %20, label %11

11:                                               ; preds = %7
  %12 = getelementptr inbounds %"class.google::protobuf::Arena", %"class.google::protobuf::Arena"* %9, i64 0, i32 0
  %13 = invoke noundef i64 @_ZNK6google8protobuf8internal9ArenaImpl14SpaceAllocatedEv(%"class.google::protobuf::internal::ArenaImpl"* noundef nonnull align 8 dereferenceable(88) %12)
          to label %20 unwind label %14

14:                                               ; preds = %11
  %15 = landingpad { i8*, i32 }
          catch i8* null
  %16 = extractvalue { i8*, i32 } %15, 0
  tail call void @__clang_call_terminate(i8* %16) #25
  unreachable

17:                                               ; preds = %4
  %18 = landingpad { i8*, i32 }
          catch i8* null
  %19 = extractvalue { i8*, i32 } %18, 0
  tail call void @_ZN6google8protobuf8internal20RepeatedPtrFieldBaseD2Ev(%"class.google::protobuf::internal::RepeatedPtrFieldBase"* noundef nonnull align 8 dereferenceable(24) %6) #24
  tail call void @__clang_call_terminate(i8* %19) #25
  unreachable

20:                                               ; preds = %7, %11
  ret void

21:                                               ; preds = %2, %1
  %22 = landingpad { i8*, i32 }
          catch i8* null
  %23 = extractvalue { i8*, i32 } %22, 0
  %24 = getelementptr inbounds %"class.tutorial::Person", %"class.tutorial::Person"* %0, i64 0, i32 1
  tail call void @_ZN6google8protobuf16RepeatedPtrFieldIN8tutorial18Person_PhoneNumberEED2Ev(%"class.google::protobuf::RepeatedPtrField"* noundef nonnull align 8 dereferenceable(24) %24) #24
  tail call void @__clang_call_terminate(i8* %23) #25
  unreachable
}

; Function Attrs: inlinehint uwtable
define linkonce_odr dso_local void @_ZN8tutorial6Person10SharedDtorEv(%"class.tutorial::Person"* noundef nonnull align 8 dereferenceable(72) %0) local_unnamed_addr #7 comdat align 2 personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*) {
  %2 = alloca %"class.google::protobuf::internal::LogMessage", align 8
  %3 = alloca %"class.google::protobuf::internal::LogFinisher", align 1
  %4 = getelementptr inbounds %"class.tutorial::Person", %"class.tutorial::Person"* %0, i64 0, i32 0, i32 0, i32 1, i32 0
  %5 = load i8*, i8** %4, align 8, !tbaa !5
  %6 = ptrtoint i8* %5 to i64
  %7 = and i64 %6, 1
  %8 = icmp eq i64 %7, 0
  %9 = and i64 %6, -2
  br i1 %8, label %14, label %10, !prof !12

10:                                               ; preds = %1
  %11 = inttoptr i64 %9 to %"struct.google::protobuf::internal::InternalMetadata::ContainerBase"*
  %12 = getelementptr inbounds %"struct.google::protobuf::internal::InternalMetadata::ContainerBase", %"struct.google::protobuf::internal::InternalMetadata::ContainerBase"* %11, i64 0, i32 0
  %13 = load %"class.google::protobuf::Arena"*, %"class.google::protobuf::Arena"** %12, align 8, !tbaa !32
  br label %16

14:                                               ; preds = %1
  %15 = inttoptr i64 %9 to %"class.google::protobuf::Arena"*
  br label %16

16:                                               ; preds = %10, %14
  %17 = phi %"class.google::protobuf::Arena"* [ %13, %10 ], [ %15, %14 ]
  %18 = icmp eq %"class.google::protobuf::Arena"* %17, null
  %19 = getelementptr inbounds %"class.google::protobuf::internal::LogFinisher", %"class.google::protobuf::internal::LogFinisher"* %3, i64 0, i32 0
  call void @llvm.lifetime.start.p0i8(i64 1, i8* nonnull %19) #24
  br i1 %18, label %24, label %20

20:                                               ; preds = %16
  %21 = bitcast %"class.google::protobuf::internal::LogMessage"* %2 to i8*
  call void @llvm.lifetime.start.p0i8(i64 56, i8* nonnull %21) #24
  call void @_ZN6google8protobuf8internal10LogMessageC1ENS0_8LogLevelEPKci(%"class.google::protobuf::internal::LogMessage"* noundef nonnull align 8 dereferenceable(56) %2, i32 noundef 3, i8* noundef getelementptr inbounds ([24 x i8], [24 x i8]* @.str.5, i64 0, i64 0), i32 noundef 483)
  %22 = invoke noundef nonnull align 8 dereferenceable(56) %"class.google::protobuf::internal::LogMessage"* @_ZN6google8protobuf8internal10LogMessagelsEPKc(%"class.google::protobuf::internal::LogMessage"* noundef nonnull align 8 dereferenceable(56) %2, i8* noundef getelementptr inbounds ([38 x i8], [38 x i8]* @.str.12, i64 0, i64 0))
          to label %23 unwind label %64

23:                                               ; preds = %20
  invoke void @_ZN6google8protobuf8internal11LogFinisheraSERNS1_10LogMessageE(%"class.google::protobuf::internal::LogFinisher"* noundef nonnull align 1 dereferenceable(1) %3, %"class.google::protobuf::internal::LogMessage"* noundef nonnull align 8 dereferenceable(56) %22)
          to label %25 unwind label %66

24:                                               ; preds = %16
  call void @llvm.lifetime.end.p0i8(i64 1, i8* nonnull %19) #24
  br label %26

25:                                               ; preds = %23
  call void @llvm.lifetime.end.p0i8(i64 1, i8* nonnull %19) #24
  call void @_ZN6google8protobuf8internal10LogMessageD1Ev(%"class.google::protobuf::internal::LogMessage"* noundef nonnull align 8 dereferenceable(56) %2) #24
  call void @llvm.lifetime.end.p0i8(i64 56, i8* nonnull %21) #24
  br label %26

26:                                               ; preds = %24, %25
  %27 = getelementptr inbounds %"class.tutorial::Person", %"class.tutorial::Person"* %0, i64 0, i32 2, i32 0
  %28 = load %"class.std::__cxx11::basic_string"*, %"class.std::__cxx11::basic_string"** %27, align 8, !tbaa !18
  %29 = icmp eq %"class.std::__cxx11::basic_string"* %28, bitcast (%"class.google::protobuf::internal::ExplicitlyConstructed.20"* @_ZN6google8protobuf8internal26fixed_address_empty_stringB5cxx11E to %"class.std::__cxx11::basic_string"*)
  %30 = icmp eq %"class.std::__cxx11::basic_string"* %28, null
  %31 = or i1 %29, %30
  br i1 %31, label %41, label %32

32:                                               ; preds = %26
  %33 = getelementptr inbounds %"class.std::__cxx11::basic_string", %"class.std::__cxx11::basic_string"* %28, i64 0, i32 0, i32 0
  %34 = load i8*, i8** %33, align 8, !tbaa !34
  %35 = getelementptr inbounds %"class.std::__cxx11::basic_string", %"class.std::__cxx11::basic_string"* %28, i64 0, i32 2
  %36 = bitcast %union.anon* %35 to i8*
  %37 = icmp eq i8* %34, %36
  br i1 %37, label %39, label %38

38:                                               ; preds = %32
  call void @_ZdlPv(i8* noundef %34) #24
  br label %39

39:                                               ; preds = %38, %32
  %40 = bitcast %"class.std::__cxx11::basic_string"* %28 to i8*
  call void @_ZdlPv(i8* noundef %40) #26
  br label %41

41:                                               ; preds = %26, %39
  %42 = getelementptr inbounds %"class.tutorial::Person", %"class.tutorial::Person"* %0, i64 0, i32 3, i32 0
  %43 = load %"class.std::__cxx11::basic_string"*, %"class.std::__cxx11::basic_string"** %42, align 8, !tbaa !18
  %44 = icmp eq %"class.std::__cxx11::basic_string"* %43, bitcast (%"class.google::protobuf::internal::ExplicitlyConstructed.20"* @_ZN6google8protobuf8internal26fixed_address_empty_stringB5cxx11E to %"class.std::__cxx11::basic_string"*)
  %45 = icmp eq %"class.std::__cxx11::basic_string"* %43, null
  %46 = or i1 %44, %45
  br i1 %46, label %56, label %47

47:                                               ; preds = %41
  %48 = getelementptr inbounds %"class.std::__cxx11::basic_string", %"class.std::__cxx11::basic_string"* %43, i64 0, i32 0, i32 0
  %49 = load i8*, i8** %48, align 8, !tbaa !34
  %50 = getelementptr inbounds %"class.std::__cxx11::basic_string", %"class.std::__cxx11::basic_string"* %43, i64 0, i32 2
  %51 = bitcast %union.anon* %50 to i8*
  %52 = icmp eq i8* %49, %51
  br i1 %52, label %54, label %53

53:                                               ; preds = %47
  call void @_ZdlPv(i8* noundef %49) #24
  br label %54

54:                                               ; preds = %53, %47
  %55 = bitcast %"class.std::__cxx11::basic_string"* %43 to i8*
  call void @_ZdlPv(i8* noundef %55) #26
  br label %56

56:                                               ; preds = %41, %54
  %57 = icmp eq %"class.tutorial::Person"* %0, bitcast (%"class.tutorial::PersonDefaultTypeInternal"* @_ZN8tutorial25_Person_default_instance_E to %"class.tutorial::Person"*)
  br i1 %57, label %70, label %58

58:                                               ; preds = %56
  %59 = getelementptr inbounds %"class.tutorial::Person", %"class.tutorial::Person"* %0, i64 0, i32 4
  %60 = load %"class.google::protobuf::Timestamp"*, %"class.google::protobuf::Timestamp"** %59, align 8, !tbaa !20
  %61 = icmp eq %"class.google::protobuf::Timestamp"* %60, null
  br i1 %61, label %70, label %62

62:                                               ; preds = %58
  call void @_ZN6google8protobuf9TimestampD1Ev(%"class.google::protobuf::Timestamp"* noundef nonnull align 8 dereferenceable(32) %60) #24
  %63 = bitcast %"class.google::protobuf::Timestamp"* %60 to i8*
  call void @_ZdlPv(i8* noundef %63) #26
  br label %70

64:                                               ; preds = %20
  %65 = landingpad { i8*, i32 }
          cleanup
  br label %68

66:                                               ; preds = %23
  %67 = landingpad { i8*, i32 }
          cleanup
  call void @llvm.lifetime.end.p0i8(i64 1, i8* nonnull %19) #24
  br label %68

68:                                               ; preds = %64, %66
  %69 = phi { i8*, i32 } [ %67, %66 ], [ %65, %64 ]
  call void @_ZN6google8protobuf8internal10LogMessageD1Ev(%"class.google::protobuf::internal::LogMessage"* noundef nonnull align 8 dereferenceable(56) %2) #24
  call void @llvm.lifetime.end.p0i8(i64 56, i8* nonnull %21) #24
  resume { i8*, i32 } %69

70:                                               ; preds = %58, %62, %56
  ret void
}

; Function Attrs: nounwind uwtable
define dso_local void @_ZN8tutorial6PersonD0Ev(%"class.tutorial::Person"* noundef nonnull align 8 dereferenceable(72) %0) unnamed_addr #6 align 2 {
  tail call void @_ZN8tutorial6PersonD2Ev(%"class.tutorial::Person"* noundef nonnull align 8 dereferenceable(72) %0) #24
  %2 = bitcast %"class.tutorial::Person"* %0 to i8*
  tail call void @_ZdlPv(i8* noundef nonnull %2) #26
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind readnone uwtable willreturn
define dso_local void @_ZN8tutorial6Person9ArenaDtorEPv(i8* nocapture noundef %0) local_unnamed_addr #5 align 2 {
  ret void
}

; Function Attrs: mustprogress nofree norecurse nounwind uwtable willreturn
define dso_local void @_ZNK8tutorial6Person13SetCachedSizeEi(%"class.tutorial::Person"* nocapture noundef nonnull writeonly align 8 dereferenceable(72) %0, i32 noundef %1) unnamed_addr #11 align 2 personality i32 (...)* @__gxx_personality_v0 {
  %3 = getelementptr inbounds %"class.tutorial::Person", %"class.tutorial::Person"* %0, i64 0, i32 6, i32 0, i32 0, i32 0
  store atomic i32 %1, i32* %3 monotonic, align 4
  ret void
}

; Function Attrs: mustprogress uwtable
define dso_local noundef nonnull align 8 dereferenceable(72) %"class.tutorial::Person"* @_ZN8tutorial6Person16default_instanceEv() local_unnamed_addr #4 align 2 {
  %1 = load atomic i32, i32* getelementptr inbounds ({ { { i32 }, i32, i32, void ()* }, [2 x i8*] }, { { { i32 }, i32, i32, void ()* }, [2 x i8*] }* @scc_info_Person_addressbook_2eproto, i64 0, i32 0, i32 0, i32 0) acquire, align 8
  %2 = icmp eq i32 %1, 0
  br i1 %2, label %4, label %3, !prof !12

3:                                                ; preds = %0
  tail call void @_ZN6google8protobuf8internal11InitSCCImplEPNS1_11SCCInfoBaseE(%"struct.google::protobuf::internal::SCCInfoBase"* noundef nonnull bitcast ({ { { i32 }, i32, i32, void ()* }, [2 x i8*] }* @scc_info_Person_addressbook_2eproto to %"struct.google::protobuf::internal::SCCInfoBase"*))
  br label %4

4:                                                ; preds = %0, %3
  ret %"class.tutorial::Person"* bitcast (%"class.tutorial::PersonDefaultTypeInternal"* @_ZN8tutorial25_Person_default_instance_E to %"class.tutorial::Person"*)
}

; Function Attrs: uwtable
define dso_local void @_ZN8tutorial6Person5ClearEv(%"class.tutorial::Person"* noundef nonnull align 8 dereferenceable(72) %0) unnamed_addr #3 align 2 personality i32 (...)* @__gxx_personality_v0 {
  %2 = getelementptr inbounds %"class.tutorial::Person", %"class.tutorial::Person"* %0, i64 0, i32 1, i32 0
  tail call void @_ZN6google8protobuf8internal20RepeatedPtrFieldBase5ClearINS0_16RepeatedPtrFieldIN8tutorial18Person_PhoneNumberEE11TypeHandlerEEEvv(%"class.google::protobuf::internal::RepeatedPtrFieldBase"* noundef nonnull align 8 dereferenceable(24) %2)
  %3 = getelementptr inbounds %"class.tutorial::Person", %"class.tutorial::Person"* %0, i64 0, i32 0, i32 0, i32 1, i32 0
  %4 = getelementptr inbounds %"class.tutorial::Person", %"class.tutorial::Person"* %0, i64 0, i32 2, i32 0
  %5 = load %"class.std::__cxx11::basic_string"*, %"class.std::__cxx11::basic_string"** %4, align 8, !tbaa !18
  %6 = icmp eq %"class.std::__cxx11::basic_string"* %5, bitcast (%"class.google::protobuf::internal::ExplicitlyConstructed.20"* @_ZN6google8protobuf8internal26fixed_address_empty_stringB5cxx11E to %"class.std::__cxx11::basic_string"*)
  br i1 %6, label %11, label %7

7:                                                ; preds = %1
  %8 = getelementptr inbounds %"class.std::__cxx11::basic_string", %"class.std::__cxx11::basic_string"* %5, i64 0, i32 1
  store i64 0, i64* %8, align 8, !tbaa !28
  %9 = getelementptr inbounds %"class.std::__cxx11::basic_string", %"class.std::__cxx11::basic_string"* %5, i64 0, i32 0, i32 0
  %10 = load i8*, i8** %9, align 8, !tbaa !34
  store i8 0, i8* %10, align 1, !tbaa !37
  br label %11

11:                                               ; preds = %7, %1
  %12 = getelementptr inbounds %"class.tutorial::Person", %"class.tutorial::Person"* %0, i64 0, i32 3, i32 0
  %13 = load %"class.std::__cxx11::basic_string"*, %"class.std::__cxx11::basic_string"** %12, align 8, !tbaa !18
  %14 = icmp eq %"class.std::__cxx11::basic_string"* %13, bitcast (%"class.google::protobuf::internal::ExplicitlyConstructed.20"* @_ZN6google8protobuf8internal26fixed_address_empty_stringB5cxx11E to %"class.std::__cxx11::basic_string"*)
  br i1 %14, label %19, label %15

15:                                               ; preds = %11
  %16 = getelementptr inbounds %"class.std::__cxx11::basic_string", %"class.std::__cxx11::basic_string"* %13, i64 0, i32 1
  store i64 0, i64* %16, align 8, !tbaa !28
  %17 = getelementptr inbounds %"class.std::__cxx11::basic_string", %"class.std::__cxx11::basic_string"* %13, i64 0, i32 0, i32 0
  %18 = load i8*, i8** %17, align 8, !tbaa !34
  store i8 0, i8* %18, align 1, !tbaa !37
  br label %19

19:                                               ; preds = %11, %15
  %20 = load i8*, i8** %3, align 8, !tbaa !5
  %21 = ptrtoint i8* %20 to i64
  %22 = and i64 %21, 1
  %23 = icmp eq i64 %22, 0
  %24 = and i64 %21, -2
  br i1 %23, label %29, label %25, !prof !12

25:                                               ; preds = %19
  %26 = inttoptr i64 %24 to %"struct.google::protobuf::internal::InternalMetadata::ContainerBase"*
  %27 = getelementptr inbounds %"struct.google::protobuf::internal::InternalMetadata::ContainerBase", %"struct.google::protobuf::internal::InternalMetadata::ContainerBase"* %26, i64 0, i32 0
  %28 = load %"class.google::protobuf::Arena"*, %"class.google::protobuf::Arena"** %27, align 8, !tbaa !32
  br label %31

29:                                               ; preds = %19
  %30 = inttoptr i64 %24 to %"class.google::protobuf::Arena"*
  br label %31

31:                                               ; preds = %25, %29
  %32 = phi %"class.google::protobuf::Arena"* [ %28, %25 ], [ %30, %29 ]
  %33 = icmp eq %"class.google::protobuf::Arena"* %32, null
  br i1 %33, label %34, label %43

34:                                               ; preds = %31
  %35 = getelementptr inbounds %"class.tutorial::Person", %"class.tutorial::Person"* %0, i64 0, i32 4
  %36 = load %"class.google::protobuf::Timestamp"*, %"class.google::protobuf::Timestamp"** %35, align 8, !tbaa !20
  %37 = icmp eq %"class.google::protobuf::Timestamp"* %36, null
  br i1 %37, label %43, label %38

38:                                               ; preds = %34
  tail call void @_ZN6google8protobuf9TimestampD1Ev(%"class.google::protobuf::Timestamp"* noundef nonnull align 8 dereferenceable(32) %36) #24
  %39 = bitcast %"class.google::protobuf::Timestamp"* %36 to i8*
  tail call void @_ZdlPv(i8* noundef %39) #26
  %40 = load i8*, i8** %3, align 8, !tbaa !5
  %41 = ptrtoint i8* %40 to i64
  %42 = and i64 %41, 1
  br label %43

43:                                               ; preds = %38, %34, %31
  %44 = phi i64 [ %42, %38 ], [ %22, %34 ], [ %22, %31 ]
  %45 = phi i64 [ %41, %38 ], [ %21, %34 ], [ %21, %31 ]
  %46 = getelementptr inbounds %"class.tutorial::Person", %"class.tutorial::Person"* %0, i64 0, i32 4
  store %"class.google::protobuf::Timestamp"* null, %"class.google::protobuf::Timestamp"** %46, align 8, !tbaa !20
  %47 = getelementptr inbounds %"class.tutorial::Person", %"class.tutorial::Person"* %0, i64 0, i32 5
  store i32 0, i32* %47, align 8, !tbaa !56
  %48 = icmp eq i64 %44, 0
  br i1 %48, label %59, label %49

49:                                               ; preds = %43
  %50 = and i64 %45, -2
  %51 = inttoptr i64 %50 to %"struct.google::protobuf::internal::InternalMetadata::Container"*
  %52 = getelementptr inbounds %"struct.google::protobuf::internal::InternalMetadata::Container", %"struct.google::protobuf::internal::InternalMetadata::Container"* %51, i64 0, i32 1
  %53 = getelementptr inbounds %"class.google::protobuf::UnknownFieldSet", %"class.google::protobuf::UnknownFieldSet"* %52, i64 0, i32 0, i32 0, i32 0, i32 0, i32 0
  %54 = load %"class.google::protobuf::UnknownField"*, %"class.google::protobuf::UnknownField"** %53, align 8, !tbaa !27
  %55 = getelementptr inbounds %"struct.google::protobuf::internal::InternalMetadata::Container", %"struct.google::protobuf::internal::InternalMetadata::Container"* %51, i64 0, i32 1, i32 0, i32 0, i32 0, i32 0, i32 1
  %56 = load %"class.google::protobuf::UnknownField"*, %"class.google::protobuf::UnknownField"** %55, align 8, !tbaa !27
  %57 = icmp eq %"class.google::protobuf::UnknownField"* %54, %56
  br i1 %57, label %59, label %58

58:                                               ; preds = %49
  tail call void @_ZN6google8protobuf15UnknownFieldSet13ClearFallbackEv(%"class.google::protobuf::UnknownFieldSet"* noundef nonnull align 8 dereferenceable(24) %52)
  br label %59

59:                                               ; preds = %43, %49, %58
  ret void
}

; Function Attrs: uwtable
define dso_local noundef i8* @_ZN8tutorial6Person14_InternalParseEPKcPN6google8protobuf8internal12ParseContextE(%"class.tutorial::Person"* noundef nonnull align 8 dereferenceable(72) %0, i8* noundef %1, %"class.google::protobuf::internal::ParseContext"* noundef %2) unnamed_addr #3 align 2 {
  %4 = alloca i8*, align 8
  store i8* %1, i8** %4, align 8, !tbaa !27
  %5 = getelementptr inbounds %"class.google::protobuf::internal::ParseContext", %"class.google::protobuf::internal::ParseContext"* %2, i64 0, i32 0
  %6 = getelementptr inbounds %"class.google::protobuf::internal::ParseContext", %"class.google::protobuf::internal::ParseContext"* %2, i64 0, i32 2
  %7 = getelementptr inbounds %"class.tutorial::Person", %"class.tutorial::Person"* %0, i64 0, i32 4
  %8 = getelementptr inbounds %"class.tutorial::Person", %"class.tutorial::Person"* %0, i64 0, i32 0, i32 0, i32 1, i32 0
  %9 = getelementptr inbounds %"class.google::protobuf::internal::ParseContext", %"class.google::protobuf::internal::ParseContext"* %2, i64 0, i32 1
  %10 = getelementptr inbounds %"class.google::protobuf::internal::ParseContext", %"class.google::protobuf::internal::ParseContext"* %2, i64 0, i32 0, i32 8
  %11 = getelementptr inbounds %"class.google::protobuf::internal::ParseContext", %"class.google::protobuf::internal::ParseContext"* %2, i64 0, i32 0, i32 4
  %12 = getelementptr inbounds %"class.google::protobuf::internal::ParseContext", %"class.google::protobuf::internal::ParseContext"* %2, i64 0, i32 0, i32 1
  %13 = getelementptr inbounds %"class.google::protobuf::internal::ParseContext", %"class.google::protobuf::internal::ParseContext"* %2, i64 0, i32 0, i32 0
  %14 = getelementptr inbounds %"class.tutorial::Person", %"class.tutorial::Person"* %0, i64 0, i32 1
  %15 = getelementptr inbounds %"class.google::protobuf::RepeatedPtrField", %"class.google::protobuf::RepeatedPtrField"* %14, i64 0, i32 0
  %16 = getelementptr inbounds %"class.tutorial::Person", %"class.tutorial::Person"* %0, i64 0, i32 1, i32 0, i32 3
  %17 = getelementptr inbounds %"class.tutorial::Person", %"class.tutorial::Person"* %0, i64 0, i32 1, i32 0, i32 1
  %18 = getelementptr inbounds %"class.tutorial::Person", %"class.tutorial::Person"* %0, i64 0, i32 1, i32 0, i32 2
  %19 = getelementptr inbounds %"class.google::protobuf::RepeatedPtrField", %"class.google::protobuf::RepeatedPtrField"* %14, i64 0, i32 0, i32 0
  %20 = getelementptr inbounds %"class.tutorial::Person", %"class.tutorial::Person"* %0, i64 0, i32 3
  %21 = getelementptr inbounds %"struct.google::protobuf::internal::ArenaStringPtr", %"struct.google::protobuf::internal::ArenaStringPtr"* %20, i64 0, i32 0
  %22 = getelementptr inbounds %"class.tutorial::Person", %"class.tutorial::Person"* %0, i64 0, i32 5
  %23 = getelementptr inbounds %"class.tutorial::Person", %"class.tutorial::Person"* %0, i64 0, i32 2
  %24 = getelementptr inbounds %"struct.google::protobuf::internal::ArenaStringPtr", %"struct.google::protobuf::internal::ArenaStringPtr"* %23, i64 0, i32 0
  %25 = getelementptr inbounds %"class.tutorial::Person", %"class.tutorial::Person"* %0, i64 0, i32 0, i32 0, i32 1
  %26 = getelementptr inbounds %"class.google::protobuf::internal::InternalMetadata", %"class.google::protobuf::internal::InternalMetadata"* %25, i64 0, i32 0
  %27 = load i32, i32* %6, align 4, !tbaa !38
  %28 = call noundef zeroext i1 @_ZN6google8protobuf8internal18EpsCopyInputStream13DoneWithCheckEPPKci(%"class.google::protobuf::internal::EpsCopyInputStream"* noundef nonnull align 8 dereferenceable(88) %5, i8** noundef nonnull %4, i32 noundef %27)
  %29 = load i8*, i8** %4, align 8, !tbaa !27
  br i1 %28, label %335, label %30

30:                                               ; preds = %3, %330
  %31 = phi i8* [ %333, %330 ], [ %29, %3 ]
  %32 = load i8, i8* %31, align 1, !tbaa !37
  %33 = zext i8 %32 to i32
  %34 = icmp sgt i8 %32, -1
  %35 = getelementptr inbounds i8, i8* %31, i64 1
  br i1 %34, label %45, label %36

36:                                               ; preds = %30
  %37 = load i8, i8* %35, align 1, !tbaa !37
  %38 = zext i8 %37 to i32
  %39 = shl nuw nsw i32 %38, 7
  %40 = add nsw i32 %33, -128
  %41 = add nsw i32 %40, %39
  %42 = icmp sgt i8 %37, -1
  br i1 %42, label %43, label %48

43:                                               ; preds = %36
  %44 = getelementptr inbounds i8, i8* %31, i64 2
  br label %45

45:                                               ; preds = %30, %43
  %46 = phi i32 [ %41, %43 ], [ %33, %30 ]
  %47 = phi i8* [ %44, %43 ], [ %35, %30 ]
  store i8* %47, i8** %4, align 8, !tbaa !27
  br label %53

48:                                               ; preds = %36
  %49 = call { i8*, i32 } @_ZN6google8protobuf8internal15ReadTagFallbackEPKcj(i8* noundef nonnull %31, i32 noundef %41)
  %50 = extractvalue { i8*, i32 } %49, 0
  %51 = extractvalue { i8*, i32 } %49, 1
  store i8* %50, i8** %4, align 8, !tbaa !27
  %52 = icmp eq i8* %50, null
  br i1 %52, label %334, label %53, !prof !41

53:                                               ; preds = %45, %48
  %54 = phi i8* [ %47, %45 ], [ %50, %48 ]
  %55 = phi i32 [ %46, %45 ], [ %51, %48 ]
  %56 = lshr i32 %55, 3
  switch i32 %56, label %305 [
    i32 1, label %57
    i32 2, label %95
    i32 3, label %123
    i32 4, label %161
    i32 5, label %247
  ]

57:                                               ; preds = %53
  %58 = and i32 %55, 255
  %59 = icmp eq i32 %58, 10
  br i1 %59, label %60, label %305, !prof !12

60:                                               ; preds = %57
  %61 = load i8*, i8** %8, align 8, !tbaa !5
  %62 = ptrtoint i8* %61 to i64
  %63 = and i64 %62, 1
  %64 = icmp eq i64 %63, 0
  %65 = and i64 %62, -2
  br i1 %64, label %70, label %66, !prof !12

66:                                               ; preds = %60
  %67 = inttoptr i64 %65 to %"struct.google::protobuf::internal::InternalMetadata::ContainerBase"*
  %68 = getelementptr inbounds %"struct.google::protobuf::internal::InternalMetadata::ContainerBase", %"struct.google::protobuf::internal::InternalMetadata::ContainerBase"* %67, i64 0, i32 0
  %69 = load %"class.google::protobuf::Arena"*, %"class.google::protobuf::Arena"** %68, align 8, !tbaa !32
  br label %72

70:                                               ; preds = %60
  %71 = inttoptr i64 %65 to %"class.google::protobuf::Arena"*
  br label %72

72:                                               ; preds = %70, %66
  %73 = phi %"class.google::protobuf::Arena"* [ %69, %66 ], [ %71, %70 ]
  %74 = load %"class.std::__cxx11::basic_string"*, %"class.std::__cxx11::basic_string"** %24, align 8, !tbaa !18
  %75 = icmp eq %"class.std::__cxx11::basic_string"* %74, bitcast (%"class.google::protobuf::internal::ExplicitlyConstructed.20"* @_ZN6google8protobuf8internal26fixed_address_empty_stringB5cxx11E to %"class.std::__cxx11::basic_string"*)
  br i1 %75, label %76, label %79

76:                                               ; preds = %72
  call void @_ZN6google8protobuf8internal14ArenaStringPtr14CreateInstanceEPNS0_5ArenaEPKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE(%"struct.google::protobuf::internal::ArenaStringPtr"* noundef nonnull align 8 dereferenceable(8) %23, %"class.google::protobuf::Arena"* noundef %73, %"class.std::__cxx11::basic_string"* noundef bitcast (%"class.google::protobuf::internal::ExplicitlyConstructed.20"* @_ZN6google8protobuf8internal26fixed_address_empty_stringB5cxx11E to %"class.std::__cxx11::basic_string"*))
  %77 = load %"class.std::__cxx11::basic_string"*, %"class.std::__cxx11::basic_string"** %24, align 8, !tbaa !18
  %78 = load i8*, i8** %4, align 8, !tbaa !27
  br label %79

79:                                               ; preds = %72, %76
  %80 = phi i8* [ %78, %76 ], [ %54, %72 ]
  %81 = phi %"class.std::__cxx11::basic_string"* [ %77, %76 ], [ %74, %72 ]
  %82 = call noundef i8* @_ZN6google8protobuf8internal24InlineGreedyStringParserEPNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEPKcPNS1_12ParseContextE(%"class.std::__cxx11::basic_string"* noundef %81, i8* noundef %80, %"class.google::protobuf::internal::ParseContext"* noundef nonnull %2)
  store i8* %82, i8** %4, align 8, !tbaa !27
  %83 = getelementptr inbounds %"class.std::__cxx11::basic_string", %"class.std::__cxx11::basic_string"* %81, i64 0, i32 0, i32 0
  %84 = load i8*, i8** %83, align 8, !tbaa !34
  %85 = getelementptr inbounds %"class.std::__cxx11::basic_string", %"class.std::__cxx11::basic_string"* %81, i64 0, i32 1
  %86 = load i64, i64* %85, align 8, !tbaa !28
  %87 = icmp slt i64 %86, 0
  br i1 %87, label %88, label %89

88:                                               ; preds = %79
  call void @_ZN6google8protobuf11StringPiece18LogFatalSizeTooBigEmPKc(i64 noundef %86, i8* noundef getelementptr inbounds ([25 x i8], [25 x i8]* @.str.15, i64 0, i64 0))
  br label %89

89:                                               ; preds = %79, %88
  %90 = call noundef zeroext i1 @_ZN6google8protobuf8internal10VerifyUTF8ENS0_11StringPieceEPKc(i8* %84, i64 %86, i8* noundef getelementptr inbounds ([21 x i8], [21 x i8]* @.str.7, i64 0, i64 0))
  %91 = xor i1 %90, true
  %92 = load i8*, i8** %4, align 8
  %93 = icmp eq i8* %92, null
  %94 = select i1 %91, i1 true, i1 %93
  br i1 %94, label %334, label %330, !prof !57

95:                                               ; preds = %53
  %96 = and i32 %55, 255
  %97 = icmp eq i32 %96, 16
  br i1 %97, label %98, label %305, !prof !12

98:                                               ; preds = %95
  %99 = load i8, i8* %54, align 1, !tbaa !37
  %100 = zext i8 %99 to i32
  %101 = and i32 %100, 128
  %102 = icmp eq i32 %101, 0
  %103 = getelementptr inbounds i8, i8* %54, i64 1
  br i1 %102, label %114, label %104

104:                                              ; preds = %98
  %105 = load i8, i8* %103, align 1, !tbaa !37
  %106 = zext i8 %105 to i32
  %107 = shl nuw nsw i32 %106, 7
  %108 = add nsw i32 %100, -128
  %109 = add nuw nsw i32 %108, %107
  %110 = and i32 %106, 128
  %111 = icmp eq i32 %110, 0
  br i1 %111, label %112, label %117

112:                                              ; preds = %104
  %113 = getelementptr inbounds i8, i8* %54, i64 2
  br label %114

114:                                              ; preds = %98, %112
  %115 = phi i32 [ %109, %112 ], [ %100, %98 ]
  %116 = phi i8* [ %113, %112 ], [ %103, %98 ]
  store i8* %116, i8** %4, align 8, !tbaa !27
  store i32 %115, i32* %22, align 8, !tbaa !56
  br label %330

117:                                              ; preds = %104
  %118 = call { i8*, i64 } @_ZN6google8protobuf8internal17VarintParseSlow64EPKcj(i8* noundef nonnull %54, i32 noundef %109)
  %119 = extractvalue { i8*, i64 } %118, 0
  %120 = extractvalue { i8*, i64 } %118, 1
  store i8* %119, i8** %4, align 8, !tbaa !27
  %121 = trunc i64 %120 to i32
  store i32 %121, i32* %22, align 8, !tbaa !56
  %122 = icmp eq i8* %119, null
  br i1 %122, label %334, label %330

123:                                              ; preds = %53
  %124 = and i32 %55, 255
  %125 = icmp eq i32 %124, 26
  br i1 %125, label %126, label %305, !prof !12

126:                                              ; preds = %123
  %127 = load i8*, i8** %8, align 8, !tbaa !5
  %128 = ptrtoint i8* %127 to i64
  %129 = and i64 %128, 1
  %130 = icmp eq i64 %129, 0
  %131 = and i64 %128, -2
  br i1 %130, label %136, label %132, !prof !12

132:                                              ; preds = %126
  %133 = inttoptr i64 %131 to %"struct.google::protobuf::internal::InternalMetadata::ContainerBase"*
  %134 = getelementptr inbounds %"struct.google::protobuf::internal::InternalMetadata::ContainerBase", %"struct.google::protobuf::internal::InternalMetadata::ContainerBase"* %133, i64 0, i32 0
  %135 = load %"class.google::protobuf::Arena"*, %"class.google::protobuf::Arena"** %134, align 8, !tbaa !32
  br label %138

136:                                              ; preds = %126
  %137 = inttoptr i64 %131 to %"class.google::protobuf::Arena"*
  br label %138

138:                                              ; preds = %136, %132
  %139 = phi %"class.google::protobuf::Arena"* [ %135, %132 ], [ %137, %136 ]
  %140 = load %"class.std::__cxx11::basic_string"*, %"class.std::__cxx11::basic_string"** %21, align 8, !tbaa !18
  %141 = icmp eq %"class.std::__cxx11::basic_string"* %140, bitcast (%"class.google::protobuf::internal::ExplicitlyConstructed.20"* @_ZN6google8protobuf8internal26fixed_address_empty_stringB5cxx11E to %"class.std::__cxx11::basic_string"*)
  br i1 %141, label %142, label %145

142:                                              ; preds = %138
  call void @_ZN6google8protobuf8internal14ArenaStringPtr14CreateInstanceEPNS0_5ArenaEPKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE(%"struct.google::protobuf::internal::ArenaStringPtr"* noundef nonnull align 8 dereferenceable(8) %20, %"class.google::protobuf::Arena"* noundef %139, %"class.std::__cxx11::basic_string"* noundef bitcast (%"class.google::protobuf::internal::ExplicitlyConstructed.20"* @_ZN6google8protobuf8internal26fixed_address_empty_stringB5cxx11E to %"class.std::__cxx11::basic_string"*))
  %143 = load %"class.std::__cxx11::basic_string"*, %"class.std::__cxx11::basic_string"** %21, align 8, !tbaa !18
  %144 = load i8*, i8** %4, align 8, !tbaa !27
  br label %145

145:                                              ; preds = %138, %142
  %146 = phi i8* [ %144, %142 ], [ %54, %138 ]
  %147 = phi %"class.std::__cxx11::basic_string"* [ %143, %142 ], [ %140, %138 ]
  %148 = call noundef i8* @_ZN6google8protobuf8internal24InlineGreedyStringParserEPNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEPKcPNS1_12ParseContextE(%"class.std::__cxx11::basic_string"* noundef %147, i8* noundef %146, %"class.google::protobuf::internal::ParseContext"* noundef nonnull %2)
  store i8* %148, i8** %4, align 8, !tbaa !27
  %149 = getelementptr inbounds %"class.std::__cxx11::basic_string", %"class.std::__cxx11::basic_string"* %147, i64 0, i32 0, i32 0
  %150 = load i8*, i8** %149, align 8, !tbaa !34
  %151 = getelementptr inbounds %"class.std::__cxx11::basic_string", %"class.std::__cxx11::basic_string"* %147, i64 0, i32 1
  %152 = load i64, i64* %151, align 8, !tbaa !28
  %153 = icmp slt i64 %152, 0
  br i1 %153, label %154, label %155

154:                                              ; preds = %145
  call void @_ZN6google8protobuf11StringPiece18LogFatalSizeTooBigEmPKc(i64 noundef %152, i8* noundef getelementptr inbounds ([25 x i8], [25 x i8]* @.str.15, i64 0, i64 0))
  br label %155

155:                                              ; preds = %145, %154
  %156 = call noundef zeroext i1 @_ZN6google8protobuf8internal10VerifyUTF8ENS0_11StringPieceEPKc(i8* %150, i64 %152, i8* noundef getelementptr inbounds ([22 x i8], [22 x i8]* @.str.8, i64 0, i64 0))
  %157 = xor i1 %156, true
  %158 = load i8*, i8** %4, align 8
  %159 = icmp eq i8* %158, null
  %160 = select i1 %157, i1 true, i1 %159
  br i1 %160, label %334, label %330, !prof !57

161:                                              ; preds = %53
  %162 = and i32 %55, 255
  %163 = icmp eq i32 %162, 34
  br i1 %163, label %164, label %305, !prof !12

164:                                              ; preds = %161
  %165 = getelementptr inbounds i8, i8* %54, i64 -1
  br label %166

166:                                              ; preds = %244, %164
  %167 = phi i8* [ %228, %244 ], [ %165, %164 ]
  %168 = getelementptr inbounds i8, i8* %167, i64 1
  store i8* %168, i8** %4, align 8, !tbaa !27
  %169 = load %"struct.google::protobuf::internal::RepeatedPtrFieldBase::Rep"*, %"struct.google::protobuf::internal::RepeatedPtrFieldBase::Rep"** %16, align 8, !tbaa !58
  %170 = icmp eq %"struct.google::protobuf::internal::RepeatedPtrFieldBase::Rep"* %169, null
  br i1 %170, label %171, label %173

171:                                              ; preds = %166
  %172 = load i32, i32* %18, align 4, !tbaa !59
  br label %187

173:                                              ; preds = %166
  %174 = load i32, i32* %17, align 8, !tbaa !60
  %175 = getelementptr inbounds %"struct.google::protobuf::internal::RepeatedPtrFieldBase::Rep", %"struct.google::protobuf::internal::RepeatedPtrFieldBase::Rep"* %169, i64 0, i32 0
  %176 = load i32, i32* %175, align 8, !tbaa !61
  %177 = icmp slt i32 %174, %176
  br i1 %177, label %178, label %184

178:                                              ; preds = %173
  %179 = add nsw i32 %174, 1
  store i32 %179, i32* %17, align 8, !tbaa !60
  %180 = sext i32 %174 to i64
  %181 = getelementptr inbounds %"struct.google::protobuf::internal::RepeatedPtrFieldBase::Rep", %"struct.google::protobuf::internal::RepeatedPtrFieldBase::Rep"* %169, i64 0, i32 1, i64 %180
  %182 = bitcast i8** %181 to %"class.tutorial::Person_PhoneNumber"**
  %183 = load %"class.tutorial::Person_PhoneNumber"*, %"class.tutorial::Person_PhoneNumber"** %182, align 8, !tbaa !27
  br label %207

184:                                              ; preds = %173
  %185 = load i32, i32* %18, align 4, !tbaa !59
  %186 = icmp eq i32 %176, %185
  br i1 %186, label %187, label %193

187:                                              ; preds = %184, %171
  %188 = phi i32 [ %172, %171 ], [ %176, %184 ]
  %189 = add nsw i32 %188, 1
  call void @_ZN6google8protobuf8internal20RepeatedPtrFieldBase7ReserveEi(%"class.google::protobuf::internal::RepeatedPtrFieldBase"* noundef nonnull align 8 dereferenceable(24) %15, i32 noundef %189)
  %190 = load %"struct.google::protobuf::internal::RepeatedPtrFieldBase::Rep"*, %"struct.google::protobuf::internal::RepeatedPtrFieldBase::Rep"** %16, align 8, !tbaa !58
  %191 = getelementptr inbounds %"struct.google::protobuf::internal::RepeatedPtrFieldBase::Rep", %"struct.google::protobuf::internal::RepeatedPtrFieldBase::Rep"* %190, i64 0, i32 0
  %192 = load i32, i32* %191, align 8, !tbaa !61
  br label %193

193:                                              ; preds = %187, %184
  %194 = phi i32 [ %192, %187 ], [ %176, %184 ]
  %195 = phi %"struct.google::protobuf::internal::RepeatedPtrFieldBase::Rep"* [ %190, %187 ], [ %169, %184 ]
  %196 = getelementptr inbounds %"struct.google::protobuf::internal::RepeatedPtrFieldBase::Rep", %"struct.google::protobuf::internal::RepeatedPtrFieldBase::Rep"* %195, i64 0, i32 0
  %197 = add nsw i32 %194, 1
  store i32 %197, i32* %196, align 8, !tbaa !61
  %198 = load %"class.google::protobuf::Arena"*, %"class.google::protobuf::Arena"** %19, align 8, !tbaa !16
  %199 = call noundef %"class.tutorial::Person_PhoneNumber"* @_ZN6google8protobuf5Arena18CreateMaybeMessageIN8tutorial18Person_PhoneNumberEJEEEPT_PS1_DpOT0_(%"class.google::protobuf::Arena"* noundef %198)
  %200 = load %"struct.google::protobuf::internal::RepeatedPtrFieldBase::Rep"*, %"struct.google::protobuf::internal::RepeatedPtrFieldBase::Rep"** %16, align 8, !tbaa !58
  %201 = load i32, i32* %17, align 8, !tbaa !60
  %202 = add nsw i32 %201, 1
  store i32 %202, i32* %17, align 8, !tbaa !60
  %203 = sext i32 %201 to i64
  %204 = getelementptr inbounds %"struct.google::protobuf::internal::RepeatedPtrFieldBase::Rep", %"struct.google::protobuf::internal::RepeatedPtrFieldBase::Rep"* %200, i64 0, i32 1, i64 %203
  %205 = bitcast i8** %204 to %"class.tutorial::Person_PhoneNumber"**
  store %"class.tutorial::Person_PhoneNumber"* %199, %"class.tutorial::Person_PhoneNumber"** %205, align 8, !tbaa !27
  %206 = load i8*, i8** %4, align 8, !tbaa !27
  br label %207

207:                                              ; preds = %178, %193
  %208 = phi i8* [ %168, %178 ], [ %206, %193 ]
  %209 = phi %"class.tutorial::Person_PhoneNumber"* [ %183, %178 ], [ %199, %193 ]
  %210 = load i8, i8* %208, align 1, !tbaa !37
  %211 = zext i8 %210 to i32
  %212 = icmp sgt i8 %210, -1
  br i1 %212, label %213, label %215

213:                                              ; preds = %207
  %214 = getelementptr inbounds i8, i8* %208, i64 1
  br label %220

215:                                              ; preds = %207
  %216 = call { i8*, i32 } @_ZN6google8protobuf8internal16ReadSizeFallbackEPKcj(i8* noundef nonnull %208, i32 noundef %211)
  %217 = extractvalue { i8*, i32 } %216, 0
  %218 = extractvalue { i8*, i32 } %216, 1
  %219 = icmp eq i8* %217, null
  br i1 %219, label %334, label %220

220:                                              ; preds = %215, %213
  %221 = phi i32 [ %211, %213 ], [ %218, %215 ]
  %222 = phi i8* [ %214, %213 ], [ %217, %215 ]
  %223 = call noundef i32 @_ZN6google8protobuf8internal18EpsCopyInputStream9PushLimitEPKci(%"class.google::protobuf::internal::EpsCopyInputStream"* noundef nonnull align 8 dereferenceable(88) %5, i8* noundef nonnull %222, i32 noundef %221)
  %224 = load i32, i32* %9, align 8, !tbaa !63
  %225 = add nsw i32 %224, -1
  store i32 %225, i32* %9, align 8, !tbaa !63
  %226 = icmp slt i32 %224, 1
  br i1 %226, label %334, label %227

227:                                              ; preds = %220
  %228 = call noundef i8* @_ZN8tutorial18Person_PhoneNumber14_InternalParseEPKcPN6google8protobuf8internal12ParseContextE(%"class.tutorial::Person_PhoneNumber"* noundef nonnull align 8 dereferenceable(32) %209, i8* noundef nonnull %222, %"class.google::protobuf::internal::ParseContext"* noundef nonnull %2)
  %229 = icmp eq i8* %228, null
  br i1 %229, label %334, label %230, !prof !41

230:                                              ; preds = %227
  %231 = load i32, i32* %9, align 8, !tbaa !63
  %232 = add nsw i32 %231, 1
  store i32 %232, i32* %9, align 8, !tbaa !63
  %233 = load i32, i32* %10, align 8, !tbaa !42
  %234 = icmp eq i32 %233, 0
  br i1 %234, label %235, label %334, !prof !12

235:                                              ; preds = %230
  %236 = load i32, i32* %11, align 4, !tbaa !64
  %237 = add nsw i32 %236, %223
  store i32 %237, i32* %11, align 4, !tbaa !64
  %238 = load i8*, i8** %12, align 8, !tbaa !65
  %239 = icmp slt i32 %237, 0
  %240 = select i1 %239, i32 %237, i32 0
  %241 = sext i32 %240 to i64
  %242 = getelementptr inbounds i8, i8* %238, i64 %241
  store i8* %242, i8** %13, align 8, !tbaa !66
  store i8* %228, i8** %4, align 8, !tbaa !27
  %243 = icmp ugt i8* %242, %228
  br i1 %243, label %244, label %330, !llvm.loop !67

244:                                              ; preds = %235
  %245 = load i8, i8* %228, align 1, !tbaa !37
  %246 = icmp eq i8 %245, 34
  br i1 %246, label %166, label %330, !llvm.loop !67

247:                                              ; preds = %53
  %248 = and i32 %55, 255
  %249 = icmp eq i32 %248, 42
  br i1 %249, label %250, label %305, !prof !12

250:                                              ; preds = %247
  %251 = load %"class.google::protobuf::Timestamp"*, %"class.google::protobuf::Timestamp"** %7, align 8, !tbaa !20
  %252 = icmp eq %"class.google::protobuf::Timestamp"* %251, null
  br i1 %252, label %253, label %269

253:                                              ; preds = %250
  %254 = load i8*, i8** %8, align 8, !tbaa !5
  %255 = ptrtoint i8* %254 to i64
  %256 = and i64 %255, 1
  %257 = icmp eq i64 %256, 0
  %258 = and i64 %255, -2
  br i1 %257, label %263, label %259, !prof !12

259:                                              ; preds = %253
  %260 = inttoptr i64 %258 to %"struct.google::protobuf::internal::InternalMetadata::ContainerBase"*
  %261 = getelementptr inbounds %"struct.google::protobuf::internal::InternalMetadata::ContainerBase", %"struct.google::protobuf::internal::InternalMetadata::ContainerBase"* %260, i64 0, i32 0
  %262 = load %"class.google::protobuf::Arena"*, %"class.google::protobuf::Arena"** %261, align 8, !tbaa !32
  br label %265

263:                                              ; preds = %253
  %264 = inttoptr i64 %258 to %"class.google::protobuf::Arena"*
  br label %265

265:                                              ; preds = %263, %259
  %266 = phi %"class.google::protobuf::Arena"* [ %262, %259 ], [ %264, %263 ]
  %267 = call noundef %"class.google::protobuf::Timestamp"* @_ZN6google8protobuf5Arena18CreateMaybeMessageINS0_9TimestampEJEEEPT_PS1_DpOT0_(%"class.google::protobuf::Arena"* noundef %266)
  store %"class.google::protobuf::Timestamp"* %267, %"class.google::protobuf::Timestamp"** %7, align 8, !tbaa !20
  %268 = load i8*, i8** %4, align 8, !tbaa !27
  br label %269

269:                                              ; preds = %250, %265
  %270 = phi i8* [ %268, %265 ], [ %54, %250 ]
  %271 = phi %"class.google::protobuf::Timestamp"* [ %267, %265 ], [ %251, %250 ]
  %272 = load i8, i8* %270, align 1, !tbaa !37
  %273 = zext i8 %272 to i32
  %274 = icmp sgt i8 %272, -1
  br i1 %274, label %275, label %277

275:                                              ; preds = %269
  %276 = getelementptr inbounds i8, i8* %270, i64 1
  br label %282

277:                                              ; preds = %269
  %278 = call { i8*, i32 } @_ZN6google8protobuf8internal16ReadSizeFallbackEPKcj(i8* noundef nonnull %270, i32 noundef %273)
  %279 = extractvalue { i8*, i32 } %278, 0
  %280 = extractvalue { i8*, i32 } %278, 1
  %281 = icmp eq i8* %279, null
  br i1 %281, label %334, label %282

282:                                              ; preds = %277, %275
  %283 = phi i32 [ %273, %275 ], [ %280, %277 ]
  %284 = phi i8* [ %276, %275 ], [ %279, %277 ]
  %285 = call noundef i32 @_ZN6google8protobuf8internal18EpsCopyInputStream9PushLimitEPKci(%"class.google::protobuf::internal::EpsCopyInputStream"* noundef nonnull align 8 dereferenceable(88) %5, i8* noundef nonnull %284, i32 noundef %283)
  %286 = load i32, i32* %9, align 8, !tbaa !63
  %287 = add nsw i32 %286, -1
  store i32 %287, i32* %9, align 8, !tbaa !63
  %288 = icmp slt i32 %286, 1
  br i1 %288, label %334, label %289

289:                                              ; preds = %282
  %290 = call noundef i8* @_ZN6google8protobuf9Timestamp14_InternalParseEPKcPNS0_8internal12ParseContextE(%"class.google::protobuf::Timestamp"* noundef nonnull align 8 dereferenceable(32) %271, i8* noundef nonnull %284, %"class.google::protobuf::internal::ParseContext"* noundef nonnull %2)
  %291 = icmp eq i8* %290, null
  br i1 %291, label %334, label %292, !prof !41

292:                                              ; preds = %289
  %293 = load i32, i32* %9, align 8, !tbaa !63
  %294 = add nsw i32 %293, 1
  store i32 %294, i32* %9, align 8, !tbaa !63
  %295 = load i32, i32* %10, align 8, !tbaa !42
  %296 = icmp eq i32 %295, 0
  br i1 %296, label %297, label %334, !prof !12

297:                                              ; preds = %292
  %298 = load i32, i32* %11, align 4, !tbaa !64
  %299 = add nsw i32 %298, %285
  store i32 %299, i32* %11, align 4, !tbaa !64
  %300 = load i8*, i8** %12, align 8, !tbaa !65
  %301 = icmp slt i32 %299, 0
  %302 = select i1 %301, i32 %299, i32 0
  %303 = sext i32 %302 to i64
  %304 = getelementptr inbounds i8, i8* %300, i64 %303
  store i8* %304, i8** %13, align 8, !tbaa !66
  store i8* %290, i8** %4, align 8, !tbaa !27
  br label %330

305:                                              ; preds = %53, %247, %161, %123, %95, %57
  %306 = and i32 %55, 7
  %307 = icmp eq i32 %306, 4
  %308 = icmp eq i32 %55, 0
  %309 = or i1 %308, %307
  br i1 %309, label %310, label %312

310:                                              ; preds = %305
  %311 = add i32 %55, -1
  store i32 %311, i32* %10, align 8, !tbaa !42
  br label %335

312:                                              ; preds = %305
  %313 = zext i32 %55 to i64
  %314 = load i8*, i8** %26, align 8, !tbaa !5
  %315 = ptrtoint i8* %314 to i64
  %316 = and i64 %315, 1
  %317 = icmp eq i64 %316, 0
  br i1 %317, label %322, label %318, !prof !41

318:                                              ; preds = %312
  %319 = and i64 %315, -2
  %320 = inttoptr i64 %319 to %"struct.google::protobuf::internal::InternalMetadata::Container"*
  %321 = getelementptr inbounds %"struct.google::protobuf::internal::InternalMetadata::Container", %"struct.google::protobuf::internal::InternalMetadata::Container"* %320, i64 0, i32 1
  br label %325

322:                                              ; preds = %312
  %323 = call noundef %"class.google::protobuf::UnknownFieldSet"* @_ZN6google8protobuf8internal16InternalMetadata27mutable_unknown_fields_slowINS0_15UnknownFieldSetEEEPT_v(%"class.google::protobuf::internal::InternalMetadata"* noundef nonnull align 8 dereferenceable(8) %25)
  %324 = load i8*, i8** %4, align 8, !tbaa !27
  br label %325

325:                                              ; preds = %318, %322
  %326 = phi i8* [ %54, %318 ], [ %324, %322 ]
  %327 = phi %"class.google::protobuf::UnknownFieldSet"* [ %321, %318 ], [ %323, %322 ]
  %328 = call noundef i8* @_ZN6google8protobuf8internal17UnknownFieldParseEmPNS0_15UnknownFieldSetEPKcPNS1_12ParseContextE(i64 noundef %313, %"class.google::protobuf::UnknownFieldSet"* noundef %327, i8* noundef %326, %"class.google::protobuf::internal::ParseContext"* noundef nonnull %2)
  store i8* %328, i8** %4, align 8, !tbaa !27
  %329 = icmp eq i8* %328, null
  br i1 %329, label %334, label %330

330:                                              ; preds = %235, %244, %155, %89, %297, %114, %325, %117
  %331 = load i32, i32* %6, align 4, !tbaa !38
  %332 = call noundef zeroext i1 @_ZN6google8protobuf8internal18EpsCopyInputStream13DoneWithCheckEPPKci(%"class.google::protobuf::internal::EpsCopyInputStream"* noundef nonnull align 8 dereferenceable(88) %5, i8** noundef nonnull %4, i32 noundef %331)
  %333 = load i8*, i8** %4, align 8, !tbaa !27
  br i1 %332, label %335, label %30

334:                                              ; preds = %325, %117, %155, %89, %48, %292, %289, %282, %277, %230, %227, %220, %215
  store i8* null, i8** %4, align 8, !tbaa !27
  br label %335

335:                                              ; preds = %330, %3, %334, %310
  %336 = phi i8* [ %54, %310 ], [ null, %334 ], [ %29, %3 ], [ %333, %330 ]
  ret i8* %336
}

; Function Attrs: mustprogress uwtable
define dso_local noundef i8* @_ZNK8tutorial6Person18_InternalSerializeEPhPN6google8protobuf2io19EpsCopyOutputStreamE(%"class.tutorial::Person"* noundef nonnull align 8 dereferenceable(72) %0, i8* noundef %1, %"class.google::protobuf::io::EpsCopyOutputStream"* noundef %2) unnamed_addr #4 align 2 {
  %4 = getelementptr inbounds %"class.tutorial::Person", %"class.tutorial::Person"* %0, i64 0, i32 2, i32 0
  %5 = load %"class.std::__cxx11::basic_string"*, %"class.std::__cxx11::basic_string"** %4, align 8, !tbaa !18
  %6 = getelementptr inbounds %"class.std::__cxx11::basic_string", %"class.std::__cxx11::basic_string"* %5, i64 0, i32 1
  %7 = load i64, i64* %6, align 8, !tbaa !28
  %8 = icmp eq i64 %7, 0
  br i1 %8, label %35, label %9

9:                                                ; preds = %3
  %10 = getelementptr inbounds %"class.std::__cxx11::basic_string", %"class.std::__cxx11::basic_string"* %5, i64 0, i32 0, i32 0
  %11 = load i8*, i8** %10, align 8, !tbaa !34
  %12 = trunc i64 %7 to i32
  %13 = tail call noundef zeroext i1 @_ZN6google8protobuf8internal14WireFormatLite16VerifyUtf8StringEPKciNS2_9OperationES4_(i8* noundef %11, i32 noundef %12, i32 noundef 1, i8* noundef getelementptr inbounds ([21 x i8], [21 x i8]* @.str.7, i64 0, i64 0))
  %14 = load %"class.std::__cxx11::basic_string"*, %"class.std::__cxx11::basic_string"** %4, align 8, !tbaa !18
  %15 = getelementptr inbounds %"class.std::__cxx11::basic_string", %"class.std::__cxx11::basic_string"* %14, i64 0, i32 1
  %16 = load i64, i64* %15, align 8, !tbaa !28
  %17 = icmp sgt i64 %16, 127
  br i1 %17, label %26, label %18, !prof !41

18:                                               ; preds = %9
  %19 = getelementptr inbounds %"class.google::protobuf::io::EpsCopyOutputStream", %"class.google::protobuf::io::EpsCopyOutputStream"* %2, i64 0, i32 0
  %20 = load i8*, i8** %19, align 8, !tbaa !44
  %21 = ptrtoint i8* %20 to i64
  %22 = ptrtoint i8* %1 to i64
  %23 = sub i64 14, %22
  %24 = add i64 %23, %21
  %25 = icmp slt i64 %24, %16
  br i1 %25, label %26, label %28, !prof !41

26:                                               ; preds = %18, %9
  %27 = tail call noundef i8* @_ZN6google8protobuf2io19EpsCopyOutputStream30WriteStringMaybeAliasedOutlineEjRKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEPh(%"class.google::protobuf::io::EpsCopyOutputStream"* noundef nonnull align 8 dereferenceable(59) %2, i32 noundef 1, %"class.std::__cxx11::basic_string"* noundef nonnull align 8 dereferenceable(32) %14, i8* noundef %1)
  br label %35

28:                                               ; preds = %18
  store i8 10, i8* %1, align 1, !tbaa !37
  %29 = getelementptr inbounds i8, i8* %1, i64 1
  %30 = trunc i64 %16 to i8
  %31 = getelementptr inbounds i8, i8* %1, i64 2
  store i8 %30, i8* %29, align 1, !tbaa !37
  %32 = getelementptr inbounds %"class.std::__cxx11::basic_string", %"class.std::__cxx11::basic_string"* %14, i64 0, i32 0, i32 0
  %33 = load i8*, i8** %32, align 8, !tbaa !34
  tail call void @llvm.memcpy.p0i8.p0i8.i64(i8* nonnull align 1 %31, i8* align 1 %33, i64 %16, i1 false)
  %34 = getelementptr inbounds i8, i8* %31, i64 %16
  br label %35

35:                                               ; preds = %28, %26, %3
  %36 = phi i8* [ %1, %3 ], [ %27, %26 ], [ %34, %28 ]
  %37 = getelementptr inbounds %"class.tutorial::Person", %"class.tutorial::Person"* %0, i64 0, i32 5
  %38 = load i32, i32* %37, align 8, !tbaa !56
  %39 = icmp eq i32 %38, 0
  br i1 %39, label %77, label %40

40:                                               ; preds = %35
  %41 = getelementptr inbounds %"class.google::protobuf::io::EpsCopyOutputStream", %"class.google::protobuf::io::EpsCopyOutputStream"* %2, i64 0, i32 0
  %42 = load i8*, i8** %41, align 8, !tbaa !44
  %43 = icmp ugt i8* %42, %36
  br i1 %43, label %47, label %44, !prof !12

44:                                               ; preds = %40
  %45 = tail call noundef i8* @_ZN6google8protobuf2io19EpsCopyOutputStream19EnsureSpaceFallbackEPh(%"class.google::protobuf::io::EpsCopyOutputStream"* noundef nonnull align 8 dereferenceable(59) %2, i8* noundef %36)
  %46 = load i32, i32* %37, align 8, !tbaa !56
  br label %47

47:                                               ; preds = %40, %44
  %48 = phi i32 [ %46, %44 ], [ %38, %40 ]
  %49 = phi i8* [ %45, %44 ], [ %36, %40 ]
  store i8 16, i8* %49, align 1, !tbaa !37
  %50 = getelementptr inbounds i8, i8* %49, i64 1
  %51 = icmp ult i32 %48, 128
  %52 = trunc i32 %48 to i8
  br i1 %51, label %53, label %55

53:                                               ; preds = %47
  store i8 %52, i8* %50, align 1, !tbaa !37
  %54 = getelementptr inbounds i8, i8* %49, i64 2
  br label %77

55:                                               ; preds = %47
  %56 = sext i32 %48 to i64
  %57 = or i8 %52, -128
  store i8 %57, i8* %50, align 1, !tbaa !37
  %58 = lshr i64 %56, 7
  %59 = icmp ult i32 %48, 16384
  br i1 %59, label %60, label %64

60:                                               ; preds = %55
  %61 = trunc i64 %58 to i8
  %62 = getelementptr inbounds i8, i8* %49, i64 2
  store i8 %61, i8* %62, align 1, !tbaa !37
  %63 = getelementptr inbounds i8, i8* %49, i64 3
  br label %77

64:                                               ; preds = %55
  %65 = getelementptr inbounds i8, i8* %49, i64 2
  br label %66

66:                                               ; preds = %66, %64
  %67 = phi i64 [ %58, %64 ], [ %71, %66 ]
  %68 = phi i8* [ %65, %64 ], [ %72, %66 ]
  %69 = trunc i64 %67 to i8
  %70 = or i8 %69, -128
  store i8 %70, i8* %68, align 1, !tbaa !37
  %71 = lshr i64 %67, 7
  %72 = getelementptr inbounds i8, i8* %68, i64 1
  %73 = icmp ugt i64 %67, 16383
  br i1 %73, label %66, label %74, !prof !41, !llvm.loop !47

74:                                               ; preds = %66
  %75 = trunc i64 %71 to i8
  %76 = getelementptr inbounds i8, i8* %68, i64 2
  store i8 %75, i8* %72, align 1, !tbaa !37
  br label %77

77:                                               ; preds = %74, %60, %53, %35
  %78 = phi i8* [ %36, %35 ], [ %54, %53 ], [ %63, %60 ], [ %76, %74 ]
  %79 = getelementptr inbounds %"class.tutorial::Person", %"class.tutorial::Person"* %0, i64 0, i32 3, i32 0
  %80 = load %"class.std::__cxx11::basic_string"*, %"class.std::__cxx11::basic_string"** %79, align 8, !tbaa !18
  %81 = getelementptr inbounds %"class.std::__cxx11::basic_string", %"class.std::__cxx11::basic_string"* %80, i64 0, i32 1
  %82 = load i64, i64* %81, align 8, !tbaa !28
  %83 = icmp eq i64 %82, 0
  br i1 %83, label %110, label %84

84:                                               ; preds = %77
  %85 = getelementptr inbounds %"class.std::__cxx11::basic_string", %"class.std::__cxx11::basic_string"* %80, i64 0, i32 0, i32 0
  %86 = load i8*, i8** %85, align 8, !tbaa !34
  %87 = trunc i64 %82 to i32
  %88 = tail call noundef zeroext i1 @_ZN6google8protobuf8internal14WireFormatLite16VerifyUtf8StringEPKciNS2_9OperationES4_(i8* noundef %86, i32 noundef %87, i32 noundef 1, i8* noundef getelementptr inbounds ([22 x i8], [22 x i8]* @.str.8, i64 0, i64 0))
  %89 = load %"class.std::__cxx11::basic_string"*, %"class.std::__cxx11::basic_string"** %79, align 8, !tbaa !18
  %90 = getelementptr inbounds %"class.std::__cxx11::basic_string", %"class.std::__cxx11::basic_string"* %89, i64 0, i32 1
  %91 = load i64, i64* %90, align 8, !tbaa !28
  %92 = icmp sgt i64 %91, 127
  br i1 %92, label %101, label %93, !prof !41

93:                                               ; preds = %84
  %94 = getelementptr inbounds %"class.google::protobuf::io::EpsCopyOutputStream", %"class.google::protobuf::io::EpsCopyOutputStream"* %2, i64 0, i32 0
  %95 = load i8*, i8** %94, align 8, !tbaa !44
  %96 = ptrtoint i8* %95 to i64
  %97 = ptrtoint i8* %78 to i64
  %98 = sub i64 14, %97
  %99 = add i64 %98, %96
  %100 = icmp slt i64 %99, %91
  br i1 %100, label %101, label %103, !prof !41

101:                                              ; preds = %93, %84
  %102 = tail call noundef i8* @_ZN6google8protobuf2io19EpsCopyOutputStream30WriteStringMaybeAliasedOutlineEjRKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEPh(%"class.google::protobuf::io::EpsCopyOutputStream"* noundef nonnull align 8 dereferenceable(59) %2, i32 noundef 3, %"class.std::__cxx11::basic_string"* noundef nonnull align 8 dereferenceable(32) %89, i8* noundef %78)
  br label %110

103:                                              ; preds = %93
  store i8 26, i8* %78, align 1, !tbaa !37
  %104 = getelementptr inbounds i8, i8* %78, i64 1
  %105 = trunc i64 %91 to i8
  %106 = getelementptr inbounds i8, i8* %78, i64 2
  store i8 %105, i8* %104, align 1, !tbaa !37
  %107 = getelementptr inbounds %"class.std::__cxx11::basic_string", %"class.std::__cxx11::basic_string"* %89, i64 0, i32 0, i32 0
  %108 = load i8*, i8** %107, align 8, !tbaa !34
  tail call void @llvm.memcpy.p0i8.p0i8.i64(i8* nonnull align 1 %106, i8* align 1 %108, i64 %91, i1 false)
  %109 = getelementptr inbounds i8, i8* %106, i64 %91
  br label %110

110:                                              ; preds = %103, %101, %77
  %111 = phi i8* [ %78, %77 ], [ %102, %101 ], [ %109, %103 ]
  %112 = getelementptr inbounds %"class.tutorial::Person", %"class.tutorial::Person"* %0, i64 0, i32 1, i32 0, i32 1
  %113 = load i32, i32* %112, align 8, !tbaa !60
  %114 = icmp eq i32 %113, 0
  br i1 %114, label %118, label %115

115:                                              ; preds = %110
  %116 = getelementptr inbounds %"class.google::protobuf::io::EpsCopyOutputStream", %"class.google::protobuf::io::EpsCopyOutputStream"* %2, i64 0, i32 0
  %117 = getelementptr inbounds %"class.tutorial::Person", %"class.tutorial::Person"* %0, i64 0, i32 1, i32 0
  br label %125

118:                                              ; preds = %163, %110
  %119 = phi i8* [ %111, %110 ], [ %165, %163 ]
  %120 = icmp ne %"class.tutorial::Person"* %0, bitcast (%"class.tutorial::PersonDefaultTypeInternal"* @_ZN8tutorial25_Person_default_instance_E to %"class.tutorial::Person"*)
  %121 = getelementptr inbounds %"class.tutorial::Person", %"class.tutorial::Person"* %0, i64 0, i32 4
  %122 = load %"class.google::protobuf::Timestamp"*, %"class.google::protobuf::Timestamp"** %121, align 8
  %123 = icmp ne %"class.google::protobuf::Timestamp"* %122, null
  %124 = select i1 %120, i1 %123, i1 false
  br i1 %124, label %168, label %209

125:                                              ; preds = %115, %163
  %126 = phi i8* [ %111, %115 ], [ %165, %163 ]
  %127 = phi i32 [ 0, %115 ], [ %166, %163 ]
  %128 = load i8*, i8** %116, align 8, !tbaa !44
  %129 = icmp ugt i8* %128, %126
  br i1 %129, label %132, label %130, !prof !12

130:                                              ; preds = %125
  %131 = tail call noundef i8* @_ZN6google8protobuf2io19EpsCopyOutputStream19EnsureSpaceFallbackEPh(%"class.google::protobuf::io::EpsCopyOutputStream"* noundef nonnull align 8 dereferenceable(59) %2, i8* noundef %126)
  br label %132

132:                                              ; preds = %125, %130
  %133 = phi i8* [ %131, %130 ], [ %126, %125 ]
  %134 = tail call noundef nonnull align 8 dereferenceable(32) %"class.tutorial::Person_PhoneNumber"* @_ZNK6google8protobuf8internal20RepeatedPtrFieldBase3GetINS0_16RepeatedPtrFieldIN8tutorial18Person_PhoneNumberEE11TypeHandlerEEERKNT_4TypeEi(%"class.google::protobuf::internal::RepeatedPtrFieldBase"* noundef nonnull align 8 dereferenceable(24) %117, i32 noundef %127)
  store i8 34, i8* %133, align 1, !tbaa !37
  %135 = getelementptr inbounds i8, i8* %133, i64 1
  %136 = getelementptr inbounds %"class.tutorial::Person_PhoneNumber", %"class.tutorial::Person_PhoneNumber"* %134, i64 0, i32 3, i32 0, i32 0, i32 0
  %137 = load atomic i32, i32* %136 monotonic, align 4
  %138 = icmp ult i32 %137, 128
  %139 = trunc i32 %137 to i8
  br i1 %138, label %140, label %142

140:                                              ; preds = %132
  store i8 %139, i8* %135, align 1, !tbaa !37
  %141 = getelementptr inbounds i8, i8* %133, i64 2
  br label %163

142:                                              ; preds = %132
  %143 = or i8 %139, -128
  store i8 %143, i8* %135, align 1, !tbaa !37
  %144 = lshr i32 %137, 7
  %145 = icmp ult i32 %137, 16384
  br i1 %145, label %146, label %150

146:                                              ; preds = %142
  %147 = trunc i32 %144 to i8
  %148 = getelementptr inbounds i8, i8* %133, i64 2
  store i8 %147, i8* %148, align 1, !tbaa !37
  %149 = getelementptr inbounds i8, i8* %133, i64 3
  br label %163

150:                                              ; preds = %142
  %151 = getelementptr inbounds i8, i8* %133, i64 2
  br label %152

152:                                              ; preds = %152, %150
  %153 = phi i32 [ %144, %150 ], [ %157, %152 ]
  %154 = phi i8* [ %151, %150 ], [ %158, %152 ]
  %155 = trunc i32 %153 to i8
  %156 = or i8 %155, -128
  store i8 %156, i8* %154, align 1, !tbaa !37
  %157 = lshr i32 %153, 7
  %158 = getelementptr inbounds i8, i8* %154, i64 1
  %159 = icmp ugt i32 %153, 16383
  br i1 %159, label %152, label %160, !prof !41, !llvm.loop !68

160:                                              ; preds = %152
  %161 = trunc i32 %157 to i8
  %162 = getelementptr inbounds i8, i8* %154, i64 2
  store i8 %161, i8* %158, align 1, !tbaa !37
  br label %163

163:                                              ; preds = %140, %146, %160
  %164 = phi i8* [ %141, %140 ], [ %149, %146 ], [ %162, %160 ]
  %165 = tail call noundef i8* @_ZNK8tutorial18Person_PhoneNumber18_InternalSerializeEPhPN6google8protobuf2io19EpsCopyOutputStreamE(%"class.tutorial::Person_PhoneNumber"* noundef nonnull align 8 dereferenceable(32) %134, i8* noundef nonnull %164, %"class.google::protobuf::io::EpsCopyOutputStream"* noundef %2)
  %166 = add nuw i32 %127, 1
  %167 = icmp eq i32 %166, %113
  br i1 %167, label %118, label %125, !llvm.loop !69

168:                                              ; preds = %118
  %169 = getelementptr inbounds %"class.google::protobuf::io::EpsCopyOutputStream", %"class.google::protobuf::io::EpsCopyOutputStream"* %2, i64 0, i32 0
  %170 = load i8*, i8** %169, align 8, !tbaa !44
  %171 = icmp ugt i8* %170, %119
  br i1 %171, label %175, label %172, !prof !12

172:                                              ; preds = %168
  %173 = tail call noundef i8* @_ZN6google8protobuf2io19EpsCopyOutputStream19EnsureSpaceFallbackEPh(%"class.google::protobuf::io::EpsCopyOutputStream"* noundef nonnull align 8 dereferenceable(59) %2, i8* noundef %119)
  %174 = load %"class.google::protobuf::Timestamp"*, %"class.google::protobuf::Timestamp"** %121, align 8, !tbaa !20
  br label %175

175:                                              ; preds = %168, %172
  %176 = phi %"class.google::protobuf::Timestamp"* [ %174, %172 ], [ %122, %168 ]
  %177 = phi i8* [ %173, %172 ], [ %119, %168 ]
  store i8 42, i8* %177, align 1, !tbaa !37
  %178 = getelementptr inbounds i8, i8* %177, i64 1
  %179 = getelementptr inbounds %"class.google::protobuf::Timestamp", %"class.google::protobuf::Timestamp"* %176, i64 0, i32 3, i32 0, i32 0, i32 0
  %180 = load atomic i32, i32* %179 monotonic, align 4
  %181 = icmp ult i32 %180, 128
  %182 = trunc i32 %180 to i8
  br i1 %181, label %183, label %185

183:                                              ; preds = %175
  store i8 %182, i8* %178, align 1, !tbaa !37
  %184 = getelementptr inbounds i8, i8* %177, i64 2
  br label %206

185:                                              ; preds = %175
  %186 = or i8 %182, -128
  store i8 %186, i8* %178, align 1, !tbaa !37
  %187 = lshr i32 %180, 7
  %188 = icmp ult i32 %180, 16384
  br i1 %188, label %189, label %193

189:                                              ; preds = %185
  %190 = trunc i32 %187 to i8
  %191 = getelementptr inbounds i8, i8* %177, i64 2
  store i8 %190, i8* %191, align 1, !tbaa !37
  %192 = getelementptr inbounds i8, i8* %177, i64 3
  br label %206

193:                                              ; preds = %185
  %194 = getelementptr inbounds i8, i8* %177, i64 2
  br label %195

195:                                              ; preds = %195, %193
  %196 = phi i32 [ %187, %193 ], [ %200, %195 ]
  %197 = phi i8* [ %194, %193 ], [ %201, %195 ]
  %198 = trunc i32 %196 to i8
  %199 = or i8 %198, -128
  store i8 %199, i8* %197, align 1, !tbaa !37
  %200 = lshr i32 %196, 7
  %201 = getelementptr inbounds i8, i8* %197, i64 1
  %202 = icmp ugt i32 %196, 16383
  br i1 %202, label %195, label %203, !prof !41, !llvm.loop !68

203:                                              ; preds = %195
  %204 = trunc i32 %200 to i8
  %205 = getelementptr inbounds i8, i8* %197, i64 2
  store i8 %204, i8* %201, align 1, !tbaa !37
  br label %206

206:                                              ; preds = %183, %189, %203
  %207 = phi i8* [ %184, %183 ], [ %192, %189 ], [ %205, %203 ]
  %208 = tail call noundef i8* @_ZNK6google8protobuf9Timestamp18_InternalSerializeEPhPNS0_2io19EpsCopyOutputStreamE(%"class.google::protobuf::Timestamp"* noundef nonnull align 8 dereferenceable(32) %176, i8* noundef nonnull %207, %"class.google::protobuf::io::EpsCopyOutputStream"* noundef %2)
  br label %209

209:                                              ; preds = %206, %118
  %210 = phi i8* [ %208, %206 ], [ %119, %118 ]
  %211 = getelementptr inbounds %"class.tutorial::Person", %"class.tutorial::Person"* %0, i64 0, i32 0, i32 0, i32 1, i32 0
  %212 = load i8*, i8** %211, align 8, !tbaa !5
  %213 = ptrtoint i8* %212 to i64
  %214 = and i64 %213, 1
  %215 = icmp eq i64 %214, 0
  br i1 %215, label %221, label %216, !prof !12

216:                                              ; preds = %209
  %217 = and i64 %213, -2
  %218 = inttoptr i64 %217 to %"struct.google::protobuf::internal::InternalMetadata::Container"*
  %219 = getelementptr inbounds %"struct.google::protobuf::internal::InternalMetadata::Container", %"struct.google::protobuf::internal::InternalMetadata::Container"* %218, i64 0, i32 1
  %220 = tail call noundef i8* @_ZN6google8protobuf8internal10WireFormat37InternalSerializeUnknownFieldsToArrayERKNS0_15UnknownFieldSetEPhPNS0_2io19EpsCopyOutputStreamE(%"class.google::protobuf::UnknownFieldSet"* noundef nonnull align 8 dereferenceable(24) %219, i8* noundef %210, %"class.google::protobuf::io::EpsCopyOutputStream"* noundef %2)
  br label %221

221:                                              ; preds = %216, %209
  %222 = phi i8* [ %220, %216 ], [ %210, %209 ]
  ret i8* %222
}

; Function Attrs: uwtable
define dso_local noundef i64 @_ZNK8tutorial6Person12ByteSizeLongEv(%"class.tutorial::Person"* noundef nonnull align 8 dereferenceable(72) %0) unnamed_addr #3 align 2 personality i32 (...)* @__gxx_personality_v0 {
  %2 = getelementptr inbounds %"class.tutorial::Person", %"class.tutorial::Person"* %0, i64 0, i32 1, i32 0, i32 1
  %3 = load i32, i32* %2, align 8, !tbaa !60
  %4 = sext i32 %3 to i64
  %5 = getelementptr inbounds %"class.tutorial::Person", %"class.tutorial::Person"* %0, i64 0, i32 1, i32 0, i32 3
  %6 = load %"struct.google::protobuf::internal::RepeatedPtrFieldBase::Rep"*, %"struct.google::protobuf::internal::RepeatedPtrFieldBase::Rep"** %5, align 8, !tbaa !58
  %7 = icmp eq %"struct.google::protobuf::internal::RepeatedPtrFieldBase::Rep"* %6, null
  %8 = getelementptr inbounds %"struct.google::protobuf::internal::RepeatedPtrFieldBase::Rep", %"struct.google::protobuf::internal::RepeatedPtrFieldBase::Rep"* %6, i64 0, i32 1, i64 0
  %9 = select i1 %7, i8** null, i8** %8
  %10 = getelementptr inbounds i8*, i8** %9, i64 %4
  %11 = icmp eq i32 %3, 0
  br i1 %11, label %12, label %19

12:                                               ; preds = %74, %1
  %13 = phi i64 [ 0, %1 ], [ %85, %74 ]
  %14 = getelementptr inbounds %"class.tutorial::Person", %"class.tutorial::Person"* %0, i64 0, i32 2, i32 0
  %15 = load %"class.std::__cxx11::basic_string"*, %"class.std::__cxx11::basic_string"** %14, align 8, !tbaa !18
  %16 = getelementptr inbounds %"class.std::__cxx11::basic_string", %"class.std::__cxx11::basic_string"* %15, i64 0, i32 1
  %17 = load i64, i64* %16, align 8, !tbaa !28
  %18 = icmp eq i64 %17, 0
  br i1 %18, label %100, label %88

19:                                               ; preds = %1, %74
  %20 = phi i64 [ %85, %74 ], [ %4, %1 ]
  %21 = phi i8** [ %86, %74 ], [ %9, %1 ]
  %22 = bitcast i8** %21 to %"class.tutorial::Person_PhoneNumber"**
  %23 = load %"class.tutorial::Person_PhoneNumber"*, %"class.tutorial::Person_PhoneNumber"** %22, align 8, !tbaa !27
  %24 = getelementptr inbounds %"class.tutorial::Person_PhoneNumber", %"class.tutorial::Person_PhoneNumber"* %23, i64 0, i32 1, i32 0
  %25 = load %"class.std::__cxx11::basic_string"*, %"class.std::__cxx11::basic_string"** %24, align 8, !tbaa !18
  %26 = getelementptr inbounds %"class.std::__cxx11::basic_string", %"class.std::__cxx11::basic_string"* %25, i64 0, i32 1
  %27 = load i64, i64* %26, align 8, !tbaa !28
  %28 = icmp eq i64 %27, 0
  br i1 %28, label %40, label %29

29:                                               ; preds = %19
  %30 = trunc i64 %27 to i32
  %31 = or i32 %30, 1
  %32 = tail call i32 @llvm.ctlz.i32(i32 %31, i1 true) #24, !range !49
  %33 = xor i32 %32, 31
  %34 = mul nuw nsw i32 %33, 9
  %35 = add nuw nsw i32 %34, 73
  %36 = lshr i32 %35, 6
  %37 = zext i32 %36 to i64
  %38 = add i64 %27, 1
  %39 = add i64 %38, %37
  br label %40

40:                                               ; preds = %29, %19
  %41 = phi i64 [ %39, %29 ], [ 0, %19 ]
  %42 = getelementptr inbounds %"class.tutorial::Person_PhoneNumber", %"class.tutorial::Person_PhoneNumber"* %23, i64 0, i32 2
  %43 = load i32, i32* %42, align 8, !tbaa !25
  %44 = icmp eq i32 %43, 0
  br i1 %44, label %59, label %45

45:                                               ; preds = %40
  %46 = icmp slt i32 %43, 0
  br i1 %46, label %56, label %47

47:                                               ; preds = %45
  %48 = or i32 %43, 1
  %49 = tail call i32 @llvm.ctlz.i32(i32 %48, i1 true) #24, !range !49
  %50 = xor i32 %49, 31
  %51 = mul nuw nsw i32 %50, 9
  %52 = add nuw nsw i32 %51, 73
  %53 = lshr i32 %52, 6
  %54 = add nuw nsw i32 %53, 1
  %55 = zext i32 %54 to i64
  br label %56

56:                                               ; preds = %47, %45
  %57 = phi i64 [ %55, %47 ], [ 11, %45 ]
  %58 = add i64 %57, %41
  br label %59

59:                                               ; preds = %56, %40
  %60 = phi i64 [ %58, %56 ], [ %41, %40 ]
  %61 = getelementptr inbounds %"class.tutorial::Person_PhoneNumber", %"class.tutorial::Person_PhoneNumber"* %23, i64 0, i32 0, i32 0, i32 1
  %62 = getelementptr inbounds %"class.google::protobuf::internal::InternalMetadata", %"class.google::protobuf::internal::InternalMetadata"* %61, i64 0, i32 0
  %63 = load i8*, i8** %62, align 8, !tbaa !5
  %64 = ptrtoint i8* %63 to i64
  %65 = and i64 %64, 1
  %66 = icmp eq i64 %65, 0
  br i1 %66, label %71, label %67, !prof !12

67:                                               ; preds = %59
  %68 = getelementptr inbounds %"class.tutorial::Person_PhoneNumber", %"class.tutorial::Person_PhoneNumber"* %23, i64 0, i32 3
  %69 = tail call noundef i64 @_ZN6google8protobuf8internal24ComputeUnknownFieldsSizeERKNS1_16InternalMetadataEmPNS1_10CachedSizeE(%"class.google::protobuf::internal::InternalMetadata"* noundef nonnull align 8 dereferenceable(8) %61, i64 noundef %60, %"class.google::protobuf::internal::CachedSize"* noundef nonnull %68)
  %70 = trunc i64 %69 to i32
  br label %74

71:                                               ; preds = %59
  %72 = trunc i64 %60 to i32
  %73 = getelementptr inbounds %"class.tutorial::Person_PhoneNumber", %"class.tutorial::Person_PhoneNumber"* %23, i64 0, i32 3, i32 0, i32 0, i32 0
  store atomic i32 %72, i32* %73 monotonic, align 4
  br label %74

74:                                               ; preds = %67, %71
  %75 = phi i32 [ %70, %67 ], [ %72, %71 ]
  %76 = phi i64 [ %69, %67 ], [ %60, %71 ]
  %77 = or i32 %75, 1
  %78 = tail call i32 @llvm.ctlz.i32(i32 %77, i1 true) #24, !range !49
  %79 = xor i32 %78, 31
  %80 = mul nuw nsw i32 %79, 9
  %81 = add nuw nsw i32 %80, 73
  %82 = lshr i32 %81, 6
  %83 = zext i32 %82 to i64
  %84 = add i64 %76, %20
  %85 = add i64 %84, %83
  %86 = getelementptr inbounds i8*, i8** %21, i64 1
  %87 = icmp eq i8** %86, %10
  br i1 %87, label %12, label %19

88:                                               ; preds = %12
  %89 = trunc i64 %17 to i32
  %90 = or i32 %89, 1
  %91 = tail call i32 @llvm.ctlz.i32(i32 %90, i1 true) #24, !range !49
  %92 = xor i32 %91, 31
  %93 = mul nuw nsw i32 %92, 9
  %94 = add nuw nsw i32 %93, 73
  %95 = lshr i32 %94, 6
  %96 = zext i32 %95 to i64
  %97 = add i64 %13, 1
  %98 = add i64 %97, %17
  %99 = add i64 %98, %96
  br label %100

100:                                              ; preds = %88, %12
  %101 = phi i64 [ %99, %88 ], [ %13, %12 ]
  %102 = getelementptr inbounds %"class.tutorial::Person", %"class.tutorial::Person"* %0, i64 0, i32 3, i32 0
  %103 = load %"class.std::__cxx11::basic_string"*, %"class.std::__cxx11::basic_string"** %102, align 8, !tbaa !18
  %104 = getelementptr inbounds %"class.std::__cxx11::basic_string", %"class.std::__cxx11::basic_string"* %103, i64 0, i32 1
  %105 = load i64, i64* %104, align 8, !tbaa !28
  %106 = icmp eq i64 %105, 0
  br i1 %106, label %119, label %107

107:                                              ; preds = %100
  %108 = trunc i64 %105 to i32
  %109 = or i32 %108, 1
  %110 = tail call i32 @llvm.ctlz.i32(i32 %109, i1 true) #24, !range !49
  %111 = xor i32 %110, 31
  %112 = mul nuw nsw i32 %111, 9
  %113 = add nuw nsw i32 %112, 73
  %114 = lshr i32 %113, 6
  %115 = zext i32 %114 to i64
  %116 = add i64 %101, 1
  %117 = add i64 %116, %105
  %118 = add i64 %117, %115
  br label %119

119:                                              ; preds = %107, %100
  %120 = phi i64 [ %118, %107 ], [ %101, %100 ]
  %121 = icmp ne %"class.tutorial::Person"* %0, bitcast (%"class.tutorial::PersonDefaultTypeInternal"* @_ZN8tutorial25_Person_default_instance_E to %"class.tutorial::Person"*)
  %122 = getelementptr inbounds %"class.tutorial::Person", %"class.tutorial::Person"* %0, i64 0, i32 4
  %123 = load %"class.google::protobuf::Timestamp"*, %"class.google::protobuf::Timestamp"** %122, align 8
  %124 = icmp ne %"class.google::protobuf::Timestamp"* %123, null
  %125 = select i1 %121, i1 %124, i1 false
  br i1 %125, label %126, label %139

126:                                              ; preds = %119
  %127 = tail call noundef i64 @_ZNK6google8protobuf9Timestamp12ByteSizeLongEv(%"class.google::protobuf::Timestamp"* noundef nonnull align 8 dereferenceable(32) %123)
  %128 = trunc i64 %127 to i32
  %129 = or i32 %128, 1
  %130 = tail call i32 @llvm.ctlz.i32(i32 %129, i1 true) #24, !range !49
  %131 = xor i32 %130, 31
  %132 = mul nuw nsw i32 %131, 9
  %133 = add nuw nsw i32 %132, 73
  %134 = lshr i32 %133, 6
  %135 = zext i32 %134 to i64
  %136 = add i64 %120, 1
  %137 = add i64 %136, %127
  %138 = add i64 %137, %135
  br label %139

139:                                              ; preds = %126, %119
  %140 = phi i64 [ %138, %126 ], [ %120, %119 ]
  %141 = getelementptr inbounds %"class.tutorial::Person", %"class.tutorial::Person"* %0, i64 0, i32 5
  %142 = load i32, i32* %141, align 8, !tbaa !56
  %143 = icmp eq i32 %142, 0
  br i1 %143, label %158, label %144

144:                                              ; preds = %139
  %145 = icmp slt i32 %142, 0
  br i1 %145, label %155, label %146

146:                                              ; preds = %144
  %147 = or i32 %142, 1
  %148 = tail call i32 @llvm.ctlz.i32(i32 %147, i1 true) #24, !range !49
  %149 = xor i32 %148, 31
  %150 = mul nuw nsw i32 %149, 9
  %151 = add nuw nsw i32 %150, 73
  %152 = lshr i32 %151, 6
  %153 = add nuw nsw i32 %152, 1
  %154 = zext i32 %153 to i64
  br label %155

155:                                              ; preds = %144, %146
  %156 = phi i64 [ %154, %146 ], [ 11, %144 ]
  %157 = add i64 %156, %140
  br label %158

158:                                              ; preds = %155, %139
  %159 = phi i64 [ %157, %155 ], [ %140, %139 ]
  %160 = getelementptr inbounds %"class.tutorial::Person", %"class.tutorial::Person"* %0, i64 0, i32 0, i32 0, i32 1
  %161 = getelementptr inbounds %"class.google::protobuf::internal::InternalMetadata", %"class.google::protobuf::internal::InternalMetadata"* %160, i64 0, i32 0
  %162 = load i8*, i8** %161, align 8, !tbaa !5
  %163 = ptrtoint i8* %162 to i64
  %164 = and i64 %163, 1
  %165 = icmp eq i64 %164, 0
  br i1 %165, label %169, label %166, !prof !12

166:                                              ; preds = %158
  %167 = getelementptr inbounds %"class.tutorial::Person", %"class.tutorial::Person"* %0, i64 0, i32 6
  %168 = tail call noundef i64 @_ZN6google8protobuf8internal24ComputeUnknownFieldsSizeERKNS1_16InternalMetadataEmPNS1_10CachedSizeE(%"class.google::protobuf::internal::InternalMetadata"* noundef nonnull align 8 dereferenceable(8) %160, i64 noundef %159, %"class.google::protobuf::internal::CachedSize"* noundef nonnull %167)
  br label %172

169:                                              ; preds = %158
  %170 = trunc i64 %159 to i32
  %171 = getelementptr inbounds %"class.tutorial::Person", %"class.tutorial::Person"* %0, i64 0, i32 6, i32 0, i32 0, i32 0
  store atomic i32 %170, i32* %171 monotonic, align 4
  br label %172

172:                                              ; preds = %169, %166
  %173 = phi i64 [ %168, %166 ], [ %159, %169 ]
  ret i64 %173
}

; Function Attrs: mustprogress uwtable
define dso_local void @_ZN8tutorial6Person9MergeFromERKN6google8protobuf7MessageE(%"class.tutorial::Person"* noundef nonnull align 8 dereferenceable(72) %0, %"class.google::protobuf::Message"* noundef nonnull align 8 dereferenceable(16) %1) unnamed_addr #4 align 2 personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*) {
  %3 = alloca %"class.google::protobuf::internal::LogMessage", align 8
  %4 = alloca %"class.google::protobuf::internal::LogFinisher", align 1
  %5 = getelementptr inbounds %"class.tutorial::Person", %"class.tutorial::Person"* %0, i64 0, i32 0
  %6 = icmp eq %"class.google::protobuf::Message"* %5, %1
  %7 = getelementptr inbounds %"class.google::protobuf::internal::LogFinisher", %"class.google::protobuf::internal::LogFinisher"* %4, i64 0, i32 0
  call void @llvm.lifetime.start.p0i8(i64 1, i8* nonnull %7) #24
  br i1 %6, label %8, label %12

8:                                                ; preds = %2
  %9 = bitcast %"class.google::protobuf::internal::LogMessage"* %3 to i8*
  call void @llvm.lifetime.start.p0i8(i64 56, i8* nonnull %9) #24
  call void @_ZN6google8protobuf8internal10LogMessageC1ENS0_8LogLevelEPKci(%"class.google::protobuf::internal::LogMessage"* noundef nonnull align 8 dereferenceable(56) %3, i32 noundef 3, i8* noundef getelementptr inbounds ([24 x i8], [24 x i8]* @.str.5, i64 0, i64 0), i32 noundef 705)
  %10 = invoke noundef nonnull align 8 dereferenceable(56) %"class.google::protobuf::internal::LogMessage"* @_ZN6google8protobuf8internal10LogMessagelsEPKc(%"class.google::protobuf::internal::LogMessage"* noundef nonnull align 8 dereferenceable(56) %3, i8* noundef getelementptr inbounds ([34 x i8], [34 x i8]* @.str.6, i64 0, i64 0))
          to label %11 unwind label %19

11:                                               ; preds = %8
  invoke void @_ZN6google8protobuf8internal11LogFinisheraSERNS1_10LogMessageE(%"class.google::protobuf::internal::LogFinisher"* noundef nonnull align 1 dereferenceable(1) %4, %"class.google::protobuf::internal::LogMessage"* noundef nonnull align 8 dereferenceable(56) %10)
          to label %13 unwind label %21

12:                                               ; preds = %2
  call void @llvm.lifetime.end.p0i8(i64 1, i8* nonnull %7) #24
  br label %14

13:                                               ; preds = %11
  call void @llvm.lifetime.end.p0i8(i64 1, i8* nonnull %7) #24
  call void @_ZN6google8protobuf8internal10LogMessageD1Ev(%"class.google::protobuf::internal::LogMessage"* noundef nonnull align 8 dereferenceable(56) %3) #24
  call void @llvm.lifetime.end.p0i8(i64 56, i8* nonnull %9) #24
  br label %14

14:                                               ; preds = %12, %13
  %15 = bitcast %"class.google::protobuf::Message"* %1 to i8*
  %16 = call i8* @__dynamic_cast(i8* nonnull %15, i8* bitcast (i8** @_ZTIN6google8protobuf7MessageE to i8*), i8* bitcast ({ i8*, i8*, i8* }* @_ZTIN8tutorial6PersonE to i8*), i64 0) #24
  %17 = icmp eq i8* %16, null
  br i1 %17, label %18, label %25

18:                                               ; preds = %14
  call void @_ZN6google8protobuf8internal13ReflectionOps5MergeERKNS0_7MessageEPS3_(%"class.google::protobuf::Message"* noundef nonnull align 8 dereferenceable(16) %1, %"class.google::protobuf::Message"* noundef nonnull %5)
  br label %27

19:                                               ; preds = %8
  %20 = landingpad { i8*, i32 }
          cleanup
  br label %23

21:                                               ; preds = %11
  %22 = landingpad { i8*, i32 }
          cleanup
  call void @llvm.lifetime.end.p0i8(i64 1, i8* nonnull %7) #24
  br label %23

23:                                               ; preds = %19, %21
  %24 = phi { i8*, i32 } [ %22, %21 ], [ %20, %19 ]
  call void @_ZN6google8protobuf8internal10LogMessageD1Ev(%"class.google::protobuf::internal::LogMessage"* noundef nonnull align 8 dereferenceable(56) %3) #24
  call void @llvm.lifetime.end.p0i8(i64 56, i8* nonnull %9) #24
  resume { i8*, i32 } %24

25:                                               ; preds = %14
  %26 = bitcast i8* %16 to %"class.tutorial::Person"*
  call void @_ZN8tutorial6Person9MergeFromERKS0_(%"class.tutorial::Person"* noundef nonnull align 8 dereferenceable(72) %0, %"class.tutorial::Person"* noundef nonnull align 8 dereferenceable(72) %26)
  br label %27

27:                                               ; preds = %25, %18
  ret void
}

; Function Attrs: mustprogress uwtable
define dso_local void @_ZN8tutorial6Person9MergeFromERKS0_(%"class.tutorial::Person"* noundef nonnull align 8 dereferenceable(72) %0, %"class.tutorial::Person"* noundef nonnull align 8 dereferenceable(72) %1) local_unnamed_addr #4 align 2 personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*) {
  %3 = alloca %"class.google::protobuf::internal::LogMessage", align 8
  %4 = alloca %"class.google::protobuf::internal::LogFinisher", align 1
  %5 = icmp eq %"class.tutorial::Person"* %1, %0
  %6 = getelementptr inbounds %"class.google::protobuf::internal::LogFinisher", %"class.google::protobuf::internal::LogFinisher"* %4, i64 0, i32 0
  call void @llvm.lifetime.start.p0i8(i64 1, i8* nonnull %6) #24
  br i1 %5, label %7, label %11

7:                                                ; preds = %2
  %8 = bitcast %"class.google::protobuf::internal::LogMessage"* %3 to i8*
  call void @llvm.lifetime.start.p0i8(i64 56, i8* nonnull %8) #24
  call void @_ZN6google8protobuf8internal10LogMessageC1ENS0_8LogLevelEPKci(%"class.google::protobuf::internal::LogMessage"* noundef nonnull align 8 dereferenceable(56) %3, i32 noundef 3, i8* noundef getelementptr inbounds ([24 x i8], [24 x i8]* @.str.5, i64 0, i64 0), i32 noundef 720)
  %9 = invoke noundef nonnull align 8 dereferenceable(56) %"class.google::protobuf::internal::LogMessage"* @_ZN6google8protobuf8internal10LogMessagelsEPKc(%"class.google::protobuf::internal::LogMessage"* noundef nonnull align 8 dereferenceable(56) %3, i8* noundef getelementptr inbounds ([34 x i8], [34 x i8]* @.str.6, i64 0, i64 0))
          to label %10 unwind label %66

10:                                               ; preds = %7
  invoke void @_ZN6google8protobuf8internal11LogFinisheraSERNS1_10LogMessageE(%"class.google::protobuf::internal::LogFinisher"* noundef nonnull align 1 dereferenceable(1) %4, %"class.google::protobuf::internal::LogMessage"* noundef nonnull align 8 dereferenceable(56) %9)
          to label %12 unwind label %68

11:                                               ; preds = %2
  call void @llvm.lifetime.end.p0i8(i64 1, i8* nonnull %6) #24
  br label %13

12:                                               ; preds = %10
  call void @llvm.lifetime.end.p0i8(i64 1, i8* nonnull %6) #24
  call void @_ZN6google8protobuf8internal10LogMessageD1Ev(%"class.google::protobuf::internal::LogMessage"* noundef nonnull align 8 dereferenceable(56) %3) #24
  call void @llvm.lifetime.end.p0i8(i64 56, i8* nonnull %8) #24
  br label %13

13:                                               ; preds = %11, %12
  %14 = getelementptr inbounds %"class.tutorial::Person", %"class.tutorial::Person"* %0, i64 0, i32 0, i32 0, i32 1
  %15 = getelementptr inbounds %"class.tutorial::Person", %"class.tutorial::Person"* %1, i64 0, i32 0, i32 0, i32 1, i32 0
  %16 = load i8*, i8** %15, align 8, !tbaa !5
  %17 = ptrtoint i8* %16 to i64
  %18 = and i64 %17, 1
  %19 = icmp eq i64 %18, 0
  br i1 %19, label %37, label %20

20:                                               ; preds = %13
  %21 = and i64 %17, -2
  %22 = inttoptr i64 %21 to %"struct.google::protobuf::internal::InternalMetadata::Container"*
  %23 = getelementptr inbounds %"struct.google::protobuf::internal::InternalMetadata::Container", %"struct.google::protobuf::internal::InternalMetadata::Container"* %22, i64 0, i32 1
  %24 = getelementptr inbounds %"class.google::protobuf::internal::InternalMetadata", %"class.google::protobuf::internal::InternalMetadata"* %14, i64 0, i32 0
  %25 = load i8*, i8** %24, align 8, !tbaa !5
  %26 = ptrtoint i8* %25 to i64
  %27 = and i64 %26, 1
  %28 = icmp eq i64 %27, 0
  br i1 %28, label %33, label %29, !prof !41

29:                                               ; preds = %20
  %30 = and i64 %26, -2
  %31 = inttoptr i64 %30 to %"struct.google::protobuf::internal::InternalMetadata::Container"*
  %32 = getelementptr inbounds %"struct.google::protobuf::internal::InternalMetadata::Container", %"struct.google::protobuf::internal::InternalMetadata::Container"* %31, i64 0, i32 1
  br label %35

33:                                               ; preds = %20
  %34 = call noundef %"class.google::protobuf::UnknownFieldSet"* @_ZN6google8protobuf8internal16InternalMetadata27mutable_unknown_fields_slowINS0_15UnknownFieldSetEEEPT_v(%"class.google::protobuf::internal::InternalMetadata"* noundef nonnull align 8 dereferenceable(8) %14)
  br label %35

35:                                               ; preds = %33, %29
  %36 = phi %"class.google::protobuf::UnknownFieldSet"* [ %32, %29 ], [ %34, %33 ]
  call void @_ZN6google8protobuf15UnknownFieldSet9MergeFromERKS1_(%"class.google::protobuf::UnknownFieldSet"* noundef nonnull align 8 dereferenceable(24) %36, %"class.google::protobuf::UnknownFieldSet"* noundef nonnull align 8 dereferenceable(24) %23)
  br label %37

37:                                               ; preds = %13, %35
  %38 = getelementptr inbounds %"class.tutorial::Person", %"class.tutorial::Person"* %0, i64 0, i32 1, i32 0
  %39 = getelementptr inbounds %"class.tutorial::Person", %"class.tutorial::Person"* %1, i64 0, i32 1, i32 0
  call void @_ZN6google8protobuf8internal20RepeatedPtrFieldBase9MergeFromINS0_16RepeatedPtrFieldIN8tutorial18Person_PhoneNumberEE11TypeHandlerEEEvRKS2_(%"class.google::protobuf::internal::RepeatedPtrFieldBase"* noundef nonnull align 8 dereferenceable(24) %38, %"class.google::protobuf::internal::RepeatedPtrFieldBase"* noundef nonnull align 8 dereferenceable(24) %39)
  %40 = getelementptr inbounds %"class.tutorial::Person", %"class.tutorial::Person"* %1, i64 0, i32 2, i32 0
  %41 = load %"class.std::__cxx11::basic_string"*, %"class.std::__cxx11::basic_string"** %40, align 8, !tbaa !18
  %42 = getelementptr inbounds %"class.std::__cxx11::basic_string", %"class.std::__cxx11::basic_string"* %41, i64 0, i32 1
  %43 = load i64, i64* %42, align 8, !tbaa !28
  %44 = icmp eq i64 %43, 0
  br i1 %44, label %72, label %45

45:                                               ; preds = %37
  %46 = getelementptr inbounds %"class.tutorial::Person", %"class.tutorial::Person"* %0, i64 0, i32 2
  %47 = getelementptr inbounds %"class.tutorial::Person", %"class.tutorial::Person"* %0, i64 0, i32 0, i32 0, i32 1, i32 0
  %48 = load i8*, i8** %47, align 8, !tbaa !5
  %49 = ptrtoint i8* %48 to i64
  %50 = and i64 %49, 1
  %51 = icmp eq i64 %50, 0
  %52 = and i64 %49, -2
  br i1 %51, label %57, label %53, !prof !12

53:                                               ; preds = %45
  %54 = inttoptr i64 %52 to %"struct.google::protobuf::internal::InternalMetadata::ContainerBase"*
  %55 = getelementptr inbounds %"struct.google::protobuf::internal::InternalMetadata::ContainerBase", %"struct.google::protobuf::internal::InternalMetadata::ContainerBase"* %54, i64 0, i32 0
  %56 = load %"class.google::protobuf::Arena"*, %"class.google::protobuf::Arena"** %55, align 8, !tbaa !32
  br label %59

57:                                               ; preds = %45
  %58 = inttoptr i64 %52 to %"class.google::protobuf::Arena"*
  br label %59

59:                                               ; preds = %57, %53
  %60 = phi %"class.google::protobuf::Arena"* [ %56, %53 ], [ %58, %57 ]
  %61 = getelementptr inbounds %"struct.google::protobuf::internal::ArenaStringPtr", %"struct.google::protobuf::internal::ArenaStringPtr"* %46, i64 0, i32 0
  %62 = load %"class.std::__cxx11::basic_string"*, %"class.std::__cxx11::basic_string"** %61, align 8, !tbaa !18
  %63 = icmp eq %"class.std::__cxx11::basic_string"* %62, bitcast (%"class.google::protobuf::internal::ExplicitlyConstructed.20"* @_ZN6google8protobuf8internal26fixed_address_empty_stringB5cxx11E to %"class.std::__cxx11::basic_string"*)
  br i1 %63, label %64, label %65

64:                                               ; preds = %59
  call void @_ZN6google8protobuf8internal14ArenaStringPtr14CreateInstanceEPNS0_5ArenaEPKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE(%"struct.google::protobuf::internal::ArenaStringPtr"* noundef nonnull align 8 dereferenceable(8) %46, %"class.google::protobuf::Arena"* noundef %60, %"class.std::__cxx11::basic_string"* noundef nonnull %41)
  br label %72

65:                                               ; preds = %59
  call void @_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE9_M_assignERKS4_(%"class.std::__cxx11::basic_string"* noundef nonnull align 8 dereferenceable(32) %62, %"class.std::__cxx11::basic_string"* noundef nonnull align 8 dereferenceable(32) %41)
  br label %72

66:                                               ; preds = %7
  %67 = landingpad { i8*, i32 }
          cleanup
  br label %70

68:                                               ; preds = %10
  %69 = landingpad { i8*, i32 }
          cleanup
  call void @llvm.lifetime.end.p0i8(i64 1, i8* nonnull %6) #24
  br label %70

70:                                               ; preds = %66, %68
  %71 = phi { i8*, i32 } [ %69, %68 ], [ %67, %66 ]
  call void @_ZN6google8protobuf8internal10LogMessageD1Ev(%"class.google::protobuf::internal::LogMessage"* noundef nonnull align 8 dereferenceable(56) %3) #24
  call void @llvm.lifetime.end.p0i8(i64 56, i8* nonnull %8) #24
  resume { i8*, i32 } %71

72:                                               ; preds = %65, %64, %37
  %73 = getelementptr inbounds %"class.tutorial::Person", %"class.tutorial::Person"* %1, i64 0, i32 3, i32 0
  %74 = load %"class.std::__cxx11::basic_string"*, %"class.std::__cxx11::basic_string"** %73, align 8, !tbaa !18
  %75 = getelementptr inbounds %"class.std::__cxx11::basic_string", %"class.std::__cxx11::basic_string"* %74, i64 0, i32 1
  %76 = load i64, i64* %75, align 8, !tbaa !28
  %77 = icmp eq i64 %76, 0
  br i1 %77, label %99, label %78

78:                                               ; preds = %72
  %79 = getelementptr inbounds %"class.tutorial::Person", %"class.tutorial::Person"* %0, i64 0, i32 3
  %80 = getelementptr inbounds %"class.tutorial::Person", %"class.tutorial::Person"* %0, i64 0, i32 0, i32 0, i32 1, i32 0
  %81 = load i8*, i8** %80, align 8, !tbaa !5
  %82 = ptrtoint i8* %81 to i64
  %83 = and i64 %82, 1
  %84 = icmp eq i64 %83, 0
  %85 = and i64 %82, -2
  br i1 %84, label %90, label %86, !prof !12

86:                                               ; preds = %78
  %87 = inttoptr i64 %85 to %"struct.google::protobuf::internal::InternalMetadata::ContainerBase"*
  %88 = getelementptr inbounds %"struct.google::protobuf::internal::InternalMetadata::ContainerBase", %"struct.google::protobuf::internal::InternalMetadata::ContainerBase"* %87, i64 0, i32 0
  %89 = load %"class.google::protobuf::Arena"*, %"class.google::protobuf::Arena"** %88, align 8, !tbaa !32
  br label %92

90:                                               ; preds = %78
  %91 = inttoptr i64 %85 to %"class.google::protobuf::Arena"*
  br label %92

92:                                               ; preds = %90, %86
  %93 = phi %"class.google::protobuf::Arena"* [ %89, %86 ], [ %91, %90 ]
  %94 = getelementptr inbounds %"struct.google::protobuf::internal::ArenaStringPtr", %"struct.google::protobuf::internal::ArenaStringPtr"* %79, i64 0, i32 0
  %95 = load %"class.std::__cxx11::basic_string"*, %"class.std::__cxx11::basic_string"** %94, align 8, !tbaa !18
  %96 = icmp eq %"class.std::__cxx11::basic_string"* %95, bitcast (%"class.google::protobuf::internal::ExplicitlyConstructed.20"* @_ZN6google8protobuf8internal26fixed_address_empty_stringB5cxx11E to %"class.std::__cxx11::basic_string"*)
  br i1 %96, label %97, label %98

97:                                               ; preds = %92
  call void @_ZN6google8protobuf8internal14ArenaStringPtr14CreateInstanceEPNS0_5ArenaEPKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE(%"struct.google::protobuf::internal::ArenaStringPtr"* noundef nonnull align 8 dereferenceable(8) %79, %"class.google::protobuf::Arena"* noundef %93, %"class.std::__cxx11::basic_string"* noundef nonnull %74)
  br label %99

98:                                               ; preds = %92
  call void @_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE9_M_assignERKS4_(%"class.std::__cxx11::basic_string"* noundef nonnull align 8 dereferenceable(32) %95, %"class.std::__cxx11::basic_string"* noundef nonnull align 8 dereferenceable(32) %74)
  br label %99

99:                                               ; preds = %98, %97, %72
  %100 = icmp ne %"class.tutorial::Person"* %1, bitcast (%"class.tutorial::PersonDefaultTypeInternal"* @_ZN8tutorial25_Person_default_instance_E to %"class.tutorial::Person"*)
  %101 = getelementptr inbounds %"class.tutorial::Person", %"class.tutorial::Person"* %1, i64 0, i32 4
  %102 = load %"class.google::protobuf::Timestamp"*, %"class.google::protobuf::Timestamp"** %101, align 8
  %103 = icmp ne %"class.google::protobuf::Timestamp"* %102, null
  %104 = select i1 %100, i1 %103, i1 false
  br i1 %104, label %105, label %131

105:                                              ; preds = %99
  %106 = getelementptr inbounds %"class.tutorial::Person", %"class.tutorial::Person"* %0, i64 0, i32 4
  %107 = load %"class.google::protobuf::Timestamp"*, %"class.google::protobuf::Timestamp"** %106, align 8, !tbaa !20
  %108 = icmp eq %"class.google::protobuf::Timestamp"* %107, null
  br i1 %108, label %109, label %126

109:                                              ; preds = %105
  %110 = getelementptr inbounds %"class.tutorial::Person", %"class.tutorial::Person"* %0, i64 0, i32 0, i32 0, i32 1, i32 0
  %111 = load i8*, i8** %110, align 8, !tbaa !5
  %112 = ptrtoint i8* %111 to i64
  %113 = and i64 %112, 1
  %114 = icmp eq i64 %113, 0
  %115 = and i64 %112, -2
  br i1 %114, label %120, label %116, !prof !12

116:                                              ; preds = %109
  %117 = inttoptr i64 %115 to %"struct.google::protobuf::internal::InternalMetadata::ContainerBase"*
  %118 = getelementptr inbounds %"struct.google::protobuf::internal::InternalMetadata::ContainerBase", %"struct.google::protobuf::internal::InternalMetadata::ContainerBase"* %117, i64 0, i32 0
  %119 = load %"class.google::protobuf::Arena"*, %"class.google::protobuf::Arena"** %118, align 8, !tbaa !32
  br label %122

120:                                              ; preds = %109
  %121 = inttoptr i64 %115 to %"class.google::protobuf::Arena"*
  br label %122

122:                                              ; preds = %120, %116
  %123 = phi %"class.google::protobuf::Arena"* [ %119, %116 ], [ %121, %120 ]
  %124 = call noundef %"class.google::protobuf::Timestamp"* @_ZN6google8protobuf5Arena18CreateMaybeMessageINS0_9TimestampEJEEEPT_PS1_DpOT0_(%"class.google::protobuf::Arena"* noundef %123)
  store %"class.google::protobuf::Timestamp"* %124, %"class.google::protobuf::Timestamp"** %106, align 8, !tbaa !20
  %125 = load %"class.google::protobuf::Timestamp"*, %"class.google::protobuf::Timestamp"** %101, align 8, !tbaa !20
  br label %126

126:                                              ; preds = %105, %122
  %127 = phi %"class.google::protobuf::Timestamp"* [ %125, %122 ], [ %102, %105 ]
  %128 = phi %"class.google::protobuf::Timestamp"* [ %124, %122 ], [ %107, %105 ]
  %129 = icmp eq %"class.google::protobuf::Timestamp"* %127, null
  %130 = select i1 %129, %"class.google::protobuf::Timestamp"* bitcast (%"class.google::protobuf::TimestampDefaultTypeInternal"* @_ZN6google8protobuf28_Timestamp_default_instance_E to %"class.google::protobuf::Timestamp"*), %"class.google::protobuf::Timestamp"* %127
  call void @_ZN6google8protobuf9Timestamp9MergeFromERKS1_(%"class.google::protobuf::Timestamp"* noundef nonnull align 8 dereferenceable(32) %128, %"class.google::protobuf::Timestamp"* noundef nonnull align 8 dereferenceable(32) %130)
  br label %131

131:                                              ; preds = %126, %99
  %132 = getelementptr inbounds %"class.tutorial::Person", %"class.tutorial::Person"* %1, i64 0, i32 5
  %133 = load i32, i32* %132, align 8, !tbaa !56
  %134 = icmp eq i32 %133, 0
  br i1 %134, label %137, label %135

135:                                              ; preds = %131
  %136 = getelementptr inbounds %"class.tutorial::Person", %"class.tutorial::Person"* %0, i64 0, i32 5
  store i32 %133, i32* %136, align 8, !tbaa !56
  br label %137

137:                                              ; preds = %135, %131
  ret void
}

declare void @_ZN6google8protobuf9Timestamp9MergeFromERKS1_(%"class.google::protobuf::Timestamp"* noundef nonnull align 8 dereferenceable(32), %"class.google::protobuf::Timestamp"* noundef nonnull align 8 dereferenceable(32)) local_unnamed_addr #0

; Function Attrs: mustprogress uwtable
define dso_local void @_ZN8tutorial6Person8CopyFromERKN6google8protobuf7MessageE(%"class.tutorial::Person"* noundef nonnull align 8 dereferenceable(72) %0, %"class.google::protobuf::Message"* noundef nonnull align 8 dereferenceable(16) %1) unnamed_addr #4 align 2 {
  %3 = getelementptr inbounds %"class.tutorial::Person", %"class.tutorial::Person"* %0, i64 0, i32 0
  %4 = icmp eq %"class.google::protobuf::Message"* %3, %1
  br i1 %4, label %6, label %5

5:                                                ; preds = %2
  tail call void @_ZN8tutorial6Person5ClearEv(%"class.tutorial::Person"* noundef nonnull align 8 dereferenceable(72) %0)
  tail call void @_ZN8tutorial6Person9MergeFromERKN6google8protobuf7MessageE(%"class.tutorial::Person"* noundef nonnull align 8 dereferenceable(72) %0, %"class.google::protobuf::Message"* noundef nonnull align 8 dereferenceable(16) %1)
  br label %6

6:                                                ; preds = %2, %5
  ret void
}

; Function Attrs: mustprogress uwtable
define dso_local void @_ZN8tutorial6Person8CopyFromERKS0_(%"class.tutorial::Person"* noundef nonnull align 8 dereferenceable(72) %0, %"class.tutorial::Person"* noundef nonnull align 8 dereferenceable(72) %1) local_unnamed_addr #4 align 2 {
  %3 = icmp eq %"class.tutorial::Person"* %1, %0
  br i1 %3, label %5, label %4

4:                                                ; preds = %2
  tail call void @_ZN8tutorial6Person5ClearEv(%"class.tutorial::Person"* noundef nonnull align 8 dereferenceable(72) %0)
  tail call void @_ZN8tutorial6Person9MergeFromERKS0_(%"class.tutorial::Person"* noundef nonnull align 8 dereferenceable(72) %0, %"class.tutorial::Person"* noundef nonnull align 8 dereferenceable(72) %1)
  br label %5

5:                                                ; preds = %2, %4
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind readnone uwtable willreturn
define dso_local noundef zeroext i1 @_ZNK8tutorial6Person13IsInitializedEv(%"class.tutorial::Person"* nocapture nonnull readnone align 8 %0) unnamed_addr #5 align 2 {
  ret i1 true
}

; Function Attrs: uwtable
define dso_local void @_ZN8tutorial6Person12InternalSwapEPS0_(%"class.tutorial::Person"* noundef nonnull align 8 dereferenceable(72) %0, %"class.tutorial::Person"* noundef %1) local_unnamed_addr #3 align 2 personality i32 (...)* @__gxx_personality_v0 {
  %3 = getelementptr inbounds %"class.tutorial::Person", %"class.tutorial::Person"* %0, i64 0, i32 0, i32 0, i32 1
  %4 = getelementptr inbounds %"class.tutorial::Person", %"class.tutorial::Person"* %1, i64 0, i32 0, i32 0, i32 1
  %5 = getelementptr inbounds %"class.google::protobuf::internal::InternalMetadata", %"class.google::protobuf::internal::InternalMetadata"* %3, i64 0, i32 0
  %6 = load i8*, i8** %5, align 8, !tbaa !5
  %7 = ptrtoint i8* %6 to i64
  %8 = and i64 %7, 1
  %9 = icmp eq i64 %8, 0
  %10 = getelementptr inbounds %"class.google::protobuf::internal::InternalMetadata", %"class.google::protobuf::internal::InternalMetadata"* %4, i64 0, i32 0
  %11 = load i8*, i8** %10, align 8, !tbaa !5
  %12 = ptrtoint i8* %11 to i64
  %13 = and i64 %12, 1
  %14 = icmp eq i64 %13, 0
  %15 = select i1 %9, i1 %14, i1 false
  br i1 %15, label %48, label %16

16:                                               ; preds = %2
  br i1 %14, label %21, label %17, !prof !41

17:                                               ; preds = %16
  %18 = and i64 %12, -2
  %19 = inttoptr i64 %18 to %"struct.google::protobuf::internal::InternalMetadata::Container"*
  %20 = getelementptr inbounds %"struct.google::protobuf::internal::InternalMetadata::Container", %"struct.google::protobuf::internal::InternalMetadata::Container"* %19, i64 0, i32 1
  br label %25

21:                                               ; preds = %16
  %22 = tail call noundef %"class.google::protobuf::UnknownFieldSet"* @_ZN6google8protobuf8internal16InternalMetadata27mutable_unknown_fields_slowINS0_15UnknownFieldSetEEEPT_v(%"class.google::protobuf::internal::InternalMetadata"* noundef nonnull align 8 dereferenceable(8) %4)
  %23 = load i8*, i8** %5, align 8, !tbaa !5
  %24 = ptrtoint i8* %23 to i64
  br label %25

25:                                               ; preds = %21, %17
  %26 = phi i64 [ %7, %17 ], [ %24, %21 ]
  %27 = phi %"class.google::protobuf::UnknownFieldSet"* [ %20, %17 ], [ %22, %21 ]
  %28 = and i64 %26, 1
  %29 = icmp eq i64 %28, 0
  br i1 %29, label %34, label %30, !prof !41

30:                                               ; preds = %25
  %31 = and i64 %26, -2
  %32 = inttoptr i64 %31 to %"struct.google::protobuf::internal::InternalMetadata::Container"*
  %33 = getelementptr inbounds %"struct.google::protobuf::internal::InternalMetadata::Container", %"struct.google::protobuf::internal::InternalMetadata::Container"* %32, i64 0, i32 1
  br label %36

34:                                               ; preds = %25
  %35 = tail call noundef %"class.google::protobuf::UnknownFieldSet"* @_ZN6google8protobuf8internal16InternalMetadata27mutable_unknown_fields_slowINS0_15UnknownFieldSetEEEPT_v(%"class.google::protobuf::internal::InternalMetadata"* noundef nonnull align 8 dereferenceable(8) %3)
  br label %36

36:                                               ; preds = %34, %30
  %37 = phi %"class.google::protobuf::UnknownFieldSet"* [ %33, %30 ], [ %35, %34 ]
  %38 = bitcast %"class.google::protobuf::UnknownFieldSet"* %37 to <2 x %"class.google::protobuf::UnknownField"*>*
  %39 = load <2 x %"class.google::protobuf::UnknownField"*>, <2 x %"class.google::protobuf::UnknownField"*>* %38, align 8, !tbaa !27
  %40 = getelementptr inbounds %"class.google::protobuf::UnknownFieldSet", %"class.google::protobuf::UnknownFieldSet"* %37, i64 0, i32 0, i32 0, i32 0, i32 0, i32 2
  %41 = load %"class.google::protobuf::UnknownField"*, %"class.google::protobuf::UnknownField"** %40, align 8, !tbaa !50
  %42 = bitcast %"class.google::protobuf::UnknownFieldSet"* %27 to <2 x %"class.google::protobuf::UnknownField"*>*
  %43 = load <2 x %"class.google::protobuf::UnknownField"*>, <2 x %"class.google::protobuf::UnknownField"*>* %42, align 8, !tbaa !27
  %44 = bitcast %"class.google::protobuf::UnknownFieldSet"* %37 to <2 x %"class.google::protobuf::UnknownField"*>*
  store <2 x %"class.google::protobuf::UnknownField"*> %43, <2 x %"class.google::protobuf::UnknownField"*>* %44, align 8, !tbaa !27
  %45 = getelementptr inbounds %"class.google::protobuf::UnknownFieldSet", %"class.google::protobuf::UnknownFieldSet"* %27, i64 0, i32 0, i32 0, i32 0, i32 0, i32 2
  %46 = load %"class.google::protobuf::UnknownField"*, %"class.google::protobuf::UnknownField"** %45, align 8, !tbaa !50
  store %"class.google::protobuf::UnknownField"* %46, %"class.google::protobuf::UnknownField"** %40, align 8, !tbaa !50
  %47 = bitcast %"class.google::protobuf::UnknownFieldSet"* %27 to <2 x %"class.google::protobuf::UnknownField"*>*
  store <2 x %"class.google::protobuf::UnknownField"*> %39, <2 x %"class.google::protobuf::UnknownField"*>* %47, align 8, !tbaa !27
  store %"class.google::protobuf::UnknownField"* %41, %"class.google::protobuf::UnknownField"** %45, align 8, !tbaa !50
  br label %48

48:                                               ; preds = %2, %36
  %49 = getelementptr inbounds %"class.tutorial::Person", %"class.tutorial::Person"* %0, i64 0, i32 1, i32 0
  %50 = getelementptr inbounds %"class.tutorial::Person", %"class.tutorial::Person"* %1, i64 0, i32 1, i32 0
  tail call void @_ZN6google8protobuf8internal20RepeatedPtrFieldBase12InternalSwapEPS2_(%"class.google::protobuf::internal::RepeatedPtrFieldBase"* noundef nonnull align 8 dereferenceable(24) %49, %"class.google::protobuf::internal::RepeatedPtrFieldBase"* noundef nonnull %50)
  %51 = getelementptr inbounds %"class.tutorial::Person", %"class.tutorial::Person"* %0, i64 0, i32 2
  %52 = getelementptr inbounds %"class.tutorial::Person", %"class.tutorial::Person"* %1, i64 0, i32 2
  %53 = getelementptr inbounds %"class.tutorial::Person", %"class.tutorial::Person"* %0, i64 0, i32 0, i32 0, i32 1, i32 0
  %54 = load i8*, i8** %53, align 8, !tbaa !5
  %55 = ptrtoint i8* %54 to i64
  %56 = and i64 %55, 1
  %57 = icmp eq i64 %56, 0
  %58 = and i64 %55, -2
  br i1 %57, label %63, label %59, !prof !12

59:                                               ; preds = %48
  %60 = inttoptr i64 %58 to %"struct.google::protobuf::internal::InternalMetadata::ContainerBase"*
  %61 = getelementptr inbounds %"struct.google::protobuf::internal::InternalMetadata::ContainerBase", %"struct.google::protobuf::internal::InternalMetadata::ContainerBase"* %60, i64 0, i32 0
  %62 = load %"class.google::protobuf::Arena"*, %"class.google::protobuf::Arena"** %61, align 8, !tbaa !32
  br label %65

63:                                               ; preds = %48
  %64 = inttoptr i64 %58 to %"class.google::protobuf::Arena"*
  br label %65

65:                                               ; preds = %59, %63
  %66 = phi %"class.google::protobuf::Arena"* [ %62, %59 ], [ %64, %63 ]
  %67 = getelementptr inbounds %"struct.google::protobuf::internal::ArenaStringPtr", %"struct.google::protobuf::internal::ArenaStringPtr"* %51, i64 0, i32 0
  %68 = load %"class.std::__cxx11::basic_string"*, %"class.std::__cxx11::basic_string"** %67, align 8, !tbaa !18
  %69 = icmp eq %"class.std::__cxx11::basic_string"* %68, bitcast (%"class.google::protobuf::internal::ExplicitlyConstructed.20"* @_ZN6google8protobuf8internal26fixed_address_empty_stringB5cxx11E to %"class.std::__cxx11::basic_string"*)
  br i1 %69, label %70, label %76

70:                                               ; preds = %65
  %71 = getelementptr inbounds %"struct.google::protobuf::internal::ArenaStringPtr", %"struct.google::protobuf::internal::ArenaStringPtr"* %52, i64 0, i32 0
  %72 = load %"class.std::__cxx11::basic_string"*, %"class.std::__cxx11::basic_string"** %71, align 8, !tbaa !18
  %73 = icmp eq %"class.std::__cxx11::basic_string"* %72, bitcast (%"class.google::protobuf::internal::ExplicitlyConstructed.20"* @_ZN6google8protobuf8internal26fixed_address_empty_stringB5cxx11E to %"class.std::__cxx11::basic_string"*)
  br i1 %73, label %88, label %74

74:                                               ; preds = %70
  tail call void @_ZN6google8protobuf8internal14ArenaStringPtr14CreateInstanceEPNS0_5ArenaEPKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE(%"struct.google::protobuf::internal::ArenaStringPtr"* noundef nonnull align 8 dereferenceable(8) %51, %"class.google::protobuf::Arena"* noundef %66, %"class.std::__cxx11::basic_string"* noundef bitcast (%"class.google::protobuf::internal::ExplicitlyConstructed.20"* @_ZN6google8protobuf8internal26fixed_address_empty_stringB5cxx11E to %"class.std::__cxx11::basic_string"*))
  %75 = load %"class.std::__cxx11::basic_string"*, %"class.std::__cxx11::basic_string"** %67, align 8, !tbaa !18
  br label %76

76:                                               ; preds = %74, %65
  %77 = phi %"class.std::__cxx11::basic_string"* [ %75, %74 ], [ %68, %65 ]
  %78 = getelementptr inbounds %"struct.google::protobuf::internal::ArenaStringPtr", %"struct.google::protobuf::internal::ArenaStringPtr"* %52, i64 0, i32 0
  %79 = load %"class.std::__cxx11::basic_string"*, %"class.std::__cxx11::basic_string"** %78, align 8, !tbaa !18
  %80 = icmp eq %"class.std::__cxx11::basic_string"* %79, bitcast (%"class.google::protobuf::internal::ExplicitlyConstructed.20"* @_ZN6google8protobuf8internal26fixed_address_empty_stringB5cxx11E to %"class.std::__cxx11::basic_string"*)
  br i1 %80, label %81, label %83

81:                                               ; preds = %76
  tail call void @_ZN6google8protobuf8internal14ArenaStringPtr14CreateInstanceEPNS0_5ArenaEPKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE(%"struct.google::protobuf::internal::ArenaStringPtr"* noundef nonnull align 8 dereferenceable(8) %52, %"class.google::protobuf::Arena"* noundef %66, %"class.std::__cxx11::basic_string"* noundef bitcast (%"class.google::protobuf::internal::ExplicitlyConstructed.20"* @_ZN6google8protobuf8internal26fixed_address_empty_stringB5cxx11E to %"class.std::__cxx11::basic_string"*))
  %82 = load %"class.std::__cxx11::basic_string"*, %"class.std::__cxx11::basic_string"** %78, align 8, !tbaa !18
  br label %83

83:                                               ; preds = %81, %76
  %84 = phi %"class.std::__cxx11::basic_string"* [ %82, %81 ], [ %79, %76 ]
  tail call void @_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE4swapERS4_(%"class.std::__cxx11::basic_string"* noundef nonnull align 8 dereferenceable(32) %77, %"class.std::__cxx11::basic_string"* noundef nonnull align 8 dereferenceable(32) %84) #24
  %85 = load i8*, i8** %53, align 8, !tbaa !5
  %86 = ptrtoint i8* %85 to i64
  %87 = and i64 %86, -2
  br label %88

88:                                               ; preds = %70, %83
  %89 = phi i64 [ %58, %70 ], [ %87, %83 ]
  %90 = phi i64 [ %55, %70 ], [ %86, %83 ]
  %91 = getelementptr inbounds %"class.tutorial::Person", %"class.tutorial::Person"* %0, i64 0, i32 3
  %92 = getelementptr inbounds %"class.tutorial::Person", %"class.tutorial::Person"* %1, i64 0, i32 3
  %93 = and i64 %90, 1
  %94 = icmp eq i64 %93, 0
  br i1 %94, label %99, label %95, !prof !12

95:                                               ; preds = %88
  %96 = inttoptr i64 %89 to %"struct.google::protobuf::internal::InternalMetadata::ContainerBase"*
  %97 = getelementptr inbounds %"struct.google::protobuf::internal::InternalMetadata::ContainerBase", %"struct.google::protobuf::internal::InternalMetadata::ContainerBase"* %96, i64 0, i32 0
  %98 = load %"class.google::protobuf::Arena"*, %"class.google::protobuf::Arena"** %97, align 8, !tbaa !32
  br label %101

99:                                               ; preds = %88
  %100 = inttoptr i64 %89 to %"class.google::protobuf::Arena"*
  br label %101

101:                                              ; preds = %95, %99
  %102 = phi %"class.google::protobuf::Arena"* [ %98, %95 ], [ %100, %99 ]
  %103 = getelementptr inbounds %"struct.google::protobuf::internal::ArenaStringPtr", %"struct.google::protobuf::internal::ArenaStringPtr"* %91, i64 0, i32 0
  %104 = load %"class.std::__cxx11::basic_string"*, %"class.std::__cxx11::basic_string"** %103, align 8, !tbaa !18
  %105 = icmp eq %"class.std::__cxx11::basic_string"* %104, bitcast (%"class.google::protobuf::internal::ExplicitlyConstructed.20"* @_ZN6google8protobuf8internal26fixed_address_empty_stringB5cxx11E to %"class.std::__cxx11::basic_string"*)
  br i1 %105, label %106, label %112

106:                                              ; preds = %101
  %107 = getelementptr inbounds %"struct.google::protobuf::internal::ArenaStringPtr", %"struct.google::protobuf::internal::ArenaStringPtr"* %92, i64 0, i32 0
  %108 = load %"class.std::__cxx11::basic_string"*, %"class.std::__cxx11::basic_string"** %107, align 8, !tbaa !18
  %109 = icmp eq %"class.std::__cxx11::basic_string"* %108, bitcast (%"class.google::protobuf::internal::ExplicitlyConstructed.20"* @_ZN6google8protobuf8internal26fixed_address_empty_stringB5cxx11E to %"class.std::__cxx11::basic_string"*)
  br i1 %109, label %121, label %110

110:                                              ; preds = %106
  tail call void @_ZN6google8protobuf8internal14ArenaStringPtr14CreateInstanceEPNS0_5ArenaEPKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE(%"struct.google::protobuf::internal::ArenaStringPtr"* noundef nonnull align 8 dereferenceable(8) %91, %"class.google::protobuf::Arena"* noundef %102, %"class.std::__cxx11::basic_string"* noundef bitcast (%"class.google::protobuf::internal::ExplicitlyConstructed.20"* @_ZN6google8protobuf8internal26fixed_address_empty_stringB5cxx11E to %"class.std::__cxx11::basic_string"*))
  %111 = load %"class.std::__cxx11::basic_string"*, %"class.std::__cxx11::basic_string"** %103, align 8, !tbaa !18
  br label %112

112:                                              ; preds = %110, %101
  %113 = phi %"class.std::__cxx11::basic_string"* [ %111, %110 ], [ %104, %101 ]
  %114 = getelementptr inbounds %"struct.google::protobuf::internal::ArenaStringPtr", %"struct.google::protobuf::internal::ArenaStringPtr"* %92, i64 0, i32 0
  %115 = load %"class.std::__cxx11::basic_string"*, %"class.std::__cxx11::basic_string"** %114, align 8, !tbaa !18
  %116 = icmp eq %"class.std::__cxx11::basic_string"* %115, bitcast (%"class.google::protobuf::internal::ExplicitlyConstructed.20"* @_ZN6google8protobuf8internal26fixed_address_empty_stringB5cxx11E to %"class.std::__cxx11::basic_string"*)
  br i1 %116, label %117, label %119

117:                                              ; preds = %112
  tail call void @_ZN6google8protobuf8internal14ArenaStringPtr14CreateInstanceEPNS0_5ArenaEPKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE(%"struct.google::protobuf::internal::ArenaStringPtr"* noundef nonnull align 8 dereferenceable(8) %92, %"class.google::protobuf::Arena"* noundef %102, %"class.std::__cxx11::basic_string"* noundef bitcast (%"class.google::protobuf::internal::ExplicitlyConstructed.20"* @_ZN6google8protobuf8internal26fixed_address_empty_stringB5cxx11E to %"class.std::__cxx11::basic_string"*))
  %118 = load %"class.std::__cxx11::basic_string"*, %"class.std::__cxx11::basic_string"** %114, align 8, !tbaa !18
  br label %119

119:                                              ; preds = %117, %112
  %120 = phi %"class.std::__cxx11::basic_string"* [ %118, %117 ], [ %115, %112 ]
  tail call void @_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE4swapERS4_(%"class.std::__cxx11::basic_string"* noundef nonnull align 8 dereferenceable(32) %113, %"class.std::__cxx11::basic_string"* noundef nonnull align 8 dereferenceable(32) %120) #24
  br label %121

121:                                              ; preds = %106, %119
  %122 = getelementptr inbounds %"class.tutorial::Person", %"class.tutorial::Person"* %0, i64 0, i32 4
  %123 = getelementptr inbounds %"class.tutorial::Person", %"class.tutorial::Person"* %1, i64 0, i32 4
  %124 = bitcast %"class.google::protobuf::Timestamp"** %122 to i64*
  %125 = load i64, i64* %124, align 8
  %126 = bitcast %"class.google::protobuf::Timestamp"** %123 to i64*
  %127 = load i64, i64* %126, align 1
  store i64 %127, i64* %124, align 8
  store i64 %125, i64* %126, align 1
  %128 = getelementptr inbounds %"class.google::protobuf::Timestamp"*, %"class.google::protobuf::Timestamp"** %122, i64 1
  %129 = getelementptr inbounds %"class.google::protobuf::Timestamp"*, %"class.google::protobuf::Timestamp"** %123, i64 1
  %130 = bitcast %"class.google::protobuf::Timestamp"** %128 to i32*
  %131 = load i32, i32* %130, align 8
  %132 = bitcast %"class.google::protobuf::Timestamp"** %129 to i32*
  %133 = load i32, i32* %132, align 1
  store i32 %133, i32* %130, align 8
  store i32 %131, i32* %132, align 1
  ret void
}

; Function Attrs: mustprogress uwtable
define dso_local { %"class.google::protobuf::Descriptor"*, %"class.google::protobuf::Reflection"* } @_ZNK8tutorial6Person11GetMetadataEv(%"class.tutorial::Person"* nocapture nonnull readnone align 8 %0) unnamed_addr #4 align 2 {
  tail call void @_ZN6google8protobuf8internal17AssignDescriptorsEPKNS1_15DescriptorTableEb(%"struct.google::protobuf::internal::DescriptorTable"* noundef nonnull @descriptor_table_addressbook_2eproto, i1 noundef zeroext false)
  %2 = load %"struct.google::protobuf::Metadata"*, %"struct.google::protobuf::Metadata"** getelementptr inbounds (%"struct.google::protobuf::internal::DescriptorTable", %"struct.google::protobuf::internal::DescriptorTable"* @descriptor_table_addressbook_2eproto, i64 0, i32 13), align 8, !tbaa !52
  %3 = getelementptr inbounds %"struct.google::protobuf::Metadata", %"struct.google::protobuf::Metadata"* %2, i64 1, i32 0
  %4 = load %"class.google::protobuf::Descriptor"*, %"class.google::protobuf::Descriptor"** %3, align 8, !tbaa.struct !54
  %5 = getelementptr inbounds %"struct.google::protobuf::Metadata", %"struct.google::protobuf::Metadata"* %2, i64 1, i32 1
  %6 = load %"class.google::protobuf::Reflection"*, %"class.google::protobuf::Reflection"** %5, align 8, !tbaa.struct !55
  %7 = insertvalue { %"class.google::protobuf::Descriptor"*, %"class.google::protobuf::Reflection"* } poison, %"class.google::protobuf::Descriptor"* %4, 0
  %8 = insertvalue { %"class.google::protobuf::Descriptor"*, %"class.google::protobuf::Reflection"* } %7, %"class.google::protobuf::Reflection"* %6, 1
  ret { %"class.google::protobuf::Descriptor"*, %"class.google::protobuf::Reflection"* } %8
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind readnone uwtable willreturn
define dso_local void @_ZN8tutorial11AddressBook21InitAsDefaultInstanceEv() local_unnamed_addr #5 align 2 {
  ret void
}

; Function Attrs: uwtable
define dso_local void @_ZN8tutorial11AddressBookC2EPN6google8protobuf5ArenaE(%"class.tutorial::AddressBook"* noundef nonnull align 8 dereferenceable(48) %0, %"class.google::protobuf::Arena"* noundef %1) unnamed_addr #3 align 2 personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*) {
  %3 = getelementptr inbounds %"class.tutorial::AddressBook", %"class.tutorial::AddressBook"* %0, i64 0, i32 0, i32 0, i32 1
  %4 = bitcast %"class.google::protobuf::internal::InternalMetadata"* %3 to %"class.google::protobuf::Arena"**
  store %"class.google::protobuf::Arena"* %1, %"class.google::protobuf::Arena"** %4, align 8, !tbaa !5
  %5 = getelementptr inbounds %"class.tutorial::AddressBook", %"class.tutorial::AddressBook"* %0, i64 0, i32 0, i32 0, i32 0
  store i32 (...)** bitcast (i8** getelementptr inbounds ({ [22 x i8*] }, { [22 x i8*] }* @_ZTVN8tutorial11AddressBookE, i64 0, inrange i32 0, i64 2) to i32 (...)**), i32 (...)*** %5, align 8, !tbaa !10
  %6 = getelementptr inbounds %"class.tutorial::AddressBook", %"class.tutorial::AddressBook"* %0, i64 0, i32 1
  %7 = getelementptr inbounds %"class.google::protobuf::RepeatedPtrField.18", %"class.google::protobuf::RepeatedPtrField.18"* %6, i64 0, i32 0, i32 0
  store %"class.google::protobuf::Arena"* %1, %"class.google::protobuf::Arena"** %7, align 8, !tbaa !16
  %8 = getelementptr inbounds %"class.tutorial::AddressBook", %"class.tutorial::AddressBook"* %0, i64 0, i32 1, i32 0, i32 1
  %9 = bitcast i32* %8 to i8*
  tail call void @llvm.memset.p0i8.i64(i8* noundef nonnull align 8 dereferenceable(20) %9, i8 0, i64 20, i1 false)
  %10 = load atomic i32, i32* getelementptr inbounds ({ { { i32 }, i32, i32, void ()* }, [1 x i8*] }, { { { i32 }, i32, i32, void ()* }, [1 x i8*] }* @scc_info_AddressBook_addressbook_2eproto, i64 0, i32 0, i32 0, i32 0) acquire, align 8
  %11 = icmp eq i32 %10, 0
  br i1 %11, label %13, label %12, !prof !12

12:                                               ; preds = %2
  invoke void @_ZN6google8protobuf8internal11InitSCCImplEPNS1_11SCCInfoBaseE(%"struct.google::protobuf::internal::SCCInfoBase"* noundef nonnull bitcast ({ { { i32 }, i32, i32, void ()* }, [1 x i8*] }* @scc_info_AddressBook_addressbook_2eproto to %"struct.google::protobuf::internal::SCCInfoBase"*))
          to label %13 unwind label %14

13:                                               ; preds = %2, %12
  ret void

14:                                               ; preds = %12
  %15 = landingpad { i8*, i32 }
          cleanup
  tail call void @_ZN6google8protobuf16RepeatedPtrFieldIN8tutorial6PersonEED2Ev(%"class.google::protobuf::RepeatedPtrField.18"* noundef nonnull align 8 dereferenceable(24) %6) #24
  resume { i8*, i32 } %15
}

; Function Attrs: nounwind uwtable
define linkonce_odr dso_local void @_ZN6google8protobuf16RepeatedPtrFieldIN8tutorial6PersonEED2Ev(%"class.google::protobuf::RepeatedPtrField.18"* noundef nonnull align 8 dereferenceable(24) %0) unnamed_addr #6 comdat align 2 personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*) {
  %2 = getelementptr inbounds %"class.google::protobuf::RepeatedPtrField.18", %"class.google::protobuf::RepeatedPtrField.18"* %0, i64 0, i32 0, i32 3
  %3 = load %"struct.google::protobuf::internal::RepeatedPtrFieldBase::Rep"*, %"struct.google::protobuf::internal::RepeatedPtrFieldBase::Rep"** %2, align 8, !tbaa !58
  %4 = icmp ne %"struct.google::protobuf::internal::RepeatedPtrFieldBase::Rep"* %3, null
  %5 = getelementptr inbounds %"class.google::protobuf::RepeatedPtrField.18", %"class.google::protobuf::RepeatedPtrField.18"* %0, i64 0, i32 0, i32 0
  %6 = load %"class.google::protobuf::Arena"*, %"class.google::protobuf::Arena"** %5, align 8
  %7 = icmp eq %"class.google::protobuf::Arena"* %6, null
  %8 = select i1 %4, i1 %7, i1 false
  br i1 %8, label %9, label %32

9:                                                ; preds = %1
  %10 = bitcast %"struct.google::protobuf::internal::RepeatedPtrFieldBase::Rep"* %3 to i8*
  %11 = getelementptr inbounds %"struct.google::protobuf::internal::RepeatedPtrFieldBase::Rep", %"struct.google::protobuf::internal::RepeatedPtrFieldBase::Rep"* %3, i64 0, i32 0
  %12 = load i32, i32* %11, align 8, !tbaa !61
  %13 = icmp sgt i32 %12, 0
  br i1 %13, label %14, label %19

14:                                               ; preds = %9
  %15 = zext i32 %12 to i64
  br label %22

16:                                               ; preds = %29
  %17 = bitcast %"struct.google::protobuf::internal::RepeatedPtrFieldBase::Rep"** %2 to i8**
  %18 = load i8*, i8** %17, align 8, !tbaa !58
  br label %19

19:                                               ; preds = %16, %9
  %20 = phi i8* [ %18, %16 ], [ %10, %9 ]
  tail call void @_ZdlPv(i8* noundef %20) #24
  %21 = load %"class.google::protobuf::Arena"*, %"class.google::protobuf::Arena"** %5, align 8, !tbaa !16
  br label %32

22:                                               ; preds = %29, %14
  %23 = phi i64 [ 0, %14 ], [ %30, %29 ]
  %24 = getelementptr inbounds %"struct.google::protobuf::internal::RepeatedPtrFieldBase::Rep", %"struct.google::protobuf::internal::RepeatedPtrFieldBase::Rep"* %3, i64 0, i32 1, i64 %23
  %25 = load i8*, i8** %24, align 8, !tbaa !27
  %26 = icmp eq i8* %25, null
  br i1 %26, label %29, label %27

27:                                               ; preds = %22
  %28 = bitcast i8* %25 to %"class.tutorial::Person"*
  tail call void @_ZN8tutorial6PersonD2Ev(%"class.tutorial::Person"* noundef nonnull align 8 dereferenceable(72) %28) #24
  tail call void @_ZdlPv(i8* noundef nonnull %25) #26
  br label %29

29:                                               ; preds = %27, %22
  %30 = add nuw nsw i64 %23, 1
  %31 = icmp eq i64 %30, %15
  br i1 %31, label %16, label %22, !llvm.loop !70

32:                                               ; preds = %19, %1
  %33 = phi %"class.google::protobuf::Arena"* [ %21, %19 ], [ %6, %1 ]
  store %"struct.google::protobuf::internal::RepeatedPtrFieldBase::Rep"* null, %"struct.google::protobuf::internal::RepeatedPtrFieldBase::Rep"** %2, align 8, !tbaa !58
  %34 = icmp eq %"class.google::protobuf::Arena"* %33, null
  br i1 %34, label %41, label %35

35:                                               ; preds = %32
  %36 = getelementptr inbounds %"class.google::protobuf::Arena", %"class.google::protobuf::Arena"* %33, i64 0, i32 0
  %37 = invoke noundef i64 @_ZNK6google8protobuf8internal9ArenaImpl14SpaceAllocatedEv(%"class.google::protobuf::internal::ArenaImpl"* noundef nonnull align 8 dereferenceable(88) %36)
          to label %41 unwind label %38

38:                                               ; preds = %35
  %39 = landingpad { i8*, i32 }
          catch i8* null
  %40 = extractvalue { i8*, i32 } %39, 0
  tail call void @__clang_call_terminate(i8* %40) #25
  unreachable

41:                                               ; preds = %32, %35
  ret void
}

; Function Attrs: uwtable
define dso_local void @_ZN8tutorial11AddressBookC2ERKS0_(%"class.tutorial::AddressBook"* noundef nonnull align 8 dereferenceable(48) %0, %"class.tutorial::AddressBook"* noundef nonnull align 8 dereferenceable(48) %1) unnamed_addr #3 align 2 personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*) {
  %3 = getelementptr inbounds %"class.tutorial::AddressBook", %"class.tutorial::AddressBook"* %0, i64 0, i32 0, i32 0, i32 1, i32 0
  store i8* null, i8** %3, align 8, !tbaa !5
  %4 = getelementptr inbounds %"class.tutorial::AddressBook", %"class.tutorial::AddressBook"* %0, i64 0, i32 0, i32 0, i32 0
  store i32 (...)** bitcast (i8** getelementptr inbounds ({ [22 x i8*] }, { [22 x i8*] }* @_ZTVN8tutorial11AddressBookE, i64 0, inrange i32 0, i64 2) to i32 (...)**), i32 (...)*** %4, align 8, !tbaa !10
  %5 = getelementptr inbounds %"class.tutorial::AddressBook", %"class.tutorial::AddressBook"* %0, i64 0, i32 1
  %6 = getelementptr inbounds %"class.google::protobuf::RepeatedPtrField.18", %"class.google::protobuf::RepeatedPtrField.18"* %5, i64 0, i32 0
  %7 = bitcast %"class.google::protobuf::RepeatedPtrField.18"* %5 to i8*
  tail call void @llvm.memset.p0i8.i64(i8* noundef nonnull align 8 dereferenceable(24) %7, i8 0, i64 24, i1 false) #24
  %8 = getelementptr inbounds %"class.tutorial::AddressBook", %"class.tutorial::AddressBook"* %1, i64 0, i32 1, i32 0
  invoke void @_ZN6google8protobuf8internal20RepeatedPtrFieldBase9MergeFromINS0_16RepeatedPtrFieldIN8tutorial6PersonEE11TypeHandlerEEEvRKS2_(%"class.google::protobuf::internal::RepeatedPtrFieldBase"* noundef nonnull align 8 dereferenceable(24) %6, %"class.google::protobuf::internal::RepeatedPtrFieldBase"* noundef nonnull align 8 dereferenceable(24) %8)
          to label %11 unwind label %9

9:                                                ; preds = %2
  %10 = landingpad { i8*, i32 }
          cleanup
  tail call void @_ZN6google8protobuf8internal20RepeatedPtrFieldBaseD2Ev(%"class.google::protobuf::internal::RepeatedPtrFieldBase"* noundef nonnull align 8 dereferenceable(24) %6) #24
  br label %39

11:                                               ; preds = %2
  %12 = getelementptr inbounds %"class.tutorial::AddressBook", %"class.tutorial::AddressBook"* %0, i64 0, i32 2, i32 0, i32 0, i32 0
  store i32 0, i32* %12, align 8, !tbaa !13
  %13 = getelementptr inbounds %"class.tutorial::AddressBook", %"class.tutorial::AddressBook"* %0, i64 0, i32 0, i32 0, i32 1
  %14 = getelementptr inbounds %"class.tutorial::AddressBook", %"class.tutorial::AddressBook"* %1, i64 0, i32 0, i32 0, i32 1, i32 0
  %15 = load i8*, i8** %14, align 8, !tbaa !5
  %16 = ptrtoint i8* %15 to i64
  %17 = and i64 %16, 1
  %18 = icmp eq i64 %17, 0
  br i1 %18, label %36, label %19

19:                                               ; preds = %11
  %20 = and i64 %16, -2
  %21 = inttoptr i64 %20 to %"struct.google::protobuf::internal::InternalMetadata::Container"*
  %22 = getelementptr inbounds %"struct.google::protobuf::internal::InternalMetadata::Container", %"struct.google::protobuf::internal::InternalMetadata::Container"* %21, i64 0, i32 1
  %23 = getelementptr inbounds %"class.google::protobuf::internal::InternalMetadata", %"class.google::protobuf::internal::InternalMetadata"* %13, i64 0, i32 0
  %24 = load i8*, i8** %23, align 8, !tbaa !5
  %25 = ptrtoint i8* %24 to i64
  %26 = and i64 %25, 1
  %27 = icmp eq i64 %26, 0
  br i1 %27, label %32, label %28, !prof !41

28:                                               ; preds = %19
  %29 = and i64 %25, -2
  %30 = inttoptr i64 %29 to %"struct.google::protobuf::internal::InternalMetadata::Container"*
  %31 = getelementptr inbounds %"struct.google::protobuf::internal::InternalMetadata::Container", %"struct.google::protobuf::internal::InternalMetadata::Container"* %30, i64 0, i32 1
  br label %34

32:                                               ; preds = %19
  %33 = invoke noundef %"class.google::protobuf::UnknownFieldSet"* @_ZN6google8protobuf8internal16InternalMetadata27mutable_unknown_fields_slowINS0_15UnknownFieldSetEEEPT_v(%"class.google::protobuf::internal::InternalMetadata"* noundef nonnull align 8 dereferenceable(8) %13)
          to label %34 unwind label %37

34:                                               ; preds = %32, %28
  %35 = phi %"class.google::protobuf::UnknownFieldSet"* [ %31, %28 ], [ %33, %32 ]
  invoke void @_ZN6google8protobuf15UnknownFieldSet9MergeFromERKS1_(%"class.google::protobuf::UnknownFieldSet"* noundef nonnull align 8 dereferenceable(24) %35, %"class.google::protobuf::UnknownFieldSet"* noundef nonnull align 8 dereferenceable(24) %22)
          to label %36 unwind label %37

36:                                               ; preds = %11, %34
  ret void

37:                                               ; preds = %34, %32
  %38 = landingpad { i8*, i32 }
          cleanup
  tail call void @_ZN6google8protobuf16RepeatedPtrFieldIN8tutorial6PersonEED2Ev(%"class.google::protobuf::RepeatedPtrField.18"* noundef nonnull align 8 dereferenceable(24) %5) #24
  br label %39

39:                                               ; preds = %9, %37
  %40 = phi { i8*, i32 } [ %38, %37 ], [ %10, %9 ]
  resume { i8*, i32 } %40
}

; Function Attrs: nounwind uwtable
define dso_local void @_ZN8tutorial11AddressBookD2Ev(%"class.tutorial::AddressBook"* noundef nonnull align 8 dereferenceable(48) %0) unnamed_addr #6 align 2 personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*) {
  %2 = alloca %"class.google::protobuf::internal::LogMessage", align 8
  %3 = alloca %"class.google::protobuf::internal::LogFinisher", align 1
  %4 = getelementptr inbounds %"class.tutorial::AddressBook", %"class.tutorial::AddressBook"* %0, i64 0, i32 0, i32 0, i32 1, i32 0
  %5 = load i8*, i8** %4, align 8, !tbaa !5
  %6 = ptrtoint i8* %5 to i64
  %7 = and i64 %6, 1
  %8 = icmp eq i64 %7, 0
  %9 = and i64 %6, -2
  br i1 %8, label %14, label %10, !prof !12

10:                                               ; preds = %1
  %11 = inttoptr i64 %9 to %"struct.google::protobuf::internal::InternalMetadata::ContainerBase"*
  %12 = getelementptr inbounds %"struct.google::protobuf::internal::InternalMetadata::ContainerBase", %"struct.google::protobuf::internal::InternalMetadata::ContainerBase"* %11, i64 0, i32 0
  %13 = load %"class.google::protobuf::Arena"*, %"class.google::protobuf::Arena"** %12, align 8, !tbaa !32
  br label %16

14:                                               ; preds = %1
  %15 = inttoptr i64 %9 to %"class.google::protobuf::Arena"*
  br label %16

16:                                               ; preds = %14, %10
  %17 = phi %"class.google::protobuf::Arena"* [ %13, %10 ], [ %15, %14 ]
  %18 = icmp eq %"class.google::protobuf::Arena"* %17, null
  %19 = getelementptr inbounds %"class.google::protobuf::internal::LogFinisher", %"class.google::protobuf::internal::LogFinisher"* %3, i64 0, i32 0
  call void @llvm.lifetime.start.p0i8(i64 1, i8* nonnull %19) #24
  br i1 %18, label %25, label %20

20:                                               ; preds = %16
  %21 = bitcast %"class.google::protobuf::internal::LogMessage"* %2 to i8*
  call void @llvm.lifetime.start.p0i8(i64 56, i8* nonnull %21) #24
  invoke void @_ZN6google8protobuf8internal10LogMessageC1ENS0_8LogLevelEPKci(%"class.google::protobuf::internal::LogMessage"* noundef nonnull align 8 dereferenceable(56) %2, i32 noundef 3, i8* noundef getelementptr inbounds ([24 x i8], [24 x i8]* @.str.5, i64 0, i64 0), i32 noundef 810)
          to label %22 unwind label %37

22:                                               ; preds = %20
  %23 = invoke noundef nonnull align 8 dereferenceable(56) %"class.google::protobuf::internal::LogMessage"* @_ZN6google8protobuf8internal10LogMessagelsEPKc(%"class.google::protobuf::internal::LogMessage"* noundef nonnull align 8 dereferenceable(56) %2, i8* noundef getelementptr inbounds ([38 x i8], [38 x i8]* @.str.12, i64 0, i64 0))
          to label %24 unwind label %27

24:                                               ; preds = %22
  invoke void @_ZN6google8protobuf8internal11LogFinisheraSERNS1_10LogMessageE(%"class.google::protobuf::internal::LogFinisher"* noundef nonnull align 1 dereferenceable(1) %3, %"class.google::protobuf::internal::LogMessage"* noundef nonnull align 8 dereferenceable(56) %23)
          to label %26 unwind label %29

25:                                               ; preds = %16
  call void @llvm.lifetime.end.p0i8(i64 1, i8* nonnull %19) #24
  br label %33

26:                                               ; preds = %24
  call void @llvm.lifetime.end.p0i8(i64 1, i8* nonnull %19) #24
  call void @_ZN6google8protobuf8internal10LogMessageD1Ev(%"class.google::protobuf::internal::LogMessage"* noundef nonnull align 8 dereferenceable(56) %2) #24
  call void @llvm.lifetime.end.p0i8(i64 56, i8* nonnull %21) #24
  br label %33

27:                                               ; preds = %22
  %28 = landingpad { i8*, i32 }
          catch i8* null
  br label %31

29:                                               ; preds = %24
  %30 = landingpad { i8*, i32 }
          catch i8* null
  call void @llvm.lifetime.end.p0i8(i64 1, i8* nonnull %19) #24
  br label %31

31:                                               ; preds = %29, %27
  %32 = phi { i8*, i32 } [ %30, %29 ], [ %28, %27 ]
  call void @_ZN6google8protobuf8internal10LogMessageD1Ev(%"class.google::protobuf::internal::LogMessage"* noundef nonnull align 8 dereferenceable(56) %2) #24
  call void @llvm.lifetime.end.p0i8(i64 56, i8* nonnull %21) #24
  br label %39

33:                                               ; preds = %26, %25
  %34 = getelementptr inbounds %"class.tutorial::AddressBook", %"class.tutorial::AddressBook"* %0, i64 0, i32 0, i32 0, i32 1
  invoke void @_ZN6google8protobuf8internal16InternalMetadata6DeleteINS0_15UnknownFieldSetEEEvv(%"class.google::protobuf::internal::InternalMetadata"* noundef nonnull align 8 dereferenceable(8) %34)
          to label %35 unwind label %37

35:                                               ; preds = %33
  %36 = getelementptr inbounds %"class.tutorial::AddressBook", %"class.tutorial::AddressBook"* %0, i64 0, i32 1
  call void @_ZN6google8protobuf16RepeatedPtrFieldIN8tutorial6PersonEED2Ev(%"class.google::protobuf::RepeatedPtrField.18"* noundef nonnull align 8 dereferenceable(24) %36) #24
  ret void

37:                                               ; preds = %20, %33
  %38 = landingpad { i8*, i32 }
          catch i8* null
  br label %39

39:                                               ; preds = %31, %37
  %40 = phi { i8*, i32 } [ %38, %37 ], [ %32, %31 ]
  %41 = extractvalue { i8*, i32 } %40, 0
  %42 = getelementptr inbounds %"class.tutorial::AddressBook", %"class.tutorial::AddressBook"* %0, i64 0, i32 1
  call void @_ZN6google8protobuf16RepeatedPtrFieldIN8tutorial6PersonEED2Ev(%"class.google::protobuf::RepeatedPtrField.18"* noundef nonnull align 8 dereferenceable(24) %42) #24
  call void @__clang_call_terminate(i8* %41) #25
  unreachable
}

; Function Attrs: nounwind uwtable
define dso_local void @_ZN8tutorial11AddressBookD0Ev(%"class.tutorial::AddressBook"* noundef nonnull align 8 dereferenceable(48) %0) unnamed_addr #6 align 2 {
  tail call void @_ZN8tutorial11AddressBookD2Ev(%"class.tutorial::AddressBook"* noundef nonnull align 8 dereferenceable(48) %0) #24
  %2 = bitcast %"class.tutorial::AddressBook"* %0 to i8*
  tail call void @_ZdlPv(i8* noundef nonnull %2) #26
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind readnone uwtable willreturn
define dso_local void @_ZN8tutorial11AddressBook9ArenaDtorEPv(i8* nocapture noundef %0) local_unnamed_addr #5 align 2 {
  ret void
}

; Function Attrs: mustprogress nofree norecurse nounwind uwtable willreturn
define dso_local void @_ZNK8tutorial11AddressBook13SetCachedSizeEi(%"class.tutorial::AddressBook"* nocapture noundef nonnull writeonly align 8 dereferenceable(48) %0, i32 noundef %1) unnamed_addr #11 align 2 personality i32 (...)* @__gxx_personality_v0 {
  %3 = getelementptr inbounds %"class.tutorial::AddressBook", %"class.tutorial::AddressBook"* %0, i64 0, i32 2, i32 0, i32 0, i32 0
  store atomic i32 %1, i32* %3 monotonic, align 8
  ret void
}

; Function Attrs: mustprogress uwtable
define dso_local noundef nonnull align 8 dereferenceable(48) %"class.tutorial::AddressBook"* @_ZN8tutorial11AddressBook16default_instanceEv() local_unnamed_addr #4 align 2 {
  %1 = load atomic i32, i32* getelementptr inbounds ({ { { i32 }, i32, i32, void ()* }, [1 x i8*] }, { { { i32 }, i32, i32, void ()* }, [1 x i8*] }* @scc_info_AddressBook_addressbook_2eproto, i64 0, i32 0, i32 0, i32 0) acquire, align 8
  %2 = icmp eq i32 %1, 0
  br i1 %2, label %4, label %3, !prof !12

3:                                                ; preds = %0
  tail call void @_ZN6google8protobuf8internal11InitSCCImplEPNS1_11SCCInfoBaseE(%"struct.google::protobuf::internal::SCCInfoBase"* noundef nonnull bitcast ({ { { i32 }, i32, i32, void ()* }, [1 x i8*] }* @scc_info_AddressBook_addressbook_2eproto to %"struct.google::protobuf::internal::SCCInfoBase"*))
  br label %4

4:                                                ; preds = %0, %3
  ret %"class.tutorial::AddressBook"* bitcast (%"class.tutorial::AddressBookDefaultTypeInternal"* @_ZN8tutorial30_AddressBook_default_instance_E to %"class.tutorial::AddressBook"*)
}

; Function Attrs: uwtable
define dso_local void @_ZN8tutorial11AddressBook5ClearEv(%"class.tutorial::AddressBook"* noundef nonnull align 8 dereferenceable(48) %0) unnamed_addr #3 align 2 {
  %2 = getelementptr inbounds %"class.tutorial::AddressBook", %"class.tutorial::AddressBook"* %0, i64 0, i32 1, i32 0
  tail call void @_ZN6google8protobuf8internal20RepeatedPtrFieldBase5ClearINS0_16RepeatedPtrFieldIN8tutorial6PersonEE11TypeHandlerEEEvv(%"class.google::protobuf::internal::RepeatedPtrFieldBase"* noundef nonnull align 8 dereferenceable(24) %2)
  %3 = getelementptr inbounds %"class.tutorial::AddressBook", %"class.tutorial::AddressBook"* %0, i64 0, i32 0, i32 0, i32 1, i32 0
  %4 = load i8*, i8** %3, align 8, !tbaa !5
  %5 = ptrtoint i8* %4 to i64
  %6 = and i64 %5, 1
  %7 = icmp eq i64 %6, 0
  br i1 %7, label %18, label %8

8:                                                ; preds = %1
  %9 = and i64 %5, -2
  %10 = inttoptr i64 %9 to %"struct.google::protobuf::internal::InternalMetadata::Container"*
  %11 = getelementptr inbounds %"struct.google::protobuf::internal::InternalMetadata::Container", %"struct.google::protobuf::internal::InternalMetadata::Container"* %10, i64 0, i32 1
  %12 = getelementptr inbounds %"class.google::protobuf::UnknownFieldSet", %"class.google::protobuf::UnknownFieldSet"* %11, i64 0, i32 0, i32 0, i32 0, i32 0, i32 0
  %13 = load %"class.google::protobuf::UnknownField"*, %"class.google::protobuf::UnknownField"** %12, align 8, !tbaa !27
  %14 = getelementptr inbounds %"struct.google::protobuf::internal::InternalMetadata::Container", %"struct.google::protobuf::internal::InternalMetadata::Container"* %10, i64 0, i32 1, i32 0, i32 0, i32 0, i32 0, i32 1
  %15 = load %"class.google::protobuf::UnknownField"*, %"class.google::protobuf::UnknownField"** %14, align 8, !tbaa !27
  %16 = icmp eq %"class.google::protobuf::UnknownField"* %13, %15
  br i1 %16, label %18, label %17

17:                                               ; preds = %8
  tail call void @_ZN6google8protobuf15UnknownFieldSet13ClearFallbackEv(%"class.google::protobuf::UnknownFieldSet"* noundef nonnull align 8 dereferenceable(24) %11)
  br label %18

18:                                               ; preds = %1, %8, %17
  ret void
}

; Function Attrs: mustprogress uwtable
define dso_local noundef i8* @_ZN8tutorial11AddressBook14_InternalParseEPKcPN6google8protobuf8internal12ParseContextE(%"class.tutorial::AddressBook"* noundef nonnull align 8 dereferenceable(48) %0, i8* noundef %1, %"class.google::protobuf::internal::ParseContext"* noundef %2) unnamed_addr #4 align 2 {
  %4 = alloca i8*, align 8
  store i8* %1, i8** %4, align 8, !tbaa !27
  %5 = getelementptr inbounds %"class.google::protobuf::internal::ParseContext", %"class.google::protobuf::internal::ParseContext"* %2, i64 0, i32 0
  %6 = getelementptr inbounds %"class.google::protobuf::internal::ParseContext", %"class.google::protobuf::internal::ParseContext"* %2, i64 0, i32 2
  %7 = load i32, i32* %6, align 4, !tbaa !38
  %8 = call noundef zeroext i1 @_ZN6google8protobuf8internal18EpsCopyInputStream13DoneWithCheckEPPKci(%"class.google::protobuf::internal::EpsCopyInputStream"* noundef nonnull align 8 dereferenceable(88) %5, i8** noundef nonnull %4, i32 noundef %7)
  br i1 %8, label %162, label %9

9:                                                ; preds = %3
  %10 = getelementptr inbounds %"class.tutorial::AddressBook", %"class.tutorial::AddressBook"* %0, i64 0, i32 0, i32 0, i32 1
  %11 = getelementptr inbounds %"class.google::protobuf::internal::InternalMetadata", %"class.google::protobuf::internal::InternalMetadata"* %10, i64 0, i32 0
  %12 = getelementptr inbounds %"class.tutorial::AddressBook", %"class.tutorial::AddressBook"* %0, i64 0, i32 1
  %13 = getelementptr inbounds %"class.google::protobuf::RepeatedPtrField.18", %"class.google::protobuf::RepeatedPtrField.18"* %12, i64 0, i32 0
  %14 = getelementptr inbounds %"class.tutorial::AddressBook", %"class.tutorial::AddressBook"* %0, i64 0, i32 1, i32 0, i32 3
  %15 = getelementptr inbounds %"class.tutorial::AddressBook", %"class.tutorial::AddressBook"* %0, i64 0, i32 1, i32 0, i32 1
  %16 = getelementptr inbounds %"class.tutorial::AddressBook", %"class.tutorial::AddressBook"* %0, i64 0, i32 1, i32 0, i32 2
  %17 = getelementptr inbounds %"class.google::protobuf::RepeatedPtrField.18", %"class.google::protobuf::RepeatedPtrField.18"* %12, i64 0, i32 0, i32 0
  %18 = getelementptr inbounds %"class.google::protobuf::internal::ParseContext", %"class.google::protobuf::internal::ParseContext"* %2, i64 0, i32 1
  %19 = getelementptr inbounds %"class.google::protobuf::internal::ParseContext", %"class.google::protobuf::internal::ParseContext"* %2, i64 0, i32 0, i32 8
  %20 = getelementptr inbounds %"class.google::protobuf::internal::ParseContext", %"class.google::protobuf::internal::ParseContext"* %2, i64 0, i32 0, i32 4
  %21 = getelementptr inbounds %"class.google::protobuf::internal::ParseContext", %"class.google::protobuf::internal::ParseContext"* %2, i64 0, i32 0, i32 1
  %22 = getelementptr inbounds %"class.google::protobuf::internal::ParseContext", %"class.google::protobuf::internal::ParseContext"* %2, i64 0, i32 0, i32 0
  br label %23

23:                                               ; preds = %9, %158
  %24 = load i8*, i8** %4, align 8, !tbaa !27
  %25 = load i8, i8* %24, align 1, !tbaa !37
  %26 = zext i8 %25 to i32
  %27 = icmp sgt i8 %25, -1
  %28 = getelementptr inbounds i8, i8* %24, i64 1
  br i1 %27, label %38, label %29

29:                                               ; preds = %23
  %30 = load i8, i8* %28, align 1, !tbaa !37
  %31 = zext i8 %30 to i32
  %32 = shl nuw nsw i32 %31, 7
  %33 = add nsw i32 %26, -128
  %34 = add nsw i32 %33, %32
  %35 = icmp sgt i8 %30, -1
  br i1 %35, label %36, label %41

36:                                               ; preds = %29
  %37 = getelementptr inbounds i8, i8* %24, i64 2
  br label %38

38:                                               ; preds = %23, %36
  %39 = phi i32 [ %34, %36 ], [ %26, %23 ]
  %40 = phi i8* [ %37, %36 ], [ %28, %23 ]
  store i8* %40, i8** %4, align 8, !tbaa !27
  br label %46

41:                                               ; preds = %29
  %42 = call { i8*, i32 } @_ZN6google8protobuf8internal15ReadTagFallbackEPKcj(i8* noundef nonnull %24, i32 noundef %34)
  %43 = extractvalue { i8*, i32 } %42, 0
  %44 = extractvalue { i8*, i32 } %42, 1
  store i8* %43, i8** %4, align 8, !tbaa !27
  %45 = icmp eq i8* %43, null
  br i1 %45, label %161, label %46, !prof !41

46:                                               ; preds = %38, %41
  %47 = phi i8* [ %40, %38 ], [ %43, %41 ]
  %48 = phi i32 [ %39, %38 ], [ %44, %41 ]
  %49 = icmp eq i32 %48, 10
  br i1 %49, label %50, label %133, !prof !71

50:                                               ; preds = %46
  %51 = getelementptr inbounds i8, i8* %47, i64 -1
  br label %52

52:                                               ; preds = %130, %50
  %53 = phi i8* [ %114, %130 ], [ %51, %50 ]
  %54 = getelementptr inbounds i8, i8* %53, i64 1
  store i8* %54, i8** %4, align 8, !tbaa !27
  %55 = load %"struct.google::protobuf::internal::RepeatedPtrFieldBase::Rep"*, %"struct.google::protobuf::internal::RepeatedPtrFieldBase::Rep"** %14, align 8, !tbaa !58
  %56 = icmp eq %"struct.google::protobuf::internal::RepeatedPtrFieldBase::Rep"* %55, null
  br i1 %56, label %57, label %59

57:                                               ; preds = %52
  %58 = load i32, i32* %16, align 4, !tbaa !59
  br label %73

59:                                               ; preds = %52
  %60 = load i32, i32* %15, align 8, !tbaa !60
  %61 = getelementptr inbounds %"struct.google::protobuf::internal::RepeatedPtrFieldBase::Rep", %"struct.google::protobuf::internal::RepeatedPtrFieldBase::Rep"* %55, i64 0, i32 0
  %62 = load i32, i32* %61, align 8, !tbaa !61
  %63 = icmp slt i32 %60, %62
  br i1 %63, label %64, label %70

64:                                               ; preds = %59
  %65 = add nsw i32 %60, 1
  store i32 %65, i32* %15, align 8, !tbaa !60
  %66 = sext i32 %60 to i64
  %67 = getelementptr inbounds %"struct.google::protobuf::internal::RepeatedPtrFieldBase::Rep", %"struct.google::protobuf::internal::RepeatedPtrFieldBase::Rep"* %55, i64 0, i32 1, i64 %66
  %68 = bitcast i8** %67 to %"class.tutorial::Person"**
  %69 = load %"class.tutorial::Person"*, %"class.tutorial::Person"** %68, align 8, !tbaa !27
  br label %93

70:                                               ; preds = %59
  %71 = load i32, i32* %16, align 4, !tbaa !59
  %72 = icmp eq i32 %62, %71
  br i1 %72, label %73, label %79

73:                                               ; preds = %70, %57
  %74 = phi i32 [ %58, %57 ], [ %62, %70 ]
  %75 = add nsw i32 %74, 1
  call void @_ZN6google8protobuf8internal20RepeatedPtrFieldBase7ReserveEi(%"class.google::protobuf::internal::RepeatedPtrFieldBase"* noundef nonnull align 8 dereferenceable(24) %13, i32 noundef %75)
  %76 = load %"struct.google::protobuf::internal::RepeatedPtrFieldBase::Rep"*, %"struct.google::protobuf::internal::RepeatedPtrFieldBase::Rep"** %14, align 8, !tbaa !58
  %77 = getelementptr inbounds %"struct.google::protobuf::internal::RepeatedPtrFieldBase::Rep", %"struct.google::protobuf::internal::RepeatedPtrFieldBase::Rep"* %76, i64 0, i32 0
  %78 = load i32, i32* %77, align 8, !tbaa !61
  br label %79

79:                                               ; preds = %73, %70
  %80 = phi i32 [ %78, %73 ], [ %62, %70 ]
  %81 = phi %"struct.google::protobuf::internal::RepeatedPtrFieldBase::Rep"* [ %76, %73 ], [ %55, %70 ]
  %82 = getelementptr inbounds %"struct.google::protobuf::internal::RepeatedPtrFieldBase::Rep", %"struct.google::protobuf::internal::RepeatedPtrFieldBase::Rep"* %81, i64 0, i32 0
  %83 = add nsw i32 %80, 1
  store i32 %83, i32* %82, align 8, !tbaa !61
  %84 = load %"class.google::protobuf::Arena"*, %"class.google::protobuf::Arena"** %17, align 8, !tbaa !16
  %85 = call noundef %"class.tutorial::Person"* @_ZN6google8protobuf5Arena18CreateMaybeMessageIN8tutorial6PersonEJEEEPT_PS1_DpOT0_(%"class.google::protobuf::Arena"* noundef %84)
  %86 = load %"struct.google::protobuf::internal::RepeatedPtrFieldBase::Rep"*, %"struct.google::protobuf::internal::RepeatedPtrFieldBase::Rep"** %14, align 8, !tbaa !58
  %87 = load i32, i32* %15, align 8, !tbaa !60
  %88 = add nsw i32 %87, 1
  store i32 %88, i32* %15, align 8, !tbaa !60
  %89 = sext i32 %87 to i64
  %90 = getelementptr inbounds %"struct.google::protobuf::internal::RepeatedPtrFieldBase::Rep", %"struct.google::protobuf::internal::RepeatedPtrFieldBase::Rep"* %86, i64 0, i32 1, i64 %89
  %91 = bitcast i8** %90 to %"class.tutorial::Person"**
  store %"class.tutorial::Person"* %85, %"class.tutorial::Person"** %91, align 8, !tbaa !27
  %92 = load i8*, i8** %4, align 8, !tbaa !27
  br label %93

93:                                               ; preds = %64, %79
  %94 = phi i8* [ %54, %64 ], [ %92, %79 ]
  %95 = phi %"class.tutorial::Person"* [ %69, %64 ], [ %85, %79 ]
  %96 = load i8, i8* %94, align 1, !tbaa !37
  %97 = zext i8 %96 to i32
  %98 = icmp sgt i8 %96, -1
  br i1 %98, label %99, label %101

99:                                               ; preds = %93
  %100 = getelementptr inbounds i8, i8* %94, i64 1
  br label %106

101:                                              ; preds = %93
  %102 = call { i8*, i32 } @_ZN6google8protobuf8internal16ReadSizeFallbackEPKcj(i8* noundef nonnull %94, i32 noundef %97)
  %103 = extractvalue { i8*, i32 } %102, 0
  %104 = extractvalue { i8*, i32 } %102, 1
  %105 = icmp eq i8* %103, null
  br i1 %105, label %161, label %106

106:                                              ; preds = %101, %99
  %107 = phi i32 [ %97, %99 ], [ %104, %101 ]
  %108 = phi i8* [ %100, %99 ], [ %103, %101 ]
  %109 = call noundef i32 @_ZN6google8protobuf8internal18EpsCopyInputStream9PushLimitEPKci(%"class.google::protobuf::internal::EpsCopyInputStream"* noundef nonnull align 8 dereferenceable(88) %5, i8* noundef nonnull %108, i32 noundef %107)
  %110 = load i32, i32* %18, align 8, !tbaa !63
  %111 = add nsw i32 %110, -1
  store i32 %111, i32* %18, align 8, !tbaa !63
  %112 = icmp slt i32 %110, 1
  br i1 %112, label %161, label %113

113:                                              ; preds = %106
  %114 = call noundef i8* @_ZN8tutorial6Person14_InternalParseEPKcPN6google8protobuf8internal12ParseContextE(%"class.tutorial::Person"* noundef nonnull align 8 dereferenceable(72) %95, i8* noundef nonnull %108, %"class.google::protobuf::internal::ParseContext"* noundef nonnull %2)
  %115 = icmp eq i8* %114, null
  br i1 %115, label %161, label %116, !prof !41

116:                                              ; preds = %113
  %117 = load i32, i32* %18, align 8, !tbaa !63
  %118 = add nsw i32 %117, 1
  store i32 %118, i32* %18, align 8, !tbaa !63
  %119 = load i32, i32* %19, align 8, !tbaa !42
  %120 = icmp eq i32 %119, 0
  br i1 %120, label %121, label %161, !prof !12

121:                                              ; preds = %116
  %122 = load i32, i32* %20, align 4, !tbaa !64
  %123 = add nsw i32 %122, %109
  store i32 %123, i32* %20, align 4, !tbaa !64
  %124 = load i8*, i8** %21, align 8, !tbaa !65
  %125 = icmp slt i32 %123, 0
  %126 = select i1 %125, i32 %123, i32 0
  %127 = sext i32 %126 to i64
  %128 = getelementptr inbounds i8, i8* %124, i64 %127
  store i8* %128, i8** %22, align 8, !tbaa !66
  store i8* %114, i8** %4, align 8, !tbaa !27
  %129 = icmp ugt i8* %128, %114
  br i1 %129, label %130, label %158, !llvm.loop !72

130:                                              ; preds = %121
  %131 = load i8, i8* %114, align 1, !tbaa !37
  %132 = icmp eq i8 %131, 10
  br i1 %132, label %52, label %158, !llvm.loop !72

133:                                              ; preds = %46
  %134 = and i32 %48, 7
  %135 = icmp eq i32 %134, 4
  %136 = icmp eq i32 %48, 0
  %137 = or i1 %136, %135
  br i1 %137, label %138, label %140

138:                                              ; preds = %133
  %139 = add i32 %48, -1
  store i32 %139, i32* %19, align 8, !tbaa !42
  br label %162

140:                                              ; preds = %133
  %141 = zext i32 %48 to i64
  %142 = load i8*, i8** %11, align 8, !tbaa !5
  %143 = ptrtoint i8* %142 to i64
  %144 = and i64 %143, 1
  %145 = icmp eq i64 %144, 0
  br i1 %145, label %150, label %146, !prof !41

146:                                              ; preds = %140
  %147 = and i64 %143, -2
  %148 = inttoptr i64 %147 to %"struct.google::protobuf::internal::InternalMetadata::Container"*
  %149 = getelementptr inbounds %"struct.google::protobuf::internal::InternalMetadata::Container", %"struct.google::protobuf::internal::InternalMetadata::Container"* %148, i64 0, i32 1
  br label %153

150:                                              ; preds = %140
  %151 = call noundef %"class.google::protobuf::UnknownFieldSet"* @_ZN6google8protobuf8internal16InternalMetadata27mutable_unknown_fields_slowINS0_15UnknownFieldSetEEEPT_v(%"class.google::protobuf::internal::InternalMetadata"* noundef nonnull align 8 dereferenceable(8) %10)
  %152 = load i8*, i8** %4, align 8, !tbaa !27
  br label %153

153:                                              ; preds = %146, %150
  %154 = phi i8* [ %47, %146 ], [ %152, %150 ]
  %155 = phi %"class.google::protobuf::UnknownFieldSet"* [ %149, %146 ], [ %151, %150 ]
  %156 = call noundef i8* @_ZN6google8protobuf8internal17UnknownFieldParseEmPNS0_15UnknownFieldSetEPKcPNS1_12ParseContextE(i64 noundef %141, %"class.google::protobuf::UnknownFieldSet"* noundef %155, i8* noundef %154, %"class.google::protobuf::internal::ParseContext"* noundef nonnull %2)
  store i8* %156, i8** %4, align 8, !tbaa !27
  %157 = icmp eq i8* %156, null
  br i1 %157, label %161, label %158

158:                                              ; preds = %121, %130, %153
  %159 = load i32, i32* %6, align 4, !tbaa !38
  %160 = call noundef zeroext i1 @_ZN6google8protobuf8internal18EpsCopyInputStream13DoneWithCheckEPPKci(%"class.google::protobuf::internal::EpsCopyInputStream"* noundef nonnull align 8 dereferenceable(88) %5, i8** noundef nonnull %4, i32 noundef %159)
  br i1 %160, label %162, label %23

161:                                              ; preds = %153, %41, %116, %113, %106, %101
  store i8* null, i8** %4, align 8, !tbaa !27
  br label %162

162:                                              ; preds = %158, %3, %161, %138
  %163 = load i8*, i8** %4, align 8, !tbaa !27
  ret i8* %163
}

; Function Attrs: mustprogress uwtable
define dso_local noundef i8* @_ZNK8tutorial11AddressBook18_InternalSerializeEPhPN6google8protobuf2io19EpsCopyOutputStreamE(%"class.tutorial::AddressBook"* noundef nonnull align 8 dereferenceable(48) %0, i8* noundef %1, %"class.google::protobuf::io::EpsCopyOutputStream"* noundef %2) unnamed_addr #4 align 2 {
  %4 = getelementptr inbounds %"class.tutorial::AddressBook", %"class.tutorial::AddressBook"* %0, i64 0, i32 1, i32 0, i32 1
  %5 = load i32, i32* %4, align 8, !tbaa !60
  %6 = icmp eq i32 %5, 0
  br i1 %6, label %10, label %7

7:                                                ; preds = %3
  %8 = getelementptr inbounds %"class.google::protobuf::io::EpsCopyOutputStream", %"class.google::protobuf::io::EpsCopyOutputStream"* %2, i64 0, i32 0
  %9 = getelementptr inbounds %"class.tutorial::AddressBook", %"class.tutorial::AddressBook"* %0, i64 0, i32 1, i32 0
  br label %17

10:                                               ; preds = %55, %3
  %11 = phi i8* [ %1, %3 ], [ %57, %55 ]
  %12 = getelementptr inbounds %"class.tutorial::AddressBook", %"class.tutorial::AddressBook"* %0, i64 0, i32 0, i32 0, i32 1, i32 0
  %13 = load i8*, i8** %12, align 8, !tbaa !5
  %14 = ptrtoint i8* %13 to i64
  %15 = and i64 %14, 1
  %16 = icmp eq i64 %15, 0
  br i1 %16, label %65, label %60, !prof !12

17:                                               ; preds = %7, %55
  %18 = phi i8* [ %1, %7 ], [ %57, %55 ]
  %19 = phi i32 [ 0, %7 ], [ %58, %55 ]
  %20 = load i8*, i8** %8, align 8, !tbaa !44
  %21 = icmp ugt i8* %20, %18
  br i1 %21, label %24, label %22, !prof !12

22:                                               ; preds = %17
  %23 = tail call noundef i8* @_ZN6google8protobuf2io19EpsCopyOutputStream19EnsureSpaceFallbackEPh(%"class.google::protobuf::io::EpsCopyOutputStream"* noundef nonnull align 8 dereferenceable(59) %2, i8* noundef %18)
  br label %24

24:                                               ; preds = %17, %22
  %25 = phi i8* [ %23, %22 ], [ %18, %17 ]
  %26 = tail call noundef nonnull align 8 dereferenceable(72) %"class.tutorial::Person"* @_ZNK6google8protobuf8internal20RepeatedPtrFieldBase3GetINS0_16RepeatedPtrFieldIN8tutorial6PersonEE11TypeHandlerEEERKNT_4TypeEi(%"class.google::protobuf::internal::RepeatedPtrFieldBase"* noundef nonnull align 8 dereferenceable(24) %9, i32 noundef %19)
  store i8 10, i8* %25, align 1, !tbaa !37
  %27 = getelementptr inbounds i8, i8* %25, i64 1
  %28 = getelementptr inbounds %"class.tutorial::Person", %"class.tutorial::Person"* %26, i64 0, i32 6, i32 0, i32 0, i32 0
  %29 = load atomic i32, i32* %28 monotonic, align 4
  %30 = icmp ult i32 %29, 128
  %31 = trunc i32 %29 to i8
  br i1 %30, label %32, label %34

32:                                               ; preds = %24
  store i8 %31, i8* %27, align 1, !tbaa !37
  %33 = getelementptr inbounds i8, i8* %25, i64 2
  br label %55

34:                                               ; preds = %24
  %35 = or i8 %31, -128
  store i8 %35, i8* %27, align 1, !tbaa !37
  %36 = lshr i32 %29, 7
  %37 = icmp ult i32 %29, 16384
  br i1 %37, label %38, label %42

38:                                               ; preds = %34
  %39 = trunc i32 %36 to i8
  %40 = getelementptr inbounds i8, i8* %25, i64 2
  store i8 %39, i8* %40, align 1, !tbaa !37
  %41 = getelementptr inbounds i8, i8* %25, i64 3
  br label %55

42:                                               ; preds = %34
  %43 = getelementptr inbounds i8, i8* %25, i64 2
  br label %44

44:                                               ; preds = %44, %42
  %45 = phi i32 [ %36, %42 ], [ %49, %44 ]
  %46 = phi i8* [ %43, %42 ], [ %50, %44 ]
  %47 = trunc i32 %45 to i8
  %48 = or i8 %47, -128
  store i8 %48, i8* %46, align 1, !tbaa !37
  %49 = lshr i32 %45, 7
  %50 = getelementptr inbounds i8, i8* %46, i64 1
  %51 = icmp ugt i32 %45, 16383
  br i1 %51, label %44, label %52, !prof !41, !llvm.loop !68

52:                                               ; preds = %44
  %53 = trunc i32 %49 to i8
  %54 = getelementptr inbounds i8, i8* %46, i64 2
  store i8 %53, i8* %50, align 1, !tbaa !37
  br label %55

55:                                               ; preds = %32, %38, %52
  %56 = phi i8* [ %33, %32 ], [ %41, %38 ], [ %54, %52 ]
  %57 = tail call noundef i8* @_ZNK8tutorial6Person18_InternalSerializeEPhPN6google8protobuf2io19EpsCopyOutputStreamE(%"class.tutorial::Person"* noundef nonnull align 8 dereferenceable(72) %26, i8* noundef nonnull %56, %"class.google::protobuf::io::EpsCopyOutputStream"* noundef %2)
  %58 = add nuw i32 %19, 1
  %59 = icmp eq i32 %58, %5
  br i1 %59, label %10, label %17, !llvm.loop !73

60:                                               ; preds = %10
  %61 = and i64 %14, -2
  %62 = inttoptr i64 %61 to %"struct.google::protobuf::internal::InternalMetadata::Container"*
  %63 = getelementptr inbounds %"struct.google::protobuf::internal::InternalMetadata::Container", %"struct.google::protobuf::internal::InternalMetadata::Container"* %62, i64 0, i32 1
  %64 = tail call noundef i8* @_ZN6google8protobuf8internal10WireFormat37InternalSerializeUnknownFieldsToArrayERKNS0_15UnknownFieldSetEPhPNS0_2io19EpsCopyOutputStreamE(%"class.google::protobuf::UnknownFieldSet"* noundef nonnull align 8 dereferenceable(24) %63, i8* noundef %11, %"class.google::protobuf::io::EpsCopyOutputStream"* noundef %2)
  br label %65

65:                                               ; preds = %60, %10
  %66 = phi i8* [ %64, %60 ], [ %11, %10 ]
  ret i8* %66
}

; Function Attrs: uwtable
define dso_local noundef i64 @_ZNK8tutorial11AddressBook12ByteSizeLongEv(%"class.tutorial::AddressBook"* noundef nonnull align 8 dereferenceable(48) %0) unnamed_addr #3 align 2 personality i32 (...)* @__gxx_personality_v0 {
  %2 = getelementptr inbounds %"class.tutorial::AddressBook", %"class.tutorial::AddressBook"* %0, i64 0, i32 1, i32 0, i32 1
  %3 = load i32, i32* %2, align 8, !tbaa !60
  %4 = sext i32 %3 to i64
  %5 = getelementptr inbounds %"class.tutorial::AddressBook", %"class.tutorial::AddressBook"* %0, i64 0, i32 1, i32 0, i32 3
  %6 = load %"struct.google::protobuf::internal::RepeatedPtrFieldBase::Rep"*, %"struct.google::protobuf::internal::RepeatedPtrFieldBase::Rep"** %5, align 8, !tbaa !58
  %7 = icmp eq %"struct.google::protobuf::internal::RepeatedPtrFieldBase::Rep"* %6, null
  %8 = getelementptr inbounds %"struct.google::protobuf::internal::RepeatedPtrFieldBase::Rep", %"struct.google::protobuf::internal::RepeatedPtrFieldBase::Rep"* %6, i64 0, i32 1, i64 0
  %9 = select i1 %7, i8** null, i8** %8
  %10 = getelementptr inbounds i8*, i8** %9, i64 %4
  %11 = icmp eq i32 %3, 0
  br i1 %11, label %12, label %20

12:                                               ; preds = %20, %1
  %13 = phi i64 [ 0, %1 ], [ %35, %20 ]
  %14 = getelementptr inbounds %"class.tutorial::AddressBook", %"class.tutorial::AddressBook"* %0, i64 0, i32 0, i32 0, i32 1
  %15 = getelementptr inbounds %"class.google::protobuf::internal::InternalMetadata", %"class.google::protobuf::internal::InternalMetadata"* %14, i64 0, i32 0
  %16 = load i8*, i8** %15, align 8, !tbaa !5
  %17 = ptrtoint i8* %16 to i64
  %18 = and i64 %17, 1
  %19 = icmp eq i64 %18, 0
  br i1 %19, label %41, label %38, !prof !12

20:                                               ; preds = %1, %20
  %21 = phi i64 [ %35, %20 ], [ %4, %1 ]
  %22 = phi i8** [ %36, %20 ], [ %9, %1 ]
  %23 = bitcast i8** %22 to %"class.tutorial::Person"**
  %24 = load %"class.tutorial::Person"*, %"class.tutorial::Person"** %23, align 8, !tbaa !27
  %25 = tail call noundef i64 @_ZNK8tutorial6Person12ByteSizeLongEv(%"class.tutorial::Person"* noundef nonnull align 8 dereferenceable(72) %24)
  %26 = trunc i64 %25 to i32
  %27 = or i32 %26, 1
  %28 = tail call i32 @llvm.ctlz.i32(i32 %27, i1 true) #24, !range !49
  %29 = xor i32 %28, 31
  %30 = mul nuw nsw i32 %29, 9
  %31 = add nuw nsw i32 %30, 73
  %32 = lshr i32 %31, 6
  %33 = zext i32 %32 to i64
  %34 = add i64 %25, %21
  %35 = add i64 %34, %33
  %36 = getelementptr inbounds i8*, i8** %22, i64 1
  %37 = icmp eq i8** %36, %10
  br i1 %37, label %12, label %20

38:                                               ; preds = %12
  %39 = getelementptr inbounds %"class.tutorial::AddressBook", %"class.tutorial::AddressBook"* %0, i64 0, i32 2
  %40 = tail call noundef i64 @_ZN6google8protobuf8internal24ComputeUnknownFieldsSizeERKNS1_16InternalMetadataEmPNS1_10CachedSizeE(%"class.google::protobuf::internal::InternalMetadata"* noundef nonnull align 8 dereferenceable(8) %14, i64 noundef %13, %"class.google::protobuf::internal::CachedSize"* noundef nonnull %39)
  br label %44

41:                                               ; preds = %12
  %42 = trunc i64 %13 to i32
  %43 = getelementptr inbounds %"class.tutorial::AddressBook", %"class.tutorial::AddressBook"* %0, i64 0, i32 2, i32 0, i32 0, i32 0
  store atomic i32 %42, i32* %43 monotonic, align 8
  br label %44

44:                                               ; preds = %41, %38
  %45 = phi i64 [ %40, %38 ], [ %13, %41 ]
  ret i64 %45
}

; Function Attrs: mustprogress uwtable
define dso_local void @_ZN8tutorial11AddressBook9MergeFromERKN6google8protobuf7MessageE(%"class.tutorial::AddressBook"* noundef nonnull align 8 dereferenceable(48) %0, %"class.google::protobuf::Message"* noundef nonnull align 8 dereferenceable(16) %1) unnamed_addr #4 align 2 personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*) {
  %3 = alloca %"class.google::protobuf::internal::LogMessage", align 8
  %4 = alloca %"class.google::protobuf::internal::LogFinisher", align 1
  %5 = getelementptr inbounds %"class.tutorial::AddressBook", %"class.tutorial::AddressBook"* %0, i64 0, i32 0
  %6 = icmp eq %"class.google::protobuf::Message"* %5, %1
  %7 = getelementptr inbounds %"class.google::protobuf::internal::LogFinisher", %"class.google::protobuf::internal::LogFinisher"* %4, i64 0, i32 0
  call void @llvm.lifetime.start.p0i8(i64 1, i8* nonnull %7) #24
  br i1 %6, label %8, label %12

8:                                                ; preds = %2
  %9 = bitcast %"class.google::protobuf::internal::LogMessage"* %3 to i8*
  call void @llvm.lifetime.start.p0i8(i64 56, i8* nonnull %9) #24
  call void @_ZN6google8protobuf8internal10LogMessageC1ENS0_8LogLevelEPKci(%"class.google::protobuf::internal::LogMessage"* noundef nonnull align 8 dereferenceable(56) %3, i32 noundef 3, i8* noundef getelementptr inbounds ([24 x i8], [24 x i8]* @.str.5, i64 0, i64 0), i32 noundef 928)
  %10 = invoke noundef nonnull align 8 dereferenceable(56) %"class.google::protobuf::internal::LogMessage"* @_ZN6google8protobuf8internal10LogMessagelsEPKc(%"class.google::protobuf::internal::LogMessage"* noundef nonnull align 8 dereferenceable(56) %3, i8* noundef getelementptr inbounds ([34 x i8], [34 x i8]* @.str.6, i64 0, i64 0))
          to label %11 unwind label %19

11:                                               ; preds = %8
  invoke void @_ZN6google8protobuf8internal11LogFinisheraSERNS1_10LogMessageE(%"class.google::protobuf::internal::LogFinisher"* noundef nonnull align 1 dereferenceable(1) %4, %"class.google::protobuf::internal::LogMessage"* noundef nonnull align 8 dereferenceable(56) %10)
          to label %13 unwind label %21

12:                                               ; preds = %2
  call void @llvm.lifetime.end.p0i8(i64 1, i8* nonnull %7) #24
  br label %14

13:                                               ; preds = %11
  call void @llvm.lifetime.end.p0i8(i64 1, i8* nonnull %7) #24
  call void @_ZN6google8protobuf8internal10LogMessageD1Ev(%"class.google::protobuf::internal::LogMessage"* noundef nonnull align 8 dereferenceable(56) %3) #24
  call void @llvm.lifetime.end.p0i8(i64 56, i8* nonnull %9) #24
  br label %14

14:                                               ; preds = %12, %13
  %15 = bitcast %"class.google::protobuf::Message"* %1 to i8*
  %16 = call i8* @__dynamic_cast(i8* nonnull %15, i8* bitcast (i8** @_ZTIN6google8protobuf7MessageE to i8*), i8* bitcast ({ i8*, i8*, i8* }* @_ZTIN8tutorial11AddressBookE to i8*), i64 0) #24
  %17 = icmp eq i8* %16, null
  br i1 %17, label %18, label %25

18:                                               ; preds = %14
  call void @_ZN6google8protobuf8internal13ReflectionOps5MergeERKNS0_7MessageEPS3_(%"class.google::protobuf::Message"* noundef nonnull align 8 dereferenceable(16) %1, %"class.google::protobuf::Message"* noundef nonnull %5)
  br label %27

19:                                               ; preds = %8
  %20 = landingpad { i8*, i32 }
          cleanup
  br label %23

21:                                               ; preds = %11
  %22 = landingpad { i8*, i32 }
          cleanup
  call void @llvm.lifetime.end.p0i8(i64 1, i8* nonnull %7) #24
  br label %23

23:                                               ; preds = %19, %21
  %24 = phi { i8*, i32 } [ %22, %21 ], [ %20, %19 ]
  call void @_ZN6google8protobuf8internal10LogMessageD1Ev(%"class.google::protobuf::internal::LogMessage"* noundef nonnull align 8 dereferenceable(56) %3) #24
  call void @llvm.lifetime.end.p0i8(i64 56, i8* nonnull %9) #24
  resume { i8*, i32 } %24

25:                                               ; preds = %14
  %26 = bitcast i8* %16 to %"class.tutorial::AddressBook"*
  call void @_ZN8tutorial11AddressBook9MergeFromERKS0_(%"class.tutorial::AddressBook"* noundef nonnull align 8 dereferenceable(48) %0, %"class.tutorial::AddressBook"* noundef nonnull align 8 dereferenceable(48) %26)
  br label %27

27:                                               ; preds = %25, %18
  ret void
}

; Function Attrs: mustprogress uwtable
define dso_local void @_ZN8tutorial11AddressBook9MergeFromERKS0_(%"class.tutorial::AddressBook"* noundef nonnull align 8 dereferenceable(48) %0, %"class.tutorial::AddressBook"* noundef nonnull align 8 dereferenceable(48) %1) local_unnamed_addr #4 align 2 personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*) {
  %3 = alloca %"class.google::protobuf::internal::LogMessage", align 8
  %4 = alloca %"class.google::protobuf::internal::LogFinisher", align 1
  %5 = icmp eq %"class.tutorial::AddressBook"* %1, %0
  %6 = getelementptr inbounds %"class.google::protobuf::internal::LogFinisher", %"class.google::protobuf::internal::LogFinisher"* %4, i64 0, i32 0
  call void @llvm.lifetime.start.p0i8(i64 1, i8* nonnull %6) #24
  br i1 %5, label %7, label %11

7:                                                ; preds = %2
  %8 = bitcast %"class.google::protobuf::internal::LogMessage"* %3 to i8*
  call void @llvm.lifetime.start.p0i8(i64 56, i8* nonnull %8) #24
  call void @_ZN6google8protobuf8internal10LogMessageC1ENS0_8LogLevelEPKci(%"class.google::protobuf::internal::LogMessage"* noundef nonnull align 8 dereferenceable(56) %3, i32 noundef 3, i8* noundef getelementptr inbounds ([24 x i8], [24 x i8]* @.str.5, i64 0, i64 0), i32 noundef 943)
  %9 = invoke noundef nonnull align 8 dereferenceable(56) %"class.google::protobuf::internal::LogMessage"* @_ZN6google8protobuf8internal10LogMessagelsEPKc(%"class.google::protobuf::internal::LogMessage"* noundef nonnull align 8 dereferenceable(56) %3, i8* noundef getelementptr inbounds ([34 x i8], [34 x i8]* @.str.6, i64 0, i64 0))
          to label %10 unwind label %40

10:                                               ; preds = %7
  invoke void @_ZN6google8protobuf8internal11LogFinisheraSERNS1_10LogMessageE(%"class.google::protobuf::internal::LogFinisher"* noundef nonnull align 1 dereferenceable(1) %4, %"class.google::protobuf::internal::LogMessage"* noundef nonnull align 8 dereferenceable(56) %9)
          to label %12 unwind label %42

11:                                               ; preds = %2
  call void @llvm.lifetime.end.p0i8(i64 1, i8* nonnull %6) #24
  br label %13

12:                                               ; preds = %10
  call void @llvm.lifetime.end.p0i8(i64 1, i8* nonnull %6) #24
  call void @_ZN6google8protobuf8internal10LogMessageD1Ev(%"class.google::protobuf::internal::LogMessage"* noundef nonnull align 8 dereferenceable(56) %3) #24
  call void @llvm.lifetime.end.p0i8(i64 56, i8* nonnull %8) #24
  br label %13

13:                                               ; preds = %11, %12
  %14 = getelementptr inbounds %"class.tutorial::AddressBook", %"class.tutorial::AddressBook"* %0, i64 0, i32 0, i32 0, i32 1
  %15 = getelementptr inbounds %"class.tutorial::AddressBook", %"class.tutorial::AddressBook"* %1, i64 0, i32 0, i32 0, i32 1, i32 0
  %16 = load i8*, i8** %15, align 8, !tbaa !5
  %17 = ptrtoint i8* %16 to i64
  %18 = and i64 %17, 1
  %19 = icmp eq i64 %18, 0
  br i1 %19, label %37, label %20

20:                                               ; preds = %13
  %21 = and i64 %17, -2
  %22 = inttoptr i64 %21 to %"struct.google::protobuf::internal::InternalMetadata::Container"*
  %23 = getelementptr inbounds %"struct.google::protobuf::internal::InternalMetadata::Container", %"struct.google::protobuf::internal::InternalMetadata::Container"* %22, i64 0, i32 1
  %24 = getelementptr inbounds %"class.google::protobuf::internal::InternalMetadata", %"class.google::protobuf::internal::InternalMetadata"* %14, i64 0, i32 0
  %25 = load i8*, i8** %24, align 8, !tbaa !5
  %26 = ptrtoint i8* %25 to i64
  %27 = and i64 %26, 1
  %28 = icmp eq i64 %27, 0
  br i1 %28, label %33, label %29, !prof !41

29:                                               ; preds = %20
  %30 = and i64 %26, -2
  %31 = inttoptr i64 %30 to %"struct.google::protobuf::internal::InternalMetadata::Container"*
  %32 = getelementptr inbounds %"struct.google::protobuf::internal::InternalMetadata::Container", %"struct.google::protobuf::internal::InternalMetadata::Container"* %31, i64 0, i32 1
  br label %35

33:                                               ; preds = %20
  %34 = call noundef %"class.google::protobuf::UnknownFieldSet"* @_ZN6google8protobuf8internal16InternalMetadata27mutable_unknown_fields_slowINS0_15UnknownFieldSetEEEPT_v(%"class.google::protobuf::internal::InternalMetadata"* noundef nonnull align 8 dereferenceable(8) %14)
  br label %35

35:                                               ; preds = %33, %29
  %36 = phi %"class.google::protobuf::UnknownFieldSet"* [ %32, %29 ], [ %34, %33 ]
  call void @_ZN6google8protobuf15UnknownFieldSet9MergeFromERKS1_(%"class.google::protobuf::UnknownFieldSet"* noundef nonnull align 8 dereferenceable(24) %36, %"class.google::protobuf::UnknownFieldSet"* noundef nonnull align 8 dereferenceable(24) %23)
  br label %37

37:                                               ; preds = %13, %35
  %38 = getelementptr inbounds %"class.tutorial::AddressBook", %"class.tutorial::AddressBook"* %0, i64 0, i32 1, i32 0
  %39 = getelementptr inbounds %"class.tutorial::AddressBook", %"class.tutorial::AddressBook"* %1, i64 0, i32 1, i32 0
  call void @_ZN6google8protobuf8internal20RepeatedPtrFieldBase9MergeFromINS0_16RepeatedPtrFieldIN8tutorial6PersonEE11TypeHandlerEEEvRKS2_(%"class.google::protobuf::internal::RepeatedPtrFieldBase"* noundef nonnull align 8 dereferenceable(24) %38, %"class.google::protobuf::internal::RepeatedPtrFieldBase"* noundef nonnull align 8 dereferenceable(24) %39)
  ret void

40:                                               ; preds = %7
  %41 = landingpad { i8*, i32 }
          cleanup
  br label %44

42:                                               ; preds = %10
  %43 = landingpad { i8*, i32 }
          cleanup
  call void @llvm.lifetime.end.p0i8(i64 1, i8* nonnull %6) #24
  br label %44

44:                                               ; preds = %40, %42
  %45 = phi { i8*, i32 } [ %43, %42 ], [ %41, %40 ]
  call void @_ZN6google8protobuf8internal10LogMessageD1Ev(%"class.google::protobuf::internal::LogMessage"* noundef nonnull align 8 dereferenceable(56) %3) #24
  call void @llvm.lifetime.end.p0i8(i64 56, i8* nonnull %8) #24
  resume { i8*, i32 } %45
}

; Function Attrs: uwtable
define dso_local void @_ZN8tutorial11AddressBook8CopyFromERKN6google8protobuf7MessageE(%"class.tutorial::AddressBook"* noundef nonnull align 8 dereferenceable(48) %0, %"class.google::protobuf::Message"* noundef nonnull align 8 dereferenceable(16) %1) unnamed_addr #3 align 2 {
  %3 = getelementptr inbounds %"class.tutorial::AddressBook", %"class.tutorial::AddressBook"* %0, i64 0, i32 0
  %4 = icmp eq %"class.google::protobuf::Message"* %3, %1
  br i1 %4, label %23, label %5

5:                                                ; preds = %2
  %6 = getelementptr inbounds %"class.tutorial::AddressBook", %"class.tutorial::AddressBook"* %0, i64 0, i32 1, i32 0
  tail call void @_ZN6google8protobuf8internal20RepeatedPtrFieldBase5ClearINS0_16RepeatedPtrFieldIN8tutorial6PersonEE11TypeHandlerEEEvv(%"class.google::protobuf::internal::RepeatedPtrFieldBase"* noundef nonnull align 8 dereferenceable(24) %6)
  %7 = getelementptr inbounds %"class.tutorial::AddressBook", %"class.tutorial::AddressBook"* %0, i64 0, i32 0, i32 0, i32 1, i32 0
  %8 = load i8*, i8** %7, align 8, !tbaa !5
  %9 = ptrtoint i8* %8 to i64
  %10 = and i64 %9, 1
  %11 = icmp eq i64 %10, 0
  br i1 %11, label %22, label %12

12:                                               ; preds = %5
  %13 = and i64 %9, -2
  %14 = inttoptr i64 %13 to %"struct.google::protobuf::internal::InternalMetadata::Container"*
  %15 = getelementptr inbounds %"struct.google::protobuf::internal::InternalMetadata::Container", %"struct.google::protobuf::internal::InternalMetadata::Container"* %14, i64 0, i32 1
  %16 = getelementptr inbounds %"class.google::protobuf::UnknownFieldSet", %"class.google::protobuf::UnknownFieldSet"* %15, i64 0, i32 0, i32 0, i32 0, i32 0, i32 0
  %17 = load %"class.google::protobuf::UnknownField"*, %"class.google::protobuf::UnknownField"** %16, align 8, !tbaa !27
  %18 = getelementptr inbounds %"struct.google::protobuf::internal::InternalMetadata::Container", %"struct.google::protobuf::internal::InternalMetadata::Container"* %14, i64 0, i32 1, i32 0, i32 0, i32 0, i32 0, i32 1
  %19 = load %"class.google::protobuf::UnknownField"*, %"class.google::protobuf::UnknownField"** %18, align 8, !tbaa !27
  %20 = icmp eq %"class.google::protobuf::UnknownField"* %17, %19
  br i1 %20, label %22, label %21

21:                                               ; preds = %12
  tail call void @_ZN6google8protobuf15UnknownFieldSet13ClearFallbackEv(%"class.google::protobuf::UnknownFieldSet"* noundef nonnull align 8 dereferenceable(24) %15)
  br label %22

22:                                               ; preds = %5, %12, %21
  tail call void @_ZN8tutorial11AddressBook9MergeFromERKN6google8protobuf7MessageE(%"class.tutorial::AddressBook"* noundef nonnull align 8 dereferenceable(48) %0, %"class.google::protobuf::Message"* noundef nonnull align 8 dereferenceable(16) %1)
  br label %23

23:                                               ; preds = %2, %22
  ret void
}

; Function Attrs: uwtable
define dso_local void @_ZN8tutorial11AddressBook8CopyFromERKS0_(%"class.tutorial::AddressBook"* noundef nonnull align 8 dereferenceable(48) %0, %"class.tutorial::AddressBook"* noundef nonnull align 8 dereferenceable(48) %1) local_unnamed_addr #3 align 2 {
  %3 = icmp eq %"class.tutorial::AddressBook"* %1, %0
  br i1 %3, label %22, label %4

4:                                                ; preds = %2
  %5 = getelementptr inbounds %"class.tutorial::AddressBook", %"class.tutorial::AddressBook"* %0, i64 0, i32 1, i32 0
  tail call void @_ZN6google8protobuf8internal20RepeatedPtrFieldBase5ClearINS0_16RepeatedPtrFieldIN8tutorial6PersonEE11TypeHandlerEEEvv(%"class.google::protobuf::internal::RepeatedPtrFieldBase"* noundef nonnull align 8 dereferenceable(24) %5)
  %6 = getelementptr inbounds %"class.tutorial::AddressBook", %"class.tutorial::AddressBook"* %0, i64 0, i32 0, i32 0, i32 1, i32 0
  %7 = load i8*, i8** %6, align 8, !tbaa !5
  %8 = ptrtoint i8* %7 to i64
  %9 = and i64 %8, 1
  %10 = icmp eq i64 %9, 0
  br i1 %10, label %21, label %11

11:                                               ; preds = %4
  %12 = and i64 %8, -2
  %13 = inttoptr i64 %12 to %"struct.google::protobuf::internal::InternalMetadata::Container"*
  %14 = getelementptr inbounds %"struct.google::protobuf::internal::InternalMetadata::Container", %"struct.google::protobuf::internal::InternalMetadata::Container"* %13, i64 0, i32 1
  %15 = getelementptr inbounds %"class.google::protobuf::UnknownFieldSet", %"class.google::protobuf::UnknownFieldSet"* %14, i64 0, i32 0, i32 0, i32 0, i32 0, i32 0
  %16 = load %"class.google::protobuf::UnknownField"*, %"class.google::protobuf::UnknownField"** %15, align 8, !tbaa !27
  %17 = getelementptr inbounds %"struct.google::protobuf::internal::InternalMetadata::Container", %"struct.google::protobuf::internal::InternalMetadata::Container"* %13, i64 0, i32 1, i32 0, i32 0, i32 0, i32 0, i32 1
  %18 = load %"class.google::protobuf::UnknownField"*, %"class.google::protobuf::UnknownField"** %17, align 8, !tbaa !27
  %19 = icmp eq %"class.google::protobuf::UnknownField"* %16, %18
  br i1 %19, label %21, label %20

20:                                               ; preds = %11
  tail call void @_ZN6google8protobuf15UnknownFieldSet13ClearFallbackEv(%"class.google::protobuf::UnknownFieldSet"* noundef nonnull align 8 dereferenceable(24) %14)
  br label %21

21:                                               ; preds = %4, %11, %20
  tail call void @_ZN8tutorial11AddressBook9MergeFromERKS0_(%"class.tutorial::AddressBook"* noundef nonnull align 8 dereferenceable(48) %0, %"class.tutorial::AddressBook"* noundef nonnull align 8 dereferenceable(48) %1)
  br label %22

22:                                               ; preds = %2, %21
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind readnone uwtable willreturn
define dso_local noundef zeroext i1 @_ZNK8tutorial11AddressBook13IsInitializedEv(%"class.tutorial::AddressBook"* nocapture nonnull readnone align 8 %0) unnamed_addr #5 align 2 {
  ret i1 true
}

; Function Attrs: uwtable
define dso_local void @_ZN8tutorial11AddressBook12InternalSwapEPS0_(%"class.tutorial::AddressBook"* noundef nonnull align 8 dereferenceable(48) %0, %"class.tutorial::AddressBook"* noundef %1) local_unnamed_addr #3 align 2 personality i32 (...)* @__gxx_personality_v0 {
  %3 = getelementptr inbounds %"class.tutorial::AddressBook", %"class.tutorial::AddressBook"* %0, i64 0, i32 0, i32 0, i32 1
  %4 = getelementptr inbounds %"class.tutorial::AddressBook", %"class.tutorial::AddressBook"* %1, i64 0, i32 0, i32 0, i32 1
  %5 = getelementptr inbounds %"class.google::protobuf::internal::InternalMetadata", %"class.google::protobuf::internal::InternalMetadata"* %3, i64 0, i32 0
  %6 = load i8*, i8** %5, align 8, !tbaa !5
  %7 = ptrtoint i8* %6 to i64
  %8 = and i64 %7, 1
  %9 = icmp eq i64 %8, 0
  %10 = getelementptr inbounds %"class.google::protobuf::internal::InternalMetadata", %"class.google::protobuf::internal::InternalMetadata"* %4, i64 0, i32 0
  %11 = load i8*, i8** %10, align 8, !tbaa !5
  %12 = ptrtoint i8* %11 to i64
  %13 = and i64 %12, 1
  %14 = icmp eq i64 %13, 0
  %15 = select i1 %9, i1 %14, i1 false
  br i1 %15, label %48, label %16

16:                                               ; preds = %2
  br i1 %14, label %21, label %17, !prof !41

17:                                               ; preds = %16
  %18 = and i64 %12, -2
  %19 = inttoptr i64 %18 to %"struct.google::protobuf::internal::InternalMetadata::Container"*
  %20 = getelementptr inbounds %"struct.google::protobuf::internal::InternalMetadata::Container", %"struct.google::protobuf::internal::InternalMetadata::Container"* %19, i64 0, i32 1
  br label %25

21:                                               ; preds = %16
  %22 = tail call noundef %"class.google::protobuf::UnknownFieldSet"* @_ZN6google8protobuf8internal16InternalMetadata27mutable_unknown_fields_slowINS0_15UnknownFieldSetEEEPT_v(%"class.google::protobuf::internal::InternalMetadata"* noundef nonnull align 8 dereferenceable(8) %4)
  %23 = load i8*, i8** %5, align 8, !tbaa !5
  %24 = ptrtoint i8* %23 to i64
  br label %25

25:                                               ; preds = %21, %17
  %26 = phi i64 [ %7, %17 ], [ %24, %21 ]
  %27 = phi %"class.google::protobuf::UnknownFieldSet"* [ %20, %17 ], [ %22, %21 ]
  %28 = and i64 %26, 1
  %29 = icmp eq i64 %28, 0
  br i1 %29, label %34, label %30, !prof !41

30:                                               ; preds = %25
  %31 = and i64 %26, -2
  %32 = inttoptr i64 %31 to %"struct.google::protobuf::internal::InternalMetadata::Container"*
  %33 = getelementptr inbounds %"struct.google::protobuf::internal::InternalMetadata::Container", %"struct.google::protobuf::internal::InternalMetadata::Container"* %32, i64 0, i32 1
  br label %36

34:                                               ; preds = %25
  %35 = tail call noundef %"class.google::protobuf::UnknownFieldSet"* @_ZN6google8protobuf8internal16InternalMetadata27mutable_unknown_fields_slowINS0_15UnknownFieldSetEEEPT_v(%"class.google::protobuf::internal::InternalMetadata"* noundef nonnull align 8 dereferenceable(8) %3)
  br label %36

36:                                               ; preds = %34, %30
  %37 = phi %"class.google::protobuf::UnknownFieldSet"* [ %33, %30 ], [ %35, %34 ]
  %38 = bitcast %"class.google::protobuf::UnknownFieldSet"* %37 to <2 x %"class.google::protobuf::UnknownField"*>*
  %39 = load <2 x %"class.google::protobuf::UnknownField"*>, <2 x %"class.google::protobuf::UnknownField"*>* %38, align 8, !tbaa !27
  %40 = getelementptr inbounds %"class.google::protobuf::UnknownFieldSet", %"class.google::protobuf::UnknownFieldSet"* %37, i64 0, i32 0, i32 0, i32 0, i32 0, i32 2
  %41 = load %"class.google::protobuf::UnknownField"*, %"class.google::protobuf::UnknownField"** %40, align 8, !tbaa !50
  %42 = bitcast %"class.google::protobuf::UnknownFieldSet"* %27 to <2 x %"class.google::protobuf::UnknownField"*>*
  %43 = load <2 x %"class.google::protobuf::UnknownField"*>, <2 x %"class.google::protobuf::UnknownField"*>* %42, align 8, !tbaa !27
  %44 = bitcast %"class.google::protobuf::UnknownFieldSet"* %37 to <2 x %"class.google::protobuf::UnknownField"*>*
  store <2 x %"class.google::protobuf::UnknownField"*> %43, <2 x %"class.google::protobuf::UnknownField"*>* %44, align 8, !tbaa !27
  %45 = getelementptr inbounds %"class.google::protobuf::UnknownFieldSet", %"class.google::protobuf::UnknownFieldSet"* %27, i64 0, i32 0, i32 0, i32 0, i32 0, i32 2
  %46 = load %"class.google::protobuf::UnknownField"*, %"class.google::protobuf::UnknownField"** %45, align 8, !tbaa !50
  store %"class.google::protobuf::UnknownField"* %46, %"class.google::protobuf::UnknownField"** %40, align 8, !tbaa !50
  %47 = bitcast %"class.google::protobuf::UnknownFieldSet"* %27 to <2 x %"class.google::protobuf::UnknownField"*>*
  store <2 x %"class.google::protobuf::UnknownField"*> %39, <2 x %"class.google::protobuf::UnknownField"*>* %47, align 8, !tbaa !27
  store %"class.google::protobuf::UnknownField"* %41, %"class.google::protobuf::UnknownField"** %45, align 8, !tbaa !50
  br label %48

48:                                               ; preds = %2, %36
  %49 = getelementptr inbounds %"class.tutorial::AddressBook", %"class.tutorial::AddressBook"* %0, i64 0, i32 1, i32 0
  %50 = getelementptr inbounds %"class.tutorial::AddressBook", %"class.tutorial::AddressBook"* %1, i64 0, i32 1, i32 0
  tail call void @_ZN6google8protobuf8internal20RepeatedPtrFieldBase12InternalSwapEPS2_(%"class.google::protobuf::internal::RepeatedPtrFieldBase"* noundef nonnull align 8 dereferenceable(24) %49, %"class.google::protobuf::internal::RepeatedPtrFieldBase"* noundef nonnull %50)
  ret void
}

; Function Attrs: mustprogress uwtable
define dso_local { %"class.google::protobuf::Descriptor"*, %"class.google::protobuf::Reflection"* } @_ZNK8tutorial11AddressBook11GetMetadataEv(%"class.tutorial::AddressBook"* nocapture nonnull readnone align 8 %0) unnamed_addr #4 align 2 {
  tail call void @_ZN6google8protobuf8internal17AssignDescriptorsEPKNS1_15DescriptorTableEb(%"struct.google::protobuf::internal::DescriptorTable"* noundef nonnull @descriptor_table_addressbook_2eproto, i1 noundef zeroext false)
  %2 = load %"struct.google::protobuf::Metadata"*, %"struct.google::protobuf::Metadata"** getelementptr inbounds (%"struct.google::protobuf::internal::DescriptorTable", %"struct.google::protobuf::internal::DescriptorTable"* @descriptor_table_addressbook_2eproto, i64 0, i32 13), align 8, !tbaa !52
  %3 = getelementptr inbounds %"struct.google::protobuf::Metadata", %"struct.google::protobuf::Metadata"* %2, i64 2, i32 0
  %4 = load %"class.google::protobuf::Descriptor"*, %"class.google::protobuf::Descriptor"** %3, align 8, !tbaa.struct !54
  %5 = getelementptr inbounds %"struct.google::protobuf::Metadata", %"struct.google::protobuf::Metadata"* %2, i64 2, i32 1
  %6 = load %"class.google::protobuf::Reflection"*, %"class.google::protobuf::Reflection"** %5, align 8, !tbaa.struct !55
  %7 = insertvalue { %"class.google::protobuf::Descriptor"*, %"class.google::protobuf::Reflection"* } poison, %"class.google::protobuf::Descriptor"* %4, 0
  %8 = insertvalue { %"class.google::protobuf::Descriptor"*, %"class.google::protobuf::Reflection"* } %7, %"class.google::protobuf::Reflection"* %6, 1
  ret { %"class.google::protobuf::Descriptor"*, %"class.google::protobuf::Reflection"* } %8
}

; Function Attrs: noinline uwtable
define dso_local noundef %"class.tutorial::Person_PhoneNumber"* @_ZN6google8protobuf5Arena18CreateMaybeMessageIN8tutorial18Person_PhoneNumberEJEEEPT_PS1_DpOT0_(%"class.google::protobuf::Arena"* noundef %0) local_unnamed_addr #16 align 2 personality i32 (...)* @__gxx_personality_v0 {
  %2 = icmp eq %"class.google::protobuf::Arena"* %0, null
  br i1 %2, label %3, label %18

3:                                                ; preds = %1
  %4 = tail call noalias noundef nonnull dereferenceable(32) i8* @_Znwm(i64 noundef 32) #27
  %5 = bitcast i8* %4 to %"class.tutorial::Person_PhoneNumber"*
  %6 = getelementptr inbounds %"class.tutorial::Person_PhoneNumber", %"class.tutorial::Person_PhoneNumber"* %5, i64 0, i32 0, i32 0, i32 1
  %7 = bitcast %"class.google::protobuf::internal::InternalMetadata"* %6 to %"class.google::protobuf::Arena"**
  store %"class.google::protobuf::Arena"* null, %"class.google::protobuf::Arena"** %7, align 8, !tbaa !5
  %8 = getelementptr inbounds %"class.tutorial::Person_PhoneNumber", %"class.tutorial::Person_PhoneNumber"* %5, i64 0, i32 0, i32 0, i32 0
  store i32 (...)** bitcast (i8** getelementptr inbounds ({ [22 x i8*] }, { [22 x i8*] }* @_ZTVN8tutorial18Person_PhoneNumberE, i64 0, inrange i32 0, i64 2) to i32 (...)**), i32 (...)*** %8, align 8, !tbaa !10
  %9 = getelementptr inbounds %"class.tutorial::Person_PhoneNumber", %"class.tutorial::Person_PhoneNumber"* %5, i64 0, i32 3, i32 0, i32 0, i32 0
  store i32 0, i32* %9, align 4, !tbaa !13
  %10 = load atomic i32, i32* getelementptr inbounds ({ { { i32 }, i32, i32, void ()* }, [0 x i8*] }, { { { i32 }, i32, i32, void ()* }, [0 x i8*] }* @scc_info_Person_PhoneNumber_addressbook_2eproto, i64 0, i32 0, i32 0, i32 0) acquire, align 8
  %11 = icmp eq i32 %10, 0
  br i1 %11, label %13, label %12, !prof !12

12:                                               ; preds = %3
  invoke void @_ZN6google8protobuf8internal11InitSCCImplEPNS1_11SCCInfoBaseE(%"struct.google::protobuf::internal::SCCInfoBase"* noundef nonnull bitcast ({ { { i32 }, i32, i32, void ()* }, [0 x i8*] }* @scc_info_Person_PhoneNumber_addressbook_2eproto to %"struct.google::protobuf::internal::SCCInfoBase"*))
          to label %13 unwind label %16

13:                                               ; preds = %12, %3
  %14 = getelementptr inbounds %"class.tutorial::Person_PhoneNumber", %"class.tutorial::Person_PhoneNumber"* %5, i64 0, i32 1, i32 0
  store %"class.std::__cxx11::basic_string"* bitcast (%"class.google::protobuf::internal::ExplicitlyConstructed.20"* @_ZN6google8protobuf8internal26fixed_address_empty_stringB5cxx11E to %"class.std::__cxx11::basic_string"*), %"class.std::__cxx11::basic_string"** %14, align 8, !tbaa !18
  %15 = getelementptr inbounds %"class.tutorial::Person_PhoneNumber", %"class.tutorial::Person_PhoneNumber"* %5, i64 0, i32 2
  br label %39

16:                                               ; preds = %12
  %17 = landingpad { i8*, i32 }
          cleanup
  tail call void @_ZdlPv(i8* noundef nonnull %4) #26
  resume { i8*, i32 } %17

18:                                               ; preds = %1
  %19 = getelementptr inbounds %"class.google::protobuf::Arena", %"class.google::protobuf::Arena"* %0, i64 0, i32 4
  %20 = load i8*, i8** %19, align 8, !tbaa !74
  %21 = icmp eq i8* %20, null
  br i1 %21, label %23, label %22, !prof !12

22:                                               ; preds = %18
  tail call void @_ZNK6google8protobuf5Arena17OnArenaAllocationEPKSt9type_infom(%"class.google::protobuf::Arena"* noundef nonnull align 8 dereferenceable(120) %0, %"class.std::type_info"* noundef bitcast ({ i8*, i8*, i8* }* @_ZTIN8tutorial18Person_PhoneNumberE to %"class.std::type_info"*), i64 noundef 32)
  br label %23

23:                                               ; preds = %22, %18
  %24 = tail call noundef i8* @_ZN6google8protobuf5Arena21AllocateAlignedNoHookEm(%"class.google::protobuf::Arena"* noundef nonnull align 8 dereferenceable(120) %0, i64 noundef 32)
  %25 = getelementptr inbounds i8, i8* %24, i64 8
  %26 = bitcast i8* %25 to %"class.google::protobuf::Arena"**
  store %"class.google::protobuf::Arena"* %0, %"class.google::protobuf::Arena"** %26, align 8, !tbaa !5
  %27 = bitcast i8* %24 to i32 (...)***
  store i32 (...)** bitcast (i8** getelementptr inbounds ({ [22 x i8*] }, { [22 x i8*] }* @_ZTVN8tutorial18Person_PhoneNumberE, i64 0, inrange i32 0, i64 2) to i32 (...)**), i32 (...)*** %27, align 8, !tbaa !10
  %28 = getelementptr inbounds i8, i8* %24, i64 28
  %29 = bitcast i8* %28 to i32*
  store i32 0, i32* %29, align 4, !tbaa !13
  %30 = load atomic i32, i32* getelementptr inbounds ({ { { i32 }, i32, i32, void ()* }, [0 x i8*] }, { { { i32 }, i32, i32, void ()* }, [0 x i8*] }* @scc_info_Person_PhoneNumber_addressbook_2eproto, i64 0, i32 0, i32 0, i32 0) acquire, align 8
  %31 = icmp eq i32 %30, 0
  br i1 %31, label %33, label %32, !prof !12

32:                                               ; preds = %23
  tail call void @_ZN6google8protobuf8internal11InitSCCImplEPNS1_11SCCInfoBaseE(%"struct.google::protobuf::internal::SCCInfoBase"* noundef nonnull bitcast ({ { { i32 }, i32, i32, void ()* }, [0 x i8*] }* @scc_info_Person_PhoneNumber_addressbook_2eproto to %"struct.google::protobuf::internal::SCCInfoBase"*))
  br label %33

33:                                               ; preds = %32, %23
  %34 = bitcast i8* %24 to %"class.tutorial::Person_PhoneNumber"*
  %35 = getelementptr inbounds i8, i8* %24, i64 16
  %36 = bitcast i8* %35 to %"class.std::__cxx11::basic_string"**
  store %"class.std::__cxx11::basic_string"* bitcast (%"class.google::protobuf::internal::ExplicitlyConstructed.20"* @_ZN6google8protobuf8internal26fixed_address_empty_stringB5cxx11E to %"class.std::__cxx11::basic_string"*), %"class.std::__cxx11::basic_string"** %36, align 8, !tbaa !18
  %37 = getelementptr inbounds i8, i8* %24, i64 24
  %38 = bitcast i8* %37 to i32*
  br label %39

39:                                               ; preds = %13, %33
  %40 = phi i32* [ %15, %13 ], [ %38, %33 ]
  %41 = phi %"class.tutorial::Person_PhoneNumber"* [ %5, %13 ], [ %34, %33 ]
  store i32 0, i32* %40, align 8, !tbaa !25
  ret %"class.tutorial::Person_PhoneNumber"* %41
}

; Function Attrs: noinline uwtable
define dso_local noundef %"class.tutorial::Person"* @_ZN6google8protobuf5Arena18CreateMaybeMessageIN8tutorial6PersonEJEEEPT_PS1_DpOT0_(%"class.google::protobuf::Arena"* noundef %0) local_unnamed_addr #16 align 2 personality i32 (...)* @__gxx_personality_v0 {
  %2 = alloca %"class.google::protobuf::Arena"*, align 8
  %3 = icmp eq %"class.google::protobuf::Arena"* %0, null
  br i1 %3, label %4, label %38

4:                                                ; preds = %1
  %5 = tail call noalias noundef nonnull dereferenceable(72) i8* @_Znwm(i64 noundef 72) #27
  %6 = bitcast i8* %5 to %"class.tutorial::Person"*
  %7 = getelementptr inbounds %"class.tutorial::Person", %"class.tutorial::Person"* %6, i64 0, i32 0, i32 0, i32 1
  %8 = bitcast %"class.google::protobuf::internal::InternalMetadata"* %7 to %"class.google::protobuf::Arena"**
  store %"class.google::protobuf::Arena"* null, %"class.google::protobuf::Arena"** %8, align 8, !tbaa !5
  %9 = getelementptr inbounds %"class.tutorial::Person", %"class.tutorial::Person"* %6, i64 0, i32 0, i32 0, i32 0
  store i32 (...)** bitcast (i8** getelementptr inbounds ({ [22 x i8*] }, { [22 x i8*] }* @_ZTVN8tutorial6PersonE, i64 0, inrange i32 0, i64 2) to i32 (...)**), i32 (...)*** %9, align 8, !tbaa !10
  %10 = getelementptr inbounds %"class.tutorial::Person", %"class.tutorial::Person"* %6, i64 0, i32 1
  %11 = getelementptr inbounds %"class.google::protobuf::RepeatedPtrField", %"class.google::protobuf::RepeatedPtrField"* %10, i64 0, i32 0, i32 0
  %12 = getelementptr inbounds %"class.tutorial::Person", %"class.tutorial::Person"* %6, i64 0, i32 6, i32 0, i32 0, i32 0
  store i32 0, i32* %12, align 4, !tbaa !13
  %13 = bitcast %"class.google::protobuf::RepeatedPtrField"* %10 to i8*
  tail call void @llvm.memset.p0i8.i64(i8* noundef nonnull align 8 dereferenceable(24) %13, i8 0, i64 24, i1 false)
  %14 = load atomic i32, i32* getelementptr inbounds ({ { { i32 }, i32, i32, void ()* }, [2 x i8*] }, { { { i32 }, i32, i32, void ()* }, [2 x i8*] }* @scc_info_Person_addressbook_2eproto, i64 0, i32 0, i32 0, i32 0) acquire, align 8
  %15 = icmp eq i32 %14, 0
  br i1 %15, label %32, label %16, !prof !12

16:                                               ; preds = %4
  invoke void @_ZN6google8protobuf8internal11InitSCCImplEPNS1_11SCCInfoBaseE(%"struct.google::protobuf::internal::SCCInfoBase"* noundef nonnull bitcast ({ { { i32 }, i32, i32, void ()* }, [2 x i8*] }* @scc_info_Person_addressbook_2eproto to %"struct.google::protobuf::internal::SCCInfoBase"*))
          to label %32 unwind label %17

17:                                               ; preds = %16
  %18 = landingpad { i8*, i32 }
          cleanup
  %19 = getelementptr inbounds %"class.google::protobuf::RepeatedPtrField", %"class.google::protobuf::RepeatedPtrField"* %10, i64 0, i32 0
  invoke void @_ZN6google8protobuf8internal20RepeatedPtrFieldBase7DestroyINS0_16RepeatedPtrFieldIN8tutorial18Person_PhoneNumberEE11TypeHandlerEEEvv(%"class.google::protobuf::internal::RepeatedPtrFieldBase"* noundef nonnull align 8 dereferenceable(24) %19)
          to label %20 unwind label %29

20:                                               ; preds = %17
  %21 = load %"class.google::protobuf::Arena"*, %"class.google::protobuf::Arena"** %11, align 8, !tbaa !16
  %22 = icmp eq %"class.google::protobuf::Arena"* %21, null
  br i1 %22, label %37, label %23

23:                                               ; preds = %20
  %24 = getelementptr inbounds %"class.google::protobuf::Arena", %"class.google::protobuf::Arena"* %21, i64 0, i32 0
  %25 = invoke noundef i64 @_ZNK6google8protobuf8internal9ArenaImpl14SpaceAllocatedEv(%"class.google::protobuf::internal::ArenaImpl"* noundef nonnull align 8 dereferenceable(88) %24)
          to label %37 unwind label %26

26:                                               ; preds = %23
  %27 = landingpad { i8*, i32 }
          catch i8* null
  %28 = extractvalue { i8*, i32 } %27, 0
  tail call void @__clang_call_terminate(i8* %28) #25
  unreachable

29:                                               ; preds = %17
  %30 = landingpad { i8*, i32 }
          catch i8* null
  %31 = extractvalue { i8*, i32 } %30, 0
  tail call void @_ZN6google8protobuf8internal20RepeatedPtrFieldBaseD2Ev(%"class.google::protobuf::internal::RepeatedPtrFieldBase"* noundef nonnull align 8 dereferenceable(24) %19) #24
  tail call void @__clang_call_terminate(i8* %31) #25
  unreachable

32:                                               ; preds = %16, %4
  %33 = getelementptr inbounds %"class.tutorial::Person", %"class.tutorial::Person"* %6, i64 0, i32 2, i32 0
  store %"class.std::__cxx11::basic_string"* bitcast (%"class.google::protobuf::internal::ExplicitlyConstructed.20"* @_ZN6google8protobuf8internal26fixed_address_empty_stringB5cxx11E to %"class.std::__cxx11::basic_string"*), %"class.std::__cxx11::basic_string"** %33, align 8, !tbaa !18
  %34 = getelementptr inbounds %"class.tutorial::Person", %"class.tutorial::Person"* %6, i64 0, i32 3, i32 0
  store %"class.std::__cxx11::basic_string"* bitcast (%"class.google::protobuf::internal::ExplicitlyConstructed.20"* @_ZN6google8protobuf8internal26fixed_address_empty_stringB5cxx11E to %"class.std::__cxx11::basic_string"*), %"class.std::__cxx11::basic_string"** %34, align 8, !tbaa !18
  %35 = getelementptr inbounds %"class.tutorial::Person", %"class.tutorial::Person"* %6, i64 0, i32 4
  %36 = bitcast %"class.google::protobuf::Timestamp"** %35 to i8*
  tail call void @llvm.memset.p0i8.i64(i8* noundef nonnull align 8 dereferenceable(12) %36, i8 0, i64 12, i1 false)
  br label %47

37:                                               ; preds = %23, %20
  tail call void @_ZdlPv(i8* noundef nonnull %5) #26
  resume { i8*, i32 } %18

38:                                               ; preds = %1
  %39 = getelementptr inbounds %"class.google::protobuf::Arena", %"class.google::protobuf::Arena"* %0, i64 0, i32 4
  %40 = load i8*, i8** %39, align 8, !tbaa !74
  %41 = icmp eq i8* %40, null
  br i1 %41, label %43, label %42, !prof !12

42:                                               ; preds = %38
  tail call void @_ZNK6google8protobuf5Arena17OnArenaAllocationEPKSt9type_infom(%"class.google::protobuf::Arena"* noundef nonnull align 8 dereferenceable(120) %0, %"class.std::type_info"* noundef bitcast ({ i8*, i8*, i8* }* @_ZTIN8tutorial6PersonE to %"class.std::type_info"*), i64 noundef 72)
  br label %43

43:                                               ; preds = %42, %38
  %44 = tail call noundef i8* @_ZN6google8protobuf5Arena21AllocateAlignedNoHookEm(%"class.google::protobuf::Arena"* noundef nonnull align 8 dereferenceable(120) %0, i64 noundef 72)
  %45 = bitcast %"class.google::protobuf::Arena"** %2 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %45) #24
  store %"class.google::protobuf::Arena"* %0, %"class.google::protobuf::Arena"** %2, align 8, !tbaa !27
  %46 = call noundef %"class.tutorial::Person"* @_ZN6google8protobuf5Arena14InternalHelperIN8tutorial6PersonEE9ConstructIJPS1_EEEPS4_PvDpOT_(i8* noundef %44, %"class.google::protobuf::Arena"** noundef nonnull align 8 dereferenceable(8) %2)
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %45) #24
  br label %47

47:                                               ; preds = %32, %43
  %48 = phi %"class.tutorial::Person"* [ %46, %43 ], [ %6, %32 ]
  ret %"class.tutorial::Person"* %48
}

; Function Attrs: noinline uwtable
define dso_local noundef %"class.tutorial::AddressBook"* @_ZN6google8protobuf5Arena18CreateMaybeMessageIN8tutorial11AddressBookEJEEEPT_PS1_DpOT0_(%"class.google::protobuf::Arena"* noundef %0) local_unnamed_addr #16 align 2 personality i32 (...)* @__gxx_personality_v0 {
  %2 = icmp eq %"class.google::protobuf::Arena"* %0, null
  br i1 %2, label %3, label %18

3:                                                ; preds = %1
  %4 = tail call noalias noundef nonnull dereferenceable(48) i8* @_Znwm(i64 noundef 48) #27
  %5 = bitcast i8* %4 to %"class.tutorial::AddressBook"*
  %6 = getelementptr inbounds %"class.tutorial::AddressBook", %"class.tutorial::AddressBook"* %5, i64 0, i32 0, i32 0, i32 1
  %7 = bitcast %"class.google::protobuf::internal::InternalMetadata"* %6 to %"class.google::protobuf::Arena"**
  store %"class.google::protobuf::Arena"* null, %"class.google::protobuf::Arena"** %7, align 8, !tbaa !5
  %8 = getelementptr inbounds %"class.tutorial::AddressBook", %"class.tutorial::AddressBook"* %5, i64 0, i32 0, i32 0, i32 0
  store i32 (...)** bitcast (i8** getelementptr inbounds ({ [22 x i8*] }, { [22 x i8*] }* @_ZTVN8tutorial11AddressBookE, i64 0, inrange i32 0, i64 2) to i32 (...)**), i32 (...)*** %8, align 8, !tbaa !10
  %9 = getelementptr inbounds %"class.tutorial::AddressBook", %"class.tutorial::AddressBook"* %5, i64 0, i32 1
  %10 = bitcast %"class.google::protobuf::RepeatedPtrField.18"* %9 to i8*
  tail call void @llvm.memset.p0i8.i64(i8* noundef nonnull align 8 dereferenceable(28) %10, i8 0, i64 28, i1 false)
  %11 = load atomic i32, i32* getelementptr inbounds ({ { { i32 }, i32, i32, void ()* }, [1 x i8*] }, { { { i32 }, i32, i32, void ()* }, [1 x i8*] }* @scc_info_AddressBook_addressbook_2eproto, i64 0, i32 0, i32 0, i32 0) acquire, align 8
  %12 = icmp eq i32 %11, 0
  br i1 %12, label %39, label %13, !prof !12

13:                                               ; preds = %3
  invoke void @_ZN6google8protobuf8internal11InitSCCImplEPNS1_11SCCInfoBaseE(%"struct.google::protobuf::internal::SCCInfoBase"* noundef nonnull bitcast ({ { { i32 }, i32, i32, void ()* }, [1 x i8*] }* @scc_info_AddressBook_addressbook_2eproto to %"struct.google::protobuf::internal::SCCInfoBase"*))
          to label %39 unwind label %14

14:                                               ; preds = %13
  %15 = landingpad { i8*, i32 }
          cleanup
  tail call void @_ZN6google8protobuf16RepeatedPtrFieldIN8tutorial6PersonEED2Ev(%"class.google::protobuf::RepeatedPtrField.18"* noundef nonnull align 8 dereferenceable(24) %9) #24
  tail call void @_ZdlPv(i8* noundef nonnull %4) #26
  br label %16

16:                                               ; preds = %34, %14
  %17 = phi { i8*, i32 } [ %15, %14 ], [ %35, %34 ]
  resume { i8*, i32 } %17

18:                                               ; preds = %1
  %19 = getelementptr inbounds %"class.google::protobuf::Arena", %"class.google::protobuf::Arena"* %0, i64 0, i32 4
  %20 = load i8*, i8** %19, align 8, !tbaa !74
  %21 = icmp eq i8* %20, null
  br i1 %21, label %23, label %22, !prof !12

22:                                               ; preds = %18
  tail call void @_ZNK6google8protobuf5Arena17OnArenaAllocationEPKSt9type_infom(%"class.google::protobuf::Arena"* noundef nonnull align 8 dereferenceable(120) %0, %"class.std::type_info"* noundef bitcast ({ i8*, i8*, i8* }* @_ZTIN8tutorial11AddressBookE to %"class.std::type_info"*), i64 noundef 48)
  br label %23

23:                                               ; preds = %22, %18
  %24 = tail call noundef i8* @_ZN6google8protobuf5Arena21AllocateAlignedNoHookEm(%"class.google::protobuf::Arena"* noundef nonnull align 8 dereferenceable(120) %0, i64 noundef 48)
  %25 = getelementptr inbounds i8, i8* %24, i64 8
  %26 = bitcast i8* %25 to %"class.google::protobuf::Arena"**
  store %"class.google::protobuf::Arena"* %0, %"class.google::protobuf::Arena"** %26, align 8, !tbaa !5
  %27 = bitcast i8* %24 to i32 (...)***
  store i32 (...)** bitcast (i8** getelementptr inbounds ({ [22 x i8*] }, { [22 x i8*] }* @_ZTVN8tutorial11AddressBookE, i64 0, inrange i32 0, i64 2) to i32 (...)**), i32 (...)*** %27, align 8, !tbaa !10
  %28 = getelementptr inbounds i8, i8* %24, i64 16
  %29 = bitcast i8* %28 to %"class.google::protobuf::Arena"**
  store %"class.google::protobuf::Arena"* %0, %"class.google::protobuf::Arena"** %29, align 8, !tbaa !16
  %30 = getelementptr inbounds i8, i8* %24, i64 24
  tail call void @llvm.memset.p0i8.i64(i8* noundef nonnull align 8 dereferenceable(20) %30, i8 0, i64 20, i1 false)
  %31 = load atomic i32, i32* getelementptr inbounds ({ { { i32 }, i32, i32, void ()* }, [1 x i8*] }, { { { i32 }, i32, i32, void ()* }, [1 x i8*] }* @scc_info_AddressBook_addressbook_2eproto, i64 0, i32 0, i32 0, i32 0) acquire, align 8
  %32 = icmp eq i32 %31, 0
  br i1 %32, label %37, label %33, !prof !12

33:                                               ; preds = %23
  invoke void @_ZN6google8protobuf8internal11InitSCCImplEPNS1_11SCCInfoBaseE(%"struct.google::protobuf::internal::SCCInfoBase"* noundef nonnull bitcast ({ { { i32 }, i32, i32, void ()* }, [1 x i8*] }* @scc_info_AddressBook_addressbook_2eproto to %"struct.google::protobuf::internal::SCCInfoBase"*))
          to label %37 unwind label %34

34:                                               ; preds = %33
  %35 = landingpad { i8*, i32 }
          cleanup
  %36 = bitcast i8* %28 to %"class.google::protobuf::RepeatedPtrField.18"*
  tail call void @_ZN6google8protobuf16RepeatedPtrFieldIN8tutorial6PersonEED2Ev(%"class.google::protobuf::RepeatedPtrField.18"* noundef nonnull align 8 dereferenceable(24) %36) #24
  br label %16

37:                                               ; preds = %33, %23
  %38 = bitcast i8* %24 to %"class.tutorial::AddressBook"*
  br label %39

39:                                               ; preds = %3, %13, %37
  %40 = phi %"class.tutorial::AddressBook"* [ %38, %37 ], [ %5, %3 ], [ %5, %13 ]
  ret %"class.tutorial::AddressBook"* %40
}

declare void @_ZNK6google8protobuf7Message11GetTypeNameB5cxx11Ev(%"class.std::__cxx11::basic_string"* sret(%"class.std::__cxx11::basic_string") align 8, %"class.google::protobuf::Message"* noundef nonnull align 8 dereferenceable(16)) unnamed_addr #0

; Function Attrs: inlinehint mustprogress uwtable
define linkonce_odr dso_local noundef %"class.tutorial::Person_PhoneNumber"* @_ZNK8tutorial18Person_PhoneNumber3NewEv(%"class.tutorial::Person_PhoneNumber"* noundef nonnull align 8 dereferenceable(32) %0) unnamed_addr #17 comdat align 2 {
  %2 = tail call noundef %"class.tutorial::Person_PhoneNumber"* @_ZN6google8protobuf5Arena18CreateMaybeMessageIN8tutorial18Person_PhoneNumberEJEEEPT_PS1_DpOT0_(%"class.google::protobuf::Arena"* noundef null)
  ret %"class.tutorial::Person_PhoneNumber"* %2
}

; Function Attrs: mustprogress uwtable
define linkonce_odr dso_local noundef %"class.tutorial::Person_PhoneNumber"* @_ZNK8tutorial18Person_PhoneNumber3NewEPN6google8protobuf5ArenaE(%"class.tutorial::Person_PhoneNumber"* noundef nonnull align 8 dereferenceable(32) %0, %"class.google::protobuf::Arena"* noundef %1) unnamed_addr #4 comdat align 2 {
  %3 = tail call noundef %"class.tutorial::Person_PhoneNumber"* @_ZN6google8protobuf5Arena18CreateMaybeMessageIN8tutorial18Person_PhoneNumberEJEEEPT_PS1_DpOT0_(%"class.google::protobuf::Arena"* noundef %1)
  ret %"class.tutorial::Person_PhoneNumber"* %3
}

declare void @_ZNK6google8protobuf7Message25InitializationErrorStringB5cxx11Ev(%"class.std::__cxx11::basic_string"* sret(%"class.std::__cxx11::basic_string") align 8, %"class.google::protobuf::Message"* noundef nonnull align 8 dereferenceable(16)) unnamed_addr #0

declare void @_ZN6google8protobuf7Message21CheckTypeAndMergeFromERKNS0_11MessageLiteE(%"class.google::protobuf::Message"* noundef nonnull align 8 dereferenceable(16), %"class.google::protobuf::MessageLite"* noundef nonnull align 8 dereferenceable(16)) unnamed_addr #0

; Function Attrs: mustprogress uwtable
define linkonce_odr dso_local noundef i32 @_ZNK8tutorial18Person_PhoneNumber13GetCachedSizeEv(%"class.tutorial::Person_PhoneNumber"* noundef nonnull align 8 dereferenceable(32) %0) unnamed_addr #4 comdat align 2 {
  %2 = getelementptr inbounds %"class.tutorial::Person_PhoneNumber", %"class.tutorial::Person_PhoneNumber"* %0, i64 0, i32 3, i32 0, i32 0, i32 0
  %3 = load atomic i32, i32* %2 monotonic, align 4
  ret i32 %3
}

; Function Attrs: mustprogress nounwind uwtable
define linkonce_odr dso_local noundef i8* @_ZNK6google8protobuf11MessageLite16InternalGetTableEv(%"class.google::protobuf::MessageLite"* noundef nonnull align 8 dereferenceable(16) %0) unnamed_addr #14 comdat align 2 {
  ret i8* null
}

declare void @_ZN6google8protobuf7Message20DiscardUnknownFieldsEv(%"class.google::protobuf::Message"* noundef nonnull align 8 dereferenceable(16)) unnamed_addr #0

declare noundef i64 @_ZNK6google8protobuf7Message13SpaceUsedLongEv(%"class.google::protobuf::Message"* noundef nonnull align 8 dereferenceable(16)) unnamed_addr #0

; Function Attrs: inlinehint mustprogress uwtable
define linkonce_odr dso_local noundef %"class.tutorial::Person"* @_ZNK8tutorial6Person3NewEv(%"class.tutorial::Person"* noundef nonnull align 8 dereferenceable(72) %0) unnamed_addr #17 comdat align 2 {
  %2 = tail call noundef %"class.tutorial::Person"* @_ZN6google8protobuf5Arena18CreateMaybeMessageIN8tutorial6PersonEJEEEPT_PS1_DpOT0_(%"class.google::protobuf::Arena"* noundef null)
  ret %"class.tutorial::Person"* %2
}

; Function Attrs: mustprogress uwtable
define linkonce_odr dso_local noundef %"class.tutorial::Person"* @_ZNK8tutorial6Person3NewEPN6google8protobuf5ArenaE(%"class.tutorial::Person"* noundef nonnull align 8 dereferenceable(72) %0, %"class.google::protobuf::Arena"* noundef %1) unnamed_addr #4 comdat align 2 {
  %3 = tail call noundef %"class.tutorial::Person"* @_ZN6google8protobuf5Arena18CreateMaybeMessageIN8tutorial6PersonEJEEEPT_PS1_DpOT0_(%"class.google::protobuf::Arena"* noundef %1)
  ret %"class.tutorial::Person"* %3
}

; Function Attrs: mustprogress nounwind uwtable
define linkonce_odr dso_local noundef i32 @_ZNK8tutorial6Person13GetCachedSizeEv(%"class.tutorial::Person"* noundef nonnull align 8 dereferenceable(72) %0) unnamed_addr #14 comdat align 2 {
  %2 = getelementptr inbounds %"class.tutorial::Person", %"class.tutorial::Person"* %0, i64 0, i32 6, i32 0, i32 0, i32 0
  %3 = load atomic i32, i32* %2 monotonic, align 4
  ret i32 %3
}

; Function Attrs: inlinehint mustprogress uwtable
define linkonce_odr dso_local noundef %"class.tutorial::AddressBook"* @_ZNK8tutorial11AddressBook3NewEv(%"class.tutorial::AddressBook"* noundef nonnull align 8 dereferenceable(48) %0) unnamed_addr #17 comdat align 2 {
  %2 = tail call noundef %"class.tutorial::AddressBook"* @_ZN6google8protobuf5Arena18CreateMaybeMessageIN8tutorial11AddressBookEJEEEPT_PS1_DpOT0_(%"class.google::protobuf::Arena"* noundef null)
  ret %"class.tutorial::AddressBook"* %2
}

; Function Attrs: mustprogress uwtable
define linkonce_odr dso_local noundef %"class.tutorial::AddressBook"* @_ZNK8tutorial11AddressBook3NewEPN6google8protobuf5ArenaE(%"class.tutorial::AddressBook"* noundef nonnull align 8 dereferenceable(48) %0, %"class.google::protobuf::Arena"* noundef %1) unnamed_addr #4 comdat align 2 {
  %3 = tail call noundef %"class.tutorial::AddressBook"* @_ZN6google8protobuf5Arena18CreateMaybeMessageIN8tutorial11AddressBookEJEEEPT_PS1_DpOT0_(%"class.google::protobuf::Arena"* noundef %1)
  ret %"class.tutorial::AddressBook"* %3
}

; Function Attrs: mustprogress nounwind uwtable
define linkonce_odr dso_local noundef i32 @_ZNK8tutorial11AddressBook13GetCachedSizeEv(%"class.tutorial::AddressBook"* noundef nonnull align 8 dereferenceable(48) %0) unnamed_addr #14 comdat align 2 {
  %2 = getelementptr inbounds %"class.tutorial::AddressBook", %"class.tutorial::AddressBook"* %0, i64 0, i32 2, i32 0, i32 0, i32 0
  %3 = load atomic i32, i32* %2 monotonic, align 8
  ret i32 %3
}

declare void @_ZN6google8protobuf8internal13VerifyVersionEiiPKc(i32 noundef, i32 noundef, i8* noundef) local_unnamed_addr #0

declare void @_ZN6google8protobuf8internal13OnShutdownRunEPFvPKvES3_(void (i8*)* noundef, i8* noundef) local_unnamed_addr #0

declare void @_ZN6google8protobuf8internal14DestroyMessageEPKv(i8* noundef) #0

; Function Attrs: noinline uwtable
define linkonce_odr dso_local void @_ZN6google8protobuf8internal14ArenaStringPtr14CreateInstanceEPNS0_5ArenaEPKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE(%"struct.google::protobuf::internal::ArenaStringPtr"* noundef nonnull align 8 dereferenceable(8) %0, %"class.google::protobuf::Arena"* noundef %1, %"class.std::__cxx11::basic_string"* noundef %2) local_unnamed_addr #16 comdat align 2 personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*) {
  %4 = alloca i64, align 8
  %5 = alloca i64, align 8
  %6 = alloca %"class.google::protobuf::internal::LogMessage", align 8
  %7 = alloca %"class.google::protobuf::internal::LogFinisher", align 1
  %8 = icmp eq %"class.std::__cxx11::basic_string"* %2, null
  %9 = getelementptr inbounds %"class.google::protobuf::internal::LogFinisher", %"class.google::protobuf::internal::LogFinisher"* %7, i64 0, i32 0
  call void @llvm.lifetime.start.p0i8(i64 1, i8* nonnull %9) #24
  br i1 %8, label %10, label %14

10:                                               ; preds = %3
  %11 = bitcast %"class.google::protobuf::internal::LogMessage"* %6 to i8*
  call void @llvm.lifetime.start.p0i8(i64 56, i8* nonnull %11) #24
  call void @_ZN6google8protobuf8internal10LogMessageC1ENS0_8LogLevelEPKci(%"class.google::protobuf::internal::LogMessage"* noundef nonnull align 8 dereferenceable(56) %6, i32 noundef 3, i8* noundef getelementptr inbounds ([43 x i8], [43 x i8]* @.str.9, i64 0, i64 0), i32 noundef 371)
  %12 = invoke noundef nonnull align 8 dereferenceable(56) %"class.google::protobuf::internal::LogMessage"* @_ZN6google8protobuf8internal10LogMessagelsEPKc(%"class.google::protobuf::internal::LogMessage"* noundef nonnull align 8 dereferenceable(56) %6, i8* noundef getelementptr inbounds ([38 x i8], [38 x i8]* @.str.10, i64 0, i64 0))
          to label %13 unwind label %87

13:                                               ; preds = %10
  invoke void @_ZN6google8protobuf8internal11LogFinisheraSERNS1_10LogMessageE(%"class.google::protobuf::internal::LogFinisher"* noundef nonnull align 1 dereferenceable(1) %7, %"class.google::protobuf::internal::LogMessage"* noundef nonnull align 8 dereferenceable(56) %12)
          to label %15 unwind label %89

14:                                               ; preds = %3
  call void @llvm.lifetime.end.p0i8(i64 1, i8* nonnull %9) #24
  br label %16

15:                                               ; preds = %13
  call void @llvm.lifetime.end.p0i8(i64 1, i8* nonnull %9) #24
  call void @_ZN6google8protobuf8internal10LogMessageD1Ev(%"class.google::protobuf::internal::LogMessage"* noundef nonnull align 8 dereferenceable(56) %6) #24
  call void @llvm.lifetime.end.p0i8(i64 56, i8* nonnull %11) #24
  br label %16

16:                                               ; preds = %14, %15
  %17 = icmp eq %"class.google::protobuf::Arena"* %1, null
  br i1 %17, label %18, label %52

18:                                               ; preds = %16
  %19 = call noalias noundef nonnull dereferenceable(32) i8* @_Znwm(i64 noundef 32) #27
  %20 = bitcast i8* %19 to %"class.std::__cxx11::basic_string"*
  %21 = getelementptr inbounds %"class.std::__cxx11::basic_string", %"class.std::__cxx11::basic_string"* %20, i64 0, i32 2
  %22 = bitcast i8* %19 to %union.anon**
  store %union.anon* %21, %union.anon** %22, align 8, !tbaa !81
  %23 = getelementptr inbounds %"class.std::__cxx11::basic_string", %"class.std::__cxx11::basic_string"* %2, i64 0, i32 0, i32 0
  %24 = load i8*, i8** %23, align 8, !tbaa !34
  %25 = getelementptr inbounds %"class.std::__cxx11::basic_string", %"class.std::__cxx11::basic_string"* %2, i64 0, i32 1
  %26 = load i64, i64* %25, align 8, !tbaa !28
  %27 = bitcast i64* %4 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %27) #24
  store i64 %26, i64* %4, align 8, !tbaa !82
  %28 = icmp ugt i64 %26, 15
  br i1 %28, label %31, label %29

29:                                               ; preds = %18
  %30 = bitcast %union.anon* %21 to i8*
  br label %37

31:                                               ; preds = %18
  %32 = invoke noundef i8* @_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE9_M_createERmm(%"class.std::__cxx11::basic_string"* noundef nonnull align 8 dereferenceable(32) %20, i64* noundef nonnull align 8 dereferenceable(8) %4, i64 noundef 0)
          to label %33 unwind label %50

33:                                               ; preds = %31
  %34 = getelementptr inbounds %"class.std::__cxx11::basic_string", %"class.std::__cxx11::basic_string"* %20, i64 0, i32 0, i32 0
  store i8* %32, i8** %34, align 8, !tbaa !34
  %35 = load i64, i64* %4, align 8, !tbaa !82
  %36 = getelementptr inbounds %"class.std::__cxx11::basic_string", %"class.std::__cxx11::basic_string"* %20, i64 0, i32 2, i32 0
  store i64 %35, i64* %36, align 8, !tbaa !37
  br label %37

37:                                               ; preds = %33, %29
  %38 = phi i8* [ %30, %29 ], [ %32, %33 ]
  switch i64 %26, label %41 [
    i64 1, label %39
    i64 0, label %42
  ]

39:                                               ; preds = %37
  %40 = load i8, i8* %24, align 1, !tbaa !37
  store i8 %40, i8* %38, align 1, !tbaa !37
  br label %42

41:                                               ; preds = %37
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 1 %38, i8* align 1 %24, i64 %26, i1 false) #24
  br label %42

42:                                               ; preds = %41, %39, %37
  %43 = getelementptr inbounds %"class.std::__cxx11::basic_string", %"class.std::__cxx11::basic_string"* %20, i64 0, i32 0, i32 0
  %44 = load i64, i64* %4, align 8, !tbaa !82
  %45 = getelementptr inbounds %"class.std::__cxx11::basic_string", %"class.std::__cxx11::basic_string"* %20, i64 0, i32 1
  store i64 %44, i64* %45, align 8, !tbaa !28
  %46 = load i8*, i8** %43, align 8, !tbaa !34
  %47 = getelementptr inbounds i8, i8* %46, i64 %44
  store i8 0, i8* %47, align 1, !tbaa !37
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %27) #24
  br label %84

48:                                               ; preds = %91, %50
  %49 = phi { i8*, i32 } [ %51, %50 ], [ %92, %91 ]
  resume { i8*, i32 } %49

50:                                               ; preds = %31
  %51 = landingpad { i8*, i32 }
          cleanup
  call void @_ZdlPv(i8* noundef nonnull %19) #26
  br label %48

52:                                               ; preds = %16
  %53 = getelementptr inbounds %"class.google::protobuf::Arena", %"class.google::protobuf::Arena"* %1, i64 0, i32 4
  %54 = load i8*, i8** %53, align 8, !tbaa !74
  %55 = icmp eq i8* %54, null
  br i1 %55, label %57, label %56, !prof !12

56:                                               ; preds = %52
  call void @_ZNK6google8protobuf5Arena17OnArenaAllocationEPKSt9type_infom(%"class.google::protobuf::Arena"* noundef nonnull align 8 dereferenceable(120) %1, %"class.std::type_info"* noundef bitcast ({ i8*, i8* }* @_ZTINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE to %"class.std::type_info"*), i64 noundef 32)
  br label %57

57:                                               ; preds = %56, %52
  %58 = getelementptr inbounds %"class.google::protobuf::Arena", %"class.google::protobuf::Arena"* %1, i64 0, i32 0
  %59 = call noundef i8* @_ZN6google8protobuf8internal9ArenaImpl28AllocateAlignedAndAddCleanupEmPFvPvE(%"class.google::protobuf::internal::ArenaImpl"* noundef nonnull align 8 dereferenceable(88) %58, i64 noundef 32, void (i8*)* noundef nonnull @_ZN6google8protobuf8internal21arena_destruct_objectINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEEEvPv)
  %60 = bitcast i8* %59 to %"class.std::__cxx11::basic_string"*
  %61 = getelementptr inbounds i8, i8* %59, i64 16
  %62 = bitcast i8* %59 to i8**
  store i8* %61, i8** %62, align 8, !tbaa !81
  %63 = getelementptr inbounds %"class.std::__cxx11::basic_string", %"class.std::__cxx11::basic_string"* %2, i64 0, i32 0, i32 0
  %64 = load i8*, i8** %63, align 8, !tbaa !34
  %65 = getelementptr inbounds %"class.std::__cxx11::basic_string", %"class.std::__cxx11::basic_string"* %2, i64 0, i32 1
  %66 = load i64, i64* %65, align 8, !tbaa !28
  %67 = bitcast i64* %5 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %67) #24
  store i64 %66, i64* %5, align 8, !tbaa !82
  %68 = icmp ugt i64 %66, 15
  br i1 %68, label %69, label %73

69:                                               ; preds = %57
  %70 = call noundef i8* @_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE9_M_createERmm(%"class.std::__cxx11::basic_string"* noundef nonnull align 8 dereferenceable(32) %60, i64* noundef nonnull align 8 dereferenceable(8) %5, i64 noundef 0)
  store i8* %70, i8** %62, align 8, !tbaa !34
  %71 = load i64, i64* %5, align 8, !tbaa !82
  %72 = bitcast i8* %61 to i64*
  store i64 %71, i64* %72, align 8, !tbaa !37
  br label %73

73:                                               ; preds = %69, %57
  %74 = phi i8* [ %70, %69 ], [ %61, %57 ]
  switch i64 %66, label %77 [
    i64 1, label %75
    i64 0, label %78
  ]

75:                                               ; preds = %73
  %76 = load i8, i8* %64, align 1, !tbaa !37
  store i8 %76, i8* %74, align 1, !tbaa !37
  br label %78

77:                                               ; preds = %73
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 1 %74, i8* align 1 %64, i64 %66, i1 false) #24
  br label %78

78:                                               ; preds = %77, %75, %73
  %79 = load i64, i64* %5, align 8, !tbaa !82
  %80 = getelementptr inbounds i8, i8* %59, i64 8
  %81 = bitcast i8* %80 to i64*
  store i64 %79, i64* %81, align 8, !tbaa !28
  %82 = load i8*, i8** %62, align 8, !tbaa !34
  %83 = getelementptr inbounds i8, i8* %82, i64 %79
  store i8 0, i8* %83, align 1, !tbaa !37
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %67) #24
  br label %84

84:                                               ; preds = %42, %78
  %85 = phi %"class.std::__cxx11::basic_string"* [ %60, %78 ], [ %20, %42 ]
  %86 = getelementptr inbounds %"struct.google::protobuf::internal::ArenaStringPtr", %"struct.google::protobuf::internal::ArenaStringPtr"* %0, i64 0, i32 0
  store %"class.std::__cxx11::basic_string"* %85, %"class.std::__cxx11::basic_string"** %86, align 8, !tbaa !18
  ret void

87:                                               ; preds = %10
  %88 = landingpad { i8*, i32 }
          cleanup
  br label %91

89:                                               ; preds = %13
  %90 = landingpad { i8*, i32 }
          cleanup
  call void @llvm.lifetime.end.p0i8(i64 1, i8* nonnull %9) #24
  br label %91

91:                                               ; preds = %87, %89
  %92 = phi { i8*, i32 } [ %90, %89 ], [ %88, %87 ]
  call void @_ZN6google8protobuf8internal10LogMessageD1Ev(%"class.google::protobuf::internal::LogMessage"* noundef nonnull align 8 dereferenceable(56) %6) #24
  call void @llvm.lifetime.end.p0i8(i64 56, i8* nonnull %11) #24
  br label %48
}

declare noundef i8* @_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE9_M_createERmm(%"class.std::__cxx11::basic_string"* noundef nonnull align 8 dereferenceable(32), i64* noundef nonnull align 8 dereferenceable(8), i64 noundef) local_unnamed_addr #0

; Function Attrs: argmemonly mustprogress nofree nounwind willreturn
declare void @llvm.memcpy.p0i8.p0i8.i64(i8* noalias nocapture writeonly, i8* noalias nocapture readonly, i64, i1 immarg) #18

declare noundef i8* @_ZN6google8protobuf5Arena21AllocateAlignedNoHookEm(%"class.google::protobuf::Arena"* noundef nonnull align 8 dereferenceable(120), i64 noundef) local_unnamed_addr #0

declare noundef i8* @_ZN6google8protobuf8internal9ArenaImpl28AllocateAlignedAndAddCleanupEmPFvPvE(%"class.google::protobuf::internal::ArenaImpl"* noundef nonnull align 8 dereferenceable(88), i64 noundef, void (i8*)* noundef) local_unnamed_addr #0

; Function Attrs: nounwind uwtable
define linkonce_odr dso_local void @_ZN6google8protobuf8internal21arena_destruct_objectINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEEEvPv(i8* noundef %0) #6 comdat personality i32 (...)* @__gxx_personality_v0 {
  %2 = bitcast i8* %0 to i8**
  %3 = load i8*, i8** %2, align 8, !tbaa !34
  %4 = getelementptr inbounds i8, i8* %0, i64 16
  %5 = icmp eq i8* %3, %4
  br i1 %5, label %7, label %6

6:                                                ; preds = %1
  tail call void @_ZdlPv(i8* noundef %3) #24
  br label %7

7:                                                ; preds = %1, %6
  ret void
}

declare void @_ZNK6google8protobuf5Arena17OnArenaAllocationEPKSt9type_infom(%"class.google::protobuf::Arena"* noundef nonnull align 8 dereferenceable(120), %"class.std::type_info"* noundef, i64 noundef) local_unnamed_addr #0

declare void @_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE9_M_assignERKS4_(%"class.std::__cxx11::basic_string"* noundef nonnull align 8 dereferenceable(32), %"class.std::__cxx11::basic_string"* noundef nonnull align 8 dereferenceable(32)) local_unnamed_addr #0

declare void @_ZN6google8protobuf8internal11InitSCCImplEPNS1_11SCCInfoBaseE(%"struct.google::protobuf::internal::SCCInfoBase"* noundef) local_unnamed_addr #0

; Function Attrs: mustprogress uwtable
define linkonce_odr dso_local noundef zeroext i1 @_ZN6google8protobuf8internal18EpsCopyInputStream13DoneWithCheckEPPKci(%"class.google::protobuf::internal::EpsCopyInputStream"* noundef nonnull align 8 dereferenceable(88) %0, i8** noundef %1, i32 noundef %2) local_unnamed_addr #4 comdat align 2 personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*) {
  %4 = alloca %"class.google::protobuf::internal::LogMessage", align 8
  %5 = alloca %"class.google::protobuf::internal::LogFinisher", align 1
  %6 = load i8*, i8** %1, align 8, !tbaa !27
  %7 = icmp eq i8* %6, null
  %8 = getelementptr inbounds %"class.google::protobuf::internal::LogFinisher", %"class.google::protobuf::internal::LogFinisher"* %5, i64 0, i32 0
  call void @llvm.lifetime.start.p0i8(i64 1, i8* nonnull %8) #24
  br i1 %7, label %9, label %13

9:                                                ; preds = %3
  %10 = bitcast %"class.google::protobuf::internal::LogMessage"* %4 to i8*
  call void @llvm.lifetime.start.p0i8(i64 56, i8* nonnull %10) #24
  call void @_ZN6google8protobuf8internal10LogMessageC1ENS0_8LogLevelEPKci(%"class.google::protobuf::internal::LogMessage"* noundef nonnull align 8 dereferenceable(56) %4, i32 noundef 3, i8* noundef getelementptr inbounds ([45 x i8], [45 x i8]* @.str.13, i64 0, i64 0), i32 noundef 209)
  %11 = invoke noundef nonnull align 8 dereferenceable(56) %"class.google::protobuf::internal::LogMessage"* @_ZN6google8protobuf8internal10LogMessagelsEPKc(%"class.google::protobuf::internal::LogMessage"* noundef nonnull align 8 dereferenceable(56) %4, i8* noundef getelementptr inbounds ([21 x i8], [21 x i8]* @.str.14, i64 0, i64 0))
          to label %12 unwind label %21

12:                                               ; preds = %9
  invoke void @_ZN6google8protobuf8internal11LogFinisheraSERNS1_10LogMessageE(%"class.google::protobuf::internal::LogFinisher"* noundef nonnull align 1 dereferenceable(1) %5, %"class.google::protobuf::internal::LogMessage"* noundef nonnull align 8 dereferenceable(56) %11)
          to label %14 unwind label %23

13:                                               ; preds = %3
  call void @llvm.lifetime.end.p0i8(i64 1, i8* nonnull %8) #24
  br label %16

14:                                               ; preds = %12
  call void @llvm.lifetime.end.p0i8(i64 1, i8* nonnull %8) #24
  call void @_ZN6google8protobuf8internal10LogMessageD1Ev(%"class.google::protobuf::internal::LogMessage"* noundef nonnull align 8 dereferenceable(56) %4) #24
  call void @llvm.lifetime.end.p0i8(i64 56, i8* nonnull %10) #24
  %15 = load i8*, i8** %1, align 8, !tbaa !27
  br label %16

16:                                               ; preds = %13, %14
  %17 = phi i8* [ %6, %13 ], [ %15, %14 ]
  %18 = getelementptr inbounds %"class.google::protobuf::internal::EpsCopyInputStream", %"class.google::protobuf::internal::EpsCopyInputStream"* %0, i64 0, i32 0
  %19 = load i8*, i8** %18, align 8, !tbaa !66
  %20 = icmp ult i8* %17, %19
  br i1 %20, label %43, label %27, !prof !12

21:                                               ; preds = %9
  %22 = landingpad { i8*, i32 }
          cleanup
  br label %25

23:                                               ; preds = %12
  %24 = landingpad { i8*, i32 }
          cleanup
  call void @llvm.lifetime.end.p0i8(i64 1, i8* nonnull %8) #24
  br label %25

25:                                               ; preds = %21, %23
  %26 = phi { i8*, i32 } [ %24, %23 ], [ %22, %21 ]
  call void @_ZN6google8protobuf8internal10LogMessageD1Ev(%"class.google::protobuf::internal::LogMessage"* noundef nonnull align 8 dereferenceable(56) %4) #24
  call void @llvm.lifetime.end.p0i8(i64 56, i8* nonnull %10) #24
  resume { i8*, i32 } %26

27:                                               ; preds = %16
  %28 = getelementptr inbounds %"class.google::protobuf::internal::EpsCopyInputStream", %"class.google::protobuf::internal::EpsCopyInputStream"* %0, i64 0, i32 1
  %29 = load i8*, i8** %28, align 8, !tbaa !65
  %30 = ptrtoint i8* %17 to i64
  %31 = ptrtoint i8* %29 to i64
  %32 = sub i64 %30, %31
  %33 = getelementptr inbounds %"class.google::protobuf::internal::EpsCopyInputStream", %"class.google::protobuf::internal::EpsCopyInputStream"* %0, i64 0, i32 4
  %34 = load i32, i32* %33, align 4, !tbaa !64
  %35 = sext i32 %34 to i64
  %36 = icmp eq i64 %32, %35
  br i1 %36, label %43, label %37

37:                                               ; preds = %27
  %38 = call { i8*, i8 } @_ZN6google8protobuf8internal18EpsCopyInputStream12DoneFallbackEPKci(%"class.google::protobuf::internal::EpsCopyInputStream"* noundef nonnull align 8 dereferenceable(88) %0, i8* noundef %17, i32 noundef %2)
  %39 = extractvalue { i8*, i8 } %38, 0
  %40 = extractvalue { i8*, i8 } %38, 1
  store i8* %39, i8** %1, align 8, !tbaa !27
  %41 = and i8 %40, 1
  %42 = icmp ne i8 %41, 0
  br label %43

43:                                               ; preds = %27, %16, %37
  %44 = phi i1 [ %42, %37 ], [ false, %16 ], [ true, %27 ]
  ret i1 %44
}

declare { i8*, i8 } @_ZN6google8protobuf8internal18EpsCopyInputStream12DoneFallbackEPKci(%"class.google::protobuf::internal::EpsCopyInputStream"* noundef nonnull align 8 dereferenceable(88), i8* noundef, i32 noundef) local_unnamed_addr #0

declare { i8*, i32 } @_ZN6google8protobuf8internal15ReadTagFallbackEPKcj(i8* noundef, i32 noundef) local_unnamed_addr #0

declare noundef zeroext i1 @_ZN6google8protobuf8internal10VerifyUTF8ENS0_11StringPieceEPKc(i8*, i64, i8* noundef) local_unnamed_addr #0

declare void @_ZN6google8protobuf11StringPiece18LogFatalSizeTooBigEmPKc(i64 noundef, i8* noundef) local_unnamed_addr #0

declare { i8*, i64 } @_ZN6google8protobuf8internal17VarintParseSlow64EPKcj(i8* noundef, i32 noundef) local_unnamed_addr #0

declare noundef i8* @_ZN6google8protobuf2io19EpsCopyOutputStream30WriteStringMaybeAliasedOutlineEjRKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEPh(%"class.google::protobuf::io::EpsCopyOutputStream"* noundef nonnull align 8 dereferenceable(59), i32 noundef, %"class.std::__cxx11::basic_string"* noundef nonnull align 8 dereferenceable(32), i8* noundef) local_unnamed_addr #0

declare noundef i8* @_ZN6google8protobuf2io19EpsCopyOutputStream19EnsureSpaceFallbackEPh(%"class.google::protobuf::io::EpsCopyOutputStream"* noundef nonnull align 8 dereferenceable(59), i8* noundef) local_unnamed_addr #0

; Function Attrs: mustprogress nofree nosync nounwind readnone speculatable willreturn
declare i32 @llvm.ctlz.i32(i32, i1 immarg) #19

; Function Attrs: nounwind
declare void @_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE4swapERS4_(%"class.std::__cxx11::basic_string"* noundef nonnull align 8 dereferenceable(32), %"class.std::__cxx11::basic_string"* noundef nonnull align 8 dereferenceable(32)) local_unnamed_addr #1

; Function Attrs: argmemonly mustprogress nofree nounwind willreturn writeonly
declare void @llvm.memset.p0i8.i64(i8* nocapture writeonly, i8, i64, i1 immarg) #20

declare void @_ZN6google8protobuf8internal20RepeatedPtrFieldBase7ReserveEi(%"class.google::protobuf::internal::RepeatedPtrFieldBase"* noundef nonnull align 8 dereferenceable(24), i32 noundef) local_unnamed_addr #0

declare noundef %"class.google::protobuf::Timestamp"* @_ZN6google8protobuf5Arena18CreateMaybeMessageINS0_9TimestampEJEEEPT_PS1_DpOT0_(%"class.google::protobuf::Arena"* noundef) local_unnamed_addr #0

; Function Attrs: inlinehint mustprogress uwtable
define linkonce_odr dso_local noundef nonnull align 8 dereferenceable(32) %"class.tutorial::Person_PhoneNumber"* @_ZNK6google8protobuf8internal20RepeatedPtrFieldBase3GetINS0_16RepeatedPtrFieldIN8tutorial18Person_PhoneNumberEE11TypeHandlerEEERKNT_4TypeEi(%"class.google::protobuf::internal::RepeatedPtrFieldBase"* noundef nonnull align 8 dereferenceable(24) %0, i32 noundef %1) local_unnamed_addr #17 comdat align 2 personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*) {
  %3 = alloca %"class.google::protobuf::internal::LogMessage", align 8
  %4 = alloca %"class.google::protobuf::internal::LogFinisher", align 1
  %5 = alloca %"class.google::protobuf::internal::LogMessage", align 8
  %6 = alloca %"class.google::protobuf::internal::LogFinisher", align 1
  %7 = icmp sgt i32 %1, -1
  %8 = getelementptr inbounds %"class.google::protobuf::internal::LogFinisher", %"class.google::protobuf::internal::LogFinisher"* %4, i64 0, i32 0
  call void @llvm.lifetime.start.p0i8(i64 1, i8* nonnull %8) #24
  br i1 %7, label %13, label %9

9:                                                ; preds = %2
  %10 = bitcast %"class.google::protobuf::internal::LogMessage"* %3 to i8*
  call void @llvm.lifetime.start.p0i8(i64 56, i8* nonnull %10) #24
  call void @_ZN6google8protobuf8internal10LogMessageC1ENS0_8LogLevelEPKci(%"class.google::protobuf::internal::LogMessage"* noundef nonnull align 8 dereferenceable(56) %3, i32 noundef 3, i8* noundef getelementptr inbounds ([46 x i8], [46 x i8]* @.str.16, i64 0, i64 0), i32 noundef 1693)
  %11 = invoke noundef nonnull align 8 dereferenceable(56) %"class.google::protobuf::internal::LogMessage"* @_ZN6google8protobuf8internal10LogMessagelsEPKc(%"class.google::protobuf::internal::LogMessage"* noundef nonnull align 8 dereferenceable(56) %3, i8* noundef getelementptr inbounds ([31 x i8], [31 x i8]* @.str.17, i64 0, i64 0))
          to label %12 unwind label %33

12:                                               ; preds = %9
  invoke void @_ZN6google8protobuf8internal11LogFinisheraSERNS1_10LogMessageE(%"class.google::protobuf::internal::LogFinisher"* noundef nonnull align 1 dereferenceable(1) %4, %"class.google::protobuf::internal::LogMessage"* noundef nonnull align 8 dereferenceable(56) %11)
          to label %14 unwind label %35

13:                                               ; preds = %2
  call void @llvm.lifetime.end.p0i8(i64 1, i8* nonnull %8) #24
  br label %15

14:                                               ; preds = %12
  call void @llvm.lifetime.end.p0i8(i64 1, i8* nonnull %8) #24
  call void @_ZN6google8protobuf8internal10LogMessageD1Ev(%"class.google::protobuf::internal::LogMessage"* noundef nonnull align 8 dereferenceable(56) %3) #24
  call void @llvm.lifetime.end.p0i8(i64 56, i8* nonnull %10) #24
  br label %15

15:                                               ; preds = %13, %14
  %16 = getelementptr inbounds %"class.google::protobuf::internal::RepeatedPtrFieldBase", %"class.google::protobuf::internal::RepeatedPtrFieldBase"* %0, i64 0, i32 1
  %17 = load i32, i32* %16, align 8, !tbaa !60
  %18 = icmp sgt i32 %17, %1
  %19 = getelementptr inbounds %"class.google::protobuf::internal::LogFinisher", %"class.google::protobuf::internal::LogFinisher"* %6, i64 0, i32 0
  call void @llvm.lifetime.start.p0i8(i64 1, i8* nonnull %19) #24
  br i1 %18, label %24, label %20

20:                                               ; preds = %15
  %21 = bitcast %"class.google::protobuf::internal::LogMessage"* %5 to i8*
  call void @llvm.lifetime.start.p0i8(i64 56, i8* nonnull %21) #24
  call void @_ZN6google8protobuf8internal10LogMessageC1ENS0_8LogLevelEPKci(%"class.google::protobuf::internal::LogMessage"* noundef nonnull align 8 dereferenceable(56) %5, i32 noundef 3, i8* noundef getelementptr inbounds ([46 x i8], [46 x i8]* @.str.16, i64 0, i64 0), i32 noundef 1694)
  %22 = invoke noundef nonnull align 8 dereferenceable(56) %"class.google::protobuf::internal::LogMessage"* @_ZN6google8protobuf8internal10LogMessagelsEPKc(%"class.google::protobuf::internal::LogMessage"* noundef nonnull align 8 dereferenceable(56) %5, i8* noundef getelementptr inbounds ([42 x i8], [42 x i8]* @.str.18, i64 0, i64 0))
          to label %23 unwind label %39

23:                                               ; preds = %20
  invoke void @_ZN6google8protobuf8internal11LogFinisheraSERNS1_10LogMessageE(%"class.google::protobuf::internal::LogFinisher"* noundef nonnull align 1 dereferenceable(1) %6, %"class.google::protobuf::internal::LogMessage"* noundef nonnull align 8 dereferenceable(56) %22)
          to label %25 unwind label %41

24:                                               ; preds = %15
  call void @llvm.lifetime.end.p0i8(i64 1, i8* nonnull %19) #24
  br label %26

25:                                               ; preds = %23
  call void @llvm.lifetime.end.p0i8(i64 1, i8* nonnull %19) #24
  call void @_ZN6google8protobuf8internal10LogMessageD1Ev(%"class.google::protobuf::internal::LogMessage"* noundef nonnull align 8 dereferenceable(56) %5) #24
  call void @llvm.lifetime.end.p0i8(i64 56, i8* nonnull %21) #24
  br label %26

26:                                               ; preds = %24, %25
  %27 = getelementptr inbounds %"class.google::protobuf::internal::RepeatedPtrFieldBase", %"class.google::protobuf::internal::RepeatedPtrFieldBase"* %0, i64 0, i32 3
  %28 = load %"struct.google::protobuf::internal::RepeatedPtrFieldBase::Rep"*, %"struct.google::protobuf::internal::RepeatedPtrFieldBase::Rep"** %27, align 8, !tbaa !58
  %29 = sext i32 %1 to i64
  %30 = getelementptr inbounds %"struct.google::protobuf::internal::RepeatedPtrFieldBase::Rep", %"struct.google::protobuf::internal::RepeatedPtrFieldBase::Rep"* %28, i64 0, i32 1, i64 %29
  %31 = bitcast i8** %30 to %"class.tutorial::Person_PhoneNumber"**
  %32 = load %"class.tutorial::Person_PhoneNumber"*, %"class.tutorial::Person_PhoneNumber"** %31, align 8, !tbaa !27
  ret %"class.tutorial::Person_PhoneNumber"* %32

33:                                               ; preds = %9
  %34 = landingpad { i8*, i32 }
          cleanup
  br label %37

35:                                               ; preds = %12
  %36 = landingpad { i8*, i32 }
          cleanup
  call void @llvm.lifetime.end.p0i8(i64 1, i8* nonnull %8) #24
  br label %37

37:                                               ; preds = %33, %35
  %38 = phi { i8*, i32 } [ %36, %35 ], [ %34, %33 ]
  call void @_ZN6google8protobuf8internal10LogMessageD1Ev(%"class.google::protobuf::internal::LogMessage"* noundef nonnull align 8 dereferenceable(56) %3) #24
  call void @llvm.lifetime.end.p0i8(i64 56, i8* nonnull %10) #24
  br label %45

39:                                               ; preds = %20
  %40 = landingpad { i8*, i32 }
          cleanup
  br label %43

41:                                               ; preds = %23
  %42 = landingpad { i8*, i32 }
          cleanup
  call void @llvm.lifetime.end.p0i8(i64 1, i8* nonnull %19) #24
  br label %43

43:                                               ; preds = %39, %41
  %44 = phi { i8*, i32 } [ %42, %41 ], [ %40, %39 ]
  call void @_ZN6google8protobuf8internal10LogMessageD1Ev(%"class.google::protobuf::internal::LogMessage"* noundef nonnull align 8 dereferenceable(56) %5) #24
  call void @llvm.lifetime.end.p0i8(i64 56, i8* nonnull %21) #24
  br label %45

45:                                               ; preds = %43, %37
  %46 = phi { i8*, i32 } [ %44, %43 ], [ %38, %37 ]
  resume { i8*, i32 } %46
}

; Function Attrs: inlinehint mustprogress uwtable
define linkonce_odr dso_local noundef nonnull align 8 dereferenceable(72) %"class.tutorial::Person"* @_ZNK6google8protobuf8internal20RepeatedPtrFieldBase3GetINS0_16RepeatedPtrFieldIN8tutorial6PersonEE11TypeHandlerEEERKNT_4TypeEi(%"class.google::protobuf::internal::RepeatedPtrFieldBase"* noundef nonnull align 8 dereferenceable(24) %0, i32 noundef %1) local_unnamed_addr #17 comdat align 2 personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*) {
  %3 = alloca %"class.google::protobuf::internal::LogMessage", align 8
  %4 = alloca %"class.google::protobuf::internal::LogFinisher", align 1
  %5 = alloca %"class.google::protobuf::internal::LogMessage", align 8
  %6 = alloca %"class.google::protobuf::internal::LogFinisher", align 1
  %7 = icmp sgt i32 %1, -1
  %8 = getelementptr inbounds %"class.google::protobuf::internal::LogFinisher", %"class.google::protobuf::internal::LogFinisher"* %4, i64 0, i32 0
  call void @llvm.lifetime.start.p0i8(i64 1, i8* nonnull %8) #24
  br i1 %7, label %13, label %9

9:                                                ; preds = %2
  %10 = bitcast %"class.google::protobuf::internal::LogMessage"* %3 to i8*
  call void @llvm.lifetime.start.p0i8(i64 56, i8* nonnull %10) #24
  call void @_ZN6google8protobuf8internal10LogMessageC1ENS0_8LogLevelEPKci(%"class.google::protobuf::internal::LogMessage"* noundef nonnull align 8 dereferenceable(56) %3, i32 noundef 3, i8* noundef getelementptr inbounds ([46 x i8], [46 x i8]* @.str.16, i64 0, i64 0), i32 noundef 1693)
  %11 = invoke noundef nonnull align 8 dereferenceable(56) %"class.google::protobuf::internal::LogMessage"* @_ZN6google8protobuf8internal10LogMessagelsEPKc(%"class.google::protobuf::internal::LogMessage"* noundef nonnull align 8 dereferenceable(56) %3, i8* noundef getelementptr inbounds ([31 x i8], [31 x i8]* @.str.17, i64 0, i64 0))
          to label %12 unwind label %33

12:                                               ; preds = %9
  invoke void @_ZN6google8protobuf8internal11LogFinisheraSERNS1_10LogMessageE(%"class.google::protobuf::internal::LogFinisher"* noundef nonnull align 1 dereferenceable(1) %4, %"class.google::protobuf::internal::LogMessage"* noundef nonnull align 8 dereferenceable(56) %11)
          to label %14 unwind label %35

13:                                               ; preds = %2
  call void @llvm.lifetime.end.p0i8(i64 1, i8* nonnull %8) #24
  br label %15

14:                                               ; preds = %12
  call void @llvm.lifetime.end.p0i8(i64 1, i8* nonnull %8) #24
  call void @_ZN6google8protobuf8internal10LogMessageD1Ev(%"class.google::protobuf::internal::LogMessage"* noundef nonnull align 8 dereferenceable(56) %3) #24
  call void @llvm.lifetime.end.p0i8(i64 56, i8* nonnull %10) #24
  br label %15

15:                                               ; preds = %13, %14
  %16 = getelementptr inbounds %"class.google::protobuf::internal::RepeatedPtrFieldBase", %"class.google::protobuf::internal::RepeatedPtrFieldBase"* %0, i64 0, i32 1
  %17 = load i32, i32* %16, align 8, !tbaa !60
  %18 = icmp sgt i32 %17, %1
  %19 = getelementptr inbounds %"class.google::protobuf::internal::LogFinisher", %"class.google::protobuf::internal::LogFinisher"* %6, i64 0, i32 0
  call void @llvm.lifetime.start.p0i8(i64 1, i8* nonnull %19) #24
  br i1 %18, label %24, label %20

20:                                               ; preds = %15
  %21 = bitcast %"class.google::protobuf::internal::LogMessage"* %5 to i8*
  call void @llvm.lifetime.start.p0i8(i64 56, i8* nonnull %21) #24
  call void @_ZN6google8protobuf8internal10LogMessageC1ENS0_8LogLevelEPKci(%"class.google::protobuf::internal::LogMessage"* noundef nonnull align 8 dereferenceable(56) %5, i32 noundef 3, i8* noundef getelementptr inbounds ([46 x i8], [46 x i8]* @.str.16, i64 0, i64 0), i32 noundef 1694)
  %22 = invoke noundef nonnull align 8 dereferenceable(56) %"class.google::protobuf::internal::LogMessage"* @_ZN6google8protobuf8internal10LogMessagelsEPKc(%"class.google::protobuf::internal::LogMessage"* noundef nonnull align 8 dereferenceable(56) %5, i8* noundef getelementptr inbounds ([42 x i8], [42 x i8]* @.str.18, i64 0, i64 0))
          to label %23 unwind label %39

23:                                               ; preds = %20
  invoke void @_ZN6google8protobuf8internal11LogFinisheraSERNS1_10LogMessageE(%"class.google::protobuf::internal::LogFinisher"* noundef nonnull align 1 dereferenceable(1) %6, %"class.google::protobuf::internal::LogMessage"* noundef nonnull align 8 dereferenceable(56) %22)
          to label %25 unwind label %41

24:                                               ; preds = %15
  call void @llvm.lifetime.end.p0i8(i64 1, i8* nonnull %19) #24
  br label %26

25:                                               ; preds = %23
  call void @llvm.lifetime.end.p0i8(i64 1, i8* nonnull %19) #24
  call void @_ZN6google8protobuf8internal10LogMessageD1Ev(%"class.google::protobuf::internal::LogMessage"* noundef nonnull align 8 dereferenceable(56) %5) #24
  call void @llvm.lifetime.end.p0i8(i64 56, i8* nonnull %21) #24
  br label %26

26:                                               ; preds = %24, %25
  %27 = getelementptr inbounds %"class.google::protobuf::internal::RepeatedPtrFieldBase", %"class.google::protobuf::internal::RepeatedPtrFieldBase"* %0, i64 0, i32 3
  %28 = load %"struct.google::protobuf::internal::RepeatedPtrFieldBase::Rep"*, %"struct.google::protobuf::internal::RepeatedPtrFieldBase::Rep"** %27, align 8, !tbaa !58
  %29 = sext i32 %1 to i64
  %30 = getelementptr inbounds %"struct.google::protobuf::internal::RepeatedPtrFieldBase::Rep", %"struct.google::protobuf::internal::RepeatedPtrFieldBase::Rep"* %28, i64 0, i32 1, i64 %29
  %31 = bitcast i8** %30 to %"class.tutorial::Person"**
  %32 = load %"class.tutorial::Person"*, %"class.tutorial::Person"** %31, align 8, !tbaa !27
  ret %"class.tutorial::Person"* %32

33:                                               ; preds = %9
  %34 = landingpad { i8*, i32 }
          cleanup
  br label %37

35:                                               ; preds = %12
  %36 = landingpad { i8*, i32 }
          cleanup
  call void @llvm.lifetime.end.p0i8(i64 1, i8* nonnull %8) #24
  br label %37

37:                                               ; preds = %33, %35
  %38 = phi { i8*, i32 } [ %36, %35 ], [ %34, %33 ]
  call void @_ZN6google8protobuf8internal10LogMessageD1Ev(%"class.google::protobuf::internal::LogMessage"* noundef nonnull align 8 dereferenceable(56) %3) #24
  call void @llvm.lifetime.end.p0i8(i64 56, i8* nonnull %10) #24
  br label %45

39:                                               ; preds = %20
  %40 = landingpad { i8*, i32 }
          cleanup
  br label %43

41:                                               ; preds = %23
  %42 = landingpad { i8*, i32 }
          cleanup
  call void @llvm.lifetime.end.p0i8(i64 1, i8* nonnull %19) #24
  br label %43

43:                                               ; preds = %39, %41
  %44 = phi { i8*, i32 } [ %42, %41 ], [ %40, %39 ]
  call void @_ZN6google8protobuf8internal10LogMessageD1Ev(%"class.google::protobuf::internal::LogMessage"* noundef nonnull align 8 dereferenceable(56) %5) #24
  call void @llvm.lifetime.end.p0i8(i64 56, i8* nonnull %21) #24
  br label %45

45:                                               ; preds = %43, %37
  %46 = phi { i8*, i32 } [ %44, %43 ], [ %38, %37 ]
  resume { i8*, i32 } %46
}

; Function Attrs: uwtable
define linkonce_odr dso_local void @_ZN6google8protobuf8internal20RepeatedPtrFieldBase5ClearINS0_16RepeatedPtrFieldIN8tutorial18Person_PhoneNumberEE11TypeHandlerEEEvv(%"class.google::protobuf::internal::RepeatedPtrFieldBase"* noundef nonnull align 8 dereferenceable(24) %0) local_unnamed_addr #3 comdat align 2 personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*) {
  %2 = alloca %"class.google::protobuf::internal::LogMessage", align 8
  %3 = alloca %"class.google::protobuf::internal::LogFinisher", align 1
  %4 = getelementptr inbounds %"class.google::protobuf::internal::RepeatedPtrFieldBase", %"class.google::protobuf::internal::RepeatedPtrFieldBase"* %0, i64 0, i32 1
  %5 = load i32, i32* %4, align 8, !tbaa !60
  %6 = icmp sgt i32 %5, -1
  %7 = getelementptr inbounds %"class.google::protobuf::internal::LogFinisher", %"class.google::protobuf::internal::LogFinisher"* %3, i64 0, i32 0
  call void @llvm.lifetime.start.p0i8(i64 1, i8* nonnull %7) #24
  br i1 %6, label %13, label %8

8:                                                ; preds = %1
  %9 = bitcast %"class.google::protobuf::internal::LogMessage"* %2 to i8*
  call void @llvm.lifetime.start.p0i8(i64 56, i8* nonnull %9) #24
  call void @_ZN6google8protobuf8internal10LogMessageC1ENS0_8LogLevelEPKci(%"class.google::protobuf::internal::LogMessage"* noundef nonnull align 8 dereferenceable(56) %2, i32 noundef 3, i8* noundef getelementptr inbounds ([46 x i8], [46 x i8]* @.str.16, i64 0, i64 0), i32 noundef 1768)
  %10 = invoke noundef nonnull align 8 dereferenceable(56) %"class.google::protobuf::internal::LogMessage"* @_ZN6google8protobuf8internal10LogMessagelsEPKc(%"class.google::protobuf::internal::LogMessage"* noundef nonnull align 8 dereferenceable(56) %2, i8* noundef getelementptr inbounds ([27 x i8], [27 x i8]* @.str.19, i64 0, i64 0))
          to label %11 unwind label %52

11:                                               ; preds = %8
  invoke void @_ZN6google8protobuf8internal11LogFinisheraSERNS1_10LogMessageE(%"class.google::protobuf::internal::LogFinisher"* noundef nonnull align 1 dereferenceable(1) %3, %"class.google::protobuf::internal::LogMessage"* noundef nonnull align 8 dereferenceable(56) %10)
          to label %12 unwind label %54

12:                                               ; preds = %11
  call void @llvm.lifetime.end.p0i8(i64 1, i8* nonnull %7) #24
  call void @_ZN6google8protobuf8internal10LogMessageD1Ev(%"class.google::protobuf::internal::LogMessage"* noundef nonnull align 8 dereferenceable(56) %2) #24
  call void @llvm.lifetime.end.p0i8(i64 56, i8* nonnull %9) #24
  br label %58

13:                                               ; preds = %1
  call void @llvm.lifetime.end.p0i8(i64 1, i8* nonnull %7) #24
  %14 = icmp eq i32 %5, 0
  br i1 %14, label %58, label %15

15:                                               ; preds = %13
  %16 = getelementptr inbounds %"class.google::protobuf::internal::RepeatedPtrFieldBase", %"class.google::protobuf::internal::RepeatedPtrFieldBase"* %0, i64 0, i32 3
  %17 = load %"struct.google::protobuf::internal::RepeatedPtrFieldBase::Rep"*, %"struct.google::protobuf::internal::RepeatedPtrFieldBase::Rep"** %16, align 8, !tbaa !58
  %18 = zext i32 %5 to i64
  br label %19

19:                                               ; preds = %49, %15
  %20 = phi i64 [ %21, %49 ], [ 0, %15 ]
  %21 = add nuw nsw i64 %20, 1
  %22 = getelementptr inbounds %"struct.google::protobuf::internal::RepeatedPtrFieldBase::Rep", %"struct.google::protobuf::internal::RepeatedPtrFieldBase::Rep"* %17, i64 0, i32 1, i64 %20
  %23 = bitcast i8** %22 to %"class.tutorial::Person_PhoneNumber"**
  %24 = load %"class.tutorial::Person_PhoneNumber"*, %"class.tutorial::Person_PhoneNumber"** %23, align 8, !tbaa !27
  %25 = getelementptr inbounds %"class.tutorial::Person_PhoneNumber", %"class.tutorial::Person_PhoneNumber"* %24, i64 0, i32 1, i32 0
  %26 = load %"class.std::__cxx11::basic_string"*, %"class.std::__cxx11::basic_string"** %25, align 8, !tbaa !18
  %27 = icmp eq %"class.std::__cxx11::basic_string"* %26, bitcast (%"class.google::protobuf::internal::ExplicitlyConstructed.20"* @_ZN6google8protobuf8internal26fixed_address_empty_stringB5cxx11E to %"class.std::__cxx11::basic_string"*)
  br i1 %27, label %32, label %28

28:                                               ; preds = %19
  %29 = getelementptr inbounds %"class.std::__cxx11::basic_string", %"class.std::__cxx11::basic_string"* %26, i64 0, i32 1
  store i64 0, i64* %29, align 8, !tbaa !28
  %30 = getelementptr inbounds %"class.std::__cxx11::basic_string", %"class.std::__cxx11::basic_string"* %26, i64 0, i32 0, i32 0
  %31 = load i8*, i8** %30, align 8, !tbaa !34
  store i8 0, i8* %31, align 1, !tbaa !37
  br label %32

32:                                               ; preds = %28, %19
  %33 = getelementptr inbounds %"class.tutorial::Person_PhoneNumber", %"class.tutorial::Person_PhoneNumber"* %24, i64 0, i32 2
  store i32 0, i32* %33, align 8, !tbaa !25
  %34 = getelementptr inbounds %"class.tutorial::Person_PhoneNumber", %"class.tutorial::Person_PhoneNumber"* %24, i64 0, i32 0, i32 0, i32 1, i32 0
  %35 = load i8*, i8** %34, align 8, !tbaa !5
  %36 = ptrtoint i8* %35 to i64
  %37 = and i64 %36, 1
  %38 = icmp eq i64 %37, 0
  br i1 %38, label %49, label %39

39:                                               ; preds = %32
  %40 = and i64 %36, -2
  %41 = inttoptr i64 %40 to %"struct.google::protobuf::internal::InternalMetadata::Container"*
  %42 = getelementptr inbounds %"struct.google::protobuf::internal::InternalMetadata::Container", %"struct.google::protobuf::internal::InternalMetadata::Container"* %41, i64 0, i32 1
  %43 = getelementptr inbounds %"class.google::protobuf::UnknownFieldSet", %"class.google::protobuf::UnknownFieldSet"* %42, i64 0, i32 0, i32 0, i32 0, i32 0, i32 0
  %44 = load %"class.google::protobuf::UnknownField"*, %"class.google::protobuf::UnknownField"** %43, align 8, !tbaa !27
  %45 = getelementptr inbounds %"struct.google::protobuf::internal::InternalMetadata::Container", %"struct.google::protobuf::internal::InternalMetadata::Container"* %41, i64 0, i32 1, i32 0, i32 0, i32 0, i32 0, i32 1
  %46 = load %"class.google::protobuf::UnknownField"*, %"class.google::protobuf::UnknownField"** %45, align 8, !tbaa !27
  %47 = icmp eq %"class.google::protobuf::UnknownField"* %44, %46
  br i1 %47, label %49, label %48

48:                                               ; preds = %39
  tail call void @_ZN6google8protobuf15UnknownFieldSet13ClearFallbackEv(%"class.google::protobuf::UnknownFieldSet"* noundef nonnull align 8 dereferenceable(24) %42)
  br label %49

49:                                               ; preds = %32, %39, %48
  %50 = icmp eq i64 %21, %18
  br i1 %50, label %51, label %19, !llvm.loop !83

51:                                               ; preds = %49
  store i32 0, i32* %4, align 8, !tbaa !60
  br label %58

52:                                               ; preds = %8
  %53 = landingpad { i8*, i32 }
          cleanup
  br label %56

54:                                               ; preds = %11
  %55 = landingpad { i8*, i32 }
          cleanup
  call void @llvm.lifetime.end.p0i8(i64 1, i8* nonnull %7) #24
  br label %56

56:                                               ; preds = %52, %54
  %57 = phi { i8*, i32 } [ %55, %54 ], [ %53, %52 ]
  call void @_ZN6google8protobuf8internal10LogMessageD1Ev(%"class.google::protobuf::internal::LogMessage"* noundef nonnull align 8 dereferenceable(56) %2) #24
  call void @llvm.lifetime.end.p0i8(i64 56, i8* nonnull %9) #24
  resume { i8*, i32 } %57

58:                                               ; preds = %12, %51, %13
  ret void
}

; Function Attrs: mustprogress uwtable
define linkonce_odr dso_local void @_ZN6google8protobuf8internal20RepeatedPtrFieldBase5ClearINS0_16RepeatedPtrFieldIN8tutorial6PersonEE11TypeHandlerEEEvv(%"class.google::protobuf::internal::RepeatedPtrFieldBase"* noundef nonnull align 8 dereferenceable(24) %0) local_unnamed_addr #4 comdat align 2 personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*) {
  %2 = alloca %"class.google::protobuf::internal::LogMessage", align 8
  %3 = alloca %"class.google::protobuf::internal::LogFinisher", align 1
  %4 = getelementptr inbounds %"class.google::protobuf::internal::RepeatedPtrFieldBase", %"class.google::protobuf::internal::RepeatedPtrFieldBase"* %0, i64 0, i32 1
  %5 = load i32, i32* %4, align 8, !tbaa !60
  %6 = icmp sgt i32 %5, -1
  %7 = getelementptr inbounds %"class.google::protobuf::internal::LogFinisher", %"class.google::protobuf::internal::LogFinisher"* %3, i64 0, i32 0
  call void @llvm.lifetime.start.p0i8(i64 1, i8* nonnull %7) #24
  br i1 %6, label %13, label %8

8:                                                ; preds = %1
  %9 = bitcast %"class.google::protobuf::internal::LogMessage"* %2 to i8*
  call void @llvm.lifetime.start.p0i8(i64 56, i8* nonnull %9) #24
  call void @_ZN6google8protobuf8internal10LogMessageC1ENS0_8LogLevelEPKci(%"class.google::protobuf::internal::LogMessage"* noundef nonnull align 8 dereferenceable(56) %2, i32 noundef 3, i8* noundef getelementptr inbounds ([46 x i8], [46 x i8]* @.str.16, i64 0, i64 0), i32 noundef 1768)
  %10 = invoke noundef nonnull align 8 dereferenceable(56) %"class.google::protobuf::internal::LogMessage"* @_ZN6google8protobuf8internal10LogMessagelsEPKc(%"class.google::protobuf::internal::LogMessage"* noundef nonnull align 8 dereferenceable(56) %2, i8* noundef getelementptr inbounds ([27 x i8], [27 x i8]* @.str.19, i64 0, i64 0))
          to label %11 unwind label %27

11:                                               ; preds = %8
  invoke void @_ZN6google8protobuf8internal11LogFinisheraSERNS1_10LogMessageE(%"class.google::protobuf::internal::LogFinisher"* noundef nonnull align 1 dereferenceable(1) %3, %"class.google::protobuf::internal::LogMessage"* noundef nonnull align 8 dereferenceable(56) %10)
          to label %12 unwind label %29

12:                                               ; preds = %11
  call void @llvm.lifetime.end.p0i8(i64 1, i8* nonnull %7) #24
  call void @_ZN6google8protobuf8internal10LogMessageD1Ev(%"class.google::protobuf::internal::LogMessage"* noundef nonnull align 8 dereferenceable(56) %2) #24
  call void @llvm.lifetime.end.p0i8(i64 56, i8* nonnull %9) #24
  br label %33

13:                                               ; preds = %1
  call void @llvm.lifetime.end.p0i8(i64 1, i8* nonnull %7) #24
  %14 = icmp eq i32 %5, 0
  br i1 %14, label %33, label %15

15:                                               ; preds = %13
  %16 = getelementptr inbounds %"class.google::protobuf::internal::RepeatedPtrFieldBase", %"class.google::protobuf::internal::RepeatedPtrFieldBase"* %0, i64 0, i32 3
  %17 = load %"struct.google::protobuf::internal::RepeatedPtrFieldBase::Rep"*, %"struct.google::protobuf::internal::RepeatedPtrFieldBase::Rep"** %16, align 8, !tbaa !58
  %18 = zext i32 %5 to i64
  br label %19

19:                                               ; preds = %19, %15
  %20 = phi i64 [ %21, %19 ], [ 0, %15 ]
  %21 = add nuw nsw i64 %20, 1
  %22 = getelementptr inbounds %"struct.google::protobuf::internal::RepeatedPtrFieldBase::Rep", %"struct.google::protobuf::internal::RepeatedPtrFieldBase::Rep"* %17, i64 0, i32 1, i64 %20
  %23 = bitcast i8** %22 to %"class.tutorial::Person"**
  %24 = load %"class.tutorial::Person"*, %"class.tutorial::Person"** %23, align 8, !tbaa !27
  tail call void @_ZN8tutorial6Person5ClearEv(%"class.tutorial::Person"* noundef nonnull align 8 dereferenceable(72) %24)
  %25 = icmp eq i64 %21, %18
  br i1 %25, label %26, label %19, !llvm.loop !84

26:                                               ; preds = %19
  store i32 0, i32* %4, align 8, !tbaa !60
  br label %33

27:                                               ; preds = %8
  %28 = landingpad { i8*, i32 }
          cleanup
  br label %31

29:                                               ; preds = %11
  %30 = landingpad { i8*, i32 }
          cleanup
  call void @llvm.lifetime.end.p0i8(i64 1, i8* nonnull %7) #24
  br label %31

31:                                               ; preds = %27, %29
  %32 = phi { i8*, i32 } [ %30, %29 ], [ %28, %27 ]
  call void @_ZN6google8protobuf8internal10LogMessageD1Ev(%"class.google::protobuf::internal::LogMessage"* noundef nonnull align 8 dereferenceable(56) %2) #24
  call void @llvm.lifetime.end.p0i8(i64 56, i8* nonnull %9) #24
  resume { i8*, i32 } %32

33:                                               ; preds = %12, %26, %13
  ret void
}

declare void @_ZN6google8protobuf15UnknownFieldSet9MergeFromERKS1_(%"class.google::protobuf::UnknownFieldSet"* noundef nonnull align 8 dereferenceable(24), %"class.google::protobuf::UnknownFieldSet"* noundef nonnull align 8 dereferenceable(24)) local_unnamed_addr #0

; Function Attrs: nounwind uwtable
define linkonce_odr dso_local void @_ZNSt6vectorIN6google8protobuf12UnknownFieldESaIS2_EED2Ev(%"class.std::vector"* noundef nonnull align 8 dereferenceable(24) %0) unnamed_addr #6 comdat align 2 personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*) {
  %2 = getelementptr inbounds %"class.std::vector", %"class.std::vector"* %0, i64 0, i32 0, i32 0, i32 0, i32 0
  %3 = load %"class.google::protobuf::UnknownField"*, %"class.google::protobuf::UnknownField"** %2, align 8, !tbaa !35
  %4 = icmp eq %"class.google::protobuf::UnknownField"* %3, null
  br i1 %4, label %7, label %5

5:                                                ; preds = %1
  %6 = bitcast %"class.google::protobuf::UnknownField"* %3 to i8*
  tail call void @_ZdlPv(i8* noundef nonnull %6) #24
  br label %7

7:                                                ; preds = %1, %5
  ret void
}

declare void @_ZN6google8protobuf15UnknownFieldSet13ClearFallbackEv(%"class.google::protobuf::UnknownFieldSet"* noundef nonnull align 8 dereferenceable(24)) local_unnamed_addr #0

; Function Attrs: noinline uwtable
define linkonce_odr dso_local noundef %"class.google::protobuf::UnknownFieldSet"* @_ZN6google8protobuf8internal16InternalMetadata27mutable_unknown_fields_slowINS0_15UnknownFieldSetEEEPT_v(%"class.google::protobuf::internal::InternalMetadata"* noundef nonnull align 8 dereferenceable(8) %0) local_unnamed_addr #16 comdat align 2 personality i32 (...)* @__gxx_personality_v0 {
  %2 = getelementptr inbounds %"class.google::protobuf::internal::InternalMetadata", %"class.google::protobuf::internal::InternalMetadata"* %0, i64 0, i32 0
  %3 = load i8*, i8** %2, align 8, !tbaa !5
  %4 = ptrtoint i8* %3 to i64
  %5 = and i64 %4, 1
  %6 = icmp eq i64 %5, 0
  %7 = and i64 %4, -2
  br i1 %6, label %12, label %8, !prof !12

8:                                                ; preds = %1
  %9 = inttoptr i64 %7 to %"struct.google::protobuf::internal::InternalMetadata::ContainerBase"*
  %10 = getelementptr inbounds %"struct.google::protobuf::internal::InternalMetadata::ContainerBase", %"struct.google::protobuf::internal::InternalMetadata::ContainerBase"* %9, i64 0, i32 0
  %11 = load %"class.google::protobuf::Arena"*, %"class.google::protobuf::Arena"** %10, align 8, !tbaa !32
  br label %14

12:                                               ; preds = %1
  %13 = inttoptr i64 %7 to %"class.google::protobuf::Arena"*
  br label %14

14:                                               ; preds = %8, %12
  %15 = phi %"class.google::protobuf::Arena"* [ %11, %8 ], [ %13, %12 ]
  %16 = icmp eq %"class.google::protobuf::Arena"* %15, null
  br i1 %16, label %17, label %19

17:                                               ; preds = %14
  %18 = tail call noalias noundef nonnull dereferenceable(32) i8* @_Znwm(i64 noundef 32) #27
  tail call void @llvm.memset.p0i8.i64(i8* noundef nonnull align 16 dereferenceable(32) %18, i8 0, i64 32, i1 false)
  br label %27

19:                                               ; preds = %14
  %20 = getelementptr inbounds %"class.google::protobuf::Arena", %"class.google::protobuf::Arena"* %15, i64 0, i32 4
  %21 = load i8*, i8** %20, align 8, !tbaa !74
  %22 = icmp eq i8* %21, null
  br i1 %22, label %24, label %23, !prof !12

23:                                               ; preds = %19
  tail call void @_ZNK6google8protobuf5Arena17OnArenaAllocationEPKSt9type_infom(%"class.google::protobuf::Arena"* noundef nonnull align 8 dereferenceable(120) %15, %"class.std::type_info"* noundef bitcast ({ i8*, i8*, i8* }* @_ZTIN6google8protobuf8internal16InternalMetadata9ContainerINS0_15UnknownFieldSetEEE to %"class.std::type_info"*), i64 noundef 32)
  br label %24

24:                                               ; preds = %23, %19
  %25 = getelementptr inbounds %"class.google::protobuf::Arena", %"class.google::protobuf::Arena"* %15, i64 0, i32 0
  %26 = tail call noundef i8* @_ZN6google8protobuf8internal9ArenaImpl28AllocateAlignedAndAddCleanupEmPFvPvE(%"class.google::protobuf::internal::ArenaImpl"* noundef nonnull align 8 dereferenceable(88) %25, i64 noundef 32, void (i8*)* noundef nonnull @_ZN6google8protobuf8internal21arena_destruct_objectINS1_16InternalMetadata9ContainerINS0_15UnknownFieldSetEEEEEvPv)
  tail call void @llvm.memset.p0i8.i64(i8* noundef nonnull align 8 dereferenceable(32) %26, i8 0, i64 32, i1 false)
  br label %27

27:                                               ; preds = %17, %24
  %28 = phi i8* [ %26, %24 ], [ %18, %17 ]
  %29 = ptrtoint i8* %28 to i64
  %30 = or i64 %29, 1
  %31 = inttoptr i64 %30 to i8*
  store i8* %31, i8** %2, align 8, !tbaa !5
  %32 = bitcast i8* %28 to %"class.google::protobuf::Arena"**
  store %"class.google::protobuf::Arena"* %15, %"class.google::protobuf::Arena"** %32, align 8, !tbaa !32
  %33 = getelementptr inbounds i8, i8* %28, i64 8
  %34 = bitcast i8* %33 to %"class.google::protobuf::UnknownFieldSet"*
  ret %"class.google::protobuf::UnknownFieldSet"* %34
}

; Function Attrs: nounwind uwtable
define linkonce_odr dso_local void @_ZN6google8protobuf8internal21arena_destruct_objectINS1_16InternalMetadata9ContainerINS0_15UnknownFieldSetEEEEEvPv(i8* noundef %0) #6 comdat personality i32 (...)* @__gxx_personality_v0 {
  %2 = getelementptr inbounds i8, i8* %0, i64 8
  %3 = bitcast i8* %2 to %"class.google::protobuf::UnknownField"**
  %4 = load %"class.google::protobuf::UnknownField"*, %"class.google::protobuf::UnknownField"** %3, align 8, !tbaa !27
  %5 = getelementptr inbounds i8, i8* %0, i64 16
  %6 = bitcast i8* %5 to %"class.google::protobuf::UnknownField"**
  %7 = load %"class.google::protobuf::UnknownField"*, %"class.google::protobuf::UnknownField"** %6, align 8, !tbaa !27
  %8 = icmp eq %"class.google::protobuf::UnknownField"* %4, %7
  br i1 %8, label %13, label %9

9:                                                ; preds = %1
  %10 = bitcast i8* %2 to %"class.google::protobuf::UnknownFieldSet"*
  invoke void @_ZN6google8protobuf15UnknownFieldSet13ClearFallbackEv(%"class.google::protobuf::UnknownFieldSet"* noundef nonnull align 8 dereferenceable(24) %10)
          to label %11 unwind label %18

11:                                               ; preds = %9
  %12 = load %"class.google::protobuf::UnknownField"*, %"class.google::protobuf::UnknownField"** %3, align 8, !tbaa !35
  br label %13

13:                                               ; preds = %11, %1
  %14 = phi %"class.google::protobuf::UnknownField"* [ %12, %11 ], [ %4, %1 ]
  %15 = icmp eq %"class.google::protobuf::UnknownField"* %14, null
  br i1 %15, label %22, label %16

16:                                               ; preds = %13
  %17 = bitcast %"class.google::protobuf::UnknownField"* %14 to i8*
  tail call void @_ZdlPv(i8* noundef nonnull %17) #24
  br label %22

18:                                               ; preds = %9
  %19 = landingpad { i8*, i32 }
          catch i8* null
  %20 = extractvalue { i8*, i32 } %19, 0
  %21 = bitcast i8* %2 to %"class.std::vector"*
  tail call void @_ZNSt6vectorIN6google8protobuf12UnknownFieldESaIS2_EED2Ev(%"class.std::vector"* noundef nonnull align 8 dereferenceable(24) %21) #24
  tail call void @__clang_call_terminate(i8* %20) #25
  unreachable

22:                                               ; preds = %13, %16
  ret void
}

; Function Attrs: nofree nounwind readonly
declare i8* @__dynamic_cast(i8*, i8*, i8*, i64) local_unnamed_addr #21

; Function Attrs: uwtable
define linkonce_odr dso_local void @_ZN6google8protobuf8internal20RepeatedPtrFieldBase7DestroyINS0_16RepeatedPtrFieldIN8tutorial18Person_PhoneNumberEE11TypeHandlerEEEvv(%"class.google::protobuf::internal::RepeatedPtrFieldBase"* noundef nonnull align 8 dereferenceable(24) %0) local_unnamed_addr #3 comdat align 2 personality i32 (...)* @__gxx_personality_v0 {
  %2 = getelementptr inbounds %"class.google::protobuf::internal::RepeatedPtrFieldBase", %"class.google::protobuf::internal::RepeatedPtrFieldBase"* %0, i64 0, i32 3
  %3 = load %"struct.google::protobuf::internal::RepeatedPtrFieldBase::Rep"*, %"struct.google::protobuf::internal::RepeatedPtrFieldBase::Rep"** %2, align 8, !tbaa !58
  %4 = icmp ne %"struct.google::protobuf::internal::RepeatedPtrFieldBase::Rep"* %3, null
  %5 = getelementptr inbounds %"class.google::protobuf::internal::RepeatedPtrFieldBase", %"class.google::protobuf::internal::RepeatedPtrFieldBase"* %0, i64 0, i32 0
  %6 = load %"class.google::protobuf::Arena"*, %"class.google::protobuf::Arena"** %5, align 8
  %7 = icmp eq %"class.google::protobuf::Arena"* %6, null
  %8 = select i1 %4, i1 %7, i1 false
  br i1 %8, label %9, label %38

9:                                                ; preds = %1
  %10 = bitcast %"struct.google::protobuf::internal::RepeatedPtrFieldBase::Rep"* %3 to i8*
  %11 = getelementptr inbounds %"struct.google::protobuf::internal::RepeatedPtrFieldBase::Rep", %"struct.google::protobuf::internal::RepeatedPtrFieldBase::Rep"* %3, i64 0, i32 0
  %12 = load i32, i32* %11, align 8, !tbaa !61
  %13 = icmp sgt i32 %12, 0
  br i1 %13, label %14, label %19

14:                                               ; preds = %9
  %15 = zext i32 %12 to i64
  br label %21

16:                                               ; preds = %35
  %17 = bitcast %"struct.google::protobuf::internal::RepeatedPtrFieldBase::Rep"** %2 to i8**
  %18 = load i8*, i8** %17, align 8, !tbaa !58
  br label %19

19:                                               ; preds = %9, %16
  %20 = phi i8* [ %18, %16 ], [ %10, %9 ]
  tail call void @_ZdlPv(i8* noundef %20) #24
  br label %38

21:                                               ; preds = %14, %35
  %22 = phi i64 [ 0, %14 ], [ %36, %35 ]
  %23 = getelementptr inbounds %"struct.google::protobuf::internal::RepeatedPtrFieldBase::Rep", %"struct.google::protobuf::internal::RepeatedPtrFieldBase::Rep"* %3, i64 0, i32 1, i64 %22
  %24 = load i8*, i8** %23, align 8, !tbaa !27
  %25 = icmp eq i8* %24, null
  br i1 %25, label %35, label %26

26:                                               ; preds = %21
  %27 = bitcast i8* %24 to %"class.tutorial::Person_PhoneNumber"*
  invoke void @_ZN8tutorial18Person_PhoneNumber10SharedDtorEv(%"class.tutorial::Person_PhoneNumber"* noundef nonnull align 8 dereferenceable(32) %27)
          to label %28 unwind label %31

28:                                               ; preds = %26
  %29 = getelementptr inbounds i8, i8* %24, i64 8
  %30 = bitcast i8* %29 to %"class.google::protobuf::internal::InternalMetadata"*
  invoke void @_ZN6google8protobuf8internal16InternalMetadata6DeleteINS0_15UnknownFieldSetEEEvv(%"class.google::protobuf::internal::InternalMetadata"* noundef nonnull align 8 dereferenceable(8) %30)
          to label %34 unwind label %31

31:                                               ; preds = %28, %26
  %32 = landingpad { i8*, i32 }
          catch i8* null
  %33 = extractvalue { i8*, i32 } %32, 0
  tail call void @__clang_call_terminate(i8* %33) #25
  unreachable

34:                                               ; preds = %28
  tail call void @_ZdlPv(i8* noundef nonnull %24) #26
  br label %35

35:                                               ; preds = %21, %34
  %36 = add nuw nsw i64 %22, 1
  %37 = icmp eq i64 %36, %15
  br i1 %37, label %16, label %21, !llvm.loop !85

38:                                               ; preds = %19, %1
  store %"struct.google::protobuf::internal::RepeatedPtrFieldBase::Rep"* null, %"struct.google::protobuf::internal::RepeatedPtrFieldBase::Rep"** %2, align 8, !tbaa !58
  ret void
}

; Function Attrs: nounwind uwtable
define linkonce_odr dso_local void @_ZN6google8protobuf8internal20RepeatedPtrFieldBaseD2Ev(%"class.google::protobuf::internal::RepeatedPtrFieldBase"* noundef nonnull align 8 dereferenceable(24) %0) unnamed_addr #6 comdat align 2 personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*) {
  %2 = getelementptr inbounds %"class.google::protobuf::internal::RepeatedPtrFieldBase", %"class.google::protobuf::internal::RepeatedPtrFieldBase"* %0, i64 0, i32 0
  %3 = load %"class.google::protobuf::Arena"*, %"class.google::protobuf::Arena"** %2, align 8, !tbaa !16
  %4 = icmp eq %"class.google::protobuf::Arena"* %3, null
  br i1 %4, label %8, label %5

5:                                                ; preds = %1
  %6 = getelementptr inbounds %"class.google::protobuf::Arena", %"class.google::protobuf::Arena"* %3, i64 0, i32 0
  %7 = invoke noundef i64 @_ZNK6google8protobuf8internal9ArenaImpl14SpaceAllocatedEv(%"class.google::protobuf::internal::ArenaImpl"* noundef nonnull align 8 dereferenceable(88) %6)
          to label %8 unwind label %9

8:                                                ; preds = %5, %1
  ret void

9:                                                ; preds = %5
  %10 = landingpad { i8*, i32 }
          catch i8* null
  %11 = extractvalue { i8*, i32 } %10, 0
  tail call void @__clang_call_terminate(i8* %11) #25
  unreachable
}

declare noundef i64 @_ZNK6google8protobuf8internal9ArenaImpl14SpaceAllocatedEv(%"class.google::protobuf::internal::ArenaImpl"* noundef nonnull align 8 dereferenceable(88)) local_unnamed_addr #0

; Function Attrs: mustprogress uwtable
define linkonce_odr dso_local noundef i32 @_ZN6google8protobuf8internal18EpsCopyInputStream9PushLimitEPKci(%"class.google::protobuf::internal::EpsCopyInputStream"* noundef nonnull align 8 dereferenceable(88) %0, i8* noundef %1, i32 noundef %2) local_unnamed_addr #4 comdat align 2 personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*) {
  %4 = alloca %"class.google::protobuf::internal::LogMessage", align 8
  %5 = alloca %"class.google::protobuf::internal::LogFinisher", align 1
  %6 = getelementptr inbounds %"class.google::protobuf::internal::LogFinisher", %"class.google::protobuf::internal::LogFinisher"* %5, i64 0, i32 0
  call void @llvm.lifetime.start.p0i8(i64 1, i8* nonnull %6) #24
  %7 = icmp ult i32 %2, 2147483632
  br i1 %7, label %12, label %8

8:                                                ; preds = %3
  %9 = bitcast %"class.google::protobuf::internal::LogMessage"* %4 to i8*
  call void @llvm.lifetime.start.p0i8(i64 56, i8* nonnull %9) #24
  call void @_ZN6google8protobuf8internal10LogMessageC1ENS0_8LogLevelEPKci(%"class.google::protobuf::internal::LogMessage"* noundef nonnull align 8 dereferenceable(56) %4, i32 noundef 3, i8* noundef getelementptr inbounds ([45 x i8], [45 x i8]* @.str.13, i64 0, i64 0), i32 noundef 128)
  %10 = invoke noundef nonnull align 8 dereferenceable(56) %"class.google::protobuf::internal::LogMessage"* @_ZN6google8protobuf8internal10LogMessagelsEPKc(%"class.google::protobuf::internal::LogMessage"* noundef nonnull align 8 dereferenceable(56) %4, i8* noundef getelementptr inbounds ([60 x i8], [60 x i8]* @.str.20, i64 0, i64 0))
          to label %11 unwind label %30

11:                                               ; preds = %8
  invoke void @_ZN6google8protobuf8internal11LogFinisheraSERNS1_10LogMessageE(%"class.google::protobuf::internal::LogFinisher"* noundef nonnull align 1 dereferenceable(1) %5, %"class.google::protobuf::internal::LogMessage"* noundef nonnull align 8 dereferenceable(56) %10)
          to label %13 unwind label %32

12:                                               ; preds = %3
  call void @llvm.lifetime.end.p0i8(i64 1, i8* nonnull %6) #24
  br label %14

13:                                               ; preds = %11
  call void @llvm.lifetime.end.p0i8(i64 1, i8* nonnull %6) #24
  call void @_ZN6google8protobuf8internal10LogMessageD1Ev(%"class.google::protobuf::internal::LogMessage"* noundef nonnull align 8 dereferenceable(56) %4) #24
  call void @llvm.lifetime.end.p0i8(i64 56, i8* nonnull %9) #24
  br label %14

14:                                               ; preds = %12, %13
  %15 = getelementptr inbounds %"class.google::protobuf::internal::EpsCopyInputStream", %"class.google::protobuf::internal::EpsCopyInputStream"* %0, i64 0, i32 1
  %16 = load i8*, i8** %15, align 8, !tbaa !65
  %17 = ptrtoint i8* %1 to i64
  %18 = ptrtoint i8* %16 to i64
  %19 = sub i64 %17, %18
  %20 = trunc i64 %19 to i32
  %21 = add nsw i32 %20, %2
  %22 = icmp slt i32 %21, 0
  %23 = select i1 %22, i32 %21, i32 0
  %24 = sext i32 %23 to i64
  %25 = getelementptr inbounds i8, i8* %16, i64 %24
  %26 = getelementptr inbounds %"class.google::protobuf::internal::EpsCopyInputStream", %"class.google::protobuf::internal::EpsCopyInputStream"* %0, i64 0, i32 0
  store i8* %25, i8** %26, align 8, !tbaa !66
  %27 = getelementptr inbounds %"class.google::protobuf::internal::EpsCopyInputStream", %"class.google::protobuf::internal::EpsCopyInputStream"* %0, i64 0, i32 4
  %28 = load i32, i32* %27, align 4, !tbaa !64
  store i32 %21, i32* %27, align 4, !tbaa !64
  %29 = sub nsw i32 %28, %21
  ret i32 %29

30:                                               ; preds = %8
  %31 = landingpad { i8*, i32 }
          cleanup
  br label %34

32:                                               ; preds = %11
  %33 = landingpad { i8*, i32 }
          cleanup
  call void @llvm.lifetime.end.p0i8(i64 1, i8* nonnull %6) #24
  br label %34

34:                                               ; preds = %30, %32
  %35 = phi { i8*, i32 } [ %33, %32 ], [ %31, %30 ]
  call void @_ZN6google8protobuf8internal10LogMessageD1Ev(%"class.google::protobuf::internal::LogMessage"* noundef nonnull align 8 dereferenceable(56) %4) #24
  call void @llvm.lifetime.end.p0i8(i64 56, i8* nonnull %9) #24
  resume { i8*, i32 } %35
}

declare { i8*, i32 } @_ZN6google8protobuf8internal16ReadSizeFallbackEPKcj(i8* noundef, i32 noundef) local_unnamed_addr #0

declare noundef i8* @_ZN6google8protobuf9Timestamp14_InternalParseEPKcPNS0_8internal12ParseContextE(%"class.google::protobuf::Timestamp"* noundef nonnull align 8 dereferenceable(32), i8* noundef, %"class.google::protobuf::internal::ParseContext"* noundef) unnamed_addr #0

declare noundef i8* @_ZNK6google8protobuf9Timestamp18_InternalSerializeEPhPNS0_2io19EpsCopyOutputStreamE(%"class.google::protobuf::Timestamp"* noundef nonnull align 8 dereferenceable(32), i8* noundef, %"class.google::protobuf::io::EpsCopyOutputStream"* noundef) unnamed_addr #0

declare noundef i64 @_ZNK6google8protobuf9Timestamp12ByteSizeLongEv(%"class.google::protobuf::Timestamp"* noundef nonnull align 8 dereferenceable(32)) unnamed_addr #0

; Function Attrs: inlinehint mustprogress uwtable
define linkonce_odr dso_local void @_ZN6google8protobuf8internal20RepeatedPtrFieldBase9MergeFromINS0_16RepeatedPtrFieldIN8tutorial18Person_PhoneNumberEE11TypeHandlerEEEvRKS2_(%"class.google::protobuf::internal::RepeatedPtrFieldBase"* noundef nonnull align 8 dereferenceable(24) %0, %"class.google::protobuf::internal::RepeatedPtrFieldBase"* noundef nonnull align 8 dereferenceable(24) %1) local_unnamed_addr #17 comdat align 2 personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*) {
  %3 = alloca %"class.google::protobuf::internal::LogMessage", align 8
  %4 = alloca %"class.google::protobuf::internal::LogFinisher", align 1
  %5 = icmp eq %"class.google::protobuf::internal::RepeatedPtrFieldBase"* %1, %0
  %6 = getelementptr inbounds %"class.google::protobuf::internal::LogFinisher", %"class.google::protobuf::internal::LogFinisher"* %4, i64 0, i32 0
  call void @llvm.lifetime.start.p0i8(i64 1, i8* nonnull %6) #24
  br i1 %5, label %7, label %11

7:                                                ; preds = %2
  %8 = bitcast %"class.google::protobuf::internal::LogMessage"* %3 to i8*
  call void @llvm.lifetime.start.p0i8(i64 56, i8* nonnull %8) #24
  call void @_ZN6google8protobuf8internal10LogMessageC1ENS0_8LogLevelEPKci(%"class.google::protobuf::internal::LogMessage"* noundef nonnull align 8 dereferenceable(56) %3, i32 noundef 3, i8* noundef getelementptr inbounds ([46 x i8], [46 x i8]* @.str.16, i64 0, i64 0), i32 noundef 1787)
  %9 = invoke noundef nonnull align 8 dereferenceable(56) %"class.google::protobuf::internal::LogMessage"* @_ZN6google8protobuf8internal10LogMessagelsEPKc(%"class.google::protobuf::internal::LogMessage"* noundef nonnull align 8 dereferenceable(56) %3, i8* noundef getelementptr inbounds ([35 x i8], [35 x i8]* @.str.21, i64 0, i64 0))
          to label %10 unwind label %17

10:                                               ; preds = %7
  invoke void @_ZN6google8protobuf8internal11LogFinisheraSERNS1_10LogMessageE(%"class.google::protobuf::internal::LogFinisher"* noundef nonnull align 1 dereferenceable(1) %4, %"class.google::protobuf::internal::LogMessage"* noundef nonnull align 8 dereferenceable(56) %9)
          to label %12 unwind label %19

11:                                               ; preds = %2
  call void @llvm.lifetime.end.p0i8(i64 1, i8* nonnull %6) #24
  br label %13

12:                                               ; preds = %10
  call void @llvm.lifetime.end.p0i8(i64 1, i8* nonnull %6) #24
  call void @_ZN6google8protobuf8internal10LogMessageD1Ev(%"class.google::protobuf::internal::LogMessage"* noundef nonnull align 8 dereferenceable(56) %3) #24
  call void @llvm.lifetime.end.p0i8(i64 56, i8* nonnull %8) #24
  br label %13

13:                                               ; preds = %11, %12
  %14 = getelementptr inbounds %"class.google::protobuf::internal::RepeatedPtrFieldBase", %"class.google::protobuf::internal::RepeatedPtrFieldBase"* %1, i64 0, i32 1
  %15 = load i32, i32* %14, align 8, !tbaa !60
  %16 = icmp eq i32 %15, 0
  br i1 %16, label %77, label %23

17:                                               ; preds = %7
  %18 = landingpad { i8*, i32 }
          cleanup
  br label %21

19:                                               ; preds = %10
  %20 = landingpad { i8*, i32 }
          cleanup
  call void @llvm.lifetime.end.p0i8(i64 1, i8* nonnull %6) #24
  br label %21

21:                                               ; preds = %17, %19
  %22 = phi { i8*, i32 } [ %20, %19 ], [ %18, %17 ]
  call void @_ZN6google8protobuf8internal10LogMessageD1Ev(%"class.google::protobuf::internal::LogMessage"* noundef nonnull align 8 dereferenceable(56) %3) #24
  call void @llvm.lifetime.end.p0i8(i64 56, i8* nonnull %8) #24
  resume { i8*, i32 } %22

23:                                               ; preds = %13
  %24 = getelementptr inbounds %"class.google::protobuf::internal::RepeatedPtrFieldBase", %"class.google::protobuf::internal::RepeatedPtrFieldBase"* %1, i64 0, i32 3
  %25 = load %"struct.google::protobuf::internal::RepeatedPtrFieldBase::Rep"*, %"struct.google::protobuf::internal::RepeatedPtrFieldBase::Rep"** %24, align 8, !tbaa !58
  %26 = call noundef i8** @_ZN6google8protobuf8internal20RepeatedPtrFieldBase14InternalExtendEi(%"class.google::protobuf::internal::RepeatedPtrFieldBase"* noundef nonnull align 8 dereferenceable(24) %0, i32 noundef %15)
  %27 = getelementptr inbounds %"class.google::protobuf::internal::RepeatedPtrFieldBase", %"class.google::protobuf::internal::RepeatedPtrFieldBase"* %0, i64 0, i32 3
  %28 = load %"struct.google::protobuf::internal::RepeatedPtrFieldBase::Rep"*, %"struct.google::protobuf::internal::RepeatedPtrFieldBase::Rep"** %27, align 8, !tbaa !58
  %29 = getelementptr inbounds %"struct.google::protobuf::internal::RepeatedPtrFieldBase::Rep", %"struct.google::protobuf::internal::RepeatedPtrFieldBase::Rep"* %28, i64 0, i32 0
  %30 = load i32, i32* %29, align 8, !tbaa !61
  %31 = getelementptr inbounds %"class.google::protobuf::internal::RepeatedPtrFieldBase", %"class.google::protobuf::internal::RepeatedPtrFieldBase"* %0, i64 0, i32 1
  %32 = load i32, i32* %31, align 8, !tbaa !60
  %33 = sub i32 %30, %32
  %34 = icmp sgt i32 %33, 0
  %35 = icmp sgt i32 %15, 0
  %36 = and i1 %34, %35
  br i1 %36, label %37, label %40

37:                                               ; preds = %23
  %38 = zext i32 %15 to i64
  %39 = zext i32 %33 to i64
  br label %46

40:                                               ; preds = %46, %23
  %41 = getelementptr inbounds %"class.google::protobuf::internal::RepeatedPtrFieldBase", %"class.google::protobuf::internal::RepeatedPtrFieldBase"* %0, i64 0, i32 0
  %42 = load %"class.google::protobuf::Arena"*, %"class.google::protobuf::Arena"** %41, align 8, !tbaa !16
  %43 = icmp slt i32 %33, %15
  br i1 %43, label %44, label %69

44:                                               ; preds = %40
  %45 = sext i32 %33 to i64
  br label %58

46:                                               ; preds = %37, %46
  %47 = phi i64 [ 0, %37 ], [ %54, %46 ]
  %48 = getelementptr inbounds %"struct.google::protobuf::internal::RepeatedPtrFieldBase::Rep", %"struct.google::protobuf::internal::RepeatedPtrFieldBase::Rep"* %25, i64 0, i32 1, i64 %47
  %49 = bitcast i8** %48 to %"class.tutorial::Person_PhoneNumber"**
  %50 = load %"class.tutorial::Person_PhoneNumber"*, %"class.tutorial::Person_PhoneNumber"** %49, align 8, !tbaa !27
  %51 = getelementptr inbounds i8*, i8** %26, i64 %47
  %52 = bitcast i8** %51 to %"class.tutorial::Person_PhoneNumber"**
  %53 = load %"class.tutorial::Person_PhoneNumber"*, %"class.tutorial::Person_PhoneNumber"** %52, align 8, !tbaa !27
  call void @_ZN6google8protobuf8internal18GenericTypeHandlerIN8tutorial18Person_PhoneNumberEE5MergeERKS4_PS4_(%"class.tutorial::Person_PhoneNumber"* noundef nonnull align 8 dereferenceable(32) %50, %"class.tutorial::Person_PhoneNumber"* noundef %53)
  %54 = add nuw nsw i64 %47, 1
  %55 = icmp ult i64 %54, %39
  %56 = icmp ult i64 %54, %38
  %57 = and i1 %55, %56
  br i1 %57, label %46, label %40, !llvm.loop !86

58:                                               ; preds = %44, %58
  %59 = phi i64 [ %45, %44 ], [ %66, %58 ]
  %60 = getelementptr inbounds %"struct.google::protobuf::internal::RepeatedPtrFieldBase::Rep", %"struct.google::protobuf::internal::RepeatedPtrFieldBase::Rep"* %25, i64 0, i32 1, i64 %59
  %61 = bitcast i8** %60 to %"class.tutorial::Person_PhoneNumber"**
  %62 = load %"class.tutorial::Person_PhoneNumber"*, %"class.tutorial::Person_PhoneNumber"** %61, align 8, !tbaa !27
  %63 = call noundef %"class.tutorial::Person_PhoneNumber"* @_ZN6google8protobuf5Arena18CreateMaybeMessageIN8tutorial18Person_PhoneNumberEJEEEPT_PS1_DpOT0_(%"class.google::protobuf::Arena"* noundef %42)
  call void @_ZN6google8protobuf8internal18GenericTypeHandlerIN8tutorial18Person_PhoneNumberEE5MergeERKS4_PS4_(%"class.tutorial::Person_PhoneNumber"* noundef nonnull align 8 dereferenceable(32) %62, %"class.tutorial::Person_PhoneNumber"* noundef %63)
  %64 = getelementptr inbounds i8*, i8** %26, i64 %59
  %65 = bitcast i8** %64 to %"class.tutorial::Person_PhoneNumber"**
  store %"class.tutorial::Person_PhoneNumber"* %63, %"class.tutorial::Person_PhoneNumber"** %65, align 8, !tbaa !27
  %66 = add nsw i64 %59, 1
  %67 = trunc i64 %66 to i32
  %68 = icmp eq i32 %15, %67
  br i1 %68, label %69, label %58, !llvm.loop !87

69:                                               ; preds = %58, %40
  %70 = load i32, i32* %31, align 8, !tbaa !60
  %71 = add nsw i32 %70, %15
  store i32 %71, i32* %31, align 8, !tbaa !60
  %72 = load %"struct.google::protobuf::internal::RepeatedPtrFieldBase::Rep"*, %"struct.google::protobuf::internal::RepeatedPtrFieldBase::Rep"** %27, align 8, !tbaa !58
  %73 = getelementptr inbounds %"struct.google::protobuf::internal::RepeatedPtrFieldBase::Rep", %"struct.google::protobuf::internal::RepeatedPtrFieldBase::Rep"* %72, i64 0, i32 0
  %74 = load i32, i32* %73, align 8, !tbaa !61
  %75 = icmp slt i32 %74, %71
  br i1 %75, label %76, label %77

76:                                               ; preds = %69
  store i32 %71, i32* %73, align 8, !tbaa !61
  br label %77

77:                                               ; preds = %76, %69, %13
  ret void
}

declare noundef i8** @_ZN6google8protobuf8internal20RepeatedPtrFieldBase14InternalExtendEi(%"class.google::protobuf::internal::RepeatedPtrFieldBase"* noundef nonnull align 8 dereferenceable(24), i32 noundef) local_unnamed_addr #0

; Function Attrs: mustprogress noinline uwtable
define linkonce_odr dso_local void @_ZN6google8protobuf8internal18GenericTypeHandlerIN8tutorial18Person_PhoneNumberEE5MergeERKS4_PS4_(%"class.tutorial::Person_PhoneNumber"* noundef nonnull align 8 dereferenceable(32) %0, %"class.tutorial::Person_PhoneNumber"* noundef %1) local_unnamed_addr #22 comdat align 2 {
  tail call void @_ZN8tutorial18Person_PhoneNumber9MergeFromERKS0_(%"class.tutorial::Person_PhoneNumber"* noundef nonnull align 8 dereferenceable(32) %1, %"class.tutorial::Person_PhoneNumber"* noundef nonnull align 8 dereferenceable(32) %0)
  ret void
}

; Function Attrs: inlinehint mustprogress uwtable
define linkonce_odr dso_local void @_ZN6google8protobuf8internal20RepeatedPtrFieldBase12InternalSwapEPS2_(%"class.google::protobuf::internal::RepeatedPtrFieldBase"* noundef nonnull align 8 dereferenceable(24) %0, %"class.google::protobuf::internal::RepeatedPtrFieldBase"* noundef %1) local_unnamed_addr #17 comdat align 2 personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*) {
  %3 = alloca %"class.google::protobuf::internal::LogMessage", align 8
  %4 = alloca %"class.google::protobuf::internal::LogFinisher", align 1
  %5 = alloca %"class.google::protobuf::internal::LogMessage", align 8
  %6 = alloca %"class.google::protobuf::internal::LogFinisher", align 1
  %7 = icmp eq %"class.google::protobuf::internal::RepeatedPtrFieldBase"* %0, %1
  %8 = getelementptr inbounds %"class.google::protobuf::internal::LogFinisher", %"class.google::protobuf::internal::LogFinisher"* %4, i64 0, i32 0
  call void @llvm.lifetime.start.p0i8(i64 1, i8* nonnull %8) #24
  br i1 %7, label %9, label %13

9:                                                ; preds = %2
  %10 = bitcast %"class.google::protobuf::internal::LogMessage"* %3 to i8*
  call void @llvm.lifetime.start.p0i8(i64 56, i8* nonnull %10) #24
  call void @_ZN6google8protobuf8internal10LogMessageC1ENS0_8LogLevelEPKci(%"class.google::protobuf::internal::LogMessage"* noundef nonnull align 8 dereferenceable(56) %3, i32 noundef 3, i8* noundef getelementptr inbounds ([46 x i8], [46 x i8]* @.str.16, i64 0, i64 0), i32 noundef 2577)
  %11 = invoke noundef nonnull align 8 dereferenceable(56) %"class.google::protobuf::internal::LogMessage"* @_ZN6google8protobuf8internal10LogMessagelsEPKc(%"class.google::protobuf::internal::LogMessage"* noundef nonnull align 8 dereferenceable(56) %3, i8* noundef getelementptr inbounds ([30 x i8], [30 x i8]* @.str.22, i64 0, i64 0))
          to label %12 unwind label %36

12:                                               ; preds = %9
  invoke void @_ZN6google8protobuf8internal11LogFinisheraSERNS1_10LogMessageE(%"class.google::protobuf::internal::LogFinisher"* noundef nonnull align 1 dereferenceable(1) %4, %"class.google::protobuf::internal::LogMessage"* noundef nonnull align 8 dereferenceable(56) %11)
          to label %14 unwind label %38

13:                                               ; preds = %2
  call void @llvm.lifetime.end.p0i8(i64 1, i8* nonnull %8) #24
  br label %15

14:                                               ; preds = %12
  call void @llvm.lifetime.end.p0i8(i64 1, i8* nonnull %8) #24
  call void @_ZN6google8protobuf8internal10LogMessageD1Ev(%"class.google::protobuf::internal::LogMessage"* noundef nonnull align 8 dereferenceable(56) %3) #24
  call void @llvm.lifetime.end.p0i8(i64 56, i8* nonnull %10) #24
  br label %15

15:                                               ; preds = %13, %14
  %16 = getelementptr inbounds %"class.google::protobuf::internal::RepeatedPtrFieldBase", %"class.google::protobuf::internal::RepeatedPtrFieldBase"* %0, i64 0, i32 0
  %17 = load %"class.google::protobuf::Arena"*, %"class.google::protobuf::Arena"** %16, align 8, !tbaa !16
  %18 = getelementptr inbounds %"class.google::protobuf::internal::RepeatedPtrFieldBase", %"class.google::protobuf::internal::RepeatedPtrFieldBase"* %1, i64 0, i32 0
  %19 = load %"class.google::protobuf::Arena"*, %"class.google::protobuf::Arena"** %18, align 8, !tbaa !16
  %20 = icmp eq %"class.google::protobuf::Arena"* %17, %19
  %21 = getelementptr inbounds %"class.google::protobuf::internal::LogFinisher", %"class.google::protobuf::internal::LogFinisher"* %6, i64 0, i32 0
  call void @llvm.lifetime.start.p0i8(i64 1, i8* nonnull %21) #24
  br i1 %20, label %26, label %22

22:                                               ; preds = %15
  %23 = bitcast %"class.google::protobuf::internal::LogMessage"* %5 to i8*
  call void @llvm.lifetime.start.p0i8(i64 56, i8* nonnull %23) #24
  call void @_ZN6google8protobuf8internal10LogMessageC1ENS0_8LogLevelEPKci(%"class.google::protobuf::internal::LogMessage"* noundef nonnull align 8 dereferenceable(56) %5, i32 noundef 3, i8* noundef getelementptr inbounds ([46 x i8], [46 x i8]* @.str.16, i64 0, i64 0), i32 noundef 2578)
  %24 = invoke noundef nonnull align 8 dereferenceable(56) %"class.google::protobuf::internal::LogMessage"* @_ZN6google8protobuf8internal10LogMessagelsEPKc(%"class.google::protobuf::internal::LogMessage"* noundef nonnull align 8 dereferenceable(56) %5, i8* noundef getelementptr inbounds ([48 x i8], [48 x i8]* @.str.23, i64 0, i64 0))
          to label %25 unwind label %42

25:                                               ; preds = %22
  invoke void @_ZN6google8protobuf8internal11LogFinisheraSERNS1_10LogMessageE(%"class.google::protobuf::internal::LogFinisher"* noundef nonnull align 1 dereferenceable(1) %6, %"class.google::protobuf::internal::LogMessage"* noundef nonnull align 8 dereferenceable(56) %24)
          to label %27 unwind label %44

26:                                               ; preds = %15
  call void @llvm.lifetime.end.p0i8(i64 1, i8* nonnull %21) #24
  br label %28

27:                                               ; preds = %25
  call void @llvm.lifetime.end.p0i8(i64 1, i8* nonnull %21) #24
  call void @_ZN6google8protobuf8internal10LogMessageD1Ev(%"class.google::protobuf::internal::LogMessage"* noundef nonnull align 8 dereferenceable(56) %5) #24
  call void @llvm.lifetime.end.p0i8(i64 56, i8* nonnull %23) #24
  br label %28

28:                                               ; preds = %26, %27
  %29 = getelementptr inbounds %"class.google::protobuf::internal::RepeatedPtrFieldBase", %"class.google::protobuf::internal::RepeatedPtrFieldBase"* %0, i64 0, i32 1
  %30 = bitcast i32* %29 to i8*
  %31 = getelementptr inbounds %"class.google::protobuf::internal::RepeatedPtrFieldBase", %"class.google::protobuf::internal::RepeatedPtrFieldBase"* %1, i64 0, i32 1
  %32 = bitcast i32* %31 to i8*
  %33 = bitcast i32* %29 to i128*
  %34 = load i128, i128* %33, align 8
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* noundef nonnull align 8 dereferenceable(16) %30, i8* noundef nonnull align 1 dereferenceable(16) %32, i64 16, i1 false) #24
  %35 = bitcast i32* %31 to i128*
  store i128 %34, i128* %35, align 1
  ret void

36:                                               ; preds = %9
  %37 = landingpad { i8*, i32 }
          cleanup
  br label %40

38:                                               ; preds = %12
  %39 = landingpad { i8*, i32 }
          cleanup
  call void @llvm.lifetime.end.p0i8(i64 1, i8* nonnull %8) #24
  br label %40

40:                                               ; preds = %36, %38
  %41 = phi { i8*, i32 } [ %39, %38 ], [ %37, %36 ]
  call void @_ZN6google8protobuf8internal10LogMessageD1Ev(%"class.google::protobuf::internal::LogMessage"* noundef nonnull align 8 dereferenceable(56) %3) #24
  call void @llvm.lifetime.end.p0i8(i64 56, i8* nonnull %10) #24
  br label %48

42:                                               ; preds = %22
  %43 = landingpad { i8*, i32 }
          cleanup
  br label %46

44:                                               ; preds = %25
  %45 = landingpad { i8*, i32 }
          cleanup
  call void @llvm.lifetime.end.p0i8(i64 1, i8* nonnull %21) #24
  br label %46

46:                                               ; preds = %42, %44
  %47 = phi { i8*, i32 } [ %45, %44 ], [ %43, %42 ]
  call void @_ZN6google8protobuf8internal10LogMessageD1Ev(%"class.google::protobuf::internal::LogMessage"* noundef nonnull align 8 dereferenceable(56) %5) #24
  call void @llvm.lifetime.end.p0i8(i64 56, i8* nonnull %23) #24
  br label %48

48:                                               ; preds = %46, %40
  %49 = phi { i8*, i32 } [ %47, %46 ], [ %41, %40 ]
  resume { i8*, i32 } %49
}

; Function Attrs: inlinehint mustprogress uwtable
define linkonce_odr dso_local void @_ZN6google8protobuf8internal20RepeatedPtrFieldBase9MergeFromINS0_16RepeatedPtrFieldIN8tutorial6PersonEE11TypeHandlerEEEvRKS2_(%"class.google::protobuf::internal::RepeatedPtrFieldBase"* noundef nonnull align 8 dereferenceable(24) %0, %"class.google::protobuf::internal::RepeatedPtrFieldBase"* noundef nonnull align 8 dereferenceable(24) %1) local_unnamed_addr #17 comdat align 2 personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*) {
  %3 = alloca %"class.google::protobuf::internal::LogMessage", align 8
  %4 = alloca %"class.google::protobuf::internal::LogFinisher", align 1
  %5 = icmp eq %"class.google::protobuf::internal::RepeatedPtrFieldBase"* %1, %0
  %6 = getelementptr inbounds %"class.google::protobuf::internal::LogFinisher", %"class.google::protobuf::internal::LogFinisher"* %4, i64 0, i32 0
  call void @llvm.lifetime.start.p0i8(i64 1, i8* nonnull %6) #24
  br i1 %5, label %7, label %11

7:                                                ; preds = %2
  %8 = bitcast %"class.google::protobuf::internal::LogMessage"* %3 to i8*
  call void @llvm.lifetime.start.p0i8(i64 56, i8* nonnull %8) #24
  call void @_ZN6google8protobuf8internal10LogMessageC1ENS0_8LogLevelEPKci(%"class.google::protobuf::internal::LogMessage"* noundef nonnull align 8 dereferenceable(56) %3, i32 noundef 3, i8* noundef getelementptr inbounds ([46 x i8], [46 x i8]* @.str.16, i64 0, i64 0), i32 noundef 1787)
  %9 = invoke noundef nonnull align 8 dereferenceable(56) %"class.google::protobuf::internal::LogMessage"* @_ZN6google8protobuf8internal10LogMessagelsEPKc(%"class.google::protobuf::internal::LogMessage"* noundef nonnull align 8 dereferenceable(56) %3, i8* noundef getelementptr inbounds ([35 x i8], [35 x i8]* @.str.21, i64 0, i64 0))
          to label %10 unwind label %17

10:                                               ; preds = %7
  invoke void @_ZN6google8protobuf8internal11LogFinisheraSERNS1_10LogMessageE(%"class.google::protobuf::internal::LogFinisher"* noundef nonnull align 1 dereferenceable(1) %4, %"class.google::protobuf::internal::LogMessage"* noundef nonnull align 8 dereferenceable(56) %9)
          to label %12 unwind label %19

11:                                               ; preds = %2
  call void @llvm.lifetime.end.p0i8(i64 1, i8* nonnull %6) #24
  br label %13

12:                                               ; preds = %10
  call void @llvm.lifetime.end.p0i8(i64 1, i8* nonnull %6) #24
  call void @_ZN6google8protobuf8internal10LogMessageD1Ev(%"class.google::protobuf::internal::LogMessage"* noundef nonnull align 8 dereferenceable(56) %3) #24
  call void @llvm.lifetime.end.p0i8(i64 56, i8* nonnull %8) #24
  br label %13

13:                                               ; preds = %11, %12
  %14 = getelementptr inbounds %"class.google::protobuf::internal::RepeatedPtrFieldBase", %"class.google::protobuf::internal::RepeatedPtrFieldBase"* %1, i64 0, i32 1
  %15 = load i32, i32* %14, align 8, !tbaa !60
  %16 = icmp eq i32 %15, 0
  br i1 %16, label %78, label %23

17:                                               ; preds = %7
  %18 = landingpad { i8*, i32 }
          cleanup
  br label %21

19:                                               ; preds = %10
  %20 = landingpad { i8*, i32 }
          cleanup
  call void @llvm.lifetime.end.p0i8(i64 1, i8* nonnull %6) #24
  br label %21

21:                                               ; preds = %17, %19
  %22 = phi { i8*, i32 } [ %20, %19 ], [ %18, %17 ]
  call void @_ZN6google8protobuf8internal10LogMessageD1Ev(%"class.google::protobuf::internal::LogMessage"* noundef nonnull align 8 dereferenceable(56) %3) #24
  call void @llvm.lifetime.end.p0i8(i64 56, i8* nonnull %8) #24
  resume { i8*, i32 } %22

23:                                               ; preds = %13
  %24 = getelementptr inbounds %"class.google::protobuf::internal::RepeatedPtrFieldBase", %"class.google::protobuf::internal::RepeatedPtrFieldBase"* %1, i64 0, i32 3
  %25 = load %"struct.google::protobuf::internal::RepeatedPtrFieldBase::Rep"*, %"struct.google::protobuf::internal::RepeatedPtrFieldBase::Rep"** %24, align 8, !tbaa !58
  %26 = call noundef i8** @_ZN6google8protobuf8internal20RepeatedPtrFieldBase14InternalExtendEi(%"class.google::protobuf::internal::RepeatedPtrFieldBase"* noundef nonnull align 8 dereferenceable(24) %0, i32 noundef %15)
  %27 = getelementptr inbounds %"class.google::protobuf::internal::RepeatedPtrFieldBase", %"class.google::protobuf::internal::RepeatedPtrFieldBase"* %0, i64 0, i32 3
  %28 = load %"struct.google::protobuf::internal::RepeatedPtrFieldBase::Rep"*, %"struct.google::protobuf::internal::RepeatedPtrFieldBase::Rep"** %27, align 8, !tbaa !58
  %29 = getelementptr inbounds %"struct.google::protobuf::internal::RepeatedPtrFieldBase::Rep", %"struct.google::protobuf::internal::RepeatedPtrFieldBase::Rep"* %28, i64 0, i32 0
  %30 = load i32, i32* %29, align 8, !tbaa !61
  %31 = getelementptr inbounds %"class.google::protobuf::internal::RepeatedPtrFieldBase", %"class.google::protobuf::internal::RepeatedPtrFieldBase"* %0, i64 0, i32 1
  %32 = load i32, i32* %31, align 8, !tbaa !60
  %33 = sub i32 %30, %32
  %34 = icmp sgt i32 %33, 0
  %35 = icmp sgt i32 %15, 0
  %36 = and i1 %35, %34
  br i1 %36, label %37, label %43

37:                                               ; preds = %23
  %38 = zext i32 %15 to i64
  %39 = zext i32 %33 to i64
  %40 = add nsw i64 %38, -1
  %41 = add nsw i64 %39, -1
  %42 = call i64 @llvm.umin.i64(i64 %40, i64 %41)
  br label %49

43:                                               ; preds = %49, %23
  %44 = getelementptr inbounds %"class.google::protobuf::internal::RepeatedPtrFieldBase", %"class.google::protobuf::internal::RepeatedPtrFieldBase"* %0, i64 0, i32 0
  %45 = load %"class.google::protobuf::Arena"*, %"class.google::protobuf::Arena"** %44, align 8, !tbaa !16
  %46 = icmp slt i32 %33, %15
  br i1 %46, label %47, label %70

47:                                               ; preds = %43
  %48 = sext i32 %33 to i64
  br label %59

49:                                               ; preds = %37, %49
  %50 = phi i64 [ 0, %37 ], [ %57, %49 ]
  %51 = getelementptr inbounds %"struct.google::protobuf::internal::RepeatedPtrFieldBase::Rep", %"struct.google::protobuf::internal::RepeatedPtrFieldBase::Rep"* %25, i64 0, i32 1, i64 %50
  %52 = bitcast i8** %51 to %"class.tutorial::Person"**
  %53 = load %"class.tutorial::Person"*, %"class.tutorial::Person"** %52, align 8, !tbaa !27
  %54 = getelementptr inbounds i8*, i8** %26, i64 %50
  %55 = bitcast i8** %54 to %"class.tutorial::Person"**
  %56 = load %"class.tutorial::Person"*, %"class.tutorial::Person"** %55, align 8, !tbaa !27
  call void @_ZN6google8protobuf8internal18GenericTypeHandlerIN8tutorial6PersonEE5MergeERKS4_PS4_(%"class.tutorial::Person"* noundef nonnull align 8 dereferenceable(72) %53, %"class.tutorial::Person"* noundef %56)
  %57 = add nuw nsw i64 %50, 1
  %58 = icmp eq i64 %50, %42
  br i1 %58, label %43, label %49, !llvm.loop !88

59:                                               ; preds = %47, %59
  %60 = phi i64 [ %48, %47 ], [ %67, %59 ]
  %61 = getelementptr inbounds %"struct.google::protobuf::internal::RepeatedPtrFieldBase::Rep", %"struct.google::protobuf::internal::RepeatedPtrFieldBase::Rep"* %25, i64 0, i32 1, i64 %60
  %62 = bitcast i8** %61 to %"class.tutorial::Person"**
  %63 = load %"class.tutorial::Person"*, %"class.tutorial::Person"** %62, align 8, !tbaa !27
  %64 = call noundef %"class.tutorial::Person"* @_ZN6google8protobuf5Arena18CreateMaybeMessageIN8tutorial6PersonEJEEEPT_PS1_DpOT0_(%"class.google::protobuf::Arena"* noundef %45)
  call void @_ZN6google8protobuf8internal18GenericTypeHandlerIN8tutorial6PersonEE5MergeERKS4_PS4_(%"class.tutorial::Person"* noundef nonnull align 8 dereferenceable(72) %63, %"class.tutorial::Person"* noundef %64)
  %65 = getelementptr inbounds i8*, i8** %26, i64 %60
  %66 = bitcast i8** %65 to %"class.tutorial::Person"**
  store %"class.tutorial::Person"* %64, %"class.tutorial::Person"** %66, align 8, !tbaa !27
  %67 = add nsw i64 %60, 1
  %68 = trunc i64 %67 to i32
  %69 = icmp eq i32 %15, %68
  br i1 %69, label %70, label %59, !llvm.loop !89

70:                                               ; preds = %59, %43
  %71 = load i32, i32* %31, align 8, !tbaa !60
  %72 = add nsw i32 %71, %15
  store i32 %72, i32* %31, align 8, !tbaa !60
  %73 = load %"struct.google::protobuf::internal::RepeatedPtrFieldBase::Rep"*, %"struct.google::protobuf::internal::RepeatedPtrFieldBase::Rep"** %27, align 8, !tbaa !58
  %74 = getelementptr inbounds %"struct.google::protobuf::internal::RepeatedPtrFieldBase::Rep", %"struct.google::protobuf::internal::RepeatedPtrFieldBase::Rep"* %73, i64 0, i32 0
  %75 = load i32, i32* %74, align 8, !tbaa !61
  %76 = icmp slt i32 %75, %72
  br i1 %76, label %77, label %78

77:                                               ; preds = %70
  store i32 %72, i32* %74, align 8, !tbaa !61
  br label %78

78:                                               ; preds = %77, %70, %13
  ret void
}

; Function Attrs: mustprogress noinline uwtable
define linkonce_odr dso_local void @_ZN6google8protobuf8internal18GenericTypeHandlerIN8tutorial6PersonEE5MergeERKS4_PS4_(%"class.tutorial::Person"* noundef nonnull align 8 dereferenceable(72) %0, %"class.tutorial::Person"* noundef %1) local_unnamed_addr #22 comdat align 2 {
  tail call void @_ZN8tutorial6Person9MergeFromERKS0_(%"class.tutorial::Person"* noundef nonnull align 8 dereferenceable(72) %1, %"class.tutorial::Person"* noundef nonnull align 8 dereferenceable(72) %0)
  ret void
}

; Function Attrs: uwtable
define linkonce_odr dso_local noundef %"class.tutorial::Person"* @_ZN6google8protobuf5Arena14InternalHelperIN8tutorial6PersonEE9ConstructIJPS1_EEEPS4_PvDpOT_(i8* noundef %0, %"class.google::protobuf::Arena"** noundef nonnull align 8 dereferenceable(8) %1) local_unnamed_addr #3 comdat align 2 personality i32 (...)* @__gxx_personality_v0 {
  %3 = load %"class.google::protobuf::Arena"*, %"class.google::protobuf::Arena"** %1, align 8, !tbaa !27
  %4 = getelementptr inbounds i8, i8* %0, i64 8
  %5 = bitcast i8* %4 to %"class.google::protobuf::Arena"**
  store %"class.google::protobuf::Arena"* %3, %"class.google::protobuf::Arena"** %5, align 8, !tbaa !5
  %6 = bitcast i8* %0 to i32 (...)***
  store i32 (...)** bitcast (i8** getelementptr inbounds ({ [22 x i8*] }, { [22 x i8*] }* @_ZTVN8tutorial6PersonE, i64 0, inrange i32 0, i64 2) to i32 (...)**), i32 (...)*** %6, align 8, !tbaa !10
  %7 = getelementptr inbounds i8, i8* %0, i64 16
  %8 = bitcast i8* %7 to %"class.google::protobuf::Arena"**
  store %"class.google::protobuf::Arena"* %3, %"class.google::protobuf::Arena"** %8, align 8, !tbaa !16
  %9 = getelementptr inbounds i8, i8* %0, i64 24
  tail call void @llvm.memset.p0i8.i64(i8* noundef nonnull align 8 dereferenceable(16) %9, i8 0, i64 16, i1 false) #24
  %10 = getelementptr inbounds i8, i8* %0, i64 68
  %11 = bitcast i8* %10 to i32*
  store i32 0, i32* %11, align 4, !tbaa !13
  %12 = load atomic i32, i32* getelementptr inbounds ({ { { i32 }, i32, i32, void ()* }, [2 x i8*] }, { { { i32 }, i32, i32, void ()* }, [2 x i8*] }* @scc_info_Person_addressbook_2eproto, i64 0, i32 0, i32 0, i32 0) acquire, align 8
  %13 = icmp eq i32 %12, 0
  br i1 %13, label %31, label %14, !prof !12

14:                                               ; preds = %2
  invoke void @_ZN6google8protobuf8internal11InitSCCImplEPNS1_11SCCInfoBaseE(%"struct.google::protobuf::internal::SCCInfoBase"* noundef nonnull bitcast ({ { { i32 }, i32, i32, void ()* }, [2 x i8*] }* @scc_info_Person_addressbook_2eproto to %"struct.google::protobuf::internal::SCCInfoBase"*))
          to label %31 unwind label %15

15:                                               ; preds = %14
  %16 = landingpad { i8*, i32 }
          cleanup
  %17 = bitcast i8* %7 to %"class.google::protobuf::internal::RepeatedPtrFieldBase"*
  invoke void @_ZN6google8protobuf8internal20RepeatedPtrFieldBase7DestroyINS0_16RepeatedPtrFieldIN8tutorial18Person_PhoneNumberEE11TypeHandlerEEEvv(%"class.google::protobuf::internal::RepeatedPtrFieldBase"* noundef nonnull align 8 dereferenceable(24) %17)
          to label %18 unwind label %27

18:                                               ; preds = %15
  %19 = load %"class.google::protobuf::Arena"*, %"class.google::protobuf::Arena"** %8, align 8, !tbaa !16
  %20 = icmp eq %"class.google::protobuf::Arena"* %19, null
  br i1 %20, label %30, label %21

21:                                               ; preds = %18
  %22 = getelementptr inbounds %"class.google::protobuf::Arena", %"class.google::protobuf::Arena"* %19, i64 0, i32 0
  %23 = invoke noundef i64 @_ZNK6google8protobuf8internal9ArenaImpl14SpaceAllocatedEv(%"class.google::protobuf::internal::ArenaImpl"* noundef nonnull align 8 dereferenceable(88) %22)
          to label %30 unwind label %24

24:                                               ; preds = %21
  %25 = landingpad { i8*, i32 }
          catch i8* null
  %26 = extractvalue { i8*, i32 } %25, 0
  tail call void @__clang_call_terminate(i8* %26) #25
  unreachable

27:                                               ; preds = %15
  %28 = landingpad { i8*, i32 }
          catch i8* null
  %29 = extractvalue { i8*, i32 } %28, 0
  tail call void @_ZN6google8protobuf8internal20RepeatedPtrFieldBaseD2Ev(%"class.google::protobuf::internal::RepeatedPtrFieldBase"* noundef nonnull align 8 dereferenceable(24) %17) #24
  tail call void @__clang_call_terminate(i8* %29) #25
  unreachable

30:                                               ; preds = %18, %21
  resume { i8*, i32 } %16

31:                                               ; preds = %2, %14
  %32 = bitcast i8* %0 to %"class.tutorial::Person"*
  %33 = getelementptr inbounds i8, i8* %0, i64 40
  %34 = bitcast i8* %33 to %"class.std::__cxx11::basic_string"**
  store %"class.std::__cxx11::basic_string"* bitcast (%"class.google::protobuf::internal::ExplicitlyConstructed.20"* @_ZN6google8protobuf8internal26fixed_address_empty_stringB5cxx11E to %"class.std::__cxx11::basic_string"*), %"class.std::__cxx11::basic_string"** %34, align 8, !tbaa !18
  %35 = getelementptr inbounds i8, i8* %0, i64 48
  %36 = bitcast i8* %35 to %"class.std::__cxx11::basic_string"**
  store %"class.std::__cxx11::basic_string"* bitcast (%"class.google::protobuf::internal::ExplicitlyConstructed.20"* @_ZN6google8protobuf8internal26fixed_address_empty_stringB5cxx11E to %"class.std::__cxx11::basic_string"*), %"class.std::__cxx11::basic_string"** %36, align 8, !tbaa !18
  %37 = getelementptr inbounds i8, i8* %0, i64 56
  tail call void @llvm.memset.p0i8.i64(i8* noundef nonnull align 8 dereferenceable(12) %37, i8 0, i64 12, i1 false)
  ret %"class.tutorial::Person"* %32
}

; Function Attrs: uwtable
define internal void @_GLOBAL__sub_I_addressbook.pb.cc() #3 section ".text.startup" {
  tail call void @_ZNSt8ios_base4InitC1Ev(%"class.std::ios_base::Init"* noundef nonnull align 1 dereferenceable(1) @_ZStL8__ioinit)
  %1 = tail call i32 @__cxa_atexit(void (i8*)* bitcast (void (%"class.std::ios_base::Init"*)* @_ZNSt8ios_base4InitD1Ev to void (i8*)*), i8* getelementptr inbounds (%"class.std::ios_base::Init", %"class.std::ios_base::Init"* @_ZStL8__ioinit, i64 0, i32 0), i8* nonnull @__dso_handle) #24
  tail call void @_ZN6google8protobuf8internal14AddDescriptorsEPKNS1_15DescriptorTableE(%"struct.google::protobuf::internal::DescriptorTable"* noundef nonnull @descriptor_table_addressbook_2eproto)
  ret void
}

; Function Attrs: nofree nosync nounwind readnone speculatable willreturn
declare i64 @llvm.umin.i64(i64, i64) #23

attributes #0 = { "frame-pointer"="none" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #1 = { nounwind "frame-pointer"="none" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #2 = { nofree nounwind }
attributes #3 = { uwtable "frame-pointer"="none" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #4 = { mustprogress uwtable "frame-pointer"="none" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #5 = { mustprogress nofree norecurse nosync nounwind readnone uwtable willreturn "frame-pointer"="none" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #6 = { nounwind uwtable "frame-pointer"="none" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #7 = { inlinehint uwtable "frame-pointer"="none" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #8 = { noinline noreturn nounwind }
attributes #9 = { nobuiltin nounwind "frame-pointer"="none" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #10 = { argmemonly mustprogress nofree nosync nounwind willreturn }
attributes #11 = { mustprogress nofree norecurse nounwind uwtable willreturn "frame-pointer"="none" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #12 = { mustprogress nofree norecurse nosync nounwind uwtable willreturn writeonly "frame-pointer"="none" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #13 = { mustprogress nofree norecurse nosync nounwind readonly uwtable willreturn "frame-pointer"="none" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #14 = { mustprogress nounwind uwtable "frame-pointer"="none" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #15 = { nobuiltin allocsize(0) "frame-pointer"="none" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #16 = { noinline uwtable "frame-pointer"="none" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #17 = { inlinehint mustprogress uwtable "frame-pointer"="none" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #18 = { argmemonly mustprogress nofree nounwind willreturn }
attributes #19 = { mustprogress nofree nosync nounwind readnone speculatable willreturn }
attributes #20 = { argmemonly mustprogress nofree nounwind willreturn writeonly }
attributes #21 = { nofree nounwind readonly }
attributes #22 = { mustprogress noinline uwtable "frame-pointer"="none" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #23 = { nofree nosync nounwind readnone speculatable willreturn }
attributes #24 = { nounwind }
attributes #25 = { noreturn nounwind }
attributes #26 = { builtin nounwind }
attributes #27 = { builtin allocsize(0) }

!llvm.module.flags = !{!0, !1, !2, !3}
!llvm.ident = !{!4}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 7, !"PIC Level", i32 2}
!2 = !{i32 7, !"PIE Level", i32 2}
!3 = !{i32 7, !"uwtable", i32 1}
!4 = !{!"Ubuntu clang version 14.0.0-1ubuntu1.1"}
!5 = !{!6, !7, i64 0}
!6 = !{!"_ZTSN6google8protobuf8internal16InternalMetadataE", !7, i64 0}
!7 = !{!"any pointer", !8, i64 0}
!8 = !{!"omnipotent char", !9, i64 0}
!9 = !{!"Simple C++ TBAA"}
!10 = !{!11, !11, i64 0}
!11 = !{!"vtable pointer", !9, i64 0}
!12 = !{!"branch_weights", i32 2000, i32 1}
!13 = !{!14, !15, i64 0}
!14 = !{!"_ZTSSt13__atomic_baseIiE", !15, i64 0}
!15 = !{!"int", !8, i64 0}
!16 = !{!17, !7, i64 0}
!17 = !{!"_ZTSN6google8protobuf8internal20RepeatedPtrFieldBaseE", !7, i64 0, !15, i64 8, !15, i64 12, !7, i64 16}
!18 = !{!19, !7, i64 0}
!19 = !{!"_ZTSN6google8protobuf8internal14ArenaStringPtrE", !7, i64 0}
!20 = !{!21, !7, i64 56}
!21 = !{!"_ZTSN8tutorial6PersonE", !22, i64 16, !19, i64 40, !19, i64 48, !7, i64 56, !15, i64 64, !23, i64 68}
!22 = !{!"_ZTSN6google8protobuf16RepeatedPtrFieldIN8tutorial18Person_PhoneNumberEEE"}
!23 = !{!"_ZTSN6google8protobuf8internal10CachedSizeE", !24, i64 0}
!24 = !{!"_ZTSSt6atomicIiE"}
!25 = !{!26, !15, i64 24}
!26 = !{!"_ZTSN8tutorial18Person_PhoneNumberE", !19, i64 16, !15, i64 24, !23, i64 28}
!27 = !{!7, !7, i64 0}
!28 = !{!29, !31, i64 8}
!29 = !{!"_ZTSNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE", !30, i64 0, !31, i64 8, !8, i64 16}
!30 = !{!"_ZTSNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE12_Alloc_hiderE", !7, i64 0}
!31 = !{!"long", !8, i64 0}
!32 = !{!33, !7, i64 0}
!33 = !{!"_ZTSN6google8protobuf8internal16InternalMetadata13ContainerBaseE", !7, i64 0}
!34 = !{!29, !7, i64 0}
!35 = !{!36, !7, i64 0}
!36 = !{!"_ZTSNSt12_Vector_baseIN6google8protobuf12UnknownFieldESaIS2_EE17_Vector_impl_dataE", !7, i64 0, !7, i64 8, !7, i64 16}
!37 = !{!8, !8, i64 0}
!38 = !{!39, !15, i64 92}
!39 = !{!"_ZTSN6google8protobuf8internal12ParseContextE", !15, i64 88, !15, i64 92, !40, i64 96}
!40 = !{!"_ZTSN6google8protobuf8internal12ParseContext4DataE", !7, i64 0, !7, i64 8}
!41 = !{!"branch_weights", i32 1, i32 2000}
!42 = !{!43, !15, i64 80}
!43 = !{!"_ZTSN6google8protobuf8internal18EpsCopyInputStreamE", !7, i64 0, !7, i64 8, !7, i64 16, !15, i64 24, !15, i64 28, !7, i64 32, !8, i64 40, !31, i64 72, !15, i64 80, !15, i64 84}
!44 = !{!45, !7, i64 0}
!45 = !{!"_ZTSN6google8protobuf2io19EpsCopyOutputStreamE", !7, i64 0, !7, i64 8, !8, i64 16, !7, i64 48, !46, i64 56, !46, i64 57, !46, i64 58}
!46 = !{!"bool", !8, i64 0}
!47 = distinct !{!47, !48}
!48 = !{!"llvm.loop.mustprogress"}
!49 = !{i32 0, i32 33}
!50 = !{!36, !7, i64 16}
!51 = !{!15, !15, i64 0}
!52 = !{!53, !7, i64 88}
!53 = !{!"_ZTSN6google8protobuf8internal15DescriptorTableE", !46, i64 0, !46, i64 1, !7, i64 8, !7, i64 16, !15, i64 24, !7, i64 32, !7, i64 40, !7, i64 48, !15, i64 56, !15, i64 60, !7, i64 64, !7, i64 72, !7, i64 80, !7, i64 88, !15, i64 96, !7, i64 104, !7, i64 112}
!54 = !{i64 0, i64 8, !27, i64 8, i64 8, !27}
!55 = !{i64 0, i64 8, !27}
!56 = !{!21, !15, i64 64}
!57 = !{!"branch_weights", i32 2002, i32 2000}
!58 = !{!17, !7, i64 16}
!59 = !{!17, !15, i64 12}
!60 = !{!17, !15, i64 8}
!61 = !{!62, !15, i64 0}
!62 = !{!"_ZTSN6google8protobuf8internal20RepeatedPtrFieldBase3RepE", !15, i64 0, !8, i64 8}
!63 = !{!39, !15, i64 88}
!64 = !{!43, !15, i64 28}
!65 = !{!43, !7, i64 8}
!66 = !{!43, !7, i64 0}
!67 = distinct !{!67, !48}
!68 = distinct !{!68, !48}
!69 = distinct !{!69, !48}
!70 = distinct !{!70, !48}
!71 = !{!"branch_weights", i32 2000, i32 2002}
!72 = distinct !{!72, !48}
!73 = distinct !{!73, !48}
!74 = !{!75, !7, i64 112}
!75 = !{!"_ZTSN6google8protobuf5ArenaE", !76, i64 0, !7, i64 88, !7, i64 96, !7, i64 104, !7, i64 112}
!76 = !{!"_ZTSN6google8protobuf8internal9ArenaImplE", !77, i64 0, !77, i64 8, !79, i64 16, !7, i64 24, !31, i64 32, !80, i64 40}
!77 = !{!"_ZTSSt6atomicIPN6google8protobuf8internal9ArenaImpl11SerialArenaEE", !78, i64 0}
!78 = !{!"_ZTSSt13__atomic_baseIPN6google8protobuf8internal9ArenaImpl11SerialArenaEE", !7, i64 0}
!79 = !{!"_ZTSSt6atomicImE"}
!80 = !{!"_ZTSN6google8protobuf8internal9ArenaImpl7OptionsE", !31, i64 0, !31, i64 8, !7, i64 16, !31, i64 24, !7, i64 32, !7, i64 40}
!81 = !{!30, !7, i64 0}
!82 = !{!31, !31, i64 0}
!83 = distinct !{!83, !48}
!84 = distinct !{!84, !48}
!85 = distinct !{!85, !48}
!86 = distinct !{!86, !48}
!87 = distinct !{!87, !48}
!88 = distinct !{!88, !48}
!89 = distinct !{!89, !48}
