
+ 环境
```shell
// protoc

sudo apt update
sudo apt install protobuf-compiler -y
sudo apt install -y libprotobuf-dev autoconf automake libtool device-tree-compiler texinfo pip git bison flex build-essential libssl-dev bc 

sudo apt-get install clang lld clang-tools -y
sudo apt install -y \
    build-essential cmake ninja-build python3 python3-pip \
    git curl zip unzip tar \
    libedit-dev libncurses5-dev libxml2-dev zlib1g-dev \
    libffi-dev libssl-dev libtinfo-dev

// llvm & mlir compile
git clone https://github.com/llvm/llvm-project.git
// 18.0.0
git checkout 186a4b3b657878ae2aea23caf684b6e103901162
cd llvm-project
mkdir build
cd build
rm -rf CMakeCache.txt CMakeFiles
cmake -G Ninja ../llvm \
  -DCMAKE_INSTALL_PREFIX=~/EECS6894-fa25/protoacc_project/install \
  -DLLVM_ENABLE_PROJECTS="mlir;clang" \
  -DLLVM_BUILD_EXAMPLES=ON \
  -DLLVM_TARGETS_TO_BUILD="Native;NVPTX;AMDGPU" \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DLLVM_INSTALL_UTILS=ON \
  -DMLIR_INCLUDE_TESTS=ON \
  -DMLIR_ENABLE_CUDA_RUNNER=OFF \
  -DCLANG_ENABLE_EXPERIMENTAL_NEW_PARSER=ON \
  -DCLANG_ENABLE_CIR=ON \
  -DLLVM_USE_LINKER=lld

ninja -j$(nproc)
sudo ninja install

// likwid 
sudo apt install likwid -y
sudo likwid-accessD start
sudo modprobe msr
sudo modprobe cpuid

```

+ protoacc-opt 编译：进入protoacc目录

```shell
mkdir build
cd build
sudo rm -rf CMakeCache.txt CMakeFiles
cmake -G Ninja .. \
	-DMLIR_DIR="~/EECS6894-fa25/protoacc_project/install/lib/cmake/mlir" \
	-DLLVM_DIR="~/EECS6894-fa25/protoacc_project/install/lib/cmake/llvm" 
	
sudo ninja -j$(nproc)
sudo ninja install protoacc-opt
```

+ 评估效果：进入HyperProtoBench目录

```shell
// 在HyperProtoBench中运行没有protoacc-opt和有的benchmark
pip install -r requirements.txt 
python3 run_all.py
```

+ 生成的roofline图见HyperProtoBench下的`roofline_compare.png`

+ `run_all.py`会自动在每个子目录下执行`make`和`make protoacc`，如果修改了protoacc-opt想要重新测试的话，可以直接在HyperProtoBench目录or在单个子目录下面`make clean`然后再执行`run_all.py`（前者是清除所有子目录下的文件，后者是单个）