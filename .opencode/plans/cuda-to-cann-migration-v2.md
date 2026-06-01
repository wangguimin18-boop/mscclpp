# MSCCL++ CUDA → CANN 迁移项目（v2 完整版）

## 项目目标

将 MSCCL++ 从 CUDA 平台迁移到华为 CANN（Ascend NPU）平台，使其可在灵衢超节点上运行。

---

## 1. CUDA 依赖完整扫描

### 1.1 CUDA 头文件引用（7 个唯一头文件，14+ 处 include）

| 类别 | 头文件 | 涉及文件 | 说明 |
|------|--------|----------|------|
| Runtime API | `cuda_runtime.h` | gpu.hpp, nvls_test.cu, peer_access_test.cu, check_nvidia_gpu.cu | 设备/内存/流管理 |
| Runtime API（纯声明） | `cuda_runtime_api.h` | gpu_data_types.hpp | CUDART_VERSION 宏和类型定义 |
| Driver API | `cuda.h` | gpu.hpp, nvls_test.cu | CUresult, CUdeviceptr, cuMem* 虚拟内存/NVLS |
| Driver 类型定义 | `cudaTypedefs.h` | nvls_test.cu | CUmemAllocationHandleType 等 |
| FP16 | `cuda_fp16.h` | gpu_data_types.hpp, switch_channel_device.hpp, executor_test_verifier.cu, python/mscclpp_benchmark/allreduce.cu | __half, __half2 |
| BF16 | `cuda_bf16.h` | gpu_data_types.hpp, executor_test_verifier.cu | __nv_bfloat16（CUDA >= 11.0） |
| FP8 | `cuda_fp8.h` | gpu_data_types.hpp | __nv_fp8_e4m3（CUDA >= 11.8） |
| CUDA C++ 标准库 | `cuda/atomic` | atomic_device.hpp | cuda::atomic, cuda::memory_order |

**注意**：项目已支持 HIP（AMD ROCm），所有 CUDA include 都有 `#if defined(MSCCLPP_DEVICE_HIP)` 条件分支。CANN 分支也已有初始桩代码（gpu.hpp 中 `MSCCLPP_DEVICE_CANN` 分支定义了类型映射：cudaError_t→aclError, cudaStream_t→aclrtStream, cudaEvent_t→aclrtNotify, CUresult→aclError 等），但此前文档未记录此进度。

**需新增 CANN 分支的 device 头文件**（此前未详细列出）：
- `switch_channel_device.hpp` — NVLS multimem PTX 指令
- `semaphore_device.hpp` — PTX `red.release.sys` / cuda::atomic
- `packet_device.hpp` — PTX volatile load/store
- `fifo_device.hpp` — PTX release store + `__CUDA_ARCH__==800` 特殊路径
- `atomic_device.hpp` — cuda::atomic 映射
- `assert_device.hpp` — `__device__ __assert_fail` 声明
- `concurrency_device.hpp` — `__syncthreads`, DeviceSyncer, DeviceSemaphore

### 1.2 CUDA Runtime API 完整分类

#### 设备管理（迁移难度：低）

| API | CANN 对应 | 调用次数 | 核心文件 |
|-----|-----------|---------|----------|
| cudaGetDevice | aclrtGetDevice | ~25 | gpu_utils.cc, proxy.cc, port_channel.cc, fifo.cc, endpoint.cc, debug.cc, registered_memory.cc, connection.cc, npkit.cc, gpu_ipc_mem.cc, nccl.cc |
| cudaSetDevice | aclrtSetDevice | ~18 | gpu_utils.cc, proxy.cc, port_channel.cc, connection.cc, 各 test/example |
| cudaGetDeviceCount | aclrtGetDeviceCount | ~6 | gpu_utils.cc, peer_access_test.cu, numa_tests.cc, ib_tests.cu |
| cudaDeviceSynchronize | aclrtSynchronize | ~40+ | 各 test/example |
| cudaGetDeviceProperties | aclrtGetDeviceInfo | ~3 | common.cc, allreduce_nvls_zero_copy.cu, npkit.cc |
| cudaDeviceCanAccessPeer | 需确认 CANN 对应 | ~4 | peer_access_test.cu, tutorial examples |
| cudaDeviceEnablePeerAccess | 需确认 CANN 对应 | 1 | connection.cc |
| cudaDeviceGetAttribute | aclrtGetDeviceInfo | 2 | nccl.cc（cudaDevAttrComputeCapabilityMajor/Minor） **←v1缺失** |
| cudaDeviceGetPCIBusId | **需确认** | 3 | common.cc, numa.cc **←v1缺失** |

**新增**：`cudaErrorPeerAccessAlreadyEnabled` 错误码（connection.cc），需映射到 CANN 错误码。

#### 内存管理（迁移难度：低-中）

| API | CANN 对应 | 调用次数 | 核心文件 |
|-----|-----------|---------|----------|
| cudaMalloc | aclrtMalloc | ~15 | gpu_utils.cc, 各 test/example |
| cudaFree | aclrtFree | ~5 | gpu_utils.cc, 各 test/example |
| cudaHostAlloc | aclrtMallocHost | ~3 | gpu_utils.cc |
| cudaFreeHost | aclrtFreeHost | 1 | gpu_utils.cc |
| cudaMemcpy | aclrtMemcpy | ~30+ | 各 test/example, connection.cc, algorithm.cc |
| cudaMemcpyAsync | aclrtMemcpyAsync | ~15 | gpu_utils.cc, context.cc, nccl.cc, collective_utils.cc, executor.cc |
| cudaMemcpyToSymbol | **需特殊处理** | ~54 | 各 test（device 常量内存赋值） |
| cudaMemcpyToSymbolAsync | **需特殊处理** | 定义于 gpu.hpp:322 **←v1缺失** |
| cudaMemset | aclrtMemset | 4 | gpu.hpp:126, nvls_test.cu, sendrecv_test.cu, allgather_test_host_offloading.cu **←v1完全缺失** |
| cudaMemsetAsync | aclrtMemsetAsync | ~8 | gpu.hpp:129, gpu_utils.cc (8处: 98,126,254,273,356,402) **←v1完全缺失** |

**关键问题**：
- `cudaMemcpyToSymbol` 用于将 host 数据拷贝到 device 全局/常量变量。CANN 无直接对应，需改为通过 device 全局指针 + aclrtMemcpy 方式。灵衢替代方案：使用 `peerMems[]` 或 `CommArgs` 传递参数到 kernel，替代 cudaMemcpyToSymbol 的"常量内存赋值"模式。
- `cudaMemset/cudaMemsetAsync` 用于 GPU 内存清零。CANN 对应 `aclrtMemset/aclrtMemsetAsync`。

#### Stream/Event 管理（迁移难度：低-中）

| API | CANN 对应 | 调用次数 | 备注 |
|-----|-----------|---------|------|
| cudaStreamCreate | aclrtCreateStream | ~10 | 直接对应 |
| cudaStreamCreateWithFlags | aclrtCreateStream | 少量 | flags 不同 |
| cudaStreamDestroy | aclrtDestroyStream | ~3 | 直接对应 |
| cudaStreamSynchronize | aclrtSynchronizeStream | ~20+ | 直接对应 |
| cudaEventCreate | aclrtCreateNotify | ~12 | Notify 而非 Event |
| cudaEventCreateWithFlags | aclrtCreateNotify(notify, ACL_NOTIFY_DEFAULT) | 少量 | HCCL 使用 flags 参数 |
| cudaEventRecord | aclrtRecordNotify | ~10 | 对应 |
| cudaEventSynchronize | aclrtWaitAndResetNotify(notify, stream, timeout) | ~10 | HCCL 带 timeout 参数 |
| cudaEventDestroy | aclrtDestroyNotify | 少量 | 对应 |
| cudaEventElapsedTime | **需确认** | 5+ | 所有 5 个 tutorial example **←v1缺失** |

#### CUDA Graph API（迁移难度：中） **←v1完全缺失**

| API | CANN 对应 | 调用次数 | 核心文件 |
|-----|-----------|---------|----------|
| cudaGraphCreate | aclrtGraphCreate? | 4+ | common.cc, bidir_port_channel.cu, bidir_memory_channel.cu, customized_allgather.cu |
| cudaGraphInstantiate | aclrtGraphInstantiate? | 4+ | common.cc, executor_test.cc, bidir_port_channel.cu |
| cudaGraphLaunch | aclrtGraphLaunch? | 4+ | common.cc, executor_test.cc, bidir_port_channel.cu |
| cudaGraphDestroy | aclrtGraphDestroy? | 3+ | common.cc, executor_test.cc |
| cudaGraphExecDestroy | aclrtGraphExecDestroy? | 2+ | common.cc, executor_test.cc |
| cudaStreamBeginCapture | aclmdlRICaptureGetInfo? | 5+ | common.cc, executor_test.cc, bidir_port_channel.cu, bidir_memory_channel.cu, customized_allgather.cu |
| cudaStreamEndCapture | 对应 | 5+ | 同上 |
| cudaStreamIsCapturing | 对应 | 1 | nccl.cc |
| cudaThreadExchangeStreamCaptureMode | aclmdlRICaptureThreadExchangeMode | 2 | proxy.cc, gpu_utils.cc |
| AvoidCudaGraphCaptureGuard | **需自行实现** | ~10 | gpu_utils.hpp, gpu_utils.cc |
| cudaStreamCaptureMode | aclmdlRICaptureThreadExchangeMode(RELAXED) | 2 | common.cc, gpu.hpp |
| cudaStreamCaptureStatus | 对应 | 1 | gpu.hpp |

**关键问题**：CUDA Graph 在测试框架和教程中广泛使用。HCCL 已有对应的 capture mode API：
- `aclmdlRICaptureGetInfo(stream, &captureStatus, &modelId)` — 查询 stream capture 状态
- `aclmdlRICaptureThreadExchangeMode(&mode)` — 设置 RELAXED 模式

MSCCL++ 的 `AvoidCudaGraphCaptureGuard` 是一个 RAII 守卫，在 proxy thread 中临时退出 graph capture 模式以调用 CUDA API。迁移时需实现对应机制。

#### IPC 内存（迁移难度：高）

| API | CANN 对应 | 核心文件 | 备注 |
|-----|-----------|----------|------|
| cudaIpcGetMemHandle | obmem_export_memory | gpu_ipc_mem.cc | 返回 token_id + uba 替代 IPC handle |
| cudaIpcOpenMemHandle | obmem_import_memory | gpu_ipc_mem.cc | 通过 token_id + uba 导入 |
| cudaIpcCloseMemHandle | obmem_release_import_memory | gpu_ipc_mem.cc | 释放导入内存 |

**新增细节**（v1 缺失）：
- `MSCCLPP_CUDAIPC_USE_DEFAULT_STREAM` 环境变量（env.cpp:64）控制 IPC 是否使用默认流
- `cudaIpcMemLazyEnablePeerAccess` 常量（gpu_ipc_mem.cc:113）作为 `cudaIpcOpenMemHandle` 的 flags
- `CudaIpcMemHandleHash/CudaIpcMemHandleEqual` 自定义哈希/相等结构（gpu_ipc_mem.cc:86-95）

#### 错误处理（迁移难度：低）

| API | CANN 对应 | 备注 |
|-----|-----------|------|
| cudaGetErrorString | aclrtGetErrorCode / 自定义映射 | |
| cudaGetLastError | 需确认 | |
| cudaSuccess | ACL_SUCCESS | |
| cudaErrorPeerAccessAlreadyEnabled | **需映射** | connection.cc **←v1缺失** |

**新增**：`MSCCLPP_CUDATHROW` / `MSCCLPP_CUTHROW` 宏（gpu_utils.hpp:17,28）是所有 CUDA/Driver API 调用的核心错误检查包装器，需映射为 `MSCCLPP_ACLTHROW` / `MSCCLPP_ACLDTHROW`。

**新增 CUDA 错误码映射**（v1 缺失）：
- CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_CONTEXT_IS_DESTROYED, CUDA_ERROR_LAUNCH_FAILED
- CUDA_ERROR_NOT_SUPPORTED, CUDA_ERROR_INVALID_VALUE, CUDA_ERROR_NOT_PERMITTED
- 均需映射到对应的 aclError 值

### 1.3 CUDA Driver API 完整分类（迁移难度：极高）

#### 虚拟内存管理

| API | CANN 对应 | 核心文件 | 备注 |
|-----|-----------|----------|------|
| cuMemCreate | **无直接对应** | gpu_utils.cc, gpu_ipc_mem.cc | 需用 aclrtMalloc 替代 |
| cuMemMap | **无直接对应** | gpu_utils.cc, gpu_ipc_mem.cc, nvls_test.cu | CANN 无虚拟内存管理 |
| cuMemUnmap | **无直接对应** | gpu_utils.cc, gpu_ipc_mem.cc | |
| cuMemAddressReserve | **无直接对应** | gpu_utils.cc, gpu_ipc_mem.cc, nvls_test.cu | CANN 无虚拟地址空间 |
| cuMemAddressFree | **无直接对应** | gpu_utils.cc, gpu_ipc_mem.cc | |
| cuMemSetAccess | **无直接对应** | gpu_utils.cc, gpu_ipc_mem.cc, nvls_test.cu | 需用 HCCL IPC 替代 |
| cuMemGetAllocationGranularity | **无对应** | gpu_utils.cc, gpu_ipc_mem.cc | 需硬编码或 aclrtGetDeviceInfo |
| cuMemGetAddressRange | **无对应** | gpu_utils.cc, utils_internal.cc, executor.cc, 各 allreduce/allgather .cu | 需自行管理地址范围 |
| cuMemRelease | **无对应** | gpu_utils.cc | 虚拟内存释放 |
| cuMemExportToShareableHandle | obmem_export_memory | gpu_ipc_mem.cc, nvls_test.cu | OBMM 替代 |
| cuMemImportFromShareableHandle | obmem_import_memory | gpu_ipc_mem.cc, nvls_test.cu | OBMM 替代 |
| cuMemRetainAllocationHandle | **无对应** | gpu_utils.cc, gpu_ipc_mem.cc | 需重新设计 |
| cuMemGetHandleForAddressRange | **无对应** | ib.cc | DMA_BUF 概念不适用 |

**新增细节**（v1 缺失）：
- `CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD` 和 `CU_MEM_RANGE_FLAG_DMA_BUF_MAPPING_TYPE_PCIE`（ib.cc）— Linux DMA_BUF 专用 flags，CANN 无对应
- `CU_DEVICE_ATTRIBUTE_FABRIC_SUPPORTED` / `DMA_BUF_SUPPORTED` / `MULTICAST_SUPPORTED`（gpu_utils.cc, gpu_ipc_mem.cc, ib.cc）— 需映射到 CANN 设备属性查询

#### NVLS Multicast（迁移难度：极高）

| API | 灵衢替代方案 | 核心文件 |
|-----|-------------|----------|
| cuMulticastCreate | OBMM export_memory（NPU A 导出共享内存） | gpu_ipc_mem.cc, nvls_test.cu |
| cuMulticastAddDevice | OBMM import_memory（其他 NPU 导入） | gpu_ipc_mem.cc, nvls_test.cu |
| cuMulticastBindAddr | peerMems[] + windowsIn/Out | gpu_ipc_mem.cc, nvls_test.cu |
| cuMulticastUnbind | obmem_release_import_memory | gpu_ipc_mem.cc |
| cuMulticastGetGranularity | OBMM 4MB 对齐（硬编码或查询） | switch_channel.cc, gpu_utils.cc, gpu_ipc_mem.cc |

#### Driver 其他 API

| API | CANN 对应 | 核心文件 |
|-----|-----------|----------|
| cuDeviceGet | aclrtGetDevice | gpu_utils.cc, ib.cc, gpu_ipc_mem.cc |
| cuDeviceGetAttribute | aclrtGetDeviceInfo | gpu_utils.cc, ib.cc, gpu_ipc_mem.cc |
| cuPointerGetAttribute | **无对应** | gpu_utils.cc — 需自行管理指针元数据 |
| cuGetErrorString | aclrtGetErrorCode / 自定义映射 | registered_memory.cc, gpu_utils_internal.hpp, gpu_ipc_mem.cc, errors.cc |

### 1.4 CUDA Device Keywords 完整映射（迁移难度：极高）

| 关键字/机制 | 说明 | 涉及范围 | Ascend C 对应 |
|-------------|------|----------|---------------|
| `__global__` | kernel 入口 | ~70+ unique kernel 定义 | Ascend C kernel 入口 |
| `__device__` | device 函数/变量 | 全部 .cu 文件 + device.hpp | Ascend C device 函数 |
| `__shared__` | shared memory | allreduce_packet.cu, allreduce_nvls_*.cu, **execution_kernel.hpp(14变量)**←v1缺失, allreduce_fullmesh.cu, allgather_fullmesh.cu | Ascend C local memory |
| `__syncthreads()` | 編程模型内线程同步 | 全部 kernel | Ascend C 同步机制（不同编程模型） |
| `blockIdx, threadIdx, blockDim, gridDim` | 线程索引 | 全部 kernel | Ascend C 6维并行（完全不同） |
| `warpSize` | warp 大小 | **实际代码搜索：0 处调用** ←v1错误声称在 allreduce .cu 中 | Ascend C 无 warp 概念 |
| `__shfl_sync()` | warp shuffle | **实际代码搜索：0 处调用** ←v1错误声称在 allreduce .cu 中 | Ascend C 无 warp shuffle |
| `atomicAdd` | 原子加 | **实际代码搜索：0 处调用** ←v1错误声称在 allreduce .cu 中 | Ascend C atomic（API不同） |
| `__launch_bounds__` | 强制内联+寄存器优化 | **54+ kernel** ←v1完全缺失 | Ascend C 无对应概念 |
| `__forceinline__` | 强制内联 | allreduce .cu | Ascend C 内联不同 |
| `__device__ __assert_fail` | device 侧断言 | assert_device.hpp ←v1缺失 | 需 Ascend C 断言替代 |

**纠正**：v1 文档声称 `atomicAdd`、`__shfl_sync()`、`warpSize` 在 allreduce .cu 中使用，但实际代码搜索未找到这些 CUDA 内建的调用。它们可能已被重构移除。本文档已修正此错误。

**新增关键发现**：`__launch_bounds__`（如 `__launch_bounds__(1024, 1)`）在 54+ kernel 上使用，控制每个线程的寄存器分配和最大线程数。Ascend C 无此概念，kernel 并行度由不同的机制控制。

### 1.5 PTX 内联汇编（迁移难度：极高） **←v1完全缺失**

这是 MSCCL++ 零拷贝通信的核心机制，也是迁移最大的技术难点：

#### MemoryChannel Semaphore PTX

```asm
// signal() — 远端原子加（GPU→远端GPU内存原子操作）
red.release.sys.global.add.u64 [remoteInboundToken], 1

// relaxed signal()
red.relaxed.sys.global.add.u64 [remoteInboundToken], 1
```

**灵衢替代**：SyncCollectives `SetOuterFlag(magic, eventID)` 写远端同步标志，或 `peerMems[]` 直接原子写入。

#### FIFO Push PTX（PortChannel GPU→Host）

```asm
// FifoDeviceHandle::push() — release 语义存储到 host-pinned 内存
st.global.release.sys.v2.u64 [triggerPtr], {fst, snd^flipMask}

// A100 特殊路径（__CUDA_ARCH__ == 800）
__threadfence_system() + relaxed store（A100 上 release 更慢）
```

**灵衢替代**：OBMM shared memory + URMA write，或通过 URPC func_call 替代 FIFO 语义。

#### SwitchChannel Multimem PTX（NVLS 专用，H100+）

```asm
// reduce — 从所有 NPU 的 multicast 内存原子读取并求和
multimem.ld_reduce.relaxed.sys.global.add.f32 %val, [mcPtr]
multimem.ld_reduce.relaxed.sys.global.add.v4.f32 {%v0,%v1,%v2,%v3}, [mcPtr]

// broadcast — 写入所有 NPU 的 multicast 内存
multimem.st.relaxed.sys.global.f32 [mcPtr], %val
multimem.st.relaxed.sys.global.v4.f32 [mcPtr], {%v0,%v1,%v2,%v3}

// reduce-add — 向所有 NPU 的 multicast 内存原子加
multimem.red.relaxed.sys.global.add.v4.f32 [mcPtr], {%v0,%v1,%v2,%v3}
```

支持类型：f32x1/x2/x4, i32x1, u32x1, f16x2/x4/x8, bf16x2/x4/x8, f8_e4m3x4/x8/x16, f8_e5m2x4/x8/x16

**灵衢替代**：OBMM 共享内存 + peerMems[] + SyncCollectives 组合机制。

#### Packet Protocol PTX

```asm
// LL16Packet volatile load/store
st.volatile.global.v4.u32 [addr], {v0,v1,v2,v3}
ld.volatile.global.v4.u32 {v0,v1,v2,v3}, [addr]
```

### 1.6 CUDA C++ Atomics 完整使用 **←v1未详细列出**

MSCCL++ 使用 `cuda::atomic_ref<T, Scope>` 的三种 scope：

| scope | 用途 | 替代 |
|-------|------|------|
| `thread_scope_device` | FIFO head、expectedInboundToken、DeviceSyncer/DeviceSemaphore | Ascend C atomic（同 NPU 内） |
| `thread_scope_system` | inboundToken、flushDonePos、tailCache（跨 NPU/跨 host-GPU） | SyncCollectives 或 Ascend C 跨 NPU atomic |
| `memory_order` | acquire/release/relaxed | Ascend C 内存序对应 |

### 1.7 NCCL Shim（迁移难度：低）

| API | HCCL 对应 | 备注 |
|-----|-----------|------|
| ncclGetUniqueId | HcclGetRootInfo / HcclCommInitClusterInfo | RootInfo 而非 UniqueId |
| ncclCommInitRank | HcclCommInitRootInfo | 参数结构不同 |
| ncclAllReduce | HcclAllReduce | dataType/reduceOp 不同 |
| ncclReduce | HcclReduce | 对应 |
| ncclBroadcast | HcclBroadcast | 对应 |
| ncclAllGather | HcclAllGather | 对应 |
| ncclReduceScatter | HcclReduceScatter | 对应 |
| ncclSend | HcclSend | 对应 |
| ncclRecv | HcclRecv | 对应 |
| ncclCommDestroy | HcclCommDestroy | 对应 |
| ncclGroupStart/End | **无对应** | HCCL 用 BatchSendRecv 替代 |

核心文件：`src/ext/nccl/nccl.cc`, `include/mscclpp/ext/nccl/nccl.h`

### 1.8 Python 层 CUDA使用 **←v1完全缺失**

#### Python 绑定（python/csrc/）

| 文件 | CUDA 内容 | 替代方案 |
|------|-----------|---------|
| executor_py.cpp | `(cudaStream_t)stream` cast | `(aclrtStream)stream` |
| algorithm.cpp | `reinterpret_cast<cudaStream_t>(stream)`, `CommUnhandledCudaError` | aclrtStream + ACL 错误枚举 |
| switch_channel_py.cpp | `NvlsConnection` class binding, `connectNvlsCollective` binding | OBMM shared memory binding |
| gpu_utils_py.cpp | `kDLCUDA` (DLDeviceType) | `kDLCANN` 或新增设备类型 |
| env_py.cpp | `cuda_ipc_use_default_stream` property | OBMM 配置参数 |
| CMakeLists.txt | `${GPU_LIBRARIES}`, `${GPU_INCLUDE_DIRS}` linking | CANN 库/头文件路径 |

#### Python _core/ 模块

| 文件 | CUDA 内容 | 替代方案 |
|------|-----------|---------|
| compiler.py | nvcc/hipcc 检测, CUDA_HOME, CUDA 编译管线 | Ascend C 编译器检测 + CANN_HOME |
| buffer.py | `cp.cuda.MemoryPointer` / `UnownedMemory` | CANN 内存管理 Python 接口 |
| algorithm.py | CUDA stream 引用, native CUDA 算法 | aclrtStream + HCCL 算法 |

### 1.9 构建系统 CUDA 配置

| 配置项 | 说明 | 位置 |
|--------|------|------|
| `MSCCLPP_USE_CUDA` option | CUDA 开关 | 根 CMakeLists.txt |
| `MSCCLPP_USE_CANN` option | **已存在** | 根 CMakeLists.txt ←v1未记录 |
| `find_package(CUDAToolkit)` | 查找 CUDA | 根 CMakeLists.txt, CheckNvidiaGpu.cmake |
| `enable_language(CUDA)` | 启用 CUDA 语言 | 根 CMakeLists.txt, CheckNvidiaGpu.cmake |
| `CMAKE_CUDA_STANDARD 17` | CUDA C++17 标准 | 根 CMakeLists.txt |
| `CMAKE_CUDA_ARCHITECTURES` | GPU 架构选择 | 根 CMakeLists.txt（sm_80/90/100/120） |
| `GPU_LIBRARIES` | CUDA::cudart + CUDA::cuda_driver [+ CCCL] | 根 CMakeLists.txt |
| `GPU_INCLUDE_DIRS` | CUDA 头文件路径 | 根 CMakeLists.txt |
| `MSCCLPP_USE_CUDA` 宏 | 编译定义 | src/core, ext/collectives, ext/nccl CMakeLists |
| nvcc 编译器 | 各 Makefile, Python JIT | examples Makefiles, python compiler.py |
| Docker CUDA 基础镜像 | cuda11.8 ~ cuda13.0 | docker/ |
| cupy-cuda11x/12x/13x | Python CUDA 包 | pyproject.toml |

---

## 2. MSCCL++ Channel 架构详细分析 **←v1严重不足**

### 2.1 MemoryChannel（零拷贝直接访问）

**核心原理**：GPU kernel 通过 IPC-mapped 指针直接读写远端 GPU 内存，使用 PTX 原子操作进行信号同步。**无 host 参与**。

**Host 侧 CUDA API**：

| 步骤 | API | 替代 |
|------|-----|------|
| 注册本地内存 | cudaIpcGetMemHandle → exchange → cudaIpcOpenMemHandle | obmem_export_memory → exchange → obmem_import_memory |
| 或 PosixFd/Fabric 路径 | cuMemCreate → cuMemExportToShareableHandle → cuMemImportFromShareableHandle → cuMemAddressReserve → cuMemMap → cuMemSetAccess | OBMM export/import 替代 |
| 分配 semaphore | cudaMalloc (inboundToken, remoteInboundToken, expectedInboundToken) | aclrtMalloc |
| P2P 访问 | cudaDeviceEnablePeerAccess | 需确认 CANN peer access |

**Device 侧机制**：

| 操作 | CUDA 实现 | 灵衢替代 |
|------|-----------|---------|
| signal() | PTX `red.release.sys.global.add.u64 [remoteInboundToken], 1` | SyncCollectives `SetOuterFlag(magic, eventID)` |
| wait() | `atomicLoad<scopeSystem, acquire>(inboundToken)` | SyncCollectives `WaitOuterFlag(magic, eventID)` |
| read() | 直接通过 IPC-mapped 指针 `*(dst_ + index)` | `peerMems[rank]` 直接访问 |
| write() | 直接通过 IPC-mapped 指针 `*(dst_ + index) = val` | `peerMems[rank]` 直接写入 |
| put/get | `copy()` 函数（多线程 stride copy，register staging） | Ascend C vector copy |

**关键数据结构**（v1 缺失）：

```cpp
// memory_channel_device.hpp
struct MemoryChannelDeviceHandle {
  MemoryDevice2DeviceSemaphoreDeviceHandle semaphore_;
  void* dst_;            // IPC-mapped 远端内存指针
  void* src_;            // 本地内存指针
  void* packetBuffer_;   // LL16Packet buffer
};
```

### 2.2 PortChannel（FIFO + Proxy 线程）

**核心原理**：GPU kernel 向 FIFO 推送 ProxyTrigger，host proxy 线程读取并执行 RDMA/cudaMemcpy 数据传输，然后通过 semaphore 信号通知 GPU。

**Host 侧完整架构**（v1 缺失）：

```
ProxyService {
  Fifo(512 entries)        — GPU 写 host-pinned 内存, host 读
  ProxyThread              — 处理 trigger, 执行数据传输
  gpuFlushDonePos          — host-pinned uint64_t, GPU spin-wait
  多个 Semaphore + Memory  — ProxyService.buildAndAddSemaphore/addMemory
}
```

**ProxyTrigger 结构**（128-bit union，v1 缺失）：

```cpp
union alignas(16) ProxyTrigger {
  uint64_t fst, snd;
  struct {
    uint32_t size, srcOffset, dstOffset;
    uint9 srcMemoryId, dstMemoryId;
    uint3 type;              // TriggerData | TriggerFlag | TriggerSync
    uint10 semaphoreId;
    uint1 reserved;
  } fields;
};
```

**FIFO 机制细节**（v1 缺失）：
- GPU 写：`atomicFetchAdd<scopeDevice>(head, 1)` → `st.global.release.sys.v2.u64`（PTX release 存储）
- Host 读：`atomicLoad<acquire>(fst)` → 读取 snd → 清零 fst → `atomicStore<release>(tail+1)`
- bit-63 flip 协议：GPU 写 `snd ^ (1<<63)`，host 通过 flip bit 判断新 entry
- FIFO 满等待：GPU 检查 `prevHead >= size + *tailCache`，同步等待 host pop

**Flush 机制**（v1 缺失）：
- GPU 推送 TriggerSync → host 请求 connection flush → completion queue drain → host 写 `flushDonePos = pos+1`
- GPU spin-wait：`atomicLoad<scopeSystem, acquire>(flushDonePos) > fifoPos`
- `flushDonePos` 是 host-pinned 内存（`gpuCallocHostShared<uint64_t>`）

**三种 Trigger 类型**：
1. **TriggerData** — proxy 执行 conn.write()（RDMA write 或 CudaIpc memcpy）
2. **TriggerFlag** — proxy 执行 semaphore->signal()（RDMA atomic CAS 或 cudaMemcpy H2D）
3. **TriggerSync** — proxy 请求 conn flush + 记录 FIFO position

**IB HostNoAtomic 模式**（v1 缺失）：
- recv 线程通过 GDRCopy BAR1 映射写 GPU 内存（非 cudaMemcpy）
- 32-bit `imm_data` from WRITE_WITH_IMM + wrap-around 检测重建 64-bit token

**灵衢替代方案**：
- FIFO → URPC func_call 或 OBMM shared memory ring buffer
- ProxyThread → 可保留（改为 URMA write + OBMM 操作），或改为 NPU kernel 自主推进
- Flush → URMA completion polling 或 SyncCollectives 等待
- Semaphore → SyncCollectives SetOuterFlag/WaitOuterFlag

### 2.3 SwitchChannel（NVLS Multicast）

**核心原理**：使用 NVIDIA NVLink SHARP (NVLS) hardware multicast，GPU kernel 通过 `multimem` PTX 指令原子读/写所有参与 GPU 的共享内存。**无显式同步**——NVLS hardware 提供一致性。

**Host 侧 CUDA API**（全部 Driver API）：

| 步骤 | API | 灵衢替代 |
|------|-----|---------|
| 创建 multicast | cuMulticastCreate | obmem_export_memory（NPU A 导出） |
| 添加设备 | cuMulticastAddDevice | obmem_import_memory（其他 NPU 导入） |
| 绑定地址 | cuMulticastBindAddr | peerMems[] + windowsIn/Out |
| 导出/导入 handle | cuMemExportToShareableHandle / cuMemImportFromShareableHandle | OBMM token_id + uba |
| 获取粒度 | cuMulticastGetGranularity | OBMM 4MB 对齐（硬编码 4MB） |

**Device 侧 PTX 指令**（v1 缺失完整列表）：

| 操作 | PTX 指令 | 灵衢替代 |
|------|---------|---------|
| reduce（读所有 NPU + 求和） | `multimem.ld_reduce.relaxed.sys.global.add.{f32,f32x4,f16x2,...}` | OBMM 共享内存读 + Ascend C reduce |
| broadcast（写所有 NPU） | `multimem.st.relaxed.sys.global.{f32,f32x4,...}` | peerMems[] 直接写 + SyncCollectives 通知 |
| reduce-add（向所有 NPU 原子加） | `multimem.red.relaxed.sys.global.add.{v4.f32,...}` | Ascend C atomic + SyncCollectives |

**灵衢替代方案确定**：

```
CUDA NVLS 流程：
1. cuMulticastCreate → 创建 multicast 内存
2. cuMulticastAddDevice → 多 GPU 加入
3. cuMulticastBindAddr → 绑定各 GPU 缓冲区地址
4. GPU kernel 通过 multimem PTX 指令原子读写

灵衢替代流程：
1. obmem_export_memory → NPU A 导出共享内存（获得 token_id + uba）
   **注意：len 必须是 4MB 整数倍！**
2. obmem_import_memory → 其他 NPU 导入（获得 mmap'd VA）
   **注意：import_cna 和 export_cna 必须不同！**
3. peerMems[] → 各 NPU kernel 通过 peerMems 直接读写共享区域
4. SyncCollectives → SetOuterFlag/WaitOuterFlag 进行同步
```

**NVLS 可用性检查**：`CUDA_NVLS_API_AVAILABLE` 宏（gpu.hpp:534-543）检查 CUDART_VERSION >= 12030 + Linux kernel >= 5.6.0。灵衢替代：检查 OBMM API 可用性。

### 2.4 Execution Kernel（通用 kernel 框架） **←v1完全缺失**

**核心文件**：`src/core/include/execution_kernel.hpp`

这是 MSCCL++ 的通用 collective algorithm 执行引擎，一个 `__global__` kernel 处理所有 channel 操作：

```cpp
template <typename T, typename PacketType = LL16Packet, bool ReuseScratch = false>
__global__ __launch_bounds__(1024, 1)
void executionKernel(int rank, T* input, T* output, T* scratch,
                     uint32_t scratchOffset, uint32_t scratchChunkSize,
                     DeviceExecutionPlan* plan, DeviceSemaphore* semaphores,
                     uint32_t localMemoryIdBegin, uint32_t flag) {
  extern __shared__ int4 sharedMem[];
  // 将 plan 复制到 shared memory
  DeviceExecutionPlan* localPlan = plan + blockIdx.x;
  for (size_t i = tid; i < sizeof(DeviceExecutionPlan)/sizeof(int4); i += blockDim.x)
    sharedMem[i] = ((int4*)localPlan)[i];
  __syncshm();
  
  // 加载 channel 指针到 shared memory
  memoryChannels_ = localPlan->channels.memoryChannels;
  portChannels_ = localPlan->channels.portChannels;
  nvlsChannels_ = localPlan->channels.nvlsChannels;
  // ... buffers, offsets
  
  // 顺序执行 operations
  for (int i = 0; i < nOperations; ) {
    executeDeviceFunction<T, PacketType, ReuseScratch>(op, input, output, scratch, &nSteps);
    i += nSteps;
  }
}
```

**14 个 `__shared__` 变量**（v1 缺失）：

```cpp
static __shared__ MemoryChannelDeviceHandle* memoryChannels_;
static __shared__ PortChannelDeviceHandle* portChannels_;
static __shared__ SwitchChannelDeviceHandle* nvlsChannels_;
static __shared__ T* buffers_[MAX_BUFFERS];
static __shared__ uint32_t offsets_[MAX_BUFFERS];
static __shared__ DeviceSyncer devSyncer_;
static __shared__ DeviceSemaphore devSemaphore_;
// ... 其他 channel/buffer 指针
```

**Operation 类型系统**（v1 缺失）：

| OperationType | 使用的 Channel | 灵衢替代 |
|---------------|---------------|---------|
| PUT, GET | MemoryChannel 或 PortChannel | OBMM + peerMems[] 或 URPC |
| PUT_WITH_SIGNAL | PortChannel (FIFO push + signal) | OBMM + SyncCollectives |
| PUT_WITH_SIGNAL_AND_FLUSH | PortChannel (FIFO + signal + flush) | OBMM + SyncCollectives + URMA poll |
| SIGNAL, WAIT | MemoryChannel 或 PortChannel semaphore | SyncCollectives |
| RELAXED_SIGNAL, RELAXED_WAIT | MemoryChannel semaphore | SyncCollectives (非阻塞版) |
| FLUSH | PortChannel flush | URMA completion poll |
| MULTI_LOAD_REDUCE_STORE | SwitchChannel (NVLS multimem) | OBMM + peerMems[] + reduce |
| BARRIER | DeviceSyncer (cross-block barrier) | Ascend C 同步 |
| SEM_ACQUIRE, SEM_RELEASE | DeviceSemaphore (inter-kernel) | Ascend C 信号量 |
| PIPELINE | 嵌套迭代 | Ascend C 循环 |

---

## 3. HCCL 源码完整分析（D:\C++\hccl）

### 3.1 HCCL 目录结构

```
hccl/
├── include/
│   ├── hccl.h              # 主 API（集合通信 + P2P）
│   ├── hccl_mc2.h          # KFC（Kernel Function Call）API ←v1缺失
├── src/
│   ├── common/
│   │   ├── adapter_acl.h/cc  # ACL runtime 适配层 ←v1缺失详细内容
│   │   ├── hccl_common.h     # 核心类型定义
│   │   ├── hcomm_dlsym/      # 动态符号加载（26 个 dlsym 文件）
│   ├── ops/                    # 集合通信算子
│   │   ├── op_common/
│   │   │   ├── template/
│   │   │   │   ├── ccu/       # CCU 微码生成 ←v1缺失详细内容
│   │   │   │   ├── aiv/       # AIV kernel ←v1缺失详细内容
│   │   │   │   ├── aicpu/
│   │   │   │   ├── dpu/       # 跨节点 ←v1缺失详细内容
│   │   │   ├── selector/      # 算法自动选择
│   │   │   ├── executor/      # 执行引擎（channel 管理）
│   ├── common/hccl_mc2.cc    # KFC API 实现
├── examples/
│   ├── 04_custom_ops_p2p/    # 含 Ascend C kernel
│   ├── 05_custom_ops_allgather/ # 含 Ascend C kernel
```

### 3.2 HCCL 公共 API 映射表

（同 v1，无遗漏）

### 3.3 HCCL 使用的 CANN Runtime API 完整列表 **←v1仅列 16 个，实际远多于此**

| ACL API | 用途 | HCCL 中的位置 | 备注 |
|---------|------|--------------|------|
| aclrtGetDevice | 获取当前设备 | 所有算子入口 | ✓ |
| aclrtSetDevice | 设置设备 | kernel 注册 | ✓ |
| aclrtMalloc | 分配设备内存 | graph 模式、examples | ✓ |
| aclrtMallocHost | 分配 host 内存 | scatter 等 | ✓ |
| aclrtFreeHost | 释放 host 内存 | scatter 等 | ✓ |
| aclrtFree | 释放设备内存 | examples | ✓ |
| aclrtMemcpy | 内存拷贝 | adapter_acl.cc | ✓ |
| aclrtMemset | 内存设置 | adapter_acl.cc | **←v1缺失** |
| aclrtCreateStream | 创建流 | examples/tests | ✓ |
| aclrtSynchronizeStream | 同步流 | examples/tests | ✓ |
| aclrtCreateNotify | 创建通知 | scatter_op.cc | ✓ 但 v1缺 flags 参数 |
| aclrtRecordNotify | 记录通知 | scatter_op.cc | ✓ |
| aclrtWaitAndResetNotify | 等待+重置通知 | scatter_op.cc | ✓ 但 v1缺 timeout 参数 |
| aclrtDestroyNotify | 销毁通知 | - | 需确认 |
| aclrtBinaryLoadFromFile | 加载 kernel 二进制 | adapter_acl.cc | ✓ 但 v1缺 load options |
| aclrtBinaryLoadFromData | 从内存加载 kernel | hccl_aiv_utils.cc | ✓ 但 v1缺嵌入方式 |
| aclrtBinaryUnLoad | 卸载 kernel 二进制 | hccl_aiv_utils.cc | **←v1缺失** |
| aclrtBinaryGetFunction | 获取 kernel 函数 | op_common、AIV utils | ✓ |
| aclrtLaunchKernelWithConfig | 启动 kernel（AICPU） | op_common | ✓ |
| aclrtLaunchKernelWithHostArgs | 启动 kernel（AIV） | AIV utils | ✓ 但 v1缺完整参数 |
| aclrtGetDeviceInfo | 查询设备属性 | adapter_acl.cc | ✓ |
| aclrtGetDevicesTopo | 获取拓扑 | adapter_acl.cc | ✓ 但 v1缺 link type 详情 |
| aclrtGetLogicDevIdByPhyDevId | 物理→逻辑设备 ID | adapter_acl.cc | **←v1缺失** |
| aclrtGetOpTimeOutInterval | 获取超时间隔 | hccl_aiv_utils.cc | **←v1缺失** |
| aclmdlRICaptureGetInfo | stream capture 状态查询 | adapter_acl.cc | **←v1缺失** |
| aclmdlRICaptureThreadExchangeMode | capture 线程交换模式 | adapter_acl.cc | **←v1缺失** |

### 3.4 HCCL IPC 机制完整详情 **←v1仅简要列出**

**HCCL 不使用 `aclrtMemExportToShareableHandle/ImportFromShareableHandle`**。

IPC 通过 HCCL 自有 dlsym 层提供，完整 API 签名：

```cpp
// 获取远端 HCCL 缓冲区
HcclResult HcclGetRemoteIpcHcclBuf(HcclComm comm, uint64_t remoteRank, void** addr, uint64_t* size);

// 通过 channel 访问远端内存
HcclResult HcclChannelGetRemoteMems(HcclComm comm, ChannelHandle channel,
    uint32_t* memNum, CommMem** remoteMems, char*** memTags);

// 注册内存供共享
HcclResult HcclCommMemReg(HcclComm comm, const char* memTag,
    const CommMem* mem, HcclMemHandle* memHandle);

// 获取设备内存 ←v1缺失
HcclResult HcclDevMemAcquire(HcclComm comm, const char* memTag,
    uint64_t* size, void** addr, bool* newCreated);
```

**新增类型定义**（v1 缺失）：

```cpp
enum CommMemType { COMM_MEM_DEVICE, COMM_MEM_HOST };
struct CommMem { CommMemType type; void* addr; uint64_t size; };
// EID addressing: COMM_ADDR_TYPE_EID, COMM_ADDR_EID_LEN = 36
```

**新增 API**（v1 缺失）：
- `HcclThreadExportToCommEngine` — 将线程 handle 导出到通信引擎
- `HcommSymWinGetPeerPointer` — 从对称窗口获取 peer 指针

### 3.5 HCCL 多引擎执行架构

（同 v1，补充以下细节）

#### AIV Kernel 完整管线 **←v1缺失**

1. **Binary Loading**（两种模式）：
   - **静态模式**：Binary `.o` 文件通过 `ld -r -b binary` 嵌入，使用 `_binary_*_bin_start/end` 符号
     `aclrtBinaryLoadFromData(embedded_start, length, &loadOptions{ACL_RT_BINARY_LOAD_OPT_LAZY_LOAD}, &binHandle)`
   - **动态模式**：从 `$ASCEND_HOME_PATH/lib64/<name>.o` 加载
     `aclrtBinaryLoadFromFile(path, &loadOptions, &binHandle)`

2. **Registration**（per device）：
   `aclrtBinaryGetFunction(binHandle, kernelName, &funcHandle)` → 存入 `(cmdType, dataType, argsType)` key 的 registry map

3. **Launch**（含 retry 机制）：
   ```cpp
   aclrtLaunchKernelWithHostArgs(funcHandle, numBlocks, stream,
       &cfg{SCHEM_MODE=1, TIMEOUT_US=calculated, ENGINE_TYPE=AIV},
       args, argsSize, nullptr, 0)
   // 如果返回 ACL_ERROR_RT_INVALID_HANDLE，重新获取 function 并 retry
   ```

4. **AivKernelArgsDef 完整结构**（v1 缺失）：
   ```cpp
   struct AivKernelArgsDef {
       const void* buffersIn;   // CCLIN 地址，所有 rank 可访问
       u64 input, output;
       u32 rank, sendRecvRemoteRank, rankSize;
       u64 xRankSize, yRankSize, zRankSize;
       u64 len;
       u32 dataType, reduceOp, root, tag;
       u64 inputSliceStride, outputSliceStride, repeatNum, inputRepeatStride, outputRepeatStride;
       bool isOpBase;
       const void* headCountMem, *tailCountMem, *addOneMem;
       u32 counterMemSize;
       bool isEnableCounter;
   };
   ```

5. **SuperKernel args**（融合 kernel，v1 缺失）：
   ```cpp
   #define SUPERKERNEL_ARGS_DEF GM_ADDR hiddenInput, GM_ADDR input, GM_ADDR output
   // hiddenInput → AivSuperKernelArgs（包含所有元数据）
   // 用于 "one-shot" 和 "two-shot" AllReduce 模式
   ```

6. **buffersIn/CCLIN 机制**（v1 缺失详细说明）：
   - `buffersIn` 是 CCLIN 注册地址，所有 rank 可访问
   - Layout: offset 0 → `GM_IN[i]`（各 rank input 地址），16KB → `GM_OUT[i]`（output 地址），32KB → TOPO info，40KB → tag/flag region

#### CCU Kernel 编程模型完整详情 **←v1缺失**

```cpp
// GroupOpSize — 处理非均匀数据大小
struct GroupOpSize {
    CcuRep::Variable addrOffset;     // 第二个 loopGroup 的偏移
    CcuRep::Variable loopParam;      // 串行重复计数
    CcuRep::Variable parallelParam;  // 并行扩展参数
    CcuRep::Variable residual;       // 尾部块大小
};

// GroupOpConfig
struct GroupOpConfig {
    uint32_t msInterleave;   // MS 步进
    uint32_t loopCount;      // 并行循环数
    uint64_t memSlice;       // 每循环字节数
};

// CCU 操作（完整签名）
GroupBroadcast(channels, dst_vec, src, goSize)
GroupBroadcastWithoutMyRank(channels, dst_vec, src, goSize)  // 跳过自身
GroupReduce(channels, dst, src_vec, goSize, dataType, outputDataType, reduceOp)
GroupReduceWithoutMyRank(channels, dst, src_vec, goSize, dataType, outputDataType, reduceOp)
GroupCopy(dst, src, goSize)  // 本地内存拷贝
GroupLocalReduce(outDst, scratch_vec, goSize, dataType, outputDataType, reduceOp)

// LoopGroup — 管线化多迭代并行
LoopGroup(loops_vec, loopCfg_var, paraCfg_var, offsetCfg_var)

// CcuRep 类型系统
LocalAddr, RemoteAddr, Variable, CompletedEvent, CcuBuf, Executor
```

#### DPU 跨节点通信完整详情 **←v1缺失**

```cpp
// 基础 API 签名
int32_t HcommSendRequest(uint64_t handle, const char* msgTag,
    const void* src, size_t sizeByte, uint32_t* msgId);
int32_t HcommWaitResponse(uint64_t handle, void* dst, size_t sizeByte, uint32_t* msgId);

// 批量传输 ←v1缺失
int32_t HcclHcommBatchTransferOnThread(ThreadHandle thread, ChannelHandle channel,
    const HcclHcommBatchTransferDesc *transferDescs, uint32_t transferDescNum);

// HcclHcommBatchTransferDesc 支持 9 种传输类型 ←v1缺失：
// WRITE, WRITE_REDUCE, WRITE_WITH_NOTIFY, WRITE_REDUCE_WITH_NOTIFY,
// READ, READ_REDUCE, NOTIFY_RECORD, NOTIFY_WAIT, NOTIFY_WAIT_WITH_DEFAULT_TIMEOUT

// 线程同步
int32_t HcommThreadSynchronize(ThreadHandle thread);
int32_t HcommThreadJoin(ThreadHandle thread, uint32_t timeout);
```

#### KFC API **←v1缺失**

```cpp
HcclKfcAllocOpArgs(void **opArgs)
HcclKfcFreeOpArgs(void *opArgs)
HcclKfcOpArgsSetSrcDataType(opArgs, srcDataType)
HcclKfcOpArgsSetDstDataType(opArgs, dstDataType)
HcclKfcOpArgsSetReduceType(opArgs, reduceType)
HcclKfcOpArgsSetCount(opArgs, count)
HcclKfcOpArgsSetAlgConfig(opArgs, algConfig)
HcclKfcOpArgsSetCommEngine(opArgs, commEngine)
HcclCreateOpResCtx(comm, opType, opArgs, &opResCtx)
```

### 3.6 HCCL 同节点通信协议 **←补充详情**

| 协议 | 值 | 说明 | 新增 |
|------|-----|------|------|
| HCCS | 默认 | 华为片间互联 | |
| HCCS_SW | - | HCCS through switch chip | **←v1缺失** |
| SIO | - | Die-to-die interconnect (Ascend 910_93) | **←v1缺失** |
| PXI | - | PCIe | **←v1缺失** |
| UBC_CTP | 4 | 统一总线 Cube CTP（Ascend 910B+） | |
| UBC_TP | 5 | 统一总线 TP（Ascend 910B+） | |
| UB_MEM | 6 | 统一总线共享内存（Ascend 910B+） | |
| RoCE | 1 | RDMA over Converged Ethernet（跨节点） | |

拓扑查询：`aclrtGetDevicesTopo(logicIdLocal, logicIdDest, &linkTypeRaw)` → 返回 HCCS/HCCS_SW/SIO/PXI

### 3.7 FP8 支持状态更新 **←v1标记"需确认"**

**已确认**：AIV copy 操作支持以下 FP8 类型：
- `fp8_e4m3fn_t`、`fp8_e5m2_t`、`fp8_e8m0_t`
- 另有 `hifloat8_t`

AIV atomic 操作支持的类型：float, half, int16_t, int32_t, int8_t, bfloat16_t, int64_t（FP8 不在 atomic 支持列表中）

---

## 4. 灵衢 UMDK 完整分析（D:\C++\umdk）

### 4.1 UMDK 架构（5 大组件）

| 组件 | 全称 | 作用 |
|------|------|------|
| **URMA** | Unified Remote Memory Access | 远端内存语义访问（RDMA 替代） |
| **CAM** | Communication Acceleration Module | MoE dispatch/combine + Ascend C kernel |
| **URPC** | Unified Remote Process Call | RPC 通信 + 消息队列（UMQ） |
| **ULOCK/DLock** | Distributed Lock | 分布式锁 + 跨 NPU 原子操作 |
| **USOCK/UMS** | UB Memory Socket | 基于 UB 的 socket 通信 |

### 4.2 OBMM 共享内存（零拷贝核心机制）完整详情 **←v1缺参数和对齐要求**

```cpp
// 导出参数
struct obmem_export_memory_param_t {
    uint64_t len;              // **必须 4MB 整数倍对齐！** ←v1缺失
    uint8_t deid[16];          // 目标 EID（16字节）
    bool cacheable;            // 缓存模式标志
};

// 导入参数
struct obmem_import_memory_param_t {
    uint16_t import_cna;       // 导入 CNA 号
    uint16_t export_cna;       // **必须与 import_cna 不同！** ←v1缺失
    uint8_t seid[16];          // 源 EID
    uint8_t deid[16];          // 目标 EID
    bool cacheable;
};

// 导出返回信息
struct obmem_export_info_t {
    uint32_t token_id;         // 导入时使用的 token
    uint64_t uba;              // UB 地址
    uint64_t size;             // 实际大小
};

// 释放
obmem_release_export_memory(handle, ptr, len)
obmem_release_import_memory(handle, ptr, len)
```

**底层机制**：OBMM 使用 `/dev/obmm_shmdev<memid>` 设备文件 + `mmap(MAP_SHARED)` 实现导出和导入。

**关键影响**：4MB 对齐要求意味着 MSCCL++ 的缓冲区分配策略必须调整——所有跨 NPU 共享的缓冲区大小必须是 4MB 的整数倍。

### 4.3 HCCL Window 零拷贝完整结构 **←v1仅列简化版**

```cpp
// Per-rank 结构 ←v1缺失
struct HcclRankRelationResV2 {
    uint32_t remoteUsrRankId;
    uint32_t remoteWorldRank;
    uint64_t windowsIn;       // Per-rank 远端数据窗口
    uint64_t windowsOut;      // Per-rank 远端输出窗口
    uint64_t windowsExp;      // Per-rank expert/status 窗口
    ListCommon nextTagRes;    // tag 资源链表
};

// 完整操作资源参数 ←v1仅列简化版
struct HcclOpResParam {
    HcclMC2WorkSpace mc2WorkSpace;
    uint32_t localUsrRankId, rankSize;
    uint64_t winSize, winExpSize;
    uint64_t localWindowsIn, localWindowsOut, localWindowsExp;  // ←v1缺失：本地窗口地址
    char hcomId[128];
    uint32_t rWinStart, rWinOffset;                            // ←v1缺失：窗口偏移参数
    uint64_t version;
    LocalResInfoV2 localRes;
    AlgoTopoInfo topoInfo;
    HcclOpConfig config;
    uint64_t hostStateInfo, aicpuStateInfo, lockAddr;           // ←v1缺失：同步状态
    uint32_t notifysize, remoteResNum;
    RemoteResPtr remoteRes[AICPU_MAX_RANK_NUM];                // ←v1缺失：远端资源数组
    HDCommunicateParams kfcControlTransferH2DParams;            // ←v1缺失：KFC host-device 通信
    HDCommunicateParams kfcStatusTransferD2HParams;
    uint64_t tinyMem, tinyMemSize;                              // ←v1缺失：AlltoAll 专用
    uint64_t zeroCopyHeadPtr, zeroCopyTailPtr;                  // ←v1缺失：环形缓冲区指针
    uint64_t zeroCopyRingBuffer;
    uint64_t zeroCopyIpcPtrs[16];
    uint32_t zeroCopyDevicePhyId[16];                           // ←v1缺失：物理设备 ID
    bool utraceStatusFlag;
};
```

**获取 Window Context 的方式**（v1 缺失）：
```cpp
// Ascend C kernel 内获取
auto winContext = AscendC::GetHcclContext<HCCL_GROUP_ID_0>();
```

### 4.4 peerMems[] — 直接跨 NPU 内存访问完整结构 **←v1缺 IPC layout 和完整结构**

```cpp
struct CommArgs {
    int rank = 0;                     // 全局 rank
    int localRank = -1;               // ←v1缺失：fullmesh 内的本地 rank
    int rankSize = 0;                 // 全局 rank size
    int localRankSize = -1;           // ←v1缺失：fullmesh 卡数
    uint32_t extraFlag = 0;           // ←v1缺失：32-bit bitmap 控制行为
    int testFlag = 0;
    GM_ADDR peerMems[CAM_MAX_RANK_SIZE] = {};  // 跨 NPU 缓冲区（CAM_MAX_RANK_SIZE = 384）
    int64_t sendCountMatrix[CAM_MAX_RANK_SIZE * CAM_MAX_RANK_SIZE] = {};  // ←v1缺失：AlltoAllVC
    int64_t sendCounts[CAM_MAX_RANK_SIZE] = {};   // ←v1缺失：AlltoAllV
    int64_t sdispls[CAM_MAX_RANK_SIZE] = {};
    int64_t recvCounts[CAM_MAX_RANK_SIZE] = {};
    int64_t rdispls[CAM_MAX_RANK_SIZE] = {};
    int64_t batchSize, hiddenSize, topk;            // ←v1缺失：MoE 参数
    int64_t sharedExpertRankNum, expertNumPerRank;
    int64_t dfx[50] = {};
};
```

**IPC 内存布局**（v1 缺失）：
- 每个 `peerMems[rank]` 的前 2MB 是 flag/sync 区域
- 然后是 100MB 数据区域（IPC_DATA_OFFSET = 2MB）

### 4.5 SyncCollectives 完整 API **←v1仅列 5 个方法，实际 20+**

```cpp
class SyncCollectives {
    void Init(int rank, int rankSize, GM_ADDR *shareAddrs, TBuf<QuePosition::VECCALC> &tBuf);
    
    // === 基本同步 ===
    void SetSyncFlag(int32_t magic, int32_t value, int32_t eventID);
    void SetSyncFlag(int32_t magic, int32_t value, int32_t eventID, int32_t rank);  // 指定 rank
    void WaitSyncFlag(int32_t magic, int32_t value, int32_t eventID);
    void WaitSyncFlag(int32_t magic, int32_t value, int32_t eventID, int32_t rank);
    void WaitSyncFlag(int32_t magic, int32_t value, int32_t eventID, int32_t rank, int64_t flagNum);  // 范围
    
    // === 同卡同步（Memory A）=== ←v1缺失整类
    void SetInnerFlag(int32_t magic, int32_t eventID);
    void SetInnerFlag(int32_t magic, int32_t eventID, int64_t setRank, int64_t setBlock);
    void WaitInnerFlag(int32_t magic, int32_t eventID, int64_t waitRank, int64_t waitBlock);
    void WaitRankInnerFlag(int32_t magic, int32_t eventID, int64_t waitRank);  // ALL blocks
    bool CheckRankInnerFlag(int32_t magic, int32_t eventID, int64_t waitRank); // 非阻塞检查
    
    // === 跨卡同步（Memory B）=== ←v1缺失整类
    void SetOuterFlag(int32_t magic, int32_t eventID);
    void SetOuterFlag(int32_t magic, int32_t eventID, int64_t setRank, int64_t setBlock);
    void WaitOuterFlag(int32_t magic, int32_t eventID, int64_t waitRank, int64_t waitBlock);
    void WaitOneRankOuterFlag(int32_t magic, int32_t eventID, int64_t rank);
    void WaitAllRankPartOuterFlag(int32_t magic, int32_t eventID, int64_t startBlock, int64_t flagNum);
    void WaitAllRankOuterFlag(int32_t magic, int32_t eventID);
    bool CheckAllRankPartOuterFlag(...);   // 非阻塞 ←v1缺失
    bool CheckAllRankOuterFlag(...);       // 非阻塞 ←v1缺失
    
    // === Chunk 级同步（MoE 专用）=== ←v1缺失
    void SetChunkFlag(int64_t rank, int64_t destRank, int64_t magic, int64_t eventId);
    int64_t GetChunkFlag(int64_t rank, int64_t destRank, int64_t magic, int64_t timeout);
    int64_t GetChunkRecvLen(...);
    
    // === 读取 flag 值 === ←v1缺失
    int64_t GetFlag(__gm__ int64_t* waitAddr);
    int64_t GetInnerFlag(int64_t waitRank, int64_t waitBlock);
    int64_t GetOuterFlag(int64_t waitRank, int64_t waitBlock);
};
```

**关键机制**（v1 缺失）：

1. **magic/value 合并**：`MergeMagicWithValue(magic, value) = (magic << 32) | value`
   - 高 32 位 = operator batch number（防过期标志匹配）
   - 低 32 位 = specific value

2. **Memory A vs Memory B**：
   - Inner flags → Memory A（同 rank 内，跨 block 同步）= MSCCL++ DeviceSyncer 替代
   - Outer flags → Memory B（跨 rank 同步）= MSCCL++ MemoryDevice2DeviceSemaphore 替代

3. **非阻塞 vs 阻塞**：
   - `Check*` → 非阻塞（返回 bool，适合条件判断）
   - `Wait*` → 阻塞轮询（适合同步等待）

4. **FLAG_UNIT_INT_NUM = 4**（128 bytes per sync unit）

5. **shareAddrs[]** 与 `peerMems[]` 不同 — shareAddrs 专门用于同步标志，peerMems 用于数据缓冲区

### 4.6 URMA 完整 API **←v1仅列 3 个，实际 40+**

```cpp
// === 初始化 ===
urma_init(conf) / urma_uninit()
urma_create_context(dev, eid_index) / urma_delete_context(ctx)

// === 单边操作 === ←v1部分列出
urma_write(jfs, target_jfr, dst_tseg, src_tseg, dst, src, len, flag, user_ctx)
urma_read(jfs, target_jfr, dst_tseg, src_tseg, dst, src, len, flag, user_ctx)

// === 双边操作 === ←v1缺失
urma_send(jfs, target_jfr, src_tseg, src, len, flag, user_ctx)
urma_recv(jfr, recv_tseg, buf, len, user_ctx)

// === 批量提交 === ←v1部分列出
urma_post_jfs_wr(jfs, wr, bad_wr)  // 支持 WRITE/READ/CAS/FAA/SEND

// === 内存段管理 === ←v1缺失
urma_register_seg(ctx, seg_cfg) → urma_target_seg_t*    // 注册本地内存段
urma_unregister_seg(target_seg)                           // 取消注册
urma_import_seg(ctx, seg, token_value, addr, flag) → urma_target_seg_t*  // 导入远端段
urma_unimport_seg(tseg)                                   // 取消导入

// === Jetty/JFR 管理 === ←v1缺失
urma_import_jfr(ctx, rjfr, token_value) → urma_target_jetty_t*
urma_import_jetty(ctx, rjetty, token_value) → urma_target_jetty_t*
urma_bind_jetty(jetty, tjetty) / urma_unbind_jetty(jetty)
urma_advise_jfr(jfs, tjfr) / urma_unadvise_jfr(jfs, tjfr)  // 建立/拆除传输通道

// === Jetty Group === ←v1部分列出
urma_create_jetty_grp(ctx, cfg) → urma_jetty_grp_t*  // 最多 max_jetty_in_jetty_grp 个 Jetty
urma_delete_jetty_grp(jetty_grp)
// 策略：URMA_JETTY_GRP_POLICY_RR / URMA_JETTY_GRP_POLICY_HASH_HINT

// === 完成轮询 === ←v1缺失
urma_poll_jfc(jfc, cr_cnt, cr)  // Completion polling

// === 资源创建/销毁 === ←v1缺失
urma_create_jfc/jfs/jfr/jetty / urma_delete_jfc/jfs/jfr/jetty

// === Token 管理 === ←v1缺失
urma_alloc_token_id(ctx) / urma_free_token_id(token_id)

// === 用户控制 === ←v1缺失
urma_user_ctl(ctx, in, out)  // UB 硬件专用用户定义控制
```

### 4.7 DLock 分布式锁 **←v1完全缺失**

```cpp
// 初始化
int dclient_lib_init(const struct client_cfg *cfg);
void dclient_lib_deinit();

// 客户端管理
int client_init(int *client_id, const char *ip_str);
int client_deinit(int client_id);
int client_heartbeat(int client_id, unsigned int timeout);

// 锁操作
int get_lock(client_id, const struct lock_desc *desc, int *lock_id);
int release_lock(client_id, int lock_id);
int trylock(client_id, const struct lock_request *req, void *result);   // 非阻塞
int lock(client_id, const struct lock_request *req, void *result);      // 阻塞
int unlock(client_id, int lock_id, void *result);

// 批量操作（最多 31 个锁）
int batch_get_lock/batch_release_lock/batch_trylock/batch_unlock(...)

// 分布式原子对象 ←v1缺失，对 MSCCL++ 有潜在价值
int umo_atomic64_create(client_id, desc, init_val, obj_id);
int umo_atomic64_destroy(client_id, obj_id);
int umo_atomic64_faa(client_id, obj_id, add_val, res_val);   // Fetch-and-Add
int umo_atomic64_cas(client_id, obj_id, cmp_val, swap_val);  // Compare-and-Swap
int umo_atomic64_get(client_id, desc, obj_id);
```

**MSCCL++ 相关性**：DLock 的 `umo_atomic64_faa/cas` 提供跨 NPU 原子操作，可能替代部分 PTX `red.release.sys` 语义。

### 4.8 URPC RPC 框架 **←v1完全缺失**

```cpp
// 初始化
int urpc_init(urpc_config_t *cfg);
void urpc_uninit();

// Channel
uint32_t urpc_channel_create();
int urpc_channel_destroy/attach/detach/refresh(...);

// Queue
uint64_t urpc_queue_create(trans_mode, cfg);
int urpc_queue_destroy/add/rm/pair/unpair(...);

// 函数注册
int urpc_func_register(info, func_id);
int urpc_func_unregister(func_id);

// 数据路径 ←核心
uint64_t urpc_func_call(chid, wr, option);           // RPC 调用
int urpc_func_poll(chid, option, msg[], max_msg_num); // 结果轮询
int urpc_func_return(qh, req_ctx, wr, option);        // 返回结果

// 内存
uint64_t urpc_mem_seg_register(va, len);
int urpc_mem_seg_remote_access_enable/disable(chid, mem_h);
```

**MSCCL++ 相关性**：URPC 可替代 PortChannel 的 FIFO + proxy 机制。`urpc_func_call` 对应 FIFO push，`urpc_func_poll` 对应 flush wait。

### 4.9 USOCK/UMS Socket over UB **←v1完全缺失**

提供 TCP-like socket 语义 over UB：
- `ums_connect/listen/bind/accept/close`
- `ums_tx/ums_rx` 数据传输
- CDC（Credit-based Data Channel）流控

**MSCCL++ 相关性**：控制平面通信（bootstrap 替代），不用于数据平面。

### 4.10 CAM MoE Dispatch/Combine **←v1完全缺失**

CAM 是 UMDK 中最接近 MSCCL++ 需求的参考实现：

```
notify_dispatch / notify_dispatch_a2   — token-based dispatch
moe_dispatch_normal / moe_dispatch_a2  — 正常 dispatch
moe_combine_normal / moe_combine_a2    — combine 操作
moe_dispatch_shmem / moe_combine_shmem — 共享内存变体
fused_deep_moe                         — GEMM + dispatch + combine 融合
```

CAM 使用完整组合：`HcclOpResParam` + `GetHcclContext<>()` + `peerMems[]` + `SyncCollectives`，是跨 NPU kernel 通信的完整参考。

---

## 5. CUDA/NVLS → 灵衢 UMDK 完整映射表

### 5.1 Runtime API 映射（完整版）

| CUDA Runtime | CANN (aclrt) | HCCL 使用 | 备注 |
|-------------|-------------|----------|------|
| cudaGetDevice | aclrtGetDevice | ✓ | 直接对应 |
| cudaSetDevice | aclrtSetDevice | ✓ | 直接对应 |
| cudaGetDeviceCount | aclrtGetDeviceCount | ✓ | 直接对应 |
| cudaDeviceSynchronize | aclrtSynchronize | - | 直接对应 |
| cudaGetDeviceProperties | aclrtGetDeviceInfo | ✓ | 参数结构不同 |
| cudaDeviceGetAttribute | aclrtGetDeviceInfo | ✓ | ←新增 |
| cudaDeviceGetPCIBusId | **需确认** | - | ←新增 |
| cudaDeviceCanAccessPeer | **需确认** | - | |
| cudaDeviceEnablePeerAccess | **需确认** | - | |
| cudaMalloc | aclrtMalloc | ✓ | 直接对应 |
| cudaFree | aclrtFree | ✓ | 直接对应 |
| cudaHostAlloc | aclrtMallocHost | ✓ | flags 不同 |
| cudaFreeHost | aclrtFreeHost | ✓ | 直接对应 |
| cudaMemcpy | aclrtMemcpy | ✓ | direction 枚举不同 |
| cudaMemcpyAsync | aclrtMemcpyAsync | - | 直接对应 |
| cudaMemcpyToSymbol | **无对应** | - | 需改为全局指针 + aclrtMemcpy 或 peerMems[] |
| cudaMemcpyToSymbolAsync | **无对应** | - | ←新增 |
| cudaMemset | aclrtMemset | ✓ | ←新增 |
| cudaMemsetAsync | aclrtMemsetAsync | ✓ | ←新增 |
| cudaStreamCreate | aclrtCreateStream | ✓ | 直接对应 |
| cudaStreamDestroy | aclrtDestroyStream | - | 直接对应 |
| cudaStreamSynchronize | aclrtSynchronizeStream | ✓ | 直接对应 |
| cudaStreamBeginCapture | aclmdlRICaptureGetInfo? | ✓ | ←新增 |
| cudaStreamEndCapture | 对应 | ✓ | ←新增 |
| cudaStreamIsCapturing | 对应 | ✓ | ←新增 |
| cudaThreadExchangeStreamCaptureMode | aclmdlRICaptureThreadExchangeMode | ✓ | ←新增 |
| cudaEventCreate | aclrtCreateNotify | ✓ | Notify 而非 Event |
| cudaEventRecord | aclrtRecordNotify | - | 对应 |
| cudaEventSynchronize | aclrtWaitAndResetNotify | - | 带 timeout |
| cudaEventDestroy | aclrtDestroyNotify | - | 对应 |
| cudaEventElapsedTime | **需确认** | - | ←新增 |
| cudaGraphCreate | aclrtGraphCreate? | - | ←新增 |
| cudaGraphInstantiate | aclrtGraphInstantiate? | - | ←新增 |
| cudaGraphLaunch | aclrtGraphLaunch? | - | ←新增 |
| cudaGraphDestroy | aclrtGraphDestroy? | - | ←新增 |
| cudaIpcGetMemHandle | obmem_export_memory | ✓ | OBMM 替代 |
| cudaIpcOpenMemHandle | obmem_import_memory | ✓ | OBMM 替代 |
| cudaIpcCloseMemHandle | obmem_release_import_memory | ✓ | OBMM 替代 |

### 5.2 Driver API 映射（完整版）

| CUDA Driver | CANN/UMDK 对应 | HCCL 使用 | 备注 |
|-------------|---------------|----------|------|
| cuMemCreate | **无直接对应** | - | 需用 aclrtMalloc 替代 |
| cuMemMap/cuMemUnmap | **无直接对应** | - | CANN 无虚拟内存管理 |
| cuMemAddressReserve/Free | **无直接对应** | - | CANN 无虚拟地址空间 |
| cuMemSetAccess | **无直接对应** | - | 需用 OBMM 替代 |
| cuMemGetAllocationGranularity | **无对应** | - | OBMM 4MB 硬编码 |
| cuMemGetAddressRange | **无对应** | - | 需自行管理地址范围 |
| cuMemRelease | **无对应** | - | |
| cuMemExportToShareableHandle | obmem_export_memory | ✓ | OBMM 替代 |
| cuMemImportFromShareableHandle | obmem_import_memory | ✓ | OBMM 替代 |
| cuMemRetainAllocationHandle | **无对应** | - | 需重新设计 |
| cuMemGetHandleForAddressRange | **无对应** | - | DMA_BUF 不适用 |
| cuMulticastCreate | OBMM export + import | ✓ | OBMM 共享内存替代 |
| cuMulticastAddDevice | obmem_import_memory | ✓ | 每个 NPU 导入 |
| cuMulticastBindAddr | peerMems[] + windowsIn/Out | ✓ | HCCL Window |
| cuMulticastUnbind | obmem_release_import_memory | ✓ | |
| cuMulticastGetGranularity | **4MB 硬编码** | ✓ | OBMM 4MB 对齐 |
| cuDeviceGet | aclrtGetDevice | ✓ | 逻辑对应 |
| cuDeviceGetAttribute | aclrtGetDeviceInfo | ✓ | 参数不同 |
| cuPointerGetAttribute | **无对应** | - | 需自行管理指针元数据 |
| cuGetErrorString | 自定义映射 | ✓ | |

### 5.3 PTX → 灵衢映射 **←v1完全缺失**

| PTX 指令 | MSCCL++ 用途 | 灵衢替代 |
|---------|-------------|---------|
| `red.release.sys.global.add.u64` | MemoryChannel semaphore signal | SyncCollectives SetOuterFlag |
| `red.relaxed.sys.global.add.u64` | MemoryChannel relaxedSignal | SyncCollectives SetOuterFlag |
| `st.global.release.sys.v2.u64` | FIFO push (GPU→Host) | OBMM + URPC func_call |
| `ld.volatile.global.v4.u32` | LL16Packet read | Ascend C volatile read |
| `st.volatile.global.v4.u32` | LL16Packet write | Ascend C volatile write |
| `multimem.ld_reduce.relaxed.sys.global.add.*` | SwitchChannel reduce | OBMM + peerMems[] + Ascend C reduce |
| `multimem.st.relaxed.sys.global.*` | SwitchChannel broadcast | peerMems[] 直接写 + SyncCollectives |
| `multimem.red.relaxed.sys.global.add.*` | SwitchChannel reduce-add | Ascend C atomic + SyncCollectives |
| `__threadfence_system()` | A100 FIFO 特殊路径 | Ascend C memory fence |

### 5.4 NCCL → HCCL API 映射

（同 v1，无遗漏）

### 5.5 CUDA Device Keywords → Ascend C 映射

| CUDA | Ascend C / HCCL | 备注 |
|------|-----------------|------|
| `__global__` | Ascend C kernel 入口 | 编程模型完全不同 |
| `__device__` | Ascend C device 函数 | 编程模型不同 |
| `__shared__` | Ascend C local memory | 对应但语义不同 |
| `__syncthreads()` | Ascend C 同步机制 | 不同 |
| `__launch_bounds__` | **无对应** | ←新增，Ascend C 无此概念 |
| blockIdx/threadIdx | Ascend C 6 维并行 | 完全不同 |
| `cuda::atomic<scopeDevice>` | Ascend C atomic (同NPU) | API 不同 |
| `cuda::atomic<scopeSystem>` | SyncCollectives 或 DLock atomic | ←补充 |
| `__device__ __assert_fail` | Ascend C assert | ←新增 |

---

## 6. 迁移难度评估（更新版）

| 难度等级 | 模块 | 说明 |
|----------|------|------|
| **低** | 设备管理 API | aclrt 有直接对应 |
| **低** | Stream/Event API | aclrt 有直接对应 |
| **低** | NCCL shim → HCCL | 接口名称替换 |
| **低** | 错误处理 | ACL_SUCCESS + 错误码映射 + MSCCLPP_ACLTHROW 宏 |
| **低** | cudaMemset/MemsetAsync | aclrtMemset/aclrtMemsetAsync |
| **中** | 内存管理（host↔device 拷贝） | cudaMemcpyToSymbol 需特殊处理 |
| **中** | 数据类型（FP16/BF16/FP8） | **FP8 已确认支持**（AIV copy 支持 fp8_e4m3/e5m2/e8m0） |
| **中** | 构建系统 | 需新增 CANN 编译链配置 |
| **中** | cuda::atomic | scopeDevice→Ascend C atomic, scopeSystem→SyncCollectives |
| **中** | CUDA Graph | CANN 有 capture mode API 但需验证 |
| **中** | Python bindings | 类型 cast + 设备类型 + 编译器检测 |
| **高** | IPC 内存机制 | OBMM 替代 CUDA IPC（4MB 对齐约束） |
| **高** | FIFO + Proxy (PortChannel) | 需 URPC/OBMM 替代或重新设计 |
| **高** | Device kernel（~70 unique kernel） | 需用 Ascend C 重写 |
| **高** | execution_kernel 框架 | Operation 类型系统 + 14 __shared__ 变量 |
| **极高** | PTX 内联汇编 | multimem/red.release/st.release 全部需替代 |
| **极高** | NVLS multicast / SwitchChannel | OBMM + peerMems[] + SyncCollectives 组合替代 |

---

## 7. 迁移策略决策（更新版）

### 7.1 核心架构选择

**推荐方案 C（混合方案）**，确认可行：

- **同节点**：保留 MSCCL++ 零拷贝设计
  - memory_channel → OBMM 共享内存 + peerMems[]
  - port_channel → **两种选择**：
    - A. 保留 proxy 模式（FIFO→URPC func_call, proxy→host thread + URMA write）
    - B. 改为 NPU kernel 自主推进（OBMM + peerMems[] + SyncCollectives）
  - switch_channel → OBMM 共享内存（多 NPU 导入同一 export）+ SyncCollectives
  - semaphore → SyncCollectives SetOuterFlag/WaitOuterFlag
- **跨节点**：URMA 单边操作（urma_write/read）

### 7.2 PortChannel 替代方案 **←v1缺失**

**选项 A：保留 proxy 模式（推荐初期）**
- FIFO → URPC func_call（GPU kernel → host proxy thread）
- Proxy data transfer → URMA write/read 或 OBMM shared memory
- Semaphore → SyncCollectives（proxy 写 NPU 同步标志）
- Flush → URMA completion polling
- 优势：改动最小，proxy thread 逻辑可保留
- 难点：URPC latency 可能高于 FIFO

**选项 B：NPU kernel 自主推进（推荐长期）**
- PortChannel 退化为 OBMM + peerMems[]（类似 MemoryChannel）
- 所有数据传输由 NPU kernel 通过 peerMems[] 直接完成
- 同步由 SyncCollectives 处理
- 优势：真正零拷贝，无 host 参与
- 难点：需重写更多 kernel，跨节点需 URMA from NPU kernel（需确认可行性）

### 7.3 cudaMemcpyToSymbol 替代方案 **←v1缺失**

**方案**：用 `CommArgs.peerMems[]` 传递参数到 kernel，替代 device 全局变量赋值。

CUDA 模式：
```cpp
__device__ int gConstMyRank;
cudaMemcpyToSymbol(gConstMyRank, &myRank, sizeof(int));
// kernel 中直接用 gConstMyRank
```

灵衢模式：
```cpp
// CommArgs 结构包含所有参数，kernel 通过 peerMems[0] 获取
struct CommArgs { int rank; int rankSize; GM_ADDR peerMems[384]; };
// kernel 入口直接接收 CommArgs 指针
```

### 7.4 OBMM 4MB 对齐影响 **←v1缺失**

所有跨 NPU 共享的缓冲区大小必须是 4MB 整数倍。MSCCL++ 当前缓冲区分配策略需调整：
- `gpuCalloc(size)` → 需改为 `gpuCallocAligned(size, 4MB)`
- 小缓冲区（如 semaphore token = 8 bytes）需合并到 4MB block 中
- 或改用 aclrtMalloc + peerMems[] offset 方式（不依赖 OBMM 共享小缓冲区）

### 7.5 文件组织策略

**推荐选项 A（条件编译）**，沿用 HIP 模式：
- 新增 `#if defined(MSCCLPP_DEVICE_CANN)` 分支
- CMake 新增 `MSCCLPP_USE_CANN` option（已存在）
- Ascend C kernel 文件（`.asc`）独立存放，但通过 CMake 条件编译集成

---

## 8. 更新后的迁移优先级（详细版）

| 优先级 | 阶段 | 内容 | 具体文件/改动 | 前置条件 |
|--------|------|------|-------------|----------|
| **P0** | 基础 API 适配 | gpu.hpp/gpu_utils 新增 CANN 分支 | gpu.hpp(类型映射已有桩), gpu_utils.cc(aclrt* API 替换), gpu_data_types.hpp(CANN 数据类型), assert_device.hpp, errors.cc(错误码映射) | 无 |
| **P0** | 错误处理宏 | MSCCLPP_CUDATHROW→MSCCLPP_ACLTHROW | gpu_utils.hpp:17,28 | 无 |
| **P0** | cudaMemset 映射 | gpu.hpp 新增 aclrtMemset | gpu.hpp:126,129, gpu_utils.cc(8处) | 无 |
| **P1** | NCCL→HCCL shim | src/ext/nccl 替换 | nccl.cc, nccl.h | P0 完成 |
| **P2** | 构建系统 | CMake 新增 CANN 编译链 | CMakeLists.txt, CheckCannGpu.cmake(新增), examples Makefiles, python/compiler.py | P0 完成 |
| **P2** | Python bindings | 类型 cast + 设备类型 | executor_py.cpp, algorithm.cpp, gpu_utils_py.cpp, switch_channel_py.cpp, env_py.cpp | P0 完成 |
| **P3** | IPC 内存 | CUDA IPC→OBMM 共享内存 | gpu_ipc_mem.cc(obmem_export/import 替代 cudaIpc*), 需处理 4MB 对齐 | P0-P2 完成 |
| **P3** | cudaMemcpyToSymbol | 改为 CommArgs 参数传递 | 所有 test .cu 文件(~54处), execution_kernel.cu | P0 完成 |
| **P4** | 同步机制 | PTX semaphore→SyncCollectives | semaphore_device.hpp(SetOuterFlag/WaitOuterFlag 替代 PTX red.release), fifo_device.hpp(URPC 替代 PTX st.release), concurrency_device.hpp(InnerFlag 替代 __syncthreads) | P3 完成 |
| **P5** | MemoryChannel kernel | Ascend C 重写（OBMM + peerMems[]） | memory_channel_device.hpp(read/write/put/get → Ascend C vector ops), copy_device.hpp → Ascend C copy | P3-P4 完成 |
| **P5** | PortChannel kernel | Ascend C 重写（两种方案选一） | port_channel_device.hpp, fifo_device.hpp → URPC func_call 或 OBMM + SyncCollectives | P3-P4 完成 |
| **P6** | SwitchChannel | OBMM + peerMems[] + SyncCollectives | switch_channel_device.hpp(multimem PTX → OBMM + peerMems + reduce), switch_channel.cc(cuMulticast* → OBMM) | P3-P4 完成 |
| **P6** | execution_kernel 框架 | Ascend C 重写（Operation 类型系统） | execution_kernel.hpp(14 __shared__ → Ascend C local memory, Operation dispatcher → Ascend C switch) | P5-P6 完成 |
| **P7** | allreduce/allgather | Ascend C 重写算法 kernel | allreduce_packet.cu, allreduce_nvls_*.cu, allreduce_fullmesh.cu, allgather_fullmesh.cu | P5-P6 完成 |
| **P8** | CUDA Graph | CANN Graph/capture API 适配 | common.cc, executor_test.cc, bidir_*_channel.cu, customized_allgather.cu, AvoidCudaGraphCaptureGuard → CANN guard | P0 完成 |

---

## 9. 待确认事项（更新版）

1. **灵衢超节点具体 NPU 型号**（910B/910-95/910-93？影响 UB 协议族和 CCU 支持）
2. **URMA 在 NPU kernel 内是否可用**（目前 UMDK 示例都在 host 侧，需确认 NPU kernel 能否调用 URMA — 如果可以，PortChannel 可完全改为 NPU 自主推进）
3. **OBMM 在灵衢超节点内的延迟特性**（vs NVLink multicast 延迟）
4. **aclrtGraph API 是否可用**（CANN Graph capture 是否支持 MSCCL++ 的使用模式）
5. **CANN peer access API**（cudaDeviceCanAccessPeer / cudaDeviceEnablePeerAccess 的 CANN 对应）
6. **aclrtMemset 是否支持设备内存清零**（HCCL adapter_acl.cc 中有使用）
7. **4MB 对齐对性能的影响**（小缓冲区分配策略）
8. **SyncCollectives 在灵衢超节点是否支持 384 卡**
9. **AIV kernel 动态加载路径是否在灵衢可用**（$ASCEND_HOME_PATH/lib64/*.o）
10. **CCU 在灵衢是否可用**（影响 SwitchChannel 是否可用 CCU 加速）

---

## 10. 下一步行动（更新版）

| 步骤 | 内容 | 前置条件 |
|------|------|----------|
| 1 | 确认灵衢硬件特性和 UMDK 可用性（第 9 节） | 用户提供 |
| 2 | P0：gpu.hpp/gpu_utils 完善已有 CANN 桩代码 | 步骤 1 |
| 3 | P0：错误处理宏 MSCCLPP_ACLTHROW | 步骤 2 |
| 4 | P0：cudaMemset → aclrtMemset 映射 | 步骤 2 |
| 5 | P1：NCCL shim → HCCL shim | 步骤 2 |
| 6 | P2：CMake 构建系统 + Python bindings | 步骤 2 |
| 7 | P3：IPC → OBMM（含 4MB 对齐策略） | 步骤 2-6 |
| 8 | P3：cudaMemcpyToSymbol → CommArgs 参数传递 | 步骤 2 |
| 9 | P4：semaphore/fifo → SyncCollectives/URPC | 步骤 7 |
| 10 | P5：MemoryChannel/PortChannel kernel Ascend C 重写 | 步骤 7-9 |
| 11 | P6：SwitchChannel OBMM + SyncCollectives | 步骤 7-9 |
| 12 | P6：execution_kernel 框架 Ascend C 重写 | 步骤 10-11 |
| 13 | P7：allreduce/allgather kernel Ascend C 重写 | 步骤 10-11 |
| 14 | P8：CUDA Graph → CANN Graph/capture | 步骤 2 |

> **可视化图表**：详见 `.opencode/plans/cuda-to-cann-migration-diagrams.md`（15 个 Mermaid 图表覆盖架构、时序、流程、依赖等所有维度）

---

## 13. P0 实施状态与代码审查（2026-06-01）

> **重要**：本节记录 P0 阶段的实际实施情况、代码审查发现的问题，以及基于 CANN 源码的战略调整。新 session 应从此节开始了解进度。

### 13.1 P0 完成状态

P0 已完成基础 API 适配层，覆盖 **11 个文件，762 行新增代码**。核心思路是在 `MSCCLPP_DEVICE_CANN` 条件下将 CUDA Runtime/Driver API 映射到 ACL 对应 API，使现有 `.cc` 代码无需修改即可编译（排除 `.cu` 文件）。

| 文件 | 改动内容 | 审查状态 |
|------|----------|----------|
| `include/mscclpp/device.hpp` | CANN 宏定义（`MSCCLPP_HOST_COMPILE`/`MSCCLPP_DEVICE_CANN`） | ✅ 正确 |
| `include/mscclpp/gpu.hpp` | ~440 行 CANN shim（类型映射 + inline 函数映射） | ✅ 已修复（问题2-4） |
| `include/mscclpp/gpu_data_types.hpp` | 软浮点 `__half`/`__bfloat16` 实现 | ✅ 已修复（问题6） |
| `src/core/gpu_utils.cc` | CANN 版 AvoidCudaGraphCaptureGuard stub + 基础内存/流操作 | ✅ 正确 |
| `src/core/include/gpu_utils_internal.hpp` | CANN teardown 错误检测 | ✅ 已修复（问题1） |
| `src/core/numa.cc` | sysfs-based NPU NUMA 检测 | ✅ 正确 |
| `CMakeLists.txt` (root) | `MSCCLPP_USE_CANN` option + 查找 CANN SDK | ✅ 已修复（问题5） |
| `src/core/CMakeLists.txt` | 排除 `.cu` + 定义 `MSCCLPP_USE_CANN` | ✅ 正确 |
| `src/ext/collectives/CMakeLists.txt` | 同上 | ✅ 正确 |
| `src/ext/nccl/CMakeLists.txt` | 同上 | ✅ 正确 |
| `include/mscclpp/gpu_utils.hpp` (GpuBuffer) | CANN 分支用 `gpuCallocShared` | ✅ 正确 |

### 13.2 P0 代码审查发现的问题（需修复）

#### 问题 1：`ACL_ERROR_RT_INNER`/`ACL_ERROR_RT_DEVICE_NOT_FOUND` 不存在

**文件**：`gpu_utils_internal.hpp:17`

**现状**：
```cpp
return err == ACL_ERROR_RT_INNER || err == ACL_ERROR_RT_DEVICE_NOT_FOUND ||
       static_cast<int>(err) >= 1000;
```

**问题**：查遍 CANN runtime `rt_error_codes.h`，不存在这两个常量。

**应改为**：
```cpp
return err == ACL_ERROR_RT_INTERNAL_ERROR || err == ACL_ERROR_RT_NO_DEVICE ||
       static_cast<int>(err) >= 1000;
```

对应常量值：`ACL_ERROR_RT_INTERNAL_ERROR = 507000`, `ACL_ERROR_RT_NO_DEVICE = 207004`

`static_cast<int>(err) >= 1000` 的兜底逻辑是合理的（自定义错误码 1000-1004 都 >= 1000）。

#### 问题 2：`aclrtWaitAndResetNotify` 传 nullptr 作为 stream

**文件**：`gpu.hpp:151`

**现状**：
```cpp
aclError ret = aclrtWaitAndResetNotify(event, nullptr, 0);
```

**问题**：CANN 实际签名 `aclrtWaitAndResetNotify(aclrtNotify notify, aclrtStream stream, uint32_t timeout)`。传 `nullptr` 作为 stream 可能不合法——ACL 文档要求有效 stream。

**当前 workaround**：先在 nullptr stream 上提交 wait 任务，再 `aclrtSynchronize()` 阻塞整个设备。

**建议**：改为传有效 stream（需保存当前 stream），或使用 `aclrtSynchronizeStream` 在特定 stream 上等待。最简单方案是 `aclrtWaitAndResetNotify(event, nullptr, 0)` 后跟 `aclrtSynchronize()` — 当前代码已这样做了，但 `nullptr` 的合法性需在灵衢硬件上验证。

#### 问题 3：`aclrtCreateNotify` 参数类型不匹配

**文件**：`gpu.hpp:141-144`

**现状**：
```cpp
inline cudaError_t cudaEventCreate(cudaEvent_t* event) { return aclrtCreateNotify(event, 0); }
inline cudaError_t cudaEventCreateWithFlags(cudaEvent_t* event, unsigned int flags) {
  (void)flags;
  return aclrtCreateNotify(event, 0);
}
```

**问题**：CANN 签名 `aclrtCreateNotify(aclrtNotify *notify, uint64_t flag)`。第二个参数是 `uint64_t`，代码传 `0`（隐式 int）。

**应改为**：`aclrtCreateNotify(event, 0ULL)` 或 `aclrtCreateNotify(event, static_cast<uint64_t>(0))`

#### 问题 4：`cuGetErrorString` 宏的参数类型

**文件**：`gpu.hpp:198`

**现状**：
```cpp
#define cuGetErrorString(err, str) do { *str = aclrtGetErrorCode(*err); } while(0)
```

**问题**：CUDA 原型 `cuGetErrorString(CUresult error, const char** pStr)`。在 CANN 中 `CUresult = aclError`。宏中 `*err` 对 `aclrtGetErrorCode(aclError)` — 需确认 `aclrtGetErrorCode` 接受值而非指针。调用处 `gpu_utils_internal.hpp:70` 为 `cuGetErrorString(&__e, &errStr)`，`*err` 即 `__e`（aclError 值），这应该正确。

**结论**：逻辑正确，但代码注释说明会更安全。

#### 问题 5：CMake 硬编码 aarch64 路径

**文件**：`CMakeLists.txt:77-78`

**现状**：
```cmake
find_path(CANN_INCLUDE_DIR NAMES acl/acl.h PATHS ${ASCEND_TOOLKIT_HOME}/aarch64/include)
find_library(CANN_RUNTIME_LIB NAMES acl_cann PATHS ${ASCEND_TOOLKIT_HOME}/aarch64/lib)
```

**问题**：x86_64 开发机上的 CANN SDK 路径为 `x86_64/include` 和 `x86_64/lib`。

**应改为**：同时搜索两种架构路径，或根据 `CMAKE_SYSTEM_PROCESSOR` 自动选择。

#### 问题 6：`__bfloat16` 转换缺少 round-to-nearest-even

**文件**：`gpu_data_types.hpp:51-53`

**现状**：
```cpp
MSCCLPP_HOST_DEVICE_INLINE explicit __bfloat16(float v) {
  union { float f; uint32_t u; } src = {v};
  __x = static_cast<uint16_t>(src.u >> 16);
}
```

**问题**：直接截断低 16 位，没有 round-to-nearest-even。对大部分数值可接受（bfloat16 精度低），但严格 IEEE 舍入需考虑被截断位的最高位。

**结论**：可接受的简化，后续可按需改进。

#### 问题 7：gpu_data_types.hpp CANN 分支缺少 FP8 类型

**现状**：CANN 分支只定义了 `__half`/`__half2`/`__bfloat16`/`__bfloat162`，没有 FP8 类型（`__fp8_e4m3`/`__fp8_e5m2`）。

**影响**：下游 `__FP8_TYPES_EXIST__` 未定义，所有 FP8 代码路径被排除。这在当前阶段（host-only compile）是正确的，因为 FP8 算法需要 device kernel。但后续 Ascend C 重写时需要补充。

### 13.3 CANN 源码关键发现

通过查阅下载的 5 个 CANN 源码仓（`cann_driver`, `cann_hccl`, `cann_hcomm`, `cann_runtime`, `cann_shmem`），有以下关键发现影响后续迁移策略：

#### 发现 1：ACLSHMEM 是真正的零拷贝公共 API

**之前计划**：用 OBMM + SyncCollectives 替代 CUDA IPC + PTX semaphore。

**新发现**：`cann_shmem` 提供了完整的 OpenSHMEM-like API，包括：
- **Host 侧**：`aclshmem_put/get`（RMA 操作）、`aclshmem_malloc`（对称堆分配）
- **Device 侧**：`aclshmem_put/get`（GM-to-GM DMA）、DMA engine 选择（RDMA/SDMA/UDMA/MTE）
- **同步**：`aclshmem_signal_op` / `aclshmem_signal_wait_until`（P2P 同步）
- **Team**：`aclshmem_team_create/destroy`（通信子组）

这比 UMDK 内部的 OBMM/SyncCollectives 更标准化、更易维护，应作为后续 IPC/同步机制替代的**首选方案**。

#### 发现 2：无独立 OBMM/SyncCollectives 公共 API

OBMM 功能在 URMA 段注册和 HCCL 内存注册内部实现，没有独立的公共头文件。SyncCollectives 同理——等效概念在 ACLSHMEM signal ops 和 hcomm notify 中。

#### 发现 3：HCCL Primitives 提供低级传输

`hcomm_primitives.h` 提供：
- `HcommLocalCopyOnThread` / `HcommLocalReduceOnThread`（本地操作）
- `HcommRemoteWriteOnChannel` / `HcommRemoteReduceOnChannel`（跨 NPU 操作）
- `HcommSignalOnChannel` / `HcommThreadNotifyRecord/Wait`（同步）

这些是 channel-based 的低级传输 API，可能替代 PortChannel 的 proxy 模式。

#### 发现 4：URMA API 定义在 hcomm test depends 中

`urma_api.h`（1138 行）位于 `cann_hcomm/hcomm/test/ut/depends/include/urma/`，这不是公共 API 目录，但提供了完整的 URMA 类型定义和函数声明，可作为跨节点 RDMA 通信的参考。

### 13.4 战略调整建议

基于 CANN 源码发现，迁移策略从 v2 计划的 OBMM+SyncCollectives 调整为：

| 原计划替代 | 新推荐替代 | 原因 |
|-----------|-----------|------|
| OBMM 共享内存 | **ACLSHMEM 对称堆** (`aclshmem_malloc/ptr`) | 公共 API，标准化，有 host+device 支持 |
| SyncCollectives OuterFlag | **ACLSHMEM signal** (`aclshmem_signal_op/wait_until`) | 公共 API，有非阻塞+阻塞模式 |
| SyncCollectives InnerFlag | Ascend C 本地同步或 `aclshmem_signal_op`（同 NPU） | |
| PTX red.release.sys | **ACLSHMEM signal_op** (SET) | 跨 NPU 原子信号 |
| PTX st.release.sys (FIFO) | **ACLSHMEM put** 或 **hcomm channel write** | 替代 FIFO push |
| peerMems[] 直接读写 | **ACLSHMEM device put/get** (GM-to-GM) | 有 DMA engine 选择 |
| NVLS multimem | **ACLSHMEM + 软件 reduce** | 无硬件 multicast，需 NPU 各自读+本地 reduce |
| URPC func_call | **hcomm channel operations** | channel-based 低级传输 |
| URMA write/read | **URMA**（跨节点，保持原计划） | URMA 是 RDMA 替代，无变化 |

### 13.5 下一步行动（P0 后更新版）

| 步骤 |
|------|------|----------|------|
| P0-fix | 修复上述 7 个代码问题（问题1-6已修复，问题7-FP8留后续） | 无 | **已完成（6/7）** |
| P0-done | gpu.hpp/gpu_utils 完善已有 CANN 桩代码 | 无 | **已完成** |
| P0-done | 错误处理宏 MSCCLPP_ACLTHROW | P0-done | **已完成**（但需修正错误码） |
| P0-done | cudaMemset → aclrtMemset 映射 | P0-done | **已完成** |
| P1 | NCCL shim → HCCL shim | P0-fix | 待做 |
| P2 | CMake 构建系统完善（x86/aarch64 双架构） | P0-fix | 待做 |
| P2 | Python bindings 类型 cast + 设备类型 | P0-fix | 待做 |
| P3 | IPC → ACLSHMEM 对称堆（替代 OBMM 方案） | P1-P2 | 待做，策略调整 |
| P3 | cudaMemcpyToSymbol → CommArgs 参数传递 | P0-fix | 待做 |
| P4 | semaphore → ACLSHMEM signal_op/wait_until | P3 | 待做，策略调整 |
| P4 | FIFO → ACLSHMEM put 或 hcomm channel | P3 | 待做，策略调整 |
| P5 | MemoryChannel kernel → ACLSHMEM device put/get | P3-P4 | 待做 |
| P5 | PortChannel kernel → ACLSHMEM + signal 或 hcomm | P3-P4 | 待做 |
| P6 | SwitchChannel → ACLSHMEM + 软件 reduce | P5 | 待做，无硬件 multicast |
| P6 | execution_kernel 框架 → Ascend C 重写 | P5 | 待做 |
| P7 | allreduce/allgather → Ascend C 重写 | P6 | 待做 |
| P8 | CUDA Graph → CANN stream capture | P0-fix | 待做 |

### 13.7 Session 修复记录（2026-06-01 P0-fix）

本次 session 完成了 P0 代码审查发现的 7 个问题中的 6 个修复：

| 问题 | 修复内容 | 修改文件 |
|------|----------|----------|
| 1 | `ACL_ERROR_RT_INNER`/`ACL_ERROR_RT_DEVICE_NOT_FOUND` → `ACL_ERROR_RT_INTERNAL_ERROR`/`ACL_ERROR_RT_NO_DEVICE` | `gpu_utils_internal.hpp` |
| 2 | `cudaEventSynchronize` 中 `nullptr` stream 注释改进，说明替代方案 | `gpu.hpp` |
| 3 | `aclrtCreateNotify(event, 0)` → `aclrtCreateNotify(event, 0ULL)` 修复参数类型 | `gpu.hpp` |
| 4 | `cuGetErrorString` 宏添加逻辑正确性注释 | `gpu.hpp` |
| 5 | CMake CANN 查找路径从硬编码 `aarch64` → 根据 `CMAKE_SYSTEM_PROCESSOR` 自动选择 `aarch64`/`x86_64` | `CMakeLists.txt` |
| 6 | `__bfloat16(float v)` 转换添加 round-to-nearest-even 舍入 | `gpu_data_types.hpp` |
| 7 | FP8 类型暂不补充，留后续 Ascend C kernel 重写时处理 | 不修改 |

**下一步**：P0-fix 完成后，可进入 P1（NCCL shim → HCCL shim）阶段。

### 13.6 CANN 源码仓库参考

| 仓库 | 路径 | 关键头文件 | 用途 |
|------|------|-----------|------|
| cann_runtime | `D:\C++\cann_runtime\runtime` | `include/external/acl/acl_rt.h` (5566行), `error_codes/rt_error_codes.h` (168行) | aclrt* 全部 Runtime API + 错误码 |
| cann_hccl | `D:\C++\cann_hccl\hccl` | `include/hccl.h` (254行), `include/hccl_mc2.h` (88行) | HCCL 集合通信 + KFC kernel API |
| cann_hcomm | `D:\C++\cann_hcomm\hcomm` | `include/hcomm_primitives.h` (509行), `include/hcomm_res_defs.h` (303行), `test/ut/depends/include/urma/urma_api.h` (1138行) | 低级传输 primitives + URMA API |
| cann_shmem | `D:\C++\cann_shmem\shmem` | `include/shmem.h`, `include/device/gm2gm/shmem_device_rma.h` (771行), `include/device/gm2gm/shmem_device_p2p_sync.h` (578行) | ACLSHMEM 零拷贝 host+device API |
| cann_driver | `D:\C++\cann_driver\driver` | `pkg_inc/ascend_hal.h`, `pkg_inc/dsmi_common_interface.h` | HAL + DSMI 设备管理 |

---

## 11. 可视化图表

详细的 Mermaid 图表集参见 `.opencode/plans/cuda-to-cann-migration-diagrams.md`，包含 15 个图表：

| 图号 | 内容 | 用途 |
|------|------|------|
| 图1 | MSCCL++ CUDA 侧整体架构 | 验证 CUDA 依赖扫描完整性 |
| 图2 | 灵衢 CANN 侧目标架构 | 验证替代方案覆盖完整性 |
| 图3 | MemoryChannel 数据流时序（CUDA vs 灵衢） | 验证零拷贝机制可行性 |
| 图4 | PortChannel 数据流时序（CUDA vs 灵衢） | 验证 FIFO/proxy 替代方案 |
| 图5 | SwitchChannel 数据流时序（CUDA vs 灵衢） | 验证 multicast 替代方案 |
| 图6 | FIFO 机制详细时序 | 验证 PortChannel 核心机制理解 |
| 图7 | SyncCollectives 机制时序 | 验证同步机制理解完整性 |
| 图8 | OBMM 共享内存流程 | 验证 IPC 替代方案可行性 |
| 图9 | AIV Kernel 加载管线 | 验证 kernel 部署策略 |
| 图10 | executionKernel Operation 处理流程 | 验证 Operation 类型系统映射 |
| 图11 | 迁移优先级依赖图 | 验证阶段划分和依赖关系 |
| 图12 | CUDA API 依赖层次 | 验证 API 映射从底层到上层完整性 |
| 图13 | 灵衢替代 API 依赖层次 | 验证灵衢替代从底层到上层覆盖 |
| 图14 | CUDA vs 灵衢对比矩阵 | 验证需求覆盖完整性（✓/⚠️ 标记） |
| 图15 | GDRCopy/IB HostNoAtomic 时序 | 验证 IB 特殊模式理解 |

---

## 12. v1 → v2 变更摘要

| 变更类型 | 数量 | 说明 |
|----------|------|------|
| **新增类别** | 6 | CUDA Graph API, PTX 内联汇编, Python 层, execution_kernel, DLock/URPC/USOCK/CAM, cudaMemset |
| **新增 API** | ~50 | CUDA侧~20(Graph/memset/elapsedTime/PCIBusId/attribute等), HCCL侧~10(capture/memset/unload等), UMDK侧~40+(URMA完整/DLock/URPC) |
| **新增数据结构** | 8 | ProxyTrigger, HcclOpResParam完整, CommArgs完整, AivKernelArgsDef, GroupOpSize/Config, obmem_param_t, SyncCollectives完整 |
| **纠正错误** | 3 | atomicAdd/__shfl_sync/warpSize 实际不存在; FP8 已确认支持; CANN桩代码已存在 |
| **新增映射** | ~30 | PTX→灵衢, Operation→灵衢, cudaMemset→aclrtMemset, Graph→CANN, Python→CANN |
| **新增策略** | 3 | PortChannel替代方案, cudaMemcpyToSymbol替代, OBMM 4MB对齐影响 |