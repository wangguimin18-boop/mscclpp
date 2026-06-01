# MSCCL++ CUDA → CANN 迁移项目

## 项目目标

将 MSCCL++ 从 CUDA 平台迁移到华为 CANN（Ascend NPU）平台，使其可在灵衢超节点上运行。

## 1. CUDA 依赖扫描结果

### 1.1 CUDA 头文件引用（7 个唯一头文件，14 处 include）

| 类别 | 头文件 | 涉及文件 | 说明 |
|------|--------|----------|------|
| Runtime API | `cuda_runtime.h` | gpu.hpp, nvls_test.cu, peer_access_test.cu, check_nvidia_gpu.cu | 设备/内存/流管理 |
| Runtime API（纯声明） | `cuda_runtime_api.h` | gpu_data_types.hpp | CUDART_VERSION 宏和类型定义 |
| Driver API | `cuda.h` | gpu.hpp, nvls_test.cu | CUresult, CUdeviceptr, cuMem* 虚拟内存/NVLS |
| Driver 类型定义 | `cudaTypedefs.h` | nvls_test.cu | CUmemAllocationHandleType 等 |
| FP16 | `cuda_fp16.h` | gpu_data_types.hpp, switch_channel_device.hpp, executor_test_verifier.cu, allreduce.cu | __half, __half2 |
| BF16 | `cuda_bf16.h` | gpu_data_types.hpp, executor_test_verifier.cu | __nv_bfloat16（CUDA >= 11.0） |
| FP8 | `cuda_fp8.h` | gpu_data_types.hpp | __nv_fp8_e4m3（CUDA >= 11.8） |
| CUDA C++ 标准库 | `cuda/atomic` | atomic_device.hpp | cuda::atomic, cuda::memory_order |

**注意**：项目已支持 HIP（AMD ROCm），所有 CUDA include 都有 `#if defined(MSCCLPP_DEVICE_HIP)` 条件分支。这为添加 CANN 条件分支提供了现成模式。

### 1.2 CUDA Runtime API 调用分类

#### 设备管理（迁移难度：低）

| API | CANN 对应 | 调用次数 | 核心文件 |
|-----|-----------|---------|----------|
| cudaGetDevice | aclrtGetDevice | ~25 | gpu_utils.cc, proxy.cc, port_channel.cc, fifo.cc, endpoint.cc, debug.cc, registered_memory.cc, connection.cc, npkit.cc, gpu_ipc_mem.cc, nccl.cc, allreduce_nvls_zero_copy.cu |
| cudaSetDevice | aclrtSetDevice | ~18 | gpu_utils.cc, proxy.cc, port_channel.cc, connection.cc, 各 test/example |
| cudaGetDeviceCount | aclrtGetDeviceCount | ~6 | gpu_utils.cc, peer_access_test.cu, numa_tests.cc, ib_tests.cu |
| cudaDeviceSynchronize | aclrtSynchronize | ~40+ | 各 test/example |
| cudaGetDeviceProperties | aclrtGetDeviceInfo | ~3 | common.cc, allreduce_nvls_zero_copy.cu, npkit.cc |
| cudaDeviceCanAccessPeer | 需确认 CANN 对应 | ~4 | peer_access_test.cu, tutorial examples |
| cudaDeviceEnablePeerAccess | 需确认 CANN 对应 | 1 | connection.cc |

#### 内存管理（迁移难度：低-中）

| API | CANN 对应 | 调用次数 | 核心文件 |
|-----|-----------|---------|----------|
| cudaMalloc | aclrtMalloc | ~15 | gpu_utils.cc, 各 test/example |
| cudaFree | aclrtFree | ~5 | gpu_utils.cc, 各 test/example |
| cudaHostAlloc | aclrtMallocHost | ~3 | gpu_utils.cc |
| cudaFreeHost | aclrtFreeHost | 1 | gpu_utils.cc |
| cudaMemcpy | aclrtMemcpy | ~30+ | 各 test/example, connection.cc, algorithm.cc |
| cudaMemcpyAsync | aclrtMemcpyAsync | ~15 | gpu_utils.cc, context.cc, nccl.cc, collective_utils.cc, executor.cc |
| cudaMemcpyToSymbol | **需特殊处理** | ~40+ | 各 test（device 常量内存赋值） |

**关键问题**：`cudaMemcpyToSymbol` 用于将 host 数据拷贝到 device 全局/常量变量。CANN 无直接对应，需改为通过 device 全局指针 + aclrtMemcpy 方式。

#### Stream/Event 管理（迁移难度：低）

| API | CANN 对应 | 调用次数 |
|-----|-----------|---------|
| cudaStreamCreate / cudaStreamCreateWithFlags | aclrtStreamCreate | ~10 |
| cudaStreamDestroy | aclrtStreamDestroy | ~3 |
| cudaStreamSynchronize | aclrtStreamSynchronize | ~20+ |
| cudaEventCreate / cudaEventCreateWithFlags | aclrtEventCreate | ~12 |
| cudaEventRecord | aclrtEventRecord | ~10 |
| cudaEventSynchronize | aclrtEventSynchronize | ~10 |
| cudaEventDestroy | aclrtEventDestroy | 少量 |

#### IPC 内存（迁移难度：高）

| API | 说明 | 核心文件 |
|-----|------|----------|
| cudaIpcGetMemHandle | 获取 IPC 内存句柄 | gpu_ipc_mem.cc |
| cudaIpcOpenMemHandle | 打开 IPC 内存句柄 | gpu_ipc_mem.cc |
| cudaIpcCloseMemHandle | 关闭 IPC 内存句柄 | gpu_ipc_mem.cc |

**关键问题**：CANN IPC 机制与 CUDA 不同。Ascend NPU 跨进程内存共享可能需通过 `aclrtMemExportToShareableHandle` / `aclrtMemImportFromShareableHandle`，需确认灵衢平台支持情况。

#### 错误处理（迁移难度：低）

| API | CANN 对应 |
|-----|-----------|
| cudaGetErrorString | aclrtGetErrorCode / 自定义映射 |
| cudaGetLastError | 需确认 |
| cudaSuccess | ACL_SUCCESS |

### 1.3 CUDA Driver API 调用分类（迁移难度：极高）

#### 虚拟内存管理

| API | 说明 | 核心文件 |
|-----|------|----------|
| cuMemCreate | 创建虚拟内存分配 | gpu_utils.cc, gpu_ipc_mem.cc |
| cuMemMap | 映射虚拟内存 | gpu_utils.cc, gpu_ipc_mem.cc, nvls_test.cu |
| cuMemUnmap | 取消映射 | gpu_utils.cc, gpu_ipc_mem.cc |
| cuMemAddressReserve | 预留虚拟地址空间 | gpu_utils.cc, gpu_ipc_mem.cc, nvls_test.cu |
| cuMemAddressFree | 释放虚拟地址 | gpu_utils.cc, gpu_ipc_mem.cc |
| cuMemSetAccess | 设置内存访问权限 | gpu_utils.cc, gpu_ipc_mem.cc, nvls_test.cu |
| cuMemGetAllocationGranularity | 获取分配粒度 | gpu_utils.cc, gpu_ipc_mem.cc |
| cuMemGetAddressRange | 获取地址范围 | gpu_utils.cc, utils_internal.cc, executor.cc, 各 allreduce/allgather .cu |
| cuMemExportToShareableHandle | 导出共享句柄（文件描述符/Fabric） | gpu_ipc_mem.cc, nvls_test.cu |
| cuMemImportFromShareableHandle | 导入共享句柄 | gpu_ipc_mem.cc, nvls_test.cu |
| cuMemRetainAllocationHandle | 保留分配句柄 | gpu_utils.cc, gpu_ipc_mem.cc |
| cuMemGetHandleForAddressRange | DMA_BUF 句柄 | ib.cc |
| cuMulticastCreate | 创建 multicast 内存 | gpu_ipc_mem.cc, nvls_test.cu |
| cuMulticastAddDevice | 添加设备到 multicast | gpu_ipc_mem.cc, nvls_test.cu |
| cuMulticastBindAddr | 绑定地址到 multicast | gpu_ipc_mem.cc, nvls_test.cu |
| cuMulticastUnbind | 解绑 multicast | gpu_ipc_mem.cc |
| cuMulticastGetGranularity | 获取 multicast 粒度 | switch_channel.cc, gpu_utils.cc, gpu_ipc_mem.cc |

**关键问题**：NVLS multicast 是 MSCCL++ switch_channel 的核心机制。CANN 无对应 API，灵衢超节点是否有类似群组内存共享/广播机制需确认。如无，需完全重新设计 switch_channel。

#### Driver 其他 API

| API | 说明 | 核心文件 |
|-----|------|----------|
| cuDeviceGet | 获取 CUdevice | gpu_utils.cc, ib.cc, gpu_ipc_mem.cc |
| cuDeviceGetAttribute | 获取设备属性 | gpu_utils.cc, ib.cc, gpu_ipc_mem.cc |
| cuPointerGetAttribute | 获取指针属性 | gpu_utils.cc |
| cuGetErrorString | Driver 错误字符串 | registered_memory.cc, gpu_utils_internal.hpp, gpu_ipc_mem.cc, errors.cc |

### 1.4 CUDA Device Keywords（迁移难度：高）

| 关键字 | 说明 | 涉及范围 |
|--------|------|----------|
| `__global__` | kernel 入口 | 53+ kernel 定义 |
| `__device__` | device 函数/变量 | 全部 .cu 文件 |
| `__shared__` | shared memory | allreduce_packet.cu, allreduce_nvls_*.cu |
| `__syncthreads()` | 线程同步 | 全部 kernel |
| `blockIdx, threadIdx, blockDim, gridDim` | 线程索引 | 全部 kernel |
| `warpSize` | warp 大小 | allreduce .cu |
| `__shfl_sync()` | warp shuffle | allreduce .cu |
| `__forceinline__` | 强制内联 | allreduce .cu |
| `atomicAdd` | 原子加 | allreduce .cu |

**关键问题**：Ascend NPU kernel 编程使用 **Ascend C** 语言，编程模型与 CUDA 不同：
- CUDA: thread/block/grid 层次 → Ascend C: 6 维并行模型
- CUDA warp → Ascend C 无 warp 概念，有 Cube/Vector 协同
- 53+ 个 CUDA kernel 全部需用 Ascend C 重写

### 1.5 NCCL Shim（迁移难度：低）

| API | 说明 | HCCL 对应 |
|-----|------|-----------|
| ncclGetUniqueId | 获取唯一 ID | HcclGetClkId |
| ncclCommInitRank | 初始化通信 | HcclCommInitRootInfo / HcclCommInitClusterInfo |
| ncclAllReduce | AllReduce | HcclAllReduce |
| ncclReduce | Reduce | HcclReduce |
| ncclBroadcast | Broadcast | HcclBroadcast |
| ncclAllGather | AllGather | HcclAllGather |
| ncclReduceScatter | ReduceScatter | HcclReduceScatter |
| ncclCommDestroy | 销毁通信 | HcclCommDestroy |
| ncclSend/ncclRecv | P2P | 需确认 HCCL 对应 |

核心文件：`src/ext/nccl/nccl.cc`, `include/mscclpp/ext/nccl/nccl.h`

### 1.6 构建系统 CUDA 配置

| 配置项 | 说明 | 位置 |
|--------|------|------|
| `MSCCLPP_USE_CUDA` option | CUDA 开关 | 根 CMakeLists.txt |
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

## 2. 迁移难度评估

| 难度等级 | 模块 | 说明 |
|----------|------|------|
| **低** | 设备管理 API | aclrt 有直接对应 API |
| **低** | Stream/Event API | aclrt 有直接对应 |
| **低** | NCCL shim → HCCL | 接口名称替换 |
| **低** | 错误处理 | ACL_SUCCESS + 错误码映射 |
| **中** | 内存管理（host<->device 拷贝） | cudaMemcpyToSymbol 需特殊处理 |
| **中** | 数据类型（FP16/BF16/FP8） | CANN 有 FP16/BF16，FP8 需确认 |
| **中** | 构建系统 | 需新增 CANN 编译链配置 |
| **中** | cuda::atomic | 需用 Ascend C 原子操作替代 |
| **高** | IPC 内存机制 | CANN IPC 差异大 |
| **高** | Device kernel（53+ 个） | 需用 Ascend C 全部重写 |
| **极高** | NVLS multicast / switch_channel | **无 CANN 对应，需重新设计** |

## 3. 迁移优先级

| 优先级 | 阶段 | 内容 | 前置条件 |
|--------|------|------|----------|
| **P0** | 基础 API 适配 | 设备管理 + 内存管理 + Stream/Event + 错误处理 | 无 |
| **P1** | NCCL → HCCL shim | src/ext/nccl 替换为 HCCL 接口 | P0 完成 |
| **P2** | 构建系统适配 | CMake 新增 CANN 编译链、宏定义 | P0 完成 |
| **P3** | IPC 内存机制 | 研究 CANN IPC，适配 gpu_ipc_mem.cc | 需 CANN IPC 文档 |
| **P4** | Channel kernel | memory_channel / port_channel kernel 用 Ascend C 重写 | P0-P2 完成 |
| **P5** | switch_channel 重设计 | 基于灵衢超节点特性设计群组通信替代 NVLS multicast | 需灵衢硬件特性文档 |
| **P6** | 算法 kernel | allreduce / allgather 全部 kernel 用 Ascend C 重写 | P4 完成 |

## 4. HCCL 源码分析结果（D:\C++\hccl）

### 4.1 HCCL 目录结构

```
hccl/
├── include/                    # 公共 API 头文件
│   ├── hccl.h                  # 主 API（集合通信 + P2P）
│   ├── hccl_mc2.h              # KFC（Kernel Function Call）API
├── src/
│   ├── common/                 # 公共基础设施
│   │   ├── adapter_acl.h/cc    # ACL runtime 适配层（封装 aclrt* 调用）
│   │   ├── hccl_common.h       # 核心类型定义
│   │   ├── hcomm_dlsym/        # 动态符号加载（26 个 dlsym 文件）
│   ├── ops/                    # 集合通信算子实现
│   │   ├── all_reduce/         # AllReduce
│   │   ├── all_gather/         # AllGather / AllGatherV
│   │   ├── all_to_all_v/       # AlltoAll / AlltoAllV / AlltoAllVC
│   │   ├── broadcast/          # Broadcast
│   │   ├── reduce/             # Reduce
│   │   ├── reduce_scatter/     # ReduceScatter / ReduceScatterV
│   │   ├── scatter/            # Scatter
│   │   ├── send/recv/          # P2P 通信
│   │   ├── batch_send_recv/    # 批量 P2P
│   │   ├── op_common/          # 共享基础设施
│   │       ├── template/       # 算法模板
│   │       │   ├── ccu/        # Cube Communication Unit（硬件加速）
│   │       │   │   ├── kernel/ # CCU 微码生成（CCU_WHILE/CCU_IF DSL）
│   │       │   ├── aiv/        # AI Vector 核心（Ascend C kernel）
│   │       │   │   ├── kernel/ # AIV kernel 实现
│   │       │   ├── aicpu/      # AICPU 模板
│   │       │   ├── dpu/        # DPU（跨节点数据传输）
│   │       ├── selector/       # 算法自动选择
│   │       ├── executor/       # 执行引擎（channel 管理）
│   ├── common/hccl_mc2.cc      # KFC API 实现
├── examples/                    # 示例代码
│   ├── 04_custom_ops_p2p/      # 自定义 P2P（含 Ascend C kernel）
│   ├── 05_custom_ops_allgather/ # 自定义 AllGather（含 Ascend C kernel）
├── test/                        # 测试
├── cmake/                       # 构建配置
```

### 4.2 HCCL 公共 API 映射表

| HCCL API | 对应 NCCL API | 说明 |
|-----------|--------------|------|
| `HcclAllReduce` | ncclAllReduce | 集合 AllReduce |
| `HcclBroadcast` | ncclBroadcast | 集合 Broadcast |
| `HcclReduceScatter` | ncclReduceScatter | 集合 ReduceScatter |
| `HcclReduceScatterV` | ncclReduceScatterV | 非均匀 ReduceScatter |
| `HcclScatter` | ncclScatter | 集合 Scatter |
| `HcclAllGather` | ncclAllGather | 集合 AllGather |
| `HcclAllGatherV` | ncclAllGatherV | 非均匀 AllGather |
| `HcclReduce` | ncclReduce | 集合 Reduce |
| `HcclAlltoAll` | ncclAlltoAll | 全互联 |
| `HcclAlltoAllV` | ncclAlltoAllV | 非均匀全互联 |
| `HcclAlltoAllVC` | - | VC 变体 |
| `HcclSend` | ncclSend | P2P 发送 |
| `HcclRecv` | ncclRecv | P2P 接收 |
| `HcclBatchSendRecv` | - | 批量 P2P |
| `HcclCommInitRootInfo` | ncclCommInitRank | 多线程初始化 |
| `HcclCommInitClusterInfo` | - | 从集群配置文件初始化 |
| `HcclCommDestroy` | ncclCommDestroy | 销毁通信域 |
| 所有 *GraphMode 变体 | - | 图模式执行（TensorFlow/PyTorch） |

**关键差异**：
- HCCL 用 `HcclRootInfo` 而非 `ncclUniqueId`
- HCCL 有 `ClusterInfo`（集群配置 JSON）初始化方式
- HCCL 有 `HcclBatchSendRecv`（NCCL 无对应）
- HCCL 所有算子都有 GraphMode 变体

### 4.3 HCCL 使用的 CANN Runtime API

| ACL API | 用途 | HCCL 中的位置 |
|---------|------|--------------|
| aclrtGetDevice | 获取当前设备 | 所有算子入口 |
| aclrtSetDevice | 设置设备 | kernel 注册 |
| aclrtMalloc | 分配设备内存 | graph 模式、examples |
| aclrtMallocHost | 分配 host 内存 | scatter 等 |
| aclrtFreeHost | 释放 host 内存 | scatter 等 |
| aclrtFree | 释放设备内存 | examples |
| aclrtMemcpy | 内存拷贝 | adapter_acl.cc |
| aclrtCreateStream | 创建流 | examples/tests |
| aclrtSynchronizeStream | 同步流 | examples/tests |
| aclrtCreateNotify | 创建通知对象 | scatter AICPU |
| aclrtBinaryLoadFromFile | 加载 kernel 二进制 | adapter_acl.cc |
| aclrtBinaryLoadFromData | 从内存加载 kernel | AIV kernel |
| aclrtBinaryGetFunction | 获取 kernel 函数 | op_common、AIV utils |
| aclrtLaunchKernelWithConfig | 启动 kernel（AICPU） | op_common |
| aclrtLaunchKernelWithHostArgs | 启动 kernel（AIV） | AIV utils |
| aclrtGetDeviceInfo | 查询设备属性 | adapter_acl.cc |
| aclrtGetDevicesTopo | 获取拓扑 | adapter_acl.cc |

### 4.4 HCCL IPC 机制（关键发现）

**HCCL 不使用 `aclrtMemExportToShareableHandle` / `aclrtMemImportFromShareableHandle`。**

IPC 类功能通过 HCCL 自有 dlsym 层提供：
- `HcclGetRemoteIpcHcclBuf(comm, remoteRank, addr, size)` — 获取远端 HCCL 缓冲区（IPC 等价）
- `HcclChannelGetRemoteMems(comm, channel, memNum, remoteMems, memTags)` — 通过 channel 访问远端内存
- `HcclCommMemReg(comm, memTag, mem, memHandle)` — 注册内存供共享

**没有找到 multicast/群组内存 API**。群组通信完全通过 HCCL 集合算子和 `HcclComm*` 生命周期管理实现。

### 4.5 HCCL 多引擎执行架构（关键发现）

| 执行引擎 | 说明 | 编程模型 |
|----------|------|----------|
| **AIV**（AI Vector） | NPU Vector Core，执行 Ascend C kernel | `aclrtBinaryLoadFromData` + `aclrtLaunchKernelWithHostArgs` |
| **CCU**（Cube Communication Unit） | 硬件加速群组通信，微码级编程 | `CCU_WHILE/CCU_IF` DSL 生成微码 |
| **AICPU** | Ascend CPU，辅助控制 | `aclrtLaunchKernelWithConfig` |
| **DPU** | 跨节点数据传输 | `HcommSendRequest/HcommWaitResponse` |
| **Host CPU** | 主机侧控制 | 标准 CPU 执行 |

### 4.6 HCCL 同节点通信机制

| 协议 | 值 | 说明 |
|------|-----|------|
| HCCS | 默认 | 华为片间互联（Chip-to-Chip Space） |
| UBC_CTP | 4 | 统一总线 Cube CTP（Ascend 910B+） |
| UBC_TP | 5 | 统一总线 TP（Ascend 910B+） |
| UB_MEM | 6 | 统一总线共享内存（Ascend 910B+） |
| RoCE | 1 | RDMA over Converged Ethernet（跨节点） |

**关键发现**：Ascend 910B/910-95 引入了 **UB（Unified Bus）** 协议族，包括 UBC_CTP、UBC_TP、UB_MEM，这是灵衢超节点的新互联机制。

### 4.7 HCCL Kernel 编程模型

**AIV kernel**（Ascend C）结构：
- `AivSuperKernelArgs` 结构体：包含 GM_ADDR buffersIn（CCLIN 地址，所有 rank 可访问）、rank、rankSize、tag、reduceOp 等
- `KERNEL_ARGS_DEF` 宏：标准化 kernel 参数传递
- `EXPORT_AIV_META_INFO(kernel_name)` 宏：元数据注册
- 信号同步：`AivNotifyType: ACK, DataSignal, Done`
- 通信模式：`CommPattern: interRank`（跨 rank）、`intraRank`（同 rank）
- 内置 reduce：`HCCL_REDUCE_SUM/MAX/MIN`

**CCU kernel**（微码）结构：
- `CcuKernelAlgBase` extends `CcuKernel`
- `GroupBroadcast/GroupReduce/GroupCopy/GroupLocalReduce` 高层操作
- `CCU_WHILE/CCU_IF` DSL 宏生成微码序列
- 按算子+拓扑分文件：`ccu_kernel_all_reduce_mesh1d.h` 等

**DPU 通信**（跨节点）：
- `HcommSendRequest/HcommWaitResponse` 请求/响应模式
- `DPURunInfo` 序列化参数
- `dpu2NpuShmemPtr` 共享内存指针

## 5. CUDA → CANN API 完整映射表

### 5.1 Runtime API 映射

| CUDA Runtime | CANN (aclrt) | HCCL 使用 | 备注 |
|-------------|-------------|----------|------|
| cudaGetDevice | aclrtGetDevice | ✓ | 直接对应 |
| cudaSetDevice | aclrtSetDevice | ✓ | 直接对应 |
| cudaGetDeviceCount | aclrtGetDeviceCount | ✓ | 直接对应 |
| cudaDeviceSynchronize | aclrtSynchronize | - | 直接对应 |
| cudaGetDeviceProperties | aclrtGetDeviceInfo | ✓ | 参数结构不同 |
| cudaMalloc | aclrtMalloc | ✓ | 直接对应 |
| cudaFree | aclrtFree | ✓ | 直接对应 |
| cudaHostAlloc | aclrtMallocHost | ✓ | flags 不同 |
| cudaFreeHost | aclrtFreeHost | ✓ | 直接对应 |
| cudaMemcpy | aclrtMemcpy | ✓ | direction 枚举不同 |
| cudaMemcpyAsync | aclrtMemcpyAsync | - | 直接对应 |
| cudaMemcpyToSymbol | **无对应** | - | 需改为全局指针 + aclrtMemcpy |
| cudaStreamCreate | aclrtCreateStream | ✓ | 直接对应 |
| cudaStreamDestroy | aclrtDestroyStream | - | 直接对应 |
| cudaStreamSynchronize | aclrtSynchronizeStream | ✓ | 直接对应 |
| cudaEventCreate | aclrtCreateNotify | ✓ | Notify 而非 Event |
| cudaEventRecord | aclrtRecordNotify | - | 对应 |
| cudaEventSynchronize | aclrtWaitAndResetNotify | - | 对应 |
| cudaIpcGetMemHandle | **HcclGetRemoteIpcHcclBuf** | ✓ | HCCL 自有 IPC |
| cudaIpcOpenMemHandle | **HcclChannelGetRemoteMems** | ✓ | HCCL 自有 IPC |
| cudaIpcCloseMemHandle | **HcclCommMemReg/unreg** | ✓ | HCCL 自有 IPC |

### 5.2 Driver API 映射

| CUDA Driver | CANN 对应 | HCCL 使用 | 备注 |
|-------------|-----------|----------|------|
| cuMemCreate | **无直接对应** | - | 需用 aclrtMalloc 替代或 HCCL 内存管理 |
| cuMemMap/cuMemUnmap | **无直接对应** | - | CANN 无虚拟内存管理 API |
| cuMemAddressReserve/Free | **无直接对应** | - | CANN 无虚拟地址空间 API |
| cuMemSetAccess | **无直接对应** | - | 需用 HCCL IPC 替代 |
| cuMemGetAllocationGranularity | **无对应** | - | 需硬编码或从 aclrtGetDeviceInfo 获取 |
| cuMemGetAddressRange | **无对应** | - | 需自行管理地址范围 |
| cuMemExportToShareableHandle | **HcclGetRemoteIpcHcclBuf** | ✓ | HCCL 自有机制 |
| cuMemImportFromShareableHandle | **HcclChannelGetRemoteMems** | ✓ | HCCL 自有机制 |
| cuMemRetainAllocationHandle | **无对应** | - | 需重新设计 |
| cuMemGetHandleForAddressRange | **无对应** | - | DMA_BUF 概念不适用 |
| cuMulticastCreate | **无对应** | - | **核心难点：需重新设计** |
| cuMulticastAddDevice | **无对应** | - | **需重新设计** |
| cuMulticastBindAddr | **无对应** | - | **需重新设计** |
| cuMulticastGetGranularity | **无对应** | - | **需重新设计** |
| cuDeviceGet | aclrtGetDevice | ✓ | 逻辑对应 |
| cuDeviceGetAttribute | aclrtGetDeviceInfo | ✓ | 参数不同 |
| cuPointerGetAttribute | **无对应** | - | 需自行管理指针元数据 |

### 5.3 NCCL → HCCL API 映射

| NCCL | HCCL | 备注 |
|------|------|------|
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

### 5.4 CUDA Device Keywords → Ascend C 映射

| CUDA | Ascend C / HCCL | 备注 |
|------|-----------------|------|
| `__global__` | Ascend C kernel 入口 | 编程模型完全不同 |
| `__device__` | Ascend C device 函数 | 编程模型不同 |
| `__shared__` | Ascend C local memory | 对应但语义不同 |
| `__syncthreads()` | Ascend C 同步机制 | 不同 |
| blockIdx/threadIdx | Ascend C 6 维并行 | 完全不同 |
| warpSize | Ascend C 无 warp | 无对应 |
| `__shfl_sync()` | Ascend C 无 warp shuffle | 无对应 |
| cuda::atomic | Ascend C atomic | API 不同 |

## 6. 迁移策略决策

### 6.1 核心架构选择

**方案 A：保留 MSCCL++ 设计（零拷贝 + 无 proxy）**
- 优势：保留 MSCCL++ 的核心创新——GPU/NPU kernel 自主推进通信
- 难点：需用 Ascend C 重写全部 kernel，IPC 用 HCCL 自有机制
- 适用：如果灵衢超节点支持 NPU kernel 直接访问远端内存

**方案 B：改为 HCCL proxy 模式**
- 优势：直接利用 HCCL 成熟的通信基础设施
- 难点：丧失 MSCCL++ 的零拷贝优势，本质上变成 HCCL 的封装
- 适用：如果 NPU kernel 无法直接控制通信进度

**方案 C：混合方案（推荐探索）**
- 同节点：保留 MSCCL++ 零拷贝设计，用 Ascend C kernel + HCCL IPC 替代 CUDA kernel + CUDA IPC
- 跨节点：利用 HCCL DPU 通信机制
- NVLS/multicast：用 CCU（Cube Communication Unit）替代或重新设计

### 6.2 switch_channel 替代方案

**选项 1：CCU 替代 NVLS**
- HCCL 的 CCU（Cube Communication Unit）提供硬件级群组通信（GroupBroadcast/GroupReduce）
- 如果灵衢支持 CCU，可映射 switch_channel 到 CCU 模式
- 需要研究 CCU 是否支持 MSCCL++ 所需的零拷贝访问模式

**选项 2：UB_MEM 替代 NVLS**
- Ascend 910B+ 的 UB_MEM（统一总线共享内存）可能提供类似 multicast 的能力
- 需确认灵衢超节点是否支持 UB_MEM

**选项 3：退化为多路 memory_channel**
- 如无硬件群组通信支持，switch_channel 退化为 N 路 memory_channel
- 性能下降但功能可用

### 6.3 文件组织策略

**选项 A：条件编译（沿用 HIP 模式）**
- 新增 `#if defined(MSCCLPP_DEVICE_CANN)` 分支
- .cu 文件内 Ascend C kernel 通过 `__CANN__` 宏切换
- CMake 新增 `MSCCLPP_USE_CANN` option
- 优势：一个代码库支持多平台

**选项 B：独立 CANN 分支**
- 新建 src/core_cann/ 目录
- .asc kernel 文件独立存放
- 优势：代码清晰，不污染原项目

**推荐：选项 A**（条件编译），因为 MSCCL++ 已有 HIP 条件编译模式可参考。

## 7. 灵衢 UMDK 分析结果（D:\C++\umdk）

### 7.1 UMDK 架构（5 大组件）

| 组件 | 全称 | 作用 |
|------|------|------|
| **URMA** | Unified Remote Memory Access | 远端内存语义访问（RDMA 替代） |
| **CAM** | Communication Acceleration Module | MoE dispatch/combine + Ascend C kernel |
| **URPC** | Unified Remote Process Call | RPC 通信 + 消息队列（UMQ） |
| **ULOCK/DLock** | Distributed Lock | 分布式锁 + 跨 NPU 原子操作 |
| **USOCK/UMS** | UB Memory Socket | 基于 UB 的 socket 通信 |

### 7.2 关键发现：NVLS multicast 的灵衢替代方案

**UMDK 没有直接的 multicast API，但提供了多种组合替代机制：**

#### 7.2.1 OBMM 共享内存（零拷贝核心机制）

| API | 作用 | 文件 |
|------|------|------|
| `obmem_export_memory(param, &handle, &exp)` | 导出本地内存供远端 NPU 访问，返回 {token_id, uba, size} | obmem_common.h |
| `obmem_import_memory(param, exp, &handle)` | 远端 NPU 导入已导出内存，获得本地 mmap'd VA | obmem_common.h |
| `obmem_release_export_memory(handle, ptr, len)` | 释放导出内存 | obmem_common.h |
| `obmem_release_import_memory(handle, ptr, len)` | 释放导入内存 | obmem_common.h |

**工作原理**：
1. NPU A 导出内存 → 获得 UB 地址(uba) + token_id
2. NPU B 导入内存（通过 uba + token_id） → 获得 mmap'd 本地虚拟地址
3. 双方访问同一物理内存 → **零拷贝共享**

**这是 CUDA IPC + cuMemMulticast 的灵衢替代方案**。

#### 7.2.2 HCCL Window 零拷贝（NPU kernel 层面）

```cpp
struct HcclOpResParam {
    uint64_t windowsIn;       // 入站数据窗口（零拷贝读）
    uint64_t windowsOut;      // 出站数据窗口（零拷贝写）
    uint64_t windowsExp;      // Expert/status 数据窗口
    uint64_t zeroCopyRingBuffer;   // 零拷贝环形缓冲区
    uint64_t zeroCopyIpcPtrs[16];  // 16 个 NPU 的 IPC 指针
};
```

#### 7.2.3 peerMems[] — 直接跨 NPU 内存访问

```cpp
constexpr int CAM_MAX_RANK_SIZE = 384;  // 最大 384 个 NPU 卡
struct CommArgs {
    GM_ADDR peerMems[CAM_MAX_RANK_SIZE]; // 直接访问任意 NPU 的全局内存
};
```

**这是灵衢超节点最关键的能力**：每个 NPU kernel 可以通过 `peerMems[rank]` 直接读写任意远端 NPU 的内存，无需远端 CPU 参与。**等同于 MSCCL++ 零拷贝设计的前提条件**。

#### 7.2.4 SyncCollectives — 跨 NPU kernel 同步

```cpp
// Ascend C kernel 内同步：
SetSyncFlag(magic, value, eventID)       // 设置同步标志（写远端 NPU 内存）
WaitSyncFlag(magic, value, eventID)      // 等待同步标志（轮询）
SetInnerFlag(magic, eventID)             // 同卡内同步
SetOuterFlag(magic, eventID)             // 跨卡同步
WaitAllRankOuterFlag(...)                // 全 rank 跨卡同步
```

**这等同于 MSCCL++ 的 GPU 侧 semaphore 信号量机制**——NPU kernel 可以自主设置和等待跨 NPU 同步标志，无需 CPU proxy。

#### 7.2.5 URMA 单边操作（跨节点 RDMA 替代）

| API | 作用 |
|------|------|
| `urma_write(jfs, tjetty, dst_tseg, src_tseg, dst, src, len, flag, ctx)` | 单边远程写 |
| `urma_read(jfs, ...)` | 单边远程读 |
| `urma_post_jfs_wr(jfs, wr, bad_wr)` | 提交 work request（支持 WRITE/READ/CAS/FAA/SEND） |

#### 7.2.6 URMA Jetty Group（类 multicast 发送）

```cpp
urma_create_jetty_grp(ctx, cfg)  // 创建 Jetty 组（最多 32 个 Jetty）
// 策略：URMA_JETTY_GRP_POLICY_RR（轮询）或 URMA_JETTY_GRP_POLICY_HASH_HINT（哈希）
```

Jetty Group 可以一次发送到组内多个 Jetty，类似 multicast 的发送端。

#### 7.2.7 CCU（Cube Communication Unit）

CCU 在 UDMA 层面提供专用 Jetty（`UDMA_CCU_JETTY_TYPE`），有更小的 WQE 大小（16 WQEBB），是硬件级群组通信加速单元。

### 7.3 CUDA/NVLS → 灵衢 UMDK 映射表

| CUDA/NVLS 概念 | 灵衢 UMDK 对应 | 说明 |
|----------------|---------------|------|
| cuMemMulticastCreate | obmem_export_memory + obmem_import_memory | OBMM 共享内存替代 multicast |
| cuMemMulticastAddDevice | obmem_import_memory（远端 CNA 导入） | 每个 NPU 导入同一导出内存 |
| cuMemMulticastBindAddr | peerMems[] + windowsIn/Out | HCCL Window + 直接内存访问 |
| cudaIpcGetMemHandle | obmem_export_memory（返回 token_id + uba） | UB 地址替代 IPC handle |
| cudaIpcOpenMemHandle | obmem_import_memory（通过 token_id + uba） | mmap 替代 IPC open |
| CUDA proxy semaphore | SyncCollectives (SetSyncFlag/WaitSyncFlag) | **NPU kernel 自主同步** |
| NVLink peer access | URMA import_seg + urma_write/read | 单边内存语义访问 |
| CUDA kernel → GPU | Ascend C kernel → NPU (AIV/CCU) | peerMems[] 提供跨 NPU 直接访问 |

### 7.4 灵衢超节点拓扑

```cpp
struct umq_node_id {
    uint32_t super_node_id;  // 灵衢超节点 ID
    uint32_t node_id;        // 超节点内的节点 ID
};
```

UMDK 明确支持超节点概念（`super_node_id`），路由基于超节点+节点双层寻址。`CAM_MAX_RANK_SIZE = 384` 表示单个超节点最多支持 384 个 NPU。

## 8. 迁移策略更新（基于 UMDK 分析）

### 8.1 核心结论：MSCCL++ 零拷贝设计**可以保留**

灵衢 UMDK 提供了 MSCCL++ 零拷贝所需的所有能力：

| MSCCL++ 需求 | 灵衢对应 | 可行性 |
|-------------|---------|--------|
| GPU kernel 直接访问远端内存 | peerMems[] + windowsIn/Out | ✓ 可行 |
| GPU kernel 自主同步（无 CPU proxy） | SyncCollectives SetSyncFlag/WaitSyncFlag | ✓ 可行 |
| IPC 内存共享 | obmem_export/import_memory | ✓ 可行 |
| Multicast 群组通信 | OBMM 共享内存 + Jetty Group | ✓ 可替代 |
| 跨节点 RDMA | URMA write/read | ✓ 可行 |

**推荐方案 C（混合方案）**：

- **同节点**：保留 MSCCL++ 零拷贝设计
  - memory_channel → OBMM 共享内存 + peerMems[]
  - port_channel → URMA import_seg + urma_write
  - switch_channel → OBMM 共享内存（多 NPU 导入同一 export）+ SyncCollectives
  - semaphore → SyncCollectives SetSyncFlag/WaitSyncFlag
- **跨节点**：URMA 单边操作（urma_write/read）

### 8.2 switch_channel 替代方案（确定）

**方案：OBMM 共享内存替代 NVLS multicast**

```
CUDA NVLS 流程：
1. cuMulticastCreate → 创建 multicast 内存
2. cuMulticastAddDevice → 多 GPU 加入
3. cuMulticastBindAddr → 绑定各 GPU 的缓冲区地址
4. 所有 GPU 通过同一 multicast 地址读写

灵衢替代流程：
1. obmem_export_memory → NPU A 导出共享内存（获得 token_id + uba）
2. obmem_import_memory → 其他 NPU 导入该内存（获得本地 mmap'd VA）
3. peerMems[] → 各 NPU kernel 通过 peerMems 直接读写共享区域
4. SyncCollectives → SetSyncFlag/WaitSyncFlag 进行同步
```

### 8.3 更新后的迁移优先级

| 优先级 | 阶段 | 内容 | UMDK 对应 |
|--------|------|------|----------|
| **P0** | 基础 API 适配 | gpu.hpp 新增 CANN 分支 | aclrt* API |
| **P1** | NCCL → HCCL shim | src/ext/nccl 替换 | HCCL API |
| **P2** | 构建系统 | CMake 新增 CANN 编译链 | CMake + UMDK |
| **P3** | IPC 内存 | CUDA IPC → OBMM 共享内存 | obmem_export/import |
| **P4** | 同步机制 | CUDA semaphore → SyncCollectives | SetSyncFlag/WaitSyncFlag |
| **P5** | Channel kernel | Ascend C 重写（peerMems[] 替代 cudaMemcpyToSymbol） | CommArgs.peerMems |
| **P6** | switch_channel | OBMM 共享内存替代 NVLS multicast | obmem + SyncCollectives |
| **P7** | allreduce/allgather | Ascend C 重写算法 kernel | AIV kernel + CCU |

## 9. 待确认事项（更新）

1. **灵衢超节点具体 NPU 型号**（910B/910-95？影响 UB 协议族支持）
2. **CANN FP8 支持**
3. **URMA 在 NPU kernel 内是否可用**（目前 UMDK 示例都在 host 侧，需确认 NPU kernel 能否调用 URMA）
4. **peerMems[] 在灵衢超节点是否支持 384 卡**（文档中 CAM_MAX_RANK_SIZE=384）
5. **OBMM 在灵衢超节点内的延迟特性**（vs NVLink multicast 延迟）

## 10. 下一步行动

| 步骤 | 内容 | 前置条件 |
|------|------|----------|
| 1 | 确认灵衢硬件特性和 UMDK 可用性（第 9 节） | 用户提供 |
| 2 | P0：gpu.hpp 新增 CANN 分支（aclrt* API 映射） | 步骤 1 |
| 3 | P1：NCCL shim → HCCL shim | 步骤 2 |
| 4 | P2：CMake 构建系统新增 CANN 编译链 | 步骤 2 |
| 5 | P3：IPC → OBMM 共享内存适配 | 步骤 2-4 |
| 6 | P4：semaphore → SyncCollectives 适配 | 步骤 5 |
| 7 | P5：memory_channel/port_channel kernel Ascend C 重写 | 步骤 5-6 |
| 8 | P6：switch_channel OBMM 替代 NVLS | 步骤 5-6 |
| 9 | P7：allreduce/allgather kernel Ascend C 重写 | 步骤 7 |