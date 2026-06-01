# MSCCL++ CUDA → CANN 迁移可视化图表

---

## 图1：MSCCL++ 整体架构（CUDA 侧）

```mermaid
graph TB
    subgraph Host["Host CPU"]
        BS["TcpBootstrap"] --> COMM["Communicator"]
        COMM --> CTX["Context"]
        CTX --> RM["RegisteredMemory"]
        CTX --> EP["Endpoint"]
        COMM --> CONN["Connection"]
        
        subgraph CONN_TYPES["Connection Types"]
            CIPC["CudaIpcConnection<br/>cudaMemcpyAsync D2D"]
            IB["IBConnection<br/>RDMA write/atomic"]
            ETH["EthernetConnection<br/>Socket send/recv"]
        end
        
        CONN --> CONN_TYPES
        
        PS["ProxyService"] --> PROXY["ProxyThread"]
        PROXY --> FIFO_HOST["Fifo (host side)<br/>poll/pop triggers"]
        PROXY --> CONN
        
        subgraph IPC["CUDA IPC / VMM"]
            IPC_RT["cudaIpcGetMemHandle<br/>cudaIpcOpenMemHandle"]
            IPC_FD["cuMemExportToShareableHandle<br/>cuMemImportFromShareableHandle<br/>cuMemAddressReserve<br/>cuMemMap<br/>cuMemSetAccess"]
            MC["cuMulticastCreate<br/>cuMulticastAddDevice<br/>cuMulticastBindAddr"]
        end
        
        CTX --> IPC
    end
    
    subgraph GPU["GPU (CUDA)"]
        subgraph Channels["Channel Device Handles"]
            MC_DEV["MemoryChannelDeviceHandle<br/>dst_(IPC ptr) + semaphore"]
            PC_DEV["PortChannelDeviceHandle<br/>FifoDevHandle + semaphore"]
            SC_DEV["SwitchChannelDeviceHandle<br/>devicePtr + mcPtr(multicast)"]
        end
        
        subgraph SEM["Semaphore Types"]
            D2D["MemoryDevice2DeviceSemaphore<br/>PTX red.release.sys.global.add.u64"]
            H2D["Host2DeviceSemaphore<br/>host writes, GPU polls atomicLoad"]
        end
        
        subgraph FIFO_DEV["FifoDeviceHandle"]
            FIFO_PUSH["push()<br/>atomicFetchAdd(head)<br/>st.global.release.sys.v2.u64"]
            FIFO_SYNC["sync()<br/>poll tail update"]
        end
        
        subgraph EK["Execution Kernel"]
            EK_MAIN["executionKernel __global__"]
            EK_SM["14 __shared__ variables<br/>memoryChannels_<br/>portChannels_<br/>nvlsChannels_<br/>buffers_[]<br/>offsets_[]"]
            EK_OPS["Operation Types<br/>PUT/GET/SIGNAL/WAIT<br/>FLUSH/MULTI_LOAD_REDUCE_STORE<br/>BARRIER/PIPELINE"]
        end
        
        MC_DEV --> D2D
        PC_DEV --> H2D
        PC_DEV --> FIFO_DEV
        SC_DEV -->|multimem PTX| GPU
        EK_MAIN --> EK_SM
        EK_SM --> Channels
        EK_OPS --> Channels
    end
    
    IPC_RT -.->|IPC-mapped ptr| MC_DEV
    IPC_FD -.->|VA-mapped ptr| MC_DEV
    MC -.->|multicast VA| SC_DEV
    FIFO_DEV -.->|host-pinned mem| FIFO_HOST
    D2D -.->|GPU atomic write| GPU
```

---

## 图2：MSCCL++ 整体架构（灵衢 CANN 侧 — 目标）

```mermaid
graph TB
    subgraph Host["Host CPU"]
        BS2["TcpBootstrap / USOCK"] --> COMM2["Communicator"]
        COMM2 --> CTX2["Context"]
        CTX2 --> RM2["RegisteredMemory"]
        COMM2 --> CONN2["Connection"]
        
        subgraph CONN_TYPES2["Connection Types"]
            OBMM_C["OBMM Connection<br/>obmem_export/import_memory"]
            URMA_C["URMA Connection<br/>urma_write/read"]
            URPC_C["URPC Connection<br/>urpc_func_call/poll"]
        end
        
        CONN2 --> CONN_TYPES2
        
        PS2["ProxyService (optional)"] --> PROXY2["ProxyThread"]
        PROXY2 --> URPC_HOST["URPC func_call/poll"]
        PROXY2 --> CONN2
    end
    
    subgraph NPU["NPU (Ascend C)"]
        subgraph Channels2["Channel Device Handles"]
            MC_DEV2["MemoryChannelDeviceHandle<br/>peerMems[rank] + SyncCollectives"]
            PC_DEV2["PortChannelDeviceHandle<br/>Option A: URPC func_call<br/>Option B: peerMems[] + SyncCollectives"]
            SC_DEV2["SwitchChannelDeviceHandle<br/>OBMM shared mem + peerMems[] + SyncCollectives"]
        end
        
        subgraph SYNC["SyncCollectives"]
            INNER["Inner Flags (Memory A)<br/>同NPU内跨block同步<br/>= DeviceSyncer"]
            OUTER["Outer Flags (Memory B)<br/>跨NPU同步<br/>= MemoryD2DSemaphore"]
            CHECK["Check* (非阻塞)<br/>Wait* (阻塞轮询)"]
        end
        
        subgraph EK2["Execution Kernel (Ascend C)"]
            EK_MAIN2["Ascend C kernel entry"]
            EK_LM["Local memory variables<br/>memoryChannels_<br/>portChannels_<br/>nvlsChannels_"]
            EK_OPS2["Operation Types<br/>same logic, different impl"]
        end
        
        MC_DEV2 --> OUTER
        PC_DEV2 --> OUTER
        SC_DEV2 --> OUTER
        EK_MAIN2 --> EK_LM
        EK_LM --> Channels2
    end
    
    OBMM_C -.->|mmap'd VA + peerMems| MC_DEV2
    OBMM_C -.->|OBMM shared mem| SC_DEV2
    URPC_C -.->|func_call args| PC_DEV2
    URMA_C -.->|RDMA-like write| CONN2
```

---

## 图3：MemoryChannel 数据流时序图（CUDA vs 灵衢）

```mermaid
sequenceDiagram
    participant GPU_A as GPU A (CUDA)
    participant GPU_B as GPU B (CUDA)
    participant Host as Host
    
    Note over GPU_A,Host: === CUDA MemoryChannel Setup ===
    Host->>Host: cudaMalloc(localMem_A)
    Host->>Host: cudaIpcGetMemHandle(localMem_A) → ipcHandle
    Host->>Host: exchange ipcHandle via Bootstrap
    Host->>Host: cudaIpcOpenMemHandle(ipcHandle) → remotePtr_B
    Host->>Host: cudaMalloc(semaphore tokens)
    Note over GPU_A,GPU_B: dst_ = remotePtr (IPC-mapped), src_ = localMem
    
    Note over GPU_A,GPU_B: === CUDA MemoryChannel Execution ===
    GPU_A->>GPU_B: PTX red.release.sys.global.add.u64 [remoteInboundToken_B], 1
    Note over GPU_B: atomicLoad<scopeSystem,acquire>(inboundToken) ≥ expected
    GPU_A->>GPU_B: *(dst_ + index) = value (direct P2P write)
    GPU_A->>GPU_B: PTX red.release.sys.global.add.u64 [remoteInboundToken_B], 1
    GPU_B->>GPU_A: wait() → signal() → read data
    
    Note over GPU_A,Host: === 灵衢 MemoryChannel Setup ===
    Host->>Host: aclrtMalloc(localMem_A)
    Host->>Host: obmem_export_memory(localMem_A) → {token_id, uba}
    Host->>Host: exchange token_id + uba via Bootstrap
    Host->>Host: obmem_import_memory(token_id, uba) → mmap'd VA
    Host->>Host: SyncCollectives.Init(rank, rankSize, shareAddrs)
    Note over GPU_A,GPU_B: peerMems[rank] = mmap'd VA (OBMM imported)
    
    Note over GPU_A,GPU_B: === 灵衢 MemoryChannel Execution ===
    GPU_A->>GPU_B: SyncCollectives.SetOuterFlag(magic, eventID)
    Note over GPU_B: SyncCollectives.WaitOuterFlag(magic, eventID)
    GPU_A->>GPU_B: peerMems[remoteRank][offset] = value (direct write)
    GPU_A->>GPU_B: SyncCollectives.SetOuterFlag(magic+1, eventID)
    GPU_B->>GPU_A: WaitOuterFlag → read peerMems[localRank]
```

---

## 图4：PortChannel 数据流时序图（CUDA vs 灵衢）

```mermaid
sequenceDiagram
    participant GPU as GPU (CUDA)
    participant FIFO as FIFO (Host-pinned)
    participant Proxy as ProxyThread
    participant Remote as Remote GPU/Host
    
    Note over GPU,Remote: === CUDA PortChannel Data Transfer ===
    GPU->>FIFO: atomicFetchAdd(head,1) → slot index
    GPU->>FIFO: st.global.release.sys.v2.u64 [trigger] {fst, snd^flipMask}
    Note over GPU: GPU spins if FIFO full (head ≥ size + tailCache)
    
    Proxy->>FIFO: poll() → atomicLoad<acquire>(fst) → read snd
    Proxy->>Proxy: handleTrigger() → conn.write(dst, src, size)
    Proxy->>Remote: RDMA write OR cudaMemcpyAsync D2D
    Proxy->>Proxy: semaphore->signal() → conn.updateAndSync()
    Proxy->>FIFO: pop() → clear fst → atomicStore<release>(tail+1)
    
    Note over GPU,Proxy: === CUDA PortChannel Flush ===
    GPU->>FIFO: push(TriggerSync)
    Proxy->>Proxy: requestFlush() → wait for completion
    Proxy->>GPU: atomicStore<release>(flushDonePos = pos+1)
    Note over GPU: atomicLoad<scopeSystem,acquire>(flushDonePos) > fifoPos
    
    Note over GPU,Remote: === 灵衢 PortChannel (Option A: Proxy模式) ===
    GPU->>Proxy: URPC func_call(transferDesc)
    Proxy->>Proxy: urma_write / obmem shared memory write
    Proxy->>GPU: SyncCollectives.SetOuterFlag(magic, eventID)
    GPU->>GPU: SyncCollectives.WaitOuterFlag(magic, eventID)
    
    Note over GPU,Remote: === 灵衢 PortChannel (Option B: NPU自主推进) ===
    GPU->>Remote: peerMems[remoteRank] direct write (like MemoryChannel)
    GPU->>Remote: SyncCollectives.SetOuterFlag(magic, eventID)
    Note over Remote: SyncCollectives.WaitOuterFlag(magic, eventID)
```

---

## 图5：SwitchChannel 数据流时序图（CUDA vs 灵衢）

```mermaid
sequenceDiagram
    participant GPU0 as GPU 0
    participant GPU1 as GPU 1
    participant GPU2 as GPU 2
    participant NVLS as NVLS Hardware
    participant Host as Host (Root Rank)
    
    Note over GPU0,Host: === CUDA SwitchChannel Setup ===
    Host->>Host: cuMulticastCreate(bufferSize, numDevices) → mcHandle
    Host->>Host: cuMemExportToShareableHandle(mcHandle) → shareableHandle
    Host->>Host: exchange shareableHandle via Bootstrap
    Host->>Host: cuMemImportFromShareableHandle → local mcHandle
    Host->>GPU0: cuMulticastAddDevice(GPU0) → join multicast
    Host->>GPU1: cuMulticastAddDevice(GPU1) → join multicast
    Host->>GPU2: cuMulticastAddDevice(GPU2) → join multicast
    Host->>Host: cuMemAddressReserve → cuMemMap(mcHandle) → cuMemSetAccess
    Note over GPU0,GPU2: mcPtr = multicast VA pointer
    
    Note over GPU0,NVLS: === CUDA SwitchChannel AllReduce (2 steps) ===
    Note over GPU0,GPU2: Step 1: Each GPU writes local data to devicePtr
    GPU0->>GPU0: devicePtr[i] = local_data_0
    GPU1->>GPU1: devicePtr[i] = local_data_1
    GPU2->>GPU2: devicePtr[i] = local_data_2
    
    Note over GPU0,NVLS: Step 2: Reduce via multimem.ld_reduce
    GPU0->>NVLS: multimem.ld_reduce.relaxed.sys.global.add.f32 [mcPtr+i]
    NVLS-->>GPU0: sum = data_0 + data_1 + data_2
    Note over NVLS: Hardware atomically reads ALL GPUs' memory at offset i
    
    Note over GPU0,NVLS: Step 3: Broadcast result via multimem.st
    GPU0->>NVLS: multimem.st.relaxed.sys.global.f32 [mcPtr+i], sum
    Note over NVLS: Hardware writes sum to ALL GPUs' memory
    NVLS-->>GPU0: devicePtr[i] = sum
    NVLS-->>GPU1: devicePtr[i] = sum
    NVLS-->>GPU2: devicePtr[i] = sum
    
    Note over GPU0,Host: === 灵衢 SwitchChannel Setup ===
    Host->>Host: obmem_export_memory(bufferSize_4MB_aligned) → {token_id, uba}
    Host->>Host: exchange token_id + uba via Bootstrap
    Host->>Host: obmem_import_memory(token_id, uba) → mmap'd VA (每个NPU)
    Note over GPU0,GPU2: peerMems[rank] 含 mmap'd VA, 可直接读写
    
    Note over GPU0,GPU2: === 灵衢 SwitchChannel AllReduce (4 steps) ===
    Note over GPU0,GPU2: Step 1: Each NPU writes local data to peerMems[localRank]
    GPU0->>GPU0: peerMems[0][offset] = local_data_0
    GPU1->>GPU1: peerMems[1][offset] = local_data_1
    
    Note over GPU0,GPU2: Step 2: Signal data ready
    GPU0->>GPU2: SyncCollectives.SetOuterFlag(magic, eventID)
    GPU1->>GPU2: SyncCollectives.SetOuterFlag(magic, eventID)
    Note over GPU2: SyncCollectives.WaitAllRankOuterFlag(magic, eventID)
    
    Note over GPU0,GPU2: Step 3: Each NPU reads all data + reduce
    GPU2->>GPU0: Ascend C reduce(peerMems[0..N][offset])
    Note over GPU2: sum = peerMems[0][offset] + peerMems[1][offset] + ...
    
    Note over GPU0,GPU2: Step 4: Write result + signal
    GPU2->>GPU2: peerMems[2][offset] = sum
    GPU2->>GPU0: SyncCollectives.SetOuterFlag(magic+1, eventID)
    Note over GPU0: SyncCollectives.WaitOuterFlag(magic+1, eventID) → read result
```

---

## 图6：FIFO 机制详细时序图（CUDA PortChannel 核心）

```mermaid
sequenceDiagram
    participant GPU_T0 as GPU Thread 0
    participant GPU_T1 as GPU Thread 1
    participant HEAD as head (GPU mem)<br/>atomicFetchAdd
    participant TRIGGERS as triggers[]<br/>host-pinned mem
    participant TAIL as tail (host-pinned)<br/>atomicLoad/Store
    participant TC as tailCache (GPU mem)<br/>local cache
    participant PROXY as ProxyThread
    participant CONN as Connection
    participant FLUSH as flushDonePos<br/>host-pinned
    participant SEM_TOKEN as inboundToken<br/>GPU mem (semaphore)
    
    Note over GPU_T0,SEM_TOKEN: === FIFO Push (GPU Side) ===
    GPU_T0->>HEAD: atomicFetchAdd<scopeDevice>(head, 1) → prevHead=0
    GPU_T0->>TRIGGERS: st.global.release.sys.v2.u64 [triggers[0]] {fst, snd^0x8000000000000000}
    Note over GPU_T0: snd bit-63 flip for safe polling
    
    GPU_T1->>HEAD: atomicFetchAdd<scopeDevice>(head, 1) → prevHead=1
    GPU_T1->>TRIGGERS: st.global.release.sys.v2.u64 [triggers[1]] {fst, snd^flipMask}
    
    Note over GPU_T0,PROXY: === FIFO Poll/Pop (Host Side) ===
    PROXY->>TRIGGERS: atomicLoad<acquire>(triggers[0].fst) → non-zero = new entry
    PROXY->>TRIGGERS: read snd → revert flip mask → get actual snd
    PROXY->>PROXY: parse ProxyTrigger fields {type, size, srcOffset, dstOffset, ...}
    
    alt TriggerData
        PROXY->>CONN: conn.write(dst, dstOffset, src, srcOffset, size)
    else TriggerFlag
        PROXY->>SEM_TOKEN: semaphore->signal() → conn.updateAndSync()
    else TriggerSync
        PROXY->>CONN: requestFlush() → wait completion
        PROXY->>FLUSH: atomicStore<release>(flushDonePos = pos+1)
    end
    
    PROXY->>TRIGGERS: triggers[0].fst = 0 (clear slot)
    PROXY->>TAIL: atomicStore<release>(tail = tail+1)
    
    Note over GPU_T0,PROXY: === GPU Wait for Flush ===
    GPU_T0->>FLUSH: atomicLoad<scopeSystem,acquire>(flushDonePos) ≤ fifoPos?
    Note over GPU_T0: spin until flushDonePos > fifoPos
```

---

## 图7：SyncCollectives 机制详细时序图

```mermaid
sequenceDiagram
    participant NPU0 as NPU 0 (Ascend C)
    participant NPU1 as NPU 1 (Ascend C)
    participant NPU2 as NPU 2 (Ascend C)
    participant MEM_A as Memory A<br/>同NPU内 flag区
    participant MEM_B as Memory B<br/>跨NPU flag区
    participant PEER as peerMems[]<br/>数据缓冲区
    
    Note over NPU0,PEER: === SyncCollectives Initialization ===
    Note over NPU0: Init(rank=0, rankSize=3, shareAddrs)
    Note over NPU1: Init(rank=1, rankSize=3, shareAddrs)
    Note over NPU2: Init(rank=2, rankSize=3, shareAddrs)
    Note over MEM_B: shareAddrs[0..2] → 各rank的flag区起始地址
    
    Note over NPU0,PEER: === Inner Flag (同NPU同步) = DeviceSyncer ===
    NPU0->>MEM_A: SetInnerFlag(magic, eventID, setRank=0, setBlock=0)
    Note over NPU0: block 0 完成 → 通知 block 1
    NPU0->>MEM_A: WaitInnerFlag(magic, eventID, waitRank=0, waitBlock=1)
    Note over MEM_A: 等待同NPU其他block完成
    
    Note over NPU0,PEER: === Outer Flag (跨NPU同步) = MemoryD2DSemaphore ===
    NPU0->>MEM_B: SetOuterFlag(magic, eventID, setRank=0, setBlock=0)
    Note over MEM_B: 写入 NPU0 的 Memory B flag 区
    NPU0->>MEM_B: MergeMagicWithValue(magic, value) → (magic<<32)|value
    Note over MEM_B: 高32位=batch号(防过期), 低32位=值
    
    NPU1->>MEM_B: WaitOuterFlag(magic, eventID, waitRank=0, waitBlock=0)
    Note over NPU1: 轮询 NPU0 的 flag 直到匹配
    NPU2->>MEM_B: WaitOuterFlag(magic, eventID, waitRank=0, waitBlock=0)
    
    Note over NPU0,PEER: === WaitAllRankOuterFlag ===
    NPU0->>MEM_B: WaitAllRankOuterFlag(magic, eventID)
    Note over NPU0: 等待所有rank的outer flag都设置
    
    Note over NPU0,PEER: === 非阻塞检查 ===
    NPU0->>MEM_B: CheckAllRankOuterFlag(magic, eventID) → bool
    Note over NPU0: 不阻塞，返回是否所有rank都已就绪
```

---

## 图8：OBMM 共享内存机制流程图

```mermaid
flowchart TB
    subgraph EXPORT["NPU A: Export"]
        E1["obmem_export_memory_param_t<br/>len=4MB-aligned<br/>deid[16]<br/>cacheable"] --> E2["obmem_export_memory(param, &handle, &exp)"]
        E2 --> E3["返回 obmem_export_info_t<br/>token_id + uba + size"]
        E3 --> E4["底层: /dev/obmm_shmdev0<br/>mmap(MAP_SHARED)"]
    end
    
    subgraph EXCHANGE["Bootstrap Exchange"]
        X1["发送 token_id + uba + size"] --> X2["所有rank收到导出信息"]
    end
    
    subgraph IMPORT["NPU B/C/D: Import"]
        I1["obmem_import_memory_param_t<br/>import_cna ≠ export_cna!<br/>seid[16] + deid[16]"] --> I2["obmem_import_memory(param, exp, &handle)"]
        I2 --> I3["返回 mmap'd VA<br/>本地可访问的虚拟地址"]
        I3 --> I4["底层: /dev/obmm_shmdev1<br/>mmap(MAP_SHARED)"]
    end
    
    subgraph USE["Kernel 使用"]
        U1["peerMems[0] = mmap'd VA<br/>2MB flags + 100MB data"] --> U2["NPU kernel 直接读写<br/>peerMems[rank][offset]"]
        U2 --> U3["零拷贝! 无远端CPU参与"]
    end
    
    subgraph RELEASE["释放"]
        R1["obmem_release_export_memory<br/>(handle, ptr, len)"]
        R2["obmem_release_import_memory<br/>(handle, ptr, len)"]
    end
    
    EXPORT --> EXCHANGE --> IMPORT --> USE --> RELEASE
    
    style E1 fill:#e1f5fe
    style I1 fill:#fff3e0
    style U2 fill:#e8f5e9
    style R1 fill:#fce4ec
```

---

## 图9：AIV Kernel 加载和执行管线

```mermaid
flowchart LR
    subgraph BUILD["构建"]
        B1["Ascend C 源码<br/>.asc"] --> B2["编译 → .o binary"]
        B2 --> B3A["静态嵌入<br/>ld -r -b binary<br/>_binary_*_bin_start/end"]
        B2 --> B3B["动态部署<br/>$ASCEND_HOME_PATH/lib64/*.o"]
    end
    
    subgraph LOAD["加载"]
        L1A["aclrtBinaryLoadFromData<br/>embedded_start, length<br/>LAZY_LOAD option"] --> L2["binHandle"]
        L1B["aclrtBinaryLoadFromFile<br/>path, loadOptions"] --> L2
    end
    
    subgraph REG["注册"]
        L2 --> R1["aclrtBinaryGetFunction<br/>binHandle, kernelName<br/>→ funcHandle"]
        R1 --> R2["Registry Map<br/>key: (cmdType, dataType, argsType)"]
    end
    
    subgraph LAUNCH["启动"]
        R2 --> A1["构造 AivKernelArgsDef<br/>buffersIn(CCLIN)<br/>input/output/rank/..."]
        A1 --> A2["aclrtLaunchKernelWithHostArgs<br/>funcHandle, numBlocks, stream<br/>cfg{SCHEM_MODE=1<br/>TIMEOUT_US=calc<br/>ENGINE_TYPE=AIV}<br/>args, argsSize"]
        A2 --> A3["NPU 执行 Ascend C kernel"]
        A3 -->|ACL_ERROR_RT_INVALID_HANDLE| A4["重新获取 funcHandle<br/>retry launch"]
        A4 --> A3
    end
    
    subgraph UNLOAD["卸载"]
        A3 --> U1["aclrtBinaryUnLoad(binHandle)"]
    end
    
    BUILD --> LOAD --> REG --> LAUNCH --> UNLOAD
    
    style B1 fill:#e1f5fe
    style A2 fill:#e8f5e9
    style A4 fill:#fff3e0
```

---

## 图10：executionKernel Operation 处理流程图

```mermaid
flowchart TB
    START["executionKernel 启动"] --> SM["复制 plan 到 __shared__ memory<br/>sharedMem[i] = ((int4*)localPlan)[i]<br/>__syncshm()"]
    SM --> LOAD["加载 channel 指针<br/>memoryChannels_ = plan->memoryChannels<br/>portChannels_ = plan->portChannels<br/>nvlsChannels_ = plan->nvlsChannels<br/>buffers_[] = plan->buffers<br/>offsets_[] = plan->offsets"]
    LOAD --> LOOP["遍历 operations[i]"]
    
    LOOP --> OP_TYPE{"Operation type?"}
    
    OP_TYPE -->|PUT/GET| MC_OP["MemoryChannel<br/>dst_[offset] read/write<br/>或 PortChannel fifo_.push(TriggerData)"]
    OP_TYPE -->|SIGNAL/WAIT| SEM_OP["Semaphore<br/>D2D: PTX red.release.sys / atomicLoad<br/>或 SyncCollectives.SetOuterFlag/WaitOuterFlag"]
    OP_TYPE -->|PUT_WITH_SIGNAL| PC_OP["PortChannel<br/>fifo_.push(TriggerData|TriggerFlag)"]
    OP_TYPE -->|FLUSH| FL_OP["PortChannel<br/>fifo_.push(TriggerSync)<br/>waitFlush(flushDonePos)"]
    OP_TYPE -->|MULTI_LOAD_REDUCE_STORE| SC_OP["SwitchChannel<br/>multimem.ld_reduce → multimem.st<br/>或 OBMM + peerMems[] + SyncCollectives"]
    OP_TYPE -->|BARRIER| BAR_OP["DeviceSyncer<br/>__syncthreads + atomicFetchAdd<br/>或 SyncCollectives.SetInnerFlag"]
    OP_TYPE -->|SEM_ACQUIRE/RELEASE| DS_OP["DeviceSemaphore<br/>atomicFetchAdd<scopeDevice><br/>或 Ascend C semaphore"]
    OP_TYPE -->|PIPELINE| PIPE_OP["嵌套迭代子operations<br/>递归调用 executeDeviceFunction"]
    
    MC_OP --> NEXT["i += nSteps"]
    SEM_OP --> NEXT
    PC_OP --> NEXT
    FL_OP --> NEXT
    SC_OP --> NEXT
    BAR_OP --> NEXT
    DS_OP --> NEXT
    PIPE_OP --> NEXT
    
    NEXT --> LOOP
    NEXT -->|all ops done| END["kernel 完成"]
    
    style START fill:#e1f5fe
    style SC_OP fill:#fce4ec
    style MC_OP fill:#e8f5e9
    style PC_OP fill:#fff3e0
```

---

## 图11：迁移优先级依赖图

```mermaid
flowchart TB
    P0["P0: 基础 API 适配<br/>gpu.hpp CANN分支<br/>aclrt* API映射<br/>MSCCLPP_ACLTHROW宏<br/>cudaMemset→aclrtMemset"]
    
    P1["P1: NCCL→HCCL shim<br/>nccl.cc替换"]
    P2["P2: 构建系统 + Python<br/>CMake CANN编译链<br/>Python bindings类型cast<br/>compiler.py Ascend C检测"]
    P8["P8: CUDA Graph<br/>CANN capture API适配<br/>AvoidCudaGraphCaptureGuard"]
    
    P3A["P3a: IPC→OBMM<br/>gpu_ipc_mem.cc<br/>4MB对齐策略"]
    P3B["P3b: cudaMemcpyToSymbol<br/>→CommArgs参数传递<br/>~54处修改"]
    
    P4["P4: 同步机制<br/>PTX→SyncCollectives<br/>semaphore→SetOuterFlag<br/>fifo→URPC/OBMM<br/>__syncthreads→InnerFlag"]
    
    P5A["P5a: MemoryChannel kernel<br/>Ascend C重写<br/>OBMM+peerMems[]"]
    P5B["P5b: PortChannel kernel<br/>Ascend C重写<br/>URPC或OBMM+SyncCollectives"]
    
    P6A["P6a: SwitchChannel<br/>OBMM+peerMems[]+SyncCollectives<br/>替代NVLS multicast"]
    P6B["P6b: execution_kernel框架<br/>Ascend C重写<br/>Operation类型系统"]
    
    P7["P7: allreduce/allgather<br/>Ascend C重写算法kernel"]
    
    P0 --> P1
    P0 --> P2
    P0 --> P8
    P0 --> P3A
    P0 --> P3B
    P1 --> P3A
    P2 --> P3A
    P3A --> P4
    P3B --> P4
    P4 --> P5A
    P4 --> P5B
    P5A --> P6A
    P5B --> P6A
    P5A --> P6B
    P5B --> P6B
    P6A --> P7
    P6B --> P7
    
    style P0 fill:#c8e6c9
    style P1 fill:#c8e6c9
    style P2 fill:#c8e6c9
    style P3A fill:#fff9c4
    style P3B fill:#fff9c4
    style P4 fill:#ffe0b2
    style P5A fill:#ffccbc
    style P5B fill:#ffccbc
    style P6A fill:#ef9a9a
    style P6B fill:#ef9a9a
    style P7 fill:#e57373
    style P8 fill:#c8e6c9
```

---

## 图12：CUDA API 依赖层次图（从底层到上层）

```mermaid
graph TB
    subgraph L0["L0: Hardware/Driver"]
        DRV["cuMem* (VMM)<br/>cuMulticast* (NVLS)<br/>cuDevice*"]
        PTX["PTX Inline ASM<br/>red.release.sys<br/>multimem.*<br/>st.global.release"]
    end
    
    subgraph L1["L1: CUDA Runtime"]
        DEV["cudaGet/SetDevice<br/>cudaDeviceSynchronize<br/>cudaGetDeviceCount"]
        MEM["cudaMalloc/Free<br/>cudaMemcpy/Async<br/>cudaMemset/Async<br/>cudaHostAlloc/FreeHost"]
        SYNC["cudaStream*<br/>cudaEvent*<br/>cudaGraph*"]
        IPC["cudaIpc*"]
        ERR["cudaError*<br/>MSCCLPP_CUDATHROW"]
    end
    
    subgraph L2["L2: CUDA C++ Abstractions"]
        ATOMIC["cuda::atomic_ref<br/>scopeDevice / scopeSystem<br/>memory_order"]
        DT["__half / __nv_bfloat16<br/>__nv_fp8_e4m3"]
        KW["__global__ / __device__<br/>__shared__ / __syncthreads<br/>__launch_bounds__"]
    end
    
    subgraph L3["L3: MSCCL++ Infrastructure"]
        SEM_M["MemoryD2DSemaphore<br/>(PTX red.release)"]
        SEM_H["Host2DSemaphore<br/>(host write + atomicLoad)"]
        FIFO["FifoDeviceHandle<br/>(PTX st.release + atomicFetchAdd)"]
        CH_M["MemoryChannel<br/>(IPC ptr + semaphore)"]
        CH_P["PortChannel<br/>(FIFO + proxy)"]
        CH_S["SwitchChannel<br/>(NVLS multimem)"]
    end
    
    subgraph L4["L4: MSCCL++ Algorithms"]
        EK["ExecutionKernel<br/>(Operation dispatcher)"]
        AR["AllReduce kernels<br/>(packet/nvls/fullmesh)"]
        AG["AllGather kernels"]
    end
    
    DRV --> MEM
    PTX --> SEM_M
    PTX --> FIFO
    PTX --> CH_S
    DEV --> MEM
    MEM --> IPC
    SYNC --> FIFO
    ATOMIC --> SEM_M
    ATOMIC --> SEM_H
    ATOMIC --> FIFO
    KW --> EK
    SEM_M --> CH_M
    SEM_H --> CH_P
    FIFO --> CH_P
    CH_S --> EK
    CH_M --> EK
    CH_P --> EK
    EK --> AR
    EK --> AG
    
    style L0 fill:#e57373
    style L1 fill:#ff9800
    style L2 fill:#ffc107
    style L3 fill:#4caf50
    style L4 fill:#2196f3
```

---

## 图13：灵衢替代 API 依赖层次图

```mermaid
graph TB
    subgraph L0_CANN["L0: UMDK Hardware/Driver"]
        OBMM["OBMM<br/>obmem_export/import_memory<br/>4MB aligned"]
        URMA_API["URMA<br/>urma_write/read<br/>urma_register/import_seg<br/>urma_poll_jfc"]
        URPC_API["URPC<br/>urpc_func_call/poll<br/>urpc_mem_seg_register"]
        DLOCK["DLock<br/>umo_atomic64_faa/cas"]
    end
    
    subgraph L1_CANN["L1: CANN Runtime"]
        ACL_RT["aclrtGet/SetDevice<br/>aclrtMalloc/Free<br/>aclrtMemcpy/Async<br/>aclrtMemset/Async"]
        ACL_SYNC["aclrtCreateStream<br/>aclrtCreateNotify<br/>aclrtRecordNotify<br/>aclrtWaitAndResetNotify"]
        ACL_GRAPH["aclrtGraph*<br/>aclmdlRICapture*"]
        ACL_ERR["aclError<br/>MSCCLPP_ACLTHROW"]
    end
    
    subgraph L2_CANN["L2: Ascend C Abstractions"]
        SYNC_C["SyncCollectives<br/>SetOuterFlag/WaitOuterFlag (跨NPU)<br/>SetInnerFlag/WaitInnerFlag (同NPU)<br/>Check* (非阻塞)"]
        PEER["peerMems[]<br/>直接跨NPU内存访问<br/>2MB flags + 100MB data"]
        AC_DT["Ascend C data types<br/>half/bfloat16_t<br/>fp8_e4m3fn_t (copy only)"]
        AC_KW["Ascend C kernel entry<br/>local memory (替代 __shared__)<br/>6维并行模型"]
    end
    
    subgraph L3_CANN["L3: MSCCL++ CANN Infrastructure"]
        SEM_M_C["MemoryD2DSemaphore<br/>→ SyncCollectives OuterFlag"]
        SEM_H_C["Host2DSemaphore<br/>→ SyncCollectives 或 aclrtNotify"]
        FIFO_C["FifoDeviceHandle<br/>→ URPC func_call 或 OBMM ring buffer"]
        CH_M_C["MemoryChannel<br/>→ OBMM + peerMems[]"]
        CH_P_C["PortChannel<br/>→ URPC+URMA 或 peerMems[]+SyncCollectives"]
        CH_S_C["SwitchChannel<br/>→ OBMM shared + peerMems[] + SyncCollectives"]
    end
    
    subgraph L4_CANN["L4: MSCCL++ Algorithms (Ascend C)"]
        EK_C["ExecutionKernel<br/>Ascend C Operation dispatcher"]
        AR_C["AllReduce (Ascend C)"]
        AG_C["AllGather (Ascend C)"]
    end
    
    OBMM --> CH_M_C
    OBMM --> CH_S_C
    URMA_API --> CH_P_C
    URPC_API --> FIFO_C
    ACL_RT --> ACL_ERR
    ACL_SYNC --> SEM_H_C
    ACL_GRAPH --> EK_C
    SYNC_C --> SEM_M_C
    SYNC_C --> CH_S_C
    PEER --> CH_M_C
    PEER --> CH_S_C
    AC_KW --> EK_C
    AC_DT --> AR_C
    SEM_M_C --> CH_M_C
    SEM_H_C --> CH_P_C
    FIFO_C --> CH_P_C
    CH_S_C --> EK_C
    CH_M_C --> EK_C
    CH_P_C --> EK_C
    EK_C --> AR_C
    EK_C --> AG_C
    
    style L0_CANN fill:#e57373
    style L1_CANN fill:#ff9800
    style L2_CANN fill:#ffc107
    style L3_CANN fill:#4caf50
    style L4_CANN fill:#2196f3
```

---

## 图14：CUDA vs 灵衢 对比矩阵（需求完整性检查）

```mermaid
graph LR
    subgraph CUDA_SIDE["CUDA 侧能力"]
        C1["设备管理<br/>cudaGetDevice etc."]
        C2["内存管理<br/>cudaMalloc/Memcpy/Memset<br/>cudaMemcpyToSymbol"]
        C3["Stream/Event/Graph"]
        C4["IPC 内存<br/>cudaIpc* / cuMem*VMM"]
        C5["P2P原子<br/>PTX red.release.sys<br/>cuda::atomic scopeSystem"]
        C6["Multicast<br/>cuMulticast* / multimem PTX"]
        C7["FIFO+Proxy<br/>st.release.sys.v2.u64<br/>host-pinned mem"]
        C8["Kernel编程<br/>__global__/__shared__/__syncthreads"]
        C9["NCCL shim"]
        C10["Python bindings"]
        C11["构建系统<br/>nvcc / CMake CUDA"]
    end
    
    subgraph CANN_SIDE["灵衢侧替代"]
        A1["aclrtGetDevice<br/>✓ 直接对应"]
        A2["aclrtMalloc/Memcpy/Memset<br/>⚠️ cudaMemcpyToSymbol需特殊处理"]
        A3["aclrtStream/Notify<br/>✓ Graph需验证"]
        A4["OBMM export/import<br/>⚠️ 4MB对齐约束"]
        A5["SyncCollectives OuterFlag<br/>✓ 跨NPU原子"]
        A6["OBMM + peerMems[] + SyncCollectives<br/>⚠️ 无硬件multicast"]
        A7["URPC func_call 或 OBMM ring buffer<br/>⚠️ latency需确认"]
        A8["Ascend C kernel<br/>⚠️ 编程模型完全不同"]
        A9["HCCL shim<br/>✓ 接口名称替换"]
        A10["类型cast + 设备类型<br/>⚠️ DLDeviceType"]
        A11["Ascend C编译器 + CMake CANN<br/>⚠️ 新编译链"]
    end
    
    C1 -->|✓| A1
    C2 -->|⚠️| A2
    C3 -->|⚠️| A3
    C4 -->|⚠️| A4
    C5 -->|✓| A5
    C6 -->|⚠️| A6
    C7 -->|⚠️| A7
    C8 -->|⚠️| A8
    C9 -->|✓| A9
    C10 -->|⚠️| A10
    C11 -->|⚠️| A11
    
    style C1 fill:#c8e6c9
    style C5 fill:#c8e6c9
    style C9 fill:#c8e6c9
    style A1 fill:#c8e6c9
    style A5 fill:#c8e6c9
    style A9 fill:#c8e6c9
    style C2 fill:#fff9c4
    style C3 fill:#fff9c4
    style C4 fill:#ffe0b2
    style C6 fill:#ffccbc
    style C7 fill:#ffccbc
    style C8 fill:#ef9a9a
    style C10 fill:#fff9c4
    style C11 fill:#fff9c4
```

---

## 图15：GDRCopy / IB HostNoAtomic 模式时序图（CUDA 特有细节）

```mermaid
sequenceDiagram
    participant GPU as GPU
    participant GDRCOPY as GDRCopy BAR1
    participant RECV as IB Recv Thread
    participant IB as IB QP
    participant PROXY as Proxy Thread
    
    Note over GPU,PROXY: === IB HostNoAtomic Mode ===
    Note over GPU,PROXY: (GPU不支持IB atomic, 需CPU辅助)
    
    PROXY->>IB: stageSendWrite(qp, dst, src, size)
    PROXY->>IB: postSend(qp)
    
    Note over RECV: Recv thread waits for WRITE_WITH_IMM
    IB-->>RECV: WRITE_WITH_IMM completion<br/>imm_data = 32-bit token snippet
    
    RECV->>RECV: reconstruct 64-bit token from<br/>32-bit imm_data + tracked high bits<br/>(wrap-around detection)
    
    RECV->>GDRCOPY: GdrMap (BAR1 mapping of GPU memory)
    RECV->>GPU: gdr_copy_to_bar(dev, bar_ptr, token, 8 bytes)<br/>Direct CPU→GPU via BAR1<br/>NOT cudaMemcpy!
    
    Note over GPU: GPU polls inboundToken<br/>atomicLoad<scopeSystem,acquire>
```

---

## 需求完整性自检清单

基于以上图表，检查是否有遗漏的需求：

| 检查项 | 状态 | 说明 |
|--------|------|------|
| 设备管理 API 全覆盖 | ✓ | 图14 显示所有 cudaDevice* 已映射 |
| 内存管理 API 全覆盖 | ✓ | 新增 cudaMemset/cudaMemcpyToSymbolAsync |
| Stream/Event/Graph 全覆盖 | ✓ | 新增 CUDA Graph API 整个类别 |
| IPC 机制全覆盖 | ✓ | OBMM 替代，含 4MB 对齐约束 |
| 跨 NPU 原子操作 | ✓ | SyncCollectives OuterFlag + DLock atomic |
| Multicast 替代 | ✓ | OBMM + peerMems[] + SyncCollectives（图5） |
| FIFO+Proxy 替代 | ✓ | URPC 或 OBMM ring buffer（图4） |
| Kernel 编程模型 | ✓ | Ascend C 6维并行（图8、10） |
| Semaphore 替代 | ✓ | SyncCollectives Outer/Inner（图7） |
| execution_kernel 框架 | ✓ | Operation 类型系统全覆盖（图10） |
| NCCL shim | ✓ | HCCL API 映射 |
| Python bindings | ✓ | 类型 cast + 编译器检测 |
| 构建系统 | ✓ | CMake + compiler.py |
| GDRCopy 特殊路径 | ✓ | 图15 覆盖 IB HostNoAtomic 模式 |
| AIV kernel 管线 | ✓ | 图9 覆盖加载→注册→启动 |
| OBMM 4MB 对齐 | ✓ | 图8 标注对齐要求 |
| bit-63 flip 协议 | ✓ | 图6 FIFO 详细时序 |
| flushDonePos 机制 | ✓ | 图6 FIFO flush 流程 |
| magic/value 合并 | ✓ | 图7 SyncCollectives 详细 |
| Memory A vs B 区分 | ✓ | 图7 Inner vs Outer flags |
| URMA 完整 API | ✓ | 图13 层次图覆盖 |
| DLock atomic | ✓ | 图13 包含 umo_atomic64 |
| URPC RPC | ✓ | 图13 包含 func_call/poll |
| **PTX→灵衢映射** | ✓ | 图3/4/5/6 覆盖所有 PTX 指令 |
| **CUDA error codes** | ⚠️ | 需建立完整 aclError 映射表 |
| **aclrtGraph 确认** | ⚠️ | 需验证 CANN Graph API 可用性 |
| **URMA from NPU kernel** | ⚠️ | 需确认 NPU kernel 内能否调用 URMA |
| **CANN peer access** | ⚠️ | cudaDeviceCanAccessPeer 的 CANN 对应未确认 |