// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef MSCCLPP_GPU_HPP_
#define MSCCLPP_GPU_HPP_

#include <mscclpp/device.hpp>

#if defined(MSCCLPP_DEVICE_CANN)

#include <acl/acl.h>
#include <unistd.h>
#include <limits.h>
#include <string>
#include <algorithm>

using cudaError_t = aclError;
using cudaEvent_t = aclrtNotify;
using cudaStream_t = aclrtStream;
using cudaMemcpyKind = aclrtMemcpyKind;
using cudaIpcMemHandle_t = uint64_t;

using CUresult = aclError;
using CUdevice = int;
using CUdeviceptr = void*;
using CUmemGenericAllocationHandle = uint64_t;

enum cudaStreamCaptureMode { cudaStreamCaptureModeGlobal = 0, cudaStreamCaptureModeRelaxed = 1 };
enum cudaStreamCaptureStatus { cudaStreamCaptureStatusNone = 0, cudaStreamCaptureStatusActive = 1 };
struct cudaGraph_t {};
struct cudaGraphExec_t {};

struct CUmemAllocationProp {
  int type;
  struct { int type; int id; } location;
  int requestedHandleTypes;
};
struct CUmemAccessDesc {
  struct { int type; int id; } location;
  int flags;
};
enum CUmemAllocationHandleType : uint64_t {
  CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR = 1,
  CU_MEM_HANDLE_TYPE_FABRIC = 8,
};
enum CUmemAllocationGranularity_flags { CU_MEM_ALLOC_GRANULARITY_MINIMUM = 0 };
enum CUmemorytype { CU_MEMORYTYPE_DEVICE = 2 };
enum CUdevice_attribute {
  CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED = 0,
  CU_DEVICE_ATTRIBUTE_DMA_BUF_SUPPORTED = 1,
  CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED = 2,
};
enum CUmulticastGranularity_flags { CU_MULTICAST_GRANULARITY_MINIMUM = 0, CU_MULTICAST_GRANULARITY_RECOMMENDED = 1 };
struct CUmulticastObjectProp {
  size_t size;
  int numDevices;
  CUmemAllocationHandleType handleTypes;
  unsigned int flags;
};

constexpr unsigned int CU_MEM_ALLOCATION_TYPE_PINNED = 1;
constexpr unsigned int CU_MEM_LOCATION_TYPE_DEVICE = 1;
constexpr unsigned int CU_MEM_ACCESS_FLAGS_PROT_READWRITE = 1;
constexpr unsigned int CU_POINTER_ATTRIBUTE_MEMORY_TYPE = 0;
constexpr unsigned int CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL = 1;

enum cudaDevice_attribute {
  cudaDevAttrComputeCapabilityMajor = 75,
  cudaDevAttrComputeCapabilityMinor = 76,
  cudaDevAttrClockRate = 84,
};

constexpr auto cudaSuccess = ACL_SUCCESS;
constexpr auto cudaErrorInvalidValue = ACL_ERROR_INVALID_PARAM;
constexpr auto cudaErrorNotSupported = ACL_ERROR_API_NOT_SUPPORT;
constexpr auto cudaErrorPeerAccessAlreadyEnabled = static_cast<cudaError_t>(1000);
constexpr auto cudaErrorContextIsDestroyed = static_cast<cudaError_t>(1001);
constexpr auto cudaErrorInvalidDevice = ACL_ERROR_INVALID_DEVICE;
constexpr auto cudaErrorCudartUnloading = static_cast<cudaError_t>(1002);
constexpr auto cudaErrorInitializationError = ACL_ERROR_RT_INTERNAL_ERROR;
constexpr auto cudaErrorDeviceUninitialized = static_cast<cudaError_t>(1003);
constexpr auto cudaErrorLaunchFailure = static_cast<cudaError_t>(1004);

constexpr auto cudaStreamNonBlocking = static_cast<unsigned int>(0);
constexpr auto cudaHostAllocMapped = static_cast<unsigned int>(0);
constexpr auto cudaHostAllocWriteCombined = static_cast<unsigned int>(0);
constexpr auto cudaMemcpyDefault = ACL_MEMCPY_DEFAULT;
constexpr auto cudaMemcpyDeviceToDevice = ACL_MEMCPY_DEVICE_TO_DEVICE;
constexpr auto cudaMemcpyHostToDevice = ACL_MEMCPY_HOST_TO_DEVICE;
constexpr auto cudaMemcpyDeviceToHost = ACL_MEMCPY_DEVICE_TO_HOST;

struct cudaDeviceProp {
  char name[256];
  int major;
  int minor;
  int multiProcessorCount;
  size_t totalGlobalMem;
  int maxThreadsPerBlock;
};

#ifndef CUDA_SUCCESS
#define CUDA_SUCCESS ACL_SUCCESS
#endif
#define CUDA_ERROR_DEINITIALIZED static_cast<aclError>(1002)
#define CUDA_ERROR_CONTEXT_IS_DESTROYED static_cast<aclError>(1001)
#define CUDA_ERROR_LAUNCH_FAILED static_cast<aclError>(1004)
#define CUDA_ERROR_NOT_SUPPORTED ACL_ERROR_API_NOT_SUPPORT
#define CUDA_ERROR_INVALID_VALUE ACL_ERROR_INVALID_PARAM

inline cudaError_t cudaGetDevice(int* device) { return aclrtGetDevice(device); }
inline cudaError_t cudaSetDevice(int device) { return aclrtSetDevice(device); }
inline cudaError_t cudaGetDeviceCount(int* count) { return aclrtGetDeviceCount(count); }
inline cudaError_t cudaDeviceSynchronize() { return aclrtSynchronize(); }
inline cudaError_t cudaGetLastError() { return ACL_SUCCESS; }
inline const char* cudaGetErrorString(cudaError_t err) { return aclrtGetErrorCode(err); }

inline cudaError_t cudaMalloc(void** ptr, size_t size) { return aclrtMalloc(ptr, size, ACL_MEM_MALLOC_HUGE_FIRST); }
inline cudaError_t cudaFree(void* ptr) { return aclrtFree(ptr); }
inline cudaError_t cudaHostAlloc(void** ptr, size_t size, unsigned int flags) {
  (void)flags;
  return aclrtMallocHost(ptr, size);
}
inline cudaError_t cudaFreeHost(void* ptr) { return aclrtFreeHost(ptr); }
inline cudaError_t cudaMemcpy(void* dst, const void* src, size_t count, cudaMemcpyKind kind) {
  return aclrtMemcpy(dst, count, src, count, kind);
}
inline cudaError_t cudaMemcpyAsync(void* dst, const void* src, size_t count, cudaMemcpyKind kind, cudaStream_t stream) {
  return aclrtMemcpyAsync(dst, count, src, count, kind, stream);
}
inline cudaError_t cudaMemset(void* ptr, int value, size_t count) {
  return aclrtMemset(ptr, count, static_cast<int32_t>(value), count);
}
inline cudaError_t cudaMemsetAsync(void* ptr, int value, size_t count, cudaStream_t stream) {
  return aclrtMemsetAsync(ptr, count, static_cast<int32_t>(value), count, stream);
}

inline cudaError_t cudaStreamCreate(cudaStream_t* stream) { return aclrtCreateStream(stream); }
inline cudaError_t cudaStreamCreateWithFlags(cudaStream_t* stream, unsigned int flags) {
  (void)flags;
  return aclrtCreateStream(stream);
}
inline cudaError_t cudaStreamDestroy(cudaStream_t stream) { return aclrtDestroyStream(stream); }
inline cudaError_t cudaStreamSynchronize(cudaStream_t stream) { return aclrtSynchronizeStream(stream); }

inline cudaError_t cudaEventCreate(cudaEvent_t* event) { return aclrtCreateNotify(event, 0ULL); }
inline cudaError_t cudaEventCreateWithFlags(cudaEvent_t* event, unsigned int flags) {
  (void)flags;
  return aclrtCreateNotify(event, 0ULL);
}
inline cudaError_t cudaEventDestroy(cudaEvent_t event) { return aclrtDestroyNotify(event); }
inline cudaError_t cudaEventRecord(cudaEvent_t event, cudaStream_t stream) { return aclrtRecordNotify(event, stream); }
inline cudaError_t cudaEventSynchronize(cudaEvent_t event) {
  // aclrtWaitAndResetNotify(notify, stream, timeout) 提交一个等待任务到指定 stream。
  // 传 nullptr 作为 stream 意味在默认 stream 上等待，但 nullptr stream 的合法性需在灵衢硬件上验证。
  // 替代方案：保存当前 stream 后在有效 stream 上调用 aclrtWaitAndResetNotify，再用 aclrtSynchronizeStream 阻塞。
  // 当前方案：nullptr stream + aclrtSynchronize()（同步整个设备）实现阻塞语义。
  aclError ret = aclrtWaitAndResetNotify(event, nullptr, 0);
  if (ret != ACL_SUCCESS) return ret;
  return aclrtSynchronize();
}

inline cudaError_t cudaDeviceCanAccessPeer(int* canAccess, int device, int peerDevice) {
  (void)device;
  (void)peerDevice;
  *canAccess = 1;
  return ACL_SUCCESS;
}
inline cudaError_t cudaDeviceEnablePeerAccess(int peerDevice, unsigned int flags) {
  (void)peerDevice;
  (void)flags;
  return ACL_SUCCESS;
}

inline cudaError_t cudaGetDeviceProperties(cudaDeviceProp* prop, int device) {
  aclrtSetDevice(device);

  // CANN aclrtGetDeviceInfo 只返回 int64_t，无设备名称查询接口
  strncpy(prop->name, "AscendNPU", 256);

  // CANN 无 major/minor 版本号概念，使用 NPU 架构信息替代
  int64_t npuArch = 0;
  aclrtGetDeviceInfo(static_cast<uint32_t>(device), ACL_DEV_ATTR_NPU_ARCH, &npuArch);
  prop->major = static_cast<int>(npuArch / 100);
  prop->minor = static_cast<int>(npuArch % 100);

  // 查询显存总量
  int64_t totalMem = 0;
  aclrtGetDeviceInfo(static_cast<uint32_t>(device), ACL_DEV_ATTR_TOTAL_GLOBAL_MEM_SIZE, &totalMem);
  prop->totalGlobalMem = static_cast<size_t>(totalMem);

  // 查询 AI Core 数量（替代 multiProcessorCount）
  int64_t aicoreCount = 0;
  aclrtGetDeviceInfo(static_cast<uint32_t>(device), ACL_DEV_ATTR_AICORE_CORE_NUM, &aicoreCount);
  prop->multiProcessorCount = static_cast<int>(aicoreCount);

  // 查询每个 Block 最大线程数
  int64_t maxThreads = 0;
  aclrtGetDeviceInfo(static_cast<uint32_t>(device), ACL_DEV_ATTR_MAX_THREADS_PER_BLOCK, &maxThreads);
  prop->maxThreadsPerBlock = static_cast<int>(maxThreads);

  return ACL_SUCCESS;
}

// cuGetErrorString(err, str): CUDA prototype is (CUresult error, const char** pStr).
// In CANN, CUresult = aclError. The macro is called as cuGetErrorString(&__e, &errStr),
// so *err dereferences the pointer to get the aclError value, which aclrtGetErrorCode accepts.
#define cuGetErrorString(err, str) do { *str = aclrtGetErrorCode(*err); } while(0)

inline CUresult cuPointerGetAttribute(void* data, CUdevice_attribute attribute, CUdeviceptr ptr) {
  (void)data;
  (void)attribute;
  (void)ptr;
  return ACL_ERROR_NOT_SUPPORTED;
}

inline CUresult cuMemGetAddressRange(void** base, size_t* size, CUdeviceptr ptr) {
  (void)base;
  (void)size;
  (void)ptr;
  return ACL_ERROR_NOT_SUPPORTED;
}

inline CUresult cuDeviceGet(CUdevice* dev, int ordinal) {
  (void)dev;
  (void)ordinal;
  return ACL_ERROR_NOT_SUPPORTED;
}

inline CUresult cuDeviceGetAttribute(int* val, CUdevice_attribute attr, CUdevice dev) {
  (void)val;
  (void)attr;
  (void)dev;
  *val = 0;
  return ACL_SUCCESS;
}

inline CUresult cuMemCreate(CUmemGenericAllocationHandle* handle, size_t size, CUmemAllocationProp* prop, unsigned long long flags) {
  (void)handle;
  (void)size;
  (void)prop;
  (void)flags;
  return ACL_ERROR_NOT_SUPPORTED;
}

inline CUresult cuMemRelease(CUmemGenericAllocationHandle handle) {
  (void)handle;
  return ACL_SUCCESS;
}

inline CUresult cuMemMap(CUdeviceptr ptr, size_t size, size_t offset, CUmemGenericAllocationHandle handle, unsigned long long flags) {
  (void)ptr;
  (void)size;
  (void)offset;
  (void)handle;
  (void)flags;
  return ACL_ERROR_NOT_SUPPORTED;
}

inline CUresult cuMemUnmap(CUdeviceptr ptr, size_t size) {
  (void)ptr;
  (void)size;
  return ACL_ERROR_NOT_SUPPORTED;
}

inline CUresult cuMemAddressReserve(CUdeviceptr* ptr, size_t size, size_t alignment, CUdeviceptr addr, unsigned long long flags) {
  (void)ptr;
  (void)size;
  (void)alignment;
  (void)addr;
  (void)flags;
  return ACL_ERROR_NOT_SUPPORTED;
}

inline CUresult cuMemAddressFree(CUdeviceptr ptr, size_t size) {
  (void)ptr;
  (void)size;
  return ACL_ERROR_NOT_SUPPORTED;
}

inline CUresult cuMemSetAccess(CUdeviceptr ptr, size_t size, CUmemAccessDesc* desc, size_t count) {
  (void)ptr;
  (void)size;
  (void)desc;
  (void)count;
  return ACL_ERROR_NOT_SUPPORTED;
}

inline CUresult cuMemGetAllocationGranularity(size_t* granularity, CUmemAllocationProp* prop, CUmemAllocationGranularity_flags flags) {
  (void)granularity;
  (void)prop;
  (void)flags;
  *granularity = 1;
  return ACL_SUCCESS;
}

inline CUresult cuMemRetainAllocationHandle(CUmemGenericAllocationHandle* handle, void* ptr) {
  (void)handle;
  (void)ptr;
  return ACL_ERROR_NOT_SUPPORTED;
}

inline CUresult cuMemExportToShareableHandle(void* shareableHandle, CUmemGenericAllocationHandle handle, CUmemAllocationHandleType handleType, unsigned long long flags) {
  (void)shareableHandle;
  (void)handle;
  (void)handleType;
  (void)flags;
  return ACL_ERROR_NOT_SUPPORTED;
}

inline CUresult cuMemImportFromShareableHandle(CUmemGenericAllocationHandle* handle, void* shareableHandle, CUmemAllocationHandleType handleType, unsigned long long flags) {
  (void)handle;
  (void)shareableHandle;
  (void)handleType;
  (void)flags;
  return ACL_ERROR_NOT_SUPPORTED;
}

inline CUresult cuMemGetHandleForAddressRange(void* handle, CUdeviceptr ptr, size_t size, CUmemAllocationHandleType handleType, unsigned long long flags) {
  (void)handle;
  (void)ptr;
  (void)size;
  (void)handleType;
  (void)flags;
  return ACL_ERROR_NOT_SUPPORTED;
}

inline cudaError_t cudaIpcGetMemHandle(cudaIpcMemHandle_t* handle, void* ptr) {
  (void)handle;
  (void)ptr;
  return ACL_ERROR_NOT_SUPPORTED;
}

inline cudaError_t cudaIpcOpenMemHandle(void** ptr, cudaIpcMemHandle_t handle, unsigned int flags) {
  (void)ptr;
  (void)handle;
  (void)flags;
  return ACL_ERROR_NOT_SUPPORTED;
}

inline cudaError_t cudaIpcCloseMemHandle(void* ptr) {
  (void)ptr;
  return ACL_ERROR_NOT_SUPPORTED;
}

inline cudaError_t cudaMemcpyToSymbol(void* symbol, const void* src, size_t count, size_t offset = 0, cudaMemcpyKind kind = cudaMemcpyHostToDevice) {
  (void)symbol;
  (void)src;
  (void)count;
  (void)offset;
  (void)kind;
  return ACL_ERROR_NOT_SUPPORTED;
}

inline cudaError_t cudaMemcpyToSymbolAsync(void* symbol, const void* src, size_t count, size_t offset = 0, cudaMemcpyKind kind = cudaMemcpyHostToDevice, cudaStream_t stream = nullptr) {
  (void)symbol;
  (void)src;
  (void)count;
  (void)offset;
  (void)kind;
  (void)stream;
  return ACL_ERROR_NOT_SUPPORTED;
}

inline CUresult cuMulticastGetGranularity(size_t* granularity, CUmulticastObjectProp* prop, CUmulticastGranularity_flags flags) {
  (void)granularity;
  (void)prop;
  (void)flags;
  *granularity = 1;
  return ACL_SUCCESS;
}

inline CUresult cuMulticastCreate(CUmemGenericAllocationHandle* handle, CUmulticastObjectProp* prop) {
  (void)handle;
  (void)prop;
  return ACL_ERROR_NOT_SUPPORTED;
}

inline CUresult cuMulticastAddDevice(CUmemGenericAllocationHandle handle, int deviceId) {
  (void)handle;
  (void)deviceId;
  return ACL_ERROR_NOT_SUPPORTED;
}

inline CUresult cuMulticastBindAddr(CUmemGenericAllocationHandle handle, size_t mcOffset, CUdeviceptr bufferAddr, size_t bufferSize, unsigned long long flags) {
  (void)handle;
  (void)mcOffset;
  (void)bufferAddr;
  (void)bufferSize;
  (void)flags;
  return ACL_ERROR_NOT_SUPPORTED;
}

inline CUresult cuMulticastUnbind(CUmemGenericAllocationHandle handle, int device, size_t mcOffset, size_t bufferSize) {
  (void)handle;
  (void)device;
  (void)mcOffset;
  (void)bufferSize;
  return ACL_ERROR_NOT_SUPPORTED;
}

inline cudaError_t cudaThreadExchangeStreamCaptureMode(cudaStreamCaptureMode* mode) {
  (void)mode;
  return ACL_SUCCESS;
}

inline cudaError_t cudaDeviceGetAttribute(int* val, cudaDevice_attribute attr, int device) {
  int64_t value = 0;
  aclError ret;
  switch (attr) {
    case cudaDevAttrComputeCapabilityMajor:
      ret = aclrtGetDeviceInfo(static_cast<uint32_t>(device), ACL_DEV_ATTR_NPU_ARCH, &value);
      *val = static_cast<int>(value / 100);
      return ret;
    case cudaDevAttrComputeCapabilityMinor:
      ret = aclrtGetDeviceInfo(static_cast<uint32_t>(device), ACL_DEV_ATTR_NPU_ARCH, &value);
      *val = static_cast<int>(value % 100);
      return ret;
    case cudaDevAttrClockRate:
      // CANN 无时钟频率属性，返回 0
      *val = 0;
      return ACL_SUCCESS;
    default:
      *val = 0;
      return ACL_SUCCESS;
  }
}

inline cudaError_t cudaDeviceGetPCIBusId(char* busId, int len, int device) {
  // Ascend NPU has no direct PCI Bus ID query API.
  // Resolve real PCI BDF via sysfs: /sys/class/davinci/davinci<d>/device → symlink to /sys/bus/pci/devices/<BDF>
  // Returns standard PCI BDF format "XXXX:XX:XX.X" so downstream code (NUMA lookup, logging) works unchanged.
  std::string devClassPath = "/sys/class/davinci/davinci" + std::to_string(device) + "/device";
  char realPath[PATH_MAX] = {};
  if (realpath(devClassPath.c_str(), realPath) != nullptr) {
    std::string rp(realPath);
    size_t pos = rp.rfind('/');
    if (pos != std::string::npos) {
      std::string pciBdf = rp.substr(pos + 1);
      snprintf(busId, len, "%s", pciBdf.c_str());
      for (int i = 0; i < len && busId[i]; i++) {
        busId[i] = static_cast<char>(std::tolower(static_cast<unsigned char>(busId[i])));
      }
      return ACL_SUCCESS;
    }
  }
  // Fallback: sysfs unavailable, return placeholder
  strncpy(busId, "0000:00:00.0", len);
  return ACL_SUCCESS;
}

inline cudaError_t cudaStreamIsCapturing(cudaStream_t stream, cudaStreamCaptureStatus* status) {
  (void)stream;
  (void)status;
  *status = cudaStreamCaptureStatusNone;
  return ACL_SUCCESS;
}

inline cudaError_t cudaEventElapsedTime(float* ms, cudaEvent_t start, cudaEvent_t end) {
  (void)ms;
  (void)start;
  (void)end;
  return ACL_ERROR_NOT_SUPPORTED;
}

#elif defined(MSCCLPP_DEVICE_HIP)

using cudaError_t = hipError_t;
using cudaEvent_t = hipEvent_t;
using cudaGraph_t = hipGraph_t;
using cudaGraphExec_t = hipGraphExec_t;
using cudaDeviceProp = hipDeviceProp_t;
using cudaStream_t = hipStream_t;
using cudaStreamCaptureMode = hipStreamCaptureMode;
using cudaStreamCaptureStatus = hipStreamCaptureStatus;
using cudaMemcpyKind = hipMemcpyKind;
using cudaIpcMemHandle_t = hipIpcMemHandle_t;

using CUresult = hipError_t;
using CUdeviceptr = hipDeviceptr_t;
using CUmemGenericAllocationHandle = hipMemGenericAllocationHandle_t;
using CUmemAllocationProp = hipMemAllocationProp;
using CUmemAccessDesc = hipMemAccessDesc;
using CUmemAllocationHandleType = hipMemAllocationHandleType;
using CUmemAllocationGranularity_flags = hipMemAllocationGranularity_flags;
using CUmemorytype = hipMemoryType;

constexpr auto cudaErrorPeerAccessAlreadyEnabled = hipErrorPeerAccessAlreadyEnabled;
constexpr auto cudaErrorContextIsDestroyed = hipErrorContextIsDestroyed;
constexpr auto cudaErrorInvalidDevice = hipErrorInvalidDevice;
constexpr auto cudaSuccess = hipSuccess;
constexpr auto cudaErrorNotSupported = hipErrorNotSupported;
constexpr auto cudaStreamNonBlocking = hipStreamNonBlocking;
constexpr auto cudaStreamCaptureModeGlobal = hipStreamCaptureModeGlobal;
constexpr auto cudaStreamCaptureModeRelaxed = hipStreamCaptureModeRelaxed;
constexpr auto cudaStreamCaptureStatusNone = hipStreamCaptureStatusNone;
constexpr auto cudaStreamCaptureStatusActive = hipStreamCaptureStatusActive;
constexpr auto cudaStreamCaptureStatusInvalidated = hipStreamCaptureStatusInvalidated;
constexpr auto cudaHostAllocMapped = hipHostMallocMapped;
constexpr auto cudaHostAllocWriteCombined = hipHostMallocWriteCombined;
constexpr auto cudaMemcpyDefault = hipMemcpyDefault;
constexpr auto cudaMemcpyDeviceToDevice = hipMemcpyDeviceToDevice;
constexpr auto cudaMemcpyHostToDevice = hipMemcpyHostToDevice;
constexpr auto cudaMemcpyDeviceToHost = hipMemcpyDeviceToHost;
constexpr auto cudaIpcMemLazyEnablePeerAccess = hipIpcMemLazyEnablePeerAccess;

constexpr auto cudaDevAttrComputeCapabilityMajor = hipDeviceAttributeComputeCapabilityMajor;
constexpr auto cudaDevAttrComputeCapabilityMinor = hipDeviceAttributeComputeCapabilityMinor;

constexpr auto CU_MEM_ALLOCATION_TYPE_PINNED = hipMemAllocationTypePinned;
constexpr auto CU_MEM_LOCATION_TYPE_DEVICE = hipMemLocationTypeDevice;
constexpr auto CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR = hipMemHandleTypePosixFileDescriptor;
constexpr auto CU_MEM_ACCESS_FLAGS_PROT_READWRITE = hipMemAccessFlagsProtReadWrite;
constexpr auto CU_MEM_ALLOC_GRANULARITY_MINIMUM = hipMemAllocationGranularityMinimum;
constexpr auto CU_MEMORYTYPE_DEVICE = hipMemoryTypeDevice;

constexpr auto CU_POINTER_ATTRIBUTE_MEMORY_TYPE = HIP_POINTER_ATTRIBUTE_MEMORY_TYPE;
constexpr auto CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL = HIP_POINTER_ATTRIBUTE_DEVICE_ORDINAL;

#ifndef CUDA_SUCCESS
#define CUDA_SUCCESS hipSuccess
#endif  // CUDA_SUCCESS
#define CUDA_ERROR_DEINITIALIZED hipErrorDeinitialized
#define CUDA_ERROR_CONTEXT_IS_DESTROYED hipErrorContextIsDestroyed
#define CUDA_ERROR_LAUNCH_FAILED hipErrorLaunchFailure
#define CUDA_ERROR_NOT_SUPPORTED hipErrorNotSupported
#define CUDA_ERROR_INVALID_VALUE hipErrorInvalidValue

#define cudaEventCreate(...) hipEventCreate(__VA_ARGS__)
#define cudaEventCreateWithFlags(...) hipEventCreateWithFlags(__VA_ARGS__)
#define cudaEventDestroy(...) hipEventDestroy(__VA_ARGS__)
#define cudaEventRecord(...) hipEventRecord(__VA_ARGS__)
#define cudaEventSynchronize(...) hipEventSynchronize(__VA_ARGS__)
#define cudaEventElapsedTime(...) hipEventElapsedTime(__VA_ARGS__)
#define cudaGetErrorString(...) hipGetErrorString(__VA_ARGS__)
#define cudaGetDevice(...) hipGetDevice(__VA_ARGS__)
#define cudaGetDeviceCount(...) hipGetDeviceCount(__VA_ARGS__)
#define cudaGetDeviceProperties(...) hipGetDeviceProperties(__VA_ARGS__)
#define cudaDeviceGetAttribute(...) hipDeviceGetAttribute(__VA_ARGS__)
#define cudaGetLastError(...) hipGetLastError(__VA_ARGS__)
#define cudaSetDevice(...) hipSetDevice(__VA_ARGS__)
#define cudaDeviceSynchronize(...) hipDeviceSynchronize(__VA_ARGS__)
#define cudaDeviceGetPCIBusId(...) hipDeviceGetPCIBusId(__VA_ARGS__)
#define cudaDeviceCanAccessPeer(...) hipDeviceCanAccessPeer(__VA_ARGS__)
#define cudaDeviceEnablePeerAccess(...) hipDeviceEnablePeerAccess(__VA_ARGS__)
#define cudaHostAlloc(...) hipHostMalloc(__VA_ARGS__)
#define cudaMalloc(...) hipMalloc(__VA_ARGS__)
#define cudaFree(...) hipFree(__VA_ARGS__)
#define cudaFreeHost(...) hipHostFree(__VA_ARGS__)
#define cudaMemset(...) hipMemset(__VA_ARGS__)
#define cudaMemsetAsync(...) hipMemsetAsync(__VA_ARGS__)
#define cudaMemcpy(...) hipMemcpy(__VA_ARGS__)
#define cudaMemcpyAsync(...) hipMemcpyAsync(__VA_ARGS__)
#define cudaMemcpyToSymbol(...) hipMemcpyToSymbol(__VA_ARGS__)
#define cudaMemcpyToSymbolAsync(...) hipMemcpyToSymbolAsync(__VA_ARGS__)
#define cudaStreamCreate(...) hipStreamCreate(__VA_ARGS__)
#define cudaStreamCreateWithFlags(...) hipStreamCreateWithFlags(__VA_ARGS__)
#define cudaStreamSynchronize(...) hipStreamSynchronize(__VA_ARGS__)
#define cudaStreamBeginCapture(...) hipStreamBeginCapture(__VA_ARGS__)
#define cudaStreamEndCapture(...) hipStreamEndCapture(__VA_ARGS__)
#define cudaStreamDestroy(...) hipStreamDestroy(__VA_ARGS__)
#define cudaStreamIsCapturing(...) hipStreamIsCapturing(__VA_ARGS__)
#define cudaGraphCreate(...) hipGraphCreate(__VA_ARGS__)
#define cudaGraphInstantiate(...) hipGraphInstantiate(__VA_ARGS__)
#define cudaGraphLaunch(...) hipGraphLaunch(__VA_ARGS__)
#define cudaGraphDestroy(...) hipGraphDestroy(__VA_ARGS__)
#define cudaGraphExecDestroy(...) hipGraphExecDestroy(__VA_ARGS__)
#define cudaThreadExchangeStreamCaptureMode(...) hipThreadExchangeStreamCaptureMode(__VA_ARGS__)
#define cudaIpcGetMemHandle(...) hipIpcGetMemHandle(__VA_ARGS__)
#define cudaIpcOpenMemHandle(...) hipIpcOpenMemHandle(__VA_ARGS__)
#define cudaIpcCloseMemHandle(...) hipIpcCloseMemHandle(__VA_ARGS__)

#define cuGetErrorString(...) hipDrvGetErrorString(__VA_ARGS__)
#define cuMemAddressReserve(...) hipMemAddressReserve(__VA_ARGS__)
#define cuMemAddressFree(...) hipMemAddressFree(__VA_ARGS__)
#define cuMemGetAddressRange(...) hipMemGetAddressRange(__VA_ARGS__)
#define cuMemCreate(...) hipMemCreate(__VA_ARGS__)
#define cuMemRelease(...) hipMemRelease(__VA_ARGS__)
#define cuMemSetAccess(...) hipMemSetAccess(__VA_ARGS__)
#define cuMemMap(...) hipMemMap(__VA_ARGS__)
#define cuMemUnmap(...) hipMemUnmap(__VA_ARGS__)
#define cuMemRetainAllocationHandle(...) hipMemRetainAllocationHandle(__VA_ARGS__)
#define cuMemExportToShareableHandle(...) hipMemExportToShareableHandle(__VA_ARGS__)
#define cuMemImportFromShareableHandle(...) hipMemImportFromShareableHandle(__VA_ARGS__)
#define cuMemGetAllocationGranularity(...) hipMemGetAllocationGranularity(__VA_ARGS__)
#define cuPointerGetAttribute(...) hipPointerGetAttribute(__VA_ARGS__)

#else  // !defined(MSCCLPP_DEVICE_CANN) && !defined(MSCCLPP_DEVICE_HIP)

#include <cuda.h>
#include <cuda_runtime.h>

#endif

// NVLS
#if defined(MSCCLPP_DEVICE_CANN)
#define CUDA_NVLS_API_AVAILABLE 0
#elif defined(MSCCLPP_DEVICE_HIP)
#define CUDA_NVLS_API_AVAILABLE 0
#define CU_MEM_HANDLE_TYPE_FABRIC ((hipMemAllocationHandleType)0x8ULL)
#else
#include <linux/version.h>
#if CUDART_VERSION < 12030
#define CU_MEM_HANDLE_TYPE_FABRIC ((CUmemAllocationHandleType)0x8ULL)
#endif
#define CUDA_NVLS_API_AVAILABLE ((CUDART_VERSION >= 12030) && (LINUX_VERSION_CODE >= KERNEL_VERSION(5, 6, 0)))
#endif

// GPU sync threads
#if defined(MSCCLPP_DEVICE_HIP)
#define __syncshm() asm volatile("s_waitcnt lgkmcnt(0) \n s_barrier");
#elif defined(MSCCLPP_DEVICE_CANN)
#define __syncshm()
#else
#define __syncshm() __syncthreads();
#endif

#endif  // MSCCLPP_GPU_HPP_
