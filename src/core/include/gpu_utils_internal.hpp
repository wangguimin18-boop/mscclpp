// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#ifndef MSCCLPP_GPU_UTILS_INTERNAL_HPP_
#define MSCCLPP_GPU_UTILS_INTERNAL_HPP_

#include <mscclpp/gpu.hpp>
#include <mscclpp/gpu_utils.hpp>

#include "logger.hpp"

namespace mscclpp {

#if defined(MSCCLPP_DEVICE_CANN)

inline bool isCudaTeardownError(cudaError_t err) {
  return err == ACL_ERROR_RT_INTERNAL_ERROR || err == ACL_ERROR_RT_NO_DEVICE ||
         static_cast<int>(err) >= 1000;
}

inline bool isCuTeardownError(CUresult r) { return isCudaTeardownError(r); }

#else

inline bool isCudaTeardownError(cudaError_t err) {
#if defined(MSCCLPP_USE_ROCM)
  return err == cudaErrorContextIsDestroyed || err == cudaErrorInvalidDevice;
#else
  return err == cudaErrorCudartUnloading || err == cudaErrorContextIsDestroyed || err == cudaErrorInitializationError ||
         err == cudaErrorInvalidDevice || err == cudaErrorLaunchFailure || err == cudaErrorDeviceUninitialized;
#endif
}

inline bool isCuTeardownError(CUresult r) {
  return r == CUDA_ERROR_DEINITIALIZED || r == CUDA_ERROR_CONTEXT_IS_DESTROYED || r == CUDA_ERROR_LAUNCH_FAILED;
}

#endif

}  // namespace mscclpp

/// Execute a CUDA runtime call and ignore teardown errors (useful in destructors).
/// Non-teardown errors will throw.
#define MSCCLPP_CUDATHROW_IGNORE_TEARDOWN(cmd) \
  do {                                         \
    cudaError_t __e = cmd;                     \
    if (mscclpp::isCudaTeardownError(__e)) {   \
      (void)cudaGetLastError();                \
    } else {                                   \
      MSCCLPP_CUDATHROW(__e);                  \
    }                                          \
  } while (false)

/// Execute a CUDA driver call and ignore teardown errors (useful in destructors).
/// Non-teardown errors will throw.
#define MSCCLPP_CUTHROW_IGNORE_TEARDOWN(cmd) \
  do {                                       \
    CUresult __e = cmd;                      \
    if (!mscclpp::isCuTeardownError(__e)) {  \
      MSCCLPP_CUTHROW(__e);                  \
    }                                        \
  } while (false)

/// Execute a CUDA driver call and log (but don't throw) on error.
#define MSCCLPP_CUTHROW_IGNORE(cmd)                                                                   \
  do {                                                                                                \
    CUresult __e = cmd;                                                                               \
    if (__e != CUDA_SUCCESS) {                                                                        \
      const char* errStr;                                                                             \
      cuGetErrorString(&__e, &errStr);                                                                \
      WARN(GPU, __FILE__, ":", __LINE__, " Cuda failure ", static_cast<int>(__e), " '", errStr, "'"); \
    }                                                                                                 \
  } while (false)

#endif  // MSCCLPP_GPU_UTILS_INTERNAL_HPP_
