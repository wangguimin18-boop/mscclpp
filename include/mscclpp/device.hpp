// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef MSCCLPP_DEVICE_HPP_
#define MSCCLPP_DEVICE_HPP_

#if defined(MSCCLPP_USE_CANN)

#define MSCCLPP_HOST_COMPILE
#define MSCCLPP_INLINE inline
#define MSCCLPP_HOST_DEVICE_INLINE inline
#define MSCCLPP_DEVICE_CANN

#elif defined(__HIP_PLATFORM_AMD__)
#include <hip/hip_runtime.h>

#define MSCCLPP_DEVICE_COMPILE
#define MSCCLPP_INLINE __forceinline__
#define MSCCLPP_DEVICE_INLINE __forceinline__ __device__
#define MSCCLPP_HOST_DEVICE_INLINE __forceinline__ __host__ __device__
#define MSCCLPP_DEVICE_HIP

#elif defined(__NVCC__)

#define MSCCLPP_DEVICE_COMPILE
#define MSCCLPP_INLINE __forceinline__
#define MSCCLPP_DEVICE_INLINE __forceinline__ __device__
#define MSCCLPP_HOST_DEVICE_INLINE __forceinline__ __host__ __device__
#define MSCCLPP_DEVICE_CUDA

#else

#define MSCCLPP_HOST_COMPILE
#define MSCCLPP_INLINE inline
#define MSCCLPP_HOST_DEVICE_INLINE inline

#endif

#endif  // MSCCLPP_DEVICE_HPP_
