// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include <numa.h>

#include <fstream>
#include <mscclpp/gpu_utils.hpp>

#include "api.h"

#if defined(MSCCLPP_DEVICE_CANN)
#include <unistd.h>
#include <limits.h>
#endif

static const std::string getBusId(int deviceId) {
  char busIdChar[] = "00000000:00:00.0";
  MSCCLPP_CUDATHROW(cudaDeviceGetPCIBusId(busIdChar, sizeof(busIdChar), deviceId));
  for (size_t i = 0; i < sizeof(busIdChar); i++) {
    busIdChar[i] = std::tolower(busIdChar[i]);
  }
  return std::string(busIdChar);
}

namespace mscclpp {

MSCCLPP_API_CPP int getDeviceNumaNode(int deviceId) {
#if defined(MSCCLPP_DEVICE_CANN)
  // Ascend NPU 通过 sysfs 查找 NUMA 信息
  // PCIe 连接的 NPU: /sys/class/davinci/davinci<d>/device 是 /sys/bus/pci/devices/<BDF> 的 symlink
  // 读取该 symlink 目标获取真实 PCI BDF，再从 /sys/bus/pci/devices/<BDF>/numa_node 获取 NUMA
  std::string devClassPath = "/sys/class/davinci/davinci" + std::to_string(deviceId) + "/device";
  char realPath[PATH_MAX] = {};
  if (realpath(devClassPath.c_str(), realPath) == nullptr) {
    throw Error("Failed to resolve device path: " + devClassPath, ErrorCode::SystemError);
  }
  std::string pciBusId;
  std::string rp(realPath);
  // realPath 格式: /sys/devices/.../0000:XX:XX.X 或 /sys/bus/pci/devices/0000:XX:XX.X
  size_t pos = rp.rfind('/');
  if (pos != std::string::npos) {
    pciBusId = rp.substr(pos + 1);
  } else {
    throw Error("Invalid device path format: " + rp, ErrorCode::SystemError);
  }
  std::string file_str = "/sys/bus/pci/devices/" + pciBusId + "/numa_node";
#else
  std::string busId = getBusId(deviceId);
  std::string file_str = "/sys/bus/pci/devices/" + busId + "/numa_node";
#endif
  std::ifstream file(file_str);
  int numaNode;
  if (file.is_open()) {
    if (!(file >> numaNode)) {
      throw Error("Failed to read NUMA node from file: " + file_str, ErrorCode::SystemError);
    }
  } else {
    throw Error("Failed to open file: " + file_str, ErrorCode::SystemError);
  }
  return numaNode;
}

MSCCLPP_API_CPP void numaBind(int node) {
  int totalNumNumaNodes = numa_num_configured_nodes();
  if (node < 0 || node >= totalNumNumaNodes) {
    throw Error(
        "Invalid NUMA node " + std::to_string(node) + ", must be between 0 and " + std::to_string(totalNumNumaNodes),
        ErrorCode::InvalidUsage);
  }
  nodemask_t mask;
  nodemask_zero(&mask);
  nodemask_set_compat(&mask, node);
  numa_bind_compat(&mask);
}

}  // namespace mscclpp
