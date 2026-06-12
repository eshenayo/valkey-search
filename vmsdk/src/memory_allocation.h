/*
 * Copyright (c) 2025, valkey-search contributors
 * All rights reserved.
 * SPDX-License-Identifier: BSD 3-Clause
 *
 */

#ifndef VMSDK_SRC_MEMORY_ALLOCATION_H_
#define VMSDK_SRC_MEMORY_ALLOCATION_H_

#include <cstdint>

namespace vmsdk {
void ResetValkeyAllocStats();
// Report used memory counter.
uint64_t GetUsedMemoryCnt();

void ReportAllocMemorySize(uint64_t size);
void ReportFreeMemorySize(uint64_t size);

int64_t GetMemoryDelta();

void SetMemoryDelta(int64_t delta);

// SVS runtime memory context. mmap/munmap calls within an SVSMemoryScope are
// tracked in both the global used_memory counter and a dedicated SVS counter.
void EnterSVSContext();
void LeaveSVSContext();
bool InSVSContext();

uint64_t GetSVSRuntimeMemoryCnt();
void ReportSVSRuntimeAlloc(uint64_t size);
void ReportSVSRuntimeFree(uint64_t size);

// Returns current process VmRSS in bytes by reading /proc/self/status.
// Returns 0 on failure (non-Linux or read error).
uint64_t GetProcessRSSBytes();

class SVSMemoryScope {
 public:
  SVSMemoryScope() { EnterSVSContext(); }
  ~SVSMemoryScope() { LeaveSVSContext(); }
  SVSMemoryScope(const SVSMemoryScope&) = delete;
  SVSMemoryScope& operator=(const SVSMemoryScope&) = delete;
};

}  // namespace vmsdk

#endif  // VMSDK_SRC_MEMORY_ALLOCATION_H_
