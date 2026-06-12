/*
 * Copyright (c) 2025, valkey-search contributors
 * All rights reserved.
 * SPDX-License-Identifier: BSD 3-Clause
 *
 */

#include "vmsdk/src/memory_allocation.h"

#include <fcntl.h>
#include <unistd.h>

#include <cstdint>
#include <cstdlib>
#include <cstring>

#include "vmsdk/src/sharded_atomic.h"

namespace vmsdk {

// Use the standard system allocator by default. Note that this is required
// since any allocation done before Valkey module initialization (namely global
// static constructors that do heap allocation, which are run on dl_open) cannot
// invoke Valkey modules api since the associated C function pointers are only
// initialized as part of the module initialization process. Refer
// https://redis.com/blog/using-the-redis-allocator-in-rust for more details.

thread_local static int64_t memory_delta = 0;
thread_local static bool in_svs_context = false;

ShardedAtomic<uint64_t> used_memory_bytes;
ShardedAtomic<uint64_t> svs_runtime_memory_bytes;

void ResetValkeyAllocStats() {
  used_memory_bytes.Reset();
  svs_runtime_memory_bytes.Reset();
  memory_delta = 0;
}

uint64_t GetUsedMemoryCnt() { return used_memory_bytes.GetTotal(); }

void ReportAllocMemorySize(uint64_t size) {
  used_memory_bytes.Add(size);

  memory_delta += static_cast<int64_t>(size);
}

void ReportFreeMemorySize(uint64_t size) {
  used_memory_bytes.Subtract(size);
  memory_delta -= static_cast<int64_t>(size);
}

int64_t GetMemoryDelta() { return memory_delta; }

void SetMemoryDelta(int64_t delta) { memory_delta = delta; }

void EnterSVSContext() { in_svs_context = true; }

void LeaveSVSContext() { in_svs_context = false; }

bool InSVSContext() { return in_svs_context; }

uint64_t GetSVSRuntimeMemoryCnt() {
  return svs_runtime_memory_bytes.GetTotal();
}

void ReportSVSRuntimeAlloc(uint64_t size) {
  svs_runtime_memory_bytes.Add(size);
}

void ReportSVSRuntimeFree(uint64_t size) {
  svs_runtime_memory_bytes.Subtract(size);
}

uint64_t GetProcessRSSBytes() {
  int fd = open("/proc/self/status", O_RDONLY);
  if (fd < 0) return 0;
  char buf[4096];
  ssize_t n = read(fd, buf, sizeof(buf) - 1);
  close(fd);
  if (n <= 0) return 0;
  buf[n] = '\0';
  const char* pos = strstr(buf, "VmRSS:");
  if (!pos) return 0;
  pos += 6;
  while (*pos == ' ' || *pos == '\t') ++pos;
  uint64_t kb = strtoull(pos, nullptr, 10);
  return kb * 1024;
}

}  // namespace vmsdk
