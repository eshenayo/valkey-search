/*
 * Copyright (c) 2026, valkey-search contributors
 * All rights reserved.
 * SPDX-License-Identifier: BSD 3-Clause
 *
 */

#ifndef VALKEYSEARCH_SRC_INDEXES_SVS_CAPABILITIES_H_
#define VALKEYSEARCH_SRC_INDEXES_SVS_CAPABILITIES_H_

#include "absl/status/status.h"
#include "src/index_schema.pb.h"

namespace valkey_search::indexes {

bool IsSvsLvqCompression(data_model::SVSCompressionType compression);
bool IsSvsLeanVecCompression(data_model::SVSCompressionType compression);

// True when the linked SVS C API can construct LVQ/LeanVec storage on this
// CPU. Probed once at runtime: the C ABI is identical across SVS builds.
bool SvsLvqLeanVecSupported();

// True when linked against the prebuilt SVS C API (SVS_C_API_PREBUILT).
bool SvsCApiPrebuilt();

// Rejects LVQ/LeanVec compressions when SvsLvqLeanVecSupported() is false.
absl::Status CheckSvsCompressionAvailable(
    data_model::SVSCompressionType compression);

}  // namespace valkey_search::indexes
#endif  // VALKEYSEARCH_SRC_INDEXES_SVS_CAPABILITIES_H_
