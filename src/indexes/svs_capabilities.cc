/*
 * Copyright (c) 2026, valkey-search contributors
 * All rights reserved.
 * SPDX-License-Identifier: BSD 3-Clause
 *
 */

#include "src/indexes/svs_capabilities.h"

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/strip.h"
#include "src/index_schema.pb.h"

#if defined(__linux__) && defined(__x86_64__)
#include <svs/c/svs_c.h>
#endif

namespace valkey_search::indexes {

bool IsSvsLvqCompression(data_model::SVSCompressionType compression) {
  switch (compression) {
    case data_model::SVS_COMPRESSION_LVQ4:
    case data_model::SVS_COMPRESSION_LVQ8:
    case data_model::SVS_COMPRESSION_LVQ4X4:
    case data_model::SVS_COMPRESSION_LVQ4X8:
      return true;
    default:
      return false;
  }
}

bool IsSvsLeanVecCompression(data_model::SVSCompressionType compression) {
  switch (compression) {
    case data_model::SVS_COMPRESSION_LEANVEC4X4:
    case data_model::SVS_COMPRESSION_LEANVEC4X8:
    case data_model::SVS_COMPRESSION_LEANVEC8X8:
      return true;
    default:
      return false;
  }
}

bool SvsLvqLeanVecSupported() {
#if defined(__linux__) && defined(__x86_64__)
  // LVQ and LeanVec share one gate in the C API, checked at storage creation.
  static const bool supported = []() {
    svs_storage_h storage =
        svs_storage_create_lvq(SVS_DATA_TYPE_UINT8, SVS_DATA_TYPE_VOID,
                               /*out_err=*/nullptr);
    if (storage == nullptr) return false;
    svs_storage_free(storage);
    return true;
  }();
  return supported;
#else
  return false;
#endif
}

bool SvsCApiPrebuilt() {
#if defined(VALKEY_SEARCH_SVS_PREBUILT) && VALKEY_SEARCH_SVS_PREBUILT
  return true;
#else
  return false;
#endif
}

absl::Status CheckSvsCompressionAvailable(
    data_model::SVSCompressionType compression) {
  if ((IsSvsLvqCompression(compression) ||
       IsSvsLeanVecCompression(compression)) &&
      !SvsLvqLeanVecSupported()) {
    return absl::InvalidArgumentError(absl::StrCat(
        "SVS_VAMANA: COMPRESSION ",
        absl::StripPrefix(data_model::SVSCompressionType_Name(compression),
                          "SVS_COMPRESSION_"),
        " requires an SVS build with LVQ/LeanVec support on an Intel CPU "
        "(build with --svs-prebuilt)."));
  }
  return absl::OkStatus();
}

}  // namespace valkey_search::indexes
