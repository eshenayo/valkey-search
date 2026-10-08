/*
 * Copyright (c) 2026, valkey-search contributors
 * All rights reserved.
 * SPDX-License-Identifier: BSD 3-Clause
 *
 */

#if defined(__linux__) && defined(__x86_64__)

#include "src/indexes/vector_svs.h"

#include <memory>
#include <string>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "src/attribute_data_type.h"
#include "src/index_schema.pb.h"
#include "src/utils/string_interning.h"
#include "testing/common.h"
#include "vmsdk/src/log.h"
#include "vmsdk/src/testing_infra/utils.h"

namespace valkey_search::indexes {

namespace {

class VectorSVSTest : public ValkeySearchTest {};

TEST_F(VectorSVSTest, BuildLogsThroughValkeyLog) {
  constexpr int kDimensions = 8;
  VMSDK_EXPECT_OK(vmsdk::InitLogging(nullptr, "debug"));
  data_model::VectorIndex proto;
  proto.set_dimension_count(kDimensions);
  proto.set_distance_metric(data_model::DISTANCE_METRIC_L2);
  auto* svs = proto.mutable_svs_vamana_algorithm();
  svs->set_graph_max_degree(16);
  svs->set_construction_window_size(32);
  svs->set_search_window_size(16);
  auto index = VectorSVS<float>::Create(
      proto, "svs_attr", data_model::ATTRIBUTE_DATA_TYPE_HASH, 0);
  VMSDK_EXPECT_OK(index);

  EXPECT_CALL(*kMockValkeyModule, Log(testing::_, testing::_, testing::_))
      .Times(testing::AnyNumber());
  EXPECT_CALL(*kMockValkeyModule,
              Log(testing::_, testing::_, testing::HasSubstr("SVS[svs_attr] ")))
      .Times(testing::AtLeast(1));
  std::vector<float> vec(kDimensions, 1.0f);
  auto res = testing_infra::AddVectorRecord(
      **index, StringInternStore::Intern("key"),
      absl::string_view(reinterpret_cast<const char*>(vec.data()),
                        vec.size() * sizeof(float)));
  VMSDK_EXPECT_OK(res);
  VMSDK_EXPECT_OK(vmsdk::InitLogging(nullptr, "notice"));
}

}  // namespace

}  // namespace valkey_search::indexes

#endif  // __linux__ && __x86_64__
