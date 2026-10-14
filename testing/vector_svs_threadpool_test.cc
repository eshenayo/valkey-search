/*
 * Copyright (c) 2026, valkey-search contributors
 * All rights reserved.
 * SPDX-License-Identifier: BSD 3-Clause
 *
 */

// Tests for the SVS_VAMANA sequential threadpool vtable and the
// create-failure counter. Built only where the SVS C API is (x86_64 Linux).

#include "src/indexes/vector_svs.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <numeric>
#include <thread>
#include <vector>

#include "absl/cleanup/cleanup.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "src/index_schema.pb.h"
#include "src/indexes/vector_base.h"
#include "src/metrics.h"
#include "src/utils/cancel.h"
#include "src/utils/string_interning.h"
#include "svs/c/svs_c.h"
#include "testing/common.h"
#include "vmsdk/src/testing_infra/utils.h"

namespace valkey_search::indexes {
namespace {

using ::testing::HasSubstr;

constexpr int kDimensions = 8;
constexpr int kNumVectors = 100;
constexpr uint32_t kGraphMaxDegree = 16;
constexpr uint32_t kConstructionWindowSize = 32;
constexpr uint32_t kSearchWindowSize = 10;

data_model::VectorIndex CreateSVSVectorIndexProto(
    int dimensions, data_model::DistanceMetric distance_metric,
    data_model::SVSCompressionType compression) {
  data_model::VectorIndex vector_index_proto;
  vector_index_proto.set_dimension_count(dimensions);
  vector_index_proto.set_distance_metric(distance_metric);
  auto svs_algorithm = std::make_unique<data_model::SVSVamanaAlgorithm>();
  svs_algorithm->set_graph_max_degree(kGraphMaxDegree);
  svs_algorithm->set_construction_window_size(kConstructionWindowSize);
  svs_algorithm->set_search_window_size(kSearchWindowSize);
  svs_algorithm->set_alpha(0.0f);
  svs_algorithm->set_compression(compression);
  vector_index_proto.set_allocated_svs_vamana_algorithm(
      svs_algorithm.release());
  return vector_index_proto;
}

InternedStringPtr IndexToKey(int i) {
  return StringInternStore::Intern(absl::StrCat(i, "_key"));
}

cancel::Token &CancelNever() {
  static cancel::Token cancel_never = cancel::Make(1000000, nullptr);
  return cancel_never;
}

// VectorSVS re-declares GetLabelCount under protected:; call it via the base.
VectorBase *AsBase(const std::shared_ptr<VectorSVS<float>> &p) {
  return p.get();
}

// -- (a) vtable shape --------------------------------------------------------

TEST(SvsThreadpoolVtable, VersionAndStructSizeMatchPinnedHeader) {
  EXPECT_EQ(kSvsThreadpoolOps.version, SVS_C_API_VERSION);
  EXPECT_EQ(kSvsThreadpoolOps.struct_size,
            sizeof(svs_threadpool_interface_ops));
  ASSERT_NE(kSvsThreadpoolOps.size, nullptr);
  ASSERT_NE(kSvsThreadpoolOps.parallel_for, nullptr);
  EXPECT_EQ(kSvsThreadpoolIface.ops, &kSvsThreadpoolOps);
  EXPECT_EQ(kSvsThreadpoolIface.self, nullptr);
}

TEST(SvsThreadpoolVtable, SizeReturnsOne) {
  EXPECT_EQ(kSvsThreadpoolOps.size(kSvsThreadpoolIface.self), 1u);
}

TEST(SvsThreadpoolVtable, InstallsOnARealBuilder) {
  svs_error_h err = svs_error_create();
  svs_algorithm_h algo = svs_algorithm_create_vamana(
      kGraphMaxDegree, kConstructionWindowSize, kSearchWindowSize, err);
  ASSERT_NE(algo, nullptr) << svs_error_get_message(err);
  svs_index_builder_h builder = svs_index_builder_create(
      SVS_DISTANCE_METRIC_EUCLIDEAN, kDimensions, algo, err);
  ASSERT_NE(builder, nullptr) << svs_error_get_message(err);
  EXPECT_TRUE(svs_index_builder_set_threadpool_custom(
      builder, &kSvsThreadpoolIface, err))
      << svs_error_get_message(err);
  svs_index_builder_free(builder);
  svs_algorithm_free(algo);
  svs_error_free(err);
}

// -- (b) parallel_for contract -----------------------------------------------

struct ParallelForRecorder {
  std::vector<size_t> seen;
  std::thread::id caller;
  bool all_on_caller_thread = true;
};

void RecordInvocation(void *param, size_t i) {
  auto *rec = static_cast<ParallelForRecorder *>(param);
  rec->seen.push_back(i);
  if (std::this_thread::get_id() != rec->caller) {
    rec->all_on_caller_thread = false;
  }
}

class SvsThreadpoolParallelFor : public ::testing::TestWithParam<size_t> {};

TEST_P(SvsThreadpoolParallelFor, InvokesEveryIndexOnceInOrderOnCallerThread) {
  const size_t n = GetParam();
  ParallelForRecorder rec;
  rec.caller = std::this_thread::get_id();
  rec.seen.reserve(n);

  svs_error_h err = svs_error_create();
  EXPECT_TRUE(kSvsThreadpoolOps.parallel_for(kSvsThreadpoolIface.self,
                                             &RecordInvocation, &rec, n, err));
  svs_error_free(err);

  std::vector<size_t> expected(n);
  std::iota(expected.begin(), expected.end(), size_t{0});
  EXPECT_EQ(rec.seen, expected);
  EXPECT_TRUE(rec.all_on_caller_thread);
}

INSTANTIATE_TEST_SUITE_P(Sizes, SvsThreadpoolParallelFor,
                         ::testing::Values(size_t{0}, size_t{1}, size_t{7},
                                           size_t{1000}));

// -- (c) build, add and search through the installed pool --------------------

class VectorSVSThreadpoolTest : public ValkeySearchTest {
 public:
  const char *attribute_identifier = "attribute_identifier_1";
  data_model::AttributeDataType attribute_data_type =
      data_model::AttributeDataType::ATTRIBUTE_DATA_TYPE_HASH;
};

class VectorSVSThreadpoolCompressionTest
    : public VectorSVSThreadpoolTest,
      public ::testing::WithParamInterface<data_model::SVSCompressionType> {};

TEST_P(VectorSVSThreadpoolCompressionTest,
       BootstrapAddAndSearchFindPlantedVector) {
  auto index = VectorSVS<float>::Create(
      CreateSVSVectorIndexProto(kDimensions, data_model::DISTANCE_METRIC_L2,
                                GetParam()),
      attribute_identifier, attribute_data_type, 0);
  VMSDK_EXPECT_OK(index);
  ASSERT_TRUE(index.ok());

  // The first AddRecord bootstraps the index; the rest go through add_points.
  auto vectors =
      DeterministicallyGenerateVectors(kNumVectors, kDimensions, 10.0);
  for (int i = 0; i < kNumVectors; ++i) {
    VMSDK_EXPECT_OK(testing_infra::AddVectorRecord(
        **index, IndexToKey(i), VectorToStr(vectors[i])));
  }
  EXPECT_EQ(AsBase(*index)->GetLabelCount(),
            static_cast<size_t>(kNumVectors));

  for (int planted : {0, 42, kNumVectors - 1}) {
    auto result =
        (*index)->Search(VectorToStr(vectors[planted]), 1, CancelNever());
    VMSDK_EXPECT_OK(result);
    ASSERT_TRUE(result.ok());
    ASSERT_EQ(result->size(), 1u) << "planted " << planted;
    EXPECT_EQ((*result)[0].external_id, IndexToKey(planted));
    EXPECT_NEAR((*result)[0].distance, 0.0f, 1e-3f);
  }
}

// SQ8 is left out: its distances are a known open item (issue #13).
INSTANTIATE_TEST_SUITE_P(
    Compression, VectorSVSThreadpoolCompressionTest,
    ::testing::Values(data_model::SVS_COMPRESSION_NONE,
                      data_model::SVS_COMPRESSION_FP16),
    [](const ::testing::TestParamInfo<data_model::SVSCompressionType> &info) {
      return data_model::SVSCompressionType_Name(info.param);
    });

// -- (d) installer failure bumps svs_create_exceptions_cnt once --------------

size_t ZeroThreadpoolSize(void * /*self*/) { return 0; }

TEST_F(VectorSVSThreadpoolTest,
       RejectedThreadpoolInstallBumpsCreateCounterOnce) {
  // The C API rejects a custom pool whose size() returns 0 at install time,
  // so the first AddRecord fails inside BootstrapIndex.
  auto *saved_size = kSvsThreadpoolOps.size;
  kSvsThreadpoolOps.size = &ZeroThreadpoolSize;
  absl::Cleanup restore = [saved_size] {
    kSvsThreadpoolOps.size = saved_size;
  };

  auto &counter = Metrics::GetStats().svs_create_exceptions_cnt;
  const uint64_t before = counter.load();

  auto index = VectorSVS<float>::Create(
      CreateSVSVectorIndexProto(kDimensions, data_model::DISTANCE_METRIC_L2,
                                data_model::SVS_COMPRESSION_NONE),
      attribute_identifier, attribute_data_type, 0);
  VMSDK_EXPECT_OK(index);
  ASSERT_TRUE(index.ok());
  // Create() does not touch SVS; nothing counted yet.
  EXPECT_EQ(counter.load(), before);

  auto vectors = DeterministicallyGenerateVectors(2, kDimensions, 10.0);
  auto result = testing_infra::AddVectorRecord(**index, IndexToKey(0),
                                               VectorToStr(vectors[0]));
  ASSERT_FALSE(result.ok());
  EXPECT_THAT(result.status().message(),
              HasSubstr("index_builder_set_threadpool_custom"));
  EXPECT_EQ(counter.load(), before + 1);
  EXPECT_EQ(AsBase(*index)->GetLabelCount(), 0u);

  // With the real vtable back the index bootstraps and the counter stays.
  std::move(restore).Invoke();
  VMSDK_EXPECT_OK(testing_infra::AddVectorRecord(**index, IndexToKey(0),
                                                 VectorToStr(vectors[0])));
  VMSDK_EXPECT_OK(testing_infra::AddVectorRecord(**index, IndexToKey(1),
                                                 VectorToStr(vectors[1])));
  EXPECT_EQ(counter.load(), before + 1);
  EXPECT_EQ(AsBase(*index)->GetLabelCount(), 2u);
}

}  // namespace
}  // namespace valkey_search::indexes
