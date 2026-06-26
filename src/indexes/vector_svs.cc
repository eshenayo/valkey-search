/*
 * Copyright (c) 2025, valkey-search contributors
 * All rights reserved.
 * SPDX-License-Identifier: BSD 3-Clause
 *
 */

#include "src/indexes/vector_svs.h"

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <limits>
#include <memory>
#include <streambuf>
#include <type_traits>
#include <utility>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#include <exception>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "src/attribute_data_type.h"
#include "src/indexes/index_base.h"
#include "src/indexes/vector_base.h"
#include "src/metrics.h"
#include "src/rdb_serialization.h"
#include "src/utils/cancel.h"
#include "src/utils/string_interning.h"
#include "src/valkey_search_options.h"
#include "third_party/hnswlib/hnswlib.h"
#include "vmsdk/src/latency_sampler.h"
#include "vmsdk/src/log.h"
#include "vmsdk/src/memory_allocation.h"
#include "vmsdk/src/utils.h"
#include "vmsdk/src/valkey_module_api/valkey_module.h"

namespace valkey_search::indexes {

namespace {

// Convert valkey-search distance metric to SVS MetricType.
svs::runtime::v0::MetricType ToSVSMetric(
    data_model::DistanceMetric metric) {
  switch (metric) {
    case data_model::DISTANCE_METRIC_L2:
      return svs::runtime::v0::MetricType::L2;
    case data_model::DISTANCE_METRIC_IP:
    case data_model::DISTANCE_METRIC_COSINE:
      // COSINE is handled by valkey-search's normalization layer,
      // which converts it to IP with normalized vectors.
      return svs::runtime::v0::MetricType::INNER_PRODUCT;
    default:
      return svs::runtime::v0::MetricType::L2;
  }
}

// Convert protobuf SVSCompressionType to svs::runtime StorageKind.
svs::runtime::v0::StorageKind ToSVSStorageKind(
    data_model::SVSCompressionType compression) {
  switch (compression) {
    case data_model::SVS_COMPRESSION_FP16:
      return svs::runtime::v0::StorageKind::FP16;
    case data_model::SVS_COMPRESSION_LVQ4:
      return svs::runtime::v0::StorageKind::LVQ4x0;
    case data_model::SVS_COMPRESSION_LVQ8:
      return svs::runtime::v0::StorageKind::LVQ8x0;
    case data_model::SVS_COMPRESSION_LVQ4X4:
      return svs::runtime::v0::StorageKind::LVQ4x4;
    case data_model::SVS_COMPRESSION_LVQ4X8:
      return svs::runtime::v0::StorageKind::LVQ4x8;
    case data_model::SVS_COMPRESSION_LEANVEC4X4:
      return svs::runtime::v0::StorageKind::LeanVec4x4;
    case data_model::SVS_COMPRESSION_LEANVEC4X8:
      return svs::runtime::v0::StorageKind::LeanVec4x8;
    case data_model::SVS_COMPRESSION_LEANVEC8X8:
      return svs::runtime::v0::StorageKind::LeanVec8x8;
    default:
      return svs::runtime::v0::StorageKind::FP32;
  }
}

const char* CompressionTypeName(data_model::SVSCompressionType type) {
  switch (type) {
    case data_model::SVS_COMPRESSION_NONE: return "NONE";
    case data_model::SVS_COMPRESSION_FP16: return "FP16";
    case data_model::SVS_COMPRESSION_LVQ4: return "LVQ4";
    case data_model::SVS_COMPRESSION_LVQ8: return "LVQ8";
    case data_model::SVS_COMPRESSION_LVQ4X4: return "LVQ4X4";
    case data_model::SVS_COMPRESSION_LVQ4X8: return "LVQ4X8";
    case data_model::SVS_COMPRESSION_LEANVEC4X4: return "LEANVEC4X4";
    case data_model::SVS_COMPRESSION_LEANVEC4X8: return "LEANVEC4X8";
    case data_model::SVS_COMPRESSION_LEANVEC8X8: return "LEANVEC8X8";
    default: return "UNKNOWN";
  }
}

}  // namespace

template <typename T>
void VectorSVS<T>::UpdateRuntimeMemoryAccounting(uint64_t rss_before) {
  uint64_t rss_after = vmsdk::GetProcessRSSBytes();
  if (rss_after <= rss_before) return;
  uint64_t delta = rss_after - rss_before;
  reported_svs_bytes_.fetch_add(delta, std::memory_order_relaxed);
  vmsdk::ReportAllocMemorySize(delta);
  vmsdk::ReportSVSRuntimeAlloc(delta);
}

template <typename T>
void VectorSVS<T>::UpdateRuntimeMemoryAccountingFree(uint64_t rss_before) {
  uint64_t rss_after = vmsdk::GetProcessRSSBytes();
  if (rss_after >= rss_before) return;
  uint64_t delta = rss_before - rss_after;
  uint64_t prev = reported_svs_bytes_.load(std::memory_order_relaxed);
  if (delta > prev) delta = prev;
  reported_svs_bytes_.fetch_sub(delta, std::memory_order_relaxed);
  vmsdk::ReportFreeMemorySize(delta);
  vmsdk::ReportSVSRuntimeFree(delta);
}

template <typename T>
VectorSVS<T>::VectorSVS(
    int dimensions, data_model::DistanceMetric distance_metric,
    const SVSBuildConfig& build_config,
    absl::string_view attribute_identifier,
    data_model::AttributeDataType attribute_data_type)
    : VectorBase(IndexerType::kSVS, dimensions, attribute_data_type,
                 attribute_identifier),
      build_config_(build_config) {}

template <typename T>
VectorSVS<T>::~VectorSVS() {
  if (svs_index_) {
    uint64_t reported = reported_svs_bytes_.exchange(0);
    if (reported > 0) {
      vmsdk::ReportFreeMemorySize(reported);
      vmsdk::ReportSVSRuntimeFree(reported);
    }
    auto status = svs::runtime::v0::DynamicVamanaIndex::destroy(svs_index_);
    if (!status.ok()) {
      VMSDK_LOG(WARNING, nullptr)
          << "SVS destroy failed: " << status.message();
    }
    svs_index_ = nullptr;
  }
}

template <typename T>
absl::StatusOr<std::shared_ptr<VectorSVS<T>>> VectorSVS<T>::Create(
    const data_model::VectorIndex& vector_index_proto,
    absl::string_view attribute_identifier,
    data_model::AttributeDataType attribute_data_type) {
  SVSBuildConfig config;

  // Extract SVS-specific params from protobuf
  if (vector_index_proto.has_svs_vamana_algorithm()) {
    const auto& svs_params = vector_index_proto.svs_vamana_algorithm();
    if (svs_params.graph_max_degree() > 0) {
      config.graph_max_degree = svs_params.graph_max_degree();
    }
    if (svs_params.construction_window_size() > 0) {
      config.construction_window_size = svs_params.construction_window_size();
    }
    if (svs_params.search_window_size() > 0) {
      config.search_window_size = svs_params.search_window_size();
    }
    if (svs_params.alpha() > 0.0f) {
      config.alpha = svs_params.alpha();
    }
    if (svs_params.compression() != data_model::SVS_COMPRESSION_NONE) {
      config.compression = svs_params.compression();
    }
    if (svs_params.leanvec_dims() > 0) {
      config.leanvec_dims = svs_params.leanvec_dims();
    }
    if (svs_params.leanvec_training_threshold() > 0) {
      config.leanvec_training_threshold =
          svs_params.leanvec_training_threshold();
    }
    if (svs_params.raw_vector_storage() ==
        data_model::RAW_VECTOR_STORAGE_DROP) {
      config.drop_intern_store = true;
    }
  }

  // SVS requires alpha <= 1.0 for MIP/Cosine distance metrics.
  if (vector_index_proto.distance_metric() ==
          data_model::DISTANCE_METRIC_IP ||
      vector_index_proto.distance_metric() ==
          data_model::DISTANCE_METRIC_COSINE) {
    if (config.alpha > 1.0f) {
      VMSDK_LOG(NOTICE, nullptr)
          << "Clamping SVS alpha from " << config.alpha
          << " to 1.0 (required for IP/COSINE metrics)";
      config.alpha = 1.0f;
    }
  }

  auto index = std::shared_ptr<VectorSVS<T>>(new VectorSVS<T>(
      vector_index_proto.dimension_count(),
      vector_index_proto.distance_metric(), config, attribute_identifier,
      attribute_data_type));

  // Initialize the VectorBase (sets distance_metric_, normalize_, space_)
  index->Init(vector_index_proto.dimension_count(),
              vector_index_proto.distance_metric(), index->space_);

  // Configure OpenMP threads used by the SVS runtime. Controlled via the
  // "--svs-omp-threads" module option. PoC default is 1 to avoid interference
  // with valkey-search's own thread pools; setting it higher may reduce
  // per-query latency at the cost of extra CPU per search. 0 means "don't
  // touch" (use the library default / OMP_NUM_THREADS env var).
#ifdef _OPENMP
  long long omp_threads = options::GetSVSOmpThreads().GetValue();
  if (omp_threads > 0) {
    omp_set_num_threads(static_cast<int>(omp_threads));
  }
#endif

  auto svs_metric = ToSVSMetric(vector_index_proto.distance_metric());

  svs::runtime::v0::VamanaIndex::BuildParams build_params;
  build_params.graph_max_degree = config.graph_max_degree;
  build_params.construction_window_size = config.construction_window_size;
  build_params.alpha = config.alpha;

  svs::runtime::v0::VamanaIndex::SearchParams search_params;
  search_params.search_window_size = config.search_window_size;

  auto storage_kind = ToSVSStorageKind(config.compression);

  svs::runtime::v0::VamanaIndex::DynamicIndexParams dyn_params;
  if (IsLeanVecCompression(config.compression)) {
    dyn_params.deferred_compression_threshold =
        config.leanvec_training_threshold;
    dyn_params.initial_storage_kind = svs::runtime::v0::StorageKind::FP32;
  }

  uint64_t rss_before = vmsdk::GetProcessRSSBytes();
  svs::runtime::v0::Status status;
  if (IsLeanVecCompression(config.compression)) {
    status = svs::runtime::v0::DynamicVamanaIndexLeanVec::build(
        &index->svs_index_,
        vector_index_proto.dimension_count(),
        svs_metric,
        storage_kind,
        config.leanvec_dims,
        build_params,
        search_params,
        dyn_params);
  } else if (dyn_params.deferred_compression_threshold > 0) {
    status = svs::runtime::v0::DynamicVamanaIndex::build(
        &index->svs_index_,
        vector_index_proto.dimension_count(),
        svs_metric,
        storage_kind,
        build_params,
        search_params,
        dyn_params);
  } else {
    status = svs::runtime::v0::DynamicVamanaIndex::build(
        &index->svs_index_,
        vector_index_proto.dimension_count(),
        svs_metric,
        storage_kind,
        build_params,
        search_params);
  }

  if (!status.ok()) {
    return absl::InternalError(
        absl::StrCat("Error building SVS Vamana index: ", status.message()));
  }
  if (index->svs_index_ == nullptr) {
    return absl::InternalError(
        "SVS build() returned OK but index pointer is null");
  }

  index->UpdateRuntimeMemoryAccounting(rss_before);

  VMSDK_LOG(NOTICE, nullptr)
      << "Created SVS Vamana index with dim="
      << vector_index_proto.dimension_count()
      << " compression=" << CompressionTypeName(config.compression)
      << " graph_max_degree=" << config.graph_max_degree
      << " construction_window_size=" << config.construction_window_size
      << " alpha=" << config.alpha
      << " search_window_size=" << config.search_window_size;

  return index;
}

// --- Mutation methods ---

template <typename T>
absl::Status VectorSVS<T>::AddRecordImpl(uint64_t internal_id,
                                          absl::string_view record) {
  try {
    absl::MutexLock lock(&index_mutex_);

    size_t label = static_cast<size_t>(internal_id);
    uint64_t rss_before = vmsdk::GetProcessRSSBytes();
    auto svs_status = svs_index_->add(
        1, &label, reinterpret_cast<const float*>(record.data()));

    if (!svs_status.ok()) {
      return absl::InternalError(
          absl::StrCat("SVS add failed: ", svs_status.message()));
    }

    UpdateRuntimeMemoryAccounting(rss_before);
    ++num_elements_;
    return absl::OkStatus();
  } catch (const std::exception& e) {
    return absl::InternalError(
        absl::StrCat("SVS add exception: ", e.what()));
  }
}

template <typename T>
absl::Status VectorSVS<T>::RemoveRecordImpl(uint64_t internal_id) {
  try {
    absl::MutexLock lock(&index_mutex_);

    size_t label = static_cast<size_t>(internal_id);
    auto svs_status = svs_index_->remove(1, &label);

    if (!svs_status.ok()) {
      return absl::InternalError(
          absl::StrCat("SVS remove failed: ", svs_status.message()));
    }

    if (num_elements_ > 0) {
      --num_elements_;
    }
    return absl::OkStatus();
  } catch (const std::exception& e) {
    return absl::InternalError(
        absl::StrCat("SVS remove exception: ", e.what()));
  }
}

template <typename T>
absl::Status VectorSVS<T>::ModifyRecordImpl(uint64_t internal_id,
                                             absl::string_view record) {
  try {
    absl::MutexLock lock(&index_mutex_);

    size_t label = static_cast<size_t>(internal_id);
    auto remove_status = svs_index_->remove(1, &label);
    if (!remove_status.ok()) {
      return absl::InternalError(
          absl::StrCat("SVS remove (modify) failed: ",
                       remove_status.message()));
    }

    auto add_status = svs_index_->add(
        1, &label, reinterpret_cast<const float*>(record.data()));
    if (!add_status.ok()) {
      if (num_elements_ > 0) {
        --num_elements_;
      }
      return absl::InternalError(
          absl::StrCat("SVS add (modify) failed: ",
                       add_status.message()));
    }

    return absl::OkStatus();
  } catch (const std::exception& e) {
    return absl::InternalError(
        absl::StrCat("SVS modify exception: ", e.what()));
  }
}

// --- Search ---

// Adapter: bridge hnswlib's BaseFilterFunctor to SVS's IDFilter interface.
class SVSIDFilterAdapter : public svs::runtime::v0::IDFilter {
 public:
  explicit SVSIDFilterAdapter(hnswlib::BaseFilterFunctor* filter)
      : filter_(filter) {}
  bool is_member(size_t id) const override {
    return (*filter_)(static_cast<hnswlib::labeltype>(id));
  }

 private:
  hnswlib::BaseFilterFunctor* filter_;
};

template <typename T>
absl::StatusOr<std::vector<Neighbor>> VectorSVS<T>::Search(
    absl::string_view query, uint64_t count,
    cancel::Token& cancellation_token,
    std::unique_ptr<hnswlib::BaseFilterFunctor> filter,
    std::optional<unsigned> search_window_size) {
  if (!IsValidSizeVector(query)) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Error parsing vector similarity query: query vector blob size (",
        query.size(), ") does not match index's expected size (",
        dimensions_ * GetDataTypeSize(), ")."));
  }

  // Start total-search timer. Captures lock-wait + core + post-processing.
  vmsdk::StopWatch total_search_timer;
  Metrics::GetStats().svs_search_cnt.fetch_add(1, std::memory_order_relaxed);

  auto perform_search =
      [this, count, &filter, &search_window_size,
       &cancellation_token](absl::string_view q)
          -> absl::StatusOr<
              std::priority_queue<std::pair<T, hnswlib::labeltype>>> {
    try {
      vmsdk::StopWatch lock_wait_timer;
      absl::ReaderMutexLock lock(&index_mutex_);
      auto lock_wait = lock_wait_timer.Duration();
      Metrics::GetStats().svs_search_lock_wait_latency.SubmitSample(lock_wait);
      Metrics::GetStats().svs_search_blackout_us_total.fetch_add(
          absl::ToInt64Microseconds(lock_wait), std::memory_order_relaxed);

      if (num_elements_ == 0) {
        return std::priority_queue<std::pair<T, hnswlib::labeltype>>();
      }

      // Clamp k to available elements.
      size_t k = std::min(static_cast<size_t>(count), num_elements_);

      // Allocate flat output arrays for SVS search results.
      std::vector<float> distances(k);
      std::vector<size_t> labels(k);

      // Build optional search params override.
      svs::runtime::v0::VamanaIndex::SearchParams params;
      svs::runtime::v0::VamanaIndex::SearchParams* params_ptr = nullptr;
      if (search_window_size.has_value()) {
        params.search_window_size = search_window_size.value();
        params_ptr = &params;
      }

      // Build optional filter adapter.
      std::unique_ptr<SVSIDFilterAdapter> svs_filter;
      if (filter) {
        svs_filter = std::make_unique<SVSIDFilterAdapter>(filter.get());
      }

      // Configure OpenMP thread count on this search thread. omp_set_num_threads
      // sets a per-thread ICV, so setting it once on the main thread during
      // index creation does not propagate to reader threads. We re-apply here
      // so the config reflects on each reader thread. 0 = library default.
#ifdef _OPENMP
      {
        long long omp_threads = options::GetSVSOmpThreads().GetValue();
        if (omp_threads > 0) {
          omp_set_num_threads(static_cast<int>(omp_threads));
        }
      }
#endif

      // Measure core SVS search time (isolated from lock wait and post-proc).
      vmsdk::StopWatch core_search_timer;
      auto status = svs_index_->search(
          1,  // single query
          reinterpret_cast<const float*>(q.data()),
          k,
          distances.data(),
          labels.data(),
          params_ptr,
          svs_filter.get());
      Metrics::GetStats().svs_search_core_latency.SubmitSample(
          core_search_timer.Duration());

      if (!status.ok()) {
        return absl::InternalError(
            absl::StrCat("SVS search failed: ", status.message()));
      }

      // SVS search() is synchronous and non-interruptible. Check the
      // cancellation token after completion to avoid wasted post-processing.
      if (cancellation_token->IsCancelled()) {
        return absl::CancelledError(
            "Search operation cancelled due to timeout");
      }

      // Convert flat arrays to priority queue (max-heap by distance).
      // Skip sentinel entries that SVS may produce for filtered searches
      // where fewer than k results match the filter.
      std::priority_queue<std::pair<T, hnswlib::labeltype>> results;
      for (size_t i = 0; i < k; ++i) {
        if (labels[i] == std::numeric_limits<size_t>::max() ||
            std::isinf(distances[i])) {
          continue;
        }
        results.emplace(distances[i],
                        static_cast<hnswlib::labeltype>(labels[i]));
      }
      return results;
    } catch (const std::exception& e) {
      return absl::InternalError(
          absl::StrCat("SVS search exception: ", e.what()));
    }
  };

  if (normalize_) {
    auto norm_record = NormalizeEmbedding(query, GetDataTypeSize());
    VMSDK_ASSIGN_OR_RETURN(
        auto search_result,
        perform_search(absl::string_view(
            reinterpret_cast<const char*>(norm_record.data()),
            norm_record.size())));
    auto reply = CreateReply(search_result);
    Metrics::GetStats().svs_vector_index_search_latency.SubmitSample(
        total_search_timer.Duration());
    return reply;
  }
  VMSDK_ASSIGN_OR_RETURN(auto search_result, perform_search(query));
  auto reply = CreateReply(search_result);
  Metrics::GetStats().svs_vector_index_search_latency.SubmitSample(
      total_search_timer.Duration());
  return reply;
}

// --- Vector tracking ---

template <typename T>
void VectorSVS<T>::TrackVector(uint64_t internal_id,
                                const InternedStringPtr& vector) {
  if (build_config_.drop_intern_store) return;
  absl::MutexLock lock(&tracked_vectors_mutex_);
  tracked_vectors_[internal_id] = vector;
}

template <typename T>
bool VectorSVS<T>::IsVectorMatch(uint64_t internal_id,
                                  const InternedStringPtr& vector) {
  if (build_config_.drop_intern_store) {
    absl::ReaderMutexLock lock(&index_mutex_);
    float dist = 0.0f;
    auto status = svs_index_->get_distance(
        static_cast<size_t>(internal_id),
        reinterpret_cast<const float*>(vector->Str().data()), &dist);
    return status.ok() && dist == 0.0f;
  }
  absl::MutexLock lock(&tracked_vectors_mutex_);
  auto it = tracked_vectors_.find(internal_id);
  if (it == tracked_vectors_.end()) {
    return false;
  }
  return it->second->Str() == vector->Str();
}

template <typename T>
void VectorSVS<T>::UnTrackVector(uint64_t internal_id) {
  if (build_config_.drop_intern_store) return;
  absl::MutexLock lock(&tracked_vectors_mutex_);
  tracked_vectors_.erase(internal_id);
}

template <typename T>
absl::StatusOr<std::pair<float, hnswlib::labeltype>>
VectorSVS<T>::ComputeDistanceFromRecordImpl(
    uint64_t internal_id, absl::string_view query) const {
  {
    absl::ReaderMutexLock lock(&index_mutex_);
    float dist = 0.0f;
    auto status = svs_index_->get_distance(
        static_cast<size_t>(internal_id),
        reinterpret_cast<const float*>(query.data()),
        &dist);
    if (status.ok()) {
      return std::pair<float, hnswlib::labeltype>{dist, internal_id};
    }
  }

  if (build_config_.drop_intern_store) {
    return absl::InternalError(
        absl::StrCat("Couldn't find internal id: ", internal_id));
  }

  absl::ReaderMutexLock lock(&tracked_vectors_mutex_);
  auto it = tracked_vectors_.find(internal_id);
  if (it == tracked_vectors_.end()) {
    return absl::InternalError(
        absl::StrCat("Couldn't find internal id: ", internal_id));
  }

  auto dist = space_->get_dist_func()(
      reinterpret_cast<const T*>(query.data()),
      reinterpret_cast<const T*>(it->second->Str().data()),
      space_->get_dist_func_param());

  return std::pair<float, hnswlib::labeltype>{dist, internal_id};
}

template <typename T>
char* VectorSVS<T>::GetValueImpl(uint64_t internal_id) const {
  if (build_config_.drop_intern_store) {
    thread_local std::vector<float> tl_buffer;
    tl_buffer.resize(dimensions_);
    absl::ReaderMutexLock lock(&index_mutex_);
    size_t id = static_cast<size_t>(internal_id);
    auto status = svs_index_->reconstruct_at(1, &id, tl_buffer.data());
    if (!status.ok()) return nullptr;
    return reinterpret_cast<char*>(tl_buffer.data());
  }
  absl::ReaderMutexLock lock(&tracked_vectors_mutex_);
  auto it = tracked_vectors_.find(internal_id);
  if (it == tracked_vectors_.end()) {
    return nullptr;
  }
  return const_cast<char*>(it->second->Str().data());
}

// --- Serialization ---

template <typename T>
void VectorSVS<T>::ToProtoImpl(
    data_model::VectorIndex* vector_index_proto) const {
  data_model::VectorDataType data_type;
  if constexpr (std::is_same_v<T, float>) {
    data_type = data_model::VectorDataType::VECTOR_DATA_TYPE_FLOAT32;
  } else {
    data_type = data_model::VectorDataType::VECTOR_DATA_TYPE_UNSPECIFIED;
  }
  vector_index_proto->set_vector_data_type(data_type);

  auto svs_algo_proto =
      std::make_unique<data_model::SVSVamanaAlgorithm>();
  svs_algo_proto->set_graph_max_degree(build_config_.graph_max_degree);
  svs_algo_proto->set_construction_window_size(
      build_config_.construction_window_size);
  svs_algo_proto->set_search_window_size(build_config_.search_window_size);
  svs_algo_proto->set_alpha(build_config_.alpha);
  svs_algo_proto->set_compression(build_config_.compression);
  svs_algo_proto->set_leanvec_dims(build_config_.leanvec_dims);
  svs_algo_proto->set_leanvec_training_threshold(
      build_config_.leanvec_training_threshold);
  svs_algo_proto->set_raw_vector_storage(
      build_config_.drop_intern_store ? data_model::RAW_VECTOR_STORAGE_DROP
                                      : data_model::RAW_VECTOR_STORAGE_KEEP);
  vector_index_proto->set_allocated_svs_vamana_algorithm(
      svs_algo_proto.release());
}

template <typename T>
int VectorSVS<T>::RespondWithInfoImpl(ValkeyModuleCtx* ctx) const {
  ValkeyModule_ReplyWithSimpleString(ctx, "data_type");
  if constexpr (std::is_same_v<T, float>) {
    ValkeyModule_ReplyWithSimpleString(
        ctx,
        LookupKeyByValue(*kVectorDataTypeByStr,
                         data_model::VectorDataType::VECTOR_DATA_TYPE_FLOAT32)
            .data());
  } else {
    ValkeyModule_ReplyWithSimpleString(ctx, "UNKNOWN");
  }
  ValkeyModule_ReplyWithSimpleString(ctx, "algorithm");

  svs::runtime::v0::StorageKind current_kind;
  {
    absl::ReaderMutexLock lock(&index_mutex_);
    current_kind = svs_index_->get_current_storage_kind();
  }

  bool is_leanvec = IsLeanVecCompression(build_config_.compression);
  int n_pairs = is_leanvec ? 10 : 8;
  ValkeyModule_ReplyWithArray(ctx, n_pairs * 2);

  ValkeyModule_ReplyWithSimpleString(ctx, "name");
  ValkeyModule_ReplyWithSimpleString(
      ctx,
      LookupKeyByValue(
          *kVectorAlgoByStr,
          data_model::VectorIndex::AlgorithmCase::kSvsVamanaAlgorithm)
          .data());
  ValkeyModule_ReplyWithSimpleString(ctx, "graph_max_degree");
  ValkeyModule_ReplyWithLongLong(ctx, build_config_.graph_max_degree);
  ValkeyModule_ReplyWithSimpleString(ctx, "construction_window_size");
  ValkeyModule_ReplyWithLongLong(ctx, build_config_.construction_window_size);
  ValkeyModule_ReplyWithSimpleString(ctx, "search_window_size");
  ValkeyModule_ReplyWithLongLong(ctx, build_config_.search_window_size);
  ValkeyModule_ReplyWithSimpleString(ctx, "alpha");
  ValkeyModule_ReplyWithDouble(ctx, build_config_.alpha);
  ValkeyModule_ReplyWithSimpleString(ctx, "compression");
  ValkeyModule_ReplyWithSimpleString(
      ctx, CompressionTypeName(build_config_.compression));
  ValkeyModule_ReplyWithSimpleString(ctx, "compression_status");
  auto target_kind = ToSVSStorageKind(build_config_.compression);
  ValkeyModule_ReplyWithSimpleString(
      ctx, (current_kind == target_kind) ? "active" : "deferred");
  ValkeyModule_ReplyWithSimpleString(ctx, "raw_vector_storage");
  ValkeyModule_ReplyWithSimpleString(
      ctx, build_config_.drop_intern_store ? "DROP" : "KEEP");

  if (is_leanvec) {
    ValkeyModule_ReplyWithSimpleString(ctx, "leanvec_dims");
    ValkeyModule_ReplyWithLongLong(ctx, build_config_.leanvec_dims);
    ValkeyModule_ReplyWithSimpleString(ctx, "deferred_compression_threshold");
    ValkeyModule_ReplyWithLongLong(
        ctx, build_config_.leanvec_training_threshold);
  }

  return 4;  // 4 top-level reply pairs: data_type + algorithm
}

// std::streambuf adapter: bridges std::ostream writes to RDBChunkOutputStream.
class RDBOstreamBuf : public std::streambuf {
 public:
  explicit RDBOstreamBuf(RDBChunkOutputStream* out)
      : out_(out) {
    buf_.resize(kStreamBufSize);
    setp(buf_.data(), buf_.data() + buf_.size());
  }

  ~RDBOstreamBuf() override { sync(); }

  absl::Status status() const { return status_; }

 protected:
  int overflow(int ch) override {
    if (!status_.ok()) return traits_type::eof();
    if (flush_buffer() != 0) return traits_type::eof();
    if (ch != traits_type::eof()) {
      *pptr() = static_cast<char>(ch);
      pbump(1);
    }
    return ch;
  }

  int sync() override {
    return flush_buffer();
  }

 private:
  static constexpr size_t kStreamBufSize = 4 * 1024 * 1024;

  int flush_buffer() {
    auto n = pptr() - pbase();
    if (n > 0) {
      status_ = out_->SaveChunk(pbase(), n);
      if (!status_.ok()) return -1;
      setp(buf_.data(), buf_.data() + buf_.size());
    }
    return 0;
  }

  RDBChunkOutputStream* out_;
  std::vector<char> buf_;
  absl::Status status_ = absl::OkStatus();
};

// std::streambuf adapter: bridges std::istream reads from RDBChunkInputStream.
class RDBIstreamBuf : public std::streambuf {
 public:
  explicit RDBIstreamBuf(RDBChunkInputStream* in)
      : in_(in) {}

  absl::Status status() const { return status_; }

 protected:
  int underflow() override {
    if (gptr() < egptr()) return traits_type::to_int_type(*gptr());
    if (in_->AtEnd()) return traits_type::eof();
    auto chunk_or = in_->LoadChunk();
    if (!chunk_or.ok()) {
      status_ = chunk_or.status();
      return traits_type::eof();
    }
    current_chunk_ = std::move(*chunk_or);
    if (!current_chunk_ || current_chunk_->empty()) {
      return traits_type::eof();
    }
    char* data = current_chunk_->data();
    setg(data, data, data + current_chunk_->size());
    return traits_type::to_int_type(*gptr());
  }

 private:
  RDBChunkInputStream* in_;
  std::unique_ptr<std::string> current_chunk_;
  absl::Status status_ = absl::OkStatus();
};

static constexpr uint32_t kSVSRDBVersion = 2;

template <typename T>
absl::Status VectorSVS<T>::SaveIndexImpl(
    RDBChunkOutputStream chunked_out) const {
  absl::ReaderMutexLock lock(&index_mutex_);

  VMSDK_RETURN_IF_ERROR(chunked_out.SaveObject(kSVSRDBVersion));

  VMSDK_RETURN_IF_ERROR(chunked_out.SaveObject(build_config_.graph_max_degree));
  VMSDK_RETURN_IF_ERROR(
      chunked_out.SaveObject(build_config_.construction_window_size));
  VMSDK_RETURN_IF_ERROR(chunked_out.SaveObject(build_config_.alpha));
  VMSDK_RETURN_IF_ERROR(
      chunked_out.SaveObject(build_config_.search_window_size));
  uint32_t compression = static_cast<uint32_t>(build_config_.compression);
  VMSDK_RETURN_IF_ERROR(chunked_out.SaveObject(compression));
  VMSDK_RETURN_IF_ERROR(chunked_out.SaveObject(build_config_.leanvec_dims));
  VMSDK_RETURN_IF_ERROR(
      chunked_out.SaveObject(build_config_.leanvec_training_threshold));
  uint8_t drop_intern = build_config_.drop_intern_store ? 1 : 0;
  VMSDK_RETURN_IF_ERROR(chunked_out.SaveObject(drop_intern));

  VMSDK_RETURN_IF_ERROR(chunked_out.SaveObject(num_elements_));

  // NOTE: SVS runtime save() currently crashes with heap corruption when
  // called in a forked BGSAVE child. This is tracked as an SVS runtime bug.
  RDBOstreamBuf ostreambuf(&chunked_out);
  std::ostream os(&ostreambuf);
  auto svs_status = svs_index_->save(os);
  os.flush();
  if (!svs_status.ok()) {
    return absl::InternalError(
        absl::StrCat("SVS save failed: ", svs_status.message()));
  }
  VMSDK_RETURN_IF_ERROR(ostreambuf.status());

  return absl::OkStatus();
}

template <typename T>
absl::StatusOr<std::shared_ptr<VectorSVS<T>>> VectorSVS<T>::LoadFromRDB(
    ValkeyModuleCtx* ctx,
    const AttributeDataType* attribute_data_type,
    const data_model::VectorIndex& vector_index_proto,
    absl::string_view attribute_identifier,
    SupplementalContentChunkIter&& iter) {
  RDBChunkInputStream input(std::move(iter));

  VMSDK_ASSIGN_OR_RETURN(auto version, input.LoadObject<uint32_t>());
  if (version != kSVSRDBVersion) {
    return absl::InvalidArgumentError(
        absl::StrCat("Unsupported SVS RDB version: ", version));
  }

  SVSBuildConfig config;
  VMSDK_ASSIGN_OR_RETURN(config.graph_max_degree,
                         input.LoadObject<size_t>());
  VMSDK_ASSIGN_OR_RETURN(config.construction_window_size,
                         input.LoadObject<size_t>());
  VMSDK_ASSIGN_OR_RETURN(config.alpha, input.LoadObject<float>());
  VMSDK_ASSIGN_OR_RETURN(config.search_window_size,
                         input.LoadObject<size_t>());
  VMSDK_ASSIGN_OR_RETURN(auto compression_val,
                         input.LoadObject<uint32_t>());
  config.compression =
      static_cast<data_model::SVSCompressionType>(compression_val);
  VMSDK_ASSIGN_OR_RETURN(config.leanvec_dims, input.LoadObject<size_t>());
  VMSDK_ASSIGN_OR_RETURN(config.leanvec_training_threshold,
                         input.LoadObject<size_t>());
  VMSDK_ASSIGN_OR_RETURN(auto drop_intern_val,
                         input.LoadObject<uint8_t>());
  config.drop_intern_store = (drop_intern_val != 0);

  VMSDK_ASSIGN_OR_RETURN(auto num_elements, input.LoadObject<size_t>());

  auto index = std::shared_ptr<VectorSVS<T>>(new VectorSVS<T>(
      vector_index_proto.dimension_count(),
      vector_index_proto.distance_metric(), config,
      attribute_identifier, attribute_data_type->ToProto()));

  index->Init(vector_index_proto.dimension_count(),
              vector_index_proto.distance_metric(), index->space_);

#ifdef _OPENMP
  long long omp_threads = options::GetSVSOmpThreads().GetValue();
  if (omp_threads > 0) {
    omp_set_num_threads(static_cast<int>(omp_threads));
  }
#endif

  auto svs_metric = ToSVSMetric(vector_index_proto.distance_metric());
  auto storage_kind = ToSVSStorageKind(config.compression);

  RDBIstreamBuf istreambuf(&input);
  std::istream is(&istreambuf);

  uint64_t rss_before = vmsdk::GetProcessRSSBytes();
  auto svs_status = svs::runtime::v0::DynamicVamanaIndex::load(
      &index->svs_index_, is, svs_metric, storage_kind);
  if (!svs_status.ok()) {
    return absl::InternalError(
        absl::StrCat("SVS load failed: ", svs_status.message()));
  }
  VMSDK_RETURN_IF_ERROR(istreambuf.status());

  index->num_elements_ = num_elements;
  index->UpdateRuntimeMemoryAccounting(rss_before);

  VMSDK_LOG(NOTICE, nullptr)
      << "Loaded SVS Vamana index from RDB: dim="
      << vector_index_proto.dimension_count()
      << " compression=" << CompressionTypeName(config.compression)
      << " num_elements=" << num_elements;

  return index;
}

// Explicit template instantiation
template class VectorSVS<float>;

}  // namespace valkey_search::indexes
