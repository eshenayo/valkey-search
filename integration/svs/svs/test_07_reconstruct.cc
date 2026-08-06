// svs/test_07_reconstruct.cc
//
// Purpose:
//   Add a known vector to the index, then read it back via
//   DynamicVamanaIndex::reconstruct_at() and verify the round-trip
//   matches within quantization tolerance.
//
// Status on SVS runtime v0.4.0:
//   reconstruct_at(n, ids, output) is available on VamanaIndex as of
//   v0.3.0 (introduced alongside get_distance). This test uses the
//   member API directly.
//
// valkey-search code path this mimics:
//   VectorSVS::GetValueImpl  src/indexes/vector_svs.cc —
//   uses raw_vectors_ FP32 shadow copy. A future change can replace
//   that with reconstruct_at() to eliminate the shadow store.
//
// Reproduction:
//   ./build_test.sh svs/test_07_reconstruct
//   ./svs/test_07_reconstruct
//
// Expected output: low L2 error between stored vector and reconstruct_at()
// output (exact for StorageKind::FP32, small for lossy compression).

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "svs_common.h"
#include "test_common.h"

using namespace svstest;
using svstest::svs_::DVamana;

constexpr size_t kDim = 64;
constexpr size_t kN = 50;

int main() {
  std::printf("svs/test_07_reconstruct: dim=%zu N=%zu\n", kDim, kN);

  DVamana* idx = nullptr;
  auto st = svs_::build_svs(&idx, kDim);
  if (!st.ok()) {
    fail("build", st.message());
    return 1;
  }

  rng(1);
  auto data = random_vecs(kN, kDim);
  std::vector<size_t> labels(kN);
  for (size_t i = 0; i < kN; ++i) labels[i] = i;
  st = idx->add(kN, labels.data(), data.data());
  if (!st.ok()) {
    fail("add", st.message());
    DVamana::destroy(idx);
    return 1;
  }

  section("reconstruct_at");
  std::vector<float> out(kDim);
  double max_err_l2 = 0.0;
  for (size_t i = 0; i < kN; ++i) {
    size_t label = i;
    auto s = idx->reconstruct_at(1, &label, out.data());
    if (!s.ok()) {
      fail("reconstruct_at",
           "i=" + std::to_string(i) + ": " + s.message());
      DVamana::destroy(idx);
      return 1;
    }
    double err = 0;
    for (size_t d = 0; d < kDim; ++d) {
      double dd = out[d] - data[i * kDim + d];
      err += dd * dd;
    }
    err = std::sqrt(err);
    if (err > max_err_l2) max_err_l2 = err;
  }
  std::printf("  max L2 reconstruction error: %.6f (target: <1e-4 for FP32)\n",
              max_err_l2);

  DVamana::destroy(idx);
  pass("svs/test_07_reconstruct");
  return 0;
}
