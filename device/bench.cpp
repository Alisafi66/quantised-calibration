// On-device latency benchmark for the quantised classifiers.
//
// Conditions are interleaved rather than run one after another. A phone or
// tablet heats up under sustained load and its governor responds, so measuring
// FP32 to completion and then INT8 would confound precision with thermal state:
// whichever ran second would look slower. Round-robin ordering spreads any
// drift across all conditions equally.
//
// Warm-up iterations are discarded because the first few runs pay for lazy
// allocation and cache population, and because the governor takes time to
// respond to a new load.
//
// Usage:
//   bench --iters 200 --warmup 20 --rounds 5 --batch 1 \
//         fp32=/data/local/tmp/fp32.onnx int8=/data/local/tmp/int8.onnx
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <string>
#include <vector>

#include "onnxruntime_cxx_api.h"

namespace {

constexpr int64_t kSeqLen = 128;

long peak_rss_kb() {
  std::ifstream status("/proc/self/status");
  std::string key;
  long value = -1;
  while (status >> key) {
    if (key == "VmHWM:") { status >> value; break; }
    status.ignore(1 << 16, '\n');
  }
  return value;
}

double percentile(std::vector<double> v, double p) {
  if (v.empty()) return -1.0;
  std::sort(v.begin(), v.end());
  const double idx = p / 100.0 * (static_cast<double>(v.size()) - 1);
  const size_t lo = static_cast<size_t>(idx);
  const size_t hi = std::min(lo + 1, v.size() - 1);
  return v[lo] + (idx - static_cast<double>(lo)) * (v[hi] - v[lo]);
}

struct Condition {
  std::string name;
  std::string path;
  std::unique_ptr<Ort::Session> session;
  std::vector<double> latencies_ms;
};

}  // namespace

int main(int argc, char** argv) {
  int iters = 200, warmup = 20, rounds = 5, batch = 1, threads = 1;
  std::vector<Condition> conditions;

  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    auto next = [&] { return std::stoi(argv[++i]); };
    if (arg == "--iters") iters = next();
    else if (arg == "--warmup") warmup = next();
    else if (arg == "--rounds") rounds = next();
    else if (arg == "--batch") batch = next();
    else if (arg == "--threads") threads = next();
    else {
      const size_t eq = arg.find('=');
      if (eq == std::string::npos) { fprintf(stderr, "bad argument: %s\n", argv[i]); return 2; }
      conditions.push_back({arg.substr(0, eq), arg.substr(eq + 1), nullptr, {}});
    }
  }
  if (conditions.empty()) { fprintf(stderr, "no conditions given\n"); return 2; }

  Ort::Env env(ORT_LOGGING_LEVEL_ERROR, "bench");
  Ort::SessionOptions options;
  options.SetIntraOpNumThreads(threads);
  options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

  for (auto& c : conditions) {
    c.session = std::make_unique<Ort::Session>(env, c.path.c_str(), options);
    c.latencies_ms.reserve(static_cast<size_t>(iters));
  }

  // One fixed input, reused everywhere, so no condition is advantaged by its data.
  const size_t n = static_cast<size_t>(batch) * static_cast<size_t>(kSeqLen);
  std::vector<int64_t> ids(n), mask(n, 1);
  for (size_t i = 0; i < n; ++i) ids[i] = static_cast<int64_t>(1000 + (i % 5000));
  const std::array<int64_t, 2> shape{batch, kSeqLen};

  auto memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  std::vector<Ort::Value> inputs;
  inputs.push_back(Ort::Value::CreateTensor<int64_t>(memory, ids.data(), n, shape.data(), 2));
  inputs.push_back(Ort::Value::CreateTensor<int64_t>(memory, mask.data(), n, shape.data(), 2));
  const char* input_names[] = {"input_ids", "attention_mask"};
  const char* output_names[] = {"logits"};

  auto run_once = [&](Condition& c) {
    const auto t0 = std::chrono::steady_clock::now();
    auto out = c.session->Run(Ort::RunOptions{nullptr}, input_names, inputs.data(), 2,
                              output_names, 1);
    const auto t1 = std::chrono::steady_clock::now();
    (void)out;
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
  };

  for (auto& c : conditions)
    for (int w = 0; w < warmup; ++w) run_once(c);

  const int per_round = std::max(1, iters / rounds);
  for (int r = 0; r < rounds; ++r)
    for (auto& c : conditions)
      for (int k = 0; k < per_round; ++k) c.latencies_ms.push_back(run_once(c));

  printf("{\n  \"batch\": %d, \"threads\": %d, \"warmup\": %d, \"rounds\": %d,\n",
         batch, threads, warmup, rounds);
  printf("  \"peak_rss_kb\": %ld,\n  \"conditions\": {\n", peak_rss_kb());
  for (size_t i = 0; i < conditions.size(); ++i) {
    auto& c = conditions[i];
    double sum = 0.0;
    for (double v : c.latencies_ms) sum += v;
    printf("    \"%s\": {\"n\": %zu, \"mean_ms\": %.4f, \"p50_ms\": %.4f, "
           "\"p95_ms\": %.4f, \"min_ms\": %.4f}%s\n",
           c.name.c_str(), c.latencies_ms.size(), sum / c.latencies_ms.size(),
           percentile(c.latencies_ms, 50), percentile(c.latencies_ms, 95),
           percentile(c.latencies_ms, 0), i + 1 < conditions.size() ? "," : "");
  }
  printf("  }\n}\n");
  return 0;
}
