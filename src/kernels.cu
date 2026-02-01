#include <vector>
#include <cuda_fp16.h>
#include <cmath>
#include <limits>
#include <algorithm>
#include <type_traits>
#include <cstdlib>
#include <cstdio>

#include "../tester/utils.h"

template <typename T>
T trace(const std::vector<T>& h_input, size_t rows, size_t cols) {
  size_t n = std::min(rows, cols);
  T sum = T(0);
  size_t max_elems = h_input.size();
  for (size_t i = 0; i < n; ++i) {
    size_t idx = i * cols + i;
    if (idx >= max_elems) break;
    sum += h_input[idx];
  }
  return sum;
}

template <typename T>
void flashAttention(const std::vector<T>& h_q, const std::vector<T>& h_k,
                    const std::vector<T>& h_v, std::vector<T>& h_o,
                    int batch_size, int target_seq_len, int src_seq_len, 
                    int query_heads, int kv_heads, int head_dim, bool is_causal) {
  if (batch_size <= 0 || target_seq_len <= 0 || src_seq_len <= 0 ||
      query_heads <= 0 || kv_heads <= 0 || head_dim <= 0) {
    return;
  }

  const size_t expect_q = static_cast<size_t>(batch_size) * target_seq_len * query_heads * head_dim;
  const size_t expect_kv = static_cast<size_t>(batch_size) * src_seq_len * kv_heads * head_dim;
  if (h_q.size() < expect_q || h_k.size() < expect_kv || h_v.size() < expect_kv || h_o.size() < expect_q) {
    return;
  }

  bool debug = false;
  const char* dbg_env = std::getenv("FLASH_ATTENTION_DEBUG");
  if (dbg_env && std::atoi(dbg_env) != 0) debug = true;
  if (debug) {
    std::fprintf(stderr, "[flashAttention] shapes: batch=%d tgt=%d src=%d qh=%d kvh=%d dim=%d causal=%d\n",
                 batch_size, target_seq_len, src_seq_len, query_heads, kv_heads, head_dim, is_causal ? 1 : 0);
  }

  using AccT = float; // strictly single-precision internal path
  const AccT inv_sqrt_d = static_cast<AccT>(1.0f / std::sqrt(static_cast<float>(head_dim)));
  const AccT neg_inf = -std::numeric_limits<AccT>::infinity();

  const size_t q_batch_stride = static_cast<size_t>(target_seq_len) * query_heads * head_dim;
  const size_t k_batch_stride = static_cast<size_t>(src_seq_len) * kv_heads * head_dim;
  const size_t q_time_stride = static_cast<size_t>(query_heads) * head_dim;
  const size_t k_time_stride = static_cast<size_t>(kv_heads) * head_dim;
  const size_t head_stride = static_cast<size_t>(head_dim);

  // temporaries reused per head
  std::vector<AccT> scores;
  std::vector<AccT> exps;
  std::vector<AccT> out_v;
  std::vector<AccT> prod_buf;    // for dot-product pairwise
  std::vector<AccT> contribs;    // for weighted contributions across src
  scores.reserve(static_cast<size_t>(src_seq_len));
  exps.reserve(static_cast<size_t>(src_seq_len));
  out_v.assign(static_cast<size_t>(head_dim), static_cast<AccT>(0));
  prod_buf.reserve(static_cast<size_t>(head_dim));
  contribs.reserve(static_cast<size_t>(src_seq_len));

  bool verbose = false;
  const char* venv = std::getenv("FLASH_ATTENTION_VERBOSE");
  if (venv && std::atoi(venv) != 0) verbose = true;

  // helper: pairwise reduce in-place on vector (length n), returns sum at index 0
  // Use std::fma for pairwise addition to control rounding behavior.
  auto pairwise_reduce = [](std::vector<AccT> &v, int n) -> AccT {
    if (n <= 0) return static_cast<AccT>(0);
    while (n > 1) {
      int m = n / 2;
      for (int i = 0; i < m; ++i) {
        // use fma(a,1,b) which computes (a*1 + b) with single rounding
        v[i] = std::fma(v[2*i], static_cast<AccT>(1.0f), v[2*i + 1]);
      }
      if (n & 1) { // odd, move last to next pos
        v[m] = v[n - 1];
        n = m + 1;
      } else {
        n = m;
      }
    }
    return v[0];
  };

  for (int b = 0; b < batch_size; ++b) {
    const size_t q_batch_offset = static_cast<size_t>(b) * q_batch_stride;
    const size_t k_batch_offset = static_cast<size_t>(b) * k_batch_stride;

    for (int t = 0; t < target_seq_len; ++t) {
      const size_t q_time_offset = q_batch_offset + static_cast<size_t>(t) * q_time_stride;

      for (int qh = 0; qh < query_heads; ++qh) {
        const size_t q_head_offset = q_time_offset + static_cast<size_t>(qh) * head_stride;
        const int kv_index = (qh * kv_heads) / std::max(1, query_heads);

        // prepare temporaries
        scores.assign(static_cast<size_t>(src_seq_len), neg_inf);
        exps.assign(static_cast<size_t>(src_seq_len), static_cast<AccT>(0));
        std::fill(out_v.begin(), out_v.end(), static_cast<AccT>(0));
        prod_buf.assign(static_cast<size_t>(head_dim), static_cast<AccT>(0));
        contribs.assign(static_cast<size_t>(src_seq_len), static_cast<AccT>(0));

        // compute dot-products: prod_buf holds per-d products, then pairwise reduce
        for (int s = 0; s < src_seq_len; ++s) {
          const size_t k_head_offset = k_batch_offset + static_cast<size_t>(s) * k_time_stride + static_cast<size_t>(kv_index) * head_stride;
          // fill prod_buf with elementwise products (use fma to compute product)
          for (int d = 0; d < head_dim; ++d) {
            AccT qv = static_cast<AccT>(h_q[q_head_offset + d]);
            AccT kv = static_cast<AccT>(h_k[k_head_offset + d]);
            prod_buf[static_cast<size_t>(d)] = std::fma(qv, kv, static_cast<AccT>(0));
          }
          // pairwise reduce prod_buf to get dot
          AccT dot = pairwise_reduce(prod_buf, head_dim);
          AccT scaled = dot * inv_sqrt_d;
          if (is_causal && s > t) scores[static_cast<size_t>(s)] = neg_inf;
          else scores[static_cast<size_t>(s)] = scaled;
        }

        // softmax (single-precision)
        AccT max_score = neg_inf;
        for (int s = 0; s < src_seq_len; ++s) {
          AccT v = scores[static_cast<size_t>(s)];
          if (v > max_score) max_score = v;
        }

        AccT sum_exp = static_cast<AccT>(0);
        if (max_score != neg_inf) {
          for (int s = 0; s < src_seq_len; ++s) {
            AccT sc = scores[static_cast<size_t>(s)];
            if (sc == neg_inf) {
              exps[static_cast<size_t>(s)] = static_cast<AccT>(0);
            } else {
              AccT e = static_cast<AccT>(::expf(sc - max_score));
              exps[static_cast<size_t>(s)] = e;
              sum_exp = std::fma(e, static_cast<AccT>(1.0f), sum_exp); // sum_exp += e using fma
            }
          }
        }

        if (sum_exp > static_cast<AccT>(0)) {
          // For each output dimension, compute pairwise sum of contributions over src positions
          for (int d = 0; d < head_dim; ++d) {
            // fill contribs[s] = w_s * v_s_d using fma for product
            for (int s = 0; s < src_seq_len; ++s) {
              AccT e = exps[static_cast<size_t>(s)];
              if (e == static_cast<AccT>(0)) {
                contribs[static_cast<size_t>(s)] = static_cast<AccT>(0);
              } else {
                AccT w = e / sum_exp;
                const size_t v_head_offset = k_batch_offset + static_cast<size_t>(s) * k_time_stride + static_cast<size_t>(kv_index) * head_stride;
                AccT vv = static_cast<AccT>(h_v[v_head_offset + d]);
                contribs[static_cast<size_t>(s)] = std::fma(w, vv, static_cast<AccT>(0));
              }
            }
            // pairwise reduce contributions to scalar and store in out_v[d]
            out_v[static_cast<size_t>(d)] = pairwise_reduce(contribs, src_seq_len);
          }
        } // else out_v remains zeros

        if (verbose) {
          if (b==0 && t==0 && qh==0) {
            int show_n = std::min(8, src_seq_len);
            std::fprintf(stderr, "[flashAttention-VERBOSE] shape b=%d t=%d qh=%d (show first %d scores/exps)\n", b, t, qh, show_n);
            for (int i = 0; i < show_n; ++i) {
              std::fprintf(stderr, "  s=%d score=%g exp=%g\n", i, static_cast<double>(scores[i]), static_cast<double>(exps[i]));
            }
            int show_o = std::min(8, head_dim);
            for (int i = 0; i < show_o; ++i) {
              std::fprintf(stderr, "  out_v[%d]=%g\n", i, static_cast<double>(out_v[i]));
            }
          }
        }

        const size_t o_head_offset = q_head_offset;
        for (int d = 0; d < head_dim; ++d) {
          AccT val = out_v[static_cast<size_t>(d)];
          if constexpr (std::is_same<T, half>::value) {
            float fval = static_cast<float>(val);
            h_o[o_head_offset + d] = __float2half(fval);
          } else {
            h_o[o_head_offset + d] = static_cast<T>(val);
          }
        }
      } // qh
    } // t
  } // b
}

// Explicit instantiations
template int trace<int>(const std::vector<int>&, size_t, size_t);
template float trace<float>(const std::vector<float>&, size_t, size_t);
template void flashAttention<float>(const std::vector<float>&, const std::vector<float>&,
  const std::vector<float>&, std::vector<float>&,
  int, int, int, int, int, int, bool);
template void flashAttention<half>(const std::vector<half>&, const std::vector<lhalf>&,
  const std::vector<half>&, std::vector<half>&,
  int, int, int, int, int, int, bool);