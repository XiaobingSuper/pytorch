#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <ATen/Parallel.h>
#include <ATen/TensorUtils.h>
#include <ATen/native/EmbeddingBag.h>

#include <TH/THBlasUtils.h>

#include <caffe2/perfkernels/embedding_lookup_idx.h>

#include <cstring>
#include <iostream>
#include <memory>
#include <sstream>
#include <vector>
#include <algorithm>


namespace {
  const int MODE_SUM = 0;
  const int MODE_MEAN = 1;
  const int MODE_MAX = 2;
}

namespace at {
namespace native {

namespace {

static inline bool isFastPathIndexSelect(const Tensor& src, Tensor& output) {
  return src.scalar_type() == kFloat && src.stride(1) == 1 && output.stride(1) == 1;
}

static inline bool isFastPathIndexSelectScale(const Tensor& src, const Tensor& scale, Tensor& output) {
  return src.scalar_type() == kFloat && src.stride(1) == 1 && output.stride(1) == 1 && scale.stride(0) == 1;
}

// This function combines index_select (using select_indices as the index) and
// index_add (using add_indices as the index), without creating an intermediary
// tensor to hold the selected embeddings
template<typename T>
void index_select_add(const Tensor &select_indices,
                             const Tensor &add_indices,
                             const Tensor &src,
                             Tensor &output,
                             const Tensor& /*offsets*/,
                             bool /*include_last_offset*/) {
  AT_ASSERT(select_indices.numel() == add_indices.numel());
  auto* add_indices_data = add_indices.data_ptr<int64_t>();
  auto* select_indices_data = select_indices.data_ptr<int64_t>();
  auto* src_data = src.data_ptr<T>();
  auto* output_data = output.data_ptr<T>();
  auto numel = add_indices.numel();
  int64_t ddim = src.size(1);
  auto src_stride0 = src.stride(0);
  auto src_stride1 = src.stride(1);
  auto output_stride0 = output.stride(0);
  auto output_stride1 = output.stride(1);

  for (int64_t i = 0; i < numel; i++) {
    THBlas_axpy<T>(ddim, 1,
            src_data + src_stride0 * select_indices_data[i], src_stride1,
            output_data + output_stride0 * add_indices_data[i], output_stride1);
  }
}

template<>
void index_select_add<float>(const Tensor &select_indices,
                             const Tensor &add_indices,
                             const Tensor &src,
                             Tensor &output,
                             const Tensor& offsets,
                             bool include_last_offset) {
  int64_t ddim = src.size(1);
  auto* src_data = src.data_ptr<float>();
  auto* select_indices_data = select_indices.data_ptr<int64_t>();
  auto* output_data = output.data_ptr<float>();

  if (isFastPathIndexSelect(src, output)) {
    int64_t output_size = offsets.numel() - 1;
    auto* offsets_data = offsets.data_ptr<int64_t>();
    std::vector<int64_t> offsets_include_last;

    if (include_last_offset) {
      output_size = offsets.numel() - 1;
    } else {
      output_size = offsets.numel();
      offsets_include_last.resize(offsets.numel() + 1);
      std::memcpy(
          offsets_include_last.data(),
          offsets.data_ptr<int64_t>(),
          sizeof(int64_t) * offsets.numel());
      offsets_include_last[offsets.numel()] = select_indices.numel();
      offsets_data = offsets_include_last.data();
    }

    at::parallel_for(
        0, output_size, 1, [&](int64_t start_idx, int64_t end_idx) {
          caffe2::EmbeddingLookupIdx(
              /*block_size=*/ddim,
              /*output_size=*/end_idx - start_idx,
              /*index_size=*/offsets_data[end_idx] - offsets_data[start_idx],
              /*data_size=*/src.size(0),
              /*input=*/src_data,
              /*indices=*/select_indices_data + offsets_data[start_idx],
              /*offsets=*/offsets_data + start_idx,
              /*weights=*/nullptr,
              /*scale_bias=*/nullptr,
              /*normalize_by_lengths=*/false,
              /*out=*/output_data + start_idx * ddim);
        });
  } else {
    AT_ASSERT(select_indices.numel() == add_indices.numel());
    auto* add_indices_data = add_indices.data_ptr<int64_t>();
    auto src_stride0 = src.stride(0);
    auto src_stride1 = src.stride(1);
    auto output_stride0 = output.stride(0);
    auto output_stride1 = output.stride(1);
    auto numel = add_indices.numel();
    for (int64_t i = 0; i < numel; i++) {
      THBlas_axpy<float>(
          ddim,
          1,
          src_data + src_stride0 * select_indices_data[i],
          src_stride1,
          output_data + output_stride0 * add_indices_data[i],
          output_stride1);
    }
  }
}

template<>
void index_select_add<int8_t>(const Tensor &select_indices,  //input
                             const Tensor &add_indices,  //offset2bag
                             const Tensor &src,  //weight
                             Tensor &output,
                             const Tensor& offsets,
                             bool include_last_offset) {
  int64_t ddim = src.size(1);
  auto src_data = src.data_ptr<qint8>();
  auto select_indices_data = select_indices.data_ptr<int64_t>();
  auto output_data = output.data_ptr<float>();
  auto scales = src.q_per_channel_scales();
  auto scales_data = scales.data_ptr<double>();
  int64_t output_size = offsets.numel() - 1;
  auto* offsets_data = offsets.data_ptr<int64_t>();
  std::vector<int64_t> offsets_include_last;
  if (include_last_offset) {
    output_size = offsets.numel() - 1;
  } else {
    output_size = offsets.numel();
    offsets_include_last.resize(offsets.numel() + 1);
    std::memcpy(
        offsets_include_last.data(),
        offsets.data_ptr<int64_t>(),
        sizeof(int64_t) * offsets.numel());
    offsets_include_last[offsets.numel()] = select_indices.numel();
    offsets_data = offsets_include_last.data();
  }
  at::parallel_for(
        0, output_size, 1, [&](int64_t start_idx, int64_t end_idx) {
      caffe2::pt_EmbeddingLookupIdx(
          /*block_size=*/src.size(1),
          /*output_size=*/end_idx - start_idx,
          /*index_size=*/offsets_data[end_idx] - offsets_data[start_idx],
          /*data_size=*/src.size(0),
          /*input=*/reinterpret_cast<int8_t*>(src_data),
          /*indices=*/select_indices_data + offsets_data[start_idx],
          /*offsets=*/offsets_data + start_idx,
          /*weights=*/nullptr,
          /*scales=*/scales_data,
          /*normalize_by_lengths=*/false,
          /*out=*/output_data + start_idx * ddim
        );
  });

}


// This function fuses the following three fns:
// index_select (using select_indices as the index)
// mul (scaling by per_sample_weights)
// index_add (using add_indices as the index)
template<typename T>
static void index_select_scale_add(const Tensor &select_indices,
                                   const Tensor &add_indices,
                                   const Tensor &scale,
                                   const Tensor &src,
                                   Tensor &output,
                                   const Tensor& /*offsets*/,
                                   bool /*include_last_offset*/) {
  AT_ASSERT(select_indices.numel() == add_indices.numel());
  auto* add_indices_data = add_indices.data_ptr<int64_t>();
  auto* select_indices_data = select_indices.data_ptr<int64_t>();
  auto* src_data = src.data_ptr<T>();
  auto* output_data = output.data_ptr<T>();
  auto numel = add_indices.numel();
  int64_t ddim = src.size(1);
  auto src_stride0 = src.stride(0);
  auto src_stride1 = src.stride(1);
  auto output_stride0 = output.stride(0);
  auto output_stride1 = output.stride(1);

  auto* scale_data = scale.data_ptr<T>();
  auto scale_stride = scale.stride(0);

  for (int64_t i = 0; i < numel; i++) {
    auto* src_base = src_data + src_stride0 * select_indices_data[i];
    auto* output_base = output_data + output_stride0 * add_indices_data[i];
    auto scale = scale_data[i * scale_stride];
    for (int64_t j = 0; j < ddim; j++) {
      output_base[j * output_stride1] += src_base[j * src_stride1] * scale;
    }
  }
}

template<>
void index_select_scale_add<float>(const Tensor &select_indices,
                                          const Tensor &add_indices,
                                          const Tensor &scale,
                                          const Tensor &src,
                                          Tensor &output,
                                          const Tensor& offsets,
                                          bool include_last_offset) {
  int64_t ddim = src.size(1);
  auto* scale_data = scale.data_ptr<float>();
  auto* select_indices_data = select_indices.data_ptr<int64_t>();
  auto* src_data = src.data_ptr<float>();
  auto* output_data = output.data_ptr<float>();

  if (isFastPathIndexSelectScale(src, scale, output)) {
    int64_t output_size = offsets.numel() - 1;
    auto* offsets_data = offsets.data_ptr<int64_t>();
    std::vector<int64_t> offsets_include_last;

    if (include_last_offset) {
      output_size = offsets.numel() - 1;
    } else {
      output_size = offsets.numel();
      offsets_include_last.resize(offsets.numel() + 1);
      std::memcpy(
          offsets_include_last.data(),
          offsets.data_ptr<int64_t>(),
          sizeof(int64_t) * offsets.numel());
      offsets_include_last[offsets.numel()] = select_indices.numel();
      offsets_data = offsets_include_last.data();
    }

    at::parallel_for(
        0, output_size, 1, [&](int64_t start_idx, int64_t end_idx) {
          caffe2::EmbeddingLookupIdx(
              /*block_size=*/ddim,
              /*output_size=*/end_idx - start_idx,
              /*index_size=*/offsets_data[end_idx] - offsets_data[start_idx],
              /*data_size=*/src.size(0),
              /*input=*/src_data,
              /*indices=*/select_indices_data + offsets_data[start_idx],
              /*offsets=*/offsets_data + start_idx,
              /*weights=*/scale_data + offsets_data[start_idx],
              /*scale_bias=*/nullptr,
              /*normalize_by_lengths=*/false,
              /*out=*/output_data + start_idx * ddim);
        });
  } else {
    AT_ASSERT(select_indices.numel() == add_indices.numel());
    auto* add_indices_data = add_indices.data_ptr<int64_t>();
    auto src_stride0 = src.stride(0);
    auto src_stride1 = src.stride(1);
    auto output_stride0 = output.stride(0);
    auto output_stride1 = output.stride(1);
    auto scale_stride = scale.stride(0);
    auto numel = add_indices.numel();


    for (int64_t i = 0; i < numel; i++) {
      auto* src_base = src_data + src_stride0 * select_indices_data[i];
      auto* output_base = output_data + output_stride0 * add_indices_data[i];
      auto scale = scale_data[i * scale_stride];
      for (int64_t j = 0; j < ddim; j++) {
        output_base[j * output_stride1] += src_base[j * src_stride1] * scale;
      }
    }
  }
}

}  // namespace

static at::Tensor make_bag_size(
    const Tensor& offsets,
    const Tensor& indices,
    const int64_t mode,
    const bool requires_grad) {
  at::Tensor bag_size;
  if (mode == MODE_MEAN || mode == MODE_MAX) {
    bag_size = at::zeros(offsets.sizes(), indices.options());
    // Compute this for MODE_MEAN and MODE_MAX (latter needed for backwards)
    if (offsets.size(0) != 1) {
      bag_size.slice(0, 0, bag_size.size(0) - 1, 1) =
          offsets.slice(0, 1, offsets.size(0), 1) -
          offsets.slice(0, 0, offsets.size(0) - 1, 1);
    }
    bag_size[-1] = indices.size(0) - offsets[-1];
  } else if (requires_grad) {
    // in MODE_SUM, only initialize bag_size if we need gradients
    bag_size = at::zeros(offsets.sizes(), indices.options());
  }
  return bag_size;
}

static Tensor apply_bag_size(const Tensor &offsets, const Tensor &indices,
                             const int64_t mode, Tensor &output,
                             const Tensor &bag_size) {
  if (mode == MODE_MEAN) {
    // Avoid dividing by 0 for empty bags.
    // Instead we want empty bags to return all 0s
    if (offsets.size(0) == 1) {
      auto bag_size_ = std::max(indices.size(0), static_cast<int64_t>(1));
      output /= bag_size_;
    } else {
      auto bag_size_ = at::max(bag_size, at::ones_like(bag_size, LEGACY_CONTIGUOUS_MEMORY_FORMAT))
                           .to(output.options())
                           .unsqueeze(1)
                           .expand_as(output);
      output /= bag_size_;
    }
  }
  return output;
}

static Tensor apply_bag_size_backward(const Tensor &offsets,
                                      const Tensor &indices, const int64_t mode,
                                      Tensor &output, const Tensor &offset2bag,
                                      const Tensor &bag_size) {
  if (mode == MODE_MEAN) {
    if (offsets.size(0) == 1) {
      auto bag_size_ = indices.size(0);
      output /= bag_size_;
    } else {
      auto inv_bag_size_ = (1 / bag_size.to(output.options()))
                             .unsqueeze(1)
                             .index_select(0, offset2bag);
      output *= inv_bag_size_;
    }
  }
  return output;
}


template <typename scalar_t>
std::tuple<Tensor, Tensor, Tensor, Tensor> embedding_bag_cpu_max(
    const Tensor& weight,
    const Tensor& indices,
    const Tensor& offset2bag,
    const Tensor& output,
    const Tensor& bag_size,
    const Tensor& offsets) {

    auto max_indices = at::zeros({offsets.size(0), weight.size(1)}, indices.options());

    int64_t numel = indices.numel();
    int64_t dims = weight.size(1);
    auto* indices_data = indices.data_ptr<int64_t>();
    auto* offset2bag_data = offset2bag.data_ptr<int64_t>();

    auto* max_indices_data = max_indices.data_ptr<int64_t>();
    auto max_indices_stride = max_indices.stride(0);

    auto* weight_data = weight.data_ptr<scalar_t>();
    auto* output_data = output.data_ptr<scalar_t>();
    auto weight_stride0 = weight.stride(0);
    auto weight_stride1 = weight.stride(1);
    auto output_stride = output.stride(0);

    for (int i = 0; i < numel; i++) {
      auto bag = offset2bag_data[i];
      auto word_idx = indices_data[i];

      for (int dim = 0; dim < dims; dim++) {
        auto& current_item = output_data[output_stride * bag + dim];
        auto weight_item = weight_data[weight_stride0 * word_idx + dim * weight_stride1];
        bool is_first_for_bag = (i == 0) || offset2bag_data[i - 1] != bag;

        if (is_first_for_bag || weight_item > current_item) {
          current_item = weight_item;
          max_indices_data[max_indices_stride * bag + dim] = word_idx;
        }
      }
    }

    return std::tuple<Tensor, Tensor, Tensor, Tensor>(output, offset2bag, bag_size, max_indices);
}

// embedding_bag wrapper to enforce contiguity in tensors other than `weight`.
// This is created to save extra `.contiguous()` call in backward.
// See NOTE [ embedding_bag Native Functions ] in native_functions.yaml for details
std::tuple<Tensor, Tensor, Tensor, Tensor>
embedding_bag(const Tensor &weight, const Tensor &indices,
              const Tensor &offsets, const bool scale_grad_by_freq,
              const int64_t mode, bool sparse,
              const Tensor &per_sample_weights,
              bool include_last_offset) {
  return at::_embedding_bag(weight, indices.contiguous(), offsets.contiguous(),
                            scale_grad_by_freq, mode, sparse, per_sample_weights, include_last_offset);
  };

// Assumes all input tensors except for `weight` are contiguous.
// See NOTE [ embedding_bag Native Functions ] in native_functions.yaml for details
std::tuple<Tensor, Tensor, Tensor, Tensor>
_embedding_bag_cpu(const Tensor &weight, const Tensor &indices,
                  const Tensor &offsets, const bool scale_grad_by_freq,
                  const int64_t mode, bool sparse,
                  const Tensor &per_sample_weights, bool include_last_offset) {
  auto indices_arg = TensorArg(indices, "indices", 1);
  checkScalarType("embedding_bag", indices_arg, kLong);
  auto offsets_arg = TensorArg(offsets, "offsets", 1);
  checkScalarType("embedding_bag", offsets_arg, kLong);
  auto weight_arg = TensorArg(weight, "weight", 1);
  checkScalarTypes("embedding_bag", weight_arg, {kFloat, kDouble, at::kQInt8});
  int64_t offset_0 = offsets.data_ptr<int64_t>()[0];
  int64_t offset_n = offsets.data_ptr<int64_t>()[offsets.size(0)-1];
  TORCH_CHECK(offset_0 == 0, "offsets[0] has to be 0, i.e., the first sequence "
                             "in the mini-batch has to start from position 0. "
                             "However, got ", offsets[0]);
  TORCH_CHECK(offset_n <= indices.size(0), "offsets[-1] can not "
               "be greater than input's length ", indices.size(0), " but got offsets[-1] of ",
               offset_n);

  if (per_sample_weights.defined()) {
    TORCH_CHECK(mode == MODE_SUM,
        "embedding_bag: per_sample_weights only supported with mode='sum'");
    auto per_input_weights_arg = TensorArg(
        per_sample_weights,"per_sample_weights", 1);
    checkSameType("embedding_bag", weight_arg, per_input_weights_arg);
    TORCH_CHECK(per_sample_weights.dim() == 1);
    TORCH_CHECK(per_sample_weights.numel() == indices.numel());
  }

  auto bag_size = make_bag_size(offsets, indices, mode, weight.requires_grad());

  if (include_last_offset) {
    TORCH_CHECK(
        offsets.size(0) >= 1,
        "include_last_offset: number of offset should be at least 1");
  }

  auto output = at::zeros(
      {include_last_offset ? offsets.size(0) - 1 : offsets.size(0),
       weight.size(1)},
      weight.is_quantized() ? at::device(c10::kCPU).dtype(c10::kFloat) : weight.options());

  // To save compute, if we are going to go down the fast path case for the 'sum'
  // mode, we skip calculating offset2bag, since it is not going to be used.
  auto fast_path_sum = [&weight, &per_sample_weights, &output]() {
    if (per_sample_weights.defined()) {
      return isFastPathIndexSelectScale(weight, per_sample_weights, output);
    } else {
      return isFastPathIndexSelect(weight, output);
    }
  };

  // Use an empty 0-element tensor as a sentinel that we have skipped the
  // creation of offset2bag because autograd chokes when trying to use an
  // undefined tensor as an input to a backward op.
  Tensor offset2bag = at::empty({0}, offsets.options());
  if (mode == MODE_MEAN || mode == MODE_MAX || !fast_path_sum()) {
    // If the last entries are empty, that the last offsets are irrelevant as they
    // won't change anything in the assignment of ID -> bag, but index_add would
    // throw out of bounds error. So to keep it simple we just add one more
    // entry to the end then get rid of it after make_offset2bag.
    offset2bag = at::zeros(
       {indices.sizes()[0] + 1}, indices.options()); // offset2bag = [0 0 0 0 0]

    make_offset2bag(offsets, indices, offset2bag);

    offset2bag.resize_({indices.sizes()[0]});
  }

  if (mode == MODE_MEAN || mode == MODE_SUM) {
   if (weight.is_quantized()) {
      AT_ASSERT(mode == MODE_SUM);
      AT_ASSERT(!per_sample_weights.defined());
      AT_ASSERT(weight.scalar_type() == at::kQInt8);
      AT_ASSERT(weight.is_contiguous() && output.is_contiguous());
      index_select_add<int8_t>(indices, offset2bag, weight, output, offsets, include_last_offset);
    } else {
      AT_DISPATCH_FLOATING_TYPES(weight.scalar_type(), "embedding_bag_cpu", [&]() {
        if (per_sample_weights.defined()) {
          AT_ASSERT(mode == MODE_SUM);
          index_select_scale_add<scalar_t>(
              indices, offset2bag, per_sample_weights, weight, output, offsets, include_last_offset);
        } else {
          index_select_add<scalar_t>(indices, offset2bag, weight, output, offsets, include_last_offset);
        }
    });
    }
    auto ret = apply_bag_size(offsets, indices, mode, output, bag_size);
    return std::tuple<Tensor, Tensor, Tensor, Tensor>(ret, offset2bag, bag_size, bag_size);
  } else { // MODE_MAX
    at::optional<Tensor> maybe_per_sample_weights;
    if (per_sample_weights.defined()) {
      maybe_per_sample_weights = per_sample_weights;
    }
    return AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      weight.scalar_type(), "embedding_bag_cpu_max", [&]() {
        return embedding_bag_cpu_max<scalar_t>(
            weight, indices, offset2bag, output, bag_size, offsets);
      }
    );
  }
}

Tensor _embedding_bag_sparse_backward_cpu_sum_fast(
    const Tensor &grad, const Tensor &indices, const Tensor &offsets, int64_t num_weights, int64_t mode, const Tensor& per_sample_weights) {

  AT_ASSERT(mode == MODE_SUM);
  AT_ASSERT((grad.scalar_type() == kFloat)&& (grad.stride(1) == 1) && !per_sample_weights.defined());

  int64_t indices_size0 = indices.size(0);
  int64_t ddim = grad.size(1);
  Tensor index_grad = at::empty({indices_size0, ddim}, grad.options());
  float* gradout_data = index_grad.data_ptr<float>();

  auto offsets_accessor = offsets.accessor<int64_t, 1>();
  auto offset_numel = offsets.numel();

  float* grad_data = grad.data_ptr<float>();
  int grad_stride0 = grad.stride(0);
  at::parallel_for(0, offset_numel, 0, [&](int64_t start, int64_t end) {
    for(auto mb = start; mb < end; mb++) {
      int64_t select_off_start = offsets_accessor[mb];
      int64_t select_off_end = (mb < (offset_numel - 1) ? offsets_accessor[mb + 1] : indices_size0);
      auto grad_block = grad_data + grad_stride0 * mb;;
      for (int64_t s = select_off_start; s < select_off_end; s++) {
        THBlas_copy<float>(ddim, grad_block, 1, gradout_data + ddim * s, 1);
      }
    }
  });

  int64_t num_features = index_grad.size(-1);
  auto weight_size = std::array<int64_t, 2>{{ num_weights, num_features }};
  auto dense_options = index_grad.options();

  if (index_grad.numel() == 0) {
    return at::_sparse_coo_tensor_unsafe(at::empty({1, 0}, indices.options()),
                                         at::empty({0, num_features}, dense_options),
                                         weight_size);
  }

  auto index = indices.reshape({1, -1});
  auto values = index_grad.reshape({-1, num_features});

  return at::_sparse_coo_tensor_unsafe(index, values, weight_size);

}

Tensor _embedding_bag_dense_backward_cpu_sum_fast(
    const Tensor &grad, const Tensor &indices, const Tensor &offsets, int64_t num_weights, int64_t mode, const Tensor& per_sample_weights) {

  AT_ASSERT(mode == MODE_SUM);
  AT_ASSERT((grad.scalar_type() == kFloat)&& (grad.stride(1) == 1) && !per_sample_weights.defined());

  int64_t indices_numel = indices.numel();
  auto offset_numel = offsets.numel();

  Tensor offset2bag;
  if (indices_numel != offset_numel) {
    offset2bag = at::zeros(
      {indices.sizes()[0] + 1}, indices.options()); // offset2bag = [0 0 0 0 0]
    make_offset2bag(offsets, indices, offset2bag);
    offset2bag.resize_({indices.sizes()[0]});
  } else {
    offset2bag = offsets;
  }

  int64_t ddim = grad.size(1);
  Tensor index_grad_weight = at::zeros({num_weights, ddim}, grad.options());

  int64_t grad_length = index_grad_weight.size(0);
  int max_threads = at::get_num_threads();
  max_threads = (grad_length < max_threads) ? grad_length : max_threads;
  int64_t avg_chunk_down = grad_length / max_threads;
  int64_t chuck_size[max_threads];
  for (auto i = 0; i < max_threads; i++) {
    chuck_size[i] = avg_chunk_down;
  }
  //make chunk balance among threads as 211
  for (auto i = 0 ; i < grad_length % max_threads ; i++) {
    chuck_size[i] += 1;
  }
  int64_t chuck_sum_size[max_threads + 1];
  chuck_sum_size[0] = 0;
  for (auto i = 1; i < max_threads; i++) {
    chuck_sum_size[i] = chuck_sum_size[i - 1] + chuck_size[i - 1];
  }
  chuck_sum_size[max_threads] = grad_length;

  auto* indices_data = indices.data_ptr<int64_t>();
  auto* offset2bag_data = offset2bag.data_ptr<int64_t>();
  auto* grad_data = grad.data_ptr<float>();
  auto* gradout_data = index_grad_weight.data_ptr<float>();
  int64_t grad_stride0 = grad.stride(0);
  at::parallel_for(0, max_threads, 0, [&](int64_t start, int64_t end) {
    for(auto k = start; k < end; k++) {
      int64_t chunk_start = chuck_sum_size[k];
      int64_t chunk_end = chuck_sum_size[k + 1];
      for (int64_t mb = 0; mb < indices_numel; mb++) {
        int64_t index = indices_data[mb];
        if (index >= chunk_start && index < chunk_end) {
          auto s = offset2bag_data[mb];
          THBlas_axpy<float>(ddim, 1.0, grad_data + grad_stride0 * s, 1, gradout_data + ddim * index, 1);
        }
      }
    }
  });

  return index_grad_weight;
}

// To save compute, if we are going to go down the fast path case for the 'sum'
// mode, we skip calculating offset2bag, since it is not going to be used.
static inline bool _embedding_bag_fast_path_sum(const Tensor& grad,
      const Tensor &indices,
      const Tensor &offset2bag,
      const Tensor& per_sample_weights,
       bool scale_grad_by_freq,
      int64_t mode) {

  if (at::get_num_threads() == 1) return false;
  if (offset2bag.numel() != 0 || indices.numel() == 0) return false;
  if (mode != MODE_SUM || grad.scalar_type() != kFloat || grad.stride(1) != 1) return false;
  if (per_sample_weights.defined() || scale_grad_by_freq) return false;
  return true;
}

// Assumes all input tensors are contiguous.
// See NOTE [ embedding_bag Native Functions ] in native_functions.yaml for details
Tensor _embedding_bag_backward_cpu(const Tensor &grad, const Tensor &indices,
                              const Tensor &offsets,
                              const Tensor &offset2bag,
                              const Tensor &bag_size_,
                              const Tensor &max_indices_,
                              int64_t num_weights,
                              bool scale_grad_by_freq, int64_t mode,
                              bool sparse,
                              const Tensor& per_sample_weights) {
  auto indices_arg = TensorArg(indices, "indices", 1);
  checkScalarType("embedding_bag", indices_arg, kLong);
  checkContiguous("embedding_bag", indices_arg);
  auto offsets_arg = TensorArg(offsets, "offsets", 1);
  checkScalarType("embedding_bag", offsets_arg, kLong);
  checkContiguous("embedding_bag", offsets_arg);

  if (_embedding_bag_fast_path_sum(grad, indices, offset2bag, per_sample_weights, scale_grad_by_freq, mode)) {
    if (sparse) {
      return _embedding_bag_sparse_backward_cpu_sum_fast(grad, indices, offsets, num_weights, mode, per_sample_weights);
    } else {
      return _embedding_bag_dense_backward_cpu_sum_fast(grad, indices, offsets, num_weights, mode, per_sample_weights);
    }
  }

  Tensor offset2bag_;
  if (indices.numel() != 0 && offset2bag.numel() == 0) {
    offset2bag_ = at::zeros(
      {indices.sizes()[0] + 1}, indices.options()); // offset2bag = [0 0 0 0 0]

    make_offset2bag(offsets, indices, offset2bag_);

    offset2bag_.resize_({indices.sizes()[0]});
  } else {
    auto offset2bag_arg = TensorArg(offset2bag, "offset2bag", 1);
    checkScalarType("embedding_bag", offset2bag_arg, kLong);
    checkContiguous("embedding_bag", offset2bag_arg);
    offset2bag_ = offset2bag;
  }

  if (sparse) {
    return at::_embedding_bag_sparse_backward(
        grad, indices, offsets, offset2bag_, bag_size_, num_weights,
        scale_grad_by_freq, mode, per_sample_weights);
  } else {
    return at::_embedding_bag_dense_backward(
        grad, indices, offsets, offset2bag_, bag_size_, max_indices_, num_weights,
        scale_grad_by_freq, mode, per_sample_weights);
  }
}

static Tensor _embedding_bag_dense_backward_cpu_max(
    const Tensor& grad,
    const Tensor& bag_size,
    const Tensor& max_indices,
    int64_t num_weights) {
  AT_ASSERT(max_indices.defined());
  auto index_grad_weight =
      at::zeros({num_weights, grad.size(1)}, grad.options());
  auto nonempty_max_indices = max_indices.index_select(0, bag_size.nonzero().view(-1));
  auto nonempty_grad = grad.index_select(0, bag_size.nonzero().view(-1));

  for (int64_t dim = 0; dim < grad.size(1); dim++) {
    index_grad_weight.select(1, dim).index_add_(
      0, nonempty_max_indices.select(1, dim), nonempty_grad.select(1, dim));
  }
  return index_grad_weight;
}

static std::vector<int64_t> compute_counts(
    int64_t num_weights,
    int64_t* indices_data,
    int64_t indices_length) {
  std::vector<int64_t> counts(num_weights, 0);
  for (int i = 0; i < indices_length; i++) {
    counts[indices_data[i]]++;
  }
  return counts;
}

// counts_uniq stores the index of the NEXT unique element
// of the (sorted) indices vector.
//
// For example:
// indices: [0, 0, 0, 1, 3, 3, 4]
// counts: [3, 1, 0, 2, 1, 0]
// counts_uniq: [3, 4, 6, 7]
//
// The unique indices can be found at index 0, 3, 4, 6.
static std::vector<int64_t> compute_counts_uniq(
    int64_t num_weights,
    int64_t* indices_data,
    int64_t indices_length,
    const std::vector<int64_t>& counts) {
  std::vector<int64_t> counts_uniq;
  counts_uniq.reserve(num_weights);
  int64_t o = 0;
  for (int64_t i = 0; i < indices_length; i += counts[indices_data[i]]) {
    counts_uniq.push_back(counts[indices_data[i]]);
    if (o > 0) {
      counts_uniq[o] += counts_uniq[o - 1];
    }
    o++;
  }
  return counts_uniq;
}

template <typename scalar_t>
void _embedding_bag_dense_backward_cpu_sum_mean(
    const Tensor& grad,
    const Tensor& indices_,
    const Tensor& offsets_,
    const Tensor& offset2bag__,
    int64_t num_weights,
    bool scale_grad_by_freq,
    int64_t mode,
    const Tensor& per_sample_weights_,
    Tensor& index_grad_weight) {

  Tensor &offset2bag_ = const_cast<Tensor &>(offset2bag__);

  auto ind_sort_ = indices_.sort();
  auto indices = std::get<0>(ind_sort_);
  auto ind_sort = std::get<1>(ind_sort_);
  auto offset2bag = offset2bag_.index_select(0, ind_sort);

  optional<Tensor> per_sample_weights;
  scalar_t* per_sample_weights_data;
  optional<int64_t> per_sample_weights_stride;
  if (per_sample_weights_.defined()) {
    per_sample_weights = per_sample_weights_.index_select(0, ind_sort);
    per_sample_weights_data = per_sample_weights->data_ptr<scalar_t>();
    per_sample_weights_stride = per_sample_weights->stride(0);
  }

  auto* indices_data = indices.data_ptr<int64_t>();
  auto* offsets_data = offsets_.data_ptr<int64_t>();
  auto* offset2bag_data = offset2bag.data_ptr<int64_t>();
  int64_t numel = indices.numel();

  auto counts = compute_counts(num_weights, indices_data, numel);
  auto next_unique_index_idx =
      compute_counts_uniq(num_weights, indices_data, numel, counts);

  int64_t ddim = grad.size(1);
  auto igwd = index_grad_weight.data_ptr<scalar_t>();
  auto gd = grad.data_ptr<scalar_t>();
  auto loop = [&](int64_t start, int64_t end) {
    for (int64_t i = start; i < end; i++) {
      int64_t start = i == 0 ? 0 : next_unique_index_idx[i - 1];
      int64_t index = indices_data[start];
      for (int64_t j = start; j < next_unique_index_idx[i]; j++) {
        int64_t source = offset2bag_data[j];
        double scale = 1.0;
        if (per_sample_weights) {
          AT_ASSERT(mode == MODE_SUM);
          scale = per_sample_weights_data[*per_sample_weights_stride * j];
        }
        if (scale_grad_by_freq) {
          scale /= counts[indices_data[i]];
        }
        if (mode == 1) { // MODE_MEAN
          if (offsets_.size(0) == 1) {
            auto bag_size = indices.size(0);
            scale /= bag_size;
          } else {
            if (source == offsets_.size(0) - 1) {
              scale /= indices.size(0) - offsets_data[offsets_.size(0) - 1];
            } else {
              scale /= offsets_data[source + 1] - offsets_data[source];
            }
          }
        }
        THBlas_axpy<scalar_t>(ddim, (scalar_t)scale, gd + ddim * source, 1,
                    igwd + ddim * index, 1);
      }
    }
  };
  if (numel > 1000) {
    at::parallel_for(0, (int64_t)next_unique_index_idx.size(), 0, loop);
  } else {
    loop(0, (int64_t)next_unique_index_idx.size());
  }
}

Tensor _embedding_bag_dense_backward_cpu(const Tensor &grad_, const Tensor &indices_,
                                  const Tensor &offsets_,
                                  const Tensor &offset2bag__,
                                  const Tensor &bag_size_,
                                  const Tensor& max_indices_, int64_t num_weights,
                                  bool scale_grad_by_freq, int64_t mode,
                                  const Tensor& per_sample_weights_) {
  // indices_, offsets_ and offset2bag__ are assumed having correct dtypes and
  // contiguous here due to the checks in _embedding_bag_backward above.
  // Also see NOTE [ embedding_bag Native Functions ] in native_functions.yaml
  // for more details.
  auto grad = grad_.contiguous();
  auto grad_arg = TensorArg(grad, "grad_", 1);
  checkScalarTypes("embedding_bag", grad_arg, {kFloat, kDouble});

  if (mode == MODE_MAX) {
    return _embedding_bag_dense_backward_cpu_max(
        grad_, bag_size_, max_indices_, num_weights);
  }
  AT_ASSERT(mode == MODE_MEAN || mode == MODE_SUM);

  auto index_grad_weight =
      at::zeros({num_weights, grad.size(1)}, grad.options());

  AT_DISPATCH_FLOATING_TYPES(grad.scalar_type(), "embedding_bag_backward", [&] {
      _embedding_bag_dense_backward_cpu_sum_mean<scalar_t>(
          grad, indices_, offsets_, offset2bag__, num_weights,
          scale_grad_by_freq, mode, per_sample_weights_, index_grad_weight);
  });
  return index_grad_weight;
}

template<typename scalar_t>
Tensor _embedding_bag_per_sample_weights_backward_cpu_template(
    const Tensor& grad,
    const Tensor& weight,  // NB: embedding table, not per_sample_weights
    const Tensor& indices,
    const Tensor& offsets,
    const Tensor& offset2bag,
    int64_t mode) {
  TORCH_CHECK(
      mode == MODE_SUM,
      "embedding_bag_backward: per_sample_weights only supported for mode='sum'");

  AT_ASSERT(grad.dim() == 2);
  auto embedding_features = grad.size(1);

  AT_ASSERT(indices.dim() == 1);
  auto num_samples = indices.size(0);

  AT_ASSERT(weight.dim() == 2);
  AT_ASSERT(weight.size(1) == embedding_features);

  auto output = at::zeros({num_samples}, grad.options());

  auto indices_arg = TensorArg(indices, "indices", 1);
  checkScalarType("embedding_bag", indices_arg, kLong);
  checkContiguous("embedding_bag", indices_arg);

  Tensor offset2bag_;
  if (indices.numel() != 0 && offset2bag.numel() == 0) {
    offset2bag_ = at::zeros(
       {indices.sizes()[0] + 1}, indices.options()); // offset2bag = [0 0 0 0 0]

    make_offset2bag(offsets, indices, offset2bag_);

    offset2bag_.resize_({indices.sizes()[0]});
  } else {
    auto offset2bag_arg = TensorArg(offset2bag, "offset2bag", 1);
    checkScalarType("embedding_bag", offset2bag_arg, kLong);
    checkContiguous("embedding_bag", offset2bag_arg);
    offset2bag_ = offset2bag;
  }

  auto* grad_data = grad.data_ptr<scalar_t>();
  auto grad_stride0 = grad.stride(0);
  auto grad_stride1 = grad.stride(1);

  auto* weight_data = weight.data_ptr<scalar_t>();
  auto weight_stride0 = weight.stride(0);
  auto weight_stride1 = weight.stride(1);

  auto* indices_data = indices.data_ptr<int64_t>();

  // The following are contiguous
  auto* output_data = output.data_ptr<scalar_t>();
  auto* offset2bag_data = offset2bag_.data_ptr<int64_t>();

  // XXX: 64 was arbitrarily chosen. There is probably a sweet spot for this number.
  parallel_for(0, num_samples, 64, [&](int64_t begin, int64_t end) {
    for (int64_t sample_idx = begin; sample_idx < end; sample_idx++) {
      auto bag_idx = offset2bag_data[sample_idx];
      auto embedding_idx = indices_data[sample_idx];

      output_data[sample_idx] = THBlas_dot<scalar_t>(
          embedding_features,
          grad_data + grad_stride0 * bag_idx, grad_stride1,
          weight_data + weight_stride0 * embedding_idx, weight_stride1);
    }
  });
  return output;
}

Tensor _embedding_bag_per_sample_weights_backward_cpu(
    const Tensor& grad,
    const Tensor& weight,  // NB: embedding table, not per_sample_weights
    const Tensor& indices,
    const Tensor& offsets,
    const Tensor& offset2bag,
    int64_t mode) {
  return AT_DISPATCH_FLOATING_TYPES(
    grad.scalar_type(), "_embedding_bag_per_sample_weights_backward_cpu", [&]() {
      return _embedding_bag_per_sample_weights_backward_cpu_template<scalar_t>(
          grad, weight, indices, offsets, offset2bag, mode);
    }
  );
}

Tensor _embedding_bag_sparse_backward(
    const Tensor &grad_, const Tensor &indices, const Tensor &offsets,
    const Tensor &offset2bag, const Tensor &bag_size_, int64_t num_weights,
    bool scale_grad_by_freq, int64_t mode, const Tensor& per_sample_weights) {
  // indices, offsets and offset2bag are assumed having correct dtypes and
  // contiguous here due to the checks in _embedding_bag_backward above.
  // Also see NOTE [ embedding_bag Native Functions ] in native_functions.yaml
  // for more details.

  Tensor grad = grad_;
  Tensor index_grad = grad_.index_select(0, offset2bag);
  index_grad = apply_bag_size_backward(offsets, indices, mode, index_grad,
                                       offset2bag, bag_size_);
  if (per_sample_weights.defined()) {
    AT_ASSERT(mode == MODE_SUM);
    index_grad.mul_(per_sample_weights.unsqueeze(1));
  }
  return native::embedding_backward(index_grad, indices, num_weights, -1,
                                    scale_grad_by_freq, true);
}
}
} // namespace at::native
