#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/native/mkldnn/Utils.h>
#include <ATen/native/Pool.h>
#include <c10/util/irange.h>

namespace at { namespace native {

std::vector<int64_t> pool_output_sizes(
    IntArrayRef input_size,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding_l,
    IntArrayRef padding_r,
    IntArrayRef dilation,
    bool ceil_mode) {
  std::vector<int64_t> output_size(input_size.size());
  // copy N and C
  output_size[0] = input_size[0];
  output_size[1] = input_size[1];

  for (const auto i : c10::irange(2, input_size.size())) {
    output_size[i] = pooling_output_shape_pad_lr<int64_t>(
      input_size[i],
      kernel_size[i - 2],
      padding_l[i - 2],
      padding_r[i - 2],
      stride[i - 2],
      dilation[i - 2],
      ceil_mode
    );
  }

   return output_size;
}

#if AT_MKLDNN_ENABLED()

#define ATTR_FUNC(NAME)                              \
  [](std::vector<c10::optional<at::Scalar>> scalars, \
     c10::optional<std::string> algorithm) {         \
    return ideep::attr_t::fuse_##NAME();             \
  }

static constexpr float kMin = -std::numeric_limits<float>::infinity();
static constexpr float kMax = std::numeric_limits<float>::infinity();

AttrFunction attr_func_none = [](std::vector<c10::optional<at::Scalar>> scalars,
                                 c10::optional<std::string> algorithm) {
  const static ideep::attr_t empty_attr = ideep::attr_t();
  return empty_attr;
};

AttrFunction attr_func_leaky_relu =
    [](std::vector<c10::optional<at::Scalar>> scalars,
       c10::optional<std::string> algorithm) {
      auto alpha_value = scalars[0].value().to<float>();
      return ideep::attr_t::fuse_relu(1.0, alpha_value);
    };

AttrFunction attr_func_hardtanh =
    [](std::vector<c10::optional<at::Scalar>> scalars,
       c10::optional<std::string> algorithm) {
      auto lower_bound_value = scalars[0].value().to<float>();
      auto upper_bound_value = scalars[1].value().to<float>();
      return ideep::attr_t::fuse_clamp(lower_bound_value, upper_bound_value);
    };

AttrFunction attr_func_gelu = [](std::vector<c10::optional<at::Scalar>> scalars,
                                 c10::optional<std::string> algorithm) {
  dnnl::algorithm gelu_type;
  if (algorithm.value() == "none") {
    gelu_type = dnnl::algorithm::eltwise_gelu_erf;
  } else if (algorithm.value() == "tanh") {
    gelu_type = dnnl::algorithm::eltwise_gelu_tanh;
  } else {
    TORCH_CHECK(
        false, "ipex::linear_gelu_run only support tanh approximate now");
  }

  return ideep::attr_t::fuse_gelu(1.0, 0.f, 0.f, gelu_type);
};

AttrFunction attr_func_clamp =
    [](std::vector<c10::optional<at::Scalar>> scalars,
       c10::optional<std::string> algorithm) {
      float lower_bound_value =
          scalars[0] ? scalars[0].value().to<float>() : kMin;
      float upper_bound_value =
          scalars[1] ? scalars[1].value().to<float>() : kMax;

      return ideep::attr_t::fuse_clamp(lower_bound_value, upper_bound_value);
    };

const std::map<std::string, AttrFunction>& fusion_attr_map() {
  static const std::map<std::string, AttrFunction> fusion_attr_map{
      {"none", attr_func_none},
      {"relu", ATTR_FUNC(relu)},
      {"sigmoid", ATTR_FUNC(sigmoid)},
      {"tanh", ATTR_FUNC(tanh)},
      {"leaky_relu", attr_func_leaky_relu},
      {"hardtanh", attr_func_hardtanh},
      {"gelu", attr_func_gelu},
      {"clamp", attr_func_clamp},
  };
  return fusion_attr_map;
};

const std::map<std::string, ideep::algorithm>& fusion_binary_alg_map() {
  static const std::map<std::string, ideep::algorithm> fusion_attr_map{
      {"add", {ideep::algorithm::binary_add}},
      {"sub", {ideep::algorithm::binary_sub}},
  };
  return fusion_attr_map;
};

#endif // AT_MKLDNN_ENABLED()

}}
