#include <vector>

#include <ATen/native/ConvUtils.h>
#include <ATen/native/mkldnn/Common.h>
#include <ATen/native/mkldnn/ConvPrepack.h>
#include <ATen/native/mkldnn/MKLDNNCommon.h>
#include <ATen/native/mkldnn/OpContext.h>
#include <ATen/native/utils/Factory.h>
#include <ATen/native/utils/ParamUtils.h>
#include <c10/util/irange.h>

#if AT_MKLDNN_ENABLED()

namespace at {
namespace native {
namespace mkldnn {
namespace internal {
namespace convolution {

c10::intrusive_ptr<mkldnn::ConvOpContext> createConvPrePackOpContext(
    Tensor weight,
    c10::optional<Tensor> bias,
    std::vector<int64_t> stride,
    std::vector<int64_t> padding,
    std::vector<int64_t> dilation,
    int64_t groups,
    std::vector<int64_t> input_size,
    std::string attr) {
  auto it_elt = fusion_attr_map.find(attr);
  auto it_binary = fusion_binary_alg_map.find(attr);
  TORCH_CHECK(
      it_elt != fusion_attr_map.end() ||
          it_binary != fusion_binary_alg_map.end(),
      "Fusion behavior undefined.");
  ideep::attr_t op_attr;
  if (it_elt != fusion_attr_map.end()) {
    op_attr = it_elt->second;
  } else {
    // TODO: what if input is nhwc but w is nchw
    bool is_channels_last =
        weight.suggest_memory_format() == at::MemoryFormat::ChannelsLast;
    ideep::tag other_tag =
        is_channels_last ? ideep::tag::nhwc : ideep::tag::any;
    auto k = weight.ndimension();
    int64_t dim = k - 2;
    const auto padding_expanded =
        expand_param_if_needed(padding, "padding", dim);
    const auto stride_expanded = expand_param_if_needed(stride, "stride", dim);
    const auto dilation_expanded =
        expand_param_if_needed(dilation, "dilation", dim);
    const auto input_size_expanded =
        expand_param_if_needed(input_size, "input_size", k);
    auto output_size = conv_output_size(
        input_size_expanded,
        weight.sizes(),
        padding_expanded,
        stride_expanded,
        dilation_expanded);
    auto other_desc = ideep::tensor::desc(
        output_size, get_mkldnn_dtype(weight.scalar_type()), other_tag);
    op_attr = ideep::attr_t::fuse_binary(it_binary->second, other_desc);
  }
  return mkldnn::MkldnnConvOpContext::create_context(
      std::move(weight),
      std::move(bias),
      std::move(padding),
      std::move(stride),
      std::move(dilation),
      groups,
      std::move(input_size),
      op_attr);
}

ContextConv create(
    const Tensor& weight,
    const c10::optional<Tensor>& bias,
    const IntArrayRef padding,
    const IntArrayRef stride,
    const IntArrayRef dilation,
    const int64_t groups,
    const IntArrayRef input_size,
    const ideep::attr_t& attr) {
  auto k = weight.ndimension();
  int64_t dim = k - 2;
  const auto padding_expanded = expand_param_if_needed(padding, "padding", dim);
  const auto stride_expanded = expand_param_if_needed(stride, "stride", dim);
  const auto dilation_expanded =
      expand_param_if_needed(dilation, "dilation", dim);
  const auto input_size_expanded =
      expand_param_if_needed(input_size, "input_size", k);

  c10::impl::ExcludeDispatchKeyGuard edkg(c10::autograd_dispatch_keyset);
  auto w = itensor_view_from_dense(weight);
  // TODO: what if input is nhwc but w is nchw
  bool is_channels_last =
      weight.suggest_memory_format() == at::MemoryFormat::ChannelsLast;
  ideep::tensor::desc expected_weight_desc =
      ideep::convolution_forward::expected_weights_desc(
          w.get_dims(),
          w.get_data_type(),
          {stride_expanded.begin(), stride_expanded.end()},
          {padding_expanded.begin(), padding_expanded.end()},
          {padding_expanded.begin(), padding_expanded.end()},
          {dilation_expanded.begin(), dilation_expanded.end()},
          groups,
          ideep::algorithm::convolution_direct,
          ideep::prop_kind::forward,
          /*x_dtype*/ w.get_data_type(),
          {input_size_expanded.begin(), input_size_expanded.end()},
          attr,
          is_channels_last);

  ideep::tensor packed_weight;
  packed_weight.init(expected_weight_desc);
  packed_weight.feed_from(w);

  return ContextConv{
      std::move(packed_weight),
      bias.has_value() ? c10::make_optional(*bias) : c10::nullopt,
      {padding_expanded.begin(), padding_expanded.end()},
      {stride_expanded.begin(), stride_expanded.end()},
      {dilation_expanded.begin(), dilation_expanded.end()},
      groups,
      std::move(attr)};
}

void _mkldnn_convolution_out(
    const ideep::tensor& x,
    ideep::tensor& y,
    const ideep::tensor& w,
    const c10::optional<ideep::tensor>& b,
    IntArrayRef padding,
    IntArrayRef stride,
    IntArrayRef dilation,
    IntArrayRef output_sizes,
    int64_t groups,
    const ideep::attr_t& attr = ideep::attr_t()) {
  if (b.has_value()) {
    ideep::convolution_forward::compute_v2(
        x,
        w,
        b.value(),
        {output_sizes.cbegin(), output_sizes.cend()},
        y,
        {stride.begin(), stride.end()},
        {dilation.begin(), dilation.end()},
        {padding.begin(), padding.end()},
        {padding.begin(), padding.end()},
        groups,
        ideep::scale_t(),
        ideep::scale_t(),
        ideep::scale_t(),
        ideep::zero_point_t(),
        ideep::zero_point_t(),
        attr);
  } else {
    ideep::convolution_forward::compute_v2(
        x,
        w,
        {output_sizes.cbegin(), output_sizes.cend()},
        y,
        {stride.begin(), stride.end()},
        {dilation.begin(), dilation.end()},
        {padding.begin(), padding.end()},
        {padding.begin(), padding.end()},
        groups,
        ideep::scale_t(),
        ideep::scale_t(),
        ideep::scale_t(),
        ideep::zero_point_t(),
        ideep::zero_point_t(),
        attr);
  }
}

void _mkldnn_convolution_binary_out(
    const ideep::tensor& x,
    const ideep::tensor& other,
    ideep::tensor& y,
    const ideep::tensor& w,
    const c10::optional<ideep::tensor>& b,
    IntArrayRef padding,
    IntArrayRef stride,
    IntArrayRef dilation,
    IntArrayRef output_sizes,
    int64_t groups,
    bool is_channels_last,
    ideep::attr_t& attr) {
  if (b.has_value()) {
    ideep::convolution_forward::compute_binary(
        x,
        other,
        w,
        b.value(),
        {output_sizes.cbegin(), output_sizes.cend()},
        y,
        {stride.begin(), stride.end()},
        {dilation.begin(), dilation.end()},
        {padding.begin(), padding.end()},
        {padding.begin(), padding.end()},
        groups,
	      is_channels_last,
        attr);
  } else {
    ideep::convolution_forward::compute_binary(
        x,
        other,
        w,
        {output_sizes.cbegin(), output_sizes.cend()},
        y,
        {stride.begin(), stride.end()},
        {dilation.begin(), dilation.end()},
        {padding.begin(), padding.end()},
        {padding.begin(), padding.end()},
        groups,
        is_channels_last,
        attr);
  }
}

void mkldnn_convolution_out(
    const Tensor& input,
    ideep::tensor& mkldnn_output,
    const ideep::tensor& mkldnn_weight,
    const c10::optional<Tensor>& bias_opt,
    IntArrayRef padding,
    IntArrayRef stride,
    IntArrayRef dilation,
    IntArrayRef output_sizes,
    int64_t groups,
    const ideep::attr_t& attr = ideep::attr_t()) {
  c10::MaybeOwned<Tensor> bias_maybe_owned =
      at::borrow_from_optional_tensor(bias_opt);
  const Tensor& bias = *bias_maybe_owned;

  c10::impl::ExcludeDispatchKeyGuard edkg(c10::autograd_dispatch_keyset);
  const ideep::tensor mkldnn_input = itensor_from_tensor(input);
  c10::optional<ideep::tensor> mkldnn_bias{c10::nullopt};
  if (bias.defined()) {
    mkldnn_bias = itensor_from_tensor(bias);
  }

  _mkldnn_convolution_out(
      mkldnn_input,
      mkldnn_output,
      mkldnn_weight,
      mkldnn_bias,
      padding,
      stride,
      dilation,
      output_sizes,
      groups,
      attr);
}

void mkldnn_convolution_binary_out(
    const Tensor& input,
    const Tensor& other,
    ideep::tensor& mkldnn_output,
    const ideep::tensor& mkldnn_weight,
    const c10::optional<Tensor>& bias_opt,
    IntArrayRef padding,
    IntArrayRef stride,
    IntArrayRef dilation,
    IntArrayRef output_sizes,
    int64_t groups,
    bool is_channels_last,
    ideep::attr_t& attr) {
  c10::MaybeOwned<Tensor> bias_maybe_owned =
      at::borrow_from_optional_tensor(bias_opt);
  const Tensor& bias = *bias_maybe_owned;
  c10::impl::ExcludeDispatchKeyGuard edkg(c10::autograd_dispatch_keyset);
  const ideep::tensor mkldnn_input = itensor_from_tensor(input);
  const ideep::tensor mkldnn_other = itensor_from_tensor(other);
  c10::optional<ideep::tensor> mkldnn_bias{c10::nullopt};
  if (bias.defined()) {
    mkldnn_bias = itensor_from_tensor(bias);
  }

  _mkldnn_convolution_binary_out(
      mkldnn_input,
      mkldnn_other,
      mkldnn_output,
      mkldnn_weight,
      mkldnn_bias,
      padding,
      stride,
      dilation,
      output_sizes,
      groups,
      is_channels_last,
      attr);
}

std::vector<int64_t> get_output_sizes(
    ContextConv& context,
    const Tensor& input) {
  const ideep::tensor& mkldnn_weight = context.weight_packed_;
  IntArrayRef padding = context.padding_;
  IntArrayRef stride = context.stride_;
  IntArrayRef dilation = context.dilation_;

  auto kernel_size = mkldnn_weight.get_dims();

  std::vector<int64_t> input_size = input.sizes().vec();
  return conv_output_size(input_size, kernel_size, padding, stride, dilation);
}

Tensor run(ContextConv& context, const Tensor& input) {
  std::vector<int64_t> output_sizes = get_output_sizes(context, input);
  auto output = at::empty(
      output_sizes,
      input.options().memory_format(input.suggest_memory_format()));

  bool is_channels_last =
      input.suggest_memory_format() == at::MemoryFormat::ChannelsLast;
  ideep::tensor y;

  c10::impl::ExcludeDispatchKeyGuard edkg(c10::autograd_dispatch_keyset);
  ideep::tensor mkldnn_output = itensor_from_tensor(output);

  if (is_channels_last) {
    mkldnn_convolution_out(
        input,
        mkldnn_output,
        context.weight_packed_,
        context.at_bias_,
        context.padding_,
        context.stride_,
        context.dilation_,
        output_sizes,
        context.groups_,
        context.attr_);
  } else {
    mkldnn_convolution_out(
        input,
        y,
        context.weight_packed_,
        context.at_bias_,
        context.padding_,
        context.stride_,
        context.dilation_,
        output_sizes,
        context.groups_,
        context.attr_);
    mkldnn_output.feed_from(y);
  }
  return output;
}

Tensor run(ContextConv& context, const Tensor& input, const Tensor& other) {
  std::vector<int64_t> output_sizes = get_output_sizes(context, input);
  auto output = at::empty(
      output_sizes,
      input.options().memory_format(input.suggest_memory_format()));

  bool is_channels_last =
      input.suggest_memory_format() == at::MemoryFormat::ChannelsLast;

  c10::impl::ExcludeDispatchKeyGuard edkg(c10::autograd_dispatch_keyset);
  ideep::tensor mkldnn_output = itensor_from_tensor(output);

  if (is_channels_last) {
    mkldnn_convolution_binary_out(
        input,
        other,
        mkldnn_output,
        context.weight_packed_,
        context.at_bias_,
        context.padding_,
        context.stride_,
        context.dilation_,
        output_sizes,
        context.groups_,
        is_channels_last,
        context.attr_);
  } else {
    ideep::tensor y;
    mkldnn_convolution_binary_out(
        input,
        other,
        y,
        context.weight_packed_,
        context.at_bias_,
        context.padding_,
        context.stride_,
        context.dilation_,
        output_sizes,
        context.groups_,
        is_channels_last,
        context.attr_);
    mkldnn_output.feed_from(y);
  }
  return output;
}

void run(ContextConv& context, const Tensor& input, void* output) {
  std::vector<int64_t> output_sizes = get_output_sizes(context, input);

  bool is_channels_last =
      input.suggest_memory_format() == at::MemoryFormat::ChannelsLast;
  ideep::tensor y;

  ideep::tag o_tag = is_channels_last ? ideep::tag::nhwc : ideep::tag::nchw;
  ideep::tensor::desc o_desc = {
      output_sizes, get_mkldnn_dtype(input.scalar_type()), o_tag};
  ideep::tensor mkldnn_output = {o_desc, output};

  if (is_channels_last) {
    mkldnn_convolution_out(
        input,
        mkldnn_output,
        context.weight_packed_,
        context.at_bias_,
        context.padding_,
        context.stride_,
        context.dilation_,
        output_sizes,
        context.groups_,
        context.attr_);
  } else {
    mkldnn_convolution_out(
        input,
        y,
        context.weight_packed_,
        context.at_bias_,
        context.padding_,
        context.stride_,
        context.dilation_,
        output_sizes,
        context.groups_,
        context.attr_);
    mkldnn_output.feed_from(y);
  }
}

void run(
    ContextConv& context,
    const Tensor& input,
    const Tensor& other,
    void* output) {
  std::vector<int64_t> output_sizes = get_output_sizes(context, input);

  bool is_channels_last =
      input.suggest_memory_format() == at::MemoryFormat::ChannelsLast;

  ideep::tag o_tag = is_channels_last ? ideep::tag::nhwc : ideep::tag::nchw;
  ideep::tensor::desc o_desc = {
      output_sizes, get_mkldnn_dtype(input.scalar_type()), o_tag};
  ideep::tensor mkldnn_output = {o_desc, output};
  if (is_channels_last) {
    mkldnn_convolution_binary_out(
        input,
        other,
        mkldnn_output,
        context.weight_packed_,
        context.at_bias_,
        context.padding_,
        context.stride_,
        context.dilation_,
        output_sizes,
        context.groups_,
        is_channels_last,
        context.attr_);
  } else {
    ideep::tensor y;
    mkldnn_convolution_binary_out(
        input,
        other,
        y,
        context.weight_packed_,
        context.at_bias_,
        context.padding_,
        context.stride_,
        context.dilation_,
        output_sizes,
        context.groups_,
        is_channels_last, 
        context.attr_);
    mkldnn_output.feed_from(y);
  }
}

Tensor conv_run(
    const Tensor& input,
    const c10::intrusive_ptr<mkldnn::ConvOpContext>& op_context) {
  return op_context->run(input);
}

Tensor conv_binary_run(
    const Tensor& input,
    const Tensor& other,
    const c10::intrusive_ptr<mkldnn::ConvOpContext>& op_context) {
  return op_context->run(input, other);
}

} // namespace convolution
} // namespace internal
} // namespace mkldnn
} // namespace native
} // namespace at

#endif // AT_MKLDNN_ENABLED()
