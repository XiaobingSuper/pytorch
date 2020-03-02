#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <ATen/Config.h>

#if !AT_MKLDNN_ENABLED()

namespace at { namespace native {

at::Tensor mkldnn_convolution(
    const Tensor& input, const Tensor& weight, const Tensor& bias,
    IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups) {
  AT_ERROR("mkldnn_convolution_forward: ATen not compiled with MKLDNN support");
}

at::Tensor mkldnn_convolution_backward_input(
    IntArrayRef input_size, const Tensor& grad_output, const Tensor& weight,
    IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups, bool bias_defined) {
  AT_ERROR("mkldnn_convolution_backward_input: ATen not compiled with MKLDNN support");
}

std::tuple<at::Tensor,at::Tensor> mkldnn_convolution_backward_weights(
    const Tensor& weight, const Tensor& grad_output, const Tensor& input,
    IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups, bool bias_defined) {
  AT_ERROR("mkldnn_convolution_backward_weights: ATen not compiled with MKLDNN support");
}

std::tuple<at::Tensor,at::Tensor,at::Tensor> mkldnn_convolution_backward(
    const Tensor& input, const Tensor& grad_output_t, const Tensor& weight,
    IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups, std::array<bool,3> output_mask) {
  AT_ERROR("mkldnn_convolution_backward: ATen not compiled with MKLDNN support");
}

}}

#else // AT_MKLDNN_EBABLED

#include <ATen/native/mkldnn/MKLDNNCommon.h>
#include <ATen/native/mkldnn/Utils.h>
#include <ATen/native/ConvUtils.h>

namespace at { namespace native {

ideep::tensor _mkldnn_conv2d(
    const ideep::tensor& x,
    const ideep::tensor& w,
    const c10::optional<ideep::tensor>& b,
    at::IntArrayRef padding,
    at::IntArrayRef stride,
    at::IntArrayRef dilation,
    int64_t groups) {

  auto kernel_size = w.get_dims();

  std::vector<int64_t> input_size = x.get_dims();
  std::vector<int64_t> output_sizes =
      conv_output_size(input_size, kernel_size, padding, stride, dilation);

  ideep::tensor y;
  if (b.has_value()) {
    ideep::convolution_forward::compute(
        x,
        w,
        b.value(),
        {output_sizes.cbegin(), output_sizes.cend()},
        y,
        {stride.begin(), stride.end()},
        {dilation.begin(), dilation.end()},
        {padding.begin(), padding.end()},
        {padding.begin(), padding.end()},
        groups);
  } else {
    ideep::convolution_forward::compute(
        x,
        w,
        {output_sizes.cbegin(), output_sizes.cend()},
        y,
        {stride.begin(), stride.end()},
        {dilation.begin(), dilation.end()},
        {padding.begin(), padding.end()},
        {padding.begin(), padding.end()},
        groups);
  }
  return y;
}

ideep::tensor _mkldnn_conv2d_backward_input(
    IntArrayRef input_sizes,
    const ideep::tensor& grady,
    const ideep::tensor& w,
    IntArrayRef padding,
    IntArrayRef stride,
    IntArrayRef dilation,
    int64_t groups) {

  ideep::tensor gradx;
  ideep::convolution_backward_data::compute<AllocForMKLDNN>(
      grady,
      w,
      {input_sizes.cbegin(), input_sizes.cend()},
      gradx,
      {stride.begin(), stride.end()},
      {dilation.begin(), dilation.end()},
      {padding.begin(), padding.end()},
      {padding.begin(), padding.end()},
      groups,
      ideep::algorithm::convolution_direct);

  return gradx;
}

std::tuple<ideep::tensor, ideep::tensor> _mkldnn_conv2d_backward_weights(
    IntArrayRef weight_sizes,
    const ideep::tensor& grady,
    const ideep::tensor& x,
    IntArrayRef padding,
    IntArrayRef stride,
    IntArrayRef dilation,
    int64_t groups,
    bool bias_defined) {

  ideep::tensor gradw, gradb;
  if (bias_defined) {
    ideep::convolution_backward_weights::compute<AllocForMKLDNN>(
        x,
        grady,
        {weight_sizes.cbegin(), weight_sizes.cend()},
        gradw,
        gradb,
        {stride.begin(), stride.end()},
        {dilation.begin(), dilation.end()},
        {padding.begin(), padding.end()},
        {padding.begin(), padding.end()},
        groups,
        ideep::algorithm::convolution_direct);
  } else {
    ideep::convolution_backward_weights::compute<AllocForMKLDNN>(
        x,
        grady,
        {weight_sizes.cbegin(), weight_sizes.cend()},
        gradw,
        {stride.begin(), stride.end()},
        {dilation.begin(), dilation.end()},
        {padding.begin(), padding.end()},
        {padding.begin(), padding.end()},
        groups,
        ideep::algorithm::convolution_direct);
  }

  return std::tuple<ideep::tensor, ideep::tensor>{gradw, gradb};
}

Tensor mkldnn_convolution(
    const Tensor& input,
    const Tensor& weight,
    const Tensor& bias,
    IntArrayRef padding,
    IntArrayRef stride,
    IntArrayRef dilation,
    int64_t groups) {

  const ideep::tensor mkldnn_input = itensor_from_tensor(input);
  const ideep::tensor mkldnn_weight = itensor_from_tensor(weight);
  c10::optional<ideep::tensor> mkldnn_bias{c10::nullopt};
  if (bias.defined()) {
    mkldnn_bias = itensor_from_tensor(bias);
  }

  ideep::tensor mkldnn_output = _mkldnn_conv2d(
      mkldnn_input,
      mkldnn_weight,
      mkldnn_bias,
      padding,
      stride,
      dilation,
      groups);

  if (input.is_mkldnn()) {
    return new_with_itensor_mkldnn(std::move(mkldnn_output), input.options());
  } else {
    return mkldnn_to_dense(
        new_with_itensor_mkldnn(std::move(mkldnn_output), input.options()));
  }
}

Tensor mkldnn_convolution_backward_input(
    IntArrayRef input_size, const Tensor& grad_output, const Tensor& weight,
    IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups, bool bias_defined) {

  const ideep::tensor mkldnn_grad_output = itensor_from_tensor(grad_output);
  const ideep::tensor mkldnn_weight = itensor_from_tensor(weight);

  ideep::tensor mkldnn_grad_input = _mkldnn_conv2d_backward_input(
      input_size,
      mkldnn_grad_output,
      mkldnn_weight,
      padding,
      stride,
      dilation,
      groups);

  if (grad_output.is_mkldnn()) {
    return new_with_itensor_mkldnn(std::move(mkldnn_grad_input), grad_output.options());
  } else {
    return mkldnn_to_dense(
        new_with_itensor_mkldnn(std::move(mkldnn_grad_input), grad_output.options()));
  }
}

std::tuple<Tensor,Tensor> mkldnn_convolution_backward_weights(
    const Tensor& weight, const Tensor& grad_output, const Tensor& input, IntArrayRef padding,
    IntArrayRef stride, IntArrayRef dilation, int64_t groups, bool bias_defined) {

  const ideep::tensor mkldnn_grad_output = itensor_from_tensor(grad_output);
  const ideep::tensor mkldnn_input = itensor_from_tensor(input);

  ideep::tensor mkldnn_grad_weight, mkldnn_grad_bias;
  std::tie(mkldnn_grad_weight, mkldnn_grad_bias) =_mkldnn_conv2d_backward_weights(
      weight.sizes(),
      mkldnn_grad_output,
      mkldnn_input,
      padding,
      stride,
      dilation,
      groups,
      bias_defined);

  ideep::tensor grad_weight, grad_bias;
  if (mkldnn_grad_weight.get_data_type() != get_mkldnn_dtype(weight.scalar_type())) {
    grad_weight.init<AllocForMKLDNN>({
        mkldnn_grad_weight.get_dims(),
        get_mkldnn_dtype(weight.scalar_type()),
        mkldnn_grad_weight.get_internal_format()});
    grad_weight.feed_from(mkldnn_grad_weight);
  } else {
    grad_weight = mkldnn_grad_weight;
  }

  if (bias_defined) {
    if (mkldnn_grad_bias.get_data_type() != get_mkldnn_dtype(weight.scalar_type())) {
      grad_bias.init<AllocForMKLDNN>(
          {mkldnn_grad_bias.get_dims(), get_mkldnn_dtype(weight.scalar_type())});
      grad_bias.feed_from(mkldnn_grad_bias);
    } else {
      grad_bias = mkldnn_grad_bias;
    }
  }

  if (weight.is_mkldnn()) {
    return std::tuple<Tensor, Tensor>{
        new_with_itensor_mkldnn(std::move(grad_weight), weight.options()),
        new_with_itensor_mkldnn(std::move(grad_bias), weight.options())};
  } else {
    return std::tuple<Tensor, Tensor>{
        mkldnn_to_dense(new_with_itensor_mkldnn(std::move(grad_weight), weight.options())),
        mkldnn_to_dense(new_with_itensor_mkldnn(std::move(grad_bias), weight.options()))};
  }
}

std::tuple<Tensor,Tensor,Tensor> mkldnn_convolution_backward(
    const Tensor& input, const Tensor& grad_output_t, const Tensor& weight, IntArrayRef padding,
    IntArrayRef stride, IntArrayRef dilation, int64_t groups, std::array<bool,3> output_mask) {

  Tensor grad_output = grad_output_t.is_mkldnn() ? grad_output_t : grad_output_t.contiguous();

  Tensor grad_input, grad_weight, grad_bias;
  if (output_mask[0]) {
    grad_input = at::mkldnn_convolution_backward_input(
        input.sizes(), grad_output, weight, padding, stride, dilation, groups, output_mask[2]);
  }
  if (output_mask[1] || output_mask[2]) {
    std::tie(grad_weight, grad_bias) = at::mkldnn_convolution_backward_weights(
        weight, grad_output, input, padding, stride, dilation, groups, output_mask[2]);
  }

  return std::make_tuple(grad_input, grad_weight, grad_bias);
}

}}  // namespace at::native

#endif
