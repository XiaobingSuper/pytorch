#include <ATen/ATen.h>
#include <ATen/Config.h>
#include <ATen/NativeFunctions.h>
#include <tuple>

#if !AT_MKLDNN_ENABLED()

namespace at {
namespace native {

std::tuple<Tensor, Tensor, Tensor> mkldnn_batch_norm(
    const Tensor& self,
    const Tensor& weight,
    const Tensor& bias,
    const Tensor& running_mean,
    const Tensor& running_var,
    bool train,
    double momentum,
    double eps) {
  AT_ERROR("mkldnn_batch_norm: ATen not compiled with MKLDNN support");
}

std::tuple<Tensor, Tensor, Tensor> mkldnn_batch_norm_backward(
    const Tensor& grad_output,
    const Tensor& input,
    const Tensor& weight,
    const Tensor& running_mean,
    const Tensor& running_var,
    const Tensor& save_mean,
    const Tensor& save_invstd,
    bool train,
    double eps,
    std::array<bool,3> grad_input_mask) {
  AT_ERROR("mkldnn_batch_norm_backward: ATen not compiled with MKLDNN support");
}

} // namespace native
} // namespace at

#else // AT_MKLDNN_EBABLED

#include <ATen/native/mkldnn/MKLDNNCommon.h>

namespace at {
namespace native {

std::tuple<Tensor, Tensor, Tensor> mkldnn_batch_norm(
    const Tensor& input,
    const Tensor& weight,
    const Tensor& bias,
    const Tensor& running_mean,
    const Tensor& running_var,
    bool train,
    double momentum,
    double eps) {
  AT_ASSERTM(input.dim() == 4 || input.dim() == 5,
             "mkldnn_batch_norm: currently mkldnn only support 2d and 3d batchnorm");
  AT_ASSERTM(weight.defined() && bias.defined(),
             "mkldnn_batch_norm: currently mkldnn only support affine model")

  ideep::tensor& x = itensor_from_mkldnn(input);
  const ideep::tensor w = get_mkldnn_tensor(weight);
  const ideep::tensor b = get_mkldnn_tensor(bias);

  bool use_running_stat = (running_mean.defined() && running_var.defined());
  ideep::tensor y;

  if (train) {
    ideep::tensor saved_mean;
    ideep::tensor saved_var;
    if (use_running_stat) {
      ideep::tensor m = itensor_view_from_dense(running_mean);
      ideep::tensor v = itensor_view_from_dense(running_var);
      ideep::batch_normalization_forward_training::compute<ideep::utils::scratch_allocator>(
          x, w, b, y, saved_mean, saved_var, m, v, momentum, eps);
    } else {
      ideep::batch_normalization_forward_training::compute<ideep::utils::scratch_allocator>(
          x, w, b, y, saved_mean, saved_var, momentum, eps);
    }
    return std::make_tuple(
        new_with_itensor_mkldnn(std::move(y), input.options()),
        new_with_itensor_mkldnn(std::move(saved_mean), input.options()),
        new_with_itensor_mkldnn(std::move(saved_var), input.options()));
  } else {
    if (use_running_stat) {
      ideep::tensor m = get_mkldnn_tensor(running_mean);
      ideep::tensor v = get_mkldnn_tensor(running_var);
      ideep::batch_normalization_forward_inference::compute<ideep::utils::scratch_allocator>(
          x, m, v, w, b, y, eps);
    } else {
      ideep::batch_normalization_forward_inference::compute<ideep::utils::scratch_allocator>(
          x, w, b, y, eps);
    }
    return std::make_tuple(
        new_with_itensor_mkldnn(std::move(y), input.options()),
        new_with_itensor_mkldnn(ideep::tensor{}, input.options()),
        new_with_itensor_mkldnn(ideep::tensor{}, input.options()));
  }
}

std::tuple<Tensor, Tensor, Tensor> mkldnn_batch_norm_backward(const Tensor& grad_output,
    const Tensor& input,
    const Tensor& weight,
    const Tensor& running_mean,
    const Tensor& running_var,
    const Tensor& save_mean,
    const Tensor& save_invstd,
    bool train,
    double eps,
    std::array<bool,3> grad_input_mask) {
  AT_ASSERTM(train, "mkldnn_batch_norm_backward: currently mkldnn only support train model");

  ideep::tensor& grady = itensor_from_mkldnn(grad_output);
  ideep::tensor& x = itensor_from_mkldnn(input);
  ideep::tensor w = itensor_view_from_dense(weight);
  ideep::tensor& m = itensor_from_mkldnn(save_mean);
  ideep::tensor& v = itensor_from_mkldnn(save_invstd);

  ideep::tensor gradx, gradw, gradb;
  ideep::batch_normalization_backward::compute<ideep::utils::scratch_allocator>(
      x, m, v, grady, w, gradx, gradw, gradb, eps);

  if (weight.is_mkldnn()) {
    return std::make_tuple(
        new_with_itensor_mkldnn(std::move(gradx), input.options()),
        new_with_itensor_mkldnn(std::move(gradw), input.options()),
        new_with_itensor_mkldnn(std::move(gradb), input.options()));
  } else {
    return std::make_tuple(
        new_with_itensor_mkldnn(std::move(gradx), input.options()),
        mkldnn_to_dense(new_with_itensor_mkldnn(std::move(gradw), input.options())),
        mkldnn_to_dense(new_with_itensor_mkldnn(std::move(gradb), input.options())));
  }
}

} // namespace native
} // namespace at

#endif // AT_MKLDNN_EBABLED
