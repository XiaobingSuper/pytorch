#include <torch/csrc/jit/tensorexpr/ir_simplifier.h>
#include <torch/csrc/jit/tensorexpr/operators/convertdtype.h>

namespace torch {
namespace jit {
namespace tensorexpr {

Tensor computeAutocastToReducedPrecision(
    const std::vector<ArgValue>& inputs,
    const std::vector<ExprHandle>& outputShape,
    const std::vector<ExprHandle>& outputStrides,
    const c10::optional<ScalarType>& outputType,
    at::Device device) {
  Dtype dtype = kFloat;
  if (outputType) {
    dtype = Dtype(*outputType);
  }
  BufHandle ResultBuf("AutocastToReducedPrecision", outputShape, dtype);
  const BufHandle InputBuf = c10::get<BufHandle>(inputs[0]);
  const auto cuda_enabled = c10::get<bool>(inputs[1]);
  const auto cpu_enabled = c10::get<bool>(inputs[2]);
  //const auto cuda_dtype = c10::get<ScalarType>(inputs[3]);
  //const auto cpu_dtype = c10::get<ScalarType>(inputs[4]);
  const auto cuda_dtype = (int64_t)at::ScalarType::BFloat16;
  const auto cpu_dtype = (int64_t)at::ScalarType::BFloat16;
  std::cout<<"11111111111111111"<<std::endl;
  return Tensor(
      ResultBuf.node(),
      ExternalCall::make(ResultBuf, "nnc_aten_autocast_to_reduced_precision", {InputBuf}, {cuda_enabled, cpu_enabled, cuda_dtype, cpu_dtype}));

}

} // namespace tensorexpr
} // namespace jit
} // namespace torch
