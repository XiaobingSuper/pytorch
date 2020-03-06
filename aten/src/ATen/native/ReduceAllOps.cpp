#include <ATen/native/ReduceAllOps.h>

#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>

namespace at {
namespace native {

DEFINE_DISPATCH(min_all_stub);
DEFINE_DISPATCH(max_all_stub);
DEFINE_DISPATCH(sum_all_stub);
DEFINE_DISPATCH(prod_all_stub);

Tensor min(const Tensor &self) {
  TORCH_CHECK(self.numel() > 0, "operation does not have an identity.");
  Tensor result = at::empty({}, self.options());
  result.fill_(0);
  min_all_stub(kCPU, result, self.contiguous());
  return result;
}

Tensor max(const Tensor &self) {
  TORCH_CHECK(self.numel() > 0, "operation does not have an identity.");
  Tensor result = at::empty({}, self.options());
  result.fill_(0);
  max_all_stub(kCPU, result, self.contiguous());
  return result;
}

// just for test, result and input have same data type

Tensor sum_cpu(const Tensor &self, c10::optional<ScalarType> dtype) {
  Tensor result = at::empty({}, self.options());
  result.fill_(0);
  sum_all_stub(kCPU, result, self.contiguous());
  return result;
}

Tensor prod_cpu(const Tensor &self, c10::optional<ScalarType> dtype) {
  Tensor result = at::empty({}, self.options());
  result.fill_(0);
  prod_all_stub(kCPU, result, self.contiguous());
  return result;
}

}} // namesapce at::native
