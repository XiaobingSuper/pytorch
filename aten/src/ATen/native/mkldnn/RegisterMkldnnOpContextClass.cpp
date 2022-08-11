#include <ATen/Tensor.h>
#include <ATen/native/mkldnn/ConvPrepack.h>
#include <ATen/native/mkldnn/OpContext.h>
#include <torch/custom_class.h>
#include <torch/library.h>

#if AT_MKLDNN_ENABLED()

namespace at {
namespace native {
namespace mkldnn {

using namespace internal::convolution;

TORCH_LIBRARY(mkldnn, m) {
  m.class_<ConvOpContext>(TORCH_SELECTIVE_CLASS("ConvOpContext"))
      .def_pickle(
          [](const c10::intrusive_ptr<ConvOpContext>& op_context)
              -> SerializationTypeConvPrePack { // __getstate__
            return op_context->unpack();
          },
          [](SerializationTypeConvPrePack state)
              -> c10::intrusive_ptr<ConvOpContext> { // __setstate__
            return createConvPrePackOpContext(
                std::move(std::get<0>(state)),
                std::move(std::get<1>(state)),
                std::move(std::get<2>(state)),
                std::move(std::get<3>(state)),
                std::move(std::get<4>(state)),
                // NOLINTNEXTLINE(performance-move-const-arg,cppcoreguidelines-avoid-magic-numbers)
                std::move(std::get<5>(state)),
                // NOLINTNEXTLINE(performance-move-const-arg,cppcoreguidelines-avoid-magic-numbers)
                std::move(std::get<6>(state)),
                // NOLINTNEXTLINE(performance-move-const-arg,cppcoreguidelines-avoid-magic-numbers)
                std::move(std::get<7>(state)));
          });
}

TORCH_LIBRARY(mkldnn_prepacked, m) {
  m.def(TORCH_SELECTIVE_SCHEMA(
      "mkldnn_prepacked::conv2d_prepack(Tensor W, Tensor? B, int[2] stride, int[2] padding, int[2] dilation, int groups, int[4] input_size, str attr) -> __torch__.torch.classes.mkldnn.ConvOpContext"));

  m.def(TORCH_SELECTIVE_SCHEMA(
      "mkldnn_prepacked::conv2d_run(Tensor X, __torch__.torch.classes.mkldnn.ConvOpContext W_prepack) -> Tensor Y"));
  m.def(TORCH_SELECTIVE_SCHEMA(
      "mkldnn_prepacked::conv2d_binary_run(Tensor X, Tensor other, __torch__.torch.classes.mkldnn.ConvOpContext W_prepack) -> Tensor Y"));
}

TORCH_LIBRARY_IMPL(mkldnn_prepacked, CPU, m) {
  m.impl(
      TORCH_SELECTIVE_NAME("mkldnn_prepacked::conv2d_prepack"),
      TORCH_FN(createConvPrePackOpContext));

  m.impl(
      TORCH_SELECTIVE_NAME("mkldnn_prepacked::conv2d_run"), TORCH_FN(conv_run));

  m.impl(
      TORCH_SELECTIVE_NAME("mkldnn_prepacked::conv2d_binary_run"),
      TORCH_FN(conv_binary_run));
}

} // namespace mkldnn
} // namespace native
} // namespace at

#endif // AT_MKLDNN_ENABLED()
