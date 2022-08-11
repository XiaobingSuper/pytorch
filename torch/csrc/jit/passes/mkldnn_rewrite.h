#pragma once

#include <ATen/Config.h>
#include <torch/csrc/jit/api/module.h>
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/passes/subgraph_rewrite.h>
#include <torch/csrc/jit/ir/subgraph_matcher.h>

#if AT_MKLDNN_ENABLED()

#include <ideep/tensor.hpp>

#endif // AT_MKLDNN_ENABLED()

namespace torch {
namespace jit {

#if AT_MKLDNN_ENABLED()

namespace mkldnn {

auto binary_filter =
    [](const torch::jit::Match& match,
       const std::unordered_map<std::string, torch::jit::Value*>& vmap) {
  auto binary_node = match.values_map.at(vmap.at("res"))->node();
  auto conv_res = binary_node->inputs().at(0);
  auto other = binary_node->inputs().at(1);
  if (!conv_res->type()->cast<TensorType>()) {
      return false;
  }
  if (other->type()->cast<TensorType>()) {
    auto conv_res_size_option = conv_res->type()
                                        ->cast<TensorType>()
                                        ->sizes()
                                        .concrete_sizes();

    auto other_size_option = other->type()
                                    ->cast<TensorType>()
                                    ->sizes()
                                    .concrete_sizes();
    // TODO: support broadcast.
    if (!conv_res_size_option.has_value() || !other_size_option.has_value()) {
    return false;
    }

    auto conv_res_size_value = conv_res_size_option.value();
    auto other_size_value = other_size_option.value();

    auto conv_res_stride_option = conv_res->type()
                                            ->cast<TensorType>()
                                            ->strides()
                                            .concrete_sizes();

    auto other_stride_option = other->type()
                                    ->cast<TensorType>()
                                    ->strides()
                                    .concrete_sizes();
    if (!conv_res_stride_option.has_value() || !other_stride_option.has_value()) {
    return false;
    }

    auto conv_res_stride_value = conv_res_stride_option.value();
    auto other_stride_value = other_stride_option.value();

    auto conv_res_dtype_option = conv_res->type()->cast<TensorType>()->scalarType();
    auto other_dtype_option = other->type()->cast<TensorType>()->scalarType();
    if (!conv_res_dtype_option || !other_dtype_option) {
    return false;
    }
    auto conv_res_device_option = conv_res->type()->cast<TensorType>()->device();
    auto other_device_option = other->type()->cast<TensorType>()->device();
    if (!conv_res_device_option || !other_device_option) {
    return false;
    }
    if (conv_res_size_value.empty() || other_size_value.empty() ||
        conv_res_size_value != other_size_value ||
        conv_res_stride_value.empty() || other_stride_value.empty() ||
        conv_res_stride_value != other_stride_value ||
        conv_res_dtype_option.value() != other_dtype_option.value() ||
        conv_res_device_option.value() != other_device_option.value())  {
      return false;
    }
  } else {
    return false;
  }

  // alpha is optional
  if (vmap.find("alpha") != vmap.end()) {
    auto alpha = toIValue(match.values_map.at(vmap.at("alpha")));
    if (alpha.has_value() && (alpha.value().isDouble() || alpha.value().isInt())) {
      if (!(alpha.value().isDouble() && alpha.value().toDouble() == 1.0) &&
          !(alpha.value().isInt() && static_cast<int>(alpha.value().toInt()) == 1)) {
        return false;
      }
    } else {
      return false;
    }
  }
  return true;
};

const static std::map<std::string, std::vector<torch::jit::MatchFilter>>
    fusion_rewrite_map = {
        {"none", {}},
        {"relu", {}},
};

const static std::map<std::string, std::vector<torch::jit::MatchFilter>>
    fusion_binary_attr_map = {
        {"add", {binary_filter}},
        {"sub", {binary_filter}},
};

} // namespace mkldnn

#endif // AT_MKLDNN_ENABLED()

void FuseConvWithBinaryOrEltwise(std::shared_ptr<Graph>& graph);

} // namespace jit
} // namespace torch
