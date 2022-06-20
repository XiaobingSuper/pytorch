#include <ATen/Config.h>
#include <ATen/code_template.h>
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/jit_log.h>
#include <torch/csrc/jit/passes/constant_propagation.h>
#include <torch/csrc/jit/passes/dead_code_elimination.h>
#include <torch/csrc/jit/passes/graph_rewrite_helper.h>
#include <torch/csrc/jit/passes/mkldnn_rewrite.h>
#include <torch/csrc/jit/tensorexpr/kernel.h>

namespace torch {
namespace jit {

#if AT_MKLDNN_ENABLED()

c10::VaryingShape<int64_t> getSizesOf(Node* n, size_t idx) {
  auto tt = n->input(idx)->type()->cast<TensorType>();
  return tt->sizes();
}

void insertPrePackedConvOpForNode(Node* n, bool use_channels_last) {
  constexpr int POS_INPUT = 0;
  constexpr int POS_WEIGHT = 1;
  if (!tensorexpr::isContiguous(
          n->input(POS_INPUT), at::MemoryFormat::ChannelsLast)) {
    GRAPH_DEBUG(
        "insertPrePackedConvOpForNode: input is not ChannelsLast contiguous");
    return;
  }

  if (!tensorexpr::isContiguous(
          n->input(POS_WEIGHT), at::MemoryFormat::ChannelsLast)) {
    GRAPH_DEBUG(
        "insertPrePackedConvOpForNode: weight is not ChannelsLast contiguous");
    return;
  }

  // Leave depthwise conv2d to NNC
  if (tensorexpr::conv2dIsSupportedJit(n)) {
    GRAPH_DEBUG("insertPrePackedConvOpForNode: leave depthwise conv2d to NNC");
    return;
  }

  WithInsertPoint guard(n);
  auto graph = n->owningGraph();

  auto input_sizes = getSizesOf(n, POS_INPUT);
  IValue input_size_value(*input_sizes.concrete_sizes());
  auto input_size = graph->insertConstant(input_size_value);

  auto prepack_node = graph->create(
      Symbol::fromQualString("mkldnn_prepacked::conv2d_prepack"), 1);

  // skip input value
  for (auto i = 1; i < n->inputs().size(); i++) {
    Value* v = n->input(i);
    prepack_node->addInput(v);
  }
  prepack_node->addInput(input_size);
  auto use_channels_last_ivalue = graph->insertConstant(IValue(use_channels_last));
  prepack_node->addInput(use_channels_last_ivalue);
  auto attr = graph->insertConstant(IValue("none"));
  prepack_node->addInput(attr);
  prepack_node->output()->setType(
      getCustomClass("__torch__.torch.classes.mkldnn.ConvOpContext"));
  graph->insertNode(prepack_node);

  auto prepack_conv = graph->insertNode(
      graph->create(Symbol::fromQualString("mkldnn_prepacked::conv2d_run"), 1));
  prepack_conv->addInput(n->input(0));
  prepack_conv->addInput(prepack_node->output());
  prepack_conv->output()->setType(n->output()->type()->cast<TensorType>());

  n->output()->replaceAllUsesWith(prepack_conv->output());
}

bool isTensorTypeCPU(Node* node) {
  for (const auto& input : node->inputs()) {
    auto type = input->type()->cast<TensorType>();
    if (!type) {
      continue;
    }
    auto device = type->device();
    if (!device) {
      return false;
    }
    if (!device->is_cpu()) {
      return false;
    }
  }
  return true;
}

void insertPrePackedConvOp(Block* b, bool use_channels_last) {
  for (Node* n : b->nodes()) {
    for (Block* b : n->blocks()) {
      insertPrePackedConvOp(b, use_channels_last);
    }

    if (n->kind() == aten::conv2d) {
      if (isTensorTypeCPU(n)) {
        insertPrePackedConvOpForNode(n, use_channels_last);
      }
    }
  }
  EliminateDeadCode(b);
}

void insertMkldnnPrePackedConv2dOp(std::shared_ptr<Graph>& graph, bool use_channels_last) {
  // Replace _convolution with conv2d
  graph_rewrite_helper::replaceConvolutionWithAtenConv(graph);

  insertPrePackedConvOp(graph->block(), use_channels_last);
}

void insertMkldnnPrePackedOps(std::shared_ptr<Graph>& graph, bool use_channels_last) {
  insertMkldnnPrePackedConv2dOp(graph, use_channels_last);
}

void insertMkldnnPrePackedOps(script::Module& module, bool use_channels_last) {
  for (auto& method : module.get_methods()) {
    auto graph = method.graph();
    insertMkldnnPrePackedOps(graph, use_channels_last);
  }
  for (script::Module m : module.children()) {
    insertMkldnnPrePackedOps(m, use_channels_last);
  }
}

void FuseReluWithPackedOps(std::shared_ptr<Graph>& graph) {
  auto conv_op_rstring = at::jit::CodeTemplate(R"(
    graph(%input, %weight, %bias, %stride:int[], %padding:int[],
          %dilation:int[], %groups:int, %input_size:int[], %use_channels_last:bool, %dummy_attr:str):
        %packed_weight_bias = mkldnn_prepacked::conv2d_prepack(
            %weight, %bias, %stride, %padding, %dilation, %groups,
            %input_size, %use_channels_last, %dummy_attr)
        %conv2d_res = mkldnn_prepacked::conv2d_run(%input, %packed_weight_bias)
        %res = aten::${op}(%conv2d_res)
        return (%res))");

  auto conv_op_fused_rstring = at::jit::CodeTemplate(R"(
    graph(%input, %weight, %bias, %stride:int[], %padding:int[],
          %dilation:int[], %groups:int, %input_size:int[], %use_channels_last:bool, %dummy_attr:str):
        %attr: str = prim::Constant[value="${op_attr}"]()
        %packed_weight_bias : __torch__.torch.classes.mkldnn.ConvOpContext = mkldnn_prepacked::conv2d_prepack(
            %weight, %bias, %stride, %padding, %dilation, %groups,
            %input_size, %use_channels_last, %attr)
        %res = mkldnn_prepacked::conv2d_run(%input, %packed_weight_bias)
        return (%res))");

  for (auto const& it : mkldnn::fusion_rewrite_map) {
    std::string op = it.first;
    if (op == std::string("none")) {
      continue;
    }

    at::jit::TemplateEnv env;
    env.s("op", op);

    at::jit::TemplateEnv env_fused;
    env_fused.s("op_attr", op);

    SubgraphRewriter rewriter;
    rewriter.RegisterRewritePattern(
        conv_op_rstring.format(env), conv_op_fused_rstring.format(env_fused));

    auto filters = it.second;
    rewriter.runOnGraph(graph, filters);
  }
}

void FuseBinaryWithPackedOps(std::shared_ptr<Graph>& graph) {
  auto conv_op_rstring = at::jit::CodeTemplate(R"(
    graph(%input, %weight, %bias, %other, %alpha, %stride:int[], %padding:int[],
          %dilation:int[], %groups:int, %input_size:int[], %use_channels_last:bool, %dummy_attr:str):
        %packed_weight_bias = mkldnn_prepacked::conv2d_prepack(
            %weight, %bias, %stride, %padding, %dilation, %groups,
            %input_size, %use_channels_last, %dummy_attr)
        %conv2d_res = mkldnn_prepacked::conv2d_run(%input, %packed_weight_bias)
        %res = aten::${op}(%conv2d_res, %other, %alpha)
        return (%res))");

  auto conv_op_fused_rstring = at::jit::CodeTemplate(R"(
    graph(%input, %weight, %bias, %other, %alpha, %stride:int[], %padding:int[],
          %dilation:int[], %groups:int, %input_size:int[], %use_channels_last:bool, %dummy_attr:str):
        %attr: str = prim::Constant[value="${op_attr}"]()
        %packed_weight_bias : __torch__.torch.classes.mkldnn.ConvOpContext = mkldnn_prepacked::conv2d_prepack(
            %weight, %bias, %stride, %padding, %dilation, %groups,
            %input_size, %use_channels_last, %attr)
        %res = mkldnn_prepacked::conv2d_binary_run(%input, %other, %packed_weight_bias)
        return (%res))");

  for (auto const& it : mkldnn::fusion_binary_attr_map) {
    std::string op = it.first;
    at::jit::TemplateEnv env;
    env.s("op", op);

    at::jit::TemplateEnv env_fused;
    env_fused.s("op_attr", op);

    SubgraphRewriter rewriter;
    rewriter.RegisterRewritePattern(
        conv_op_rstring.format(env), conv_op_fused_rstring.format(env_fused));

    auto filter = [](const Match& match,
                     const std::unordered_map<std::string, Value*>& vmap) {
      auto conv_res = toIValue(match.values_map.at(vmap.at("conv2d_res")));
      auto other = toIValue(match.values_map.at(vmap.at("other")));
      return true;
      if (!conv_res.has_value() || !conv_res.value().isTensor()) {
        return false;
      }
      const at::Tensor& conv_res_value = conv_res.value().toTensor();
      if (other.has_value() && other.value().isTensor()) {
        const at::Tensor& other_value = other.value().toTensor();
        // TODO: support broadcast.
        if (other_value.sizes() != conv_res_value.sizes() ||
            other_value.dtype() != conv_res_value.dtype() ||
            !other_value.is_contiguous() ||
            other_value.suggest_memory_format() !=
                conv_res_value.suggest_memory_format() ||
            other_value.device() != conv_res_value.device()) {
          return false;
        }
      } else {
        return false;
      }
      // alpha is optional
      if (vmap.find("alpha") != vmap.end()) {
        auto alpha = toIValue(match.values_map.at(vmap.at("alpha")));
        if (alpha.has_value() && alpha.value().isDouble()) {
          auto alpha_ = alpha.value().toDouble();
          if (alpha_ != 1.0) {
            return false;
          }
        } else {
          return false;
        }
      }
      return true;
    };
    rewriter.runOnGraph(graph, filter);
  }
}

void PrePackingOpsFolder(Block* b) {
  auto is_foldable_op = [](const Node* n) -> bool {
    return (
        n->kind() ==
        Symbol::fromQualString("mkldnn_prepacked::conv2d_prepack"));
  };

  std::unordered_set<Node*> nodes_to_delete;
  for (Node* n : b->nodes()) {
    for (Block* block : n->blocks()) {
      PrePackingOpsFolder(block);
    }
    if (is_foldable_op(n)) {
      auto optional_outputs = torch::jit::runNodeIfInputsAreConstant(n);
      if (optional_outputs) {
        auto outputs = optional_outputs.value();
        TORCH_CHECK(outputs.size() == 1, "Prepack ops have single output");
        Value* prepack_op_value = n->output(0);
        auto graph = n->owningGraph();
        WithInsertPoint ins(prepack_op_value->node());
        auto weak_class_obj =
            outputs[0].toObject()->copy_to_weak_compilation_ref();
        Value* packed_weight = graph->insertConstant(weak_class_obj)
                                   ->setType(n->output(0)->type());
        prepack_op_value->replaceAllUsesWith(packed_weight);
        nodes_to_delete.insert(n);
      }
    }
  }
  for (auto n : nodes_to_delete) {
    n->removeAllInputs();
  }
  for (auto n : nodes_to_delete) {
    n->destroy();
  }
}

void FoldPrePackingOps(std::shared_ptr<Graph>& graph) {
  PrePackingOpsFolder(graph->block());
}

void FuseConvWithBinaryOrEltwise(std::shared_ptr<Graph>& graph, bool use_channels_last) {
  GRAPH_DEBUG(
      "Before insertMkldnnPrePackedOps. Beginning of FuseConvWithBinaryOrEltwise\n",
      *graph);
  insertMkldnnPrePackedOps(graph, use_channels_last);
  GRAPH_DEBUG(
      "After insertMkldnnPrePackedOps, before FuseReluWithPackedOps\n", *graph);
  FuseReluWithPackedOps(graph);
  GRAPH_DEBUG(
      "After FuseReluWithPackedOps, before FuseBinaryWithPackedOps\n", *graph);
  FuseBinaryWithPackedOps(graph);
  GRAPH_DEBUG(
      "After FuseBinaryWithPackedOps, before FoldPrePackingOps\n", *graph);
  FoldPrePackingOps(graph);
  GRAPH_DEBUG("After FoldPrePackingOps. End of FuseConvWithEltwise\n", *graph);
}

#else

void FuseConvWithBinaryOrEltwise(std::shared_ptr<Graph>& graph) {
  GRAPH_DEBUG("MKLDNN Not enabled");
}

#endif // AT_MKLDNN_ENABLED()

} // namespace jit
} // namespace torch
