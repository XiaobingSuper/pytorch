# mypy: allow-untyped-defs
from __future__ import annotations

import dataclasses
import hashlib
from typing import Any

import torch
from torch._inductor.codegen.flydsl.flydsl_op_overrides import (
    FlyDSLCSEVariable,
    FlyDSLOpOverrides,
    upcast_compute_type,
)
from torch._inductor.ir import ComputedBuffer, Pointwise
from torch._inductor.scheduler import BaseSchedulerNode
from torch._inductor.virtualized import V
from torch.utils._sympy.value_ranges import ValueRanges


def _flydsl_op_name(target: Any) -> str | None:
    if isinstance(target, torch._ops.OpOverload):
        op_name = target.overloadpacket.__name__
    elif isinstance(target, str):
        op_name = target
    else:
        op_name = target.__name__ if callable(target) else None
    return "truediv" if op_name == "div" else op_name


class FlyDSLGemmEpilogueBody:
    def __init__(self) -> None:
        self.lines: list[str] = []

    def writeline(self, line: str) -> None:
        self.lines.append(line)


class FlyDSLGemmEpilogueCSE:
    def __init__(self) -> None:
        self.index = 0

    def generate(self, body, expr, *, bounds=None, dtype=None, shape=None):
        name = f"tmp{self.index}"
        self.index += 1
        body.writeline(f"{name} = {expr}")
        return FlyDSLCSEVariable(
            name,
            ValueRanges.unknown() if bounds is None else bounds,
            dtype=dtype,
            shape=shape,
        )


class FlyDSLGemmEpilogueKernel:
    def __init__(self) -> None:
        self.body = FlyDSLGemmEpilogueBody()
        self.cse = FlyDSLGemmEpilogueCSE()


class FlyDSLGemmEpilogueOpOverrides(FlyDSLOpOverrides):
    # Aten add/sub carry alpha as schema sugar; FlyDSL only needs the scaled RHS.
    @staticmethod
    def add(a: Any, b: Any, *, alpha: Any = 1) -> Any:
        rhs = b if alpha == 1 else FlyDSLOpOverrides.mul(b, alpha)
        return FlyDSLOpOverrides.add(a, rhs)

    @staticmethod
    def sub(a: Any, b: Any, *, alpha: Any = 1) -> Any:
        rhs = b if alpha == 1 else FlyDSLOpOverrides.mul(b, alpha)
        return FlyDSLOpOverrides.sub(a, rhs)


class FlyDSLSchedulerEpilogueOpOverrides(FlyDSLGemmEpilogueOpOverrides):
    def __init__(self, env: dict[str, Any]) -> None:
        self.env = env

    def load(self, name: str, index: Any) -> Any:
        if name not in self.env:
            raise NotImplementedError(
                f"FlyDSL hgemm epilogue only supports accumulator-only reads; "
                f"unexpected read from {name!r}"
            )
        return self.env[name]


@dataclasses.dataclass(frozen=True)
class FlyDSLGemmOutputPlan:
    """Classify the GEMM epilogue output into a main result and aux returns."""

    output: torch.fx.Node
    aux_outputs: tuple[torch.fx.Node, ...] = ()


def output_plan(graph_module: torch.fx.GraphModule) -> FlyDSLGemmOutputPlan:
    output_nodes = [node for node in graph_module.graph.nodes if node.op == "output"]
    if len(output_nodes) != 1:
        raise NotImplementedError("FlyDSL GEMM epilogue expects one output node")

    output_value = output_nodes[0].args[0]
    if isinstance(output_value, (tuple, list)):
        if len(output_value) == 1:
            output_value = output_value[0]
        else:
            output, *aux_outputs = output_value
            return FlyDSLGemmOutputPlan(output, tuple(aux_outputs))

    if not isinstance(output_value, torch.fx.Node):
        raise NotImplementedError("FlyDSL GEMM epilogue expects a tensor output")
    return FlyDSLGemmOutputPlan(output_value)


def gemm_node(graph_module: torch.fx.GraphModule, gemm_op: Any) -> torch.fx.Node:
    expected_name = _flydsl_op_name(gemm_op)
    gemm_nodes = [
        node
        for node in graph_module.graph.nodes
        if node.op == "call_function" and _flydsl_op_name(node.target) == expected_name
    ]
    if len(gemm_nodes) != 1:
        raise NotImplementedError("FlyDSL GEMM epilogue expects one GEMM node")
    return gemm_nodes[0]


def _flydsl_arg(value: Any, env: dict[torch.fx.Node, Any]) -> Any:
    if isinstance(value, torch.fx.Node):
        if value in env:
            return env[value]
        raise NotImplementedError(
            f"unsupported FlyDSL GEMM epilogue dependency: {value.format_node()}"
        )
    if isinstance(
        value,
        (
            int,
            float,
            bool,
            torch.dtype,
            torch.device,
            torch.layout,
            torch.memory_format,
        ),
    ):
        return value
    if isinstance(value, (tuple, list)):
        return type(value)(_flydsl_arg(item, env) for item in value)
    raise NotImplementedError(f"unsupported FlyDSL GEMM epilogue constant: {value!r}")


def _flydsl_call(target: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
    op_name = _flydsl_op_name(target)
    if op_name in {"sum", "mean", "amax", "amin", "prod"}:
        raise NotImplementedError(
            f"unsupported FlyDSL GEMM epilogue reduction: {target}"
        )
    if op_name is None:
        raise NotImplementedError(f"unsupported FlyDSL GEMM epilogue op: {target}")
    try:
        op = getattr(V.get_ops_handler(), op_name)
    except AttributeError:
        raise NotImplementedError(
            f"unsupported FlyDSL GEMM epilogue op: {target}"
        ) from None
    return op(*args, **kwargs)


def _flydsl_method_call(target: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
    if target == "to":
        if "dtype" in kwargs:
            dtype = kwargs["dtype"]
        elif args and isinstance(args[-1], torch.dtype):
            *args, dtype = args
        else:
            raise NotImplementedError("FlyDSL GEMM epilogue to() requires dtype")
        if len(args) != 1:
            raise NotImplementedError("unsupported FlyDSL GEMM epilogue to() form")
        return FlyDSLGemmEpilogueOpOverrides.to_dtype(
            args[0], dtype, use_compute_types=False
        )
    raise NotImplementedError(f"unsupported FlyDSL GEMM epilogue method: {target}")


def materialize_flydsl_scheduler_epilogue(
    original_buffer_name: str,
    epilogue_nodes: list[BaseSchedulerNode],
) -> tuple[str, str]:
    """Build a FlyDSL epilogue callable from fused Inductor scheduler nodes.

    This initial hgemm integration supports epilogues that read only the GEMM
    accumulator.  Extra epilogue tensor reads require extending the hgemm launch
    ABI with aux tensors.
    """
    if not epilogue_nodes:
        return "", (
            "HAS_EPILOGUE: fx.Constexpr = False\n"
            "EPILOGUE_FN = None\n"
            "EPILOGUE_ARG_KINDS: fx.Constexpr = ()\n"
            "EPILOGUE_ARG_DTYPES: fx.Constexpr = ()\n"
            "RETURNS_AUX: fx.Constexpr = False\n"
            "AUX_OUT_DTYPE: fx.Constexpr = 'f16'\n"
        )

    kernel = FlyDSLGemmEpilogueKernel()
    env: dict[str, Any] = {
        original_buffer_name: FlyDSLCSEVariable(
            "acc", ValueRanges.unknown(), dtype=torch.float32, shape=(1,)
        )
    }

    with V.set_kernel_handler(kernel), V.set_ops_handler(
        FlyDSLSchedulerEpilogueOpOverrides(env)
    ):
        for node in epilogue_nodes:
            if node.is_reduction():
                raise NotImplementedError("FlyDSL hgemm epilogue reductions unsupported")
            for scheduler_node in node.get_nodes():
                ir_node = scheduler_node.node
                if not isinstance(ir_node, ComputedBuffer) or not isinstance(
                    ir_node.data, Pointwise
                ):
                    raise NotImplementedError(
                        f"unsupported FlyDSL hgemm epilogue node: {ir_node}"
                    )
                result = ir_node.data.inner_fn(*ir_node.data.inner_fn_args())
                env[ir_node.get_name()] = result

    final_name = epilogue_nodes[-1].get_name()
    if final_name not in env:
        raise AssertionError(f"missing final epilogue value for {final_name}")

    body = "\n".join(f"    {line}" for line in kernel.body.lines)
    if body:
        body += "\n"
    result = env[final_name]
    key_payload = f"{original_buffer_name}\n{body}\nreturn {result}"
    key = hashlib.sha256(key_payload.encode()).hexdigest()[:16]
    name = f"flydsl_hgemm_epilogue_{key}"
    return (
        name,
        f"@flyc.jit\ndef {name}(acc):\n"
        f"{body}    return {result}\n\n"
        "HAS_EPILOGUE: fx.Constexpr = True\n"
        f"EPILOGUE_FN = {name}\n"
        "EPILOGUE_ARG_KINDS: fx.Constexpr = ()\n"
        "EPILOGUE_ARG_DTYPES: fx.Constexpr = ()\n"
        "RETURNS_AUX: fx.Constexpr = False\n"
        "AUX_OUT_DTYPE: fx.Constexpr = 'f16'\n",
    )


def materialize_flydsl_gemm_epilogue(
    graph_module: torch.fx.GraphModule,
    gemm_op: Any,
    epilogue_arg_placeholders: tuple[torch.fx.Node, ...] = (),
) -> tuple[str, str]:
    """Build a generated FlyDSL epilogue callable from an FX GEMM body."""
    gemm = gemm_node(graph_module, gemm_op)
    outputs = output_plan(graph_module)
    kernel = FlyDSLGemmEpilogueKernel()
    env: dict[torch.fx.Node, Any] = {
        gemm: FlyDSLCSEVariable(
            "acc", ValueRanges.unknown(), dtype=torch.float32, shape=(1,)
        )
    }

    with V.set_kernel_handler(kernel), V.set_ops_handler(
        FlyDSLGemmEpilogueOpOverrides()
    ):
        for index, node in enumerate(epilogue_arg_placeholders):
            epilogue_arg_meta = node.meta["val"]
            physical_dtype = (
                torch.uint8
                if epilogue_arg_meta.dtype is torch.bool
                else epilogue_arg_meta.dtype
            )
            logical_dtype = upcast_compute_type(epilogue_arg_meta.dtype)
            env[node] = FlyDSLCSEVariable(
                f"aux{index}",
                ValueRanges.unknown(),
                dtype=physical_dtype,
                shape=(1,),
            )
            if logical_dtype != physical_dtype:
                env[node] = FlyDSLGemmEpilogueOpOverrides.to_dtype(
                    env[node], logical_dtype, use_compute_types=False
                )

        for node in graph_module.graph.nodes:
            if node is gemm or node.op in ("placeholder", "output"):
                continue
            with V.set_current_node(node):
                node_args = tuple(_flydsl_arg(arg, env) for arg in node.args)
                node_kwargs = {
                    key: _flydsl_arg(value, env) for key, value in node.kwargs.items()
                }
                if node.op == "call_function":
                    env[node] = _flydsl_call(node.target, node_args, node_kwargs)
                    continue
                if node.op == "call_method":
                    env[node] = _flydsl_method_call(
                        node.target, node_args, node_kwargs
                    )
                    continue
                raise NotImplementedError(
                    f"unsupported FlyDSL GEMM epilogue node: {node.format_node()}"
                )

    body = "\n".join(f"    {line}" for line in kernel.body.lines)
    if body:
        body += "\n"
    aux_args = [f"aux{index}" for index in range(len(epilogue_arg_placeholders))]
    epilogue_params = ", ".join(["acc", *aux_args])
    result = _flydsl_arg(outputs.output, env)
    if outputs.aux_outputs:
        aux_results = [_flydsl_arg(aux_output, env) for aux_output in outputs.aux_outputs]
        result = f"({', '.join(str(item) for item in (result, *aux_results))})"

    key_payload = f"{graph_module.code}\n{body}\nreturn {result}"
    key = hashlib.sha256(key_payload.encode()).hexdigest()[:16]
    name = f"flydsl_gemm_epilogue_{key}"
    return (
        name,
        "import flydsl.compiler as flyc\n"
        "import flydsl.expr as fx\n\n"
        f"@flyc.jit\ndef {name}({epilogue_params}):\n"
        f"{body}    return {result}\n",
    )


def materialize_flydsl_flex_gemm_epilogue(
    graph_module: torch.fx.GraphModule,
    gemm_op: Any,
    epilogue_arg_placeholders: tuple[torch.fx.Node, ...] = (),
) -> tuple[str, str]:
    """Build an embeddable FlyDSL epilogue callable from a FlexGEMM body."""
    gemm = gemm_node(graph_module, gemm_op)
    outputs = output_plan(graph_module)
    if len(outputs.aux_outputs) > 1:
        raise NotImplementedError("FlyDSL FlexGEMM supports at most one aux output")

    kernel = FlyDSLGemmEpilogueKernel()
    env: dict[torch.fx.Node, Any] = {
        gemm: FlyDSLCSEVariable(
            "acc", ValueRanges.unknown(), dtype=torch.float32, shape=(1,)
        )
    }

    with V.set_kernel_handler(kernel), V.set_ops_handler(
        FlyDSLGemmEpilogueOpOverrides()
    ):
        for index, node in enumerate(epilogue_arg_placeholders):
            epilogue_arg_meta = node.meta["val"]
            physical_dtype = (
                torch.uint8
                if epilogue_arg_meta.dtype is torch.bool
                else epilogue_arg_meta.dtype
            )
            logical_dtype = upcast_compute_type(epilogue_arg_meta.dtype)
            env[node] = FlyDSLCSEVariable(
                f"aux{index}",
                ValueRanges.unknown(),
                dtype=physical_dtype,
                shape=(1,),
            )
            if logical_dtype != physical_dtype:
                env[node] = FlyDSLGemmEpilogueOpOverrides.to_dtype(
                    env[node], logical_dtype, use_compute_types=False
                )

        for node in graph_module.graph.nodes:
            if node is gemm or node.op in ("placeholder", "output"):
                continue
            with V.set_current_node(node):
                node_args = tuple(_flydsl_arg(arg, env) for arg in node.args)
                node_kwargs = {
                    key: _flydsl_arg(value, env) for key, value in node.kwargs.items()
                }
                if node.op == "call_function":
                    env[node] = _flydsl_call(node.target, node_args, node_kwargs)
                    continue
                if node.op == "call_method":
                    env[node] = _flydsl_method_call(
                        node.target, node_args, node_kwargs
                    )
                    continue
                raise NotImplementedError(
                    f"unsupported FlyDSL FlexGEMM epilogue node: {node.format_node()}"
                )

    body = "\n".join(f"    {line}" for line in kernel.body.lines)
    if body:
        body += "\n"
    result = _flydsl_arg(outputs.output, env)
    if outputs.aux_outputs:
        aux_result = _flydsl_arg(outputs.aux_outputs[0], env)
        result = f"({result}, {aux_result})"
    key_payload = f"{graph_module.code}\n{body}\nreturn {result}"
    key = hashlib.sha256(key_payload.encode()).hexdigest()[:16]
    name = f"flydsl_flex_gemm_epilogue_{key}"
    epilogue_params = ", ".join(
        ["acc", *(f"aux{index}" for index in range(len(epilogue_arg_placeholders)))]
    )
    return (
        name,
        f"@flyc.jit\ndef {name}({epilogue_params}):\n"
        f"{body}    return {result}\n\n"
        "HAS_EPILOGUE: fx.Constexpr = True\n"
        f"EPILOGUE_FN = {name}\n",
    )
