# mypy: allow-untyped-defs
import contextlib
import logging
from collections.abc import Callable
from typing import Any
from unittest.mock import patch

import torch

from torch._inductor.codegen.common import IndentedBuffer, Kernel
from torch._inductor.ir import BaseView, Buffer, ExternKernel, MutableBox, ReinterpretView
from torch._inductor.utils import OrderedSet
from torch._inductor.virtualized import V


MAIN_SUFFIX = "main"

log = logging.getLogger(__name__)
kernel_code_log = torch._logging.getArtifactLogger(__name__, "kernel_code")


class FlyDSLKernelWrapper:
    """Wrapper to provide the `.run()` interface expected by Inductor."""

    def __init__(self, kernel_fn: Callable[..., Any], kernel_path: str | None = None):
        self.kernel_fn = kernel_fn
        self.kernel_path = kernel_path
        kernel_code_log.info("FlyDSL kernel path: %s", kernel_path)

    def run(self, *args, stream=None, **kwargs):
        return self.kernel_fn(*args, stream=stream, **kwargs)


class FlyDSLTemplateKernel(Kernel):
    """Minimal template kernel implementation for FlyDSL Inductor demos."""

    def __init__(
        self,
        kernel_name: str,
        input_nodes: list[Buffer],
        output_node: Buffer,
        subgraphs: list[Buffer] | None = None,
    ) -> None:
        super().__init__()
        self.kernel_name = kernel_name
        self.input_nodes = input_nodes
        self.output_node = output_node
        self.subgraphs = subgraphs
        self.render_hooks: dict[str, Callable[[], str]] = {}
        self.prologue_fused_inputs: OrderedSet[str] = OrderedSet()
        self.prologue_fused_inputs_preserve_zero: OrderedSet[str] = OrderedSet()
        self._template_input_args: list[tuple[str, Buffer]] = []
        self._seen_input_args: OrderedSet[str] = OrderedSet()

    @contextlib.contextmanager
    def _patch_get_dtype_for_args(self):
        original_get_dtype = V.graph.get_dtype

        def get_dtype(name: str) -> torch.dtype:
            for arg_name, input_node in self._template_input_args:
                if name == arg_name:
                    return input_node.get_dtype()
            return original_get_dtype(name)

        with patch.object(V.graph, "get_dtype", get_dtype):
            yield

    @staticmethod
    def _get_reinterpret_view(node) -> ReinterpretView | None:
        while isinstance(node, MutableBox):
            node = node.data
        if isinstance(node, BaseView):
            return ExternKernel.convert_to_reinterpret_view(node)
        return None

    def gen_imports(self) -> str:
        imports = IndentedBuffer()
        imports.splice(
            """
            import torch
            import flydsl.compiler as flyc
            import flydsl.expr as fx
            """
        )
        return imports.getvalue()

    def gen_defines(self, **kwargs) -> str:
        params = IndentedBuffer()
        for name, val in kwargs.items():
            params.writeline(f"{name}: fx.Constexpr = {val!r}")
        return params.getvalue()

    def gen_epilogue(self) -> str:
        explicit_epilogue_source = getattr(self, "explicit_epilogue_source", None)
        if explicit_epilogue_source is not None:
            epilogue_arg_kinds = getattr(self, "explicit_epilogue_arg_kinds", ())
            epilogue_arg_dtypes = getattr(self, "explicit_epilogue_arg_dtypes", ())
            returns_aux = getattr(self, "explicit_returns_aux", False)
            aux_out_dtype = getattr(self, "explicit_aux_out_dtype", "f16")
            return (
                explicit_epilogue_source
                + f"EPILOGUE_ARG_KINDS: fx.Constexpr = {tuple(epilogue_arg_kinds)!r}\n"
                + f"EPILOGUE_ARG_DTYPES: fx.Constexpr = {tuple(epilogue_arg_dtypes)!r}\n"
                + f"RETURNS_AUX: fx.Constexpr = {returns_aux!r}\n"
                + f"AUX_OUT_DTYPE: fx.Constexpr = {aux_out_dtype!r}\n"
            )

        epilogue_nodes = getattr(self, "epilogue_nodes", None)
        if not epilogue_nodes:
            return (
                "HAS_EPILOGUE: fx.Constexpr = False\n"
                "EPILOGUE_FN = None\n"
                "EPILOGUE_ARG_KINDS: fx.Constexpr = ()\n"
                "EPILOGUE_ARG_DTYPES: fx.Constexpr = ()\n"
                "RETURNS_AUX: fx.Constexpr = False\n"
                "AUX_OUT_DTYPE: fx.Constexpr = 'f16'\n"
            )

        from torch._inductor.kernel.flydsl.epilogue import (
            materialize_flydsl_scheduler_epilogue,
        )

        _, code = materialize_flydsl_scheduler_epilogue(
            self.original_output_name, list(epilogue_nodes)
        )
        return code

    def render(self, template, **kwargs):
        from torch._inductor.select_algorithm import PartialRender

        self._template_kwargs = dict(kwargs)
        template_env = {
            "def_kernel": self.def_kernel,
            "gen_defines": lambda: self.gen_defines(**kwargs),
            "gen_epilogue": self.gen_epilogue,
            "get_epilogue_args": self.get_epilogue_args,
            "get_aux_output": self.get_aux_output,
            "get_output": self.get_output,
        }

        rendered_code = template.render(
            kernel_name=self.kernel_name,
            input_nodes=self.input_nodes,
            output_node=self.output_node,
            **template_env,
            **kwargs,
        )
        return PartialRender(self.gen_imports() + rendered_code, self.render_hooks)

    def def_kernel(self, *argnames):
        renames = IndentedBuffer(initial_indent=1)
        self._template_input_args = []
        self._seen_input_args = OrderedSet()

        for i, input_node in enumerate(self.input_nodes):
            buf_name = input_node.get_name()
            self.args.input(buf_name)
            if i < len(argnames):
                template_name = argnames[i]
                arg_name = f"arg_{template_name}"
                renames.writeline(f"{template_name} = {arg_name}")
            else:
                arg_name = f"arg{i}"
            self.args.input_buffers[buf_name] = arg_name
            self._template_input_args.append((arg_name, input_node))
            self._seen_input_args.add(arg_name)

        if self.output_node:
            self.args.output(self.output_node.get_name())

        def hook():
            code = IndentedBuffer()
            params = [arg_name for arg_name, _ in self._template_input_args]
            with self._patch_get_dtype_for_args():
                arg_defs, _, _, _ = self.args.python_argdefs()
            for arg_def in arg_defs:
                if arg_def.full_name() not in self._seen_input_args:
                    params.append(arg_def.full_name())
            params.append("stream")
            code.writeline(
                f"def {self.kernel_name}_{MAIN_SUFFIX}({', '.join(params)}):"
            )
            with code.indent():
                code.splice(renames.getvalue())
            return code.getvalue()

        if "<DEF_KERNEL>" in self.render_hooks:
            raise AssertionError("<DEF_KERNEL> hook already registered")
        self.render_hooks["<DEF_KERNEL>"] = hook
        return "<DEF_KERNEL>"

    def get_epilogue_args(self):
        epilogue_arg_count = getattr(self, "explicit_epilogue_arg_count", None)
        arg_names = [arg_name for arg_name, _ in self._template_input_args[2:]]
        if epilogue_arg_count is not None:
            arg_names = arg_names[:epilogue_arg_count]
        if not arg_names:
            return "()"
        return f"({', '.join(arg_names)},)"

    def get_aux_output(self):
        aux_out_index = getattr(self, "explicit_aux_out_index", None)
        if aux_out_index is None:
            return "None"
        return self._template_input_args[aux_out_index][0]

    def get_output(self):
        if not self.output_node:
            raise AssertionError("Output node must exist to get output buffer name")
        output = self.args.output_buffers.get(self.output_node.get_name(), None)
        if output is None:
            raise ValueError(f"Output buffer '{self.output_node.get_name()}' not found")
        return output

    def call_kernel(self, name: str, node=None):
        wrapper = V.graph.wrapper_code
        call_args = []
        arg_types = []

        for _, input_node in self._template_input_args:
            reinterpret_view = self._get_reinterpret_view(input_node)
            call_args.append(
                reinterpret_view.codegen_reference()
                if reinterpret_view is not None
                else input_node.get_name()
            )
            arg_types.append(V.graph.get_dtype(input_node.get_name()))

        with self._patch_get_dtype_for_args():
            orig_arg_defs, orig_call_args, _, orig_arg_types = (
                self.args.python_argdefs()
            )
        for arg_def, call_arg, arg_type in zip(
            orig_arg_defs, orig_call_args, orig_arg_types
        ):
            if arg_def.full_name() in self._seen_input_args:
                continue
            call_args.append(call_arg)
            arg_types.append(arg_type)

        wrapper.generate_kernel_call(name, call_args, triton=True, arg_types=arg_types)
