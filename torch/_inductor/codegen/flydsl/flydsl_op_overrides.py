# mypy: allow-untyped-defs
"""
FlyDSL-specific operation overrides for pointwise expressions.

This module maps Inductor ops into Python source expressions that are valid
inside a FlyDSL JIT function using ``import flydsl.expr as fx``.
"""

from __future__ import annotations

import math
from typing import Any

import sympy

import torch
from torch._inductor.codegen.common import CSEVariable, OpOverrides
from torch._inductor.utils import get_bounds_index_expr
from torch._inductor.virtualized import OpsValue, V
from torch.utils._sympy.value_ranges import ValueRanges


class FlyDSLCSEVariable(CSEVariable):
    def __init__(
        self,
        name: str,
        bounds: ValueRanges[sympy.Expr],
        dtype: torch.dtype | None = None,
        shape=None,
        *,
        is_scalar_expr: bool = False,
        index_expr: sympy.Expr | None = None,
    ) -> None:
        super().__init__(name, bounds, dtype, shape)
        self.is_scalar_expr = is_scalar_expr
        self.index_expr = index_expr


FlyDSLArg = CSEVariable | OpsValue | str | bool | float | int


def upcast_compute_type(dtype: torch.dtype) -> torch.dtype:
    """Maybe upcast [b]float16 to float32 for expression compute."""
    if dtype in (torch.float16, torch.bfloat16):
        return torch.float32
    return dtype


def _optional_dtype(name: str) -> torch.dtype | None:
    return getattr(torch, name, None)


class FlyDSLOpOverrides(OpOverrides):
    """
    Map Inductor pointwise ops to FlyDSL expression syntax.

    The generated code assumes the template wrapper imports ``flydsl.expr`` as
    ``fx``.  The class intentionally does not import FlyDSL at module import time
    so PyTorch remains importable when FlyDSL is not installed.
    """

    TORCH_TO_FLYDSL_DTYPE = {
        dtype: flydsl_dtype
        for dtype, flydsl_dtype in [
            (torch.float16, "fx.Float16"),
            (torch.bfloat16, "fx.BFloat16"),
            (torch.float32, "fx.Float32"),
            (torch.float64, "fx.Float64"),
            (torch.int8, "fx.Int8"),
            (torch.int16, "fx.Int16"),
            (torch.int32, "fx.Int32"),
            (torch.int64, "fx.Int64"),
            (torch.uint8, "fx.Uint8"),
            (_optional_dtype("uint16"), "fx.Uint16"),
            (_optional_dtype("uint32"), "fx.Uint32"),
            (_optional_dtype("uint64"), "fx.Uint64"),
            (torch.bool, "fx.Boolean"),
        ]
        if dtype is not None
    }

    @staticmethod
    def _get_cse_var(arg: FlyDSLArg) -> CSEVariable | None:
        if isinstance(arg, CSEVariable):
            return arg
        if isinstance(arg, OpsValue) and isinstance(arg.value, CSEVariable):
            return arg.value
        return None

    @staticmethod
    def _as_expr(arg: FlyDSLArg) -> str:
        cse_var = FlyDSLOpOverrides._get_cse_var(arg)
        if cse_var is not None:
            return str(cse_var)
        return str(arg)

    @staticmethod
    def _is_scalar_expr(arg: FlyDSLArg) -> bool:
        cse_var = FlyDSLOpOverrides._get_cse_var(arg)
        return isinstance(cse_var, FlyDSLCSEVariable) and cse_var.is_scalar_expr

    @staticmethod
    def _index_expr(arg: FlyDSLArg) -> sympy.Expr | None:
        cse_var = FlyDSLOpOverrides._get_cse_var(arg)
        if isinstance(cse_var, FlyDSLCSEVariable):
            return cse_var.index_expr
        if isinstance(arg, (int, sympy.Integer)):
            return sympy.Integer(arg)
        if isinstance(arg, str) and arg.lstrip("-").isdigit():
            return sympy.Integer(int(arg))
        return None

    @staticmethod
    def _extract_dtype_bounds_shape(
        *args: FlyDSLArg,
    ) -> tuple[torch.dtype | None, ValueRanges[sympy.Expr], Any]:
        for arg in args:
            cse_var = FlyDSLOpOverrides._get_cse_var(arg)
            if cse_var is not None:
                return cse_var.dtype, cse_var.bounds, cse_var.shape
        return None, ValueRanges.unknown(), None

    @staticmethod
    def _cse_generate(
        expr: str,
        *args: FlyDSLArg,
        dtype: torch.dtype | None = None,
        shape=None,
        is_scalar_expr: bool | None = None,
    ) -> FlyDSLArg:
        if not any(FlyDSLOpOverrides._get_cse_var(arg) is not None for arg in args):
            return expr

        inferred_dtype, bounds, inferred_shape = (
            FlyDSLOpOverrides._extract_dtype_bounds_shape(*args)
        )
        result = V.kernel.cse.generate(
            V.kernel.body,
            expr,
            bounds=bounds,
            dtype=dtype if dtype is not None else inferred_dtype,
            shape=shape if shape is not None else inferred_shape,
        )
        if isinstance(result, FlyDSLCSEVariable):
            if is_scalar_expr is None:
                is_scalar_expr = all(
                    FlyDSLOpOverrides._is_scalar_expr(arg)
                    for arg in args
                    if FlyDSLOpOverrides._get_cse_var(arg) is not None
                )
            result.is_scalar_expr = bool(is_scalar_expr)
        return result

    @staticmethod
    def _apply_unary_op(x: FlyDSLArg, op_format: str) -> FlyDSLArg:
        result_expr = op_format.format(x=FlyDSLOpOverrides._as_expr(x))
        return FlyDSLOpOverrides._cse_generate(result_expr, x)

    @staticmethod
    def _apply_binary_op(
        a: FlyDSLArg,
        b: FlyDSLArg,
        op_format: str,
        index_expr_fn=None,
        *,
        dtype: torch.dtype | None = None,
    ) -> FlyDSLArg:
        result_expr = op_format.format(
            a=FlyDSLOpOverrides._as_expr(a),
            b=FlyDSLOpOverrides._as_expr(b),
        )
        result = FlyDSLOpOverrides._cse_generate(result_expr, a, b, dtype=dtype)
        if index_expr_fn is not None and isinstance(result, FlyDSLCSEVariable):
            a_expr = FlyDSLOpOverrides._index_expr(a)
            b_expr = FlyDSLOpOverrides._index_expr(b)
            if a_expr is not None and b_expr is not None:
                result.index_expr = V.graph.sizevars.simplify(
                    index_expr_fn(a_expr, b_expr)
                )
        return result

    @staticmethod
    def _apply_ternary_op(
        a: FlyDSLArg,
        b: FlyDSLArg,
        c: FlyDSLArg,
        op_format: str,
        *,
        dtype: torch.dtype | None = None,
    ) -> FlyDSLArg:
        result_expr = op_format.format(
            a=FlyDSLOpOverrides._as_expr(a),
            b=FlyDSLOpOverrides._as_expr(b),
            c=FlyDSLOpOverrides._as_expr(c),
        )
        return FlyDSLOpOverrides._cse_generate(result_expr, a, b, c, dtype=dtype)

    @staticmethod
    def _cast_expr(expr: str, dtype: torch.dtype) -> str:
        flydsl_type = FlyDSLOpOverrides.TORCH_TO_FLYDSL_DTYPE.get(dtype)
        if flydsl_type is None:
            raise NotImplementedError(
                f"FlyDSL dtype cast not implemented for torch dtype: {dtype}"
            )
        return f"{flydsl_type}({expr})"

    @staticmethod
    def constant(value: bool | float | int, dtype: torch.dtype) -> str:
        if value == float("-inf"):
            return "float('-inf')"
        if value == float("inf"):
            return "float('inf')"
        if isinstance(value, float) and math.isnan(value):
            return "float('nan')"
        return repr(value)

    @staticmethod
    def index_expr(expr: sympy.Expr, dtype: torch.dtype) -> FlyDSLArg:
        if isinstance(expr, (int, sympy.Integer)):
            return FlyDSLOpOverrides.constant(int(expr), dtype)

        idx_str = V.kernel.kexpr(V.kernel.rename_indexing(expr))
        result = V.kernel.cse.generate(
            V.kernel.body,
            idx_str,
            bounds=get_bounds_index_expr(expr),
            dtype=dtype,
        )
        if isinstance(result, FlyDSLCSEVariable):
            result.is_scalar_expr = True
            result.index_expr = V.graph.sizevars.simplify(expr)
        return result

    @staticmethod
    def value_expr(expr: sympy.Expr, dtype: torch.dtype) -> FlyDSLArg:
        return FlyDSLOpOverrides.index_expr(expr, dtype)

    @staticmethod
    def add(a: FlyDSLArg, b: FlyDSLArg) -> FlyDSLArg:
        return FlyDSLOpOverrides._apply_binary_op(
            a, b, "({a} + {b})", lambda a_expr, b_expr: a_expr + b_expr
        )

    @staticmethod
    def mul(a: FlyDSLArg, b: FlyDSLArg) -> FlyDSLArg:
        return FlyDSLOpOverrides._apply_binary_op(
            a, b, "({a} * {b})", lambda a_expr, b_expr: a_expr * b_expr
        )

    @staticmethod
    def sub(a: FlyDSLArg, b: FlyDSLArg) -> FlyDSLArg:
        return FlyDSLOpOverrides._apply_binary_op(
            a, b, "({a} - {b})", lambda a_expr, b_expr: a_expr - b_expr
        )

    @staticmethod
    def truediv(a: FlyDSLArg, b: FlyDSLArg) -> FlyDSLArg:
        return FlyDSLOpOverrides._apply_binary_op(a, b, "({a} / {b})")

    @staticmethod
    def floordiv(a: FlyDSLArg, b: FlyDSLArg) -> FlyDSLArg:
        return FlyDSLOpOverrides._apply_binary_op(
            a, b, "({a} // {b})", lambda a_expr, b_expr: a_expr // b_expr
        )

    @staticmethod
    def mod(a: FlyDSLArg, b: FlyDSLArg) -> FlyDSLArg:
        return FlyDSLOpOverrides._apply_binary_op(a, b, "({a} % {b})")

    @staticmethod
    def remainder(a: FlyDSLArg, b: FlyDSLArg) -> FlyDSLArg:
        return FlyDSLOpOverrides._apply_binary_op(
            a, b, "({a} % {b})", lambda a_expr, b_expr: a_expr % b_expr
        )

    @staticmethod
    def pow(a: FlyDSLArg, b: FlyDSLArg) -> FlyDSLArg:
        return FlyDSLOpOverrides._apply_binary_op(a, b, "fx.powf({a}, {b})")

    @staticmethod
    def fma(a: FlyDSLArg, b: FlyDSLArg, c: FlyDSLArg) -> FlyDSLArg:
        return FlyDSLOpOverrides._apply_ternary_op(a, b, c, "fx.fma({a}, {b}, {c})")

    @staticmethod
    def neg(x: FlyDSLArg) -> FlyDSLArg:
        return FlyDSLOpOverrides._apply_unary_op(x, "(-{x})")

    @staticmethod
    # pyrefly: ignore [bad-override]
    def abs(x: FlyDSLArg) -> FlyDSLArg:
        cse_var = FlyDSLOpOverrides._get_cse_var(x)
        dtype = cse_var.dtype if cse_var is not None else None
        op_name = "fx.absi" if dtype is not None and not dtype.is_floating_point else "fx.absf"
        return FlyDSLOpOverrides._apply_unary_op(x, f"{op_name}({{x}})")

    @staticmethod
    # pyrefly: ignore [bad-override]
    def exp(x: FlyDSLArg) -> FlyDSLArg:
        return FlyDSLOpOverrides._apply_unary_op(x, "fx.exp({x})")

    @staticmethod
    # pyrefly: ignore [bad-override]
    def exp2(x: FlyDSLArg) -> FlyDSLArg:
        return FlyDSLOpOverrides._apply_unary_op(x, "fx.exp2({x})")

    @staticmethod
    def expm1(x: FlyDSLArg) -> FlyDSLArg:
        return FlyDSLOpOverrides._apply_unary_op(x, "fx.expm1({x})")

    @staticmethod
    # pyrefly: ignore [bad-override]
    def sqrt(x: FlyDSLArg) -> FlyDSLArg:
        return FlyDSLOpOverrides._apply_unary_op(x, "fx.sqrt({x})")

    @staticmethod
    # pyrefly: ignore [bad-override]
    def rsqrt(x: FlyDSLArg) -> FlyDSLArg:
        return FlyDSLOpOverrides._apply_unary_op(x, "fx.rsqrt({x})")

    @staticmethod
    # pyrefly: ignore [bad-override]
    def log(x: FlyDSLArg) -> FlyDSLArg:
        return FlyDSLOpOverrides._apply_unary_op(x, "fx.log({x})")

    @staticmethod
    # pyrefly: ignore [bad-override]
    def log2(x: FlyDSLArg) -> FlyDSLArg:
        return FlyDSLOpOverrides._apply_unary_op(x, "fx.log2({x})")

    @staticmethod
    # pyrefly: ignore [bad-override]
    def log10(x: FlyDSLArg) -> FlyDSLArg:
        return FlyDSLOpOverrides._apply_unary_op(x, "fx.log10({x})")

    @staticmethod
    def log1p(x: FlyDSLArg) -> FlyDSLArg:
        return FlyDSLOpOverrides._apply_unary_op(x, "fx.log1p({x})")

    @staticmethod
    # pyrefly: ignore [bad-override]
    def sin(x: FlyDSLArg) -> FlyDSLArg:
        return FlyDSLOpOverrides._apply_unary_op(x, "fx.sin({x})")

    @staticmethod
    # pyrefly: ignore [bad-override]
    def cos(x: FlyDSLArg) -> FlyDSLArg:
        return FlyDSLOpOverrides._apply_unary_op(x, "fx.cos({x})")

    @staticmethod
    # pyrefly: ignore [bad-override]
    def tan(x: FlyDSLArg) -> FlyDSLArg:
        return FlyDSLOpOverrides._apply_unary_op(x, "fx.tan({x})")

    @staticmethod
    # pyrefly: ignore [bad-override]
    def asin(x: FlyDSLArg) -> FlyDSLArg:
        return FlyDSLOpOverrides._apply_unary_op(x, "fx.asin({x})")

    @staticmethod
    # pyrefly: ignore [bad-override]
    def acos(x: FlyDSLArg) -> FlyDSLArg:
        return FlyDSLOpOverrides._apply_unary_op(x, "fx.acos({x})")

    @staticmethod
    # pyrefly: ignore [bad-override]
    def atan(x: FlyDSLArg) -> FlyDSLArg:
        return FlyDSLOpOverrides._apply_unary_op(x, "fx.atan({x})")

    @staticmethod
    # pyrefly: ignore [bad-override]
    def atan2(a: FlyDSLArg, b: FlyDSLArg) -> FlyDSLArg:
        return FlyDSLOpOverrides._apply_binary_op(a, b, "fx.atan2({a}, {b})")

    @staticmethod
    # pyrefly: ignore [bad-override]
    def erf(x: FlyDSLArg) -> FlyDSLArg:
        return FlyDSLOpOverrides._apply_unary_op(x, "fx.erf({x})")

    @staticmethod
    def tanh(x: FlyDSLArg) -> FlyDSLArg:
        return FlyDSLOpOverrides._apply_unary_op(x, "fx.tanh({x})")

    @staticmethod
    def sigmoid(x: FlyDSLArg) -> FlyDSLArg:
        x_expr = FlyDSLOpOverrides._as_expr(x)
        result_expr = f"(fx.Float32(1.0) / (fx.Float32(1.0) + fx.exp(-{x_expr})))"
        return FlyDSLOpOverrides._cse_generate(result_expr, x, dtype=torch.float32)

    @staticmethod
    def relu(x: FlyDSLArg) -> FlyDSLArg:
        return FlyDSLOpOverrides.maximum(x, 0)

    @staticmethod
    # pyrefly: ignore [bad-override]
    def floor(x: FlyDSLArg) -> FlyDSLArg:
        return FlyDSLOpOverrides._apply_unary_op(x, "fx.floor({x})")

    @staticmethod
    # pyrefly: ignore [bad-override]
    def ceil(x: FlyDSLArg) -> FlyDSLArg:
        return FlyDSLOpOverrides._apply_unary_op(x, "fx.ceil({x})")

    @staticmethod
    def trunc(x: FlyDSLArg) -> FlyDSLArg:
        return FlyDSLOpOverrides._apply_unary_op(x, "fx.trunc({x})")

    @staticmethod
    # pyrefly: ignore [bad-override]
    def minimum(a: FlyDSLArg, b: FlyDSLArg) -> FlyDSLArg:
        return FlyDSLOpOverrides.where(FlyDSLOpOverrides.lt(a, b), a, b)

    @staticmethod
    # pyrefly: ignore [bad-override]
    def maximum(a: FlyDSLArg, b: FlyDSLArg) -> FlyDSLArg:
        return FlyDSLOpOverrides.where(FlyDSLOpOverrides.gt(a, b), a, b)

    @staticmethod
    # pyrefly: ignore [bad-override]
    def where(condition: FlyDSLArg, a: FlyDSLArg, b: FlyDSLArg) -> FlyDSLArg:
        result_expr = (
            f"({FlyDSLOpOverrides._as_expr(condition)}).select("
            f"{FlyDSLOpOverrides._as_expr(a)}, {FlyDSLOpOverrides._as_expr(b)})"
        )
        dtype, _, shape = FlyDSLOpOverrides._extract_dtype_bounds_shape(a, b, condition)
        return FlyDSLOpOverrides._cse_generate(
            result_expr,
            condition,
            a,
            b,
            dtype=dtype,
            shape=shape,
        )

    @staticmethod
    def logical_and(x0: FlyDSLArg, x1: FlyDSLArg) -> FlyDSLArg:
        return FlyDSLOpOverrides._apply_binary_op(
            x0, x1, "({a}).__fly_and__({b})", dtype=torch.bool
        )

    @staticmethod
    def logical_or(x0: FlyDSLArg, x1: FlyDSLArg) -> FlyDSLArg:
        return FlyDSLOpOverrides._apply_binary_op(
            x0, x1, "({a}).__fly_or__({b})", dtype=torch.bool
        )

    @staticmethod
    def logical_xor(x0: FlyDSLArg, x1: FlyDSLArg) -> FlyDSLArg:
        return FlyDSLOpOverrides._apply_binary_op(
            x0, x1, "(({a}).__fly_bool__() ^ ({b}).__fly_bool__())", dtype=torch.bool
        )

    @staticmethod
    # pyrefly: ignore [bad-override]
    def logical_not(a: FlyDSLArg) -> FlyDSLArg:
        return FlyDSLOpOverrides._apply_unary_op(a, "({x}).__fly_not__()")

    @staticmethod
    # pyrefly: ignore [bad-override]
    def bitwise_and(x: FlyDSLArg, y: FlyDSLArg) -> FlyDSLArg:
        return FlyDSLOpOverrides._apply_binary_op(x, y, "({a} & {b})")

    @staticmethod
    # pyrefly: ignore [bad-override]
    def bitwise_or(x: FlyDSLArg, y: FlyDSLArg) -> FlyDSLArg:
        return FlyDSLOpOverrides._apply_binary_op(x, y, "({a} | {b})")

    @staticmethod
    # pyrefly: ignore [bad-override]
    def bitwise_xor(x: FlyDSLArg, y: FlyDSLArg) -> FlyDSLArg:
        return FlyDSLOpOverrides._apply_binary_op(x, y, "({a} ^ {b})")

    @staticmethod
    # pyrefly: ignore [bad-override]
    def bitwise_not(x: FlyDSLArg) -> FlyDSLArg:
        return FlyDSLOpOverrides._apply_unary_op(x, "(~{x})")

    @staticmethod
    # pyrefly: ignore [bad-override]
    def bitwise_left_shift(x: FlyDSLArg, y: FlyDSLArg) -> FlyDSLArg:
        return FlyDSLOpOverrides._apply_binary_op(x, y, "({a} << {b})")

    @staticmethod
    # pyrefly: ignore [bad-override]
    def bitwise_right_shift(x: FlyDSLArg, y: FlyDSLArg) -> FlyDSLArg:
        return FlyDSLOpOverrides._apply_binary_op(x, y, "({a} >> {b})")

    @staticmethod
    def eq(a: FlyDSLArg, b: FlyDSLArg) -> FlyDSLArg:
        return FlyDSLOpOverrides._apply_binary_op(a, b, "({a} == {b})", dtype=torch.bool)

    @staticmethod
    def ne(a: FlyDSLArg, b: FlyDSLArg) -> FlyDSLArg:
        return FlyDSLOpOverrides._apply_binary_op(a, b, "({a} != {b})", dtype=torch.bool)

    @staticmethod
    def lt(a: FlyDSLArg, b: FlyDSLArg) -> FlyDSLArg:
        return FlyDSLOpOverrides._apply_binary_op(a, b, "({a} < {b})", dtype=torch.bool)

    @staticmethod
    def le(a: FlyDSLArg, b: FlyDSLArg) -> FlyDSLArg:
        return FlyDSLOpOverrides._apply_binary_op(a, b, "({a} <= {b})", dtype=torch.bool)

    @staticmethod
    def gt(a: FlyDSLArg, b: FlyDSLArg) -> FlyDSLArg:
        return FlyDSLOpOverrides._apply_binary_op(a, b, "({a} > {b})", dtype=torch.bool)

    @staticmethod
    def ge(a: FlyDSLArg, b: FlyDSLArg) -> FlyDSLArg:
        return FlyDSLOpOverrides._apply_binary_op(a, b, "({a} >= {b})", dtype=torch.bool)

    @staticmethod
    def to_dtype(
        x: FlyDSLArg,
        dtype: torch.dtype,
        src_dtype: torch.dtype | None = None,
        use_compute_types: bool = True,
    ) -> FlyDSLArg:
        if use_compute_types:
            dtype = upcast_compute_type(dtype)

        expr = FlyDSLOpOverrides._cast_expr(FlyDSLOpOverrides._as_expr(x), dtype)
        return FlyDSLOpOverrides._cse_generate(expr, x, dtype=dtype)

    @staticmethod
    def convert_element_type(x: FlyDSLArg, dtype: torch.dtype) -> FlyDSLArg:
        return FlyDSLOpOverrides.to_dtype(x, dtype)

    @staticmethod
    def _to_copy(x: FlyDSLArg, *, dtype: torch.dtype, **kwargs: Any) -> FlyDSLArg:
        unsupported_kwargs = {
            key: value
            for key, value in kwargs.items()
            if value not in (None, False, torch.preserve_format)
        }
        if unsupported_kwargs:
            raise NotImplementedError(
                f"unsupported FlyDSL _to_copy kwargs: {unsupported_kwargs}"
            )
        return FlyDSLOpOverrides.to_dtype(x, dtype)

    @staticmethod
    def clamp(
        x: FlyDSLArg,
        min: FlyDSLArg | None = None,
        max: FlyDSLArg | None = None,
    ) -> FlyDSLArg:
        result = x
        if min is not None:
            result = FlyDSLOpOverrides.maximum(result, min)
        if max is not None:
            result = FlyDSLOpOverrides.minimum(result, max)
        return result

    @staticmethod
    def clamp_min(x: FlyDSLArg, min: FlyDSLArg) -> FlyDSLArg:
        return FlyDSLOpOverrides.maximum(x, min)

    @staticmethod
    def clamp_max(x: FlyDSLArg, max: FlyDSLArg) -> FlyDSLArg:
        return FlyDSLOpOverrides.minimum(x, max)
