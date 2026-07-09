# mypy: allow-untyped-defs
from __future__ import annotations

import sympy

import torch
from torch._higher_order_ops.flex_gemm import flex_gemm
from torch.fx import symbolic_trace
from torch._inductor.codegen.flydsl.flydsl_op_overrides import (
    FlyDSLCSEVariable,
    FlyDSLOpOverrides,
)
from torch._inductor.kernel.flydsl.epilogue import (
    materialize_flydsl_flex_gemm_epilogue,
    materialize_flydsl_gemm_epilogue,
)
from torch._inductor.test_case import TestCase, run_tests
from torch._inductor.virtualized import ops, V
from torch.utils._sympy.value_ranges import ValueRanges


class _FlyDSLBody:
    def __init__(self) -> None:
        self.lines: list[str] = []

    def writeline(self, line: str) -> None:
        self.lines.append(line)

    def getvalue(self) -> str:
        return "\n".join(self.lines)


class _FlyDSLCSE:
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


class _FlyDSLKernel:
    def __init__(self) -> None:
        self.body = _FlyDSLBody()
        self.cse = _FlyDSLCSE()


class TestFlyDSLOpOverrides(TestCase):
    def test_direct_expression_lowering(self):
        self.assertEqual(FlyDSLOpOverrides.exp2("acc"), "fx.exp2(acc)")
        self.assertEqual(FlyDSLOpOverrides.rsqrt("acc"), "fx.rsqrt(acc)")
        self.assertEqual(FlyDSLOpOverrides.log2("acc"), "fx.log2(acc)")
        self.assertEqual(FlyDSLOpOverrides.atan2("y", "x"), "fx.atan2(y, x)")
        self.assertEqual(FlyDSLOpOverrides.floor("x"), "fx.floor(x)")
        self.assertEqual(
            FlyDSLOpOverrides.logical_xor("mask0", "mask1"),
            "((mask0).__fly_bool__() ^ (mask1).__fly_bool__())",
        )
        self.assertEqual(
            FlyDSLOpOverrides.where("cond", "lhs", "rhs"),
            "(cond).select(lhs, rhs)",
        )

    def test_dtype_cast_lowering(self):
        self.assertEqual(
            FlyDSLOpOverrides.to_dtype(
                "acc", torch.float16, use_compute_types=False
            ),
            "fx.Float16(acc)",
        )
        self.assertEqual(
            FlyDSLOpOverrides.to_dtype("acc", torch.float16),
            "fx.Float32(acc)",
        )
        self.assertEqual(
            FlyDSLOpOverrides.to_dtype(
                "mask", torch.bool, use_compute_types=False
            ),
            "fx.Boolean(mask)",
        )

    def test_ops_handler_demo(self):
        with V.set_ops_handler(FlyDSLOpOverrides()):
            expr = ops.rsqrt(ops.add(ops.exp2("acc"), "bias"))
            mask = ops.logical_xor(ops.gt("acc", 0), "mask")

        self.assertEqual(str(expr), "fx.rsqrt((fx.exp2(acc) + bias))")
        self.assertEqual(
            str(mask),
            "(((acc > 0)).__fly_bool__() ^ (mask).__fly_bool__())",
        )

    def test_cse_expression_lowering(self):
        kernel = _FlyDSLKernel()
        acc = FlyDSLCSEVariable(
            "acc", ValueRanges.unknown(), dtype=torch.float32, shape=(1,)
        )
        bias = FlyDSLCSEVariable(
            "bias", ValueRanges.unknown(), dtype=torch.float32, shape=(1,)
        )

        with V.set_kernel_handler(kernel):
            exp2 = FlyDSLOpOverrides.exp2(acc)
            result = FlyDSLOpOverrides.rsqrt(FlyDSLOpOverrides.add(exp2, bias))

        self.assertIsInstance(result, FlyDSLCSEVariable)
        self.assertEqual(str(result), "tmp2")
        self.assertEqual(
            kernel.body.getvalue(),
            "\n".join(
                [
                    "tmp0 = fx.exp2(acc)",
                    "tmp1 = (tmp0 + bias)",
                    "tmp2 = fx.rsqrt(tmp1)",
                ]
            ),
        )

    def test_index_expr_metadata(self):
        kernel = _FlyDSLKernel()
        x = FlyDSLCSEVariable(
            "x",
            ValueRanges.unknown(),
            dtype=torch.int64,
            is_scalar_expr=True,
            index_expr=sympy.Symbol("x", integer=True),
        )

        class _SizeVars:
            @staticmethod
            def simplify(expr):
                return sympy.simplify(expr)

        class _Graph:
            sizevars = _SizeVars()

        with V.set_graph_handler(_Graph()), V.set_kernel_handler(kernel):
            result = FlyDSLOpOverrides.add(x, 1)

        self.assertIsInstance(result, FlyDSLCSEVariable)
        self.assertEqual(result.index_expr, sympy.Symbol("x", integer=True) + 1)


class TestFlyDSLGemmEpilogueFusion(TestCase):
    def test_symbolic_mm_epilogue_generates_flydsl_code(self):
        def fn(a, b, bias, mask):
            acc = torch.mm(a, b)
            value = torch.rsqrt(torch.exp2(acc + bias))
            pred = torch.logical_xor(acc > 0, mask)
            return torch.where(pred, value, acc)

        graph_module = symbolic_trace(fn)
        placeholders = {
            node.target: node for node in graph_module.graph.nodes if node.op == "placeholder"
        }
        placeholders["bias"].meta["val"] = torch.empty(
            (16, 16), dtype=torch.float16
        )
        placeholders["mask"].meta["val"] = torch.empty((16, 16), dtype=torch.bool)

        name, code = materialize_flydsl_gemm_epilogue(
            graph_module,
            torch.mm,
            epilogue_arg_placeholders=(
                placeholders["bias"],
                placeholders["mask"],
            ),
        )

        self.assertTrue(name.startswith("flydsl_gemm_epilogue_"))
        self.assertIn("import flydsl.compiler as flyc", code)
        self.assertIn("import flydsl.expr as fx", code)
        self.assertIn("@flyc.jit", code)
        self.assertIn("def " + name + "(acc, aux0, aux1):", code)
        self.assertIn("fx.Float32(aux0)", code)
        self.assertIn("fx.Boolean(aux1)", code)
        self.assertIn("fx.exp2", code)
        self.assertIn("fx.rsqrt", code)
        self.assertIn("__fly_bool__() ^", code)
        self.assertIn(".select(", code)
        self.assertIn("return", code)

    def test_symbolic_mm_epilogue_cast_method(self):
        def fn(a, b):
            return torch.mm(a, b).to(torch.float16)

        graph_module = symbolic_trace(fn)
        name, code = materialize_flydsl_gemm_epilogue(graph_module, torch.mm)

        self.assertTrue(name.startswith("flydsl_gemm_epilogue_"))
        self.assertIn("fx.Float16(acc)", code)
        self.assertIn("return tmp0", code)


class TestFlyDSLFlexGemmPath(TestCase):
    def test_flex_gemm_epilogue_generates_embeddable_flydsl_code(self):
        def fn(a, b):
            acc = torch.mm(a, b)
            return torch.rsqrt(torch.exp2(acc))

        graph_module = symbolic_trace(fn)
        name, code = materialize_flydsl_flex_gemm_epilogue(graph_module, torch.mm)

        self.assertTrue(name.startswith("flydsl_flex_gemm_epilogue_"))
        self.assertIn("@flyc.jit", code)
        self.assertIn("def " + name + "(acc):", code)
        self.assertIn("fx.exp2(acc)", code)
        self.assertIn("fx.rsqrt", code)
        self.assertIn("HAS_EPILOGUE: fx.Constexpr = True", code)
        self.assertIn(f"EPILOGUE_FN = {name}", code)

    def test_flex_gemm_frontend_accepts_flydsl_backend(self):
        a = torch.randn(4, 8)
        b = torch.randn(8, 6)

        def epilogue(acc):
            return torch.rsqrt(torch.exp2(acc))

        actual = flex_gemm(
            torch.mm,
            (a, b),
            epilogue,
            kernel_options={"backend": "FLYDSL"},
        )

        torch.testing.assert_close(actual, epilogue(a @ b))


if __name__ == "__main__":
    run_tests()
