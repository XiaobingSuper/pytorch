# Owner(s): ["module: inductor"]
import unittest

import torch

from torch._inductor.test_case import TestCase


try:
    import flydsl  # noqa: F401

    HAS_FLYDSL = True
except ImportError:
    HAS_FLYDSL = False

if HAS_FLYDSL:
    from torch._inductor.codegen.flydsl import flydsl_utils
    from torch._inductor.codegen.flydsl.flydsl_kernel import FlyDSLTemplateKernel


class TestFlyDSLTemplate(TestCase):
    def test_gen_imports(self):
        if not HAS_FLYDSL:
            self.skipTest("requires flydsl")

        kernel = FlyDSLTemplateKernel(
            kernel_name="test_kernel",
            input_nodes=[],
            output_node=None,
        )

        imports = kernel.gen_imports()

        self.assertIn("import torch", imports)
        self.assertIn("import flydsl.compiler as flyc", imports)
        self.assertIn("import flydsl.expr as fx", imports)

    @unittest.skipUnless(HAS_FLYDSL, "requires flydsl")
    @unittest.skipUnless(torch.cuda.is_available(), "CUDA/ROCm not available")
    @unittest.skipIf(torch.version.hip is None, "requires ROCm")
    @torch._inductor.config.patch(
        max_autotune_gemm=True,
        max_autotune_gemm_backends="FLYDSL",
    )
    def test_flydsl_hgemm_transposed_rhs_e2e(self):
        from torch._inductor.utils import run_and_get_code

        if not flydsl_utils.runtime_available():
            self.skipTest("FlyDSL runtime unavailable")

        def fn(a, b):
            return torch.mm(a, b.t())

        for dtype in (torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype):
                a = torch.randn(32, 128, device="cuda", dtype=dtype)
                b = torch.randn(128, 128, device="cuda", dtype=dtype)

                compiled_fn = torch.compile(fn, backend="inductor")
                result, (code,) = run_and_get_code(compiled_fn, a, b)

                self.assertIn("async_compile.flydsl", code)
                self.assertIn("_hgemm_splitk_mm", code)
                self.assertIn("TILE_M: fx.Constexpr", code)
                self.assertIn("STAGES: fx.Constexpr", code)
                self.assertIn("BLOCK_N_WARPS: fx.Constexpr", code)
                self.assertIn("BLOCK_K_WARPS: fx.Constexpr", code)
                self.assertTrue(torch.allclose(result, fn(a, b), atol=3e-2, rtol=3e-2))

    @unittest.skipUnless(HAS_FLYDSL, "requires flydsl")
    @unittest.skipUnless(torch.cuda.is_available(), "CUDA/ROCm not available")
    @unittest.skipIf(torch.version.hip is None, "requires ROCm")
    @torch._inductor.config.patch(
        max_autotune_gemm=True,
        max_autotune_gemm_backends="FLYDSL",
    )
    def test_flydsl_flex_gemm_epilogue_e2e(self):
        from torch._higher_order_ops.flex_gemm import flex_gemm
        from torch._inductor.utils import run_and_get_code

        if not flydsl_utils.runtime_available():
            self.skipTest("FlyDSL runtime unavailable")

        def epilogue(acc):
            return acc + 1

        def fn(a, b):
            return flex_gemm(
                torch.mm,
                (a, b),
                epilogue,
                kernel_options={"backend": "FLYDSL"},
            )

        a = torch.randn(32, 128, device="cuda", dtype=torch.float16) * 0.01
        b = torch.randn(128, 128, device="cuda", dtype=torch.float16) * 0.01

        compiled_fn = torch.compile(fn, backend="inductor", fullgraph=True)
        result, (code,) = run_and_get_code(compiled_fn, a, b)

        self.assertIn("async_compile.flydsl", code)
        self.assertIn("HAS_EPILOGUE: fx.Constexpr = True", code)
        self.assertIn("EPILOGUE_FN", code)
        self.assertTrue(torch.allclose(result, fn(a, b), atol=3e-2, rtol=3e-2))

    @unittest.skipUnless(HAS_FLYDSL, "requires flydsl")
    @unittest.skipUnless(torch.cuda.is_available(), "CUDA/ROCm not available")
    @unittest.skipIf(torch.version.hip is None, "requires ROCm")
    @torch._inductor.config.patch(
        max_autotune_gemm=True,
        max_autotune_gemm_backends="FLYDSL",
    )
    def test_flydsl_flex_gemm_captured_row_epilogue_e2e(self):
        from torch._higher_order_ops.flex_gemm import flex_gemm
        from torch._inductor.utils import run_and_get_code

        if not flydsl_utils.runtime_available():
            self.skipTest("FlyDSL runtime unavailable")

        def epilogue(acc, row_scale):
            return acc * row_scale + 1

        def fn(a, b, row_scale):
            return flex_gemm(
                torch.mm,
                (a, b),
                lambda acc: epilogue(acc, row_scale),
                kernel_options={"backend": "FLYDSL"},
            )

        a = torch.randn(32, 128, device="cuda", dtype=torch.float16) * 0.01
        b = torch.randn(128, 128, device="cuda", dtype=torch.float16) * 0.01
        row_scale = torch.randn(1, 128, device="cuda", dtype=torch.float16) * 0.01 + 1

        compiled_fn = torch.compile(fn, backend="inductor", fullgraph=True)
        result, (code,) = run_and_get_code(compiled_fn, a, b, row_scale)

        self.assertIn("async_compile.flydsl", code)
        self.assertIn("HAS_EPILOGUE: fx.Constexpr = True", code)
        self.assertIn("EPILOGUE_ARG_KINDS: fx.Constexpr = ('row',)", code)
        self.assertIn("EPILOGUE_ARG_DTYPES: fx.Constexpr = ('f16',)", code)
        self.assertTrue(torch.allclose(result, fn(a, b, row_scale), atol=3e-2, rtol=3e-2))

    @unittest.skipUnless(HAS_FLYDSL, "requires flydsl")
    @unittest.skipUnless(torch.cuda.is_available(), "CUDA/ROCm not available")
    @unittest.skipIf(torch.version.hip is None, "requires ROCm")
    @torch._inductor.config.patch(
        max_autotune_gemm=True,
        max_autotune_gemm_backends="FLYDSL",
    )
    def test_flydsl_flex_gemm_math_epilogue_e2e(self):
        from torch._higher_order_ops.flex_gemm import flex_gemm
        from torch._inductor.utils import run_and_get_code

        if not flydsl_utils.runtime_available():
            self.skipTest("FlyDSL runtime unavailable")

        def epilogue(acc):
            return torch.rsqrt(torch.exp2(acc))

        def fn(a, b):
            return flex_gemm(
                torch.mm,
                (a, b),
                epilogue,
                kernel_options={"backend": "FLYDSL"},
            )

        a = torch.randn(32, 128, device="cuda", dtype=torch.float16) * 0.01
        b = torch.randn(128, 128, device="cuda", dtype=torch.float16) * 0.01

        compiled_fn = torch.compile(fn, backend="inductor", fullgraph=True)
        result, (code,) = run_and_get_code(compiled_fn, a, b)

        self.assertIn("async_compile.flydsl", code)
        self.assertIn("fx.exp2", code)
        self.assertIn("fx.rsqrt", code)
        self.assertTrue(torch.allclose(result, fn(a, b), atol=3e-2, rtol=3e-2))

    @unittest.skipUnless(HAS_FLYDSL, "requires flydsl")
    @unittest.skipUnless(torch.cuda.is_available(), "CUDA/ROCm not available")
    @unittest.skipIf(torch.version.hip is None, "requires ROCm")
    @torch._inductor.config.patch(
        max_autotune_gemm=True,
        max_autotune_gemm_backends="FLYDSL",
    )
    def test_flydsl_flex_gemm_aux_output_e2e(self):
        from torch._higher_order_ops.flex_gemm import flex_gemm
        from torch._inductor.utils import run_and_get_code

        if not flydsl_utils.runtime_available():
            self.skipTest("FlyDSL runtime unavailable")

        def epilogue(acc):
            return acc + 1, acc * 2

        def fn(a, b):
            return flex_gemm(
                torch.mm,
                (a, b),
                epilogue,
                kernel_options={"backend": "FLYDSL"},
            )

        a = torch.randn(32, 128, device="cuda", dtype=torch.float16) * 0.01
        b = torch.randn(128, 128, device="cuda", dtype=torch.float16) * 0.01

        compiled_fn = torch.compile(fn, backend="inductor", fullgraph=True)
        (main_out, aux_out), (code,) = run_and_get_code(compiled_fn, a, b)
        expected_main, expected_aux = fn(a, b)

        self.assertIn("async_compile.flydsl", code)
        self.assertIn("RETURNS_AUX: fx.Constexpr = True", code)
        self.assertTrue(torch.allclose(main_out, expected_main, atol=3e-2, rtol=3e-2))
        self.assertTrue(torch.allclose(aux_out, expected_aux, atol=3e-2, rtol=3e-2))


if __name__ == "__main__":
    from torch._inductor.test_case import run_tests

    run_tests()
