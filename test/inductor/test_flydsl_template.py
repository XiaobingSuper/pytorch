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
        max_autotune=True,
        max_autotune_gemm_backends="FLYDSL",
        autotune_fallback_to_aten=False,
    )
    def test_flydsl_grouped_mm_e2e(self):
        import torch.nn.functional as F

        from torch._inductor.utils import run_and_get_code

        if not flydsl_utils.runtime_available():
            self.skipTest("FlyDSL runtime unavailable")

        group_size, k, n = 4, 128, 128
        m_sizes = torch.tensor([16, 32, 48, 64], dtype=torch.int32, device="cuda")
        offs = torch.cumsum(m_sizes, dim=0).to(torch.int32)
        total_m = int(m_sizes.sum().item())

        def fn(a, b, offsets):
            return F.grouped_mm(a, b, offs=offsets)

        dtype = torch.bfloat16
        a = torch.randn(total_m, k, device="cuda", dtype=dtype) * 0.1
        b = torch.randn(group_size, k, n, device="cuda", dtype=dtype) * 0.01

        compiled_fn = torch.compile(fn, backend="inductor", fullgraph=True)
        result, (code,) = run_and_get_code(compiled_fn, a, b, offs)

        self.assertIn("async_compile.flydsl", code)
        self.assertIn("_flydsl_grouped_mm", code)
        self.assertTrue(torch.allclose(result, fn(a, b, offs), atol=3e-2, rtol=3e-2))

    @unittest.skipUnless(HAS_FLYDSL, "requires flydsl")
    @unittest.skipUnless(torch.cuda.is_available(), "CUDA/ROCm not available")
    @unittest.skipIf(torch.version.hip is None, "requires ROCm")
    def test_flydsl_grouped_tile_map(self):
        from torch._inductor.kernel.vendored_templates.flydsl.kernels import (
            compile_grouped_tile_map_kernel,
        )
        from torch._inductor.kernel.vendored_templates.flydsl.kernels.tensor_shim import (
            _run_compiled,
        )

        if not flydsl_utils.runtime_available():
            self.skipTest("FlyDSL runtime unavailable")

        tile_m = 16
        n_blocks = 3
        m_sizes = torch.tensor([0, 1, 17, 32], dtype=torch.int32, device="cuda")
        offs = torch.cumsum(m_sizes, dim=0).to(torch.int32)
        stream = torch.cuda.current_stream().cuda_stream

        expected = []
        for group_idx, m_g in enumerate(m_sizes.cpu().tolist()):
            for tile_m_idx in range((m_g + tile_m - 1) // tile_m):
                for tile_n_idx in range(n_blocks):
                    expected.append((group_idx, tile_m_idx, tile_n_idx))

        max_m_tiles_upper = (int(m_sizes.sum().item()) + tile_m - 1) // tile_m
        capacity = max(1, int(m_sizes.numel()) * max_m_tiles_upper * n_blocks)
        tile_groups = torch.empty((capacity,), dtype=torch.int32, device="cuda")
        tile_ms = torch.empty((capacity,), dtype=torch.int32, device="cuda")
        tile_ns = torch.empty((capacity,), dtype=torch.int32, device="cuda")
        total_tiles = torch.zeros((1,), dtype=torch.int32, device="cuda")

        exe = compile_grouped_tile_map_kernel(tile_m, int(m_sizes.numel()))
        _run_compiled(
            exe,
            offs,
            tile_groups,
            tile_ms,
            tile_ns,
            total_tiles,
            n_blocks,
            max_m_tiles_upper,
            capacity,
            stream,
        )
        torch.cuda.synchronize()

        actual_total = int(total_tiles.item())
        self.assertEqual(actual_total, len(expected))
        actual = list(
            zip(
                tile_groups[:actual_total].cpu().tolist(),
                tile_ms[:actual_total].cpu().tolist(),
                tile_ns[:actual_total].cpu().tolist(),
            )
        )
        self.assertEqual(actual, expected)


if __name__ == "__main__":
    from torch._inductor.test_case import run_tests

    run_tests()
