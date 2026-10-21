"""Hardware tests: skip unsupported environments, fail real launch regressions."""

import importlib
import importlib.util
import sys
import unittest

from attention_kernel_lab.torch_ops import attention, reference_attention

if importlib.util.find_spec("pytest") is not None:
    import pytest
    pytestmark = pytest.mark.gpu

torch = importlib.import_module("torch") if importlib.util.find_spec("torch") is not None else None


def unavailable_reason():
    if torch is None:
        return "optional PyTorch dependency is not installed"
    if not sys.platform.startswith("linux"):
        return "GPU kernel targets Linux"
    if torch.version.hip is not None:
        return "ROCm is outside this project's target"
    if not torch.cuda.is_available():
        return "CUDA device is unavailable"
    if torch.cuda.get_device_capability()[0] < 8:
        return "NVIDIA compute capability 8.0+ is required"
    if importlib.util.find_spec("triton") is None:
        return "optional Triton dependency is not installed"
    return None


REASON = unavailable_reason()


@unittest.skipIf(REASON is not None, REASON or "")
class GPUAttentionTests(unittest.TestCase):
    def tensors(self, nq, nk, d, dtype, multiplier=1.0):
        generator = torch.Generator(device="cuda").manual_seed(2026)
        return tuple(
            torch.randn((2, 2, n, d), device="cuda", dtype=dtype, generator=generator) * multiplier
            for n in (nq, nk, nk)
        )

    def check_case(self, nq, nk, d, dtype, causal, block_m=32, block_n=32, scale=None):
        q, k, v = self.tensors(nq, nk, d, dtype)
        actual = attention(q, k, v, backend="triton", causal=causal, scale=scale, block_m=block_m, block_n=block_n)
        expected = reference_attention(q, k, v, causal=causal, scale=scale)
        torch.cuda.synchronize()
        self.assertEqual(actual.dtype, dtype)
        self.assertEqual(actual.shape, q.shape)
        self.assertTrue(torch.isfinite(actual).all().item())
        tolerance = 2e-2 if dtype == torch.bfloat16 else 3e-3
        torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)
        relative_l2 = (actual.float() - expected.float()).norm() / expected.float().norm().clamp_min(1e-12)
        self.assertLess(relative_l2.item(), tolerance)

    def test_irregular_rectangular_and_causal(self):
        for nq, nk in ((1, 1), (1, 37), (17, 5), (33, 65), (71, 33), (113, 203), (1023, 1024)):
            for causal in (False, True):
                with self.subTest(nq=nq, nk=nk, causal=causal):
                    self.check_case(nq, nk, 32, torch.float16, causal)

    def test_head_dimensions_and_dtypes(self):
        for d in (16, 32, 64, 128):
            for dtype in (torch.float16, torch.bfloat16):
                for causal in (False, True):
                    with self.subTest(d=d, dtype=dtype, causal=causal):
                        self.check_case(67, 79, d, dtype, causal)

    def test_tile_variants(self):
        for block_m, block_n in ((16, 16), (16, 64), (32, 32), (64, 16), (64, 64)):
            with self.subTest(block_m=block_m, block_n=block_n):
                self.check_case(65, 97, 64, torch.float16, True, block_m, block_n, 0.2)

    def test_large_logits_remain_finite(self):
        q, k, v = self.tensors(35, 37, 32, torch.float16)
        q, k = q * 8, k * 8
        actual = attention(q, k, v, backend="triton", causal=True)
        expected = reference_attention(q, k, v, causal=True)
        torch.cuda.synchronize()
        self.assertTrue(torch.isfinite(actual).all().item())
        torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)

    def test_gpu_contract_rejections(self):
        q, k, v = self.tensors(3, 5, 16, torch.float16)
        with self.assertRaisesRegex(RuntimeError, "inference-only"):
            attention(q.requires_grad_(), k, v, backend="triton")
        with self.assertRaisesRegex(RuntimeError, "float16 or bfloat16"):
            attention(q.detach().float(), k.float(), v.float(), backend="triton")
        for d in (24, 59, 111):
            with self.subTest(d=d):
                q, k, v = self.tensors(3, 5, d, torch.float16)
                with self.assertRaisesRegex(RuntimeError, "head_dim"):
                    attention(q, k, v, backend="triton")


if __name__ == "__main__":
    unittest.main()
