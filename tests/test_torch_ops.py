"""Tensor contract and CPU oracle tests; no GPU is required."""

import importlib
import importlib.util
import math
import unittest
from unittest.mock import patch

from attention_kernel_lab import torch_ops

HAS_TORCH = importlib.util.find_spec("torch") is not None
torch = importlib.import_module("torch") if HAS_TORCH else None


@unittest.skipUnless(HAS_TORCH, "optional PyTorch dependency is not installed")
class TorchReferenceTests(unittest.TestCase):
    def tensors(self, nq=5, nk=7, d=16, dtype=None):
        generator = torch.Generator().manual_seed(31)
        dtype = dtype or torch.float32
        return (
            torch.randn(2, 3, nq, d, generator=generator, dtype=dtype),
            torch.randn(2, 3, nk, d, generator=generator, dtype=dtype),
            torch.randn(2, 3, nk, d, generator=generator, dtype=dtype),
        )

    def test_oracle_matches_sdpa(self):
        for nq, nk in ((1, 1), (5, 7), (9, 3), (33, 35)):
            for causal in (False, True):
                with self.subTest(nq=nq, nk=nk, causal=causal):
                    q, k, v = self.tensors(nq=nq, nk=nk)
                    actual = torch_ops.reference_attention(q, k, v, causal=causal)
                    expected = torch.nn.functional.scaled_dot_product_attention(
                        q, k, v, is_causal=causal, dropout_p=0.0
                    )
                    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)

    def test_upper_left_causal_alignment(self):
        q = torch.zeros(1, 1, 3, 2, dtype=torch.float64)
        k = torch.zeros(1, 1, 5, 2, dtype=torch.float64)
        v = torch.tensor([[[[2., 4.], [4., 6.], [6., 8.], [50., 60.], [70., 80.]]]], dtype=torch.float64)
        actual = torch_ops.reference_attention(q, k, v, causal=True)
        expected = torch.tensor([[[[2., 4.], [3., 5.], [4., 6.]]]], dtype=torch.float64)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)

    def test_dtype_and_custom_scale(self):
        for dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
            with self.subTest(dtype=dtype):
                q, k, v = self.tensors(dtype=dtype)
                actual = torch_ops.reference_attention(q, k, v, scale=0.25)
                accum_dtype = torch.float64 if dtype == torch.float64 else torch.float32
                expected = torch.softmax(q.to(accum_dtype) @ k.to(accum_dtype).transpose(-2, -1) * 0.25, -1) @ v.to(accum_dtype)
                self.assertEqual(actual.dtype, dtype)
                torch.testing.assert_close(actual, expected.to(dtype), atol=0, rtol=0)

    def test_stable_large_scores(self):
        q, k, v = self.tensors(dtype=torch.float64)
        actual = torch_ops.reference_attention(q * 100, k * 100, v, causal=True)
        self.assertTrue(torch.isfinite(actual).all().item())
        self.assertLessEqual(actual.abs().max().item(), v.abs().max().item() + 1e-12)

    def test_reference_preserves_autograd(self):
        q, k, v = self.tensors()
        q.requires_grad_()
        torch_ops.attention(q, k, v, backend="reference").sum().backward()
        self.assertIsNotNone(q.grad)
        self.assertTrue(torch.isfinite(q.grad).all().item())

    def test_auto_reference_on_cpu(self):
        q, k, v = self.tensors()
        self.assertIn("CUDA", torch_ops.triton_unavailable_reason(q, k, v))
        torch.testing.assert_close(torch_ops.attention(q, k, v), torch_ops.reference_attention(q, k, v))
        with self.assertRaisesRegex(RuntimeError, "CUDA"):
            torch_ops.attention(q, k, v, backend="triton")

    def test_auto_does_not_hide_eligible_launch_failure(self):
        q, k, v = self.tensors()
        class FailingKernel:
            @staticmethod
            def launch_attention(*args):
                raise RuntimeError("sentinel compile failure")
        original_import = importlib.import_module
        def import_kernel(name):
            return FailingKernel if name == "attention_kernel_lab.triton_kernel" else original_import(name)
        with patch.object(torch_ops, "_triton_reason", return_value=None), patch.object(torch_ops.importlib, "import_module", side_effect=import_kernel):
            with self.assertRaisesRegex(RuntimeError, "sentinel compile failure"):
                torch_ops.attention(q, k, v, backend="auto")

    def test_rejects_invalid_tensor_contracts(self):
        q, k, v = self.tensors()
        cases = (
            (q[0], k, v, ValueError),
            (q, k[:, :2].contiguous(), v, ValueError),
            (q, k[:, :, :-1].contiguous(), v, ValueError),
            (q, k, v[:, :, :, :-1].contiguous(), ValueError),
            (q[:, :, :0], k, v, ValueError),
            (q, k.double(), v, TypeError),
            (q.long(), k.long(), v.long(), TypeError),
            (q.transpose(-2, -1), k, v, ValueError),
            (q, k, torch.empty(v.shape, device="meta"), ValueError),
            (None, k, v, TypeError),
        )
        for index, (bad_q, bad_k, bad_v, error) in enumerate(cases):
            with self.subTest(case=index), self.assertRaises(error):
                torch_ops.reference_attention(bad_q, bad_k, bad_v)

    def test_rejects_noncontiguous_equal_shape(self):
        q, k, v = self.tensors(nq=16, nk=16, d=16)
        with self.assertRaisesRegex(ValueError, "contiguous"):
            torch_ops.reference_attention(q.transpose(-2, -1), k, v)

    def test_rejects_bad_options(self):
        q, k, v = self.tensors()
        for scale in (0, -1, math.inf, -math.inf, math.nan):
            with self.subTest(scale=scale), self.assertRaises(ValueError):
                torch_ops.reference_attention(q, k, v, scale=scale)
        for scale in (True, "0.5", 1j):
            with self.subTest(scale=scale), self.assertRaises(TypeError):
                torch_ops.reference_attention(q, k, v, scale=scale)
        for options in ({"backend": "other"}, {"block_m": 7}, {"block_n": 128}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                torch_ops.attention(q, k, v, **options)
        for options in ({"causal": 1}, {"block_m": True}, {"block_n": 32.0}):
            with self.subTest(options=options), self.assertRaises(TypeError):
                torch_ops.attention(q, k, v, **options)


if __name__ == "__main__":
    unittest.main()
