"""CPU checks for reporting math and bounded benchmark configurations."""

import unittest

from attention_kernel_lab.benchmark import BenchmarkConfig, logical_flops, percentile


class BenchmarkHelpersTests(unittest.TestCase):
    def test_percentile_interpolation(self):
        self.assertAlmostEqual(percentile([4.0, 1.0, 3.0, 2.0], 0.2), 1.6)
        self.assertEqual(percentile([8.0], 0.8), 8.0)
        with self.assertRaises(ValueError):
            percentile([], 0.5)
        with self.assertRaises(ValueError):
            percentile([1.0], 1.1)

    def test_logical_flops_match_explicit_rectangular_mask(self):
        for nq, nk in ((1, 7), (7, 1), (3, 9), (9, 3), (8, 8)):
            with self.subTest(nq=nq, nk=nk):
                allowed = sum(j <= i for i in range(nq) for j in range(nk))
                self.assertEqual(logical_flops(2, 3, nq, nk, 32, True), 4 * 2 * 3 * allowed * 32)
                self.assertEqual(logical_flops(2, 3, nq, nk, 32, False), 4 * 2 * 3 * nq * nk * 32)

    def test_config_rejects_unsafe_reference_and_invalid_runs(self):
        BenchmarkConfig().validate()
        for kwargs in (
            {"nq": 0}, {"warmup": 0}, {"samples": 0}, {"dtype": "fp32"},
            {"scale": float("nan")}, {"scale": 0}, {"scale": -0.5},
            {"scale": True}, {"scale": "1"}, {"block_m": 8},
            {"block_n": 128}, {"nq": 65536, "nk": 65536},
            {"batch": True}, {"heads": 1.5}, {"warmup": 2.5},
            {"causal": 1}, {"include_eager": 0}, {"seed": 2**64},
            {"batch": 65536, "heads": 1, "nq": 1, "nk": 1},
            {"batch": 1, "heads": 1, "nq": 1, "nk": 2_000_000},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                BenchmarkConfig(**kwargs).validate()


if __name__ == "__main__":
    unittest.main()
