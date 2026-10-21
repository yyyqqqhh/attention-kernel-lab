import math
import random
import unittest
from decimal import Decimal, localcontext

from attention_kernel_lab.online import dense_attention, online_attention


def decimal_oracle(q, k, v, causal, scale):
    with localcontext() as context:
        context.prec = 70
        cast = lambda x: Decimal(str(x))
        result = []
        for i, row in enumerate(q):
            end = min(i + 1, len(k)) if causal else len(k)
            scores = [sum(cast(a) * cast(b) for a, b in zip(row, key)) * cast(scale)
                      for key in k[:end]]
            maximum = max(scores)
            weights = [(score - maximum).exp() for score in scores]
            denominator = sum(weights)
            result.append([float(sum(weights[j] * cast(v[j][d]) for j in range(end))
                                 / denominator) for d in range(len(row))])
        return result


class OnlineAttentionTests(unittest.TestCase):
    def assertMatrixClose(self, actual, expected, tolerance=2e-12):
        self.assertEqual(len(actual), len(expected))
        for row, reference in zip(actual, expected):
            self.assertEqual(len(row), len(reference))
            for value, target in zip(row, reference):
                self.assertTrue(math.isclose(value, target, rel_tol=tolerance, abs_tol=tolerance),
                                (value, target))

    def test_against_independent_decimal_oracle(self):
        rng = random.Random(734)
        for nq, nk, d in ((1, 1, 1), (3, 7, 4), (9, 3, 5), (7, 11, 8)):
            q = [[rng.uniform(-3, 3) for _ in range(d)] for _ in range(nq)]
            k = [[rng.uniform(-3, 3) for _ in range(d)] for _ in range(nk)]
            v = [[rng.uniform(-2, 2) for _ in range(d)] for _ in range(nk)]
            for causal in (False, True):
                expected = decimal_oracle(q, k, v, causal, 0.37)
                self.assertMatrixClose(dense_attention(q, k, v, causal=causal, scale=0.37), expected)
                for block_size in (1, 2, 3, 8, 32):
                    with self.subTest(shape=(nq, nk, d), causal=causal, block=block_size):
                        self.assertMatrixClose(online_attention(q, k, v, causal=causal,
                                                              scale=0.37, block_size=block_size), expected)

    def test_large_logits_and_running_maximum_changes(self):
        q, k, v = [[1.0]], [[-1000.0], [0.0], [1000.0]], [[1.0], [2.0], [3.0]]
        self.assertEqual(online_attention(q, k, v, block_size=1), [[3.0]])
        self.assertEqual(dense_attention(q, k, v), [[3.0]])

    def test_upper_left_causal_excludes_future_values(self):
        q = [[1.0], [1.0]]
        k = [[0.0], [0.0], [0.0], [0.0]]
        v = [[2.0], [4.0], [1e9], [-1e9]]
        self.assertEqual(online_attention(q, k, v, causal=True, block_size=3), [[2.0], [3.0]])

    def test_uniform_scores_average_values(self):
        q, k = [[0.0, 0.0]], [[5.0, -3.0], [9.0, 2.0], [-7.0, 8.0]]
        v = [[1.0, 2.0], [4.0, 5.0], [7.0, 8.0]]
        self.assertMatrixClose(online_attention(q, k, v), [[4.0, 5.0]])

    def test_key_translation_leaves_probabilities_unchanged(self):
        q = [[0.5, -0.75], [2.0, 0.0]]
        k, v = [[1.0, 2.0], [-3.0, 4.0]], [[1.0, 0.0], [0.0, 1.0]]
        translated = [[a + 128.0, b - 64.0] for a, b in k]
        self.assertMatrixClose(online_attention(q, translated, v, block_size=1),
                               online_attention(q, k, v, block_size=1))

    def test_repeated_keys_split_weight_without_changing_result(self):
        q, k, v = [[0.3, -0.4]], [[1.0, 2.0], [3.0, 4.0]], [[-2.0, 1.0], [1.0, 2.0]]
        self.assertMatrixClose(online_attention(q, k + k, v + v, block_size=3),
                               dense_attention(q, k, v))

    def test_value_translation(self):
        q, k, v = [[2.0]], [[-1.0], [1.0]], [[3.0], [5.0]]
        shifted = online_attention(q, k, [[13.0], [15.0]], block_size=1)
        self.assertMatrixClose(shifted, [[online_attention(q, k, v)[0][0] + 10.0]])

    def test_reject_invalid_inputs(self):
        valid = [[1.0, 2.0]]
        for function in (dense_attention, online_attention):
            for bad in ([], [[]], [[1.0], [1.0, 2.0]], [[math.inf, 1.0]], [[math.nan, 1.0]]):
                with self.subTest(function=function.__name__, bad=bad):
                    with self.assertRaises(ValueError):
                        function(bad, valid, valid)
            for bad in ([[True, 1.0]], [["1", 1.0]]):
                with self.assertRaises(TypeError):
                    function(bad, valid, valid)
            with self.assertRaises(ValueError):
                function(valid, valid, valid + valid)
            with self.assertRaises(ValueError):
                function(valid, [[1.0]], valid)
            with self.assertRaises(TypeError):
                function(valid, valid, valid, causal=1)
            for scale in (0.0, -1.0, math.inf, math.nan):
                with self.assertRaises(ValueError):
                    function(valid, valid, valid, scale=scale)
            with self.assertRaises(TypeError):
                function(valid, valid, valid, scale=True)

    def test_reject_invalid_block_size(self):
        for block in (0, -1, 1.5, True):
            with self.assertRaises(ValueError):
                online_attention([[1.0]], [[1.0]], [[1.0]], block_size=block)

    def test_overflow_fails_explicitly(self):
        for function in (dense_attention, online_attention):
            with self.assertRaisesRegex(ValueError, "overflowed"):
                function([[1e200]], [[1e200]], [[1.0]])


if __name__ == "__main__":
    unittest.main()
