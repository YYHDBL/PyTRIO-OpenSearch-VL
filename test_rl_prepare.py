import math
import io
import unittest

import numpy as np
import pytrio as trio
from PIL import Image


class RLPreparationTest(unittest.TestCase):
    def test_multiturn_uses_original_prefix_and_masks_fatal_steps(self):
        from rl_prepare import build_trajectory_datums

        first = trio.ModelInput.from_ints([11, 12])
        second = trio.ModelInput.from_ints([11, 12, 21, 22, 31, 32])
        steps = [{"prompt": first, "tokens": [21, 22], "logprobs": [-0.5, -0.6]},
                 {"prompt": second, "tokens": [41, 42, 43], "logprobs": [-0.7, -0.8, -0.9]}]
        datums = build_trajectory_datums(steps, 0.4, fatal_start=1)
        self.assertEqual(len(datums), 1)
        self.assertEqual(datums[0].loss_fn_inputs["target_tokens"].data, [0, 21, 22])
        self.assertEqual(len(build_trajectory_datums(steps, -0.2, fatal_start=None)), 2)
        self.assertEqual(build_trajectory_datums(steps, 0.4, fatal_start=0), [])
        steps[1]["prompt"] = trio.ModelInput.from_ints([11, 12, 21, 99, 31, 32])
        with self.assertRaises(ValueError):
            build_trajectory_datums(steps, 0.4, fatal_start=None)

    def test_multiturn_preserves_image_chunk_identity(self):
        from rl_prepare import build_trajectory_datums

        image = io.BytesIO()
        Image.new("RGB", (8, 8), "red").save(image, format="PNG")
        original = trio.ImageChunk(data=image.getvalue(), format="png", expected_tokens=3)
        first = trio.ModelInput(chunks=[trio.types.EncodedTextChunk(tokens=[11]), original])
        second = trio.ModelInput(chunks=[trio.types.EncodedTextChunk(tokens=[11]), original,
                                         trio.types.EncodedTextChunk(tokens=[21, 22, 31])])
        steps = [{"prompt": first, "tokens": [21, 22], "logprobs": [-0.5, -0.6]},
                 {"prompt": second, "tokens": [41], "logprobs": [-0.7]}]
        datums = build_trajectory_datums(steps, 0.4, None)
        self.assertEqual(len(datums), 2)
        self.assertEqual(datums[1].loss_fn_inputs["target_tokens"].data, [0] * 6 + [41])
        different = io.BytesIO()
        Image.new("RGB", (8, 8), "blue").save(different, format="PNG")
        second.chunks[1] = trio.ImageChunk(data=different.getvalue(), format="png", expected_tokens=3)
        with self.assertRaises(ValueError):
            build_trajectory_datums(steps, 0.4, None)

    def test_rloo_uses_complete_group_before_fatal_clamp(self):
        from rl_prepare import rloo_advantages

        actual = rloo_advantages([0.0, 0.2, 0.6, 1.0], [True, False, False, True])
        np.testing.assert_allclose(actual, [0.0, -0.3333333333, 0.2, 0.7333333333])
        np.testing.assert_allclose(rloo_advantages([0.5] * 4, [False] * 4), [0] * 4)
        with self.assertRaises(ValueError):
            rloo_advantages([0.0, 1.0], [False, False])
        with self.assertRaises(ValueError):
            rloo_advantages([0.0, 0.2, 0.6, math.nan], [False] * 4)

    def test_multimodal_targets_and_masks_preserve_token_alignment(self):
        from rl_prepare import build_datum

        image = io.BytesIO()
        Image.new("RGB", (8, 8)).save(image, format="PNG")
        prompt = trio.ModelInput(chunks=[trio.types.EncodedTextChunk(tokens=[11, 12]),
                                        trio.ImageChunk(data=image.getvalue(), format="png", expected_tokens=3),
                                        trio.types.EncodedTextChunk(tokens=[13])])
        datum = build_datum(prompt, [21, 22, 23], [-0.5, -0.6, -0.7], 0.4)
        self.assertEqual(len(datum.model_input), 8)
        self.assertEqual(datum.loss_fn_inputs["target_tokens"].data, [0, 0, 0, 0, 0, 21, 22, 23])
        np.testing.assert_allclose(datum.loss_fn_inputs["logprobs"].data, [0, 0, 0, 0, 0, -0.5, -0.6, -0.7])
        np.testing.assert_allclose(datum.loss_fn_inputs["advantages"].data, [0, 0, 0, 0, 0, 0.4, 0.4, 0.4])
        with self.assertRaises(ValueError):
            build_datum(prompt, [21, 22], [-0.5], 1.0)
        with self.assertRaises(ValueError):
            build_datum(prompt, [21], [float("nan")], 1.0)
