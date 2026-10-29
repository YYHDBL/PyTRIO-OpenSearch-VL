import unittest

import torch


class RLLossTest(unittest.TestCase):
    def test_token_clipping_and_trajectory_reduction_have_real_gradients(self):
        from rl_loss import make_ppo_loss

        current = torch.tensor([0., 0., 0.], requires_grad=True)
        meta = [{"start": 1, "old": [0., 0.], "advantage": 0.5, "normalizer": 2}]
        loss, _ = make_ppo_loss(meta, trajectories=1)([None], [current])
        self.assertAlmostEqual(loss.item(), -0.5)
        loss.backward()
        torch.testing.assert_close(current.grad, torch.tensor([0., -0.25, -0.25]))

    def test_microbatch_losses_sum_to_full_loss(self):
        from rl_loss import make_ppo_loss

        meta = [{"start": 0, "old": [0.], "advantage": a, "normalizer": 1} for a in [0.3, -0.4]]
        values = [torch.tensor([0.], requires_grad=True) for _ in meta]
        whole, _ = make_ppo_loss(meta, 2)([None, None], values)
        pieces = [make_ppo_loss([m], 2)([None], [v])[0] for m, v in zip(meta, values)]
        torch.testing.assert_close(sum(pieces), whole)

    def test_positive_and_negative_clipping_match_author_dual_clip(self):
        from rl_loss import make_ppo_loss

        positive = torch.tensor([1.0], requires_grad=True)
        negative = torch.tensor([2.0], requires_grad=True)
        p, _ = make_ppo_loss([{"start": 0, "old": [0.], "advantage": 1., "normalizer": 1}], 1)([None], [positive])
        n, _ = make_ppo_loss([{"start": 0, "old": [0.], "advantage": -1., "normalizer": 1}], 1)([None], [negative])
        self.assertAlmostEqual(p.item(), -1.28, places=5)
        self.assertAlmostEqual(n.item(), 3.0, places=5)
