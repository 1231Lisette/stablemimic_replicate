import unittest

try:
    import torch
except ModuleNotFoundError:
    torch = None

from stablemimic.eval import classify_fallen_orientation, matched_push_protocol


class EvaluationProtocolTests(unittest.TestCase):
    @unittest.skipIf(torch is None, "torch is not installed")
    def test_classify_fallen_orientation(self) -> None:
        labels = classify_fallen_orientation(torch.tensor([
            [-1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, -1.0],
        ]))
        self.assertEqual(labels.tolist(), [0, 1, 2, 3, 4])

    def test_matched_push_protocol(self) -> None:
        events = matched_push_protocol()
        self.assertEqual(len(events), 100)
        self.assertEqual({event.direction_xy for event in events}, {(1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0)})
        for direction in {event.direction_xy for event in events}:
            self.assertEqual(sum(event.direction_xy == direction for event in events), 25)
        self.assertAlmostEqual(min(event.force_newtons for event in events), 525.0)
        self.assertAlmostEqual(max(event.force_newtons for event in events), 575.0)
        self.assertTrue(all(event.duration_seconds == 0.2 for event in events))
