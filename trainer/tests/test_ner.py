"""Unit tests for the token-classification (NER) trainer pure functions.

Tests span-to-token alignment (with offset_mapping), BIO decoding,
sliding windows, and strict entity metrics.
"""

from __future__ import annotations

import unittest
from trainer.tasks.ner import (
    build_label_map,
    decode_bio_spans,
    entity_metrics,
    spans_to_token_labels,
    windows,
)


class TestNERAlignment(unittest.TestCase):
    def test_span_to_token_labels_simple(self):
        text = "Acme Corp paid $100."
        # Character offsets simulated from a fast tokenizer:
        # "Acme" (0..4), "Corp" (5..9), "paid" (10..14), "$" (15..16), "100" (16..19), "." (19..20)
        offsets = [(0, 4), (5, 9), (10, 14), (15, 16), (16, 19), (19, 20)]
        spans = [
            {"start": 0, "end": 9, "label": "ORG"},       # "Acme Corp" spans 2 tokens
            {"start": 15, "end": 19, "label": "MONEY"},   # "$100" spans 2 tokens
        ]
        label_map = {
            "O": 0,
            "B-ORG": 1, "I-ORG": 2,
            "B-MONEY": 3, "I-MONEY": 4,
        }

        labels = spans_to_token_labels(text, spans, offsets, label_map, "BIO")
        self.assertEqual(labels, [1, 2, 0, 3, 4, 0])

    def test_decode_bio_spans(self):
        id_to_label = {
            0: "O",
            1: "B-ORG", 2: "I-ORG",
            3: "B-MONEY", 4: "I-MONEY",
        }
        offsets = [(0, 4), (5, 9), (10, 14), (15, 16), (16, 19), (19, 20)]
        token_labels = [1, 2, 0, 3, 4, 0]

        spans = decode_bio_spans(token_labels, offsets, id_to_label)
        self.assertEqual(len(spans), 2)
        self.assertEqual(spans[0]["start"], 0)
        self.assertEqual(spans[0]["end"], 9)
        self.assertEqual(spans[0]["label"], "ORG")

        self.assertEqual(spans[1]["start"], 15)
        self.assertEqual(spans[1]["end"], 19)
        self.assertEqual(spans[1]["label"], "MONEY")

    def test_windows(self):
        # 10 tokens, window=4, stride=1 (step=3)
        # Windows: [0..4], [3..7], [6..10]
        w = windows(10, 4, 1)
        self.assertEqual(w, [(0, 4), (3, 7), (6, 10)])

        # Single window when num_tokens <= max_length
        self.assertEqual(windows(5, 10, 2), [(0, 5)])

    def test_entity_metrics(self):
        gold = [
            [{"start": 0, "end": 4, "label": "ORG"}, {"start": 10, "end": 15, "label": "LOC"}],
        ]
        # Exact match for ORG, missing LOC, extra false positive
        preds = [
            [{"start": 0, "end": 4, "label": "ORG"}, {"start": 20, "end": 25, "label": "MISC"}],
        ]

        m = entity_metrics(gold, preds)
        # TP=1 (ORG), FP=1 (MISC), FN=1 (LOC)
        # Precision = 1/2 = 0.5, Recall = 1/2 = 0.5, F1 = 0.5
        self.assertEqual(m["micro_f1"], 0.5)
        self.assertEqual(m["precision"], 0.5)
        self.assertEqual(m["recall"], 0.5)
        self.assertIn("ORG", m["per_entity"])
        self.assertEqual(m["per_entity"]["ORG"]["f1"], 1.0)


if __name__ == "__main__":
    unittest.main()