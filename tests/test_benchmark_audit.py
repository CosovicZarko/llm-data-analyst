import unittest
import pandas as pd
from benchmark_audit_scoring import UNORDERED_CASE_IDS, tied_group_top, has_rank_ties


class AuditScoringTests(unittest.TestCase):
    def setUp(self):
        self.gold = pd.DataFrame({"category": ["Export", "Login", "Payment"], "count": [12, 12, 12]})

    def test_equivalent_tie_top_one(self):
        self.assertTrue(tied_group_top(pd.DataFrame({"cat": ["Payment"], "n": [12]}), self.gold, 1, ascending=False))

    def test_rejects_wrong_value(self):
        self.assertFalse(tied_group_top(pd.DataFrame({"cat": ["Payment"], "n": [11]}), self.gold, 1, ascending=False))

    def test_rejects_non_group(self):
        self.assertFalse(tied_group_top(pd.DataFrame({"cat": ["Random"], "n": [12]}), self.gold, 1, ascending=False))

    def test_rejects_multiple_rows_for_top_one(self):
        self.assertFalse(tied_group_top(self.gold, self.gold, 1, ascending=False))

    def test_rejects_wrong_ranking_order(self):
        ranked = pd.DataFrame({"key": ["A", "B", "C"], "num": [9, 8, 7]})
        candidate = pd.DataFrame({"key": ["B", "A"], "num": [8, 9]})
        self.assertFalse(tied_group_top(candidate, ranked, 2, ascending=False))

    def test_ambiguous_topk_boundary(self):
        ranked = pd.DataFrame({"key": ["A", "B", "C", "D"], "num": [9, 8, 8, 7]})
        candidate = pd.DataFrame({"key": ["A", "C"], "num": [9, 8]})
        self.assertTrue(tied_group_top(candidate, ranked, 2, ascending=False))
        self.assertFalse(tied_group_top(pd.DataFrame({"key": ["C", "D"], "num": [8, 7]}), ranked, 2, ascending=False))

    def test_correct_topk_ties_swapped_columns(self):
        candidate = pd.DataFrame({"n": [12, 12], "category": ["Login", "Payment"]})
        self.assertTrue(tied_group_top(candidate, self.gold, 2, ascending=False))

    def test_detect_ties(self):
        self.assertTrue(has_rank_ties(self.gold, 1))

    def test_unrequested_order_overrides(self):
        self.assertIn("sta-14", UNORDERED_CASE_IDS)
        self.assertIn("sup-13", UNORDERED_CASE_IDS)
        self.assertNotIn("ret-12", UNORDERED_CASE_IDS)


if __name__ == "__main__":
    unittest.main()
