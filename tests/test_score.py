import json
import unittest

import score

GOOD = {"schema_version": "1.0", "domain": "rahmantraders.com", "notice_type": "expiry",
        "expires_on": "2027-01-14", "registrar": "Namewell", "amount_due": 13.99,
        "currency": "USD", "action_required": True}


def rec(raw, done="stop"):
    return {"id": "n01", "raw": raw, "done_reason": done}


class ScoreTests(unittest.TestCase):
    def test_perfect_output(self):
        s = score.score_record(rec(json.dumps(GOOD)))
        self.assertTrue(s["valid"])
        self.assertTrue(all(s["correct"].values()))

    def test_fence_forgiven_but_logged(self):
        s = score.score_record(rec("```json\n" + json.dumps(GOOD) + "\n```"))
        self.assertTrue(s["valid"] and s["fenced"])

    def test_prose_around_json_is_invalid(self):
        s = score.score_record(rec("Here you go: " + json.dumps(GOOD)))
        self.assertFalse(s["valid"])
        self.assertFalse(any(s["correct"].values()))

    def test_truncated_is_invalid_even_if_it_parses(self):
        self.assertFalse(score.score_record(rec(json.dumps(GOOD), done="length"))["valid"])

    def test_schema_violation_scores_zero(self):
        bad = {**GOOD, "notice_type": "reminder"}
        s = score.score_record(rec(json.dumps(bad)))
        self.assertFalse(s["valid"])
        self.assertFalse(s["correct"]["domain"])

    def test_field_rules(self):
        s = score.score_record(rec(json.dumps({**GOOD, "registrar": "  namewell ",
                                               "amount_due": 13.990001,
                                               "expires_on": "2027-01-15"})))
        self.assertTrue(s["correct"]["registrar"] and s["correct"]["amount_due"])
        self.assertFalse(s["correct"]["expires_on"])

    def test_refusal_detected(self):
        self.assertTrue(score.score_record(rec("I cannot help with that."))["refusal"])


if __name__ == "__main__":
    unittest.main()
