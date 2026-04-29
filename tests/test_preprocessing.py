"""Smoke tests for the preprocessing module."""

import unittest

from src.preprocessing import Vectorizer, clean_text


class TestPreprocessing(unittest.TestCase):
    def test_clean_text_removes_urls(self):
        out = clean_text("Click here https://bad.example.com to win!")
        self.assertNotIn("http", out)
        self.assertNotIn("example", out)  # filtered after URL strip + tokenization
        self.assertIn("click", out)

    def test_clean_text_lowercases(self):
        self.assertEqual(clean_text("HELLO World").split(), ["hello", "world"])

    def test_clean_text_handles_non_string(self):
        self.assertEqual(clean_text(None), "")
        self.assertEqual(clean_text(123), "")

    def test_vectorizer_round_trip(self):
        vec = Vectorizer(max_features=50)
        X_train = vec.fit_transform(["buy now", "team meeting at noon", "free money"])
        X_test = vec.transform(["urgent money offer"])
        self.assertEqual(X_train.shape[0], 3)
        self.assertEqual(X_test.shape[0], 1)
        self.assertEqual(X_train.shape[1], X_test.shape[1])


if __name__ == "__main__":
    unittest.main()
