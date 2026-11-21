"""python3 -m unittest scripts/test_number_words.py"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(__file__))
from number_words import to_digits  # noqa: E402


class SwahiliAmounts(unittest.TestCase):
    def check(self, text, want):
        self.assertEqual(to_digits(text), want)

    def test_scale_then_multiplier(self):
        self.check("Weka elfu ishirini", "Weka 20000")
        self.check("kiasi laki tano", "kiasi 500000")
        self.check("Kiasi ni elfu sabini", "Kiasi ni 70000")
        self.check("shilingi milioni moja", "shilingi 1000000")

    def test_na_joins_parts(self):
        self.check("Laki tatu na elfu hamsini", "350000")
        self.check("Milioni moja na laki tano", "1500000")
        self.check("elfu mbili na mia tano", "2500")

    def test_compound_multiplier(self):
        self.check("elfu kumi na tano", "15000")
        self.check("elfu mia tano", "500000")
        self.check("elfu thelathini na nane", "38000")

    def test_hundreds(self):
        self.check("mia tano", "500")
        self.check("mia moja na hamsini", "150")

    def test_digits_in_a_phrase(self):
        self.check("milioni 2", "2000000")
        self.check("laki 1", "100000")

    def test_scale_alone_means_one(self):
        self.check("elfu", "1000")

    def test_words_without_a_scale_stay(self):
        self.check("Nenda mara moja", "Nenda mara moja")
        self.check("Tuma fomu", "Tuma fomu")

    def test_rest_of_the_text_is_unchanged(self):
        self.check("Nifungulie fomu ya mkopo, jina ni Asha Juma, kiasi laki tano",
                   "Nifungulie fomu ya mkopo, jina ni Asha Juma, kiasi 500000")
        self.check("Namba yangu ya simu ni 0754 111 222", "Namba yangu ya simu ni 0754 111 222")

    def test_punctuation_ends_a_number(self):
        self.check("kiasi laki moja, simu 0755 999 000", "kiasi 100000, simu 0755 999 000")

    def test_na_between_a_number_and_a_word_stays(self):
        self.check("elfu kumi na uweke", "10000 na uweke")


class EnglishAmounts(unittest.TestCase):
    def check(self, text, want):
        self.assertEqual(to_digits(text), want)

    def test_amounts(self):
        self.check("two million shillings", "2000000 shillings")
        self.check("Make it three hundred thousand", "Make it 300000")
        self.check("one hundred thousand", "100000")
        self.check("one million five hundred thousand", "1500000")
        self.check("two hundred and fifty thousand", "250000")
        self.check("a thousand", "1000")

    def test_without_a_scale(self):
        self.check("Take me to my profile", "Take me to my profile")
        self.check("one more thing", "one more thing")


if __name__ == "__main__":
    unittest.main()
