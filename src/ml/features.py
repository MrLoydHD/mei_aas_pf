"""
Feature extraction for DGA detection.
Extracts various statistical, linguistic, and TF-IDF features from domain names.
"""

import math
import os
from collections import Counter
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


class DomainFeatureExtractor:
    """
    Extract features from domain names for DGA detection.

    Features include:
    - Length-based features
    - Character distribution features
    - Entropy measures
    - N-gram frequencies
    - Vowel/consonant ratios
    - Dictionary word presence
    - TF-IDF similarity scores (from notebook approach)
    """

    VOWELS = set('aeiou')
    CONSONANTS = set('bcdfghjklmnpqrstvwxyz')

    def __init__(
        self,
        word_dict_path: Optional[str] = None,
        legit_domains_path: Optional[str] = None,
        use_tfidf: bool = True
    ):
        """
        Initialize the feature extractor.

        Args:
            word_dict_path: Path to dictionary file for word-based features
            legit_domains_path: Path to legitimate domains CSV for TF-IDF
            use_tfidf: Whether to compute TF-IDF features
        """
        self.word_dict = set()
        self.use_tfidf = use_tfidf

        # TF-IDF vectorizers
        self.legit_tfidf_vectorizer = None
        self.legit_tfidf_weights = None
        self.word_tfidf_vectorizer = None
        self.word_tfidf_weights = None

        if word_dict_path:
            self.load_dictionary(word_dict_path)
            if use_tfidf:
                self._fit_word_tfidf(word_dict_path)

        if legit_domains_path and use_tfidf:
            self._fit_legit_tfidf(legit_domains_path)

        # Common bigrams in English
        self.common_bigrams = {
            'th', 'he', 'in', 'en', 'nt', 're', 'er', 'an', 'ti', 'es',
            'on', 'at', 'se', 'nd', 'or', 'ar', 'al', 'te', 'co', 'de',
            'to', 'ra', 'et', 'ed', 'it', 'sa', 'em', 'ro'
        }

        # Common trigrams in English
        self.common_trigrams = {
            'the', 'and', 'tha', 'ent', 'ing', 'ion', 'tio', 'for',
            'nde', 'has', 'nce', 'edt', 'tis', 'oft', 'sth', 'men'
        }

    def _fit_legit_tfidf(self, legit_domains_path: str) -> None:
        """Fit TF-IDF vectorizer on legitimate domains."""
        try:
            print(f"Fitting TF-IDF on legitimate domains from {legit_domains_path}...")
            legit_df = pd.read_csv(legit_domains_path)
            legit_domains = legit_df['domain'].dropna().drop_duplicates().tolist()

            self.legit_tfidf_vectorizer = TfidfVectorizer(
                analyzer='char',
                ngram_range=(3, 5),
                min_df=1e-6,
                max_df=1.0
            )

            tfidf_matrix = self.legit_tfidf_vectorizer.fit_transform(legit_domains)
            self.legit_tfidf_weights = tfidf_matrix.sum(axis=0).getA1()
            print(f"Legit TF-IDF fitted with {len(self.legit_tfidf_vectorizer.get_feature_names_out())} features")
        except Exception as e:
            print(f"Warning: Could not fit legit TF-IDF: {e}")
            self.legit_tfidf_vectorizer = None

    def _fit_word_tfidf(self, word_dict_path: str) -> None:
        """Fit TF-IDF vectorizer on English words."""
        try:
            print(f"Fitting TF-IDF on English words from {word_dict_path}...")
            words = []
            with open(word_dict_path, 'r', encoding='utf-8') as f:
                for line in f:
                    word = line.strip().lower()
                    if word and word.isalpha():
                        words.append(word)

            self.word_tfidf_vectorizer = TfidfVectorizer(
                analyzer='char',
                ngram_range=(3, 5),
                min_df=1e-6,
                max_df=1.0
            )

            tfidf_matrix = self.word_tfidf_vectorizer.fit_transform(words)
            self.word_tfidf_weights = tfidf_matrix.sum(axis=0).getA1()
            print(f"Word TF-IDF fitted with {len(self.word_tfidf_vectorizer.get_feature_names_out())} features")
        except Exception as e:
            print(f"Warning: Could not fit word TF-IDF: {e}")
            self.word_tfidf_vectorizer = None

    def load_dictionary(self, path: str) -> None:
        """Load word dictionary from file."""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                self.word_dict = set(
                    word.strip().lower()
                    for word in f
                    if len(word.strip()) >= 3
                )
        except FileNotFoundError:
            print(f"Warning: Dictionary file not found at {path}")

    def calculate_entropy(self, text: str) -> float:
        """
        Calculate Shannon entropy of a string.
        Higher entropy indicates more randomness (common in DGA domains).
        """
        if not text:
            return 0.0

        freq = Counter(text)
        length = len(text)
        entropy = 0.0

        for count in freq.values():
            if count > 0:
                prob = count / length
                entropy -= prob * math.log2(prob)

        return entropy

    def get_ngrams(self, text: str, n: int) -> List[str]:
        """Extract n-grams from text."""
        return [text[i:i+n] for i in range(len(text) - n + 1)]

    def calculate_vowel_ratio(self, text: str) -> float:
        """Calculate ratio of vowels to total characters."""
        if not text:
            return 0.0

        text_lower = text.lower()
        vowel_count = sum(1 for c in text_lower if c in self.VOWELS)
        return vowel_count / len(text)

    def calculate_consonant_ratio(self, text: str) -> float:
        """Calculate ratio of consonants to total characters."""
        if not text:
            return 0.0

        text_lower = text.lower()
        consonant_count = sum(1 for c in text_lower if c in self.CONSONANTS)
        return consonant_count / len(text)

    def calculate_digit_ratio(self, text: str) -> float:
        """Calculate ratio of digits to total characters."""
        if not text:
            return 0.0

        digit_count = sum(1 for c in text if c.isdigit())
        return digit_count / len(text)

    def count_consecutive_consonants(self, text: str) -> int:
        """Count maximum consecutive consonants."""
        text_lower = text.lower()
        max_consec = 0
        current = 0

        for c in text_lower:
            if c in self.CONSONANTS:
                current += 1
                max_consec = max(max_consec, current)
            else:
                current = 0

        return max_consec

    def count_consecutive_digits(self, text: str) -> int:
        """Count maximum consecutive digits."""
        max_consec = 0
        current = 0

        for c in text:
            if c.isdigit():
                current += 1
                max_consec = max(max_consec, current)
            else:
                current = 0

        return max_consec

    def has_dictionary_words(self, text: str, min_word_len: int = 4) -> Tuple[int, float]:
        """
        Check if domain contains dictionary words.

        Returns:
            Tuple of (word_count, coverage_ratio)
        """
        if not self.word_dict:
            return 0, 0.0

        text_lower = text.lower()
        word_count = 0
        covered_chars = 0

        # Check for words of various lengths
        for word_len in range(min_word_len, len(text_lower) + 1):
            for i in range(len(text_lower) - word_len + 1):
                substring = text_lower[i:i+word_len]
                if substring in self.word_dict:
                    word_count += 1
                    covered_chars = max(covered_chars, word_len)

        coverage = covered_chars / len(text_lower) if text_lower else 0.0
        return word_count, coverage

    def calculate_bigram_score(self, text: str) -> float:
        """
        Calculate score based on common English bigrams.
        Higher score = more English-like = more likely legitimate.
        """
        if len(text) < 2:
            return 0.0

        text_lower = text.lower()
        bigrams = self.get_ngrams(text_lower, 2)

        if not bigrams:
            return 0.0

        common_count = sum(1 for bg in bigrams if bg in self.common_bigrams)
        return common_count / len(bigrams)

    def calculate_trigram_score(self, text: str) -> float:
        """
        Calculate score based on common English trigrams.
        """
        if len(text) < 3:
            return 0.0

        text_lower = text.lower()
        trigrams = self.get_ngrams(text_lower, 3)

        if not trigrams:
            return 0.0

        common_count = sum(1 for tg in trigrams if tg in self.common_trigrams)
        return common_count / len(trigrams)

    def calculate_legit_tfidf_score(self, domain: str) -> float:
        """Calculate TF-IDF similarity score with legitimate domains."""
        if self.legit_tfidf_vectorizer is None or self.legit_tfidf_weights is None:
            return 0.0

        try:
            tfidf_vec = self.legit_tfidf_vectorizer.transform([domain])
            score = float(self.legit_tfidf_weights @ tfidf_vec.T)
            return score
        except Exception:
            return 0.0

    def calculate_word_tfidf_score(self, domain: str) -> float:
        """Calculate TF-IDF similarity score with English words."""
        if self.word_tfidf_vectorizer is None or self.word_tfidf_weights is None:
            return 0.0

        try:
            tfidf_vec = self.word_tfidf_vectorizer.transform([domain])
            score = float(self.word_tfidf_weights @ tfidf_vec.T)
            return score
        except Exception:
            return 0.0

    def extract_features(self, domain: str) -> Dict[str, float]:
        """
        Extract all features from a single domain.

        Args:
            domain: Domain name (without TLD)

        Returns:
            Dictionary of feature names to values
        """
        # Clean domain - remove TLD and subdomains, focus on main part
        domain_clean = domain.lower().replace('-', '').replace('_', '')

        # Length features
        length = len(domain_clean)

        # Character distribution
        unique_chars = len(set(domain_clean))
        unique_ratio = unique_chars / length if length > 0 else 0

        # Entropy
        entropy = self.calculate_entropy(domain_clean)
        normalized_entropy = entropy / math.log2(26) if length > 0 else 0

        # Character type ratios
        vowel_ratio = self.calculate_vowel_ratio(domain_clean)
        consonant_ratio = self.calculate_consonant_ratio(domain_clean)
        digit_ratio = self.calculate_digit_ratio(domain_clean)

        # Consecutive character features
        max_consonants = self.count_consecutive_consonants(domain_clean)
        max_digits = self.count_consecutive_digits(domain_clean)

        # N-gram scores
        bigram_score = self.calculate_bigram_score(domain_clean)
        trigram_score = self.calculate_trigram_score(domain_clean)

        # Dictionary features
        word_count, word_coverage = self.has_dictionary_words(domain_clean)

        # Special character features (in original domain)
        has_numbers = 1 if any(c.isdigit() for c in domain) else 0
        has_hyphen = 1 if '-' in domain else 0

        # Character variety (number of unique character types)
        has_upper = any(c.isupper() for c in domain)
        has_lower = any(c.islower() for c in domain)
        has_digit = any(c.isdigit() for c in domain)
        char_variety = sum([has_upper, has_lower, has_digit])

        features = {
            'length': length,
            'unique_chars': unique_chars,
            'unique_ratio': unique_ratio,
            'entropy': entropy,
            'normalized_entropy': normalized_entropy,
            'vowel_ratio': vowel_ratio,
            'consonant_ratio': consonant_ratio,
            'digit_ratio': digit_ratio,
            'max_consecutive_consonants': max_consonants,
            'max_consecutive_digits': max_digits,
            'bigram_score': bigram_score,
            'trigram_score': trigram_score,
            'dictionary_word_count': word_count,
            'dictionary_coverage': word_coverage,
            'has_numbers': has_numbers,
            'has_hyphen': has_hyphen,
            'char_variety': char_variety,
        }

        # Add TF-IDF features if available
        if self.use_tfidf:
            legit_tfidf = self.calculate_legit_tfidf_score(domain_clean)
            word_tfidf = self.calculate_word_tfidf_score(domain_clean)

            features['legit_tfidf'] = legit_tfidf
            features['word_tfidf'] = word_tfidf
            # Log-transformed versions (as in the notebook)
            features['log_legit_tfidf'] = np.log1p(legit_tfidf)
            features['log_word_tfidf'] = np.log1p(word_tfidf)

        return features

    def extract_features_batch(self, domains: List[str]) -> pd.DataFrame:
        """
        Extract features from multiple domains.

        Args:
            domains: List of domain names

        Returns:
            DataFrame with features for each domain
        """
        features_list = [self.extract_features(d) for d in domains]
        return pd.DataFrame(features_list)

    def get_feature_names(self) -> List[str]:
        """Return list of feature names."""
        base_features = [
            'length', 'unique_chars', 'unique_ratio', 'entropy',
            'normalized_entropy', 'vowel_ratio', 'consonant_ratio',
            'digit_ratio', 'max_consecutive_consonants', 'max_consecutive_digits',
            'bigram_score', 'trigram_score', 'dictionary_word_count',
            'dictionary_coverage', 'has_numbers', 'has_hyphen', 'char_variety'
        ]

        if self.use_tfidf:
            base_features.extend([
                'legit_tfidf', 'word_tfidf', 'log_legit_tfidf', 'log_word_tfidf'
            ])

        return base_features


def remove_outliers(df: pd.DataFrame, column: str, q1: float = 0.20, q3: float = 0.80) -> pd.DataFrame:
    """
    Remove outliers from a DataFrame column using IQR method.
    From notebook approach.

    Args:
        df: Input DataFrame
        column: Column name to check for outliers
        q1: Lower quantile (default 0.20)
        q3: Upper quantile (default 0.80)

    Returns:
        DataFrame with outliers removed
    """
    quartile1 = df[column].quantile(q1)
    quartile3 = df[column].quantile(q3)
    iqr = quartile3 - quartile1
    lower_limit = quartile1 - 1.5 * iqr
    upper_limit = quartile3 + 1.5 * iqr

    return df[~((df[column] <= lower_limit) | (df[column] >= upper_limit))]
