"""Train-only lexical context retrieval. No gold corrections or topic keyword rules."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import unicodedata

import joblib
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

# Function words contribute little topic information; these are not topic mappings.
STOP_WORDS = "và là của có một những các được cho với trong trên này đó tôi bạn họ em anh chị mình chúng ta thì đã đang sẽ cũng rất về ở khi để từ đến như hay hoặc nhưng nên vì không còn vào ra lại thêm thấy cần muốn xem mua chọn".split()


@dataclass
class ContextProfile:
    vectorizer: object
    vector: object
    neighbors: list
    reason: str

    @property
    def available(self):
        return self.vector is not None and bool(self.neighbors)

    def score(self, text):
        if not self.available:
            return 0.0
        return float((self.vectorizer.transform([text]) @ self.vector.T)[0, 0])

    def describe(self):
        return {"available": self.available, "method": "train_only_tfidf_neighbor_centroid",
                "reason": self.reason, "neighbor_count": len(self.neighbors),
                "top_neighbors": self.neighbors[:5], "calibrated_probability": False}


class ContextIndex:
    def __init__(self, vectorizer, matrix, metadata, options=None):
        self.vectorizer, self.matrix, self.metadata = vectorizer, matrix, metadata
        self.options = options or {}

    @classmethod
    def build(cls, corpus_path, metadata, options):
        # Fit only this known train-side corpus. Context benchmark files are never read here.
        sentences = Path(corpus_path).read_text(encoding="utf-8").splitlines()
        vectorizer = TfidfVectorizer(lowercase=True, token_pattern=r"(?u)\b[^\W\d_]+\b",
                                     ngram_range=(1, 2), min_df=options.get("min_df", 2),
                                     max_features=options.get("max_features", 60000),
                                     stop_words=STOP_WORDS, sublinear_tf=True, dtype=np.float32)
        matrix = vectorizer.fit_transform(sentences).tocsr()
        return cls(vectorizer, matrix, {**metadata, "n_sentences": len(sentences),
                                       "n_features": matrix.shape[1]}, options)

    def save(self, path):
        joblib.dump(self, path, compress=3)

    @classmethod
    def load(cls, path):
        return joblib.load(path)

    def profile(self, previous="", following=""):
        if len(previous) > 4000 or len(following) > 4000:
            raise ValueError("Each neighboring context is limited to 4,000 characters.")
        query = unicodedata.normalize("NFC", previous + " " + following).strip()
        if not query:
            return ContextProfile(self.vectorizer, None, [], "No neighboring context supplied.")
        vector = self.vectorizer.transform([query])
        if not vector.nnz:
            return ContextProfile(self.vectorizer, None, [], "Context has no indexed content features.")
        matches = (self.matrix @ vector.T).tocoo()
        minimum = self.options.get("minimum_similarity", 0.08)
        eligible = [(float(score), int(row)) for row, score in zip(matches.row, matches.data) if score >= minimum]
        eligible.sort(key=lambda item: (-item[0], item[1]))
        chosen = eligible[:self.options.get("neighbors", 64)]
        if not chosen:
            return ContextProfile(self.vectorizer, None, [], "No train-corpus neighbors above retrieval threshold.")
        weights = np.array([score for score, _ in chosen], dtype=np.float32)
        rows = [row for _, row in chosen]
        centroid = csr_matrix(weights.reshape(1, -1)) @ self.matrix[rows]
        centroid = normalize(centroid, norm="l2", copy=False)
        return ContextProfile(self.vectorizer, centroid,
                              [{"corpus_row": row, "similarity": score} for score, row in chosen],
                              "Lexical support from train-corpus sentences; not a semantic correctness guarantee.")
