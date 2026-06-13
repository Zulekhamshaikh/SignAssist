# StaticNorm.py — FINAL: BETTER ANCHOR
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class HandNormalizer132(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def _norm_hand(self, hand):
        if np.all(hand == 0):
            return hand
        wrist = hand[0]
        # Use wrist if valid, else first non-zero landmark
        if np.var(wrist) < 1e-6 or np.all(wrist == 0):
            valid = hand[np.any(hand != 0, axis=1)]
            anchor = valid.mean(axis=0) if len(valid) > 0 else hand.mean(axis=0)
        else:
            anchor = wrist
        centered = hand - anchor
        scale = np.ptp(centered[:, :2], axis=0).max()
        scale = max(scale, 1e-6)
        normalized = centered / scale
        normalized[:, 2] = np.clip(normalized[:, 2], -1.5, 1.5)
        return normalized

    def transform(self, X):
        X = np.asarray(X, dtype=np.float32)
        landmarks = X[:, :126].reshape(-1, 2, 21, 3)
        relative = X[:, 126:]
        
        for i in range(2):
            for j in range(landmarks.shape[0]):
                landmarks[j, i] = self._norm_hand(landmarks[j, i])
        
        normalized_landmarks = landmarks.reshape(-1, 126)
        return np.hstack([normalized_landmarks, relative])