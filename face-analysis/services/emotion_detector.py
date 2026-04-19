from typing import Any, Iterable, List, Optional, Tuple

import numpy as np

from config import canonical_emotion_label
from hsemotion_onnx.facial_emotions import HSEmotionRecognizer


class EmotionDetector:
    def __init__(self, model_name: str = "enet_b0_8_best_afew", debug: bool = False) -> None:
        self.model = HSEmotionRecognizer(model_name=model_name)
        self._debug = debug

    def predict(self, face_bgr: np.ndarray) -> Tuple[str, float]:
        prediction = self._run_model(face_bgr)
        if self._debug:
            print(f"Raw preds: {prediction}")
        emotion, confidence = self._parse_prediction(prediction)
        return emotion, confidence

    def _run_model(self, face_bgr: np.ndarray) -> Any:
        for method_name in ("predict_emotions", "predict"):
            method = getattr(self.model, method_name, None)
            if method is None:
                continue

            for kwargs in ({"logits": False}, {}, {"logits": True}):
                try:
                    return method(face_bgr, **kwargs)
                except TypeError:
                    continue

            try:
                return method(face_bgr)
            except Exception as exc:
                raise RuntimeError(f"Emotion model inference failed: {exc}") from exc

        if callable(self.model):
            try:
                return self.model(face_bgr)
            except Exception as exc:
                raise RuntimeError(f"Emotion model inference failed: {exc}") from exc

        raise RuntimeError("No compatible prediction method found for HSEmotion model")

    def _parse_prediction(self, prediction: Any) -> Tuple[str, float]:
        labels = self._labels_from_model()

        if isinstance(prediction, str):
            return prediction.title(), 1.0

        if isinstance(prediction, dict):
            emotion = prediction.get("emotion") or prediction.get("label")
            confidence = prediction.get("confidence") or prediction.get("score")
            if emotion is None:
                emotion, confidence = self._extract_from_scores(prediction, labels)
            return self._normalize_emotion(emotion), self._normalize_confidence(confidence)

        if isinstance(prediction, tuple) and len(prediction) >= 1:
            emotion_candidate = prediction[0]
            confidence = None

            if len(prediction) >= 2:
                second = prediction[1]
                if isinstance(second, (dict, list, tuple, np.ndarray)):
                    parsed_emotion, parsed_conf = self._extract_from_scores(second, labels)
                    if isinstance(emotion_candidate, str):
                        return self._normalize_emotion(emotion_candidate), self._normalize_confidence(parsed_conf)
                    return self._normalize_emotion(parsed_emotion), self._normalize_confidence(parsed_conf)
                confidence = second

            if isinstance(emotion_candidate, str):
                return self._normalize_emotion(emotion_candidate), self._normalize_confidence(confidence)

            emotion, parsed_conf = self._extract_from_scores(emotion_candidate, labels)
            return self._normalize_emotion(emotion), self._normalize_confidence(parsed_conf)

        if isinstance(prediction, (list, tuple, np.ndarray)):
            emotion, confidence = self._extract_from_scores(prediction, labels)
            return self._normalize_emotion(emotion), self._normalize_confidence(confidence)

        return "Unknown", 0.0

    def _labels_from_model(self) -> Optional[List[str]]:
        for attr_name in ("idx_to_class", "class_names", "emotions", "labels"):
            value = getattr(self.model, attr_name, None)
            if value is None:
                continue

            if isinstance(value, dict) and value:
                max_index = max(value.keys())
                return [str(value[i]) for i in range(max_index + 1) if i in value]

            if isinstance(value, Iterable) and not isinstance(value, (str, bytes, dict)):
                labels = [str(x) for x in value]
                if labels:
                    return labels

        return None

    def _extract_from_scores(self, scores_obj: Any, labels: Optional[List[str]]) -> Tuple[str, float]:
        if isinstance(scores_obj, dict):
            if not scores_obj:
                return "Unknown", 0.0

            best_label = max(scores_obj, key=scores_obj.get)
            best_score = float(scores_obj[best_label])
            return str(best_label), best_score

        scores = np.asarray(scores_obj, dtype=float).flatten()
        if scores.size == 0:
            return "Unknown", 0.0

        # Convert logits to probabilities if values are outside a typical probability range.
        if np.min(scores) < 0 or np.max(scores) > 1:
            exp_scores = np.exp(scores - np.max(scores))
            probs = exp_scores / np.sum(exp_scores)
        else:
            total = np.sum(scores)
            probs = scores / total if total > 0 else scores

        best_idx = int(np.argmax(probs))
        best_score = float(probs[best_idx])

        if labels and best_idx < len(labels):
            return labels[best_idx], best_score
        return str(best_idx), best_score

    @staticmethod
    def _normalize_emotion(emotion: Any) -> str:
        if emotion is None:
            return "Unknown"
        canonical = canonical_emotion_label(str(emotion))
        return canonical.title()

    @staticmethod
    def _normalize_confidence(confidence: Any) -> float:
        if confidence is None:
            return 1.0

        try:
            value = float(confidence)
        except (TypeError, ValueError):
            return 1.0

        if value < 0:
            return 0.0
        if value > 1:
            return 1.0
        return value
