"""Detect emotions using the Skills Network Watson NLP service."""

import json
from typing import Any

import requests

API_URL = (
    "https://sn-watson-emotion.labs.skills.network/"
    "v1/watson.runtime.nlp.v1/NlpService/EmotionPredict"
)
MODEL_ID = "emotion_aggregated-workflow_lang_en_stock"
EMOTIONS = ("anger", "disgust", "fear", "joy", "sadness")


def _empty_result() -> dict[str, None]:
    """Return the standard result structure with empty values."""
    return {emotion: None for emotion in (*EMOTIONS, "dominant_emotion")}


def emotion_detector(text_to_analyze: str) -> dict[str, Any]:
    """Return emotion scores and the dominant emotion for the supplied text."""
    response = requests.post(
        API_URL,
        headers={"grpc-metadata-mm-model-id": MODEL_ID},
        json={"raw_document": {"text": text_to_analyze}},
        timeout=30,
    )

    if response.status_code == 400:
        return _empty_result()

    response.raise_for_status()
    response_data = json.loads(response.text)
    emotion_scores = response_data["emotionPredictions"][0]["emotion"]
    result = {emotion: emotion_scores[emotion] for emotion in EMOTIONS}
    result["dominant_emotion"] = max(EMOTIONS, key=emotion_scores.__getitem__)
    return result
