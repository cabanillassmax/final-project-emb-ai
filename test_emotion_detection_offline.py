"""Offline tests for emotion detection and the Flask interface."""

import json
import unittest
from unittest.mock import Mock, patch

from EmotionDetection import emotion_detector
from server import INVALID_TEXT_MESSAGE, app

EMOTIONS = ("anger", "disgust", "fear", "joy", "sadness")
API_URL = (
    "https://sn-watson-emotion.labs.skills.network/"
    "v1/watson.runtime.nlp.v1/NlpService/EmotionPredict"
)
MODEL_ID = "emotion_aggregated-workflow_lang_en_stock"


def _mock_response(scores, status_code=200):
    """Build a requests-like response for an isolated unit test."""
    response = Mock()
    response.status_code = status_code
    response.text = json.dumps(
        {"emotionPredictions": [{"emotion": scores}]}
    )
    response.raise_for_status.return_value = None
    return response


class EmotionDetectorTests(unittest.TestCase):
    """Test response formatting for the five required emotions."""

    def _assert_dominant_emotion(self, mock_post, text, expected_emotion):
        """Run the detector against a deterministic mocked service response."""
        scores = {emotion: 0.1 for emotion in EMOTIONS}
        scores[expected_emotion] = 0.9
        mock_post.return_value = _mock_response(scores)

        result = emotion_detector(text)

        self.assertEqual(result["dominant_emotion"], expected_emotion)
        actual_emotions = tuple(
            key for key in result if key != "dominant_emotion"
        )
        self.assertEqual(actual_emotions, EMOTIONS)
        mock_post.assert_called_once_with(
            API_URL,
            headers={"grpc-metadata-mm-model-id": MODEL_ID},
            json={"raw_document": {"text": text}},
            timeout=30,
        )

    @patch("EmotionDetection.emotion_detection.requests.post")
    def test_joy_is_dominant(self, mock_post):
        """Detect joy for the required positive statement."""
        self._assert_dominant_emotion(
            mock_post, "I am glad this happened", "joy"
        )

    @patch("EmotionDetection.emotion_detection.requests.post")
    def test_anger_is_dominant(self, mock_post):
        """Detect anger for the required angry statement."""
        self._assert_dominant_emotion(
            mock_post, "I am really mad about this", "anger"
        )

    @patch("EmotionDetection.emotion_detection.requests.post")
    def test_disgust_is_dominant(self, mock_post):
        """Detect disgust for the required disgusted statement."""
        self._assert_dominant_emotion(
            mock_post,
            "I feel disgusted just hearing about this",
            "disgust",
        )

    @patch("EmotionDetection.emotion_detection.requests.post")
    def test_sadness_is_dominant(self, mock_post):
        """Detect sadness for the required sad statement."""
        self._assert_dominant_emotion(
            mock_post, "I am so sad about this", "sadness"
        )

    @patch("EmotionDetection.emotion_detection.requests.post")
    def test_fear_is_dominant(self, mock_post):
        """Detect fear for the required fearful statement."""
        self._assert_dominant_emotion(
            mock_post, "I am really afraid that this will happen", "fear"
        )

    @patch("EmotionDetection.emotion_detection.requests.post")
    def test_bad_request_returns_none_for_every_value(self, mock_post):
        """Return the required empty result when the service rejects input."""
        mock_post.return_value = _mock_response({}, status_code=400)

        result = emotion_detector("")

        self.assertEqual(
            result,
            {
                "anger": None,
                "disgust": None,
                "fear": None,
                "joy": None,
                "sadness": None,
                "dominant_emotion": None,
            },
        )


class EmotionDetectorRouteTests(unittest.TestCase):
    """Test Flask response formatting and invalid-input handling."""

    def setUp(self):
        """Create a Flask test client."""
        self.client = app.test_client()

    def test_home_page_renders_text_input(self):
        """Serve the page with the multiline text input."""
        response = self.client.get("/")

        self.assertEqual(response.status_code, 200)
        self.assertIn(b'<textarea class="form-control"', response.data)

    def test_client_script_is_served(self):
        """Serve the browser script used to call the detector route."""
        response = self.client.get("/static/mywebscript.js")

        self.assertEqual(response.status_code, 200)
        self.assertIn(b"encodeURIComponent(textToAnalyze)", response.data)
        response.close()

    @patch("server.emotion_detector")
    def test_valid_result_is_formatted(self, mock_detector):
        """Return the required readable string for a valid result."""
        mock_detector.return_value = {
            "anger": 0.1,
            "disgust": 0.2,
            "fear": 0.3,
            "joy": 0.9,
            "sadness": 0.4,
            "dominant_emotion": "joy",
        }

        response = self.client.get(
            "/emotionDetector",
            query_string={"textToAnalyze": "I think I am having fun"},
        )

        self.assertEqual(response.status_code, 200)
        self.assertIn(b"The dominant emotion is joy.", response.data)

    @patch("server.emotion_detector")
    def test_invalid_result_displays_error_message(self, mock_detector):
        """Display the required message when the detector returns no result."""
        mock_detector.return_value = {
            "anger": None,
            "disgust": None,
            "fear": None,
            "joy": None,
            "sadness": None,
            "dominant_emotion": None,
        }

        response = self.client.get("/emotionDetector")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_data(as_text=True), INVALID_TEXT_MESSAGE)
        mock_detector.assert_called_once_with("")


if __name__ == "__main__":
    unittest.main()
