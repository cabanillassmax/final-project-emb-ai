"""Live Watson NLP tests required by the course assignment."""

import unittest

from EmotionDetection import emotion_detector


class WatsonEmotionDetectionTests(unittest.TestCase):
    """Verify the five required emotions with the live Skills Network API."""

    def test_joy_is_dominant(self):
        """Detect joy in the required positive statement."""
        result = emotion_detector("I am glad this happened")
        self.assertEqual(result["dominant_emotion"], "joy")

    def test_anger_is_dominant(self):
        """Detect anger in the required angry statement."""
        result = emotion_detector("I am really mad about this")
        self.assertEqual(result["dominant_emotion"], "anger")

    def test_disgust_is_dominant(self):
        """Detect disgust in the required disgusted statement."""
        result = emotion_detector(
            "I feel disgusted just hearing about this"
        )
        self.assertEqual(result["dominant_emotion"], "disgust")

    def test_sadness_is_dominant(self):
        """Detect sadness in the required sad statement."""
        result = emotion_detector("I am so sad about this")
        self.assertEqual(result["dominant_emotion"], "sadness")

    def test_fear_is_dominant(self):
        """Detect fear in the required fearful statement."""
        result = emotion_detector("I am really afraid that this will happen")
        self.assertEqual(result["dominant_emotion"], "fear")


if __name__ == "__main__":
    unittest.main()
