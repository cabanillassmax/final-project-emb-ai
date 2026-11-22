"""Flask application for the Watson NLP emotion detector."""

from flask import Flask, render_template, request

from EmotionDetection import emotion_detector

app = Flask(__name__)
INVALID_TEXT_MESSAGE = "Invalid text! Please try again!"


@app.route("/")
def render_index_page():
    """Render the emotion detection page."""
    return render_template("index.html")


@app.route("/emotionDetector")
def emotion_detector_route():
    """Analyze the submitted text and return a readable result."""
    text_to_analyze = request.args.get("textToAnalyze", default="", type=str)
    result = emotion_detector(text_to_analyze)

    if result["dominant_emotion"] is None:
        return INVALID_TEXT_MESSAGE

    return (
        "For the given statement, the system response is "
        f"'anger': {result['anger']}, 'disgust': {result['disgust']}, "
        f"'fear': {result['fear']}, 'joy': {result['joy']} and "
        f"'sadness': {result['sadness']}. "
        f"The dominant emotion is {result['dominant_emotion']}."
    )


if __name__ == "__main__":
    app.run(host="localhost", port=5000)
