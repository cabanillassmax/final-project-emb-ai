function RunSentimentAnalysis() {
    const textToAnalyze = document.getElementById("textToAnalyze").value;
    const systemResponse = document.getElementById("system_response");
    const xhttp = new XMLHttpRequest();

    xhttp.onreadystatechange = function() {
        if (this.readyState === 4) {
            systemResponse.textContent = this.status === 200
                ? this.responseText
                : `Request failed with status ${this.status}. Please try again.`;
        }
    };
    xhttp.open(
        "GET",
        `/emotionDetector?textToAnalyze=${encodeURIComponent(textToAnalyze)}`,
        true
    );
    xhttp.send();
}
