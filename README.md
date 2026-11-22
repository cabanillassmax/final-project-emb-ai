# Final project

Watson NLP emotion detection app built with Flask for the Coursera final project.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-dev.txt
```

## Run locally

```powershell
python server.py
```

Open `http://localhost:5000` in a browser. The Watson NLP endpoint is only
available from the Skills Network Theia environment, so live emotion detection
and the course screenshots must be verified there.

The local unit tests mock the external API and can be run offline:

```powershell
python -m unittest -v test_emotion_detection_offline.py
```

Run the required static analysis with:

```powershell
python -m pylint server.py
```

In the Skills Network Theia environment, run the five course tests against the
live Watson NLP API:

```bash
python3 -m unittest -v test_emotion_detection.py
```
