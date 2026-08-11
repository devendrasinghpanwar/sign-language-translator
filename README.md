# Sign Language Translator — Live Web Version

![Tests](https://github.com/YOUR-USERNAME/sign-language-translator/actions/workflows/tests.yml/badge.svg)

A real-time sign language recognizer that runs live in the browser.
Built for learning + a fast, genuinely live deployment you can send to
recruiters.

**Live demo:** https://huggingface.co/spaces/YOUR-USERNAME/sign-language-translator
*(update this link once deployed — see Part 2 below)*

## What makes this "live"?

This uses **Gradio** — a Python library that turns a normal Python
function into a website automatically — and **Hugging Face Spaces**,
which hosts it for free and gives you a public URL like:

```
https://huggingface.co/spaces/yourname/sign-language-translator
```

Anyone with that link can open it in their browser, use their webcam,
and see live predictions. No web development knowledge needed.

## How it works (the ML side)

1. `collect_data.py` — uses MediaPipe to find 21 points on your hand
   and saves them as numbers (not images) into `hand_data.csv`
2. `train_classifier.py` — trains a Random Forest model (a simpler,
   beginner-friendly alternative to deep learning) on that data
3. `app.py` — loads the trained model and wraps it in a Gradio
   interface so your webcam feed gets predicted live in the browser

## Part 1: Run it locally first

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Step 1 — Collect your data
```bash
python collect_data.py
```
Show each sign, press its key to save a sample. Aim for 30-50 samples
per sign, varying your hand angle/distance slightly each time.

### Step 2 — Train the model
```bash
python train_classifier.py
```
You'll see an accuracy score printed. 85%+ is solid for a first version.

### Step 3 — Test the web app locally
```bash
python app.py
```
Open the URL it prints in your browser (usually `http://127.0.0.1:7860`)
and test it with your webcam before deploying.

## Testing

This project has a real test suite (12 tests) covering:
- Landmark-extraction logic (`tests/test_utils.py`) — verifies the
  MediaPipe-to-numbers conversion is correct, handles edge cases
  (empty hands), and preserves ordering
- Model training pipeline (`tests/test_model_pipeline.py`) — verifies
  the classifier trains correctly and achieves expected accuracy on
  known synthetic data, catching bugs like mislabeled/misaligned data

Tests use small, synthetic/fake data so they run in ~3 seconds without
needing a webcam or real recorded dataset — a standard practice for
testing ML pipelines.

### Run the tests

```bash
pip install pytest pandas scikit-learn numpy
pytest tests/ -v
```

### Continuous Integration

Tests run automatically on every push via GitHub Actions
(`.github/workflows/tests.yml`). Once you push this project to GitHub,
you'll see a "Tests" check on every commit, and the badge at the top
of this README will show passing/failing status live.

## Part 2: Deploy it live (this is the recruiter-facing part)

### Step 1 — Create a free Hugging Face account
Go to https://huggingface.co/join

### Step 2 — Create a new Space
1. Go to https://huggingface.co/new-space
2. Give it a name (e.g. `sign-language-translator`)
3. Choose **Gradio** as the SDK
4. Choose **Public** visibility
5. Click "Create Space"

### Step 3 — Upload your files
In your new Space, click "Files" → "Add file" → "Upload files", and
upload:
- `app.py`
- `requirements.txt`
- `sign_classifier.joblib` (the trained model from Step 2 above)

(You do NOT need to upload `collect_data.py`, `train_classifier.py`,
or `hand_data.csv` — the live app only needs the trained model.)

### Step 4 — Wait for it to build
Hugging Face automatically installs your requirements and starts the
app. This takes 2-5 minutes. You'll see build logs on the page.

### Step 5 — You're live!
Your app is now at:
```
https://huggingface.co/spaces/YOUR-USERNAME/sign-language-translator
```
This is the link you put on your resume, LinkedIn, and share with
recruiters.

## What to say to recruiters

> "I built and deployed a real-time sign language recognition web app
> using Python, OpenCV, MediaPipe for hand-landmark extraction, and a
> scikit-learn classifier — deployed live with Gradio on Hugging Face
> Spaces."

This is 100% true, demonstrates real skills (CV, ML, deployment), and
you built every part of it yourself.

## Next steps to grow this project (after it's live)

Once the live link is working and on your resume, you can keep
improving it — each of these is a great "I kept iterating on it"
talking point in interviews:

- Add more signs/letters to increase vocabulary
- Try a neural network (`MLPClassifier` from scikit-learn) instead of
  Random Forest and compare accuracy
- Add confidence-based feedback ("show your hand more clearly")
- Eventually add motion-based word recognition (the LSTM approach) as
  a "v2" — mention this as a roadmap item in interviews, it shows
  you understand there's more depth beyond the MVP

## Tech Stack

Python · OpenCV · MediaPipe · scikit-learn (Random Forest) · Gradio ·
Hugging Face Spaces (deployment)
