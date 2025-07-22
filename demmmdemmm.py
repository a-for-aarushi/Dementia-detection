import numpy as np
import assemblyai as aai
import soundfile as sf
import librosa
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import tempfile
import os

# Set AssemblyAI API key
aai.settings.api_key = "API KEY"  # Replace with your API key

# Classification logic
def classify_cognitive_risk(pause_rate, repetition_rate, wpm, pitch_std, incomplete_rate):
    score = 0
    if pause_rate > 5: score += 1
    if repetition_rate > 5: score += 1
    if wpm < 100 or wpm > 190: score += 1
    if pitch_std < 30: score += 1
    if incomplete_rate > 50: score += 1

    if score <= 1:
        return "Low Risk"
    elif score in [2, 3]:
        return "Moderate Risk"
    else:
        return "High Risk"

# Analysis function
def analyze_audio(file_path):
    transcriber = aai.Transcriber()
    transcript = transcriber.transcribe(file_path)

    y, sr = sf.read(file_path)
    duration = len(y) / sr

    words = transcript.words
    hesitations = {'uh', 'um', 'hmm'}
    annotated = []
    prev_end = 0
    prev_phrases_2, prev_phrases_3 = [], []
    long_pauses = []
    hesitation_count = 0
    repetition_count = 0
    phrase_repetition_count = 0

    for i, word_obj in enumerate(words):
        word = word_obj.text.strip().lower()
        start = word_obj.start / 1000.0
        end = word_obj.end / 1000.0
        pause = start - prev_end

        if pause > 1.2:
            long_pauses.append(pause)
        if word in hesitations:
            hesitation_count += 1
        if i > 0 and word == words[i-1].text.strip().lower():
            repetition_count += 1
        if i > 1:
            phrase_2 = (words[i-2].text.strip().lower(), words[i-1].text.strip().lower())
            if phrase_2 in prev_phrases_2:
                phrase_repetition_count += 1
            prev_phrases_2.append(phrase_2)
            if len(prev_phrases_2) > 20:
                prev_phrases_2.pop(0)
        if i > 2:
            phrase_3 = (
                words[i-3].text.strip().lower(),
                words[i-2].text.strip().lower(),
                words[i-1].text.strip().lower()
            )
            if phrase_3 in prev_phrases_3:
                phrase_repetition_count += 1
            prev_phrases_3.append(phrase_3)
            if len(prev_phrases_3) > 20:
                prev_phrases_3.pop(0)

        annotated.append(word)
        prev_end = end

    total_words = len(words)
    wpm = (total_words / duration) * 60

    pitches, magnitudes = librosa.piptrack(y=y.T[0] if y.ndim > 1 else y, sr=sr)
    pitch_values = pitches[magnitudes > np.median(magnitudes)]
    pitch_std = np.std(pitch_values)

    annotated_text = " ".join(annotated)
    sentences = annotated_text.split(".")
    incomplete_sentence_count = sum(
        1 for s in sentences if s.strip() and not s.strip().endswith((".", "?", "!"))
    )

    total_pauses = len(long_pauses)
    pauses_per_minute = (total_pauses / duration) * 60
    repetition_rate = ((repetition_count + phrase_repetition_count) / total_words) * 100
    incomplete_rate = (incomplete_sentence_count / len(sentences)) * 100

    risk = classify_cognitive_risk(pauses_per_minute, repetition_rate, wpm, pitch_std, incomplete_rate)

    return {
        "pause_count": total_pauses,
        "pauses_per_minute": round(pauses_per_minute, 2),
        "repetition_rate": round(repetition_rate, 2),
        "speech_rate_wpm": round(wpm, 2),
        "pitch_variability_std": round(pitch_std, 2),
        "incomplete_sentence_rate": round(incomplete_rate, 2),
        "risk_level": risk
    }

# FastAPI app
app = FastAPI(title="Cognitive Risk Analysis API")

@app.post("/analyze")
async def analyze_voice(file: UploadFile = File(...)):
    try:
        # Save uploaded file to a temp location
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
            temp_audio.write(await file.read())
            temp_audio_path = temp_audio.name

        # Analyze the audio file
        result = analyze_audio(temp_audio_path)

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

    finally:
        if os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)

    return result
@app.get("/")
def home():
    return {"message": "Hello from FastAPI on Render!"}
