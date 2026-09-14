import os
import shutil
import tempfile
import torch
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from infer import predict_speech_disorder

app = FastAPI(
    title="Pediatric Speech-Language Pathology AI",
    description="Age-Aware Feature Fusion (AAFF) Clinical Decision Support System",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

CHECKPOINT_PATH = "./results_controlled_baselines/model_c_aaff_best.pt"
MODEL_CHECKPOINT = "microsoft/wavlm-large"

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Pediatric SLP AI - Clinical Decision Support</title>
    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-primary: #0b0f19;
            --bg-card: rgba(18, 24, 38, 0.85);
            --border-color: rgba(255, 255, 255, 0.08);
            --accent-cyan: #06b6d4;
            --accent-blue: #3b82f6;
            --accent-indigo: #6366f1;
            --success-color: #10b981;
            --warning-color: #f59e0b;
            --danger-color: #ef4444;
            --text-main: #f3f4f6;
            --text-muted: #9ca3af;
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: 'Plus Jakarta Sans', sans-serif;
        }

        body {
            background-color: var(--bg-primary);
            background-image: 
                radial-gradient(at 0% 0%, rgba(59, 130, 246, 0.15) 0px, transparent 50%),
                radial-gradient(at 100% 100%, rgba(99, 102, 241, 0.12) 0px, transparent 50%);
            min-height: 100vh;
            color: var(--text-main);
            display: flex;
            flex-direction: column;
            align-items: center;
            padding: 2.5rem 1rem;
        }

        .container {
            width: 100%;
            max-width: 960px;
            display: flex;
            flex-direction: column;
            gap: 2rem;
        }

        header {
            text-align: center;
        }

        .badge-tag {
            display: inline-flex;
            align-items: center;
            gap: 0.5rem;
            background: rgba(59, 130, 246, 0.12);
            color: #60a5fa;
            border: 1px solid rgba(59, 130, 246, 0.3);
            border-radius: 9999px;
            padding: 0.35rem 1rem;
            font-size: 0.8rem;
            font-weight: 600;
            letter-spacing: 0.05em;
            text-transform: uppercase;
            margin-bottom: 1rem;
        }

        h1 {
            font-size: 2.4rem;
            font-weight: 800;
            background: linear-gradient(135deg, #ffffff 0%, #cbd5e1 50%, #94a3b8 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            letter-spacing: -0.02em;
            margin-bottom: 0.5rem;
        }

        .subtitle {
            color: var(--text-muted);
            font-size: 1.05rem;
            max-width: 650px;
            margin: 0 auto;
            line-height: 1.6;
        }

        .main-card {
            background: var(--bg-card);
            backdrop-filter: blur(16px);
            border: 1px solid var(--border-color);
            border-radius: 1.25rem;
            padding: 2rem;
            box-shadow: 0 20px 40px -15px rgba(0, 0, 0, 0.5);
        }

        .grid-2 {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 1.5rem;
        }

        @media (max-width: 768px) {
            .grid-2 { grid-template-columns: 1fr; }
        }

        .input-group {
            display: flex;
            flex-direction: column;
            gap: 0.6rem;
        }

        label {
            font-size: 0.9rem;
            font-weight: 600;
            color: #e5e7eb;
            display: flex;
            justify-content: space-between;
        }

        .upload-zone {
            border: 2px dashed rgba(255, 255, 255, 0.15);
            border-radius: 0.85rem;
            padding: 1.75rem 1rem;
            text-align: center;
            cursor: pointer;
            transition: all 0.2s ease;
            background: rgba(255, 255, 255, 0.02);
            position: relative;
        }

        .upload-zone:hover, .upload-zone.dragover {
            border-color: var(--accent-cyan);
            background: rgba(6, 182, 212, 0.05);
        }

        .upload-icon {
            font-size: 2rem;
            margin-bottom: 0.5rem;
        }

        .upload-text {
            font-size: 0.9rem;
            color: var(--text-muted);
        }

        .upload-filename {
            color: var(--accent-cyan);
            font-weight: 600;
            margin-top: 0.5rem;
            font-size: 0.95rem;
            word-break: break-all;
        }

        input[type="file"] {
            display: none;
        }

        .range-container {
            background: rgba(255, 255, 255, 0.03);
            border: 1px solid var(--border-color);
            border-radius: 0.85rem;
            padding: 1.25rem;
            display: flex;
            flex-direction: column;
            gap: 1rem;
        }

        .range-display {
            display: flex;
            justify-content: space-between;
            align-items: baseline;
        }

        .age-val {
            font-size: 1.8rem;
            font-weight: 800;
            color: var(--accent-cyan);
        }

        input[type="range"] {
            -webkit-appearance: none;
            width: 100%;
            height: 6px;
            background: rgba(255, 255, 255, 0.1);
            border-radius: 3px;
            outline: none;
        }

        input[type="range"]::-webkit-slider-thumb {
            -webkit-appearance: none;
            width: 20px;
            height: 20px;
            border-radius: 50%;
            background: var(--accent-cyan);
            cursor: pointer;
            box-shadow: 0 0 10px rgba(6, 182, 212, 0.5);
            transition: transform 0.1s ease;
        }

        input[type="range"]::-webkit-slider-thumb:hover {
            transform: scale(1.15);
        }

        .sample-buttons {
            display: flex;
            gap: 0.6rem;
            flex-wrap: wrap;
            margin-top: 0.5rem;
        }

        .btn-sample {
            background: rgba(255, 255, 255, 0.06);
            border: 1px solid var(--border-color);
            color: var(--text-main);
            border-radius: 0.5rem;
            padding: 0.45rem 0.85rem;
            font-size: 0.8rem;
            font-weight: 500;
            cursor: pointer;
            transition: all 0.2s ease;
        }

        .btn-sample:hover {
            background: rgba(59, 130, 246, 0.2);
            border-color: var(--accent-blue);
        }

        .btn-submit {
            background: linear-gradient(135deg, var(--accent-blue), var(--accent-indigo));
            color: #ffffff;
            border: none;
            border-radius: 0.85rem;
            padding: 1rem 2rem;
            font-size: 1.05rem;
            font-weight: 700;
            cursor: pointer;
            width: 100%;
            margin-top: 1.5rem;
            box-shadow: 0 10px 20px -5px rgba(59, 130, 246, 0.4);
            transition: all 0.2s ease;
            display: flex;
            justify-content: center;
            align-items: center;
            gap: 0.5rem;
        }

        .btn-submit:hover {
            transform: translateY(-2px);
            box-shadow: 0 15px 25px -5px rgba(59, 130, 246, 0.5);
        }

        .btn-submit:disabled {
            opacity: 0.5;
            cursor: not-allowed;
            transform: none;
        }

        .result-panel {
            margin-top: 2rem;
            padding: 1.75rem;
            border-radius: 1rem;
            border: 1px solid var(--border-color);
            background: rgba(255, 255, 255, 0.02);
            display: none;
            animation: fadeIn 0.3s ease forwards;
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }

        .prediction-badge {
            display: inline-flex;
            align-items: center;
            padding: 0.5rem 1.25rem;
            border-radius: 9999px;
            font-size: 1.1rem;
            font-weight: 800;
            letter-spacing: 0.02em;
            margin-bottom: 1rem;
        }

        .badge-disorder {
            background: rgba(239, 68, 68, 0.15);
            color: #f87171;
            border: 1px solid rgba(239, 68, 68, 0.4);
        }

        .badge-typical {
            background: rgba(16, 185, 129, 0.15);
            color: #34d399;
            border: 1px solid rgba(16, 185, 129, 0.4);
        }

        .progress-bar-container {
            width: 100%;
            height: 10px;
            background: rgba(255, 255, 255, 0.08);
            border-radius: 5px;
            overflow: hidden;
            margin: 0.5rem 0 1.25rem;
        }

        .progress-fill {
            height: 100%;
            border-radius: 5px;
            transition: width 0.6s cubic-bezier(0.4, 0, 0.2, 1);
        }

        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 1rem;
            margin-top: 1rem;
        }

        .stat-box {
            background: rgba(0, 0, 0, 0.25);
            border: 1px solid var(--border-color);
            border-radius: 0.75rem;
            padding: 1rem;
        }

        .stat-box .title {
            font-size: 0.75rem;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 0.25rem;
        }

        .stat-box .val {
            font-size: 1.15rem;
            font-weight: 700;
            color: #ffffff;
        }

        .spinner {
            border: 3px solid rgba(255, 255, 255, 0.1);
            border-top: 3px solid #ffffff;
            border-radius: 50%;
            width: 20px;
            height: 20px;
            animation: spin 0.8s linear infinite;
        }

        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }

        footer {
            color: var(--text-muted);
            font-size: 0.85rem;
            text-align: center;
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <div class="badge-tag">Clinical Decision Support AI</div>
            <h1>Pediatric Speech Pathology Classifier</h1>
            <p class="subtitle">
                Age-Aware Feature Fusion (AAFF) on frozen WavLM-Large representations for automated pediatric speech disorder screening.
            </p>
        </header>

        <main class="main-card">
            <form id="predictionForm" onsubmit="submitForm(event)">
                <div class="grid-2">
                    <!-- Audio Input Column -->
                    <div class="input-group">
                        <label>
                            <span>Child Voice Recording (.wav)</span>
                            <span style="font-weight: 400; color: var(--text-muted);">16 kHz Mono</span>
                        </label>
                        <div class="upload-zone" id="dropZone" onclick="document.getElementById('audioInput').click()">
                            <div class="upload-icon">🎙️</div>
                            <div class="upload-text">Click or drag & drop a .wav recording here</div>
                            <div class="upload-filename" id="fileNameDisplay"></div>
                            <input type="file" id="audioInput" name="audio" accept=".wav,audio/wav" onchange="handleFileSelect(this)">
                        </div>

                        <div>
                            <span style="font-size: 0.8rem; color: var(--text-muted);">Or select generated test audio:</span>
                            <div class="sample-buttons">
                                <button type="button" class="btn-sample" onclick="selectSample('sample_child_speech.wav')">Sample Voice 1 (3.5s)</button>
                                <button type="button" class="btn-sample" onclick="selectSample('sample_child_speech_alt.wav')">Sample Voice 2 (4.0s)</button>
                            </div>
                        </div>
                    </div>

                    <!-- Age & Model Config Column -->
                    <div class="input-group">
                        <label>
                            <span>Child Age (Chronological)</span>
                            <span style="font-weight: 400; color: var(--text-muted);">Training Bounds: [5.67 - 12.83]</span>
                        </label>
                        <div class="range-container">
                            <div class="range-display">
                                <span class="age-val" id="ageDisplay">7.50</span>
                                <span style="font-size: 0.95rem; color: var(--text-muted);">years old</span>
                            </div>
                            <input type="range" id="ageInput" name="age" min="5.0" max="14.0" step="0.1" value="7.5" oninput="updateAge(this.value)">
                            <div style="display: flex; justify-content: space-between; font-size: 0.75rem; color: var(--text-muted);">
                                <span>5.0 yrs</span>
                                <span>Norm Anchor: 5.67</span>
                                <span>14.0 yrs</span>
                            </div>
                        </div>

                        <div class="stat-box" style="margin-top: 0.5rem;">
                            <div class="title">Active Trained Checkpoint</div>
                            <div class="val" style="font-size: 0.9rem; font-family: monospace; color: var(--accent-cyan);">
                                model_c_aaff_best.pt
                            </div>
                        </div>
                    </div>
                </div>

                <button type="submit" class="btn-submit" id="submitBtn">
                    <span id="btnText">Run AAFF Neural Inference</span>
                    <div id="btnSpinner" class="spinner" style="display: none;"></div>
                </button>
            </form>

            <!-- Results Panel -->
            <div class="result-panel" id="resultPanel">
                <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap;">
                    <div>
                        <div style="font-size: 0.8rem; color: var(--text-muted); text-transform: uppercase; margin-bottom: 0.35rem;">Diagnostic Classification</div>
                        <div class="prediction-badge" id="predictionBadge">SPEECH DISORDER</div>
                    </div>
                    <div style="text-align: right;">
                        <div style="font-size: 0.8rem; color: var(--text-muted); text-transform: uppercase;">Confidence</div>
                        <div style="font-size: 1.8rem; font-weight: 800; color: #ffffff;" id="confidenceDisplay">0.00%</div>
                    </div>
                </div>

                <div style="margin-top: 0.75rem;">
                    <div style="display: flex; justify-content: space-between; font-size: 0.85rem; font-weight: 600;">
                        <span style="color: #34d399;">Typically Developing: <span id="probTypical">0%</span></span>
                        <span style="color: #f87171;">Speech Disorder: <span id="probDisorder">0%</span></span>
                    </div>
                    <div class="progress-bar-container">
                        <div class="progress-fill" id="progressFill" style="width: 50%; background: var(--accent-blue);"></div>
                    </div>
                </div>

                <div class="stats-grid">
                    <div class="stat-box">
                        <div class="title">Audio Duration</div>
                        <div class="val" id="durationDisplay">0.00s</div>
                    </div>
                    <div class="stat-box">
                        <div class="title">Normalized Age</div>
                        <div class="val" id="normAgeDisplay">0.0000</div>
                    </div>
                    <div class="stat-box">
                        <div class="title">WavLM Backbone</div>
                        <div class="val" style="font-size: 0.85rem;">1024-d Frozen</div>
                    </div>
                    <div class="stat-box">
                        <div class="title">Inference Device</div>
                        <div class="val" id="deviceDisplay">CPU</div>
                    </div>
                </div>
            </div>
        </main>

        <footer>
            Pediatric Speech-Language Pathology Research • AAFF Architecture • Group 8
        </footer>
    </div>

    <script>
        let selectedAudioFile = null;
        let selectedSampleName = null;

        function updateAge(val) {
            document.getElementById('ageDisplay').innerText = parseFloat(val).toFixed(2);
        }

        function handleFileSelect(input) {
            if (input.files && input.files[0]) {
                selectedAudioFile = input.files[0];
                selectedSampleName = null;
                document.getElementById('fileNameDisplay').innerText = "Selected: " + selectedAudioFile.name;
            }
        }

        function selectSample(name) {
            selectedAudioFile = null;
            selectedSampleName = name;
            document.getElementById('fileNameDisplay').innerText = "Selected Pre-loaded Sample: " + name;
        }

        // Drag & drop support
        const dropZone = document.getElementById('dropZone');
        ['dragenter', 'dragover'].forEach(eventName => {
            dropZone.addEventListener(eventName, (e) => {
                e.preventDefault();
                dropZone.classList.add('dragover');
            }, false);
        });
        ['dragleave', 'drop'].forEach(eventName => {
            dropZone.addEventListener(eventName, (e) => {
                e.preventDefault();
                dropZone.classList.remove('dragover');
            }, false);
        });
        dropZone.addEventListener('drop', (e) => {
            const dt = e.dataTransfer;
            if (dt.files && dt.files[0]) {
                selectedAudioFile = dt.files[0];
                selectedSampleName = null;
                document.getElementById('fileNameDisplay').innerText = "Dropped: " + selectedAudioFile.name;
            }
        });

        async function submitForm(e) {
            e.preventDefault();
            const submitBtn = document.getElementById('submitBtn');
            const btnText = document.getElementById('btnText');
            const btnSpinner = document.getElementById('btnSpinner');
            const age = document.getElementById('ageInput').value;

            if (!selectedAudioFile && !selectedSampleName) {
                alert("Please select a .wav audio file or click one of the sample voice buttons.");
                return;
            }

            submitBtn.disabled = true;
            btnText.innerText = "Extracting Features & Running AAFF...";
            btnSpinner.style.display = "block";

            try {
                const formData = new FormData();
                formData.append('age', age);
                if (selectedAudioFile) {
                    formData.append('file', selectedAudioFile);
                } else {
                    formData.append('sample_name', selectedSampleName);
                }

                const response = await fetch('/predict', {
                    method: 'POST',
                    body: formData
                });

                if (!response.ok) {
                    const err = await response.json();
                    throw new Error(err.detail || "Inference failed");
                }

                const res = await response.json();

                // Display result panel
                const panel = document.getElementById('resultPanel');
                panel.style.display = 'block';

                const badge = document.getElementById('predictionBadge');
                badge.innerText = res.prediction.toUpperCase();
                if (res.prediction === "Speech Disorder") {
                    badge.className = "prediction-badge badge-disorder";
                    document.getElementById('progressFill').style.background = "var(--danger-color)";
                    document.getElementById('progressFill').style.width = res.confidence + "%";
                } else {
                    badge.className = "prediction-badge badge-typical";
                    document.getElementById('progressFill').style.background = "var(--success-color)";
                    document.getElementById('progressFill').style.width = res.confidence + "%";
                }

                document.getElementById('confidenceDisplay').innerText = res.confidence.toFixed(2) + "%";
                document.getElementById('probTypical').innerText = res.prob_typical.toFixed(1) + "%";
                document.getElementById('probDisorder').innerText = res.prob_disorder.toFixed(1) + "%";
                document.getElementById('durationDisplay').innerText = res.duration.toFixed(2) + "s";
                document.getElementById('normAgeDisplay').innerText = res.normalized_age.toFixed(4);
                document.getElementById('deviceDisplay').innerText = res.device.toUpperCase();

                panel.scrollIntoView({ behavior: 'smooth' });
            } catch (err) {
                alert("Error during model inference: " + err.message);
            } finally {
                submitBtn.disabled = false;
                btnText.innerText = "Run AAFF Neural Inference";
                btnSpinner.style.display = "none";
            }
        }
    </script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
async def index():
    return HTMLResponse(content=HTML_TEMPLATE)

@app.get("/samples/{filename}")
async def get_sample(filename: str):
    if not os.path.exists(filename):
        raise HTTPException(status_code=404, detail="Sample file not found")
    return FileResponse(filename, media_type="audio/wav")

@app.post("/predict")
async def predict(
    age: float = Form(...),
    file: UploadFile = File(None),
    sample_name: str = Form(None)
):
    temp_path = None
    try:
        if file and file.filename:
            suffix = os.path.splitext(file.filename)[1] or ".wav"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                shutil.copyfileobj(file.file, tmp)
                temp_path = tmp.name
            target_audio = temp_path
        elif sample_name:
            if not os.path.exists(sample_name):
                raise HTTPException(status_code=400, detail=f"Sample file {sample_name} not found")
            target_audio = sample_name
        else:
            raise HTTPException(status_code=400, detail="No audio file or sample provided")

        device_name = "cuda" if torch.cuda.is_available() else "cpu"
        result = predict_speech_disorder(
            audio_path=target_audio,
            age=age,
            checkpoint_path=CHECKPOINT_PATH,
            model_checkpoint=MODEL_CHECKPOINT
        )
        safe_result = {
            k: (float(v) if hasattr(v, "item") or isinstance(v, float) else v)
            for k, v in result.items()
        }
        safe_result["device"] = device_name
        return JSONResponse(content=safe_result)

    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
