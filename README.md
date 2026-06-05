```markdown
# IELTS Speaking Assessor – AI-Powered Pronunciation & Fluency Evaluation

A complete, zero‑shot IELTS speaking assessment system that evaluates all four criteria using deep learning, acoustic feature engineering, and reinforcement learning.

- **Pronunciation** – Wav2Vec2, HuBERT, whisper + pitch, pause, energy analysis
- **Fluency & Coherence** – disfluency detection, speech rate, discourse markers
- **Lexical Resource** – vocabulary diversity, academic words, collocations, idioms
- **Grammatical Range & Accuracy** – error detection, complex sentences, structure variety

The system records your speech, transcribes it with OpenAI Whisper, extracts deep speech representations, and generates IELTS band scores (1–9) with detailed feedback. It also includes a **Reinforcement Learning (RL) agent** that learns from your assessments to improve over time.

---

## Features

- 🎤 **Real‑time recording** – press Enter to start/stop, auto‑normalisation
- 📝 **Automatic transcription** – using OpenAI Whisper (base model)
- 🧠 **Zero‑shot assessors** – no training data required; based on official IELTS band descriptors
- 🤖 **Reinforcement Learning** – DQN agent that calibrates scores using experience replay
- 📊 **Comprehensive reports** – scores, official descriptors, strengths, areas for improvement, and study recommendations
- 💾 **Model persistence** – saves RL model and calibration data for continuous improvement
- 📁 **Export to JSON / text** – all results saved with timestamps

---

## How It Works

1. **Record** – speak into your microphone (at least 10 seconds).
2. **Transcribe** – Whisper converts speech to text.
3. **Feature Engineering (FE)** – extract acoustic (pitch, pause ratio, energy) and text (lexical diversity, grammar patterns) features.
4. **Zero‑shot assessment** – rule‑based scoring using official IELTS descriptors.
5. **Reinforcement Learning** – the RL agent predicts bands and is trained to minimise error against FE scores.
6. **Feedback** – detailed report with band scores and actionable advice.

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/ielts-speaking-assessor.git
cd ielts-speaking-assessor
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. (Optional) Pre‑download NLTK data

The script will download it automatically, but you can do it manually:

```python
import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')
```

---

## Usage

Run the main script:

```bash
python transition.py
```

Then follow the interactive menu:

- **Option 1** – Record and assess (trains the RL model on your speech)
- **Option 2** – Assess an existing WAV file
- **Option 3** – Exit

The system will output:

- Band scores for each criterion (1–9)
- A text‑based visualisation (pitch range, intelligibility, pause ratio)
- A detailed feedback report with strengths, areas for improvement, and study recommendations
- Saved JSON and text files in the `reports/` folder (created automatically)

---

## Example Output

```
📊 PRONUNCIATION ANALYSIS:
  • Intelligibility: 78%
  • Pitch Range: 142 Hz
  • Pause Ratio: 16%
  • Composite Score: 0.812

🎯 IELTS BAND: 7/9

📋 OFFICIAL IELTS DESCRIPTOR:
• Generally clear pronunciation
• Minor issues don't affect understanding

✅ STRENGTHS:
  • High intelligibility - speech is clear
  • Good phonological range
  • Appropriate use of pauses

🎯 AREAS FOR IMPROVEMENT:
  • Increase pitch range (current: 142Hz, target: >150Hz for Band 8)

💡 PRACTICE RECOMMENDATIONS:
1. Practice sentence stress and rhythm
2. Work on connected speech features (linking, assimilation)
3. Record yourself with different emotions
4. Get feedback from native speakers
```

---

## Model Calibration & RL Training

- The system starts with zero‑shot rule‑based assessment.
- Each time you confirm that the assessment is accurate, the RL agent stores the experience and retrains incrementally.
- After ~2000 calibrations, the RL agent takes over 70% of the scoring weight, becoming highly personalised.
- The RL model is saved as `ielts_rl_model.pth` and loaded automatically on next run.

---

## Requirements

- Python 3.8+
- Microphone (for recording)
- ~4GB RAM (8GB recommended for transformer models)
- GPU optional (falls back to CPU)

---

## File Structure

```
.
├── transition.py          # Main script
├── requirements.txt       # Dependencies
├── README.md              # This file
├── reports/               # Generated assessment reports (created on run)
└── ielts_rl_model.pth     # Saved RL model (created after training)
```

---

## Limitations

- Requires a quiet environment for accurate pitch analysis.
- Lexical and grammar assessment is based on written transcripts; speaking naturally helps.
- The RL agent needs several calibrations to become accurate – use the “calibrate” option when it appears.

---

## Future Improvements

- Support for multiple languages (currently English only)
- Integration with a web frontend
- Fine‑tuned transformer for grammar error detection

---

## License

MIT – free to use, modify, and distribute.

---

## Author

**Johnny Kozman**  
[GitHub](https://github.com/JohnnyKozman2007)

---

*This tool is for educational and practice purposes. It is not an official IELTS certification.*
```
