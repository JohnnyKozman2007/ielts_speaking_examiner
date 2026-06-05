import whisper
import librosa
import numpy as np
from scipy import signal
import warnings
warnings.filterwarnings('ignore')
import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio
from transformers import Wav2Vec2Processor, Wav2Vec2Model
import librosa
import sounddevice as sd
import wave
import tempfile
import os
from datetime import datetime
import json
from pathlib import Path
import re
import statistics
from collections import defaultdict, deque
from typing import Dict, List, Tuple, Any
from nltk import word_tokenize, pos_tag
from nltk.corpus import stopwords
import nltk
import random

# Download NLTK data if not present
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('taggers/averaged_perceptron_tagger')
    nltk.data.find('corpora/stopwords')
except:
    import ssl
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context
    nltk.download('punkt', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
    nltk.download('stopwords', quiet=True)

# ========================
# FIX JAVA PATH (ADD THIS FIRST!)
# ========================
java_path = r"C:\Program Files\Eclipse Adoptium\jdk-25.0.1.8-hotspot\bin"
java_home = r"C:\Program Files\Eclipse Adoptium\jdk-25.0.1.8-hotspot"

if java_path not in os.environ['PATH']:
    os.environ['PATH'] = java_path + ';' + os.environ['PATH']
os.environ['JAVA_HOME'] = java_home

# ========================
# GLOBAL CONFIGURATION
# ========================
WHISPER_MODEL = None  # Will be cached
SAMPLE_RATE = 16000
OPTIMAL_PAUSE_MIN = 0.10  # 10% minimum pauses
OPTIMAL_PAUSE_MAX = 0.20  # 20% maximum pauses

# ========================
# REINFORCEMENT LEARNING CONFIG
# ========================
RL_STATE_DIM = 28  # Features from FE approach
RL_ACTION_DIM = 4   # 4 IELTS criteria
RL_HIDDEN_DIM = 256
RL_LEARNING_RATE = 0.001
RL_GAMMA = 0.99
RL_MEMORY_SIZE = 1000
RL_BATCH_SIZE = 32
RL_TARGET_UPDATE = 10
RL_EPSILON_START = 1.0
RL_EPSILON_END = 0.01
RL_EPSILON_DECAY = 0.995

# ========================
# CACHE WHISPER MODEL
# ========================
def get_whisper_model():
    """Cache Whisper model to avoid reloading"""
    global WHISPER_MODEL
    if WHISPER_MODEL is None:
        print("🔄 Loading Whisper model...")
        WHISPER_MODEL = whisper.load_model("base")
        print("✅ Whisper model loaded")
    return WHISPER_MODEL

# ========================
# AUDIO PRE-PROCESSING
# ========================
def normalize_audio(audio):
    """Normalize audio volume to standard level"""
    if len(audio) == 0:
        return audio
    
    # Normalize to [-1, 1] range
    max_val = np.max(np.abs(audio))
    if max_val > 0:
        audio = audio / max_val
    
    # Apply gentle compression (soft limit)
    threshold = 0.7
    audio = np.where(np.abs(audio) > threshold, 
                    np.sign(audio) * (threshold + (np.abs(audio) - threshold) * 0.5),
                    audio)
    
    return audio

def remove_silence(audio, sr):
    """Remove excessive leading/trailing silence"""
    # Calculate energy
    energy = librosa.feature.rms(y=audio)[0]
    threshold = np.percentile(energy, 10)
    
    # Find first and last non-silent frames
    non_silent = np.where(energy > threshold)[0]
    if len(non_silent) == 0:
        return audio
    
    # Convert frames to samples
    hop_length = len(audio) // len(energy)
    start_sample = max(0, non_silent[0] * hop_length - int(0.1 * sr))  # Keep 100ms before
    end_sample = min(len(audio), non_silent[-1] * hop_length + int(0.1 * sr))  # Keep 100ms after
    
    return audio[start_sample:end_sample]

# ========================
# RECORDING FUNCTION
# ========================
def record_and_transcribe():
    """
    Record audio and transcribe accurately
    Press Enter to start, Enter to stop
    Returns: (wav_filename, transcript_text)
    """
    
    print("🎤 VOICE RECORDER")
    print("-" * 40)
    input("Press ENTER to start recording...")
    
    sample_rate = SAMPLE_RATE
    is_recording = True
    audio_chunks = []
    
    def audio_callback(indata, frames, time_info, status):
        """Record audio chunks"""
        if is_recording:
            audio_chunks.append(indata.copy())
            duration = len(np.concatenate(audio_chunks)) / sample_rate
            print(f"\r⏺️ Recording: {duration:.1f} seconds", end='')
    
    print("\n🎤 Recording... Speak clearly. Press ENTER to stop.")
    print("-" * 40)
    
    stream = sd.InputStream(
        callback=audio_callback,
        channels=1,
        samplerate=sample_rate,
        blocksize=int(sample_rate * 0.5)
    )
    
    stream.start()
    input()
    is_recording = False
    stream.stop()
    stream.close()
    
    print("\n\n⏹️ Stopped recording")
    
    if not audio_chunks:
        print("❌ No audio recorded")
        return None, ""
    
    full_audio = np.concatenate(audio_chunks, axis=0)
    duration = len(full_audio) / sample_rate
    
    if duration < 1.0:
        print(f"⚠️  Recording too short: {duration:.1f}s (minimum 1.0s)")
        return None, ""
    
    # Normalize audio
    full_audio = normalize_audio(full_audio)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    wav_filename = f"recording_{timestamp}.wav"
    audio_int16 = np.int16(full_audio * 32767)
    
    with wave.open(wav_filename, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio_int16.tobytes())
    
    print(f"✅ Audio saved: {wav_filename}")
    print(f"⏱️  Duration: {duration:.1f} seconds")
    
    # Transcribe
    print("\n🎯 Transcribing...")
    
    try:
        model = get_whisper_model()
        
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_wav = tmp.name
            with wave.open(tmp_wav, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(sample_rate)
                wf.writeframes(audio_int16.tobytes())
        
        result = model.transcribe(tmp_wav, language="en")
        full_transcript = result["text"].strip()
        
        os.unlink(tmp_wav)
        
    except Exception as e:
        print(f"❌ Transcription error: {e}")
        full_transcript = ""
    
    if full_transcript:
        transcript_filename = f"transcript_{timestamp}.txt"
        with open(transcript_filename, 'w', encoding='utf-8') as f:
            f.write(full_transcript)
        
        print(f"📝 Transcript saved: {transcript_filename}")
        print("\n" + "=" * 40)
        print("✅ TRANSCRIPTION COMPLETE")
        print("=" * 40)
        print(f"\n📄 TRANSCRIPT:")
        print("-" * 40)
        print(full_transcript)
        print("-" * 40)
    else:
        print("⚠️  Could not transcribe audio")
    
    print(f"\n🎯 Done! Returned: WAV file and transcript text")
    return wav_filename, full_transcript

# ========================
# PRONUNCIATION ASSESSMENT FUNCTIONS
# ========================
def calculate_pause_score(pause_ratio):
    """Score pauses: optimal 10-20%, penalize both too few and too many"""
    if pause_ratio < OPTIMAL_PAUSE_MIN:
        # Too few pauses (rushing speech)
        return pause_ratio / OPTIMAL_PAUSE_MIN
    elif pause_ratio > OPTIMAL_PAUSE_MAX:
        # Too many pauses (choppy speech)
        max_penalty = 0.4  # Maximum pause ratio we consider
        if pause_ratio >= max_penalty:
            return 0.0
        else:
            return 1.0 - ((pause_ratio - OPTIMAL_PAUSE_MAX) / (max_penalty - OPTIMAL_PAUSE_MAX))
    else:
        # Perfect amount of pauses
        return 1.0

def get_ielts_descriptors():
    """Return IELTS pronunciation descriptors for each band"""
    return {
        9: "• Uses a full range of phonological features\n• Flexible use of connected speech\n• Can be effortlessly understood\n• Accent has no effect on intelligibility",
        8: "• Uses a wide range of phonological features\n• Flexible use of stress and intonation\n• Can be easily understood\n• Accent has minimal effect",
        7: "• Shows all positive features of Band 6 and some of Band 8\n• Generally clear pronunciation",
        6: "• Uses a range of phonological features, but control is variable\n• Individual words may be mispronounced but rarely impede communication",
        5: "• Displays features of Band 4 and some of Band 6\n• Basic pronunciation with noticeable issues",
        4: "• Uses some acceptable phonological features, but range is limited\n• Frequent mispronunciation causing lack of clarity",
        3: "• Limited phonological features\n• Pronunciation significantly affects intelligibility",
        2: "• Very limited phonological features\n• Speech is often unintelligible",
        1: "• Essentially no phonological features\n• Speech is totally incoherent"
    }

def generate_detailed_feedback(band, features, composite_score, transcript, duration):
    """Generate comprehensive feedback report"""
    
    words = transcript.split()
    wpm = len(words) / (duration / 60) if duration > 0 else 0
    
    feedback = f"🎯 IELTS PRONUNCIATION ASSESSMENT REPORT\n"
    feedback += "=" * 70 + "\n\n"
    
    # Band Score
    feedback += f"📊 OVERALL BAND SCORE: {band}/9\n\n"
    
    # IELTS Descriptors
    descriptors = get_ielts_descriptors()
    feedback += "📋 IELTS CRITERIA MATCHED:\n"
    feedback += descriptors.get(band, "") + "\n\n"
    
    # Detailed Metrics
    feedback += "🔍 DETAILED ANALYSIS:\n"
    feedback += f"  • Intelligibility: {features.get('intelligibility', 0):.1%} (words understood)\n"
    feedback += f"  • Pitch Range: {features.get('pitch_range', 0):.0f} Hz (optimal >100Hz)\n"
    feedback += f"  • Pitch Variation: {features.get('pitch_variation', 0):.3f} (higher = more expressive)\n"
    feedback += f"  • Pause Ratio: {features.get('pause_ratio', 0):.1%} (optimal: 10-20%)\n"
    feedback += f"  • Stress Variation: {features.get('energy_variance', 0):.4f} (higher = better stress)\n"
    feedback += f"  • Speaking Rate: {wpm:.0f} WPM (optimal: 120-180)\n"
    feedback += f"  • Rhythm Regularity: {features.get('rhythm_regularity', 0):.3f}\n"
    feedback += f"  • Composite Score: {composite_score:.3f}/1.0\n\n"
    
    # Strengths
    feedback += "✅ STRENGTHS:\n"
    strengths = []
    
    if features.get('intelligibility', 0) > 0.8:
        strengths.append("High intelligibility - speech is clear")
    if features.get('pitch_range', 0) > 100:
        strengths.append("Good phonological range")
    if features.get('pitch_variation', 0) > 0.2:
        strengths.append("Effective intonation variation")
    if OPTIMAL_PAUSE_MIN <= features.get('pause_ratio', 0) <= OPTIMAL_PAUSE_MAX:
        strengths.append("Appropriate use of pauses")
    if features.get('energy_variance', 0) > 0.002:
        strengths.append("Good word stress patterns")
    if 120 <= wpm <= 180:
        strengths.append("Natural speaking pace")
    
    if strengths:
        for strength in strengths:
            feedback += f"  • {strength}\n"
    else:
        feedback += "  • Basic pronunciation achieved\n"
    feedback += "\n"
    
    # Areas for Improvement
    feedback += "🎯 AREAS FOR IMPROVEMENT:\n"
    improvements = []
    
    if band < 9:
        if features.get('pitch_range', 0) < 100:
            improvements.append(f"Increase pitch range (current: {features.get('pitch_range', 0):.0f}Hz, target: >100Hz)")
        if features.get('pitch_variation', 0) < 0.15:
            improvements.append("Practice varying your intonation more")
        if features.get('pause_ratio', 0) < OPTIMAL_PAUSE_MIN:
            improvements.append("Add more natural pauses for better rhythm")
        elif features.get('pause_ratio', 0) > OPTIMAL_PAUSE_MAX:
            improvements.append("Reduce unnecessary pauses")
        if features.get('energy_variance', 0) < 0.001:
            improvements.append("Work on emphasizing important words")
        if wpm < 100:
            improvements.append("Increase speaking rate slightly")
        elif wpm > 200:
            improvements.append("Slow down for better clarity")
        if features.get('intelligibility', 0) < 0.7:
            improvements.append("Focus on clearer articulation of individual sounds")
    
    if improvements:
        for improvement in improvements:
            feedback += f"  • {improvement}\n"
    else:
        feedback += "  • Maintain current pronunciation quality\n"
    feedback += "\n"
    
    # Practice Recommendations
    feedback += "💡 PRACTICE RECOMMENDATIONS:\n"
    
    if band < 7:
        feedback += "1. Record and listen to yourself daily\n"
        feedback += "2. Practice minimal pairs (ship/sheep, bat/bet)\n"
        feedback += "3. Use 'shadowing' technique with native speakers\n"
        feedback += "4. Focus on word stress patterns\n"
    elif band < 9:
        feedback += "1. Practice sentence stress and rhythm\n"
        feedback += "2. Work on connected speech features (linking, assimilation)\n"
        feedback += "3. Record yourself with different emotions\n"
        feedback += "4. Get feedback from native speakers\n"
    else:
        feedback += "1. Maintain your excellent pronunciation\n"
        feedback += "2. Focus on accent reduction if desired\n"
        feedback += "3. Practice public speaking in English\n"
    
    # Confidence Score
    confidence = min(0.95, 0.7 + (composite_score * 0.25))
    feedback += f"\n📈 ASSESSMENT CONFIDENCE: {confidence:.1%}"
    
    return feedback

def visualize_features(features, band):
    """Create simple text visualization of features"""
    print("\n📊 PRONUNCIATION FEATURE VISUALIZATION:")
    print("-" * 50)
    
    # Pitch Range
    pitch = features.get('pitch_range', 0)
    pitch_bar = "█" * int(pitch / 10) + " " * (20 - int(pitch / 10))
    print(f"Pitch Range:    [{pitch_bar}] {pitch:.0f} Hz")
    
    # Intelligibility
    intel = features.get('intelligibility', 0)
    intel_bar = "█" * int(intel * 20) + " " * (20 - int(intel * 20))
    print(f"Intelligibility:[{intel_bar}] {intel:.1%}")
    
    # Pause Ratio
    pause = features.get('pause_ratio', 0)
    pause_pos = int((pause / 0.4) * 20)
    pause_bar = [" "] * 20
    if 0 <= pause_pos < 20:
        pause_bar[pause_pos] = "↑"
    pause_str = "".join(pause_bar)
    print(f"Pause Ratio:    [{pause_str}] {pause:.1%}")
    print(f"                {'10%':<5}{'20%':^10}{'30%':>5} (optimal)")
    
    # Band Indicator
    print(f"\n🎯 IELTS BAND: {band}/9")
    band_bar = ["○"] * 9
    band_bar[band-1] = "●"
    print(f"Band Scale:     [{' '.join(band_bar)}]")
    print(f"                1 2 3 4 5 6 7 8 9")

def assess_ielts_pronunciation(wav_file_path, return_detailed=False):
    """
    Comprehensive IELTS pronunciation assessment
    Returns band score (1-9) or detailed report
    """
    
    try:
        # 1. LOAD AND PRE-PROCESS AUDIO
        audio, sample_rate = librosa.load(wav_file_path, sr=SAMPLE_RATE)
        duration = len(audio) / sample_rate
        
        if duration < 1.0:
            print(f"⚠️  Audio too short: {duration:.1f}s (need at least 1.0s)")
            if return_detailed:
                return 4, {"error": "Audio too short", "duration": duration}
            return 4
        
        # Remove excessive silence and normalize
        audio = remove_silence(audio, sample_rate)
        audio = normalize_audio(audio)
        duration = len(audio) / sample_rate
        
        # 2. WHISPER INTELLIGIBILITY
        model = get_whisper_model()
        result = model.transcribe(wav_file_path)
        transcript = result["text"].strip()
        words = transcript.split()
        word_count = len(words)
        
        # Intelligibility score
        expected_words = max(1, duration * 2.5)
        word_ratio = min(1.0, word_count / expected_words)
        
        # 3. PITCH FEATURES
        pitches, voiced_flags, voiced_probs = librosa.pyin(
            audio, 
            fmin=librosa.note_to_hz('C2'), 
            fmax=librosa.note_to_hz('C7'),
            sr=sample_rate
        )
        
        valid_pitches = pitches[~np.isnan(pitches)]
        
        if len(valid_pitches) > 10:
            pitch_min = np.min(valid_pitches)
            pitch_max = np.max(valid_pitches)
            pitch_range = pitch_max - pitch_min
            pitch_mean = np.mean(valid_pitches)
            pitch_std = np.std(valid_pitches)
            pitch_variation = pitch_std / (pitch_mean + 1e-10)
        else:
            pitch_range = 0
            pitch_std = 0
            pitch_variation = 0
        
        # 4. RHYTHM & PAUSE FEATURES
        frame_length = int(0.025 * sample_rate)
        hop_length = int(0.01 * sample_rate)
        rms_energy = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]
        
        silence_threshold = np.percentile(rms_energy, 20)
        is_silence = rms_energy < silence_threshold
        pause_ratio = np.sum(is_silence) / len(is_silence)
        
        # Rhythm regularity
        energy_acorr = np.correlate(rms_energy - np.mean(rms_energy), 
                                   rms_energy - np.mean(rms_energy), mode='full')
        energy_acorr = energy_acorr[len(energy_acorr)//2:]
        rhythm_regularity = np.mean(energy_acorr[:20]) if len(energy_acorr) > 20 else 0
        
        # 5. STRESS & INTONATION
        energy_std = np.std(rms_energy)
        energy_variance = np.var(rms_energy)
        
        # 6. SPEAKING RATE
        onset_env = librosa.onset.onset_strength(y=audio, sr=sample_rate)
        pulse = librosa.beat.plp(onset_envelope=onset_env, sr=sample_rate)
        peaks = signal.find_peaks(pulse)[0]
        
        if duration > 0:
            syllable_rate = len(peaks) / duration
            syllable_rate_score = min(1.0, syllable_rate / 5.0)
        else:
            syllable_rate_score = 0
        
        # 7. COMPOSITE SCORE
        weights = {
            'intelligibility': 0.35,
            'pitch_range': 0.20,
            'pitch_variation': 0.15,
            'pause_ratio': 0.10,
            'energy_variance': 0.10,
            'speaking_rate': 0.05,
            'rhythm_regularity': 0.05
        }
        
        normalized_features = {
            'intelligibility': min(1.0, word_ratio),
            'pitch_range': min(1.0, pitch_range / 200),
            'pitch_variation': min(1.0, pitch_variation * 10),
            'pause_ratio': calculate_pause_score(pause_ratio),
            'energy_variance': min(1.0, energy_variance * 100),
            'speaking_rate': syllable_rate_score,
            'rhythm_regularity': min(1.0, rhythm_regularity * 1000)
        }
        
        composite_score = 0
        for feature, weight in weights.items():
            composite_score += normalized_features[feature] * weight
        
        # 8. MAP TO IELTS BANDS
        raw_features = {
            'intelligibility': word_ratio,
            'pitch_range': pitch_range,
            'pitch_variation': pitch_variation,
            'pause_ratio': pause_ratio,
            'energy_variance': energy_variance,
            'speaking_rate': syllable_rate,
            'rhythm_regularity': rhythm_regularity
        }
        
        if composite_score >= 0.95:
            if pitch_range >= 150 and word_ratio >= 0.9 and calculate_pause_score(pause_ratio) >= 0.9:
                band = 9
            else:
                band = 8
        elif composite_score >= 0.85:
            if pitch_range >= 120 and word_ratio >= 0.85:
                band = 8
            else:
                band = 7
        elif composite_score >= 0.75:
            if word_ratio >= 0.75:
                band = 7
            else:
                band = 6
        elif composite_score >= 0.65:
            band = 6
        elif composite_score >= 0.55:
            band = 5
        elif composite_score >= 0.45:
            band = 4
        elif composite_score >= 0.35:
            band = 3
        elif composite_score >= 0.25:
            band = 2
        else:
            band = 1
        
        # 9. DIAGNOSTIC OUTPUT
        print(f"\n📊 PRONUNCIATION ANALYSIS:")
        print(f"  • Intelligibility: {word_ratio:.1%}")
        print(f"  • Pitch Range: {pitch_range:.0f} Hz")
        print(f"  • Pause Ratio: {pause_ratio:.1%}")
        print(f"  • Composite Score: {composite_score:.3f}")
        
        visualize_features(raw_features, band)
        
        # 10. RETURN RESULTS
        if return_detailed:
            detailed_report = {
                "band": band,
                "composite_score": float(composite_score),
                "features": {k: float(v) for k, v in raw_features.items()},
                "normalized_features": {k: float(v) for k, v in normalized_features.items()},
                "transcript": transcript,
                "duration": float(duration),
                "word_count": word_count,
                "wpm": float(len(words) / (duration / 60) if duration > 0 else 0),
                "feedback_text": generate_detailed_feedback(band, raw_features, composite_score, transcript, duration)
            }
            return band, detailed_report
        else:
            return band
    
    except Exception as e:
        print(f"❌ Error in assessment: {str(e)}")
        if return_detailed:
            return 4, {"error": str(e), "band": 4}
        return 4

def export_report(detailed_report, output_dir="reports"):
    """Export detailed assessment report to files"""
    
    # Create output directory
    Path(output_dir).mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_filename = f"ielts_pronunciation_{timestamp}"
    
    # 1. Save JSON data
    json_file = f"{output_dir}/{base_filename}.json"
    with open(json_file, 'w', encoding='utf-8') as f:
        # Convert numpy types to Python types for JSON serialization
        json_data = json.loads(json.dumps(detailed_report, default=str))
        json.dump(json_data, f, indent=2, ensure_ascii=False)
    
    # 2. Save text report
    text_file = f"{output_dir}/{base_filename}.txt"
    with open(text_file, 'w', encoding='utf-8') as f:
        f.write(detailed_report.get("feedback_text", ""))
    
    # 3. Save transcript
    transcript_file = f"{output_dir}/{base_filename}_transcript.txt"
    with open(transcript_file, 'w', encoding='utf-8') as f:
        f.write(detailed_report.get("transcript", ""))
    
    print(f"📁 Reports saved to '{output_dir}/':")
    print(f"   • {base_filename}.json (detailed data)")
    print(f"   • {base_filename}.txt (feedback report)")
    print(f"   • {base_filename}_transcript.txt (transcript)")
    
    return json_file

# ========================
# ZERO-SHOT PRONUNCIATION SCORER
# ========================
class ZeroShotPronunciationScorer:
    """
    TRUE zero-shot pronunciation scoring:
    1. Uses pre-trained speech model (wav2vec2) trained on general speech tasks
    2. NO IELTS training data
    3. Returns number (0-1) that you can map to IELTS bands
    """
    
    def __init__(self):
        # Load pre-trained model (trained on general speech, NOT pronunciation assessment)
        self.model_name = "facebook/wav2vec2-base"
        self.processor = Wav2Vec2Processor.from_pretrained(self.model_name)
        self.model = Wav2Vec2Model.from_pretrained(self.model_name)
        self.model.eval()  # Evaluation mode
        
    def extract_speech_features(self, audio_path):
        """Extract features using pre-trained model"""
        # Load audio
        audio, sr = librosa.load(audio_path, sr=16000)
        
        # Preprocess for wav2vec2
        inputs = self.processor(audio, sampling_rate=sr, return_tensors="pt", padding=True)
        
        with torch.no_grad():
            # Get hidden states (model's understanding of speech)
            outputs = self.model(**inputs)
            hidden_states = outputs.last_hidden_state  # [batch, time, features]
        
        return hidden_states.squeeze(0).numpy()
    
    def score_pronunciation(self, audio_path):
        """
        Zero-shot pronunciation score (0-1)
        Model INFERS pronunciation quality from general speech understanding
        """
        # 1. Get speech features from pre-trained model
        features = self.extract_speech_features(audio_path)
        
        # 2. Calculate "pronunciation confidence" score
        # Higher variance = clearer speech (hypothesis)
        feature_variance = np.var(features, axis=0).mean()
        
        # Energy concentration (clear speech has focused energy)
        energy = np.abs(features).mean(axis=1)
        energy_skew = np.abs(librosa.feature.spectral_rolloff(
            y=energy.reshape(1, -1), sr=16000
        )).mean()
        
        # 3. Combine scores (tuned on general principles, NOT IELTS data)
        clarity_score = min(1.0, feature_variance * 100)  # 0-1 range
        energy_score = min(1.0, energy_skew / 5000)  # 0-1 range
        
        # Final score: weighted combination
        final_score = (clarity_score * 0.6) + (energy_score * 0.4)
        
        return float(final_score)
    
    def score_to_band(self, score):
        """Map your 0-1 score to IELTS bands 1-9"""
        # You define the mapping
        if score >= 0.90: return 9
        elif score >= 0.80: return 8
        elif score >= 0.70: return 7
        elif score >= 0.60: return 6
        elif score >= 0.50: return 5
        elif score >= 0.40: return 4
        elif score >= 0.30: return 3
        elif score >= 0.20: return 2
        else: return 1

# ========================
# ZERO-SHOT IELTS GRAMMAR ASSESSOR
# ========================
class ZeroShotIELTSGrammar:
    """
    Zero-shot approach to IELTS Grammatical Range & Accuracy
    No training needed, based on official IELTS criteria
    """
    
    def __init__(self):
        print("🔄 Initializing Zero-Shot IELTS Grammar Assessor...")
        
        # IELTS Official Band Descriptors for Grammar
        self.band_descriptors = {
            9: {
                "range": "• Uses a full range of structures naturally and appropriately\n• Produces rare minor errors only",
                "accuracy": "• Errors are so rare they are difficult to spot\n• Structures are precise and accurate at all times",
                "score_ranges": {
                    "error_density": (0, 1.0),      # errors per 100 words
                    "complex_ratio": (0.8, 1.0),    # complex sentences
                    "structures_used": (12, 14),    # different structures
                    "sentence_variety": (0.9, 1.0)  # sentence type variety
                }
            },
            8: {
                "range": "• Uses a wide range of structures flexibly\n• Majority of sentences are error-free",
                "accuracy": "• Makes only very occasional errors\n• Errors do not reduce communication",
                "score_ranges": {
                    "error_density": (1.0, 3.0),
                    "complex_ratio": (0.6, 0.8),
                    "structures_used": (9, 12),
                    "sentence_variety": (0.8, 0.9)
                }
            },
            7: {
                "range": "• Uses a variety of complex structures\n• Produces frequent error-free sentences",
                "accuracy": "• Has good control of grammar and punctuation\n• May make a few errors",
                "score_ranges": {
                    "error_density": (3.0, 6.0),
                    "complex_ratio": (0.4, 0.6),
                    "structures_used": (7, 9),
                    "sentence_variety": (0.6, 0.8)
                }
            },
            6: {
                "range": "• Uses a mix of simple and complex sentence forms\n• Some complex structures are accurate",
                "accuracy": "• Makes some errors in grammar and punctuation\n• Errors rarely reduce communication",
                "score_ranges": {
                    "error_density": (6.0, 10.0),
                    "complex_ratio": (0.3, 0.4),
                    "structures_used": (5, 7),
                    "sentence_variety": (0.4, 0.6)
                }
            },
            5: {
                "range": "• Uses only a limited range of structures\n• Attempts complex sentences but with errors",
                "accuracy": "• Basic sentence forms are fairly accurate\n• Complex structures usually contain errors",
                "score_ranges": {
                    "error_density": (10.0, 15.0),
                    "complex_ratio": (0.2, 0.3),
                    "structures_used": (3, 5),
                    "sentence_variety": (0.3, 0.4)
                }
            },
            4: {
                "range": "• Uses only basic sentence forms\n• Subordinate clauses are rare",
                "accuracy": "• Errors are frequent and may cause strain for reader\n• Basic sentence forms may contain errors",
                "score_ranges": {
                    "error_density": (15.0, 25.0),
                    "complex_ratio": (0.1, 0.2),
                    "structures_used": (1, 3),
                    "sentence_variety": (0.1, 0.3)
                }
            }
        }
        
        # IELTS-specific grammar patterns (zero-shot rules)
        self.ielts_patterns = {
            # HIGH SEVERITY (Band reduction patterns)
            "high_severity": [
                # Tense shifting in narrative (very bad for IELTS)
                (r'(last\s+year|yesterday|ago)\s+(am|is|are|has|have)\b', 2.0, "Tense shift: Past time requires past tense"),
                (r'\b(if\s+I\s+have.*I\s+would)\b', 1.5, "Mixed conditional: Should be 'If I had... I would'"),
                
                # Basic SVA errors (Band 5 max if present)
                (r'\b(he|she|it)\s+(have|do)\b', 1.5, "Subject-verb agreement: Use 'has' or 'does'"),
                (r'\b(they|we|you)\s+(has|does)\b', 1.5, "Subject-verb agreement: Use 'have' or 'do'"),
                
                # Article errors with proper nouns
                (r'\b(the\s+India|the\s+China|the\s+Japan)\b', 1.0, "No article with country names (usually)"),
            ],
            
            # MEDIUM SEVERITY (Common IELTS errors)
            "medium_severity": [
                # Preposition errors
                (r'\b(depend|rely)\s+of\b', 0.5, "Preposition: Should be 'depend on'"),
                (r'\b(interested)\s+for\b', 0.5, "Preposition: Should be 'interested in'"),
                
                # Countable/uncountable
                (r'\b(many)\s+(information|advice|news)\b', 0.5, "Uncountable noun: Use 'much' or 'a lot of'"),
                
                # Word form errors
                (r'\b(very)\s+(importance|beautifully)\s+(noun)', 0.5, "Word form: Use adjective before noun"),
            ],
            
            # LOW SEVERITY (Minor errors)
            "low_severity": [
                # Article a/an
                (r'\b(a)\s+[aeiou][a-z]*\b', 0.2, "Article: Use 'an' before vowel sound"),
                (r'\b(an)\s+[^aeiou\s][a-z]*\b', 0.2, "Article: Use 'a' before consonant sound"),
                
                # Informal contractions (not good for IELTS writing)
                (r"\b(gonna|wanna|gotta)\b", 0.2, "Avoid informal contractions in IELTS"),
            ]
        }
        
        print("✅ Zero-Shot IELTS Grammar Assessor ready")
    
    def detect_errors_zero_shot(self, text: str) -> Dict:
        """
        Zero-shot error detection using pattern matching
        No ML models, just IELTS-specific rules
        """
        errors = {
            "high_severity": [],
            "medium_severity": [],
            "low_severity": [],
            "total_score_deduction": 0.0
        }
        
        # Convert to lowercase for matching (but keep original for context)
        text_lower = text.lower()
        
        # Check each pattern category
        for severity, patterns in self.ielts_patterns.items():
            for pattern, penalty, message in patterns:
                matches = re.finditer(pattern, text_lower, re.IGNORECASE)
                for match in matches:
                    # Get context (3 words before and after)
                    start = max(0, match.start() - 30)
                    end = min(len(text), match.end() + 30)
                    context = text[start:end]
                    
                    errors[severity].append({
                        "pattern": pattern,
                        "message": message,
                        "context": context,
                        "penalty": penalty
                    })
                    errors["total_score_deduction"] += penalty
        
        return errors
    
    def analyze_grammar_range(self, text: str) -> Dict:
        """
        Analyze grammatical range without ML
        Based on sentence structure and variety
        """
        # Basic sentence splitting
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if not sentences:
            return {
                "sentence_count": 0,
                "avg_words": 0,
                "complex_sentences": 0,
                "complex_ratio": 0,
                "structures_detected": [],
                "structure_count": 0
            }
        
        # Count words and analyze sentence complexity
        total_words = sum(len(s.split()) for s in sentences)
        avg_words = total_words / len(sentences)
        
        # Detect complex sentences (simple heuristic)
        complex_sentences = 0
        for sentence in sentences:
            # Complex if has subordinate conjunction or relative pronoun
            if (re.search(r'\b(although|because|since|if|when|while|that|which|who)\b', sentence, re.IGNORECASE) and
                len(sentence.split()) > 8):
                complex_sentences += 1
        
        complex_ratio = complex_sentences / len(sentences)
        
        # Detect grammar structures (simplified patterns)
        structures_detected = []
        
        # Check for common IELTS structures
        structure_checks = [
            ("simple_present", lambda t: bool(re.search(r'\b(am|is|are|do|does)\b', t, re.IGNORECASE))),
            ("simple_past", lambda t: bool(re.search(r'\b(was|were|did|had|ed\b)', t, re.IGNORECASE))),
            ("present_perfect", lambda t: bool(re.search(r'\b(have|has)\s+\w+ed\b', t, re.IGNORECASE)) or 
                                         bool(re.search(r'\b(have|has)\s+been\b', t, re.IGNORECASE))),
            ("future", lambda t: bool(re.search(r'\b(will|shall|going to)\b', t, re.IGNORECASE))),
            ("conditionals", lambda t: bool(re.search(r'\b(if.*would|if.*will|unless.*)\b', t, re.IGNORECASE))),
            ("passive", lambda t: bool(re.search(r'\b(am|is|are|was|were)\s+\w+ed\b', t, re.IGNORECASE)) and
                                 not re.search(r'\b(am|is|are|was|were)\s+going\b', t, re.IGNORECASE)),
            ("relative_clauses", lambda t: bool(re.search(r'\b(that|which|who|whom)\b', t, re.IGNORECASE))),
            ("modal_verbs", lambda t: bool(re.search(r'\b(can|could|may|might|must|should|would)\b', t, re.IGNORECASE))),
            ("comparatives", lambda t: bool(re.search(r'\b(more|less|than|er\b)\b', t, re.IGNORECASE))),
        ]
        
        for struct_name, check_func in structure_checks:
            if check_func(text):
                structures_detected.append(struct_name)
        
        return {
            "sentence_count": len(sentences),
            "avg_words": avg_words,
            "complex_sentences": complex_sentences,
            "complex_ratio": complex_ratio,
            "structures_detected": structures_detected,
            "structure_count": len(structures_detected)
        }
    
    def calculate_band_score(self, errors: Dict, range_analysis: Dict, word_count: int) -> Tuple[int, Dict]:
        """
        Calculate IELTS band based on zero-shot analysis
        """
        # Calculate error density
        total_errors = (len(errors["high_severity"]) + 
                       len(errors["medium_severity"]) + 
                       len(errors["low_severity"]))
        
        error_density = (total_errors / word_count) * 100 if word_count > 0 else 100
        
        # Get metrics from range analysis
        complex_ratio = range_analysis["complex_ratio"]
        structures_used = range_analysis["structure_count"]
        
        # Calculate base scores
        error_score = self._error_score(error_density, errors["total_score_deduction"])
        range_score = self._range_score(complex_ratio, structures_used)
        
        # Combine scores (60% accuracy, 40% range - based on IELTS weighting)
        composite_score = (error_score * 0.6) + (range_score * 0.4)
        
        # Map to IELTS bands
        band = self._map_to_band(composite_score, error_density, complex_ratio, structures_used)
        
        # Detailed breakdown
        breakdown = {
            "composite_score": round(composite_score, 3),
            "error_density": round(error_density, 1),
            "complex_ratio": round(complex_ratio, 2),
            "structures_used": structures_used,
            "total_errors": total_errors,
            "error_breakdown": {
                "high": len(errors["high_severity"]),
                "medium": len(errors["medium_severity"]),
                "low": len(errors["low_severity"])
            }
        }
        
        return band, breakdown
    
    def _error_score(self, error_density: float, penalty: float) -> float:
        """Score for grammatical accuracy"""
        # Normalize error density to 0-1 scale (0=best, 1=worst)
        normalized_errors = min(1.0, error_density / 30.0)  # 30 errors/100 words = worst
        
        # Apply penalty from high-severity errors
        penalty_factor = min(1.0, penalty / 5.0)  # Max 5 penalty points
        
        # Combine (more weight to high-severity errors)
        score = 1.0 - (normalized_errors * 0.7 + penalty_factor * 0.3)
        return max(0.0, min(1.0, score))
    
    def _range_score(self, complex_ratio: float, structures_used: int) -> float:
        """Score for grammatical range"""
        # Sentence complexity score
        complexity_score = min(1.0, complex_ratio / 0.8)  # 80% complex = perfect
        
        # Structure variety score (14 possible structures)
        variety_score = min(1.0, structures_used / 12.0)  # 12+ structures = perfect
        
        # Combine scores
        score = (complexity_score * 0.6) + (variety_score * 0.4)
        return max(0.0, min(1.0, score))
    
    def _map_to_band(self, composite: float, error_density: float, 
                    complex_ratio: float, structures: int) -> int:
        """Map scores to IELTS bands with realistic thresholds"""
        
        # Band 9 criteria (extremely strict)
        if (composite >= 0.95 and 
            error_density <= 1.0 and 
            complex_ratio >= 0.8 and 
            structures >= 10):
            return 9
        
        # Band 8 criteria
        elif (composite >= 0.85 and 
              error_density <= 3.0 and 
              complex_ratio >= 0.6 and 
              structures >= 8):
            return 8
        
        # Band 7 criteria
        elif (composite >= 0.75 and 
              error_density <= 6.0 and 
              complex_ratio >= 0.4 and 
              structures >= 6):
            return 7
        
        # Band 6 criteria
        elif (composite >= 0.65 and 
              error_density <= 10.0 and 
              complex_ratio >= 0.3 and 
              structures >= 4):
            return 6
        
        # Band 5 criteria
        elif (composite >= 0.55 and 
              error_density <= 15.0):
            return 5
        
        # Band 4 criteria
        elif (composite >= 0.45 and 
              error_density <= 25.0):
            return 4
        
        # Band 3 criteria
        elif composite >= 0.35:
            return 3
        
        # Band 2 criteria
        elif composite >= 0.25:
            return 2
        
        # Band 1
        else:
            return 1
    
    def generate_feedback(self, band: int, errors: Dict, 
                         range_analysis: Dict, breakdown: Dict) -> str:
        """Generate IELTS-style feedback"""
        
        feedback = "\n🎯 IELTS GRAMMATICAL RANGE & ACCURACY ASSESSMENT\n"
        feedback += "=" * 70 + "\n\n"
        
        # Band score
        feedback += f"📊 OVERALL BAND SCORE: {band}/9\n\n"
        
        # Official descriptor
        if band in self.band_descriptors:
            desc = self.band_descriptors[band]
            feedback += "📋 OFFICIAL IELTS DESCRIPTOR:\n"
            feedback += f"{desc['range']}\n"
            feedback += f"{desc['accuracy']}\n\n"
        
        # Detailed metrics
        feedback += "🔍 DETAILED ANALYSIS:\n"
        feedback += f"  • Error Density: {breakdown['error_density']} per 100 words\n"
        feedback += f"  • Complex Sentences: {range_analysis['complex_ratio']:.1%}\n"
        feedback += f"  • Grammar Structures Used: {breakdown['structures_used']}/14\n"
        feedback += f"  • Sentence Variety: {range_analysis['sentence_count']} sentences\n"
        feedback += f"  • Average Length: {range_analysis['avg_words']:.1f} words/sentence\n"
        feedback += f"  • Composite Score: {breakdown['composite_score']:.3f}\n\n"
        
        # Error breakdown
        if breakdown['total_errors'] > 0:
            feedback += "⚠️  ERROR ANALYSIS:\n"
            feedback += f"  • High Severity: {breakdown['error_breakdown']['high']}\n"
            feedback += f"  • Medium Severity: {breakdown['error_breakdown']['medium']}\n"
            feedback += f"  • Low Severity: {breakdown['error_breakdown']['low']}\n\n"
            
            # Show specific high/medium errors
            if errors["high_severity"]:
                feedback += "🔴 CRITICAL ERRORS (Band Limiting):\n"
                for i, error in enumerate(errors["high_severity"][:3], 1):
                    feedback += f"  {i}. {error['message']}\n"
                    feedback += f"     Example: \"...{error['context']}...\"\n"
                feedback += "\n"
            
            if errors["medium_severity"]:
                feedback += "🟡 COMMON IELTS ERRORS:\n"
                for i, error in enumerate(errors["medium_severity"][:3], 1):
                    feedback += f"  {i}. {error['message']}\n"
                feedback += "\n"
        
        # Grammar structures found
        if range_analysis["structures_detected"]:
            feedback += "✅ GRAMMAR STRUCTURES IDENTIFIED:\n"
            structures = range_analysis["structures_detected"]
            for i in range(0, len(structures), 4):
                line = structures[i:i+4]
                feedback += f"  • " + " • ".join(line) + "\n"
            feedback += "\n"
        
        # Improvement suggestions based on band
        feedback += "🎯 AREAS FOR IMPROVEMENT:\n"
        
        if band <= 5:
            feedback += "  1. Master basic sentence structure (Subject + Verb + Object)\n"
            feedback += "  2. Practice simple past and present tenses consistently\n"
            feedback += "  3. Avoid tense shifting within paragraphs\n"
            feedback += "  4. Learn basic articles (a/an/the) rules\n"
        elif band <= 6.5:
            feedback += "  1. Work on complex sentences with 'although', 'because', 'which'\n"
            feedback += "  2. Practice different conditional forms (if...will, if...would)\n"
            feedback += "  3. Use a variety of sentence beginnings\n"
            feedback += "  4. Improve preposition accuracy (depend on, interested in)\n"
        elif band <= 7.5:
            feedback += "  1. Refine use of advanced structures (passive, perfect tenses)\n"
            feedback += "  2. Work on subtle article usage (the for specific references)\n"
            feedback += "  3. Practice mixed conditionals and hypothetical language\n"
            feedback += "  4. Eliminate minor punctuation errors\n"
        else:
            feedback += "  1. Maintain your excellent grammatical range\n"
            feedback += "  2. Focus on native-like fluency and natural phrasing\n"
            feedback += "  3. Practice academic writing style for Task 2\n"
            feedback += "  4. Get native speaker feedback for subtle improvements\n"
        
        return feedback
    
    def assess(self, text: str) -> Tuple[int, str, Dict]:
        """
        Main assessment function
        Returns: (band_score, feedback, detailed_analysis)
        """
        if not text or len(text.split()) < 10:
            return 3, "Insufficient text for assessment", {}
        
        word_count = len(text.split())
        
        # Step 1: Zero-shot error detection
        errors = self.detect_errors_zero_shot(text)
        
        # Step 2: Grammar range analysis
        range_analysis = self.analyze_grammar_range(text)
        
        # Step 3: Calculate band score
        band, breakdown = self.calculate_band_score(errors, range_analysis, word_count)
        
        # Step 4: Generate feedback
        feedback = self.generate_feedback(band, errors, range_analysis, breakdown)
        
        # Detailed analysis for debugging
        detailed = {
            "band": band,
            "word_count": word_count,
            "errors": errors,
            "range_analysis": range_analysis,
            "breakdown": breakdown
        }
        
        return band, feedback, detailed

# ========================
# GRAMMAR ASSESSOR WRAPPER
# ========================
class GrammarIELTSAssessor:
    """
    Wrapper for Zero-Shot IELTS Grammar Assessor
    """
    
    def __init__(self):
        print("\n🔄 Loading Zero-Shot IELTS Grammar Assessor...")
        self.assessor = ZeroShotIELTSGrammar()
        print("✅ IELTS Grammar Assessor ready")
    
    def assess_grammatical_band(self, transcript):
        """
        Main function: Assess Grammatical Range & Accuracy (Band 1-9)
        """
        if not transcript or len(transcript.split()) < 10:
            return 3, "Insufficient text for grammar analysis"
        
        try:
            band, feedback, detailed = self.assessor.assess(transcript)
            return band, feedback
        except Exception as e:
            print(f"⚠️  Grammar assessment error: {e}")
            return 4, "Grammar assessment failed"

# ========================
# ZERO-SHOT FLUENCY & COHERENCE ASSESSOR
# ========================
class ZeroShotFluencyCoherence:
    """
    Zero-shot assessment of IELTS Speaking Fluency & Coherence
    Based on official IELTS band descriptors
    """
    
    def __init__(self):
        print("🔄 Initializing Fluency & Coherence Assessor...")
        
        # Official IELTS Band Descriptors for Fluency & Coherence
        self.band_descriptors = {
            9: {
                "fluency": "• Speaks fluently with only rare repetition or self-correction\n• Any hesitation is content-related",
                "coherence": "• Speaks coherently with fully appropriate cohesive features\n• Develops topics fully and appropriately",
                "indicators": {
                    "speech_rate": (4.0, 5.0),        # syllables per second
                    "pause_ratio": (0.05, 0.15),      # % of time pausing
                    "hesitation_words": (0, 1),       # um/uh per 100 words
                    "repetitions": (0, 1),            # repetitions per 100 words
                    "self_corrections": (0, 1),       # self-corrections per 100 words
                    "cohesive_devices": (8, 12),      # linking words per 100 words
                    "topic_development": (0.9, 1.0)   # topic coherence score
                }
            },
            8: {
                "fluency": "• Speaks fluently with only occasional repetition\n• Hesitation is usually content-related",
                "coherence": "• Sequences information and ideas logically\n• Manages all aspects of cohesion well",
                "indicators": {
                    "speech_rate": (3.5, 4.5),
                    "pause_ratio": (0.10, 0.25),
                    "hesitation_words": (1, 3),
                    "repetitions": (1, 3),
                    "self_corrections": (1, 3),
                    "cohesive_devices": (6, 10),
                    "topic_development": (0.8, 0.9)
                }
            },
            7: {
                "fluency": "• Speaks at length without noticeable effort\n• May demonstrate language-related hesitation",
                "coherence": "• Sequences information and ideas logically\n• Uses a range of cohesive devices appropriately",
                "indicators": {
                    "speech_rate": (3.0, 4.0),
                    "pause_ratio": (0.15, 0.30),
                    "hesitation_words": (3, 6),
                    "repetitions": (3, 6),
                    "self_corrections": (3, 6),
                    "cohesive_devices": (5, 8),
                    "topic_development": (0.7, 0.8)
                }
            },
            6: {
                "fluency": "• Is willing to speak at length but may lose coherence\n• Uses repetition and self-correction",
                "coherence": "• Sequences information and ideas coherently\n• Uses cohesive devices effectively",
                "indicators": {
                    "speech_rate": (2.5, 3.5),
                    "pause_ratio": (0.20, 0.35),
                    "hesitation_words": (6, 10),
                    "repetitions": (6, 10),
                    "self_corrections": (6, 10),
                    "cohesive_devices": (4, 6),
                    "topic_development": (0.6, 0.7)
                }
            },
            5: {
                "fluency": "• Usually maintains flow of speech but uses repetition\n• Slow speech may cause strain for listener",
                "coherence": "• Presents information with some organization\n• May overuse certain cohesive devices",
                "indicators": {
                    "speech_rate": (2.0, 3.0),
                    "pause_ratio": (0.25, 0.40),
                    "hesitation_words": (10, 15),
                    "repetitions": (10, 15),
                    "self_corrections": (10, 15),
                    "cohesive_devices": (3, 5),
                    "topic_development": (0.5, 0.6)
                }
            },
            4: {
                "fluency": "• Cannot respond without noticeable pauses\n• May speak slowly with frequent repetition",
                "coherence": "• Presents information but ideas are not arranged coherently",
                "indicators": {
                    "speech_rate": (1.5, 2.5),
                    "pause_ratio": (0.35, 0.50),
                    "hesitation_words": (15, 25),
                    "repetitions": (15, 25),
                    "self_corrections": (15, 25),
                    "cohesive_devices": (1, 3),
                    "topic_development": (0.4, 0.5)
                }
            }
        }
        
        # Fluency patterns to detect
        self.fluency_patterns = {
            "hesitation_words": [
                r'\b(um|uh|er|ah|mm|hmm)\b',
                r'\b(like|you know|I mean|sort of|kind of)\b'  # Filler phrases
            ],
            
            "repetitions": [
                r'\b(\w+)\s+\1\b',  # Word repetition (the the)
                r'\b(I|the|and|but)\s+\1\b'  # Common word repetition
            ],
            
            "self_corrections": [
                r'\b(I mean|sorry|actually|I meant|or rather)\b',
                r'\b(\w+)\s+(no|sorry)\s+(\w+)\b'  # Self-correction pattern
            ],
            
            "false_starts": [
                r'\b(And|But|So|Well)\s+(um|uh|er)',
                r'\b(I|We|They)\s+(was|were|is|are)\s+(um|uh)'
            ]
        }
        
        # Coherence patterns (discourse markers)
        self.cohesive_devices = {
            "adding": ["and", "also", "furthermore", "moreover", "in addition"],
            "contrasting": ["but", "however", "although", "on the other hand", "nevertheless"],
            "cause_effect": ["because", "so", "therefore", "as a result", "consequently"],
            "sequencing": ["first", "second", "then", "next", "finally"],
            "exemplifying": ["for example", "for instance", "such as", "like"],
            "summarizing": ["in conclusion", "to sum up", "overall", "in summary"],
            "referencing": ["this", "that", "these", "those", "it", "they"],
            "time_markers": ["when", "while", "after", "before", "during", "since"]
        }
        
        # Topic development patterns
        self.topic_patterns = {
            "topic_shifts": [
                r'\b(by the way|anyway|oh|well)\s+(did|do|have)',  # Abrupt topic change
                r'\b(also|and)\s+(I|we)\s+(have|like|want)'  # Adding unrelated ideas
            ],
            "elaboration": [
                r'\b(because|since|as)\s+',  # Giving reasons
                r'\b(for example|for instance|such as)\s+',  # Giving examples
                r'\b(which means|that is|in other words)\s+'  # Clarifying
            ]
        }
        
        print("✅ Fluency & Coherence Assessor ready")
    
    def analyze_fluency(self, transcript: str, audio_duration: float = None) -> Dict:
        """
        Analyze fluency features from transcript
        """
        words = transcript.split()
        word_count = len(words)
        
        # Calculate basic metrics
        metrics = {
            "word_count": word_count,
            "unique_words": len(set(words)),
            "lexical_diversity": len(set(words)) / max(1, word_count)
        }
        
        # Detect hesitation patterns
        hesitations = 0
        for pattern in self.fluency_patterns["hesitation_words"]:
            matches = re.findall(pattern, transcript.lower())
            hesitations += len(matches)
        
        # Detect repetitions
        repetitions = 0
        for pattern in self.fluency_patterns["repetitions"]:
            matches = re.findall(pattern, transcript.lower())
            repetitions += len(matches)
        
        # Detect self-corrections
        self_corrections = 0
        for pattern in self.fluency_patterns["self_corrections"]:
            matches = re.findall(pattern, transcript.lower())
            self_corrections += len(matches)
        
        # Detect false starts
        false_starts = 0
        for pattern in self.fluency_patterns["false_starts"]:
            matches = re.findall(pattern, transcript.lower())
            false_starts += len(matches)
        
        # Calculate rates per 100 words
        metrics.update({
            "hesitation_rate": (hesitations / word_count) * 100 if word_count > 0 else 0,
            "repetition_rate": (repetitions / word_count) * 100 if word_count > 0 else 0,
            "self_correction_rate": (self_corrections / word_count) * 100 if word_count > 0 else 0,
            "false_start_rate": (false_starts / word_count) * 100 if word_count > 0 else 0,
            "total_disfluencies": hesitations + repetitions + self_corrections + false_starts,
            "disfluency_rate": ((hesitations + repetitions + self_corrections + false_starts) / word_count) * 100 if word_count > 0 else 0
        })
        
        # If audio duration is provided, calculate speech rate
        if audio_duration and audio_duration > 0:
            syllables = self._estimate_syllables(transcript)
            metrics["speech_rate_syllables"] = syllables / audio_duration
            metrics["speech_rate_words"] = word_count / audio_duration * 60  # WPM
        
        return metrics
    
    def analyze_coherence(self, transcript: str) -> Dict:
        """
        Analyze coherence and cohesion features
        """
        words = transcript.split()
        word_count = len(words)
        
        # Count cohesive devices
        cohesive_counts = {}
        total_cohesive = 0
        
        for category, markers in self.cohesive_devices.items():
            count = 0
            for marker in markers:
                # Use word boundaries to avoid partial matches
                pattern = r'\b' + re.escape(marker) + r'\b'
                matches = re.findall(pattern, transcript.lower())
                count += len(matches)
            
            cohesive_counts[category] = count
            total_cohesive += count
        
        # Analyze topic development
        topic_shifts = 0
        for pattern in self.topic_patterns["topic_shifts"]:
            matches = re.findall(pattern, transcript.lower())
            topic_shifts += len(matches)
        
        elaborations = 0
        for pattern in self.topic_patterns["elaboration"]:
            matches = re.findall(pattern, transcript.lower())
            elaborations += len(matches)
        
        # Calculate sentence complexity for coherence
        sentences = re.split(r'[.!?]+', transcript)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        avg_sentence_length = statistics.mean([len(s.split()) for s in sentences]) if sentences else 0
        
        # Topic coherence score (simplified)
        if len(sentences) > 1:
            # Check if sentences share common topics
            topic_words = set()
            for sentence in sentences[:3]:  # Check first 3 sentences
                words = sentence.lower().split()[:5]  # First 5 words
                topic_words.update(words)
            
            topic_overlap = len(topic_words) / (len(sentences[:3]) * 5) if sentences else 0
        else:
            topic_overlap = 0
        
        metrics = {
            "cohesive_devices_total": total_cohesive,
            "cohesive_devices_rate": (total_cohesive / word_count) * 100 if word_count > 0 else 0,
            "cohesive_categories": cohesive_counts,
            "topic_shifts": topic_shifts,
            "elaborations": elaborations,
            "elaboration_ratio": elaborations / max(1, len(sentences)),
            "avg_sentence_length": avg_sentence_length,
            "topic_coherence": min(1.0, topic_overlap * 1.5),  # Normalize
            "sentence_count": len(sentences)
        }
        
        return metrics
    
    def calculate_band_score(self, fluency_metrics: Dict, 
                           coherence_metrics: Dict, 
                           word_count: int) -> Tuple[int, Dict]:
        """
        Calculate IELTS band for Fluency & Coherence
        """
        # Extract key metrics
        disfluency_rate = fluency_metrics.get("disfluency_rate", 100)
        speech_rate = fluency_metrics.get("speech_rate_words", 0)  # WPM
        
        cohesive_rate = coherence_metrics.get("cohesive_devices_rate", 0)
        topic_coherence = coherence_metrics.get("topic_coherence", 0)
        elaboration_ratio = coherence_metrics.get("elaboration_ratio", 0)
        
        # Calculate fluency score (0-1, 1=best)
        if speech_rate > 0:
            fluency_score = self._calculate_fluency_score(disfluency_rate, speech_rate)
        else:
            # Fallback if no audio duration
            fluency_score = self._calculate_fluency_text_only(disfluency_rate)
        
        # Calculate coherence score (0-1, 1=best)
        coherence_score = self._calculate_coherence_score(cohesive_rate, 
                                                         topic_coherence, 
                                                         elaboration_ratio)
        
        # Combined score (equal weighting for Fluency and Coherence)
        composite_score = (fluency_score + coherence_score) / 2
        
        # Map to band
        band = self._map_to_band(composite_score, disfluency_rate, 
                                cohesive_rate, topic_coherence)
        
        # Detailed breakdown
        breakdown = {
            "composite_score": round(composite_score, 3),
            "fluency_score": round(fluency_score, 3),
            "coherence_score": round(coherence_score, 3),
            "disfluency_rate": round(disfluency_rate, 1),
            "speech_rate_wpm": round(speech_rate, 1) if speech_rate > 0 else "N/A",
            "cohesive_rate": round(cohesive_rate, 1),
            "topic_coherence": round(topic_coherence, 2),
            "elaboration_ratio": round(elaboration_ratio, 2)
        }
        
        return band, breakdown
    
    def _calculate_fluency_score(self, disfluency_rate: float, speech_rate: float) -> float:
        """Calculate fluency score from metrics"""
        # Normalize disfluency (lower is better)
        disfluency_norm = max(0, 1 - (disfluency_rate / 30.0))
        
        # Normalize speech rate (optimal: 140-180 WPM)
        if speech_rate < 100:
            speech_norm = speech_rate / 100.0
        elif speech_rate > 200:
            speech_norm = max(0, 1 - ((speech_rate - 180) / 100.0))
        else:
            speech_norm = 1.0
        
        # Weight disfluency more heavily (70%)
        score = (disfluency_norm * 0.7) + (speech_norm * 0.3)
        return max(0, min(1, score))
    
    def _calculate_fluency_text_only(self, disfluency_rate: float) -> float:
        """Calculate fluency score without audio (text only)"""
        # Based only on disfluencies in transcript
        return max(0, 1 - (disfluency_rate / 40.0))
    
    def _calculate_coherence_score(self, cohesive_rate: float, 
                                  topic_coherence: float, 
                                  elaboration_ratio: float) -> float:
        """Calculate coherence score from metrics"""
        # Normalize cohesive device rate (optimal: 6-10 per 100 words)
        if cohesive_rate < 3:
            cohesive_norm = cohesive_rate / 3.0
        elif cohesive_rate > 12:
            cohesive_norm = max(0, 1 - ((cohesive_rate - 12) / 10.0))
        else:
            cohesive_norm = 1.0
        
        # Combine scores
        score = (cohesive_norm * 0.4) + (topic_coherence * 0.4) + (elaboration_ratio * 0.2)
        return max(0, min(1, score))
    
    def _map_to_band(self, composite: float, disfluency_rate: float,
                    cohesive_rate: float, topic_coherence: float) -> int:
        """Map to IELTS bands with realistic thresholds"""
        
        # Band 9
        if (composite >= 0.95 and 
            disfluency_rate <= 3.0 and 
            cohesive_rate >= 8.0 and 
            topic_coherence >= 0.9):
            return 9
        
        # Band 8
        elif (composite >= 0.85 and 
              disfluency_rate <= 6.0 and 
              cohesive_rate >= 6.0 and 
              topic_coherence >= 0.8):
            return 8
        
        # Band 7
        elif (composite >= 0.75 and 
              disfluency_rate <= 12.0 and 
              cohesive_rate >= 5.0 and 
              topic_coherence >= 0.7):
            return 7
        
        # Band 6
        elif (composite >= 0.65 and 
              disfluency_rate <= 18.0 and 
              cohesive_rate >= 4.0 and 
              topic_coherence >= 0.6):
            return 6
        
        # Band 5
        elif (composite >= 0.55 and 
              disfluency_rate <= 25.0):
            return 5
        
        # Band 4
        elif (composite >= 0.45 and 
              disfluency_rate <= 35.0):
            return 4
        
        # Band 3
        elif composite >= 0.35:
            return 3
        
        # Band 2
        elif composite >= 0.25:
            return 2
        
        # Band 1
        else:
            return 1
    
    def _estimate_syllables(self, text: str) -> int:
        """Simple syllable estimation for speech rate calculation"""
        vowels = 'aeiouy'
        words = text.lower().split()
        
        syllable_count = 0
        for word in words:
            if len(word) <= 3:
                syllable_count += 1
            else:
                # Simple vowel counting
                count = 0
                prev_char = ''
                for char in word:
                    if char in vowels and prev_char not in vowels:
                        count += 1
                    prev_char = char
                
                # Adjustments
                if word.endswith('e'):
                    count -= 1
                if word.endswith('le') and len(word) > 2:
                    count += 1
                
                syllable_count += max(1, count)
        
        return syllable_count
    
    def generate_feedback(self, band: int, fluency_metrics: Dict, 
                         coherence_metrics: Dict, breakdown: Dict) -> str:
        """Generate IELTS-style feedback for Fluency & Coherence"""
        
        feedback = "\n🎯 IELTS FLUENCY & COHERENCE ASSESSMENT\n"
        feedback += "=" * 70 + "\n\n"
        
        # Band score
        feedback += f"📊 OVERALL BAND SCORE: {band}/9\n\n"
        
        # Official descriptor
        if band in self.band_descriptors:
            desc = self.band_descriptors[band]
            feedback += "📋 OFFICIAL IELTS DESCRIPTOR:\n"
            feedback += f"Fluency: {desc['fluency']}\n"
            feedback += f"Coherence: {desc['coherence']}\n\n"
        
        # Detailed metrics
        feedback += "🔍 DETAILED ANALYSIS:\n"
        
        # Fluency metrics
        feedback += "FLUENCY METRICS:\n"
        if breakdown.get('speech_rate_wpm') != "N/A":
            feedback += f"  • Speech Rate: {breakdown['speech_rate_wpm']} WPM\n"
        feedback += f"  • Disfluency Rate: {breakdown['disfluency_rate']} per 100 words\n"
        feedback += f"  • Hesitations: {fluency_metrics.get('hesitation_rate', 0):.1f} per 100 words\n"
        feedback += f"  • Repetitions: {fluency_metrics.get('repetition_rate', 0):.1f} per 100 words\n"
        feedback += f"  • Self-corrections: {fluency_metrics.get('self_correction_rate', 0):.1f} per 100 words\n"
        
        # Coherence metrics
        feedback += "\nCOHERENCE METRICS:\n"
        feedback += f"  • Cohesive Devices: {breakdown['cohesive_rate']} per 100 words\n"
        feedback += f"  • Topic Coherence: {breakdown['topic_coherence']:.1%}\n"
        feedback += f"  • Elaboration Ratio: {breakdown['elaboration_ratio']:.2f}\n"
        feedback += f"  • Sentence Variety: {coherence_metrics.get('avg_sentence_length', 0):.1f} words/sentence\n"
        feedback += f"  • Composite Score: {breakdown['composite_score']:.3f}\n\n"
        
        # Show cohesive devices used
        cohesive_cats = coherence_metrics.get('cohesive_categories', {})
        if any(count > 0 for count in cohesive_cats.values()):
            feedback += "✅ COHESIVE DEVICES IDENTIFIED:\n"
            for category, count in cohesive_cats.items():
                if count > 0:
                    feedback += f"  • {category.title()}: {count}\n"
            feedback += "\n"
        
        # Show specific disfluencies if high
        if breakdown['disfluency_rate'] > 10:
            feedback += "⚠️  FLUENCY ISSUES DETECTED:\n"
            
            if fluency_metrics.get('hesitation_rate', 0) > 5:
                feedback += f"  • High hesitation rate: {fluency_metrics['hesitation_rate']:.1f}/100 words\n"
                feedback += "    - Practice speaking without fillers (um, uh, you know)\n"
            
            if fluency_metrics.get('repetition_rate', 0) > 5:
                feedback += f"  • Frequent repetitions: {fluency_metrics['repetition_rate']:.1f}/100 words\n"
                feedback += "    - Practice rephrasing instead of repeating words\n"
            
            if fluency_metrics.get('self_correction_rate', 0) > 5:
                feedback += f"  • Many self-corrections: {fluency_metrics['self_correction_rate']:.1f}/100 words\n"
                feedback += "    - Plan your sentences before speaking\n"
            
            feedback += "\n"
        
        # Improvement suggestions based on band
        feedback += "🎯 AREAS FOR IMPROVEMENT:\n"
        
        if band <= 5:
            feedback += "  1. Practice speaking at a steady pace (aim for 120-150 WPM)\n"
            feedback += "  2. Reduce filler words (um, uh) - use pauses instead\n"
            feedback += "  3. Use basic linking words: and, but, because, so\n"
            feedback += "  4. Practice 1-minute speeches on familiar topics\n"
        
        elif band <= 6.5:
            feedback += "  1. Increase speaking rate to natural speed (150-180 WPM)\n"
            feedback += "  2. Use a wider range of linking words (however, therefore, furthermore)\n"
            feedback += "  3. Practice speaking for 2 minutes without stopping\n"
            feedback += "  4. Work on smooth transitions between ideas\n"
        
        elif band <= 7.5:
            feedback += "  1. Refine natural rhythm and intonation\n"
            feedback += "  2. Use discourse markers appropriately (in addition, on the other hand)\n"
            feedback += "  3. Practice handling unfamiliar topics smoothly\n"
            feedback += "  4. Work on eliminating all unnecessary repetitions\n"
        
        else:
            feedback += "  1. Maintain your excellent fluency and coherence\n"
            feedback += "  2. Focus on natural, native-like speech patterns\n"
            feedback += "  3. Practice impromptu speaking on complex topics\n"
            feedback += "  4. Record and analyze your speech for subtle improvements\n"
        
        return feedback
    
    def assess(self, transcript: str, audio_duration: float = None) -> Tuple[int, str, Dict]:
        """
        Main assessment function for Fluency & Coherence
        """
        if not transcript or len(transcript.split()) < 10:
            return 3, "Insufficient speech for assessment", {}
        
        # Step 1: Analyze fluency
        fluency_metrics = self.analyze_fluency(transcript, audio_duration)
        
        # Step 2: Analyze coherence
        coherence_metrics = self.analyze_coherence(transcript)
        
        # Step 3: Calculate band score
        word_count = len(transcript.split())
        band, breakdown = self.calculate_band_score(fluency_metrics, 
                                                  coherence_metrics, 
                                                  word_count)
        
        # Step 4: Generate feedback
        feedback = self.generate_feedback(band, fluency_metrics, 
                                        coherence_metrics, breakdown)
        
        # Detailed analysis
        detailed = {
            "band": band,
            "fluency_metrics": fluency_metrics,
            "coherence_metrics": coherence_metrics,
            "breakdown": breakdown
        }
        
        return band, feedback, detailed

# ========================
# FLUENCY & COHERENCE WRAPPER
# ========================
class FluencyCoherenceIELTSAssessor:
    """
    Wrapper for Zero-Shot Fluency & Coherence Assessor
    """
    
    def __init__(self):
        print("\n🔄 Loading Zero-Shot Fluency & Coherence Assessor...")
        self.assessor = ZeroShotFluencyCoherence()
        print("✅ IELTS Fluency & Coherence Assessor ready")
    
    def assess_fluency_coherence_band(self, transcript, audio_duration=None):
        """
        Main function: Assess Fluency & Coherence (Band 1-9)
        """
        if not transcript or len(transcript.split()) < 10:
            return 3, "Insufficient speech for assessment"
        
        try:
            band, feedback, detailed = self.assessor.assess(transcript, audio_duration)
            return band, feedback
        except Exception as e:
            print(f"⚠️  Fluency assessment error: {e}")
            return 4, "Fluency assessment failed"

# ========================
# ZERO-SHOT LEXICAL RESOURCE ASSESSOR
# ========================
class ZeroShotLexicalResource:
    """
    Zero-shot assessment of IELTS Lexical Resource (Vocabulary)
    Based on official IELTS band descriptors
    """
    
    def __init__(self):
        print("🔄 Initializing Lexical Resource Assessor...")
        
        # Official IELTS Band Descriptors for Lexical Resource
        self.band_descriptors = {
            9: {
                "description": "• Uses a wide range of vocabulary with very natural and sophisticated control\n• Uses rare words and idioms appropriately\n• Rare minor errors only as 'slips'",
                "indicators": {
                    "lexical_diversity": (0.7, 1.0),     # Type-token ratio
                    "advanced_words": (0.25, 0.35),      # % of advanced vocabulary
                    "collocations": (0.8, 1.0),          # Appropriate collocations
                    "idioms_used": (3, 10),              # Number of idioms
                    "precision_score": (0.9, 1.0)        # Word choice precision
                }
            },
            8: {
                "description": "• Uses a wide range of vocabulary fluently and flexibly\n• Uses uncommon lexical items skillfully\n• Occasional inaccuracies in word choice",
                "indicators": {
                    "lexical_diversity": (0.6, 0.75),
                    "advanced_words": (0.20, 0.30),
                    "collocations": (0.7, 0.85),
                    "idioms_used": (2, 5),
                    "precision_score": (0.8, 0.9)
                }
            },
            7: {
                "description": "• Uses sufficient range of vocabulary for flexibility and precision\n• Uses less common lexical items with awareness of style\n• Occasional errors in word choice",
                "indicators": {
                    "lexical_diversity": (0.5, 0.65),
                    "advanced_words": (0.15, 0.25),
                    "collocations": (0.6, 0.75),
                    "idioms_used": (1, 3),
                    "precision_score": (0.7, 0.85)
                }
            },
            6: {
                "description": "• Uses adequate range of vocabulary for the task\n• Attempts to use less common vocabulary but with inaccuracies\n• Some errors in word choice and collocation",
                "indicators": {
                    "lexical_diversity": (0.4, 0.55),
                    "advanced_words": (0.10, 0.20),
                    "collocations": (0.5, 0.65),
                    "idioms_used": (0, 2),
                    "precision_score": (0.6, 0.75)
                }
            },
            5: {
                "description": "• Uses limited range of vocabulary, minimally adequate\n• Noticeable errors in word choice and collocation\n• May use simple vocabulary repetitively",
                "indicators": {
                    "lexical_diversity": (0.3, 0.45),
                    "advanced_words": (0.05, 0.15),
                    "collocations": (0.4, 0.55),
                    "idioms_used": (0, 1),
                    "precision_score": (0.5, 0.65)
                }
            },
            4: {
                "description": "• Uses only basic vocabulary, may be repetitive\n• Limited control of word formation\n• Errors cause strain for listener",
                "indicators": {
                    "lexical_diversity": (0.2, 0.35),
                    "advanced_words": (0, 0.10),
                    "collocations": (0.3, 0.45),
                    "idioms_used": (0, 0),
                    "precision_score": (0.4, 0.55)
                }
            }
        }
        
        # Word lists for analysis
        self.basic_words = set([
            'the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have', 'i',
            'it', 'for', 'not', 'on', 'with', 'he', 'as', 'you', 'do', 'at',
            'this', 'but', 'his', 'by', 'from', 'they', 'we', 'say', 'her', 'she',
            'or', 'an', 'will', 'my', 'one', 'all', 'would', 'there', 'their', 'what',
            'so', 'up', 'out', 'if', 'about', 'who', 'get', 'which', 'go', 'me',
            'when', 'make', 'can', 'like', 'time', 'no', 'just', 'him', 'know', 'take',
            'people', 'into', 'year', 'your', 'good', 'some', 'could', 'them', 'see', 'other',
            'than', 'then', 'now', 'look', 'only', 'come', 'its', 'over', 'think', 'also',
            'back', 'after', 'use', 'two', 'how', 'our', 'work', 'first', 'well', 'way',
            'even', 'new', 'want', 'because', 'any', 'these', 'give', 'day', 'most', 'us'
        ])
        
        self.advanced_words = set([
            'nevertheless', 'consequently', 'furthermore', 'moreover', 'therefore',
            'however', 'although', 'despite', 'whereas', 'nonetheless',
            'consequence', 'significant', 'essential', 'crucial', 'fundamental',
            'phenomenon', 'contemporary', 'perspective', 'approach', 'methodology',
            'innovation', 'technology', 'development', 'environment', 'sustainable',
            'globalization', 'economic', 'political', 'cultural', 'social',
            'challenge', 'opportunity', 'potential', 'capacity', 'capability',
            'strategy', 'implementation', 'evaluation', 'assessment', 'analysis',
            'concept', 'theory', 'principle', 'philosophy', 'ideology',
            'communication', 'collaboration', 'cooperation', 'negotiation', 'mediation',
            'transform', 'evolve', 'develop', 'progress', 'advance',
            'complex', 'complicated', 'sophisticated', 'intricate', 'elaborate',
            'effective', 'efficient', 'productive', 'innovative', 'creative'
        ])
        
        # Common collocations
        self.common_collocations = [
            ('make', 'decision'), ('take', 'break'), ('do', 'homework'),
            ('heavy', 'rain'), ('strong', 'coffee'), ('fast', 'food'),
            ('high', 'quality'), ('deep', 'breath'), ('bright', 'future'),
            ('big', 'mistake'), ('small', 'talk'), ('quick', 'response'),
            ('long', 'time'), ('short', 'term'), ('good', 'luck'),
            ('bad', 'habit'), ('hard', 'work'), ('soft', 'voice'),
            ('sweet', 'dreams'), ('bitter', 'truth')
        ]
        
        # Common idioms
        self.common_idioms = [
            r'piece of cake', r'break the ice', r'hit the nail', r'cost an arm',
            r'beat around the bush', r'bite the bullet', r'call it a day',
            r'cut corners', r'get out of hand', r'give the benefit',
            r'go the extra mile', r'hit the sack', r'it\'s not rocket',
            r'kill two birds', r'let the cat out', r'miss the boat',
            r'on the ball', r'pull someone\'s leg', r'speak of the devil',
            r'straight from the horse', r'the last straw', r'under the weather'
        ]
        
        print("✅ Lexical Resource Assessor ready")
    
    def analyze_vocabulary(self, text: str) -> Dict:
        """
        Analyze vocabulary features from text
        """
        # Tokenize and clean text
        words = [word.lower() for word in word_tokenize(text) if word.isalpha()]
        word_count = len(words)
        
        if word_count == 0:
            return {
                "word_count": 0,
                "lexical_diversity": 0,
                "basic_word_ratio": 0,
                "advanced_word_ratio": 0,
                "collocation_score": 0,
                "idiom_count": 0,
                "precision_score": 0,
                "unique_words": 0
            }
        
        # Calculate lexical diversity (Type-Token Ratio)
        unique_words = set(words)
        lexical_diversity = len(unique_words) / word_count
        
        # Analyze word levels
        basic_words = [w for w in words if w in self.basic_words]
        advanced_words = [w for w in words if w in self.advanced_words]
        other_words = [w for w in words if w not in self.basic_words and w not in self.advanced_words]
        
        basic_ratio = len(basic_words) / word_count
        advanced_ratio = len(advanced_words) / word_count
        other_ratio = len(other_words) / word_count
        
        # Calculate collocation score
        collocation_score = self._calculate_collocation_score(text)
        
        # Count idioms
        idiom_count = 0
        for idiom in self.common_idioms:
            if re.search(idiom, text.lower()):
                idiom_count += 1
        
        # Calculate word precision score (based on variety and appropriateness)
        precision_score = self._calculate_precision_score(words, unique_words, advanced_ratio)
        
        # Average word length (simple proxy for vocabulary sophistication)
        avg_word_length = sum(len(w) for w in words) / word_count
        
        return {
            "word_count": word_count,
            "unique_words": len(unique_words),
            "lexical_diversity": lexical_diversity,
            "basic_word_ratio": basic_ratio,
            "advanced_word_ratio": advanced_ratio,
            "other_word_ratio": other_ratio,
            "collocation_score": collocation_score,
            "idiom_count": idiom_count,
            "precision_score": precision_score,
            "avg_word_length": avg_word_length,
            "vocabulary_richness": min(1.0, (advanced_ratio * 2) + (lexical_diversity * 0.5))
        }
    
    def _calculate_collocation_score(self, text: str) -> float:
        """Calculate collocation appropriateness score"""
        text_lower = text.lower()
        found_collocations = 0
        
        for word1, word2 in self.common_collocations:
            pattern = rf'\b{word1}\s+\w+\s+{word2}\b|\b{word1}\s+{word2}\b'
            if re.search(pattern, text_lower):
                found_collocations += 1
        
        total_collocations = len(self.common_collocations)
        return found_collocations / max(1, total_collocations) * 0.8
    
    def _calculate_precision_score(self, words: List[str], unique_words: set, advanced_ratio: float) -> float:
        """Calculate word choice precision score"""
        # Factors:
        # 1. Repetition penalty (repeating same words too often)
        word_counts = {}
        for word in words:
            word_counts[word] = word_counts.get(word, 0) + 1
        
        repetition_penalty = sum(1 for count in word_counts.values() if count > 3) / len(words)
        
        # 2. Variety bonus (using different words)
        variety_bonus = len(unique_words) / len(words)
        
        # 3. Advanced word bonus
        advanced_bonus = advanced_ratio * 2
        
        # Combine scores
        precision = (variety_bonus * 0.5) + (advanced_bonus * 0.3) - (repetition_penalty * 0.2)
        return max(0, min(1, precision))
    
    def calculate_band_score(self, vocabulary_metrics: Dict) -> Tuple[int, Dict]:
        """
        Calculate IELTS band for Lexical Resource
        """
        lexical_diversity = vocabulary_metrics.get("lexical_diversity", 0)
        advanced_ratio = vocabulary_metrics.get("advanced_word_ratio", 0)
        collocation_score = vocabulary_metrics.get("collocation_score", 0)
        idiom_count = vocabulary_metrics.get("idiom_count", 0)
        precision_score = vocabulary_metrics.get("precision_score", 0)
        
        # Calculate composite score with weights
        weights = {
            "lexical_diversity": 0.25,
            "advanced_ratio": 0.25,
            "collocation_score": 0.20,
            "idiom_count": 0.15,
            "precision_score": 0.15
        }
        
        # Normalize idiom count (0-3 = 0-1 scale)
        normalized_idioms = min(1.0, idiom_count / 3.0)
        
        scores = {
            "lexical_diversity": lexical_diversity,
            "advanced_ratio": min(1.0, advanced_ratio * 4),  # Scale up
            "collocation_score": collocation_score,
            "idiom_count": normalized_idioms,
            "precision_score": precision_score
        }
        
        composite_score = sum(scores[feature] * weight for feature, weight in weights.items())
        
        # Map to band
        band = self._map_to_band(composite_score, lexical_diversity, advanced_ratio, idiom_count)
        
        breakdown = {
            "composite_score": round(composite_score, 3),
            "lexical_diversity": round(lexical_diversity, 3),
            "advanced_word_ratio": round(advanced_ratio, 3),
            "collocation_score": round(collocation_score, 3),
            "idiom_count": idiom_count,
            "precision_score": round(precision_score, 3),
            "vocabulary_richness": round(vocabulary_metrics.get("vocabulary_richness", 0), 3)
        }
        
        return band, breakdown
    
    def _map_to_band(self, composite: float, lexical_diversity: float, 
                    advanced_ratio: float, idiom_count: int) -> int:
        """Map to IELTS bands for Lexical Resource"""
        
        # Band 9
        if (composite >= 0.90 and 
            lexical_diversity >= 0.65 and 
            advanced_ratio >= 0.20 and 
            idiom_count >= 2):
            return 9
        
        # Band 8
        elif (composite >= 0.80 and 
              lexical_diversity >= 0.55 and 
              advanced_ratio >= 0.15):
            return 8
        
        # Band 7
        elif (composite >= 0.70 and 
              lexical_diversity >= 0.45 and 
              advanced_ratio >= 0.10):
            return 7
        
        # Band 6
        elif (composite >= 0.60 and 
              lexical_diversity >= 0.35):
            return 6
        
        # Band 5
        elif (composite >= 0.50 and 
              lexical_diversity >= 0.25):
            return 5
        
        # Band 4
        elif (composite >= 0.40):
            return 4
        
        # Band 3
        elif (composite >= 0.30):
            return 3
        
        # Band 2
        elif (composite >= 0.20):
            return 2
        
        # Band 1
        else:
            return 1
    
    def generate_feedback(self, band: int, vocabulary_metrics: Dict, 
                         breakdown: Dict) -> str:
        """Generate IELTS-style feedback for Lexical Resource"""
        
        feedback = "\n🎯 IELTS LEXICAL RESOURCE ASSESSMENT\n"
        feedback += "=" * 70 + "\n\n"
        
        # Band score
        feedback += f"📊 OVERALL BAND SCORE: {band}/9\n\n"
        
        # Official descriptor
        if band in self.band_descriptors:
            desc = self.band_descriptors[band]
            feedback += "📋 OFFICIAL IELTS DESCRIPTOR:\n"
            feedback += f"{desc['description']}\n\n"
        
        # Detailed metrics
        feedback += "🔍 DETAILED ANALYSIS:\n"
        feedback += f"  • Lexical Diversity: {breakdown['lexical_diversity']:.3f} (higher = more varied vocabulary)\n"
        feedback += f"  • Advanced Vocabulary: {vocabulary_metrics.get('advanced_word_ratio', 0):.1%}\n"
        feedback += f"  • Collocation Accuracy: {breakdown['collocation_score']:.1%}\n"
        feedback += f"  • Idioms Used: {breakdown['idiom_count']}\n"
        feedback += f"  • Word Precision: {breakdown['precision_score']:.1%}\n"
        feedback += f"  • Vocabulary Richness: {breakdown['vocabulary_richness']:.1%}\n"
        feedback += f"  • Composite Score: {breakdown['composite_score']:.3f}/1.0\n\n"
        
        # Vocabulary breakdown
        basic_ratio = vocabulary_metrics.get('basic_word_ratio', 0)
        advanced_ratio = vocabulary_metrics.get('advanced_word_ratio', 0)
        other_ratio = vocabulary_metrics.get('other_word_ratio', 0)
        
        feedback += "📊 VOCABULARY DISTRIBUTION:\n"
        feedback += f"  • Basic Words: {basic_ratio:.1%}\n"
        feedback += f"  • Advanced Words: {advanced_ratio:.1%}\n"
        feedback += f"  • Other Words: {other_ratio:.1%}\n\n"
        
        # Show advanced words if found
        if advanced_ratio > 0.1:
            feedback += "✅ ADVANCED VOCABULARY IDENTIFIED:\n"
            # You could add logic here to list the advanced words found
            feedback += "  • Some advanced vocabulary detected\n"
            feedback += "  • Good use of less common words\n\n"
        
        # Show idioms if found
        if breakdown['idiom_count'] > 0:
            feedback += "✅ IDIOMS USED:\n"
            feedback += f"  • {breakdown['idiom_count']} idiom(s) detected\n"
            feedback += "  • Appropriate use of idiomatic expressions\n\n"
        
        # Improvement suggestions
        feedback += "🎯 AREAS FOR IMPROVEMENT:\n"
        
        if band <= 5:
            feedback += "  1. Expand basic vocabulary range\n"
            feedback += "  2. Learn common collocations (e.g., 'make a decision', 'take a break')\n"
            feedback += "  3. Practice using synonyms to avoid repetition\n"
            feedback += "  4. Study topic-specific vocabulary lists\n"
        
        elif band <= 6.5:
            feedback += "  1. Incorporate more advanced vocabulary\n"
            feedback += "  2. Work on collocation accuracy\n"
            feedback += "  3. Learn and use 2-3 common idioms appropriately\n"
            feedback += "  4. Practice paraphrasing using different words\n"
        
        elif band <= 7.5:
            feedback += "  1. Refine use of sophisticated vocabulary\n"
            feedback += "  2. Master academic and formal vocabulary\n"
            feedback += "  3. Use idioms naturally and appropriately\n"
            feedback += "  4. Work on precise word choice for different contexts\n"
        
        else:
            feedback += "  1. Maintain excellent vocabulary range\n"
            feedback += "  2. Focus on nuanced word choice\n"
            feedback += "  3. Expand knowledge of specialized vocabulary\n"
            feedback += "  4. Practice using idioms and proverbs naturally\n"
        
        # Study recommendations
        feedback += "\n📚 STUDY RECOMMENDATIONS:\n"
        if band < 7:
            feedback += "  1. Use vocabulary flashcards (Anki, Quizlet)\n"
            feedback += "  2. Read English newspapers/articles daily\n"
            feedback += "  3. Keep a vocabulary journal\n"
            feedback += "  4. Practice with IELTS vocabulary lists\n"
        else:
            feedback += "  1. Read academic journals in your field\n"
            feedback += "  2. Study vocabulary in context (phrases not isolated words)\n"
            feedback += "  3. Learn word families and derivatives\n"
            feedback += "  4. Practice using new words in speaking and writing\n"
        
        return feedback
    
    def assess(self, text: str) -> Tuple[int, str, Dict]:
        """
        Main assessment function for Lexical Resource
        """
        if not text or len(text.split()) < 10:
            return 3, "Insufficient text for vocabulary assessment", {}
        
        # Step 1: Analyze vocabulary
        vocabulary_metrics = self.analyze_vocabulary(text)
        
        # Step 2: Calculate band score
        band, breakdown = self.calculate_band_score(vocabulary_metrics)
        
        # Step 3: Generate feedback
        feedback = self.generate_feedback(band, vocabulary_metrics, breakdown)
        
        # Detailed analysis
        detailed = {
            "band": band,
            "vocabulary_metrics": vocabulary_metrics,
            "breakdown": breakdown
        }
        
        return band, feedback, detailed

# ========================
# LEXICAL RESOURCE WRAPPER
# ========================
class LexicalResourceIELTSAssessor:
    """
    Wrapper for Zero-Shot Lexical Resource Assessor
    """
    
    def __init__(self):
        print("\n🔄 Loading Zero-Shot Lexical Resource Assessor...")
        self.assessor = ZeroShotLexicalResource()
        print("✅ IELTS Lexical Resource Assessor ready")
    
    def assess_lexical_band(self, transcript):
        """
        Main function: Assess Lexical Resource (Band 1-9)
        """
        if not transcript or len(transcript.split()) < 10:
            return 3, "Insufficient text for vocabulary analysis"
        
        try:
            band, feedback, detailed = self.assessor.assess(transcript)
            return band, feedback
        except Exception as e:
            print(f"⚠️  Vocabulary assessment error: {e}")
            return 4, "Vocabulary assessment failed"

# ========================
# REINFORCEMENT LEARNING MODEL
# ========================
class IELTSRLModel(nn.Module):
    """Deep Q-Network for IELTS Assessment"""
    
    def __init__(self, state_dim, action_dim, hidden_dim=RL_HIDDEN_DIM):
        super(IELTSRLModel, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim)
        )
    
    def forward(self, state):
        return self.network(state)

class ReplayBuffer:
    """Experience replay buffer for DQN"""
    
    def __init__(self, capacity=RL_MEMORY_SIZE):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, min(len(self.buffer), batch_size))
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.array(states), np.array(actions), np.array(rewards), 
                np.array(next_states), np.array(dones))
    
    def __len__(self):
        return len(self.buffer)

class IELTSPronunciationRLAgent:
    """RL Agent for learning IELTS assessment"""
    
    def __init__(self, state_dim=RL_STATE_DIM, action_dim=RL_ACTION_DIM):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Q-networks
        self.policy_net = IELTSRLModel(state_dim, action_dim).to(self.device)
        self.target_net = IELTSRLModel(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        # Optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=RL_LEARNING_RATE)
        
        # Experience replay
        self.memory = ReplayBuffer(RL_MEMORY_SIZE)
        
        # Training parameters
        self.epsilon = RL_EPSILON_START
        self.steps_done = 0
        self.update_count = 0
        
        # Feature extractors for FE approach
        self.grammar_assessor = GrammarIELTSAssessor()
        self.fluency_assessor = FluencyCoherenceIELTSAssessor()
        self.lexical_assessor = LexicalResourceIELTSAssessor()
        
        print(f"✅ IELTS RL Agent initialized on {self.device}")
    
    def extract_features(self, wav_file, transcript, audio_duration=None):
        """Extract features from FE approach for RL state"""
        features = []
        
        try:
            # 1. Pronunciation features
            pron_band, pron_details = assess_ielts_pronunciation(wav_file, return_detailed=True)
            if "features" in pron_details:
                pron_features = [
                    pron_details["features"].get("intelligibility", 0),
                    pron_details["features"].get("pitch_range", 0) / 500,  # Normalize
                    pron_details["features"].get("pitch_variation", 0),
                    pron_details["features"].get("pause_ratio", 0),
                    pron_details["features"].get("energy_variance", 0) * 100,
                    pron_details["features"].get("speaking_rate", 0) / 10,  # Normalize
                    pron_details["features"].get("rhythm_regularity", 0)
                ]
                features.extend(pron_features)
            
            # 2. Grammar features
            grammar_band, grammar_feedback = self.grammar_assessor.assess_grammatical_band(transcript)
            grammar_assessor = ZeroShotIELTSGrammar()
            grammar_result = grammar_assessor.assess(transcript)
            if len(grammar_result) > 2:
                grammar_details = grammar_result[2]
                grammar_features = [
                    grammar_details.get("breakdown", {}).get("error_density", 0) / 30,
                    grammar_details.get("breakdown", {}).get("complex_ratio", 0),
                    grammar_details.get("breakdown", {}).get("structures_used", 0) / 14,
                    grammar_details.get("breakdown", {}).get("sentence_variety", 0),
                    len(grammar_details.get("errors", {}).get("high_severity", [])) / 10
                ]
                features.extend(grammar_features)
            
            # 3. Fluency features
            fluency_band, fluency_feedback = self.fluency_assessor.assess_fluency_coherence_band(transcript, audio_duration)
            fluency_assessor = ZeroShotFluencyCoherence()
            fluency_result = fluency_assessor.assess(transcript, audio_duration)
            if len(fluency_result) > 2:
                fluency_details = fluency_result[2]
                fluency_features = [
                    fluency_details.get("fluency_metrics", {}).get("disfluency_rate", 0) / 30,
                    fluency_details.get("fluency_metrics", {}).get("speech_rate_words", 0) / 200,
                    fluency_details.get("coherence_metrics", {}).get("cohesive_devices_rate", 0) / 12,
                    fluency_details.get("coherence_metrics", {}).get("topic_coherence", 0),
                    fluency_details.get("coherence_metrics", {}).get("elaboration_ratio", 0)
                ]
                features.extend(fluency_features)
            
            # 4. Lexical features
            lexical_band, lexical_feedback = self.lexical_assessor.assess_lexical_band(transcript)
            lexical_assessor = ZeroShotLexicalResource()
            lexical_result = lexical_assessor.assess(transcript)
            if len(lexical_result) > 2:
                lexical_details = lexical_result[2]
                lexical_features = [
                    lexical_details.get("vocabulary_metrics", {}).get("lexical_diversity", 0),
                    lexical_details.get("vocabulary_metrics", {}).get("advanced_word_ratio", 0),
                    lexical_details.get("vocabulary_metrics", {}).get("collocation_score", 0),
                    lexical_details.get("vocabulary_metrics", {}).get("idiom_count", 0) / 3,
                    lexical_details.get("vocabulary_metrics", {}).get("precision_score", 0)
                ]
                features.extend(lexical_features)
            
            # Add basic features
            words = transcript.split()
            word_count = len(words)
            unique_words = len(set(words))
            
            features.extend([
                word_count / 100,  # Normalize
                unique_words / max(1, word_count),
                len(transcript) / 1000,  # Normalize
                audio_duration / 60 if audio_duration else 0
            ])
            
            # Pad if needed
            while len(features) < self.state_dim:
                features.append(0.0)
            
            # Truncate if too long
            features = features[:self.state_dim]
            
            return np.array(features, dtype=np.float32), {
                "pronunciation": pron_band,
                "grammar": grammar_band,
                "fluency": fluency_band,
                "lexical": lexical_band
            }
            
        except Exception as e:
            print(f"⚠️ Feature extraction error: {e}")
            # Return default features
            default_features = np.zeros(self.state_dim, dtype=np.float32)
            default_bands = {
                "pronunciation": 4,
                "grammar": 4,
                "fluency": 4,
                "lexical": 4
            }
            return default_features, default_bands
    
    def select_action(self, state, evaluate=False):
        """Select action using epsilon-greedy policy"""
        if not evaluate and random.random() < self.epsilon:
            # Random action (exploration)
            return np.random.uniform(1, 9, size=self.action_dim)
        
        # Greedy action (exploitation)
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_tensor)
            action = q_values.cpu().numpy()[0]
            # Scale to band scores (1-9)
            action = np.clip(action, 1, 9)
            return action
    
    def compute_reward(self, rl_bands, fe_bands):
        """Compute reward based on how close RL bands are to FE bands"""
        # Mean squared error between RL and FE bands
        mse = np.mean([(rl_bands[i] - fe_bands[i]) ** 2 for i in range(len(rl_bands))])
        
        # Reward is negative MSE (we want to minimize error)
        reward = -mse
        
        # Bonus for exact match
        if np.allclose(rl_bands, fe_bands, atol=0.5):
            reward += 1.0
        
        return float(reward)
    
    def update(self, batch_size=RL_BATCH_SIZE):
        """Update the network using experience replay"""
        if len(self.memory) < batch_size:
            return
        
        # Sample batch
        states, actions, rewards, next_states, dones = self.memory.sample(batch_size)
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device).unsqueeze(1)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device).unsqueeze(1)
        
        # Current Q values
        current_q = self.policy_net(states)
        
        # Next Q values from target network
        with torch.no_grad():
            next_q = self.target_net(next_states)
            max_next_q = next_q.max(1)[0].unsqueeze(1)
        
        # Compute target Q values
        target_q = rewards + RL_GAMMA * max_next_q * (1 - dones)
        
        # Compute loss (Mean Squared Error)
        loss = nn.MSELoss()(current_q, target_q)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        
        # Update target network
        self.update_count += 1
        if self.update_count % RL_TARGET_UPDATE == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        
        # Decay epsilon
        self.epsilon = max(RL_EPSILON_END, self.epsilon * RL_EPSILON_DECAY)
        
        return loss.item()
    
    def save_model(self, path="ielts_rl_model.pth"):
        """Save the model"""
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'steps_done': self.steps_done
        }, path)
        print(f"💾 Model saved to {path}")
    
    def load_model(self, path="ielts_rl_model.pth"):
        """Load the model"""
        if os.path.exists(path):
            checkpoint = torch.load(path, map_location=self.device)
            self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
            self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.epsilon = checkpoint['epsilon']
            self.steps_done = checkpoint['steps_done']
            print(f"📂 Model loaded from {path}")
        else:
            print(f"⚠️  No model found at {path}, starting fresh")
    
    def train_episode(self, wav_file, transcript, audio_duration=None):
        """Train on one episode (one recording)"""
        # Extract features and get FE bands
        state, fe_bands_dict = self.extract_features(wav_file, transcript, audio_duration)
        
        # Convert FE bands to array
        fe_bands = np.array([
            fe_bands_dict["pronunciation"],
            fe_bands_dict["grammar"],
            fe_bands_dict["fluency"],
            fe_bands_dict["lexical"]
        ])
        
        # Select action (RL bands)
        rl_bands = self.select_action(state)
        
        # Compute reward
        reward = self.compute_reward(rl_bands, fe_bands)
        
        # Store in replay buffer (use current state as next state since it's one-step)
        self.memory.push(state, rl_bands, reward, state, True)
        
        # Update network
        loss = self.update()
        
        self.steps_done += 1
        
        return {
            "state": state,
            "rl_bands": rl_bands,
            "fe_bands": fe_bands,
            "reward": reward,
            "loss": loss,
            "epsilon": self.epsilon
        }
    
    def assess_with_rl(self, wav_file, transcript, audio_duration=None):
        """Assess using RL model (without training)"""
        # Extract features
        state, _ = self.extract_features(wav_file, transcript, audio_duration)
        
        # Get RL bands
        rl_bands = self.select_action(state, evaluate=True)
        
        return {
            "pronunciation": float(rl_bands[0]),
            "grammar": float(rl_bands[1]),
            "fluency": float(rl_bands[2]),
            "lexical": float(rl_bands[3]),
            "overall": float(np.mean(rl_bands))
        }

# ========================
# COMPREHENSIVE IELTS ASSESSMENT WITH RL
# ========================
class ComprehensiveIELTSAssessorWithRL:
    """
    Complete IELTS Speaking Assessment with RL integration
    """
    
    def __init__(self, use_rl=True):
        print("🔄 Initializing Comprehensive IELTS Speaking Assessor with RL...")
        self.grammar_assessor = GrammarIELTSAssessor()
        self.fluency_assessor = FluencyCoherenceIELTSAssessor()
        self.lexical_assessor = LexicalResourceIELTSAssessor()
        self.use_rl = use_rl
        
        if use_rl:
            self.rl_agent = IELTSPronunciationRLAgent()
            # Try to load existing model
            self.rl_agent.load_model()
        
        print("✅ Comprehensive IELTS Assessor with RL ready")
    
    def assess_all_criteria(self, wav_file, transcript, audio_duration, train_rl=True):
        """
        Assess all 4 IELTS speaking criteria with optional RL training
        """
        results = {}
        
        print("\n" + "=" * 60)
        print("🔍 ASSESSING ALL IELTS CRITERIA...")
        print("=" * 60)
        
        # 1. Assess with FE approach (for reward computation)
        print("\n🎯 1. ASSESSING WITH FEATURE ENGINEERING...")
        try:
            pronunciation_band, pronunciation_details = assess_ielts_pronunciation(wav_file, return_detailed=True)
            results["fe_pronunciation"] = {
                "band": pronunciation_band,
                "feedback": pronunciation_details.get("feedback_text", ""),
                "details": pronunciation_details
            }
            print(f"   ✅ FE Pronunciation Band: {pronunciation_band}/9")
        except Exception as e:
            print(f"   ❌ FE Pronunciation assessment failed: {e}")
            results["fe_pronunciation"] = {"band": 4, "feedback": "Assessment failed"}
        
        # 2. Assess Grammar
        print("\n🎯 2. ASSESSING GRAMMATICAL RANGE & ACCURACY...")
        try:
            grammar_band, grammar_feedback = self.grammar_assessor.assess_grammatical_band(transcript)
            results["fe_grammar"] = {
                "band": grammar_band,
                "feedback": grammar_feedback
            }
            print(f"   ✅ FE Grammar Band: {grammar_band}/9")
        except Exception as e:
            print(f"   ❌ FE Grammar assessment failed: {e}")
            results["fe_grammar"] = {"band": 4, "feedback": "Assessment failed"}
        
        # 3. Assess Fluency & Coherence
        print("\n🎯 3. ASSESSING FLUENCY & COHERENCE...")
        try:
            fluency_band, fluency_feedback = self.fluency_assessor.assess_fluency_coherence_band(transcript, audio_duration)
            results["fe_fluency_coherence"] = {
                "band": fluency_band,
                "feedback": fluency_feedback
            }
            print(f"   ✅ FE Fluency & Coherence Band: {fluency_band}/9")
        except Exception as e:
            print(f"   ❌ FE Fluency assessment failed: {e}")
            results["fe_fluency_coherence"] = {"band": 4, "feedback": "Assessment failed"}
        
        # 4. Assess Lexical Resource
        print("\n🎯 4. ASSESSING LEXICAL RESOURCE...")
        try:
            lexical_band, lexical_feedback = self.lexical_assessor.assess_lexical_band(transcript)
            results["fe_lexical_resource"] = {
                "band": lexical_band,
                "feedback": lexical_feedback
            }
            print(f"   ✅ FE Lexical Resource Band: {lexical_band}/9")
        except Exception as e:
            print(f"   ❌ FE Vocabulary assessment failed: {e}")
            results["fe_lexical_resource"] = {"band": 4, "feedback": "Assessment failed"}
        
        # Calculate FE overall band
        fe_bands = [
            results["fe_pronunciation"]["band"],
            results["fe_grammar"]["band"],
            results["fe_fluency_coherence"]["band"],
            results["fe_lexical_resource"]["band"]
        ]
        fe_overall_band = sum(fe_bands) / len(fe_bands)
        results["fe_overall_band"] = fe_overall_band
        
        # 5. RL Assessment (if enabled)
        if self.use_rl:
            print("\n🎯 5. ASSESSING WITH REINFORCEMENT LEARNING...")
            try:
                rl_assessment = self.rl_agent.assess_with_rl(wav_file, transcript, audio_duration)
                results["rl_assessment"] = rl_assessment
                print(f"   ✅ RL Pronunciation Band: {rl_assessment['pronunciation']:.1f}/9")
                print(f"   ✅ RL Grammar Band: {rl_assessment['grammar']:.1f}/9")
                print(f"   ✅ RL Fluency Band: {rl_assessment['fluency']:.1f}/9")
                print(f"   ✅ RL Lexical Band: {rl_assessment['lexical']:.1f}/9")
                print(f"   ✅ RL Overall Band: {rl_assessment['overall']:.1f}/9")
                
                # Calculate error between RL and FE
                rl_bands_array = np.array([
                    rl_assessment['pronunciation'],
                    rl_assessment['grammar'],
                    rl_assessment['fluency'],
                    rl_assessment['lexical']
                ])
                fe_bands_array = np.array(fe_bands)
                error = np.mean(np.abs(rl_bands_array - fe_bands_array))
                results["rl_fe_error"] = error
                print(f"   📊 RL-FE Error: {error:.3f}")
                
                # Train RL agent if requested
                if train_rl:
                    print("\n🎯 6. TRAINING RL AGENT...")
                    training_result = self.rl_agent.train_episode(wav_file, transcript, audio_duration)
                    results["rl_training"] = training_result
                    print(f"   ✅ RL Training Reward: {training_result['reward']:.3f}")
                    print(f"   ✅ RL Loss: {training_result['loss']:.5f}" if training_result['loss'] else "   ✅ RL Loss: N/A")
                    print(f"   ✅ RL Epsilon: {training_result['epsilon']:.3f}")
                    
                    # Save model periodically
                    if self.rl_agent.steps_done % 10 == 0:
                        self.rl_agent.save_model()
                
            except Exception as e:
                print(f"   ❌ RL assessment/training failed: {e}")
                results["rl_assessment"] = {"error": str(e)}
        
        return results
    
    def generate_comprehensive_report(self, results, transcript, wav_file):
        """
        Generate comprehensive report with FE and RL results
        """
        report = "=" * 80 + "\n"
        report += "🎯 COMPREHENSIVE IELTS SPEAKING ASSESSMENT REPORT (FE + RL)\n"
        report += "=" * 80 + "\n\n"
        
        # Overall bands comparison
        report += "📊 OVERALL BANDS COMPARISON:\n"
        report += "=" * 40 + "\n"
        
        if "fe_overall_band" in results:
            report += f"  • FE (Feature Engineering):    {results['fe_overall_band']:.1f}/9\n"
        
        if "rl_assessment" in results and "overall" in results["rl_assessment"]:
            report += f"  • RL (Reinforcement Learning): {results['rl_assessment']['overall']:.1f}/9\n"
        
        if "rl_fe_error" in results:
            report += f"  • RL-FE Error:                 {results['rl_fe_error']:.3f}\n"
        
        report += "\n"
        
        # Band breakdown comparison
        report += "📈 BAND BREAKDOWN COMPARISON:\n"
        report += "=" * 40 + "\n"
        report += "  Criterion           FE Band   RL Band   Difference\n"
        report += "  ------------------  --------  --------  ----------\n"
        
        criteria = ["pronunciation", "grammar", "fluency_coherence", "lexical_resource"]
        rl_criteria = ["pronunciation", "grammar", "fluency", "lexical"]
        
        for fe_crit, rl_crit in zip(criteria, rl_criteria):
            fe_band = results.get(f"fe_{fe_crit}", {}).get("band", "N/A")
            rl_band = results.get("rl_assessment", {}).get(rl_crit, "N/A")
            
            if fe_band != "N/A" and rl_band != "N/A":
                diff = abs(float(rl_band) - float(fe_band))
                report += f"  {fe_crit.replace('_', ' ').title():18} {fe_band:8.1f} {rl_band:8.1f} {diff:10.2f}\n"
            else:
                report += f"  {fe_crit.replace('_', ' ').title():18} {str(fe_band):8} {str(rl_band):8} {'N/A':10}\n"
        
        report += "\n"
        
        # RL Learning Stats
        if "rl_training" in results:
            report += "🤖 RL LEARNING STATISTICS:\n"
            report += "=" * 40 + "\n"
            report += f"  • Reward: {results['rl_training']['reward']:.3f}\n"
            if results['rl_training'].get('loss'):
                report += f"  • Loss: {results['rl_training']['loss']:.5f}\n"
            report += f"  • Epsilon: {results['rl_training']['epsilon']:.3f}\n"
            report += f"  • Steps Done: {self.rl_agent.steps_done if hasattr(self, 'rl_agent') else 'N/A'}\n"
            report += "\n"
        
        # Add FE feedback from each criterion
        report += "🔍 DETAILED FE FEEDBACK BY CRITERION:\n"
        report += "=" * 80 + "\n\n"
        
        for crit in criteria:
            crit_name = crit.replace('_', ' ').title()
            if f"fe_{crit}" in results and "feedback" in results[f"fe_{crit}"]:
                report += f"{crit_name}:\n"
                report += "-" * 40 + "\n"
                report += results[f"fe_{crit}"]["feedback"] + "\n\n"
        
        # Improvement plan based on both FE and RL
        report += "💡 COMBINED IMPROVEMENT PLAN:\n"
        report += "=" * 80 + "\n\n"
        
        # Identify weakest area from FE
        fe_bands_dict = {}
        for crit in criteria:
            if f"fe_{crit}" in results:
                fe_bands_dict[crit] = results[f"fe_{crit}"]["band"]
        
        if fe_bands_dict:
            weakest_fe = min(fe_bands_dict, key=fe_bands_dict.get)
            report += f"🎯 FE PRIORITY AREA: {weakest_fe.replace('_', ' ').title()} (Band: {fe_bands_dict[weakest_fe]})\n\n"
        
        # Identify weakest area from RL
        if "rl_assessment" in results:
            rl_bands = results["rl_assessment"]
            rl_criteria_simple = {
                "pronunciation": rl_bands.get("pronunciation", 4),
                "grammar": rl_bands.get("grammar", 4),
                "fluency": rl_bands.get("fluency", 4),
                "lexical": rl_bands.get("lexical", 4)
            }
            weakest_rl = min(rl_criteria_simple, key=rl_criteria_simple.get)
            report += f"🎯 RL PRIORITY AREA: {weakest_rl.title()} (Band: {rl_criteria_simple[weakest_rl]:.1f})\n\n"
        
        # Combined recommendations
        report += "WEEKLY PRACTICE PLAN:\n"
        report += "1. Record yourself daily and compare FE vs RL assessments\n"
        report += "2. Focus on your weakest area identified above\n"
        report += "3. Use specific exercises for that criterion\n"
        report += "4. Track your progress with both FE and RL scores\n"
        report += "5. The RL model will improve as you use it more\n"
        
        # RL-specific note
        if self.use_rl:
            report += "\n"
            report += "🤖 RL MODEL NOTE:\n"
            report += "The reinforcement learning model is learning from your assessments.\n"
            report += "As you use it more, it will become more accurate and personalized.\n"
            report += f"Current training steps: {self.rl_agent.steps_done if hasattr(self, 'rl_agent') else 'N/A'}\n"
        
        return report

# ========================
# MAIN EXECUTION WITH RL
# ========================
if __name__ == "__main__":
    print("=" * 80)
    print("🎯 AI IELTS COMPREHENSIVE SPEAKING ASSESSOR WITH RL")
    print("=" * 80)
    print("📋 Features:")
    print("   1. Feature Engineering (FE) assessment")
    print("   2. Reinforcement Learning (RL) assessment")
    print("   3. RL learns from FE assessments")
    print("   4. Online training with user recordings")
    print("=" * 80)
    
    # Pre-load Whisper model
    get_whisper_model()
    
    # Initialize comprehensive assessor with RL
    assessor = ComprehensiveIELTSAssessorWithRL(use_rl=True)
    
    # Main loop
    episode_count = 0
    while True:
        print(f"\n{'=' * 60}")
        print(f"EPISODE {episode_count + 1}")
        print('=' * 60)
        
        # Ask user what they want to do
        print("\nOptions:")
        print("1. Record and assess (train RL)")
        print("2. Assess existing WAV file")
        print("3. Exit")
        
        choice = input("\nSelect option (1-3): ").strip()
        
        if choice == "3":
            print("👋 Exiting...")
            break
        
        if choice == "1":
            # Record and transcribe
            result = record_and_transcribe()
            
            if result[0] and result[1]:
                wav_file, transcript = result
                
                # Get audio duration for assessment
                try:
                    audio, sr = librosa.load(wav_file, sr=SAMPLE_RATE)
                    audio_duration = len(audio) / sr
                except:
                    audio_duration = None
                
                # Assess with FE and train RL
                all_results = assessor.assess_all_criteria(
                    wav_file, transcript, audio_duration, train_rl=True
                )
                
                # Generate and display report
                print("\n" + "=" * 80)
                print("📊 ASSESSMENT RESULTS")
                print("=" * 80)
                
                report = assessor.generate_comprehensive_report(all_results, transcript, wav_file)
                print(report)
                
                # Save report
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                report_file = f"ielts_assessment_episode_{episode_count + 1}_{timestamp}.txt"
                with open(report_file, 'w', encoding='utf-8') as f:
                    f.write(report)
                print(f"📄 Report saved to: {report_file}")
                
                # Save RL model
                if hasattr(assessor, 'rl_agent'):
                    assessor.rl_agent.save_model(f"ielts_rl_model_episode_{episode_count + 1}.pth")
                
                episode_count += 1
        
        elif choice == "2":
            # Assess existing file
            wav_file = input("Enter WAV file path: ").strip()
            
            if not os.path.exists(wav_file):
                print(f"❌ File not found: {wav_file}")
                continue
            
            # Transcribe
            print("\n🎯 Transcribing...")
            try:
                model = get_whisper_model()
                result = model.transcribe(wav_file)
                transcript = result["text"].strip()
                print(f"✅ Transcript: {transcript[:100]}...")
            except Exception as e:
                print(f"❌ Transcription error: {e}")
                continue
            
            # Get audio duration
            try:
                audio, sr = librosa.load(wav_file, sr=SAMPLE_RATE)
                audio_duration = len(audio) / sr
            except:
                audio_duration = None
            
            # Assess with FE only (no RL training)
            print("\n📊 Assessing with Feature Engineering...")
            all_results = assessor.assess_all_criteria(
                wav_file, transcript, audio_duration, train_rl=False
            )
            
            # Display results
            print("\n" + "=" * 80)
            print("📊 FE ASSESSMENT RESULTS")
            print("=" * 80)
            
            print(f"Overall Band: {all_results.get('fe_overall_band', 'N/A'):.1f}/9")
            print(f"Pronunciation: {all_results.get('fe_pronunciation', {}).get('band', 'N/A')}/9")
            print(f"Grammar: {all_results.get('fe_grammar', {}).get('band', 'N/A')}/9")
            print(f"Fluency: {all_results.get('fe_fluency_coherence', {}).get('band', 'N/A')}/9")
            print(f"Lexical: {all_results.get('fe_lexical_resource', {}).get('band', 'N/A')}/9")
        
        else:
            print("❌ Invalid choice. Please select 1, 2, or 3.")
    
    print(f"\n✅ Session complete. Total episodes: {episode_count}")
    if hasattr(assessor, 'rl_agent'):
        print(f"🤖 RL Model trained for {assessor.rl_agent.steps_done} steps")
        assessor.rl_agent.save_model("ielts_rl_model_final.pth")