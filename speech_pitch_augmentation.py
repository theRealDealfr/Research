"""
speech_pitch_augmentation.py
-----------------------------
Reusable module for gender-stratified pitch shifting data augmentation
in pediatric speech-language pathology tasks.

Following:
"Multimodal LLMs are not all you need for Pediatric Speech Language Pathology"
(Fürst, Steindl, Schäfer, arXiv:2604.26568)

Key Rules:
- Stratification:
    Male ('M')   -> Shift pitch UP (+semitones)
    Female ('F') -> Shift pitch DOWN (-semitones)
- Missing/Unknown Gender:
    Explicitly handled (raises ValueError or skips; never silently imputed).
- Invariants:
    Target duration and sampling rate are strictly preserved.
    Metadata (speaker_id, age, gender, disorder label) remain unchanged.
"""

import numpy as np
import warnings

try:
    from audiomentations import PitchShift
    AUDIOMENTATIONS_AVAILABLE = True
except ImportError:
    AUDIOMENTATIONS_AVAILABLE = False

try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False


def pitch_shift_audio(
    audio: np.ndarray,
    sample_rate: int = 16000,
    shift_semitones: float = 0.0,
    backend: str = "audiomentations"
) -> np.ndarray:
    """
    Apply pitch shifting to a 1D audio array without changing its target duration.

    Parameters:
    -----------
    audio : np.ndarray
        Mono 1D float waveform.
    sample_rate : int
        Sample rate in Hz (default: 16000).
    shift_semitones : float
        Number of semitones to shift (positive for up, negative for down).
    backend : str
        'audiomentations' (preferred/fast) or 'librosa'.

    Returns:
    --------
    np.ndarray
        Pitch-shifted waveform with identical length, sample rate, and dtype.
    """
    if not isinstance(audio, np.ndarray):
        audio = np.asarray(audio, dtype=np.float32)

    if audio.ndim != 1:
        if audio.ndim == 2:
            audio = audio.mean(axis=1 if audio.shape[0] > audio.shape[1] else 0)
        audio = audio.squeeze()

    orig_len = len(audio)
    if orig_len == 0 or shift_semitones == 0.0:
        return audio.copy().astype(np.float32)

    try:
        if backend == "audiomentations" and AUDIOMENTATIONS_AVAILABLE:
            # Audiomentations takes exact min/max bounds
            shift = float(shift_semitones)
            augmenter = PitchShift(min_semitones=shift, max_semitones=shift, p=1.0)
            shifted = augmenter(samples=audio.astype(np.float32), sample_rate=sample_rate)
        elif LIBROSA_AVAILABLE:
            shifted = librosa.effects.pitch_shift(
                y=audio.astype(np.float32),
                sr=sample_rate,
                n_steps=float(shift_semitones)
            )
        elif AUDIOMENTATIONS_AVAILABLE:
            shift = float(shift_semitones)
            augmenter = PitchShift(min_semitones=shift, max_semitones=shift, p=1.0)
            shifted = augmenter(samples=audio.astype(np.float32), sample_rate=sample_rate)
        else:
            raise RuntimeError("Neither audiomentations nor librosa is available for pitch shifting.")

        # Ensure exact length preservation
        if len(shifted) < orig_len:
            shifted = np.pad(shifted, (0, orig_len - len(shifted)), mode='constant')
        elif len(shifted) > orig_len:
            shifted = shifted[:orig_len]

        return shifted.astype(np.float32)

    except Exception as e:
        warnings.warn(f"Pitch shift of {shift_semitones} st failed with error: {e}. Returning original audio.")
        return audio.copy().astype(np.float32)


def apply_gender_stratified_pitch_augmentation(
    audio: np.ndarray,
    sample_rate: int,
    gender: str,
    shift_min: float = 6.0,
    shift_max: float = 8.0,
    rng: np.random.RandomState = None
) -> np.ndarray:
    """
    Applies gender-stratified pitch augmentation:
    - Male ('M'): shifts pitch UP by a random value in [+shift_min, +shift_max] semitones.
    - Female ('F'): shifts pitch DOWN by a random value in [-shift_max, -shift_min] semitones.
    - Missing / Unexpected gender: raises ValueError explicitly.

    Parameters:
    -----------
    audio : np.ndarray
        Mono waveform.
    sample_rate : int
        Audio sampling rate.
    gender : str
        Gender identifier ('M' or 'F').
    shift_min : float
        Minimum absolute semitones shift (default 6.0 from paper baseline).
    shift_max : float
        Maximum absolute semitones shift (default 8.0 from paper baseline).
    rng : np.random.RandomState
        Random state for reproducibility.

    Returns:
    --------
    np.ndarray
        Augmented waveform.
    """
    if gender is None or str(gender).strip().upper() not in ("M", "F"):
        raise ValueError(
            f"Explicit gender handling: Sample gender must be 'M' or 'F', got '{gender}'. "
            "Do not silently assign or impute missing genders."
        )

    g = str(gender).strip().upper()
    if rng is None:
        rng = np.random.RandomState()

    shift_amount = rng.uniform(shift_min, shift_max)
    if g == "M":
        semitones = shift_amount       # Shift pitch UP
    else:
        semitones = -shift_amount      # Shift pitch DOWN

    return pitch_shift_audio(
        audio=audio,
        sample_rate=sample_rate,
        shift_semitones=semitones
    )
