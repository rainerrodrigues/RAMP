import os
import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew
from scipy.fft import fft

RAW_DATA_DIR = "data/raw/bearing_cwru"
PROCESSED_DIR = "data/processed"
OUTPUT_FILE = os.path.join(PROCESSED_DIR, "features.csv")


def compute_time_features(signal):
    """Extract time-domain vibration features"""
    return {
        "mean": np.mean(signal),
        "std": np.std(signal),
        "rms": np.sqrt(np.mean(signal**2)),
        "kurtosis": kurtosis(signal),
        "skewness": skew(signal),
        "peak_to_peak": np.ptp(signal),
        "max": np.max(signal),
        "min": np.min(signal),
    }


def compute_frequency_features(signal, sampling_rate=12000):
    """Extract frequency-domain features using FFT"""
    fft_vals = np.abs(fft(signal))
    freqs = np.fft.fftfreq(len(signal), 1 / sampling_rate)

    # Keep only positive frequencies
    positive_freqs = freqs[: len(freqs) // 2]
    positive_fft = fft_vals[: len(fft_vals) // 2]

    dominant_freq = positive_freqs[np.argmax(positive_fft)]
    spectral_energy = np.sum(positive_fft**2)

    return {
        "dominant_frequency": dominant_freq,
        "spectral_energy": spectral_energy,
    }


def load_signal(file_path):
    """Load vibration signal from CSV or TXT"""
    data = np.loadtxt(file_path)
    return data


def extract_features():
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    feature_rows = []

    for root, _, files in os.walk(RAW_DATA_DIR):
        for file in files:
            if file.endswith(".csv") or file.endswith(".txt"):
                file_path = os.path.join(root, file)

                try:
                    signal = load_signal(file_path)

                    time_feats = compute_time_features(signal)
                    freq_feats = compute_frequency_features(signal)

                    features = {**time_feats, **freq_feats}
                    features["file_name"] = file

                    # Simple label inference (you can improve later)
                    if "normal" in file.lower():
                        features["label"] = "normal"
                    else:
                        features["label"] = "fault"

                    feature_rows.append(features)

                except Exception as e:
                    print(f"Error processing {file}: {e}")

    df = pd.DataFrame(feature_rows)
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"✅ Features saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    extract_features()
