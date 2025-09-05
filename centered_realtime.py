# To run:
# python centered_realtime.py --source csv --plot --loop --data_path .
# python centered_realtime.py --source xdf --plot --loop --data_path .
# python centered_realtime.py --source xdf --feedback --port COM4 --baud 9600 --loop --data_path .

import os
import time
import argparse
from typing import Tuple, Optional, Iterable, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.signal import filtfilt, iirnotch, iirfilter, sosfiltfilt, butter
from sklearn.ensemble import RandomForestClassifier
from mne.time_frequency import psd_array_multitaper

def try_import(name):
    try:
        return __import__(name)
    except Exception:
        return None

pyxdf = try_import("pyxdf")
serial = try_import("serial")

# ---------- Configurações ----------
DEFAULT_FS = 256  # Hz
NOTCH_FREQ = 50
QUALITY_FACTOR = 40
ORDER = 8
LOWCUT = 4
HIGHCUT = 90

CHUNK_SIZE_SECONDS = 3
BETA_BAND = (12, 35)

TRIALS_DICT = {"neutral": 0, "relaxed": 1, "concentrating": 2}
CHANNEL_NAMES = ["TP9", "AF7", "AF8", "TP10"]

# ---------- Plot globals ----------
_PLOT_INITIALIZED = False
_FIG = None
_AX = None

def init_plot():
    global _PLOT_INITIALIZED, _FIG, _AX
    if not _PLOT_INITIALIZED:
        plt.ion()
        _FIG, _AX = plt.subplots(figsize=(9, 4))
        _PLOT_INITIALIZED = True

# ---------- Filtros ----------
def design_filters(fs: int):
    b_notch, a_notch = iirnotch(NOTCH_FREQ, QUALITY_FACTOR, fs)
    sos_lp = iirfilter(ORDER, HIGHCUT, btype="lowpass", analog=False, ftype="butter", fs=fs, output="sos")
    b_hp, a_hp = butter(ORDER, LOWCUT, btype="highpass", fs=fs)
    return (b_notch, a_notch), sos_lp, (b_hp, a_hp)

def apply_filters(df: pd.DataFrame, fs: int, filt_notch, sos_lp, filt_hp) -> pd.DataFrame:
    """low-pass -> high-pass -> notch """
    b_notch, a_notch = filt_notch
    b_hp, a_hp = filt_hp
    lp = pd.DataFrame(sosfiltfilt(sos_lp, df.values, axis=0), columns=df.columns)
    lphp = pd.DataFrame(filtfilt(b_hp, a_hp, lp.values, axis=0), columns=df.columns)
    out = pd.DataFrame(index=df.index, columns=df.columns)
    for ch in lphp.columns:
        out[ch] = filtfilt(b_notch, a_notch, lphp[ch].values)
    return out

# ---------- Features ----------
from numpy import trapezoid
def integrate_band(freq: np.ndarray, psd: np.ndarray, low: float, high: float) -> float:
    mask = (freq >= low) & (freq <= high)
    if not np.any(mask):
        return 0.0
    return float(trapezoid(psd[mask], x=freq[mask]))

def extract_features_beta(df: pd.DataFrame, fs: int, band: Tuple[float, float] = BETA_BAND) -> np.ndarray:
    beta_powers: List[float] = []
    for col in df.columns:
        psd_mt, freq_mt = psd_array_multitaper(df[col].values, fs, normalization="full", verbose=0)
        beta_powers.append(integrate_band(freq_mt, psd_mt, band[0], band[1]))
    return np.array(beta_powers, dtype=float).reshape(1, -1)

# ---------- Windowing ----------
def iter_windows(df: pd.DataFrame, win: int, hop: int) -> Iterable[pd.DataFrame]:
    n = len(df)
    if n < win:
        return
    for i in range(0, n - win + 1, hop):
        yield df.iloc[i:i+win, :]

# ---------- Loaders ----------
def load_csv(path: str) -> Tuple[pd.DataFrame, Optional[str], int]:
    df = pd.read_csv(path)
    data = df.iloc[:, 1:5].copy()
    if len(data.columns) >= 4:
        data.columns = CHANNEL_NAMES
    estado = None
    try:
        estado = os.path.basename(path).split("-")[1]
    except Exception:
        pass
    return data, estado, DEFAULT_FS

def load_xdf(path: str) -> Tuple[pd.DataFrame, Optional[str], int]:
    if pyxdf is None:
        raise ImportError("pyxdf não disponível. Instala 'pyxdf' ou usa --source csv.")
    streams, _ = pyxdf.load_xdf(path)
    eeg_stream = None
    for s in streams:
        if s.get("info", {}).get("type", [""])[0] == "EEG":
            eeg_stream = s
            break
    if eeg_stream is None:
        raise ValueError("Não encontrei stream EEG no XDF.")

    eeg = np.asarray(eeg_stream["time_series"])
    df = pd.DataFrame(eeg)
    if df.shape[1] >= 4:
        df = df.iloc[:, :4].copy()
    df.columns = CHANNEL_NAMES

    # tenta obter sample rate do XDF, senão usa DEFAULT_FS
    fs = DEFAULT_FS
    try:
        srate = eeg_stream["info"]["nominal_srate"][0]
        fs = int(float(srate)) if srate not in (None, "", "NaN") else DEFAULT_FS
    except Exception:
        fs = DEFAULT_FS

    # rótulo a partir do basename (ex: relaxed.xdf)
    estado = os.path.splitext(os.path.basename(path))[0]
    return df, estado, fs

# ---------- Plot helper ----------
def live_plot(df: pd.DataFrame, fs: int):
    init_plot()
    x = np.arange(len(df)) / fs
    _AX.cla()
    for ch in df.columns:
        _AX.plot(x, df[ch].values, label=ch)
    _AX.legend(loc="upper left")
    _AX.set_xlabel("Time (s)")
    _AX.set_ylabel("Amplitude (a.u.)")
    _FIG.tight_layout()
    _FIG.canvas.draw()
    _FIG.canvas.flush_events()
    plt.pause(0.05)

def send_feedback(ser, y_pred_num: int):
    if ser is None:
        return
    ser.write(b"1" if y_pred_num == 1 else b"0")
    time.sleep(0.1)

def list_candidate_files(root: str, source: str) -> list:
    files = []
    for f in os.listdir(root):
        if source == "csv":
            if f.endswith("1.csv") and "subject" in f:
                files.append(os.path.join(root, f))
        else:
            if f.lower().endswith(".xdf"):
                files.append(os.path.join(root, f))
    return sorted(files)

def process_file(path: str,
                 source: str,
                 clf: RandomForestClassifier,
                 do_plot: bool = False,
                 ser=None):
    
    if source == "csv":
        data, estado, fs = load_csv(path)
    else:
        data, estado, fs = load_xdf(path)

    chunk_samples = int(fs * CHUNK_SIZE_SECONDS)
    hop = chunk_samples // 2

    if len(data) < chunk_samples:
        print(f"[skip] {os.path.basename(path)} curto ({len(data)} samples < {chunk_samples}).")
        return

    filt_notch, sos_lp, filt_hp = design_filters(fs)

    for chunk in iter_windows(data, chunk_samples, hop):
        filtered = apply_filters(chunk, fs, filt_notch, sos_lp, filt_hp)

        if do_plot:
            live_plot(filtered, fs)

        X = extract_features_beta(filtered, fs, BETA_BAND)

        y_label_str = estado if isinstance(estado, str) else "neutral"
        y_list = [y_label_str]
        clf.fit(X, y_list)
        y_pred = clf.predict(X)[0]
        y_num = TRIALS_DICT.get(y_pred, 0)

        print(f"Pred: {y_num} —  estado = {estado}")
        send_feedback(ser, y_num)
        time.sleep(0.3)

def main():
    ap = argparse.ArgumentParser(description="Unified realtime EEG runner")
    ap.add_argument("--data_path", default=os.getcwd(), help="Pasta a varrer por ficheiros")
    ap.add_argument("--source", choices=["csv", "xdf"], required=True, help="csv (dataset) ou xdf (ourdata)")
    ap.add_argument("--feedback", action="store_true", help="Ativa envio série para Arduino")
    ap.add_argument("--port", default="COM4", help="Porta série (se --feedback)")
    ap.add_argument("--baud", type=int, default=9600, help="Baud rate")
    ap.add_argument("--plot", action="store_true", help="Plot em tempo real")
    ap.add_argument("--loop", action="store_true", help="Ciclar varredura da pasta")
    ap.add_argument("--once", action="store_true", help="Uma passagem e sair")
    args = ap.parse_args()

    # serial opcional
    ser = None
    if args.feedback:
        if serial is None:
            raise ImportError("pyserial não disponível. Instala 'pyserial' ou remove --feedback.")
        ser = serial.Serial(args.port, args.baud)

    # classificador
    clf = RandomForestClassifier(max_depth=None, min_samples_leaf=1, min_samples_split=2, n_estimators=100)

    try:
        if args.once:
            files = list_candidate_files(args.data_path, args.source)
            if not files:
                print(f"[info] Sem ficheiros em {args.data_path} para source={args.source}.")
            for f in files:
                process_file(f, args.source, clf, args.plot, ser)
        else:
            while True:
                files = list_candidate_files(args.data_path, args.source)
                for f in files:
                    process_file(f, args.source, clf, args.plot, ser)
                time.sleep(5)
    finally:
        if ser is not None:
            ser.close()

if __name__ == "__main__":
    main()
