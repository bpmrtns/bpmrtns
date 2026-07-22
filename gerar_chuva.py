#!/usr/bin/env python3
"""Gera um audio de chuva continua (sem goteiras) e salva em MP3.

A chuva e sintetizada como ruido branco filtrado: uma banda media suave
(o "chiado" da chuva caindo) mais um leve corpo grave (rumor distante),
com variacoes de intensidade muito lentas para soar natural. Nao ha
nenhum evento impulsivo — nada de pingos ou goteiras.

Uso: python3 gerar_chuva.py [saida.mp3] [duracao_segundos]
"""

import sys

import lameenc
import numpy as np
from scipy import signal

SAMPLE_RATE = 44100
CHUNK_SECONDS = 30
BITRATE_KBPS = 128
FADE_SECONDS = 4.0
TARGET_RMS_DB = -20.0


def main() -> None:
    out_path = sys.argv[1] if len(sys.argv) > 1 else "chuva_1_hora.mp3"
    duration = float(sys.argv[2]) if len(sys.argv) > 2 else 3600.0

    rng = np.random.default_rng(20260722)

    # Filtros (aplicados em cascata, com estado preservado entre blocos
    # para o audio ser perfeitamente continuo).
    # Banda principal do chiado: 300 Hz – 6 kHz, com topo dobrado para suavizar.
    sos_hiss = signal.butter(2, [300, 6000], btype="bandpass", fs=SAMPLE_RATE, output="sos")
    sos_soft = signal.butter(2, 7500, btype="lowpass", fs=SAMPLE_RATE, output="sos")
    # Corpo grave sutil (rumor da chuva ao longe): passa-baixa em 400 Hz.
    sos_body = signal.butter(2, 400, btype="lowpass", fs=SAMPLE_RATE, output="sos")

    n_channels = 2
    zi_hiss = [signal.sosfilt_zi(sos_hiss) * 0 for _ in range(n_channels)]
    zi_soft = [signal.sosfilt_zi(sos_soft) * 0 for _ in range(n_channels)]
    zi_body = [signal.sosfilt_zi(sos_body) * 0 for _ in range(n_channels)]

    # Envelope de intensidade: caminhada aleatoria bem lenta (ondas de
    # ~1 minuto), suavizada, com profundidade de ~±1.5 dB. Gerado inteiro
    # de uma vez numa taxa de controle baixa e interpolado por bloco.
    ctrl_rate = 4.0
    n_ctrl = int(duration * ctrl_rate) + 2
    walk = np.cumsum(rng.standard_normal(n_ctrl))
    b_env, a_env = signal.butter(2, 1.0 / 60.0, fs=ctrl_rate)
    walk = signal.filtfilt(b_env, a_env, walk)
    walk = (walk - walk.mean()) / (walk.std() + 1e-9)
    env_ctrl_db = walk * 1.5
    env_ctrl = 10.0 ** (env_ctrl_db / 20.0)

    target_rms = 10.0 ** (TARGET_RMS_DB / 20.0)
    gain = None

    encoder = lameenc.Encoder()
    encoder.set_bit_rate(BITRATE_KBPS)
    encoder.set_in_sample_rate(SAMPLE_RATE)
    encoder.set_channels(n_channels)
    encoder.set_quality(2)

    total_samples = int(duration * SAMPLE_RATE)
    chunk_samples = CHUNK_SECONDS * SAMPLE_RATE
    fade_samples = int(FADE_SECONDS * SAMPLE_RATE)
    written = 0

    with open(out_path, "wb") as f:
        while written < total_samples:
            n = min(chunk_samples, total_samples - written)
            chans = []
            for ch in range(n_channels):
                white = rng.standard_normal(n).astype(np.float64)
                hiss, zi_hiss[ch] = signal.sosfilt(sos_hiss, white, zi=zi_hiss[ch])
                hiss, zi_soft[ch] = signal.sosfilt(sos_soft, hiss, zi=zi_soft[ch])
                low = rng.standard_normal(n).astype(np.float64)
                body, zi_body[ch] = signal.sosfilt(sos_body, low, zi=zi_body[ch])
                chans.append(hiss + 0.35 * body)

            block = np.stack(chans, axis=1)

            if gain is None:
                gain = target_rms / (np.sqrt(np.mean(block**2)) + 1e-12)
            block *= gain

            t = (written + np.arange(n)) / SAMPLE_RATE
            env = np.interp(t * ctrl_rate, np.arange(n_ctrl), env_ctrl)
            block *= env[:, None]

            # Fade de entrada e saida para comecar e terminar suave.
            idx = written + np.arange(n)
            fade = np.minimum(idx / fade_samples, (total_samples - 1 - idx) / fade_samples)
            block *= np.clip(fade, 0.0, 1.0)[:, None]

            block = np.tanh(block * 1.2) / 1.2  # limitador suave, evita picos
            pcm = np.clip(block * 32767.0, -32768, 32767).astype(np.int16)
            f.write(encoder.encode(pcm.tobytes()))
            written += n
            print(f"\r{written / SAMPLE_RATE / 60:6.1f} min gerados", end="", flush=True)

        f.write(encoder.flush())
    print(f"\nPronto: {out_path}")


if __name__ == "__main__":
    main()
