#!/usr/bin/env python3
"""Gera um audio de chuva natural e continua, salvo em MP3.

A chuva soa natural porque e feita da textura granular real da chuva:
milhares de micro-impactos de gotas por segundo (um processo de Poisson
denso), cada um sendo um curtissimo estouro de ruido com decaimento. Essa
granularidade densa cria o "crepitar" caracteristico — nao ha nenhum pingo
isolado (nada de goteira). Por cima vem um "lençol" de ruido rosa filtrado
(o murmurinho das gotas distantes) e um corpo grave suave (rumor ao longe).

Camadas:
  1. Sizzle: gotas densas (impulsos aleatorios) convoluidas com um kernel
     curto de ruido bandpass com decaimento — o crepitar da chuva.
  2. Wash: ruido rosa filtrado, o lençol de fundo continuo.
  3. Body: passa-baixa suave, o rumor grave distante.

Uso: python3 gerar_chuva.py [saida.mp3] [duracao_segundos]
"""

import sys

import lameenc
import numpy as np
from scipy import signal

SAMPLE_RATE = 44100
CHUNK_SECONDS = 20
BITRATE_KBPS = 160
FADE_SECONDS = 4.0
TARGET_RMS_DB = -20.0

# Densidade de gotas por segundo (por canal). Alta o bastante para que
# gotas individuais se fundam num crepitar continuo, sem virar goteira.
DROPS_PER_SEC = 13000


def pink_noise(n, rng, state):
    """Ruido rosa via filtro; mantem estado entre blocos para continuidade."""
    white = rng.standard_normal(n)
    b = [0.049922035, -0.095993537, 0.050612699, -0.004408786]
    a = [1, -2.494956002, 2.017265875, -0.522189400]
    out, state = signal.lfilter(b, a, white, zi=state)
    return out, state


def make_drop_kernel(rng):
    """Kernel curto de uma gota: estouro de ruido bandpass com decaimento.

    Usa ruido (nao senoide) para evitar um 'ping' tonal artificial. Uma
    pequena colecao de kernels com frequencias/durações variadas da a
    variedade natural das gotas.
    """
    kernels = []
    # Gotas graves e arredondadas (nada de agudos sibilantes) -> som quente.
    for fc, dur_ms in [(400, 12), (650, 10), (950, 8), (1400, 7)]:
        n = int(SAMPLE_RATE * dur_ms / 1000)
        burst = rng.standard_normal(n)
        sos = signal.butter(2, [fc * 0.6, fc * 1.6], btype="bandpass",
                            fs=SAMPLE_RATE, output="sos")
        burst = signal.sosfilt(sos, burst)
        env = np.exp(-np.linspace(0, 4, n))  # decaimento suave, gota arredondada
        k = burst * env
        k /= np.sqrt(np.sum(k**2)) + 1e-12
        kernels.append(k)
    return kernels


def main():
    out_path = sys.argv[1] if len(sys.argv) > 1 else "chuva_1_hora.mp3"
    duration = float(sys.argv[2]) if len(sys.argv) > 2 else 3600.0

    rng = np.random.default_rng(20260723)
    n_channels = 2

    drop_kernels = make_drop_kernel(rng)
    max_klen = max(len(k) for k in drop_kernels)

    # Estados de filtro por canal (para continuidade perfeita entre blocos).
    pink_state = [signal.lfilter_zi([0.049922035, -0.095993537, 0.050612699,
                  -0.004408786], [1, -2.494956002, 2.017265875, -0.522189400]) * 0
                  for _ in range(n_channels)]
    sos_wash = signal.butter(2, [150, 1600], btype="bandpass",
                             fs=SAMPLE_RATE, output="sos")
    zi_wash = [signal.sosfilt_zi(sos_wash) * 0 for _ in range(n_channels)]
    sos_body = signal.butter(2, 250, btype="lowpass", fs=SAMPLE_RATE, output="sos")
    zi_body = [signal.sosfilt_zi(sos_body) * 0 for _ in range(n_channels)]
    # Passa-baixa forte (ordem 4) corta os agudos que soam como chiado.
    sos_tame = signal.butter(4, 2800, btype="lowpass", fs=SAMPLE_RATE, output="sos")
    zi_tame = [signal.sosfilt_zi(sos_tame) * 0 for _ in range(n_channels)]
    # Cauda da convolucao das gotas que transborda para o proximo bloco.
    drop_tail = [np.zeros(max_klen) for _ in range(n_channels)]

    # Envelope lento de intensidade (ondas de chuva, ~±2 dB).
    ctrl_rate = 4.0
    n_ctrl = int(duration * ctrl_rate) + 2
    walk = np.cumsum(rng.standard_normal(n_ctrl))
    b_env, a_env = signal.butter(2, 1.0 / 45.0, fs=ctrl_rate)
    walk = signal.filtfilt(b_env, a_env, walk)
    walk = (walk - walk.mean()) / (walk.std() + 1e-9)
    env_ctrl = 10.0 ** ((walk * 2.0) / 20.0)

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
                # 1) Sizzle: trem de impulsos de gotas -> convolucao.
                n_drops = rng.poisson(DROPS_PER_SEC * n / SAMPLE_RATE)
                pos = rng.integers(0, n, n_drops)
                amp = rng.gamma(2.0, 0.5, n_drops)  # amplitudes variadas
                impulses = np.zeros(n + max_klen)
                np.add.at(impulses, pos, amp)
                # Distribui as gotas entre os kernels disponiveis.
                sizzle = np.zeros(n + max_klen)
                which = rng.integers(0, len(drop_kernels), n_drops)
                for ki, k in enumerate(drop_kernels):
                    sel = pos[which == ki]
                    if len(sel) == 0:
                        continue
                    imp = np.zeros(n + max_klen)
                    np.add.at(imp, sel, amp[which == ki])
                    sizzle += signal.fftconvolve(imp, k)[:n + max_klen]
                sizzle[:max_klen] += drop_tail[ch]
                drop_tail[ch] = sizzle[n:n + max_klen].copy()
                sizzle = sizzle[:n]

                # 2) Wash: lençol de ruido rosa filtrado.
                pink, pink_state[ch] = pink_noise(n, rng, pink_state[ch])
                wash, zi_wash[ch] = signal.sosfilt(sos_wash, pink, zi=zi_wash[ch])

                # 3) Body: rumor grave distante.
                low = rng.standard_normal(n)
                body, zi_body[ch] = signal.sosfilt(sos_body, low, zi=zi_body[ch])

                mix = 0.7 * sizzle + 1.0 * wash + 0.7 * body
                mix, zi_tame[ch] = signal.sosfilt(sos_tame, mix, zi=zi_tame[ch])
                chans.append(mix)

            block = np.stack(chans, axis=1)

            if gain is None:
                gain = target_rms / (np.sqrt(np.mean(block**2)) + 1e-12)
            block *= gain

            t = (written + np.arange(n)) / SAMPLE_RATE
            env = np.interp(t * ctrl_rate, np.arange(n_ctrl), env_ctrl)
            block *= env[:, None]

            idx = written + np.arange(n)
            fade = np.minimum(idx / fade_samples,
                              (total_samples - 1 - idx) / fade_samples)
            block *= np.clip(fade, 0.0, 1.0)[:, None]

            block = np.tanh(block * 1.1) / 1.1
            pcm = np.clip(block * 32767.0, -32768, 32767).astype(np.int16)
            f.write(encoder.encode(pcm.tobytes()))
            written += n
            print(f"\r{written / SAMPLE_RATE / 60:6.1f} min gerados", end="", flush=True)

        f.write(encoder.flush())
    print(f"\nPronto: {out_path}")


if __name__ == "__main__":
    main()
