"""
Funções de apoio para o experimento de confusão com 16-QAM, SPE e
embaralhamento espectral de amplitude.

A ideia deste arquivo é deixar apenas o que o main realmente usa. No arquivo
original havia várias versões repetidas da mesma função. Em Python, quando uma
função aparece mais de uma vez com o mesmo nome, a última definição sobrescreve
as anteriores. Por isso, aqui fica uma versão única e mais direta de cada etapa.
"""

import math
import random
from typing import Iterable

import numpy as np
from Crypto.Cipher import AES


# =============================================================================
# Utilidades básicas
# =============================================================================


def generate_random_bits(num_bits: int) -> list[int]:
    """Gera uma sequência aleatória de 0 e 1 com o tamanho pedido."""
    return [random.randint(0, 1) for _ in range(num_bits)]


def hamming_distance(bits_a: Iterable[int], bits_b: Iterable[int]) -> int:
    """Conta em quantas posições duas sequências de bits são diferentes."""
    return sum(int(a) ^ int(b) for a, b in zip(bits_a, bits_b))


# =============================================================================
# AES em modo CTR para gerar a sequência pseudoaleatória de fases
# =============================================================================


def _bits_to_bytes(bits: str) -> bytes:
    """
    Converte uma string de bits em bytes.

    Exemplo: "01000001" vira b"A".
    """
    if len(bits) % 8 != 0:
        bits = bits.ljust(len(bits) + (8 - len(bits) % 8), "0")

    return bytes(int(bits[i:i + 8], 2) for i in range(0, len(bits), 8))


def encrypt_aes_ctr_keystream(
    key_bits: str,
    num_output_bits: int,
    nonce_bits: str,
    counter_bits: str,
) -> str:
    """
    Usa AES-CTR para gerar bits de keystream.

    Aqui o AES funciona como um gerador determinístico de bits. Com a mesma
    chave, nonce e contador, a saída é sempre a mesma. Esses bits depois viram
    fases aplicadas no espectro do sinal.
    """
    key_bytes = _bits_to_bytes(key_bits)
    nonce_bytes = _bits_to_bytes(nonce_bits)
    counter_int = int(counter_bits, 2)

    num_bytes = math.ceil(num_output_bits / 8)

    cipher = AES.new(
        key_bytes,
        AES.MODE_CTR,
        nonce=nonce_bytes,
        initial_value=counter_int,
    )

    # Criptografar zeros em CTR é uma forma simples de obter o keystream.
    keystream_bytes = cipher.encrypt(b"\x00" * num_bytes)
    keystream_bits = "".join(format(byte, "08b") for byte in keystream_bytes)

    return keystream_bits[:num_output_bits]


def loop_aes_phase_blocos(
    key_bits: str,
    message_bits: str,
    nonce_bits: str,
    counter_bits: str,
    block_size_bits: int = 128,
    n_fases: int | None = None,
    bits_per_phase: int = 8,
) -> tuple[list[float], str]:
    """
    Gera as fases usadas na SPE.

    O processo é este:
    1. define quantos bits serão necessários;
    2. gera esses bits com AES-CTR;
    3. separa os bits em grupos, por exemplo grupos de 8 bits;
    4. transforma cada grupo em um número inteiro;
    5. transforma esse número em uma fase entre 0 e 2*pi.

    O parâmetro message_bits foi mantido porque ele ajuda a indicar o tamanho
    da sequência que precisa ser gerada.
    """
    if n_fases is None:
        n_fases = math.ceil(len(message_bits) / bits_per_phase)

    total_bits_needed = n_fases * bits_per_phase
    phases_bits: list[str] = []

    while len(phases_bits) < total_bits_needed:
        bits_to_generate = min(block_size_bits, total_bits_needed - len(phases_bits))

        bloco_bits = encrypt_aes_ctr_keystream(
            key_bits=key_bits,
            num_output_bits=bits_to_generate,
            nonce_bits=nonce_bits,
            counter_bits=counter_bits,
        )

        phases_bits.extend(bloco_bits)

        # Cada bloco usa o próximo contador do AES-CTR.
        counter_decimal = int(counter_bits, 2) + 1
        counter_bits = bin(counter_decimal)[2:].zfill(len(counter_bits))

    phases: list[float] = []
    n_phase_levels = 2 ** bits_per_phase

    for i in range(0, total_bits_needed, bits_per_phase):
        bits_k = phases_bits[i:i + bits_per_phase]
        phase_index = int("".join(bits_k), 2)
        phase = (2 * math.pi / n_phase_levels) * phase_index
        phases.append(phase)

    return phases, counter_bits


# =============================================================================
# Modulação 16-QAM e filtro RRC
# =============================================================================


def rrc_filter(beta: float, sps: int, span: int) -> np.ndarray:
    """
    Cria o filtro Root Raised Cosine usado para limitar a banda do sinal.

    beta é o roll-off. sps é o número de amostras por símbolo. span controla
    quantos símbolos entram no comprimento do filtro.
    """
    beta = float(beta)

    if not 0 <= beta <= 1:
        raise ValueError("O roll-off beta precisa estar entre 0 e 1.")

    n = span * sps
    t = np.arange(-n / 2, n / 2 + 1) / sps
    h = np.zeros_like(t, dtype=float)

    for i, ti in enumerate(t):
        if np.isclose(ti, 0.0):
            h[i] = 1.0 - beta + 4 * beta / np.pi

        elif beta != 0 and np.isclose(abs(ti), 1 / (4 * beta)):
            h[i] = (
                beta / np.sqrt(2)
                * (
                    (1 + 2 / np.pi) * np.sin(np.pi / (4 * beta))
                    + (1 - 2 / np.pi) * np.cos(np.pi / (4 * beta))
                )
            )

        else:
            numerator = (
                np.sin(np.pi * ti * (1 - beta))
                + 4 * beta * ti * np.cos(np.pi * ti * (1 + beta))
            )
            denominator = np.pi * ti * (1 - (4 * beta * ti) ** 2)
            h[i] = numerator / denominator

    # Normaliza a energia para o filtro não mudar a potência do sinal à toa.
    return h / np.sqrt(np.sum(h ** 2))


def gerar_16qam(
    bits: Iterable[int],
    sps: int,
    fs: float,
    rolloff: float = 0.35,
    span: int = 10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Gera o sinal 16-QAM no tempo.

    Passo a passo:
    1. agrupa os bits de 4 em 4;
    2. transforma cada grupo em um símbolo complexo 16-QAM;
    3. faz upsampling, colocando zeros entre os símbolos;
    4. aplica o filtro RRC;
    5. ajusta a escala para manter os níveis da constelação bem posicionados.
    """
    bits_array = np.array([int(bit) for bit in bits], dtype=int)

    if len(bits_array) % 4 != 0:
        padding = 4 - (len(bits_array) % 4)
        bits_array = np.concatenate([bits_array, np.zeros(padding, dtype=int)])

    grupos = bits_array.reshape(-1, 4)
    simbolos: list[complex] = []

    # Bits b0 b1 definem o eixo I. Bits b2 b3 definem o eixo Q.
    for b0, b1, b2, b3 in grupos:
        dec_i = 2 * b0 + b1
        dec_q = 2 * b2 + b3

        I = 2 * dec_i - 3
        Q = 2 * dec_q - 3

        simbolos.append(I + 1j * Q)

    simbolos_array = np.array(simbolos, dtype=complex)

    # Trem de impulsos: símbolo em uma amostra e zeros entre símbolos.
    sinal_16qam = np.zeros(len(simbolos_array) * sps, dtype=complex)
    sinal_16qam[::sps] = simbolos_array

    h_rrc = rrc_filter(rolloff, sps, span)
    sinal_filtrado = np.convolve(sinal_16qam, h_rrc, mode="same")

    # Compensação simples de escala após o filtro.
    samples = sinal_filtrado[::sps]
    rms_samples = np.sqrt(np.mean(np.abs(samples) ** 2))
    rms_ref = np.sqrt(np.mean(np.abs(simbolos_array) ** 2))

    if rms_samples > 0:
        sinal_filtrado = sinal_filtrado * (rms_ref / rms_samples)

    t = np.arange(len(sinal_16qam)) / fs

    return t, sinal_16qam, sinal_filtrado, simbolos_array, h_rrc


# =============================================================================
# Funções espectrais da SPE e do embaralhamento
# =============================================================================


def spectrum_shifted(signal_time: np.ndarray) -> np.ndarray:
    """Leva o sinal para a frequência e coloca o zero da frequência no centro."""
    return np.fft.fftshift(np.fft.fft(np.asarray(signal_time)))


def time_from_spectrum_shifted(spectrum: np.ndarray) -> np.ndarray:
    """Volta para o tempo a partir de um espectro centralizado."""
    return np.fft.ifft(np.fft.ifftshift(np.asarray(spectrum)))


def calculate_ncs(nsa_p: int, sps: int, rolloff: float) -> tuple[int, int, int]:
    """
    Calcula quantos bins centrais entram na SPE.

    A conta usa o tamanho do sinal, a quantidade de amostras por símbolo e o
    roll-off do filtro. O retorno é:
        ncs, índice inicial, índice final exclusivo.
    """
    ncs = math.ceil((nsa_p + 1) * (1 + rolloff) / sps)
    n1 = math.floor((nsa_p - ncs) / 2)
    n2 = n1 + ncs

    if n1 < 0 or n2 > nsa_p:
        raise ValueError(
            f"Índices SPE inválidos: nsa_p={nsa_p}, ncs={ncs}, n1={n1}, n2={n2}"
        )

    return ncs, n1, n2


def central_indices_from_ncs(n_total: int, ncs: int) -> tuple[np.ndarray, int, int]:
    """Retorna os índices centrais onde a SPE será aplicada."""
    n1 = int((n_total - ncs) // 2)
    n2 = n1 + int(ncs)
    positions = np.arange(n1, n2)
    return positions, n1, n2


def apply_phase_on_indices(
    spectrum: np.ndarray,
    positions: np.ndarray,
    phases: Iterable[float],
) -> np.ndarray:
    """Multiplica os bins escolhidos por exp(j*fase)."""
    spectrum_out = np.array(spectrum, dtype=complex, copy=True)
    phases_array = np.asarray(list(phases), dtype=float)

    if len(positions) != len(phases_array):
        raise ValueError(
            f"positions e phases precisam ter o mesmo tamanho. "
            f"Recebi {len(positions)} posições e {len(phases_array)} fases."
        )

    spectrum_out[positions] *= np.exp(1j * phases_array)
    return spectrum_out


def ranking_vector(values: Iterable[float], len_vetor: int | None = None) -> np.ndarray:
    """
    Cria uma permutação determinística a partir das fases.

    Na prática, fases menores aparecem antes. O mergesort mantém a ordem estável
    quando aparecem valores iguais.
    """
    values_array = np.asarray(list(values), dtype=float)

    if len_vetor is not None:
        values_array = values_array[:len_vetor]

    return np.argsort(values_array, kind="mergesort")


def shuffle_amplitude_on_indices(
    spectrum: np.ndarray,
    positions: np.ndarray,
    permutation: Iterable[int],
) -> np.ndarray:
    """
    Embaralha apenas as amplitudes dos bins centrais.

    A fase local de cada bin é preservada. O que muda é qual amplitude fica em
    cada posição. Isso acrescenta uma segunda camada de alteração espectral.
    """
    spectrum_out = np.array(spectrum, dtype=complex, copy=True)
    z = spectrum_out[positions]

    amplitudes = np.abs(z)
    local_phases = np.angle(z)
    permutation_array = np.asarray(list(permutation), dtype=int)[:len(amplitudes)]

    if (
        permutation_array.size != len(amplitudes)
        or np.any(permutation_array < 0)
        or np.any(permutation_array >= len(amplitudes))
    ):
        raise ValueError("A permutação do embaralhamento tem índices inválidos.")

    shuffled_amplitudes = amplitudes[permutation_array]
    spectrum_out[positions] = shuffled_amplitudes * np.exp(1j * local_phases)

    return spectrum_out


def unshuffle_amplitude_on_indices(
    spectrum: np.ndarray,
    positions: np.ndarray,
    permutation: Iterable[int],
) -> np.ndarray:
    """
    Desfaz o embaralhamento de amplitude.

    Esta função não é usada no experimento de confusão, mas fica pronta para
    testes de reversibilidade. Ela aplica a permutação inversa nas amplitudes.
    """
    spectrum_out = np.array(spectrum, dtype=complex, copy=True)
    z = spectrum_out[positions]

    amplitudes = np.abs(z)
    local_phases = np.angle(z)
    permutation_array = np.asarray(list(permutation), dtype=int)[:len(amplitudes)]

    inverse_permutation = np.empty_like(permutation_array)
    inverse_permutation[permutation_array] = np.arange(len(permutation_array))

    restored_amplitudes = amplitudes[inverse_permutation]
    spectrum_out[positions] = restored_amplitudes * np.exp(1j * local_phases)

    return spectrum_out


# =============================================================================
# Demodulação 16-QAM e apoio para contadores
# =============================================================================


def demodulate_16qam_gray_corrigido(
    signal: np.ndarray,
    sps: int,
    k: int = 0,
    normalize: bool = False,
    n_bits: int | None = None,
) -> tuple[list[int], int]:
    """
    Detector ML para 16-QAM.

    O nome foi mantido para não quebrar chamadas antigas do main. A função
    monta os 16 pontos possíveis da constelação, compara cada amostra recebida
    com todos esses pontos e escolhe o ponto mais próximo.
    """
    pontos: list[complex] = []
    labels: list[list[int]] = []

    for sym in range(16):
        i_idx = sym % 4
        q_idx = sym // 4

        I = 2 * i_idx - 3
        Q = 2 * q_idx - 3

        pontos.append(I + 1j * Q)
        labels.append([
            (sym >> 3) & 1,
            (sym >> 2) & 1,
            (sym >> 1) & 1,
            sym & 1,
        ])

    pontos_array = np.array(pontos, dtype=complex)
    labels_array = np.array(labels, dtype=int)

    symbols_rx = signal[k::sps]

    if len(symbols_rx) == 0:
        return [], k

    if normalize:
        rms_rx = np.sqrt(np.mean(np.abs(symbols_rx) ** 2))
        rms_ref = np.sqrt(np.mean(np.abs(pontos_array) ** 2))

        if rms_rx > 0:
            symbols_rx = symbols_rx * (rms_ref / rms_rx)

    bits_out: list[int] = []

    for z in symbols_rx:
        idx = np.argmin(np.abs(z - pontos_array) ** 2)
        bits_out.extend(labels_array[idx].tolist())

    if n_bits is not None:
        bits_out = bits_out[:n_bits]

    return bits_out, k


def counters_per_word(n_elements: int, bits_per_phase: int) -> int:
    """Calcula quantos blocos de contador AES uma palavra consome."""
    return math.ceil((n_elements * bits_per_phase) / 128)
