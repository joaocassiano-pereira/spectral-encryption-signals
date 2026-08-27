# functions_espectros_limpo_comentado_RCF.py
# =============================================================================
# Funções de apoio para geração, proteção e recuperação de sinais 16-QAM.
#
# Este arquivo concentra as partes usadas pelo main:
#   - geração de bits;
#   - geração de fases com AES-CTR;
#   - modulação 16-QAM;
#   - filtro RCF;
#   - FFT/IFFT centralizadas;
#   - mudança de fase no espectro;
#   - embaralhamento e desembaralhamento de amplitude.
#
# Os comentários foram escritos para explicar o fluxo do método de forma direta.
# =============================================================================

import math
import random

import numpy as np
from Crypto.Cipher import AES

from Data_L import input_data as inp


# =============================================================================
# Geração de dados e conversões simples
# =============================================================================

def generate_random_bits(num_bits):
    """Gera uma sequência aleatória de bits com tamanho definido."""
    return [random.randint(0, 1) for _ in range(num_bits)]


def int_to_64bit_string(value):
    """Converte um número inteiro para uma string binária com 64 bits."""
    return bin(int(value))[2:].zfill(64)


def bits_to_bytes(bit_string):
    """Converte uma string de bits em bytes."""
    if len(bit_string) % 8 != 0:
        bit_string = bit_string.ljust(len(bit_string) + (8 - len(bit_string) % 8), "0")

    return bytes(
        int(bit_string[i:i + 8], 2)
        for i in range(0, len(bit_string), 8)
    )


# =============================================================================
# AES-CTR usado para gerar a sequência de fase
# =============================================================================

def aes_ctr_keystream_bits(key_bits, nonce_bits, counter_bits, num_bits):
    """
    Gera uma sequência pseudoaleatória de bits usando AES em modo CTR.

    Aqui o AES não é usado para guardar uma mensagem textual. Ele serve para
    gerar uma sequência controlada pela chave, pelo nonce e pelo contador.
    Essa sequência depois vira a chave de fase do método espectral.
    """
    key_bytes = bits_to_bytes(key_bits)
    nonce_bytes = bits_to_bytes(nonce_bits)

    counter_int = int(counter_bits, 2)
    num_bytes = math.ceil(num_bits / 8)

    cipher = AES.new(
        key_bytes,
        AES.MODE_CTR,
        nonce=nonce_bytes,
        initial_value=counter_int,
    )

    keystream = cipher.encrypt(b"\x00" * num_bytes)
    keystream_bits = "".join(format(byte, "08b") for byte in keystream)

    return keystream_bits[:num_bits]


def aes_phase_levels_from_ctr(
    key_bits,
    nonce_bits,
    counter_bits,
    n_phases,
    bits_per_phase=8,
):
    """
    Gera fases no intervalo [0, 2*pi).

    Cada fase usa bits_per_phase bits. Com 8 bits por fase, por exemplo,
    existem 256 níveis possíveis de fase.
    """
    total_bits = n_phases * bits_per_phase

    phase_bits = aes_ctr_keystream_bits(
        key_bits=key_bits,
        nonce_bits=nonce_bits,
        counter_bits=counter_bits,
        num_bits=total_bits,
    )

    n_levels = 2 ** bits_per_phase
    phases = []

    for i in range(0, total_bits, bits_per_phase):
        group = phase_bits[i:i + bits_per_phase]
        phase_index = int(group, 2)
        phase = (2 * math.pi / n_levels) * phase_index
        phases.append(phase)

    # Atualiza o contador pelo número de blocos AES consumidos.
    blocks_used = math.ceil(total_bits / 128)
    updated_counter = int(counter_bits, 2) + blocks_used
    updated_counter_bits = bin(updated_counter)[2:].zfill(len(counter_bits))

    return phases, updated_counter_bits


# =============================================================================
# Filtro RCF
# =============================================================================

def rcf_taps(alpha, sps, span):
    """
    Gera os coeficientes do filtro RCF.

    alpha controla o roll-off, sps define quantas amostras existem por símbolo
    e span define a duração do filtro em símbolos.
    """
    alpha = float(max(alpha, 1e-12))

    n_samples = span * sps
    n = np.arange(-n_samples // 2, n_samples // 2 + 1)
    x = n / float(sps)

    sinc_part = np.sinc(x)
    cosine_part = np.cos(np.pi * alpha * x)
    denominator = 1.0 - (2.0 * alpha * x) ** 2

    h = np.empty_like(x, dtype=float)
    h[:] = sinc_part * cosine_part / denominator

    # Corrige o ponto central.
    h[n == 0] = 1.0

    # Corrige os pontos onde a fórmula teria divisão por zero.
    singular_point = 1.0 / (2.0 * alpha)
    singular_mask = np.isclose(np.abs(x), singular_point, atol=1e-9)

    if np.any(singular_mask):
        h[singular_mask] = (alpha / 2.0) * np.sin(np.pi / (2.0 * alpha))

    # Normaliza a energia do filtro.
    h = h / np.sqrt(np.sum(h ** 2))

    return h


# Mantém esse nome como atalho, caso algum arquivo antigo ainda chame rc_taps.
rc_taps = rcf_taps


# =============================================================================
# Modulação 16-QAM
# =============================================================================

def modulate_16qam_natural_for_spectra(
    bits,
    symbol_rate,
    sps,
    alpha=0.02,
    span=8,
    gain=1.0,
):
    """
    Gera o sinal 16-QAM antes e depois do filtro RCF.

    A cada grupo de 4 bits:
        - os 2 primeiros bits definem a componente I;
        - os 2 últimos bits definem a componente Q.

    Retorna:
        signal_before_filter: referência temporal com pulsos retangulares, usada
            para visualizar a PSD antes do RCF (envoltória sinc²);
        signal_after_filter: sinal obtido pelo fluxo digital convencional
            símbolos -> upsampling por inserção de zeros -> RCF;
        time_vector: vetor de tempo.

    Observação importante:
        O sinal retangular retornado como signal_before_filter é usado somente
        para a Figura 5/diagnóstico espectral. O RCF continua sendo excitado
        pelo trem de impulsos sobreamostrado, evitando a cascata indevida de
        um pulso retangular com o Raised Cosine.
    """
    bits = np.asarray(bits, dtype=int).reshape(-1)

    # A 16-QAM usa 4 bits por símbolo.
    # Se sobrar bit incompleto, ele é descartado para fechar os grupos.
    if len(bits) % 4 != 0:
        bits = bits[:len(bits) - (len(bits) % 4)]

    bit_groups = bits.reshape(-1, 4)
    symbols = []

    # Mapeamento Gray 2 bits -> nível PAM-4:
    # 00 -> -3, 01 -> -1, 11 -> +1, 10 -> +3.
    gray_level = {(0, 0): -3, (0, 1): -1, (1, 1): 1, (1, 0): 3}

    for b0, b1, b2, b3 in bit_groups:
        i_level = gray_level[(int(b0), int(b1))]
        q_level = gray_level[(int(b2), int(b3))]
        symbols.append(i_level + 1j * q_level)

    symbols = np.asarray(symbols, dtype=complex)

    # -------------------------------------------------------------------------
    # Sinal de referência ANTES do RCF para a Figura 5:
    # cada símbolo é mantido durante Ts (pulso retangular / zero-order hold).
    # A magnitude do espectro desse pulso possui envoltória proporcional a
    # sinc²(f*Ts) na PSD, com zeros em ±Rs, ±2Rs, ...
    # -------------------------------------------------------------------------
    signal_before_filter = np.repeat(symbols, sps).astype(complex)

    # -------------------------------------------------------------------------
    # Fluxo efetivo do pulse shaping:
    # símbolos -> upsampling por inserção de zeros -> filtro Raised Cosine.
    # Não usamos o sinal retangular acima como entrada do RCF, pois isso
    # acrescentaria uma resposta sinc extra ao sistema.
    # -------------------------------------------------------------------------
    impulse_train = np.zeros(len(symbols) * sps, dtype=complex)
    impulse_train[::sps] = symbols

    h_rcf = rcf_taps(float(max(alpha, 1e-6)), sps, span)
    signal_after_filter = np.convolve(impulse_train, h_rcf, mode="same") * gain

    # Ajuste simples de escala para manter os pontos próximos da constelação.
    samples = signal_after_filter[::sps]
    rms_samples = np.sqrt(np.mean(np.abs(samples) ** 2))
    rms_reference = np.sqrt(np.mean(np.abs(symbols) ** 2))

    if rms_samples > 0:
        signal_after_filter = signal_after_filter * (rms_reference / rms_samples)

    time_vector = np.arange(signal_after_filter.size) / (symbol_rate * sps)

    return signal_before_filter, signal_after_filter, time_vector


# =============================================================================
# Operações no domínio da frequência
# =============================================================================

def spectrum_shifted(signal_time):
    """Calcula a FFT centralizada mantendo o tamanho natural do sinal."""
    return np.fft.fftshift(np.fft.fft(np.asarray(signal_time)))


def time_from_spectrum_shifted(spectrum_shifted_in):
    """Volta do espectro centralizado para o domínio do tempo com preservação de fase."""
    return np.fft.ifft(np.fft.ifftshift(np.asarray(spectrum_shifted_in)))

def calculate_center_bin_count(num_samples, sps, rolloff):
    """
    Calcula quantos bins centrais serão usados na proteção espectral.

    A quantidade cresce com o roll-off e diminui quando há mais amostras
    por símbolo.
    """
    ncs = math.ceil((num_samples + 1) * (1 + rolloff) / sps)

    n1 = math.floor((num_samples - ncs) / 2)
    n2 = n1 + ncs

    if n1 < 0 or n2 > num_samples:
        raise ValueError(
            f"Faixa central inválida: num_samples={num_samples}, "
            f"ncs={ncs}, n1={n1}, n2={n2}."
        )

    return ncs, n1, n2


# Mantém compatibilidade com nomes usados em versões anteriores.
matlab_like_ncs = calculate_center_bin_count


def central_indices_from_ncs(nfft, ncs):
    """Retorna os índices da faixa central do espectro."""
    n1 = math.floor((nfft - ncs) / 2)
    n2 = n1 + ncs
    return np.arange(n1, n2), n1, n2


def apply_phase_on_indices(spectrum_shifted_in, positions, key_phases):
    """
    Aplica mudança de fase apenas nos índices escolhidos.

    A amplitude de cada bin é mantida. O que muda é o ângulo daquele ponto
    no espectro.
    """
    spectrum_out = np.array(spectrum_shifted_in, dtype=complex, copy=True)

    positions = np.asarray(positions, dtype=int)
    key_phases = np.asarray(key_phases, dtype=float)[:len(positions)]

    spectrum_out[positions] = spectrum_out[positions] * np.exp(1j * key_phases)

    return spectrum_out


def remove_phase_on_indices(spectrum_shifted_in, positions, key_phases):
    """
    Remove a fase aplicada anteriormente.

    É a operação inversa da mudança de fase: usa exp(-j*fase).
    """
    spectrum_out = np.array(spectrum_shifted_in, dtype=complex, copy=True)

    positions = np.asarray(positions, dtype=int)
    key_phases = np.asarray(key_phases, dtype=float)[:len(positions)]

    spectrum_out[positions] = spectrum_out[positions] * np.exp(-1j * key_phases)

    return spectrum_out


# =============================================================================
# Embaralhamento e desembaralhamento de amplitude
# =============================================================================

def ranking_vector(values, vector_length=None):
    """
    Cria uma ordem determinística a partir dos valores recebidos.

    A saída é uma permutação 1-based, ou seja, começa em 1. Isso foi mantido
    porque deixa o embaralhamento fácil de ler nos laços abaixo.
    """
    values = np.asarray(values)

    if vector_length is None:
        vector_length = len(values)

    vector_length = int(vector_length)
    order = sorted(range(vector_length), key=lambda idx: (values[idx], idx))

    ranking = [0] * vector_length
    for rank, original_index in enumerate(order, start=1):
        ranking[original_index] = rank

    return ranking


def validate_permutation(permutation, expected_size):
    """Garante que a permutação tenha o tamanho e os limites corretos."""
    permutation = np.asarray(permutation, dtype=int)[:expected_size]

    is_valid = (
        permutation.size == expected_size
        and np.all(permutation >= 1)
        and np.all(permutation <= expected_size)
        and len(np.unique(permutation)) == expected_size
    )

    if not is_valid:
        raise ValueError("Vetor de embaralhamento inválido.")

    return permutation


def shuffle_amplitude_on_indices(spectrum_shifted_in, positions, permutation):
    """
    Embaralha as amplitudes dos bins centrais e preserva as fases locais.

    A fase que já existe em cada posição continua naquela posição.
    O que muda de lugar são as amplitudes.
    """
    spectrum_out = np.array(spectrum_shifted_in, dtype=complex, copy=True)
    positions = np.asarray(positions, dtype=int)

    center_values = spectrum_out[positions]

    amplitudes = np.abs(center_values)
    phases = np.angle(center_values)

    permutation = validate_permutation(permutation, len(amplitudes))

    shuffled_amplitudes = np.zeros_like(amplitudes)

    for source_index in range(len(amplitudes)):
        target_index = permutation[source_index] - 1
        shuffled_amplitudes[target_index] = amplitudes[source_index]

    spectrum_out[positions] = shuffled_amplitudes * np.exp(1j * phases)

    return spectrum_out


def deshuffle_amplitude_on_indices(spectrum_shifted_in, positions, permutation):
    """
    Desfaz o embaralhamento de amplitude.

    Usa a mesma permutação da etapa de embaralhamento, mas no sentido inverso.
    """
    spectrum_out = np.array(spectrum_shifted_in, dtype=complex, copy=True)
    positions = np.asarray(positions, dtype=int)

    center_values = spectrum_out[positions]

    shuffled_amplitudes = np.abs(center_values)
    phases = np.angle(center_values)

    permutation = validate_permutation(permutation, len(shuffled_amplitudes))

    restored_amplitudes = np.zeros_like(shuffled_amplitudes)

    for source_index in range(len(shuffled_amplitudes)):
        target_index = permutation[source_index] - 1
        restored_amplitudes[source_index] = shuffled_amplitudes[target_index]

    spectrum_out[positions] = restored_amplitudes * np.exp(1j * phases)

    return spectrum_out


# =============================================================================
# Funções antigas mantidas só por compatibilidade
# =============================================================================

def pulse(p_type):
    """Gera um pulso quadrado simples."""
    if p_type == "square":
        time = np.linspace(0, inp.Tsymbol, inp.spsymbol)
        pulse_values = np.ones(inp.spsymbol)
        return time, pulse_values

    raise ValueError(f"Tipo de pulso não suportado: {p_type}")


def inband_indices(symbol_rate, sps, alpha, nfft, fftshifted=False):
    """Retorna os índices que ficam dentro da banda ocupada pelo sinal."""
    sampling_frequency = symbol_rate * sps

    frequencies = np.fft.fftfreq(nfft, d=1.0 / sampling_frequency)

    if fftshifted:
        frequencies = np.fft.fftshift(frequencies)

    total_frequency = 0.5 * (1.0 + alpha) * symbol_rate
    inband_mask = np.abs(frequencies) <= total_frequency
    indices = np.where(inband_mask)[0]

    return indices, frequencies
