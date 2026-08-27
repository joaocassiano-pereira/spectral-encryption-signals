"""
Experimento de difusão para 16-QAM com SPE e embaralhamento espectral.

Neste experimento a chave fica fixa dentro de cada rodada. O que muda é a
mensagem: a cada teste, um único bit da mensagem original é invertido. Depois
disso, o código mede o quanto a saída muda. Essa medida é a distância de
Hamming entre os bits obtidos com a mensagem original e os bits obtidos com a
mensagem alterada.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from Data_L import input_data as inp
from Func_L import functions as fun


# =============================================================================
# Configuração do experimento
# =============================================================================

NUM_BITS = 256
SPS = 4
ROLLOFF = 0.02
BITS_PER_PHASE = 8
N_WORDS = 400

OUTPUT_DIR = Path("data_hist_difusao_mensagem_com_embaralhamento")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# Chave AES original do experimento.
# São 256 bits, então ela é compatível com AES-256.
KEY_ORIGINAL = [
    0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 1, 1,
    0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1,
    0, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0,
    1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0,
    0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1,
    1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1,
    1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 0,
    0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0,
    1, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1,
    1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1,
    0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1,
    0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1,
    0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0,
    0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0,
    1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1,
]


# =============================================================================
# Funções pequenas para deixar o main mais legível
# =============================================================================


def int_to_64bit_string(value: int) -> str:
    """Converte um inteiro para uma string binária com 64 bits."""
    return bin(int(value))[2:].zfill(64)


def flip_bit(bits: list[int], index: int) -> list[int]:
    """Copia uma sequência de bits e inverte apenas a posição escolhida."""
    bits_flipped = bits[:]
    bits_flipped[index] = 1 - bits_flipped[index]
    return bits_flipped


def bits_to_string(bits: list[int]) -> str:
    """Transforma uma lista de 0 e 1 em uma string binária."""
    return "".join(map(str, bits))


def processa_palavra_spe(
    mensagem_bits: list[int],
    nonce_bits: str,
    counter_start_decimal: int,
    key_bits: str,
    n_fases: int | None = None,
) -> tuple[list[int], dict]:
    """
    Processa uma mensagem pelo fluxo completo do experimento.

    O caminho é este:
    1. gera o sinal 16-QAM a partir dos bits;
    2. aplica o filtro RCF;
    3. leva o sinal para o domínio da frequência;
    4. escolhe os bins centrais do espectro;
    5. gera fases com AES-CTR;
    6. aplica essas fases nos bins centrais;
    7. embaralha as amplitudes nesses mesmos bins;
    8. volta para o domínio do tempo;
    9. demodula e recupera os bits observados na saída.
    """
    # Primeiro, a mensagem vira um sinal 16-QAM filtrado.
    t, sinal_sem_filtro, sinal_filtrado, simbolos, h_rcf = fun.gerar_16qam(
        mensagem_bits,
        SPS,
        inp.Rsymbol,
        ROLLOFF,
    )

    # Ncs define quantos bins centrais do espectro serão alterados.
    ncs, _, _ = fun.calculate_ncs(
        nsa_p=len(sinal_filtrado),
        sps=SPS,
        rolloff=ROLLOFF,
    )

    if n_fases is None:
        n_fases = ncs

    # Esta sequência de zeros não é a mensagem transmitida. Ela só define o
    # tamanho do keystream necessário para gerar todas as fases.
    message_bits_for_aes = "0" * max(128, n_fases * BITS_PER_PHASE)
    counter_bits = int_to_64bit_string(counter_start_decimal)

    key_phases_aes, counter_final = fun.loop_aes_phase_blocos(
        key_bits=key_bits,
        message_bits=message_bits_for_aes,
        nonce_bits=nonce_bits,
        counter_bits=counter_bits,
        block_size_bits=128,
        n_fases=n_fases,
        bits_per_phase=BITS_PER_PHASE,
    )

    # O sinal filtrado vai para o espectro centralizado.
    spectrum_original = fun.spectrum_shifted(sinal_filtrado)

    positions_h, n1_spe, n2_spe = fun.central_indices_from_ncs(
        len(sinal_filtrado),
        ncs,
    )

    positions_h = np.asarray(positions_h, dtype=int)

    # A quantidade de fases precisa bater com a quantidade real de bins usados.
    n_bins_spe = len(positions_h)
    key_phases_bins = np.asarray(key_phases_aes[:n_bins_spe])

    # Primeira camada: altera a fase dos bins centrais.
    spectrum_after_phase = fun.apply_phase_on_indices(
        spectrum=spectrum_original,
        positions=positions_h,
        phases=key_phases_bins,
    )

    # Segunda camada: embaralha as amplitudes usando uma ordem derivada das fases.
    permutation = fun.ranking_vector(key_phases_bins, n_bins_spe)

    spectrum_after_shuffle = fun.shuffle_amplitude_on_indices(
        spectrum=spectrum_after_phase,
        positions=positions_h,
        permutation=permutation,
    )

    # Depois das duas alterações espectrais, o sinal volta para o tempo.
    signal_crypto = fun.time_from_spectrum_shifted(spectrum_after_shuffle)

    # Recupera os bits por decisão ML em 16-QAM.
    bits_recebidos, k = fun.demodulate_16qam_gray_corrigido(
        signal_crypto,
        SPS,
        k=0,
        normalize=False,
        n_bits=NUM_BITS,
    )

    debug = {
        "t": t,
        "sinal_sem_filtro": sinal_sem_filtro,
        "sinal_filtrado": sinal_filtrado,
        "simbolos": simbolos,
        "h_rcf": h_rcf,
        "espectro_antes_spe": spectrum_original,
        "espectro_apos_fase": spectrum_after_phase,
        "espectro_apos_embaralhamento": spectrum_after_shuffle,
        "sinal_apos_fase_e_embaralhamento": signal_crypto,
        "key_phases_aes": key_phases_aes,
        "key_phases_bins": key_phases_bins,
        "permutation": permutation,
        "ncs": ncs,
        "n_bins_spe": n_bins_spe,
        "n1": n1_spe,
        "n2": n2_spe,
        "counter_final": counter_final,
        "k_demod": k,
    }

    return bits_recebidos, debug


# =============================================================================
# Loop principal de difusão
# =============================================================================


def roda_experimento_difusao() -> None:
    """Executa todas as rodadas e salva uma planilha por palavra testada."""
    key_bits_original = bits_to_string(KEY_ORIGINAL)

    for loop_hist in range(N_WORDS):
        # A chave fica fixa dentro da rodada.
        mensagem_original = fun.generate_random_bits(NUM_BITS)

        # O nonce muda a cada palavra para evitar repetir exatamente o mesmo keystream.
        # Uso loop_hist + 1 para não começar com nonce zero.
        nonce_bits = int_to_64bit_string(loop_hist + 1)

        # Primeiro processa a mensagem original com a chave fixa.
        # Esse resultado vira a referência da rodada.
        bits_ref, debug_ref = processa_palavra_spe(
            mensagem_bits=mensagem_original,
            nonce_bits=nonce_bits,
            counter_start_decimal=0,
            key_bits=key_bits_original,
        )

        ncs = debug_ref["ncs"]
        counters_per_word = fun.counters_per_word(ncs, BITS_PER_PHASE)

        resultados = []

        # Agora inverte um bit da mensagem por vez e compara com a saída de referência.
        for bit_index in range(len(mensagem_original)):
            mensagem_alterada = flip_bit(mensagem_original, bit_index)

            # Cada mensagem alterada começa em um contador diferente para separar os testes.
            counter_start_alt = (bit_index + 1) * counters_per_word

            bits_alt, _ = processa_palavra_spe(
                mensagem_bits=mensagem_alterada,
                nonce_bits=nonce_bits,
                counter_start_decimal=counter_start_alt,
                key_bits=key_bits_original,
                n_fases=ncs,
            )

            difusao = fun.hamming_distance(bits_ref, bits_alt)

            resultados.append({
                "Mensagem Original": mensagem_original,
                "Mensagem alterada": mensagem_alterada,
                "Mensagem criptografada original": bits_ref,
                "Mensagem alterada criptografada": bits_alt,
                "Chave Original": KEY_ORIGINAL,
                "Chave alterada": KEY_ORIGINAL,
                "Bit da mensagem alterado": bit_index,
                "Difusão": difusao,
                "Difusão normalizada": difusao / NUM_BITS,
            })

        df = pd.DataFrame(resultados)

        # Nome limpo para salvar corretamente dentro da pasta de saída.
        out_file = OUTPUT_DIR / f"data{loop_hist}.xlsx"
        df.to_excel(out_file, index=False)

        print(
            f"loop_hist={loop_hist + 1}/{N_WORDS} | "
            f"media difusão/Nr = {df['Difusão normalizada'].mean():.6f} | "
            f"SPE+embaralhamento | Ncs = {ncs} | arquivo = {out_file}"
        )


if __name__ == "__main__":
    roda_experimento_difusao()
