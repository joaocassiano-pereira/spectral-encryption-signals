
# main_constelacoes_estilo_artigo.py
# =============================================================================
#   - espectro médio à esquerda
#   - constelação em dispersão à direita
# para as etapas do sistema 16-QAM com criptografia espectral.
# =============================================================================

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec

from Data_L import input_data as inp
from Func_L import functions as fun


# =============================================================================
# Configurações principais
# =============================================================================

NUM_WORDS = 300         
NUM_BITS = 1024          # bits por rodada
SPS = 8                  # amostras por símbolo
ROLLOFF = 0.02
SPAN = 160
BITS_PER_PHASE = 8

# Faixa do gráfico espectral (GHz)
SPECTRUM_XLIM_GHZ = (-20, 20)

# Escala da constelação
# None = calcula automaticamente uma mesma janela I/Q para todas as etapas.
CONSTELLATION_LIMIT = None
CONSTELLATION_PERCENTILE = 99.5   # evita que poucos outliers dominem o limite
CONSTELLATION_MARGIN = 1.10       # 10% de margem visual
CONSTELLATION_SCATTER_ROUNDS = 12 # quantas rodadas coloridas sobrepor na constelação
CONSTELLATION_SCATTER_SIZE = 6
CONSTELLATION_SCATTER_ALPHA = 0.65

# Escala do eixo Y do espectro (unidades arbitrárias)
SPECTRUM_ARB_MAX = 50.0

# Quantas rodadas coloridas mostrar por espectro.
# As demais continuam contribuindo para a média, mas não são desenhadas.
SPECTRUM_COLOR_ROUNDS = 12

# Paleta viva semelhante aos gráficos interativos antigos
VIVID_COLORS = [
    "#FF0055", "#00E5FF", "#76FF03", "#D500F9", "#FF9100",
    "#2979FF", "#FFEA00", "#00E676", "#FF1744", "#AA00FF",
    "#00B0FF", "#1DE9B6", "#FF6D00", "#C6FF00", "#F50057",
    "#651FFF", "#00C853", "#FF3D00", "#304FFE", "#64DD17"
]

# Salvar figura em PNG
SAVE_FIGURE = True
OUTPUT_FIGURE = "figura_espectros_constelacoes_abnt.png"

# Mostrar figura
SHOW_FIGURE = True


# =============================================================================
# Funções auxiliares
# =============================================================================

def compute_average_magnitude(spectra):
    """
    Calcula a média ponto a ponto da magnitude do espectro.
    """
    if len(spectra) == 0:
        raise ValueError("Lista de espectros vazia.")

    stacked = np.vstack([np.abs(np.asarray(s).ravel()) for s in spectra])
    mean_mag = np.mean(stacked, axis=0)

    # Suavização leve para reduzir serrilhado visual
    kernel = np.ones(5) / 5.0
    mean_mag = np.convolve(mean_mag, kernel, mode="same")
    return mean_mag


def collect_scaled_spectra(spectra, scale_factor, max_rounds=20):
    """
    Retorna algumas magnitudes individuais já escaladas para o mesmo eixo
    usado pela curva média. Isso permite ver amostras/bins mudando de posição.
    """
    selected = spectra[:max_rounds]
    curves = []
    for s in selected:
        mag = np.abs(np.asarray(s).ravel()) * scale_factor
        curves.append(mag)
    return curves


def collect_symbols(signals, sps=1, drop_zero=False, tol=1e-12, offset=0):
    """
    Junta símbolos de várias rodadas para formar a nuvem da constelação.
    """
    all_symbols = []

    for sig in signals:
        sig = np.asarray(sig).ravel()

        if sps > 1:
            symbols = sig[offset::sps]
        else:
            symbols = sig

        if drop_zero:
            symbols = symbols[np.abs(symbols) > tol]

        all_symbols.append(symbols)

    if not all_symbols:
        return np.array([], dtype=complex)

    return np.concatenate(all_symbols)


def collect_round_symbols(signals, sps=1, drop_zero=False, tol=1e-12, offset=0, max_rounds=None):
    """
    Retorna uma lista de vetores de símbolos, um por rodada, para plot scatter colorido.
    """
    collected = []
    selected = signals if max_rounds is None else signals[:max_rounds]

    for sig in selected:
        sig = np.asarray(sig).ravel()
        symbols = sig[offset::sps] if sps > 1 else sig
        if drop_zero:
            symbols = symbols[np.abs(symbols) > tol]
        collected.append(symbols)

    return collected


def automatic_constellation_limit(stages):
    """
    Calcula um único limite 
    """
    values = []

    for _, _, _, signals in stages:
        symbols = collect_symbols(
            signals,
            sps=SPS,
            drop_zero=True,
            offset=0
        )
        if symbols.size:
            values.append(np.abs(np.real(symbols)))
            values.append(np.abs(np.imag(symbols)))

    if not values:
        return 5.0

    values = np.concatenate(values)
    lim = np.percentile(values, CONSTELLATION_PERCENTILE)
    lim = max(float(lim) * CONSTELLATION_MARGIN, 4.5)

    print(f"Limite automático das constelações: ±{lim:.3f}")
    return lim


# =============================================================================
# Processamento principal
# =============================================================================

def run_experiment():
    spectra_after_rcf = []
    spectra_after_phase = []
    spectra_after_shuffle = []
    spectra_after_deshuffle = []
    spectra_recovered = []

    const_after_rcf = []
    const_after_phase = []
    const_after_shuffle = []
    const_after_deshuffle = []
    const_recovered = []

    frequencies = None
    nonce_bits_base = "1111000011110001111100101111001111110100111101011111011011110111"

    for word_index in range(NUM_WORDS):
        original_bits = fun.generate_random_bits(NUM_BITS)
        key_bits = "".join(map(str, fun.generate_random_bits(256)))
        counter_bits = fun.int_to_64bit_string(word_index)

        # 1) Modulação + filtro RCF
        signal_before, signal_after_rcf, time_vector = fun.modulate_16qam_natural_for_spectra(
            original_bits,
            inp.Rsymbol,
            SPS,
            alpha=ROLLOFF,
            span=SPAN,
            gain=1.0,
        )

        num_samples = len(signal_after_rcf)
        dt = time_vector[1] - time_vector[0]
        frequencies = np.fft.fftshift(np.fft.fftfreq(num_samples, d=dt))

        spectrum_rcf = fun.spectrum_shifted(signal_after_rcf)

        # 2) Região central
        ncs, _, _ = fun.calculate_center_bin_count(num_samples, SPS, ROLLOFF)
        center_positions, _, _ = fun.central_indices_from_ncs(num_samples, ncs)

        # 3) Fases
        key_phases, _ = fun.aes_phase_levels_from_ctr(
            key_bits=key_bits,
            nonce_bits=nonce_bits_base,
            counter_bits=counter_bits,
            n_phases=ncs,
            bits_per_phase=BITS_PER_PHASE,
        )

        # 4) Fase
        spectrum_phase = fun.apply_phase_on_indices(
            spectrum_rcf,
            center_positions,
            key_phases,
        )

        # 5) Embaralhamento das componentes espectrais
        # A amostra complexa é deslocada como um todo: amplitude e fase vão juntas.
        permutation = fun.ranking_vector(key_phases, len(key_phases))
        spectrum_shuffle = fun.shuffle_amplitude_on_indices(
            spectrum_phase,
            center_positions,
            permutation,
        )

        # 6) Desembaralhamento das componentes espectrais
        # A operação inversa recoloca cada amostra complexa em sua posição original.
        spectrum_deshuffle = fun.deshuffle_amplitude_on_indices(
            spectrum_shuffle,
            center_positions,
            permutation,
        )

        # 7) Recuperação
        spectrum_recovered = fun.remove_phase_on_indices(
            spectrum_deshuffle,
            center_positions,
            key_phases,
        )

        # 8) Volta para o domínio do tempo
        signal_rcf_time = fun.time_from_spectrum_shifted(spectrum_rcf)
        signal_phase_time = fun.time_from_spectrum_shifted(spectrum_phase)
        signal_shuffle_time = fun.time_from_spectrum_shifted(spectrum_shuffle)
        signal_deshuffle_time = fun.time_from_spectrum_shifted(spectrum_deshuffle)
        signal_recovered_time = fun.time_from_spectrum_shifted(spectrum_recovered)

        # Armazenamento
        spectra_after_rcf.append(spectrum_rcf)
        spectra_after_phase.append(spectrum_phase)
        spectra_after_shuffle.append(spectrum_shuffle)
        spectra_after_deshuffle.append(spectrum_deshuffle)
        spectra_recovered.append(spectrum_recovered)

        const_after_rcf.append(signal_rcf_time)
        const_after_phase.append(signal_phase_time)
        const_after_shuffle.append(signal_shuffle_time)
        const_after_deshuffle.append(signal_deshuffle_time)
        const_recovered.append(signal_recovered_time)

    return {
        "frequencies": frequencies,
        "spectra_after_rcf": spectra_after_rcf,
        "spectra_after_phase": spectra_after_phase,
        "spectra_after_shuffle": spectra_after_shuffle,
        "spectra_after_deshuffle": spectra_after_deshuffle,
        "spectra_recovered": spectra_recovered,
        "const_after_rcf": const_after_rcf,
        "const_after_phase": const_after_phase,
        "const_after_shuffle": const_after_shuffle,
        "const_after_deshuffle": const_after_deshuffle,
        "const_recovered": const_recovered,
    }


# =============================================================================
# Figura final no estilo do artigo
# =============================================================================

def plot_article_style_figure(results):
    f = np.asarray(results["frequencies"]).ravel()
    f_ghz = f / 1e9

    stages = [
        ("(a)", "Após o RCF", results["spectra_after_rcf"], results["const_after_rcf"]),
        ("(b)", "Após encriptação de fase", results["spectra_after_phase"], results["const_after_phase"]),
        ("(c)", "Após embaralhamento espectral", results["spectra_after_shuffle"], results["const_after_shuffle"]),
        ("(d)", "Após desembaralhamento", results["spectra_after_deshuffle"], results["const_after_deshuffle"]),
        ("(e)", "Sinal recuperado", results["spectra_recovered"], results["const_recovered"]),
    ]

    # Média da magnitude para cada etapa
    avg_spectra = []
    for _, _, spectra, _ in stages:
        avg_spectra.append(compute_average_magnitude(spectra))

    # Escala global em unidades arbitrárias, comum a todas as etapas
    global_max = max(np.max(x) for x in avg_spectra)
    if global_max <= 0:
        raise ValueError("Não foi possível normalizar os espectros: magnitude máxima inválida.")
    scale_factor = SPECTRUM_ARB_MAX / global_max
    avg_spectra_scaled = [x * scale_factor for x in avg_spectra]

    # Curvas individuais coloridas por etapa
    colored_spectra = []
    for _, _, spectra, _ in stages:
        colored_spectra.append(
            collect_scaled_spectra(
                spectra,
                scale_factor=scale_factor,
                max_rounds=SPECTRUM_COLOR_ROUNDS
            )
        )

    # Define uma MESMA janela I/Q para todas as etapas.
    # Isso evita que as constelações criptografadas fiquem fora do quadro.
    constellation_lim = (
        automatic_constellation_limit(stages)
        if CONSTELLATION_LIMIT is None
        else float(CONSTELLATION_LIMIT)
    )

    # Símbolos por rodada para scatter colorido das constelações
    scatter_constellations = []
    for _, _, _, signals in stages:
        scatter_constellations.append(
            collect_round_symbols(
                signals,
                sps=SPS,
                drop_zero=True,
                offset=0,
                max_rounds=CONSTELLATION_SCATTER_ROUNDS
            )
        )

    nrows = len(stages)
    fig = plt.figure(figsize=(10, 10.8))
    gs = gridspec.GridSpec(
        nrows=nrows,
        ncols=3,
        width_ratios=[6.4, 1.35, 0.08],
        hspace=0.28,
        wspace=0.08
    )

    # Região mostrada do espectro
    mask = (f_ghz >= SPECTRUM_XLIM_GHZ[0]) & (f_ghz <= SPECTRUM_XLIM_GHZ[1])

    for i, ((letter, stage_title, _, _), mean_mag, individual_curves, constellation_rounds) in enumerate(
        zip(stages, avg_spectra_scaled, colored_spectra, scatter_constellations)
    ):
        ax_spec = fig.add_subplot(gs[i, 0])
        ax_const = fig.add_subplot(gs[i, 1])
        cax = fig.add_subplot(gs[i, 2])

        # Espectros individuais coloridos.
        # Assim é possível enxergar diferenças entre rodadas e, especialmente,
        # a redistribuição das componentes espectrais depois do embaralhamento.
        for j, curve in enumerate(individual_curves):
            ax_spec.plot(
                f_ghz[mask],
                curve[mask],
                color=VIVID_COLORS[j % len(VIVID_COLORS)],
                linewidth=0.8,
                alpha=0.40
            )

        # Curva média em preto por cima, como referência
        ax_spec.plot(
            f_ghz[mask],
            mean_mag[mask],
            color="black",
            linewidth=1.6,
            label="Média"
        )

        ax_spec.set_xlim(SPECTRUM_XLIM_GHZ)
        # dá uma pequena folga para picos individuais
        panel_max = max(
            [np.max(mean_mag[mask])] +
            [np.max(c[mask]) for c in individual_curves]
        )
        ax_spec.set_ylim(0, panel_max * 1.05)
        ax_spec.tick_params(direction="in", top=True, right=True, length=4)
        ax_spec.grid(False)

        # Letra da subfigura dentro do painel
        ax_spec.text(
            0.03, 0.10, letter,
            transform=ax_spec.transAxes,
            fontsize=11,
            fontweight="bold"
        )

        if i < nrows - 1:
            ax_spec.set_xticklabels([])
        else:
            ax_spec.set_xlabel("Frequência (GHz)", fontsize=11)

        if i == nrows // 2:
            ax_spec.set_ylabel("Amplitude média (u.a.)", fontsize=11)
        else:
            ax_spec.set_ylabel("")

        # Constelação com bolinhas coloridas (scatter), como nos seus gráficos antigos.
        for j, symbols in enumerate(constellation_rounds):
            if len(symbols) == 0:
                continue
            ax_const.scatter(
                np.real(symbols),
                np.imag(symbols),
                s=CONSTELLATION_SCATTER_SIZE,
                alpha=CONSTELLATION_SCATTER_ALPHA,
                color=VIVID_COLORS[j % len(VIVID_COLORS)],
                edgecolors='none'
            )

        ax_const.set_xticks([])
        ax_const.set_yticks([])
        ax_const.set_xlim(-constellation_lim, constellation_lim)
        ax_const.set_ylim(-constellation_lim, constellation_lim)
        ax_const.set_aspect('equal', adjustable='box')
        for spine in ax_const.spines.values():
            spine.set_visible(True)

        # A terceira coluna vira apenas espaço em branco para manter o alinhamento
        cax.axis('off')

    # Ajuste manual porque grids com colorbars não combinam bem com tight_layout().
    plt.subplots_adjust(left=0.09, right=0.97, bottom=0.08, top=0.98, hspace=0.24, wspace=0.08)

    if SAVE_FIGURE:
        fig.savefig(OUTPUT_FIGURE, dpi=300, bbox_inches="tight")
        print(f"Figura salva em: {Path(OUTPUT_FIGURE).resolve()}")
        print()
        print("Legenda sugerida para inserir no Word:")
        print(
            "Figura X – Espectros e constelações do sinal nas diferentes etapas do "
            "processamento: (a) após o filtro RCF; (b) após a encriptação de fase; "
            "(c) após o embaralhamento das componentes espectrais; (d) após o desembaralhamento "
            "espectral; e (e) após a recuperação do sinal."
        )
        print("Fonte: Elaborado pelo autor (2026).")

    if SHOW_FIGURE:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    results = run_experiment()
    plot_article_style_figure(results)
