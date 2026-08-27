# main.py
# =============================================================================
# Geração de espectros e constelações do sistema 16-QAM com criptografia
# espectral de fase e embaralhamento de amplitude.
# =============================================================================

from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import plotly.io as pio

from Data_L import input_data as inp
from Func_L import functions as fun


# =============================================================================
# Configurações principais do experimento
# =============================================================================

NUM_WORDS = 200    
NUM_BITS = 1024       # 1024 bits (256 símbolos) para alta resolução no espectro

SPS = 8              # 8 amostras/símbolo: fs = 224 GHz e Nyquist = ±112 GHz para Rs=28 Gbaud
ROLLOFF = 0.02
SPAN = 160            # Span do filtro RCF
BITS_PER_PHASE = 8

SAVE_HTML = False
SHOW_POINTWISE_MEAN_SPECTRA = True
SHOW_POINTWISE_MEAN_CONSTELLATIONS = False
SHOW_CURSOR_GUIDES = True

OUTPUT_HTML_DIR = Path("figuras_interativas")
OUTPUT_HTML_DIR.mkdir(parents=True, exist_ok=True)

pio.renderers.default = "browser"


# =============================================================================
# Funções auxiliares para os gráficos (com régua granular ajustada)
# =============================================================================

def safe_filename(title):
    name = "".join(c if c.isalnum() else "_" for c in title.lower())
    while "__" in name:
        name = name.replace("__", "_")
    return name.strip("_") or "figura"


def plot_spectra_interactive(
    frequencies,
    spectra,
    labels=None,
    title="Espectro de frequência",
    figsize=(11, 5),
    stem_stride=1,
    linewidth=1.5,
    xscale="linear",
    xlim=(-60, 60),      # Janela de visualização em GHz
    ylim=None,           # Limite Y automático ou especificado
    x_dtick=1.0,         # Marcador principal a cada 1.0 GHz
    grid=True,
):
    f = np.asarray(frequencies).ravel()
    n_points = f.size

    if isinstance(spectra, (list, tuple)):
        spectra_list = [np.asarray(s).ravel() for s in spectra]
    else:
        arr = np.asarray(spectra)
        if arr.ndim == 1:
            spectra_list = [arr.ravel()]
        elif arr.ndim == 2 and arr.shape[0] == n_points:
            spectra_list = [arr[:, k].ravel() for k in range(arr.shape[1])]
        elif arr.ndim == 2 and arr.shape[1] == n_points:
            spectra_list = [arr[k, :].ravel() for k in range(arr.shape[0])]
        else:
            raise ValueError("Formato de espectro inválido para este gráfico.")

    magnitudes = [np.abs(s) for s in spectra_list]

    if labels is None or len(labels) != len(magnitudes):
        labels = [f"Rodada {i + 1}" for i in range(len(magnitudes))]

    indices = np.arange(0, n_points, stem_stride)
    f_hz = f[indices]
    f_ghz = f_hz / 1e9

    fig = go.Figure()

    def build_stem_vectors(x_vals, y_vals, idx_vals, f_hz_vals):
        n = len(x_vals)
        x_stem = np.empty(n * 3)
        y_stem = np.empty(n * 3)
        
        x_stem[0::3] = x_vals
        x_stem[1::3] = x_vals
        x_stem[2::3] = None

        y_stem[0::3] = 0
        y_stem[1::3] = y_vals
        y_stem[2::3] = None

        custom_stem = np.empty((n * 3, 3))
        custom_stem[0::3] = np.stack([idx_vals, f_hz_vals, x_vals], axis=-1)
        custom_stem[1::3] = np.stack([idx_vals, f_hz_vals, x_vals], axis=-1)
        custom_stem[2::3] = np.nan

        return x_stem, y_stem, custom_stem

    vivid_colors = [
        "#FF0055", "#00E5FF", "#76FF03", "#D500F9", "#FF9100", 
        "#2979FF", "#FFEA00", "#00E676", "#FF1744", "#AA00FF", 
        "#00B0FF", "#1DE9B6", "#FF6D00", "#C6FF00", "#F50057", "#651FFF"
    ]

    for i, (label, magnitude) in enumerate(zip(labels, magnitudes)):
        y_plot = magnitude[indices]
        x_stem, y_stem, custom_stem = build_stem_vectors(f_ghz, y_plot, indices, f_hz)
        color = vivid_colors[i % len(vivid_colors)]

        fig.add_trace(go.Scattergl(
            x=x_stem,
            y=y_stem,
            mode="lines",
            name=label,
            opacity=0.6,
            line=dict(width=linewidth, color=color),
            customdata=custom_stem,
            connectgaps=False,
            hovertemplate=(
                f"<b>{label}</b><br>"
                "índice = %{customdata[0]}<br>"
                "frequência = %{customdata[1]:.6e} Hz<br>"
                "frequência = %{customdata[2]:.6f} GHz<br>"
                "magnitude = %{y:.6e}"
                "<extra></extra>"
            ),
        ))

    if SHOW_POINTWISE_MEAN_SPECTRA and len(magnitudes) > 0:
        stacked = np.vstack([m[indices] for m in magnitudes])
        mean_curve = np.mean(stacked, axis=0)
        std_curve = np.std(stacked, axis=0, ddof=1) if len(magnitudes) > 1 else np.zeros_like(mean_curve)

        custom_mean = np.stack([indices, f_hz, f_ghz, std_curve], axis=-1)

        fig.add_trace(go.Scattergl(
            x=f_ghz,
            y=mean_curve,
            mode="lines",
            name="Média ponto a ponto",
            line=dict(width=3.0, color="#000000"),
            customdata=custom_mean,
            hovertemplate=(
                "<b>Média ponto a ponto</b><br>"
                "índice = %{customdata[0]}<br>"
                "frequência = %{customdata[1]:.6e} Hz<br>"
                "frequência = %{customdata[2]:.6f} GHz<br>"
                "magnitude média = %{y:.6e}<br>"
                "desvio padrão = %{customdata[3]:.6e}"
                "<extra></extra>"
            ),
        ))

    # Configuração de Régua Granular e Limites Fixo/Ajustados
    fig.update_layout(
        title=dict(text=title, x=0.5, font=dict(size=16)),
        xaxis=dict(
            title=dict(
                    text="Frequência (GHz)",
                    font=dict(size=20)
                ),
            type="log" if xscale == "log" else "linear",
            range=xlim if xlim is not None else None,
            dtick=x_dtick,          # Rótulo numérico impresso (ex: a cada 2.0 GHz)
            tickangle=0,            # Mantém a numeração 100% na horizontal
            showgrid=grid,
            gridcolor="rgba(200, 200, 200, 0.4)",
            
            
            # CONFIGURAÇÃO DA SUBESCALA DE 0.25 GHz
            minor=dict(
                dtick=0.25,                  # Passo da subescala (risquinhos a cada 0.25 GHz)
                ticks="outside",             # Desenha os risquinhos para fora do eixo
                ticklen=3,                   # Tamanho do risquinho secundário
                tickwidth=1,                 # Espessura do risquinho secundário
                tickcolor="black",
                showgrid=grid,               # Exibe as linhas de grade finas no fundo
                gridcolor="rgba(230, 230, 230, 0.25)",
                gridwidth=0.5
            ),
            
            ticks="outside",
            ticklen=7,
            tickwidth=1.5,
            showline=True,
            linecolor="black",
            mirror=True
        ),
        yaxis=dict(         
            title=dict(
                    text="Magnitude (u.a.)",
                    font=dict(size=20)
                ),
            range=ylim if ylim is not None else None,
            showgrid=grid,
            gridcolor="rgba(200, 200, 200, 0.4)",
            minor=dict(
                showgrid=grid,
                gridcolor="rgba(230, 230, 230, 0.3)",
                gridwidth=0.5
            ),
            ticks="outside",
            ticklen=6,
            tickwidth=1.5,
            showline=True,
            linecolor="black",
            mirror=True
        ),
        hovermode="x unified",
        dragmode="zoom",
        template="plotly_white",
        width=int(figsize[0] * 110),
        height=int(figsize[1] * 110),
        margin=dict(l=60, r=40, t=50, b=50),
        legend=dict(
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor="gray",
            borderwidth=1,
        ),
    )

    if SAVE_HTML:
        fig.write_html(OUTPUT_HTML_DIR / f"{safe_filename(title)}.html", auto_open=True, config={"scrollZoom": True})
    else:
        fig.show(config={"scrollZoom": True})



def plot_psd_before_filter(
    frequencies,
    spectra,
    symbol_rate,
    sps,
    title="PSD linear não normalizada antes do filtro RCF: simulação e sinc² teórica",
    xlim=None,
):
    """
    Compara, em ESCALA LINEAR, a PSD média simulada do sinal com a referência teórica sinc².

    A curva simulada é mostrada em escala linear sem normalização. A sinc² teórica\n    é escalada pelo pico bruto da simulação apenas para permitir comparação visual\n    na mesma escala vertical.
    """
    f = np.asarray(frequencies, dtype=float).ravel()
    spectra_list = [np.asarray(x).ravel() for x in spectra]
    if not spectra_list:
        raise ValueError("Nenhum espectro disponível para calcular a PSD.")

    # Potência espectral média simulada: E{|X(f)|²}.
    # IMPORTANTE: aqui NÃO há normalização da curva simulada.
    power = np.vstack([np.abs(x) ** 2 for x in spectra_list])
    psd_mean = np.mean(power, axis=0)

    # Suavização leve entre bins vizinhos apenas para reduzir serrilhado.
    smooth_bins = 5
    kernel = np.ones(smooth_bins) / smooth_bins
    psd_mean = np.convolve(psd_mean, kernel, mode="same")

    # Pico bruto da simulação. Ele é usado somente para colocar as referências
    # teóricas na mesma escala vertical, sem alterar os valores da simulação.
    psd_peak = max(np.max(psd_mean), np.finfo(float).tiny)

    # Referência contínua do pulso retangular de duração Ts = 1/Rs.
    # A sinc² tem pico unitário por definição; multiplicamos pelo pico bruto
    # da simulação apenas para permitir comparação visual na mesma escala.
    psd_sinc2 = (np.sinc(f / float(symbol_rate)) ** 2) * psd_peak

    # Frequência de amostragem usada apenas para definir Nyquist
    # e para o diagnóstico dos múltiplos de Rs.
    fs = float(symbol_rate) * float(sps)

    f_ghz = f / 1e9

    # Diagnóstico numérico nos múltiplos da taxa de símbolos.
    print("\n=== Verificação da Figura 5 - escala linear sem normalização ===")
    print(f"Rs = {symbol_rate/1e9:.3f} Gbaud | SPS = {sps} | fs = {fs/1e9:.3f} GHz")
    for k in (1, 2, 3):
        target = k * symbol_rate
        if target <= fs / 2:
            idx = int(np.argmin(np.abs(f - target)))
            print(f"PSD simulada bruta em +{k}Rs ({f[idx]/1e9:.3f} GHz): {psd_mean[idx]:.6f}")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=f_ghz, y=psd_mean, mode="lines",
        name="PSD média",
        line=dict(width=2.6),
    ))
    fig.add_trace(go.Scatter(
        x=f_ghz, y=psd_sinc2, mode="lines",
        name="sinc² teórica",
        line=dict(width=2.0, dash="dash"),
    ))
    nyquist = fs / 2.0
    if xlim is None:
        shown = min(3.5 * symbol_rate, 0.95 * nyquist)
        xlim = (-shown / 1e9, shown / 1e9)

    max_abs_hz = max(abs(xlim[0]), abs(xlim[1])) * 1e9
    max_k = int(np.floor(max_abs_hz / symbol_rate))
    for k in range(1, max_k + 1):
        for sign in (-1, 1):
            x = sign * k * symbol_rate / 1e9
            if xlim[0] <= x <= xlim[1]:
                label = f"-{k}Rs" if sign < 0 else f"{k}Rs"
                fig.add_vline(
                    x=x,
                    line_width=1,
                    line_dash="dash",
                    annotation_text=label,
                    annotation_position="top",
                )

    fig.update_layout(
        title=dict(text=title, x=0.5),
        xaxis=dict(
            title=dict(
                    text="Frequência (GHz)",
                    font=dict(size=20)
                ),
            range=list(xlim),
            dtick=14,
            showgrid=True,
            minor=dict(dtick=2, showgrid=True),
        ),
        yaxis=dict(
            title=dict(
                    text="Potência espectral média (u.a.)",
                    font=dict(size=20)
                ),
            showgrid=True,
        ),
        hovermode="x unified",
        template="plotly_white",
        width=1250,
        height=650,
        legend=dict(
        bgcolor="rgba(255,255,255,0.85)",
        font=dict(size=20)
        ),
    )

    if SAVE_HTML:
        fig.write_html(
            OUTPUT_HTML_DIR / f"{safe_filename(title)}.html",
            auto_open=True,
            config={"scrollZoom": True},
        )
    else:
        fig.show(config={"scrollZoom": True})

def plot_constellations_interactive(
    signals,
    sps=1,
    labels=None,
    title="Constelações sobrepostas",
    drop_zero=False,
    tol=1e-12,
    alpha=0.6,
    size=10,
):
    if isinstance(signals, np.ndarray) and signals.ndim == 1:
        signals = [signals]

    if labels is None:
        labels = [f"Rodada {i + 1}" for i in range(len(signals))]

    processed = []

    for signal in signals:
        signal = np.asarray(signal).ravel()
        symbols = signal[::sps] if sps > 1 else signal

        if drop_zero:
            symbols = symbols[np.abs(symbols) > tol]

        processed.append(symbols)

    non_empty = [p for p in processed if len(p)]

    if non_empty:
        all_real = np.concatenate([np.real(p) for p in non_empty])
        all_imag = np.concatenate([np.imag(p) for p in non_empty])
        limit = max(np.max(np.abs(all_real)), np.max(np.abs(all_imag)))
        padding = 0.15 * limit if limit > 0 else 1.0
    else:
        limit = 1.0
        padding = 1.0

    axis_min = -limit - padding
    axis_max = limit + padding

    fig = go.Figure()

    for label, symbols in zip(labels, processed):
        if len(symbols) == 0:
            continue

        fig.add_trace(go.Scatter(
            x=np.real(symbols),
            y=np.imag(symbols),
            mode="markers",
            name=label,
            opacity=alpha,
            marker=dict(size=size),
            hovertemplate=(
                f"<b>{label}</b><br>"
                "I = %{x:.6f}<br>"
                "Q = %{y:.6f}"
                "<extra></extra>"
            ),
        ))

    fig.update_layout(
        title=dict(text=title, x=0.5),
        xaxis=dict(title="Componente I", range=[axis_min, axis_max], scaleanchor="y", scaleratio=1),
        yaxis=dict(title="Componente Q", range=[axis_min, axis_max]),
        template="plotly_white",
        width=780,
        height=780,
    )

    if SAVE_HTML:
        fig.write_html(OUTPUT_HTML_DIR / f"{safe_filename(title)}.html", auto_open=True)
    else:
        fig.show()


# =============================================================================
# Processamento principal
# =============================================================================

def run_experiment():
    spectra_before_filter = []
    spectra_after_rcf = []
    spectra_after_phase = []
    spectra_after_shuffle = []
    spectra_after_deshuffle = []
    spectra_recovered = []

    const_before_filter = []
    const_after_rcf = []
    const_after_phase = []
    const_after_shuffle = []
    const_after_deshuffle = []
    const_recovered = []

    frequencies = None
    last_time_vector = None

    nonce_bits_base = "1111000011110001111100101111001111110100111101011111011011110111"

    for word_index in range(NUM_WORDS):
        original_bits = fun.generate_random_bits(NUM_BITS)
        key_bits = "".join(map(str, fun.generate_random_bits(256)))
        counter_bits = fun.int_to_64bit_string(word_index)

        # 1) Modulação e Filtro RCF
        signal_before, signal_after_rcf, time_vector = fun.modulate_16qam_natural_for_spectra(
            original_bits,
            inp.Rsymbol,
            SPS,
            alpha=ROLLOFF,
            span=SPAN,
            gain=1.0,
        )

        last_time_vector = time_vector
        num_samples = len(signal_after_rcf)

        # Vetor de frequência natural baseado no número de amostras reais
        dt = time_vector[1] - time_vector[0]
        frequencies = np.fft.fftshift(np.fft.fftfreq(num_samples, d=dt))

        # 2) Espectro Shifted
        spectrum_before = fun.spectrum_shifted(signal_before)
        spectrum_rcf = fun.spectrum_shifted(signal_after_rcf)

        # 3) Bins centrais
        ncs, n1, n2 = fun.calculate_center_bin_count(num_samples, SPS, ROLLOFF)
        center_positions, _, _ = fun.central_indices_from_ncs(num_samples, ncs)

        # 4) Fases do AES
        key_phases, _ = fun.aes_phase_levels_from_ctr(
            key_bits=key_bits,
            nonce_bits=nonce_bits_base,
            counter_bits=counter_bits,
            n_phases=ncs,
            bits_per_phase=BITS_PER_PHASE,
        )

        # 5) Aplicação de fase
        spectrum_phase = fun.apply_phase_on_indices(
            spectrum_rcf,
            center_positions,
            key_phases,
        )

        # 6) Embaralhamento
        permutation = fun.ranking_vector(key_phases, len(key_phases))

        spectrum_shuffle = fun.shuffle_amplitude_on_indices(
            spectrum_phase,
            center_positions,
            permutation,
        )

        # 7) Desembaralhamento
        spectrum_deshuffle = fun.deshuffle_amplitude_on_indices(
            spectrum_shuffle,
            center_positions,
            permutation,
        )

        # 8) Remoção de fase
        spectrum_recovered = fun.remove_phase_on_indices(
            spectrum_deshuffle,
            center_positions,
            key_phases,
        )

        # 9) Retorno ao domínio do tempo perfeito
        signal_before_time = fun.time_from_spectrum_shifted(spectrum_before)
        signal_rcf_time = fun.time_from_spectrum_shifted(spectrum_rcf)
        signal_phase_time = fun.time_from_spectrum_shifted(spectrum_phase)
        signal_shuffle_time = fun.time_from_spectrum_shifted(spectrum_shuffle)
        signal_deshuffle_time = fun.time_from_spectrum_shifted(spectrum_deshuffle)
        signal_recovered_time = fun.time_from_spectrum_shifted(spectrum_recovered)

        spectra_before_filter.append(spectrum_before)
        spectra_after_rcf.append(spectrum_rcf)
        spectra_after_phase.append(spectrum_phase)
        spectra_after_shuffle.append(spectrum_shuffle)
        spectra_after_deshuffle.append(spectrum_deshuffle)
        spectra_recovered.append(spectrum_recovered)

        const_before_filter.append(signal_before_time)
        const_after_rcf.append(signal_rcf_time)
        const_after_phase.append(signal_phase_time)
        const_after_shuffle.append(signal_shuffle_time)
        const_after_deshuffle.append(signal_deshuffle_time)
        const_recovered.append(signal_recovered_time)

    return {
        "frequencies": frequencies,
        "spectra_before_filter": spectra_before_filter,
        "spectra_after_rcf": spectra_after_rcf,
        "spectra_after_phase": spectra_after_phase,
        "spectra_after_shuffle": spectra_after_shuffle,
        "spectra_after_deshuffle": spectra_after_deshuffle,
        "spectra_recovered": spectra_recovered,
        "const_before_filter": const_before_filter,
        "const_after_rcf": const_after_rcf,
        "const_after_phase": const_after_phase,
        "const_after_shuffle": const_after_shuffle,
        "const_after_deshuffle": const_after_deshuffle,
        "const_recovered": const_recovered,
    }


def plot_all_results(results):
    frequencies = results["frequencies"]

    # Figura 5: avaliação espectral correta do pulso retangular.
    # A janela ±100 GHz mostra os zeros em ±28, ±56 e ±84 GHz.
    plot_psd_before_filter(
        frequencies,
        results["spectra_before_filter"],
        symbol_rate=inp.Rsymbol,
        sps=SPS,
        title="PSD antes do filtro RCF — simulação e sinc² teórica",
        xlim=(-100, 100),
    )
    plot_spectra_interactive(frequencies, results["spectra_after_rcf"], title="Espectros depois do filtro RCF", xlim=(-22, 22), x_dtick=2.0)
    plot_spectra_interactive(frequencies, results["spectra_after_phase"], title="Espectros após mudança de fase", xlim=(-22, 22), x_dtick=2.0)
    plot_spectra_interactive(frequencies, results["spectra_after_shuffle"], title="Espectros após embaralhamento", xlim=(-22, 22), x_dtick=2.0)
    plot_spectra_interactive(frequencies, results["spectra_after_deshuffle"], title="Espectros após desembaralhamento", xlim=(-22, 22), x_dtick=2.0)
    plot_spectra_interactive(frequencies, results["spectra_recovered"], title="Espectros recuperados", xlim=(-22, 22), x_dtick=2.0)

    plot_constellations_interactive(results["const_before_filter"], SPS, drop_zero=True, title="Constelações antes do filtro")
    plot_constellations_interactive(results["const_after_rcf"], SPS, drop_zero=True, title="Constelações depois do filtro RCF")
    plot_constellations_interactive(results["const_after_phase"], SPS, drop_zero=True, title="Constelações após mudança de fase")
    plot_constellations_interactive(results["const_after_shuffle"], SPS, drop_zero=True, title="Constelações após embaralhamento")
    plot_constellations_interactive(results["const_after_deshuffle"], SPS, drop_zero=True, title="Constelações após desembaralhamento")
    plot_constellations_interactive(results["const_recovered"], SPS, drop_zero=True, title="Constelações do sinal recuperado")

if __name__ == "__main__":
    experiment_results = run_experiment()
    plot_all_results(experiment_results)