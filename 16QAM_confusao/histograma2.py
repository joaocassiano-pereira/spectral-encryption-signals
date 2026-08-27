import os
import math
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

pio.renderers.default = "browser"

# =========================
# CONFIGURAÇÕES
# =========================

directory = r'D:\Rec\Mestrado\Codigo Final\16QAM_difusao\data_hist'

Nr = 256
Nw = 100

scenario_name = r'16QAM SPE confusion'

values = []

# =========================
# LEITURA DOS ARQUIVOS
# =========================

for filename in os.listdir(directory):
    if filename.endswith(".xlsx"):
        filepath = os.path.join(directory, filename)
        df = pd.read_excel(filepath)

        # Coluna 6 = "Confusão"
        column_values = df.iloc[:, 6].values
        values.extend(column_values)

values_df = pd.DataFrame(values, columns=["d"])
values_df = values_df.dropna()
values_df = values_df[values_df["d"] != 0]

d_values = values_df["d"].astype(int).to_numpy()

# =========================
# ESTATÍSTICAS DA SIMULAÇÃO
# =========================

d_norm = d_values / Nr

mean_sim = np.mean(d_norm)
std_sim = np.std(d_norm, ddof=1)

total_samples = len(d_values)

# =========================
# BINOMIAL IDEAL
# =========================

p = 0.5

k = np.arange(0, Nr + 1)

pmf_binomial = np.array([
    math.comb(Nr, int(ki)) * (p ** ki) * ((1 - p) ** (Nr - ki))
    for ki in k
])

x_binomial = k / Nr

mean_ideal = p
std_ideal = math.sqrt(Nr * p * (1 - p)) / Nr

# =========================
# SIMULAÇÃO EM PROBABILIDADE
# =========================

counts = np.bincount(d_values, minlength=Nr + 1)
prob_sim = counts / counts.sum()

x_sim = np.arange(0, Nr + 1) / Nr

# =========================
# FIGURA INTERATIVA
# =========================

fig = go.Figure()

# Binomial ideal em barras
fig.add_trace(go.Bar(
    x=x_binomial,
    y=pmf_binomial,
    name="Ideal Binomial",
    marker=dict(
        color="orange",
        line=dict(color="black", width=0.5)
    ),
    opacity=0.85,
    width=1 / Nr * 0.85,
    customdata=np.stack([k, x_binomial], axis=-1),
    hovertemplate=(
        "<b>Ideal Binomial</b><br>"
        "d = %{customdata[0]}<br>"
        "d/Nr = %{customdata[1]:.4f}<br>"
        "Probability = %{y:.6f}"
        "<extra></extra>"
    )
))

# Simulação em linha com marcadores
fig.add_trace(go.Scatter(
    x=x_sim,
    y=prob_sim,
    mode="lines+markers",
    name="Simulation",
    line=dict(color="blue", width=3),
    marker=dict(
        color="white",
        size=8,
        line=dict(color="blue", width=2)
    ),
    customdata=np.stack([np.arange(0, Nr + 1), x_sim, counts], axis=-1),
    hovertemplate=(
        "<b>Simulation</b><br>"
        "d = %{customdata[0]}<br>"
        "d/Nr = %{customdata[1]:.4f}<br>"
        "Count = %{customdata[2]}<br>"
        "Probability = %{y:.6f}"
        "<extra></extra>"
    )
))

# =========================
# CAIXA DE TEXTO
# =========================

stats_text = (
    f"Scenario: {scenario_name}<br>"
    f"N<sub>r</sub> = {Nr}, N<sub>w</sub> = {Nw}<br><br>"
    f"Simulation:<br>"
    f"Mean(d/N<sub>r</sub>) = {mean_sim:.4f}<br>"
    f"Std(d/N<sub>r</sub>) = {std_sim:.4f}<br><br>"
    f"Ideal:<br>"
    f"Mean(d/N<sub>r</sub>) = {mean_ideal:.4f}<br>"
    f"Std(d/N<sub>r</sub>) = {std_ideal:.4f}<br><br>"
    f"Total samples = {total_samples}"
)

fig.add_annotation(
    x=0.33,
    y=max(max(prob_sim), max(pmf_binomial)) * 0.95,
    xref="x",
    yref="y",
    text=stats_text,
    showarrow=False,
    align="left",
    bordercolor="gray",
    borderwidth=1,
    bgcolor="white",
    font=dict(size=14, color="black")
)

# =========================
# LAYOUT
# =========================

fig.update_layout(
    title=dict(
        text=f"Diffusion Histogram - {scenario_name}",
        x=0.5,
        font=dict(size=20)
    ),
    xaxis=dict(
        title="Normalized Hamming Distance d/Nr",
        range=[0.3, 0.7],
        dtick=0.05,
        showgrid=True,
        gridcolor="rgba(0,0,0,0.15)"
    ),
    yaxis=dict(
        title="Probability",
        range=[0, max(max(prob_sim), max(pmf_binomial)) * 1.15],
        dtick=0.005,
        showgrid=True,
        gridcolor="rgba(0,0,0,0.15)"
    ),
    legend=dict(
        x=0.82,
        y=0.98,
        bgcolor="white",
        bordercolor="gray",
        borderwidth=1
    ),
    hovermode="x unified",
    bargap=0,
    plot_bgcolor="white",
    paper_bgcolor="white",
    width=1400,
    height=750
)

fig.show()