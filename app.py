import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import plotly.express as px
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances
from datetime import datetime

# =====================
# CONFIG & STYLES
# =====================
st.set_page_config(
    page_title="Alpha Signal Discovery via Autoencoders",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🧬"
)

DARK_BG = "#181A20"
DARK_CARD = "#23272F"
ACCENT = "#00FFC6"

st.markdown(f"""
    <style>
    .reportview-container {{ background: {DARK_BG}; }}
    .sidebar .sidebar-content {{ background: {DARK_CARD}; }}
    .stCard {{ background: {DARK_CARD}; border-radius: 16px; }}
    .stMetric {{ color: {ACCENT}; }}
    .block-container {{ padding-top: 2rem; }}
    .css-1d391kg {{ background: {DARK_BG}; }}
    .css-1v0mbdj {{ background: {DARK_BG}; }}
    .css-1cpxqw2 {{ background: {DARK_BG}; }}
    </style>
""", unsafe_allow_html=True)

# =====================
# DATA PIPELINE
# =====================
def generate_signals(n_samples=500, n_features=6, seed=42):
    np.random.seed(seed)
    # Ensure enough samples for rolling window
    window = st.session_state.get('window', 30)
    n_samples = max(n_samples, window + 200)
    dates = pd.date_range(end=datetime.today(), periods=n_samples)
    df = pd.DataFrame(index=dates)
    walk = np.cumsum(np.random.randn(n_samples))
    df['momentum'] = pd.Series(walk).rolling(20).mean().fillna(0)
    df['volatility'] = pd.Series(walk).rolling(20).std().fillna(0)
    df['trend'] = np.linspace(-1, 1, n_samples) + 0.2 * np.random.randn(n_samples)
    df['carry'] = np.sin(np.linspace(0, 8 * np.pi, n_samples)) + 0.1 * np.random.randn(n_samples)
    df['skew'] = pd.Series(walk).rolling(20).skew().fillna(0)
    df['kurtosis'] = pd.Series(walk).rolling(20).kurt().fillna(0)
    return df

def preprocess_signals(df, window=60):
    scaler = StandardScaler()
    # Rolling window alignment
    df_rolled = df.rolling(window).mean().dropna()
    if df_rolled.shape[0] == 0:
        st.warning(f"Not enough data for rolling window of size {window}. Please reduce the window size or provide more data.")
        st.stop()
    normed = pd.DataFrame(scaler.fit_transform(df_rolled), index=df_rolled.index, columns=df_rolled.columns)
    return normed

# =====================
# AUTOENCODER MODEL
# =====================
class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, input_dim)
        )
    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat, z

def train_autoencoder(X, latent_dim, epochs=100, lr=1e-3):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Autoencoder(X.shape[1], latent_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    for epoch in range(epochs):
        optimizer.zero_grad()
        x_hat, _ = model(X_tensor)
        loss = criterion(x_hat, X_tensor)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        _, Z = model(X_tensor)
    return model, Z.cpu().numpy()

# =====================
# SIDEBAR
# =====================
signal_source = st.sidebar.radio("Signal Source", ["Upload CSV", "Generate Signals"], index=1, key="signal_source_radio")
df_raw = None

# SIDEBAR (single set, all widgets with unique keys)
st.sidebar.title("Alpha Signal Discovery")
latent_dim = st.sidebar.slider("Latent Dimension Size", min_value=2, max_value=8, value=3, key="latent_dim")
window = st.sidebar.slider("Training Window Size", min_value=5, max_value=120, value=10, key="window")
df_raw = None
selected_signals = []
if signal_source == "Upload CSV":
    uploaded = st.sidebar.file_uploader("Upload Engineered Signals CSV", type=["csv"], key="csv_uploader")
    if uploaded:
        df_raw = pd.read_csv(uploaded, index_col=0, parse_dates=True)
        selected_signals = st.sidebar.multiselect("Select Signals", df_raw.columns.tolist(), default=df_raw.columns.tolist(), key="signal_select")
else:
    df_raw = generate_signals(n_samples=window+200)
    selected_signals = st.sidebar.multiselect("Select Signals", df_raw.columns.tolist(), default=df_raw.columns.tolist(), key="signal_select")

if not selected_signals:
    st.warning("Select at least one signal.")
    st.stop()

# =====================
# DATA PREPROCESSING
# =====================
df = df_raw[selected_signals]
df_proc = preprocess_signals(df, window=window)

# =====================
# MODEL TRAINING
# =====================
model, Z = train_autoencoder(df_proc.values, latent_dim, epochs=150, lr=1e-3)
Z_df = pd.DataFrame(Z, index=df_proc.index, columns=[f"Factor {i+1}" for i in range(latent_dim)])

# =====================
# FACTOR METRICS
# =====================
factor_returns = Z_df.diff().fillna(0)
factor_vols = factor_returns.std()
factor_means = Z_df.mean()
factor_corr = Z_df.corr()

# =====================
# MAIN LAYOUT
# =====================
st.title("🧬 Alpha Signal Discovery via Autoencoders")
st.markdown("""
    <h3 style='color:#00FFC6;font-weight:600;'>Latent Alpha Factor Analysis</h3>
    <p style='color:#B0B0B0;'>Discover hidden drivers of financial returns using deep representation learning. Explore latent factor structure, dynamics, and relationships in a cinematic quant dashboard.</p>
""", unsafe_allow_html=True)

# Metric Cards
cols = st.columns(latent_dim)
for i in range(latent_dim):
    cols[i].metric(f"Factor {i+1} Mean", f"{factor_means[i]:.3f}", f"Vol: {factor_vols[i]:.3f}")

st.markdown("---")

# =====================
# 3D LATENT FACTOR SPACE
# =====================
if latent_dim >= 3:
    fig_3d = go.Figure()
    fig_3d.add_trace(go.Scatter3d(
        x=Z_df.iloc[:,0], y=Z_df.iloc[:,1], z=Z_df.iloc[:,2],
        mode='markers+lines',
        marker=dict(size=5, color=Z_df.index.astype(int), colorscale='Viridis', opacity=0.8),
        line=dict(color=ACCENT, width=2),
        text=Z_df.index.strftime('%Y-%m-%d'),
        hoverinfo='text+x+y+z',
    ))
    fig_3d.update_layout(
        title="3D Latent Factor Space",
        scene=dict(
            xaxis_title='Factor 1',
            yaxis_title='Factor 2',
            zaxis_title='Factor 3',
            bgcolor=DARK_BG,
        ),
        paper_bgcolor=DARK_BG,
        font=dict(color=ACCENT),
        margin=dict(l=0, r=0, b=0, t=40),
        width=1200,
        height=600,
    )
    fig_3d.update_traces(marker=dict(line=dict(width=0.5, color=DARK_CARD)))
    st.plotly_chart(fig_3d, use_container_width=True)

# =====================
# ANIMATED EVOLUTION
# =====================
frames = []
if latent_dim >= 3:
    for i in range(0, len(Z_df), max(1, len(Z_df)//50)):
        frame = go.Frame(
            data=[go.Scatter3d(
                x=Z_df.iloc[:i+1,0], y=Z_df.iloc[:i+1,1], z=Z_df.iloc[:i+1,2],
                mode='markers+lines',
                marker=dict(size=5, color=Z_df.index[:i+1].astype(int), colorscale='Viridis', opacity=0.8),
                line=dict(color=ACCENT, width=2),
                text=Z_df.index[:i+1].strftime('%Y-%m-%d'),
                hoverinfo='text+x+y+z',
            )],
            name=str(i)
        )
        frames.append(frame)
    anim_fig = go.Figure(
        data=[go.Scatter3d(
            x=Z_df.iloc[:,0], y=Z_df.iloc[:,1], z=Z_df.iloc[:,2],
            mode='markers+lines',
            marker=dict(size=5, color=Z_df.index.astype(int), colorscale='Viridis', opacity=0.8),
            line=dict(color=ACCENT, width=2),
            text=Z_df.index.strftime('%Y-%m-%d'),
            hoverinfo='text+x+y+z',
        )],
        frames=frames
    )
    anim_fig.update_layout(
        title="Animated Latent Factor Evolution",
        scene=dict(
            xaxis_title='Factor 1',
            yaxis_title='Factor 2',
            zaxis_title='Factor 3',
            bgcolor=DARK_BG,
        ),
        paper_bgcolor=DARK_BG,
        font=dict(color=ACCENT),
        margin=dict(l=0, r=0, b=0, t=40),
        width=1200,
        height=600,
        updatemenus=[dict(
            type="buttons",
            showactive=False,
            buttons=[dict(label="Play", method="animate", args=[None, {"frame": {"duration": 50, "redraw": True}, "fromcurrent": True, "transition": {"duration": 0}}])]
        )]
    )
    st.plotly_chart(anim_fig, use_container_width=True)

# =====================
# FACTOR RETURN TIME SERIES
# =====================
ret_fig = px.line(factor_returns, labels={"value": "Return", "index": "Date"}, title="Factor Return Time Series")
ret_fig.update_layout(
    paper_bgcolor=DARK_BG,
    plot_bgcolor=DARK_BG,
    font=dict(color=ACCENT),
    margin=dict(l=0, r=0, b=0, t=40),
    width=1200,
    height=400,
)
st.plotly_chart(ret_fig, use_container_width=True)

# =====================
# FACTOR CORRELATION HEATMAP
# =====================
heatmap_fig = px.imshow(factor_corr, text_auto=True, color_continuous_scale='Viridis', title="Factor Correlation Heatmap")
heatmap_fig.update_layout(
    paper_bgcolor=DARK_BG,
    font=dict(color=ACCENT),
    margin=dict(l=0, r=0, b=0, t=40),
    width=800,
    height=600,
)
st.plotly_chart(heatmap_fig, use_container_width=True)

# =====================
# FOOTER
# =====================
st.markdown("""
    <div style='text-align:center;color:#888;margin-top:2em;'>
    <b>Alpha Signal Discovery via Autoencoders</b> | Quant Research Lab | Powered by Streamlit, PyTorch, Plotly
    </div>
""", unsafe_allow_html=True)
