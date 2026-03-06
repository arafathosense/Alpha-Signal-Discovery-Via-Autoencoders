import pandas as pd
import numpy as np
import plotly.graph_objs as go
import plotly.offline as pyo
from datetime import datetime
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim

# ========== DATA GENERATION ==========
def generate_signals(n_samples=200, seed=42):
    np.random.seed(seed)
    dates = pd.date_range(end=datetime.today(), periods=n_samples)
    df = pd.DataFrame(index=dates)
    walk = np.cumsum(np.random.randn(n_samples))
    df['momentum'] = walk
    df['volatility'] = np.abs(np.random.randn(n_samples))
    df['trend'] = np.linspace(-1, 1, n_samples) + 0.2 * np.random.randn(n_samples)
    df['carry'] = np.sin(np.linspace(0, 8 * np.pi, n_samples)) + 0.1 * np.random.randn(n_samples)
    df['skew'] = np.random.randn(n_samples)
    df['kurtosis'] = np.random.randn(n_samples)
    return df

# ========== PREPROCESSING ==========
def preprocess_signals(df):
    scaler = StandardScaler()
    normed = pd.DataFrame(scaler.fit_transform(df), index=df.index, columns=df.columns)
    normed = normed.replace([np.inf, -np.inf], 0).fillna(0)
    return normed

# ========== AUTOENCODER ==========
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

# ========== MAIN EXECUTION ==========
if __name__ == "__main__":
    df = generate_signals()
    df_proc = preprocess_signals(df)
    latent_dim = 3
    model, Z = train_autoencoder(df_proc.values, latent_dim, epochs=100, lr=1e-3)
    Z_df = pd.DataFrame(Z, index=df_proc.index, columns=[f"Factor {i+1}" for i in range(latent_dim)])

    # 3D Plot
    fig = go.Figure()
    fig.add_trace(go.Scatter3d(
        x=Z_df.iloc[:,0], y=Z_df.iloc[:,1], z=Z_df.iloc[:,2],
        mode='markers+lines',
        marker=dict(size=5, color=Z_df.index.astype(int), colorscale='Viridis', opacity=0.8),
        line=dict(color='cyan', width=2),
        text=Z_df.index.strftime('%Y-%m-%d'),
        hoverinfo='text+x+y+z',
    ))
    fig.update_layout(
        title="3D Latent Factor Space",
        scene=dict(
            xaxis_title='Factor 1',
            yaxis_title='Factor 2',
            zaxis_title='Factor 3',
        ),
        width=900,
        height=600,
    )
    pyo.plot(fig, filename="latent_factor_3d.html", auto_open=True)
    print("Graph loaded: latent_factor_3d.html")
