import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
from torch.utils.data import DataLoader, TensorDataset

# --- CONFIGURAÇÕES ---
# O arquivo CSV deve estar na mesma pasta ou o caminho deve ser alterado aqui
PATH_DATASET = '2023_Yellow_Taxi_Trip_Data_20260225.csv'

def run_pipeline():
    # 1. Carregamento Seguro
    if not os.path.exists(PATH_DATASET):
        print(f"Erro: O arquivo {PATH_DATASET} não foi encontrado na raiz do projeto.")
        return

    print("Lendo dataset (1M rows)...")
    df = pd.read_csv(PATH_DATASET, encoding='latin1', nrows=1000000, low_memory=False)

    # 2. Pré-processamento e Limpeza
    cols_numericas = ['total_amount', 'trip_distance', 'passenger_count', 'extra', 
                      'mta_tax', 'tip_amount', 'tolls_amount', 'improvement_surcharge', 
                      'congestion_surcharge', 'Airport_fee', 'airport_fee']
    
    for col in cols_numericas:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    df['tpep_pickup_datetime'] = pd.to_datetime(df['tpep_pickup_datetime'], errors='coerce')
    df['tpep_dropoff_datetime'] = pd.to_datetime(df['tpep_dropoff_datetime'], errors='coerce')
    df = df.dropna(subset=['tpep_pickup_datetime', 'tpep_dropoff_datetime', 'total_amount'])

    df['hour'] = df['tpep_pickup_datetime'].dt.hour
    df['day_of_week'] = df['tpep_pickup_datetime'].dt.dayofweek
    df['duration_min'] = (df['tpep_dropoff_datetime'] - df['tpep_pickup_datetime']).dt.total_seconds() / 60

    # Filtros de Outliers
    df = df[(df['total_amount'] > 2.5) & (df['total_amount'] < 150)]
    df = df[(df['trip_distance'] > 0.1) & (df['trip_distance'] < 40)]
    df = df[(df['duration_min'] > 1) & (df['duration_min'] < 120)]

    # 3. Engenharia de Features
    features = ['trip_distance', 'passenger_count', 'hour', 'day_of_week', 'duration_min',
                'PULocationID', 'DOLocationID', 'RatecodeID', 'VendorID', 'payment_type',
                'extra', 'mta_tax', 'improvement_surcharge', 'congestion_surcharge']

    if 'airport_fee' in df.columns: features.append('airport_fee')
    else: features.append('extra')

    X = df[features].fillna(0).values
    y = df['total_amount'].values.reshape(-1, 1)

    # 4. Preparação para Deep Learning
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler_x = StandardScaler()
    X_train = scaler_x.fit_transform(X_train)
    X_test = scaler_x.transform(X_test)

    scaler_y = StandardScaler() 
    y_train_scaled = scaler_y.fit_transform(y_train)

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train_scaled, dtype=torch.float32)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)

    # 5. Arquitetura da Rede
    class TaxiNet(nn.Module):
        def __init__(self, input_size):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_size, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 1)
            )
        def forward(self, x): return self.net(x)

    model = TaxiNet(len(features))
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # 6. Treinamento
    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=2048, shuffle=True)
    history = []

    print("Iniciando Treinamento...")
    for epoch in range(100):
        epoch_losses = []
        for batch_x, batch_y in train_loader:
            pred = model(batch_x)
            loss = criterion(pred, batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())
        
        avg_loss = np.mean(epoch_losses)
        history.append(avg_loss)
        if (epoch+1) % 10 == 0:
            print(f"Época {epoch+1}/100 | Loss: {avg_loss:.6f}")

    # 7. Avaliação e Exportação de Gráficos
    model.eval()
    with torch.no_grad():
        y_pred_scaled = model(X_test_t).numpy()
        y_pred = scaler_y.inverse_transform(y_pred_scaled)

    print(f"\n--- MÉTRICAS FINAIS ---")
    print(f"R² Score: {r2_score(y_test, y_pred):.4f}")
    print(f"MAE: ${mean_absolute_error(y_test, y_pred):.2f}")

    # Salvar gráficos como imagem para o GitHub
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history)
    plt.title('Curva de Aprendizado')
    plt.subplot(1, 2, 2)
    plt.scatter(y_test[:500], y_pred[:500], alpha=0.5, color='orange')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.title('Real vs Predição')
    plt.savefig('resultado_modelo.png')
    print("\nGráfico salvo como 'resultado_modelo.png'")

if __name__ == "__main__":
    run_pipeline()
