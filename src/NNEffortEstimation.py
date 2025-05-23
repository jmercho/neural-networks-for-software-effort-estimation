# Modelos de redes neuronales para la estimación de esfuerzo en proyectos de desarrollo de software
# Autor: John Jairo Mercado
# Fecha: 2025-04-04

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from scipy.io import arff
import os
import joblib  
import argparse 

from sklearn.preprocessing import StandardScaler, PowerTransformer, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
from torch.utils.data import DataLoader, TensorDataset
import warnings
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
warnings.filterwarnings('ignore')

# Configuración general
torch.manual_seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Creación de directorios
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
METRICS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results', 'metrics')
PLOTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results', 'plots')
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models', 'saved_models')

# Asegurar que los directorios existan
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(os.path.join(DATA_DIR, 'datasets'), exist_ok=True)
os.makedirs(METRICS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# 1. Preprocesamiento de datos ================================================
def load_and_process(dataset_path, y_col):
    # Adjust path to use data directory
    full_path = os.path.join(DATA_DIR, 'datasets', dataset_path)
    data = arff.loadarff(full_path)
    df = pd.DataFrame(data[0])
    
    # Extraer el nombre del dataset desde la ruta
    dataset_name = os.path.basename(dataset_path).split('.')[0]
    metrics_file = os.path.join(METRICS_DIR, f"{dataset_name}_metrics.txt")
    
    # Convertir bytes a float y manejar valores faltantes
    for col in df.columns:
        if df[col].dtype == 'object':
            try:
                df[col] = df[col].str.decode('utf-8').astype(float)
            except:
                continue
    
    # Obtener el tamaño original del dataset
    original_shape = df.shape
    
    # Seleccionar solo columnas numéricas (excluyendo la variable objetivo)
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    x_cols = [col for col in numeric_cols if col != y_col]
    
    # Eliminar filas con valores NaN o infinitos
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=x_cols + [y_col])
    
    # Eliminar ceros de la variable objetivo para evitar división por cero en MMRE
    original_after_nan = len(df)
    df = df[df[y_col] > 0]
    zeros_removed = original_after_nan - len(df)
    
    # Calcular correlaciones con la variable objetivo para la generación de informes
    correlations = df[x_cols + [y_col]].corr()[y_col].abs().sort_values(ascending=False)
    
    # Obtener estadísticas iniciales del dataset
    with open(metrics_file, 'w', encoding='utf-8') as f:
        f.write(f"{'='*50}\n")
        f.write(f"METRICS DEL DATASET: {dataset_name.upper()}\n")
        f.write(f"{'='*50}\n\n")
        
        f.write(f"Tamaño original del dataset: {original_shape}\n")
        f.write(f"Tamaño del dataset después de eliminar valores NaN: {original_after_nan} filas\n")
        f.write(f"Tamaño del dataset después de eliminar ceros en la variable objetivo: {len(df)} filas\n")
        f.write(f"Total de características utilizadas: {len(x_cols)}\n")
        f.write(f"Características: {', '.join(x_cols)}\n")
        f.write(f"Target: {y_col}\n\n")
        
        f.write("Correlaciones con la variable objetivo:\n")
        for feature, corr in correlations.items():
            if feature != y_col:  # Omitir la columna objetivo
                f.write(f"- {feature}: {corr:.4f}\n")
        f.write("\n")
        
        f.write("Estadísticas del dataset antes del preprocesamiento:\n")
        stats_before = df[x_cols + [y_col]].describe()
        f.write(stats_before.to_string())
        f.write("\n\n")
    
    print(f"Métricas iniciales guardadas en {metrics_file}")
    
    # Imprimir estadísticas del dataset para debugging
    print("\nEstadísticas del dataset antes del preprocesamiento:")
    print(df[x_cols + [y_col]].describe())
    print(f"Usando todas las {len(x_cols)} características para el entrenamiento")
    
    # Usar IQR para la eliminación de outliers, y ajustar el factor de acuerdo al tamaño del dataset
    Q1 = df[x_cols + [y_col]].quantile(0.25)
    Q3 = df[x_cols + [y_col]].quantile(0.75)
    IQR = Q3 - Q1
    
    # Usar un umbral menos agresivo para la detección de outliers para datasets pequeños
    iqr_factor = 3 if len(df) < 100 else 2
    print(f"Usando un factor de IQR de {iqr_factor} para la detección de outliers")
    
    outlier_mask = ~((df[x_cols + [y_col]] < (Q1 - iqr_factor * IQR)) | 
                     (df[x_cols + [y_col]] > (Q3 + iqr_factor * IQR))).any(axis=1)
    
    original_row_count = len(df)
    df = df[outlier_mask]
    outliers_removed = original_row_count - len(df)
    
    print(f"Después de la eliminación de outliers: {len(df)} muestras restantes")
    
    # Aplicar características polinómicas
    poly = PolynomialFeatures(2, include_bias=False, interaction_only=True)
    X_poly = poly.fit_transform(df[x_cols].values)
    
    # Aplicar transformación logarítmica para mejor escalado
    X = np.log1p(X_poly)
    y = np.log1p(df[y_col].values.reshape(-1, 1))
    
    # Aplicar estandarización
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # Ajustar la estratificación según el tamaño del dataset
    test_size = 0.2
    sample_count = len(df)
    expected_test_samples = int(sample_count * test_size)
    
    # Estratificación según el tamaño del dataset
    if sample_count < 20:  # Dataset muy pequeño
        print(f"Dataset demasiado pequeño ({sample_count} muestras), usando simple división entrenamiento/prueba sin estratificación")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, shuffle=True)
    else:
        # Determinar el número óptimo de bins para la estratificación
        max_bins = min(4, expected_test_samples)
        print(f"Usando {max_bins} bins para la estratificación")
        
        # Crear bins con el número ajustado
        y_bins = pd.qcut(df[y_col], q=max_bins, labels=False, duplicates='drop')
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y_bins)
    
    # Estadísticas después del preprocesamiento
    print("\nEstadísticas del dataset después del preprocesamiento:")
    print(f"Forma de X: {X.shape}")
    print(f"Forma de X_train: {X_train.shape}, forma de X_test: {X_test.shape}")
    print(f"Forma de y_train: {y_train.shape}, forma de y_test: {y_test.shape}")
    
    # Guardar estadísticas después del preprocesamiento
    with open(metrics_file, 'a', encoding='utf-8') as f:
        f.write(f"Outliers eliminados: {outliers_removed} ({outliers_removed/original_row_count:.2%} de los datos originales)\n\n")
        f.write(f"Características polinómicas (solo interacciones): Dimensiones de entrada {len(x_cols)} -> {X.shape[1]}\n\n")
        
        f.write("Estadísticas del dataset después del preprocesamiento:\n")
        f.write(f"Forma de X: {X.shape}\n")
        f.write(f"Media de X: {np.mean(X, axis=0)}\n")
        f.write(f"Desviación estándar de X: {np.std(X, axis=0)}\n")
        f.write(f"Media de y: {np.mean(y)}\n")
        f.write(f"Desviación estándar de y: {np.std(y)}\n\n")
        
        f.write(f"Tamaño del conjunto de entrenamiento: {X_train.shape[0]}\n")
        f.write(f"Tamaño del conjunto de prueba: {X_test.shape[0]}\n")
    
    return (X_train, y_train), (X_test, y_test), scaler

# 2. Modelos ==================================================================
# MLP con Optimización Evolutiva
class EvolutiveMLP(nn.Module):
    def __init__(self, input_size, hidden_layers, activation='relu', dropout_rate=0.1):
        super().__init__()
        self.input_norm = nn.BatchNorm1d(input_size)
        
        # Crear ModuleList para almacenar las capas de manera correcta
        self.layers = nn.ModuleList()
        prev_size = input_size
        
        for neurons in hidden_layers:
            self.layers.append(nn.Linear(prev_size, neurons))
            self.layers.append(nn.BatchNorm1d(neurons))
            self.layers.append(nn.ReLU() if activation == 'relu' else nn.Tanh())
            self.layers.append(nn.Dropout(dropout_rate))
            prev_size = neurons
        
        self.output = nn.Linear(prev_size, 1)
    
    def forward(self, x):
        x = self.input_norm(x)
        for layer in self.layers:
            x = layer(x)
        return self.output(x)

# Modelo híbrido CNN-GRU
class CNNGRU(nn.Module):
    def __init__(self, input_size, conv_filters, gru_units, dropout_rate=0.1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(input_size, conv_filters, kernel_size=2, padding=1),
            nn.BatchNorm1d(conv_filters),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.MaxPool1d(kernel_size=1)
        )
        self.gru = nn.GRU(conv_filters, gru_units, batch_first=True, 
                         bidirectional=True)
        self.fc = nn.Linear(gru_units * 2, 1)
    
    def forward(self, x):
        # Forma de x: [batch_size, input_features]
        # Añadir dimensión de secuencia
        x = x.unsqueeze(-1)  # Ahora: [batch_size, input_features, 1]
        x = self.conv(x)
        x = x.permute(0, 2, 1)  # Cambiar a [batch_size, seq_len, features] para GRU
        _, hn = self.gru(x)
        hn = torch.cat((hn[-2], hn[-1]), dim=1)  # Combinar estados bidireccionales
        return self.fc(hn)

# 3. Entrenamiento y Métricas =================================================
def train_model(model, X_train, y_train, epochs=10000, lr=0.001, patience=100, plot=False):
    model.to(device)
    criterion = nn.HuberLoss(delta=0.5)  # Usar pérdida Huber en lugar de MSE
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.001) # Usar AdamW como optimizador, que desacopla el decaimiento del peso en la actualización del gradiente.
    
    # Usar scheduler de cosine annealing: Se ajusta la tasa de aprendizaje siguiendo una curva coseno.

    # eta_t = eta_min + 0.5 * (eta_max - eta_min) * (1 + cos(t/T_max * pi))
    # eta_t: Tasa de aprendizaje en la época t
    # eta_min: Tasa de aprendizaje mínima
    # eta_max: Tasa de aprendizaje máxima
    # T_max: Número máximo de épocas
    # t: Número de épocas actual

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-6
    )
    
    # Añadir clip de gradientes
    max_grad_norm = 1.0
    
    # Crear conjuntos de entrenamiento y validación
    train_size = int(0.8 * len(X_train))
    train_data = TensorDataset(
        torch.FloatTensor(X_train[:train_size]).to(device), 
        torch.FloatTensor(y_train[:train_size]).to(device)
    )
    val_data = TensorDataset(
        torch.FloatTensor(X_train[train_size:]).to(device), 
        torch.FloatTensor(y_train[train_size:]).to(device)
    )
    
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=32, shuffle=False)
    
    best_loss = float('inf')
    patience_counter = 0
    best_model_state = model.state_dict().copy()  # Inicializar con el estado actual
    
    # Inicializar diccionario de historial
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_mae': [],
        'val_mae': []
    }
    
    try:
        # Comprobar datos de entrada
        if torch.any(torch.isnan(torch.FloatTensor(X_train))):
            raise ValueError("Valores NaN encontrados en los datos de entrenamiento")
        if torch.any(torch.isnan(torch.FloatTensor(y_train))):
            raise ValueError("Valores NaN encontrados en los datos de objetivo")
        
        for epoch in range(epochs):
            # Fase de entrenamiento
            model.train()
            train_loss = 0
            train_mae = 0
            for inputs, targets in train_loader:
                optimizer.zero_grad()
                outputs = model(inputs)
                
                # Comprobar NaN en outputs
                if torch.any(torch.isnan(outputs)):
                    print(f"NaN detectado en outputs en la época {epoch}")
                    print(f"Valores de entrada: {inputs}")
                    raise ValueError("NaN en salidas del modelo")
                
                loss = criterion(outputs, targets)
                loss.backward()
                
                # Clip de gradientes: Evitar gradientes excesivamente grandes que pueden causar explosiones de gradientes, mediante la normalización de los gradientes.
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                
                optimizer.step()
                
                # Comprobar valor de pérdida
                if np.isnan(loss.item()):
                    print(f"Pérdida NaN en la época {epoch}")
                    raise ValueError("Pérdida NaN detectada")
                
                train_loss += loss.item()
                train_mae += torch.mean(torch.abs(outputs - targets)).item()
            
            train_loss /= len(train_loader)
            train_mae /= len(train_loader)
            
            # Fase de validación
            model.eval()
            val_loss = 0
            val_mae = 0
            with torch.no_grad():
                for inputs, targets in val_loader:
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    val_loss += loss.item()
                    val_mae += torch.mean(torch.abs(outputs - targets)).item()
            
            val_loss /= len(val_loader)
            val_mae /= len(val_loader)
            
            # Actualizar tasa de aprendizaje
            scheduler.step()
            
            # Guardar historial
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['train_mae'].append(train_mae)
            history['val_mae'].append(val_mae)
            
            # Comprobar detención temprana
            if val_loss < best_loss:
                best_loss = val_loss
                best_model_state = model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Detención temprana en la época {epoch}")
                    break
            
            if epoch % 10 == 0:
                print(f"Época {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")
                
    except Exception as e:
        print(f"Error durante el entrenamiento: {str(e)}")
        if best_model_state is None:
            best_model_state = model.state_dict().copy()
    
    # Cargar el mejor estado del modelo
    model.load_state_dict(best_model_state)
    
    # Plot if requested: Graficar la pérdida y el MAE durante el entrenamiento y la validación.
    if plot:
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(history['train_loss'], label='Entrenamiento')
        plt.plot(history['val_loss'], label='Validación')
        plt.title('Pérdida del Modelo')
        plt.ylabel('Pérdida')
        plt.xlabel('Época')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(history['train_mae'], label='Entrenamiento')
        plt.plot(history['val_mae'], label='Validación')
        plt.title('MAE del Modelo')
        plt.ylabel('MAE')
        plt.xlabel('Época')
        plt.legend()
        
        plt.tight_layout()
        plt.show()
    
    return model, history

# 3. Evaluación del Modelo =====================================================
def evaluate_model(model, X_test, y_test):
    with torch.no_grad():
        test_tensor = torch.FloatTensor(X_test).to(device)
        preds = model(test_tensor)
        preds = preds.cpu().numpy()
    
    # Calcular métricas
    mae = np.mean(np.abs(preds - y_test))
    mse = np.mean((preds - y_test)**2)
    rmse = np.sqrt(mse)
    
    relative_errors = np.abs((y_test - preds) / y_test)
    mmre = np.mean(relative_errors)
    mdmre = np.median(relative_errors)
    pred25 = np.mean(relative_errors <= 0.25) * 100
    pred30 = np.mean(relative_errors <= 0.30) * 100
    
    return {
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'MMRE': mmre,
        'MdMRE': mdmre,
        'PRED(25)': pred25,
        'PRED(30)': pred30
    }

# 4. Guardar el Modelo =========================================================
def save_model(model, params, dataset_name, model_type):
    # Crear el nombre del archivo del modelo
    filename = f"{dataset_name}_{model_type}.pth"
    filepath = os.path.join(MODELS_DIR, filename)
    
    # Guardar el estado del modelo y los hiperparámetros
    save_dict = {
        'model_state': model.state_dict(),
        'hyperparameters': params,
        'architecture': str(model)
    }
    torch.save(save_dict, filepath)
    print(f"Modelo guardado en {filepath}")

def find_best_hyperparameters(X_train, y_train, X_val, y_val, model_type='mlp', dataset_name='', input_size=1):
    best_metrics = float('inf')
    best_params = None
    best_model = None
    best_history = None
    
    print(f"Tamaño de entrada: {input_size}")
    print(f"Forma de X_train: {X_train.shape}")
    
    if model_type == 'mlp':
        hidden_layers_options = [
            [64], [128],
            [64, 32], [128, 64],
            [128, 64, 32]
        ]
        activations = ['relu', 'tanh']
        learning_rates = [0.001, 0.0005, 0.0001]
        dropout_rates = [0.1, 0.2, 0.3]
        
        for hidden in hidden_layers_options:
            for act in activations:
                for lr in learning_rates:
                    for dropout in dropout_rates:
                        try:
                            model = EvolutiveMLP(input_size=input_size, hidden_layers=hidden, 
                                               activation=act, dropout_rate=dropout)
                            model, history = train_model(model, X_train, y_train, 
                                                       epochs=10000, lr=lr, patience=100, plot=False)
                            metrics = evaluate_model(model, X_val, y_val)
                            
                            if metrics['RMSE'] < best_metrics:
                                best_metrics = metrics['RMSE']
                                best_params = {
                                    'hidden_layers': hidden, 
                                    'activation': act, 
                                    'lr': lr,
                                    'dropout': dropout
                                }
                                best_model = model
                                best_history = history
                                print(f"Nuevo mejor modelo encontrado: RMSE = {best_metrics:.4f}")
                        except Exception as e:
                            print(f"Error con los hiperparámetros {hidden}, {act}, {lr}, {dropout}: {str(e)}")
                            continue
    
    else:  # CNN-GRU
        conv_filters_options = [16, 32, 64]
        gru_units_options = [16, 32, 64]
        learning_rates = [0.001, 0.0005, 0.0001]
        dropout_rates = [0.1, 0.2, 0.3]
        
        for filters in conv_filters_options:
            for units in gru_units_options:
                for lr in learning_rates:
                    for dropout in dropout_rates:
                        try:
                            model = CNNGRU(input_size=input_size, conv_filters=filters, 
                                         gru_units=units, dropout_rate=dropout)
                            model, history = train_model(model, X_train, y_train, 
                                                       epochs=10000, lr=lr, patience=100, plot=False)
                            metrics = evaluate_model(model, X_val, y_val)
                            
                            if metrics['RMSE'] < best_metrics:
                                best_metrics = metrics['RMSE']
                                best_params = {
                                    'conv_filters': filters,
                                    'gru_units': units,
                                    'lr': lr,
                                    'dropout': dropout
                                }
                                best_model = model
                                best_history = history
                                print(f"Nuevo mejor modelo encontrado: RMSE = {best_metrics:.4f}")
                        except Exception as e:
                            print(f"Error con los hiperparámetros {filters}, {units}, {lr}, {dropout}: {str(e)}")
                            continue
    
    # Graficar solo para el mejor modelo
    if best_model is not None:
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(best_history['train_loss'], label='Entrenamiento')
        plt.plot(best_history['val_loss'], label='Validación')
        plt.title(f'Pérdida del Mejor Modelo {model_type.upper()} - {dataset_name}')
        plt.ylabel('Pérdida')
        plt.xlabel('Época')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(best_history['train_mae'], label='Entrenamiento')
        plt.plot(best_history['val_mae'], label='Validación')
        plt.title(f'MAE del Mejor Modelo {model_type.upper()} - {dataset_name}')
        plt.ylabel('MAE')
        plt.xlabel('Época')
        plt.legend()
        
        plt.tight_layout()
        plot_path = os.path.join(PLOTS_DIR, f'{dataset_name}_{model_type}_training_plot.png')
        plt.savefig(plot_path)
        plt.close()
    
    if best_model is None:
        raise ValueError("No se encontró un modelo válido durante la búsqueda de hiperparámetros")
    
    return best_model, best_params

# 4. Pipeline Principal =======================================================
def save_results_to_file(results, filename=None):
    if filename is None:
        filename = os.path.join(METRICS_DIR, 'final_metrics.txt')
    
    with open(filename, 'w') as f:
        for dataset, data in results.items():
            f.write(f"\n{'='*50}\n")
            f.write(f"Resultados para {dataset}:\n\n")
            
            for model_name, model_data in data.items():
                f.write(f"{model_name}:\n")
                f.write("Mejores hiperparámetros:\n")
                for param, value in model_data['params'].items():
                    f.write(f"- {param}: {value}\n")
                
                f.write("\nMétricas:\n")
                for metric, value in model_data['metrics'].items():
                    f.write(f"- {metric}: {value:.4f}\n")
                f.write("\n")

if __name__ == "__main__":
    # Configurar el parser de argumentos
    parser = argparse.ArgumentParser(description='Entrenar modelos de redes neuronales para estimación de esfuerzo de software')
    parser.add_argument('--datasets', nargs='+', type=str, 
                        help='Lista de conjuntos de datos a usar (predeterminado: todos los conjuntos de datos disponibles)')
    parser.add_argument('--models', nargs='+', type=str, choices=['mlp', 'cnn_gru'], 
                        help='Modelos a entrenar (predeterminado: ambos (mlp y cnn_gru))')
    
    args = parser.parse_args()
    
    datasets = {
        'albrecht': ('albrecht.arff', 'Effort'),    
        'china': ('china.arff', 'Effort'),
        'isbsg': ('isbsg.arff', 'NormalisedWorkEffortLevel1'),    
        'kitchenham': ('kitchenham_normalized.arff', 'Actual.effort'),    
        'maxwell': ('maxwell.arff', 'Effort'),    
        'miyazaki94': ('miyazaki94.arff', 'MM'),    
        'nasa93': ('nasa93.arff', 'act_effort'),
        'desharnais': ('desharnais.arff', 'Effort')
    }
    
    # Filtrar conjuntos de datos si se especifica
    if args.datasets:
        filtered_datasets = {}
        for name in args.datasets:
            if name in datasets:
                filtered_datasets[name] = datasets[name]
            else:
                print(f"Advertencia: El conjunto de datos '{name}' no se encontró. Conjuntos de datos disponibles: {', '.join(datasets.keys())}")
        datasets = filtered_datasets
        
        if not datasets:
            print("No se especificaron conjuntos de datos válidos.")
            exit(1)
    
    # Determinar qué modelos entrenar
    train_mlp = True
    train_cnn_gru = True
    
    if args.models:
        train_mlp = 'mlp' in args.models
        train_cnn_gru = 'cnn_gru' in args.models
    
    results = {}
    
    for name, (path, y_col) in datasets.items():
        print(f"\nProcesando conjunto de datos: {name}")
        
        # No necesitamos usar load_and_process con la ruta completa ya que la función la construye internamente
        (X_train, y_train), (X_test, y_test), scaler = load_and_process(path, y_col)
        
        # Actualizar el tamaño de entrada para los modelos usando el tamaño real después de las características polinómicas
        input_size = X_train.shape[1] 
        print(f"Tamaño de entrada actual después de las características polinómicas: {input_size}")
        
        # Dividir los datos de entrenamiento en conjuntos de entrenamiento y validación
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
        
        results[name] = {}
        
        # Entrenar MLP
        if train_mlp:
            print("Optimizando MLP...")
            best_mlp, mlp_params = find_best_hyperparameters(X_train, y_train, X_val, y_val, 
                                                           model_type='mlp', dataset_name=name,
                                                           input_size=input_size)
            best_mlp, mlp_history = train_model(best_mlp, X_train, y_train, epochs=10000, lr=mlp_params['lr'])
            mlp_metrics = evaluate_model(best_mlp, X_test, y_test)
            
            # Guardar el modelo MLP y el historial
            save_model(best_mlp, mlp_params, name, 'mlp')
            history_path = os.path.join(MODELS_DIR, f"{name}_mlp_history.npy")
            np.save(history_path, mlp_history)
            
            results[name]['MLP+Evolutivo'] = {
                'params': mlp_params,
                'metrics': mlp_metrics
            }
        
        # Entrenar CNN-GRU
        if train_cnn_gru:
            print("Optimizando CNN-GRU...")
            best_cnn_gru, cnn_gru_params = find_best_hyperparameters(X_train, y_train, X_val, y_val, 
                                                                   model_type='cnn_gru', dataset_name=name,
                                                                   input_size=input_size)
            best_cnn_gru, cnn_gru_history = train_model(best_cnn_gru, X_train, y_train, epochs=10000, lr=cnn_gru_params['lr'])
            cnn_gru_metrics = evaluate_model(best_cnn_gru, X_test, y_test)
            
            # Guardar el modelo CNN-GRU y el historial
            save_model(best_cnn_gru, cnn_gru_params, name, 'cnn_gru')
            history_path = os.path.join(MODELS_DIR, f"{name}_cnn_gru_history.npy")
            np.save(history_path, cnn_gru_history)
            
            results[name]['CNN+GRU'] = {
                'params': cnn_gru_params,
                'metrics': cnn_gru_metrics
            }
        
        # Guardar el escalador para este conjunto de datos
        scaler_filename = os.path.join(MODELS_DIR, f"{name}_scaler.pkl")
        joblib.dump(scaler, scaler_filename)
    
    # Imprimir resultados
    for dataset, data in results.items():
        print(f"\n{'='*50}")
        print(f"Resultados para {dataset}:")
        
        for model_name, model_data in data.items():
            print(f"\n{model_name}:")
            print("Mejores hiperparámetros:")
            for param, value in model_data['params'].items():
                print(f"- {param}: {value}")
            
            print("\nMétricas:")
            for metric, value in model_data['metrics'].items():
                print(f"- {metric}: {value:.4f}")
    
    # Guardar resultados en un archivo
    save_results_to_file(results)