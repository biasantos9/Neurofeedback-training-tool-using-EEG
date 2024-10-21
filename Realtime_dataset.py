import os
import pandas as pd
import numpy as np
from scipy.signal import filtfilt, iirnotch, iirfilter, sosfiltfilt, butter
from sklearn.ensemble import RandomForestClassifier
from mne.time_frequency import psd_array_multitaper
import time
import matplotlib.pyplot as plt

data_path = os.getcwd()

dataframes_original = {}

# Mapear os estados 
trials_dict = {'neutral': 0, 'relaxed': 1, 'concentrating': 2}

# Parâmetros dos filtros
notch_freq = 50 # Notch
quality_factor = 40
fs = 256  # Sampling rate in Hz
time_interval = 1.0 / fs  # Time interval between samples
highcut = 90 # Low-pass
lowcut = 4 # High-pass
order = 8

# Aplicação dos parâmetros dos filtros
b_notch, a_notch = iirnotch(notch_freq, quality_factor, fs)
sos = iirfilter(order, highcut, btype='lowpass', analog=False, ftype='butter', fs=256, output='sos')
b_hp, a_hp = butter(order, lowcut, btype='highpass', fs=256)

# Create RF classifier
rf_classifier = RandomForestClassifier(max_depth= None, min_samples_leaf=1, min_samples_split=2, n_estimators=100)

# Variáveis globais para armazenar os dados filtrados
predicted_labels = []

# Tamanho do chunk em segundos
chunk_size_seconds = 3
chunk_size_samples = fs * chunk_size_seconds
overlap_samples = chunk_size_samples // 2 # Sobreposição entre os chunks (50%)

# Banda de interesse
beta = (12, 35)

# Função para carregar os dados de um arquivo CSV
def load_data(file_path):
    df = pd.read_csv(file_path)
    data = df.iloc[:, 1:5] 
    num_samples = df.shape[0]  # Obtém o número de samples
    return data, num_samples

# Low-pass, high-pass e notch
def apply_filters(data):
    data_filtrado = pd.DataFrame(sosfiltfilt(sos, data.values, axis=0), columns=data.columns)
    data_filtrado_lphp = pd.DataFrame(filtfilt(b_hp, a_hp, data_filtrado.values, axis=0), columns=data.columns)
    filtered_channel = pd.DataFrame()
    for channel in data_filtrado_lphp.columns:
        filtered_data = pd.DataFrame(filtfilt(b_notch, a_notch, data_filtrado_lphp[channel]))
        filtered_channel[channel] = filtered_data.values.flatten()
    return filtered_channel

def calculate_average_power(freq, magnitude, low_freq, high_freq):
    mask = (freq >= low_freq) & (freq <= high_freq)
    freq_interval = freq[mask]
    magnitude_interval = magnitude[mask]
    average_power = np.trapz(magnitude_interval, x=freq_interval)
    return average_power

def extract_features(data):
    multitaper_features = []
    beta_powers = []  # Lista para armazenar os valores de beta_power
    
    for column in data.columns:
        psd_mt, freq_mt = psd_array_multitaper(data[column], fs, normalization='full', verbose=0)
        multitaper_features.append(psd_mt)
        beta_power = calculate_average_power(freq_mt, psd_mt, beta[0], beta[1])
        beta_powers.append(beta_power)
    
    multitaper_features = np.vstack(multitaper_features).T
    beta_powers = np.array(beta_powers).reshape(1, -1)  # Transformar em array numpy e remodelar
    return multitaper_features, beta_powers

def process_files(data_path):
    time_elapsed = 0  # Initializing time_elapsed
    for file in os.listdir(data_path):
        if file.endswith('1.csv') and 'subject' in file:
            file_path = os.path.join(data_path, file)
            estado = file.split('-')[1]
            sujeito = file.split('subject')[1][0]
            key = (estado, sujeito)
            if key not in dataframes_original:
                # Load data
                data, num_samples = load_data(file_path)

                # Check data length
                num_samples = len(data)
                if num_samples < chunk_size_samples:
                    print("Data too short, skipping...")
                    continue

                # Iterar sobre os chunks
                for i in range(0, num_samples - chunk_size_samples + 1, overlap_samples):
                    chunk_data = data.iloc[i:i + chunk_size_samples, :]
                    
                    # Preprocessing
                    filtered_data = apply_filters(chunk_data)

                    # Extract features
                    multitaper_features, beta_powers = extract_features(filtered_data)

                    if len(multitaper_features) > 0:

                        x = np.arange(len(filtered_data))/fs
                        y1 = filtered_data.iloc[:,0]
                        y2 = filtered_data.iloc[:,1]
                        y3 = filtered_data.iloc[:,2]
                        y4 = filtered_data.iloc[:,3]

                        plt.cla()
                        plt.plot(x, y1, label='TP9')
                        plt.plot(x, y2, label='AF7')
                        plt.plot(x, y3, label='AF8')
                        plt.plot(x, y4, label='TP10')

                        plt.legend(loc='upper left')
                        plt.tight_layout()
                        plt.draw()
                        plt.pause(0.05)
                        
                        all_data = np.array(beta_powers)
                        labels = [estado] * all_data.shape[0]
                        rf_classifier.fit(all_data, labels)
                        
                        # Previsões
                        y_pred = rf_classifier.predict(all_data)
                        y_pred_num = trials_dict[y_pred[0]]
                        print("Predicted label:", y_pred_num)

                        time.sleep(0.3)

while True: # Executar a função para processar os arquivos e plotar em tempo real
    process_files(data_path)
    time.sleep(5)  # Espera por 5 segundos antes de verificar novamente os arquivos
