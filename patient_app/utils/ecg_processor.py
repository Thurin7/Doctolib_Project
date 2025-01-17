import pandas as pd
import numpy as np
from scipy.signal import find_peaks, savgol_filter
from datetime import datetime

class ECGProcessor:
    def __init__(self, sample_rate=120):
        self.sample_rate = sample_rate
        
    def load_data(self, filename):
        df = pd.read_csv(filename)
        return df['ECG_Value'].values
    
    def analyze_cycle_distance(self, signal, visualize=True):
        processed_signal = (signal - np.mean(signal)) / np.std(signal)
        processed_signal = savgol_filter(processed_signal, window_length=51, polyorder=3)
        processed_signal = processed_signal - np.mean(processed_signal)
        
        autocorr = np.correlate(processed_signal, processed_signal, mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        autocorr = autocorr / autocorr[0]
        
        time_axis = np.arange(len(autocorr)) / self.sample_rate
        min_samples = int(0.2 * self.sample_rate)
        peaks, properties = find_peaks(autocorr[min_samples:],
                                     height=0.3,
                                     distance=min_samples,
                                     prominence=0.2)
        peaks = peaks + min_samples
        
        if len(peaks) > 0:
            cycle_length = peaks[0]
            period = cycle_length / self.sample_rate
            print(f"Période détectée : {period:.3f} secondes")
            print(f"Fréquence cardiaque estimée : {60/period:.1f} BPM")
            return cycle_length
        return None

    def find_r_peaks(self, signal, cycle_length=None):
        if cycle_length is None:
            min_distance = 30
        else:
            min_distance = int(0.8 * cycle_length)
            
        peaks, _ = find_peaks(signal,
                            height=350,
                            distance=min_distance,
                            prominence=50)
        return peaks

    def extract_cycles(self, signal, r_peaks, before_r=20, after_r=30, target_length=182):
        cycles = []
        valid_peaks = []
        
        for peak in r_peaks:
            if peak - before_r >= 0 and peak + after_r < len(signal):
                # Extraire le cycle
                cycle = signal[peak - before_r:peak + after_r]
                cycle_min = np.min(cycle)
                cycle_max = np.max(cycle)
                cycle_norm = (cycle - cycle_min) / (cycle_max - cycle_min)
                
                # Redimensionner le cycle
                x_original = np.arange(len(cycle_norm))
                x_new = np.linspace(0, len(cycle_norm)-1, target_length)
                cycle_resampled = np.interp(x_new, x_original, cycle_norm)
                
                cycles.append(cycle_resampled)
                valid_peaks.append(peak)
        
        return np.array(cycles), np.array(valid_peaks)
    
    def save_cycles(self, cycles, filename=None):
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"ecg_cycles_{timestamp}.csv"
        
        # Utiliser le chemin complet vers le dossier media
        from django.conf import settings
        import os
        
        # Créer un sous-dossier 'processed_ecg' dans media
        processed_dir = os.path.join(settings.MEDIA_ROOT, 'processed_ecg')
        os.makedirs(processed_dir, exist_ok=True)
        
        # Chemin complet du fichier
        full_path = os.path.join(processed_dir, filename)
        
        # Sauvegarder le fichier
        np.savetxt(full_path, cycles, delimiter=',')
        print(f"Cycles sauvegardés dans {full_path}")
        return full_path