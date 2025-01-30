import pandas as pd
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, butter, filtfilt

from scipy.signal import find_peaks, savgol_filter
from datetime import datetime
from django.views.generic import TemplateView
from django.contrib.auth.mixins import LoginRequiredMixin
import os
from django.conf import settings

class ECGProcessor:
    def __init__(self):
        # Paramètres globaux pour l'analyse ECG
        self.sampling_freq = 500.0  # Fréquence d'échantillonnage en Hz

    def load_data(self, file_path):
        """
        Charge les données ECG à partir d'un fichier CSV.
        """
        try:
            # Charger avec pandas pour voir la structure
            data = pd.read_csv(file_path)
            print("\n=== Debug chargement données ===")
            print("Colonnes disponibles:", data.columns.tolist())
            print("Aperçu des premières valeurs:")
            print(data.head())
            
            # Prendre la colonne des valeurs ECG
            signal = data.iloc[:, 1].values  # Deuxième colonne
            
            print(f"Signal extrait : {len(signal)} points")
            print(f"Valeurs min/max : {np.min(signal):.2f}/{np.max(signal):.2f}")
            
            return signal
            
        except Exception as e:
            print(f"Erreur de chargement : {e}")
            raise

    def filter_signal(self, signal):
        """
        Filtre amélioré avec correction de la ligne de base
        """
        # Utiliser la fréquence d'échantillonnage de la classe
        freq_ech = self.sampling_freq
        lowcut = 0.5     
        highcut = 40.0    
        order = 2        

        nyquist = 0.5 * freq_ech
        low = lowcut / nyquist
        high = highcut / nyquist

        b, a = butter(order, [low, high], btype='band')
        filtered = filtfilt(b, a, signal)
        
        # Correction de la ligne de base
        window = 50  # Taille de la fenêtre pour la moyenne mobile
        baseline = np.convolve(filtered, np.ones(window)/window, mode='same')
        corrected = filtered - baseline
        
        return corrected

    def analyze_cycle_distance(self, signal):
        """Analyse adaptée aux données Arduino"""
        try:
            print("\n=== Debug détection des cycles ===")
            print(f"1. Signal d'entrée : shape={signal.shape}, type={type(signal)}")
            
            plt.figure(figsize=(15, 10))
            
            # Signal brut
            plt.subplot(4, 1, 1)
            try:
                plt.plot(signal)
                print("2. Plot du signal brut réussi")
            except Exception as e:
                print(f"Erreur plot signal brut : {e}")
            plt.title("Signal brut Arduino")
            plt.grid(True)

            # Signal filtré
            try:
                filtered_signal = self.filter_signal(signal)
                print("3. Filtrage réussi")
                plt.subplot(4, 1, 2)
                plt.plot(filtered_signal)
                print("4. Plot du signal filtré réussi")
            except Exception as e:
                print(f"Erreur filtrage/plot : {e}")
            plt.title("Signal filtré")
            plt.grid(True)

            # Signal normalisé
            try:
                normalized_signal = (filtered_signal - np.min(filtered_signal)) / (np.max(filtered_signal) - np.min(filtered_signal))
                print("5. Normalisation réussie")
                plt.subplot(4, 1, 3)
                plt.plot(normalized_signal)
                print("6. Plot du signal normalisé réussi")
            except Exception as e:
                print(f"Erreur normalisation/plot : {e}")
            plt.title("Signal normalisé")
            plt.grid(True)

            # Détection adaptative des pics
            try:
                peaks, _ = find_peaks(normalized_signal, 
                                    height=0.4,
                                    distance=30,
                                    prominence=0.3)
                print(f"7. Détection des pics réussie : {len(peaks)} pics trouvés")
            except Exception as e:
                print(f"Erreur détection pics : {e}")

            plt.subplot(4, 1, 4)
            try:
                plt.plot(normalized_signal)
                plt.plot(peaks, normalized_signal[peaks], "rx")
                print("8. Plot final réussi")
            except Exception as e:
                print(f"Erreur plot final : {e}")
            plt.title(f"Détection des pics")
            plt.legend([f"Pics R détectés: {len(peaks)}"])
            plt.grid(True)

            plt.tight_layout()
            
            # Sauvegarder pour debug avec plus de détails sur le chemin
            try:
                debug_path = os.path.join(settings.MEDIA_ROOT, 'debug')
                os.makedirs(debug_path, exist_ok=True)
                debug_file = os.path.join(debug_path, 'debug_arduino.png')
                print(f"9. Tentative de sauvegarde dans : {debug_file}")
                plt.savefig(debug_file)
                print("10. Sauvegarde réussie")
            except Exception as e:
                print(f"Erreur sauvegarde : {e}")
            finally:
                plt.close()

            if len(peaks) >= 2:
                distances = np.diff(peaks)
                mean_distance = np.mean(distances)
                std_distance = np.std(distances)
                print(f"\nStatistiques des cycles détectés:")
                print(f"Nombre de pics : {len(peaks)}")
                print(f"Distance moyenne : {mean_distance:.2f} ± {std_distance:.2f}")
                print(f"Fréquence cardiaque estimée : {60 * self.sampling_freq / mean_distance:.1f} BPM")
                return mean_distance
            else:
                print("Aucun cycle détecté!")
                return None

        except Exception as e:
            print(f"Erreur générale dans analyze_cycle_distance : {e}")
            raise

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
    
    def save_cycles(self, cycles, original_filename):
        # Créer le dossier processed_ecg s'il n'existe pas
        processed_dir = os.path.join(settings.MEDIA_ROOT, 'processed_ecg')
        os.makedirs(processed_dir, exist_ok=True)
        
        # Générer un nom de fichier unique avec timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"processed_{timestamp}_{os.path.splitext(original_filename)[0]}.csv"
        
        # Chemin complet du fichier
        full_path = os.path.join(processed_dir, filename)
        
        # Sauvegarder le fichier
        np.savetxt(full_path, cycles, delimiter=',')
        print(f"Cycles traités sauvegardés dans {full_path}")
        return full_path

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
    
    def save_cycles(self, cycles, original_filename):
        # Créer le dossier processed_ecg s'il n'existe pas
        processed_dir = os.path.join(settings.MEDIA_ROOT, 'processed_ecg')
        os.makedirs(processed_dir, exist_ok=True)
        
        # Générer un nom de fichier unique avec timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"processed_{timestamp}_{os.path.splitext(original_filename)[0]}.csv"
        
        # Chemin complet du fichier
        full_path = os.path.join(processed_dir, filename)
        
        # Sauvegarder le fichier
        np.savetxt(full_path, cycles, delimiter=',')
        print(f"Cycles traités sauvegardés dans {full_path}")
        return full_path