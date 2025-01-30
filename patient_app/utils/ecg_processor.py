import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, butter, filtfilt
from datetime import datetime
import os
from django.conf import settings


class ECGProcessor:
    def __init__(self):
        self.sampling_freq = 120.0 
        self.target_length = 182
        self.before_r = 20    
        self.after_r = 20    
        self.peak_params = {
        'height': 0.4,
        'distance': 30,    # Distance minimale entre pics (~60ms à 500Hz)
        'prominence': 0.3
        }
        self.r_peak_params = {
            'height': 0.6,
            'distance': 30,
            'prominence': 0.4
        }

    def load_data(self, file_path):
        """Charge et valide les données ECG depuis un fichier CSV."""
        try:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Le fichier {file_path} n'existe pas")
            
            data = pd.read_csv(file_path)
            print("\n=== Debug chargement données ===")
            print("Colonnes disponibles:", data.columns.tolist())
            
            required_columns = ['Timestamp', 'ECG_Value']
            missing_columns = [col for col in required_columns if col not in data.columns]
            if missing_columns:
                raise ValueError(f"Colonnes manquantes dans le CSV: {missing_columns}")
            
            signal = data['ECG_Value'].values
            if len(signal) == 0:
                raise ValueError("Le signal ECG est vide")
                
            print("Aperçu des premières valeurs:")
            print(data.head())
            print(f"Signal extrait : {len(signal)} points")
            print(f"Valeurs min/max : {np.min(signal):.2f}/{np.max(signal):.2f}")
            
            return signal
            
        except pd.errors.EmptyDataError:
            raise ValueError("Le fichier CSV est vide")
        except pd.errors.ParserError as e:
            raise ValueError(f"Erreur de parsing du CSV: {str(e)}")
        except Exception as e:
            print(f"Erreur inattendue lors du chargement: {str(e)}")
            raise

    def filter_signal(self, signal):
        """Filtre le signal avec correction de la ligne de base."""
        try:
            # Paramètres du filtre
            nyquist = 0.5 * self.sampling_freq
            low = 0.5 / nyquist
            high = 40.0 / nyquist
            order = 2

            # Filtrage passe-bande
            b, a = butter(order, [low, high], btype='band')
            filtered = filtfilt(b, a, signal)
            
            # Correction de la ligne de base
            window = 50
            baseline = np.convolve(filtered, np.ones(window)/window, mode='same')
            corrected = filtered - baseline
            
            return corrected
            
        except Exception as e:
            print(f"Erreur lors du filtrage: {str(e)}")
            raise

    def detect_peaks(self, signal, is_normalized=False):
        """Méthode unifiée de détection des pics R"""
        try:
            if not is_normalized:
                signal = (signal - np.min(signal)) / (np.max(signal) - np.min(signal))
            
            peaks, properties = find_peaks(signal,
                                         height=self.peak_params['height'],
                                         distance=self.peak_params['distance'],
                                         prominence=self.peak_params['prominence'])
            
            return peaks, properties
            
        except Exception as e:
            print(f"Erreur dans detect_peaks: {str(e)}")
            raise

    def find_r_peaks(self, signal, cycle_length=None):
        """Détecte spécifiquement les pics R dans le signal."""
        try:
            filtered_signal = self.filter_signal(signal)
            normalized_signal = (filtered_signal - np.min(filtered_signal)) / (np.max(filtered_signal) - np.min(filtered_signal))
            
            if cycle_length is not None and isinstance(cycle_length, (int, float)):
                self.r_peak_params['distance'] = int(0.8 * float(cycle_length))
            else:
                print("Attention : cycle_length non valide, utilisation de la distance par défaut")
            
            peaks, properties = find_peaks(normalized_signal,
                                         height=self.r_peak_params['height'],
                                         distance=self.r_peak_params['distance'],
                                         prominence=self.r_peak_params['prominence'])
            
            print(f"Pics R détectés : {len(peaks)}")
            if properties:
                print(f"Propriétés des pics : {properties}")
                
            return peaks
            
        except Exception as e:
            print(f"Erreur dans find_r_peaks : {e}")
            raise

    def analyze_cycle_distance(self, signal):
        """Analyse la distance entre les cycles et retourne la distance moyenne."""
        try:
            filtered_signal = self.filter_signal(signal)
            normalized_signal = (filtered_signal - np.min(filtered_signal)) / (np.max(filtered_signal) - np.min(filtered_signal))
            
            peaks, properties = self.detect_peaks(normalized_signal, is_normalized=True)
            
            self._create_debug_plots(signal, filtered_signal, normalized_signal, peaks)
            stats = self._calculate_statistics(peaks)
            
            # On retourne uniquement la distance moyenne pour la compatibilité
            return stats['mean_distance'] if stats['mean_distance'] is not None else None
            
        except Exception as e:
            print(f"Erreur dans analyze_cycle_distance : {e}")
            raise

    def _create_debug_plots(self, raw_signal, filtered_signal, normalized_signal, peaks):
        """Méthode séparée pour la création des visualisations"""
        try:
            plt.figure(figsize=(15, 10))
            
            plt.subplot(4, 1, 1)
            plt.plot(raw_signal)
            plt.title("Signal brut Arduino")
            plt.grid(True)

            plt.subplot(4, 1, 2)
            plt.plot(filtered_signal)
            plt.title("Signal filtré")
            plt.grid(True)

            plt.subplot(4, 1, 3)
            plt.plot(normalized_signal)
            plt.title("Signal normalisé")
            plt.grid(True)

            plt.subplot(4, 1, 4)
            plt.plot(normalized_signal)
            plt.plot(peaks, normalized_signal[peaks], "rx")
            plt.title(f"Détection des pics ({len(peaks)} pics)")
            plt.grid(True)

            plt.tight_layout()
            debug_path = os.path.join(settings.MEDIA_ROOT, 'debug')
            os.makedirs(debug_path, exist_ok=True)
            plt.savefig(os.path.join(debug_path, 'debug_arduino.png'))
            plt.close()
            
        except Exception as e:
            print(f"Erreur lors de la création des plots: {str(e)}")
            raise

    def _calculate_statistics(self, peaks):
        """Méthode séparée pour le calcul des statistiques"""
        stats = {
            'total_peaks': len(peaks),
            'mean_distance': None,
            'std_distance': None,
            'bpm': None
        }
        
        if len(peaks) >= 2:
            distances = np.diff(peaks)
            stats['mean_distance'] = float(np.mean(distances))  # Conversion explicite en float
            stats['std_distance'] = float(np.std(distances))    # Conversion explicite en float
            stats['bpm'] = 60 * self.sampling_freq / stats['mean_distance']
            
            print(f"\nStatistiques des cycles :")
            print(f"Pics détectés : {stats['total_peaks']}")
            print(f"Distance moyenne : {stats['mean_distance']:.2f} ± {stats['std_distance']:.2f}")
            print(f"Fréquence cardiaque : {stats['bpm']:.1f} BPM")
            
        return stats

    def extract_cycles(self, signal, peaks):
        """Extrait et normalise les cycles autour des pics R"""
        try:
            cycles = []
            valid_peaks = []
            
            print(f"Total des pics R : {len(peaks)}")
            print(f"Longueur du signal : {len(signal)}")
            
            for peak in peaks:
                if peak - self.before_r >= 0 and peak + self.after_r < len(signal):
                    cycle = signal[peak - self.before_r:peak + self.after_r]
                    
                    cycle_min = np.min(cycle)
                    cycle_max = np.max(cycle)
                    cycle_norm = (cycle - cycle_min) / (cycle_max - cycle_min)
                    
                    x_original = np.arange(len(cycle_norm))
                    x_new = np.linspace(0, len(cycle_norm)-1, self.target_length)
                    cycle_resampled = np.interp(x_new, x_original, cycle_norm)
                    
                    cycles.append(cycle_resampled)
                    valid_peaks.append(peak)
                else:
                    print(f"Pic {peak} ignoré : hors limites")
            
            if not cycles:
                raise ValueError("Aucun cycle valide n'a pu être extrait")
                
            print(f"Cycles valides extraits : {len(cycles)}")
            return np.array(cycles), np.array(valid_peaks)
            
        except Exception as e:
            print(f"Erreur lors de l'extraction des cycles : {str(e)}")
            raise

    def save_cycles(self, cycles, original_filename):
            """Sauvegarde les cycles traités dans un fichier CSV.
            
            Args:
                cycles (numpy.ndarray): Tableau des cycles traités
                original_filename (str): Nom du fichier original
                
            Returns:
                str: Chemin du fichier sauvegardé
            """
            try:
                # Création du répertoire de sortie
                processed_dir = os.path.join(settings.MEDIA_ROOT, 'processed_ecg')
                os.makedirs(processed_dir, exist_ok=True)
                
                # Génération du nom de fichier avec timestamp
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"processed_{timestamp}_{os.path.splitext(original_filename)[0]}.csv"
                full_path = os.path.join(processed_dir, filename)
                
                # Debug avant sauvegarde
                print(f"Sauvegarde des cycles : shape={cycles.shape}")
                print(f"Valeurs min/max : {np.min(cycles)}, {np.max(cycles)}")
                
                # Sauvegarde des cycles
                np.savetxt(full_path, cycles, delimiter=',')
                print(f"Cycles sauvegardés dans {full_path}")
                
                return full_path
                
            except Exception as e:
                print(f"Erreur lors de la sauvegarde des cycles : {str(e)}")
                raise