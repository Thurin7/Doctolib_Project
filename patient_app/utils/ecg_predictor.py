import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import seaborn as sns
from django.conf import settings
from sklearn.preprocessing import StandardScaler
from joblib import load

class ECGPredictor:
    def __init__(self, model_path=None, scaler_path=None):
        """Initialise le prédicteur ECG avec le modèle et le scaler."""
        try:
            # Charger le modèle
            self.model = tf.keras.models.load_model(model_path)
            print("Architecture du modèle:")
            self.model.summary()
            self.model.compile(
                optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
            
            # Charger le scaler
            self.scaler = load(scaler_path)
            print("\nType de scaler:", type(self.scaler))
            
            # Seuil optimal calculé pendant l'entraînement
            self.optimal_threshold = 0.038
            
            print("Modèle et scaler chargés avec succès.")
        
        except Exception as e:
            print(f"Erreur lors du chargement : {e}")
            raise

    def generate_plots(self, cycles, probas_malade, classifications):
        """Génère les visualisations des cycles ECG."""
        try:
            plt.style.use('ggplot')
            fig, ax = plt.subplots(figsize=(12, 6))

            n_normaux = sum(1 for x in classifications if x == 0)
            n_anormaux = sum(1 for x in classifications if x == 1)

            # Génération sûre des couleurs
            cold_colors = plt.cm.Blues(np.linspace(0.3, 0.8, max(n_normaux, 1)))
            warm_colors = plt.cm.Reds(np.linspace(0.3, 0.8, max(n_anormaux, 1)))
            
            cold_index, warm_index = 0, 0
            cycle_labels = []

            for i, cycle in enumerate(cycles):
                if classifications[i] == 0:
                    color = cold_colors[min(cold_index, len(cold_colors)-1)]
                    cold_index += 1
                else:
                    color = warm_colors[min(warm_index, len(warm_colors)-1)]
                    warm_index += 1

                label = f"Cycle {i+1}"
                cycle_labels.append((label, color))
                ax.plot(cycle, alpha=0.85, color=color, linewidth=1.5, label=label)

            ax.set_title("Superposition des cycles ECG", fontsize=14, fontweight='bold')
            ax.set_xlabel("Échantillons", fontsize=12)
            ax.set_ylabel("Amplitude normalisée", fontsize=12)
            ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)

            # Légendes
            legend_labels = {
                "Cycles Normaux": 'blue',
                "Cycles Anormaux": 'red'
            }
            handles = [plt.Line2D([0], [0], color=color, linewidth=2, label=label) 
                    for label, color in legend_labels.items()]
            classification_legend = ax.legend(handles=handles, loc="lower right",
                                           fontsize=10, frameon=True)

            handles = [plt.Line2D([0], [0], color=color, linewidth=1.5, label=label) 
                    for label, color in cycle_labels]
            cycle_legend = ax.legend(handles=handles, loc="upper right", fontsize=8, 
                                   frameon=True, ncol=3, title="Cycles détaillés")

            ax.add_artist(cycle_legend)
            ax.add_artist(classification_legend)

            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=200, bbox_inches='tight')
            plt.close(fig)
            buffer.seek(0)

            return buffer.getvalue()

        except Exception as e:
            print(f"Erreur détaillée dans generate_plots : {e}")
            import traceback
            traceback.print_exc()
            return None

    def calculate_confidence(self, probas_malade):
        """Calcule un score de confiance basé sur les probabilités."""
        certainty_scores = [abs(p - 0.5) * 2 for p in probas_malade]
        return np.mean(certainty_scores)

    def _generate_interpretation(self, stats, risk_type):
        """Génère l'interprétation basée sur les statistiques et le niveau de risque."""
        base_stats = f"""Analyse détaillée :
    - Total des cycles analysés : {stats['total_cycles']}
    - Cycles normaux : {stats['normal_cycles']}
    - Cycles anormaux : {stats['abnormal_cycles']}
    - Ratio d'anomalies : {stats['abnormal_ratio']*100:.1f}%\n"""

        if risk_type == 'normal':
            return base_stats + """
Conclusion :
L'analyse n'a révélé aucune anomalie significative. L'ECG présente un rythme régulier avec des cycles normaux.

Recommandation : 
Aucune action médicale urgente n'est requise."""
        
        elif risk_type == 'attention':
            return base_stats + """
Conclusion :
Quelques irrégularités ont été détectées dans le rythme cardiaque. Bien que non critiques, ces anomalies méritent attention.

Recommandation : 
Une consultation médicale de contrôle est recommandée pour évaluer ces irrégularités."""
        
        else:  # urgent
            return base_stats + """
Conclusion :
Des anomalies significatives ont été détectées dans le rythme cardiaque. Le nombre de cycles anormaux est important.

RECOMMANDATION URGENTE : 
Une consultation médicale rapide est nécessaire pour évaluer ces anomalies."""

    def analyze_personal_ecg(self, cycles):
        """Analyse complète des cycles ECG avec validation et métriques détaillées."""
        try:
            # 1. Validation des données d'entrée
            if not isinstance(cycles, np.ndarray):
                raise ValueError("Les cycles doivent être un tableau numpy")
            if cycles.shape[1] != 182:
                raise ValueError(f"Chaque cycle doit avoir 182 points (reçu: {cycles.shape[1]})")
            
            print("Debug - Données brutes:")
            print(f"Shape des cycles: {cycles.shape}")
            print(f"Amplitude min-max brute: {np.min(cycles):.3f} - {np.max(cycles):.3f}")

            # 2. Normalisation avec validation
            cycles_normalises = np.zeros_like(cycles)
            invalid_cycles = []
            
            for i, cycle in enumerate(cycles):
                amplitude = np.max(cycle) - np.min(cycle)
                if amplitude > 0:
                    cycles_normalises[i] = (cycle - np.min(cycle)) / amplitude
                else:
                    invalid_cycles.append(i)
                    print(f"Attention: Cycle {i} invalide (amplitude nulle)")

            if invalid_cycles:
                print(f"Attention: {len(invalid_cycles)} cycles invalides détectés")
                valid_mask = ~np.isin(np.arange(len(cycles)), invalid_cycles)
                cycles_normalises = cycles_normalises[valid_mask]

            print("\nDebug - Après normalisation:")
            print(f"Shape des cycles normalisés: {cycles_normalises.shape}")
            print(f"Amplitude min-max normalisée: {np.min(cycles_normalises):.3f} - {np.max(cycles_normalises):.3f}")

            # 3. Préparation pour le modèle
            X = cycles_normalises.reshape(-1, 182, 1)
            if np.any(np.isnan(X)):
                raise ValueError("Valeurs NaN détectées après préparation")

            # 4. Prédictions
            predictions = self.model.predict(X, verbose=0)
            probas_malade = predictions[:, 1]
            classifications = (probas_malade >= self.optimal_threshold).astype(int)

            # Debug des prédictions
            print("\nDebug - Prédictions par cycle:")
            for i, (proba, classif) in enumerate(zip(probas_malade, classifications)):
                print(f"Cycle {i+1:2d}: Proba malade = {proba:.3f}, Classification = {'MALADE' if classif else 'SAIN'}")

            # 5. Calcul des statistiques
            stats = self._calculate_detailed_stats(probas_malade, classifications)
            
            # 6. Évaluation du risque
            risk_assessment = self._assess_risk_level(stats)

            # 7. Génération des visualisations
            plots_data = self.generate_plots(cycles_normalises, probas_malade, classifications)

            # 8. Préparation des résultats détaillés
            cycles_details = []
            for i, (proba, classification) in enumerate(zip(probas_malade, classifications)):
                cycles_details.append({
                    'cycle_num': i + 1,
                    'probability': float(proba),
                    'classification': 'Anomalie' if classification == 1 else 'Normal',
                    'classification_color': 'red' if classification == 1 else 'green'
                })

            results = {
                'plots': plots_data,
                'risk_level': risk_assessment['risk_level'],
                'conclusion': risk_assessment['conclusion'],
                'confidence_score': self.calculate_confidence(probas_malade),
                'interpretation': risk_assessment['interpretation'],
                'cycles_details': cycles_details,
                'stats': stats,
                'warnings': [f"Cycle invalide #{i}" for i in invalid_cycles] if invalid_cycles else []
            }

            return results

        except Exception as e:
            print(f"Erreur dans analyze_personal_ecg : {e}")
            raise

    def _calculate_detailed_stats(self, probas_malade, classifications):
        """Calcule des statistiques détaillées sur les prédictions."""
        n_sains = np.sum(classifications == 0)
        n_malades = np.sum(classifications == 1)
        
        return {
            'normal_cycles': int(n_sains),
            'abnormal_cycles': int(n_malades),
            'total_cycles': len(classifications),
            'abnormal_ratio': float(n_malades / len(classifications)),
            'mean_probability': float(np.mean(probas_malade)),
            'max_probability': float(np.max(probas_malade)),
            'consecutive_abnormal': self._count_consecutive_abnormal(classifications)
        }

    def _assess_risk_level(self, stats):
        ratio = stats['abnormal_ratio']
        consecutive_abnormal = stats['consecutive_abnormal']
        max_proba = stats['max_probability']
        
        # Risque élevé uniquement si plusieurs conditions sont remplies
        if (ratio > 0.3 and max_proba >= 0.7) or consecutive_abnormal >= 2:
            return {
                'risk_level': 'HIGH',
                'conclusion': 'ECG À CONTRÔLER D\'URGENCE',
                'interpretation': self._generate_interpretation(stats, 'urgent')
            }
        # Risque moyen pour des anomalies isolées mais significatives
        elif ratio > 0.1 or max_proba >= 0.7:
            return {
                'risk_level': 'MEDIUM',
                'conclusion': 'ECG À CONTRÔLER',
                'interpretation': self._generate_interpretation(stats, 'attention')
            }
        else:
            return {
                'risk_level': 'LOW',
                'conclusion': 'ECG NORMAL',
                'interpretation': self._generate_interpretation(stats, 'normal')
            }

    def _count_consecutive_abnormal(self, classifications):
        """Compte le nombre maximum de cycles anormaux consécutifs."""
        max_consecutive = current_consecutive = 0
        for c in classifications:
            if c == 1:
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 0
        return max_consecutive

    def get_model_summary(self):
        """Retourne un résumé du modèle."""
        if hasattr(self, 'model'):
            return self.model.summary()
        return "Aucun modèle n'a été chargé."