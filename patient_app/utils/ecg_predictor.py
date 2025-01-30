import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')  # Ajoutez cette ligne AVANT d'importer pyplot
import matplotlib.pyplot as plt
import io
import seaborn as sns
from django.conf import settings
from sklearn.preprocessing import StandardScaler
from joblib import load

class ECGPredictor:
    def __init__(self, model_path=None, scaler_path=None):
        """
        Initialise le prédicteur ECG avec le modèle et le scaler.
        """
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
        try:
            # Utilisation du style médical ggplot
            plt.style.use('ggplot')
            fig, ax = plt.subplots(figsize=(12, 6))

            # Nombre de cycles normaux et anormaux
            n_normaux = sum(1 for x in classifications if x == 0)
            n_anormaux = sum(1 for x in classifications if x == 1)

            # Génération sûre des couleurs
            cold_colors = plt.cm.Blues(np.linspace(0.3, 0.8, max(n_normaux, 1)))
            warm_colors = plt.cm.Reds(np.linspace(0.3, 0.8, max(n_anormaux, 1)))
            
            cold_index, warm_index = 0, 0  # Indices pour suivre les couleurs
            cycle_labels = []  # Pour stocker les labels des cycles

            for i, cycle in enumerate(cycles):
                if classifications[i] == 0:
                    color = cold_colors[min(cold_index, len(cold_colors)-1)]
                    cold_index += 1
                else:
                    color = warm_colors[min(warm_index, len(warm_colors)-1)]
                    warm_index += 1

                # Ajouter chaque cycle avec sa couleur dans la légende
                label = f"Cycle {i+1}"
                cycle_labels.append((label, color))  # Stocke pour la légende complète
                
                ax.plot(cycle, alpha=0.85, color=color, linewidth=1.5, label=label)

            # Paramètres visuels
            ax.set_title("Superposition des cycles ECG", fontsize=14, fontweight='bold')
            ax.set_xlabel("Échantillons", fontsize=12)
            ax.set_ylabel("Amplitude normalisée", fontsize=12)
            ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)

            # Légende principale
            legend_labels = {
                "Cycles Normaux": 'blue',
                "Cycles Anormaux": 'red'
            }
            handles = [plt.Line2D([0], [0], color=color, linewidth=2, label=label) 
                    for label, color in legend_labels.items()]
            classification_legend = ax.legend(handles=handles, loc="lower right", fontsize=10, frameon=True)

            # Légende des cycles
            handles = [plt.Line2D([0], [0], color=color, linewidth=1.5, label=label) 
                    for label, color in cycle_labels]
            cycle_legend = ax.legend(handles=handles, loc="upper right", fontsize=8, 
                                    frameon=True, ncol=3, title="Cycles détaillés")

            # Ajouter la légende de classification après celle des cycles
            ax.add_artist(cycle_legend)
            ax.add_artist(classification_legend)

            # Sauvegarde du plot dans un buffer mémoire
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=200, bbox_inches='tight')
            plt.close(fig)
            buffer.seek(0)

            return buffer.getvalue()  # Retourne l'image encodée en bytes

        except Exception as e:
            print(f"Erreur détaillée dans generate_plots : {e}")
            import traceback
            traceback.print_exc()
            return None

    def calculate_confidence(self, probas_malade):
        """
        Calcule un score de confiance basé sur les probabilités du modèle.
        """
        certainty_scores = [abs(p - 0.5) * 2 for p in probas_malade]  # Transforme [0-1] en score de certitude
        confidence_score = np.mean(certainty_scores)  # Moyenne des scores de certitude
        return confidence_score

    
    def analyze_personal_ecg(self, cycles):

        """
        Analyse complète des cycles ECG avec visualisations et métriques détaillées
        """
        # Debug avant tout traitement
        print("Debug - Données brutes:")
        print(f"Shape des cycles: {cycles.shape}")
        print(f"Amplitude min-max brute: {np.min(cycles):.3f} - {np.max(cycles):.3f}")

        # Normalisation manuelle cycle par cycle
        cycles_normalises = np.zeros_like(cycles)
        for i in range(len(cycles)):
            cycle = cycles[i]
            min_val = np.min(cycle)
            max_val = np.max(cycle)
            if max_val - min_val > 0:  # Éviter division par zéro
                cycles_normalises[i] = (cycle - min_val) / (max_val - min_val)

        # Debug après normalisation
        print("\nDebug - Après normalisation manuelle:")
        print(f"Shape des cycles normalisés: {cycles_normalises.shape}")
        print(f"Amplitude min-max normalisée: {np.min(cycles_normalises):.3f} - {np.max(cycles_normalises):.3f}")

        # Préparation pour le modèle
        X = cycles_normalises.reshape(-1, 182, 1)
        
        # Prédictions
        predictions = self.model.predict(X)
        probas_malade = predictions[:, 1]
        classifications = (probas_malade >= self.optimal_threshold).astype(int)

        # Debug des prédictions
        print("\nDebug - Prédictions par cycle:")
        for i, (proba, classif) in enumerate(zip(probas_malade, classifications)):
            print(f"Cycle {i+1:2d}: Proba malade = {proba:.3f}, Classification = {'MALADE' if classif else 'SAIN'}")
        
        # Statistiques
        n_sains = np.sum(classifications == 0)
        n_malades = np.sum(classifications == 1)
        ratio_malades = n_malades / len(cycles)

        #Score de confiance
        confidence_score = self.calculate_confidence(probas_malade)

        # Génération des plots - Correction ici
        plots_data = self.generate_plots(cycles, probas_malade, classifications)
        
        # Formatage des résultats par cycle
        cycles_details = []
        for i, (proba, classification) in enumerate(zip(probas_malade, classifications)):
            cycles_details.append({
                'cycle_num': i + 1,
                'probability': float(proba),
                'classification': 'Anomalie' if classification == 1 else 'Normal',
                'classification_color': 'red' if classification == 1 else 'green'
            })
        
        # Détermination du niveau de risque et de l'interprétation
        if ratio_malades == 0 :
            risk_level = 'LOW'
            conclusion = "ECG NORMAL"
            interpretation = f"""Analyse détaillée :
    - Total des cycles analysés : {len(cycles)}
    - Cycles normaux : {n_sains}
    - Cycles anormaux : {n_malades}
    - Ratio d'anomalies : {ratio_malades*100:.1f}%

Conclusion :
L'analyse n'a révélé aucune anomalie significative. L'ECG présente un rythme régulier avec des cycles normaux.

Recommandation : 
Aucune action médicale urgente n'est requise."""

        elif 0 < ratio_malades < 0.3 :
            risk_level = 'MEDIUM'
            conclusion = "ECG À CONTRÔLER"
            interpretation = f"""Analyse détaillée :
    - Total des cycles analysés : {len(cycles)}
    - Cycles normaux : {n_sains}
    - Cycles anormaux : {n_malades}
    - Ratio d'anomalies : {ratio_malades*100:.1f}%

Conclusion :
Quelques irrégularités ont été détectées dans le rythme cardiaque. Bien que non critiques, ces anomalies méritent attention.

Recommandation : 
Une consultation médicale de contrôle est recommandée pour évaluer ces irrégularités."""

        else:
            risk_level = 'HIGH'
            conclusion = "ECG À CONTRÔLER D'URGENCE"
            interpretation = f"""Analyse détaillée :
    - Total des cycles analysés : {len(cycles)}
    - Cycles normaux : {n_sains}
    - Cycles anormaux : {n_malades}
    - Ratio d'anomalies : {ratio_malades*100:.1f}%

Conclusion :
Des anomalies significatives ont été détectées dans le rythme cardiaque. Le nombre de cycles anormaux est important.

RECOMMANDATION URGENTE : 
Une consultation médicale rapide est nécessaire pour évaluer ces anomalies."""

        # Générer les plots AVANT de créer le dictionnaire results
        plots_data = self.generate_plots(cycles, probas_malade, classifications)

        # Préparation des résultats
        results = {
            'plots': plots_data,  # Utiliser plots_data ici
            'risk_level': risk_level,
            'conclusion': conclusion,
            'confidence_score': confidence_score,
            'interpretation': interpretation,
            'probabilites': probas_malade.tolist(),
            'classifications': classifications.tolist(),
            'cycles_details': cycles_details,
            'stats': {
                'normal_cycles': int(n_sains),
                'abnormal_cycles': int(n_malades),
                'total_cycles': len(cycles),
                'abnormal_ratio': float(ratio_malades)
            }
        }
        
        return results

    def get_model_summary(self):
        """
        Retourne un résumé du modèle.
        """
        if hasattr(self, 'model'):
            return self.model.summary()
        return "Aucun modèle n'a été chargé."