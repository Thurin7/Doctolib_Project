import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')  # Ajoutez cette ligne AVANT d'importer pyplot
import matplotlib.pyplot as plt
import io
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
            plt.figure(figsize=(20, 15))
            
            # Superposition des cycles avec classification
            plt.subplot(2, 2, 1)
            for i in range(len(cycles)):
                color = 'red' if classifications[i] == 1 else 'blue'
                plt.plot(cycles[i], alpha=0.3, color=color)
            plt.title('Classification des cycles (rouge=anomalie, bleu=normal)')
            plt.grid(True)
            
            # Distribution des probabilités
            plt.subplot(2, 2, 2)
            plt.hist(probas_malade, bins=20, color='blue', alpha=0.7)
            plt.axvline(x=self.optimal_threshold, color='r', linestyle='--', 
                    label=f'Seuil ({self.optimal_threshold})')
            plt.title('Distribution des probabilités d\'anomalie')
            plt.legend()
            
            # Évolution des probabilités
            plt.subplot(2, 2, 3)
            plt.plot(probas_malade, 'bo-', label='Probabilité')
            plt.axhline(y=self.optimal_threshold, color='r', linestyle='--', label='Seuil')
            plt.fill_between(range(len(probas_malade)), probas_malade, self.optimal_threshold,
                            where=(probas_malade >= self.optimal_threshold),
                            color='red', alpha=0.3, label='Zones anormales')
            plt.title('Probabilités par cycle')
            plt.legend()
            
            # Sauvegarder le plot
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
            plt.close('all')
            buffer.seek(0)
            
            # Convertir en bytes et retourner
            return buffer.getvalue()

        except Exception as e:
            print(f"Erreur lors de la génération des plots : {e}")
            return None


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
        predictions = self.model.predict(X, verbose=0)
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
        if ratio_malades < 0.1:
            risk_level = 'LOW'
            conclusion = "ECG NORMAL"
            confidence_score = 0.9
            interpretation = f"""Analyse détaillée :
    - Total des cycles analysés : {len(cycles)}
    - Cycles normaux : {n_sains}
    - Cycles anormaux : {n_malades}
    - Ratio d'anomalies : {ratio_malades*100:.1f}%

    Conclusion :
    L'analyse n'a révélé aucune anomalie significative. L'ECG présente un rythme régulier avec des cycles normaux.

    Recommandation : 
    Aucune action médicale urgente n'est requise."""

        elif ratio_malades < 0.3:
            risk_level = 'MEDIUM'
            conclusion = "ECG À CONTRÔLER"
            confidence_score = 0.7
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
            confidence_score = 0.5
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