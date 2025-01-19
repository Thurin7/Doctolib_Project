import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
from django.conf import settings
from sklearn.preprocessing import StandardScaler

class ECGPredictor:
    def __init__(self, model_path=None, scaler_path=None):
        """
        Initialise le prédicteur ECG avec le modèle et le scaler.
        
        :param model_path: Chemin vers le modèle
        :param scaler_path: Chemin vers le scaler pré-entraîné
        """
        # Charger le modèle
        self.model = tf.keras.models.load_model(model_path)
        
        # Charger le scaler
        from joblib import load
        self.scaler = load(scaler_path)
        
        # Seuil optimal calculé pendant l'entraînement
        self.optimal_threshold = 0.038  # À ajuster selon vos résultats Colab
        
        try:
            # Charger le modèle
            self.model = tf.keras.models.load_model(model_path)
            self.model.compile(
                optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
            
            # Charger le scaler
            from joblib import load
            self.scaler = load(scaler_path)
            
            print("Modèle et scaler chargés avec succès.")
        
        except Exception as e:
            print(f"Erreur lors du chargement : {e}")
            raise

    def analyze_personal_ecg(self, cycles, optimal_threshold=0.038):
        """
        Analyse des cycles ECG avec une classification progressive
        
        :param cycles: Tableau NumPy des cycles ECG
        :param optimal_threshold: Seuil de classification (défaut: 0.038)
        :return: Dictionnaire de résultats
        """
        # Normalisation avec le scaler
        cycles_normalises = self.scaler.transform(cycles)
        
        # Reshape pour le modèle
        X = cycles_normalises.reshape(-1, 182, 1)
        
        # Prédictions
        predictions = self.model.predict(X)
        probas_malade = predictions[:, 1]
        
        # Identification des cycles malades
        cycles_malades = probas_malade >= optimal_threshold
        n_cycles_malades = np.sum(cycles_malades)
        
        # Préparation des résultats
        results = {
            'conclusion': "",
            'confidence_score': 0,
            'interpretation': "",
            'stats': {
                'total_cycles': len(cycles),
                'cycles_malades': int(n_cycles_malades),
                'pct_cycles_malades': float(n_cycles_malades/len(cycles)*100)
            },
            'probabilites_cycles': probas_malade.tolist()
        }
        
        # Logique de classification
        if n_cycles_malades == 0:
            results['conclusion'] = "ECG SAIN"
            results['confidence_score'] = 100.0
            results['interpretation'] = f"""
            Analyse détaillée :
            - Total des cycles : {len(cycles)}
            - Tous les cycles semblent normaux
            
            Recommandation : Aucune action médicale requise
            """
        elif n_cycles_malades == 1:
            results['conclusion'] = "ECG À CONTRÔLER"
            results['confidence_score'] = 50.0
            results['interpretation'] = f"""
            Analyse détaillée :
            - Total des cycles : {len(cycles)}
            - Cycles potentiellement anormaux : 1
            
            Recommandation : Consultation médicale recommandée pour un contrôle
            """
        elif n_cycles_malades >= 3:
            results['conclusion'] = "ECG À CONTRÔLER D'URGENCE"
            results['confidence_score'] = 0.0
            results['interpretation'] = f"""
            Analyse détaillée :
            - Total des cycles : {len(cycles)}
            - Cycles potentiellement anormaux : {n_cycles_malades}
            
            RECOMMANDATION URGENTE : Consultation médicale immédiate requise
            """
        else:  # 2 cycles malades
            results['conclusion'] = "ECG À CONTRÔLER"
            results['confidence_score'] = 25.0
            results['interpretation'] = f"""
            Analyse détaillée :
            - Total des cycles : {len(cycles)}
            - Cycles potentiellement anormaux : 2
            
            Recommandation : Consultation médicale rapide recommandée
            """
        
        return results

    def get_model_summary(self):
        """
        Retourne un résumé du modèle.
        
        :return: Résumé du modèle
        """
        if hasattr(self, 'model'):
            return self.model.summary()
        else:
            print("Aucun modèle n'a été chargé.")
            return None

# Exemple d'utilisation
# predictor = ECGPredictor()
# probas, classifications = predictor.analyze_personal_ecg(cycles)