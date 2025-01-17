from django.shortcuts import render
from django.views.generic.edit import FormView
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
from django.conf import settings
import os
from .forms import ECGUploadForm
from .utils.ecg_processor import ECGProcessor
from .models import ECG

class ECGUploadView(FormView):
    template_name = 'patient_app/upload.html'
    form_class = ECGUploadForm
    success_url = '/upload/success/'  # Nous changerons cela plus tard

    def form_valid(self, form):
        # Récupérer le fichier uploadé
        file = form.cleaned_data['ecg_file']
        
        # Sauvegarder temporairement le fichier
        path = default_storage.save(f'tmp/{file.name}', ContentFile(file.read()))
        tmp_file = os.path.join(settings.MEDIA_ROOT, path)
        
        try:
            # Traiter l'ECG
            processor = ECGProcessor()
            signal = processor.load_data(tmp_file)
            
            # Analyser le cycle cardiaque
            cycle_length = processor.analyze_cycle_distance(signal)
            
            if cycle_length:
                # Trouver les pics R
                r_peaks = processor.find_r_peaks(signal, cycle_length)
                
                # Extraire les cycles
                cycles, valid_peaks = processor.extract_cycles(signal, r_peaks)
                
                # Sauvegarder les cycles
                cycles_file = processor.save_cycles(cycles)
                
                # Créer une entrée dans la base de données
                ecg = ECG.objects.create(
                    patient=self.request.user.patient_profile,  # Supposant que l'utilisateur est connecté
                    ecg_data=file.read(),
                    record_id=1,  # À gérer de manière appropriée
                    confidence_score=0.85,  # À calculer selon vos critères
                    interpretation="ECG traité avec succès",
                )
                
                # Nettoyer les fichiers temporaires
                default_storage.delete(path)
                os.remove(cycles_file)
                
                return super().form_valid(form)
            
        except Exception as e:
            # Gérer les erreurs
            print(f"Erreur lors du traitement de l'ECG: {str(e)}")
            default_storage.delete(path)
            return self.form_invalid(form)
            
        finally:
            # S'assurer que le fichier temporaire est supprimé
            if os.path.exists(tmp_file):
                os.remove(tmp_file)
        
        return super().form_valid(form)