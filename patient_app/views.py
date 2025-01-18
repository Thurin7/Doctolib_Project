from django.shortcuts import render
from django.views.generic.base import TemplateView
from django.views.generic.edit import FormView
from django.contrib.auth.mixins import LoginRequiredMixin
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
from django.conf import settings
from django.urls import reverse_lazy
from django.contrib import messages
import os
from .forms import ECGUploadForm
from .utils.ecg_processor import ECGProcessor
from .models import ECG
from django.utils import timezone


class ECGUploadView(FormView):
    template_name = 'patient_app/upload.html'
    form_class = ECGUploadForm
    success_url = reverse_lazy('patient_app:upload_success')

    def form_valid(self, form):
        print("\n=== DÉBUT DU TRAITEMENT ===")
        file = form.cleaned_data['ecg_file']
        print(f"Fichier reçu: {file.name}")
        
        path = default_storage.save(f'tmp/{file.name}', ContentFile(file.read()))
        tmp_file = os.path.join(settings.MEDIA_ROOT, path)
        print(f"Fichier temporaire créé: {tmp_file}")
        
        try:
            print("\n=== TRAITEMENT ECG ===")
            processor = ECGProcessor()
            signal = processor.load_data(tmp_file)
            print(f"Signal chargé, longueur: {len(signal)}")
            
            cycle_length = processor.analyze_cycle_distance(signal)
            print(f"Cycle length calculé: {cycle_length}")
            
            if cycle_length:
                print("\n=== ANALYSE DES CYCLES ===")
                r_peaks = processor.find_r_peaks(signal, cycle_length)
                print(f"Nombre de pics R trouvés: {len(r_peaks)}")
                
                cycles, valid_peaks = processor.extract_cycles(signal, r_peaks)
                print(f"Nombre de cycles extraits: {len(cycles)}")
                
                cycles_file = processor.save_cycles(cycles)
                print(f"Cycles sauvegardés dans: {cycles_file}")

                print("\n=== CRÉATION EN BASE DE DONNÉES ===")
                try:
                    # Modification ici : on rend le patient optionnel temporairement
                    ecg = ECG.objects.create(
                        record_id=2,  # Pour test
                        ecg_data=file.read(),
                        diagnosis_date=timezone.now(),
                        confidence_score=0.85,
                        interpretation=f"""
                            ECG analysé avec succès
                            Nombre de cycles détectés : {len(cycles)}
                            Fréquence cardiaque estimée : {60/cycle_length:.1f} BPM
                        """,
                        patient_notified=True,
                        doctor_notified=False,
                    )
                    print(f"ECG créé avec succès, ID: {ecg.diagnosis_id}")
                except Exception as db_error:
                    print(f"ERREUR lors de la création en BDD: {str(db_error)}")
                    raise db_error

                return super().form_valid(form)
            
        except Exception as e:
            print(f"\n=== ERREUR ===")
            print(f"Type d'erreur: {type(e).__name__}")
            print(f"Message d'erreur: {str(e)}")
            print(f"Détails: {e.__dict__}")
            return self.form_invalid(form)

class ECGUploadSuccessView(TemplateView):
    template_name = 'patient_app/upload_success.html'