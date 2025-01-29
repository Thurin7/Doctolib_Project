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
from .utils.ecg_predictor import ECGPredictor
from .models import ECG
from django.utils import timezone
from django.shortcuts import render, redirect


class ECGUploadView(FormView):
    template_name = 'patient_app/upload.html'
    form_class = ECGUploadForm
    success_url = reverse_lazy('patient_app:upload_success')

    def form_valid(self, form):
        file = form.cleaned_data['ecg_file']
        
        # Créer le dossier tmp s'il n'existe pas
        tmp_dir = os.path.join(settings.MEDIA_ROOT, 'tmp')
        os.makedirs(tmp_dir, exist_ok=True)
        
        # Sauvegarder dans le dossier tmp
        tmp_path = os.path.join('tmp', file.name)
        path = default_storage.save(tmp_path, ContentFile(file.read()))
        tmp_file = os.path.join(settings.MEDIA_ROOT, path)
        
        try:
            self.request.session['uploaded_ecg_tmp_file'] = tmp_file
            self.request.session['uploaded_ecg_filename'] = file.name
            return super().form_valid(form)
        except Exception as e:
            if os.path.exists(tmp_file):
                os.remove(tmp_file)
            messages.error(self.request, "Erreur lors du téléchargement de l'ECG")
            return self.form_invalid(form)

class ECGUploadSuccessView(TemplateView):
    template_name = 'patient_app/upload_success.html'

    def get(self, request, *args, **kwargs):
        tmp_file = request.session.get('uploaded_ecg_tmp_file')
        filename = request.session.get('uploaded_ecg_filename')
        
        if not tmp_file or not filename:
            messages.error(request, "Aucun fichier ECG à traiter")
            return redirect('patient_app:upload')
        
        try:
            # Traitement de l'ECG
            processor = ECGProcessor()
            signal = processor.load_data(tmp_file)
            cycle_length = processor.analyze_cycle_distance(signal)
            
            if cycle_length:
                r_peaks = processor.find_r_peaks(signal, cycle_length)
                cycles, valid_peaks = processor.extract_cycles(signal, r_peaks)
                
                # Sauvegarder les cycles traités
                processed_file_path = processor.save_cycles(cycles, filename)
                
                # Chemin vers le modèle et le scaler (à ajuster)
                model_path = os.path.join(settings.BASE_DIR, 'patient_app', 'utils', 'model_1.h5')
                scaler_path = os.path.join(settings.BASE_DIR, 'patient_app', 'utils', 'ecg_scaler.joblib')
                
                # Initialisation du prédicteur
                predictor = ECGPredictor(
                    model_path=model_path,
                    scaler_path=scaler_path
                )
                
                # Analyse
                results = predictor.analyze_personal_ecg(cycles)

                # Création de l'ECG avec le fichier original
                with open(tmp_file, 'rb') as file:
                    risk_level = 'LOW' if results['conclusion'] == 'ECG SAIN' else 'HIGH'
                    ecg = ECG.objects.create(
                        ecg_data=file.read(),
                        processed_data_path=processed_file_path,  # Ajouter ce champ au modèle ECG
                        diagnosis_date=timezone.now(),
                        confidence_score=results['confidence_score'],
                        interpretation=results['interpretation'],
                        risk_level=risk_level,
                        patient_notified=True,
                        doctor_notified=False,
                    )

            else:
                messages.error(request, "Aucun cycle cardiaque détecté")

        except Exception as e:
            messages.error(request, f"Erreur lors du traitement de l'ECG : {str(e)}")
            print(f"Erreur de traitement ECG : {e}")
        
        finally:
            # Nettoyage du fichier temporaire
            if tmp_file and os.path.exists(tmp_file):
                os.remove(tmp_file)

            # Nettoyer la session
            request.session.pop('uploaded_ecg_tmp_file', None)
            request.session.pop('uploaded_ecg_filename', None)

        return super().get(request, *args, **kwargs)

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        try:
            latest_ecg = ECG.objects.latest('created_at')
            context['ecg'] = latest_ecg
        except ECG.DoesNotExist:
            context['ecg'] = None
        return context