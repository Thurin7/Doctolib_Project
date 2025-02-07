from django.shortcuts import render
from django.views.generic.base import TemplateView
from django.views.generic.edit import FormView
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
from django.conf import settings
from django.urls import reverse_lazy
from django.contrib import messages
from django.utils import timezone
from django.shortcuts import render, redirect
import os
import base64
from .forms import ECGUploadForm
from .utils.ecg_processor import ECGProcessor
from .utils.ecg_predictor import ECGPredictor
from .models import ECG
from django.views.generic import DetailView, ListView
from .models import ECG

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
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        # Récupérer les 5 dernières analyses ECG, sans filtrage
        context['previous_ecgs'] = ECG.objects.order_by('-diagnosis_date')[:5]
        return context

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
            
            if not cycle_length:
                messages.error(request, "Aucun cycle cardiaque détecté")
                return redirect('patient_app:upload')

            # Extraction des cycles
            r_peaks = processor.find_r_peaks(signal, cycle_length)
            cycles, valid_peaks = processor.extract_cycles(signal, r_peaks)
            processed_file_path = processor.save_cycles(cycles, filename)
            print(f"Chemin du fichier de cycles : {processed_file_path}")


            # Configuration des chemins pour le modèle et le scaler
            model_path = os.path.join(settings.BASE_DIR, 'patient_app', 'models', 'ecg_model_m1.h5')
            scaler_path = os.path.join(settings.BASE_DIR, 'patient_app', 'models', 'ecg_scaler_m1.joblib')
            
            # Initialisation et analyse avec le nouveau prédicteur
            predictor = ECGPredictor(model_path=model_path, scaler_path=scaler_path)
            results = predictor.analyze_personal_ecg(cycles)

            # Lecture du fichier ECG
            with open(tmp_file, 'rb') as file:
                ecg_data = file.read()

            # Conversion des plots en bytes si nécessaire
            plots_data = results.get('plots')
            if plots_data and not isinstance(plots_data, bytes):
                plots_data = bytes(plots_data)

            print("Debug - Interpretation:", results.get('interpretation'))

            # Dans la vue, avant de créer l'ECG
            print("Debug - Plots data type:", type(results.get('plots')))
            print("Debug - Plots data size:", len(results.get('plots')) if results.get('plots') else 'None')
            print("Debug - Interpretation:", results.get('interpretation'))

            # Création de l'ECG avec les plots

            results = predictor.analyze_personal_ecg(cycles)
            json_path = processor.save_analysis_results(cycles, results, filename)

            ecg = ECG.objects.create(
            ecg_data=ecg_data,
            processed_data_path=processed_file_path,
            diagnosis_date=timezone.now(),
            confidence_score=results['confidence_score'],
            interpretation=results['interpretation'],
            risk_level=results['risk_level'],
            plots=plots_data,
            patient_notified=True,
            doctor_notified=results['risk_level'] == 'HIGH',
            cycles_analysis_path=json_path  # Ajoutez ce champ au modèle ECG
        )


            # Stockage des cycles_details dans la session pour l'affichage
            request.session['cycles_details'] = results.get('cycles_details', [])
            request.session['analyzed_ecg_id'] = ecg.diagnosis_id

        except Exception as e:
            import traceback
            print("Erreur détaillée avec traceback complet:")
            print(traceback.format_exc())
            messages.error(request, f"Erreur lors du traitement de l'ECG : {str(e)}")
            return redirect('patient_app:upload')
        
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
            # Récupérer l'ECG analysé
            ecg_id = self.request.session.get('analyzed_ecg_id')
           
            if ecg_id:
                ecg = ECG.objects.get(diagnosis_id=ecg_id)
                context['ecg'] = ecg

                # Convertir le score de confiance en pourcentage
                ecg.confidence_score = ecg.confidence_score * 100
                
                context['cycles_details'] = self.request.session.get('cycles_details', [])
              
                if ecg.plots:
                    context['plots'] = base64.b64encode(ecg.plots).decode('utf-8')

                # Ajouter des messages selon le niveau de risque
                if ecg.risk_level == 'HIGH':
                    messages.warning(self.request, "⚠️ Attention : Anomalies significatives détectées. Consultation médicale recommandée.")
                elif ecg.risk_level == 'MEDIUM':
                    messages.info(self.request, "ℹ️ Quelques irrégularités détectées. Surveillance recommandée.")
                else:
                    messages.success(self.request, "✅ ECG normal. Aucune anomalie significative détectée.")
                
            # Nettoyer la session après utilisation
            self.request.session.pop('analyzed_ecg_id', None)
            self.request.session.pop('cycles_details', None)
            
        except ECG.DoesNotExist:
            context['ecg'] = None
            messages.error(self.request, "Résultats de l'analyse non trouvés")
        
        return context

class ECGDetailView(DetailView):
    model = ECG
    template_name = 'patient_app/ecg_detail.html'
    context_object_name = 'ecg'
    pk_url_kwarg = 'pk'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        
        # Convertir le score de confiance en pourcentage
        ecg = self.object
        print(f"ECG ID: {ecg.diagnosis_id}")
        print(f"Doctor notes: {ecg.doctor_notes}")
        print(f"Risk level: {ecg.risk_level}")
        ecg.confidence_score = ecg.confidence_score * 100
        
        # Ajouter les plots en base64 si disponible
        if ecg.plots:
            context['plots'] = base64.b64encode(ecg.plots).decode('utf-8')
        
        # Récupérer les détails des cycles
        try:
            # Ajoutez votre logique pour extraire les détails des cycles
            # Par exemple, si vous avez une méthode dans votre modèle ECG
            context['cycles_details'] = ecg.get_cycle_details()
        except Exception:
            context['cycles_details'] = []
        
        print(f"Doctor notes: {ecg.doctor_notes}")
        
        return context
    
class ECGHistoryView(ListView):
    model = ECG
    template_name = 'patient_app/ecg_history.html'
    context_object_name = 'ecgs'
    paginate_by = 10

    def get_queryset(self):
        # Récupérer tous les ECG, triés par date de diagnostic (les plus récents en premier)
        return ECG.objects.order_by('-diagnosis_date')