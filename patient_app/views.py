from django.shortcuts import render
from django.views.generic.edit import FormView
from .forms import ECGUploadForm

class ECGUploadView(FormView):
    template_name = 'patient_app/upload.html'
    form_class = ECGUploadForm
    success_url = '/upload/success/'  # On changera ça plus tard

    def form_valid(self, form):
        file = form.cleaned_data['ecg_file']
        # Pour l'instant, on affiche juste un message de succès
        # On ajoutera le traitement plus tard
        return super().form_valid(form)