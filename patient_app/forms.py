from django import forms

class ECGUploadForm(forms.Form):
    ecg_file = forms.FileField(
        label='Fichier ECG',
        help_text='Uploadez votre fichier ECG Arduino (.csv)'
    )