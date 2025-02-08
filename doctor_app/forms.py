from django import forms
from patient_app.models import ECG

class DoctorNoteForm(forms.ModelForm):
    class Meta:
        model = ECG
        fields = ['doctor_notes']
        widgets = {
            'doctor_notes': forms.Textarea(attrs={'rows': 4, 'class': 'w-full border rounded p-2'})
        }