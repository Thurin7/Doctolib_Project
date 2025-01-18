from django.db import models
from django.utils import timezone
from account_app.models import Patient

class ECG(models.Model):
    diagnosis_id = models.AutoField(primary_key=True)
    record_id = models.IntegerField(unique=True)
    patient = models.ForeignKey(
        Patient,
        on_delete=models.CASCADE,
        related_name='ecgs',
        null=True,  # Ajout de ces deux lignes
        blank=True  # pour rendre le champ optionnel
    )
    ecg_data = models.BinaryField()
    confidence_score = models.FloatField(null=True, blank=True)
    interpretation = models.TextField(null=True, blank=True)
    patient_notified = models.BooleanField(default=False)
    doctor_notified = models.BooleanField(default=False)
    doctor_notes = models.TextField(null=True, blank=True)
    diagnosis_date = models.DateTimeField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"ECG #{self.diagnosis_id} - Patient: {self.patient.last_name} - Date: {self.created_at}"

    class Meta:
        verbose_name = "ECG"
        verbose_name_plural = "ECGs"
        ordering = ['-created_at']