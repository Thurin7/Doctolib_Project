from django.db import models
from django.utils import timezone
from account_app.models import Patient
import pandas as pd
import json


class ECG(models.Model):
    RISK_LEVELS = [
        ('LOW', 'Faible'),
        ('MEDIUM', 'Moyen'),
        ('HIGH', 'Élevé'),
    ]

    diagnosis_id = models.AutoField(primary_key=True)
    patient = models.ForeignKey(
        Patient,
        on_delete=models.CASCADE,
        related_name='ecgs',
        null=True,
        blank=True
    )
    ecg_data = models.BinaryField()
    processed_data_path = models.CharField(max_length=255, null=True, blank=True)
    confidence_score = models.FloatField(null=True, blank=True)
    cycles_analysis_path = models.CharField(max_length=255, null=True, blank=True)
    interpretation = models.TextField(null=True, blank=True)
    risk_level = models.CharField(  # Ajout du champ risk_level
        max_length=10,
        choices=RISK_LEVELS,
        default='LOW'
    )
    plots = models.BinaryField(editable=True, null=True, blank=True)  # Ajoutez editable=True
    patient_notified = models.BooleanField(default=False)
    doctor_notified = models.BooleanField(default=False)
    doctor_notes = models.TextField(null=True, blank=True)
    diagnosis_date = models.DateTimeField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        patient_name = f"{self.patient.last_name}" if self.patient else "Inconnu"
        return f"ECG #{self.diagnosis_id} - Patient: {patient_name} - Date: {self.created_at}"
    
    def get_cycle_details(self):
        try:
            with open(self.cycles_analysis_path, 'r') as f:
                analysis = json.load(f)
            return analysis.get('cycles_details', [])
        except Exception as e:
            print(f"Erreur lors de la lecture des détails des cycles : {e}")
            return []

    class Meta:
        verbose_name = "ECG"
        verbose_name_plural = "ECGs"
        ordering = ['-created_at']