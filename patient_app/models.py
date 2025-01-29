from django.db import models
from django.utils import timezone
from account_app.models import Patient

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
    confidence_score = models.FloatField(null=True, blank=True)
    interpretation = models.TextField(null=True, blank=True)
    risk_level = models.CharField(  # Ajout du champ risk_level
        max_length=10,
        choices=RISK_LEVELS,
        default='LOW'
    )
    plots = models.BinaryField(null=True, blank=True)  # Ajout du champ plots
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