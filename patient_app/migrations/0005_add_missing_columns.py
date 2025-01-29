# Première migration (0005_add_missing_columns.py)
from django.db import migrations, models

class Migration(migrations.Migration):

    dependencies = [
        ('patient_app', '0004_ecg_processed_data_path'),
    ]

    operations = [
        migrations.AddField(
            model_name='ecg',
            name='plots',
            field=models.BinaryField(blank=True, null=True),
            preserve_default=True,
        ),
        migrations.AddField(
            model_name='ecg',
            name='risk_level',
            field=models.CharField(choices=[('LOW', 'Faible'), ('MEDIUM', 'Moyen'), ('HIGH', 'Élevé')], default='LOW', max_length=10),
            preserve_default=True,
        ),
    ]