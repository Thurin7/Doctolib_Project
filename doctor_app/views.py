from django.views.generic import TemplateView, ListView
from django.db.models import Case, When, Value, IntegerField, Count
from django.urls import reverse
from django.http import JsonResponse
from django.views.decorators.http import require_http_methods
from django.views.decorators.csrf import csrf_protect
from django.contrib.auth.mixins import LoginRequiredMixin
from patient_app.models import ECG
import json


class DoctorDashboardView(LoginRequiredMixin, TemplateView):
    template_name = 'doctor_app/dashboard.html'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        doctor = self.request.user.doctor

        # Filtrer les ECG pour n'inclure que ceux des patients du médecin connecté
        ecgs = ECG.objects.filter(patient__doctor=doctor)

        # Calculer les statistiques
        risk_counts = ecgs.values('risk_level').annotate(count=Count('risk_level'))
        risk_dict = {item['risk_level']: item['count'] for item in risk_counts}

        context['total_ecgs'] = ecgs.count()
        context['high_risk_count'] = risk_dict.get('HIGH', 0)
        context['medium_risk_count'] = risk_dict.get('MEDIUM', 0)
        context['low_risk_count'] = risk_dict.get('LOW', 0)

        # Récupérer les 5 ECG les plus récents, tous niveaux de risque confondus
        context['recent_ecgs'] = ecgs.order_by(
            Case(
                When(risk_level='HIGH', then=Value(0)),
                When(risk_level='MEDIUM', then=Value(1)),
                When(risk_level='LOW', then=Value(2)),
                default=Value(3),
                output_field=IntegerField()
            ),
            '-diagnosis_date'
        )[:5]

        context['show_all_ecgs_url'] = reverse('doctor_app:ecg_history')
        return context

class DoctorECGHistoryView(LoginRequiredMixin, ListView):
    model = ECG
    template_name = 'doctor_app/ecg_history.html'
    context_object_name = 'ecgs'
    paginate_by = 10

    def get_queryset(self):
        # Filtrer les ECG pour n'inclure que ceux des patients du médecin connecté
        return ECG.objects.filter(patient__doctor=self.request.user.doctor).select_related('patient').order_by(
            Case(
                When(risk_level='HIGH', then=Value(0)),
                When(risk_level='MEDIUM', then=Value(1)),
                default=Value(2),
                output_field=IntegerField()
            ), 
            '-diagnosis_date'
        )

@csrf_protect
@require_http_methods(["POST"])
def save_doctor_note(request, pk):
    try:
        data = json.loads(request.body)
        ecg = ECG.objects.get(pk=pk)
        ecg.doctor_notes = data.get('doctor_notes', '')
        ecg.save()
        return JsonResponse({'status': 'success'})
    except ECG.DoesNotExist:
        return JsonResponse({'status': 'error', 'message': 'ECG not found'}, status=404)
    except json.JSONDecodeError:
        return JsonResponse({'status': 'error', 'message': 'Invalid JSON'}, status=400)
    except Exception as e:
        return JsonResponse({'status': 'error', 'message': str(e)}, status=500)