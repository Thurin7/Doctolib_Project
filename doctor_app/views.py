from django.views.generic import TemplateView, ListView
from django.db.models import Case, When, Value, IntegerField
from django.urls import reverse
from django.http import JsonResponse
from django.views.decorators.http import require_http_methods
from django.views.decorators.csrf import csrf_protect
from patient_app.models import ECG
import json

class DoctorDashboardView(TemplateView):
    template_name = 'doctor_app/dashboard.html'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['high_risk_ecgs'] = ECG.objects.order_by(
            Case(
                When(risk_level='HIGH', then=Value(0)),
                When(risk_level='MEDIUM', then=Value(1)),
                default=Value(2),
                output_field=IntegerField()
            ), 
            '-diagnosis_date'
        )[:5]
        context['show_all_ecgs_url'] = reverse('doctor_app:ecg_history')
        context['total_ecgs'] = ECG.objects.count()
        context['high_risk_count'] = ECG.objects.filter(risk_level='HIGH').count()
        context['medium_risk_count'] = ECG.objects.filter(risk_level='MEDIUM').count()
        return context

class ECGHistoryView(ListView):
    model = ECG
    template_name = 'doctor_app/ecg_history.html'
    context_object_name = 'ecgs'
    paginate_by = 10

    def get_queryset(self):
        return ECG.objects.order_by(
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