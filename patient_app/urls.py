from django.urls import path
from .views import ECGUploadView, ECGUploadSuccessView, ECGDetailView, ECGHistoryView

app_name = 'patient_app'

urlpatterns = [
    path('upload/', ECGUploadView.as_view(), name='upload'),
    path('upload/success/', ECGUploadSuccessView.as_view(), name='upload_success'),
    path('ecg/detail/<int:pk>/', ECGDetailView.as_view(), name='ecg_detail'),
    path('ecg/history/', ECGHistoryView.as_view(), name='ecg_history'),
]