from django.urls import path
from .views import ECGUploadView, ECGUploadSuccessView

app_name = 'patient_app'

urlpatterns = [
    path('upload/', ECGUploadView.as_view(), name='upload'),
    path('upload/success/', ECGUploadSuccessView.as_view(), name='upload_success'),
]