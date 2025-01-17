from django.urls import path
from .views import ECGUploadView

app_name = 'patient_app'

urlpatterns = [
    path('upload/', ECGUploadView.as_view(), name='upload'),
]