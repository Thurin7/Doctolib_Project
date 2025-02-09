from django.urls import path, include
from . import views

app_name = 'doctor_app'

urlpatterns = [
    path('dashboard/', views.DoctorDashboardView.as_view(), name='dashboard'),
    path('ecg-history/', views.DoctorECGHistoryView.as_view(), name='ecg_history'),
    path('ecg/<int:pk>/note/', views.save_doctor_note, name='add_note'),
    path('patient/', include('patient_app.urls')),
]