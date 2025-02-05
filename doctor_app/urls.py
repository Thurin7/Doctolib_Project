from django.urls import path
from .views import DoctorDashboardView, ECGHistoryView
from . import views


app_name = 'doctor_app'

urlpatterns = [
    path('dashboard/', views.DoctorDashboardView.as_view(), name='dashboard'),
    path('ecg-history/', views.ECGHistoryView.as_view(), name='ecg_history'),
    path('ecg/<int:pk>/note/', views.save_doctor_note, name='add_note'),
]