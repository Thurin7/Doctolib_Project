
from django.contrib.auth.decorators import login_required
from django.shortcuts import render, redirect
from django.contrib.auth import login
from django.views.generic import CreateView
from .forms import UserRegistrationForm, DoctorRegistrationForm, PatientRegistrationForm
from .models import User, Doctor, Patient

class UserRegistrationView(CreateView):
    template_name = 'account_app/register.html'
    form_class = UserRegistrationForm
    success_url = '/account/complete_profile/'

    def form_valid(self, form):
        # Créer l'utilisateur
        user = form.save()
        login(self.request, user)
        return super().form_valid(form)


@login_required
def complete_profile(request):
    user = request.user
    
    # Vérifie que l'utilisateur est authentifié
    if not hasattr(user, 'role'):
        return redirect('account_app:login')
    
    if request.method == 'POST':
        if user.role == 'DOCTOR':
            form = DoctorRegistrationForm(request.POST)
            if form.is_valid():
                doctor = form.save(commit=False)
                doctor.user = user
                doctor.first_name = user.first_name
                doctor.last_name = user.last_name
                doctor.save()
                return redirect('account_app:dashboard')
        elif user.role == 'PATIENT':
            form = PatientRegistrationForm(request.POST)
            if form.is_valid():
                patient = form.save(commit=False)
                patient.user = user
                patient.first_name = user.first_name
                patient.last_name = user.last_name
                patient.save()
                return redirect('account_app:dashboard')
    else:
        if user.role == 'DOCTOR':
            form = DoctorRegistrationForm()
        elif user.role == 'PATIENT':
            form = PatientRegistrationForm()
        else:
            return redirect('account_app:login')

    return render(request, 'account_app/complete_profile.html', {'form': form})

def dashboard(request):
    # Vue du tableau de bord qui diffère selon le rôle
    user = request.user
    if user.role == 'DOCTOR':
        return redirect('doctor_app:dashboard')
    elif user.role == 'PATIENT':
        return redirect('patient_app:upload')
    else:
        # Pour les admin, ajouter une vue admin si nécessaire
        return redirect('admin:index')