from django.contrib.auth import login, logout
from django.shortcuts import render, redirect
from django.views.generic import CreateView
from django.contrib import messages
from .forms import UserRegistrationForm
from .models import User, Patient, Doctor

class UserRegistrationView(CreateView):
    template_name = 'account_app/register.html'
    form_class = UserRegistrationForm

    def form_valid(self, form):
        # Créer l'utilisateur
        user = form.save()
        
        # Récupérer les données du formulaire
        role = form.cleaned_data.get('role')

        try:
            if role == 'PATIENT':
                # Créer le profil patient
                Patient.objects.create(
                    user=user,
                    social_security_number=form.cleaned_data.get('social_security_number'),
                    first_name=user.first_name,
                    last_name=user.last_name,
                    birth_date=form.cleaned_data.get('birth_date'),
                    gender=form.cleaned_data.get('gender'),
                    phone=form.cleaned_data.get('patient_phone'),
                    address=form.cleaned_data.get('address')
                )
            elif role == 'DOCTOR':
                # Créer le profil médecin
                Doctor.objects.create(
                    user=user,
                    license_number=form.cleaned_data.get('license_number'),
                    first_name=user.first_name,
                    last_name=user.last_name,
                    specialty=form.cleaned_data.get('specialty'),
                    phone=form.cleaned_data.get('doctor_phone')
                )

            # Déconnexion explicite
            logout(self.request)

            # Ajouter un message de succès
            messages.success(self.request, 'Compte créé avec succès. Veuillez vous connecter.')
            
            # Redirection vers la page de login
            return redirect('account_app:login')

        except Exception as e:
            # En cas d'erreur, supprimer l'utilisateur créé
            user.delete()
            messages.error(self.request, f'Erreur lors de la création du profil : {str(e)}')
            return self.form_invalid(form)

    def get_success_url(self):
        return None

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