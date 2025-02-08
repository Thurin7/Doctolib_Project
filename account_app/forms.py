from django import forms
from django.contrib.auth.forms import UserCreationForm, AuthenticationForm
from django.core.validators import RegexValidator
from .models import User, Doctor, Patient

class LoginForm(AuthenticationForm):
    username = forms.EmailField(
        widget=forms.EmailInput(attrs={
            'class': 'w-full px-3 py-2 border rounded-lg',
            'placeholder': 'Email'
        })
    )
    password = forms.CharField(
        widget=forms.PasswordInput(attrs={
            'class': 'w-full px-3 py-2 border rounded-lg',
            'placeholder': 'Mot de passe'
        })
    )

class UserRegistrationForm(UserCreationForm):
    ROLE_CHOICES = [
        ('PATIENT', 'Patient'),
        ('DOCTOR', 'Médecin')
    ]

    role = forms.ChoiceField(
        choices=ROLE_CHOICES, 
        widget=forms.Select(attrs={
            'class': 'w-full px-3 py-2 border rounded-lg'
        })
    )
    
    first_name = forms.CharField(
        widget=forms.TextInput(attrs={
            'class': 'w-full px-3 py-2 border rounded-lg',
            'placeholder': 'Prénom'
        })
    )
    last_name = forms.CharField(
        widget=forms.TextInput(attrs={
            'class': 'w-full px-3 py-2 border rounded-lg',
            'placeholder': 'Nom'
        })
    )
    email = forms.EmailField(
        widget=forms.EmailInput(attrs={
            'class': 'w-full px-3 py-2 border rounded-lg',
            'placeholder': 'Email'
        })
    )
    password1 = forms.CharField(
        label='Mot de passe',
        widget=forms.PasswordInput(attrs={
            'class': 'w-full px-3 py-2 border rounded-lg',
            'placeholder': 'Mot de passe'
        })
    )
    password2 = forms.CharField(
        label='Confirmation du mot de passe',
        widget=forms.PasswordInput(attrs={
            'class': 'w-full px-3 py-2 border rounded-lg',
            'placeholder': 'Confirmez le mot de passe'
        })
    )
    
    # Champs spécifiques aux patients
    social_security_number = forms.CharField(
        required=False,
        validators=[
            RegexValidator(
                regex=r'^\d{15}$', 
                message="Le numéro de sécurité sociale doit contenir exactement 15 chiffres"
            )
        ],
        widget=forms.TextInput(attrs={
            'class': 'w-full px-3 py-2 border rounded-lg',
            'placeholder': 'Numéro de sécurité sociale'
        }),
        help_text="Numéro de sécurité sociale de 15 chiffres"
    )
    birth_date = forms.DateField(
        required=False,
        widget=forms.DateInput(attrs={
            'type': 'date',
            'class': 'w-full px-3 py-2 border rounded-lg'
        })
    )
    gender = forms.ChoiceField(
        required=False,
        choices=[('M', 'Masculin'), ('F', 'Féminin'), ('O', 'Autre')],
        widget=forms.Select(attrs={
            'class': 'w-full px-3 py-2 border rounded-lg'
        })
    )
    patient_phone = forms.CharField(
        required=False,
        widget=forms.TextInput(attrs={
            'class': 'w-full px-3 py-2 border rounded-lg',
            'placeholder': 'Numéro de téléphone'
        })
    )
    address = forms.CharField(
        required=False,
        widget=forms.Textarea(attrs={
            'class': 'w-full px-3 py-2 border rounded-lg',
            'placeholder': 'Adresse',
            'rows': 3
        })
    )
    
    # Champs spécifiques aux médecins
    license_number = forms.CharField(
        required=False,
        validators=[
            RegexValidator(
                regex=r'^\d{11}$', 
                message="Le numéro RPPS doit contenir exactement 11 chiffres"
            )
        ],
        widget=forms.TextInput(attrs={
            'class': 'w-full px-3 py-2 border rounded-lg',
            'placeholder': 'Numéro RPPS'
        }),
        help_text="Numéro RPPS de 11 chiffres"
    )
    specialty = forms.CharField(
        required=False,
        widget=forms.TextInput(attrs={
            'class': 'w-full px-3 py-2 border rounded-lg',
            'placeholder': 'Spécialité'
        })
    )
    doctor_phone = forms.CharField(
        required=False,
        widget=forms.TextInput(attrs={
            'class': 'w-full px-3 py-2 border rounded-lg',
            'placeholder': 'Numéro de téléphone'
        })
    )

    class Meta:
        model = User
        fields = ['first_name', 'last_name', 'email', 'role', 'password1', 'password2']

    def clean(self):
        cleaned_data = super().clean()
        role = cleaned_data.get('role')

        if role == 'PATIENT':
            # Validation des champs patients
            required_patient_fields = [
                'social_security_number', 'birth_date', 
                'gender', 'patient_phone', 'address'
            ]
            for field in required_patient_fields:
                if not cleaned_data.get(field):
                    self.add_error(field, f"Ce champ est obligatoire pour les patients")

        elif role == 'DOCTOR':
            # Validation des champs médecins
            required_doctor_fields = [
                'license_number', 'specialty', 'doctor_phone'
            ]
            for field in required_doctor_fields:
                if not cleaned_data.get(field):
                    self.add_error(field, f"Ce champ est obligatoire pour les médecins")

        return cleaned_data