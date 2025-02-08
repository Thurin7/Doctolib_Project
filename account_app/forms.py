from django.contrib.auth.forms import AuthenticationForm
from django import forms
from django.contrib.auth.forms import UserCreationForm
from .models import User, Doctor, Patient

class UserRegistrationForm(UserCreationForm):
    email = forms.EmailField(
        required=True,
        widget=forms.EmailInput(attrs={
            'class': 'mt-1 block w-full rounded-lg border-gray-300 shadow-sm focus:border-sky-500 focus:ring-sky-500 transition-colors duration-200',
            'placeholder': 'exemple@email.com',
            'id': 'registration_email'
        })
    )
    first_name = forms.CharField(
        required=True,
        widget=forms.TextInput(attrs={
            'class': 'mt-1 block w-full rounded-lg border-gray-300 shadow-sm focus:border-sky-500 focus:ring-sky-500 transition-colors duration-200',
            'placeholder': 'Prénom',
            'id': 'registration_first_name'
        })
    )
    last_name = forms.CharField(
        required=True,
        widget=forms.TextInput(attrs={
            'class': 'mt-1 block w-full rounded-lg border-gray-300 shadow-sm focus:border-sky-500 focus:ring-sky-500 transition-colors duration-200',
            'placeholder': 'Nom',
            'id': 'registration_last_name'
        })
    )
    role = forms.ChoiceField(
        choices=User.ROLE_CHOICES,
        required=True,
        widget=forms.Select(attrs={
            'class': 'mt-1 block w-full rounded-lg border-gray-300 shadow-sm focus:border-sky-500 focus:ring-sky-500 transition-colors duration-200',
            'id': 'registration_role'
        })
    )

    class Meta:
        model = User
        fields = ('email', 'first_name', 'last_name', 'role', 'password1', 'password2')

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Personnaliser les widgets des champs password
        self.fields['password1'].widget.attrs.update({
            'class': 'mt-1 block w-full rounded-lg border-gray-300 shadow-sm focus:border-sky-500 focus:ring-sky-500 transition-colors duration-200',
            'id': 'registration_password1'
        })
        self.fields['password2'].widget.attrs.update({
            'class': 'mt-1 block w-full rounded-lg border-gray-300 shadow-sm focus:border-sky-500 focus:ring-sky-500 transition-colors duration-200',
            'id': 'registration_password2'
        })

class DoctorRegistrationForm(forms.ModelForm):
    class Meta:
        model = Doctor
        fields = ('license_number', 'specialty', 'phone')
        widgets = {
            'license_number': forms.TextInput(attrs={
                'class': 'mt-1 block w-full rounded-lg border-gray-300 shadow-sm focus:border-sky-500 focus:ring-sky-500 transition-colors duration-200',
                'placeholder': 'Numéro RPPS',
                'id': 'doctor_license_number'
            }),
            'specialty': forms.TextInput(attrs={
                'class': 'mt-1 block w-full rounded-lg border-gray-300 shadow-sm focus:border-sky-500 focus:ring-sky-500 transition-colors duration-200',
                'placeholder': 'Spécialité',
                'id': 'doctor_specialty'
            }),
            'phone': forms.TextInput(attrs={
                'class': 'mt-1 block w-full rounded-lg border-gray-300 shadow-sm focus:border-sky-500 focus:ring-sky-500 transition-colors duration-200',
                'placeholder': 'Téléphone',
                'id': 'doctor_phone'
            })
        }

class PatientRegistrationForm(forms.ModelForm):
    GENDER_CHOICES = [
        ('M', 'Masculin'),
        ('F', 'Féminin'),
        ('O', 'Autre')
    ]

    gender = forms.ChoiceField(
        choices=GENDER_CHOICES,
        required=True,
        widget=forms.Select(attrs={
            'class': 'mt-1 block w-full rounded-lg border-gray-300 shadow-sm focus:border-sky-500 focus:ring-sky-500 transition-colors duration-200',
            'id': 'patient_gender'
        })
    )

    class Meta:
        model = Patient
        fields = ['social_security_number', 'birth_date', 'gender', 'phone', 'address']
        widgets = {
            'social_security_number': forms.TextInput(attrs={
                'class': 'mt-1 block w-full rounded-lg border-gray-300 shadow-sm focus:border-sky-500 focus:ring-sky-500 transition-colors duration-200',
                'placeholder': 'Numéro de sécurité sociale',
                'id': 'patient_ssn'
            }),
            'birth_date': forms.DateInput(attrs={
                'class': 'mt-1 block w-full rounded-lg border-gray-300 shadow-sm focus:border-sky-500 focus:ring-sky-500 transition-colors duration-200',
                'type': 'date',
                'id': 'patient_birth_date'
            }),
            'phone': forms.TextInput(attrs={
                'class': 'mt-1 block w-full rounded-lg border-gray-300 shadow-sm focus:border-sky-500 focus:ring-sky-500 transition-colors duration-200',
                'placeholder': 'Téléphone',
                'id': 'patient_phone'
            }),
            'address': forms.Textarea(attrs={
                'class': 'mt-1 block w-full rounded-lg border-gray-300 shadow-sm focus:border-sky-500 focus:ring-sky-500 transition-colors duration-200',
                'placeholder': 'Adresse',
                'rows': '3',
                'id': 'patient_address'
            })
        }

    class Meta:
        model = Patient
        fields = ('social_security_number', 'birth_date', 'gender', 'phone', 'address')
        widgets = {
            'social_security_number': forms.TextInput(attrs={
                'class': 'mt-1 block w-full rounded-lg border-gray-300 shadow-sm focus:border-sky-500 focus:ring-sky-500 transition-colors duration-200',
                'placeholder': 'Numéro de sécurité sociale',
                'id': 'patient_ssn'
            }),
            'birth_date': forms.DateInput(attrs={
                'class': 'mt-1 block w-full rounded-lg border-gray-300 shadow-sm focus:border-sky-500 focus:ring-sky-500 transition-colors duration-200',
                'type': 'date',
                'id': 'patient_birth_date'
            }),
            'phone': forms.TextInput(attrs={
                'class': 'mt-1 block w-full rounded-lg border-gray-300 shadow-sm focus:border-sky-500 focus:ring-sky-500 transition-colors duration-200',
                'placeholder': 'Téléphone',
                'id': 'patient_phone'
            }),
            'address': forms.Textarea(attrs={
                'class': 'mt-1 block w-full rounded-lg border-gray-300 shadow-sm focus:border-sky-500 focus:ring-sky-500 transition-colors duration-200',
                'placeholder': 'Adresse',
                'rows': '3',
                'id': 'patient_address'
            })
        }

class LoginForm(AuthenticationForm):
    username = forms.EmailField(
        widget=forms.EmailInput(attrs={
            'class': 'mt-1 block w-full rounded-lg border-gray-300 shadow-sm focus:border-sky-500 focus:ring-sky-500 transition-colors duration-200',
            'placeholder': 'Votre email',
            'id': 'login_email'
        })
    )
    password = forms.CharField(
        widget=forms.PasswordInput(attrs={
            'class': 'mt-1 block w-full rounded-lg border-gray-300 shadow-sm focus:border-sky-500 focus:ring-sky-500 transition-colors duration-200',
            'placeholder': 'Votre mot de passe',
            'id': 'login_password'
        })
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Remplacer le champ username par email pour l'authentification
        self.fields['username'].label = 'Email'