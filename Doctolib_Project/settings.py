import os
from pathlib import Path

# Build paths inside the project like this: BASE_DIR / 'subdir'.
BASE_DIR = Path(__file__).resolve().parent.parent

# SECURITY WARNING: keep the secret key used in production secret!
SECRET_KEY = 'django-insecure-fca%yz6*^&^8&9x^ns_nd=fms679!p6l+p!60!h+c)bysm$k@1'

# SECURITY WARNING: don't run with debug turned on in production!
DEBUG = True

ALLOWED_HOSTS = []

# Application definition
INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    
    # Applications personnalisées
    'account_app',
    'doctor_app',
    'patient_app',
]

MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]

ROOT_URLCONF = 'Doctolib_Project.urls'

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [BASE_DIR / 'templates'],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
            ],
        },
    },
]

WSGI_APPLICATION = 'Doctolib_Project.wsgi.application'

# Database
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': BASE_DIR / 'db.sqlite3',
    }
}

# Validation des mots de passe
AUTH_PASSWORD_VALIDATORS = [
    {
        'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.CommonPasswordValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.NumericPasswordValidator',
    },
]

# Internationalisation
LANGUAGE_CODE = 'fr-fr'  # Changé de 'en-us' à 'fr-fr'
TIME_ZONE = 'Europe/Paris'  # Changé de 'UTC' à 'Europe/Paris'
USE_I18N = True
USE_TZ = True

# Fichiers statiques
STATIC_URL = 'static/'

# Modèle utilisateur personnalisé
AUTH_USER_MODEL = 'account_app.User'

# Gestion des médias
MEDIA_URL = '/media/'
MEDIA_ROOT = BASE_DIR / 'media'

# Créer les répertoires temporaires
os.makedirs(MEDIA_ROOT / 'tmp', exist_ok=True)
os.makedirs(MEDIA_ROOT / 'processed_ecg', exist_ok=True)

# Configurations de connexion
LOGIN_URL = 'account_app:login'
LOGIN_REDIRECT_URL = 'account_app:dashboard'
LOGOUT_REDIRECT_URL = 'account_app:login'

# Clé par défaut pour les modèles
DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'

# Messages framework configuration
from django.contrib.messages import constants as messages

MESSAGE_TAGS = {
    messages.DEBUG: 'alert-info',
    messages.INFO: 'alert-info',
    messages.SUCCESS: 'alert-success',
    messages.WARNING: 'alert-warning',
    messages.ERROR: 'alert-danger',
}

# Désactiver les messages de framework par défaut
MESSAGE_STORAGE = 'django.contrib.messages.storage.session.SessionStorage'