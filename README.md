# Doctolib Project

## Installation

1. Cloner le projet
```bash
git clone [votre-lien-github]
cd Doctolib_Project
```

2. Créer et activer l'environnement virtuel
```bash
python3 -m venv env
source env/bin/activate  # Sur MacOS/Linux
# ou
.\env\Scripts\activate  # Sur Windows
```

3. Installer les dépendances
```bash
pip install django
```

4. Appliquer les migrations
```bash
python manage.py migrate
```

5. Créer un superuser
```bash
python manage.py createsuperuser
```
Note : Pour le rôle, utiliser 'ADMIN' (en majuscules)

6. Lancer le serveur
```bash
python manage.py runserver
```

## Structure du projet

- `account_app/` : Gestion des utilisateurs (User, Patient, Doctor)
- `patient_app/` : Gestion des ECG

## Base de données

### Models principaux :
- User : Modèle d'authentification personnalisé
- Patient : Stockage des informations patients
- Doctor : Stockage des informations médecins
- ECG : Stockage et analyse des ECG

### Relations :
- Un médecin peut avoir plusieurs patients
- Un patient a un seul médecin
- Un patient peut avoir plusieurs ECG

## Contributeurs
- Lucien LACHARMOISE
- Clément ASENSIO
- Emma COCO
- Cheryle Adebada

## Notes de développement
- Utiliser Python 3.9+
- Django 4.2+