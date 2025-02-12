# Doctolib Project

## 📋 Table des matières
- [Installation](#installation)
- [Structure du projet](#structure-du-projet)
- [Base de données](#base-de-données)
- [Contributeurs](#contributeurs)
- [Notes de développement](#notes-de-développement)

---

## 🚀 Installation

### Prérequis
- **Python 3.9+**
- **pip** (gestionnaire de paquets Python)

### Étapes d'installation

1. **Cloner le projet**
   ```bash
   git clone [votre-lien-github]
   cd Doctolib_Project
   ```

2. **Créer et activer l'environnement virtuel**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate  # Sur MacOS/Linux
   # ou
   .\.venv\Scripts\activate  # Sur Windows
   ```

3. **Installer les dépendances**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configurer l'environnement**  
   Créez un fichier `.env` à la racine du projet et ajoutez les variables d'environnement nécessaires.

5. **Appliquer les migrations**
   ```bash
   python manage.py migrate
   ```

6. **Créer un superuser**
   ```bash
   python manage.py createsuperuser
   ```
   📌 **Note :** Pour le rôle, utilisez `'ADMIN'` (en majuscules).

7. **Créer le dossier pour les sessions**
   ```bash
   mkdir session_files
   chmod 755 session_files
   ```

8. **Lancer le serveur**
   ```bash
   python manage.py runserver
   ```

---

## 🏗️ Structure du projet

- **`account_app/`** : Gestion des utilisateurs (User, Patient, Doctor)
- **`patient_app/`** : Gestion des ECG
- **`doctor_app/`** : Fonctionnalités spécifiques aux médecins

---

## 💾 Base de données

### Models principaux :
- **User** : Modèle d'authentification personnalisé
- **Patient** : Stockage des informations patients
- **Doctor** : Stockage des informations médecins
- **ECG** : Stockage et analyse des ECG

### Relations :
- Un **médecin** peut avoir **plusieurs patients**.
- Un **patient** a **un seul médecin**.
- Un **patient** peut avoir **plusieurs ECG**.

---

## 👥 Contributeurs
- **Lucien LACHARMOISE**
- **Clément ASENSIO**
- **Emma COCO**

---

## 📝 Notes de développement
- **Utiliser Python 3.9+**
- **Django 4.2+**
- Les **sessions** sont stockées dans des fichiers plutôt que dans la base de données.
- Consultez `requirements.txt` pour la liste complète des dépendances et leurs versions.
- Pour toute question ou problème, n'hésitez pas à **ouvrir une issue** sur le dépôt GitHub du projet.

**Bonne utilisation ! 🎉**

**Études menées lors de l'analyse des données MIT-BIH et de l'entrainement des réseaux de neurones"**

[Projet Doctolib - Rapport.docx](https://github.com/user-attachments/files/18769086/Projet.Doctolib.-.Rapport.docx)

[Rapport Doctolib_Réseaux de neurones.pdf](https://github.com/user-attachments/files/18769091/Rapport.Doctolib_Reseaux.de.neurones.pdf)




