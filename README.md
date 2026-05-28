# DermoAI — Détection du Cancer de la Peau par IA

Application web Flask intégrant un modèle **VGG16** pour la détection du cancer de la peau (mélanome).

> 🏫 TD 8 — ENSTAB · Module : Introduction à l'IA · Année 2025/2026

---

## 🚀 Fonctionnalités

- 🔐 **Authentification** sécurisée (session Flask)
- 📊 **Dashboard** avec statistiques en temps réel
- 🔬 **Analyse IA** — upload d'image + diagnostic instantané (VGG16)
- 🗂️ **Historique patients** avec recherche et filtre
- 🎨 **Interface premium** — Dark mode, glassmorphism, animations

---

## 🛠️ Stack technique

| Composant | Technologie |
|---|---|
| Backend | Python 3 + Flask |
| IA | TensorFlow / Keras — VGG16 |
| Base de données | SQLite (intégré) |
| Frontend | HTML5 + CSS3 (Vanilla, glassmorphism) |

---

## ⚙️ Installation

```bash
# 1. Cloner le dépôt
git clone https://github.com/garmayosr-afk/dermoai.git
cd dermoai

# 2. Créer un environnement virtuel
python -m venv venv
.\venv\Scripts\activate     # Windows
# source venv/bin/activate  # Linux/Mac

# 3. Installer les dépendances
pip install flask tensorflow numpy pillow

# 4. Placer le modèle dans model/
# → model/mon_modele_melanome_vgg16.h5

# 5. Lancer l'application
python app.py
```

L'application sera accessible sur **http://127.0.0.1:5000**

---

## 🔑 Connexion par défaut

| Identifiant | Mot de passe |
|---|---|
| `admin` | `1234` |

---

## 📁 Structure du projet

```
dermoai/
├── app.py                    ← Application Flask principale
├── database.sql              ← Schéma MySQL (référence)
├── requirements.txt          ← Dépendances Python
├── model/
│   └── mon_modele_melanome_vgg16.h5  ← Modèle VGG16 (à fournir)
├── static/
│   ├── style.css             ← Design premium dark
│   └── uploads/              ← Images analysées (auto-créé)
└── templates/
    ├── login.html
    ├── dashboard.html
    ├── predict.html
    ├── result.html
    └── patients.html
```

---

## 🤖 Modèle IA

- Architecture : **VGG16** (poids ImageNet + fine-tuning mélanome)
- Entrée : image 224×224 pixels normalisée (÷255)
- Sortie : probabilité sigmoid → **Malignant** (>0.5) / **Benign** (≤0.5)
