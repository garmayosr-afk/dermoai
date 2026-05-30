# DermoAI — Détection du Cancer de la Peau par IA

Application web Flask intégrant un modèle **VGG16** pour la détection du cancer de la peau (mélanome).

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

##app demo: 
<img width="1837" height="878" alt="Screenshot 2026-05-30 201436" src="https://github.com/user-attachments/assets/6d84583f-ed00-4d03-b56b-3248e98bf368" />
<img width="1841" height="889" alt="Screenshot 2026-05-30 201458" src="https://github.com/user-attachments/assets/001df2b8-cd8d-4ae2-a20f-917ba26047a3" />
<img width="1838" height="886" alt="Screenshot 2026-05-30 201514" src="https://github.com/user-attachments/assets/9135d3cc-8633-4096-b3c2-259f8c03621c" />
<img width="1827" height="879" alt="Screenshot 2026-05-30 201532" src="https://github.com/user-attachments/assets/854ceaa5-4c6c-4417-98a3-a04f95594387" />
<img width="1833" height="881" alt="Screenshot 2026-05-30 201616" src="https://github.com/user-attachments/assets/47e35009-346a-46be-b011-a69ea36e96eb" />
<img width="1823" height="887" alt="Screenshot 2026-05-30 201723" src="https://github.com/user-attachments/assets/7a2a7ce1-cbbd-4982-825d-5e3500774314" />
<img width="1834" height="885" alt="Screenshot 2026-05-30 201744" src="https://github.com/user-attachments/assets/dee4f1d0-8a90-4d41-a7a7-9f2d3cef5c1e" />




## 🤖 Modèle IA

- Architecture : **VGG16** (poids ImageNet + fine-tuning mélanome)
- Entrée : image 224×224 pixels normalisée (÷255)
- Sortie : probabilité sigmoid → **Malignant** (>0.5) / **Benign** (≤0.5)
