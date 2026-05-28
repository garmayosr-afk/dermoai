# -*- coding: utf-8 -*-
import sys, io, os, sqlite3, numpy as np
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

from flask import Flask, render_template, request, redirect, session, flash, g
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Dense, Flatten, Dropout, Input
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image

app = Flask(__name__)
app.secret_key = "dermoai_secret_2025"

UPLOAD_FOLDER = "static/uploads/"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

DB_PATH = "skin_cancer.db"

# ── SQLite helpers ────────────────────────────────────────
def get_db():
    db = getattr(g, '_database', None)
    if db is None:
        db = g._database = sqlite3.connect(DB_PATH)
        db.row_factory = sqlite3.Row
    return db

@app.teardown_appcontext
def close_db(exception):
    db = getattr(g, '_database', None)
    if db is not None:
        db.close()

def init_db():
    with app.app_context():
        db = sqlite3.connect(DB_PATH)
        db.execute("""CREATE TABLE IF NOT EXISTS users (
            id       INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            password TEXT NOT NULL
        )""")
        db.execute("""CREATE TABLE IF NOT EXISTS patients (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            name        TEXT,
            age         INTEGER,
            result      TEXT,
            probability REAL,
            image_path  TEXT,
            created_at  DATETIME DEFAULT CURRENT_TIMESTAMP
        )""")
        # Insert default admin user if not exists
        existing = db.execute("SELECT * FROM users WHERE username='admin'").fetchone()
        if not existing:
            db.execute("INSERT INTO users (username, password) VALUES ('admin','1234')")
        db.commit()
        db.close()
        print("[OK] Base de donnees SQLite initialisee.")

# ── Rebuild VGG16 + load weights ─────────────────────────
def build_model(weights_path):
    inputs  = Input(shape=(224, 224, 3), name="input_layer")
    base    = VGG16(weights=None, include_top=False, input_tensor=inputs)
    x       = Flatten(name="flatten")(base.output)
    x       = Dense(256, activation="relu", name="dense")(x)
    x       = Dropout(0.5, name="dropout")(x)
    outputs = Dense(1, activation="sigmoid", name="dense_1")(x)
    model   = Model(inputs=inputs, outputs=outputs)
    model.load_weights(weights_path, by_name=True, skip_mismatch=True)
    return model

print("[INFO] Chargement du modele VGG16...")
model = build_model("model/mon_modele_melanome_vgg16.h5")
print("[OK] Modele charge avec succes !")

init_db()

# ── LOGIN ────────────────────────────────────────────────
@app.route("/", methods=["GET", "POST"])
def login():
    if "user" in session:
        return redirect("/dashboard")
    if request.method == "POST":
        user = request.form["username"]
        pwd  = request.form["password"]
        db   = get_db()
        row  = db.execute(
            "SELECT * FROM users WHERE username=? AND password=?", (user, pwd)
        ).fetchone()
        if row:
            session["user"] = user
            flash("Connexion reussie", "success")
            return redirect("/dashboard")
        else:
            flash("Identifiants incorrects", "danger")
    return render_template("login.html")

# ── DASHBOARD ────────────────────────────────────────────
@app.route("/dashboard")
def dashboard():
    if "user" not in session:
        return redirect("/")
    db        = get_db()
    total     = db.execute("SELECT COUNT(*) FROM patients").fetchone()[0]
    malignant = db.execute("SELECT COUNT(*) FROM patients WHERE result='Malignant'").fetchone()[0]
    benign    = total - malignant
    return render_template("dashboard.html",
                           user=session["user"],
                           total=total,
                           malignant=malignant,
                           benign=benign)

# ── PREDICT ──────────────────────────────────────────────
@app.route("/predict", methods=["GET", "POST"])
def predict():
    if "user" not in session:
        return redirect("/")
    if request.method == "POST":
        try:
            name = request.form["name"]
            age  = request.form["age"]
            file = request.files["image"]

            if file.filename == "":
                flash("Veuillez choisir une image", "warning")
                return redirect("/predict")

            path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            file.save(path)

            img  = image.load_img(path, target_size=(224, 224))
            img  = image.img_to_array(img) / 255.0
            img  = np.expand_dims(img, axis=0)
            pred = model.predict(img)[0][0]
            res  = "Malignant" if pred > 0.5 else "Benign"

            db = get_db()
            db.execute(
                "INSERT INTO patients (name, age, result, probability, image_path) VALUES (?,?,?,?,?)",
                (name, age, res, float(pred), path)
            )
            db.commit()

            flash("Analyse reussie", "success")
            return render_template("result.html",
                                   result=res,
                                   prob=round(pred * 100, 2),
                                   img=path,
                                   name=name,
                                   age=age)
        except Exception as e:
            flash("Erreur systeme : " + str(e), "danger")
            return redirect("/predict")

    return render_template("predict.html")

# ── PATIENTS ─────────────────────────────────────────────
@app.route("/patients")
def patients():
    if "user" not in session:
        return redirect("/")
    db   = get_db()
    rows = db.execute("SELECT * FROM patients ORDER BY created_at DESC").fetchall()
    data = [dict(r) for r in rows]
    return render_template("patients.html", patients=data)

# ── LOGOUT ───────────────────────────────────────────────
@app.route("/logout")
def logout():
    session.clear()
    flash("Vous avez ete deconnecte", "info")
    return redirect("/")

if __name__ == "__main__":
    app.run(debug=True)
