# -*- coding: utf-8 -*-
import sys, io, os, sqlite3, numpy as np
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

from flask import Flask, render_template, request, redirect, session, flash, g
from datetime import datetime
from PIL import Image, ImageFilter, ImageStat
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

# ── XAI HELPERS ──────────────────────────────────────────
XAI_LAYER = 'block5_conv3'   # Last conv layer of VGG16

def generate_gradcam_heatmap(mdl, img_array, layer_name=XAI_LAYER):
    """Return a 2-D numpy heatmap (values 0-1) using Grad-CAM."""
    grad_model = tf.keras.models.Model(
        inputs=mdl.inputs,
        outputs=[mdl.get_layer(layer_name).output, mdl.output]
    )
    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(img_array)
        class_score = preds[:, 0]
    grads = tape.gradient(class_score, conv_out)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_out = conv_out[0]
    heatmap = conv_out @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0)
    heatmap = heatmap / (tf.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()

def _resize_heatmap(heatmap, size=(224, 224)):
    """Resize a 2-D float heatmap to the target size via PIL."""
    h_img = Image.fromarray((heatmap * 255).astype(np.uint8), mode='L')
    return np.array(h_img.resize(size, Image.BILINEAR)).astype(np.float32) / 255.0

def _apply_colormap(heatmap_2d, cmap_name='jet'):
    """Apply matplotlib colormap; fallback to manual jet if unavailable."""
    try:
        import matplotlib.cm as cm
        cmap = cm.get_cmap(cmap_name)
        return (cmap(heatmap_2d)[:, :, :3] * 255).astype(np.uint8)
    except Exception:
        # Manual jet fallback
        h, w = heatmap_2d.shape
        out = np.zeros((h, w, 3), dtype=np.uint8)
        for i in range(h):
            for j in range(w):
                v = float(heatmap_2d[i, j])
                out[i, j] = (
                    int(np.clip(1.5 - abs(4*v-3), 0, 1)*255),
                    int(np.clip(1.5 - abs(4*v-2), 0, 1)*255),
                    int(np.clip(1.5 - abs(4*v-1), 0, 1)*255),
                )
        return out

def save_gradcam_overlay(orig_path, heatmap, out_path, alpha=0.45):
    """Grad-CAM: jet heatmap blended over the original image."""
    orig = Image.open(orig_path).convert('RGB').resize((224, 224))
    orig_arr = np.array(orig, dtype=np.float32)
    colored = _apply_colormap(_resize_heatmap(heatmap), 'jet').astype(np.float32)
    overlay = np.clip(orig_arr*(1-alpha) + colored*alpha, 0, 255).astype(np.uint8)
    Image.fromarray(overlay).save(out_path)

def save_pure_heatmap(heatmap, out_path):
    """Save the raw Grad-CAM heatmap with an inferno colormap."""
    colored = _apply_colormap(_resize_heatmap(heatmap), 'inferno')
    Image.fromarray(colored).save(out_path)

def save_attention_map(orig_path, heatmap, out_path):
    """Attention map: original pixels weighted by Grad-CAM intensity."""
    orig = Image.open(orig_path).convert('RGB').resize((224, 224))
    orig_arr = np.array(orig, dtype=np.float32)
    attention = 0.15 + 0.85 * _resize_heatmap(heatmap)   # keep some context
    weighted = np.clip(orig_arr * attention[:, :, np.newaxis], 0, 255).astype(np.uint8)
    Image.fromarray(weighted).save(out_path)

def assess_image_quality(path):
    """Estimate whether the uploaded lesion image is sharp, exposed, and usable."""
    img = Image.open(path).convert("RGB")
    width, height = img.size
    gray = img.convert("L")
    stat = ImageStat.Stat(gray)
    brightness = stat.mean[0]
    contrast = stat.stddev[0]
    edges = gray.filter(ImageFilter.FIND_EDGES)
    sharpness = ImageStat.Stat(edges).stddev[0]

    score = 100
    tips = []
    if min(width, height) < 224:
        score -= 24
        tips.append("Utilisez une image d'au moins 224 x 224 pixels.")
    if brightness < 55:
        score -= 18
        tips.append("Image sombre : ameliorez l'eclairage.")
    elif brightness > 205:
        score -= 18
        tips.append("Image surexposee : reduisez le flash ou les reflets.")
    if contrast < 28:
        score -= 18
        tips.append("Contraste faible : centrez la lesion sur une peau visible.")
    if sharpness < 16:
        score -= 22
        tips.append("Flou possible : reprenez l'image avec une meilleure mise au point.")
    if not tips:
        tips.append("Qualite adaptee pour un premier screening IA.")

    score = max(0, min(100, int(round(score))))
    label = "Excellent" if score >= 85 else "Bon" if score >= 70 else "Moyen" if score >= 55 else "A revoir"
    return {
        "score": score,
        "label": label,
        "tips": tips,
        "brightness": round(brightness, 1),
        "contrast": round(contrast, 1),
        "sharpness": round(sharpness, 1),
        "resolution": f"{width} x {height}",
    }

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
            image_quality = assess_image_quality(path)
            if image_quality["score"] < 55:
                flash("Qualite d'image faible : verifiez la lumiere, le focus et la resolution.", "warning")

            img_arr = image.load_img(path, target_size=(224, 224))
            img_arr = image.img_to_array(img_arr) / 255.0
            img_arr = np.expand_dims(img_arr, axis=0)
            pred = model.predict(img_arr)[0][0]
            res  = "Malignant" if pred > 0.5 else "Benign"

            db = get_db()
            db.execute(
                "INSERT INTO patients (name, age, result, probability, image_path) VALUES (?,?,?,?,?)",
                (name, age, res, float(pred), path)
            )
            db.commit()

            # ── Generate XAI visualisations ──────────────────
            base     = os.path.splitext(file.filename)[0]
            ext      = os.path.splitext(file.filename)[1] or '.png'
            gc_path  = os.path.join(UPLOAD_FOLDER, f"gradcam_{base}{ext}")
            hm_path  = os.path.join(UPLOAD_FOLDER, f"heatmap_{base}{ext}")
            att_path = os.path.join(UPLOAD_FOLDER, f"attention_{base}{ext}")

            try:
                heatmap = generate_gradcam_heatmap(model, img_arr)
                save_gradcam_overlay(path, heatmap, gc_path)
                save_pure_heatmap(heatmap, hm_path)
                save_attention_map(path, heatmap, att_path)
                xai_ok = True
            except Exception as xai_err:
                print("[XAI] Error:", xai_err)
                xai_ok = False

            flash("Analyse reussie", "success")
            return render_template("result.html",
                                   result=res,
                                   prob=round(pred * 100, 2),
                                   img=path,
                                   name=name,
                                   age=age,
                                   gradcam=gc_path  if xai_ok else None,
                                   heatmap=hm_path  if xai_ok else None,
                                   attention=att_path if xai_ok else None,
                                   image_quality=image_quality)
        except Exception as e:
            flash("Erreur systeme : " + str(e), "danger")
            return redirect("/predict")

    latest_scans = get_db().execute("SELECT * FROM patients ORDER BY created_at DESC LIMIT 3").fetchall()
    return render_template("predict.html", latest_scans=latest_scans)

# ── PATIENTS ─────────────────────────────────────────────
@app.route("/patients")
def patients():
    if "user" not in session:
        return redirect("/")
    db   = get_db()
    rows = db.execute("SELECT * FROM patients ORDER BY created_at DESC").fetchall()
    data = []
    for r in rows:
        row = dict(r)
        if row.get("created_at") and isinstance(row["created_at"], str):
            try:
                row["created_at"] = datetime.strptime(row["created_at"], "%Y-%m-%d %H:%M:%S")
            except ValueError:
                row["created_at"] = None
        data.append(row)
    return render_template("patients.html", patients=data)

# ── LOGOUT ───────────────────────────────────────────────
@app.route("/logout")
def logout():
    session.clear()
    flash("Vous avez ete deconnecte", "info")
    return redirect("/")

if __name__ == "__main__":
    app.run(debug=True)
