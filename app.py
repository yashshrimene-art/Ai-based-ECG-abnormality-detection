# streamlit.py
import streamlit as st
import numpy as np
import pandas as pd
import pickle
from PIL import Image
import matplotlib.pyplot as plt
import json, os, hashlib
from datetime import datetime, timedelta

# ---------------- CONFIG ----------------
st.set_page_config(page_title="ECG Detection", layout="wide")

# ---------------- STYLE ----------------
st.markdown("""
<style>
html, body {
    background: linear-gradient(135deg,#fff1f7,#ffe4ec);
    font-family: 'Segoe UI';
}
.header {
    font-size:32px;
    font-weight:700;
    color:#ec4899;
}
.card {
    background:white;
    padding:20px;
    border-radius:15px;
    box-shadow:0px 5px 15px rgba(0,0,0,0.05);
    margin-bottom:20px;
}
.good {background:#ecfdf5;padding:10px;border-left:5px solid green;}
.bad {background:#fef2f2;padding:10px;border-left:5px solid red;}
.alert {background:#fff0f3;padding:10px;border-left:5px solid #ec4899;}
</style>
""", unsafe_allow_html=True)

# ---------------- FILES ----------------
USER="users.json"
DATA="data.json"

for f in [USER, DATA]:
    if not os.path.exists(f):
        json.dump({}, open(f, "w"))

def load(f): return json.load(open(f))
def save(f, d): json.dump(d, open(f, "w"))
def hash_pwd(p): return hashlib.sha256(p.encode()).hexdigest()

# ---------------- SESSION ----------------
if "login" not in st.session_state:
    st.session_state.login = False
    st.session_state.user = None

# ---------------- LOAD MODELS ----------------
# CSV MODEL
try:
    rf_model = pickle.load(open("ecg_model_new.pkl", "rb"))
    csv_ok = True
except:
    csv_ok = False

# IMAGE MODEL
from tensorflow.keras.models import load_model

try:
    cnn_model = load_model("ecg_image_model.h5", compile=False)
    img_ok = True
except:
    img_ok = False

# LOAD CLASS LABELS
try:
    class_indices = pickle.load(open("class_indices.pkl", "rb"))
    classes = {v: k for k, v in class_indices.items()}
except:
    classes = {}

# ---------------- MENU ----------------
menu = st.sidebar.radio("Menu", [
    "Login", "Signup", "Home", "Profile", "Dashboard", "Analyze ECG", "Logout"
])

# ---------------- HOME ----------------
if menu == "Home":
    st.markdown('<div class="header">ECG Detection</div>', True)

    st.markdown('<div class="card">', True)
    st.write("""
    💗 AI-powered ECG Detection System  
    ✔ Upload ECG CSV or Image  
    ✔ Instant Diagnosis  
    ✔ Track Heart Health  
    ✔ Smart Reminders  
    """)
    st.markdown('</div>', True)

# ---------------- PROFILE ----------------
elif menu == "Profile":
    if not st.session_state.login:
        st.warning("Login first")
    else:
        db = load(DATA)
        user = db.setdefault(st.session_state.user, {})

        st.subheader("Patient Profile")
        age = st.number_input("Age", 1, 100, 25)
        gender = st.selectbox("Gender", ["Male", "Female", "Other"])

        if st.button("Save Profile"):
            user["age"] = age
            user["gender"] = gender
            save(DATA, db)
            st.success("Profile Saved")

# ---------------- DASHBOARD ----------------
elif menu == "Dashboard":
    if not st.session_state.login:
        st.warning("Login required")
    else:
        db = load(DATA)
        user = db.get(st.session_state.user, {})

        hist = user.get("history", [])
        nxt = user.get("next", None)

        st.metric("Total Tests", len(hist))
        st.metric("Abnormal Cases", hist.count("Abnormal"))

        if nxt:
            nxt_date = datetime.strptime(nxt, "%Y-%m-%d")
            if nxt_date.date() == (datetime.today() + timedelta(days=1)).date():
                st.markdown('<div class="alert">🔔 ECG test tomorrow</div>', True)

        st.subheader("Schedule Next ECG")
        d = st.date_input("Select Date")

        if st.button("Save Date"):
            user["next"] = str(d)
            save(DATA, db)
            st.success("Saved")

# ---------------- ANALYZE ----------------
elif menu == "Analyze ECG":
    if not st.session_state.login:
        st.warning("Login required")
    else:
        st.markdown('<div class="card">', True)

        option = st.radio("Select Input Type", ["ECG CSV", "ECG Image"])

        # ===== CSV =====
        if option == "ECG CSV":
            file = st.file_uploader("Upload ECG CSV", type=["csv"])

            if file and csv_ok:
                data = pd.read_csv(file, header=None)

                fig, ax = plt.subplots(figsize=(10, 4))
                ax.plot(data.iloc[0, :-1])
                ax.set_title("ECG Signal")
                st.pyplot(fig)

                sample = data.iloc[0, :-1].values.reshape(1, -1)

                if st.button("Analyze CSV"):
                    pred = rf_model.predict(sample)[0]
                    probs = rf_model.predict_proba(sample)[0]

                    confidence = float(np.max(probs))
                    label = "Normal" if pred == 0 else "Abnormal"

                    st.subheader("Report")
                    st.write("Diagnosis:", label)
                    st.write("Confidence:", round(confidence, 2))

                    if label == "Normal":
                        st.markdown('<div class="good">✔ Normal ECG</div>', True)
                    else:
                        st.markdown('<div class="bad">⚠ Abnormal ECG</div>', True)

        # ===== IMAGE =====
        elif option == "ECG Image":
            file = st.file_uploader("Upload ECG Image", type=["jpg", "png", "jpeg"])

            if file:
                if not img_ok:
                    st.error("Image model not available")
                else:
                    image = Image.open(file).convert("RGB").resize((128, 128))
                    st.image(image)

                    img = np.array(image) / 255.0
                    img = np.expand_dims(img, axis=0)

                    if st.button("Analyze Image"):
                        pred = cnn_model.predict(img)
                        class_idx = np.argmax(pred)
                        confidence = float(np.max(pred))

                        # 🔥 CLASS → NORMAL / ABNORMAL
                        raw_label = classes.get(class_idx, "Unknown")

                        if raw_label.lower() == "n":
                            final_label = "Normal"
                        else:
                            final_label = "Abnormal"

                        st.subheader("Report")
                        st.write("Diagnosis:", final_label)
                        st.write("Confidence:", round(confidence, 2))

                        if final_label == "Normal":
                            st.markdown('<div class="good">✔ Normal ECG</div>', True)
                        else:
                            st.markdown('<div class="bad">⚠ Abnormal ECG</div>', True)

        st.markdown('</div>', True)

# ---------------- LOGIN ----------------
elif menu == "Login":
    u = st.text_input("Username")
    p = st.text_input("Password", type="password")

    if st.button("Login"):
        users = load(USER)
        if u in users and users[u] == hash_pwd(p):
            st.session_state.login = True
            st.session_state.user = u
            st.success("Login Successful")
        else:
            st.error("Invalid Credentials")

# ---------------- SIGNUP ----------------
elif menu == "Signup":
    u = st.text_input("Username")
    p = st.text_input("Password", type="password")

    if st.button("Register"):
        users = load(USER)
        users[u] = hash_pwd(p)
        save(USER, users)
        st.success("Account Created")

# ---------------- LOGOUT ----------------
elif menu == "Logout":
    st.session_state.login = False
    st.success("Logged out")