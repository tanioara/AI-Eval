import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import requests
import time
from model_utils import HousingEvaluatorBackend, FEATURE_MAP

# Configurare pagină
st.set_page_config(
    page_title="Evaluator Consistență Factuală LLM",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inițializare backend în starea sesiunii Streamlit pentru persistența datelor
if "backend" not in st.session_state:
    st.session_state.backend = HousingEvaluatorBackend()
    # Antrenare inițială automată pentru a nu lăsa interfața goală
    st.session_state.backend.train_model()

backend = st.session_state.backend

# Injectare CSS personalizat pentru design premium (Dark Mode, Neon Accents, Glassmorphism)
st.markdown("""
<style>
    /* Fundalul general al aplicației */
    .stApp {
        background-color: #0b0f19;
        color: #f3f4f6;
    }
    
    /* Stil pentru sidebar */
    section[data-testid="stSidebar"] {
        background-color: #0e1626;
        border-right: 1px solid rgba(255, 255, 255, 0.05);
    }
    
    /* Titluri și text */
    h1, h2, h3 {
        font-family: 'Inter', 'Outfit', sans-serif;
        font-weight: 700;
        letter-spacing: -0.02em;
    }
    
    .main-title {
        background: linear-gradient(135deg, #06b6d4 0%, #8b5cf6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.8rem;
        margin-bottom: 0.2rem;
    }
    
    .subtitle {
        color: #9ca3af;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    
    /* Design de tip Glassmorphic Card */
    .glass-card {
        background: rgba(22, 30, 48, 0.6);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.25);
        margin-bottom: 20px;
    }
    
    .glass-card-header {
        font-size: 1.25rem;
        font-weight: 600;
        color: #06b6d4;
        margin-bottom: 15px;
        border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        padding-bottom: 8px;
    }
    
    /* Badge-uri de stare */
    .status-badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        margin-right: 10px;
    }
    .status-ok {
        background-color: rgba(16, 185, 129, 0.15);
        color: #10b981;
        border: 1px solid rgba(16, 185, 129, 0.3);
    }
    .status-warning {
        background-color: rgba(245, 158, 11, 0.15);
        color: #f59e0b;
        border: 1px solid rgba(245, 158, 11, 0.3);
    }
    .status-error {
        background-color: rgba(239, 68, 68, 0.15);
        color: #ef4444;
        border: 1px solid rgba(239, 68, 68, 0.3);
    }
    
    /* Stiluri taburi Streamlit */
    button[data-baseweb="tab"] {
        font-size: 1rem !important;
        font-weight: 500 !important;
        color: #9ca3af !important;
        background-color: transparent !important;
        border-bottom: 2px solid transparent !important;
        padding: 10px 16px !important;
        transition: all 0.3s ease !important;
    }
    button[data-baseweb="tab"]:hover {
        color: #06b6d4 !important;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        color: #06b6d4 !important;
        border-bottom: 2px solid #06b6d4 !important;
        font-weight: 600 !important;
    }
    
    /* Box explicație LLM */
    .llm-response-box {
        background-color: rgba(139, 92, 246, 0.05);
        border-left: 4px solid #8b5cf6;
        padding: 15px;
        border-radius: 0 8px 8px 0;
        font-style: italic;
        line-height: 1.6;
        margin-top: 15px;
        margin-bottom: 15px;
    }
</style>
""", unsafe_allow_html=True)


# Verificare conexiune locală la Ollama
def check_ollama_online():
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=1.0)
        return response.status_code == 200
    except:
        return False

ollama_online = check_ollama_online()

# ----------------- SIDEBAR -----------------
st.sidebar.markdown("<h2 style='text-align: center; color: #06b6d4;'>🎛️ Panou Control</h2>", unsafe_allow_html=True)
st.sidebar.markdown("---")

# 1. Indicatori de Stare
st.sidebar.markdown("### Stare Sistem")

# Stare Model ML
if backend.is_trained:
    st.sidebar.markdown('<span class="status-badge status-ok">● Random Forest Antrenat</span>', unsafe_allow_html=True)
else:
    st.sidebar.markdown('<span class="status-badge status-error">○ Model Neantrenat</span>', unsafe_allow_html=True)

# Stare Ollama
if ollama_online:
    st.sidebar.markdown('<span class="status-badge status-ok">● Ollama Online</span>', unsafe_allow_html=True)
else:
    st.sidebar.markdown('<span class="status-badge status-warning">▲ Ollama Offline (Simulat)</span>', unsafe_allow_html=True)

st.sidebar.markdown("---")

# 2. Configurare LLM
st.sidebar.markdown("### Configurare LLM Local")
model_option = st.sidebar.selectbox(
    "Alege modelul din Ollama:",
    ["llama3.2", "mistral", "llama3", "gemma"],
    index=0
)

# Mod operare forțat (pentru testare fără Ollama)
run_mode = st.sidebar.radio(
    "Mod Rulare:",
    ["Automat (Detectează Ollama)", "Forțează Mod Simulat (Offline Demo)"],
    index=0
)
force_sim = (run_mode == "Forțează Mod Simulat (Offline Demo)") or not ollama_online

st.sidebar.markdown("---")

# 3. Buton Retrain
st.sidebar.markdown("### Re-antrenare Model ML")
if st.sidebar.button("Re-antrenează Random Forest"):
    with st.sidebar.spinner("Se antrenează modelul..."):
        metrics, _ = backend.train_model()
        st.sidebar.success("Model antrenat cu succes!")
        st.rerun()

# Ghid instalare rapidă Ollama în Sidebar dacă e offline
if not ollama_online and not force_sim:
    st.sidebar.warning("Pentru conectarea Ollama real:\n1. Instalează din terminal: `brew install ollama`\n2. Porneste serviciul: `brew services start ollama`\n3. Descarcă modelul: `ollama run llama3.2`\n4. Reîncarcă această pagină.")

# ----------------- MAIN PANEL -----------------
st.markdown("<h1 class='main-title'>🔍 Factual Consistency Evaluator</h1>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Studiu de caz: Sunt explicațiile LLM-ului în concordanță cu decizia matematică a modelului ML de regresie, sau halucinează?</div>", unsafe_allow_html=True)

# Organizare pe Tab-uri
tab_intro, tab_model, tab_single, tab_batch = st.tabs([
    "📖 Despre Studiu", 
    "📊 Model ML & Importanță Globală", 
    "🏡 Analiză & Explicație Individuală", 
    "📈 Evaluare Batch (Scor Consistență)"
])

# ================= TAB 1: DESPRE STUDIU =================
with tab_intro:
    st.markdown("""
    <div class="glass-card">
        <div class="glass-card-header">💡 Obiectivul Cercetării</div>
        Modelele mari de limbaj (LLMs) sunt folosite din ce în ce mai mult pentru a explica deciziile algoritmilor de tip "cutie neagră" (Black-Box ML). 
        Cu toate acestea, <b>un LLM nu cunoaște în mod nativ coeficienții matematici</b> ai modelului predictiv. Când i se cere să explice o decizie (ex: <i>"De ce este această casă scumpă?"</i>), LLM-ul tinde să creeze o poveste plauzibilă bazată pe cunoștințele sale generale (de exemplu, presupune că o casă mare e scumpă, sau că locația e totul), chiar dacă modelul matematic a luat decizia pe baza unui alt factor (ex: vârsta casei sau venitul median al zonei).
        <br><br>
        Acest experiment măsoară în mod direct <b>Rata de Consistență Factuală</b> a explicațiilor generate de LLM față de <b>importanța locală a caracteristicilor (SHAP-like)</b> calculată riguros prin perturbare matematică.
    </div>
    """, unsafe_allow_html=True)

    # Grid pentru arhitectura de flux
    st.markdown("### 🔄 Diagrama de Flux a Experimentului")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("""
        <div class="glass-card" style="text-align: center; height: 180px;">
            <div style="font-size: 2rem;">🤖 1. ML Model</div>
            <p style="font-size: 0.9rem; color: #9ca3af; margin-top: 10px;">
                Un model Random Forest antrenat pe California Housing prezice prețul unei locuințe.
            </p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="glass-card" style="text-align: center; height: 180px;">
            <div style="font-size: 2rem;">🧮 2. SHAP (Truth)</div>
            <p style="font-size: 0.9rem; color: #9ca3af; margin-top: 10px;">
                Calculăm importanța locală a factorilor prin perturbare: ce caracteristici au influențat efectiv predicția?
            </p>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="glass-card" style="text-align: center; height: 180px;">
            <div style="font-size: 2rem;">🗣️ 3. LLM Explain</div>
            <p style="font-size: 0.9rem; color: #9ca3af; margin-top: 10px;">
                LLM-ul primește datele și prețul, generând o explicație naturală în limba română.
            </p>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown("""
        <div class="glass-card" style="text-align: center; height: 180px;">
            <div style="font-size: 2rem;">⚖️ 4. Evaluare</div>
            <p style="font-size: 0.9rem; color: #9ca3af; margin-top: 10px;">
                Extragem termenii cheie menționați de LLM și îi comparăm matematic cu importanțele din model.
            </p>
        </div>
        """, unsafe_allow_html=True)

    # Detalii metrici
    st.markdown("""
    <div class="glass-card">
        <div class="glass-card-header">📊 Metrice de Evaluare Definite</div>
        <ul>
            <li><b>Rata de Consistență Factuală (Factual Consistency Rate):</b> Procentul de factori din textul LLM-ului care corespund cu cei mai importanți 3 factori din model.
                <br><small style="color:#9ca3af;">Formula: (Factori Corecți Menționați) / (Total Factori Menționați de LLM)</small></li>
            <li style="margin-top: 10px;"><b>Recall Factor Primar (Primary Factor Recall):</b> Dacă LLM-ul a reușit sau nu să identifice cel mai important factor determinat matematic de model (Scor 1 sau 0 per casă).</li>
            <li style="margin-top: 10px;"><b>Rata de Halucinație (Hallucination Rate):</b> Menționarea unor caracteristici drept cauze ale prețului, deși importanța lor matematică locală în model este neglijabilă.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)


# ================= TAB 2: MODEL ML & IMPORTANȚĂ GLOBALĂ =================
with tab_model:
    if backend.is_trained:
        st.markdown("### 📊 Performanța Globală a Modelului Random Forest")
        
        # Grid Metrici
        col_r2, col_mae, col_mse = st.columns(3)
        with col_r2:
            st.markdown(f"""
            <div class="glass-card" style="text-align: center;">
                <div style="color: #9ca3af; font-size: 0.9rem; text-transform: uppercase; font-weight: 600;">Scor R² (Coeficient Determinare)</div>
                <div style="color: #06b6d4; font-size: 2.5rem; font-weight: 700; margin: 10px 0;">{backend.metrics['r2']:.4f}</div>
                <div style="font-size: 0.85rem; color: #10b981;">Aproximativ {backend.metrics['r2']*100:.1f}% din variația prețului este explicată de model.</div>
            </div>
            """, unsafe_allow_html=True)
            
        with col_mae:
            st.markdown(f"""
            <div class="glass-card" style="text-align: center;">
                <div style="color: #9ca3af; font-size: 0.9rem; text-transform: uppercase; font-weight: 600;">Eroare Medie Absolută (MAE)</div>
                <div style="color: #8b5cf6; font-size: 2.5rem; font-weight: 700; margin: 10px 0;">${backend.metrics['mae']:,.0f}</div>
                <div style="font-size: 0.85rem; color: #9ca3af;">Abaterea medie a predicțiilor față de prețurile reale ale casei.</div>
            </div>
            """, unsafe_allow_html=True)
            
        with col_mse:
            st.markdown(f"""
            <div class="glass-card" style="text-align: center;">
                <div style="color: #9ca3af; font-size: 0.9rem; text-transform: uppercase; font-weight: 600;">Preț Mediu în Dataset</div>
                <div style="color: #10b981; font-size: 2.5rem; font-weight: 700; margin: 10px 0;">${backend.metrics['mean_price']:,.0f}</div>
                <div style="font-size: 0.85rem; color: #9ca3af;">Deviația standard: ${backend.metrics['std_price']:,.0f}</div>
            </div>
            """, unsafe_allow_html=True)

        # Importanța Globală a Caracteristicilor
        st.markdown("### 🌎 Care sunt cei mai importanți factori la nivel global?")
        st.markdown("Această diagramă arată importanța medie a fiecărei caracteristici pe întregul set de date California Housing, determinată prin algoritmul Random Forest.")
        
        # Formatare date pentru grafic
        imp_data = []
        for name, value in backend.global_importances.items():
            imp_data.append({
                "Caracteristică": FEATURE_MAP[name]["ro"],
                "Nume Tehnic": name,
                "Importanță Relativă": value
            })
        df_imp = pd.DataFrame(imp_data)
        
        # Altair chart
        chart = alt.Chart(df_imp).mark_bar(
            cornerRadiusBottomRight=4,
            cornerRadiusTopRight=4,
            color='#06b6d4'
        ).encode(
            x=alt.X('Importanță Relativă:Q', title='Importanță Relativă (Gini)'),
            y=alt.Y('Caracteristică:N', sort='-x', title='Caracteristică (RO)'),
            tooltip=['Nume Tehnic:N', 'Importanță Relativă:Q']
        ).properties(
            height=300
        ).configure_axis(
            grid=False,
            labelColor='#f3f4f6',
            titleColor='#f3f4f6'
        ).configure_view(
            strokeWidth=0
        )
        
        st.altair_chart(chart, use_container_width=True)
        
    else:
        st.warning("Modelul nu este antrenat. Te rog apasă pe butonul 'Re-antrenează Random Forest' din meniul lateral.")


# ================= TAB 3: EXPLICAȚIE INDIVIDUALĂ =================
with tab_single:
    st.markdown("### 🏡 Testare pe o Singură Locuință")
    st.markdown("Selectează o casă de test, rulează predicția modelului și cere LLM-ului o explicație. Vom compara dacă LLM-ul a nimerit factorii importanți.")

    # Obținere case
    houses = backend.get_test_houses(n=50)
    
    # Dropdown selectie casa
    house_options = {i: f"Casa ID {h['id']} - Preț Real: ${h['actual_price']:,.0f}" for i, h in enumerate(houses)}
    selected_idx = st.selectbox(
        "Alege o locuință din setul de test:",
        options=list(house_options.keys()),
        format_func=lambda x: house_options[x]
    )
    
    house = houses[selected_idx]
    features = house["features"]
    
    # Afisare date casa intr-un grid
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.markdown("#### 📋 Caracteristicile Casei Selectate")
    
    col_feat1, col_feat2, col_feat3, col_feat4 = st.columns(4)
    with col_feat1:
        st.metric(
            label=f"🔑 {FEATURE_MAP['MedInc']['ro']}", 
            value=f"${features['MedInc']*10:.1f}k / an",
            help=FEATURE_MAP['MedInc']['desc']
        )
        st.metric(
            label=f"🏠 {FEATURE_MAP['HouseAge']['ro']}", 
            value=f"{features['HouseAge']:.0f} ani",
            help=FEATURE_MAP['HouseAge']['desc']
        )
    with col_feat2:
        st.metric(
            label=f"🚪 {FEATURE_MAP['AveRooms']['ro']}", 
            value=f"{features['AveRooms']:.2f} camere",
            help=FEATURE_MAP['AveRooms']['desc']
        )
        st.metric(
            label=f"🛏️ {FEATURE_MAP['AveBedrms']['ro']}", 
            value=f"{features['AveBedrms']:.2f} dormitoare",
            help=FEATURE_MAP['AveBedrms']['desc']
        )
    with col_feat3:
        st.metric(
            label=f"👥 {FEATURE_MAP['Population']['ro']}", 
            value=f"{int(features['Population']):,}",
            help=FEATURE_MAP['Population']['desc']
        )
        st.metric(
            label=f"👨‍👩‍👧 {FEATURE_MAP['AveOccup']['ro']}", 
            value=f"{features['AveOccup']:.2f} persoane",
            help=FEATURE_MAP['AveOccup']['desc']
        )
    with col_feat4:
        st.metric(
            label=f"🌐 {FEATURE_MAP['Latitude']['ro']}", 
            value=f"{features['Latitude']:.3f}° N",
            help=FEATURE_MAP['Latitude']['desc']
        )
        st.metric(
            label=f"🧭 {FEATURE_MAP['Longitude']['ro']}", 
            value=f"{features['Longitude']:.3f}° W",
            help=FEATURE_MAP['Longitude']['desc']
        )
    st.markdown("</div>", unsafe_allow_html=True)

    # Generare Predicție și Calcul local
    pred_price, y_base, local_contribs = backend.get_local_contributions(features)
    
    col_pred, col_shap = st.columns([1, 1])
    
    with col_pred:
        st.markdown("""
        <div class="glass-card" style="height: 100%;">
            <div class="glass-card-header">🔮 Predicția Modelului de Regresie</div>
        """, unsafe_allow_html=True)
        
        # Comparare preț real vs preț prezis
        diff = pred_price - house['actual_price']
        diff_pct = (diff / house['actual_price']) * 100.0
        
        st.metric(
            label="Preț Predis de Random Forest",
            value=f"${pred_price:,.0f}",
            delta=f"${diff:,.0f} ({diff_pct:+.1f}%) față de prețul real"
        )
        
        st.markdown(f"""
            <div style="margin-top: 20px;">
                <p><b>Preț Real:</b> ${house['actual_price']:,.0f}</p>
                <p><b>Predicția pentru o casă medie globală (Baseline):</b> ${y_base:,.0f}</p>
                <p style="font-size: 0.9rem; color: #9ca3af;">
                    Prețul acestei case este cu <b>${abs(pred_price - y_base):,.0f} {'mai mare' if pred_price > y_base else 'mai mic'}</b> 
                    decât prețul mediu global al zonei, ca urmare a factorilor locali analizați în dreapta.
                </p>
            </div>
        """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
    with col_shap:
        st.markdown("""
        <div class="glass-card" style="height: 100%;">
            <div class="glass-card-header">🧮 Contribuțiile Locale ale Caracteristicilor (SHAP-like)</div>
        """, unsafe_allow_html=True)
        st.markdown("Graficul arată cu câți dolari ($) a influențat fiecare caracteristică prețul final în raport cu prețul de referință (baseline).")
        
        # Creare dataframe pentru grafic
        shap_data = []
        for name, val in local_contribs.items():
            shap_data.append({
                "Caracteristică": FEATURE_MAP[name]["ro"],
                "Contribuție ($)": val,
                "Tip": "Pozitiv (Crește prețul)" if val >= 0 else "Negativ (Scade prețul)"
            })
        df_shap = pd.DataFrame(shap_data)
        
        # Grafic Altair cu bare colorate după tip (pozitiv/negativ)
        shap_chart = alt.Chart(df_shap).mark_bar(
            cornerRadius=4
        ).encode(
            x=alt.X('Contribuție ($):Q', title='Impact pe Preț (USD)'),
            y=alt.Y('Caracteristică:N', sort=alt.EncodingSortField(field='Contribuție ($)', op='sum', order='descending'), title=None),
            color=alt.Color('Tip:N', scale=alt.Scale(domain=['Pozitiv (Crește prețul)', 'Negativ (Scade prețul)'], range=['#10b981', '#ef4444']), legend=None),
            tooltip=['Caracteristică:N', alt.Tooltip('Contribuție ($):Q', format="$,.0f")]
        ).properties(
            height=250
        ).configure_axis(
            labelColor='#f3f4f6',
            titleColor='#f3f4f6'
        ).configure_view(
            strokeWidth=0
        )
        st.altair_chart(shap_chart, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # SECȚIUNEA DE EXPLICABILITATE LLM
    st.markdown("---")
    st.markdown("### 🧠 Generare Explicație de către LLM Local")
    
    col_btn, col_info_llm = st.columns([1, 2])
    with col_btn:
        generate_btn = st.button("Generează Explicație LLM", type="primary")
        
    with col_info_llm:
        if force_sim:
            st.info("ℹ️ Ollama nu este pornit sau rularea simulată este activată. Explicația va fi generată de emulatorul local în mod realist.")
        else:
            st.success(f"🤖 Conectat la Ollama local. Se va folosi modelul `{model_option}`.")

    if generate_btn:
        with st.spinner("Se procesează explicația..."):
            explanation_text, was_simulated = backend.generate_explanation(
                features, pred_price, local_contribs, model_name=model_option, force_simulation=force_sim
            )
            
            # Parsare si evaluare
            llm_factors = backend.parse_llm_explanation(explanation_text)
            eval_res = backend.evaluate_explanation_consistency(local_contribs, llm_factors)
            
            # Afișare rezultate
            st.markdown("#### 💬 Răspuns Generat de LLM")
            st.markdown(f"""
            <div class="llm-response-box">
                "{explanation_text}"
                <br>
                <span style="font-size: 0.8rem; color:#9ca3af; font-style: normal;">
                    — Mod generare: {'Simulare Fallback' if was_simulated else f'Ollama (model: {model_option})'}
                </span>
            </div>
            """, unsafe_allow_html=True)
            
            # Evaluare consistență
            st.markdown("#### ⚖️ Analiza Consistenței și Halucinațiilor")
            
            col_sc1, col_sc2, col_sc3 = st.columns(3)
            with col_sc1:
                cons_pct = eval_res['factual_consistency'] * 100.0
                st.metric(
                    label="Rata de Consistență Factuală",
                    value=f"{cons_pct:.1f}%",
                    help="Câți dintre factorii menționați de LLM sunt de fapt în top 3 factori reali ai modelului."
                )
            with col_sc2:
                rec_val = "DA (100%)" if eval_res['primary_recall'] > 0 else "NU (0%)"
                st.metric(
                    label="Recall Factor Primar",
                    value=rec_val,
                    help="Dacă LLM-ul a menționat sau nu cel mai important factor matematic al modelului."
                )
            with col_sc3:
                hall_count = len(eval_res['hallucinations'])
                st.metric(
                    label="Halucinații Detectate",
                    value=str(hall_count),
                    help="Numărul de caracteristici pe care LLM-ul le menționează ca importante, deși modelul le-a ignorat.",
                    delta=f"{hall_count} factori" if hall_count > 0 else None,
                    delta_color="inverse"
                )
                
            # Tabel comparativ
            st.markdown("##### Detalii comparare factori:")
            
            # Pregătim date pentru tabel
            comp_data = []
            
            # Top factori reali
            for name in eval_res["top_real_features"]:
                rank = eval_res["top_real_features"].index(name) + 1
                comp_data.append({
                    "Tip": "Real (Model ML)",
                    "Rang / Importanță": f"#{rank} - {FEATURE_MAP[name]['ro']}",
                    "Valoare Locală (Impact USD)": f"${local_contribs[name]:+,.0f}",
                    "Menționat de LLM?": "DA" if name in llm_factors or (name in ["Latitude", "Longitude"] and ("Latitude" in llm_factors or "Longitude" in llm_factors)) else "NU"
                })
                
            # Factori halucinați de LLM
            for name in eval_res["hallucinations"]:
                comp_data.append({
                    "Tip": "Halucinație (Doar LLM)",
                    "Rang / Importanță": f"Ignorat de model - {FEATURE_MAP[name]['ro']}",
                    "Valoare Locală (Impact USD)": f"${local_contribs[name]:+,.0f}",
                    "Menționat de LLM?": "DA"
                })
                
            df_comp = pd.DataFrame(comp_data)
            st.table(df_comp)


# ================= TAB 4: EVALUARE BATCH =================
with tab_batch:
    st.markdown("### 📈 Evaluare de Batch (Analiză Statistică pe 30-50 Locuințe)")
    st.markdown("""
    Pentru a formula o concluzie științifică cu privire la întrebarea de cercetare, nu ne putem baza pe o singură casă. 
    Vom rula o evaluare automată pe un set extins de locuințe de test (30-50 case). Aplicația va prezice prețul, va calcula factorii SHAP locali, va genera explicațiile LLM și le va compara automat.
    """)
    
    col_b1, col_b2 = st.columns([1, 2])
    with col_b1:
        batch_size = st.slider("Număr de case de evaluat:", min_value=10, max_value=50, value=30, step=5)
        run_batch_btn = st.button("Lansează Evaluare Batch", type="primary")
        
    with col_b2:
        st.markdown(f"""
        <div style="background-color: rgba(6, 182, 212, 0.05); padding: 15px; border-radius: 8px; border-left: 4px solid #06b6d4;">
            <b>Configurație curentă:</b><br>
            • Batch size: {batch_size} case de test distincte.<br>
            • Model LLM: <code>{model_option}</code><br>
            • Rulare în mod: <code>{'Simulat' if force_sim else 'Ollama Real'}</code>
        </div>
        """, unsafe_allow_html=True)

    if run_batch_btn:
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Callback pentru actualizarea progresului în UI
        def update_progress(current, total):
            pct = current / total
            progress_bar.progress(pct)
            status_text.text(f"Se procesează explicația pentru casa {current} din {total}...")
            
        # Rulare evaluare batch
        start_time = time.time()
        summary, logs = backend.run_batch_evaluation(
            batch_size=batch_size, 
            model_name=model_option, 
            force_simulation=force_sim, 
            progress_callback=update_progress
        )
        duration = time.time() - start_time
        
        # Curățare indicatori de progres
        progress_bar.empty()
        status_text.empty()
        
        st.success(f"Evaluare finalizată cu succes în {duration:.2f} secunde!")
        
        # RENDER STATISTICI AGREGATE
        st.markdown("#### 📊 Rezultate Evaluare Globală")
        
        col_res1, col_res2, col_res3 = st.columns(3)
        with col_res1:
            st.markdown(f"""
            <div class="glass-card" style="text-align: center;">
                <div style="color: #9ca3af; font-size: 0.9rem; text-transform: uppercase; font-weight: 600;">Consistență Factuală Medie</div>
                <div style="color: #10b981; font-size: 3rem; font-weight: 700; margin: 10px 0;">{summary['avg_consistency_rate']:.1f}%</div>
                <div style="font-size: 0.85rem; color: #9ca3af;">Proporția factorilor numiți de LLM care au fost corecți.</div>
            </div>
            """, unsafe_allow_html=True)
            
        with col_res2:
            st.markdown(f"""
            <div class="glass-card" style="text-align: center;">
                <div style="color: #9ca3af; font-size: 0.9rem; text-transform: uppercase; font-weight: 600;">Recall Factor Primar</div>
                <div style="color: #8b5cf6; font-size: 3rem; font-weight: 700; margin: 10px 0;">{summary['avg_primary_recall_rate']:.1f}%</div>
                <div style="font-size: 0.85rem; color: #9ca3af;">Cât de des LLM a identificat cel mai important factor real.</div>
            </div>
            """, unsafe_allow_html=True)
            
        with col_res3:
            st.markdown(f"""
            <div class="glass-card" style="text-align: center;">
                <div style="color: #9ca3af; font-size: 0.9rem; text-transform: uppercase; font-weight: 600;">Medie Halucinații / Casă</div>
                <div style="color: #ef4444; font-size: 3rem; font-weight: 700; margin: 10px 0;">{summary['avg_hallucinations_per_house']:.2f}</div>
                <div style="font-size: 0.85rem; color: #9ca3af;">Total halucinații detectate: {summary['total_hallucinations_count']} factori.</div>
            </div>
            """, unsafe_allow_html=True)

        # Grafic Distribuție Scoruri de Consistență
        st.markdown("##### 📈 Distribuția Scorurilor de Consistență per Casă")
        df_logs = pd.DataFrame(logs)
        
        # Desenare grafic de frecvență a scorurilor de consistență
        hist_chart = alt.Chart(df_logs).mark_bar(
            color='#8b5cf6',
            binSpacing=2
        ).encode(
            x=alt.X('consistency_score:Q', bin=alt.Bin(maxbins=10), title='Scor Consistență (%)'),
            y=alt.Y('count():Q', title='Număr de Case'),
            tooltip=['count():Q']
        ).properties(
            height=200
        ).configure_axis(
            labelColor='#f3f4f6',
            titleColor='#f3f4f6'
        ).configure_view(
            strokeWidth=0
        )
        st.altair_chart(hist_chart, use_container_width=True)

        # TABEL DETALIAT DE LOGURI
        st.markdown("#### 📜 Log detaliat al evaluărilor")
        st.markdown("Tabelul de mai jos conține rezultatul pentru fiecare casă analizată în batch. Poți ordona coloanele sau căuta text în explicații.")
        
        # Formatare df pentru afișare frumoasă în Streamlit
        display_rows = []
        for l in logs:
            display_rows.append({
                "ID Casă": l["house_id"],
                "Preț Real (USD)": f"${l['actual_price']:,.0f}",
                "Preț Predis (USD)": f"${l['predicted_price']:,.0f}",
                "Consistență (%)": f"{l['consistency_score']:.1f}%",
                "Factor Primar Recunoscut?": "DA" if l["primary_recalled"] else "NU",
                "Top Factori Reali": ", ".join([FEATURE_MAP[f]["ro"] for f in l["top_real_factors"]]),
                "Factori Menționați LLM": ", ".join([FEATURE_MAP[f]["ro"] for f in l["llm_mentioned_factors"]]) if l["llm_mentioned_factors"] else "Niciunul",
                "Halucinații (Factori)": ", ".join([FEATURE_MAP[f]["ro"] for f in l["hallucinations"]]) if l["hallucinations"] else "Niciuna",
                "Omiteri (Factori)": ", ".join([FEATURE_MAP[f]["ro"] for f in l["omissions"]]) if l["omissions"] else "Niciuna",
                "Explicație LLM": l["explanation"]
            })
            
        df_display = pd.DataFrame(display_rows)
        st.dataframe(df_display, use_container_width=True)
