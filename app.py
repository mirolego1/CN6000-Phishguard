import streamlit as st
import joblib
import numpy as np
import re
from pathlib import Path

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PhishGuard",
    page_icon="🛡️",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
    background-color: #0d0d0d !important;
    color: #f0f0f0 !important;
}
.stApp { background-color: #0d0d0d !important; }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1.5rem !important; max-width: 760px !important; }

/* Header */
.pg-header { text-align: center; padding: 1.5rem 0 1rem 0; }
.pg-logo { font-size: 3rem; margin-bottom: 0.3rem; }
.pg-title { font-size: 2.6rem; font-weight: 900; letter-spacing: -0.04em; color: #ffffff; margin: 0; }
.pg-title span { color: #6c63ff; }
.pg-sub { font-size: 0.78rem; color: #555; letter-spacing: 0.15em; text-transform: uppercase; margin-top: 0.4rem; }
.pg-divider { height: 1px; background: #1e1e1e; margin: 1.2rem 0; }

/* Stats bar */
.stats-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 0.8rem; margin-bottom: 1.2rem; }
.stat-box { background: #111; border: 1px solid #1e1e1e; border-radius: 12px; padding: 0.9rem; text-align: center; }
.stat-val { font-size: 1.4rem; font-weight: 800; color: #6c63ff; line-height: 1.1; }
.stat-lbl { font-size: 0.62rem; color: #444; text-transform: uppercase; letter-spacing: 0.1em; margin-top: 0.2rem; }

/* Input card */
.input-card { background: #111; border: 1px solid #1e1e1e; border-radius: 14px; padding: 1.4rem; margin-bottom: 1rem; }
.card-label { font-size: 0.68rem; font-weight: 600; letter-spacing: 0.15em; text-transform: uppercase; color: #444; margin-bottom: 0.8rem; }

/* Streamlit input overrides */
.stTextInput > div > div > input,
.stTextArea > div > div > textarea {
    background: #0d0d0d !important;
    border: 1px solid #222 !important;
    border-radius: 8px !important;
    color: #f0f0f0 !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.88rem !important;
}
.stTextInput > div > div > input:focus,
.stTextArea > div > div > textarea:focus {
    border-color: #6c63ff !important;
    box-shadow: 0 0 0 2px rgba(108,99,255,0.2) !important;
}
.stTextInput label, .stTextArea label {
    font-size: 0.72rem !important; font-weight: 600 !important;
    letter-spacing: 0.1em !important; text-transform: uppercase !important;
    color: #555 !important;
}

/* Button */
.stButton, .stButton > div, .stButton > div > div { width: 100% !important; }
.stButton > button {
    background: #6c63ff !important; color: #fff !important;
    border: none !important; border-radius: 10px !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.95rem !important; font-weight: 700 !important;
    padding: 0.75rem 2rem !important; width: 100% !important;
    min-width: 100% !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 4px 20px rgba(108,99,255,0.3) !important;
    margin-top: 0.4rem !important;
}
.stButton > button:hover {
    background: #7c74ff !important;
    box-shadow: 0 6px 28px rgba(108,99,255,0.45) !important;
}

/* Verdict banner */
.verdict-phishing {
    background: linear-gradient(135deg, #1a0508 0%, #110205 100%);
    border: 1.5px solid #ff4d6d;
    border-radius: 14px; padding: 1.6rem; margin-bottom: 0.8rem;
}
.verdict-legit {
    background: linear-gradient(135deg, #021510 0%, #011009 100%);
    border: 1.5px solid #00d68f;
    border-radius: 14px; padding: 1.6rem; margin-bottom: 0.8rem;
}
.verdict-suspicious {
    background: linear-gradient(135deg, #150e00 0%, #0f0900 100%);
    border: 1.5px solid #ffb347;
    border-radius: 14px; padding: 1.6rem; margin-bottom: 0.8rem;
}
.verdict-text-phishing { font-size: 2rem; font-weight: 900; letter-spacing: -0.02em; color: #ff4d6d; margin: 0; }
.verdict-text-legit { font-size: 2rem; font-weight: 900; letter-spacing: -0.02em; color: #00d68f; margin: 0; }
.verdict-text-suspicious { font-size: 2rem; font-weight: 900; letter-spacing: -0.02em; color: #ffb347; margin: 0; }
.verdict-sub { font-size: 0.78rem; color: #888; margin-top: 0.3rem; text-transform: uppercase; letter-spacing: 0.1em; }

/* Result cards */
.result-card { background: #111; border: 1px solid #1e1e1e; border-radius: 12px; padding: 1.2rem; margin-bottom: 0.8rem; }
.result-card.url-header { border-bottom-left-radius: 0; border-bottom-right-radius: 0; margin-bottom: 0; border-bottom: none; padding-bottom: 0.6rem; }
.url-flagged { border-top-left-radius: 0 !important; border-top-right-radius: 0 !important; }
.url-item-last { border-bottom-left-radius: 12px; border-bottom-right-radius: 12px; overflow: hidden; margin-bottom: 0.8rem; }
.result-card-title { font-size: 0.68rem; font-weight: 600; letter-spacing: 0.15em; text-transform: uppercase; color: #444; margin-bottom: 0.8rem; }

/* Confidence bar */
.conf-track { background: #2a2a2a; border-radius: 999px; height: 10px; overflow: hidden; margin: 0.5rem 0; }
.conf-fill-danger { height: 100%; border-radius: 999px; background: linear-gradient(90deg, #ff4d6d, #ff8fa3); }
.conf-fill-safe { height: 100%; border-radius: 999px; background: linear-gradient(90deg, #00d68f, #7fffd4); }
.conf-fill-warn { height: 100%; border-radius: 999px; background: linear-gradient(90deg, #ffb347, #ffd700); }
.conf-num-danger { font-size: 1.6rem; font-weight: 800; color: #ff4d6d; }
.conf-num-safe { font-size: 1.6rem; font-weight: 800; color: #00d68f; }
.conf-num-warn { font-size: 1.6rem; font-weight: 800; color: #ffb347; }

/* Risk badge */
.badge-high { display:inline-block; background: rgba(255,77,109,0.12); color: #ff4d6d; border: 1px solid #ff4d6d; border-radius: 999px; padding: 0.3rem 1rem; font-size: 0.75rem; font-weight: 700; letter-spacing: 0.1em; text-transform: uppercase; }
.badge-medium { display:inline-block; background: rgba(255,179,71,0.12); color: #ffb347; border: 1px solid #ffb347; border-radius: 999px; padding: 0.3rem 1rem; font-size: 0.75rem; font-weight: 700; letter-spacing: 0.1em; text-transform: uppercase; }
.badge-low { display:inline-block; background: rgba(0,214,143,0.12); color: #00d68f; border: 1px solid #00d68f; border-radius: 999px; padding: 0.3rem 1rem; font-size: 0.75rem; font-weight: 700; letter-spacing: 0.1em; text-transform: uppercase; }
.badge-override { display:inline-block; background: rgba(255,179,71,0.12); color: #ffb347; border: 1px solid #ffb347; border-radius: 999px; padding: 0.3rem 1rem; font-size: 0.75rem; font-weight: 700; letter-spacing: 0.1em; text-transform: uppercase; margin-left: 0.5rem; }

/* Word pills */
.pill-danger { display:inline-block; background: rgba(255,77,109,0.1); color: #ff8fa3; border: 1px solid rgba(255,77,109,0.25); border-radius: 6px; padding: 0.25rem 0.65rem; margin: 0.2rem; font-size: 0.78rem; font-weight: 500; }
.pill-safe { display:inline-block; background: rgba(0,214,143,0.1); color: #7fffd4; border: 1px solid rgba(0,214,143,0.25); border-radius: 6px; padding: 0.25rem 0.65rem; margin: 0.2rem; font-size: 0.78rem; font-weight: 500; }

/* URL items */
.url-flagged { background: rgba(255,77,109,0.05); border: 1px solid rgba(255,77,109,0.2); border-radius: 8px; padding: 0.7rem 0.9rem; margin: 0.4rem 0; }
.url-clean { background: rgba(0,214,143,0.04); border: 1px solid rgba(0,214,143,0.15); border-radius: 8px; padding: 0.7rem 0.9rem; margin: 0.4rem 0; }
.url-text { font-size: 0.75rem; color: #ff8fa3; word-break: break-all; font-family: monospace; }
.url-text-clean { font-size: 0.75rem; color: #7fffd4; word-break: break-all; font-family: monospace; }
.url-signal { font-size: 0.68rem; color: #666; margin-top: 0.3rem; }

/* Footer */
.pg-footer { text-align: center; font-size: 0.65rem; color: #2a2a2a; letter-spacing: 0.1em; margin-top: 2rem; padding-top: 1.2rem; border-top: 1px solid #1a1a1a; }
</style>
""", unsafe_allow_html=True)

# ── Model loading ──────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    if not Path("svm_model.pkl").exists() or not Path("tfidf_vectorizer.pkl").exists():
        return None, None
    return joblib.load("svm_model.pkl"), joblib.load("tfidf_vectorizer.pkl")

# ── Helpers ────────────────────────────────────────────────────────────────────
def get_top_words(text, vectorizer, model, n=10):
    try:
        vec = vectorizer.transform([text])
        names = vectorizer.get_feature_names_out()
        scores = vec.toarray()[0] * model.coef_[0]
        idx = np.argsort(scores)[::-1][:n]
        return [names[i] for i in idx if vec.toarray()[0][i] > 0][:n]
    except Exception:
        return []

def extract_urls(text):
    return re.findall(r'https?://[^\s<>"\'()]+', text, re.IGNORECASE)

def analyse_urls(urls):
    KEYWORDS = ['verify','confirm','secure','login','signin','account','update',
                'validate','authenticate','suspended','unlock','restore','billing',
                'payment','alert','limited','unusual','review','activate','recover',
                'password','credential','bank','paypal','amazon','apple','microsoft',
                'google','netflix','support','helpdesk','refund','invoice']
    BRANDS = {'amazon':'amazon.com','paypal':'paypal.com','apple':'apple.com',
              'microsoft':'microsoft.com','google':'google.com','netflix':'netflix.com',
              'facebook':'facebook.com','instagram':'instagram.com','hsbc':'hsbc.com',
              'barclays':'barclays.co.uk','lloyds':'lloyds.com',
              'halifax':'halifax.co.uk','natwest':'natwest.com'}
    TLDS = ['.xyz','.top','.click','.link','.win','.loan','.download',
            '.gq','.tk','.ml','.cf','.ga']
    
    flagged, total = [], 0
    for url in urls:
        ul, signals, score = url.lower(), [], 0
        try:
            domain = re.search(r'https?://([^/]+)', ul).group(1)
        except Exception:
            domain = ul
            
        if re.search(r'https?://\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', ul):
            signals.append("IP address used as domain"); score += 40
            
        for tld in TLDS:
            if domain.endswith(tld):
                signals.append(f"Suspicious TLD ({tld})"); score += 25; break
                
        for brand, real in BRANDS.items():
            if brand in ul and real not in domain:
                signals.append(f"Possible {brand.title()} impersonation"); score += 35; break
                
        hits = [k for k in KEYWORDS if k in ul]
        if len(hits) >= 2:
            signals.append(f"Suspicious keywords: {', '.join(hits[:3])}"); score += 20
        elif len(hits) == 1:
            score += 8
            
        if len(url) > 100:
            signals.append("Unusually long URL"); score += 10
        if domain.count('.') - 1 >= 3:
            signals.append("Excessive subdomains"); score += 15
        if '-' in domain.split('/')[0]:
            signals.append("Hyphenated domain"); score += 15
            
        if score > 0:
            flagged.append({'url': url, 'score': score, 'signals': signals})
            total += score
            
    return {'flagged': flagged, 'total_score': total, 'url_count': len(urls), 'flagged_count': len(flagged)}

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="pg-header">
    <div class="pg-title">Phish<span>Guard</span></div>
    <div class="pg-sub">Email Threat Detection &nbsp;·&nbsp; SVM + URL Analysis</div>
</div>
<div class="pg-divider"></div>
""", unsafe_allow_html=True)

model, vectorizer = load_model()
if model is None:
    st.error("Model files not found. Please ensure svm_model.pkl and tfidf_vectorizer.pkl are in the same folder as app.py.")
    st.stop()

# ── Stats bar ──────────────────────────────────────────────────────────────────
st.markdown("""
<div class="stats-grid">
    <div class="stat-box"><div class="stat-val">99.02%</div><div class="stat-lbl">Accuracy</div></div>
    <div class="stat-box"><div class="stat-val">99.02%</div><div class="stat-lbl">F1-Score</div></div>
    <div class="stat-box"><div class="stat-val">56,478</div><div class="stat-lbl">Trained On</div></div>
    <div class="stat-box"><div class="stat-val">SVM</div><div class="stat-lbl">Model</div></div>
</div>
""", unsafe_allow_html=True)

# ── Input ──────────────────────────────────────────────────────────────────────
subject = st.text_input("Subject Line", placeholder="Enter the email subject...")
body = st.text_area("Email Body", placeholder="Paste the full email body here...", height=180)
analyse = st.button("🔍 Analyse Email")

# ── Analysis ───────────────────────────────────────────────────────────────────
if analyse:
    if not subject.strip() and not body.strip():
        st.warning("Please enter a subject or body to analyse.")
    else:
        with st.spinner("Analysing..."):
            combined = re.sub(r'\s+', ' ', f"{subject} {body}").strip()
            vec = vectorizer.transform([combined])
            pred = model.predict(vec)[0]
            is_phish = bool(pred == 1)
            
            try:
                dec = model.decision_function(vec)[0]
                conf = float(1 / (1 + np.exp(-dec)))
                if not is_phish:
                    conf = 1 - conf
            except Exception:
                conf = 0.95
                
            conf_pct = round(conf * 100, 1)
            urls = extract_urls(combined)
            url_res = analyse_urls(urls)
            url_flag = url_res['flagged_count'] > 0
            override = not is_phish and url_flag and url_res['total_score'] >= 30
            top_words = get_top_words(combined, vectorizer, model, n=10)

            # Risk level
            if override:
                risk, risk_cls = "High", "badge-high"
            elif is_phish:
                risk = "High" if conf >= 0.90 else ("Medium" if conf >= 0.70 else "Low")
                risk_cls = "badge-high" if risk == "High" else ("badge-medium" if risk == "Medium" else "badge-low")
            else:
                risk, risk_cls = "Low", "badge-low"

            # ── Verdict banner ─────────────────────────────────────────────────
            if override:
                v_cls, v_text, v_icon = "verdict-suspicious", "verdict-text-suspicious", "⚠"
                v_head, v_sub = "SUSPICIOUS — URL THREAT", "ML marked legitimate · suspicious URLs detected"
                c_fill, c_num, p_pill = "conf-fill-warn", "conf-num-warn", "pill-danger"
            elif is_phish:
                v_cls, v_text, v_icon = "verdict-phishing", "verdict-text-phishing", "⚠"
                v_head, v_sub = "PHISHING DETECTED", "This email shows characteristics of a phishing attempt"
                c_fill, c_num, p_pill = "conf-fill-danger", "conf-num-danger", "pill-danger"
            else:
                v_cls, v_text, v_icon = "verdict-legit", "verdict-text-legit", "✓"
                v_head, v_sub = "LEGITIMATE EMAIL", "This email appears to be safe"
                c_fill, c_num, p_pill = "conf-fill-safe", "conf-num-safe", "pill-safe"

            st.markdown(f"""
            <div class="{v_cls}">
                <div class="{v_text}">{v_icon} {v_head}</div>
                <div class="verdict-sub">{v_sub}</div>
            </div>
            """, unsafe_allow_html=True)

            # ── Three result cards in columns ──────────────────────────────────
            col1, col2 = st.columns(2)
            with col1:
                col_conf_color = '#ff4d6d' if (is_phish or override) else '#00d68f'
                conf_level_text = 'High' if conf >= 0.90 else ('Medium' if conf >= 0.70 else 'Low')
                st.markdown(f"""
                <div class="result-card">
                    <div class="result-card-title">ML Confidence</div>
                    <div class="{c_num}">{conf_pct}%</div>
                    <div class="conf-track">
                        <div class="{c_fill}" style="width:{conf_pct}%"></div>
                    </div>
                    <div style="margin-top:0.8rem;padding-top:0.8rem;border-top:1px solid #1a1a1a;">
                        <div style="font-size:0.62rem;color:#444;text-transform:uppercase;letter-spacing:0.1em;">Confidence Level</div>
                        <div style="font-size:1rem;font-weight:800;color:{col_conf_color};margin-top:0.2rem;">{conf_level_text}</div>
                        <div style="font-size:0.7rem;color:#444;">Based on TF-IDF features</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            with col2:
                override_badge = '<span class="badge-override">URL OVERRIDE</span>' if override else ''
                risk_desc = ("Phishing detected by ML" if is_phish and not override else "URLs escalated risk" if override else "No threats detected")
                url_scan_colour = "#ff4d6d" if url_res["flagged_count"] > 0 else "#00d68f"
                url_scan_count = f'{url_res["flagged_count"]} flagged'
                urls_total = f'{url_res["url_count"]} URL{"s" if url_res["url_count"] != 1 else ""} scanned'
                
                # Render the card in one clean block to prevent tag mismatch
                card_html = f"""
                <div class="result-card">
                    <div class="result-card-title">Risk Level</div>
                    <div style="margin-top:0.5rem;">
                        <span class="{risk_cls}">{risk}</span> {override_badge}
                    </div>
                    <div style="font-size:0.72rem;color:#444;margin-top:0.5rem;">{risk_desc}</div>
                    <div style="margin-top:0.8rem;padding-top:0.8rem;border-top:1px solid #1a1a1a;">
                        <div style="font-size:0.62rem;color:#444;text-transform:uppercase;letter-spacing:0.1em;">URL Scan</div>
                        <div style="font-size:1rem;font-weight:800;color:{url_scan_colour};margin-top:0.2rem;">{url_scan_count}</div>
                        <div style="font-size:0.7rem;color:#444;">{urls_total}</div>
                    </div>
                </div>
                """
                st.markdown(card_html, unsafe_allow_html=True)

            # ── Top words card ─────────────────────────────────────────────────
            if top_words:
                pills = "".join(f'<span class="{p_pill}">{w}</span>' for w in top_words)
                st.markdown(f"""
                <div class="result-card">
                    <div class="result-card-title">🔍 Top Influencing Terms</div>
                    <div style="margin-top:0.4rem;">{pills}</div>
                </div>
                """, unsafe_allow_html=True)

            # ── URL analysis card ──────────────────────────────────────────────
            if urls:
                flagged_set = {f['url'] for f in url_res['flagged']}
                st.markdown(f'''<div class="result-card url-header"><div class="result-card-title">🔗 URL Analysis — {url_res["flagged_count"]} flagged / {url_res["url_count"]} found</div></div>''', unsafe_allow_html=True)
                for item in url_res['flagged']:
                    sigs = ' · '.join(item['signals']) if item['signals'] else 'Suspicious signals detected'
                    url_html = f'''<div class="url-flagged" style="border-top:none;border-top-left-radius:0;border-top-right-radius:0;margin-top:0;"><div class="url-text">{item["url"]}</div><div class="url-signal">⚠ {sigs}</div></div>'''
                    st.markdown(url_html, unsafe_allow_html=True)
                for u in urls:
                    if u not in flagged_set:
                        clean_html = f'''<div class="url-clean" style="border-top:none;border-top-left-radius:0;border-top-right-radius:0;margin-top:0;"><div class="url-text-clean">{u}</div><div class="url-signal">✓ No suspicious signals</div></div>'''
                        st.markdown(clean_html, unsafe_allow_html=True)
            elif not urls and (is_phish or override):
                st.markdown('''<div class="result-card"><div class="result-card-title">🔗 URL Analysis</div><div style="font-size:0.8rem;color:#555;margin-top:0.4rem;">No URLs found in this email.</div></div>''', unsafe_allow_html=True)

# ── How it works ───────────────────────────────────────────────────────────────
with st.expander("ℹ How PhishGuard works"):
    st.markdown("""
    <div style="font-size:0.82rem;line-height:1.9;color:#555;">
        <strong style="color:#f0f0f0;">Model:</strong> Support Vector Machine (LinearSVC)<br>
        <strong style="color:#f0f0f0;">Feature Extraction:</strong> TF-IDF (10,000 features · 1-2 word n-grams)<br>
        <strong style="color:#f0f0f0;">Training Data:</strong> CEAS_08 + Kaggle — 56,478 emails (50/50 balanced)<br>
        <strong style="color:#f0f0f0;">Train/Test Split:</strong> 80% training / 20% testing (stratified)<br>
        <br>
        <strong style="color:#f0f0f0;">Detection Pipeline:</strong><br>
        &nbsp;&nbsp;1. Email text is vectorised using TF-IDF<br>
        &nbsp;&nbsp;2. SVM classifies the email as phishing or legitimate<br>
        &nbsp;&nbsp;3. URLs are independently analysed for suspicious signals<br>
        &nbsp;&nbsp;4. If ML says legitimate but URLs are suspicious, verdict is escalated<br>
        <br>
        <strong style="color:#f0f0f0;">Risk levels:</strong><br>
        &nbsp;&nbsp;• <span style="color:#ff4d6d;">High</span> — Phishing ≥90% confidence or URL override<br>
        &nbsp;&nbsp;• <span style="color:#ffb347;">Medium</span> — Phishing 70–89% confidence<br>
        &nbsp;&nbsp;• <span style="color:#00d68f;">Low</span> — Legitimate with no suspicious URL signals
    </div>
    """, unsafe_allow_html=True)

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="pg-footer">
    PhishGuard · BSc Computer Science Final Year Project · CN6000<br>
    SVM · CEAS_08 + Kaggle Dataset · TF-IDF
</div>
""", unsafe_allow_html=True)