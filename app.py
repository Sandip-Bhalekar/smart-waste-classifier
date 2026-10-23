import json
import uuid
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path

import altair as alt
import pandas as pd
import requests
import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

PROJECT = Path(__file__).resolve().parent
CHECKPOINT_PATH = PROJECT / "checkpoints/multilabel/multilabel_best.pth"
THRESHOLD_PATH = PROJECT / "checkpoints/multilabel/thresholds_v2.json"

# Each material gets its own accent colour, used consistently in cards, bars and charts.
MATERIAL_GUIDANCE = {
    "glass": {
        "title": "Glass", "icon": "🫙", "short": "Bottles and jars", "color": "#0e9aa7", "bin": "Glass bin",
        "recycling": "Separate glass bottles and jars where local glass recycling is available. Keep reusable containers intact when possible.",
        "disposal": "Handle broken glass carefully. Do not place broken glass loose in recycling bins unless your local service explicitly permits it.",
    },
    "paper": {
        "title": "Paper", "icon": "📄", "short": "Paper and newspapers", "color": "#d99a1e", "bin": "Paper bin",
        "recycling": "Keep clean, dry paper separate for paper recycling. Remove non-paper components where practical.",
        "disposal": "Greasy or heavily food-contaminated paper may not be accepted for recycling. Check local collection rules.",
    },
    "plastic": {
        "title": "Plastic", "icon": "🧴", "short": "Bottles and containers", "color": "#3b6fd4", "bin": "Plastic bin",
        "recycling": "Empty containers and check the plastic type and local recycling rules. Separate caps or labels only when your local service requests it.",
        "disposal": "Keep plastic waste out of drains and waterways. If the item is not accepted for recycling, use the appropriate local waste collection service.",
    },
    "cardboard": {
        "title": "Cardboard", "icon": "📦", "short": "Boxes and cartons", "color": "#a8703c", "bin": "Paper / cardboard bin",
        "recycling": "Flatten clean cardboard boxes and keep them dry. Remove excess tape and other packaging where practical.",
        "disposal": "Food-soiled or wet cardboard may not be accepted in paper recycling. Follow local collection rules.",
    },
    "metal": {
        "title": "Metal", "icon": "🥫", "short": "Cans and metal packaging", "color": "#64748b", "bin": "Metal bin",
        "recycling": "Separate accepted metal cans and packaging for metal recycling. Empty containers before placing them in the appropriate bin.",
        "disposal": "Handle sharp metal edges carefully. Batteries, aerosol cans, and hazardous metal-containing items may require special collection.",
    },
}

st.set_page_config(
    page_title="EcoSort | Smart Waste Assistant",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------------------------------------------------------------
# Styling. Pair this with .streamlit/config.toml (light theme) so far fewer overrides
# are needed than before.
# ----------------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Plus+Jakarta+Sans:wght@600;700;800&display=swap');
    :root {
        --green-900:#0f3d2e; --green-800:#14573f; --green-700:#16704f; --green-600:#1a8a61;
        --green-100:#dff3e8; --green-50:#f0f9f4;
        --bg:#f4f7f5; --card:#ffffff; --border:#e0e9e3; --ink:#1b2a23; --muted:#5d7066;
        --radius:18px; --shadow:0 1px 2px rgba(15,61,46,.04), 0 8px 24px rgba(15,61,46,.06);
    }
    html, body, [class*="css"], .stApp { font-family:'Inter',sans-serif; }
    .stApp { background:var(--bg) !important; color:var(--ink); }
    .stApp h1, .stApp h2, .stApp h3, .stApp h4 { font-family:'Plus Jakarta Sans',sans-serif; color:var(--green-900) !important; letter-spacing:-.02em; }
    .stApp p, .stApp li, .stApp label, .stApp [data-testid="stMarkdownContainer"],
    .stApp [data-testid="stCaptionContainer"] { color:var(--ink); }
    .block-container { max-width:1200px; padding:2rem 2rem 3rem; }
    #MainMenu, footer { visibility:hidden; }
    [data-testid="stHeader"] { background:#f4f7f5 !important; border-bottom:1px solid #e0e9e3; }
    /* Toolbar icons (Share, star, edit, GitHub, menu): solid dark so they never vanish on the light header. */
    [data-testid="stToolbar"] svg, [data-testid="stToolbar"] svg *,
    [data-testid="stHeader"] svg, [data-testid="stHeader"] svg * { fill:#1b2a23 !important; color:#1b2a23 !important; }
    [data-testid="stToolbar"] a, [data-testid="stToolbar"] button,
    [data-testid="stToolbar"] span, [data-testid="stToolbar"] p { color:#1b2a23 !important; }
    [data-testid="stToolbar"] button:hover, [data-testid="stToolbar"] a:hover { background:#e3eee7 !important; border-radius:8px; }
    /* Stronger, version-proof header rules */
    header, header[data-testid="stHeader"], [data-testid="stAppToolbar"], [data-testid="stDecoration"] { background:#f4f7f5 !important; background-image:none !important; }
    header *, [data-testid="stAppToolbar"] *, [data-testid="stMainMenu"] *, [data-testid="stToolbarActions"] * {
        color:#1b2a23 !important; fill:#1b2a23 !important; stroke:#1b2a23 !important; opacity:1 !important;
    }
    [data-testid="stDecoration"] { display:none; }

    /* ---------- Sidebar ---------- */
    section[data-testid="stSidebar"],
    section[data-testid="stSidebar"] > div,
    section[data-testid="stSidebar"] [data-testid="stSidebarContent"],
    section[data-testid="stSidebar"] [data-testid="stSidebarUserContent"],
    [data-testid="stSidebar"] { background:#0f3d2e !important; background-image:linear-gradient(180deg,#0f3d2e 0%,#12281f 100%) !important; border-right:0; }
    section[data-testid="stSidebar"] * { color:#e8f3ec !important; }
    section[data-testid="stSidebar"] [data-testid="stCaptionContainer"],
    section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] *,
    section[data-testid="stSidebar"] .brand-tag,
    section[data-testid="stSidebar"] .side-mat small { color:#a9c7b6 !important; }
    section[data-testid="stSidebar"] [data-testid="stSidebarCollapseButton"] button,
    section[data-testid="stSidebar"] [data-testid="stSidebarCollapseButton"] svg { color:#fff !important; fill:#fff !important; }
    [data-testid="stExpandSidebarButton"], [data-testid="stSidebarCollapsedControl"] { color:#0f3d2e !important; }
    [data-testid="stToolbar"], [data-testid="stToolbar"] * , [data-testid="stHeader"] button { color:#1b2a23 !important; }
    section[data-testid="stSidebar"] h1, section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3, section[data-testid="stSidebar"] strong { color:#fff !important; }
    section[data-testid="stSidebar"] hr { border-color:rgba(255,255,255,.12) !important; }
    section[data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] { gap:.35rem; }
    section[data-testid="stSidebar"] [data-testid="stRadio"] label {
        background:rgba(255,255,255,.05); border:1px solid rgba(255,255,255,.08);
        border-radius:12px; padding:.6rem .8rem; width:100%; transition:all .15s ease;
    }
    section[data-testid="stSidebar"] [data-testid="stRadio"] label:hover { background:rgba(255,255,255,.1); }
    section[data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) { background:#1a8a61; border-color:#34b585; }
    section[data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) * { color:#fff !important; font-weight:600; }
    section[data-testid="stSidebar"] [data-testid="stRadio"] label > div:first-child { display:none; }
    .brand { display:flex; align-items:center; gap:.65rem; margin-bottom:.2rem; }
    .brand-logo { width:40px; height:40px; border-radius:12px; background:linear-gradient(135deg,#34b585,#1a8a61); display:grid; place-items:center; font-size:1.3rem; }
    .brand-name { font-family:'Plus Jakarta Sans',sans-serif; font-weight:800; font-size:1.3rem; color:#fff; line-height:1; }
    .brand-tag { font-size:.75rem; color:#9fbdad; }
    .side-mat { display:flex; align-items:center; gap:.6rem; padding:.35rem 0; }
    .side-dot { width:9px; height:9px; border-radius:99px; flex:none; }
    .side-mat small { color:#9fbdad !important; display:block; font-size:.75rem; line-height:1.1; }
    .user-card { background:rgba(255,255,255,.07); border-radius:12px; padding:.65rem .8rem; margin-bottom:.6rem; font-size:.82rem; }
    .user-card b { color:#fff; display:block; word-break:break-all; }

    /* ---------- Hero ---------- */
    .hero { position:relative; overflow:hidden; background:linear-gradient(120deg,#0f3d2e 0%,#16704f 60%,#34b585 120%); border-radius:26px; padding:2.4rem 2.6rem; margin-bottom:1.6rem; box-shadow:var(--shadow); }
    .hero::after { content:"♻️"; position:absolute; right:2rem; top:50%; transform:translateY(-50%) rotate(-12deg); font-size:9rem; opacity:.13; }
    .hero, .hero * { color:#fff !important; }
    .hero h1 { font-size:clamp(1.9rem,3.6vw,2.9rem); line-height:1.12; margin:.3rem 0 .7rem; }
    .hero p { color:#d7efe2 !important; font-size:1.02rem; max-width:620px; margin:0; }
    .eyebrow { text-transform:uppercase; letter-spacing:.14em; font-size:.72rem; font-weight:700; color:#a8e6c6 !important; }
    .hero-pills { display:flex; flex-wrap:wrap; gap:.5rem; margin-top:1.1rem; }
    .pill { background:rgba(255,255,255,.14); border:1px solid rgba(255,255,255,.25); border-radius:99px; padding:.3rem .8rem; font-size:.8rem; }

    /* ---------- Generic cards ---------- */
    .section-kicker { color:var(--green-700) !important; font-size:.72rem; font-weight:700; text-transform:uppercase; letter-spacing:.12em; margin-bottom:.2rem; }
    .subtle { color:var(--muted) !important; font-size:.93rem; }
    .card { background:var(--card); border:1px solid var(--border); border-radius:var(--radius); padding:1.15rem 1.25rem; box-shadow:var(--shadow); margin-bottom:.8rem; }
    .card, .card * { color:var(--ink); }
    .card h3 { margin:.2rem 0 .3rem; }
    .empty { text-align:center; padding:2.2rem 1.5rem; border-style:dashed; }
    .empty .big { font-size:2.4rem; }

    /* ---------- Steps ---------- */
    .steps { display:grid; grid-template-columns:repeat(4,1fr); gap:.8rem; margin-bottom:1.4rem; }
    .step { background:var(--card); border:1px solid var(--border); border-radius:14px; padding:.9rem 1rem; }
    .step-num { width:26px; height:26px; border-radius:99px; background:var(--green-100); color:var(--green-800) !important; font-weight:700; font-size:.8rem; display:grid; place-items:center; margin-bottom:.45rem; }
    .step b { color:var(--green-900); font-size:.9rem; display:block; }
    .step span { color:var(--muted); font-size:.8rem; }

    /* ---------- Results ---------- */
    .summary { background:linear-gradient(135deg,#f0f9f4,#e3f4ea); border:1px solid #c5e5d2; border-radius:var(--radius); padding:1.1rem 1.25rem; margin-bottom:1rem; }
    .summary, .summary * { color:var(--green-900); }
    .summary .label { font-size:.72rem; font-weight:700; text-transform:uppercase; letter-spacing:.1em; color:var(--green-700) !important; }
    .summary .headline { font-family:'Plus Jakarta Sans',sans-serif; font-size:1.35rem; font-weight:800; margin:.15rem 0 .55rem; }
    .chip { display:inline-flex; align-items:center; gap:.35rem; padding:.3rem .75rem; border-radius:99px; font-size:.83rem; font-weight:600; margin:0 .4rem .4rem 0; border:1px solid; }
    .mat-head { display:flex; align-items:center; justify-content:space-between; gap:.6rem; }
    .mat-title { display:flex; align-items:center; gap:.65rem; }
    .mat-icon { width:42px; height:42px; border-radius:12px; display:grid; place-items:center; font-size:1.4rem; }
    .mat-name { font-weight:700; font-size:1.02rem; color:var(--green-900); line-height:1.15; }
    .mat-sub { font-size:.8rem; color:var(--muted) !important; }
    .badge { font-size:.72rem; font-weight:700; padding:.28rem .6rem; border-radius:99px; white-space:nowrap; }
    .badge-yes { background:#e0f5e9; color:#14673f !important; }
    .badge-no { background:#eef1ef; color:#60706a !important; }
    .meter { position:relative; height:9px; background:#e8eeea; border-radius:99px; margin:.8rem 0 .35rem; }
    .meter-fill { height:100%; border-radius:99px; }
    .meter-tick { position:absolute; top:-4px; width:2px; height:17px; background:#1b2a23; opacity:.55; border-radius:2px; }
    .meter-meta { display:flex; justify-content:space-between; font-size:.76rem; color:var(--muted) !important; }
    .guide { border-left:4px solid; border-radius:6px; background:#fafcfb; padding:.7rem .95rem; margin:.55rem 0; font-size:.92rem; }
    .guide b { display:block; font-size:.76rem; text-transform:uppercase; letter-spacing:.08em; margin-bottom:.15rem; }
    .cloud-status { display:flex; align-items:center; gap:.5rem; border-radius:12px; padding:.6rem .9rem; font-size:.85rem; font-weight:500; margin-bottom:.9rem; }
    .cloud-ok { background:#e5f6ec; border:1px solid #b9dfc9; color:#14573f; }
    .cloud-warn { background:#fff6e0; border:1px solid #f0d99b; color:#7a5200; }

    /* ---------- Widgets ---------- */
    div[data-testid="stFileUploader"] { background:var(--card); border:2px dashed #a9cdb8; border-radius:var(--radius); padding:.9rem 1rem; }
    div[data-testid="stFileUploader"] section { background:#f6faf8 !important; border:0 !important; border-radius:12px; }
    div[data-testid="stFileUploader"] small, div[data-testid="stFileUploader"] span { color:var(--muted) !important; }
    div[data-testid="stFileUploader"] button { background:var(--green-700) !important; color:#fff !important; border:0 !important; border-radius:10px !important; font-weight:600 !important; }
    div[data-testid="stFileUploader"] button * { color:#fff !important; }
    div.stButton > button, div[data-testid="stFormSubmitButton"] button, div.stDownloadButton > button {
        border-radius:12px !important; font-weight:650 !important; min-height:2.8rem;
        background:#fff !important; color:var(--green-900) !important; border:1px solid #b9cfc0 !important; transition:all .15s ease;
    }
    div.stButton > button *, div[data-testid="stFormSubmitButton"] button *, div.stDownloadButton > button * { color:inherit !important; }
    div.stButton > button[kind="primary"], div[data-testid="stFormSubmitButton"] button[kind="primary"] {
        background:linear-gradient(135deg,#1a8a61,#14573f) !important; color:#fff !important; border:0 !important; box-shadow:0 6px 16px rgba(22,112,79,.28);
    }
    div.stButton > button[kind="primary"] *, div[data-testid="stFormSubmitButton"] button[kind="primary"] * { color:#fff !important; }
    div.stButton > button:hover, div.stDownloadButton > button:hover { border-color:var(--green-700) !important; transform:translateY(-1px); }
    section[data-testid="stSidebar"] div.stButton > button { background:rgba(255,255,255,.1) !important; color:#fff !important; border:1px solid rgba(255,255,255,.2) !important; }
    div[data-testid="stTextInput"] input { background:#fff !important; color:var(--ink) !important; }
    div[data-testid="stTextInput"] [data-baseweb="input"] { background:#fff !important; border-radius:10px !important; }
    div[data-testid="stTextInput"] button { background:#e8f4ec !important; color:#174b36 !important; border-radius:0 10px 10px 0 !important; }
    div[data-testid="stTextInput"] button svg, div[data-testid="stTextInput"] button svg path { fill:#174b36 !important; color:#174b36 !important; }
    div[data-baseweb="select"] > div { background:#fff !important; border-radius:10px !important; border-color:#cbd8ce !important; }
    div[data-baseweb="select"] * { color:var(--ink) !important; }
    [data-testid="stMetric"] { background:#fff; border:1px solid var(--border); border-radius:16px; padding:1rem 1.15rem; box-shadow:var(--shadow); }
    [data-testid="stMetricLabel"] p { color:var(--muted) !important; font-size:.82rem; }
    [data-testid="stMetricValue"] { color:var(--green-900) !important; font-family:'Plus Jakarta Sans',sans-serif; font-weight:800; }
    div[data-testid="stExpander"] { background:#fff; border:1px solid var(--border); border-radius:14px; }
    div[data-testid="stExpander"] summary p { color:var(--green-900) !important; font-weight:600; }
    button[data-baseweb="tab"] { font-weight:600; }
    button[data-baseweb="tab"][aria-selected="true"] { color:var(--green-700) !important; }
    div[data-baseweb="tab-highlight"] { background:var(--green-700) !important; }
    [data-testid="stAlert"] { border-radius:12px; }
    [data-testid="stDataFrame"] { border:1px solid var(--border); border-radius:14px; overflow:hidden; }
    [data-testid="stImage"] img { border-radius:var(--radius); box-shadow:var(--shadow); }
    .footer { border-top:1px solid var(--border); margin-top:2.5rem; padding-top:1rem; color:var(--muted) !important; font-size:.8rem; text-align:center; }

    @media (max-width:900px) { .steps { grid-template-columns:repeat(2,1fr); } .hero::after { display:none; } }
    @media (max-width:600px) { .block-container { padding:1rem 1rem 2rem; } .hero { padding:1.6rem 1.4rem; } .steps { grid-template-columns:1fr; } }
    </style>
    """,
    unsafe_allow_html=True,
)


# ----------------------------------------------------------------------------------
# Small HTML helpers
# ----------------------------------------------------------------------------------
def hex_to_rgba(hex_color, alpha):
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def chip_html(material):
    item = MATERIAL_GUIDANCE[material]
    c = item["color"]
    return (
        f'<span class="chip" style="background:{hex_to_rgba(c, .1)};border-color:{hex_to_rgba(c, .35)};color:{c}">'
        f'{item["icon"]} {item["title"]}</span>'
    )


def material_card_html(material, score, threshold, detected):
    item = MATERIAL_GUIDANCE[material]
    c = item["color"]
    pct = max(0, min(100, score * 100))
    tick = max(0, min(100, threshold * 100))
    badge = '<span class="badge badge-yes">Detected</span>' if detected else '<span class="badge badge-no">Below threshold</span>'
    return (
        '<div class="card">'
        '<div class="mat-head">'
        '<div class="mat-title">'
        f'<div class="mat-icon" style="background:{hex_to_rgba(c, .12)}">{item["icon"]}</div>'
        f'<div><div class="mat-name">{item["title"]}</div><div class="mat-sub">{item["short"]}</div></div>'
        '</div>'
        f'{badge}'
        '</div>'
        f'<div class="meter"><div class="meter-fill" style="width:{pct:.1f}%;background:{c}"></div>'
        f'<div class="meter-tick" style="left:{tick:.1f}%"></div></div>'
        f'<div class="meter-meta"><span>Threshold {threshold:.0%}</span><span><b>Model score {score:.1%}</b></span></div>'
        '</div>'
    )


def guide_html(label, text, color):
    return f'<div class="guide" style="border-color:{color}"><b style="color:{color}">{label}</b>{text}</div>'


def render_footer(text):
    st.markdown(f'<div class="footer">{text}</div>', unsafe_allow_html=True)


# ----------------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        '<div class="brand"><div class="brand-logo">♻️</div>'
        '<div><div class="brand-name">EcoSort</div><div class="brand-tag">Smart Waste Assistant</div></div></div>',
        unsafe_allow_html=True,
    )
    st.write("")
    page = st.radio(
        "Navigation",
        ["🔍  Waste Classifier", "📊  History & Analytics"],
        label_visibility="collapsed",
    )
    st.divider()
    st.markdown("**Supported materials**")
    for key, item in MATERIAL_GUIDANCE.items():
        st.markdown(
            f'<div class="side-mat"><span class="side-dot" style="background:{item["color"]}"></span>'
            f'<div>{item["icon"]} {item["title"]}<small>{item["short"]}</small></div></div>',
            unsafe_allow_html=True,
        )
    st.divider()
    st.caption("Model: EfficientNet-B0 · Multi-label classification")


# ----------------------------------------------------------------------------------
# History dashboard
# ----------------------------------------------------------------------------------
def render_history_dashboard():
    st.markdown('<div class="section-kicker">Cloud records</div>', unsafe_allow_html=True)
    st.title("History & analytics")
    st.markdown('<p class="subtle">Review your stored analyses and material trends.</p>', unsafe_allow_html=True)
    with st.spinner("Loading cloud prediction history…"):
        rows, error = fetch_prediction_history(st.session_state.supabase_access_token, st.session_state.supabase_user["id"])
    if error:
        st.info(error)
        st.markdown(
            "**Your history could not be loaded.** Check that the signed-in role has `SELECT` permission and that "
            "the RLS policy allows `auth.uid() = user_id`. This dashboard intentionally does not bypass "
            "row-level security or expose other users’ records."
        )
        return
    if not rows:
        st.markdown(
            '<div class="card empty"><div class="big">🗂️</div><h3>No analyses yet</h3>'
            '<p class="subtle">Analyze an image on the Waste Classifier page to create your first record.</p></div>',
            unsafe_allow_html=True,
        )
        return

    materials = list(MATERIAL_GUIDANCE.keys())
    titles = {m: MATERIAL_GUIDANCE[m]["title"] for m in materials}
    colors = [MATERIAL_GUIDANCE[m]["color"] for m in materials]

    df = pd.DataFrame(rows)
    df["created"] = pd.to_datetime(df.get("created_at"), errors="coerce", utc=True)
    df["predicted_materials"] = df["predicted_materials"].apply(lambda v: v if isinstance(v, list) else [])

    # ---- Filters
    f1, f2 = st.columns([2, 1])
    selected = f1.multiselect(
        "Filter by material", materials, format_func=lambda m: f"{MATERIAL_GUIDANCE[m]['icon']} {titles[m]}",
        placeholder="All materials",
    )
    valid_dates = df["created"].dropna()
    date_range = None
    if not valid_dates.empty:
        lo, hi = valid_dates.min().date(), valid_dates.max().date()
        date_range = f2.date_input("Date range", value=(lo, hi), min_value=lo, max_value=hi)

    view = df
    if selected:
        view = view[view["predicted_materials"].apply(lambda ms: any(m in ms for m in selected))]
    if date_range and isinstance(date_range, tuple) and len(date_range) == 2:
        d = view["created"].dt.date
        view = view[(d >= date_range[0]) & (d <= date_range[1])]

    if view.empty:
        st.info("No analyses match the current filters.")
        return

    counts = {m: 0 for m in materials}
    for ms in view["predicted_materials"]:
        for m in ms:
            if m in counts:
                counts[m] += 1
    top = max(counts, key=counts.get) if any(counts.values()) else None
    avg_labels = sum(counts.values()) / len(view)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Saved analyses", len(view))
    c2.metric("Material labels", sum(counts.values()))
    c3.metric("Most common", f"{MATERIAL_GUIDANCE[top]['icon']} {titles[top]}" if top else "—")
    c4.metric("Avg. labels / scan", f"{avg_labels:.1f}")
    st.write("")

    col_a, col_b = st.columns([1.1, 1], gap="large")
    chart_data = pd.DataFrame({"Material": [titles[m] for m in materials], "Predictions": [counts[m] for m in materials]})
    with col_a:
        st.subheader("Material distribution")
        bars = (
            alt.Chart(chart_data)
            .mark_bar(cornerRadiusTopLeft=8, cornerRadiusTopRight=8, size=44)
            .encode(
                x=alt.X("Material:N", sort=None, axis=alt.Axis(labelAngle=0, title=None, labelColor="#34483d", domainColor="#dce8df", tickColor="#dce8df")),
                y=alt.Y("Predictions:Q", scale=alt.Scale(domainMin=0), axis=alt.Axis(format="d", tickMinStep=1, title=None, labelColor="#52665b", gridColor="#edf2ee", domain=False)),
                color=alt.Color("Material:N", scale=alt.Scale(domain=[titles[m] for m in materials], range=colors), legend=None),
                tooltip=["Material:N", alt.Tooltip("Predictions:Q", format="d")],
            )
            .properties(height=290, background="#ffffff", padding={"left": 14, "right": 14, "top": 14, "bottom": 14})
            .configure_view(stroke="#e1e9e3", fill="#ffffff")
        )
        st.altair_chart(bars, use_container_width=True, theme=None)

    with col_b:
        st.subheader("Prediction activity")
        dates = view["created"].dropna().dt.date
        if not dates.empty:
            activity = dates.value_counts().sort_index().rename_axis("Date").reset_index(name="Analyses")
            activity["Date"] = pd.to_datetime(activity["Date"])
            line = (
                alt.Chart(activity)
                .mark_area(line={"color": "#16845b", "strokeWidth": 3}, point=alt.OverlayMarkDef(filled=True, size=70, color="#16845b"),
                           color=alt.Gradient(gradient="linear", stops=[alt.GradientStop(color="rgba(22,132,91,.28)", offset=0), alt.GradientStop(color="rgba(22,132,91,0)", offset=1)], x1=1, x2=1, y1=1, y2=0))
                .encode(
                    x=alt.X("Date:T", axis=alt.Axis(format="%d %b", title=None, grid=False, labelColor="#52665b", domainColor="#dce8df")),
                    y=alt.Y("Analyses:Q", scale=alt.Scale(domainMin=0), axis=alt.Axis(tickMinStep=1, format="d", title=None, labelColor="#52665b", gridColor="#edf2ee", domain=False)),
                    tooltip=[alt.Tooltip("Date:T", format="%d %b %Y"), alt.Tooltip("Analyses:Q", format="d")],
                )
                .properties(height=290, background="#ffffff", padding={"left": 14, "right": 14, "top": 14, "bottom": 14})
                .configure_view(stroke="#e1e9e3", fill="#ffffff")
            )
            st.altair_chart(line, use_container_width=True, theme=None)
        else:
            st.info("No timestamps available.")

    st.subheader("Recent analyses")
    table = pd.DataFrame({
        "ID": view["id"],
        "Predicted materials": view["predicted_materials"].apply(
            lambda ms: ", ".join(MATERIAL_GUIDANCE.get(m, {}).get("title", m) for m in ms) or "No category crossed threshold"),
        "Created": view["created"].dt.tz_convert("Asia/Kolkata").dt.strftime("%d %b %Y, %I:%M %p"),
        "Image stored": view["image_path"].apply(lambda p: "✅" if p else "—") if "image_path" in view else "—",
    })
    st.dataframe(table, use_container_width=True, hide_index=True)

    # ---- Private image preview gallery
    st.subheader("🖼️ Your saved prediction images")
    st.markdown(
        '<p class="subtle">Only images belonging to your signed-in account are shown. '
        'Preview links expire automatically after 10 minutes.</p>',
        unsafe_allow_html=True,
    )

    image_rows = view[view["image_path"].notna() & (view["image_path"].astype(str).str.strip() != "")].head(30)
    if image_rows.empty:
        st.info("No stored images are available for the selected history records.")
    else:
        st.caption(f"Showing up to 30 saved images ({len(image_rows)} in this view).")
        gallery_cols = st.columns(3, gap="medium")
        for index, (_, record) in enumerate(image_rows.iterrows()):
            with gallery_cols[index % 3]:
                image_url, image_error = create_signed_image_url(
                    record["image_path"],
                    st.session_state.supabase_access_token,
                    expires_in=600,
                )
                with st.container(border=True):
                    if image_url:
                        st.image(image_url, use_container_width=True)
                    else:
                        st.info("Image preview unavailable.")
                        st.caption(image_error or "No additional details.")

                    predicted = record.get("predicted_materials", [])
                    predicted_label = ", ".join(
                        MATERIAL_GUIDANCE.get(m, {}).get("title", m) for m in predicted
                    ) or "No category crossed threshold"
                    st.markdown(f"**{predicted_label}**")

                    created_at = record.get("created")
                    if pd.notna(created_at):
                        st.caption(
                            "Analyzed " +
                            created_at.tz_convert("Asia/Kolkata").strftime("%d %b %Y, %I:%M %p")
                        )
                    st.caption(f"Record ID: {record.get('id', '—')}")

    st.download_button("⬇️ Download as CSV", table.to_csv(index=False).encode("utf-8"), "ecosort_history.csv", "text/csv")
    st.caption("Material counts reflect model predictions, not verified ground truth or model accuracy.")


@st.cache_resource
def load_model():
    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    classes = checkpoint["classes"]
    model = models.efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(classes))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    with open(THRESHOLD_PATH, "r", encoding="utf-8") as f:
        thresholds = json.load(f)
    return model, classes, thresholds


# ----------------------------------------------------------------------------------
# Supabase helpers (unchanged behaviour)
# ----------------------------------------------------------------------------------
def get_supabase_credentials():
    """Read Supabase credentials from Streamlit secrets without exposing them."""
    try:
        url = st.secrets["SUPABASE_URL"].rstrip("/")
        key = st.secrets["SUPABASE_KEY"]
        if not url or not key:
            return None, None
        return url, key
    except Exception:
        return None, None


def auth_headers(access_token=None):
    """Headers for Supabase Auth/Data API; use the user's JWT when signed in."""
    supabase_url, supabase_key = get_supabase_credentials()
    if not supabase_url or not supabase_key:
        return None, None
    return supabase_url, {
        "apikey": supabase_key,
        "Authorization": f"Bearer {access_token or supabase_key}",
        "Content-Type": "application/json",
    }


def supabase_auth_request(path, payload):
    supabase_url, supabase_key = get_supabase_credentials()
    if not supabase_url or not supabase_key:
        return None, "Supabase secrets are not configured."
    try:
        response = requests.post(
            f"{supabase_url}/auth/v1/{path}",
            headers={"apikey": supabase_key, "Content-Type": "application/json"},
            json=payload,
            timeout=15,
        )
        try:
            data = response.json()
        except ValueError:
            data = {}
        if response.ok:
            return data, None
        message = data.get("msg") or data.get("message") or data.get("error_description") or data.get("error") or f"HTTP {response.status_code}"
        return None, str(message)
    except requests.RequestException:
        return None, "Could not reach Supabase Authentication."


def render_auth_gate():
    """Render sign-in/sign-up and return True only when a valid session exists."""
    if "supabase_access_token" not in st.session_state:
        st.session_state.supabase_access_token = None
        st.session_state.supabase_user = None
        st.session_state.supabase_refresh_token = None

    if st.session_state.supabase_access_token and st.session_state.supabase_user:
        with st.sidebar:
            st.divider()
            email = st.session_state.supabase_user.get("email", "user")
            st.markdown(f'<div class="user-card">Signed in as<b>{email}</b></div>', unsafe_allow_html=True)
            if st.button("Sign out", use_container_width=True):
                st.session_state.supabase_access_token = None
                st.session_state.supabase_refresh_token = None
                st.session_state.supabase_user = None
                st.rerun()
        return True

    st.markdown(
        """
        <div class="hero">
          <div class="eyebrow">Secure access</div>
          <h1>Welcome to EcoSort</h1>
          <p>Sign in or create an account to classify waste and keep your prediction history safely tied to you.</p>
          <div class="hero-pills"><span class="pill">🔒 Private by default</span><span class="pill">☁️ Cloud history</span><span class="pill">♻️ Recycling guidance</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    _, mid, _ = st.columns([1, 1.6, 1])
    with mid:
        login_tab, signup_tab = st.tabs(["Sign in", "Create account"])
        with login_tab:
            with st.form("supabase_signin_form"):
                email = st.text_input("Email address", key="signin_email", placeholder="you@example.com")
                password = st.text_input("Password", type="password", key="signin_password")
                submitted = st.form_submit_button("Sign in", type="primary", use_container_width=True)
            if submitted:
                if not email.strip() or not password:
                    st.error("Enter your email address and password.")
                else:
                    data, error = supabase_auth_request("token?grant_type=password", {"email": email.strip(), "password": password})
                    if error:
                        st.error(f"Sign-in failed: {error}")
                    elif data and data.get("access_token") and data.get("user"):
                        st.session_state.supabase_access_token = data["access_token"]
                        st.session_state.supabase_refresh_token = data.get("refresh_token")
                        st.session_state.supabase_user = data["user"]
                        st.rerun()
                    else:
                        st.error("Sign-in did not return a valid session. If email confirmation is enabled, confirm your email first.")
        with signup_tab:
            with st.form("supabase_signup_form"):
                new_email = st.text_input("Email address", key="signup_email", placeholder="you@example.com")
                new_password = st.text_input("Password (at least 8 characters)", type="password", key="signup_password")
                confirm_password = st.text_input("Confirm password", type="password", key="signup_confirm_password")
                submitted_signup = st.form_submit_button("Create account", type="primary", use_container_width=True)
            if submitted_signup:
                if not new_email.strip() or not new_password:
                    st.error("Enter an email address and password.")
                elif new_password != confirm_password:
                    st.error("The passwords do not match.")
                elif len(new_password) < 8:
                    st.error("Use a password with at least 8 characters.")
                else:
                    data, error = supabase_auth_request("signup", {"email": new_email.strip(), "password": new_password})
                    if error:
                        st.error(f"Account creation failed: {error}")
                    elif data and data.get("access_token") and data.get("user"):
                        st.session_state.supabase_access_token = data["access_token"]
                        st.session_state.supabase_refresh_token = data.get("refresh_token")
                        st.session_state.supabase_user = data["user"]
                        st.rerun()
                    else:
                        st.success("Account created. Check your email for a confirmation link, then return here and sign in.")
        st.caption("🔒 Your prediction history is protected by database policies. Use your own account to access it.")
    return False


def upload_image_to_supabase(image, original_name, access_token):
    """Upload an image to the private Supabase Storage bucket and return its path."""
    supabase_url, supabase_key = get_supabase_credentials()
    if not supabase_url or not supabase_key:
        return None, "Supabase secrets are not configured."

    suffix = Path(original_name).suffix.lower()
    content_types = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".webp": "image/webp"}
    content_type = content_types.get(suffix)
    if content_type is None:
        return None, "Unsupported image format for cloud storage."

    image_path = f"uploads/{datetime.now(timezone.utc):%Y/%m/%d}/{uuid.uuid4().hex}{suffix}"
    endpoint = f"{supabase_url}/storage/v1/object/waste-images/{image_path}"
    headers = {
        "apikey": supabase_key,
        "Authorization": f"Bearer {access_token}",
        "Content-Type": content_type,
        "x-upsert": "false",
    }
    try:
        response = requests.post(endpoint, headers=headers, data=image_to_bytes(image, suffix), timeout=20)
        if response.ok:
            return image_path, "Image uploaded to private cloud storage."
        return None, f"Image upload failed (HTTP {response.status_code})."
    except requests.RequestException:
        return None, "Could not reach Supabase Storage."


def image_to_bytes(image, suffix):
    """Serialize a PIL image using its allowed upload format."""
    buffer = BytesIO()
    image_format = {".jpg": "JPEG", ".jpeg": "JPEG", ".png": "PNG", ".webp": "WEBP"}[suffix]
    image.save(buffer, format=image_format)
    return buffer.getvalue()


def create_signed_image_url(image_path, access_token, expires_in=600):
    """Create a short-lived signed URL for an object in the private Storage bucket.

    Older records may contain a bucket prefix or a leading slash in image_path,
    so try a small set of normalized keys before returning the Storage error.
    """
    if not image_path:
        return None, "No stored image path."

    supabase_url, supabase_key = get_supabase_credentials()
    if not supabase_url or not supabase_key:
        return None, "Supabase credentials are not configured."

    raw_path = str(image_path).strip().replace("\\", "/")
    candidates = []
    for candidate in (raw_path, raw_path.lstrip("/")):
        candidate = candidate.strip("/")
        if candidate.startswith("waste-images/"):
            candidate = candidate[len("waste-images/"):]
        if candidate.startswith("public/waste-images/"):
            candidate = candidate[len("public/waste-images/"):]
        if candidate and candidate not in candidates:
            candidates.append(candidate)

    headers = {
        "apikey": supabase_key,
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
    }
    last_error = "No object key candidates were available."

    for clean_path in candidates:
        endpoint = f"{supabase_url}/storage/v1/object/sign/waste-images/{clean_path}"
        try:
            response = requests.post(
                endpoint,
                headers=headers,
                json={"expiresIn": int(expires_in)},
                timeout=12,
            )
            try:
                data = response.json()
            except ValueError:
                data = {}

            if not response.ok:
                last_error = f"HTTP {response.status_code}: {(response.text or 'No details')[:350]}"
                continue

            signed_path = data.get("signedURL") or data.get("signedUrl")
            if not signed_path:
                last_error = "Supabase did not return a signed image URL."
                continue
            if signed_path.startswith(("http://", "https://")):
                return signed_path, None
            if not signed_path.startswith("/"):
                signed_path = "/" + signed_path
            if signed_path.startswith("/storage/v1/"):
                return f"{supabase_url}{signed_path}", None
            return f"{supabase_url}/storage/v1{signed_path}", None
        except requests.RequestException:
            last_error = "Could not reach Supabase Storage."

    return None, (
        "Could not create a private image link. The saved database path may not match "
        f"the exact object key in the waste-images bucket. Saved path: {raw_path!r}. "
        f"Last Storage response: {last_error}"
    )


def save_prediction_to_supabase(predicted_materials, prediction_scores, image_path, access_token, user_id):
    """Save one prediction record and optional private storage path through Supabase's REST Data API."""
    supabase_url, supabase_key = get_supabase_credentials()
    if not supabase_url or not supabase_key:
        return False, "Supabase secrets are not configured. Prediction was not saved to cloud history."

    endpoint = f"{supabase_url}/rest/v1/prediction_history"
    headers = {
        "apikey": supabase_key,
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
        "Prefer": "return=minimal",
    }
    payload = {
        "user_id": user_id,
        "predicted_materials": predicted_materials,
        "prediction_scores": prediction_scores,
    }
    if image_path:
        payload["image_path"] = image_path
    try:
        response = requests.post(endpoint, headers=headers, json=payload, timeout=12)
        if response.ok:
            return True, "Prediction saved to cloud history."
        return False, f"Cloud history save failed (HTTP {response.status_code}). Check Supabase table permissions and RLS policy."
    except requests.RequestException:
        return False, "Could not reach Supabase. The prediction is available, but was not saved to cloud history."


def fetch_prediction_history(access_token, user_id, limit=200):
    """Fetch history only when the database's RLS policy explicitly permits it."""
    supabase_url, supabase_key = get_supabase_credentials()
    if not supabase_url or not supabase_key:
        return None, "Supabase credentials are not configured."
    endpoint = f"{supabase_url}/rest/v1/prediction_history"
    headers = {
        "apikey": supabase_key,
        "Authorization": f"Bearer {access_token}",
        "Accept": "application/json",
    }
    params = {
        "select": "id,predicted_materials,prediction_scores,image_path,created_at",
        "user_id": f"eq.{user_id}",
        "order": "created_at.desc",
        "limit": str(limit),
    }
    try:
        response = requests.get(endpoint, headers=headers, params=params, timeout=12)
        if response.ok:
            return response.json(), None
        if response.status_code in (401, 403):
            return None, "Database read access is disabled by Row Level Security. For privacy, this dashboard will not make your prediction history public."
        return None, f"Could not load cloud history (HTTP {response.status_code})."
    except requests.RequestException:
        return None, "Could not reach Supabase to load history."


# ----------------------------------------------------------------------------------
# Page dispatch
# ----------------------------------------------------------------------------------
if not render_auth_gate():
    st.stop()

if "History" in page:
    render_history_dashboard()
    render_footer("EcoSort · Student project prototype · Prediction trends are not a formal accuracy evaluation.")
    st.stop()

st.markdown(
    """
    <div class="hero">
      <div class="eyebrow">AI-powered material recognition</div>
      <h1>Give waste a<br>better next life.</h1>
      <p>Upload a photo to identify likely waste materials and get practical recycling and disposal guidance in seconds.</p>
      <div class="hero-pills"><span class="pill">♻️ 5 material categories</span><span class="pill">🧠 Multi-label detection</span><span class="pill">☁️ Auto-saved history</span></div>
    </div>
    <div class="steps">
      <div class="step"><div class="step-num">1</div><b>Upload</b><span>Add a clear photo</span></div>
      <div class="step"><div class="step-num">2</div><b>Analyze</b><span>Run the classifier</span></div>
      <div class="step"><div class="step-num">3</div><b>Review</b><span>See likely materials</span></div>
      <div class="step"><div class="step-num">4</div><b>Recycle</b><span>Follow the guidance</span></div>
    </div>
    """,
    unsafe_allow_html=True,
)

IMAGE_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

st.markdown('<div class="section-kicker">Start a new scan</div>', unsafe_allow_html=True)
st.subheader("Upload your waste image")
st.markdown('<p class="subtle">For best results, use a clear, well-lit photo where the waste is visible and the background is simple.</p>', unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Choose an image (JPG, JPEG, PNG or WEBP)",
    type=["jpg", "jpeg", "png", "webp"],
    help="Use one image at a time. Mixed-waste photos may be harder to classify accurately.",
    label_visibility="collapsed",
)

if uploaded_file is None:
    st.markdown(
        '<div class="card empty"><div class="big">🖼️</div><h3>Your analysis will appear here</h3>'
        '<p class="subtle">Upload an image above to see the predicted materials and disposal advice.</p></div>',
        unsafe_allow_html=True,
    )
    render_footer("EcoSort · Student project prototype · Always check local recycling rules.")
else:
    try:
        image = Image.open(uploaded_file).convert("RGB")
        scan_left, scan_right = st.columns([0.9, 1.1], gap="large")

        with scan_left:
            st.markdown('<div class="section-kicker">01 / Your image</div>', unsafe_allow_html=True)
            st.image(image, use_container_width=True)
            st.caption(f"📎 {uploaded_file.name} · {image.width} × {image.height} px")
            analyze = st.button("✨ Analyze waste", type="primary", use_container_width=True)

        if analyze:
            with st.spinner("Analyzing materials…"):
                model, classes, thresholds = load_model()
                tensor = IMAGE_TRANSFORM(image).unsqueeze(0)
                with torch.inference_mode():
                    scores = torch.sigmoid(model(tensor))[0].tolist()

            results = sorted(
                [(material, float(score), float(thresholds[material])) for material, score in zip(classes, scores)],
                key=lambda row: row[1],
                reverse=True,
            )
            predictions = [row for row in results if row[1] >= row[2]]

            image_path, upload_message = upload_image_to_supabase(image, uploaded_file.name, st.session_state.supabase_access_token)
            saved, save_message = save_prediction_to_supabase(
                [material for material, score, threshold in predictions],
                {material: round(score, 6) for material, score, threshold in results},
                image_path=image_path,
                access_token=st.session_state.supabase_access_token,
                user_id=st.session_state.supabase_user["id"],
            )

            with scan_right:
                st.markdown('<div class="section-kicker">02 / Analysis result</div>', unsafe_allow_html=True)

                # Cloud status
                if saved and image_path:
                    st.markdown('<div class="cloud-status cloud-ok">☁️ Saved to cloud history · image stored privately</div>', unsafe_allow_html=True)
                elif saved:
                    st.markdown(f'<div class="cloud-status cloud-ok">☁️ Saved to cloud history · {upload_message}</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="cloud-status cloud-warn">⚠️ {save_message} {upload_message}</div>', unsafe_allow_html=True)

                # Summary
                if predictions:
                    chips = "".join(chip_html(m) for m, _, _ in predictions)
                    best = predictions[0]
                    plural = "material" if len(predictions) == 1 else "materials"
                    st.markdown(
                        f'<div class="summary"><div class="label">Result</div>'
                        f'<div class="headline">{len(predictions)} {plural} detected</div>{chips}'
                        f'<div class="subtle" style="margin-top:.2rem">Strongest signal: <b>{MATERIAL_GUIDANCE[best[0]]["title"]}</b> ({best[1]:.0%})</div></div>',
                        unsafe_allow_html=True,
                    )
                else:
                    st.warning("No category crossed its detection threshold.")
                    st.write("Try a brighter, closer image with less background clutter.")

                tab_materials, tab_guide, tab_scores = st.tabs(["Likely materials", "Recycling guide", "All scores"])

                with tab_materials:
                    if predictions:
                        for material, score, threshold in predictions:
                            st.markdown(material_card_html(material, score, threshold, True), unsafe_allow_html=True)
                    else:
                        st.info("Nothing to show here. Check the *All scores* tab to see how close each category came.")

                with tab_guide:
                    if predictions:
                        for material, score, threshold in predictions:
                            item = MATERIAL_GUIDANCE[material]
                            with st.expander(f"{item['icon']} {item['title']} — how to handle it", expanded=len(predictions) == 1 or material == predictions[0][0]):
                                st.markdown(f'**Suggested bin:** {item["bin"]}')
                                st.markdown(guide_html("♻️ Recycling", item["recycling"], "#16704f"), unsafe_allow_html=True)
                                st.markdown(guide_html("⚠️ Disposal & safety", item["disposal"], "#b4690e"), unsafe_allow_html=True)
                    else:
                        st.info("No category-specific advice was selected for this image.")

                with tab_scores:
                    for material, score, threshold in results:
                        st.markdown(material_card_html(material, score, threshold, score >= threshold), unsafe_allow_html=True)
                    st.caption("The dark tick on each bar marks that category's detection threshold.")

                st.caption("Scores are model outputs, not calibrated probabilities of correctness.")

        else:
            with scan_right:
                st.markdown('<div class="section-kicker">02 / Analysis result</div>', unsafe_allow_html=True)
                st.markdown(
                    '<div class="card" style="padding:1.6rem"><h3>Ready when you are</h3>'
                    '<p class="subtle">Select <b>Analyze waste</b> to run the model and view likely material categories along with recycling advice.</p></div>',
                    unsafe_allow_html=True,
                )

        render_footer("EcoSort is a student project prototype. Predictions can be incomplete or incorrect, especially for cluttered scenes. Scores are not calibrated probabilities. Recycling rules vary by location.")
    except Exception as exc:
        st.error(f"Could not analyze this image: {exc}")
        st.caption("Check the application logs if the issue persists.")
