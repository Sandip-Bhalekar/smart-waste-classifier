import json
from pathlib import Path

import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

PROJECT = Path(__file__).resolve().parent
CHECKPOINT_PATH = PROJECT / "checkpoints/multilabel/multilabel_best.pth"
THRESHOLD_PATH = PROJECT / "checkpoints/multilabel/thresholds_v2.json"

MATERIAL_GUIDANCE = {
    "glass": {
        "title": "Glass",
        "icon": "🫙",
        "short": "Bottles and jars",
        "recycling": "Separate glass bottles and jars where local glass recycling is available. Keep reusable containers intact when possible.",
        "disposal": "Handle broken glass carefully. Do not place broken glass loose in recycling bins unless your local service explicitly permits it.",
    },
    "paper": {
        "title": "Paper",
        "icon": "📄",
        "short": "Paper and newspapers",
        "recycling": "Keep clean, dry paper separate for paper recycling. Remove non-paper components where practical.",
        "disposal": "Greasy or heavily food-contaminated paper may not be accepted for recycling. Check local collection rules.",
    },
    "plastic": {
        "title": "Plastic",
        "icon": "🧴",
        "short": "Bottles and containers",
        "recycling": "Empty containers and check the plastic type and local recycling rules. Separate caps or labels only when your local service requests it.",
        "disposal": "Keep plastic waste out of drains and waterways. If the item is not accepted for recycling, use the appropriate local waste collection service.",
    },
    "cardboard": {
        "title": "Cardboard",
        "icon": "📦",
        "short": "Boxes and cartons",
        "recycling": "Flatten clean cardboard boxes and keep them dry. Remove excess tape and other packaging where practical.",
        "disposal": "Food-soiled or wet cardboard may not be accepted in paper recycling. Follow local collection rules.",
    },
    "metal": {
        "title": "Metal",
        "icon": "🥫",
        "short": "Cans and metal packaging",
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

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');

    :root { --eco-green: #16845b; --eco-dark: #12352a; --eco-mint: #e9f7ef; }
    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
    .stApp { background: #f6f8f6; }
    [data-testid="stHeader"] { background: rgba(246,248,246,.9); }
    .block-container { max-width: 1200px; padding-top: 1.7rem; padding-bottom: 3rem; }
    h1, h2, h3 { font-family: 'Manrope', sans-serif; color: #153b2e; letter-spacing: -0.03em; }
    .hero {
        background: linear-gradient(125deg, #12352a 0%, #176447 58%, #2b8b62 100%);
        padding: 2rem 2.1rem; border-radius: 24px; color: white; margin-bottom: 1.3rem;
        box-shadow: 0 12px 30px rgba(18,53,42,.12);
    }
    .hero h1 { color: white; font-size: clamp(2rem, 4vw, 3rem); line-height: 1.12; margin: .35rem 0 .7rem; }
    .hero p { color: #e0f2e8; font-size: 1.02rem; max-width: 760px; margin-bottom: 0; }
    .eyebrow { text-transform: uppercase; letter-spacing: .14em; font-size: .75rem; font-weight: 700; color: #b9e7cd; }
    .pill { display:inline-block; padding: .35rem .7rem; border: 1px solid rgba(255,255,255,.28); border-radius: 99px; color:#effcf3; font-size:.78rem; margin-top:1rem; }
    .section-kicker { color: #16845b; font-size: .75rem; font-weight: 800; text-transform: uppercase; letter-spacing: .12em; margin-bottom: .2rem; }
    .subtle { color: #61756b; font-size: .93rem; }
    .result-card { background: white; border: 1px solid #e2ebe4; border-radius: 17px; padding: 1rem 1.05rem; margin-bottom: .7rem; }
    .result-title { color:#173d2f; font-weight:700; font-size:1.05rem; }
    .result-meta { color:#708078; font-size:.8rem; }
    .material-icon { font-size:1.45rem; margin-right:.35rem; }
    .status-good { background:#e5f6eb; color:#176541; padding:.35rem .65rem; border-radius:99px; font-size:.77rem; font-weight:700; }
    .status-check { background:#fff4dc; color:#8a5a00; padding:.35rem .65rem; border-radius:99px; font-size:.77rem; font-weight:700; }
    div[data-testid="stFileUploader"] { background:white; border:1.5px dashed #b8d5c3; border-radius:18px; padding:1rem; }
    div[data-testid="stFileUploader"] section { border:0; }
    div.stButton > button { border-radius: 10px; font-weight:700; min-height:2.8rem; }
    div[data-testid="stExpander"] { background:white; border:1px solid #e2ebe4; border-radius:13px; }
    [data-testid="stMetric"] { background:white; border:1px solid #e2ebe4; border-radius:15px; padding:1rem; }
    .footer { border-top:1px solid #e1e9e3; margin-top:2.5rem; padding-top:1rem; color:#708078; font-size:.8rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("## ♻️ EcoSort")
    st.caption("Smart Waste Assistant")
    st.divider()
    st.markdown("### How it works")
    st.markdown(
        "1. Upload a waste photo\n"
        "2. Run the material classifier\n"
        "3. Review likely materials\n"
        "4. Follow recycling guidance"
    )
    st.divider()
    st.markdown("### Supported materials")
    for key, item in MATERIAL_GUIDANCE.items():
        st.markdown(f"{item['icon']} **{item['title']}**  \n<span style='color:#708078;font-size:.82rem'>{item['short']}</span>", unsafe_allow_html=True)
    st.divider()
    st.caption("Model: EfficientNet-B0 · Multi-label classification")

st.markdown(
    """
    <div class="hero">
      <div class="eyebrow">AI-powered material recognition</div>
      <h1>Give waste a<br>better next life.</h1>
      <p>Upload a photo to identify likely waste materials and get practical recycling and disposal guidance.</p>
      <span class="pill">♻️ 5 material categories · Image-based analysis</span>
    </div>
    """,
    unsafe_allow_html=True,
)

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

IMAGE_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

left_intro, right_intro = st.columns([1.25, 1])
with left_intro:
    st.markdown('<div class="section-kicker">Start a new scan</div>', unsafe_allow_html=True)
    st.subheader("Upload your waste image")
    st.markdown('<p class="subtle">For best results, use a clear, well-lit photo where the waste is visible.</p>', unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Choose an image (JPG, JPEG, PNG or WEBP)",
    type=["jpg", "jpeg", "png", "webp"],
    help="Use one image at a time. Mixed-waste photos may be harder to classify accurately.",
)

if uploaded_file is None:
    st.markdown(
        """
        <div class="result-card" style="text-align:center;padding:2rem">
          <div style="font-size:2.2rem">🖼️</div>
          <h3 style="margin:.5rem 0">Your analysis will appear here</h3>
          <p class="subtle">Upload an image above to see the predicted materials and disposal advice.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown('<div class="footer">EcoSort · Student project prototype · Always check local recycling rules.</div>', unsafe_allow_html=True)
else:
    try:
        image = Image.open(uploaded_file).convert("RGB")
        scan_left, scan_right = st.columns([0.95, 1.05], gap="large")

        with scan_left:
            st.markdown('<div class="section-kicker">01 / Your image</div>', unsafe_allow_html=True)
            st.image(image, use_container_width=True)
            st.caption(f"Image: {uploaded_file.name} · {image.width} × {image.height} px")
            analyze = st.button("✨ Analyze waste", type="primary", use_container_width=True)

        if analyze:
            with st.spinner("Analyzing materials…"):
                model, classes, thresholds = load_model()
                tensor = IMAGE_TRANSFORM(image).unsqueeze(0)
                with torch.inference_mode():
                    scores = torch.sigmoid(model(tensor))[0].tolist()

            results = sorted(
                [(material, float(score), float(thresholds[material]))
                 for material, score in zip(classes, scores)],
                key=lambda row: row[1],
                reverse=True,
            )
            predictions = [row for row in results if row[1] >= row[2]]

            with scan_right:
                st.markdown('<div class="section-kicker">02 / Analysis result</div>', unsafe_allow_html=True)
                if predictions:
                    st.success(f"{len(predictions)} material category/categories detected")
                    st.markdown("### Likely materials")
                    for material, score, threshold in predictions:
                        item = MATERIAL_GUIDANCE[material]
                        badge = "Above threshold" if score >= threshold else "Review"
                        st.markdown(
                            f"""
                            <div class="result-card">
                              <div style="display:flex;align-items:center;justify-content:space-between;gap:.5rem">
                                <div><span class="material-icon">{item['icon']}</span><span class="result-title">{item['title']}</span></div>
                                <span class="status-good">{badge}</span>
                              </div>
                              <div class="result-meta" style="margin:.45rem 0 .3rem">{item['short']}</div>
                              <div style="height:7px;background:#e8efe9;border-radius:99px;overflow:hidden">
                                <div style="height:7px;width:{max(0,min(100,score*100)):.1f}%;background:#16845b;border-radius:99px"></div>
                              </div>
                              <div class="result-meta" style="text-align:right;margin-top:.3rem">Model score: {score:.1%}</div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                else:
                    st.warning("No category crossed its detection threshold.")
                    st.write("Try a brighter, closer image with less background clutter.")

                st.markdown("### Recycling guide")
                if predictions:
                    for material, score, threshold in predictions:
                        item = MATERIAL_GUIDANCE[material]
                        with st.expander(f"{item['icon']} {item['title']} — how to handle it", expanded=True):
                            st.markdown("**Recycling**")
                            st.write(item["recycling"])
                            st.markdown("**Disposal & safety**")
                            st.write(item["disposal"])
                else:
                    st.info("No category-specific advice was selected for this image.")

                with st.expander("See all model scores"):
                    for material, score, threshold in results:
                        st.write(f"**{MATERIAL_GUIDANCE[material]['title']}** · {score:.1%} score · {threshold:.0%} threshold")
                        st.progress(max(0.0, min(1.0, score)))
                st.caption("Scores are model outputs, not calibrated probabilities of correctness.")

        else:
            with scan_right:
                st.markdown('<div class="section-kicker">02 / Analysis result</div>', unsafe_allow_html=True)
                st.markdown(
                    """
                    <div class="result-card" style="padding:1.5rem">
                      <h3 style="margin-top:0">Ready when you are</h3>
                      <p class="subtle">Select <b>Analyze waste</b> to run the model and view likely material categories.</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        st.markdown(
            '<div class="footer">EcoSort is a student project prototype. Predictions can be incomplete or incorrect, especially for cluttered scenes. Scores are not calibrated probabilities. Recycling rules vary by location.</div>',
            unsafe_allow_html=True,
        )
    except Exception as exc:
        st.error(f"Could not analyze this image: {exc}")
        st.caption("Check the application logs if the issue persists.")
