
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
        "recycling": (
            "Separate glass bottles and jars where local glass recycling "
            "is available. Keep reusable containers intact when possible."
        ),
        "disposal": (
            "Handle broken glass carefully and follow local waste rules. "
            "Do not place broken glass loose in recycling bins unless "
            "your local service explicitly permits it."
        ),
    },
    "paper": {
        "title": "Paper",
        "recycling": (
            "Keep clean, dry paper separate for paper recycling. Flatten "
            "paper packaging and remove non-paper components where practical."
        ),
        "disposal": (
            "Greasy or heavily food-contaminated paper may not be accepted "
            "for recycling. Check local collection rules."
        ),
    },
    "plastic": {
        "title": "Plastic",
        "recycling": (
            "Empty containers and check the plastic type and local recycling "
            "rules. Separate caps or labels only when your local service "
            "requests it."
        ),
        "disposal": (
            "Keep plastic waste out of drains and waterways. If the item "
            "is not accepted for recycling, use the appropriate local waste "
            "collection service."
        ),
    },
    "cardboard": {
        "title": "Cardboard",
        "recycling": (
            "Flatten clean cardboard boxes and keep them dry. Remove "
            "excess tape and other packaging where practical."
        ),
        "disposal": (
            "Food-soiled or wet cardboard may not be accepted in paper "
            "recycling. Follow local collection rules."
        ),
    },
    "metal": {
        "title": "Metal",
        "recycling": (
            "Separate accepted metal cans and packaging for metal recycling. "
            "Empty containers before placing them in the appropriate bin."
        ),
        "disposal": (
            "Handle sharp metal edges carefully. Aerosol cans, batteries, "
            "and hazardous metal-containing items may require special "
            "collection rather than ordinary recycling."
        ),
    },
}

@st.cache_resource
def load_model():
    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location="cpu",
        weights_only=False,
    )

    classes = checkpoint["classes"]

    model = models.efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(
        model.classifier[1].in_features, len(classes)
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    with open(THRESHOLD_PATH, "r", encoding="utf-8") as f:
        thresholds = json.load(f)

    return model, classes, thresholds

IMAGE_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

st.set_page_config(
    page_title="Smart Waste Assistant",
    page_icon="♻️",
    layout="wide",
)

st.title("♻️ Smart Waste Classification Assistant")
st.write(
    "Upload a photograph containing waste to identify likely materials "
    "and view general recycling and disposal guidance."
)

st.info(
    "This model predicts material categories present in the image. "
    "It does not locate individual objects, and predictions can be "
    "incomplete or incorrect, especially in cluttered scenes."
)

uploaded_file = st.file_uploader(
    "Upload a waste image",
    type=["jpg", "jpeg", "png", "webp"],
)

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")

        left, right = st.columns([1, 1])

        with left:
            st.subheader("Uploaded image")
            st.image(image, use_container_width=True)

        with st.spinner("Analyzing image..."):
            model, classes, thresholds = load_model()
            tensor = IMAGE_TRANSFORM(image).unsqueeze(0)

            with torch.inference_mode():
                scores = torch.sigmoid(model(tensor))[0].tolist()

        predictions = []

        for material, score in zip(classes, scores):
            threshold = float(thresholds[material])

            if score >= threshold:
                predictions.append((material, score, threshold))

        with right:
            st.subheader("Detected materials")

            if predictions:
                for material, score, threshold in sorted(
                    predictions, key=lambda x: x[1], reverse=True
                ):
                    st.markdown(
                        f"**{material.title()}** — model score: {score:.1%}"
                    )
                    st.progress(min(max(score, 0.0), 1.0))

                st.caption(
                    "Scores are model outputs, not calibrated probabilities "
                    "of correctness."
                )
            else:
                st.warning(
                    "No material exceeded its detection threshold. "
                    "Try a clearer image with waste occupying more of the frame."
                )

        st.divider()
        st.subheader("Recycling and disposal guidance")

        if predictions:
            for material, score, threshold in sorted(
                predictions, key=lambda x: x[1], reverse=True
            ):
                guidance = MATERIAL_GUIDANCE[material]

                with st.expander(guidance["title"], expanded=True):
                    st.markdown("**Recycling**")
                    st.write(guidance["recycling"])
                    st.markdown("**Disposal and safety**")
                    st.write(guidance["disposal"])
        else:
            st.write(
                "No material-specific guidance was selected because "
                "the model did not detect a category above its threshold."
            )

        with st.expander("All material scores"):
            for material, score in zip(classes, scores):
                st.write(
                    f"{material.title()}: {score:.1%} "
                    f"(threshold: {float(thresholds[material]):.1%})"
                )

        st.caption(
            "Recommendations are general guidance. Accepted materials and "
            "collection procedures vary by location; check your local "
            "waste-management service."
        )

    except Exception as exc:
        st.error(f"Could not analyze this image: {exc}")
        st.exception(exc)
