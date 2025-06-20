import streamlit as st
import json
import torch
from sentence_transformers import SentenceTransformer, util
import os
from openai import OpenAI
from dotenv import load_dotenv

# ------------------ 🔐 Load OpenAI API Key ------------------
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ------------------ 🧠 Load Embedding Model ------------------
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2", device='cpu')

model = load_model()

# ------------------ 📂 Load Data ------------------
@st.cache_data
def load_data():
    with open("questionnaire_with_embeddings.json", "r", encoding="utf-8") as f:
        return json.load(f)

data = load_data()

# ------------------ 🧠 GPT Classifier ------------------
def classify_with_gpt(question, ideal_answer, user_answer, model_name="gpt-3.5-turbo"):
    prompt = f"""
You are a data compliance officer. Classify the user's answer based on how well it aligns with the expected answer.

---
Question:
{question}

Expected (ideal) answer:
{ideal_answer}

User's answer:
{user_answer}

Classify the user's answer into one of the following:
- Fully Compliant
- Partially Compliant
- Not Compliant
- Not Applicable

Respond with ONLY the label.
"""
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"❌ GPT Error: {str(e)}"

# ------------------ 🎨 Streamlit UI ------------------

st.set_page_config(page_title="Compliance Classifier", layout="centered")
st.title("🔐 Data Compliance Classifier")

# 1. Select Question
questions = [item["question"] for item in data]
selected_idx = st.selectbox("📋 Select a question", range(len(questions)), format_func=lambda i: questions[i])
entry = data[selected_idx]

# ✅ Show full question clearly
st.markdown(f"### ❓ Selected Question:\n{entry['question']}")

# 2. Show Ideal Answer
if st.checkbox("👁 Show expected (ideal) answer"):
    st.markdown(f"**Expected Answer:** {entry['ideal_answer']}")

# 3. User Answer
user_input = st.text_area("✍️ Your Answer")

# 4. Choose Method
mode = st.radio("⚙️ Choose classification method:", ["Embedding Model", "GPT Model"])

# 5. Classify Button
if st.button("🔍 Classify Answer"):
    if not user_input.strip():
        st.warning("Please type an answer before classification.")
    else:
        if mode == "Embedding Model":
            ideal_embedding = torch.tensor(entry["ideal_embedding"])
            user_embedding = model.encode(user_input, convert_to_tensor=True)
            score = util.cos_sim(user_embedding, ideal_embedding).item()

            # Label Logic
            if "not applicable" in user_input.lower() or user_input.strip().lower() in ["na", "n/a"]:
                label = "🚫 Not Applicable"
            elif score >= 0.85:
                label = "✅ Fully Compliant"
            elif score >= 0.65:
                label = "🟡 Partially Compliant"
            else:
                label = "❌ Not Compliant"

            st.markdown(f"### 📊 Result: {label}")
            st.markdown(f"**Cosine Similarity Score:** `{score:.4f}`")

        else:
            label = classify_with_gpt(entry["question"], entry["ideal_answer"], user_input)
            st.markdown(f"### 🤖 GPT Classification Result: {label}")

        # Ground Truth
        st.markdown(f"**🧾 Ground Truth Label:** `{entry['label']}`")
