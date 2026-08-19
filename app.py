
import streamlit as st
import joblib
from google import genai


# ── Page Config ───────────────────────────────
st.set_page_config(
    page_title="Review Analyzer",
    page_icon="💬"
)

st.title("💬 Customer Review Analyzer")


# ── Sidebar ───────────────────────────────────
with st.sidebar:

    st.header("⚙️ Settings")

    gemini_api_key = st.text_input(
        "Gemini API Key",
        type="password",
        placeholder="AIza..."
    )

    st.markdown("---")

    st.header("🤖 Select Model")

    selected_model = st.selectbox(
        "Gemini Model",
        [
            "gemini-2.5-flash",
            "gemini-2.5-flash-lite"
        ]
    )


# ── Load Trained ML Model & Vectorizer ───────
@st.cache_resource
def load_model():

    vectorizer = joblib.load(
        "tfidf_vectorizer.pkl"
    )

    model = joblib.load(
        "linear_svc_model.pkl"
    )

    return vectorizer, model


try:

    vectorizer, model = load_model()

    st.sidebar.success(
        "✅ ML Model Loaded Successfully"
    )

except FileNotFoundError:

    st.sidebar.error(
        "❌ tfidf_vectorizer.pkl or "
        "linear_svc_model.pkl not found."
    )

    st.stop()


# ── Generate 3 Replies Using Gemini ──────────
def generate_gemini_replies(
    review,
    analysis,
    api_key,
    model_name
):

    client = genai.Client(
        api_key=api_key
    )

    prompt = f"""
You are a professional customer support agent.

Customer Review:
"{review}"

Predicted Analysis:
- Sentiment: {analysis['sentiment']}
- Issue: {analysis['issue']}
- Priority: {analysis['priority']}

Generate exactly 3 different customer support replies.

Rules for ALL replies:

1. Start with:
Dear Customer,

2. Use "we" instead of "I".

3. If the sentiment is negative, apologize sincerely.

4. Address the customer's issue appropriately.

5. Be polite, empathetic and professional.

6. Keep each reply within 3-4 lines.

7. End each reply with:
Warm regards,
Support Team

Make the 3 replies slightly different in wording.

Return ONLY the 3 replies.

Separate them using:

---REPLY 2---

and

---REPLY 3---
"""

    response = client.models.generate_content(
        model=model_name,
        contents=prompt
    )

    text = response.text.strip()

    # Split Gemini response into 3 replies
    parts = text.split("---REPLY 2---")

    if len(parts) == 2:

        reply1 = parts[0].strip()

        remaining = parts[1].split(
            "---REPLY 3---"
        )

        if len(remaining) == 2:

            reply2 = remaining[0].strip()
            reply3 = remaining[1].strip()

            return [
                reply1,
                reply2,
                reply3
            ]

    # If Gemini formatting is unexpected
    return [
        text,
        text,
        text
    ]


# ── Fallback Replies ──────────────────────────
def generate_fallback_replies(
    review,
    analysis
):

    sentiment = str(
        analysis["sentiment"]
    ).lower()

    issue = str(
        analysis["issue"]
    )

    priority = str(
        analysis["priority"]
    )

    # Negative sentiment
    if "negative" in sentiment:

        replies = [

            f"""Dear Customer,

We sincerely apologize for the inconvenience regarding your {issue.lower()} issue. We understand your concern and will work to resolve it as quickly as possible. Your case has been marked as {priority.lower()} priority.

Warm regards,
Support Team""",

            f"""Dear Customer,

We are sorry to hear about your experience with the {issue.lower()} issue. We understand how frustrating this can be and will do our best to assist you promptly.

Warm regards,
Support Team""",

            f"""Dear Customer,

We sincerely regret the inconvenience caused by this {issue.lower()} issue. We appreciate you bringing this to our attention and will ensure that the matter receives appropriate attention.

Warm regards,
Support Team"""
        ]

    # Positive sentiment
    elif "positive" in sentiment:

        replies = [

            f"""Dear Customer,

Thank you for sharing your positive experience with us. We are glad to know that your experience regarding {issue.lower()} was satisfactory. We truly appreciate your support.

Warm regards,
Support Team""",

            f"""Dear Customer,

Thank you for your kind feedback. We are pleased to hear about your experience and appreciate you taking the time to share it with us.

Warm regards,
Support Team""",

            f"""Dear Customer,

We greatly appreciate your positive feedback. It is wonderful to know that we were able to provide a satisfactory experience. Thank you for choosing us.

Warm regards,
Support Team"""
        ]

    # Neutral sentiment
    else:

        replies = [

            f"""Dear Customer,

Thank you for contacting us regarding your {issue.lower()} concern. We understand your request and will review the matter carefully to provide the appropriate assistance.

Warm regards,
Support Team""",

            f"""Dear Customer,

Thank you for bringing this matter to our attention. We understand your concern regarding {issue.lower()} and will work towards providing an appropriate resolution.

Warm regards,
Support Team""",

            f"""Dear Customer,

We appreciate you sharing your feedback with us. We have noted your {issue.lower()} concern and will take the necessary steps to assist you.

Warm regards,
Support Team"""
        ]

    return replies


# ── Full Processing Pipeline ──────────────────
def process_review(
    text,
    api_key,
    model_name
):

    # Convert text to TF-IDF
    vectorized_text = vectorizer.transform(
        [text]
    )

    # Predict using MultiOutput LinearSVC
    prediction = model.predict(
        vectorized_text
    )[0]

    sentiment = prediction[0]
    issue = prediction[1]
    priority = prediction[2]

    analysis = {

        "sentiment": sentiment,

        "issue": issue,

        "priority": priority
    }


    # ── Try Gemini ─────────────────────────────

    if api_key:

        try:

            replies = generate_gemini_replies(
                text,
                analysis,
                api_key,
                model_name
            )

            source = "Gemini AI"

        except Exception as e:

            st.warning(
                "⚠️ Gemini API unavailable. "
                "Using built-in replies instead."
            )

            replies = generate_fallback_replies(
                text,
                analysis
            )

            source = "Built-in Fallback"


    # ── No API Key ────────────────────────────

    else:

        replies = generate_fallback_replies(
            text,
            analysis
        )

        source = "Built-in Fallback"


    return {

        "review": text,

        "sentiment": sentiment,

        "issue": issue,

        "priority": priority,

        "replies": replies,

        "source": source
    }


# ── Main UI ───────────────────────────────────
st.subheader(
    "Enter Customer Review"
)

review_text = st.text_area(
    "",
    height=150,
    placeholder=(
        "Example: The product arrived damaged "
        "and I want a refund immediately."
    )
)


if st.button(
    "Analyze Review",
    type="primary",
    use_container_width=True
):

    if not review_text.strip():

        st.warning(
            "⚠️ Please enter a review."
        )

    else:

        with st.spinner(
            "Analyzing review..."
        ):

            try:

                result = process_review(
                    review_text,
                    gemini_api_key,
                    selected_model
                )


                # ── Display Analysis ─────────────

                st.markdown("---")

                col1, col2, col3 = st.columns(3)

                col1.metric(
                    "😊 Sentiment",
                    result["sentiment"]
                )

                col2.metric(
                    "📌 Issue",
                    result["issue"]
                )

                col3.metric(
                    "🚨 Priority",
                    result["priority"]
                )


                # ── Reply Section ───────────────

                st.markdown("---")

                st.subheader(
                    "🤖 Customer Support Replies"
                )

                st.caption(
                    f"Generated using: "
                    f"{result['source']}"
                )


                # ── Three Reply Options ─────────

                for i, reply in enumerate(
                    result["replies"],
                    start=1
                ):

                    st.markdown(
                        f"### 💬 Reply Option {i}"
                    )

                    st.info(reply)

                    st.download_button(
                        label=(
                            f"⬇️ Download Reply {i}"
                        ),
                        data=reply,
                        file_name=(
                            f"customer_reply_{i}.txt"
                        ),
                        mime="text/plain",
                        key=f"download_{i}"
                    )

            except Exception as e:

                st.error(
                    f"❌ Error: {e}"
                )