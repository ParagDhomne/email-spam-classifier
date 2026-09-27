import streamlit as st
import joblib


# --------------------------------------------------
# Load model
# --------------------------------------------------

model = joblib.load("models/spam_classifier_svm.joblib")


# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Email Spam Classifier",
    page_icon="📧",
    layout="centered"
)


# --------------------------------------------------
# Header
# --------------------------------------------------

st.title("📧 Email Spam Classifier")

st.markdown(
    """
    This application uses a **TF-IDF + Linear SVM** machine learning
    pipeline to classify emails as **Spam** or **Ham**.
    """
)

st.divider()


# --------------------------------------------------
# Example emails
# --------------------------------------------------

st.subheader("Try an Example")

example = st.selectbox(
    "Choose an example:",
    [
        "None",
        "Spam Example",
        "Ham Example"
    ]
)


# --------------------------------------------------
# Email examples
# --------------------------------------------------

spam_example = """
CONGRATULATIONS!

You have won $1,000,000.

Click here immediately to claim your FREE CASH PRIZE.

This offer expires TODAY.
"""

ham_example = """
Hi John,

Can we meet tomorrow at 10 AM to discuss the project?

Please let me know if the time works for you.

Regards,
Parag
"""


# --------------------------------------------------
# Email input
# --------------------------------------------------

if example == "Spam Example":
    default_text = spam_example

elif example == "Ham Example":
    default_text = ham_example

else:
    default_text = ""


email_text = st.text_area(
    "Email Content",
    value=default_text,
    height=250,
    placeholder="Paste the email content here..."
)


# --------------------------------------------------
# Classification
# --------------------------------------------------

if st.button("🔍 Classify Email", use_container_width=True):

    if not email_text.strip():

        st.warning("⚠️ Please enter an email first.")

    else:

        prediction = model.predict([email_text])[0]
        score = model.decision_function([email_text])[0]


        # ------------------------------------------
        # Result
        # ------------------------------------------

        st.subheader("Prediction")

        if prediction == "spam":

            st.error("🚨 SPAM")

        else:

            st.success("✅ HAM")


        # ------------------------------------------
        # Model information
        # ------------------------------------------

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Prediction",
                prediction.upper()
            )

        with col2:
            st.metric(
                "Decision Score",
                f"{score:.4f}"
            )


        # ------------------------------------------
        # Explain decision score
        # ------------------------------------------

        with st.expander("ℹ️ What is the Decision Score?"):

            st.write(
                """
                The Linear SVM produces a decision score that indicates
                which side of the classification boundary the email falls on.
                """
            )

            st.write(
                """
                **Positive score → Spam**
                
                **Negative score → Ham**
                
                A score close to zero means the model is less certain about
                the classification.
                """
            )


# --------------------------------------------------
# Footer
# --------------------------------------------------

st.divider()

st.caption(
    "Machine Learning Model: TF-IDF + Linear SVM | "
    "Dataset: Enron Spam Dataset"
)