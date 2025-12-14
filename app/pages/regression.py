import streamlit as st

from components.plots import render_placeholder_plot


def render_page() -> None:
    st.title("Regression")
    st.subheader("Regression Experiments")

    st.markdown(
        "This page will host regression workflows "
        "(e.g., predicting continuous targets)."
    )

    st.markdown("### Data Input")
    uploaded_file = st.file_uploader(
        "Upload a CSV dataset (optional, for now)",
        type=["csv"],
    )

    if uploaded_file is not None:
        st.success("File uploaded. You can hook this into your pipeline later.")
        # df = pd.read_csv(uploaded_file)
        # ...regression-specific preprocessing/EDA...
    else:
        st.info("No dataset uploaded yet.")

    st.markdown("### Models & Results")
    render_placeholder_plot("Regression model performance plots will appear here.")


# Ensure content is rendered when this script is used as a Streamlit page
render_page()
