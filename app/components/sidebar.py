import streamlit as st


def render_sidebar() -> str:
    """Render the main sidebar and return the selected page."""
    with st.sidebar:
        st.title("ML Dashboard")
        st.markdown("Select a page:")

        page = st.radio(
            "Navigation",
            options=["Overview", "Classification", "Regression"],
            index=0,
        )

    return page
