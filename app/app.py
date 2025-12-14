import streamlit as st


def render_overview_page() -> None:
    """Landing / overview content."""
    st.title("ML Dashboard")
    st.subheader("Overview")
    st.write(
        "This dashboard will host experiments for both **classification** and "
        "**regression** problems.\n\n"
        "Use the navigation on the left (Streamlit pages) to switch between pages."
    )


def main() -> None:
    st.set_page_config(
        page_title="ML Dashboard",
        page_icon="📊",
        layout="wide",
    )

    render_overview_page()


if __name__ == "__main__":
    main()
