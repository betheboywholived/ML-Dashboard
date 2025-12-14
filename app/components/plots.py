import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def render_placeholder_plot(message: str = "Add plots here later.") -> None:
    """Simple placeholder to reserve space for plots."""
    st.info(message)


def example_plotly_placeholder(df: pd.DataFrame | None = None) -> None:
    """
    Example Plotly plot placeholder.

    If no DataFrame is provided, a tiny dummy dataset is used just to
    reserve the space and validate that Plotly is wired correctly.
    """
    if df is None or df.empty:
        df = pd.DataFrame(
            {
                "x": [1, 2, 3, 4],
                "y": [1, 4, 9, 16],
                "series": ["A", "A", "B", "B"],
            }
        )

    fig = px.line(
        df,
        x="x",
        y="y",
        color="series",
        title="Plotly placeholder chart",
        markers=True,
    )

    st.plotly_chart(fig, use_container_width=True)


def example_metric_cards_placeholder() -> None:
    """Example of where Plotly-based KPI/metric visuals could go."""
    fig = go.Figure()
    fig.add_trace(
        go.Indicator(
            mode="number+delta",
            value=0.85,
            delta={"reference": 0.8},
            title={"text": "Accuracy"},
        )
    )
    st.plotly_chart(fig, use_container_width=True)
