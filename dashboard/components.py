from __future__ import annotations

import streamlit as st


def metric_card(title: str, value: str, hint: str, *, tone: str = "neutral") -> None:
    """Render a polished metric card with a muted tone."""
    color = {
        "neutral": "var(--ink)",
        "positive": "var(--save)",
        "warning": "var(--spend)",
        "danger": "var(--fail)",
    }.get(tone, "var(--ink)")
    st.markdown(
        f"""
        <div class="panel metric-card">
            <div class="metric-title">{title}</div>
            <div class="metric-value" style="color:{color};">{value}</div>
            <div class="metric-hint">{hint}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def section_title(title: str, subtitle: str | None = None) -> None:
    """Render a simple section heading."""
    st.markdown(f"<div class='section-title'>{title}</div>", unsafe_allow_html=True)
    if subtitle:
        st.caption(subtitle)


def status_badge(text: str, *, tone: str = "neutral") -> None:
    """Render a compact badge used for state and health labels."""
    color = {
        "neutral": "var(--accent)",
        "positive": "var(--save)",
        "warning": "var(--spend)",
        "danger": "var(--fail)",
    }.get(tone, "var(--accent)")
    st.markdown(
        f"<span class='badge' style='border-color:{color}; color:{color};'>{text}</span>",
        unsafe_allow_html=True,
    )
