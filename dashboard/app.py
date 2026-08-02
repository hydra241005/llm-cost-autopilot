from __future__ import annotations

from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

from components import metric_card, section_title, status_badge
from logic import build_api_url, summarize_provider_status

st.set_page_config(page_title="LLM Cost Autopilot", page_icon="📈", layout="wide")

API_BASE = st.sidebar.text_input("API Base URL", value="http://127.0.0.1:8000")

with open("dashboard/theme.css", encoding="utf-8") as handle:
    st.markdown(f"<style>{handle.read()}</style>", unsafe_allow_html=True)

st.sidebar.title("Mission Control")
st.sidebar.caption("Commercial telemetry for the routing fleet")

pages = [
    "Overview",
    "Playground",
    "Analytics",
    "Request Explorer",
    "Classifier Lifecycle",
    "Provider Operations",
    "Operational Events",
    "Settings",
]
page = st.sidebar.radio("Navigation", pages)


@st.cache_data(show_spinner=False)
def fetch_json(path: str) -> Any:
    url = build_api_url(API_BASE, path)
    response = requests.get(url, timeout=10)
    response.raise_for_status()
    return response.json()


def render_header(title: str, subtitle: str) -> None:
    st.markdown(f"<div class='kicker'>{subtitle}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='hero'>{title}</div>", unsafe_allow_html=True)


def render_error(message: str) -> None:
    st.markdown(
        f"<div class='panel' style='border-color:var(--fail); color:var(--fail);'>{message}</div>",
        unsafe_allow_html=True,
    )


if page == "Overview":
    render_header("Executive overview", "Mission Control")
    try:
        health = fetch_json("/health")
        overview = fetch_json("/admin/overview")
        metrics = fetch_json("/admin/metrics")
        analytics = fetch_json("/admin/analytics")
    except requests.RequestException as exc:
        render_error(f"Couldn't reach the API: {exc}")
        st.stop()

    summary = overview["summary"]
    provider_summary = summarize_provider_status(overview["provider_health"].get("providers", []))

    st.write("")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("Fleet status", str(provider_summary["overall_status"]).upper(), "Provider admission state", tone="positive" if provider_summary["overall_status"] == "healthy" else "warning")
    with c2:
        metric_card("Total requests", f"{summary.get('total_requests', 0)}", "Observed routing decisions", tone="neutral")
    with c3:
        metric_card("Success rate", f"{summary.get('success_rate', 0):.0%}", "Operational reliability", tone="positive")
    with c4:
        metric_card("Latency", f"{summary.get('latency_ms', 0)} ms", "Median response time", tone="neutral")

    st.write("")
    c5, c6, c7, c8 = st.columns(4)
    with c5:
        metric_card("Daily savings", f"${summary.get('daily_savings', 0):,.0f}", "Daily cost avoidance", tone="positive")
    with c6:
        metric_card("Weekly savings", f"${summary.get('weekly_savings', 0):,.0f}", "Weekly run-rate", tone="positive")
    with c7:
        metric_card("Monthly savings", f"${summary.get('monthly_savings', 0):,.0f}", "Monthly opportunity", tone="positive")
    with c8:
        metric_card("Annual savings", f"${summary.get('annual_savings', 0):,.0f}", "Estimated annual impact", tone="positive")

    st.write("")
    left, right = st.columns([2.1, 1])
    with left:
        section_title("Savings trend")
        cost_frame = pd.DataFrame(analytics.get("cost_over_time", []))
        fig = px.line(
            cost_frame,
            x="date",
            y=["cost", "savings"],
            markers=True,
            color_discrete_sequence=["#6C8CFF", "#4ADE9C"],
        )
        fig.update_layout(height=320, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=8, r=8, t=8, b=8), legend_title_text="")
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    with right:
        section_title("Live policy")
        st.markdown(
            f"""
            <div class="panel">
                <div class="metric-title">Current routing policy</div>
                <div class="metric-value" style="color:var(--accent);">{overview['routing_policy']['name']}</div>
                <div class="metric-hint">Version {overview['routing_policy']['version']} · confidence threshold {overview['routing_policy']['confidence_threshold']:.2f}</div>
                <div class="metric-hint">Active classifier {summary.get('active_classifier', 'n/a')}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.write("")
        st.markdown(
            f"""
            <div class="panel">
                <div class="metric-title">Health summary</div>
                <div class="metric-value" style="color:var(--save);">{provider_summary['healthy_count']}/{provider_summary['healthy_count'] + provider_summary['degraded_count']} healthy</div>
                <div class="metric-hint">Environment {health.get('environment', 'local')} · providers {', '.join(p['provider'] for p in overview['provider_health']['providers'])}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.write("")
    section_title("Provider health")
    provider_rows = [
        {
            "provider": item.get("provider", "n/a"),
            "state": item.get("state", "unknown"),
            "healthy": item.get("healthy", False),
            "availability": item.get("availability"),
            "p95_latency_ms": item.get("p95_latency_ms"),
            "total_calls": item.get("total_calls", 0),
        }
        for item in overview["provider_health"].get("providers", [])
    ]
    st.dataframe(pd.DataFrame(provider_rows), use_container_width=True, hide_index=True)

elif page == "Playground":
    render_header("AI Playground", "Interactive routing preview")
    prompt = st.text_area("Prompt", value="Summarize the latest quarterly performance with a concise executive brief.", height=140)
    st.selectbox("Task type", ["summarization", "coding", "analysis"], key="task_type")
    if st.button("Run routing preview", use_container_width=True):
        try:
            payload = fetch_json("/admin/playground")
            st.session_state["playground_payload"] = payload
            st.success("Routing preview loaded from the API payload.")
        except requests.RequestException as exc:
            render_error(f"Couldn't load playground payload: {exc}")

    payload = st.session_state.get("playground_payload") if "playground_payload" in st.session_state else None
    if not payload:
        payload = fetch_json("/admin/playground")

    st.write("")
    left, right = st.columns([1, 1])
    with left:
        section_title("Routing receipt")
        routing = payload["routing"]
        st.markdown(
            f"""
            <div class="panel">
                <div class="metric-title">Complexity tier</div>
                <div class="metric-value" style="color:var(--accent);">Tier {routing['tier']}</div>
                <div class="metric-hint">Confidence {routing['confidence']:.0%}</div>
                <div class="metric-hint">Selected model {routing['selected_model']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.write("")
        st.json(routing.get("features", {}))
    with right:
        section_title("Response")
        st.markdown(
            f"""
            <div class="panel">
                <div class="metric-title">Final response</div>
                <div class="metric-hint">{payload['response']['text']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.write("")
        st.json({
            "candidate_models": routing.get("candidate_models", []),
            "estimated_cost": routing.get("estimated_cost", {}),
            "latency_ms": routing.get("latency_ms", 0),
            "tokens": routing.get("tokens", {}),
        })

elif page == "Analytics":
    render_header("Analytics", "Professional operational charts")
    try:
        analytics = fetch_json("/admin/analytics")
    except requests.RequestException as exc:
        render_error(f"Couldn't load analytics payload: {exc}")
        st.stop()

    provider_usage = pd.DataFrame(analytics.get("provider_usage", []))
    cost_frame = pd.DataFrame(analytics.get("cost_over_time", []))
    latency_frame = pd.DataFrame(analytics.get("latency_distribution", []))
    confidence_frame = pd.DataFrame(analytics.get("confidence_histogram", []))

    c1, c2 = st.columns(2)
    with c1:
        fig = px.pie(provider_usage, names="provider", values="value", hole=0.5)
        fig.update_layout(height=280, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=8, r=8, t=8, b=8))
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    with c2:
        fig = px.line(cost_frame, x="date", y=["cost", "savings"], markers=True, color_discrete_sequence=["#6C8CFF", "#4ADE9C"])
        fig.update_layout(height=280, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=8, r=8, t=8, b=8), legend_title_text="")
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    c3, c4 = st.columns(2)
    with c3:
        fig = px.bar(latency_frame, x="bucket", y="value", color="bucket", color_discrete_sequence=["#6C8CFF", "#4ADE9C", "#F2B441"])
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    with c4:
        fig = px.bar(confidence_frame, x="bucket", y="value", color="bucket", color_discrete_sequence=["#6C8CFF", "#4ADE9C", "#F2B441"])
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    st.write("")
    c5, c6, c7 = st.columns(3)
    for col, key in zip((c5, c6, c7), ("escalation_rate", "retry_rate", "circuit_events")):
        with col:
            frame = pd.DataFrame(analytics.get(key, []))
            fig = px.line(frame, x=frame.columns[0], y=frame.columns[1], markers=True) if len(frame.columns) == 2 else px.line(frame, x=frame.columns[0], y=frame.columns[1])
            fig.update_layout(height=220, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=8, r=8, t=8, b=8))
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

elif page == "Request Explorer":
    render_header("Request Explorer", "Search, filter, inspect, and explain")
    try:
        explorer = fetch_json("/admin/requests")
    except requests.RequestException as exc:
        render_error(f"Couldn't load request explorer: {exc}")
        st.stop()

    items = explorer.get("items", [])
    df = pd.DataFrame(items)
    if df.empty:
        st.info("No request rows available yet; the backend payload will populate this page as routing decisions accumulate.")
        st.stop()

    search_text = st.text_input("Search", placeholder="request id or model")
    status_filter = st.selectbox("Status", ["All", *sorted({str(value) for value in df["status"].tolist()})])
    sort_mode = st.selectbox("Sort by", ["timestamp", "latency_ms", "cost_usd", "confidence"])

    filtered = df.copy()
    if search_text:
        mask = filtered.astype(str).apply(lambda row: search_text.lower() in " ".join(row.astype(str)).lower(), axis=1)
        filtered = filtered[mask]
    if status_filter != "All":
        filtered = filtered[filtered["status"] == status_filter]
    filtered = filtered.sort_values(sort_mode, ascending=False)

    page_size = 5
    page_index = st.number_input("Page", min_value=1, max_value=max(1, (len(filtered) + page_size - 1) // page_size), step=1)
    start = (page_index - 1) * page_size
    page_df = filtered.iloc[start:start + page_size]

    st.dataframe(page_df, use_container_width=True, hide_index=True)
    selected_id = st.selectbox("Inspect request", page_df["request_id"].tolist())

    if selected_id:
        try:
            explain = fetch_json(f"/admin/routing/explain/{selected_id}")
            st.json(explain)
        except requests.HTTPError as exc:
            # Friendly message when placeholder/sample IDs have no explainability
            status_code = None
            try:
                status_code = exc.response.status_code
            except Exception:
                status_code = None
            if status_code == 404:
                st.info("Run a Playground request to generate explainability data for this request.")
            else:
                render_error(f"Couldn't load explainability: {exc}")
        except requests.RequestException as exc:
            render_error(f"Couldn't load explainability: {exc}")

elif page == "Classifier Lifecycle":
    render_header("Classifier Lifecycle", "Production, candidates, shadow evaluation, and A/B testing")
    try:
        compare = fetch_json("/admin/classifiers/compare")
        classifiers = fetch_json("/v1/classifiers")
    except requests.RequestException as exc:
        render_error(f"Couldn't load classifier lifecycle payload: {exc}")
        st.stop()

    section_title("Current production")
    lifecycle_df = pd.DataFrame(compare.get("lifecycle", []))
    st.dataframe(lifecycle_df, use_container_width=True, hide_index=True)

    st.write("")
    c1, c2 = st.columns(2)
    with c1:
        section_title("Shadow evaluation")
        st.metric("Shadow evaluations", compare.get("shadow_evaluations", 0))
        st.metric("A/B decisions", compare.get("ab_decisions", 0))
    with c2:
        section_title("Confusion matrix")
        z = [[12, 2], [1, 85]]
        fig = go.Figure(data=go.Heatmap(z=z, x=["Predicted low", "Predicted high"], y=["Actual low", "Actual high"], colorscale="Mint"))
        fig.update_layout(height=220, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=8, r=8, t=8, b=8))
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    st.write("")
    section_title("Classifier versions")
    st.dataframe(pd.DataFrame(classifiers), use_container_width=True, hide_index=True)

elif page == "Provider Operations":
    render_header("Provider Operations", "Live health, latency, availability, retries, and circuit state")
    try:
        operations = fetch_json("/admin/providers/operations")
    except requests.RequestException as exc:
        render_error(f"Couldn't load provider operations payload: {exc}")
        st.stop()

    provider_df = pd.DataFrame(operations.get("providers", []))
    st.dataframe(provider_df, use_container_width=True, hide_index=True)

    st.write("")
    fig = px.bar(provider_df, x="provider", y="p95_latency_ms", color="provider", color_discrete_sequence=["#6C8CFF", "#4ADE9C", "#F2B441"])
    fig.update_layout(height=260, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=8, r=8, t=8, b=8))
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

elif page == "Operational Events":
    render_header("Operational Events", "Timeline, severity, filtering, and expandable detail")
    try:
        overview = fetch_json("/admin/overview")
        metrics = fetch_json("/admin/metrics")
    except requests.RequestException as exc:
        render_error(f"Couldn't load operational events: {exc}")
        st.stop()

    c1, c2, c3 = st.columns(3)
    with c1:
        metric_card("Events", str(metrics.get("events", 0)), "Structured operational records", tone="neutral")
    with c2:
        metric_card("Promotions", str(metrics.get("promotions", 0)), "Classifier promotions and rollbacks", tone="positive")
    with c3:
        metric_card("Circuit events", str(metrics.get("circuit_events", 0)), "Breaker transitions", tone="warning")

    st.write("")
    for event in overview.get("recent_events", []):
        with st.expander(f"{event.get('title', 'Event')} · {event.get('severity', 'info')}"):
            st.write(event.get("detail", ""))
            st.caption(event.get("timestamp", ""))

elif page == "Settings":
    render_header("Settings", "Providers, routing thresholds, feature flags, and environment")
    health = fetch_json("/health")
    overview = fetch_json("/admin/overview")
    st.markdown(
        """
        <div class="panel">
            <div class="metric-title">Configuration preview</div>
            <div class="metric-hint">This view consumes backend configuration and exposes a read-only control surface for the Mission Control experience.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.write("")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.selectbox("Environment", [health.get("environment", "local")], disabled=True)
        st.selectbox("Providers", ["ollama", "openai", "anthropic"], disabled=True)
    with c2:
        st.text_input("Routing threshold", value=str(overview["routing_policy"].get("confidence_threshold", 0.6)), disabled=True)
        st.text_input("Cost limit", value="$5,000 / day", disabled=True)
    with c3:
        st.checkbox("Feature flags enabled", value=True, disabled=True)
        st.checkbox("Health alerts enabled", value=True, disabled=True)
