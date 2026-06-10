"""
app.py – RAG Pipeline Data Observability Dashboard
Run: streamlit run app.py
"""
from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime

import pandas as pd
import streamlit as st

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RAG Pipeline Dashboard",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.metric-card {
    background: linear-gradient(135deg, #1e1e2e 0%, #2a2a3e 100%);
    border: 1px solid #3a3a5c;
    border-radius: 12px;
    padding: 20px 24px;
    margin: 4px 0;
}
.metric-label { color: #a0a0c0; font-size: 12px; font-weight: 500; text-transform: uppercase; letter-spacing: 1px; }
.metric-value { color: #e0e0ff; font-size: 28px; font-weight: 700; margin: 4px 0; }
.metric-delta-pos { color: #4ade80; font-size: 13px; }
.metric-delta-neg { color: #f87171; font-size: 13px; }

.section-header {
    background: linear-gradient(90deg, #6366f1, #8b5cf6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 22px;
    font-weight: 700;
    margin: 24px 0 12px;
}

.badge-pass { background: #166534; color: #4ade80; padding: 2px 10px; border-radius: 999px; font-size: 12px; font-weight: 600; }
.badge-fail { background: #7f1d1d; color: #f87171; padding: 2px 10px; border-radius: 999px; font-size: 12px; font-weight: 600; }
.badge-fresh { background: #164e63; color: #38bdf8; padding: 2px 10px; border-radius: 999px; font-size: 12px; }
.badge-stale { background: #78350f; color: #fbbf24; padding: 2px 10px; border-radius: 999px; font-size: 12px; }

.pipeline-step {
    background: #1e1e2e;
    border-left: 3px solid #6366f1;
    padding: 10px 16px;
    margin: 6px 0;
    border-radius: 0 8px 8px 0;
    font-size: 14px;
    color: #c0c0e0;
}

[data-testid="stSidebar"] { background: #13131f; }
[data-testid="stSidebar"] * { color: #c0c0e0 !important; }
</style>
""", unsafe_allow_html=True)

# ── Data loading ───────────────────────────────────────────────────────────────
DATA = Path("data")

@st.cache_data
def load_json(path: str):
    p = DATA / path
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))

@st.cache_data
def load_csv(path: str):
    p = DATA / path
    if not p.exists():
        return None
    return pd.read_csv(p)

baseline_metrics   = load_json("results/baseline_metrics.json") or {}
corrupted_metrics  = load_json("results/corrupted_metrics.json") or {}
repaired_metrics   = load_json("results/repaired_metrics.json") or {}
baseline_answers   = load_json("results/baseline_answers.json") or []
corrupted_answers  = load_json("results/corrupted_answers.json") or []
repaired_answers   = load_json("results/repaired_answers.json") or []
baseline_quality   = load_json("quality/baseline.json") or {}
corrupted_quality  = load_json("quality/corrupted.json") or {}
repaired_quality   = load_json("quality/repaired.json") or {}
freshness_baseline = load_json("quality/freshness_report.json") or {}
freshness_corrupt  = load_json("quality/freshness_corrupted.json") or {}
freshness_repair   = load_json("quality/freshness_repaired.json") or {}
corruption_log     = load_json("results/corruption_log.json") or {}
demo_answers       = load_json("results/agent_demo_answers.json") or []
test_set           = load_json("eval/test_set.json") or []
df_baseline        = load_csv("clean/papers_clean.csv")
df_corrupted_csv   = load_csv("clean/papers_clean_corrupted.csv")
df_repaired_csv    = load_csv("clean/papers_clean_repaired.csv")

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔬 RAG Pipeline")
    st.markdown("**Data Observability Dashboard**")
    st.markdown("---")
    page = st.radio(
        "Navigate",
        ["📊 Overview", "📥 Data Ingestion", "⚡ Corruption Analysis", "📋 Papers Explorer", "💬 Agent QA", "📄 Reports"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.markdown("**Source:** Crossref REST API")
    st.markdown("**Model:** deepseek-v4-flash")
    st.markdown("**Embeddings:** MiniLM-L6-v2")
    st.markdown("**Vector DB:** ChromaDB")
    st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1: OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if page == "📊 Overview":
    st.markdown("# 🔬 RAG Pipeline Observability")
    st.markdown("End-to-end data quality monitoring for a Retrieval-Augmented Generation system built on **Crossref** scholarly papers.")
    st.markdown("---")

    # Pipeline flow diagram
    st.markdown('<div class="section-header">Pipeline Architecture</div>', unsafe_allow_html=True)
    cols = st.columns(6)
    steps = [
        ("📡", "Fetch", "Crossref API"),
        ("🧹", "Clean", "Normalize & filter"),
        ("🔢", "Embed", "MiniLM + ChromaDB"),
        ("🤖", "Agent", "RAG Q&A"),
        ("📊", "Evaluate", "Metrics & judge"),
        ("🩺", "Observe", "Quality & freshness"),
    ]
    for col, (icon, title, desc) in zip(cols, steps):
        with col:
            st.markdown(f"<div style='text-align:center;padding:12px;background:#1e1e2e;border-radius:10px;border:1px solid #3a3a5c'>"
                        f"<div style='font-size:28px'>{icon}</div>"
                        f"<div style='color:#e0e0ff;font-weight:600;font-size:14px;margin-top:4px'>{title}</div>"
                        f"<div style='color:#7070a0;font-size:11px'>{desc}</div></div>", unsafe_allow_html=True)

    st.markdown("---")

    # Key metrics
    st.markdown('<div class="section-header">Baseline Performance</div>', unsafe_allow_html=True)
    c1, c2, c3, c4, c5 = st.columns(5)
    metrics_map = [
        (c1, "Papers Indexed", len(df_baseline) if df_baseline is not None else 0, None, "📄"),
        (c2, "Retrieval Hit Rate", baseline_metrics.get("retrieval_hit_rate", 0), None, "🎯"),
        (c3, "Token F1", baseline_metrics.get("mean_token_f1", 0), None, "📝"),
        (c4, "Judge Accuracy", baseline_metrics.get("judge_accuracy", 0), None, "⚖️"),
        (c5, "Judge Score", baseline_metrics.get("mean_judge_score", 0), "/ 5.0", "⭐"),
    ]
    for col, label, value, suffix, icon in metrics_map:
        with col:
            fmt = f"{value:.4f}" if isinstance(value, float) else str(value)
            st.markdown(
                f"<div class='metric-card'>"
                f"<div class='metric-label'>{icon} {label}</div>"
                f"<div class='metric-value'>{fmt}{suffix or ''}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

    st.markdown("")

    # Quality overview
    st.markdown('<div class="section-header">Data Quality Overview</div>', unsafe_allow_html=True)
    qcols = st.columns(3)
    for col, (label, qdata, fresh) in zip(qcols, [
        ("Baseline", baseline_quality, freshness_baseline),
        ("Corrupted", corrupted_quality, freshness_corrupt),
        ("Repaired", repaired_quality, freshness_repair),
    ]):
        with col:
            p = qdata.get("passed", 0)
            total = p + qdata.get("failed", 0)
            sr = qdata.get("success_rate", 0)
            color = "#4ade80" if sr == 1.0 else "#fbbf24" if sr >= 0.6 else "#f87171"
            fresh_status = "FRESH" if fresh.get("is_fresh") else "STALE"
            fresh_color = "#38bdf8" if fresh.get("is_fresh") else "#fbbf24"
            st.markdown(
                f"<div class='metric-card'>"
                f"<div class='metric-label'>📊 {label}</div>"
                f"<div style='color:{color};font-size:22px;font-weight:700;margin:6px 0'>{p}/{total} checks</div>"
                f"<div style='color:{color};font-size:13px'>Quality: {sr:.0%}</div>"
                f"<div style='color:{fresh_color};font-size:13px;margin-top:4px'>Freshness: {fresh_status}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2: DATA INGESTION
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📥 Data Ingestion":
    st.markdown("# 📥 Data Ingestion")
    st.markdown("---")

    col1, col2 = st.columns([1, 2])
    with col1:
        st.markdown("### Source Config")
        st.info("**API:** Crossref REST API\n\n**Query:** agentic retrieval augmented generation large language model\n\n**Filter:** from-pub-date (180 days), has-abstract:true\n\n**Max results:** 24")

        if df_baseline is not None:
            st.markdown("### Dataset Stats")
            st.metric("Clean Records", len(df_baseline))
            st.metric("Unique Authors", df_baseline["authors_joined"].str.split(", ").explode().nunique() if "authors_joined" in df_baseline.columns else "N/A")
            if "age_days" in df_baseline.columns:
                st.metric("Avg Age (days)", f"{df_baseline['age_days'].mean():.0f}")
            if "summary_chars" in df_baseline.columns:
                st.metric("Avg Summary Length", f"{df_baseline['summary_chars'].mean():.0f} chars")

    with col2:
        if df_baseline is not None:
            st.markdown("### Publication Timeline")
            if "published" in df_baseline.columns:
                pub_counts = df_baseline["published"].str[:7].value_counts().sort_index()
                pub_df = pd.DataFrame({"Month": pub_counts.index, "Papers": pub_counts.values})
                st.bar_chart(pub_df.set_index("Month"))

            st.markdown("### Age Distribution (days)")
            if "age_days" in df_baseline.columns:
                age_df = df_baseline[["age_days"]].copy()
                st.bar_chart(age_df["age_days"].value_counts().sort_index())

    if df_baseline is not None:
        st.markdown("### 📄 Cleaned Papers")
        show_cols = ["title", "published", "age_days", "authors_joined", "summary_chars"]
        show_cols = [c for c in show_cols if c in df_baseline.columns]
        st.dataframe(
            df_baseline[show_cols].rename(columns={
                "title": "Title", "published": "Published",
                "age_days": "Age (days)", "authors_joined": "Authors",
                "summary_chars": "Summary Length"
            }),
            use_container_width=True,
            height=400,
        )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3: CORRUPTION ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "⚡ Corruption Analysis":
    st.markdown("# ⚡ Corruption & Repair Analysis")
    st.markdown("---")

    # Corruption log
    if corruption_log:
        st.markdown("### 🔥 Corruption Applied")
        clog_cols = st.columns(3)
        corruptions = corruption_log.get("corruptions", [])
        for i, c in enumerate(corruptions):
            with clog_cols[i % 3]:
                ctype = c.get("type", "").replace("_", " ").title()
                count = c.get("count", "?")
                st.markdown(
                    f"<div class='pipeline-step'><b>{ctype}</b><br>"
                    f"<span style='color:#6366f1'>Count: {count}</span></div>",
                    unsafe_allow_html=True,
                )
        st.markdown(f"**Original rows:** {corruption_log.get('original_rows', '?')} → **Corrupted rows:** {corruption_log.get('corrupted_rows', '?')}")
    
    st.markdown("---")

    # Metrics comparison
    st.markdown("### 📊 Metrics Comparison: Baseline vs Corrupted vs Repaired")

    metric_keys = [
        ("retrieval_hit_rate", "Retrieval Hit Rate"),
        ("mean_token_f1", "Mean Token F1"),
        ("judge_accuracy", "Judge Accuracy"),
        ("mean_judge_score", "Mean Judge Score"),
    ]

    # Bar chart data
    chart_data = []
    for key, label in metric_keys:
        for phase, mdata in [("Baseline", baseline_metrics), ("Corrupted", corrupted_metrics), ("Repaired", repaired_metrics)]:
            val = mdata.get(key)
            if val is not None:
                chart_data.append({"Metric": label, "Phase": phase, "Value": float(val)})

    if chart_data:
        chart_df = pd.DataFrame(chart_data)
        pivot = chart_df.pivot(index="Metric", columns="Phase", values="Value")
        st.bar_chart(pivot)

    # Table
    st.markdown("### 📋 Detailed Comparison Table")
    table_rows = []
    for key, label in metric_keys:
        b = baseline_metrics.get(key)
        c = corrupted_metrics.get(key)
        r = repaired_metrics.get(key)
        def f(v):
            return f"{float(v):.4f}" if v is not None else "N/A"
        delta_c = f"{float(c)-float(b):+.4f}" if b is not None and c is not None else "N/A"
        delta_r = f"{float(r)-float(c):+.4f}" if c is not None and r is not None else "N/A"
        table_rows.append({
            "Metric": label,
            "Baseline": f(b),
            "Corrupted": f(c),
            "Δ (C-B)": delta_c,
            "Repaired": f(r),
            "Δ (R-C)": delta_r,
        })
    st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)

    # Quality comparison
    st.markdown("---")
    st.markdown("### 🩺 Data Quality Checks")
    qcols = st.columns(3)
    for col, (phase, qdata) in zip(qcols, [
        ("Baseline", baseline_quality),
        ("Corrupted", corrupted_quality),
        ("Repaired", repaired_quality),
    ]):
        with col:
            st.markdown(f"**{phase}**")
            for check in qdata.get("checks", []):
                badge = "✅" if check["passed"] else "❌"
                name = check["name"].replace("_", " ").title()
                detail = check.get("detail", "")
                st.markdown(f"{badge} **{name}**  \n<small style='color:#808080'>{detail}</small>", unsafe_allow_html=True)

    # Freshness
    st.markdown("---")
    st.markdown("### 🕐 Freshness Report")
    fcols = st.columns(3)
    for col, (phase, fresh) in zip(fcols, [
        ("Baseline", freshness_baseline),
        ("Corrupted", freshness_corrupt),
        ("Repaired", freshness_repair),
    ]):
        with col:
            is_fresh = fresh.get("is_fresh", False)
            status_color = "#4ade80" if is_fresh else "#fbbf24"
            status = "FRESH ✅" if is_fresh else "STALE ⚠️"
            st.markdown(f"**{phase}**")
            st.markdown(f"<span style='color:{status_color};font-weight:600'>{status}</span>", unsafe_allow_html=True)
            st.markdown(f"- Latest: `{fresh.get('latest_published', 'N/A')}`")
            st.markdown(f"- Oldest: `{fresh.get('oldest_published', 'N/A')}`")
            st.markdown(f"- Stale rows: {fresh.get('stale_rows', 0)} ({fresh.get('stale_pct', 0):.1f}%)")
            st.markdown(f"- Total rows: {fresh.get('total_rows', 'N/A')}")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4: PAPERS EXPLORER
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📋 Papers Explorer":
    st.markdown("# 📋 Papers Explorer")
    st.markdown("---")

    dataset_choice = st.selectbox("Select dataset", ["Baseline", "Corrupted", "Repaired"])
    df_map = {"Baseline": df_baseline, "Corrupted": df_corrupted_csv, "Repaired": df_repaired_csv}
    df = df_map[dataset_choice]

    if df is not None:
        # Search
        search = st.text_input("🔍 Search by title or summary", "")
        if search:
            mask = (
                df["title"].str.contains(search, case=False, na=False) |
                df["summary"].str.contains(search, case=False, na=False)
            )
            df = df[mask]

        st.markdown(f"**{len(df)} papers**")

        show_cols = ["title", "published", "age_days", "authors_joined", "summary_chars"]
        show_cols = [c for c in show_cols if c in df.columns]
        selected = st.dataframe(
            df[show_cols].rename(columns={
                "title": "Title", "published": "Published",
                "age_days": "Age (days)", "authors_joined": "Authors",
                "summary_chars": "Summary Length"
            }),
            use_container_width=True,
            selection_mode="single-row",
            on_select="rerun",
            height=350,
        )

        # Detail view
        if selected and selected.selection.rows:
            idx = selected.selection.rows[0]
            row = df.iloc[idx]
            st.markdown("---")
            st.markdown(f"### 📖 {row.get('title', 'N/A')}")
            col1, col2 = st.columns([1, 2])
            with col1:
                st.markdown(f"**DOI:** `{row.get('paper_id', 'N/A')}`")
                st.markdown(f"**Published:** {row.get('published', 'N/A')}")
                st.markdown(f"**Age:** {row.get('age_days', 'N/A')} days")
                st.markdown(f"**Authors:** {row.get('authors_joined', 'N/A')}")
                st.markdown(f"**Categories:** {row.get('categories_joined', 'N/A') or '—'}")
                if row.get("abs_url"):
                    st.markdown(f"[🔗 Open paper]({row['abs_url']})")
            with col2:
                st.markdown("**Abstract:**")
                st.markdown(f"> {row.get('summary', 'N/A')[:600]}...")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5: AGENT QA
# ══════════════════════════════════════════════════════════════════════════════
elif page == "💬 Agent QA":
    st.markdown("# 💬 Agent Q&A Results")
    st.markdown("---")

    # Demo answers
    if demo_answers:
        st.markdown("### 🎯 Demo Agent Answers (Baseline)")
        for item in demo_answers:
            with st.expander(f"❓ {item.get('question', '')[:80]}..."):
                st.markdown(f"**Answer:** {item.get('answer', 'N/A')}")

    # Evaluation answers by phase
    st.markdown("---")
    st.markdown("### 📊 Evaluation Answers Comparison")

    phase_choice = st.selectbox("Select phase", ["Baseline", "Corrupted", "Repaired"])
    answers_map = {"Baseline": baseline_answers, "Corrupted": corrupted_answers, "Repaired": repaired_answers}
    answers = answers_map[phase_choice]

    if answers:
        filter_type = st.multiselect(
            "Filter by question type",
            options=list(set(a.get("question_type", "") for a in answers)),
            default=list(set(a.get("question_type", "") for a in answers)),
        )
        answers = [a for a in answers if a.get("question_type") in filter_type]

        for item in answers:
            hit = item.get("retrieval_hit", False)
            f1 = item.get("token_f1", 0)
            judge_score = item.get("judge", {}).get("score", "?")
            correct = item.get("judge", {}).get("correct", False)
            hit_icon = "✅" if hit else "❌"
            correct_icon = "✅" if correct else "❌"

            with st.expander(f"{hit_icon} [{item.get('question_type','?').upper()}] {item.get('question','')[:70]}..."):
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"**Ground Truth:** {item.get('ground_truth', 'N/A')[:200]}")
                    st.markdown(f"**Answer:** {item.get('answer', 'N/A')[:200]}")
                with col2:
                    st.markdown(f"**Retrieval Hit:** {hit_icon}")
                    st.markdown(f"**Token F1:** `{f1:.4f}`")
                    st.markdown(f"**Judge Score:** {judge_score}/5 {correct_icon}")
                    reasoning = item.get("judge", {}).get("reasoning", "")
                    if reasoning:
                        st.caption(f"*{reasoning[:150]}*")
    else:
        st.info("No evaluation answers found for this phase.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 6: REPORTS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📄 Reports":
    st.markdown("# 📄 Pipeline Reports")
    st.markdown("---")

    tab1, tab2 = st.tabs(["📊 Phase 1 – Baseline Report", "⚡ Corruption Report"])

    with tab1:
        report_path = Path("data/reports/phase1_report.md")
        if report_path.exists():
            st.markdown(report_path.read_text(encoding="utf-8"))
        else:
            st.warning("Phase 1 report not found. Run `python script/run_phase1.py` first.")

    with tab2:
        report_path = Path("data/reports/corruption_report.md")
        if report_path.exists():
            st.markdown(report_path.read_text(encoding="utf-8"))
        else:
            st.warning("Corruption report not found. Run `python script/run_corruption_flow.py` first.")
