"""
app.py – RAG Pipeline Data Observability Dashboard
Run: streamlit run app.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from datetime import datetime
import pandas as pd
import streamlit as st

# Add src/ to path so we can import project modules for the chatbot
SRC_DIR = Path(__file__).resolve().parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# ── Page Configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RAG Pipeline Observability Dashboard",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS for Premium Look ───────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Outfit', sans-serif; }

.section-header {
    font-size: 20px;
    font-weight: 600;
    color: #6366f1;
    margin-top: 10px;
    margin-bottom: 15px;
    border-left: 4px solid #6366f1;
    padding-left: 10px;
}

.qa-card {
    background-color: #111827;
    border: 1px solid #1f2937;
    border-radius: 8px;
    padding: 15px;
    margin-bottom: 12px;
}

.metrics-box {
    background-color: #1e1b4b;
    border-left: 4px solid #818cf8;
    padding: 12px;
    border-radius: 0 8px 8px 0;
    margin: 8px 0;
}
</style>
""", unsafe_allow_html=True)

# ── Data Loading & Caching ─────────────────────────────────────────────────────
DATA = Path("data")

@st.cache_data
def load_json(path: str):
    p = DATA / path
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None

@st.cache_data
def load_csv(path: str):
    p = DATA / path
    if not p.exists():
        return None
    try:
        return pd.read_csv(p)
    except Exception:
        return None

# Load data
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

# Map answers for easy Q&A side-by-side comparison
baseline_dict = {a["question"]: a for a in baseline_answers if "question" in a}
corrupted_dict = {a["question"]: a for a in corrupted_answers if "question" in a}
repaired_dict = {a["question"]: a for a in repaired_answers if "question" in a}

# ── Sidebar Navigation ────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("<h2 style='text-align: center; color: #818cf8;'>🔬 RAG Pipeline</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; font-size: 14px; color: #9ca3af;'>Data Observability Dashboard</p>", unsafe_allow_html=True)
    st.markdown("---")
    page = st.radio(
        "Menu",
        ["📊 Overview", "📥 Ingestion & Clean", "⚡ Corruption & Repair", "📋 Papers Explorer", "💬 Agent QA", "🗨️ Chatbot", "📄 Reports"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.markdown("**Core Tech Stack:**")
    st.caption("• Embeddings: MiniLM-L6-v2")
    st.caption("• Vector Database: ChromaDB")
    st.caption("• LLM Judge: Gemini / DeepSeek")
    st.caption("• Framework: LangChain")
    st.markdown("---")
    st.caption(f"Last sync: {datetime.now().strftime('%Y-%m-%d %H:%M')}")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1: OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if page == "📊 Overview":
    st.markdown("# 📊 Pipeline Overview")
    st.markdown("Giám sát và kiểm định chất lượng dữ liệu đầu-cuối của hệ thống **Retrieval-Augmented Generation (RAG)** dựa trên tài liệu Crossref.")
    st.markdown("---")

    # Pipeline Architecture flowchart using HTML/CSS
    st.markdown('<div class="section-header">Luồng Xử Lý Hệ Thống (Pipeline Architecture)</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="display: flex; flex-direction: row; flex-wrap: wrap; gap: 8px; justify-content: space-between; align-items: center; margin: 15px 0;">
      <div style="flex: 1; min-width: 140px; padding: 12px; background: #131b2e; border: 1px solid #6366f1; border-radius: 8px; text-align: center;">
         <div style="font-size: 22px;">📡</div>
         <div style="font-weight: 600; color: #fff; margin-top: 4px; font-size: 14px;">1. Ingestion</div>
         <div style="font-size: 11px; color: #9ca3af;">Crossref API</div>
      </div>
      <div style="color: #6366f1; font-weight: bold; font-size: 18px; padding: 0 4px;">➔</div>
      <div style="flex: 1; min-width: 140px; padding: 12px; background: #131b2e; border: 1px solid #3b82f6; border-radius: 8px; text-align: center;">
         <div style="font-size: 22px;">🧹</div>
         <div style="font-weight: 600; color: #fff; margin-top: 4px; font-size: 14px;">2. Clean</div>
         <div style="font-size: 11px; color: #9ca3af;">Chuẩn hóa & Lọc</div>
      </div>
      <div style="color: #3b82f6; font-weight: bold; font-size: 18px; padding: 0 4px;">➔</div>
      <div style="flex: 1; min-width: 140px; padding: 12px; background: #131b2e; border: 1px solid #10b981; border-radius: 8px; text-align: center;">
         <div style="font-size: 22px;">🔢</div>
         <div style="font-weight: 600; color: #fff; margin-top: 4px; font-size: 14px;">3. Embed & DB</div>
         <div style="font-size: 11px; color: #9ca3af;">MiniLM + ChromaDB</div>
      </div>
      <div style="color: #10b981; font-weight: bold; font-size: 18px; padding: 0 4px;">➔</div>
      <div style="flex: 1; min-width: 140px; padding: 12px; background: #131b2e; border: 1px solid #8b5cf6; border-radius: 8px; text-align: center;">
         <div style="font-size: 22px;">🤖</div>
         <div style="font-weight: 600; color: #fff; margin-top: 4px; font-size: 14px;">4. AI Agent</div>
         <div style="font-size: 11px; color: #9ca3af;">RAG Q&A</div>
      </div>
      <div style="color: #8b5cf6; font-weight: bold; font-size: 18px; padding: 0 4px;">➔</div>
      <div style="flex: 1; min-width: 140px; padding: 12px; background: #131b2e; border: 1px solid #f97316; border-radius: 8px; text-align: center;">
         <div style="font-size: 22px;">⚖️</div>
         <div style="font-weight: 600; color: #fff; margin-top: 4px; font-size: 14px;">5. Evaluation</div>
         <div style="font-size: 11px; color: #9ca3af;">LLM Judge</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # Baseline Performance Metrics
    st.markdown('<div class="section-header">Hiệu năng Baseline (Khi dữ liệu sạch)</div>', unsafe_allow_html=True)
    c1, c2, c3, c4, c5 = st.columns(5)
    
    with c1:
        with st.container(border=True):
            st.metric(label="📄 Papers Indexed", value=len(df_baseline) if df_baseline is not None else 0)
    with c2:
        with st.container(border=True):
            st.metric(label="🎯 Retrieval Hit Rate", value=f"{baseline_metrics.get('retrieval_hit_rate', 0):.2%}")
    with c3:
        with st.container(border=True):
            st.metric(label="📝 Token F1 Score", value=f"{baseline_metrics.get('mean_token_f1', 0):.2%}")
    with c4:
        with st.container(border=True):
            st.metric(label="⚖️ Judge Accuracy", value=f"{baseline_metrics.get('judge_accuracy', 0):.2%}")
    with c5:
        with st.container(border=True):
            st.metric(label="⭐ Avg Judge Score", value=f"{baseline_metrics.get('mean_judge_score', 0):.2f} / 5.0")

    st.markdown("---")

    # Quality & Freshness overview
    st.markdown('<div class="section-header">Trạng thái Chất lượng & Độ tươi mới dữ liệu</div>', unsafe_allow_html=True)
    qcols = st.columns(3)
    phases_info = [
        ("🟢 Baseline (Sạch)", baseline_quality, freshness_baseline),
        ("🔴 Corrupted (Lỗi)", corrupted_quality, freshness_corrupt),
        ("🔵 Repaired (Đã sửa)", repaired_quality, freshness_repair),
    ]
    for col, (label, qdata, fresh) in zip(qcols, phases_info):
        with col:
            with st.container(border=True):
                st.markdown(f"#### {label}")
                passed = qdata.get("passed", 0)
                total = passed + qdata.get("failed", 0)
                sr = qdata.get("success_rate", 0.0)
                is_fresh = fresh.get("is_fresh", False)
                
                st.markdown(f"**Kiểm định chất lượng:** `{passed}/{total}` checks passed")
                st.progress(sr)
                
                if is_fresh:
                    st.success("Freshness: FRESH ✅")
                else:
                    st.warning("Freshness: STALE ⚠️")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2: INGESTION & CLEAN
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📥 Ingestion & Clean":
    st.markdown("# 📥 Ingestion & Cleaning Details")
    st.markdown("Theo dõi thông tin cấu hình thu thập từ API Crossref và kết quả chuẩn hóa dữ liệu sạch.")
    st.markdown("---")

    col1, col2 = st.columns([1, 2])
    with col1:
        with st.container(border=True):
            st.markdown("### ⚙️ Source Configurations")
            st.write("**API Nguồn:** `Crossref REST API`")
            st.write("**Từ khóa Query:** `agentic retrieval augmented generation large language model`")
            st.write("**Bộ lọc API:** `has-abstract:true, from-pub-date` (180 ngày trước)")
            st.write("**Ngưỡng tươi mới (Freshness Threshold):** `180` ngày")

        if df_baseline is not None:
            with st.container(border=True):
                st.markdown("### 📊 Dataset Statistics")
                c_a, c_b = st.columns(2)
                c_a.metric("Clean Records", len(df_baseline))
                c_b.metric("Authors Count", df_baseline["authors_joined"].str.split(", ").explode().nunique() if "authors_joined" in df_baseline.columns else 0)
                
                c_c, c_d = st.columns(2)
                c_c.metric("Avg Age (Days)", f"{df_baseline['age_days'].mean():.0f}")
                c_d.metric("Avg Abstract Length", f"{df_baseline['summary_chars'].mean():.0f} ký tự")

    with col2:
        if df_baseline is not None:
            st.markdown("### 📅 Phân bố thời gian xuất bản (Timeline)")
            if "published" in df_baseline.columns:
                pub_counts = df_baseline["published"].str[:7].value_counts().sort_index()
                pub_df = pd.DataFrame({"Month": pub_counts.index, "Papers": pub_counts.values})
                st.bar_chart(pub_df.set_index("Month"), color="#6366f1")

            st.markdown("### 🕒 Phân bố ngày tuổi dữ liệu (Age Distribution)")
            if "age_days" in df_baseline.columns:
                age_df = df_baseline[["age_days"]].copy()
                st.bar_chart(age_df["age_days"].value_counts().sort_index(), color="#8b5cf6")

    if df_baseline is not None:
        st.markdown("---")
        st.markdown("### 📄 Danh sách tài liệu sạch (Cleaned Dataset)")
        show_cols = ["title", "published", "age_days", "authors_joined", "summary_chars"]
        show_cols = [c for c in show_cols if c in df_baseline.columns]
        st.dataframe(
            df_baseline[show_cols].rename(columns={
                "title": "Tiêu đề", "published": "Ngày xuất bản",
                "age_days": "Ngày tuổi", "authors_joined": "Tác giả",
                "summary_chars": "Độ dài tóm tắt"
            }),
            use_container_width=True,
            height=300,
        )

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3: CORRUPTION & REPAIR
# ══════════════════════════════════════════════════════════════════════════════
elif page == "⚡ Corruption & Repair":
    st.markdown("# ⚡ Corruption & Repair Analysis")
    st.markdown("Trang phân tích tác động lỗi dữ liệu lên hiệu suất RAG và hiệu quả của việc khôi phục.")
    st.markdown("---")

    tab_metrics, tab_quality, tab_corruption_log = st.tabs([
        "📊 So sánh Hiệu năng (Performance Impact)",
        "🩺 Kiểm định Chất lượng (Quality & Freshness)",
        "🔥 Nhật ký Lỗi & Sửa (Corruption & Repair Log)"
    ])

    # 1. Performance Impact Tab
    with tab_metrics:
        st.markdown("### Biến thiên Hiệu suất AI Agent qua 3 Giai đoạn")
        
        metric_keys = [
            ("retrieval_hit_rate", "Retrieval Hit Rate"),
            ("mean_token_f1", "Token F1 Score"),
            ("judge_accuracy", "Judge Accuracy"),
            ("mean_judge_score", "Avg Judge Score (Normalised)"),
        ]

        chart_data = []
        for key, label in metric_keys:
            # Scale Mean Judge Score (out of 5.0) to 0.0 - 1.0 for standard visual scaling
            b_val = baseline_metrics.get(key, 0)
            c_val = corrupted_metrics.get(key, 0)
            r_val = repaired_metrics.get(key, 0)
            
            if key == "mean_judge_score":
                b_val = b_val / 5.0
                c_val = c_val / 5.0
                r_val = r_val / 5.0

            chart_data.append({"Metric": label, "Phase": "Baseline (Sạch)", "Value": float(b_val)})
            chart_data.append({"Metric": label, "Phase": "Corrupted (Lỗi)", "Value": float(c_val)})
            chart_data.append({"Metric": label, "Phase": "Repaired (Đã sửa)", "Value": float(r_val)})

        if chart_data:
            chart_df = pd.DataFrame(chart_data)
            pivot = chart_df.pivot(index="Metric", columns="Phase", values="Value")
            # Enforce column sorting order
            pivot = pivot[["Baseline (Sạch)", "Corrupted (Lỗi)", "Repaired (Đã sửa)"]]
            st.bar_chart(pivot, height=350)

        st.markdown("### Bảng So sánh Chỉ số Chi tiết (Delta Comparison)")
        table_rows = []
        for key, label in metric_keys:
            b = baseline_metrics.get(key)
            c = corrupted_metrics.get(key)
            r = repaired_metrics.get(key)
            
            def fmt(v):
                return f"{float(v):.4f}" if v is not None else "N/A"
            
            delta_c = f"{float(c)-float(b):+.4f}" if b is not None and c is not None else "N/A"
            delta_r = f"{float(r)-float(c):+.4f}" if c is not None and r is not None else "N/A"
            
            table_rows.append({
                "Chỉ số": label,
                "Baseline (Sạch)": fmt(b) if key != "mean_judge_score" else f"{float(b):.2f}/5.0",
                "Corrupted (Lỗi)": fmt(c) if key != "mean_judge_score" else f"{float(c):.2f}/5.0",
                "Δ Lỗi vs Sạch": delta_c,
                "Repaired (Đã sửa)": fmt(r) if key != "mean_judge_score" else f"{float(r):.2f}/5.0",
                "Δ Sau Sửa vs Lỗi": delta_r,
            })
        st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)

        st.info("💡 **Nhận xét:** Khi dữ liệu bị corrupt, Retrieval Hit Rate giảm đáng kể dẫn đến việc sinh câu trả lời bị sai lệch (Judge Accuracy & Token F1 giảm mạnh). Khi thực hiện sửa lỗi bằng cách re-ingest (tái nạp dữ liệu sạch) từ tệp dữ liệu gốc ban đầu, toàn bộ các chỉ số của hệ thống được khôi phục hoàn hảo 100% về mức tương đương với Baseline.")

    # 2. Quality & Freshness Tab
    with tab_quality:
        st.markdown("### So sánh 6 Chiều Chất lượng Dữ liệu (Data Quality Dimensions)")
        q_rows = []
        
        # Gather all checks from baseline (source of truth for dimension names)
        baseline_checks = baseline_quality.get("checks", [])
        
        for chk in baseline_checks:
            chk_name = chk["name"]
            dimension_label = chk.get("dimension", chk_name.replace("_", " ").title())
            
            def find_chk(q_dict, _name=chk_name):
                for c in q_dict.get("checks", []):
                    if c["name"] == _name:
                        return "✅ PASS" if c["passed"] else "❌ FAIL"
                return "—"
            
            def find_chk_detail(q_dict, _name=chk_name):
                for c in q_dict.get("checks", []):
                    if c["name"] == _name:
                        return c.get("detail", "")
                return ""
            
            q_rows.append({
                "Dimension": dimension_label,
                "Baseline": find_chk(baseline_quality),
                "Corrupted": find_chk(corrupted_quality),
                "Repaired": find_chk(repaired_quality),
                "Chi tiết (pha Corrupted)": find_chk_detail(corrupted_quality),
            })
            
        st.dataframe(pd.DataFrame(q_rows), use_container_width=True, hide_index=True)

        st.markdown("---")
        st.markdown("### Báo cáo Độ tươi mới Dữ liệu (Freshness Comparison)")
        f_cols = st.columns(3)
        fresh_info = [
            ("Baseline (Sạch)", freshness_baseline),
            ("Corrupted (Lỗi)", freshness_corrupt),
            ("Repaired (Đã sửa)", freshness_repair),
        ]
        for col, (phase_name, f_data) in zip(f_cols, fresh_info):
            with col:
                with st.container(border=True):
                    is_fresh = f_data.get("is_fresh", False)
                    status_text = "FRESH ✅" if is_fresh else "STALE ⚠️"
                    st.markdown(f"#### {phase_name}")
                    st.markdown(f"Trạng thái: **{status_text}**")
                    st.write(f"- Ngày xuất bản mới nhất: `{f_data.get('latest_published', 'N/A')}`")
                    st.write(f"- Ngày xuất bản cũ nhất: `{f_data.get('oldest_published', 'N/A')}`")
                    st.write(f"- Số dòng bị cũ: {f_data.get('stale_rows', 0)} ({f_data.get('stale_pct', 0):.1f}%)")

    # 3. Corruption & Repair Log Tab
    with tab_corruption_log:
        col_c_left, col_c_right = st.columns(2)
        
        with col_c_left:
            with st.container(border=True):
                st.markdown("### 🔥 6 Loại Lỗi đã chèn (Corruption Applied)")
                if corruption_log:
                    corruptions = corruption_log.get("corruptions", [])
                    for c in corruptions:
                        ctype = c.get("type", "").replace("_", " ").title()
                        count = c.get("count", "?")
                        st.markdown(f"• **{ctype}**: Tác động `{count}` dòng dữ liệu.")
                else:
                    st.write("Không tìm thấy nhật ký lỗi.")
                    
        with col_c_right:
            with st.container(border=True):
                st.markdown("### 🛠️ Chiến lược sửa lỗi (Repair Strategy)")
                st.markdown("""
                **Phương pháp sửa lỗi (Repair Strategy):**
                - Sửa lỗi bằng cách chạy lại toàn bộ quy trình **Làm sạch & Chuẩn hóa (Re-ingestion & Cleaning)** từ tệp nguồn dữ liệu thô gốc (`crossref_records.json`).
                
                **Kết quả khôi phục:**
                - Phục hồi **100%** dữ liệu sạch ban đầu.
                - Khắc phục hoàn toàn tất cả 6 loại lỗi của pha Corruption: điền lại các tóm tắt bị trống/nhiễu, loại bỏ các bản ghi trùng lặp, khôi phục các bản ghi bị xóa, và hoàn tác việc cắt ngắn tiêu đề bài báo.
                """)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4: PAPERS EXPLORER
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📋 Papers Explorer":
    st.markdown("# 📋 Khám phá Tập dữ liệu (Papers Explorer)")
    st.markdown("Tìm kiếm và xem chi tiết dữ liệu văn bản được nạp vào vector database qua các pha.")
    st.markdown("---")

    dataset_choice = st.selectbox("Chọn tập dữ liệu để xem", ["Baseline (Dữ liệu sạch)", "Corrupted (Dữ liệu lỗi)", "Repaired (Dữ liệu đã sửa)"])
    df_map = {
        "Baseline (Dữ liệu sạch)": df_baseline, 
        "Corrupted (Dữ liệu lỗi)": df_corrupted_csv, 
        "Repaired (Dữ liệu đã sửa)": df_repaired_csv
    }
    df = df_map[dataset_choice]

    if df is not None:
        # Search box
        search = st.text_input("🔍 Nhập từ khóa tìm kiếm trong tiêu đề hoặc tóm tắt", "")
        if search:
            mask = (
                df["title"].str.contains(search, case=False, na=False) |
                df["summary"].str.contains(search, case=False, na=False)
            )
            df = df[mask]

        st.markdown(f"Tìm thấy **{len(df)}** tài liệu:")

        show_cols = ["title", "published", "age_days", "authors_joined", "summary_chars"]
        show_cols = [c for c in show_cols if c in df.columns]
        
        selected = st.dataframe(
            df[show_cols].rename(columns={
                "title": "Tiêu đề", "published": "Ngày xuất bản",
                "age_days": "Ngày tuổi", "authors_joined": "Tác giả",
                "summary_chars": "Độ dài tóm tắt"
            }),
            use_container_width=True,
            selection_mode="single-row",
            on_select="rerun",
            height=300,
        )

        # Detail view card
        if selected and selected.selection.rows:
            idx = selected.selection.rows[0]
            row = df.iloc[idx]
            st.markdown("---")
            with st.container(border=True):
                st.markdown(f"### 📖 {row.get('title', 'N/A')}")
                col_d1, col_d2 = st.columns([1, 2])
                with col_d1:
                    st.write(f"**Mã DOI:** `{row.get('paper_id', 'N/A')}`")
                    st.write(f"**Ngày xuất bản:** {row.get('published', 'N/A')}")
                    st.write(f"**Ngày tuổi:** {row.get('age_days', 'N/A')} ngày")
                    st.write(f"**Tác giả:** {row.get('authors_joined', 'N/A')}")
                    st.write(f"**Danh mục:** {row.get('categories_joined', 'N/A') or '—'}")
                    if row.get("abs_url"):
                        st.markdown(f"[🔗 Xem Link Gốc Paper]({row['abs_url']})")
                with col_d2:
                    st.markdown("**Nội dung Tóm tắt (Abstract):**")
                    st.markdown(f"> {row.get('summary', 'N/A')}")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5: AGENT QA
# ══════════════════════════════════════════════════════════════════════════════
elif page == "💬 Agent QA":
    st.markdown("# 💬 Agent Q&A Results & Analysis")
    st.markdown("So sánh chi tiết câu trả lời của AI Agent trước và sau khi chất lượng dữ liệu bị biến đổi.")
    st.markdown("---")

    tab_compare, tab_demo = st.tabs(["📊 So sánh Q&A 3 Pha (Side-by-Side Comparison)", "💬 Demo Q&A Baseline"])

    # 1. Side-by-Side Comparison Tab
    with tab_compare:
        if baseline_dict:
            question_list = list(baseline_dict.keys())
            selected_q = st.selectbox("Chọn một câu hỏi trong bộ kiểm thử (test set) để so sánh:", question_list)
            
            b_item = baseline_dict.get(selected_q, {})
            c_item = corrupted_dict.get(selected_q, {})
            r_item = repaired_dict.get(selected_q, {})
            
            st.markdown("#### 🎯 Đáp án chuẩn (Ground Truth)")
            st.info(b_item.get("ground_truth", "Không có Ground Truth"))
            
            st.markdown("---")
            col_b, col_c, col_r = st.columns(3)
            
            # Helper to draw result card
            def draw_answer_card(col, title, item, color_theme):
                with col:
                    st.markdown(f"<h4 style='color:{color_theme}'>{title}</h4>", unsafe_allow_html=True)
                    if not item:
                        st.warning("Không có dữ liệu cho pha này.")
                        return
                    
                    hit = item.get("retrieval_hit", False)
                    f1 = item.get("token_f1", 0.0)
                    judge_score = item.get("judge", {}).get("score", 0)
                    correct = item.get("judge", {}).get("correct", False)
                    
                    st.markdown(f"**AI trả lời:**  \n{item.get('answer', 'N/A')}")
                    
                    # Status Box
                    hit_icon = "✅ Tìm thấy" if hit else "❌ Không tìm thấy"
                    correct_icon = "✅ Đúng (Correct)" if correct else "❌ Sai (Incorrect)"
                    
                    st.markdown(f"""
                    <div class='metrics-box'>
                    <b>Tìm kiếm (Retrieval):</b> {hit_icon}<br>
                    <b>Điểm F1:</b> <code>{f1:.4f}</code><br>
                    <b>Chất lượng (LLM Judge):</b> {judge_score}/5.0 ({correct_icon})
                    </div>
                    """, unsafe_allow_html=True)
                    
                    reasoning = item.get("judge", {}).get("reasoning", "")
                    if reasoning:
                        st.caption(f"*Lý do chấm điểm: {reasoning}*")

            draw_answer_card(col_b, "🟢 Baseline (Dữ liệu Sạch)", b_item, "#4ade80")
            draw_answer_card(col_c, "🔴 Corrupted (Dữ liệu Lỗi)", c_item, "#f87171")
            draw_answer_card(col_r, "🔵 Repaired (Đã Sửa Lỗi)", r_item, "#60a5fa")
            
        else:
            st.warning("Không tìm thấy bộ dữ liệu Q&A. Hãy chạy python script/run_phase1.py trước.")

    # 2. Demo Q&A Tab
    with tab_demo:
        if demo_answers:
            st.markdown("### Câu hỏi mẫu của AI Agent khi tương tác thử nghiệm")
            for item in demo_answers:
                with st.expander(f"❓ {item.get('question', '')}"):
                    st.markdown(f"**Câu trả lời của Agent:**  \n{item.get('answer', 'N/A')}")
        else:
            st.info("Không tìm thấy dữ liệu demo. Chạy Phase 1 để tự động sinh câu trả lời demo.")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 6: REPORTS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📄 Reports":
    st.markdown("# 📄 Báo cáo Pipeline chi tiết (Markdown Reports)")
    st.markdown("Xem trực tiếp nội dung các file báo cáo markdown được sinh tự động bởi hệ thống quan sát dữ liệu.")
    st.markdown("---")

    tab_r1, tab_r2 = st.tabs(["📊 Báo cáo Phase 1 (Baseline Report)", "⚡ Báo cáo Pha 2 (Corruption Report)"])

    with tab_r1:
        report_path = Path("data/reports/phase1_report.md")
        if report_path.exists():
            st.markdown(report_path.read_text(encoding="utf-8"))
        else:
            st.warning("Không tìm thấy báo cáo Phase 1. Hãy chạy `python script/run_phase1.py` trước.")

    with tab_r2:
        report_path = Path("data/reports/corruption_report.md")
        if report_path.exists():
            st.markdown(report_path.read_text(encoding="utf-8"))
        else:
            st.warning("Không tìm thấy báo cáo so sánh Pha 2. Hãy chạy `python script/run_corruption_flow.py` trước.")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 7: CHATBOT
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🗨️ Chatbot":
    st.markdown("# 🗨️ RAG Chatbot")
    st.markdown("Trò chuyện trực tiếp với AI Agent về kho tài liệu khoa học đã được nạp vào hệ thống.")
    st.markdown("---")

    # ── Load index and LLM (cached) ──────────────────────────────────────────
    @st.cache_resource(show_spinner="Đang tải mô hình embedding và kho dữ liệu...")
    def load_chatbot_resources():
        """Load ChromaDB index and LLM for the chatbot."""
        try:
            from core.config import load_settings
            from retrieval.index import LocalEmbeddingIndex
            from retrieval.llm import build_llm

            settings = load_settings()
            embeddings_path = settings.paths.embeddings_json
            if not embeddings_path.exists():
                return None, None, "Chưa có dữ liệu embedding. Hãy chạy `python script/run_phase1.py` trước."
            index = LocalEmbeddingIndex.load(settings, embeddings_path)
            llm = build_llm(settings=settings, temperature=0.3)
            return index, llm, None
        except Exception as e:
            return None, None, f"Lỗi khi tải chatbot: {e}"

    index, llm, load_error = load_chatbot_resources()

    if load_error:
        st.error(load_error)
    else:
        # ── Sample questions ─────────────────────────────────────────────────
        st.markdown('<div class="section-header">💡 Câu hỏi mẫu (chọn để hỏi nhanh)</div>', unsafe_allow_html=True)

        sample_questions = [
            "What is the paper about Hallucination in Large Language Models?",
            "Who authored the paper about Retrieval-Augmented Generation?",
            "What are the latest trends in RAG systems?",
            "List the papers related to LLM evaluation",
            "Summarize the research on agentic AI systems",
        ]

        # Render sample question buttons in a row
        sample_cols = st.columns(len(sample_questions))
        selected_sample = None
        for i, (col, sq) in enumerate(zip(sample_cols, sample_questions)):
            with col:
                short_label = sq[:35] + "..." if len(sq) > 35 else sq
                if st.button(f"💬 {short_label}", key=f"sample_{i}", use_container_width=True):
                    selected_sample = sq

        st.markdown("---")

        # ── Chat history ─────────────────────────────────────────────────────
        if "chat_messages" not in st.session_state:
            st.session_state.chat_messages = []

        # Display chat history
        for msg in st.session_state.chat_messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        # ── Handle input ─────────────────────────────────────────────────────
        user_input = st.chat_input("Nhập câu hỏi về kho tài liệu...")

        # Use sample question if clicked
        if selected_sample:
            user_input = selected_sample

        if user_input:
            # Append user message
            st.session_state.chat_messages.append({"role": "user", "content": user_input})
            with st.chat_message("user"):
                st.markdown(user_input)

            # Generate response
            with st.chat_message("assistant"):
                with st.spinner("Đang tìm kiếm và sinh câu trả lời..."):
                    try:
                        # Step 1: Semantic search for relevant papers
                        search_results = index.search(user_input, top_k=4)

                        if not search_results:
                            answer_text = "Không tìm thấy tài liệu liên quan trong kho dữ liệu."
                            st.markdown(answer_text)
                        else:
                            # Step 2: Build context from retrieved papers
                            context_parts = []
                            for j, result in enumerate(search_results, 1):
                                context_parts.append(
                                    f"[Paper {j}] {result.title}\n"
                                    f"Score: {result.score:.4f}\n"
                                    f"{result.content}"
                                )
                            context_text = "\n\n".join(context_parts)

                            # Step 3: Ask LLM with context
                            prompt = (
                                f"You are a helpful research assistant. Answer the user's question based on the following papers from the indexed corpus.\n\n"
                                f"--- Retrieved Papers ---\n{context_text}\n--- End ---\n\n"
                                f"User question: {user_input}\n\n"
                                f"Instructions: Answer concisely and accurately based on the papers above. "
                                f"If the papers don't contain enough info, say so. Cite paper titles when possible."
                            )
                            llm_response = llm.invoke(prompt)
                            answer_text = llm_response.content if hasattr(llm_response, 'content') else str(llm_response)

                            st.markdown(answer_text)

                            # Show retrieved sources in an expander
                            with st.expander("📚 Nguồn tài liệu đã tìm thấy (Retrieved Sources)", expanded=False):
                                for j, result in enumerate(search_results, 1):
                                    st.markdown(
                                        f"**{j}. {result.title}**  \n"
                                        f"DOI: `{result.paper_id}` · Relevance: `{result.score:.4f}`  \n"
                                        f"*{result.metadata.get('summary', '')[:200]}...*"
                                    )
                                    st.markdown("---")

                    except Exception as e:
                        answer_text = f"⚠️ Lỗi khi sinh câu trả lời: {e}"
                        st.error(answer_text)

            # Append assistant message
            st.session_state.chat_messages.append({"role": "assistant", "content": answer_text})

        # ── Clear chat button ────────────────────────────────────────────────
        if st.session_state.chat_messages:
            if st.button("🗑️ Xóa lịch sử hội thoại", use_container_width=True):
                st.session_state.chat_messages = []
                st.rerun()
