
import os

os.environ["OMP_NUM_THREADS"] = "1"

import asyncio
import streamlit as st
from pathlib import Path
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

from config import model_config, rag_config, debate_config, validate_config
from models.factory import create_model
from rag.document_loader import DocumentLoader
from rag.text_splitter import TextSplitter
from rag.vector_store import VectorStore
from rag.retriever import Retriever
from agents.triage_agent import TriageAgent
from agents.expert_agent import ExpertAgent
from agents.skeptic_agent import SkepticAgent
from agents.synthesizer_agent import SynthesizerAgent
from agents.retrieval_agent import RetrievalAgent
from debate.protocol import CouncilProtocol

# Page config
st.set_page_config(
    page_title="Council Mode -- Multi-Agent Debate",
    page_icon="C",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    .stApp {
        font-family: 'Inter', sans-serif;
    }
    
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem;
        font-weight: 700;
    }
    
    .sub-header {
        text-align: center;
        color: #888;
        font-size: 1.1rem;
        margin-top: -1rem;
        margin-bottom: 2rem;
    }
    
    .expert-card {
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        border-radius: 12px;
        padding: 1.5rem;
        margin: 0.5rem 0;
        border-left: 4px solid;
    }
    
    .expert-1 { border-left-color: #00d2ff; }
    .expert-2 { border-left-color: #ff6b6b; }
    .expert-3 { border-left-color: #feca57; }
    
    .skeptic-card {
        background: linear-gradient(135deg, #2d1b69, #1a1a2e);
        border-radius: 12px;
        padding: 1.5rem;
        margin: 0.5rem 0;
        border-left: 4px solid #ff4757;
    }
    
    .consensus-card {
        background: linear-gradient(135deg, #0a3d0c, #1a1a2e);
        border-radius: 12px;
        padding: 1.5rem;
        margin: 0.5rem 0;
        border-left: 4px solid #2ed573;
    }
    
    .confidence-high { color: #2ed573; }
    .confidence-medium { color: #feca57; }
    .confidence-low { color: #ff4757; }
    
    .metric-box {
        background: rgba(255,255,255,0.05);
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def init_rag():
    """Initialize RAG pipeline (cached)."""
    loader = DocumentLoader()
    splitter = TextSplitter(chunk_size=rag_config.chunk_size, chunk_overlap=rag_config.chunk_overlap)
    vector_store = VectorStore(
        embedding_model=rag_config.embedding_model,
        persist_dir=str(rag_config.vector_db_dir),
    )
    
    docs_dir = rag_config.documents_dir
    if docs_dir.exists() and any(docs_dir.iterdir()) and vector_store.total_vectors == 0:
        documents = loader.load_directory(str(docs_dir))
        chunks = splitter.split_documents(documents)
        vector_store.add_chunks(chunks)
    
    return Retriever(vector_store, top_k=rag_config.top_k)


@st.cache_resource
def init_protocol(_retriever):
    """Initialize the Council Protocol (cached)."""
    triage_model = create_model(model_config.triage_model)
    
    # Dynamically create experts based on config
    experts = []
    expert_models = [
        model_config.expert_model_1,
        model_config.expert_model_2,
        model_config.expert_model_3,
    ]
    for i in range(min(debate_config.num_experts, 3)):
        experts.append(ExpertAgent(i + 1, create_model(expert_models[i])))
    
    skeptic = SkepticAgent(create_model(model_config.skeptic_model))
    synthesizer = SynthesizerAgent(create_model(model_config.synthesizer_model))
    retrieval_agent = RetrievalAgent(_retriever)
    
    return CouncilProtocol(
        triage_agent=TriageAgent(triage_model),
        experts=experts,
        skeptic=skeptic,
        synthesizer=synthesizer,
        retrieval_agent=retrieval_agent,
        direct_answer_model=triage_model,
        max_rounds=debate_config.max_rounds,
        consensus_threshold=debate_config.consensus_threshold,
    )


def render_sidebar():
    """Render the sidebar with configuration and document management."""
    with st.sidebar:
        st.markdown("## Configuration")
        
        st.markdown("### Models")
        st.text(f"Triage: {model_config.triage_model}")
        
        expert_models = [model_config.expert_model_1, model_config.expert_model_2, model_config.expert_model_3]
        for i in range(debate_config.num_experts):
            st.text(f"Expert {i+1}: {expert_models[i]}")
            
        st.text(f"Skeptic: {model_config.skeptic_model}")
        st.text(f"Synthesizer: {model_config.synthesizer_model}")
        
        st.markdown("### RAG Settings")
        st.text(f"Embedding: {rag_config.embedding_model}")
        st.text(f"Chunk size: {rag_config.chunk_size}")
        st.text(f"Top-K: {rag_config.top_k}")
        
        st.markdown("### Debate Settings")
        st.text(f"Num Experts: {debate_config.num_experts}")
        st.text(f"Max rounds: {debate_config.max_rounds}")
        st.text(f"Consensus threshold: {debate_config.consensus_threshold}")
        
        st.markdown("---")
        
        # Document upload
        st.markdown("### Upload Documents")
        uploaded_files = st.file_uploader(
            "Add PDFs for RAG grounding",
            type=["pdf"],
            accept_multiple_files=True,
        )
        
        if uploaded_files:
            docs_dir = rag_config.documents_dir
            docs_dir.mkdir(parents=True, exist_ok=True)
            
            for f in uploaded_files:
                filepath = docs_dir / f.name
                with open(filepath, "wb") as out:
                    out.write(f.getbuffer())
                st.success(f"Saved: {f.name}")
            
            if st.button("Re-index Documents"):
                st.cache_resource.clear()
                st.rerun()


def render_result(result):
    """Render the Council Mode result."""
    
    # Triage badge
    if result.triage.is_high_stakes:
        st.markdown("### HIGH STAKES -- Full Council Debate Triggered")
    else:
        st.markdown("### LOW STAKES -- Direct Answer")
    
    # Original Intent (Baseline) Display
    if result.baseline_answer:
        with st.expander("🔍 VIEW ORIGINAL INTENT (Zero-Shot Baseline)", expanded=True):
            st.info("This is what a standard LLM would have answered without RAG evidence or Multi-Agent debate.")
            st.markdown(result.baseline_answer)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Classification", result.triage.classification)
    with col2:
        st.metric("Triage Confidence", f"{result.triage.confidence:.0%}")
    with col3:
        if result.was_debated:
            st.metric("Debate Rounds", result.total_rounds)
    
    st.markdown("---")
    
    if not result.was_debated:
        st.markdown("### Answer")
        st.markdown(result.direct_answer)
    else:
        # Show debate rounds in expanders
        for round_result in result.rounds:
            with st.expander(f"Round {round_result.round_num}", expanded=(round_result.round_num == 1)):
                # Expert responses
                expert_cols = st.columns(len(round_result.expert_responses))
                labels = ["[Blue]", "[Red]", "[Gold]"]
                
                for i, (col, resp) in enumerate(zip(expert_cols, round_result.expert_responses)):
                    with col:
                        st.markdown(f"#### Expert {resp.expert_id}")
                        st.markdown(f"*{resp.expert_name}* -- `{resp.model_name}`")
                        
                        # Show reasoning in an expander to avoid cluttering the view
                        if getattr(resp, 'reasoning', None):
                            with st.expander("Show Thinking Process"):
                                st.markdown(resp.reasoning)
                        
                        st.markdown(resp.answer)
                
                # Skeptic analysis
                if round_result.skeptic_analysis:
                    st.markdown("#### Skeptic's Critique")
                    analysis = round_result.skeptic_analysis
                    
                    if analysis.contradictions:
                        st.warning(f"**Contradictions:** {len(analysis.contradictions)}")
                        for c in analysis.contradictions:
                            st.markdown(f"- {c}")
                    
                    if analysis.unsupported_claims:
                        st.info(f"**Unsupported Claims:** {len(analysis.unsupported_claims)}")
                    
                    if analysis.needs_re_retrieval:
                        st.warning("Re-retrieval requested with refined queries")
        
        # Final synthesis
        if result.synthesis:
            st.markdown("---")
            st.markdown("## Final Consensus")
            
            # Confidence display
            conf = result.synthesis.confidence_score
            if conf >= 80:
                conf_label = "HIGH"
            elif conf >= 60:
                conf_label = "MEDIUM"
            else:
                conf_label = "LOW"
            st.markdown(f"### Confidence: {conf}% ({conf_label})")
            st.progress(conf / 100)
            
            st.markdown(result.synthesis.final_answer)
            
            # Sources
            if result.synthesis.source_attribution:
                with st.expander("Source Attribution"):
                    for src in result.synthesis.source_attribution:
                        st.markdown(f"- {src}")

def render_benchmark_viz(report):
    """Render Plotly visualizations for benchmark results."""
    st.markdown("#### Comparative Analysis")
    
    # 1. Performance Comparison (F1 Score)
    b_f1 = report.baseline_metrics.get("avg_f1_score", 0)
    c_f1 = report.council_metrics.get("avg_f1_score", 0)
    
    fig_perf = go.Figure(data=[
        go.Bar(name='Single Agent (Baseline)', x=['F1 Score'], y=[b_f1], marker_color='#636EFA'),
        go.Bar(name='Council Mode', x=['F1 Score'], y=[c_f1], marker_color='#00CC96')
    ])
    fig_perf.update_layout(
        title=f"Performance Comparison: {report.dataset_name}",
        barmode='group',
        yaxis_title="Score",
        height=400,
        template="plotly_dark"
    )
    
    # 2. Latency Comparison
    b_time = report.total_baseline_time_s
    c_time = report.total_council_time_s
    
    fig_time = go.Figure(data=[
        go.Bar(name='Baseline', x=['Total Time'], y=[b_time], marker_color='#636EFA'),
        go.Bar(name='Council', x=['Total Time'], y=[c_time], marker_color='#EF553B')
    ])
    fig_time.update_layout(
        title="Efficiency (Total Seconds)",
        barmode='group',
        yaxis_title="Seconds (Lower is Better)",
        height=400,
        template="plotly_dark"
    )
    
    vcol1, vcol2 = st.columns(2)
    with vcol1:
        st.plotly_chart(fig_perf, width="stretch")
    with vcol2:
        st.plotly_chart(fig_time, width="stretch")
    
    # 3. Improvement Gauge
    improvement = report.f1_improvement
    fig_gauge = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = improvement,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "F1 Performance Improvement %"},
        delta = {'reference': 0, 'increasing': {'color': "green"}},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': "#00CC96"},
            'steps': [
                {'range': [0, 20], 'color': "rgba(255, 0, 0, 0.1)"},
                {'range': [20, 50], 'color': "rgba(255, 255, 0, 0.1)"},
                {'range': [50, 100], 'color': "rgba(0, 255, 0, 0.1)"}
            ],
        }
    ))
    fig_gauge.update_layout(height=300, template="plotly_dark")
    st.plotly_chart(fig_gauge, width="stretch")


def render_benchmark_tab():
    """Render the benchmark evaluation tab."""
    st.markdown("### Benchmark Evaluation")
    st.markdown("Run Council Mode vs Single-Agent Baseline on standard datasets.")
    
    dataset_options = {
        "halueval": "HaluEval 2.0 -- Hallucination Testing",
        "hotpotqa": "HotpotQA -- Multi-hop RAG",
        "squad": "SQuAD 2.0 -- Unanswerable Questions",
        "legal": "Legal RAG Bench -- Complex Reasoning",
        "all": "All Datasets",
    }
    
    col1, col2 = st.columns([3, 1])
    with col1:
        selected = st.selectbox(
            "Select Dataset",
            options=list(dataset_options.keys()),
            format_func=lambda x: dataset_options[x],
        )
    with col2:
        sample_limit = st.number_input("Samples", min_value=1, max_value=500, value=10, step=5)
    
    if st.button("Run Benchmark", type="primary", width="stretch"):
        from evaluation.benchmark_runner import BenchmarkRunner
        from evaluation.baseline import SingleAgentBaseline
        
        try:
            retriever = init_rag()
            protocol = init_protocol(retriever)
            baseline_model = create_model(model_config.triage_model)
            baseline = SingleAgentBaseline(baseline_model)
            runner = BenchmarkRunner(protocol=protocol, baseline=baseline)
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            def progress_cb(current, total, msg):
                progress_bar.progress(current / total)
                status_text.text(f"[{current}/{total}] {msg}")
            
            if selected == "all":
                reports = asyncio.run(runner.run_all_datasets(
                    sample_limit=sample_limit,
                    progress_callback=progress_cb,
                ))
            else:
                report = asyncio.run(runner.run_dataset(
                    selected, sample_limit=sample_limit,
                    progress_callback=progress_cb,
                ))
                reports = [report]
            
            progress_bar.progress(1.0)
            status_text.text("Benchmark complete!")
            
            for report in reports:
                st.markdown(f"---\n#### {report.dataset_name}")
                
                mc1, mc2, mc3, mc4 = st.columns(4)
                with mc1:
                    st.metric("Samples", report.total_samples)
                with mc2:
                    b_f1 = report.baseline_metrics.get("avg_f1_score", 0)
                    st.metric("Baseline F1", f"{b_f1:.4f}")
                with mc3:
                    c_f1 = report.council_metrics.get("avg_f1_score", 0)
                    st.metric("Council F1", f"{c_f1:.4f}")
                with mc4:
                    st.metric("F1 Improvement", f"{report.f1_improvement:+.2f}%")
                
                mc5, mc6, mc7 = st.columns(3)
                with mc5:
                    st.metric("RRH", f"{report.rrh:+.2f}%")
                with mc6:
                    st.metric("Baseline Time", f"{report.total_baseline_time_s:.1f}s")
                with mc7:
                    st.metric("Council Time", f"{report.total_council_time_s:.1f}s")
                
                # Render Charts
                render_benchmark_viz(report)
                
                # Save report
                filepath = runner.save_report(report)
                st.success(f"Report saved: {filepath}")
                
                # Show individual results
                with st.expander(f"Sample Results ({report.dataset_name})"):
                    for sr in report.sample_results:
                        st.markdown(f"**Q:** {sr['question'][:150]}")
                        st.markdown(f"**Ground Truth:** {sr['ground_truth'][:150]}")
                        st.markdown(f"**Baseline:** {sr['baseline_answer'][:200]}")
                        st.markdown(f"**Council:** {sr['council_answer'][:200]}")
                        st.markdown("---")
        
        except Exception as e:
            st.error(f"Benchmark error: {e}")
            st.exception(e)


def main():
    """Main Streamlit app."""
    st.markdown('<h1 class="main-header">Council Mode</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Multi-Agent Debate System -- Reasoning through debate, not token prediction</p>', unsafe_allow_html=True)
    
    render_sidebar()
    
    # Validate config
    warnings = validate_config()
    for w in warnings:
        st.warning(w)
    
    # Tabs
    tab_query, tab_data, tab_benchmark = st.tabs(["Ask the Council", "Knowledge Base", "Benchmark Evaluation"])
    
    with tab_query:
        # Initialize
        try:
            retriever = init_rag()
            protocol = init_protocol(retriever)
        except Exception as e:
            st.error(f"Initialization error: {e}")
            st.info("Make sure Ollama is running and/or GEMINI_API_KEY is set in .env")
            return
        
        # Query input
        st.markdown("### Ask the Council")
        
        col1, col2 = st.columns([4, 1])
        with col1:
            query = st.text_area(
                "Enter your query",
                placeholder="Ask a factual question that requires verified accuracy...",
                height=100,
                label_visibility="collapsed",
            )
        with col2:
            force_debate = st.checkbox("Force Debate", help="Skip triage and force full council debate")
            submit = st.button("Convene Council", type="primary", width="stretch")
        
        if submit and query:
            with st.spinner("The Council is deliberating..."):
                try:
                    result = asyncio.run(protocol.process_query(query, force_debate=force_debate))
                    render_result(result)
                except Exception as e:
                    st.error(f"Error during debate: {e}")
                    st.exception(e)
    
    with tab_data:
        st.markdown("### Knowledge Base Insights")
        retriever = init_rag()
        vs = retriever.vector_store
        
        if vs.total_vectors > 0:
            dcol1, dcol2 = st.columns(2)
            with dcol1:
                st.metric("Total Documents", len(list(rag_config.documents_dir.iterdir())) if rag_config.documents_dir.exists() else 0)
                st.metric("Total Chunks/Vectors", vs.total_vectors)
            
            with dcol2:
                # Chunk Size Distribution Visualization
                chunks = vs.all_chunks if hasattr(vs, 'all_chunks') else []
                if chunks:
                    lengths = [len(c.page_content) for c in chunks]
                    fig_dist = px.histogram(lengths, nbins=20, title="Chunk Size Distribution", labels={'value': 'Characters'})
                    fig_dist.update_layout(template="plotly_dark", showlegend=False)
                    st.plotly_chart(fig_dist, width="stretch")
            
            st.markdown("#### Topic Map (Top Keywords)")
            # Simple keyword extraction for visualization
            all_text = " ".join([c.page_content[:100] for c in (vs.all_chunks[:100] if hasattr(vs, 'all_chunks') else [])])
            if all_text:
                from wordcloud import WordCloud
                import matplotlib.pyplot as plt
                
                wc = WordCloud(background_color="#1a1a2e", colormap="viridis", width=800, height=400).generate(all_text)
                fig_wc, ax = plt.subplots()
                ax.imshow(wc, interpolation='bilinear')
                ax.axis("off")
                fig_wc.patch.set_facecolor('#1a1a2e')
                st.pyplot(fig_wc)
        else:
            st.info("Upload documents in the sidebar to see data visualizations.")
            
    with tab_benchmark:
        render_benchmark_tab()
    
    # Chat history
    if "history" not in st.session_state:
        st.session_state.history = []


if __name__ == "__main__":
    main()
