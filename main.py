
import os

os.environ["OMP_NUM_THREADS"] = "1"

import asyncio
import sys
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.prompt import Prompt
from rich.table import Table

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
from utils.logger import get_logger

logger = get_logger("system")
console = Console()


def print_banner():
    """Print a simple text welcome message."""
    print("\n" + "="*60)
    print(" COUNCIL MODE -- Multi-Agent Debate System")
    print(" Status: Ready (Simple Mode)")
    print("="*60 + "\n")


def setup_rag_pipeline():
    """Initialize the RAG pipeline."""
    print("Setting up RAG Pipeline...")
    
    # Document loader
    loader = DocumentLoader()
    docs_dir = rag_config.documents_dir
    
    # Load documents if directory has files
    documents = []
    if docs_dir.exists() and any(docs_dir.iterdir()):
        documents = loader.load_directory(str(docs_dir))
        print(f"  Loaded {len(documents)} documents")
    else:
        print(f"  No documents in {docs_dir}. Add PDFs for grounding.")
    
    # Text splitter
    splitter = TextSplitter(
        chunk_size=rag_config.chunk_size,
        chunk_overlap=rag_config.chunk_overlap,
    )
    
    # Vector store
    vector_store = VectorStore(
        embedding_model=rag_config.embedding_model,
        persist_dir=str(rag_config.vector_db_dir),
    )
    
    # If we have new documents and no existing index, build it
    if documents and vector_store.total_vectors == 0:
        chunks = splitter.split_documents(documents)
        vector_store.add_chunks(chunks)
        print(f"  Indexed {vector_store.total_vectors} vectors")
    elif vector_store.total_vectors > 0:
        print(f"  Loaded existing index: {vector_store.total_vectors} vectors")
    
    # Retriever
    retriever = Retriever(vector_store, top_k=rag_config.top_k)
    
    return retriever


def setup_agents(retriever):
    """Initialize all agents with their assigned models."""
    print("\nInitializing Agents...")
    
    # Create models
    triage_model = create_model(model_config.triage_model)
    expert_model_1 = create_model(model_config.expert_model_1)
    expert_model_2 = create_model(model_config.expert_model_2)
    expert_model_3 = create_model(model_config.expert_model_3)
    skeptic_model = create_model(model_config.skeptic_model)
    synthesizer_model = create_model(model_config.synthesizer_model)
    
    # Create agents
    triage_agent = TriageAgent(triage_model)
    print(f"  Triage Agent: {model_config.triage_model}")
    
    # Dynamically create experts based on config
    experts = []
    expert_models = [
        model_config.expert_model_1,
        model_config.expert_model_2,
        model_config.expert_model_3,
    ]
    
    expert_labels = ["The Analyst", "The Researcher", "The Specialist"]
    
    for i in range(min(debate_config.num_experts, 3)):
        model_name = expert_models[i]
        experts.append(ExpertAgent(i + 1, create_model(model_name)))
        print(f"  Expert {i+1} ({expert_labels[i]}): {model_name}")
    
    skeptic = SkepticAgent(skeptic_model)
    print(f"  Skeptic Agent: {model_config.skeptic_model}")
    
    synthesizer = SynthesizerAgent(synthesizer_model)
    print(f"  Synthesizer Agent: {model_config.synthesizer_model}")
    
    retrieval_agent = RetrievalAgent(retriever)
    print(f"  Retrieval Agent: ready")
    
    # Create protocol
    protocol = CouncilProtocol(
        triage_agent=triage_agent,
        experts=experts,
        skeptic=skeptic,
        synthesizer=synthesizer,
        retrieval_agent=retrieval_agent,
        direct_answer_model=triage_model,  # Reuse for low-stakes
        max_rounds=debate_config.max_rounds,
        consensus_threshold=debate_config.consensus_threshold,
    )
    
    return protocol


def display_result(result):
    """Display the Council Mode result in a formatted way."""
    console.print()
    
    # Triage info
    triage_table = Table(title="Query Triage")
    triage_table.add_column("Field", style="cyan")
    triage_table.add_column("Value", style="white")
    triage_table.add_row("Classification", result.triage.classification)
    triage_table.add_row("Confidence", f"{result.triage.confidence:.0%}")
    triage_table.add_row("Reason", result.triage.reason)
    console.print(triage_table)
    
    if not result.was_debated:
        # Low stakes -- direct answer
        console.print(Panel(
            Markdown(result.direct_answer),
            title="Direct Answer (Low Stakes)",
            border_style="green",
        ))
    else:
        # High stakes -- full debate result
        console.print(Panel(
            f"Debate completed in {result.total_rounds} rounds",
            title="Council Debate",
            border_style="yellow",
        ))
        
        # Final synthesis
        if result.synthesis:
            console.print(Panel(
                Markdown(result.synthesis.final_answer),
                title=f"Final Consensus (Confidence: {result.synthesis.confidence_score}%)",
                border_style="bold green",
            ))
            
            # Source attribution
            if result.synthesis.source_attribution:
                console.print("\n[bold cyan]Source Attribution:[/bold cyan]")
                for src in result.synthesis.source_attribution:
                    console.print(f"  - {src}")


def ingest_documents():
    """Standalone document ingestion command."""
    console.print("[bold yellow]Document Ingestion Mode[/bold yellow]")
    
    loader = DocumentLoader()
    splitter = TextSplitter(
        chunk_size=rag_config.chunk_size,
        chunk_overlap=rag_config.chunk_overlap,
    )
    vector_store = VectorStore(
        embedding_model=rag_config.embedding_model,
        persist_dir=str(rag_config.vector_db_dir),
    )
    
    docs_dir = rag_config.documents_dir
    if not docs_dir.exists() or not any(docs_dir.iterdir()):
        console.print(f"[red]No documents found in {docs_dir}[/red]")
        console.print("Add PDF files to this directory and run again.")
        return
    
    documents = loader.load_directory(str(docs_dir))
    chunks = splitter.split_documents(documents)
    
    vector_store.clear()
    vector_store.add_chunks(chunks)
    
    console.print(f"[bold green]Ingested {len(documents)} documents -> {vector_store.total_vectors} vectors[/bold green]")


async def interactive_mode():
    """Run the interactive CLI mode."""
    print_banner()
    
    # Validate config
    warnings = validate_config()
    for w in warnings:
        console.print(f"  {w}", style="yellow")
    
    # Setup
    retriever = setup_rag_pipeline()
    protocol = setup_agents(retriever)
    
    console.print("\n[bold green]Council Mode is ready![/bold green]")
    console.print("Type your query below. Use '/quit' to exit, '/debate' to force debate mode.\n")
    
    while True:
        try:
            query = Prompt.ask("[bold cyan]You[/bold cyan]")
            
            if query.strip().lower() in ("/quit", "/exit", "/q"):
                console.print("[yellow]Goodbye![/yellow]")
                break
            
            if not query.strip():
                continue
            
            force_debate = False
            if query.startswith("/debate "):
                force_debate = True
                query = query[8:]
            
            with console.status("[bold yellow]Council is deliberating...[/bold yellow]"):
                result = await protocol.process_query(query, force_debate=force_debate)
            
            display_result(result)
            console.print()
            
        except KeyboardInterrupt:
            console.print("\n[yellow]Interrupted. Goodbye![/yellow]")
            break
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
            logger.error(f"Error processing query: {e}", exc_info=True)


def list_available_datasets():
    """List all available benchmark datasets."""
    from data.datasets.registry import list_datasets
    
    table = Table(title="Available Benchmark Datasets")
    table.add_column("Key", style="cyan", width=12)
    table.add_column("Dataset", style="white")
    table.add_column("Best For", style="yellow")
    
    dataset_info = {
        "halueval": ("HaluEval 2.0", "Hallucination Testing"),
        "hotpotqa": ("HotpotQA", "Multi-hop RAG"),
        "squad":    ("SQuAD 2.0", "Unanswerable Questions"),
        "legal":    ("Legal RAG Bench", "Complex Reasoning"),
    }
    
    for key, (name, best_for) in dataset_info.items():
        table.add_row(key, name, best_for)
    
    console.print(table)
    console.print("\n[dim]Usage: python main.py benchmark <dataset_key> [--samples N][/dim]")
    console.print("[dim]       python main.py benchmark all --samples 25[/dim]")


async def run_benchmark(dataset_key: str, sample_limit: int = 25):
    """Run benchmark evaluation on a dataset."""
    from evaluation.benchmark_runner import BenchmarkRunner
    from evaluation.baseline import SingleAgentBaseline
    
    console.print(Panel(
        f"Dataset: [bold]{dataset_key}[/bold]\n"
        f"Samples: {sample_limit}\n"
        f"Mode: Council vs Single-Agent Baseline",
        title="Council Mode -- Benchmark",
        border_style="cyan",
    ))
    
    # Validate config
    warnings = validate_config()
    for w in warnings:
        console.print(f"  {w}", style="yellow")
    
    # Setup
    retriever = setup_rag_pipeline()
    protocol = setup_agents(retriever)
    
    # Create baseline
    baseline_model = create_model(model_config.triage_model)
    baseline = SingleAgentBaseline(baseline_model)
    
    # Create runner
    runner = BenchmarkRunner(protocol=protocol, baseline=baseline)
    
    def progress(current, total, msg):
        console.print(f"  [{current}/{total}] {msg}", style="dim")
    
    if dataset_key == "all":
        reports = await runner.run_all_datasets(
            sample_limit=sample_limit,
            progress_callback=progress,
        )
        for report in reports:
            runner.print_report(report)
            runner.save_report(report)
    else:
        report = await runner.run_dataset(
            dataset_key, sample_limit=sample_limit,
            progress_callback=progress,
        )
        runner.print_report(report)
        filepath = runner.save_report(report)
        console.print(f"\n[bold green]Report saved: {filepath}[/bold green]")


def main():
    """Main entry point."""
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == "ingest":
            ingest_documents()
            return
        elif command == "datasets":
            list_available_datasets()
            return
        elif command == "benchmark":
            # Parse benchmark arguments
            dataset_key = sys.argv[2] if len(sys.argv) > 2 else "all"
            sample_limit = 25  # default
            if "--samples" in sys.argv:
                idx = sys.argv.index("--samples")
                if idx + 1 < len(sys.argv):
                    sample_limit = int(sys.argv[idx + 1])
            asyncio.run(run_benchmark(dataset_key, sample_limit))
            return
        elif command == "help":
            console.print(Panel(
                "Usage:\n"
                "  python main.py              -- Interactive CLI mode\n"
                "  python main.py ingest       -- Ingest documents into vector DB\n"
                "  python main.py datasets     -- List available benchmark datasets\n"
                "  python main.py benchmark    -- Run all benchmarks (25 samples each)\n"
                "  python main.py benchmark halueval --samples 50\n"
                "  python main.py benchmark hotpotqa --samples 100\n"
                "  python main.py benchmark squad --samples 50\n"
                "  python main.py benchmark legal --samples 30\n"
                "  python main.py benchmark all --samples 25\n"
                "  streamlit run app.py        -- Web UI mode\n",
                title="Council Mode -- Help",
            ))
            return
    
    asyncio.run(interactive_mode())


if __name__ == "__main__":
    main()
