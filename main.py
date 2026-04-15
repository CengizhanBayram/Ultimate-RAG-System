#!/usr/bin/env python3
"""
RAG System CLI Entry Point.

Usage:
    python main.py --query "Pro paket fiyatı nedir?"
    python main.py --interactive
    python main.py --rebuild-index
    python main.py --run-tests
    python main.py --watch
    python main.py --benchmark
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path

# Force UTF-8 output on Windows so Turkish characters (₺, ş, ğ, ü…) render correctly
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Ensure src is importable when run from project root
sys.path.insert(0, str(Path(__file__).parent))

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich import box

console = Console()


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


def build_pipeline():
    from src.pipeline import RAGPipeline
    return RAGPipeline()


def print_response(result, question: str) -> None:
    """Print a GeneratorResponse using Rich formatting."""
    # Answer panel
    answer_text = result.answer
    console.print(
        Panel(
            answer_text,
            title=f"[bold cyan]Yanıt[/bold cyan]",
            subtitle=f"Latency: {result.latency_ms:.0f}ms | "
                     f"Chunks: {result.chunk_count} | "
                     f"Cache: {'HIT' if result.was_cache_hit else 'MISS'}",
            border_style="cyan",
            padding=(1, 2),
        )
    )

    # Evaluation scores
    ev = result.evaluation
    score_color = "green" if ev.passes_guardrail else "red"
    console.print(
        f"[{score_color}]Faithfulness: {ev.faithfulness_score:.2f} | "
        f"Relevance: {ev.relevance_score:.2f} | "
        f"Guardrail: {'✓ PASS' if ev.passes_guardrail else '✗ FAIL'}[/{score_color}]"
    )

    # Sources
    if result.sources_used:
        console.print(f"[dim]Sources: {', '.join(result.sources_used)}[/dim]")

    # Conflicts
    if result.conflicts_resolved:
        console.print(
            Panel(
                "\n".join(f"• {c}" for c in result.conflicts_resolved),
                title="[yellow]Çözülen Çakışmalar[/yellow]",
                border_style="yellow",
            )
        )

    # Queries used
    if len(result.queries_used) > 1:
        console.print(
            f"[dim]Expanded queries ({len(result.queries_used)}): "
            f"{' | '.join(result.queries_used[:3])}...[/dim]"
        )


def cmd_single_query(question: str) -> None:
    setup_logging("WARNING")
    console.print(Panel(f"[bold]Soru:[/bold] {question}", border_style="blue"))
    console.print("[yellow]Index yükleniyor / oluşturuluyor...[/yellow]")

    pipeline = build_pipeline()
    pipeline.build_index()

    console.print("[yellow]Sorgu işleniyor...[/yellow]")
    result = pipeline.query(question)
    print_response(result, question)


def cmd_interactive(with_watcher: bool = False) -> None:
    setup_logging("WARNING")
    console.print(
        Panel(
            "[bold green]RAG Sistemi — Etkileşimli Mod[/bold green]\n"
            "Çıkmak için 'quit' veya 'exit' yazın.",
            border_style="green",
        )
    )

    pipeline = build_pipeline()
    console.print("[yellow]Index yükleniyor...[/yellow]")
    pipeline.build_index()
    console.print("[green]Hazır![/green]\n")

    if with_watcher:
        from src.watcher import DataFileWatcher
        from src.config import settings
        watcher = DataFileWatcher(pipeline, settings.data_dir)
        watcher.start()

    try:
        while True:
            try:
                question = console.input("[bold blue]Sorunuz:[/bold blue] ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if question.lower() in ("quit", "exit", "q", ""):
                break
            result = pipeline.query(question)
            print_response(result, question)
            console.print()
    finally:
        if with_watcher:
            watcher.stop()


def cmd_rebuild_index() -> None:
    setup_logging("INFO")
    pipeline = build_pipeline()
    console.print("[yellow]Index yeniden oluşturuluyor...[/yellow]")
    t0 = time.perf_counter()
    pipeline.build_index(force_rebuild=True)
    elapsed = time.perf_counter() - t0
    console.print(f"[green]Index {elapsed:.2f}s içinde oluşturuldu.[/green]")


def cmd_run_tests() -> None:
    import subprocess
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/", "-v", "--tb=short"],
        cwd=Path(__file__).parent,
    )
    sys.exit(result.returncode)


def cmd_benchmark() -> None:
    """Run all 10 test cases and print a Rich table with pass/fail."""
    setup_logging("WARNING")

    TEST_CASES = [
        {
            "id": 1,
            "query": "Pro paket aylık ücreti nedir?",
            "must_contain": ["599", "Pro"],
            "must_source": ["paket_fiyatlari.csv"],
        },
        {
            "id": 2,
            "query": "Pro paketimi iptal edersem param ne zaman gelir?",
            "must_contain": ["30"],
            "must_source": ["guncellemeler.json"],
        },
        {
            "id": 3,
            "query": "Basic paket iade süresi kaç gün?",
            "must_contain": ["7"],
            "must_source": ["guncellemeler.json"],
        },
        {
            "id": 4,
            "query": "Standart sözleşme iade maddesi ne diyor?",
            "must_contain": ["14", "güncellem"],
            "must_source": ["sozlesme.txt"],
        },
        {
            "id": 5,
            "query": "Basic ve Enterprise farkı nedir?",
            "must_contain": ["Basic", "Enterprise"],
            "must_source": ["paket_fiyatlari.csv"],
        },
        {
            "id": 6,
            "query": "Veri silme hakkım var mı?",
            "must_contain": ["KVKK", "silme"],
            "must_source": ["sozlesme.txt"],
        },
        {
            "id": 7,
            "query": "Hangi paket 7/24 destek sunuyor?",
            "must_contain": ["7/24"],
            "must_source": ["paket_fiyatlari.csv"],
        },
        {
            "id": 8,
            "query": "Sözleşmeyi feshedersem tazminat öder miyim?",
            "must_contain": ["fesih", "tazminat"],
            "must_source": ["sozlesme.txt"],
        },
        {
            "id": 9,
            "query": "Şu an kaç kullanıcı ekleyebilirim?",
            "must_contain": ["Basic", "Pro", "Enterprise"],
            "must_source": ["paket_fiyatlari.csv"],
        },
        {
            "id": 10,
            "query": "Aydan aya fiyat değişti mi?",
            "must_contain": ["UPD-003", "fiyat"],
            "must_source": ["guncellemeler.json"],
        },
    ]

    console.print("[yellow]Index yükleniyor...[/yellow]")
    pipeline = build_pipeline()
    pipeline.build_index()
    console.print("[green]Hazır — benchmark başlatılıyor...[/green]\n")

    table = Table(
        title="RAG Benchmark Sonuçları",
        box=box.ROUNDED,
        show_lines=True,
    )
    table.add_column("#", style="cyan", width=3)
    table.add_column("Sorgu", width=40)
    table.add_column("Kaynak", width=8)
    table.add_column("Değer", width=8)
    table.add_column("Faith.", width=7)
    table.add_column("Durum", width=6)

    passed = 0
    total = len(TEST_CASES)

    for tc in TEST_CASES:
        try:
            result = pipeline.query(tc["query"])
            answer_lower = result.answer.lower()

            source_ok = any(
                s in result.sources_used for s in tc["must_source"]
            )
            value_ok = all(
                kw.lower() in answer_lower for kw in tc["must_contain"]
            )
            faith_ok = result.evaluation.faithfulness_score >= 0.70

            status_ok = source_ok and value_ok and faith_ok
            if status_ok:
                passed += 1

            status_str = "[green]PASS[/green]" if status_ok else "[red]FAIL[/red]"
            src_str = "[green]✓[/green]" if source_ok else "[red]✗[/red]"
            val_str = "[green]✓[/green]" if value_ok else "[red]✗[/red]"
            faith_str = f"{result.evaluation.faithfulness_score:.2f}"

            table.add_row(
                str(tc["id"]),
                tc["query"][:38] + "…" if len(tc["query"]) > 38 else tc["query"],
                src_str,
                val_str,
                faith_str,
                status_str,
            )
        except Exception as e:
            table.add_row(
                str(tc["id"]),
                tc["query"][:38],
                "[red]ERR[/red]",
                "[red]ERR[/red]",
                "N/A",
                "[red]ERROR[/red]",
            )
            console.print(f"[red]Test {tc['id']} hatası: {e}[/red]")

    console.print(table)
    score_color = "green" if passed == total else ("yellow" if passed >= total * 0.7 else "red")
    console.print(
        f"\n[{score_color}]Sonuç: {passed}/{total} test geçti "
        f"({passed/total*100:.0f}%)[/{score_color}]"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Production RAG System CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Örnekler:
  python main.py --query "Pro paket fiyatı nedir?"
  python main.py --interactive
  python main.py --watch
  python main.py --rebuild-index
  python main.py --benchmark
  python main.py --run-tests
        """,
    )
    parser.add_argument("--query", "-q", type=str, help="Tek seferlik sorgu")
    parser.add_argument(
        "--interactive", "-i", action="store_true", help="Etkileşimli mod"
    )
    parser.add_argument(
        "--watch",
        action="store_true",
        help="Etkileşimli mod + otomatik index yenileme",
    )
    parser.add_argument(
        "--rebuild-index", action="store_true", help="Index'i yeniden oluştur"
    )
    parser.add_argument(
        "--run-tests", action="store_true", help="pytest testlerini çalıştır"
    )
    parser.add_argument(
        "--benchmark", action="store_true", help="10 sorgu benchmark"
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Log seviyesi",
    )

    args = parser.parse_args()

    if args.run_tests:
        cmd_run_tests()
    elif args.rebuild_index:
        setup_logging(args.log_level)
        cmd_rebuild_index()
    elif args.benchmark:
        cmd_benchmark()
    elif args.query:
        setup_logging(args.log_level)
        cmd_single_query(args.query)
    elif args.watch:
        cmd_interactive(with_watcher=True)
    elif args.interactive:
        cmd_interactive(with_watcher=False)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
