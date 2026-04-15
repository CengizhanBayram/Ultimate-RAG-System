"""
Streamlit Web UI for the RAG System.

Run:
    streamlit run app.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import streamlit as st

# ── Page config (must be first Streamlit call) ─────────────────────────────
st.set_page_config(
    page_title="RAG Destek Sistemi",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ── Pipeline singleton ──────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def get_pipeline():
    from src.pipeline import RAGPipeline
    p = RAGPipeline()
    p.build_index()
    return p


# ── Session state init ──────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "model_loaded" not in st.session_state:
    st.session_state.model_loaded = False
if "pending_query" not in st.session_state:
    st.session_state.pending_query = None


# ── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("🔧 RAG Kontrol Paneli")
    st.divider()

    # Data source status
    st.subheader("📂 Veri Kaynakları")
    data_dir = Path("./data")
    files = [
        ("sozlesme.txt", "Sözleşme"),
        ("paket_fiyatlari.csv", "Fiyat Tablosu"),
        ("guncellemeler.json", "Güncelleme Logu"),
    ]
    for fname, label in files:
        fpath = data_dir / fname
        if fpath.exists():
            mtime = time.strftime("%d.%m.%Y %H:%M", time.localtime(fpath.stat().st_mtime))
            size_kb = fpath.stat().st_size / 1024
            st.success(f"✅ **{label}** (`{fname}`)\n{mtime} — {size_kb:.1f} KB")
        else:
            st.error(f"❌ **{label}** (`{fname}`) bulunamadı")

    st.divider()

    # Index freshness
    st.subheader("🗄️ FAISS Index")
    faiss_path = Path("./index_cache/faiss.index")
    if faiss_path.exists():
        mtime = time.strftime("%d.%m.%Y %H:%M", time.localtime(faiss_path.stat().st_mtime))
        st.info(f"Son güncelleme: {mtime}")
    else:
        st.warning("Index henüz oluşturulmadı")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 Index Yenile", use_container_width=True):
            with st.spinner("Yeniden oluşturuluyor..."):
                try:
                    p = get_pipeline()
                    p.cache.invalidate_all()
                    p.build_index(force_rebuild=True)
                    st.cache_resource.clear()
                    st.session_state.model_loaded = False
                    st.success("Tamamlandı!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Hata: {e}")
    with col2:
        if st.button("🗑️ Cache Temizle", use_container_width=True):
            try:
                get_pipeline().cache.invalidate_all()
                st.success("Temizlendi!")
            except Exception as e:
                st.error(f"Hata: {e}")

    st.divider()

    # Clear chat
    if st.session_state.messages:
        if st.button("💬 Sohbeti Temizle", use_container_width=True, type="secondary"):
            st.session_state.messages = []
            st.session_state.pending_query = None
            st.rerun()

    st.divider()

    # Settings
    with st.expander("⚙️ Sistem Ayarları"):
        from src.config import settings
        st.markdown(f"""
| Parametre | Değer |
|---|---|
| Embedding | `{settings.embed_model.split('/')[-1]}` |
| Cross-Encoder | `{settings.cross_encoder_model.split('/')[-1]}` |
| Gemini Modeli | `{settings.gemini_model.split('/')[-1]}` |
| Top-K Retrieval | `{settings.top_k_retrieval}` |
| Top-K Final | `{settings.top_k_final}` |
| BM25 Ağırlık | `{settings.bm25_weight}` |
| Dense Ağırlık | `{settings.dense_weight}` |
| Faithfulness Eşiği | `{settings.faithfulness_threshold}` |
| Sorgu Genişletme | `{'Aktif' if settings.query_expansion_enabled else 'Pasif'}` |
""")


# ── Main Area ───────────────────────────────────────────────────────────────
st.title("🤖 Müşteri Destek RAG Sistemi")
st.caption("Sözleşme, fiyat tablosu ve güncelleme loglarına dayalı akıllı destek asistanı")

# ── İlk sorgu uyarısı ───────────────────────────────────────────────────────
if not st.session_state.model_loaded:
    st.info(
        "⏳ **İlk sorgu ~30–45 saniye sürebilir** — BGE-M3 dil modeli belleğe yükleniyor. "
        "Model yüklendikten sonra sonraki sorgular çok daha hızlı yanıt verir "
        "(genellikle 5–15 saniye, önbellekte anında).",
        icon="ℹ️",
    )

# ── Chat geçmişini göster ───────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        # Yanıt metadatasını da göster (varsa)
        if msg["role"] == "assistant" and "meta" in msg:
            meta = msg["meta"]
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Gecikme", f"{meta['latency_ms']:.0f}ms")
            c2.metric("Doğruluk", f"{meta['faithfulness']:.0%}")
            c3.metric("Chunk", meta["chunk_count"])
            c4.metric("Cache", "HIT" if meta["cache_hit"] else "MISS")
            if meta.get("sources"):
                with st.expander("📎 Kullanılan Kaynaklar"):
                    for src in meta["sources"]:
                        st.write(f"- `{src}`")
            if meta.get("conflicts"):
                with st.expander(f"⚠️ Çözülen Çakışmalar ({len(meta['conflicts'])})"):
                    for c in meta["conflicts"]:
                        st.warning(c)
            if meta.get("queries") and len(meta["queries"]) > 1:
                with st.expander(f"🔍 Genişletilmiş Sorgular ({len(meta['queries'])})"):
                    for i, q in enumerate(meta["queries"]):
                        st.write(f"{'**Orijinal:**' if i == 0 else f'`{i}.`'} {q}")

# ── Sorgu al: chat input VEYA örnek butonu ──────────────────────────────────
user_input = st.chat_input("Sorunuzu yazın... (örn: Pro paket aylık ücreti nedir?)")

# Örnek buton tıklandıysa pending_query'den al
if st.session_state.pending_query and not user_input:
    user_input = st.session_state.pending_query
    st.session_state.pending_query = None

# ── Yanıt üretimi ────────────────────────────────────────────────────────────
if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        spinner_text = (
            "⏳ BGE-M3 modeli yükleniyor ve sorgu işleniyor... (ilk sorgu uzun sürebilir)"
            if not st.session_state.model_loaded
            else "🔍 Sorgu işleniyor..."
        )
        with st.spinner(spinner_text):
            try:
                pipeline = get_pipeline()
                result = pipeline.query(user_input)
                st.session_state.model_loaded = True

                # Ana yanıt
                st.markdown(result.answer)

                # Metrikler
                ev = result.evaluation
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Gecikme", f"{result.latency_ms:.0f}ms")
                c2.metric(
                    "Doğruluk",
                    f"{ev.faithfulness_score:.0%}",
                    delta="✓ PASS" if ev.passes_guardrail else "✗ FAIL",
                    delta_color="normal" if ev.passes_guardrail else "inverse",
                )
                c3.metric("Chunk", result.chunk_count)
                c4.metric("Cache", "HIT" if result.was_cache_hit else "MISS")

                # Kaynaklar
                if result.sources_used:
                    with st.expander("📎 Kullanılan Kaynaklar"):
                        for src in result.sources_used:
                            st.write(f"- `{src}`")

                # Çakışmalar
                if result.conflicts_resolved:
                    with st.expander(f"⚠️ Çözülen Çakışmalar ({len(result.conflicts_resolved)})"):
                        for conflict in result.conflicts_resolved:
                            st.warning(conflict)

                # Genişletilmiş sorgular
                if len(result.queries_used) > 1:
                    with st.expander(f"🔍 Genişletilmiş Sorgular ({len(result.queries_used)})"):
                        for i, q in enumerate(result.queries_used):
                            st.write(f"{'**Orijinal:**' if i == 0 else f'`{i}.`'} {q}")

                # Guardrail uyarısı
                if not ev.passes_guardrail:
                    st.error(
                        f"🚨 Guardrail UYARISI — Yanıt doğruluk eşiğinin altında "
                        f"(Faithfulness: {ev.faithfulness_score:.2f})"
                    )
                elif ev.unfaithful_claims:
                    st.warning(
                        f"⚠️ {len(ev.unfaithful_claims)} doğrulanamayan iddia tespit edildi: "
                        + ", ".join(ev.unfaithful_claims[:2])
                    )

                # Mesajı metadata ile kaydet (geçmişte tekrar göstermek için)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": result.answer,
                    "meta": {
                        "latency_ms": result.latency_ms,
                        "faithfulness": ev.faithfulness_score,
                        "chunk_count": result.chunk_count,
                        "cache_hit": result.was_cache_hit,
                        "sources": result.sources_used,
                        "conflicts": result.conflicts_resolved,
                        "queries": result.queries_used,
                    },
                })

            except Exception as e:
                error_msg = f"❌ Bir hata oluştu: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})


# ── Örnek sorular (sadece boş ekranda göster) ────────────────────────────────
if not st.session_state.messages:
    st.divider()
    st.subheader("💡 Örnek Sorular")
    st.caption("Bir soruya tıkla, anında yanıtlanır.")

    example_queries = [
        ("💰", "Pro paket aylık ücreti nedir?"),
        ("🔄", "Basic paket iade süresi kaç gün?"),
        ("💾", "Enterprise paket depolama kapasitesi nedir?"),
        ("🔒", "Veri silme hakkım var mı? KVKK kapsamında ne yapabilirim?"),
        ("❌", "Pro paketimi iptal edersem param ne zaman gelir?"),
        ("📞", "Hangi paket 7/24 destek sunuyor?"),
        ("📊", "Basic ve Enterprise farkı nedir?"),
        ("⚖️", "Sözleşmeyi feshedersem tazminat öder miyim?"),
    ]

    cols = st.columns(2)
    for i, (icon, q) in enumerate(example_queries):
        with cols[i % 2]:
            if st.button(f"{icon} {q}", key=f"ex_{i}", use_container_width=True):
                st.session_state.pending_query = q
                st.rerun()
