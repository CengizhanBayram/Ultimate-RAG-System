import re
import json
import logging
from pathlib import Path
from uuid import uuid4
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from typing import List

import pandas as pd

from .models import Document

logger = logging.getLogger(__name__)

# Approximate token count: words * 1.3
def _approx_tokens(text: str) -> float:
    return len(text.split()) * 1.3


class DocumentLoader:
    """Parallel multi-format loader for .txt, .csv, .json data files."""

    def load_txt(self, filepath: Path) -> List[Document]:
        """Parse sozlesme.txt by Madde boundaries into one Document per article."""
        text = filepath.read_text(encoding="utf-8")

        # Extract header date
        date_match = re.search(r"Sözleşme Tarihi:\s*(\d{4}-\d{2}-\d{2})", text)
        belge_tarihi = date_match.group(1) if date_match else "unknown"

        # Split by Madde boundaries
        pattern = re.compile(r"(Madde\s+\d+(?:\.\d+)*)\s*:", re.IGNORECASE)
        matches = list(pattern.finditer(text))

        documents = []

        if not matches:
            # Fallback: treat entire file as one document
            doc = Document(
                content=text.strip(),
                metadata={
                    "source": filepath.name,
                    "type": "sozlesme",
                    "madde": "genel",
                    "belge_tarihi": belge_tarihi,
                    "chunk_id": str(uuid4()),
                    "priority_tier": 3,
                },
            )
            documents.append(doc)
            return documents

        for i, match in enumerate(matches):
            madde_label = match.group(1).strip()
            # Normalize: "Madde 4.1" -> "4.1"
            madde_num = re.sub(r"(?i)madde\s+", "", madde_label).strip()

            start = match.start()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            article_text = text[start:end].strip()

            if _approx_tokens(article_text) <= 600:
                # Short enough: one document
                doc = Document(
                    content=article_text,
                    metadata={
                        "source": filepath.name,
                        "type": "sozlesme",
                        "madde": madde_num,
                        "belge_tarihi": belge_tarihi,
                        "chunk_id": str(uuid4()),
                        "priority_tier": 3,
                    },
                )
                documents.append(doc)
            else:
                # Long article: sub-chunk with overlap
                sub_chunks = self._semantic_split(article_text, max_tokens=600)
                for j, chunk_text in enumerate(sub_chunks):
                    doc = Document(
                        content=chunk_text,
                        metadata={
                            "source": filepath.name,
                            "type": "sozlesme",
                            "madde": madde_num,
                            "belge_tarihi": belge_tarihi,
                            "chunk_id": str(uuid4()),
                            "priority_tier": 3,
                            "sub_chunk_index": j,
                            "total_sub_chunks": len(sub_chunks),
                        },
                    )
                    documents.append(doc)

        logger.info(f"Loaded {len(documents)} documents from {filepath.name}")
        return documents

    def load_csv(self, filepath: Path) -> List[Document]:
        """Read paket_fiyatlari.csv. Each row = exactly one Document, never split."""
        # Read raw lines to skip comment rows
        lines = filepath.read_text(encoding="utf-8").splitlines()
        non_comment = [ln for ln in lines if not ln.strip().startswith("#")]
        csv_text = "\n".join(non_comment)

        from io import StringIO
        df = pd.read_csv(StringIO(csv_text))

        documents = []
        for _, row in df.iterrows():
            content = (
                f"Paket: {row['paket_adi']} | "
                f"Aylık: ₺{row['aylik_fiyat_tl']} | "
                f"Yıllık: ₺{row['yillik_fiyat_tl']} | "
                f"Kullanıcı: {row['kullanici_limiti']} | "
                f"Depolama: {row['depolama_gb']}GB | "
                f"API/Ay: {row['api_cagrisi_limiti_aylik']} | "
                f"Destek: {row['destek_tipi']} | "
                f"SLA: %{row['sla_uptime_yuzde']} | "
                f"Özellikler: {row['ozellikler']}"
            )
            doc = Document(
                content=content,
                metadata={
                    "source": filepath.name,
                    "type": "fiyat_tablosu",
                    "paket": str(row["paket_adi"]),
                    "son_guncelleme": str(row["son_guncelleme_tarihi"]),
                    "chunk_id": str(uuid4()),
                    "priority_tier": 2,
                    "aylik_fiyat_tl": row["aylik_fiyat_tl"],
                    "yillik_fiyat_tl": row["yillik_fiyat_tl"],
                },
            )
            documents.append(doc)

        logger.info(f"Loaded {len(documents)} documents from {filepath.name}")
        return documents

    def load_json(self, filepath: Path) -> List[Document]:
        """Load guncellemeler.json — each entry = one Document."""
        with open(filepath, encoding="utf-8") as f:
            entries = json.load(f)

        documents = []
        for entry in entries:
            paket_str = entry.get("etkilenen_paket")
            paket_part = f", Paket: {paket_str}" if paket_str else ""

            content = (
                f"[{entry['id']} | {entry['tarih']}] {entry['degisiklik']}\n"
                f"Etkilenen: Madde {entry.get('etkilenen_madde', 'N/A')}{paket_part}\n"
                f"Değişim: {entry['onceki_deger']} → {entry['yeni_deger']} | "
                f"Onaylayan: {entry['onaylayan']}"
            )

            try:
                tarih_date = date.fromisoformat(entry["tarih"])
            except (ValueError, KeyError):
                tarih_date = None

            doc = Document(
                content=content,
                metadata={
                    "source": filepath.name,
                    "type": "guncelleme_logu",
                    "guncelleme_id": entry["id"],
                    "tarih": entry["tarih"],
                    "tarih_date": tarih_date,
                    "etkilenen_paket": entry.get("etkilenen_paket"),
                    "etkilened_paket": entry.get("etkilened_paket"),  # typo variant in data
                    "etkilenen_madde": entry.get("etkilenen_madde"),
                    "onceki_deger": entry["onceki_deger"],
                    "yeni_deger": entry["yeni_deger"],
                    "chunk_id": str(uuid4()),
                    "priority_tier": 1,
                },
            )
            documents.append(doc)

        logger.info(f"Loaded {len(documents)} documents from {filepath.name}")
        return documents

    def load_all_parallel(self, data_dir: Path) -> List[Document]:
        """Load all three data files in parallel using ThreadPoolExecutor."""
        tasks = [
            (self.load_txt,  data_dir / "sozlesme.txt"),
            (self.load_csv,  data_dir / "paket_fiyatlari.csv"),
            (self.load_json, data_dir / "guncellemeler.json"),
        ]
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = {executor.submit(fn, path): path for fn, path in tasks}
            results = []
            for future in as_completed(futures):
                try:
                    results.extend(future.result())
                except Exception as e:
                    logger.error(f"Loader failed for {futures[future]}: {e}")

        logger.info(f"Loaded {len(results)} documents from {len(tasks)} sources")
        return results

    def _semantic_split(self, text: str, max_tokens: int) -> List[str]:
        """Split long text by sentence boundaries with overlap (last 2 sentences)."""
        # Turkish sentence splitter: split on . ! ? ; followed by whitespace
        sentence_pattern = re.compile(r"(?<=[.!?;])\s+")
        sentences = sentence_pattern.split(text)
        sentences = [s.strip() for s in sentences if s.strip()]

        chunks = []
        current_sentences: List[str] = []
        current_tokens = 0.0

        for sentence in sentences:
            sent_tokens = _approx_tokens(sentence)
            if current_tokens + sent_tokens > max_tokens and current_sentences:
                chunks.append(" ".join(current_sentences))
                # Overlap: keep last 2 sentences
                overlap = current_sentences[-2:] if len(current_sentences) >= 2 else current_sentences[-1:]
                current_sentences = overlap
                current_tokens = sum(_approx_tokens(s) for s in current_sentences)
            current_sentences.append(sentence)
            current_tokens += sent_tokens

        if current_sentences:
            chunks.append(" ".join(current_sentences))

        return chunks if chunks else [text]
