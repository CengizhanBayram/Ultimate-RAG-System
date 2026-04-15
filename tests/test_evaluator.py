"""Tests for RAGEvaluator (mocked Claude calls)."""
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.evaluator import RAGEvaluator
from src.models import Document, ScoredDocument, EvaluationResult


def _make_settings():
    s = MagicMock()
    s.google_api_key = "test-key"
    s.gemini_model = "gemini-2.5-flash-preview-04-17"
    s.faithfulness_threshold = 0.70
    s.relevance_threshold = 0.50
    return s


def _make_chunk(content):
    return ScoredDocument(
        document=Document(
            content=content,
            metadata={"chunk_id": "x", "source": "test.txt", "type": "sozlesme"},
        ),
        retrieval_score=0.8,
    )


def _mock_response(faithfulness, relevance, unfaithful=None, explanation="ok"):
    data = {
        "faithfulness_score": faithfulness,
        "relevance_score": relevance,
        "unfaithful_claims": unfaithful or [],
        "explanation": explanation,
    }
    content = MagicMock()
    content.text = json.dumps(data)
    msg = MagicMock()
    msg.content = [content]
    return msg


class TestRAGEvaluator:
    def test_passes_when_scores_above_threshold(self):
        evaluator = RAGEvaluator(_make_settings())
        with patch.object(evaluator.client.messages, "create",
                          return_value=_mock_response(0.9, 0.8)):
            result = evaluator.evaluate(
                "test query", "test answer", [_make_chunk("test content")]
            )
        assert result.passes_guardrail is True
        assert result.faithfulness_score == 0.9
        assert result.relevance_score == 0.8

    def test_fails_when_faithfulness_below_threshold(self):
        evaluator = RAGEvaluator(_make_settings())
        with patch.object(evaluator.client.messages, "create",
                          return_value=_mock_response(0.5, 0.8)):
            result = evaluator.evaluate("q", "a", [_make_chunk("ctx")])
        assert result.passes_guardrail is False

    def test_fails_when_relevance_below_threshold(self):
        evaluator = RAGEvaluator(_make_settings())
        with patch.object(evaluator.client.messages, "create",
                          return_value=_mock_response(0.9, 0.3)):
            result = evaluator.evaluate("q", "a", [_make_chunk("ctx")])
        assert result.passes_guardrail is False

    def test_unfaithful_claims_returned(self):
        evaluator = RAGEvaluator(_make_settings())
        with patch.object(evaluator.client.messages, "create",
                          return_value=_mock_response(0.8, 0.7, ["claim X"])):
            result = evaluator.evaluate("q", "a", [_make_chunk("ctx")])
        assert "claim X" in result.unfaithful_claims

    def test_fails_open_on_api_error(self):
        evaluator = RAGEvaluator(_make_settings())
        with patch.object(evaluator.client.messages, "create",
                          side_effect=Exception("API down")):
            result = evaluator.evaluate("q", "a", [_make_chunk("ctx")])
        assert result.passes_guardrail is True
        assert result.faithfulness_score == 1.0

    def test_handles_markdown_wrapped_json(self):
        evaluator = RAGEvaluator(_make_settings())
        raw = '```json\n{"faithfulness_score": 0.95, "relevance_score": 0.8, "unfaithful_claims": [], "explanation": "good"}\n```'
        content = MagicMock()
        content.text = raw
        msg = MagicMock()
        msg.content = [content]
        with patch.object(evaluator.client.messages, "create", return_value=msg):
            result = evaluator.evaluate("q", "a", [_make_chunk("ctx")])
        assert result.faithfulness_score == 0.95
