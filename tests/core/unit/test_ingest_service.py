"""Unit tests for IngestionService."""

from src.core.ingest_service import IngestionService
from tests.helpers import DummyIngestor


def test_ingestion_service_delegates():
    ingestor = DummyIngestor()
    service = IngestionService(ingestor)
    result = service.run("data", "vs", ["txt"])
    assert result == "vectorstore"
    assert ingestor.calls == [("data", "vs", ("txt",))]
