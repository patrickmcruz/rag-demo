from src.ingest import IngestionService


class DummyIngestor:
    def __init__(self):
        self.calls = []

    def ingest(self, data_dir, persist_dir, file_types=None):
        self.calls.append((data_dir, persist_dir, tuple(file_types) if file_types else None))
        return "ok"


def test_ingestion_service_delegates():
    ingestor = DummyIngestor()
    service = IngestionService(ingestor)
    result = service.run("data", "vs", ["txt"])
    assert result == "ok"
    assert ingestor.calls == [("data", "vs", ("txt",))]
