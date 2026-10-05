"""End-to-end ingestion pipeline (runs inside a Celery worker).

    parse -> clean -> chunk -> extract entities -> extract relationships
          -> resolve entities -> build graph -> embed -> index -> complete
"""

from __future__ import annotations

import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from typing import Any

from app.core.config import Settings
from app.core.errors import DocumentProcessingError
from app.core.logging import get_logger
from app.graph.builder import GraphBuilder
from app.ingestion.chunker import TokenChunker
from app.ingestion.embedding import Embedder
from app.ingestion.entity_resolver import EntityResolver
from app.ingestion.loader import FileStorage
from app.ingestion.metadata import chunk_records, document_metadata
from app.ingestion.parser import ParseOptions, parse_document
from app.ingestion.relationship_extractor import ChunkExtraction, GraphExtractor
from app.models.job import JobStage
from app.utils.text import clean_text

logger = get_logger(__name__)

StageCallback = Callable[[str, dict[str, Any]], None]


@dataclass
class IngestionResult:
    title: str | None
    page_count: int | None
    chunks: int
    entities: int
    relationships: int
    mentions: int
    merges: int
    metadata: dict[str, Any] = field(default_factory=dict)
    timings_ms: dict[str, int] = field(default_factory=dict)

    def stats(self) -> dict[str, Any]:
        data = asdict(self)
        data.pop("metadata")
        return data


class IngestionPipeline:
    def __init__(
        self,
        settings: Settings,
        storage: FileStorage,
        embedder: Embedder,
        extractor_factory: Callable[[], GraphExtractor],
        resolver: EntityResolver,
        builder: GraphBuilder,
    ) -> None:
        self.settings = settings
        self.storage = storage
        self.embedder = embedder
        self.extractor_factory = extractor_factory
        self.resolver = resolver
        self.builder = builder
        self.chunker = TokenChunker(settings.chunk_size, settings.chunk_overlap)

    def run(
        self,
        *,
        tenant_id: str,
        document_id: str,
        filename: str,
        file_type: str,
        storage_path: str,
        on_stage: StageCallback | None = None,
    ) -> IngestionResult:
        timings: dict[str, int] = {}
        notify = on_stage or (lambda stage, detail: None)
        clock = time.perf_counter()

        def mark(stage: str, **detail: Any) -> None:
            nonlocal clock
            now = time.perf_counter()
            timings[stage] = int((now - clock) * 1000)
            clock = now
            notify(stage, detail)

        # 1. Parse
        mark(JobStage.PARSING)
        data = self.storage.load(storage_path)
        parsed = parse_document(data, file_type, filename, ParseOptions(
            ocr_enabled=self.settings.ocr_enabled, ocr_language=self.settings.ocr_language,
            ocr_dpi=self.settings.ocr_dpi, ocr_min_page_chars=self.settings.ocr_min_page_chars))

        # 2. Clean
        mark(JobStage.CLEANING, blocks=len(parsed.blocks), pages=parsed.page_count)
        for block in parsed.blocks:
            block.text = clean_text(block.text)
        parsed.blocks = [b for b in parsed.blocks if b.text]
        meta = document_metadata(parsed, filename, file_type)

        # 3. Chunk
        mark(JobStage.CHUNKING)
        chunks = self.chunker.chunk(parsed.blocks)
        if not chunks:
            raise DocumentProcessingError("Document produced no chunks")
        records = chunk_records(
            chunks, tenant_id=tenant_id, document_id=document_id, source_filename=filename, title=parsed.title
        )

        # 4/5. Entities + relationships (one joint pass per chunk; bounded concurrency for LLM calls)
        mark(JobStage.EXTRACTING_ENTITIES, chunks=len(records))
        extractor = self.extractor_factory()
        extractor.prime([r["text"] for r in records])

        def extract(record: dict[str, Any]) -> ChunkExtraction:
            return extractor.extract(record["text"], record["id"])

        workers = max(1, self.settings.llm_max_concurrency) if extractor.name == "llm" else 1
        with ThreadPoolExecutor(max_workers=workers) as pool:
            extractions = list(pool.map(extract, records))
        n_entity_mentions = sum(len(x.entities) for x in extractions)
        mark(JobStage.EXTRACTING_RELATIONSHIPS, entity_mentions=n_entity_mentions)
        triples = [(r["id"], x.entities, x.relationships) for r, x in zip(records, extractions, strict=True)]

        # 6. Resolve
        mark(JobStage.RESOLVING_ENTITIES, relationship_mentions=sum(len(x.relationships) for x in extractions))
        resolution = self.resolver.resolve(tenant_id, triples)

        # 7. Graph (idempotent: drop any previous version of this document first)
        mark(JobStage.BUILDING_GRAPH, entities=len(resolution.entities), relationships=len(resolution.relationships))
        self.builder.reset_document(tenant_id, document_id)
        stats = self.builder.build(
            tenant_id=tenant_id,
            document_id=document_id,
            filename=filename,
            title=parsed.title,
            file_type=file_type,
            chunks=records,
            resolution=resolution,
        )

        # 8. Embeddings (batched)
        mark(JobStage.EMBEDDING)
        texts = [self._embedding_text(r) for r in records]
        embeddings: list[list[float]] = []
        batch = max(1, self.settings.embedding_batch_size)
        for i in range(0, len(texts), batch):
            embeddings.extend(self.embedder.embed_documents(texts[i : i + batch]))

        # 9. Vector index
        mark(JobStage.INDEXING, embeddings=len(embeddings))
        self.builder.index_embeddings(tenant_id, [r["id"] for r in records], embeddings)
        mark(JobStage.COMPLETED)

        result = IngestionResult(
            title=parsed.title,
            page_count=parsed.page_count,
            chunks=stats.chunks,
            entities=stats.entities,
            relationships=stats.relationships,
            mentions=stats.mentions,
            merges=resolution.merges,
            metadata={**meta, "embedding_model": self.embedder.name, "extractor": extractor.name},
            timings_ms=timings,
        )
        logger.info("ingestion_completed", extra={"document_id": document_id, **result.stats()})
        return result

    @staticmethod
    def _embedding_text(record: dict[str, Any]) -> str:
        # Prepend lightweight context so chunks embed with their document/section semantics.
        prefix = " / ".join(str(x) for x in (record.get("document_title"), record.get("section")) if x)
        return f"{prefix}\n{record['text']}" if prefix else record["text"]
