"""Create a demo tenant/user and ingest the sample corpus through the real pipeline.

    python -m app.scripts.seed_demo            (inside the backend container: `docker compose exec backend python -m app.scripts.seed_demo`)

Idempotent: re-running skips documents that were already ingested for the demo tenant.
"""

from __future__ import annotations

import hashlib
import os
import uuid
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy import select

from app.core.config import get_settings
from app.core.security import hash_password
from app.db.postgres import sync_session_scope
from app.db.redis import invalidate_tenant_cache_sync
from app.ingestion.loader import FileStorage, validate_upload
from app.models.document import Document, DocumentStatus
from app.models.job import IngestionJob, JobStage, JobStatus
from app.models.tenant import Tenant
from app.models.user import User
from app.workers.tasks import build_pipeline

DEMO_EMAIL = os.getenv("DEMO_EMAIL", "demo@techcorp.com")
DEMO_PASSWORD = os.getenv("DEMO_PASSWORD", "DemoPassw0rd")
SAMPLES = Path(__file__).resolve().parents[2] / "data" / "samples"


def main() -> None:
    settings = get_settings()
    with sync_session_scope() as db:
        user = db.execute(select(User).where(User.email == DEMO_EMAIL)).scalar_one_or_none()
        if user is None:
            tenant = Tenant(name="TechCorp Demo", slug=f"techcorp-demo-{uuid.uuid4().hex[:6]}")
            db.add(tenant)
            db.flush()
            user = User(tenant_id=tenant.id, email=DEMO_EMAIL, full_name="Demo User",
                        password_hash=hash_password(DEMO_PASSWORD), role="admin")
            db.add(user)
            db.flush()
            print(f"Created demo user {DEMO_EMAIL}")
        tenant_id, user_id = user.tenant_id, user.id
    storage = FileStorage(settings.upload_dir)
    pipeline = build_pipeline()
    for path in sorted(SAMPLES.iterdir()):
        data = path.read_bytes()
        checksum = hashlib.sha256(data).hexdigest()
        with sync_session_scope() as db:
            if db.execute(select(Document.id).where(Document.tenant_id == tenant_id, Document.checksum == checksum)).first():
                print(f"skip {path.name} (already ingested)")
                continue
        validated = validate_upload(path.name, None, data, settings.max_upload_bytes)
        doc_id = uuid.uuid4()
        stored = storage.save(str(tenant_id), str(doc_id), validated.file_type, validated.data)
        result = pipeline.run(tenant_id=str(tenant_id), document_id=str(doc_id), filename=validated.filename,
                              file_type=validated.file_type, storage_path=stored)
        now = datetime.now(UTC)
        with sync_session_scope() as db:
            db.add(Document(id=doc_id, tenant_id=tenant_id, uploaded_by=user_id, filename=validated.filename,
                            file_type=validated.file_type, size_bytes=validated.size, checksum=validated.checksum,
                            storage_path=stored, status=DocumentStatus.COMPLETED, title=result.title,
                            page_count=result.page_count, chunk_count=result.chunks, entity_count=result.entities,
                            relationship_count=result.relationships, metadata_=result.metadata))
            db.flush()
            db.add(IngestionJob(tenant_id=tenant_id, document_id=doc_id, status=JobStatus.COMPLETED,
                                stage=JobStage.COMPLETED, progress=100, attempts=1, stats=result.stats(),
                                stage_history=[], started_at=now, finished_at=now))
        print(f"ingested {path.name}: {result.chunks} chunks, {result.entities} entities, "
              f"{result.relationships} relationships")
    invalidate_tenant_cache_sync(settings.redis_url, str(tenant_id))
    print(f"Demo ready - log in as {DEMO_EMAIL} / {DEMO_PASSWORD}")


if __name__ == "__main__":
    main()
