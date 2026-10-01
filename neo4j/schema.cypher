// Agentic GraphRAG - Neo4j schema (applied automatically at backend startup, idempotent).
// Vector dimension must equal EMBEDDING_DIMENSIONS (1536 for text-embedding-3-small).

CREATE CONSTRAINT entity_id IF NOT EXISTS FOR (e:Entity) REQUIRE e.id IS UNIQUE;
CREATE CONSTRAINT chunk_id IF NOT EXISTS FOR (c:Chunk) REQUIRE c.id IS UNIQUE;
CREATE CONSTRAINT document_id IF NOT EXISTS FOR (d:Document) REQUIRE d.id IS UNIQUE;

// Tenant-scoped lookup indexes: every query filters on tenant_id first.
CREATE INDEX entity_tenant IF NOT EXISTS FOR (e:Entity) ON (e.tenant_id);
CREATE INDEX entity_tenant_norm IF NOT EXISTS FOR (e:Entity) ON (e.tenant_id, e.normalized_name);
CREATE INDEX entity_tenant_type IF NOT EXISTS FOR (e:Entity) ON (e.tenant_id, e.type);
CREATE INDEX chunk_tenant IF NOT EXISTS FOR (c:Chunk) ON (c.tenant_id);
CREATE INDEX chunk_tenant_document IF NOT EXISTS FOR (c:Chunk) ON (c.tenant_id, c.document_id);
CREATE INDEX document_tenant IF NOT EXISTS FOR (d:Document) ON (d.tenant_id);

// HNSW vector index over chunk embeddings.
CREATE VECTOR INDEX chunk_embedding_index IF NOT EXISTS
FOR (c:Chunk) ON (c.embedding)
OPTIONS {indexConfig: {`vector.dimensions`: 1536, `vector.similarity_function`: 'cosine'}};

// Full-text (BM25) indexes for keyword retrieval and entity linking.
CREATE FULLTEXT INDEX chunk_text_fulltext IF NOT EXISTS FOR (c:Chunk) ON EACH [c.text];
CREATE FULLTEXT INDEX entity_name_fulltext IF NOT EXISTS FOR (e:Entity) ON EACH [e.name, e.aliases_text];
