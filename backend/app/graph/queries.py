"""Parameterised Cypher used by the repository.

Every query is tenant-scoped: nodes are matched with ``tenant_id: $tenant_id``
and relationship types/labels are only ever interpolated from whitelists.
"""

from __future__ import annotations

# ------------------------------------------------------------------ writes
UPSERT_DOCUMENT = """
MERGE (d:Document {id: $document_id})
ON CREATE SET d.tenant_id = $tenant_id, d.created_at = datetime()
WITH d WHERE d.tenant_id = $tenant_id
SET d.filename = $filename, d.title = $title, d.file_type = $file_type, d.updated_at = datetime()
RETURN d.id AS id
"""

DELETE_DOCUMENT_CHUNKS = """
MATCH (c:Chunk {tenant_id: $tenant_id, document_id: $document_id})
DETACH DELETE c
"""

WRITE_CHUNKS = """
MATCH (d:Document {id: $document_id, tenant_id: $tenant_id})
UNWIND $chunks AS row
MERGE (c:Chunk {id: row.id})
ON CREATE SET c.tenant_id = $tenant_id
WITH d, c, row WHERE c.tenant_id = $tenant_id
SET c.document_id = $document_id, c.chunk_index = row.chunk_index, c.text = row.text,
    c.token_count = row.token_count, c.page_number = row.page_number, c.page_end = row.page_end,
    c.section = row.section, c.source_filename = row.source_filename,
    c.document_title = row.document_title
MERGE (d)-[:CONTAINS]->(c)
RETURN count(c) AS written
"""

SET_CHUNK_EMBEDDINGS = """
UNWIND $rows AS row
MATCH (c:Chunk {id: row.id, tenant_id: $tenant_id})
CALL db.create.setNodeVectorProperty(c, 'embedding', row.embedding)
RETURN count(c) AS written
"""

UPSERT_ENTITIES = """
UNWIND $entities AS row
MERGE (e:Entity {id: row.id})
ON CREATE SET e.tenant_id = $tenant_id, e.name = row.name, e.type = row.type,
              e.normalized_name = row.normalized_name, e.description = row.description,
              e.aliases = row.aliases, e.document_ids = [row.document_id], e.created_at = datetime()
ON MATCH SET e.description = CASE WHEN coalesce(e.description, '') = '' THEN row.description ELSE e.description END,
             e.aliases = reduce(acc = coalesce(e.aliases, []), a IN row.aliases |
                                CASE WHEN a IN acc THEN acc ELSE acc + a END)[..25],
             e.document_ids = CASE WHEN row.document_id IN coalesce(e.document_ids, [])
                                   THEN e.document_ids ELSE coalesce(e.document_ids, []) + row.document_id END
WITH e, row WHERE e.tenant_id = $tenant_id
SET e.aliases_text = reduce(s = '', a IN coalesce(e.aliases, []) | s + ' ' + a), e.updated_at = datetime()
SET e:{label}
RETURN count(e) AS written
"""

WRITE_MENTIONS = """
UNWIND $mentions AS row
MATCH (c:Chunk {id: row.chunk_id, tenant_id: $tenant_id})
MATCH (e:Entity {id: row.entity_id, tenant_id: $tenant_id})
MERGE (c)-[:MENTIONS]->(e)
RETURN count(*) AS written
"""

UPSERT_RELATIONSHIPS = """
UNWIND $rels AS row
MATCH (s:Entity {id: row.source_id, tenant_id: $tenant_id})
MATCH (t:Entity {id: row.target_id, tenant_id: $tenant_id})
MERGE (s)-[r:{rel_type}]->(t)
ON CREATE SET r.tenant_id = $tenant_id, r.chunk_ids = [], r.document_ids = [], r.evidence = row.evidence,
              r.created_at = datetime()
SET r.chunk_ids = reduce(acc = coalesce(r.chunk_ids, []), c IN row.chunk_ids |
                         CASE WHEN c IN acc THEN acc ELSE acc + c END),
    r.document_ids = CASE WHEN $document_id IN r.document_ids THEN r.document_ids
                          ELSE r.document_ids + $document_id END,
    r.evidence = coalesce(r.evidence, row.evidence)
RETURN count(r) AS written
"""

DELETE_DOCUMENT = """
MATCH (d:Document {id: $document_id, tenant_id: $tenant_id})
OPTIONAL MATCH (d)-[:CONTAINS]->(c:Chunk)
DETACH DELETE c, d
"""

PRUNE_DOCUMENT_RELATIONSHIPS = """
MATCH (s:Entity {tenant_id: $tenant_id})-[r]->(t:Entity {tenant_id: $tenant_id})
WHERE $document_id IN r.document_ids
SET r.document_ids = [x IN r.document_ids WHERE x <> $document_id],
    r.chunk_ids = [x IN r.chunk_ids WHERE NOT x STARTS WITH $chunk_prefix]
WITH r WHERE size(r.document_ids) = 0
DELETE r
"""

PRUNE_DOCUMENT_ENTITIES = """
MATCH (e:Entity {tenant_id: $tenant_id})
WHERE $document_id IN e.document_ids
SET e.document_ids = [x IN e.document_ids WHERE x <> $document_id]
WITH e WHERE size(e.document_ids) = 0
DETACH DELETE e
"""

FIND_ENTITIES_BY_KEYS = """
MATCH (e:Entity {tenant_id: $tenant_id})
WHERE e.normalized_name IN $keys
RETURN e.id AS id, e.name AS name, e.type AS type, e.normalized_name AS normalized_name,
       e.description AS description
"""

FIND_ENTITY_CANDIDATES = """
MATCH (e:Entity {tenant_id: $tenant_id, type: $type})
WHERE any(tok IN $tokens WHERE e.normalized_name CONTAINS tok)
RETURN e.id AS id, e.name AS name, e.type AS type, e.normalized_name AS normalized_name,
       e.description AS description
LIMIT $limit
"""

# ------------------------------------------------------------------- reads
_CHUNK_RETURN = """
RETURN node.id AS chunk_id, node.document_id AS document_id, node.text AS text, score,
       node.source_filename AS source_filename, node.page_number AS page_number,
       node.section AS section, node.chunk_index AS chunk_index, node.document_title AS document_title
"""

CHUNK_FILTERS = """
  ($document_ids IS NULL OR node.document_id IN $document_ids)
  AND ($filenames IS NULL OR node.source_filename IN $filenames)
  AND ($page_from IS NULL OR node.page_number >= $page_from)
  AND ($page_to IS NULL OR node.page_number <= $page_to)
  AND ($section IS NULL OR toLower(coalesce(node.section, '')) CONTAINS toLower($section))
"""

COUNT_TENANT_CHUNKS = "MATCH (c:Chunk {tenant_id: $tenant_id}) RETURN count(c) AS n"

# Exact k-NN restricted to the tenant (used while a tenant's corpus is small).
VECTOR_SEARCH_EXACT = (
    """
MATCH (node:Chunk {tenant_id: $tenant_id})
WHERE node.embedding IS NOT NULL AND """
    + CHUNK_FILTERS
    + """
WITH node, vector.similarity.cosine(node.embedding, $embedding) AS score
ORDER BY score DESC LIMIT $top_k
"""
    + _CHUNK_RETURN
)

# Approximate k-NN through the HNSW index, over-fetching then filtering by tenant.
VECTOR_SEARCH_ANN = (
    """
CALL db.index.vector.queryNodes($index_name, $candidates, $embedding) YIELD node, score
WITH node, score WHERE node.tenant_id = $tenant_id AND """
    + CHUNK_FILTERS
    + """
WITH node, score ORDER BY score DESC LIMIT $top_k
"""
    + _CHUNK_RETURN
)

FULLTEXT_CHUNKS = (
    """
CALL db.index.fulltext.queryNodes($index_name, $query, {limit: $candidates}) YIELD node, score
WITH node, score WHERE node.tenant_id = $tenant_id AND """
    + CHUNK_FILTERS
    + """
WITH node, score ORDER BY score DESC LIMIT $top_k
"""
    + _CHUNK_RETURN
)

CHUNKS_BY_IDS = """
MATCH (node:Chunk {tenant_id: $tenant_id}) WHERE node.id IN $chunk_ids
WITH node, 1.0 AS score
""" + _CHUNK_RETURN

CHUNKS_MENTIONING = """
MATCH (node:Chunk {tenant_id: $tenant_id})-[:MENTIONS]->(e:Entity {tenant_id: $tenant_id})
WHERE e.id IN $entity_ids
WITH node, count(DISTINCT e) AS hits
WITH node, toFloat(hits) / $n_entities AS score
ORDER BY score DESC, node.chunk_index ASC LIMIT $top_k
""" + _CHUNK_RETURN

LINK_ENTITIES_EXACT = """
UNWIND $names AS q
MATCH (e:Entity {tenant_id: $tenant_id})
WHERE e.normalized_name = q.key OR toLower(e.name) = q.lower OR q.lower IN [a IN coalesce(e.aliases, []) | toLower(a)]
RETURN q.raw AS query, e.id AS id, e.name AS name, e.type AS type, e.description AS description, 1.0 AS score
"""

LINK_ENTITIES_FUZZY = """
UNWIND $names AS q
MATCH (e:Entity {tenant_id: $tenant_id})
WHERE any(tok IN q.tokens WHERE e.normalized_name CONTAINS tok)
WITH q, e, size([tok IN q.tokens WHERE e.normalized_name CONTAINS tok]) AS hit
RETURN q.raw AS query, e.id AS id, e.name AS name, e.type AS type, e.description AS description,
       toFloat(hit) / size(q.tokens) AS score
ORDER BY score DESC LIMIT $limit
"""

ENTITIES_IN_TEXT = """
MATCH (e:Entity {tenant_id: $tenant_id})
WHERE size(e.normalized_name) > 1 AND (
      (' ' + $text + ' ') CONTAINS (' ' + e.normalized_name + ' ')
   OR any(a IN coalesce(e.aliases, []) WHERE size(a) > 2 AND (' ' + $text + ' ') CONTAINS (' ' + toLower(a) + ' ')))
RETURN e.id AS id, e.name AS name, e.type AS type, e.description AS description, 1.0 AS score
LIMIT 25
"""

# 1-hop facts around linked entities (both directions), optional relationship filter.
NEIGHBORHOOD = """
MATCH (s:Entity {tenant_id: $tenant_id})-[r]->(t:Entity {tenant_id: $tenant_id})
WHERE (s.id IN $entity_ids OR t.id IN $entity_ids)
  AND ($rel_types IS NULL OR type(r) IN $rel_types)
RETURN s.name AS source, s.type AS source_type, type(r) AS relationship, t.name AS target,
       t.type AS target_type, r.evidence AS evidence, r.chunk_ids AS chunk_ids,
       r.document_ids AS document_ids, s.id AS source_id, t.id AS target_id, 1 AS hops
LIMIT $limit
"""

# Entities connected to *all* anchor entities (e.g. a project managed by Rahul that uses Kafka).
COMMON_NEIGHBORS = """
UNWIND $anchors AS anchor
MATCH (a:Entity {id: anchor.id, tenant_id: $tenant_id})-[r]-(m:Entity {tenant_id: $tenant_id})
WHERE NOT m.id IN $entity_ids AND (anchor.rels IS NULL OR type(r) IN anchor.rels)
WITH m, count(DISTINCT a) AS anchors
WHERE anchors >= $min_anchors
RETURN m.id AS id, m.name AS name, m.type AS type, anchors
ORDER BY anchors DESC LIMIT $limit
"""

# Multi-hop paths between anchor pairs, bounded length, tenant-checked on every node.
PATHS_BETWEEN = """
MATCH (a:Entity {tenant_id: $tenant_id}), (b:Entity {tenant_id: $tenant_id})
WHERE a.id IN $entity_ids AND b.id IN $entity_ids AND a.id < b.id
MATCH p = allShortestPaths((a)-[*..{max_hops}]-(b))
WHERE all(n IN nodes(p) WHERE n.tenant_id = $tenant_id AND n:Entity)
WITH p LIMIT $limit
UNWIND relationships(p) AS r
WITH DISTINCT r, length(p) AS hops
WITH startNode(r) AS s, r, endNode(r) AS t, hops
RETURN s.name AS source, s.type AS source_type, type(r) AS relationship, t.name AS target,
       t.type AS target_type, r.evidence AS evidence, r.chunk_ids AS chunk_ids,
       r.document_ids AS document_ids, s.id AS source_id, t.id AS target_id, hops
"""

# ------------------------------------------------------------ graph explorer
GRAPH_STATS = """
CALL () { MATCH (e:Entity {tenant_id: $tenant_id}) RETURN count(e) AS entities }
CALL () { MATCH (:Entity {tenant_id: $tenant_id})-[r]->(:Entity {tenant_id: $tenant_id}) RETURN count(r) AS relationships }
CALL () { MATCH (c:Chunk {tenant_id: $tenant_id}) RETURN count(c) AS chunks }
CALL () { MATCH (d:Document {tenant_id: $tenant_id}) RETURN count(d) AS documents }
RETURN entities, relationships, chunks, documents
"""

ENTITIES_BY_TYPE = """
MATCH (e:Entity {tenant_id: $tenant_id}) RETURN e.type AS type, count(*) AS n
"""

RELATIONSHIPS_BY_TYPE = """
MATCH (:Entity {tenant_id: $tenant_id})-[r]->(:Entity {tenant_id: $tenant_id}) RETURN type(r) AS type, count(*) AS n
"""

SEARCH_ENTITIES = """
MATCH (e:Entity {tenant_id: $tenant_id})
WHERE ($q IS NULL OR toLower(e.name) CONTAINS toLower($q) OR e.normalized_name CONTAINS toLower($q))
  AND ($types IS NULL OR e.type IN $types)
OPTIONAL MATCH (e)-[r]-(:Entity {tenant_id: $tenant_id})
WITH e, count(r) AS degree
RETURN e.id AS id, e.name AS name, e.type AS type, e.description AS description,
       coalesce(e.aliases, []) AS aliases, degree, coalesce(e.document_ids, []) AS document_ids
ORDER BY degree DESC, e.name ASC SKIP $offset LIMIT $limit
"""

ENTITY_BY_ID = """
MATCH (e:Entity {id: $entity_id, tenant_id: $tenant_id})
OPTIONAL MATCH (e)-[r]-(:Entity {tenant_id: $tenant_id})
WITH e, count(r) AS degree
RETURN e.id AS id, e.name AS name, e.type AS type, e.description AS description,
       coalesce(e.aliases, []) AS aliases, degree, coalesce(e.document_ids, []) AS document_ids
"""

ENTITY_SOURCES = """
MATCH (c:Chunk {tenant_id: $tenant_id})-[:MENTIONS]->(e:Entity {id: $entity_id, tenant_id: $tenant_id})
RETURN c.id AS chunk_id, c.document_id AS document_id, c.source_filename AS source_filename,
       c.page_number AS page_number, c.section AS section, left(c.text, 300) AS snippet
ORDER BY c.source_filename, c.chunk_index LIMIT $limit
"""

SUBGRAPH = """
MATCH (s:Entity {tenant_id: $tenant_id})-[r]->(t:Entity {tenant_id: $tenant_id})
WHERE ($entity_id IS NULL OR s.id = $entity_id OR t.id = $entity_id)
  AND ($types IS NULL OR (s.type IN $types AND t.type IN $types))
RETURN s.id AS source_id, s.name AS source, s.type AS source_type, type(r) AS relationship,
       t.id AS target_id, t.name AS target, t.type AS target_type, r.evidence AS evidence,
       r.chunk_ids AS chunk_ids, r.document_ids AS document_ids
LIMIT $limit
"""

EXPAND_SUBGRAPH = """
MATCH (center:Entity {id: $entity_id, tenant_id: $tenant_id})
MATCH p = (center)-[*1..{depth}]-(n:Entity {tenant_id: $tenant_id})
WHERE all(x IN nodes(p) WHERE x.tenant_id = $tenant_id)
WITH p LIMIT $limit
UNWIND relationships(p) AS r
WITH DISTINCT r
WITH startNode(r) AS s, r, endNode(r) AS t
RETURN s.id AS source_id, s.name AS source, s.type AS source_type, type(r) AS relationship,
       t.id AS target_id, t.name AS target, t.type AS target_type, r.evidence AS evidence,
       r.chunk_ids AS chunk_ids, r.document_ids AS document_ids
"""
