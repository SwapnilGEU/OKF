"""Qdrant Cloud: chunk search (okf_chunks) and topic-page search (okf_topics).

Same query as 01/07: dense (MiniLM) + sparse (BM25) prefetch, then a ColBERT rerank, all with
Qdrant's cloud inference (the models run in Qdrant, nothing to download here).

If Qdrant can't be reached, `ok` is False and the pipeline falls back to keyword search + links.
"""

from __future__ import annotations

import hashlib
import uuid

from .bundle import Bundle
from .config import Settings


class VectorStore:
    def __init__(self, settings: Settings):
        self.s = settings
        self.client = None
        self.ok = False
        self.topics_ok = False
        self.error = None
        if not settings.use_vectors:
            self.error = "use_vectors is off"
            return
        if not settings.qdrant_url:
            self.error = "QDRANT_URL is not set"
            return
        try:
            from qdrant_client import QdrantClient
            self.client = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key or None,
                                       cloud_inference=True, timeout=60)
            self.client.count(settings.chunk_collection, exact=False)
            self.ok = True
            self.topics_ok = self.client.collection_exists(settings.topic_collection)
        except Exception as exc:                      # no network, wrong key, collection missing...
            self.error = f"{type(exc).__name__}: {exc}"[:300]

    def search(self, collection: str, query: str, top_k: int, id_field: str) -> list[str]:
        from qdrant_client import models
        s = self.s
        result = self.client.query_points(
            collection,
            prefetch=[models.Prefetch(query=models.Document(text=query, model=s.dense_model), using="dense", limit=s.prefetch_limit),
                      models.Prefetch(query=models.Document(text=query, model=s.sparse_model), using="sparse", limit=s.prefetch_limit)],
            query=models.Document(text=query, model=s.late_interaction_model),
            using="multi", limit=top_k, with_payload=[id_field],
        )
        return [p.payload[id_field] for p in result.points]

    def search_chunks(self, query: str, top_k: int) -> list[str]:
        return self.search(self.s.chunk_collection, query, top_k, "chunk_id")

    def search_topics(self, query: str, top_k: int) -> list[str]:
        return self.search(self.s.topic_collection, query, top_k, "concept_id")

    # --- maintenance: keep okf_topics in sync with the bundle (what 07 step 4 did) --------------
    def sync_topics(self, bundle: Bundle) -> dict:
        """Create okf_topics if needed, upload changed topic pages, delete removed ones."""
        from qdrant_client import models
        s, client = self.s, self.client
        if not client.collection_exists(s.topic_collection):
            client.create_collection(
                s.topic_collection,
                vectors_config={
                    "dense": models.VectorParams(size=384, distance=models.Distance.COSINE),
                    "multi": models.VectorParams(size=96, distance=models.Distance.COSINE,
                                                 multivector_config=models.MultiVectorConfig(comparator=models.MultiVectorComparator.MAX_SIM),
                                                 hnsw_config=models.HnswConfigDiff(m=0)),
                },
                sparse_vectors_config={"sparse": models.SparseVectorParams(modifier=models.Modifier.IDF)},
            )
        stored, offset = {}, None
        while True:
            points, offset = client.scroll(s.topic_collection, limit=256, offset=offset,
                                           with_payload=["content_hash"], with_vectors=False)
            stored.update({str(p.id): (p.payload or {}).get("content_hash") for p in points})
            if offset is None:
                break

        def point_id(cid):
            return str(uuid.uuid5(uuid.NAMESPACE_URL, f"okf_topic:{cid}"))

        def digest(cid):
            return hashlib.sha256(bundle.topic_text(cid).encode()).hexdigest()

        wanted = {point_id(cid): cid for cid in bundle.wiki_ids}
        todo = [cid for pid, cid in wanted.items() if stored.get(pid) != digest(cid)]
        stale = [pid for pid in stored if pid not in wanted]
        if stale:
            client.delete(s.topic_collection, points_selector=models.PointIdsList(points=stale))
        for i in range(0, len(todo), 32):
            client.upsert(s.topic_collection, wait=True, points=[
                models.PointStruct(
                    id=point_id(cid),
                    vector={"dense": models.Document(text=bundle.topic_text(cid), model=s.dense_model),
                            "sparse": models.Document(text=bundle.topic_text(cid), model=s.sparse_model),
                            "multi": models.Document(text=bundle.topic_text(cid), model=s.late_interaction_model)},
                    payload={"concept_id": cid, "title": bundle[cid].title, "type": bundle[cid].type,
                             "content_hash": digest(cid)},
                ) for cid in todo[i:i + 32]])
        self.topics_ok = True
        return {"uploaded": len(todo), "unchanged": len(wanted) - len(todo), "deleted": len(stale)}
