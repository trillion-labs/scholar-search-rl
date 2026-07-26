import dataclasses
import logging

from pymilvus import Collection, CollectionSchema, DataType, FieldSchema, connections, utility

log = logging.getLogger(__name__)

PAPERS_COLLECTION = "papers"
CHUNKS_COLLECTION = "chunks"
DIM = 1024
HNSW_INDEX = {
    "index_type": "HNSW",
    "metric_type": "IP",
    "params": {"M": 32, "efConstruction": 256},
}
SPARSE_INDEX = {
    "index_type": "SPARSE_INVERTED_INDEX",
    "metric_type": "IP",
    "params": {},
}


@dataclasses.dataclass(frozen=True)
class MilvusEndpoint:
    uri: str = "http://localhost:19530"
    alias: str = "default"


def connect(endpoint: MilvusEndpoint) -> None:
    if endpoint.alias in connections.list_connections():
        connections.disconnect(endpoint.alias)
    host_port = endpoint.uri.removeprefix("http://").removeprefix("https://")
    host, _, port = host_port.partition(":")
    connections.connect(alias=endpoint.alias, host=host, port=port or "19530")
    log.info("connected to milvus @ %s (alias=%s)", endpoint.uri, endpoint.alias)


def papers_schema() -> CollectionSchema:
    return CollectionSchema(
        fields=[
            FieldSchema("corpus_id", DataType.INT64, is_primary=True, auto_id=False),
            FieldSchema("embedding", DataType.FLOAT_VECTOR, dim=DIM),
            FieldSchema("sparse_embedding", DataType.SPARSE_FLOAT_VECTOR),
            FieldSchema("year", DataType.INT32),
            FieldSchema("venue", DataType.VARCHAR, max_length=512),
            FieldSchema("citationcount", DataType.INT32),
            FieldSchema("classification", DataType.VARCHAR, max_length=256),
        ],
        description="paper-level dense + sparse embeddings with scalar filters",
    )


def chunks_schema() -> CollectionSchema:
    return CollectionSchema(
        fields=[
            FieldSchema("chunk_id", DataType.INT64, is_primary=True, auto_id=False),
            FieldSchema("paper_corpus_id", DataType.INT64),
            FieldSchema("section_idx", DataType.INT32),
            FieldSchema("embedding", DataType.FLOAT_VECTOR, dim=DIM),
        ],
        description="chunk-level dense embeddings",
    )


def ensure_collection(name: str, schema: CollectionSchema) -> Collection:
    if utility.has_collection(name):
        return Collection(name=name)
    coll = Collection(name=name, schema=schema)
    log.info("created collection %s", name)
    return coll


def build_indexes(coll: Collection, fields: dict[str, dict]) -> None:
    existing = {idx.field_name for idx in coll.indexes}
    for field, params in fields.items():
        if field in existing:
            log.info("index on %s.%s already exists", coll.name, field)
            continue
        log.info("building %s index on %s.%s ...", params["index_type"], coll.name, field)
        coll.create_index(field_name=field, index_params=params)
    coll.flush()
    log.info("indexes on %s done", coll.name)


def load(coll: Collection) -> None:
    coll.load()
    log.info("loaded %s into memory (num_entities=%d)", coll.name, coll.num_entities)
