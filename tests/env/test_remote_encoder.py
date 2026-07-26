import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from s2cs.env.encoder import BatchedEncoder, OpenAIEmbeddings, RemoteBGEM3


class _FakeEmbedServer:
    """Stdlib HTTP server speaking the embed_server.py /encode protocol.

    For each text it returns a dense vector filled with float(len(text)) and a
    sparse weight {str(len(text)): 1.0}, and records every text it received (per
    server instance) so tests can assert routing and order preservation.
    """

    def __init__(self, dim: int = 4):
        self.dim = dim
        self.received: list[list[str]] = []  # one entry per /encode request
        self.lock = threading.Lock()
        server = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_a):
                pass

            def do_GET(self):
                if self.path == "/health":
                    self.send_response(200)
                    self.end_headers()
                    self.wfile.write(b"ok")
                else:
                    self.send_response(404)
                    self.end_headers()

            def do_POST(self):
                if self.path != "/encode":
                    self.send_response(404)
                    self.end_headers()
                    return
                n = int(self.headers.get("Content-Length", 0))
                req = json.loads(self.rfile.read(n))
                texts = list(req["texts"])
                with server.lock:
                    server.received.append(texts)
                result = {"dense": [[float(len(t))] * server.dim for t in texts]}
                if req.get("return_sparse", False):
                    result["sparse"] = [{str(len(t)): 1.0} for t in texts]
                else:
                    result["sparse"] = []
                body = json.dumps(result).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        self._httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self._httpd.server_address[1]}"
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        self._thread.start()

    @property
    def total_texts(self) -> int:
        with self.lock:
            return sum(len(t) for t in self.received)

    def close(self):
        self._httpd.shutdown()


@pytest.fixture
def servers():
    srvs = [_FakeEmbedServer(dim=4), _FakeEmbedServer(dim=4)]
    yield srvs
    for s in srvs:
        s.close()


def test_hybrid_encode_shape_matches_bgem3(servers):
    enc = RemoteBGEM3([servers[0].url])
    out = enc.encode(["abc"], return_dense=True, return_sparse=True, return_colbert_vecs=False)
    assert out["dense_vecs"][0].tolist() == [3.0, 3.0, 3.0, 3.0]
    assert out["lexical_weights"][0] == {"3": 1.0}


def test_dense_only_encode(servers):
    enc = RemoteBGEM3([servers[0].url])
    out = enc.encode(["abcd"], return_dense=True, return_sparse=False, return_colbert_vecs=False)
    assert out["dense_vecs"][0].tolist() == [4.0, 4.0, 4.0, 4.0]


def test_batch_split_across_servers_preserves_order(servers):
    enc = RemoteBGEM3([servers[0].url, servers[1].url])
    texts = ["a", "bb", "ccc", "dddd"]  # distinct lengths → identifiable
    out = enc.encode(texts, return_dense=True, return_sparse=True, return_colbert_vecs=False)
    assert [v.tolist()[0] for v in out["dense_vecs"]] == [1.0, 2.0, 3.0, 4.0]
    assert [d for d in out["lexical_weights"]] == [{"1": 1.0}, {"2": 1.0}, {"3": 1.0}, {"4": 1.0}]
    # both servers actually shared the batch
    assert servers[0].total_texts >= 1 and servers[1].total_texts >= 1


def test_single_text_calls_round_robin_across_servers(servers):
    enc = RemoteBGEM3([servers[0].url, servers[1].url])
    for t in ["a", "bb", "ccc", "dddd"]:
        enc.encode([t], return_dense=True, return_sparse=False, return_colbert_vecs=False)
    assert servers[0].total_texts == 2 and servers[1].total_texts == 2


def test_plugs_into_batched_encoder(servers):
    enc = BatchedEncoder(RemoteBGEM3([servers[0].url, servers[1].url]), max_batch=8, wait_ms=5)
    dense, sparse = enc.encode_hybrid("hello")  # len 5
    assert dense == [5.0, 5.0, 5.0, 5.0]
    assert sparse == {5: 1.0}


class _FakeOpenAIEmbedServer:
    """Stdlib server speaking the OpenAI /v1/embeddings protocol.

    Each text -> embedding [float(len(text))] * dim. Returns items in REVERSED
    order with explicit "index" fields, so a correct client must re-sort by index.
    """

    def __init__(self, dim: int = 4):
        self.dim = dim
        server = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_a):
                pass

            def do_POST(self):
                if self.path != "/embeddings":
                    self.send_response(404)
                    self.end_headers()
                    return
                n = int(self.headers.get("Content-Length", 0))
                texts = list(json.loads(self.rfile.read(n))["input"])
                data = [
                    {"object": "embedding", "index": i, "embedding": [float(len(t))] * server.dim}
                    for i, t in enumerate(texts)
                ]
                body = json.dumps({"object": "list", "data": list(reversed(data))}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        self._httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self._httpd.server_address[1]}"
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        self._thread.start()

    def close(self):
        self._httpd.shutdown()


@pytest.fixture
def openai_server():
    s = _FakeOpenAIEmbedServer(dim=4)
    yield s
    s.close()


def test_openai_embeddings_dense_and_empty_sparse(openai_server):
    enc = OpenAIEmbeddings(openai_server.url, "baai/bge-m3", "test-key")
    out = enc.encode(["abc"], return_dense=True, return_sparse=True)
    assert out["dense_vecs"][0].tolist() == [3.0, 3.0, 3.0, 3.0]
    assert out["lexical_weights"] == [{}]  # no learned-sparse over HTTP


def test_openai_embeddings_restores_input_order(openai_server):
    # Server replies reversed; the client must re-sort by "index".
    enc = OpenAIEmbeddings(openai_server.url, "baai/bge-m3", "test-key")
    out = enc.encode(["a", "bb", "ccc", "dddd"])
    assert [v.tolist()[0] for v in out["dense_vecs"]] == [1.0, 2.0, 3.0, 4.0]


def test_openai_embeddings_dense_only_through_batched_encoder(openai_server):
    enc = BatchedEncoder(OpenAIEmbeddings(openai_server.url, "baai/bge-m3", "test-key"), max_batch=8, wait_ms=5)
    dense, sparse = enc.encode_hybrid("hello")  # len 5
    assert dense == [5.0, 5.0, 5.0, 5.0]
    assert sparse == {}  # empty -> search_papers takes the dense-only path
