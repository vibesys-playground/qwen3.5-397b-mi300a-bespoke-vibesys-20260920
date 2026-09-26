"""Hermetic CPU tests for `/debug/fault`'s request handling (no HTTP server, no model).

`debug_fault` is exercised directly against a fake `aiohttp.web.Request`-shaped object (only
`.json()` is called on it), following `test_score_endpoint.py`'s pattern of testing the
handler as a pure function of its inputs rather than booting a real server. `make_app`'s
env-var gating (`SEED_FAULT_ENDPOINT=1` required to register the route at all) is not
re-tested here: it is a one-line `if` around `add_routes`, and route registration is not
something a unit test on `debug_fault` itself can observe.

    /tmp/torchenv/bin/python -m pytest \\
        examples/model-serving/qwen3.5-397b-a17b-mi300a-bespoke/seed_tests/test_debug_fault_endpoint.py
"""

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import model as seed_model  # noqa: E402

import server  # noqa: E402


class _FakeRequest:
    def __init__(self, body: dict) -> None:
        self._body = body

    async def json(self) -> dict:
        return self._body


def _call(body: dict) -> tuple[int, dict]:
    import asyncio

    response = asyncio.run(server.debug_fault(_FakeRequest(body)))
    return response.status, json.loads(response.body.decode())


@pytest.fixture(autouse=True)
def _clean_flags():
    yield
    for _name, (attr, kind) in server.FAULT_FLAGS.items():
        if kind is bool:
            setattr(seed_model, attr, False)
    seed_model.FAULT_FLIP_CAUSAL_MASK_LAYER = 0


def test_flip_a_bool_fault_on_and_off() -> None:
    status, payload = _call({"name": "drop_expert", "value": True})
    assert status == 200
    assert payload["faults"]["drop_expert"] is True
    assert seed_model.FAULT_DROP_EXPERT is True

    status, payload = _call({"name": "drop_expert", "value": False})
    assert status == 200
    assert payload["faults"]["drop_expert"] is False
    assert seed_model.FAULT_DROP_EXPERT is False


def test_set_causal_mask_layer_int() -> None:
    status, payload = _call({"name": "flip_causal_mask_layer", "value": 3})
    assert status == 200
    assert payload["faults"]["flip_causal_mask_layer"] == 3
    assert seed_model.FAULT_FLIP_CAUSAL_MASK_LAYER == 3


def test_reset_all_clears_every_boolean_flag() -> None:
    _call({"name": "rope_base", "value": True})
    _call({"name": "skip_deltanet_gate", "value": True})
    status, payload = _call({"name": "all", "value": False})
    assert status == 200
    assert all(v is False for k, v in payload["faults"].items() if k != "flip_causal_mask_layer")
    assert seed_model.FAULT_ROPE_BASE is False
    assert seed_model.FAULT_SKIP_DELTANET_GATE is False


@pytest.mark.parametrize(
    ("body", "message_fragment"),
    [
        ({"name": "not_a_fault", "value": True}, "unknown fault"),
        ({"name": "drop_expert", "value": "yes"}, "takes a bool"),
        ({"name": "flip_causal_mask_layer", "value": True}, "takes an integer"),
        ({"name": "flip_causal_mask_layer", "value": 1.5}, "takes an integer"),
        ({"name": "all", "value": True}, "only supports value=false"),
    ],
)
def test_rejects_invalid_requests(body: dict, message_fragment: str) -> None:
    status, payload = _call(body)
    assert status == 400
    assert message_fragment in payload["error"]
