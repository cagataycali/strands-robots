#!/usr/bin/env python3
"""
GR00T Client Implementation

Provides ZMQ-based client for communicating with GR00T inference services.
Uses Isaac-GR00T's native client when available, falls back to embedded implementation.

When `gr00t` is installed:
  - Uses `gr00t.policy.server_client.PolicyClient` directly
  - Full compatibility with Isaac-GR00T server protocol

When `gr00t` is NOT installed:
  - Uses lightweight embedded client (ZMQ + msgpack)
  - Compatible with the same server protocol

SPDX-License-Identifier: Apache-2.0
"""

import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Try to use Isaac-GR00T native client; fall back to embedded implementation
# ---------------------------------------------------------------------------
_USING_GROOT_CLIENT = False

try:
    from gr00t.policy.server_client import MsgSerializer  # noqa: F401
    from gr00t.policy.server_client import PolicyClient as _NativePolicyClient

    _USING_GROOT_CLIENT = True
    logger.debug("Using gr00t.policy.server_client (Isaac-GR00T installed)")

    class ExternalRobotInferenceClient(_NativePolicyClient):
        """Thin wrapper around Isaac-GR00T's PolicyClient.

        Provides the ``get_action(observations)`` convenience method expected
        by ``Gr00tPolicy``.
        """

        def __init__(self, host: str = "localhost", port: int = 5555, timeout_ms: int = 15000, api_token: str = None):
            super().__init__(host=host, port=port, timeout_ms=timeout_ms, api_token=api_token, strict=False)

        def get_action(self, observations: Dict[str, Any]) -> Dict[str, Any]:
            """Get actions via the native PolicyClient protocol."""
            action, _info = self._get_action(observations)
            return action

except ImportError:
    logger.debug("Isaac-GR00T not installed – using embedded client")

    import io
    import json

    import msgpack
    import numpy as np
    import zmq

    from .data_config import ModalityConfig

    class MsgSerializer:
        """Message serializer for ZMQ communication with GR00T inference service."""

        @staticmethod
        def to_bytes(data: dict) -> bytes:
            return msgpack.packb(data, default=MsgSerializer.encode_custom_classes)

        @staticmethod
        def from_bytes(data: bytes) -> dict:
            return msgpack.unpackb(data, object_hook=MsgSerializer.decode_custom_classes)

        @staticmethod
        def decode_custom_classes(obj):
            if "__ModalityConfig_class__" in obj:
                obj = ModalityConfig(**json.loads(obj["as_json"]))
            if "__ndarray_class__" in obj:
                obj = np.load(io.BytesIO(obj["as_npy"]), allow_pickle=False)
            return obj

        @staticmethod
        def encode_custom_classes(obj):
            if isinstance(obj, ModalityConfig):
                return {"__ModalityConfig_class__": True, "as_json": obj.model_dump_json()}
            if isinstance(obj, np.ndarray):
                output = io.BytesIO()
                np.save(output, obj, allow_pickle=False)
                return {"__ndarray_class__": True, "as_npy": output.getvalue()}
            return obj

    class ExternalRobotInferenceClient:
        """Embedded client for GR00T inference services (fallback)."""

        def __init__(self, host: str = "localhost", port: int = 5555, timeout_ms: int = 15000, api_token: str = None):
            self.context = zmq.Context()
            self.host = host
            self.port = port
            self.timeout_ms = timeout_ms
            self.api_token = api_token
            self._init_socket()

        def _init_socket(self):
            self.socket = self.context.socket(zmq.REQ)
            self.socket.connect(f"tcp://{self.host}:{self.port}")

        def ping(self) -> bool:
            try:
                self.call_endpoint("ping", requires_input=False)
                return True
            except zmq.error.ZMQError:
                self._init_socket()
                return False

        def kill_server(self):
            self.call_endpoint("kill", requires_input=False)

        def call_endpoint(self, endpoint: str, data: dict | None = None, requires_input: bool = True) -> dict:
            request: dict = {"endpoint": endpoint}
            if requires_input:
                request["data"] = data
            if self.api_token:
                request["api_token"] = self.api_token

            self.socket.send(MsgSerializer.to_bytes(request))
            message = self.socket.recv()
            response = MsgSerializer.from_bytes(message)

            if isinstance(response, dict) and "error" in response:
                raise RuntimeError(f"Server error: {response['error']}")
            return response

        def get_action(self, observations: Dict[str, Any]) -> Dict[str, Any]:
            """Get actions from the GR00T policy server."""
            return self.call_endpoint("get_action", observations)

        def __del__(self):
            if hasattr(self, "socket"):
                self.socket.close()
            if hasattr(self, "context"):
                self.context.term()


__all__ = ["ExternalRobotInferenceClient", "MsgSerializer"]
