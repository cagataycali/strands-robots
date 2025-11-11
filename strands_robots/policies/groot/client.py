#!/usr/bin/env python3
"""
GR00T Client Implementation

Embedded Isaac-GR00T client components to eliminate external dependencies.
Extracted from Isaac-GR00T to provide complete abstraction.

SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
"""

import io
import json
from dataclasses import dataclass
from typing import Any, Callable, Dict

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


@dataclass
class EndpointHandler:
    """Handler for inference service endpoints."""

    handler: Callable
    requires_input: bool = True


class BaseInferenceClient:
    """Base client for communicating with GR00T inference services."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 5555,
        timeout_ms: int = 15000,
        api_token: str = None,
    ):
        self.context = zmq.Context()
        self.host = host
        self.port = port
        self.timeout_ms = timeout_ms
        self.api_token = api_token
        self._init_socket()

    def _init_socket(self):
        """Initialize or reinitialize the socket with current settings"""
        self.socket = self.context.socket(zmq.REQ)
        self.socket.connect(f"tcp://{self.host}:{self.port}")

    def ping(self) -> bool:
        """Ping the inference service to check connectivity."""
        try:
            self.call_endpoint("ping", requires_input=False)
            return True
        except zmq.error.ZMQError:
            self._init_socket()  # Recreate socket for next attempt
            return False

    def kill_server(self):
        """Kill the inference server."""
        self.call_endpoint("kill", requires_input=False)

    def call_endpoint(self, endpoint: str, data: dict | None = None, requires_input: bool = True) -> dict:
        """Call an endpoint on the inference server.

        Args:
            endpoint: The name of the endpoint.
            data: The input data for the endpoint.
            requires_input: Whether the endpoint requires input data.
        """
        request: dict = {"endpoint": endpoint}
        if requires_input:
            request["data"] = data
        if self.api_token:
            request["api_token"] = self.api_token

        self.socket.send(MsgSerializer.to_bytes(request))
        message = self.socket.recv()
        response = MsgSerializer.from_bytes(message)

        if "error" in response:
            raise RuntimeError(f"Server error: {response['error']}")
        return response

    def __del__(self):
        """Cleanup resources on destruction"""
        if hasattr(self, "socket"):
            self.socket.close()
        if hasattr(self, "context"):
            self.context.term()


class ExternalRobotInferenceClient(BaseInferenceClient):
    """Client for communicating with GR00T inference services.

    This class provides the interface to GR00T policy inference servers,
    allowing robot control through natural language instructions.
    """

    def get_action(self, observations: Dict[str, Any]) -> Dict[str, Any]:
        """Get actions from the GR00T policy server.

        Args:
            observations: Robot observations dict containing camera feeds,
                         robot state, and language instructions.
                         Format defined by the policy's modality configuration.

        Returns:
            Action chunk containing robot actions for execution.
        """
        return self.call_endpoint("get_action", observations)


__all__ = ["ExternalRobotInferenceClient", "BaseInferenceClient", "MsgSerializer"]
