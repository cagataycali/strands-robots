#!/usr/bin/env python3
"""
GR00T Inference Service Management Tool

Manages GR00T policy inference services running in Docker containers.
Uses Isaac-GR00T's native inference service for proper ZMQ communication.
"""

import subprocess
import socket
import time
import json
from typing import Dict, Any
from strands import tool


@tool
def gr00t_inference(
    action: str,
    checkpoint_path: str = None,
    policy_name: str = None,
    port: int = None,
    data_config: str = "so100_dualcam",
    embodiment_tag: str = "new_embodiment",
    denoising_steps: int = 4,
    host: str = "0.0.0.0",
    container_name: str = None,
    timeout: int = 60,
) -> Dict[str, Any]:
    """
    Manage GR00T inference services in Docker containers using Isaac-GR00T native scripts.

    Args:
        action: Action to perform
            - "start": Start inference service with checkpoint
            - "stop": Stop inference service on port
            - "status": Check status of service on port
            - "list": List all running services
            - "restart": Restart service with new checkpoint
            - "find_containers": Find available isaac-gr00t containers
        checkpoint_path: Path to model checkpoint (for start/restart)
        policy_name: Name for the policy service (for registration)
        port: Port for inference service
        data_config: GR00T data config (so100_dualcam, so100, fourier_gr1_arms_only, etc.)
        embodiment_tag: Embodiment tag for model
        denoising_steps: Number of denoising steps
        host: Host to bind service to
        container_name: Specific container name
        timeout: Timeout for operations

    Returns:
        Dict with status and information about the operation
    """

    if action == "find_containers":
        return _find_gr00t_containers()
    elif action == "list":
        return _list_running_services()
    elif action == "status":
        if port is None:
            return {"status": "error", "message": "Port required for status check"}
        return _check_service_status(port)
    elif action == "stop":
        if port is None:
            return {"status": "error", "message": "Port required to stop service"}
        return _stop_service(port)
    elif action == "start":
        if checkpoint_path is None:
            return {"status": "error", "message": "Checkpoint path required to start service"}
        if port is None:
            return {"status": "error", "message": "Port required to start service"}
        return _start_service(
            checkpoint_path,
            port,
            data_config,
            embodiment_tag,
            denoising_steps,
            host,
            container_name,
            policy_name,
            timeout,
        )
    elif action == "restart":
        if checkpoint_path is None or port is None:
            return {"status": "error", "message": "Checkpoint path and port required for restart"}
        # Stop existing service and start new one
        _stop_service(port)
        time.sleep(2)  # Brief pause
        return _start_service(
            checkpoint_path,
            port,
            data_config,
            embodiment_tag,
            denoising_steps,
            host,
            container_name,
            policy_name,
            timeout,
        )
    else:
        return {"status": "error", "message": f"Unknown action: {action}"}


def _find_gr00t_containers() -> Dict[str, Any]:
    """Find available isaac-gr00t containers."""
    try:
        result = subprocess.run(
            ["docker", "ps", "-a", "--format", "{{.Names}}\\t{{.Status}}\\t{{.Ports}}"],
            capture_output=True,
            text=True,
            check=True,
        )

        containers = []
        for line in result.stdout.strip().split("\n"):
            if line and ("isaac" in line.lower() and "gr00t" in line.lower()):
                parts = line.split("\t")
                containers.append({"name": parts[0], "status": parts[1], "ports": parts[2] if len(parts) > 2 else ""})

        return {"status": "success", "containers": containers, "message": f"Found {len(containers)} GR00T containers"}

    except subprocess.CalledProcessError as e:
        return {"status": "error", "message": f"Failed to find containers: {e}"}


def _list_running_services() -> Dict[str, Any]:
    """List all running GR00T inference services by checking ZMQ ports."""
    try:
        services = []
        common_ports = [5555, 5556, 5557, 5558, 8000, 8001, 8002, 8003]

        for port in common_ports:
            if _is_zmq_service_running(port):
                services.append({"port": port, "protocol": "ZMQ", "status": "running"})

        return {"status": "success", "services": services, "message": f"Found {len(services)} running ZMQ services"}

    except Exception as e:
        return {"status": "error", "message": f"Failed to list services: {e}"}


def _is_zmq_service_running(port: int) -> bool:
    """Check if ZMQ service is running on port."""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1)
        result = sock.connect_ex(("localhost", port))
        sock.close()
        return result == 0
    except:
        return False


def _check_service_status(port: int) -> Dict[str, Any]:
    """Check status of ZMQ service on specific port."""
    if _is_zmq_service_running(port):
        return {"status": "success", "port": port, "service_status": "running", "protocol": "ZMQ"}
    else:
        return {
            "status": "error",
            "port": port,
            "service_status": "not_running",
            "message": f"No ZMQ service running on port {port}",
        }


def _stop_service(port: int) -> Dict[str, Any]:
    """Stop service running on specific port."""
    try:
        # Find process using the port
        result = subprocess.run(["lsof", "-t", f"-i:{port}"], capture_output=True, text=True)

        if result.returncode == 0:
            pids = result.stdout.strip().split("\n")
            for pid in pids:
                if pid:
                    subprocess.run(["kill", "-TERM", pid], check=True)

            # Wait for graceful shutdown
            time.sleep(2)

            # Force kill if still running
            result = subprocess.run(["lsof", "-t", f"-i:{port}"], capture_output=True, text=True)

            if result.returncode == 0:
                pids = result.stdout.strip().split("\n")
                for pid in pids:
                    if pid:
                        subprocess.run(["kill", "-KILL", pid], check=True)

            return {"status": "success", "port": port, "message": f"Service on port {port} stopped"}
        else:
            return {"status": "success", "port": port, "message": f"No service running on port {port}"}

    except Exception as e:
        return {"status": "error", "message": f"Failed to stop service: {e}"}


def _start_service(
    checkpoint_path: str,
    port: int,
    data_config: str,
    embodiment_tag: str,
    denoising_steps: int,
    host: str,
    container_name: str,
    policy_name: str,
    timeout: int,
) -> Dict[str, Any]:
    """Start GR00T inference service using Isaac-GR00T's native inference service."""
    try:
        # Find container if not specified
        if container_name is None:
            containers = _find_gr00t_containers()
            if containers["status"] == "error":
                return containers

            running_containers = [c for c in containers["containers"] if "Up" in c["status"]]
            if not running_containers:
                return {"status": "error", "message": "No running GR00T containers found"}

            container_name = running_containers[0]["name"]

        # Build Isaac-GR00T inference service command
        cmd = [
            "docker",
            "exec",
            "-d",
            container_name,
            "python",
            "/opt/Isaac-GR00T/scripts/inference_service.py",
            "--server",
            "--model-path",
            checkpoint_path,
            "--port",
            str(port),
            "--host",
            host,
            "--data-config",
            data_config,
            "--embodiment-tag",
            embodiment_tag,
            "--denoising-steps",
            str(denoising_steps),
        ]

        # Start service
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)

        # Wait for ZMQ service to start
        start_time = time.time()
        while time.time() - start_time < timeout:
            if _is_zmq_service_running(port):
                return {
                    "status": "success",
                    "port": port,
                    "checkpoint_path": checkpoint_path,
                    "container_name": container_name,
                    "policy_name": policy_name,
                    "protocol": "ZMQ",
                    "data_config": data_config,
                    "embodiment_tag": embodiment_tag,
                    "message": f"GR00T ZMQ service started on port {port}",
                }
            time.sleep(1)

        return {"status": "error", "message": f"ZMQ service failed to start within {timeout} seconds"}

    except subprocess.CalledProcessError as e:
        return {"status": "error", "message": f"Failed to start service: {e}"}
    except Exception as e:
        return {"status": "error", "message": f"Unexpected error: {e}"}


if __name__ == "__main__":
    print("🐳 GR00T Inference Service Manager (Isaac-GR00T Native)")
    print("Uses Isaac-GR00T's ZMQ-based inference service")

    # Example usage
    examples = [
        "gr00t_inference(action='find_containers')",
        "gr00t_inference(action='start', checkpoint_path='/data/checkpoints/gr00t-wave/checkpoint-300000', port=5555, policy_name='wave_model')",
        "gr00t_inference(action='list')",
        "gr00t_inference(action='status', port=5555)",
        "gr00t_inference(action='stop', port=5555)",
    ]

    for example in examples:
        print(f"  {example}")
