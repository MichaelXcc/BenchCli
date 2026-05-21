"""Thin wrapper around the docker SDK for vLLM container lifecycle."""
from __future__ import annotations

import os
import shlex
import subprocess
from dataclasses import dataclass
from typing import Iterable, Optional

import docker
from docker.errors import APIError, ImageNotFound, NotFound
from docker.models.containers import Container

from .config import ServeConfig


class DockerError(RuntimeError):
    """Raised when a docker operation cannot be completed."""


@dataclass
class ContainerInfo:
    name: str
    id: str
    status: str
    image: str
    ports: dict


class DockerManager:
    """High-level operations on the vLLM container managed by BenchCli."""

    def __init__(self) -> None:
        try:
            self.client = docker.from_env()
            self.client.ping()
        except Exception as exc:  # noqa: BLE001 — surface any docker connectivity issue
            raise DockerError(
                "Cannot reach the Docker daemon. Is Docker running and your user permitted to use it?"
            ) from exc

    # -- queries --------------------------------------------------------------

    def find(self, name: str) -> Optional[Container]:
        try:
            return self.client.containers.get(name)
        except NotFound:
            return None

    def info(self, name: str) -> Optional[ContainerInfo]:
        c = self.find(name)
        if c is None:
            return None
        c.reload()
        return ContainerInfo(
            name=c.name,
            id=c.short_id,
            status=c.status,
            image=(c.image.tags[0] if c.image and c.image.tags else c.image.id),
            ports=c.attrs.get("NetworkSettings", {}).get("Ports", {}) or {},
        )

    # -- lifecycle ------------------------------------------------------------

    @staticmethod
    def _gpu_device_request(gpus: str) -> docker.types.DeviceRequest:
        spec = gpus.strip().lower()
        if spec == "all":
            return docker.types.DeviceRequest(count=-1, capabilities=[["gpu"]])
        if spec.isdigit():
            return docker.types.DeviceRequest(count=int(spec), capabilities=[["gpu"]])
        if spec.startswith("device="):
            spec = spec.removeprefix("device=")
        device_ids = [device.strip() for device in spec.split(",") if device.strip()]
        if device_ids and all(device.isdigit() for device in device_ids):
            return docker.types.DeviceRequest(device_ids=device_ids, capabilities=[["gpu"]])
        raise DockerError(
            "Invalid GPU spec. Use 'all', 'none', a count like '2', or GPU ids like '0,1'."
        )

    def ensure_image(self, image: str, pull_if_missing: bool = True) -> None:
        try:
            self.client.images.get(image)
        except ImageNotFound:
            if not pull_if_missing:
                raise DockerError(f"Image {image!r} not present locally.")
            self.client.images.pull(image)

    def start_vllm(self, cfg: ServeConfig, pull_if_missing: bool = True) -> Container:
        existing = self.find(cfg.container_name)
        if existing is not None:
            existing.reload()
            if existing.status == "running":
                return existing
            existing.remove(force=True)

        self.ensure_image(cfg.image, pull_if_missing=pull_if_missing)

        env: dict[str, str] = {}
        if cfg.hf_token:
            env["HUGGING_FACE_HUB_TOKEN"] = cfg.hf_token

        device_requests = []
        if cfg.gpus and cfg.gpus.lower() != "none":
            device_requests.append(self._gpu_device_request(cfg.gpus))

        volumes = {}
        if cfg.hf_cache_dir:
            os.makedirs(cfg.hf_cache_dir, exist_ok=True)
            volumes[cfg.hf_cache_dir] = {"bind": "/root/.cache/huggingface", "mode": "rw"}
        if cfg.model_mount_dir:
            volumes[cfg.model_mount_dir] = {"bind": cfg.container_model_root, "mode": "ro"}
        if cfg.workspace_dir:
            os.makedirs(cfg.workspace_dir, exist_ok=True)
            volumes[cfg.workspace_dir] = {"bind": cfg.container_workspace_dir, "mode": "rw"}

        try:
            container = self.client.containers.run(
                image=cfg.image,
                command=cfg.vllm_command(),
                name=cfg.container_name,
                detach=True,
                ports={f"{cfg.container_port}/tcp": cfg.host_port},
                environment=env,
                volumes=volumes,
                device_requests=device_requests or None,
                ipc_mode="host",
                shm_size="16g",
            )
        except APIError as exc:
            raise DockerError(f"Failed to start container: {exc.explanation or exc}") from exc
        return container

    def stop(self, name: str, remove: bool = True) -> bool:
        c = self.find(name)
        if c is None:
            return False
        try:
            c.stop(timeout=10)
        except APIError:
            pass
        if remove:
            try:
                c.remove(force=True)
            except APIError:
                pass
        return True

    # -- exec / logs ----------------------------------------------------------

    def stream_logs(self, name: str, tail: int = 100) -> Iterable[bytes]:
        c = self.find(name)
        if c is None:
            raise DockerError(f"Container {name!r} not found.")
        return c.logs(stream=True, follow=True, tail=tail)

    def exec_interactive(
        self,
        name: str,
        command: list[str],
        workdir: Optional[str] = None,
    ) -> int:
        """Run `docker exec -it <name> <command>` so the user sees live output.

        Using the docker CLI here keeps TTY behavior simple for benchmark output.
        """
        c = self.find(name)
        if c is None:
            raise DockerError(f"Container {name!r} not found.")
        argv = ["docker", "exec", "-it"]
        if workdir:
            argv += ["-w", workdir]
        argv += [name, *command]
        printable = " ".join(shlex.quote(p) for p in argv)
        print(f"$ {printable}")
        return subprocess.call(argv)
