"""Configuration models and defaults for OpenBench."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field


DEFAULT_IMAGE = "vllm/vllm-openai:latest"
DEFAULT_CONTAINER_NAME = "openbench-vllm"
DEFAULT_HOST_PORT = 8000
DEFAULT_CONTAINER_PORT = 8000
DEFAULT_HF_CACHE = str(Path.home() / ".cache" / "huggingface")
DEFAULT_CONTAINER_MODEL_ROOT = "/models"
DEFAULT_WORKSPACE_MOUNT = "/workspace/openbench"
DEFAULT_DATASET_DIR = "datasets"


class ServeConfig(BaseModel):
    """Parameters used to launch a vLLM inference container."""

    image: str = DEFAULT_IMAGE
    container_name: str = DEFAULT_CONTAINER_NAME
    model: str = Field(..., description="HF model id or local path mounted into the container.")
    host_port: int = DEFAULT_HOST_PORT
    container_port: int = DEFAULT_CONTAINER_PORT
    gpus: str = Field(
        "all",
        description="GPU spec. Use 'all', 'none', a count like '2', or GPU ids like '0,1'.",
    )
    hf_cache_dir: str = DEFAULT_HF_CACHE
    hf_token: Optional[str] = None
    model_mount_dir: Optional[str] = Field(
        None,
        description="Host directory mounted into the container when serving a local model.",
    )
    container_model_root: str = DEFAULT_CONTAINER_MODEL_ROOT
    workspace_dir: str = Field(
        default_factory=lambda: str(Path.cwd()),
        description="Host working directory mounted into the container.",
    )
    container_workspace_dir: str = DEFAULT_WORKSPACE_MOUNT
    extra_args: list[str] = Field(
        default_factory=list,
        description="Extra args appended to `vllm serve` inside the container.",
    )

    def vllm_command(self) -> list[str]:
        cmd = ["--model", self.model, "--host", "0.0.0.0", "--port", str(self.container_port)]
        cmd.extend(self.extra_args)
        return cmd


class BenchConfig(BaseModel):
    """Parameters for `vllm bench serve` executed inside the container."""

    model: str
    bench_command: list[str] = Field(default_factory=lambda: ["vllm", "bench", "serve"])
    backend: str = "openai"
    base_url: Optional[str] = "http://localhost:8000"
    host: str = "127.0.0.1"
    port: int = 8000
    endpoint: str = "/v1/completions"
    dataset_name: str = "random"
    dataset_path: Optional[str] = None
    num_prompts: int = 200
    request_rate: float = float("inf")
    max_concurrency: Optional[int] = None
    save_result: bool = False
    result_dir: Optional[str] = None
    extra_args: list[str] = Field(default_factory=list)

    def vllm_bench_command(self) -> list[str]:
        cmd = [
            *self.bench_command,
            "--model",
            self.model,
            "--backend",
            self.backend,
            "--dataset-name",
            self.dataset_name,
            "--num-prompts",
            str(self.num_prompts),
            "--endpoint",
            self.endpoint,
        ]
        if self.base_url:
            cmd += ["--base-url", self.base_url]
        else:
            cmd += ["--host", self.host, "--port", str(self.port)]
        if self.dataset_path:
            cmd += ["--dataset-path", self.dataset_path]
        if self.request_rate != float("inf"):
            cmd += ["--request-rate", str(self.request_rate)]
        if self.max_concurrency is not None:
            cmd += ["--max-concurrency", str(self.max_concurrency)]
        if self.save_result:
            cmd.append("--save-result")
            if self.result_dir:
                cmd += ["--result-dir", self.result_dir]
        cmd.extend(self.extra_args)
        return cmd
