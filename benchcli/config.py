"""Configuration models and defaults for BenchCli."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field


DEFAULT_IMAGE = "vllm/vllm-openai:latest"
DEFAULT_CONTAINER_NAME = "benchcli-vllm"
DEFAULT_HOST_PORT = 8000
DEFAULT_CONTAINER_PORT = 8000
DEFAULT_HF_CACHE = str(Path.home() / ".cache" / "huggingface")
DEFAULT_CONTAINER_MODEL_ROOT = "/models"


class ServeConfig(BaseModel):
    """Parameters used to launch a vLLM inference container."""

    image: str = DEFAULT_IMAGE
    container_name: str = DEFAULT_CONTAINER_NAME
    model: str = Field(..., description="HF model id or local path mounted into the container.")
    host_port: int = DEFAULT_HOST_PORT
    container_port: int = DEFAULT_CONTAINER_PORT
    gpus: str = Field("all", description="Value passed to `--gpus`. Use 'all', a count, or 'none'.")
    hf_cache_dir: str = DEFAULT_HF_CACHE
    hf_token: Optional[str] = None
    model_mount_dir: Optional[str] = Field(
        None,
        description="Host directory mounted into the container when serving a local model.",
    )
    container_model_root: str = DEFAULT_CONTAINER_MODEL_ROOT
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
    base_url: str = "http://localhost:8000"
    dataset_name: str = "random"
    num_prompts: int = 200
    request_rate: float = float("inf")
    max_concurrency: Optional[int] = None
    random_input_len: int = 1024
    random_output_len: int = 128
    save_result: bool = False
    result_dir: Optional[str] = None
    extra_args: list[str] = Field(default_factory=list)

    def vllm_bench_command(self) -> list[str]:
        cmd = [
            "vllm",
            "bench",
            "serve",
            "--model",
            self.model,
            "--base-url",
            self.base_url,
            "--dataset-name",
            self.dataset_name,
            "--num-prompts",
            str(self.num_prompts),
        ]
        if self.request_rate != float("inf"):
            cmd += ["--request-rate", str(self.request_rate)]
        if self.max_concurrency is not None:
            cmd += ["--max-concurrency", str(self.max_concurrency)]
        if self.dataset_name == "random":
            cmd += [
                "--random-input-len",
                str(self.random_input_len),
                "--random-output-len",
                str(self.random_output_len),
            ]
        if self.save_result:
            cmd.append("--save-result")
            if self.result_dir:
                cmd += ["--result-dir", self.result_dir]
        cmd.extend(self.extra_args)
        return cmd
