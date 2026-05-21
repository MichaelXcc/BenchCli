"""Interactive prompts and rich output helpers."""
from __future__ import annotations

from typing import Optional

import questionary
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from .config import (
    DEFAULT_CONTAINER_NAME,
    DEFAULT_HF_CACHE,
    DEFAULT_HOST_PORT,
    DEFAULT_IMAGE,
    BenchConfig,
    ServeConfig,
)
from .docker_manager import ContainerInfo

console = Console()


# -- banners --------------------------------------------------------------


def banner() -> None:
    console.print(
        Panel.fit(
            "[bold cyan]BenchCli[/bold cyan]  ·  vLLM container & benchmark helper\n"
            "[dim]Use arrow keys to navigate. Ctrl-C to quit.[/dim]",
            border_style="cyan",
        )
    )


def show_status(info: Optional[ContainerInfo]) -> None:
    if info is None:
        console.print("[yellow]No managed vLLM container is running.[/yellow]")
        return
    table = Table(title="vLLM container", show_header=False, border_style="cyan")
    table.add_row("name", info.name)
    table.add_row("id", info.id)
    table.add_row("status", info.status)
    table.add_row("image", info.image)
    if info.ports:
        mappings = []
        for container_port, host_bindings in info.ports.items():
            if not host_bindings:
                continue
            for hb in host_bindings:
                mappings.append(f"{hb.get('HostIp', '0.0.0.0')}:{hb['HostPort']} → {container_port}")
        if mappings:
            table.add_row("ports", "\n".join(mappings))
    console.print(table)


# -- main menu ------------------------------------------------------------


MENU_SERVE = "Start vLLM inference server"
MENU_BENCH = "Run `vllm bench serve`"
MENU_STATUS = "Show container status"
MENU_LOGS = "Tail container logs"
MENU_STOP = "Stop and remove container"
MENU_QUIT = "Quit"


def main_menu() -> str:
    choice = questionary.select(
        "What do you want to do?",
        choices=[
            MENU_SERVE,
            MENU_BENCH,
            MENU_STATUS,
            MENU_LOGS,
            MENU_STOP,
            questionary.Separator(),
            MENU_QUIT,
        ],
    ).ask()
    return choice or MENU_QUIT


# -- prompts --------------------------------------------------------------


def _ask_text(message: str, default: str = "") -> str:
    return (questionary.text(message, default=default).ask() or default).strip()


def _ask_int(message: str, default: int) -> int:
    raw = questionary.text(
        message,
        default=str(default),
        validate=lambda v: v.isdigit() or "Please enter a positive integer.",
    ).ask()
    return int(raw) if raw else default


def _ask_optional_int(message: str, default: Optional[int] = None) -> Optional[int]:
    raw = questionary.text(
        message + " (blank to skip)",
        default="" if default is None else str(default),
        validate=lambda v: v == "" or v.isdigit() or "Enter an integer or leave blank.",
    ).ask()
    if not raw:
        return None
    return int(raw)


def _ask_float(message: str, default: float) -> float:
    raw = questionary.text(
        message,
        default="inf" if default == float("inf") else str(default),
    ).ask()
    if raw is None or raw.strip() == "":
        return default
    raw = raw.strip()
    if raw.lower() in {"inf", "infinity"}:
        return float("inf")
    try:
        return float(raw)
    except ValueError:
        return default


def prompt_serve_config(default_model: Optional[str] = None) -> ServeConfig:
    image = _ask_text("Docker image", default=DEFAULT_IMAGE)
    container_name = _ask_text("Container name", default=DEFAULT_CONTAINER_NAME)
    model = _ask_text("Model (HF id or local path)", default=default_model or "")
    while not model:
        console.print("[red]Model is required.[/red]")
        model = _ask_text("Model (HF id or local path)", default="")
    host_port = _ask_int("Host port", default=DEFAULT_HOST_PORT)
    gpus = _ask_text("GPUs (all / count / none)", default="all")
    hf_cache_dir = _ask_text("Host HF cache dir", default=DEFAULT_HF_CACHE)
    hf_token = _ask_text("HF token (blank if not needed)", default="") or None
    extra = _ask_text("Extra `vllm serve` args", default="")
    extra_args = extra.split() if extra else []
    return ServeConfig(
        image=image,
        container_name=container_name,
        model=model,
        host_port=host_port,
        container_port=8000,
        gpus=gpus,
        hf_cache_dir=hf_cache_dir,
        hf_token=hf_token,
        extra_args=extra_args,
    )


def prompt_bench_config(default_model: str) -> BenchConfig:
    model = _ask_text("Model served by vLLM", default=default_model)
    base_url = _ask_text("Base URL (inside container)", default="http://localhost:8000")
    dataset_name = questionary.select(
        "Dataset",
        choices=["random", "sharegpt", "sonnet", "hf"],
        default="random",
    ).ask() or "random"
    num_prompts = _ask_int("Number of prompts", default=200)
    request_rate = _ask_float("Request rate (req/s, 'inf' for max)", default=float("inf"))
    max_concurrency = _ask_optional_int("Max concurrency", default=None)
    random_input_len = 1024
    random_output_len = 128
    if dataset_name == "random":
        random_input_len = _ask_int("Random input length", default=1024)
        random_output_len = _ask_int("Random output length", default=128)
    save_result = questionary.confirm("Save result JSON?", default=False).ask() or False
    result_dir = None
    if save_result:
        result_dir = _ask_text("Result dir (inside container)", default="/tmp/vllm-bench")
    extra = _ask_text("Extra `vllm bench serve` args", default="")
    extra_args = extra.split() if extra else []
    return BenchConfig(
        model=model,
        base_url=base_url,
        dataset_name=dataset_name,
        num_prompts=num_prompts,
        request_rate=request_rate,
        max_concurrency=max_concurrency,
        random_input_len=random_input_len,
        random_output_len=random_output_len,
        save_result=save_result,
        result_dir=result_dir,
        extra_args=extra_args,
    )


def confirm(message: str, default: bool = False) -> bool:
    return bool(questionary.confirm(message, default=default).ask())
