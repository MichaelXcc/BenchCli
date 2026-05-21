"""BenchCli command-line entry point."""
from __future__ import annotations

import urllib.request
from typing import Optional

import typer

from . import ui
from .config import (
    DEFAULT_CONTAINER_NAME,
    DEFAULT_HOST_PORT,
    DEFAULT_IMAGE,
    DEFAULT_WORKSPACE_MOUNT,
    BenchConfig,
    ServeConfig,
)
from .docker_manager import DockerError, DockerManager

app = typer.Typer(
    add_completion=False,
    no_args_is_help=False,
    help="Interactive CLI for vLLM containers and `vllm bench serve`.",
)


# -- helpers ----------------------------------------------------------------


def _manager() -> DockerManager:
    try:
        return DockerManager()
    except DockerError as exc:
        ui.console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc


def _resolve_running_model(dm: DockerManager, container_name: str) -> Optional[str]:
    """Best-effort: read the model arg from the container's command."""
    c = dm.find(container_name)
    if c is None:
        return None
    cmd = c.attrs.get("Config", {}).get("Cmd") or []
    if "--model" in cmd:
        model_index = cmd.index("--model") + 1
        if model_index < len(cmd):
            return cmd[model_index]
    # Expected: ["vllm", "serve", "<model>", ...]
    if len(cmd) >= 3 and cmd[0] == "vllm" and cmd[1] == "serve":
        return cmd[2]
    return None


# -- subcommands ------------------------------------------------------------


@app.command()
def serve(
    model: Optional[str] = typer.Option(None, help="Model HF id or local path."),
    image: str = typer.Option(DEFAULT_IMAGE, help="Docker image."),
    container_name: str = typer.Option(DEFAULT_CONTAINER_NAME, help="Container name."),
    host_port: int = typer.Option(DEFAULT_HOST_PORT, help="Host port to expose."),
    gpus: str = typer.Option(
        "all",
        help="GPU spec: 'all', 'none', a count like '2', or GPU ids like '0,1'.",
    ),
    interactive: bool = typer.Option(True, help="Prompt for missing fields."),
) -> None:
    """Start a vLLM inference container."""
    dm = _manager()
    if interactive and (not model):
        cfg = ui.prompt_serve_config(default_model=model)
    else:
        if not model:
            ui.console.print("[red]--model is required when --no-interactive.[/red]")
            raise typer.Exit(code=2)
        cfg = ServeConfig(
            image=image,
            container_name=container_name,
            model=model,
            host_port=host_port,
            gpus=gpus,
        )
    ui.console.print(f"[cyan]Starting vLLM container [bold]{cfg.container_name}[/bold]…[/cyan]")
    try:
        container = dm.start_vllm(cfg)
    except DockerError as exc:
        ui.console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc
    ui.console.print(
        f"[green]Started[/green] {container.name} "
        f"on http://localhost:{cfg.host_port}  (id={container.short_id})"
    )
    ui.console.print(
        "[dim]Note: the model still needs to load. Use `benchcli logs` to follow startup.[/dim]"
    )


@app.command()
def bench(
    model: Optional[str] = typer.Option(None, help="Model name passed to `vllm bench serve`."),
    container_name: str = typer.Option(DEFAULT_CONTAINER_NAME, help="Container to exec into."),
    interactive: bool = typer.Option(True, help="Prompt for parameters."),
) -> None:
    """Run `vllm bench serve` inside the running container."""
    dm = _manager()
    info = dm.info(container_name)
    if info is None or info.status != "running":
        ui.console.print(
            f"[red]Container {container_name!r} is not running. Start it with `benchcli serve` first.[/red]"
        )
        raise typer.Exit(code=1)

    default_model = model or _resolve_running_model(dm, container_name) or ""
    if interactive:
        cfg = ui.prompt_bench_config(default_model=default_model)
    else:
        if not default_model:
            ui.console.print("[red]--model is required when --no-interactive.[/red]")
            raise typer.Exit(code=2)
        cfg = BenchConfig(model=default_model)

    rc = dm.exec_interactive(
        container_name,
        cfg.vllm_bench_command(),
        workdir=DEFAULT_WORKSPACE_MOUNT,
    )
    if rc != 0:
        ui.console.print(f"[yellow]Benchmark exited with code {rc}.[/yellow]")
        raise typer.Exit(code=rc)


@app.command("download-dataset")
def download_dataset() -> None:
    """Download a benchmark dataset to the host workspace."""
    url, output_path = ui.prompt_dataset_download()
    ui.console.print(f"[cyan]Downloading[/cyan] {url}")
    try:
        urllib.request.urlretrieve(url, output_path)
    except Exception as exc:  # noqa: BLE001
        ui.console.print(f"[red]Failed to download dataset: {exc}[/red]")
        raise typer.Exit(code=1) from exc
    ui.console.print(f"[green]Downloaded[/green] {output_path}")


@app.command()
def status(
    container_name: str = typer.Option(DEFAULT_CONTAINER_NAME, help="Container name."),
) -> None:
    """Show the managed container's status."""
    dm = _manager()
    ui.show_status(dm.info(container_name))


@app.command()
def logs(
    container_name: str = typer.Option(DEFAULT_CONTAINER_NAME, help="Container name."),
    tail: int = typer.Option(200, help="Number of lines to start with."),
) -> None:
    """Tail logs from the vLLM container."""
    dm = _manager()
    try:
        for chunk in dm.stream_logs(container_name, tail=tail):
            try:
                print(chunk.decode("utf-8", errors="replace"), end="")
            except Exception:
                pass
    except DockerError as exc:
        ui.console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc
    except KeyboardInterrupt:
        pass


@app.command()
def stop(
    container_name: str = typer.Option(DEFAULT_CONTAINER_NAME, help="Container name."),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation."),
) -> None:
    """Stop and remove the managed container."""
    dm = _manager()
    info = dm.info(container_name)
    if info is None:
        ui.console.print(f"[yellow]No container named {container_name!r}.[/yellow]")
        return
    if not yes and not ui.confirm(f"Stop and remove {container_name}?", default=False):
        ui.console.print("[dim]Cancelled.[/dim]")
        return
    if dm.stop(container_name):
        ui.console.print(f"[green]Removed {container_name}.[/green]")


# -- interactive default ---------------------------------------------------


def _interactive_loop() -> None:
    dm = _manager()
    local_model_root: Optional[str] = None
    selected_local_model: Optional[str] = None
    ui.banner()
    while True:
        choice = ui.main_menu(
            local_model_root=local_model_root,
            selected_local_model=selected_local_model,
        )
        if choice == ui.MENU_QUIT:
            return
        try:
            if choice == ui.MENU_SERVE:
                cfg = ui.prompt_serve_config(
                    local_model_root=local_model_root,
                    selected_local_model=selected_local_model,
                )
                if cfg.model_mount_dir:
                    local_model_root = cfg.model_mount_dir
                    selected_local_model = cfg.model
                ui.console.print(f"[cyan]Starting {cfg.container_name}…[/cyan]")
                container = dm.start_vllm(cfg)
                ui.console.print(
                    f"[green]Started[/green] {container.name} on http://localhost:{cfg.host_port}"
                )
            elif choice == ui.MENU_BENCH:
                info = dm.info(DEFAULT_CONTAINER_NAME)
                if info is None or info.status != "running":
                    ui.console.print(
                        "[red]No running vLLM container. Start one first.[/red]"
                    )
                    continue
                default_model = _resolve_running_model(dm, DEFAULT_CONTAINER_NAME) or ""
                cfg = ui.prompt_bench_config(default_model=default_model)
                dm.exec_interactive(
                    DEFAULT_CONTAINER_NAME,
                    cfg.vllm_bench_command(),
                    workdir=DEFAULT_WORKSPACE_MOUNT,
                )
            elif choice == ui.MENU_DOWNLOAD_DATASET:
                url, output_path = ui.prompt_dataset_download()
                ui.console.print(f"[cyan]Downloading[/cyan] {url}")
                try:
                    urllib.request.urlretrieve(url, output_path)
                except Exception as exc:  # noqa: BLE001
                    ui.console.print(f"[red]Failed to download dataset: {exc}[/red]")
                    continue
                ui.console.print(f"[green]Downloaded[/green] {output_path}")
            elif choice == ui.MENU_MODEL_ROOT:
                selected_local_model, local_model_root = ui.prompt_local_model(
                    default_root=local_model_root
                )
                ui.console.print(
                    f"[green]Selected local model {selected_local_model} from {local_model_root}.[/green]"
                )
            elif choice == ui.MENU_STATUS:
                ui.show_status(dm.info(DEFAULT_CONTAINER_NAME))
            elif choice == ui.MENU_LOGS:
                try:
                    for chunk in dm.stream_logs(DEFAULT_CONTAINER_NAME, tail=200):
                        print(chunk.decode("utf-8", errors="replace"), end="")
                except KeyboardInterrupt:
                    ui.console.print("\n[dim]Stopped tailing logs.[/dim]")
            elif choice == ui.MENU_STOP:
                if ui.confirm(f"Stop and remove {DEFAULT_CONTAINER_NAME}?", default=False):
                    if dm.stop(DEFAULT_CONTAINER_NAME):
                        ui.console.print(f"[green]Removed {DEFAULT_CONTAINER_NAME}.[/green]")
                    else:
                        ui.console.print("[yellow]Container not found.[/yellow]")
        except DockerError as exc:
            ui.console.print(f"[red]{exc}[/red]")
        except ui.BackRequested:
            ui.console.print("\n[dim]Cancelled.[/dim]")
        except KeyboardInterrupt:
            ui.console.print("\n[dim]Cancelled.[/dim]")


@app.callback(invoke_without_command=True)
def _root(ctx: typer.Context) -> None:
    """Launch the interactive menu when no subcommand is given."""
    if ctx.invoked_subcommand is None:
        try:
            _interactive_loop()
        except KeyboardInterrupt:
            ui.console.print("\n[dim]Bye.[/dim]")


def main() -> None:  # pragma: no cover — convenience for `python -m benchcli`
    app()


if __name__ == "__main__":  # pragma: no cover
    main()
