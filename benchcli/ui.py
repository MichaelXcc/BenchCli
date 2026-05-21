"""Interactive prompts and rich output helpers."""
from __future__ import annotations

from pathlib import Path
import shlex
from typing import Optional

import questionary
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from .config import (
    DEFAULT_CONTAINER_NAME,
    DEFAULT_CONTAINER_MODEL_ROOT,
    DEFAULT_HF_CACHE,
    DEFAULT_HOST_PORT,
    DEFAULT_IMAGE,
    BenchConfig,
    ServeConfig,
)
from .docker_manager import ContainerInfo

console = Console()
MODEL_CONFIG_FILE = "config.json"
MODEL_WEIGHT_SUFFIXES = (".safetensors", ".bin", ".gguf")
CUSTOM_SERVE_ARGS = "__custom_serve_args__"

VLLM_MODEL_CONFIG_OPTIONS = [
    {
        "flag": "--runner",
        "zh": "模型运行器类型；通常保持 auto，生成模型可用 generate。",
        "choices": ["auto", "draft", "generate", "pooling"],
    },
    {
        "flag": "--convert",
        "zh": "用适配器转换模型用途；例如把生成模型转为 embed/classify。",
        "choices": ["auto", "classify", "embed", "none"],
    },
    {"flag": "--tokenizer", "zh": "指定 tokenizer 名称或路径；不填时使用模型路径。"},
    {
        "flag": "--tokenizer-mode",
        "zh": "tokenizer 加载模式；auto 会按模型自动选择。",
        "choices": ["auto", "deepseek_v32", "deepseek_v4", "fastokens", "hf", "mistral", "slow"],
    },
    {
        "flag": "--trust-remote-code",
        "zh": "信任并执行模型仓库中的自定义代码；只在可信模型上启用。",
        "kind": "bool",
    },
    {
        "flag": "--dtype",
        "zh": "模型权重和激活的数据类型；auto 会按模型配置选择。",
        "choices": ["auto", "bfloat16", "float", "float16", "float32", "half"],
    },
    {"flag": "--seed", "zh": "随机种子，用于复现采样行为。"},
    {"flag": "--hf-config-path", "zh": "指定 Hugging Face config 名称或路径。"},
    {
        "flag": "--allowed-local-media-path",
        "zh": "允许多模态请求读取的服务端本地媒体目录；有安全风险。",
    },
    {"flag": "--allowed-media-domains", "zh": "限制多模态媒体 URL 只能来自指定域名。"},
    {"flag": "--revision", "zh": "Hugging Face 模型版本、分支、标签或 commit。"},
    {"flag": "--code-revision", "zh": "Hugging Face 模型代码版本、分支、标签或 commit。"},
    {"flag": "--tokenizer-revision", "zh": "Hugging Face tokenizer 版本、分支、标签或 commit。"},
    {
        "flag": "--max-model-len",
        "zh": "模型上下文长度，支持 1k/1K/25.6k/auto 等写法。",
    },
    {"flag": "--tensor-parallel-size", "zh": "张量并行使用的 GPU 数量；例如 1、2、4、8。"},
    {
        "flag": "--gpu-memory-utilization",
        "zh": "每张 GPU 可用于模型执行的显存比例；例如 0.9 或 0.95。",
    },
    {"flag": "--quantization", "zh": "权重量化方式；不填时按模型配置自动判断。"},
    {"flag": "--quantization-config", "zh": "量化配置 JSON 或键值参数。"},
    {
        "flag": "--allow-deprecated-quantization",
        "zh": "允许使用已废弃的量化方法。",
        "kind": "bool",
    },
    {
        "flag": "--enforce-eager",
        "zh": "强制 PyTorch eager 模式，禁用 CUDA graph。",
        "kind": "bool",
    },
    {
        "flag": "--enable-return-routed-experts",
        "zh": "返回 MoE routed experts 信息。",
        "kind": "bool",
    },
    {"flag": "--max-logprobs", "zh": "请求 logprobs 时最多返回的 log probability 数量。"},
    {
        "flag": "--logprobs-mode",
        "zh": "logprobs 返回 raw/processed、logits/logprobs 的哪一种。",
        "choices": ["raw_logprobs", "processed_logprobs", "raw_logits", "processed_logits"],
    },
    {
        "flag": "--use-fp64-gumbel",
        "zh": "采样 Gumbel 噪声使用 FP64，减少 tie 但更慢。",
        "kind": "bool",
    },
    {
        "flag": "--disable-sliding-window",
        "zh": "禁用 sliding window，并按窗口大小限制上下文。",
        "kind": "bool",
    },
    {
        "flag": "--skip-tokenizer-init",
        "zh": "跳过 tokenizer/detokenizer 初始化；请求需传 token ids。",
        "kind": "bool",
    },
    {
        "flag": "--enable-prompt-embeds",
        "zh": "允许请求传入 prompt_embeds；仅给可信调用方使用。",
        "kind": "bool",
    },
    {"flag": "--served-model-name", "zh": "API 中暴露的模型名；可与实际路径不同。"},
    {
        "flag": "--config-format",
        "zh": "模型配置格式。",
        "choices": ["auto", "hf", "mistral"],
    },
    {"flag": "--hf-token", "zh": "访问 Hugging Face 远程文件的 token。", "remote_only": True},
    {"flag": "--hf-overrides", "zh": "传给 Hugging Face config 的覆盖项，通常是 JSON。"},
    {"flag": "--pooler-config", "zh": "pooling 模型的 pooler 配置 JSON。"},
    {
        "flag": "--generation-config",
        "zh": "generation config 路径；auto 从模型读取，vllm 使用 vLLM 默认。",
    },
    {"flag": "--override-generation-config", "zh": "覆盖 generation config 的 JSON。"},
    {
        "flag": "--enable-sleep-mode",
        "zh": "启用 sleep mode；仅 cuda/hip 平台支持。",
        "kind": "bool",
    },
    {
        "flag": "--enable-cumem-allocator",
        "zh": "启用 cumem allocator，支持更高级 GPU 内存分配。",
        "kind": "bool",
    },
    {
        "flag": "--model-impl",
        "zh": "选择模型实现；auto 会优先 vLLM 实现再回退 Transformers。",
        "choices": ["auto", "terratorch", "transformers", "vllm"],
    },
    {"flag": "--override-attention-dtype", "zh": "覆盖 attention 使用的数据类型。"},
    {"flag": "--logits-processors", "zh": "一个或多个 logits processor 的完整类名。"},
    {"flag": "--io-processor-plugin", "zh": "启动模型时加载的 IOProcessor 插件名。"},
    {"flag": "--renderer-num-workers", "zh": "renderer 线程池 worker 数。"},
]


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
MENU_MODEL_ROOT = "Set local model directory and select model"
MENU_STATUS = "Show container status"
MENU_LOGS = "Tail container logs"
MENU_STOP = "Stop and remove container"
MENU_QUIT = "Quit"


def main_menu(
    local_model_root: Optional[str] = None,
    selected_local_model: Optional[str] = None,
) -> str:
    message = "What do you want to do?"
    if local_model_root:
        message += f"  [model dir: {local_model_root}]"
    if selected_local_model:
        message += f" [model: {selected_local_model}]"
    choice = questionary.select(
        message,
        choices=[
            MENU_SERVE,
            MENU_BENCH,
            MENU_MODEL_ROOT,
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


def _prompt_vllm_serve_extra_args(is_local_model: bool) -> list[str]:
    available_options = [
        option
        for option in VLLM_MODEL_CONFIG_OPTIONS
        if not (is_local_model and option.get("remote_only"))
    ]
    choices = [
        questionary.Choice(
            title=f"{option['flag']} - {option['zh']}",
            value=option["flag"],
        )
        for option in available_options
    ]
    choices.append(
        questionary.Choice(
            title="自定义原始参数 - 手动输入其它 vllm serve 参数",
            value=CUSTOM_SERVE_ARGS,
        )
    )

    selected_flags = questionary.checkbox(
        "Extra `vllm serve` args (space to select, enter to continue)",
        choices=choices,
    ).ask() or []

    option_by_flag = {option["flag"]: option for option in available_options}
    args: list[str] = []
    for flag in selected_flags:
        if flag == CUSTOM_SERVE_ARGS:
            raw = _ask_text("Custom extra args", default="")
            if raw:
                args.extend(shlex.split(raw))
            continue

        option = option_by_flag[flag]
        if option.get("kind") == "bool":
            args.append(flag)
            continue

        if option.get("choices"):
            value = questionary.select(
                f"{flag} - {option['zh']}",
                choices=option["choices"],
                default=option["choices"][0],
            ).ask()
        else:
            value = _ask_text(f"{flag} - {option['zh']}", default="")
        if value:
            args.extend([flag, value])
    return args


def _looks_like_model_dir(path: Path) -> bool:
    if not (path / MODEL_CONFIG_FILE).is_file():
        return False
    return any(
        child.is_file() and child.suffix.lower() in MODEL_WEIGHT_SUFFIXES
        for child in path.iterdir()
    )


def _discover_model_dirs(root: Path) -> list[Path]:
    model_dirs: set[Path] = set()
    for config_file in root.rglob(MODEL_CONFIG_FILE):
        model_dir = config_file.parent
        try:
            if _looks_like_model_dir(model_dir):
                model_dirs.add(model_dir)
        except OSError:
            continue
    return sorted(model_dirs, key=lambda p: str(p.relative_to(root)).lower())


def _container_model_path(model_dir: Path, mount_root: Path) -> str:
    relative = model_dir.relative_to(mount_root)
    container_path = Path(DEFAULT_CONTAINER_MODEL_ROOT)
    if str(relative) != ".":
        container_path /= relative
    return container_path.as_posix()


def _local_model_label(model: str, root: Optional[str]) -> str:
    if root and model.startswith(DEFAULT_CONTAINER_MODEL_ROOT):
        relative = model.removeprefix(DEFAULT_CONTAINER_MODEL_ROOT).lstrip("/")
        return relative or Path(root).name
    return model


def prompt_local_model_root(default: Optional[str] = None) -> str:
    while True:
        root_raw = _ask_text("Local model root directory", default=default or "")
        if not root_raw:
            console.print("[red]Local model root directory is required.[/red]")
            continue
        root = Path(root_raw).expanduser()
        if root.is_dir():
            return str(root.resolve())
        console.print(f"[red]{root_raw!r} is not a directory.[/red]")


def _select_model_from_root(root: Path) -> tuple[str, str]:
    model_dirs = _discover_model_dirs(root)
    if not model_dirs:
        console.print(
            "[yellow]No model directories found under this directory.[/yellow]"
        )
        raise KeyboardInterrupt

    console.print(f"[cyan]Found {len(model_dirs)} local model(s) under {root}.[/cyan]")
    choices = [
        questionary.Choice(
            title=str(model_dir.relative_to(root)),
            value=model_dir,
        )
        for model_dir in model_dirs
    ]
    selected = questionary.select("Select local model", choices=choices).ask()
    if selected is None:
        raise KeyboardInterrupt
    mount_root = root.resolve()
    return _container_model_path(selected.resolve(), mount_root), str(mount_root)


def prompt_local_model(default_root: Optional[str] = None) -> tuple[str, str]:
    root = Path(prompt_local_model_root(default=default_root))
    return _select_model_from_root(root)


def _prompt_manual_model(default_model: Optional[str] = None) -> str:
    model = _ask_text("Model (HF id or local path)", default=default_model or "")
    while not model:
        console.print("[red]Model is required.[/red]")
        model = _ask_text("Model (HF id or local path)", default="")
    return model


def _prompt_model(
    default_model: Optional[str] = None,
    local_model_root: Optional[str] = None,
    selected_local_model: Optional[str] = None,
) -> tuple[str, Optional[str]]:
    use_selected_choice = "Use selected local model"
    local_choice = "Select local model from configured directory"
    change_choice = "Change local model directory and select model"
    manual_choice = "Enter HF id or path manually"
    choices = []
    if selected_local_model and local_model_root:
        label = _local_model_label(selected_local_model, local_model_root)
        choices.append(questionary.Choice(f"{use_selected_choice} ({label})", use_selected_choice))
        choices.append(local_choice)
        choices.append(change_choice)
    elif local_model_root:
        choices.append(questionary.Choice(f"{local_choice} ({local_model_root})", local_choice))
        choices.append(change_choice)
    else:
        choices.append(questionary.Choice("Set local model directory and select model", change_choice))
    choices.append(manual_choice)

    method = questionary.select(
        "Model",
        choices=choices,
    ).ask() or manual_choice

    if method == use_selected_choice and selected_local_model and local_model_root:
        return selected_local_model, local_model_root

    if method == manual_choice:
        return _prompt_manual_model(default_model=default_model), None

    try:
        if method == change_choice or not local_model_root:
            return prompt_local_model(default_root=local_model_root)
        return _select_model_from_root(Path(local_model_root))
    except KeyboardInterrupt:
        console.print("[yellow]Falling back to manual model input.[/yellow]")
        return _prompt_manual_model(default_model=default_model), None


def prompt_serve_config(
    default_model: Optional[str] = None,
    local_model_root: Optional[str] = None,
    selected_local_model: Optional[str] = None,
) -> ServeConfig:
    image = _ask_text("Docker image", default=DEFAULT_IMAGE)
    container_name = _ask_text("Container name", default=DEFAULT_CONTAINER_NAME)
    model, model_mount_dir = _prompt_model(
        default_model=default_model,
        local_model_root=local_model_root,
        selected_local_model=selected_local_model,
    )
    host_port = _ask_int("Host port", default=DEFAULT_HOST_PORT)
    gpus = _ask_text("GPUs (all / count / none)", default="all")
    if model_mount_dir:
        hf_cache_dir = ""
        hf_token = None
    else:
        hf_cache_dir = _ask_text("Host HF cache dir", default=DEFAULT_HF_CACHE)
        hf_token = _ask_text("HF token (blank if not needed)", default="") or None
    extra_args = _prompt_vllm_serve_extra_args(is_local_model=bool(model_mount_dir))
    return ServeConfig(
        image=image,
        container_name=container_name,
        model=model,
        host_port=host_port,
        container_port=8000,
        gpus=gpus,
        hf_cache_dir=hf_cache_dir,
        hf_token=hf_token,
        model_mount_dir=model_mount_dir,
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
