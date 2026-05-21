# OpenBench

An interactive CLI for working with vLLM containers. Wraps the common workflows
of starting a vLLM inference server in Docker and running `vllm bench serve`
against it, with a Claude-style interactive menu.

## Features

- Start / stop / inspect a vLLM Docker container with a chosen image and model.
- Scan a local model root directory in the interactive flow and select any
  discovered model directory under it.
- Run `vllm bench serve` inside the running container with guided parameters.
- Download benchmark datasets on the host into `datasets/`, then use them from
  inside the container through the workspace mount.
- Interactive main menu (arrow keys), or direct subcommands for scripting.

## Install (dev)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

If the environment cannot download build dependencies such as `hatchling`,
install through the legacy editable path instead. If both `setuptools` and
`wheel` are already available:

```bash
pip install -e . --no-use-pep517
```

If `wheel` is not installed but `setuptools` is available:

```bash
python setup.py develop
```

If dependency downloads are also blocked, install the runtime dependencies from
a reachable mirror or wheel directory first, then run one of the commands above.

Then run:

```bash
openbench           # interactive menu
openbench serve     # start vLLM server
openbench bench     # run `vllm bench serve` inside the container
openbench download-dataset  # download a benchmark dataset to ./datasets
openbench status    # show current container
openbench stop      # stop and remove the container
```

The legacy `benchcli` console command is still installed as a compatibility
alias, but new usage should prefer `openbench`.

In the interactive menu, choose `Set local model directory and select model` to
set or change the host directory used to discover local models. OpenBench then
shows every discovered model under that directory and asks which one to use.
When starting a server, the `Model` prompt lets you use that selected local
model, choose another model from the same directory, change the directory, or
enter an HF id/path manually. OpenBench detects model directories that contain
`config.json` and a weight file such as `.safetensors`, `.bin`, or `.gguf`,
mounts the chosen root at `/models` inside the container, and passes the
selected model's container path to `vllm serve`.

The `Extra vllm serve args` step shows selectable ModelConfig options from the
vLLM serve CLI docs with Chinese descriptions. Select one or more options, then
OpenBench will ask for the required value for each option. For local models, HF
cache and HF token prompts are skipped.
OpenBench starts the vLLM image with `--model <model>` and includes options such
as `--tensor-parallel-size` and `--gpu-memory-utilization` in the selectable
extra arguments.
During interactive prompts, enter `:back` in text or number fields to return to
the previous step. Selection prompts include a `← Back` option.
For the GPU prompt, enter `all` for every GPU, `none` for CPU/no GPU, a count
such as `2`, or explicit GPU ids such as `0` or `0,1`.

When a vLLM container is started, OpenBench mounts the current working directory
into the container at `/workspace/openbench`. Benchmark dataset downloads are
stored under `./datasets` on the host, so they are visible inside the container
as `/workspace/openbench/datasets/...`. The `Run vllm bench serve` flow shows the
resolved benchmark configuration and command before executing it in the running
container.
For older vLLM images that do not include `vllm bench serve`, OpenBench probes the
container and can use the legacy-compatible command
`python -m vllm.benchmarks.benchmark_serving` instead. You can also enter a
custom benchmark command in the interactive benchmark flow.

## Requirements

- Docker daemon reachable from the current user.
- A vLLM image available locally or pullable (default: `vllm/vllm-openai:latest`).
- An NVIDIA GPU + nvidia-container-toolkit if you want GPU inference.
