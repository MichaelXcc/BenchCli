# BenchCli

An interactive CLI for working with vLLM containers. Wraps the common workflows
of starting a vLLM inference server in Docker and running `vllm bench serve`
against it, with a Claude-style interactive menu.

## Features

- Start / stop / inspect a vLLM Docker container with a chosen image and model.
- Scan a local model root directory in the interactive flow and select any
  discovered model directory under it.
- Run `vllm bench serve` inside the running container with guided parameters.
- Interactive main menu (arrow keys), or direct subcommands for scripting.

## Install (dev)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Then run:

```bash
benchcli           # interactive menu
benchcli serve     # start vLLM server
benchcli bench     # run `vllm bench serve` inside the container
benchcli status    # show current container
benchcli stop      # stop and remove the container
```

In the interactive menu, choose `Set local model directory and select model` to
set or change the host directory used to discover local models. BenchCli then
shows every discovered model under that directory and asks which one to use.
When starting a server, the `Model` prompt lets you use that selected local
model, choose another model from the same directory, change the directory, or
enter an HF id/path manually. BenchCli detects model directories that contain
`config.json` and a weight file such as `.safetensors`, `.bin`, or `.gguf`,
mounts the chosen root at `/models` inside the container, and passes the
selected model's container path to `vllm serve`.

The `Extra vllm serve args` step shows selectable ModelConfig options from the
vLLM serve CLI docs with Chinese descriptions. Select one or more options, then
BenchCli will ask for the required value for each option. For local models, HF
cache and HF token prompts are skipped.

## Requirements

- Docker daemon reachable from the current user.
- A vLLM image available locally or pullable (default: `vllm/vllm-openai:latest`).
- An NVIDIA GPU + nvidia-container-toolkit if you want GPU inference.
