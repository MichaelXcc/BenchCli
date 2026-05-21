# BenchCli

An interactive CLI for working with vLLM containers. Wraps the common workflows
of starting a vLLM inference server in Docker and running `vllm bench serve`
against it, with a Claude-style interactive menu.

## Features

- Start / stop / inspect a vLLM Docker container with a chosen image and model.
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

## Requirements

- Docker daemon reachable from the current user.
- A vLLM image available locally or pullable (default: `vllm/vllm-openai:latest`).
- An NVIDIA GPU + nvidia-container-toolkit if you want GPU inference.
