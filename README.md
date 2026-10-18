# Self-host Qwen3.8-27B-Turbo-Brilliance-50X-Reasoning-Instruct-NEO-MAX-MTP-GGUF on Kaggle T4 × 2 GPU

This notebook is the **fresh-install** version for a new Kaggle notebook. It:

1. Pulls Pre-complied llama-server.
2. Downloads **Qwen3.8-27B-Turbo-Brilliance-50X-Reasoning-Instruct-NEO-MAX-MTP-GGUF** by `DavidAU` from `Hugging Face`.
3. Serves it through **OpenAI-compatible** and **Anthropic-compatible** endpoints.
4. Connects a **Cloudflare Tunnel**.




---

## Before running any cells

In Kaggle, create a new notebook and configure:

| Setting | Value |
|---|---|
| **Accelerator** | GPU T4 × 2 |
| **Internet** | On |
| **Secrets** | `LLAMA_API_KEY` and `CF_TUNNEL_TOKEN`, both enabled for notebook access |

We use **`Qwen3.8-27B-Turbo-Brilliance-50X-Reasoning-Instruct-NEO-MAX-MTP-GGUF`**, listed at **17.1 GB**. This quantization fits the download/storage budget, though with a greater quality trade-off. Tested on GPU T4 × 2 and produced ~10-20 tokens/sec on Small tasks on full 256K context and 1 parallel. More than 1 parallel and it crashes. Output is brilliance  but on Long running process model seems to "Hallucinate" it keeps repeating itself in loop.

---

## Cell 1 — Check GPU and CUDA

```python
import os
import sys
import shutil
import subprocess
from pathlib import Path

print("Python:", sys.version)
print("Working directory:", Path("/kaggle/working"))


subprocess.run(["nvidia-smi"], check=True)

NVCC = shutil.which("nvcc")
if not NVCC:
    raise RuntimeError(
        "CUDA compiler nvcc was not found. "
        "Enable GPU T4 x2 in Kaggle Notebook Settings."
    )

subprocess.run([NVCC, "--version"], check=True)
print("\nGPU and CUDA checks passed.")
```

You should see **two Tesla T4 GPUs** and the installed CUDA compiler.

---

## Cell 2 — Install build dependencies

```python
import os
import subprocess

apt_env = os.environ.copy()
apt_env["DEBIAN_FRONTEND"] = "noninteractive"

subprocess.run(
    ["apt-get", "update", "-qq"],
    check=True,
    env=apt_env,
)

subprocess.run(
    [
        "apt-get", "install", "-y", "-qq",
        "build-essential",
        "cmake",
        "git",
        "pkg-config",
        "libcurl4-openssl-dev",
        "libssl-dev",
        "curl",
    ],
    check=True,
    env=apt_env,
)

subprocess.run(["cmake", "--version"], check=True)
subprocess.run(["git", "--version"], check=True)
print("Build dependencies installed.")
```

> **Note:** You do **not** need Ollama, Docker, or the `zstd` package for this setup.

---

## Cell 3 — Clone compiled `llama.cpp`

This compiles specifically for the Tesla T4's CUDA compute capability **7.5**. It's already prebuilt



```python
import os
import shutil
import subprocess
from pathlib import Path

SD_DIR = Path("/kaggle/working/stable-diffusion.cpp")
SD_BUILD = SD_DIR / "build"
BUILD_BIN = SD_BUILD / "bin"

RUNTIME_ROOT = Path("/kaggle/working/sdcpp-runtime")
RUNTIME_BIN = RUNTIME_ROOT / "bin"
SD_SERVER = RUNTIME_BIN / "sd-server"

shutil.rmtree(SD_DIR, ignore_errors=True)

subprocess.run(
    [
        "git", "clone", "--recursive",
        "https://github.com/leejet/stable-diffusion.cpp",
        str(SD_DIR),
    ],
    check=True,
)
commit_date = subprocess.run(
    ["git", "log", "-1", "--format=%cs"],
    cwd=SD_DIR,
    capture_output=True,
    text=True,
    check=True,
).stdout.strip()

commit_hash = subprocess.run(
    ["git", "rev-parse", "--short", "HEAD"],
    cwd=SD_DIR,
    capture_output=True,
    text=True,
    check=True,
).stdout.strip()

print("Source commit:", commit_hash)
print("Commit date:", commit_date)

if commit_date < "2026-10-06":
    raise RuntimeError(
        "The checked-out source is older than 2026-10-06. "
        "Update stable-diffusion.cpp before building."
    )

subprocess.run(
    [
        "cmake",
        "-S", str(SD_DIR),
        "-B", str(SD_BUILD),
        "-DSD_CUDA=ON",
        "-DSD_BUILD_EXAMPLES=ON",
        "-DSD_SERVER_BUILD_FRONTEND=OFF",
        "-DCMAKE_BUILD_TYPE=Release",
        "-DCMAKE_CUDA_ARCHITECTURES=75",
    ],
    check=True,
)

subprocess.run(
    [
        "cmake", "--build", str(SD_BUILD),
        "--config", "Release",
        "--target", "sd-server",
        "-j", "2",
    ],
    check=True,
)

built_server = BUILD_BIN / "sd-server"
assert built_server.is_file(), f"Missing build output: {built_server}"

build_env = os.environ.copy()
build_env["LD_LIBRARY_PATH"] = (
    str(BUILD_BIN) + os.pathsep
    + build_env.get("LD_LIBRARY_PATH", "")
)

subprocess.run(
    [str(built_server), "--help"],
    env=build_env,
    check=False,
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL,
)

shutil.rmtree(RUNTIME_ROOT, ignore_errors=True)
RUNTIME_BIN.mkdir(parents=True, exist_ok=True)

shutil.copytree(
    BUILD_BIN,
    RUNTIME_BIN,
    symlinks=True,
    dirs_exist_ok=True,
)

SD_SERVER = RUNTIME_BIN / "sd-server"
assert SD_SERVER.is_file(), "sd-server was not restored."

runtime_env = os.environ.copy()
runtime_env["LD_LIBRARY_PATH"] = (
    str(RUNTIME_BIN) + os.pathsep
    + runtime_env.get("LD_LIBRARY_PATH", "")
)
os.environ["LD_LIBRARY_PATH"] = runtime_env["LD_LIBRARY_PATH"]

deps = subprocess.run(
    ["ldd", str(SD_SERVER)],
    capture_output=True,
    text=True,
    env=runtime_env,
)

dependency_output = deps.stdout + deps.stderr
print(dependency_output)

if "not found" in dependency_output or "file too short" in dependency_output:
    raise RuntimeError("The compiled server has missing or broken libraries.")

subprocess.run(
    [str(SD_SERVER), "--help"],
    env=runtime_env,
    check=True,
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL,
)
shutil.rmtree(SD_DIR, ignore_errors=True)

print("\nCUDA image server compiled successfully.")
print("Server:", SD_SERVER)
print(
    "Free disk space:",
    f"{shutil.disk_usage('/kaggle/working').free / (1024**3):.2f} GiB",
)
```

---

## Cell 4 — Verify the compiled server and disk space

```python
import os
import shutil
import subprocess

runtime_env = os.environ.copy()
runtime_env["LD_LIBRARY_PATH"] = (
    str(RUNTIME_BIN)
    + os.pathsep
    + runtime_env.get("LD_LIBRARY_PATH", "")
)

subprocess.run(
    [str(LLAMA_SERVER), "--version"],
    check=True,
    env=runtime_env,
)

subprocess.run(
    [str(LLAMA_SERVER), "--list-devices"],
    check=True,
    env=runtime_env,
)

disk = shutil.disk_usage("/kaggle/working")
free_gib = disk.free / (1024 ** 3)
print(f"Free working-directory space: {free_gib:.2f} GiB")

if free_gib < 14:
    raise RuntimeError(
        "Less than 14 GiB of free space is available. "
        "Free storage before downloading the selected GGUF."
    )
```

This cell verifies that the build can run and that **both CUDA devices** are visible, then checks for sufficient space for the selected model.

---

## Cell 5 — Download the Thinking model

The selected public repository is **`DavidAU/Qwen3.8-27B-Turbo-Brilliance-50X-Reasoning-Instruct-NEO-MAX-MTP-GGUF`**. The exact `Q4_K_M` filename is listed in its model files.

```python
import os
import sys
import shutil
import subprocess
import importlib.util
from pathlib import Path

MODEL_DIR = Path("/kaggle/working/models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_REPO = (
    "DavidAU/"
    "Qwen3.8-27B-Turbo-Brilliance-50X-Reasoning-Instruct-NEO-MAX-MTP-GGUF"
)

MODEL_FILE = (
    "Qwen3.8-27B-Turbo-Brilliance-50X-RI-NEO-MAX-MTP-Q4_K_M.gguf"
)

MODEL_PATH = MODEL_DIR / MODEL_FILE

# Disable Xet's separate chunk cache to reduce additional disk consumption.
os.environ["HF_HUB_DISABLE_XET"] = "1"

if importlib.util.find_spec("huggingface_hub") is None:
    subprocess.run(
        [
            sys.executable, "-m", "pip", "install",
            "--no-cache-dir", "huggingface_hub"
        ],
        check=True
    )

from huggingface_hub import hf_hub_download

free_gib = shutil.disk_usage(MODEL_DIR).free / (1024**3)
print(f"Available before model download: {free_gib:.2f} GiB")

if free_gib < 17.5 and not MODEL_PATH.exists():
    raise RuntimeError(
        "Not enough free disk for this download. "
        "Use the smaller Q4_K_M variant shown in the fallback below."
    )

if MODEL_PATH.exists() and MODEL_PATH.stat().st_size > 16 * 1024**3:
    print("Model file already exists; skipping download.")
else:
    print("Downloading the Turbo Brilliance Q4_K_M model...")
    print("Keep this notebook session running until the download finishes.")

    downloaded_path = hf_hub_download(
        repo_id=MODEL_REPO,
        filename=MODEL_FILE,
        local_dir=str(MODEL_DIR)
    )

    MODEL_PATH = Path(downloaded_path)

if not MODEL_PATH.exists():
    raise RuntimeError("Model file was not found after downloading.")

size_gib = MODEL_PATH.stat().st_size / (1024**3)
print(f"\nModel saved: {MODEL_PATH}")
print(f"Model size: {size_gib:.2f} GiB")
print(
    "Remaining disk:",
    f"{shutil.disk_usage('/kaggle/working').free / (1024**3):.2f} GiB"
)

if size_gib < 16:
    raise RuntimeError(
        "Downloaded file is unexpectedly small. "
        "Check whether the download completed."
    )
```

---

## Cell 6 — Load Kaggle Secrets

```python
from kaggle_secrets import UserSecretsClient

secrets = UserSecretsClient()
LLAMA_API_KEY = secrets.get_secret("LLAMA_API_KEY")
CF_TUNNEL_TOKEN = secrets.get_secret("CF_TUNNEL_TOKEN")

assert LLAMA_API_KEY, "LLAMA_API_KEY is missing in Kaggle Secrets."
assert CF_TUNNEL_TOKEN, "CF_TUNNEL_TOKEN is missing in Kaggle Secrets."
print("API key and Cloudflare Tunnel token loaded.")
```

---

## Cell 7 — Launch Qwen with 256K context

The server exposes both **OpenAI-compatible routes** and **Anthropic's Messages API**. Current `llama.cpp` supports quantized K/V caches, Flash Attention, Jinja templates, and API-key authentication.

```python
import os
import subprocess
import time
import requests
from pathlib import Path

assert LLAMA_SERVER.is_file()
assert MODEL_PATH.is_file()
assert LLAMA_API_KEY

# Stop a previous server tracked by this notebook if this cell is rerun.
try:
    if llama_proc.poll() is None:
        llama_proc.terminate()
        llama_proc.wait(timeout=10)
except (NameError, Exception):
    pass

SERVER_LOG = Path("/tmp/llama-server.log")
server_log_handle = open(SERVER_LOG, "w")

server_env = os.environ.copy()
server_env["LD_LIBRARY_PATH"] = (
    str(RUNTIME_BIN)
    + os.pathsep
    + server_env.get("LD_LIBRARY_PATH", "")
)

llama_proc = subprocess.Popen(
    [
        str(LLAMA_SERVER),
        "--model", str(MODEL_PATH),
        "--alias", "qwen-thinking",
        "--host", "127.0.0.1",
        "--port", "8080",
        "--ctx-size", "262144",
        "--n-gpu-layers", "999",
        "--flash-attn",
        "--cache-type-k", "q8_0",
        "--cache-type-v", "q8_0",
        "--jinja",
        "--api-key", LLAMA_API_KEY,
    ],
    stdout=server_log_handle,
    stderr=subprocess.STDOUT,
    env=server_env,
    start_new_session=True,
)

# Wait for the server to become reachable.
deadline = time.time() + 600
while time.time() < deadline:
    if llama_proc.poll() is not None:
        print(SERVER_LOG.read_text(errors="replace")[-10000:])
        raise RuntimeError("llama-server exited during startup.")
    try:
        r = requests.get("http://127.0.0.1:8080/health", timeout=5)
        if r.status_code == 200:
            print("llama-server is up.")
            break
    except requests.RequestException:
        pass
    time.sleep(5)
else:
    raise RuntimeError("Timed out waiting for llama-server to start.")
```

> **Note:** 256K is the *requested* context size, not a guarantee that it fits. If loading fails with CUDA out-of-memory, inspect `/tmp/llama-server.log`. The quantized cache reduces memory use, but the GPU allocation can still fail on two T4s. Reduce to 128K and 1 parallel process.

---

## Cell 8 — Test the OpenAI-compatible API locally

```python
LOCAL_BASE_URL = "http://127.0.0.1:8080/v1"

openai_headers = {
    "Authorization": f"Bearer {LLAMA_API_KEY}",
    "Content-Type": "application/json",
}

response = requests.get(
    f"{LOCAL_BASE_URL}/models",
    headers=openai_headers,
    timeout=30,
)
print("HTTP status:", response.status_code)
response.raise_for_status()
print(response.json())
```

Run a test completion:

```python
response = requests.post(
    f"{LOCAL_BASE_URL}/chat/completions",
    headers=openai_headers,
    json={
        "model": "qwen-thinking",
        "messages": [
            {
                "role": "user",
                "content": "Write a Python binary search function."
            }
        ],
        "temperature": 0.6,
        "max_tokens": 300,
    },
    timeout=300,
)
print("HTTP status:", response.status_code)
response.raise_for_status()
print(response.json()["choices"][0]["message"]["content"])
```

---

## Cell 9 — Test the Anthropic-compatible API locally

```python
ANTHROPIC_URL = "http://127.0.0.1:8080/v1/messages"

anthropic_headers = {
    "x-api-key": LLAMA_API_KEY,
    "anthropic-version": "2023-06-01",
    "Content-Type": "application/json",
}

response = requests.post(
    ANTHROPIC_URL,
    headers=anthropic_headers,
    json={
        "model": "qwen-thinking",
        "max_tokens": 300,
        "messages": [
            {
                "role": "user",
                "content": "Explain binary search briefly."
            }
        ],
    },
    timeout=300,
)
print("HTTP status:", response.status_code)
response.raise_for_status()
print(response.json())
```

---

## Cell 10 — Install Cloudflare Tunnel

```python
import platform
import urllib.request
import subprocess
from pathlib import Path

architecture = platform.machine().lower()
if architecture in ("x86_64", "amd64"):
    binary_name = "cloudflared-linux-amd64"
elif architecture in ("aarch64", "arm64"):
    binary_name = "cloudflared-linux-arm64"
else:
    raise RuntimeError(f"Unsupported architecture: {architecture}")

CLOUDFLARED = Path("/kaggle/working/cloudflared")

url = (
    "https://github.com/cloudflare/cloudflared/"
    f"releases/latest/download/{binary_name}"
)
print("Downloading cloudflared...")
urllib.request.urlretrieve(url, CLOUDFLARED)
CLOUDFLARED.chmod(0o755)

subprocess.run(
    [str(CLOUDFLARED), "--version"],
    check=True,
)
print("cloudflared installed.")
```

---

## Cell 11 — Start Cloudflare Tunnel

> **Before this cell:** create or reuse a **named tunnel** in Cloudflare and configure a **published application route** for your hostname to `http://localhost:8080`.

```python
import os
import subprocess
import time
from pathlib import Path

assert CLOUDFLARED.is_file()
assert CF_TUNNEL_TOKEN

try:
    if tunnel_proc.poll() is None:
        tunnel_proc.terminate()
        tunnel_proc.wait(timeout=10)
except (NameError, Exception):
    pass

TUNNEL_LOG = Path("/tmp/cloudflared.log")
tunnel_log_handle = open(TUNNEL_LOG, "w")

tunnel_proc = subprocess.Popen(
    [
        str(CLOUDFLARED),
        "tunnel",
        "--no-autoupdate",
        "run",
        "--token",
        CF_TUNNEL_TOKEN,
    ],
    stdout=tunnel_log_handle,
    stderr=subprocess.STDOUT,
    start_new_session=True,
)

time.sleep(8)

if tunnel_proc.poll() is not None:
    tunnel_log_handle.flush()
    print(TUNNEL_LOG.read_text(errors="replace")[-10000:])
    raise RuntimeError("Cloudflare Tunnel exited during startup.")

print("Cloudflare Tunnel started.")
```

---

## Cell 12 — Test your public endpoints

> Replace the placeholder hostname with your actual Cloudflare hostname.

```python
PUBLIC_BASE_URL = "https://ai.yourdomain.com"

response = requests.get(
    f"{PUBLIC_BASE_URL}/health",
    timeout=30,
)
print("Public health status:", response.status_code)
print(response.text)
response.raise_for_status()
```

Test OpenAI-compatible chat:

```python
response = requests.post(
    f"{PUBLIC_BASE_URL}/v1/chat/completions",
    headers=openai_headers,
    json={
        "model": "qwen-thinking",
        "messages": [
            {
                "role": "user",
                "content": "Reply with: Qwen API online."
            }
        ],
        "max_tokens": 50,
        "temperature": 0.6,
    },
    timeout=300,
)
print("OpenAI API status:", response.status_code)
response.raise_for_status()
print(response.json()["choices"][0]["message"]["content"])
```

Test the Anthropic Messages endpoint:

```python
response = requests.post(
    f"{PUBLIC_BASE_URL}/v1/messages",
    headers=anthropic_headers,
    json={
        "model": "qwen-thinking",
        "max_tokens": 100,
        "messages": [
            {
                "role": "user",
                "content": "Reply with: Anthropic endpoint online."
            }
        ],
    },
    timeout=300,
)
print("Anthropic API status:", response.status_code)
response.raise_for_status()
print(response.json())
```

---

## Cell 13 — Optional process monitor

Only run this cell **after** the local and public API tests pass. It keeps the notebook execution active while the model server and tunnel are alive.

```python
import time
from pathlib import Path

print("Qwen API and Cloudflare Tunnel are running.")
print("Monitoring both processes.")

while True:
    if llama_proc.poll() is not None:
        print(
            Path("/tmp/llama-server.log").read_text(
                errors="replace"
            )[-10000:]
        )
        raise RuntimeError("llama-server stopped.")

    if tunnel_proc.poll() is not None:
        print(
            Path("/tmp/cloudflared.log").read_text(
                errors="replace"
            )[-10000:]
        )
        raise RuntimeError("Cloudflare Tunnel stopped.")

    time.sleep(60)
```

---

## Final order

| # | Step |
|---|---|
| 1 | Check GPU and CUDA |
| 2 | Install build dependencies |
| 3 | Clone and compile `llama.cpp` |
| 4 | Verify compiled server and disk |
| 5 | Download Qwen Thinking GGUF |
| 6 | Load Kaggle Secrets |
| 7 | Start `llama-server` with 256K context |
| 8 | Test local OpenAI API |
| 9 | Test local Anthropic API |
| 10 | Install `cloudflared` |
| 11 | Start Cloudflare Tunnel |
| 12 | Test both public endpoints |
| 13 | Optional process monitor |
