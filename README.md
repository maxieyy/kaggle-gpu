# Fresh-Install `llama.cpp` CUDA Build for Kaggle Notebook (Qwen3 Thinking + Cloudflare Tunnel)

This notebook is the **fresh-install** version for a new Kaggle notebook. It:

1. Clones [`llama.cpp`](https://github.com/ggml-org/llama.cpp) from GitHub and compiles the **CUDA server from source** — no precompiled runtime Dataset is used.
2. Downloads **Qwen3 Thinking** from Hugging Face.
3. Serves it through **OpenAI-compatible** and **Anthropic-compatible** endpoints.
4. Connects a **Cloudflare Tunnel**.

The build settings follow the official [`llama.cpp` CUDA build guide](https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md).

---

## Before running any cells

In Kaggle, create a new notebook and configure:

| Setting | Value |
|---|---|
| **Accelerator** | GPU T4 × 2 |
| **Internet** | On |
| **Secrets** | `LLAMA_API_KEY` and `CF_TUNNEL_TOKEN`, both enabled for notebook access |

We use **`Qwen3-30B-A3B-Thinking-2507-Q3_K_S.gguf`**, listed at **13.292 GB**. This quantization fits the download/storage budget better than `Q4_K_M`, though with a greater quality trade-off.

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

# Verify that Kaggle gave this notebook GPU access.
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

## Cell 3 — Clone and compile `llama.cpp`

This compiles specifically for the Tesla T4's CUDA compute capability **7.5**. It uses **two build jobs** to limit memory pressure.

After compiling, it keeps the runnable server and its shared libraries, and removes the source/build intermediates to free disk space for the model.

```python
import os
import shutil
import subprocess
from pathlib import Path

REPO_DIR = Path("/kaggle/working/llama.cpp")
BUILD_DIR = REPO_DIR / "build"
BUILD_BIN = BUILD_DIR / "bin"
RUNTIME_DIR = Path("/kaggle/working/llama-runtime")
RUNTIME_BIN = RUNTIME_DIR / "bin"
RUNTIME_SERVER = RUNTIME_BIN / "llama-server"

# Clone if not already present.
if not REPO_DIR.is_dir():
    subprocess.run(
        [
            "git", "clone", "--depth", "1",
            "https://github.com/ggml-org/llama.cpp.git",
            str(REPO_DIR),
        ],
        check=True,
    )

# Configure the CUDA build for T4 (compute capability 7.5).
subprocess.run(
    [
        "cmake",
        "-S", str(REPO_DIR),
        "-B", str(BUILD_DIR),
        "-DGGML_CUDA=ON",
        "-DCMAKE_CUDA_ARCHITECTURES=75",
        "-DLLAMA_CURL=ON",
        "-DCMAKE_BUILD_TYPE=Release",
    ],
    check=True,
)

# Build with two jobs to limit memory pressure on the T4 host.
subprocess.run(
    [
        "cmake",
        "--build", str(BUILD_DIR),
        "--config", "Release",
        "--target", "llama-server",
        "-j", "2",
    ],
    check=True,
)

built_server = BUILD_BIN / "llama-server"
assert built_server.is_file(), (
    f"Compilation succeeded but server is missing: {built_server}"
)

# Verify the compiled binary before removing build intermediates.
build_env = os.environ.copy()
build_env["LD_LIBRARY_PATH"] = (
    str(BUILD_BIN)
    + os.pathsep
    + build_env.get("LD_LIBRARY_PATH", "")
)

subprocess.run(
    [str(built_server), "--version"],
    check=True,
    env=build_env,
)

subprocess.run(
    [str(built_server), "--list-devices"],
    check=True,
    env=build_env,
)

# Move the complete runtime directory, preserving the libraries
# and symlinks beside the executable.
shutil.rmtree(RUNTIME_DIR, ignore_errors=True)
RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
shutil.move(str(BUILD_BIN), str(RUNTIME_BIN))

# Free the repository, object files, and intermediate build files.
shutil.rmtree(REPO_DIR, ignore_errors=True)

LLAMA_SERVER = RUNTIME_SERVER
assert LLAMA_SERVER.is_file(), "llama-server was not created."

os.environ["LD_LIBRARY_PATH"] = (
    str(RUNTIME_BIN)
    + os.pathsep
    + os.environ.get("LD_LIBRARY_PATH", "")
)

print("\nFresh CUDA build ready:", LLAMA_SERVER)
```

> If compilation fails, the cell stops and shows the CMake or compiler error. **Fix that error before moving on.**

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

The selected public repository is **`tensorblock/Qwen_Qwen3-30B-A3B-Thinking-2507-GGUF`**. The exact `Q3_K_S` filename is listed in its model files.

```python
import os
import sys
import subprocess
from pathlib import Path
import shutil

try:
    from huggingface_hub import hf_hub_download
except ImportError:
    subprocess.run(
        [
            sys.executable, "-m", "pip", "install", "-q",
            "huggingface_hub",
        ],
        check=True,
    )
    from huggingface_hub import hf_hub_download

MODEL_REPO = (
    "tensorblock/Qwen_Qwen3-30B-A3B-Thinking-2507-GGUF"
)
MODEL_FILE = "Qwen3-30B-A3B-Thinking-2507-Q3_K_S.gguf"
MODEL_DIR = Path("/kaggle/working/models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = MODEL_DIR / MODEL_FILE

free_gib = shutil.disk_usage(MODEL_DIR).free / (1024 ** 3)
print(f"Free space before download: {free_gib:.2f} GiB")

if not MODEL_PATH.is_file():
    if free_gib < 14:
        raise RuntimeError(
            "Not enough free disk space to download Q3_K_S safely."
        )
    print("Downloading Thinking GGUF from Hugging Face...")
    downloaded_path = hf_hub_download(
        repo_id=MODEL_REPO,
        filename=MODEL_FILE,
        local_dir=str(MODEL_DIR),
    )
    MODEL_PATH = Path(downloaded_path)

assert MODEL_PATH.is_file(), "Model download failed."

print("\nModel ready:", MODEL_PATH)
print(f"Model size: {MODEL_PATH.stat().st_size / (1024**3):.2f} GiB")
print(
    f"Remaining space: "
    f"{shutil.disk_usage(MODEL_DIR).free / (1024**3):.2f} GiB"
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

> **Note:** 256K is the *requested* context size, not a guarantee that it fits. If loading fails with CUDA out-of-memory, inspect `/tmp/llama-server.log`. The quantized cache reduces memory use, but the GPU allocation can still fail on two T4s.

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
