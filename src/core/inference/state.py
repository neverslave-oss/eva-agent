"""Single source of truth for the model server's shared mutable state.

Extracted from the original model_server.py monolith (refactor). All engine
code reads and writes this module instead of declaring `global _X` and reaching
across a 3.7k-line file. Keeping it as plain module attributes (mirroring the
original globals 1:1) makes the migration mechanical and low-risk: each accessor
in the engine becomes `state.model`, `state.processor`, etc.

The vLLM engine owns its own state inside vllm.py; this holder covers the
transformers/HF path, the model-slot registry, config, locking, adapters, and
backend capability flags.
"""

import threading

# --- config / startup -------------------------------------------------------
config = None
lazy_config_path = "config.yaml"          # set at startup; used by lazy loads
lazy_model_override = None                # set at startup via --model

# --- main HF model ----------------------------------------------------------
model = None
processor = None          # AutoProcessor (Gemma/Qwen) or AutoTokenizer (Nemotron)
drafter = None
drafter_tokenizer = None

# --- backend capability flags ----------------------------------------------
is_nemotron = False
nemotron_mode = "linear_spec"             # ar | diffusion | linear_spec
nemotron_block_length = None
nemotron_threshold = None
model_supports_tools = True               # True only for Gemma 4+ native parse_response
audio_capable = False                     # main model natively handles audio
native_agentic = False                    # main model handles the full agentic flow
is_omni = False                           # Qwen2.5-Omni (any-to-any)
is_janus = False                          # DeepSeek Janus/Janus-Pro
janus_processor = None                    # VLChatProcessor (carries tokenizer)

# --- multimodal / tool-calling slots ---------------------------------------
mm_model = None
mm_processor = None
tool_calling_model = None
tool_calling_processor = None
tool_calling_slot_loaded = False

# --- slot registry + locking ------------------------------------------------
slot_registry = None
load_lock = threading.Lock()
infer_lock = threading.RLock()

# --- LoRA adapters ----------------------------------------------------------
loaded_adapters: dict[str, str] = {}
current_adapter_name: str | None = None
