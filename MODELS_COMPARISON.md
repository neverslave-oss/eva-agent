I now have detailed model card information. Let me summarize the key capabilities:

Gemma 4 E2B-it (2.3B effective):

Modalities: Text, Image, Audio (E2B/E4B/12B support audio)
Native function calling / tool use
128K context, 35 layers
Vision encoder ~150M, Audio encoder ~300M
Thinking mode, coding, agentic
Qwen3.5-0.8B:

Causal LM with Vision Encoder (image-text-to-text)
0.8B params, 262K context
Multimodal (vision + text), tool calling
Note: this is the tool_calling slot model
Qwen2.5-Omni-3B (6B params total):

Any-to-Any multimodal: text, images, audio, video
Generates text AND natural speech responses (streaming)
Thinker-Talker architecture, TMRoPE
Real-time voice/video chat
Strong across all modalities; outperforms Qwen2-Audio in audio, comparable to Qwen2.5-VL-7B in vision
Can disable talker to save ~2GB VRAM (text-only)
Audio output via speaker param
This is the true "omni" model — vision + audio + text + speech generation
Janus-Pro-7B:

Unified understanding AND generation MLLM
Decoupled visual encoding (SigLIP-L vision encoder for understanding, LlamaGen tokenizer for generation)
Can generate images (text-to-image) AND understand images
Based on DeepSeek-LLM-7b-base
Multimodal: understanding + image generation — not just vision understanding
Nemotron-Labs-Diffusion-3B (4B params):

Tri-mode language model: AR decoding, diffusion parallel decoding, self-speculation
Text generation only (no vision/audio)
Optimized for decode efficiency/speed
Now I have the model capabilities. This is a large task. Given the complexity (building a benchmark harness across 4 modalities and multiple models, with a 4090 GPU), let me be strategic.

The key constraint: the model server is currently running in lazy mode with Gemma (audio slot) as the thought slot. To benchmark other models, I need to use the model server's slot system and inference methods. The models available and their capabilities:

Model	Text	Tools	Vision	Audio	Image-gen
Gemma 4 E2B-it	✅	✅	✅	✅	❌
Qwen3.5-0.8B	✅	✅	✅	❌	❌
Qwen2.5-Omni-3B	✅	✅	✅	✅	❌ (video+speech)
Janus-Pro-7B	✅	❌	✅	❌	✅
Nemotron-Diffusion-3B	✅	❌	❌	❌	❌
