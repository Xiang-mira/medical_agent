# VoxTell / Qwen Text Encoder Architecture Note

## Current answer for teacher questions

### 1. Is Qwen frozen?

Yes. In the project fine-tuning path, Qwen is a frozen text encoder.

- `scripts/train_voxtell_prompt_student.py` computes all prompt embeddings in `build_prompt_embeddings()` under `@torch.inference_mode()`.
- The Qwen `AutoModel` is set to `eval()` and `requires_grad_(False)`.
- Qwen parameters are never passed into the optimizer. The optimizer is built only from `VoxTellModel` parameters with `requires_grad=True`.
- In inference, `third_party/VoxTell/voxtell/inference/predictor.py` also sets the text backbone to `eval()` and `requires_grad_(False)`.

If `--freeze-encoder` is used in the project trainer, that flag freezes the VoxTell image encoder parameters whose names start with `encoder.`. It does not change Qwen, because Qwen is already outside the trainable segmentation network.

### 2. What comes out after the prompt enters Qwen?

The prompt is not used as token IDs directly by VoxTell. It is converted into a pooled dense text vector.

Flow:

1. The raw prompt is wrapped as an instruction/query string:
   `Instruct: Given an anatomical term query, retrieve the precise anatomical entity and location it represents\nQuery: <prompt>`.
2. `AutoTokenizer` tokenizes the wrapped text with padding/truncation up to 8192 tokens.
3. `AutoModel` returns hidden states, including `last_hidden_state`.
4. `last_token_pool(last_hidden_state, attention_mask)` selects the last valid token representation.
5. The project stores one pooled embedding per prompt. With Qwen3-Embedding-4B, this is a 2560-dimensional vector, shaped as `(1, 1, 2560)` in the training cache for one prompt.

So the immediate output consumed by VoxTell is a pooled text embedding, not the full token sequence.

### 3. How does text feature fuse with CT image feature?

VoxTell uses a prompt-conditioned decoder with transformer cross-attention and mask-embedding fusion.

Detailed flow:

1. CT volume enters a `ResidualEncoder`, producing multi-scale 3D image features.
2. A selected encoder feature map, configured here as decoder layer 4, is reshaped from `(B, C, D, H, W)` to a sequence and projected to `query_dim=2048`.
3. The pooled Qwen embedding `(B, N, 2560)` is projected by `project_text_embed`: `2560 -> 2048 hidden -> query_dim`.
4. The projected text embeddings are used as transformer decoder target/query tokens.
5. The projected CT feature sequence is used as transformer memory, with 3D positional encoding.
6. `TransformerDecoderLayer` performs multi-head cross-attention from text query to image memory. The output is a fused text-image mask embedding for each prompt.
7. The fused mask embeddings are projected to decoder channel dimensions for multiple U-Net decoder stages.
8. During decoding, each prompt embedding conditions image features with `torch.einsum`. Intermediate stages concatenate prompt-conditioned fusion features back into the decoder; the final stage uses `einsum` to produce per-prompt mask logits.

This is not a simple "CT + prompt -> mask" black box. The prompt becomes a Qwen pooled embedding, then a trainable VoxTell text query, then a cross-attended mask embedding, then a multi-scale decoder conditioning vector.

### 4. Trainable vs frozen components in the current project trainer

Frozen:

- Qwen tokenizer/model/text encoder.
- Cached text embeddings are detached CPU tensors.

Trainable by default:

- VoxTell image encoder.
- VoxTell decoder.
- `project_bottleneck_embed`.
- `project_text_embed`.
- transformer decoder layers.
- `project_to_decoder_channels`.

Trainable when `--freeze-encoder` is enabled:

- VoxTell modules except parameters whose names start with `encoder.`.
- This still leaves text projection, transformer decoder, mask projection, and decoder-side conditioning trainable.

## Fixed implementation risk

The prompt embedding cache now stores metadata:

- text model name/path
- prompt hash
- cache format version
- text encoder policy

This prevents accidentally reusing embeddings produced by a different Qwen model or a different prompt set. Legacy caches can only be reused by explicitly setting `MEDAI_ALLOW_LEGACY_PROMPT_CACHE=1`.
