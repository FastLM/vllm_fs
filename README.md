<!-- markdownlint-disable MD001 MD041 -->
<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/vllm-project/vllm/main/docs/assets/logos/vllm-logo-text-dark.png">
    <img alt="vLLM" src="https://raw.githubusercontent.com/vllm-project/vllm/main/docs/assets/logos/vllm-logo-text-light.png" width=55%>
  </picture>
</p>

<h3 align="center">
Easy, fast, and cheap LLM serving for everyone
</h3>

<p align="center">
| <a href="https://docs.vllm.ai"><b>Documentation</b></a> | <a href="https://blog.vllm.ai/"><b>Blog</b></a> | <a href="https://arxiv.org/abs/2309.06180"><b>Paper</b></a> | <a href="https://x.com/vllm_project"><b>Twitter/X</b></a> | <a href="https://discuss.vllm.ai"><b>User Forum</b></a> | <a href="https://slack.vllm.ai"><b>Developer Slack</b></a> |
</p>

🔥 We have built a vLLM website to help you get started with vLLM. Please visit [vllm.ai](https://vllm.ai) to learn more.
For events, please visit [vllm.ai/events](https://vllm.ai/events) to join us.

---

## About

vLLM is a fast and easy-to-use library for LLM inference and serving.

Originally developed in the [Sky Computing Lab](https://sky.cs.berkeley.edu) at UC Berkeley, vLLM has grown into one of the most active open-source AI projects built and maintained by a diverse community of many dozens of academic institutions and companies from over 2000 contributors.

vLLM is fast with:

- State-of-the-art serving throughput
- Efficient management of attention key and value memory with [**PagedAttention**](https://blog.vllm.ai/2023/06/20/vllm.html)
- Continuous batching of incoming requests, chunked prefill, prefix caching
- Fast and flexible model execution with piecewise and full CUDA/HIP graphs
- Quantization: FP8, MXFP8/MXFP4, NVFP4, INT8, INT4, GPTQ/AWQ, GGUF, compressed-tensors, ModelOpt, TorchAO, and [more](https://docs.vllm.ai/en/latest/features/quantization/index.html)
- Optimized attention kernels including FlashAttention, FlashInfer, TRTLLM-GEN, FlashMLA, and Triton
- Optimized GEMM/MoE kernels for various precisions using CUTLASS, TRTLLM-GEN, CuTeDSL
- Speculative decoding including n-gram, suffix, EAGLE, DFlash
- Automatic kernel generation and graph-level transformations using torch.compile
- Disaggregated prefill, decode, and encode

vLLM is flexible and easy to use with:

- Seamless integration with popular Hugging Face models
- High-throughput serving with various decoding algorithms, including *parallel sampling*, *beam search*, and more
- Tensor, pipeline, data, expert, and context parallelism for distributed inference
- Streaming outputs
- Generation of structured outputs using xgrammar or guidance
- Tool calling and reasoning parsers
- OpenAI-compatible API server, plus Anthropic Messages API and gRPC support
- Efficient multi-LoRA support for dense and MoE layers
- Support for NVIDIA GPUs, AMD GPUs, Intel GPUs, and x86/ARM/PowerPC CPUs. Additionally, diverse hardware plugins such as Google TPUs, Intel Gaudi, IBM Spyre, Huawei Ascend, Rebellions NPU, Apple Silicon, MetaX GPU, and more.

vLLM seamlessly supports 200+ model architectures on Hugging Face, including:

- Decoder-only LLMs (e.g., Llama, Qwen, Gemma)
- Mixture-of-Expert LLMs (e.g., Mixtral, DeepSeek-V3, Qwen-MoE, GPT-OSS)
- Hybrid attention and state-space models (e.g., Mamba, Qwen3.5)
- Multi-modal models (e.g., LLaVA, Qwen-VL, Pixtral)
- Embedding and retrieval models (e.g., E5-Mistral, GTE, ColBERT)
- Reward and classification models (e.g., Qwen-Math)

Find the full list of supported models [here](https://docs.vllm.ai/en/latest/models/supported_models.html).

## ForkServe

ForkServe maps an agentic step to a rooted token tree. Let \(b\) be KV bytes per token (one sequence, all layers), \(P\) the scheduler page size, \(L=|x|\) the trunk length, \(k\) the number of live children, \(\ell_i\) the residual of child \(i\), and \(r=L \bmod P\) the unaligned tail. After a winner \(\star\) is bound, only \(L+\ell_\star\) remains live.

\[
M_{\mathrm{clone}}=b\Bigl(kL+\sum_{i=1}^{k}\ell_i\Bigr),\qquad
M_{\mathrm{CoW}}=b\Bigl(L+\sum_{i=1}^{k}\ell_i\Bigr),\qquad
M_{\mathrm{spine}}=b(L+\ell_\star).
\]

### Automatic Prefix Caching (APC)

vLLM V1 APC is a content-addressed index of *full* pages. The key of a page is \(h=\mathrm{hash}(h_{\mathrm{parent}},\,t_{0:P},\,\mathrm{extra})\). `get_computed_blocks` returns a prefix whose length is a multiple of \(P\), capped at \(n-1\). Partial-hit CoW copies \(P\) rows, not the occupied prefix \(n_{\mathrm{valid}}\). v1 block tables are append-only: one hash may map to several physical pages until the owner is freed.

Sharing is discovered by hashing the *new* prompt. There is no parent pointer. If \(k\) siblings look up before any page is published (same-batch / first fan-out), every lookup misses and APC stores \(M_{\mathrm{clone}}\). On an aligned hash-hit (\(r=0\)) APC stores \(M_{\mathrm{CoW}}\). On an unaligned hash-hit the tail is private per child:

\[
M_{\mathrm{APC}}^{\mathrm{hit,}\,r>0}
=b\Bigl(L+\sum_{i=1}^{k}\ell_i+(k-1)r\Bigr).
\]

### Location-addressed CoW

ForkServe addresses pages by the parent table, not by content hash. A logical table is \(\sigma_v=(\pi(v),\,\mathrm{alias\_len},\,\mathrm{residual})\). `fork` aliases \(\sigma_{\pi(v)}\) *before* \(\ell_i\) exists and increments the parent frames (extra-pin on owned, non-RO pages so a parent `free` does not drop the trunk). A write that lands on \(\mathrm{ro}\lor\mathrm{ref}>1\) allocates a residual page and copies \(n_{\mathrm{valid}}\) rows. Abort decrefs residual pages only; cost \(\Theta(\ell_i/P)\), independent of \(L\).

Committed full pages are published into APC. Speculative pages are never hashed and never sampled. Intra-session fan-out does not walk \(h\) on the allocate path. Ordinary requests without a parent pointer still use APC unchanged.

### Adaptive tail

Let \(\ell\) be a common residual length. Frame counts for the unaligned tail:

\[
N_{\mathrm{pack}}=k\left\lceil\frac{r+\ell}{P}\right\rceil,\qquad
N_{\mathrm{freeze}}=1+k\left\lceil\frac{\ell}{P}\right\rceil.
\]

Choose \(\arg\min\{N_{\mathrm{pack}},N_{\mathrm{freeze}}\}\). Pack copies the \(r\) occupied rows into each child's first residual page. Freeze marks the parent's last page RO with \(n_{\mathrm{valid}}=r\) and shares it once.

PagedAttention maps token \(t\) as \((\mathrm{block\_table}[\lfloor t/P\rfloor],\,t\bmod P)\) and assumes every page except the last is full. A frozen page with \(n_{\mathrm{valid}}=r<P\) in the *middle* of a child sequence is therefore a correctness bug unless the slot map is fork-aware. The implemented path is pack plus row copy; freeze is accounted for comparison only.

### Implementation

| Component | Role |
|---|---|
| `KVCacheBlock.ro`, `.n_valid`, `.is_speculative` | CoW bit, occupancy, eviction class |
| `ForkServeTracker.try_alias` | bind \(\sigma_{\pi(v)}\) without walking \(h\) |
| `pin_readonly` / extra-pin | \(\mathrm{ref}\) independent of parent request lifetime |
| `cow_copy_kv_rows` | copy \(n_{\mathrm{valid}}\) rows, not \(P\) |
| APC `cached_block_hash_to_block` | secondary, commit-time index |

### Evaluation

Protocol: \(P=16\). APC: construct \(k\) children, call `get_computed_blocks` on all, then allocate (hashes unpublished \(\Rightarrow\) miss). ForkServe: allocate parent \(x\), then children \((x,\rho_i)\) with `forkserve_parent`. Live **blk** \(=|\{p:\mathrm{ref}(p)>0\}|\) in a one-layer pool. GiB uses Llama-3-8B GQA bf16,

\[
b=2\cdot 32\cdot 8\cdot 128\cdot 2=128\,\mathrm{KiB/tok},\qquad
\mathrm{GiB}=\frac{b}{2^{30}}\times\text{allocated slots}.
\]

Allocated slots:

\[
S_{\mathrm{clone}}=k\left\lceil\frac{L+\ell}{P}\right\rceil P,\qquad
S_{\mathrm{pack}}=(L-r)+k\left\lceil\frac{r+\ell}{P}\right\rceil P.
\]

Saving on live frames: \(1-N_{\mathrm{FS}}/N_{\mathrm{APC}}\).

| Workload | \(L\) | \(k\) | \(\ell\) | \(N_{\mathrm{APC}}\) | \(N_{\mathrm{FS}}\) | \(1-N_{\mathrm{FS}}/N_{\mathrm{APC}}\) | APC GiB | FS GiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| aligned_fanout_4 | 2048 | 4 | 64 | 528 | 144 | 72.7% | 1.03 | 0.28 |
| aligned_fanout_8 | 4096 | 8 | 128 | 2112 | 320 | 84.8% | 4.12 | 0.62 |
| unaligned_pack | 2050 | 4 | 48 | 528 | 145 | 72.5% | 1.03 | 0.28 |
| tot_wide | 8192 | 8 | 32 | 4112 | 528 | 87.2% | 8.03 | 1.03 |
| short_residual | 1024 | 6 | 4 | 390 | 70 | 82.1% | 0.76 | 0.14 |
| planner_specialists | 16384 | 4 | 256 | 4160 | 1088 | 73.8% | 8.12 | 2.12 |

```bash
source "$HOME/envs/forkserve/bin/activate"
python -m pytest tests/v1/core/test_forkserve.py -q --noconftest
python benchmarks/forkserve/compare_apc.py
```

## Getting Started

Install vLLM with [`uv`](https://docs.astral.sh/uv/) (recommended) or `pip`:

```bash
uv pip install vllm
```

Or [build from source](https://docs.vllm.ai/en/latest/getting_started/installation/gpu/index.html#build-wheel-from-source) for development.

Visit our [documentation](https://docs.vllm.ai/en/latest/) to learn more.

- [Installation](https://docs.vllm.ai/en/latest/getting_started/installation.html)
- [Quickstart](https://docs.vllm.ai/en/latest/getting_started/quickstart.html)
- [List of Supported Models](https://docs.vllm.ai/en/latest/models/supported_models.html)

## Contributing

We welcome and value any contributions and collaborations.
Please check out [Contributing to vLLM](https://docs.vllm.ai/en/latest/contributing/index.html) for how to get involved.

## Citation

If you use vLLM for your research, please cite our [paper](https://arxiv.org/abs/2309.06180):

```bibtex
@inproceedings{kwon2023efficient,
  title={Efficient Memory Management for Large Language Model Serving with PagedAttention},
  author={Woosuk Kwon and Zhuohan Li and Siyuan Zhuang and Ying Sheng and Lianmin Zheng and Cody Hao Yu and Joseph E. Gonzalez and Hao Zhang and Ion Stoica},
  booktitle={Proceedings of the ACM SIGOPS 29th Symposium on Operating Systems Principles},
  year={2023}
}
```

## Contact Us

<!-- --8<-- [start:contact-us] -->
- For technical questions and feature requests, please use GitHub [Issues](https://github.com/vllm-project/vllm/issues)
- For discussing with fellow users, please use the [vLLM Forum](https://discuss.vllm.ai)
- For coordinating contributions and development, please use [Slack](https://slack.vllm.ai)
- For security disclosures, please use GitHub's [Security Advisories](https://github.com/vllm-project/vllm/security/advisories) feature
- For collaborations and partnerships, please contact us at [collaboration@vllm.ai](mailto:collaboration@vllm.ai)
<!-- --8<-- [end:contact-us] -->

## Media Kit

- If you wish to use vLLM's logo, please refer to [our media kit repo](https://github.com/vllm-project/media-kit)
