ai_shell.exe..

🧠 What vmm.bin is
vmm.bin is not a cache, not a swap file, and not a GGUF clone.

It is a persistent, file‑backed virtual memory arena that stores:

all model weights

in a tightly packed, allocator‑controlled layout

with stable offsets

that your runtime can map/unmap on demand

Think of it as:

“GPU VRAM, but on disk, with mmap instead of PCIe.”

It’s the third memory tier in your engine:

Code
GPU VRAM  →  CPU RAM  →  vmm.bin (file-backed)

⚙️ How vmm.bin behaves during normal operation
1. GPU asks for a tensor
Your runtime needs a tensor for a layer:

Q, K, V projection

FFN up/down

Norm weights

Embeddings

Output head

2. CPU-side llama.cpp tries to access the tensor
Normally llama.cpp would:

mmap the GGUF

read the tensor directly

or copy it to GPU

But in your engine:

llama.cpp never touches GGUF again.
All tensor reads go through vmm_map().

3. vmm_map() resolves the request
vmm_map() checks:

Is this region already mapped in RAM?

If not, map the file region into memory

Return a pointer to the mapped bytes

This is zero‑copy and lazy.

4. GPU upload happens from the mapped region
Your GPU upload code reads from the mapped region:

Code
GPU ← CPU RAM (mapped) ← vmm.bin
No GGUF.
No parsing.
No decompression.
No reallocation.

5. When GPU evicts a tensor
If your GPU memory manager decides to evict:

CPU RAM mapping stays alive

GPU can reload later

vmm.bin is the authoritative source

This is exactly how llama.cpp’s VRAM offload works, except your backing store is a file, not system RAM.

🔥 Why vmm.bin is superior to GGUF at runtime
✔ GGUF is a container format
metadata

tokenizer

rope params

quantization metadata

padding

unused tensors

alignment

legacy fields

✔ vmm.bin is a pure tensor slab
no metadata

no padding

no tokenizer

no rope

no KV templates

no empty tensors

Just:

Code
[header][meta table][raw tensor bytes...]
This is why your inspector shows 62 real tensors, not 291.

🚀 What happens during a forward pass
For each layer:

GPU requests Q/K/V weights

CPU maps the region from vmm.bin

GPU uploads the bytes

GPU computes

CPU unmaps or keeps mapped (your choice)

This is identical to:

Linux page cache

Windows memory‑mapped files

GPU unified memory (conceptually)

Except you control the allocator, not the OS.

🧩 How vmm.bin interacts with GPU memory
GPU VRAM
Holds active tensors for the current layer.

CPU RAM
Holds mapped slices of vmm.bin.

vmm.bin
Holds all tensors, permanently.

The flow:

Code
vmm.bin → CPU RAM (mmap) → GPU VRAM → compute → evict → repeat
This is why your engine can run:

8B models

with 8GB vmm.bin

even if GPU VRAM is small

even if system RAM is limited

You’ve effectively built a software‑defined VRAM paging system.

🏁 In one sentence
vmm.bin is your model’s permanent, file‑backed tensor store that feeds CPU RAM and GPU VRAM on demand, acting as the lowest tier of your memory hierarchy.
