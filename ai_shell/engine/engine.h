#pragma once

#include <stddef.h>
#include <stdint.h>

#include "../loader/gguf_loader.h"
#include "../vmm/vmm.h"
#include "llama.h"

#ifdef __cplusplus
extern "C" {
#endif

    // Forward declaration
    struct llama_model;
    struct llama_context;

    typedef struct engine {
        engine_gguf_loader_t* loader;
        vmm_t* vmm;

        struct llama_model* tok_model;
        struct llama_context* tok_ctx;

    } engine_t;

    // Open / close engine
    engine_t* engine_open(const char* path);
    void       engine_close(engine_t* e);

    // Forward pass
    int engine_forward(engine_t* e, const float* input, float* output);

    // High‑level inference
    int engine_infer(engine_t* e, const char* prompt, char* out, int out_size);

#ifdef __cplusplus
}
#endif
