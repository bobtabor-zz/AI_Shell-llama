#include "engine.h"

#include "../loader/gguf_loader.h"
#include "../vmm/vmm.h"
#include "llama.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>



// -----------------------------------------------------------------------------
// Engine open
// -----------------------------------------------------------------------------

engine_t* engine_open(const char* path) {
    engine_t* e = calloc(1, sizeof(engine_t));
    if (!e) return NULL;

    // load GGUF metadata
    e->loader = engine_gguf_open(path);
    if (!e->loader) {
        fprintf(stderr, "engine_open: failed to load GGUF\n");
        free(e);
        return NULL;
    }

    // init VMM
    e->vmm = vmm_create(path);
    if (!e->vmm) {
        fprintf(stderr, "engine_open: failed to init VMM\n");
        engine_gguf_close(e->loader);
        free(e);
        return NULL;
    }

    // load llama.cpp tokenizer model (vocab only)
    struct llama_model_params mp = llama_model_default_params();
    mp.vocab_only = true;

    e->tok_model = llama_load_model_from_file(path, mp);
    if (!e->tok_model) {
        fprintf(stderr, "engine_open: failed to load tokenizer model\n");
        vmm_destroy(e->vmm);
        engine_gguf_close(e->loader);
        free(e);
        return NULL;
    }

    struct llama_context_params cp = llama_context_default_params();
    e->tok_ctx = llama_new_context_with_model(e->tok_model, cp);
    if (!e->tok_ctx) {
        fprintf(stderr, "engine_open: failed to create tokenizer context\n");
        llama_free_model(e->tok_model);
        vmm_destroy(e->vmm);
        engine_gguf_close(e->loader);
        free(e);
        return NULL;
    }

    return e;
}

// -----------------------------------------------------------------------------
// Engine close
// -----------------------------------------------------------------------------

void engine_close(engine_t* e) {
    if (!e) return;

    if (e->tok_ctx)   llama_free(e->tok_ctx);
    if (e->tok_model) llama_free_model(e->tok_model);
    if (e->vmm)       vmm_destroy(e->vmm);
    if (e->loader)    engine_gguf_close(e->loader);

    free(e);
}

// -----------------------------------------------------------------------------
// Tensor lookup
// -----------------------------------------------------------------------------

static engine_gguf_tensor_info_t* find_tensor(engine_t* e, const char* name) {
    engine_gguf_tensor_info_t* tins = NULL;
    uint64_t count = 0;

    engine_gguf_get_tensors(e->loader, &tins, &count);

    for (uint64_t i = 0; i < count; ++i) {
        if (strcmp(tins[i].name, name) == 0)
            return &tins[i];
    }
    return NULL;
}

// -----------------------------------------------------------------------------
// Forward pass
// -----------------------------------------------------------------------------

int engine_forward(engine_t* e, const float* input, float* output) {
    engine_gguf_tensor_info_t* ti =
        find_tensor(e, "model.layers.0.attn.q_proj.weight");

    if (!ti) {
        fprintf(stderr, "engine_forward: missing tensor\n");
        return 0;
    }

    const float* w = (const float*)vmm_map(
        e->vmm,
        ti->offset + e->loader->hdr.tensor_data_offset,
        engine_gguf_tensor_nbytes(ti)
    );

    if (!w) {
        fprintf(stderr, "engine_forward: vmm_map failed\n");
        return 0;
    }

    // your math here

    vmm_unmap(e->vmm, (void*)w, engine_gguf_tensor_nbytes(ti));
    return 1;
}

// -----------------------------------------------------------------------------
// Chat inference
// -----------------------------------------------------------------------------

int engine_infer(engine_t* e, const char* prompt, char* out, int out_size) {
    // FIX: Added 'struct' keyword here
    const struct llama_vocab* vocab = llama_model_get_vocab(e->tok_model);

    llama_token tokens[512];
    int ntok = llama_tokenize(
        vocab,
        prompt,
        (int32_t)strlen(prompt),
        tokens,
        512,
        true, // add_special (BOS)
        false // parse_special
    );

    if (ntok <= 0) return 0;

    float logits[4096];
    for (int i = 0; i < ntok; ++i) {
        float input[4096] = { 0 };
        input[tokens[i]] = 1.0f;

        if (!engine_forward(e, input, logits))
            return 0;
    }

    // Standard greedy sampling (assuming e->tok_ctx is initialized)
        // sample next token: greedy argmax over logits
    int   n_vocab = llama_n_vocab(e->tok_model);
    int   best_id = 0;
    float best_logit = -FLT_MAX;

    for (int i = 0; i < n_vocab; ++i) {
        if (logits[i] > best_logit) {
            best_logit = logits[i];
            best_id = i;
        }
    }

    llama_token next = (llama_token)best_id;


    int n = llama_detokenize(
        vocab,
        &next,
        1,
        out,
        out_size,
        false, // render_special
        false  // strip_buffer
    );

    return (n > 0);
}