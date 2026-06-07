#include <stdio.h>
#include "engine_kv.h"
#include "llama.h"   // whatever header exposes llama_kv_cache_clear / ctx

void engine_kv_clear(engine_t* e) {
    if (!e) return;
    engine_recreate_context(e);   // <- real KV clear via new context
}

void engine_kv_mark_prompt(engine_t* e, int64_t n_prompt) {
    if (!e) return;
    e->kv_valid = true;
    e->kv_len = n_prompt;
    e->pos = n_prompt;
}

void engine_kv_advance(engine_t* e, int64_t n_gen) {
    if (!e || !e->kv_valid) return;
    e->kv_len += n_gen;
    e->pos += n_gen;
}

void engine_kv_debug(const engine_t* e) {
    if (!e) return;
   /* fprintf(stderr, "[kv] valid=%d len=%lld pos=%lld\n",
        e->kv_valid ? 1 : 0,
        (long long)e->kv_len,
        (long long)e->pos);*/
}


//void engine_recreate_context(engine_t* e) {
//    if (!e || !e->model) return;
//
//    struct llama_context_params cp = llama_context_default_params();
//
//    // REMOVE cp.seed — it no longer exists
//    // cp.seed = 0;
//
//    cp.n_ctx = 2048;   // or whatever you want
//
//    if (e->ctx) {
//        llama_free(e->ctx);
//    }
//
//    e->ctx = llama_new_context_with_model(e->model, cp);
//
//    e->pos = 0;
//    e->kv_len = 0;
//    e->kv_valid = false;
//}

void engine_recreate_context(engine_t* e)
{
    if (!e || !e->model) return;

    fprintf(stderr, "[engine] Recreating context (KV cache clear)...\n");

    // Manual struct initialization - safest for your broken header
    struct llama_context_params cp = { 0 };   // zero all fields

    cp.n_ctx = 4096;      // ← change as needed
    cp.n_batch = 512;
    cp.n_ubatch = 512;
    cp.n_threads = 0;         // auto
    cp.n_threads_batch = 0;
    //cp.flash_attn = true;

    if (e->ctx) {
        llama_free(e->ctx);
        e->ctx = NULL;
    }

    e->ctx = llama_new_context_with_model(e->model, cp);

    if (e->ctx) {
        e->pos = 0;
        e->kv_len = 0;
        e->kv_valid = false;
        fprintf(stderr, "[engine] Context recreated successfully\n");
    }
    else {
        fprintf(stderr, "[engine] FAILED to recreate context!\n");
    }
}
