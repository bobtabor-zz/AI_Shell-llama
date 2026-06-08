#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h> // Required for time()

#include "llama.h"
#include "engine.h"
#include "engine_kv.h"
#include "ggml-backend.h"


// Paste these definitions at the top of engine.c below your includes
static void llama_batch_clear(struct llama_batch* batch) {
    batch->n_tokens = 0;
}

static void llama_batch_add(
    struct llama_batch* batch,
    llama_token          id,
    llama_pos            pos,
    const int* seq_ids,
    bool                 logits)
{
    batch->token[batch->n_tokens] = id;
    batch->pos[batch->n_tokens] = pos;
    batch->n_seq_id[batch->n_tokens] = 1;
    batch->seq_id[batch->n_tokens] = seq_ids;
    batch->logits[batch->n_tokens] = logits ? 1 : 0;
    batch->n_tokens++;
}

//
//char* plugin_ddg(int argc, char** argv) {
//    static char dummy[] = "{\"error\": \"ddg plugin not implemented\"}";
//    return dummy;
//}
//


void sanitize(char* s) {
    for (int i = 0; s[i]; i++) {
        if (s[i] == '<' || s[i] == '|' || s[i] == '>') {
            s[i] = ' ';
        }
    }
}


void engine_append_turn(engine_t* e, const char* role, const char* text) {
    if (e->n_turns >= MAX_TURNS) {
        for (int i = 1; i < e->n_turns; ++i)
            e->turns[i - 1] = e->turns[i];
        e->n_turns--;
    }

    engine_turn_t* t = &e->turns[e->n_turns++];

    memset(t, 0, sizeof(*t));   // <-- REQUIRED

    strncpy(t->role, role, sizeof(t->role) - 1);
    strncpy(t->text, text, sizeof(t->text) - 1);
}


void engine_build_prompt(engine_t* e, char* out, size_t out_sz) {
    out[0] = 0;  // <-- REQUIRED

    strcat(out, "<|system|>\nYou are a helpful assistant.\n\n");

    for (int i = 0; i < e->n_turns; i++) {
        strcat(out, "<|");
        strcat(out, e->turns[i].role);
        strcat(out, "|>\n");
        strcat(out, e->turns[i].text);
        strcat(out, "\n\n");
    }

    strcat(out, "<|assistant|>\n");
}



static bool engine_has_gpu(void) {
    // Call the exact function verified by your dumpbin check
    if (llama_supports_gpu_offload()) {
        printf("Engine Status: GPU offloading is supported by this llama.dll build!\n");
        return true;
    }

    printf("Engine Status: CPU-only fallback active.\n");
    return false;
}


void engine_close(engine_t* e) {
    if (!e) return;
    if (e->ctx)   llama_free(e->ctx);
    if (e->model) llama_free_model(e->model);
    free(e);
}

static llama_token sample_next(engine_t* e, float temp, int top_k, float top_p) {
    if (!e || !e->ctx) return -1;

    struct llama_sampler_chain_params sparams = llama_sampler_chain_default_params();
    struct llama_sampler* smpl = llama_sampler_chain_init(sparams);

    if (temp > 0.0f) {
        if (top_k > 0) {
            llama_sampler_chain_add(smpl, llama_sampler_init_top_k(top_k));
        }
        if (top_p < 1.0f) {
            llama_sampler_chain_add(smpl, llama_sampler_init_top_p(top_p, 1));
        }

        llama_sampler_chain_add(smpl, llama_sampler_init_temp(temp));

        // FIX: Use the current system time as a dynamic seed.
        // This stops the text from being repetitive ("weird text") 
        // while safely keeping the required distribution sampler active.
        uint32_t dynamic_seed = (uint32_t)time(NULL);
        llama_sampler_chain_add(smpl, llama_sampler_init_dist(dynamic_seed));
    }
    else {
        // If temperature is 0, use greedy sampling (always picks the absolute highest probability)
        llama_sampler_chain_add(smpl, llama_sampler_init_greedy());
    }

    // This will no longer crash
    llama_token tok = llama_sampler_sample(smpl, e->ctx, -1);

    llama_sampler_free(smpl);
    return tok;
}



// ===================================================================
// Generate
// ===================================================================
//int engine_generate(
//    engine_t* e,
//    const char* prompt,
//    char* out,
//    size_t out_size,
//    int max_tokens,
//    float temp,
//    int top_k,
//    float top_p,
//    bool stream)
//{
//    if (!e || !e->model || !e->ctx || !prompt || !out || out_size == 0)
//        return -1;
//
//    // ===================================================================
//    // FAST MODE OVERRIDES
//    // ===================================================================
//    {
//        max_tokens = 16;
//        temp = 0.7f;
//        top_k = 20;
//        top_p = 0.9f;
//    }
//
//    out[0] = '\0';
//
//    // -------------------------------------------------------------------
//    // Tokenize prompt
//    // -------------------------------------------------------------------
//    const int max_tokens_prompt = 4096;
//    llama_token* tokens = (llama_token*)malloc(max_tokens_prompt * sizeof(llama_token));
//    if (!tokens)
//        return -1;
//
//    const struct llama_vocab* vocab = llama_model_get_vocab(e->model);
//
//    int n_prompt = llama_tokenize(
//        vocab,
//        prompt,
//        (int32_t)strlen(prompt),
//        tokens,
//        max_tokens_prompt,
//        true,
//        false
//    );
//
//    if (n_prompt <= 0) {
//        fprintf(stderr, "[engine] tokenize failed\n");
//        free(tokens);
//        return -1;
//    }
//
//    // -------------------------------------------------------------------
//    // Build and decode prompt batch
//    // -------------------------------------------------------------------
//    llama_batch batch = llama_batch_init(1024, 0, 1);
//
//    int64_t cur_pos = 0;   // ⭐ THIS WAS YOUR ORIGINAL WORKAROUND
//
//    for (int i = 0; i < n_prompt; ++i) {
//        llama_batch_add(
//            &batch,
//            tokens[i],
//            (llama_pos)cur_pos,
//            (const int[]) {
//            0
//        },
//            i == n_prompt - 1
//        );
//        cur_pos++;   // ⭐ STRICTLY INCREASING
//    }
//
//    if (llama_decode(e->ctx, batch) != 0) {
//        fprintf(stderr, "[engine] decode(prompt) failed\n");
//        free(tokens);
//        return -1;
//    }
//
//    // -------------------------------------------------------------------
//    // Generation loop
//    // -------------------------------------------------------------------
//    size_t out_len = 0;
//    int n_gen = 0;
//
//    while (n_gen < max_tokens && out_len + 32 < out_size) {
//
//        llama_token tok = sample_next(e, temp, top_k, top_p);
//        if (tok == llama_token_eos(e->model))
//            break;
//
//        char buf[128];
//        int n = llama_token_to_piece(
//            llama_model_get_vocab(e->model),
//            tok,
//            buf,
//            sizeof(buf),
//            0,
//            false
//        );
//        if (n <= 0)
//            break;
//
//        if (out_len + (size_t)n >= out_size)
//            break;
//
//        memcpy(out + out_len, buf, n);
//        out_len += n;
//        out[out_len] = '\0';
//
//        if (stream) {
//            fwrite(buf, 1, n, stdout);
//            fflush(stdout);
//        }
//
//        // feed back token
//        llama_batch_clear(&batch);
//        llama_batch_add(
//            &batch,
//            tok,
//            (llama_pos)cur_pos,
//            (const int[]) {
//            0
//        },
//            true
//        );
//
//        if (llama_decode(e->ctx, batch) != 0) {
//            fprintf(stderr, "[engine] decode(next token) failed\n");
//            break;
//        }
//
//        ++n_gen;
//        ++cur_pos;   // ⭐ STRICTLY INCREASING
//
//        if (strstr(out, "\nUser:") || strstr(out, "\nAI:"))
//            break;
//    }
//
//    free(tokens);
//
//    // ⭐ NO KV BOOKKEEPING
//    // ⭐ NO e->pos MATH
//    // ⭐ NO e->kv_len
//    // ⭐ NO e->kv_valid
//
//    return (int)out_len;
//}
//
 
// 
// grock one
// 
//int engine_generate(
//    engine_t* e,
//    const char* prompt,
//    char* out,
//    size_t out_size,
//    int max_tokens,
//    float temp,
//    int top_k,
//    float top_p,
//    bool stream)
//{
//    if (!e || !e->model || !e->ctx || !prompt || !out || out_size == 0)
//        return -1;
//
//    out[0] = '\0';
//    if (prompt[0] == '\0') return 0;
//
//    // ====================== FAST SETTINGS ======================
//    max_tokens = 16;      // increase later when stable
//    temp = 0.7f;
//    top_k = 20;
//    top_p = 0.9f;
//
//    // Clear previous state for this new query
//    engine_kv_clear(e);
//
//    // ====================== TOKENIZE ======================
//    const int max_prompt_tokens = 4096;
//    llama_token* tokens = malloc(max_prompt_tokens * sizeof(llama_token));
//    if (!tokens) return -1;
//
//    const int max_tokens_prompt = 4096;
//
//	const struct llama_vocab* vocab = llama_model_get_vocab(e->model);
//    int n_prompt = llama_tokenize(
//                vocab,
//                prompt,
//                (int32_t)strlen(prompt),
//                tokens,
//                max_tokens_prompt,
//                true,
//                false
//            );
//
//    if (n_prompt <= 0) {
//        fprintf(stderr, "[engine] tokenize failed\n");
//        free(tokens);
//        return -1;
//    }
//
//    // ====================== BATCH ======================
//    llama_batch batch = llama_batch_init(1024, 0, 1);
//
//    // Add prompt
//    for (int i = 0; i < n_prompt; ++i) {
//        llama_batch_add(&batch, tokens[i], i, (const int[]) { 0 }, i == n_prompt - 1);
//    }
//
//    if (llama_decode(e->ctx, batch) != 0) {
//        fprintf(stderr, "[engine] decode(prompt) failed\n");
//        goto cleanup;
//    }
//
//    engine_kv_mark_prompt(e, n_prompt);
//
//    // ====================== GENERATION LOOP ======================
//    size_t out_len = 0;
//    int n_gen = 0;
//
//    while (n_gen < max_tokens && out_len + 32 < out_size) {
//        llama_token tok = sample_next(e, temp, top_k, top_p);
//        if (tok == llama_token_eos(e->model)) break;
//
//        char buf[128] = { 0 };
//        int n = llama_token_to_piece(llama_model_get_vocab(e->model),
//            tok, buf, sizeof(buf), 0, false);
//        if (n <= 0) break;
//
//        if (out_len + (size_t)n >= out_size) break;
//
//        memcpy(out + out_len, buf, n);
//        out_len += n;
//        out[out_len] = '\0';
//
//        if (stream) {
//            fwrite(buf, 1, n, stdout);
//            fflush(stdout);
//        }
//
//        // Next token
//        llama_batch_clear(&batch);
//        llama_batch_add(&batch, tok, n_prompt + n_gen, (const int[]) { 0 }, true);
//
//        if (llama_decode(e->ctx, batch) != 0) {
//            fprintf(stderr, "[engine] decode(next) failed\n");
//            break;
//        }
//
//        ++n_gen;
//        engine_kv_advance(e, 1);
//
//        // Simple stop conditions
//        if (strstr(out, "\n\n") || strstr(out, "\n"))
//            break;
//    }
//  
//cleanup:
//   // llama_batch_free(batch);
//    free(tokens);
//
//    engine_kv_debug(e);
//    return (int)out_len;
//}

//int engine_generate(
//    engine_t* e,
//    const char* prompt,
//    char* out,
//    size_t out_size,
//    int max_tokens,
//    float temp,
//    int top_k,
//    float top_p,
//    bool stream)
//{
//    if (!e || !e->model || !e->ctx || !prompt || !out || out_size == 0)
//        return -1;
//
//    out[0] = '\0';
//    if (prompt[0] == '\0') return 0;
//
//    // Fast defaults
//    max_tokens = 128;      // you can increase this later
//    temp = 0.75f;
//    top_k = 40;
//    top_p = 0.9f;
//
//    // Clear KV cache before every new generation (your requirement)
//    engine_kv_clear(e);        // this will call engine_recreate_context
//
//    // ====================== TOKENIZE ======================
//    const int max_tokens_prompt = 4096;
//    llama_token* tokens = malloc(max_tokens_prompt * sizeof(llama_token));
//    if (!tokens) return -1;
//
//    //int n_prompt = llama_tokenize(
//    //    e->model,                    // ← model, not vocab
//    //    prompt,
//    //    tokens,
//    //    max_prompt,
//    //    true,   // add_special
//    //    false   // parse_special
//    //);
//
//    const struct llama_vocab* vocab = llama_model_get_vocab(e->model);
//        int n_prompt = llama_tokenize(
//                    vocab,
//                    prompt,
//                    (int32_t)strlen(prompt),
//                    tokens,
//                    max_tokens_prompt,
//                    true,
//                    false
//                );
//
//
//    if (n_prompt <= 0) {
//        fprintf(stderr, "[engine] tokenize failed\n");
//        free(tokens);
//        return -1;
//    }
//
//    // ====================== BATCH ======================
//    llama_batch batch = llama_batch_init(1024, 0, 1);
//
//    // Add prompt tokens (positions start from 0 after clear)
//    for (int i = 0; i < n_prompt; ++i) {
//        llama_batch_add(&batch, tokens[i], i, (const int[]) { 0 }, i == n_prompt - 1);
//    }
//
//    if (llama_decode(e->ctx, batch) != 0) {
//        fprintf(stderr, "[engine] decode(prompt) failed\n");
//        goto cleanup;
//    }
//
//    // ====================== GENERATION ======================
//    size_t out_len = 0;
//    int n_gen = 0;
//    int64_t current_pos = n_prompt;   // next position
//
//    while (n_gen < max_tokens && out_len + 32 < out_size) {
//        llama_token tok = sample_next(e, temp, top_k, top_p);
//        if (tok == llama_token_eos(e->model)) break;
//
//        char buf[128] = { 0 };
//        int n = llama_token_to_piece(
//            llama_model_get_vocab(e->model),
//            tok, buf, sizeof(buf), 0, false
//        );
//
//        if (n <= 0) break;
//        if (out_len + (size_t)n >= out_size) break;
//
//        memcpy(out + out_len, buf, n);
//        out_len += n;
//        out[out_len] = '\0';
//
//        if (stream) {
//            fwrite(buf, 1, n, stdout);
//            fflush(stdout);
//        }
//
//        // Next token
//        llama_batch_clear(&batch);
//        llama_batch_add(&batch, tok, current_pos, (const int[]) { 0 }, true);
//
//        if (llama_decode(e->ctx, batch) != 0) {
//            fprintf(stderr, "[engine] decode(next) failed\n");
//            break;
//        }
//
//        current_pos++;
//        n_gen++;
//
//        // Early stop
//        if (strstr(out, "\n\n") || strstr(out, "\nUser:"))
//            break;
//    }
//
//cleanup:
//   // llama_batch_free(batch);
//    free(tokens);
//
//    return (int)out_len;
//}

//int engine_generate(
//    engine_t* e,
//    const char* prompt,
//    char* out,
//    size_t out_size,
//    int max_tokens,
//    float temp,
//    int top_k,
//    float top_p,
//    bool stream)
//{
//    if (!e || !e->model || !e->ctx || !prompt || !out || out_size == 0)
//        return -1;
//
//    out[0] = '\0';
//    if (prompt[0] == '\0') return 0;
//
//    // ====================== FAST SETTINGS ======================
//    max_tokens = 16;      // bump later if you want
//    temp = 0.7f;
//    top_k = 20;
//    top_p = 0.9f;
//
//    
//        // ❌ NO engine_kv_clear(e) HERE ANYMORE replaced by llama_memory_seq_rm
//    // Context is already recreated by the caller (chat / summarize).
//    //engine_kv_clear(e);
//
//    // ====================== TOKENIZE ======================
//    const int max_prompt_tokens = 4096;
//    llama_token* tokens = malloc(max_prompt_tokens * sizeof(llama_token));
//    if (!tokens) return -1;
//
//    const struct llama_vocab* vocab = llama_model_get_vocab(e->model);
//
//    int n_prompt = llama_tokenize(
//        vocab,
//        prompt,
//        (int32_t)strlen(prompt),
//        tokens,
//        max_prompt_tokens,
//        true,
//        false
//    );
//
//    if (n_prompt <= 0) {
//        fprintf(stderr, "[engine] tokenize failed\n");
//        free(tokens);
//        return -1;
//    }
//
//    // ====================== BATCH ======================
//    llama_batch batch = llama_batch_init(1024, 0, 1);
//
//    int64_t cur_pos = 0;  // fresh context → start at 0
//
//    // Add prompt
//    for (int i = 0; i < n_prompt; ++i) {
//        llama_batch_add(
//            &batch,
//            tokens[i],
//            (llama_pos)cur_pos,
//            (const int[]) {
//            0
//        },          // single sequence
//            i == n_prompt - 1
//        );
//        cur_pos++;
//    }
//
//    if (llama_decode(e->ctx, batch) != 0) {
//        fprintf(stderr, "[engine] decode(prompt) failed\n");
//        free(tokens);
//        return -1;
//    }
//
//    // ❌ NO engine_kv_mark_prompt(e, n_prompt);
//
//    // ====================== GENERATION LOOP ======================
//    size_t out_len = 0;
//    int    n_gen = 0;
//
//    while (n_gen < max_tokens && out_len + 32 < out_size) {
//        llama_token tok = sample_next(e, temp, top_k, top_p);
//        if (tok == llama_token_eos(e->model))
//            break;
//
//        char buf[128] = { 0 };
//        int n = llama_token_to_piece(
//            llama_model_get_vocab(e->model),
//            tok,
//            buf,
//            sizeof(buf),
//            0,
//            false
//        );
//        if (n <= 0) break;
//
//        if (out_len + (size_t)n >= out_size) break;
//
//        memcpy(out + out_len, buf, n);
//        out_len += n;
//        out[out_len] = '\0';
//
//        if (stream) {
//            fwrite(buf, 1, n, stdout);
//            fflush(stdout);
//        }
//
//        llama_batch_clear(&batch);
//        llama_batch_add(
//            &batch,
//            tok,
//            (llama_pos)cur_pos,
//            (const int[]) {
//            0
//        },
//            true
//        );
//
//        if (llama_decode(e->ctx, batch) != 0) {
//            fprintf(stderr, "[engine] decode(next) failed\n");
//            break;
//        }
//
//        ++n_gen;
//        ++cur_pos;
//
//        if (strstr(out, "\n\n") || strstr(out, "\n"))
//            break;
//    }
//
//    free(tokens);
//
//    // ❌ NO engine_kv_advance(e, 1);
//    // ❌ NO engine_kv_debug(e);
//
//    return (int)out_len;
//}

int engine_generate(
    engine_t* e,
    const char* prompt,
    char* out,
    size_t out_size,
    int max_tokens,
    float temp,
    int top_k,
    float top_p,
    bool stream)
{
    if (!e || !e->model || !e->ctx || !prompt || !out || out_size == 0)
        return -1;

    out[0] = '\0';
    if (prompt[0] == '\0') return 0;

    // fast defaults
    max_tokens = 128;
    temp = 0.7f;
    top_k = 20;
    top_p = 0.9f;

    // IMPORTANT:
    // - DO NOT reset KV here
    // - DO NOT reset position here
    // If you want a fresh context, call engine_kv_clear(e) BEFORE this.

    // ====================== TOKENIZE ======================
    const int max_prompt_tokens = 4096;
    llama_token* tokens = malloc(max_prompt_tokens * sizeof(llama_token));
    if (!tokens) return -1;

    const struct llama_vocab* vocab = llama_model_get_vocab(e->model);

    int n_prompt = llama_tokenize(
        vocab,
        prompt,
        (int32_t)strlen(prompt),
        tokens,
        max_prompt_tokens,
        true,
        false
    );

    if (n_prompt <= 0) {
        fprintf(stderr, "[engine] tokenize failed\n");
        free(tokens);
        return -1;
    }

    // ====================== BATCH (PROMPT) ======================
    llama_batch batch = llama_batch_init(1024, 0, 1);

    int64_t cur_pos = e->pos;   // <-- start where we left off

    for (int i = 0; i < n_prompt; ++i) {
        llama_batch_add(
            &batch,
            tokens[i],
            (llama_pos)cur_pos,
            (const int[]) {
            e->seq_id
        },   // single sequence
            i == n_prompt - 1             // logits on last prompt token
        );
        cur_pos++;
    }

    if (llama_decode(e->ctx, batch) != 0) {
        fprintf(stderr, "[engine] decode(prompt) failed\n");
        llama_batch_free(batch);
        free(tokens);
        return -1;
    }

    // ====================== GENERATION LOOP ======================
    size_t out_len = 0;
    int    n_gen = 0;

    while (n_gen < max_tokens && out_len + 8 < out_size) { //32
        llama_token tok = sample_next(e, temp, top_k, top_p);
        if (tok == llama_token_eos(e->model))
            break;

        char buf[128] = { 0 };
        int n = llama_token_to_piece(
            llama_model_get_vocab(e->model),
            tok,
            buf,
            sizeof(buf),
            0,
            false
        );
        if (n <= 0) break;
        if (out_len + (size_t)n >= out_size) break;

        // ⭐⭐⭐ INSERT THE ROLE‑TAG BLOCKER HERE ⭐⭐⭐
        if (strstr(buf, "<|") || strstr(buf, "|>")) {
            // Skip tokens that would corrupt your prompt
            continue;
        }
        // ⭐⭐⭐ END INSERT ⭐⭐⭐

        memcpy(out + out_len, buf, n);
        out_len += n;
        out[out_len] = '\0';

        if (stream) {
            fwrite(buf, 1, n, stdout);
            fflush(stdout);
        }

        llama_batch_clear(&batch);
        llama_batch_add(
            &batch,
            tok,
            (llama_pos)cur_pos,          // <-- continue at cur_pos
            (const int[]) {
            e->seq_id
        },
            true                         // need logits for next token
        );

        if (llama_decode(e->ctx, batch) != 0) {
            fprintf(stderr, "[engine] decode(next) failed\n");
            break;
        }

        ++n_gen;
        ++cur_pos;

        /*if (strstr(out, "\n\n") || strstr(out, "\n"))
            break;*/
    }

    //vsafe_engine_batch_free(batch);

    ///llama_batch_free(batch);
    free(tokens);

    // persist new position for next call
    e->pos = cur_pos;

    return (int)out_len;
}



engine_t* engine_open(const char* path) {
    fprintf(stderr, "[engine] loading model: %s\n", path);

    // -----------------------------
    // Model params
    // -----------------------------
    struct llama_model_params mp = llama_model_default_params();

    // Always request GPU layers; llama.cpp will ignore if unsupported
    mp.n_gpu_layers = 999;

    struct llama_model* model = llama_load_model_from_file(path, mp);
    if (!model) {
        fprintf(stderr, "[engine] llama_load_model_from_file failed\n");
        return NULL;
    }

    // -----------------------------
    // GPU detection (your working function)
    // -----------------------------
    if (engine_has_gpu()) {
        fprintf(stderr, "[engine] GPU backend detected — enabling full GPU offload\n");
    }
    else {
        fprintf(stderr, "[engine] No GPU backend — falling back to CPU\n");
        mp.n_gpu_layers = 0;   // clarity only; llama.cpp already ignores it
    }

    // -----------------------------
    // Context params (CPU optimized)
    // -----------------------------
    struct llama_context_params cp = llama_context_default_params();
    cp.n_ctx = 2048;
    cp.n_batch = 1024;   // CPU throughput boost
    cp.n_threads = 0;      // use all cores

    struct llama_context* ctx = llama_new_context_with_model(model, cp);
    if (!ctx) {
        fprintf(stderr, "[engine] llama_new_context_with_model failed\n");
        llama_free_model(model);
        return NULL;
    }

    // -----------------------------
    // Allocate engine_t
    // -----------------------------
    engine_t* e = (engine_t*)calloc(1, sizeof(engine_t));
    if (!e) {
        llama_free(ctx);
        llama_free_model(model);
        return NULL;
    }

    e->model = model;
    e->ctx = ctx;
    e->pos = 0;   // initialize KV position

    fprintf(stderr, "[engine] model loaded successfully!\n");
    return e;
}

// Hard-coded fast settings
//max_tokens = 16;   // was 256 or higher — this is the BIG speed win
//temp = 0.7f;
//top_k = 20;
//top_p = 0.9f;

//int engine_chat(engine_t* e,
//    const char* user_input,
//    char* out,
//    size_t out_size)
//{
//    if (!e || !e->model || !e->ctx || !user_input || !out || out_size == 0)
//        return -1;
//
//    // ⭐ ALWAYS RESET KV BEFORE ANY CHAT TURN
//    // This prevents "X vs Y" KV position mismatch errors.
//   // engine_kv_clear(e);
//   // engine_recreate_context(e);
//
//    // ============================================================
//    // DDG SEARCH MODE
//    // ============================================================
//    if (strncmp(user_input, "search:", 7) == 0) {
//        const char* q = user_input + 7;
//        while (*q == ' ') ++q;
//
//        char* argv[1];
//        argv[0] = (char*)q;
//
//        char* ddg_json = plugin_ddg(1, argv);
//        if (!ddg_json) {
//            snprintf(out, out_size, "Search failed.");
//            engine_append_turn(e, "AI", out);
//            return 0;
//        }
//
//        char prompt[8192];
//        snprintf(prompt, sizeof(prompt),
//            "You are a helpful assistant.\n"
//            "Here is JSON search result data:\n%s\n\n"
//            "Summarize the most relevant answer for the user.\nAI:",
//            ddg_json);
//
//        int rc = engine_generate(
//            e,
//            prompt,
//            out,
//            out_size,
//            64,
//            0.7f,
//            20,
//            0.9f,
//            true   // ⭐ STREAM
//        );
//
//        //engine_append_turn(e, "AI", out);
//        return (rc < 0) ? -1 : 0;
//    }
//
//    // ============================================================
//    // NORMAL CHAT MODE
//    // ============================================================
//    char prompt[8192];
//    engine_build_prompt(e, user_input, prompt, sizeof(prompt));
//
//    int rc = engine_generate(
//        e,
//        prompt,
//        out,
//        out_size,
//        64,
//        0.7f,
//        20,
//        0.9f,
//        true   // ⭐ STREAM
//    );
//
//    if (rc < 0) {
//        fprintf(stderr, "[engine] engine_generate(chat) failed\n");
//        return -1;
//    }
//
//    engine_append_turn(e, "AI", out);
//    return 0;
//}

int engine_chat(engine_t* e,
    const char* user_msg,
    char* out,
    size_t out_size)
{
    // First turn → reset KV + pos
    /*if (e->n_turns != 0) {
        engine_kv_clear(e);
    }*/

    // Add user turn
    engine_append_turn(e, "user", user_msg);

    // Build prompt
    char prompt[65536];
    engine_build_prompt(e, prompt, sizeof(prompt));

    // STREAMING OUTPUT
    printf("AI> ");
    fflush(stdout);

    out[0] = 0;
    int n = engine_generate(
        e,
        prompt,
        out,
        out_size,
        128,
        0.7f,
        20,
        0.9f,
        true   // tickertape streaming
    );

    if (n <= 0)
        return -1;

    // Save assistant turn
    sanitize(out);
    engine_append_turn(e, "assistant", out);

    printf("\n");
    return 0;
}

