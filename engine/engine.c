#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h> // Required for time()

#include "llama.h"
#include "engine.h"
#include "engine_kv.h"
#include "ggml-backend.h"
#include "../include/plugin.h"

char* g_plugin_result = NULL;


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

//void llama_kv_cache_seq_rm(
//    struct llama_context* ctx,
//    llama_seq_id    seq_id,  // The sequence ID to target (-1 for ALL sequences)
//    int32_t        p0,      // Starting token position (-1 for the very beginning)
//    int32_t        p1       // Ending token position (-1 for up to the very end)
//);
//
//static inline void engine_kv_clear(struct llama_context* ctx) {
//    llama_kv_cache_seq_rm(ctx, -1, 0, INT32_MAX);
//}

//static inline void engine_kv_clear(struct llama_context* ctx) {
//    // Universal KV reset for all llama.cpp versions
//    llama_set_state_data(ctx, NULL);
//}



char* engine_json_extract_string(char* s, char* out, size_t out_sz)
{
    size_t pos = 0;
    s++; // skip opening quote

    while (*s && pos < out_sz - 1) {
        if (*s == '\\') {
            s++;
            if (*s == 'n') out[pos++] = '\n';
            else if (*s == 't') out[pos++] = '\t';
            else if (*s == '\\') out[pos++] = '\\';
            else if (*s == '"') out[pos++] = '"';
            else out[pos++] = *s;
            s++;
        }
        else if (*s == '"') {
            s++; // closing quote
            break;
        }
        else {
            out[pos++] = *s++;
        }
    }

    out[pos] = 0;
    return s;
}




void sanitize(char* s) {
    for (int i = 0; s[i]; i++) {
        if (s[i] == '<' || s[i] == '|' || s[i] == '>') {
            s[i] = ' ';
        }
    }
}

//void engine_reset_context(engine_t* e) {
//    if (e->ctx) {
//        llama_free(e->ctx);
//        e->ctx = NULL;
//    }
//
//    struct llama_context_params params;
//
//    params.n_ctx = e->n_ctx;
//    params.n_parts = -1;
//    params.seed = 0;
//    params.f16_kv = 1;
//    params.logits_all = 0;
//    params.vocab_only = 0;
//    params.use_mmap = 1;
//    params.use_mlock = 0;
//
//    e->ctx = llama_init_from_file(e->model_path, params);
//}



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
    //out[0] = 0;  // <-- REQUIRED
    // FULL CLEAR — required
    memset(out, 0, out_sz);
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



//void engine_append_turn_html(engine_t* e, const char* role, const char* text) {
//    if (e->html_n_turns >= MAX_TURNS) {
//        for (int i = 1; i < e->html_n_turns; ++i)
//            e->html_turns[i - 1] = e->html_turns[i];
//        e->html_n_turns--;
//    }
//
//    html_turn_t* t = &e->html_turns[e->html_n_turns++];
//
//    memset(t, 0, sizeof(*t));   // <-- REQUIRED
//
//    strncpy(t->role, role, sizeof(t->role) - 1);
//    strncpy(t->text, text, sizeof(t->text) - 1);
//}

void engine_append_turn_html(engine_t* e, const char* role, const char* text) {
    if (!role || !text) return;

    // Clamp turn count
    if (e->html_n_turns >= MAX_TURNS) {
        // Shift left
        for (int i = 1; i < MAX_TURNS; i++) {
            e->html_turns[i - 1] = e->html_turns[i];
        }
        e->html_n_turns = MAX_TURNS - 1;
    }

    html_turn_t* t = &e->html_turns[e->html_n_turns++];

    // Zero the struct
    memset(t, 0, sizeof(*t));

    // Safe copy
    strncpy(t->role, role, sizeof(t->role) - 1);
    strncpy(t->text, text, sizeof(t->text) - 1);

    // Ensure null termination
    t->role[sizeof(t->role) - 1] = 0;
    t->text[sizeof(t->text) - 1] = 0;
}



//void engine_build_prompt_html(engine_t* e, char* out, size_t out_sz) {
//    out[0] = 0;  // <-- REQUIRED
//
//    strcat(out, "<|system|>\nYou are a helpful assistant.\n\n");
//
//    for (int i = 0; i < e->html_n_turns; i++) {
//        strcat(out, "<|");
//        strcat(out, e->html_turns[i].role);
//        strcat(out, "|>\n");
//        strcat(out, e->html_turns[i].text);
//        strcat(out, "\n\n");
//    }
//
//    strcat(out, "<|assistant|>\n");
//}

void engine_build_prompt_html(engine_t* e, char* out, size_t out_sz)
{
    if (!out || out_sz == 0) return;
    memset(out, 0, out_sz);

    // Prepare messages
    struct llama_chat_message messages[64];
    int n_messages = 0;

    // System message
    messages[n_messages].role = "system";
    messages[n_messages].content = "You are a helpful, friendly assistant. "
        "Respond naturally in plain text. "
        "Never output links, references, or long unrelated text unless asked.";
    n_messages++;

    // Add conversation history
    for (int i = 0; i < e->html_n_turns && n_messages < 60; i++) {
        html_turn_t* t = &e->html_turns[i];
        if (!t->role[0] || !t->text[0]) continue;

        messages[n_messages].role = t->role;
        messages[n_messages].content = t->text;
        n_messages++;
    }

    // Apply Llama 3.1 template (correct signature)
    int32_t ret = llama_chat_apply_template(
        NULL,                          // tmpl = NULL → use model's default template
        messages,
        n_messages,
        true,                          // add_generation_prompt = true
        out,
        (int32_t)out_sz
    );

    if (ret < 0) {
        // Fallback if template fails
        strncpy(out, "<|begin_of_text|><|start_header_id|>assistant<|end_header_id|>\n", out_sz - 1);
    }
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

    engine_kv_clear(e->ctx);

    if (llama_decode(e->ctx, batch) != 0) {
        fprintf(stderr, "[engine] decode(prompt) failed\n");
        //llama_batch_free(batch);
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

// Reset the context / KV cache
//void engine_reset_context(engine_t* e)
//{
//    if (!e || !e->ctx) return;
//
//    struct llama_memory* mem = llama_get_memory(e->ctx);
//    if (mem) {
//        llama_memory_clear(mem, true);        // Current recommended way (2026)
//    }
//
//    // Also reset your custom tracking variables
//    e->pos = 0;
//    // e->seq_id = 0;     // Uncomment only if you want to reset sequence ID too
//}

// Reset the context / KV cache - Best for Llama 3.1

// Reset the context / KV cache - Compatible with latest llama.cpp (June 2026)
// Best version right now
void engine_reset_context(engine_t* e)
{
    if (!e || !e->ctx) return;

    struct llama_memory* mem = llama_get_memory(e->ctx);
    if (mem) {
        llama_memory_clear(mem, true);
    }

    e->pos = 0;
    // Do NOT let seq_id grow forever
    e->seq_id = 0;        // ← Reset to 0 every time (safest)
}

#pragma message("CHECKING LLAMA API...")

#ifdef LLAMA_CONTEXT_PARAMS
#pragma message("HAS LLAMA_CONTEXT_PARAMS")
#else
#pragma message("NO LLAMA_CONTEXT_PARAMS")
#endif

int engine_generate_html(
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
    if (!e || !e->model || !e->ctx || !prompt || !out || out_size == 0) return -1;
    out[0] = '\0';
    if (prompt[0] == '\0') return 0;

    max_tokens = 512;   // better default

    const struct llama_vocab* vocab = llama_model_get_vocab(e->model);

    // 1. Tokenize
       
    // 2. Full reset before every prompt
    engine_reset_context(e);
    e->pos = 0;
    e->seq_id++;                     // increment every new message

    // 3. Tokenize
    const int max_prompt_tokens = 4096;
    llama_token* tokens = malloc(max_prompt_tokens * sizeof(llama_token));
    if (!tokens) return -1;

    int n_prompt = llama_tokenize(vocab, prompt, (int32_t)strlen(prompt),
        tokens, max_prompt_tokens, true, false);

    if (n_prompt <= 0) {
        free(tokens);
        return -1;
    }

    // 4. Create and fill batch (correct modern way)
    struct llama_batch batch = llama_batch_init(n_prompt, 0, 1);  // n_seq_max = 1

    for (int i = 0; i < n_prompt; i++) {
        batch.token[i] = tokens[i];
        batch.pos[i] = i;                    // absolute position
        batch.n_seq_id[i] = 1;
        //batch.seq_id[i][0] = e->seq_id;            // ← This was the main crash source
        batch.seq_id[i][0] = 0;           // ← Always use 0 for simple chat
        batch.logits[i] = (i == n_prompt - 1) ? 1 : 0;
    }
    batch.n_tokens = n_prompt;

    // Decode prompt
    if (llama_decode(e->ctx, batch) != 0) {
        printf("ERROR: llama_decode failed on prompt (n_tokens=%d)\n", n_prompt);
        llama_batch_free(batch);
        free(tokens);
        return -1;
    }

    llama_batch_free(batch);
    free(tokens);

    e->pos = n_prompt;

    // 5. Generation
    size_t out_len = 0;
    int n_gen = 0;

    while (n_gen < max_tokens && out_len + 32 < out_size) {
        llama_token tok = sample_next(e, temp, top_k, top_p);

        if (tok == llama_token_eos(e->model) || tok < 0)
            break;

        char buf[256] = { 0 };
        int n = llama_token_to_piece(vocab, tok, buf, sizeof(buf), 0, false);
        if (n <= 0) break;

        // Append text (skip special tokens)
        if (n > 0 && !strstr(buf, "<|") && !strchr(buf, '|')) {
            if (out_len + (size_t)n < out_size) {
                memcpy(out + out_len, buf, n);
                out_len += n;
                out[out_len] = '\0';
            }
        }

        // === CORRECT SINGLE TOKEN BATCH ===
        llama_pos      pos = e->pos;
        int32_t        n_seq = 1;
        llama_seq_id   seq_id = e->seq_id;
        llama_seq_id* seq_ptr = &seq_id;           // ← Important

        struct llama_batch loop_batch = llama_batch_init(1, 0, 1);

        loop_batch.token[0] = tok;
        loop_batch.pos[0] = pos;
        loop_batch.n_seq_id[0] = n_seq;
        //loop_batch.seq_id[0][0] = seq_id;           // Correct way
        loop_batch.seq_id[0][0] = 0;     // ← Use 0
        loop_batch.logits[0] = 1;

        if (llama_decode(e->ctx, loop_batch) != 0) {
            llama_batch_free(loop_batch);
            break;
        }

        llama_batch_free(loop_batch);

        e->pos++;
        n_gen++;
    }
    return (int)out_len;
}


//int engine_generate_html(
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
//    if (!e || !e->model || !e->ctx || !prompt || !out || out_size == 0) return -1;
//    out[0] = '\0';
//    if (prompt[0] == '\0') return 0;
//
//    max_tokens = 512;   // you can make this a parameter later 128
//    temp = 0.7f;
//    top_k = 20;
//    top_p = 0.9f;
//
//    const struct llama_vocab* vocab = llama_model_get_vocab(e->model);
//
//    // 1. Tokenize full prompt
//    const int max_prompt_tokens = 4096;
//    llama_token* tokens = malloc(max_prompt_tokens * sizeof(llama_token));
//    if (!tokens) return -1;
//
//    int n_prompt = llama_tokenize(vocab, prompt, (int32_t)strlen(prompt),
//        tokens, max_prompt_tokens, true, false);
//
//    if (n_prompt <= 0) {
//        free(tokens);
//        return -1;
//    }
//
//    // 2. Full reset (recommended for your use case)
//    engine_reset_context(e);
//    struct llama_memory* mem = llama_get_memory(e->ctx);
//    if (mem) {
//        llama_memory_clear(mem, true);        // this is the current way
//    }
//
//    // 3. Evaluate prompt (one big batch)
//    struct llama_batch batch = llama_batch_init(n_prompt, 0, 1);
//
//    for (int i = 0; i < n_prompt; i++) {
//        batch.token[i] = tokens[i];
//        batch.pos[i] = i;
//        batch.n_seq_id[i] = 1;
//        batch.seq_id[i][0] = 0;               // sequence 0
//        batch.logits[i] = (i == n_prompt - 1) ? 1 : 0;
//    }
//    batch.n_tokens = n_prompt;
//
//    if (llama_decode(e->ctx, batch) != 0) {
//        llama_batch_free(batch);
//        free(tokens);
//        return -1;
//    }
//
//    e->pos = n_prompt;
//    llama_batch_free(batch);
//    free(tokens);
//
//    // 4. Generation loop
//    size_t out_len = 0;
//    int n_gen = 0;
//
//    while (n_gen < max_tokens && out_len + 64 < out_size) {
//        llama_token tok = sample_next(e, temp, top_k, top_p);
//
//        if (tok == llama_token_eos(e->model)) break;
//
//        char buf[128] = { 0 };
//        int n = llama_token_to_piece(vocab, tok, buf, sizeof(buf), 0, false);
//        if (n <= 0) break;
//
//        // Skip unwanted role tags in output
//        if (!strstr(buf, "<|")) {
//            if (out_len + (size_t)n >= out_size) break;
//            memcpy(out + out_len, buf, n);
//            out_len += n;
//            out[out_len] = '\0';
//        }
//
//        // Decode next token
//        llama_pos      pos = e->pos;
//        int32_t        n_seq = 1;
//        llama_seq_id   seq = e->seq_id;
//        llama_seq_id* seq_ptr = &seq;
//        int8_t         logits = 1;
//
//        struct llama_batch loop_batch = {
//            .n_tokens = 1,
//            .token = &tok,
//            .embd = NULL,
//            .pos = &pos,
//            .n_seq_id = &n_seq,
//            .seq_id = &seq_ptr,
//            .logits = &logits,
//        };
//
//        if (llama_decode(e->ctx, loop_batch) != 0) break;
//
//        e->pos++;
//        n_gen++;
//    }
//
//    return (int)out_len;
//}
// 
// 
// 
// ===================================================================
// Generate html
// ===================================================================
//int engine_generate_html(
//    engine_t* e,
//    const char* prompt,
//    char* out,
//    size_t      out_size,
//    int         max_tokens,
//    float       temp,
//    int         top_k,
//    float       top_p,
//    bool        stream
//) {
//    if (!e || !e->model || !e->ctx || !prompt || !out || out_size == 0) return -1;
//    out[0] = '\0';
//    if (prompt[0] == '\0') return 0;
//
//    max_tokens = 128;
//    temp = 0.7f;
//    top_k = 20;
//    top_p = 0.9f;
//
//    const struct llama_vocab* vocab = llama_model_get_vocab(e->model);
//
//    // ============================================================
//    // 1. MEASURE IMMUTABLE SYSTEM PROMPT
//    // ============================================================
//    const char* system_str = "<|system|>\nYou are a helpful assistant.\n\n";
//    //llama_token sys_token_buf;
//    llama_token sys_token_buf[128] = { 0 };
//
//    int32_t system_prompt_len = llama_tokenize(
//        vocab,
//        system_str,
//        (int32_t)strlen(system_str),
//        sys_token_buf, // Now safely passes the address of our 128-slot array
//        128,
//        true,
//        false
//    );
//
//    if (system_prompt_len <= 0) return -1;
//
//    engine_reset_context(e); 
//
//    // ============================================================
//    // 2. DISCARD HISTORICAL LAYERS FROM CONTEXT CACHE
//    // ============================================================
//    struct llama_memory* mem = llama_get_memory(e->ctx);
//    if (mem) {
//        int32_t max_context_bound = llama_n_ctx(e->ctx);
//
//        // If an HTTP thread crossover, plugin execution, or state desync is caught:
//        if (system_prompt_len <= 0 || system_prompt_len >= max_context_bound || e->pos < system_prompt_len) {
//
//            //  THE TRUE UNIFIED MEMORY SYSTEM CLEAN RESET:
//            // By using llama_memory_clear with 'true', the library drops all active attention 
//            // allocations across every sequence simultaneously. It does zero range testing, 
//            // completely avoiding the ggml-base.dll abort() crash!
//            llama_memory_clear(mem, true);
//
//            // Force the prompt tracking string to be fully re-evaluated from the absolute beginning
//            system_prompt_len = 0;
//            e->pos = 0;
//        }
//        else {
//            // Normal conversation turn: Safe to remove trailing elements of the current active sequence
//            // Using -1 for the final parameter in your era means "up to the end of the line"
//            llama_memory_seq_rm(mem, 0, system_prompt_len, -1);
//        }
//    }
//
//    // ============================================================
//    // 3. PARSE FULL COMBINED PROMPT STRING
//    // ============================================================
//    const int max_prompt_tokens = 4096;
//    llama_token* tokens = malloc(max_prompt_tokens * sizeof(llama_token));
//    if (!tokens) return -1;
//
//    int n_prompt = llama_tokenize(
//        vocab, prompt, (int32_t)strlen(prompt),
//        tokens, max_prompt_tokens,
//        true,
//        false
//    );
//    if (n_prompt <= 0) {
//        free(tokens);
//        return -1;
//    }
//
//    if (n_prompt < system_prompt_len) {
//        system_prompt_len = 0;
//    }
//
//    int32_t tokens_to_add = n_prompt - system_prompt_len;
//
//
//// ============================================================
//// 4. PROMPT EVALUATION PHASE (FIXED ARRAY BASELINE OFFSET POINTER)
//// ============================================================
//    int decode_rc = 0;
//    int64_t cur_pos = system_prompt_len;
//
//    if (tokens_to_add > 0) {
//#define PROMPT_MAX_CAP 2048
//
//        if (tokens_to_add > PROMPT_MAX_CAP) {
//            free(tokens);
//            return -1;
//        }
//
//        llama_pos      p_pos[PROMPT_MAX_CAP];
//        int32_t        p_n_seq[PROMPT_MAX_CAP];
//        llama_seq_id   p_seq_ids[PROMPT_MAX_CAP];
//        llama_seq_id* p_seq_ptrs[PROMPT_MAX_CAP];
//        int8_t         p_logits[PROMPT_MAX_CAP];
//
//        struct llama_batch prompt_batch;
//
//        prompt_batch.n_tokens = 0;
//
//        //  FIX: Point to the raw base address of the tokens array!
//        // The loop below will use token indexing to read tokens starting from the system prompt length offset.
//        prompt_batch.token = tokens;
//        prompt_batch.embd = NULL;
//        prompt_batch.pos = p_pos;
//        prompt_batch.n_seq_id = p_n_seq;
//        prompt_batch.seq_id = p_seq_ptrs;
//        prompt_batch.logits = p_logits;
//
//        for (int i = 0; i < tokens_to_add; ++i) {
//            // Calculate the actual token array offset position index cleanly
//            int token_array_index = system_prompt_len + i;
//
//            p_pos[i] = (llama_pos)cur_pos;
//            p_n_seq[i] = 1;
//            p_seq_ids[i] = 0; // Keep sequence 0 bound
//            p_seq_ptrs[i] = &p_seq_ids[i];
//            p_logits[i] = (i == tokens_to_add - 1) ? 1 : 0;
//
//            // Copy the token from its offset location to the active batch slot position
//            // This guarantees llama_decode reads real tokens instead of unallocated heap space!
//            tokens[i] = tokens[token_array_index];
//
//            prompt_batch.n_tokens++;
//            cur_pos++;
//        }
//
//        if (prompt_batch.n_tokens > 0) {
//            decode_rc = llama_decode(e->ctx, prompt_batch);
//        }
//
//#undef PROMPT_MAX_CAP
//    }
//
//    free(tokens);
//
//    if (decode_rc != 0) {
//        return -1;
//    }
//
//    e->pos = cur_pos;
//
//
//
//// ============================================================
//// 5. GENERATION / SAMPLING LOOP (PURE STACK C REUSE)
//// ============================================================
//    size_t out_len = 0;
//    int n_gen = 0;
//
//    while (n_gen < max_tokens && out_len + 8 < out_size) {
//        llama_token tok = sample_next(e, temp, top_k, top_p);
//
//        // Stop on EOS
//        if (tok == llama_token_eos(e->model)) break;
//
//        // Convert token to text
//        char buf[128] = { 0 };
//        int n = llama_token_to_piece(vocab, tok, buf, sizeof(buf), 0, false);
//        if (n <= 0) break;
//
//        // BLOCK ONLY EXACT ROLE TAG TOKENS
//        bool skip_output = false;
//        if (strcmp(buf, "<|assistant|>") == 0 || strcmp(buf, "<|user|>") == 0 ||
//            strcmp(buf, "<|system|>") == 0 || strcmp(buf, "assistant") == 0 ||
//            strcmp(buf, "/assistant") == 0) {
//            skip_output = true;
//        }
//
//        // APPEND OUTPUT (ONLY IF NOT SKIPPED)
//        if (!skip_output) {
//            if (out_len + (size_t)n >= out_size) break;
//            memcpy(out + out_len, buf, n);
//            out_len += n;
//            out[out_len] = '\0';
//        }
//
//        // ============================================================
//        // FIX: LINE-BY-LINE MSVC C ASSIGNMENT (NO COMPOUND LITERALS)
//        // ============================================================
//        // Declare backing arrays explicitly on the thread stack
//        llama_pos      stack_pos = (llama_pos)cur_pos;
//        int32_t        stack_n_seq = 1;
//        llama_seq_id   stack_seq_id = e->seq_id;
//        llama_seq_id* stack_seq_ptr = &stack_seq_id;
//        int8_t         stack_output = 1; // 1 means generate logits for this token
//
//        // Allocate a blank struct on the stack
//        struct llama_batch loop_batch;
//
//        // Populate fields line-by-line to comply with MSVC's standard C engine
//        loop_batch.n_tokens = 1;
//        loop_batch.token = &tok;
//        loop_batch.embd = NULL;
//        loop_batch.pos = &stack_pos;
//        loop_batch.n_seq_id = &stack_n_seq;
//        loop_batch.seq_id = &stack_seq_ptr;
//
//        //  UNCOMMENTED AND MAPPED TO NATIVE FIELD NAME:
//        loop_batch.logits = &stack_output;
//
//        // Pass the stack-allocated structure directly to decode
//        if (llama_decode(e->ctx, loop_batch) != 0) {
//            break; // Exit if context length is exceeded or desynced
//        }
//
//        ++n_gen;
//        ++cur_pos;
//    }
//
//    // Persist final position pointer safely back into state engine
//    e->pos = cur_pos;
//
//    return (int)out_len;
//}




//
//// ===================================================================
//// Generate html
//// ===================================================================
//int engine_generate_html(
//    engine_t* e,
//    const char* prompt,
//    char* out,
//    size_t out_size,
//    int max_tokens,
//    float temp,
//    int top_k,
//    float top_p,
//    bool stream   // MUST be false for HTML
//) {
//    if (!e || !e->model || !e->ctx || !prompt || !out || out_size == 0)
//        return -1;
//
//    out[0] = '\0';
//    if (prompt[0] == '\0')
//        return 0;
//
//    // HTML defaults
//    max_tokens = 128;
//    temp = 0.7f;
//    top_k = 20;
//    top_p = 0.9f;
//
//    // ============================================================
//    // TOKENIZE PROMPT
//    // ============================================================
//    const int max_prompt_tokens = 4096;
//    llama_token* tokens = malloc(max_prompt_tokens * sizeof(llama_token));
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
//        max_prompt_tokens,
//        true,   // add BOS
//        false   // no special tokens
//    );
//
//    if (n_prompt <= 0) {
//        free(tokens);
//        return -1;
//    }
//
//    // ============================================================
//    // BATCH INIT
//    // ============================================================
//    llama_batch batch = llama_batch_init(1024, 0, 1);
//
//    int64_t cur_pos = e->pos;   // HTML chat resets pos before calling this
//
//    // ============================================================
//    // FEED PROMPT TOKENS
//    // ============================================================
//    for (int i = 0; i < n_prompt; ++i) {
//        llama_batch_add(
//            &batch,
//            tokens[i],
//            (llama_pos)cur_pos,
//            (const int[]) {
//            e->seq_id
//        },   // ⭐ correct KV namespace
//            i == n_prompt - 1             // logits on last prompt token
//        );
//        cur_pos++;
//    }
//    ///////////////////////////////////////////////////////////////
//   //struct llama_kv_cache_seq_rm(e->ctx, -1, -1, -1);
//   // engine_kv_clear(e->ctx);
//
//    //////////////////////////////////////////////////
//
//    if (llama_decode(e->ctx, batch) != 0) {
//        //llama_batch_free(batch);
//        free(tokens);
//        return -1;
//    }
//
//    // ============================================================
//    // GENERATION LOOP (patched, no infinite loops)
//    // ============================================================
//    size_t out_len = 0;
//    int n_gen = 0;
//
//    while (n_gen < max_tokens && out_len + 8 < out_size) {
//
//        llama_token tok = sample_next(e, temp, top_k, top_p);
//
//        // Stop on EOS
//        if (tok == llama_token_eos(e->model))
//            break;
//
//        // Convert token to text
//        char buf[128] = { 0 };
//        int n = llama_token_to_piece(
//            vocab,
//            tok,
//            buf,
//            sizeof(buf),
//            0,
//            false
//        );
//
//        if (n <= 0)
//            break;
//
//        // ============================================================
//        // BLOCK ONLY EXACT ROLE TAG TOKENS — NOT SUBSTRINGS
//        // ============================================================
//        bool skip_output = false;
//
//        if (strcmp(buf, "<|assistant|>") == 0 ||
//            strcmp(buf, "<|user|>") == 0 ||
//            strcmp(buf, "<|system|>") == 0 ||
//            strcmp(buf, "assistant") == 0 ||
//            strcmp(buf, "/assistant") == 0)
//        {
//            skip_output = true;
//        }
//
//        // ============================================================
//        // APPEND OUTPUT (ONLY IF NOT SKIPPED)
//        // ============================================================
//        if (!skip_output) {
//            if (out_len + (size_t)n >= out_size)
//                break;
//
//            memcpy(out + out_len, buf, n);
//            out_len += n;
//            out[out_len] = '\0';
//        }
//
//        // ============================================================
//        // ALWAYS ADVANCE THE MODEL — EVEN IF SKIPPED
//        // ============================================================
//        llama_batch_clear(&batch);
//        llama_batch_add(
//            &batch,
//            tok,
//            (llama_pos)cur_pos,
//            (const int[]) {
//            e->seq_id
//        },   // same KV namespace
//            true
//        );
//
//        if (llama_decode(e->ctx, batch) != 0)
//            break;
//
//        ++n_gen;
//        ++cur_pos;
//    }
//
//
//    free(tokens);
//
//    // Persist new position
//    e->pos = cur_pos;
//
//    return (int)out_len;
//}



// ===================================================================
//     Working Open
/// ===================================================================
engine_t* engine_open(const char* path) {
    fprintf(stderr, "[engine] loading model: %s\n", path);

    // -----------------------------
    // Model params
    // -----------------------------
    struct llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 999;

    struct llama_model* model = llama_load_model_from_file(path, mp);
    if (!model) {
        fprintf(stderr, "[engine] llama_load_model_from_file failed\n");
        return NULL;
    }

    if (engine_has_gpu()) {
        fprintf(stderr, "[engine] GPU backend detected — enabling full GPU offload\n");
    }
    else {
        fprintf(stderr, "[engine] No GPU backend — falling back to CPU\n");
        mp.n_gpu_layers = 0;
    }

    // -----------------------------
    // Context params (CPU optimized)
    // -----------------------------
    struct llama_context_params cp = llama_context_default_params();
    cp.n_ctx = 2048;
    cp.n_batch = 1024;
    cp.n_threads = 0;

    // ⭐ REQUIRED FOR seq_id++ TO WORK ⭐
    cp.n_seq_max = 4;   // <-- THIS FIXES YOUR ERROR

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
    e->pos = 0;
    e->seq_id = 0;   // ⭐ REPL uses sequence 0

    fprintf(stderr, "[engine] model loaded successfully!\n");
    return e;
}



int engine_chat(engine_t* e,
    const char* user_msg,
    char* out,
    size_t out_size)
{
   
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

//////////////////////////////////////
////////// HTML Only
////////////////////////////
int engine_chat_html(
    engine_t* e,
    const char* user_input,
    char* out,
    size_t out_size)
{
    int rc = 0;

    // ⭐ 2. NORMAL CHAT LOGIC CONTINUES BELOW
    // build prompt, call engine_generate_html, etc.

    if (!e || !user_input || !out || out_size == 0)
        return -1;

    out[0] = '\0';

    // === FORCE CLEAN STATE ===
   
    //e->pos = 0;
    engine_reset_context(e);           // ← Make sure this does llama_memory_clear + kv_cache_clear
     //e->seq_id++;

    //
    // ⭐ 1. Reset KV cache for HTML chat
    //    (modern llama.cpp: seq_id++ = new KV namespace)
    //

    //e->seq_id++;
    //e->pos = 0;

// ============================================================
// HTML Plugin Router (argc/argv version)
// ============================================================
	char* Plugin_result = NULL;
	char* temp = NULL; 
	const char* query = NULL;
    const char* user_input_query = NULL;
    if (_strnicmp(user_input, "ddg ", 4) == 0) {
        query = user_input + 4;
        user_input_query = query;
        temp = _strdup(query);
        char* token = strtok(temp, " ");
        char* argvv[32];
        int argcv = 0;

        while (token && argcv < 32) {
            argvv[argcv++] = token;
            token = strtok(NULL, " ");
        }

        plugin_fn fn = plugin_lookup("ddg");
        if (!fn) {
            printf("ERR unknown_plugin\n");
            return;
        }

        Plugin_result = fn(argcv, argvv);
        if (Plugin_result) {}
          //  printf("%s\n", Plugin_result);
        else {
            free(Plugin_result);
            free(temp);
            return 0;
        }
    }
      
    if (_strnicmp(user_input, "websearch ", 10) == 0) {
        query = user_input + 10;
        user_input_query = query;
         temp = _strdup(query);
        char* token = strtok(temp, " ");
        char* argvv[32];
        int argcv = 0;

        while (token && argcv < 32) {
            argvv[argcv++] = token;
            token = strtok(NULL, " ");
        }

        plugin_fn fn = plugin_lookup("websearch");
        if (!fn) {
            printf("ERR unknown_plugin\n");
            return;
        }

        Plugin_result = fn(argcv, argvv);
        if (Plugin_result){}
           // printf("%s\n", Plugin_result);
        else {
            free(Plugin_result);
            free(temp);
            return 0;
        }      
    }

    if (_strnicmp(user_input, "summarize_file ", 15) == 0) {

        const char* raw = user_input + 15;
        while (*raw == ' ' || *raw == '\t') raw++;

        // Clean JSON escapes
        char clean_path[4096];
        engine_json_extract_string((char*)raw - 1, clean_path, sizeof(clean_path));

        // Run plugin directly
        char* argvv[1];
        argvv[0] = clean_path;

        g_plugin_result = plugin_summarize_file_html(1, argvv);

        //return 0;   // ⭐ DO NOT fall through to chat LLM

        if (g_plugin_result) {/// working but cut off
            char combined[65536];

            // Combine plugin output + rc (converted to string) + LLM output
            snprintf(combined, sizeof(combined),
                "%s\n[rc=%d]\n%s",
                g_plugin_result, rc, out);

            // Copy back into out buffer
            strncpy(out, combined, out_size - 1);
            out[out_size - 1] = 0;

            free(g_plugin_result);
        }

        return rc;
    }
      
    //
    // ⭐ 2. Append USER turn to HTML history
    //
    if (user_input_query)
    {
        user_input = NULL;
		user_input = user_input_query;
    }
    else 
    {
        user_input;
    }
   

    engine_append_turn_html(e, "user", user_input);

    //
    // ⭐ 3. Build prompt from HTML history only
    //
    char prompt[65536];
    engine_build_prompt_html(e, prompt, sizeof(prompt));

    //
    // ⭐ 4. Generate (NO STREAMING TO TERMINAL)
    //
    rc = engine_generate_html(
        e,
        prompt,
        out,
        out_size,
        128,      // max tokens
        0.7f,     // temp
        20,       // top_k
        0.9f,     // top_p
        false     // stream = false (HTML must NOT print to terminal)
    );

    if (rc < 0)
        return rc;

    //
    // ⭐ 5. Append ASSISTANT turn to HTML history
    //
    engine_append_turn_html(e, "assistant", out);

   /* if (Plugin_result) {
	rc = Plugin_result + '\n' + rc;
    }
    else
    {
        return rc;
    }*/
    
    if (Plugin_result) {/// working but cut off
        char combined[65536];

        // Combine plugin output + rc (converted to string) + LLM output
        snprintf(combined, sizeof(combined),
            "%s\n[rc=%d]\n%s",
            Plugin_result, rc, out);

        // Copy back into out buffer
        strncpy(out, combined, out_size - 1);
        out[out_size - 1] = 0;

        free(Plugin_result);
    }

    return rc;

}

///////////////////////////////////////////////////////////////////////////////////////////////////////

int engine_generate_plugin(
    engine_t* e,
    const char* prompt,
    char* out,
    size_t out_size,
    int max_tokens,
    float temp,
    int top_k,
    float top_p
) {
    if (!e || !e->model || !e->ctx || !prompt || !out || out_size == 0)
        return -1;

    out[0] = '\0';
    if (prompt[0] == '\0')
        return 0;

    const int max_prompt_tokens = 4096;
    llama_token* tokens = malloc(max_prompt_tokens * sizeof(llama_token));
    if (!tokens)
        return -1;

    const struct llama_vocab* vocab = llama_model_get_vocab(e->model);

    int n_prompt = llama_tokenize(
        vocab,
        prompt,
        (int32_t)strlen(prompt),
        tokens,
        max_prompt_tokens,
        true,   // add BOS
        false   // no special tokens
    );

    if (n_prompt <= 0) {
        free(tokens);
        return -1;
    }

    // 🔹 Plugin: always start from pos 0, seq_id 0
    llama_batch batch = llama_batch_init(1024, 0, 1);
    int64_t cur_pos = 0;
    int32_t seq_id = 0;

    for (int i = 0; i < n_prompt; ++i) {
        llama_batch_add(
            &batch,
            tokens[i],
            (llama_pos)cur_pos,
            (const int[]) {
            seq_id
        },
            i == n_prompt - 1
        );
        cur_pos++;
    }

    if (llama_decode(e->ctx, batch) != 0) {
        llama_batch_free(batch);
        free(tokens);
        return -1;
    }

    // TODO: your token sampling loop here, same pattern, but using cur_pos++, seq_id=0

    llama_batch_free(batch);
    free(tokens);
    return 0;
}

