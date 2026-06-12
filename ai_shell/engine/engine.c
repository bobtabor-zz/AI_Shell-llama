// engine.c – incremental llama.cpp-style engine for Llama-3-Instruct

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>

#include "llama.h"
#include "engine.h"   // defines engine_t, html_turn_t, engine_turn_t, MAX_TURNS
#include "../include/plugin.h"

char* g_plugin_result = NULL;


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


// ------------------------------------------------------------
// Runtime init
// ------------------------------------------------------------

void engine_init_runtime(engine_t* e) {
    if (!e) return;

    e->seq_id = 0;
    e->html_seq_id = 0;
    e->pos = 0;

    e->temp = 0.7f;
    e->top_k = 20;
    e->top_p = 0.9f;

    e->kv_valid = false;
    e->kv_len = 0;

    e->html_n_turns = 0;
    e->n_turns = 0;
}

// ------------------------------------------------------------
// Llama-3 chat template wrappers (manual, no <|begin_of_text|>)
// ------------------------------------------------------------

static void engine_wrap_llama3_system(char* dst, size_t dst_size, const char* system_text) {
    snprintf(dst, dst_size,
        "<|start_header_id|>system<|end_header_id|>\n\n" // Fixed: Added extra \n
        "%s<|eot_id|>\n",
        system_text
    );
}

static void engine_wrap_llama3_user(char* dst, size_t dst_size, const char* user_text) {
    snprintf(dst, dst_size,
        "<|start_header_id|>user<|end_header_id|>\n\n" // Fixed: Added extra \n
        "%s<|eot_id|>\n"
        "<|start_header_id|>assistant<|end_header_id|>\n\n", // Fixed: Added extra \n
        user_text
    );
}


// ------------------------------------------------------------
// Simple sampler (greedy; swap with your sampler if desired)
// ------------------------------------------------------------

static llama_token engine_sample_next(engine_t* e) {
    const struct llama_vocab* vocab = llama_model_get_vocab(e->model);
    const float* logits = llama_get_logits(e->ctx);
    int n_vocab = llama_vocab_n_tokens(vocab);

    int best_id = 0;
    float best_logit = logits[0];
    for (int i = 1; i < n_vocab; ++i) {
        if (logits[i] > best_logit) {
            best_logit = logits[i];
            best_id = i;
        }
    }
    return (llama_token)best_id;
}

// ------------------------------------------------------------
// Feed system prompt (templated) ONCE per reset/open
// ------------------------------------------------------------

int engine_feed_system_prompt(engine_t* e, const char* system_text) {
    if (!e || !e->model || !e->ctx) return -1;

    const struct llama_vocab* vocab = llama_model_get_vocab(e->model);

    char buf[4096];
    engine_wrap_llama3_system(buf, sizeof(buf), system_text);

    llama_token tokens[1024];
    int n_tokens = llama_tokenize(
        vocab,
        buf,
        (int32_t)strlen(buf),
        tokens,
        1024,
        false,  // add_special: false (we wrap them manually)
        true    // FIX: parse_special MUST be true to recognize Llama 3 <|control|> tags!
    );
    if (n_tokens <= 0) return -1;

    struct llama_batch batch = (struct llama_batch){ 0 };

    llama_pos      pos_arr[1024];
    int32_t        n_seq_arr[1024];
    llama_seq_id   seq_id_arr[1024];
    llama_seq_id* seq_ptr_arr[1024];
    int8_t         logits_arr[1024];

    // Array bindings matching your llama.cpp layout specification
    batch.token = tokens;
    batch.pos = pos_arr;
    batch.n_seq_id = n_seq_arr;
    batch.seq_id = seq_ptr_arr;
    batch.logits = logits_arr;
    batch.n_tokens = 0;

    llama_pos pos = e->pos;

    for (int i = 0; i < n_tokens; ++i) {
        pos_arr[i] = pos;
        n_seq_arr[i] = 1;
        seq_id_arr[i] = e->seq_id;
        seq_ptr_arr[i] = &seq_id_arr[i];
        logits_arr[i] = (i == n_tokens - 1) ? 1 : 0; // Request logits only for the very last token

        batch.n_tokens++;
        pos++;
    }

    // Direct initialization pass evaluation
    int decode_rc = llama_decode(e->ctx, batch);
    //if (decode_rc != 0) {
    //    return -1; // Aborts back to engine_open cleanly if initialization fails
    //}
    int32_t n_past = batch.pos[0];
    if (decode_rc != 0) {
        llama_memory_t mem = llama_get_memory(e->ctx);
        if (mem) {
            const int32_t system_prompt_tokens = 512;
            const int32_t discard_chunk = 512;

            const int32_t remove_start = system_prompt_tokens;
            const int32_t remove_end = system_prompt_tokens + discard_chunk;

            // 1. Remove the targeted history block
            //llama_memory_seq_rm(mem, 0, remove_start, remove_end);
            llama_memory_seq_rm(mem, e->seq_id, remove_start, remove_end);

            // 2. Slide subsequent context slots back
            //llama_memory_seq_add(mem, 0, remove_end, -1, -discard_chunk);

            llama_memory_seq_add(mem, e->seq_id, remove_end, -1, -discard_chunk);

            // 3. RE-ALIGNMENT FIX:
            // Read where the KV cache physically ends after the shift.
            // For sequence 0, the next valid token position MUST be exactly 512 (X + 1)
            e->pos -= discard_chunk;

            // Explicitly write the corrected positions directly into the batch arrays
            for (int32_t i = 0; i < batch.n_tokens; ++i) {
                batch.pos[i] = e->pos + i;
            }

            // 4. Retry the decoding operations with clean positions
            decode_rc = llama_decode(e->ctx, batch);
            if (decode_rc != 0) {
                batch.n_tokens = 0;
                return decode_rc;
            }
        }
        else {
            batch.n_tokens = 0;
            return decode_rc;
        }
    }

    // Update your application context states
    e->pos = pos;
    e->kv_valid = true;
    e->kv_len = pos;

    return 0;
}



// ------------------------------------------------------------
// Feed user message (templated) incrementally - STABLE NATIVE BATCH
// ------------------------------------------------------------

int engine_feed_user(engine_t* e, const char* user_text_raw) {
    if (!e || !e->model || !e->ctx || !user_text_raw) return -1;

    const struct llama_vocab* vocab = llama_model_get_vocab(e->model);

    // 1. Calculate length and dynamically allocate the wrapped buffer string
    size_t raw_len = strlen(user_text_raw);
    size_t wrapped_size = raw_len + 4096;
    char* wrapped = (char*)malloc(wrapped_size);
    if (!wrapped) return -1;

    engine_wrap_llama3_user(wrapped, wrapped_size, user_text_raw);

    // 2. Perform a dry-run token count request
    int n_tokens = llama_tokenize(vocab, wrapped, (int32_t)strlen(wrapped), NULL, 0, false, true);
    if (n_tokens < 0) n_tokens = -n_tokens;
    if (n_tokens <= 0 || n_tokens > 2048) { // Security bounds checking safety check
        free(wrapped);
        return -1;
    }

    // 3. Dynamically allocate a clean array for token identifiers
    llama_token* tokens = (llama_token*)malloc(sizeof(llama_token) * n_tokens);
    if (!tokens) {
        free(wrapped);
        return -1;
    }

    int actual_tokens = llama_tokenize(vocab, wrapped, (int32_t)strlen(wrapped), tokens, n_tokens, false, true);
    free(wrapped);
    if (actual_tokens <= 0) {
        free(tokens);
        return -1;
    }

    // =========================================================================
    // FIX: STACK ALLOCATE THE ARRAYS TO ELIMINATE LLAMA_BATCH_FREE() COMPLETELY
    // =========================================================================
    struct llama_batch batch = (struct llama_batch){ 0 };

    // Dynamically size these stack arrays using Visual Studio VLA or standard local buffers
    // 2048 provides an immense safety ceiling for common incoming user turns
#define USER_FEED_MAX_TOKENS 2048
    if (actual_tokens > USER_FEED_MAX_TOKENS) {
        free(tokens);
        return -1;
    }

    llama_pos      pos_arr[USER_FEED_MAX_TOKENS];
    int32_t        n_seq_arr[USER_FEED_MAX_TOKENS];
    llama_seq_id   seq_id_arr[USER_FEED_MAX_TOKENS];
    llama_seq_id* seq_ptr_arr[USER_FEED_MAX_TOKENS];
    int8_t         logits_arr[USER_FEED_MAX_TOKENS];

    // Assign your framework's exact pointer properties safely
    batch.token = tokens;
    batch.pos = pos_arr;
    batch.n_seq_id = n_seq_arr;
    batch.seq_id = seq_ptr_arr;
    batch.logits = logits_arr;
    batch.n_tokens = 0;

    llama_pos current_pos = e->pos;

    // 5. Explicit assignment mapping
    for (int i = 0; i < actual_tokens; ++i) {
        pos_arr[i] = current_pos;
        n_seq_arr[i] = 1;
        seq_id_arr[i] = e->seq_id;
        seq_ptr_arr[i] = &seq_id_arr[i];
        logits_arr[i] = (i == actual_tokens - 1) ? 1 : 0; // Logits requested for sampler only

        batch.n_tokens++;
        current_pos++;
    }

    // 6. Execute inference decoding safely
    int decode_rc = llama_decode(e->ctx, batch);

    // 7. Dynamic memory cleanup 
    free(tokens);

    // REMOVED: llama_batch_free(batch) is gone. 
    // The memory drops out of local stack scopes automatically without ntdll.dll tracking problems!

    //if (decode_rc != 0) {
    //    return decode_rc; // Let the main generator handle the context shift if it overflows here
    //}
    int32_t n_past = batch.pos[0];
    if (decode_rc != 0) {
        llama_memory_t mem = llama_get_memory(e->ctx);
        if (mem) {
            const int32_t system_prompt_tokens = 512;
            const int32_t discard_chunk = 512;

            const int32_t remove_start = system_prompt_tokens;
            const int32_t remove_end = system_prompt_tokens + discard_chunk;

            // 1. Remove the targeted history block            
            llama_memory_seq_rm(mem, e->seq_id, remove_start, remove_end);

            // 2. Slide subsequent context slots back     
            llama_memory_seq_add(mem, e->seq_id, remove_end, -1, -discard_chunk);

            // 3. RE-ALIGNMENT FIX:
            // Read where the KV cache physically ends after the shift.
            // For sequence 0, the next valid token position MUST be exactly 512 (X + 1)
            e->pos -= discard_chunk;

            // Explicitly write the corrected positions directly into the batch arrays
            for (int32_t i = 0; i < batch.n_tokens; ++i) {
                batch.pos[i] = e->pos + i;
            }

            // 4. Retry the decoding operations with clean positions
            decode_rc = llama_decode(e->ctx, batch);
            if (decode_rc != 0) {
                batch.n_tokens = 0;
                return decode_rc;
            }
        }
        else {
            batch.n_tokens = 0;
            return decode_rc;
        }
    }


    // 8. Commit tracker values on success states
    e->pos = current_pos;
    e->kv_valid = true;
    e->kv_len = current_pos;

    return 0;
}



// ------------------------------------------------------------
// Generate assistant reply incrementally - STABLE NATIVE BATCH
// ------------------------------------------------------------

int engine_generate_reply(
    engine_t* e,
    char* out,
    size_t out_size,
    int max_tokens)
{
    if (!e || !e->model || !e->ctx || !out || out_size == 0) return -1;

    const struct llama_vocab* vocab = llama_model_get_vocab(e->model);
    size_t out_len = 0;
    int n_gen = 0;

    out[0] = '\0';

    llama_token eos_tok = llama_token_eos(vocab);
    llama_token eot_tok = llama_token_eot(vocab);

    while (n_gen < max_tokens && out_len + 8 < out_size) {
        llama_token tok = engine_sample_next(e);

        if (tok == eos_tok || tok == eot_tok || llama_token_is_control(vocab, tok) || llama_token_is_eog(vocab, tok)) {
            break;
        }

        char buf[128] = { 0 };
        int n = llama_token_to_piece(vocab, tok, buf, sizeof(buf), 0, false);
        if (n <= 0) {
            break;
        }

        if (out_len + (size_t)n >= out_size) {
            break;
        }

        memcpy(out + out_len, buf, n);
        out_len += n;
        out[out_len] = '\0';

        if (strstr(out, "<|eot_id|") != NULL) {
            char* p = strstr(out, "<|eot_id|");
            *p = '\0';
            out_len = strlen(out);
            break;
        }
        if (strstr(out, "<|start_header_id") != NULL) {
            char* p = strstr(out, "<|start_header_id");
            *p = '\0';
            out_len = strlen(out);
            break;
        }

        // 3. Initialize a MANAGED Single-Token Batch structure natively
        struct llama_batch batch = llama_batch_init(1, 0, 1);

        // 4. Native internal parameter mapping loop for a single token
        int32_t token_index = batch.n_tokens;
        batch.token[token_index] = tok;
        batch.pos[token_index] = e->pos;
        batch.n_seq_id[token_index] = 1;
        batch.seq_id[token_index][0] = e->seq_id;
        batch.logits[token_index] = 1; // Always request logits for consecutive generations
        batch.n_tokens++;

        // 5. Decode token tracking parameters safely
        int decode_rc = 0;
        decode_rc = llama_decode(e->ctx, batch);
        if (decode_rc != 0) {
            llama_memory_t mem = llama_get_memory(e->ctx);
            if (mem) {
                // 1. Fetch the exact active context size allocated for this session
                int32_t active_ctx_size = llama_n_ctx(e->ctx);

                // 2. Safely lock your system prompt reservation limits
                const int32_t system_prompt_tokens = 512;

                // 3. ADAPTIVE SHIFTING MATH:
                // Clear out roughly 25% of the total window size, with a fallback ceiling
                int32_t discard_chunk = active_ctx_size / 4;
                if (discard_chunk < 512)  discard_chunk = 512;
                if (discard_chunk > 2048) discard_chunk = 2048; // Don't wipe too much chat memory at once

                const int32_t remove_start = system_prompt_tokens;
                const int32_t remove_end = system_prompt_tokens + discard_chunk;

                // Verify that we aren't trying to remove more than the total context window size
                if (remove_end >= active_ctx_size) {
                    // Panic check: model window is too small to even hold these constraints
                    llama_batch_free(batch);
                    return decode_rc;
                }

                // 4. Remove and slide context memory safely
                llama_memory_seq_rm(mem, e->seq_id, remove_start, remove_end);
                llama_memory_seq_add(mem, e->seq_id, remove_end, -1, -discard_chunk);

                // 5. Re-align your structural engine history tracking positions
                e->pos -= discard_chunk;

                // Restore the batch token counter back to 1 explicitly
                batch.n_tokens = 1;

                // Explicitly assign to index [0] of the batch array pointers.
                batch.token[0] = tok;
                batch.pos[0] = e->pos;
                batch.n_seq_id[0] = 1;
                batch.seq_id[0][0] = e->seq_id;
                batch.logits[0] = 1;

                // Retry decoding the token onto the cleared cache timelines
                decode_rc = llama_decode(e->ctx, batch);
                if (decode_rc != 0) {
                    llama_batch_free(batch);
                    return decode_rc;
                }
            }
            else {
                llama_batch_free(batch);
                return decode_rc;
            }
        }

        // 6. Free managed batch tracking objects
        llama_batch_free(batch);

        if (decode_rc != 0) {
            break;
        }

        e->pos++;
        e->kv_len = e->pos;
        n_gen++;
    }

    return (int)out_len;
}


// ------------------------------------------------------------
// HTML chat entry point - Complete Separated Plugin & UI Merge Pipeline
// ------------------------------------------------------------

int engine_chat_html(
    engine_t* e,
    const char* user_input,
    char* out,
    size_t out_size)
{
    if (!e || !user_input || !out || out_size == 0)
        return -1;

    out[0] = '\0';

    char* Plugin_result = NULL;
    char* temp_str = NULL;
    const char* extracted_query = user_input; // Default if no plugin matches
    bool plugin_executed = false;
    char* temp = NULL;

    // 1. Check for "ddg " plugin
    if (strncmp(user_input, "ddg ", 4) == 0) {
        extracted_query = user_input + 4;
        while (*extracted_query == ' ' || *extracted_query == '\t') extracted_query++; // strip leading spaces

        plugin_fn fn = plugin_lookup("ddg");
        if (fn) {
            // Pass the entire clean query as a single argument instead of splitting words!
            char* argvv[1];
            argvv[0] = (char*)extracted_query;
            Plugin_result = fn(1, argvv);
            plugin_executed = true;
        }
    }
  
    else if (strncmp(user_input, "websearch ", 10) == 0) {
        
        extracted_query = user_input + 10;
        temp = _strdup(extracted_query);
        char* token = strtok(temp, " ");
        char* argvv[32];
        int argcv = 0;

        while (token && argcv < 32) {
            argvv[argcv++] = token;
            token = strtok(NULL, " ");
        }

        plugin_fn fn = plugin_lookup("websearch");
        if (fn)
        {
            Plugin_result = fn(argcv, argvv);
            if (Plugin_result) {
                // printf("%s\n", Plugin_result);
                plugin_executed = true;
            }
                
        } 
               
    }
    // 3. Check for "summarize_file " plugin
    else if (strncmp(user_input, "summarize_file ", 15) == 0) {
        const char* raw = user_input + 15;
        while (*raw == ' ' || *raw == '\t') raw++;

        char clean_path[4096];
        engine_json_extract_string((char*)raw - 1, clean_path, sizeof(clean_path));

        char* argvv[1];
        argvv[0] = clean_path;

        Plugin_result = plugin_summarize_file_html(1, argvv);
        extracted_query = "Please synthesize and summarize this file contents.";
        plugin_executed = true;
    }

    // ------------------------------------------------------------
    // Dynamic Ingestion Layer (Background Data Context Feed)
    // ------------------------------------------------------------

    // If a plugin executed successfully, inject data into model KV cache first
    if (plugin_executed && Plugin_result) {
        // Wrapped in system header formatting so LLM treats it as an objective environment fact
        size_t system_block_size = strlen(Plugin_result) + 1024;
        char* system_payload = (char*)malloc(system_block_size);

        if (system_payload) {
            snprintf(system_payload, system_block_size,
                "<|start_header_id|>system<|end_header_id|>\n"
                "Search results background context:\n%s<|eot_id|>\n",
                Plugin_result
            );

            // Feed the background system context block into the model
            //engine_feed_user(e, system_payload);
            free(system_payload);
        }

        // Note: Do NOT free Plugin_result here yet! We keep it to print to the UI.
        if (temp_str) {
            free(temp_str);
            temp_str = NULL;
        }
    }
   
    // ------------------------------------------------------------
    // Feed Clean User Turn (No text mixed into the question)
    // ------------------------------------------------------------

    // This calls engine_feed_user with ONLY your isolated question (e.g. "how fast is a f-16?")
    int feed_rc = engine_feed_user(e, extracted_query);

    if (feed_rc != 0) {
        // 1. Clean up local plugin memory to prevent leaks
        if (plugin_executed && Plugin_result) {
            free(Plugin_result);
        }

        // 2. Flush engine tokens and clear the KV cache
        engine_reset(e);
        return -1;
    }

    // ------------------------------------------------------------
    // Inference Response & UI Compilation Layer
    // ------------------------------------------------------------


    // FIXED: Allocated a complete heap block instead of a broken single-byte variable
    char* model_reply = (char*)malloc(out_size);
    if (!model_reply) {
        if (plugin_executed && Plugin_result) free(Plugin_result);
        return -1;
    }
    model_reply[0] = '\0';

    int rc = engine_generate_reply(e, model_reply, out_size, 256);
    if (rc < 0) {
        free(model_reply);
        if (plugin_executed && Plugin_result) free(Plugin_result);
        return rc;
    }

    // Format output based on whether a plugin was run
    if (plugin_executed && Plugin_result) {
        snprintf(out, out_size,
            "=== Search Results Applied ===\n%s\n"
            "==============================\n\n"
            "AI Assistant:\n%s",
            Plugin_result, model_reply
        );

        free(Plugin_result);
    }
    else {
        strncpy(out, model_reply, out_size - 1);
        out[out_size - 1] = '\0';
    }

    free(model_reply);

    return (int)strlen(out);
}
 


// ------------------------------------------------------------
// Reset conversation
// ------------------------------------------------------------

void engine_reset(engine_t* e) {
    if (!e || !e->ctx) return;

    struct llama_memory* mem = llama_get_memory(e->ctx);
    if (mem) {
        llama_memory_seq_rm(mem, e->seq_id, 0, INT32_MAX);
    }

    e->pos = 0;
    e->kv_valid = false;
    e->kv_len = 0;

    e->html_n_turns = 0;
    e->n_turns = 0;

    engine_feed_system_prompt(e, "You are a helpful assistant.");
}

// ------------------------------------------------------------
// Open / close
// ------------------------------------------------------------

engine_t* engine_open(const char* model_path) {
    engine_t* e = calloc(1, sizeof(engine_t));
    if (!e) return NULL;

    engine_init_runtime(e);

    struct llama_model_params mparams = llama_model_default_params();
    e->model = llama_load_model_from_file(model_path, mparams);
    if (!e->model) {
        free(e);
        return NULL;
    }

    struct llama_context_params cparams = llama_context_default_params();

    // DYNAMIC METADATA EXTRACTION
    // Read the native maximum training context limit embedded directly inside the GGUF file
    int32_t model_train_ctx = llama_model_n_ctx_train(e->model);

    if (model_train_ctx > 0) {
        // Successfully pulled from GGUF metadata. Assign it directly!
        cparams.n_ctx = model_train_ctx;
        printf("[SUCCESS] GGUF metadata found! Context length auto-populated to: %d tokens.\n", cparams.n_ctx);
    }
    else {
        // Fallback guard condition in case the metadata key is missing
        cparams.n_ctx = 4096;
        printf("[WARN] GGUF metadata context length unavailable. Falling back to default: 4096 tokens.\n");
    }

    // Keep sequence scaling optimized for your singular tracking stream
    cparams.n_seq_max = 1;

    // Allocate memory matching the precise dynamic token ceiling
    e->ctx = llama_new_context_with_model(e->model, cparams);
    if (!e->ctx) {
        llama_free_model(e->model);
        free(e);
        return NULL;
    }

    if (engine_feed_system_prompt(e, "You are a helpful assistant.") != 0) {
        llama_free(e->ctx);
        llama_free_model(e->model);
        free(e);
        return NULL;
    }

    return e;
}


void engine_close(engine_t* e) {
    if (!e) return;

    if (e->ctx) {
        llama_free(e->ctx);
        e->ctx = NULL;
    }

    if (e->model) {
        llama_free_model(e->model);
        e->model = NULL;
    }

    free(e);
}
