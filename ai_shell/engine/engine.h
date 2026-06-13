#pragma once
#include <stddef.h>
#include "llama.h"
#include "../vmm/vmm.h"

#ifdef __cplusplus
extern "C" {
#endif
#define MAX_TURNS 32

    typedef struct {
        char role[8];   // "User" / "AI"
        char text[4096];
    } engine_turn_t;

    typedef struct {
        char role[8];   // "User" / "AI"
        char text[4096];
    } html_turn_t;

    typedef struct engine {
        struct llama_model* model;
        struct llama_context* ctx;
        //int64_t pos;
        //char conversation[16384];

        html_turn_t html_turns[128];
        int html_n_turns;

        int html_seq_id;   // always 0 for HTML chat

        engine_turn_t turns[MAX_TURNS];
        int n_turns;

        int32_t seq_id;   // single stable sequence
        llama_pos pos;    // current position in this sequence
        // --- Sampling parameters (needed for incremental generation) ---
        float temp;
        int   top_k;
        float top_p;

        vmm_model_t* vmm_model;

        // ⭐ KV subsystem
        
        bool    kv_valid;      // is there a meaningful sequence in KV?
        int64_t kv_len;        // how many tokens are currently in KV?

    } engine_t;

    void engine_recreate_context(engine_t* e);

    // Inside engine.h
    void engine_reset(engine_t* e);

   int engine_chat(engine_t* e,
        const char* user_input,
        char* out,
        size_t out_size);

   int engine_chat_html(engine_t* e,
       const char* user_input,
       char* out,
       size_t out_size);

    /*void engine_kv_cache_clear(engine_t* e);*/

    void cmd_open(int argc, char** argv);
    void cmd_close(int argc, char** argv);
    void cmd_infer(int argc, char** argv);

    engine_t* engine_open(const char* path);
    void      engine_close(engine_t* e);

    int engine_generate(
        engine_t* e,
        const char* prompt,
        char* out,
        size_t out_size,
        int max_tokens,
        float temp,
        int top_k,
        float top_p,
        bool stream
    );


#ifdef __cplusplus
}
#endif
