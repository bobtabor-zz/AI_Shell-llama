typedef struct engine {
    struct llama_model* model;
    struct llama_context* ctx;

    int32_t seq_id;   // single stable sequence
    llama_pos pos;    // current position in this sequence

    float temp;
    int top_k;
    float top_p;
} engine_t;


engine_t* engine_create(const char* model_path) {
    engine_t* e = calloc(1, sizeof(engine_t));
    if (!e) return NULL;

    struct llama_model_params mparams = llama_model_default_params();
    e->model = llama_load_model_from_file(model_path, mparams);
    if (!e->model) { free(e); return NULL; }

    struct llama_context_params cparams = llama_context_default_params();
    // IMPORTANT: allow at least 1 sequence
    cparams.n_seq_max = 1;
    e->ctx = llama_new_context_with_model(e->model, cparams);
    if (!e->ctx) { llama_free_model(e->model); free(e); return NULL; }

    e->seq_id = 0;
    e->pos = 0;

    e->temp = 0.7f;
    e->top_k = 20;
    e->top_p = 0.9f;

    return e;
}

static int tokenize_text(
    const struct llama_vocab* vocab,
    const char* text,
    llama_token* out,
    int max_tokens)
{
    return llama_tokenize(
        vocab,
        text,
        (int32_t)strlen(text),
        out,
        max_tokens,
        true,   // add BOS if needed
        false   // no special tokens
    );
}

static llama_token sample_next_token(engine_t* e) {
    const struct llama_vocab* vocab = llama_model_get_vocab(e->model);
    const int n_vocab = llama_vocab_n_tokens(vocab);

    // get logits for last token
    const float* logits = llama_get_logits(e->ctx);

    // simple greedy (replace with your temp/top_k/top_p sampler)
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


int engine_feed_system_prompt(engine_t* e, const char* system_str) {
    const struct llama_vocab* vocab = llama_model_get_vocab(e->model);

    llama_token tokens[1024];
    int n_tokens = tokenize_text(vocab, system_str, tokens, 1024);
    if (n_tokens <= 0) return -1;

    struct llama_batch batch = { 0 };
    batch.n_tokens = 0;

    llama_pos pos = e->pos;

    // stack arrays for batch fields
    llama_pos      pos_arr[1024];
    int32_t        n_seq_arr[1024];
    llama_seq_id   seq_id_arr[1024];
    llama_seq_id* seq_ptr_arr[1024];
    int8_t         logits_arr[1024];

    batch.token = tokens;
    batch.pos = pos_arr;
    batch.n_seq_id = n_seq_arr;
    batch.seq_id = seq_ptr_arr;
    batch.logits = logits_arr;

    for (int i = 0; i < n_tokens; ++i) {
        pos_arr[i] = pos;
        n_seq_arr[i] = 1;
        seq_id_arr[i] = e->seq_id;
        seq_ptr_arr[i] = &seq_id_arr[i];
        logits_arr[i] = (i == n_tokens - 1) ? 1 : 0; // logits on last

        batch.n_tokens++;
        pos++;
    }

    if (llama_decode(e->ctx, batch) != 0) {
        return -1;
    }

    e->pos = pos;
    return 0;
}


int engine_feed_user(engine_t* e, const char* user_text) {
    const struct llama_vocab* vocab = llama_model_get_vocab(e->model);

    llama_token tokens[2048];
    int n_tokens = tokenize_text(vocab, user_text, tokens, 2048);
    if (n_tokens <= 0) return -1;

    struct llama_batch batch = { 0 };
    batch.n_tokens = 0;

    llama_pos      pos_arr[2048];
    int32_t        n_seq_arr[2048];
    llama_seq_id   seq_id_arr[2048];
    llama_seq_id* seq_ptr_arr[2048];
    int8_t         logits_arr[2048];

    batch.token = tokens;
    batch.pos = pos_arr;
    batch.n_seq_id = n_seq_arr;
    batch.seq_id = seq_ptr_arr;
    batch.logits = logits_arr;

    llama_pos pos = e->pos;

    for (int i = 0; i < n_tokens; ++i) {
        pos_arr[i] = pos;
        n_seq_arr[i] = 1;
        seq_id_arr[i] = e->seq_id;
        seq_ptr_arr[i] = &seq_id_arr[i];
        logits_arr[i] = (i == n_tokens - 1) ? 1 : 0;

        batch.n_tokens++;
        pos++;
    }

    if (llama_decode(e->ctx, batch) != 0) {
        return -1;
    }

    e->pos = pos;
    return 0;
}


int engine_generate_reply(
    engine_t* e,
    char* out,
    size_t out_size,
    int max_tokens)
{
    const struct llama_vocab* vocab = llama_model_get_vocab(e->model);
    size_t out_len = 0;
    int n_gen = 0;

    while (n_gen < max_tokens && out_len + 8 < out_size) {
        llama_token tok = sample_next_token(e);

        if (tok == llama_token_eos(e->model)) break;

        char buf[128] = { 0 };
        int n = llama_token_to_piece(vocab, tok, buf, sizeof(buf), 0, false);
        if (n <= 0) break;

        if (out_len + (size_t)n >= out_size) break;
        memcpy(out + out_len, buf, n);
        out_len += n;
        out[out_len] = '\0';

        // feed this assistant token back into KV
        struct llama_batch batch = { 0 };

        llama_pos      pos_val = e->pos;
        int32_t        n_seq_val = 1;
        llama_seq_id   seq_id_val = e->seq_id;
        llama_seq_id* seq_ptr_val = &seq_id_val;
        int8_t         logits_val = 1;

        batch.n_tokens = 1;
        batch.token = &tok;
        batch.pos = &pos_val;
        batch.n_seq_id = &n_seq_val;
        batch.seq_id = &seq_ptr_val;
        batch.logits = &logits_val;

        if (llama_decode(e->ctx, batch) != 0) {
            break;
        }

        e->pos++;
        n_gen++;
    }

    return (int)out_len;
}


int engine_chat_turn(
    engine_t* e,
    const char* user_input,
    char* out,
    size_t out_size)
{
    // 1) feed user message
    if (engine_feed_user(e, user_input) != 0)
        return -1;

    // 2) generate assistant reply
    int rc = engine_generate_reply(e, out, out_size, 128);
    return rc;
}


void engine_reset(engine_t* e) {
    struct llama_memory* mem = llama_get_memory(e->ctx);
    if (mem) {
        llama_memory_seq_rm(mem, e->seq_id, 0, INT32_MAX);
    }
    e->pos = 0;
}
