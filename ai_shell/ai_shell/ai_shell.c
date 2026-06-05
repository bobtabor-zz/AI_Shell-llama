#include "../engine/engine.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// -----------------------------------------------------------------------------
// Global engine instance
// -----------------------------------------------------------------------------

static engine_t * g_engine = NULL;

// -----------------------------------------------------------------------------
// Utility
// -----------------------------------------------------------------------------

static void trim(char * s) {
    size_t n = strlen(s);
    while (n > 0 && (s[n - 1] == '\n' || s[n - 1] == '\r')) {
        s[--n] = 0;
    }
}

// -----------------------------------------------------------------------------
// Commands
// -----------------------------------------------------------------------------

void cmd_open(int argc, char** argv) {
    if (argc < 3) {
        printf("ERR missing_args\n");
        return;
    }

    const char* slot = argv[1];
    const char* path = argv[2];   // <‑‑ THIS is the fix

    engine_t* e = engine_open(path);
    if (!e) {
        printf("ERR failed_to_open\n");
        return;
    }

    g_engine = e;   // or whatever your global is
    printf("OK MODEL %s\n", slot);
}


static void cmd_close(int argc, char ** argv) {
    (void) argc;
    (void) argv;

    if (!g_engine) {
        printf("ERR no_model\n");
        return;
    }

    engine_close(g_engine);
    g_engine = NULL;

    printf("OK\n");
}

static void cmd_infer(int argc, char ** argv) {
    if (argc < 2) {
        printf("ERR usage: INFER <prompt>\n");
        return;
    }

    if (!g_engine) {
        printf("ERR no_model\n");
        return;
    }

    // Reconstruct prompt from argv[1..]
    char prompt[4096] = { 0 };
    for (int i = 1; i < argc; i++) {
        strcat(prompt, argv[i]);
        if (i + 1 < argc) {
            strcat(prompt, " ");
        }
    }

    char out[4096];
    if (!engine_infer(g_engine, prompt, out, sizeof(out))) {
        printf("ERR infer_failed\n");
        return;
    }

    printf("OK %s\n", out);
}

static void cmd_ping(int argc, char ** argv) {
    (void) argc;
    (void) argv;
    printf("PONG\n");
}

static void cmd_stats(int argc, char ** argv) {
    (void) argc;
    (void) argv;

    printf("OK\n");
    printf("ENGINE %s\n", g_engine ? "LOADED" : "NONE");
    printf("END\n");
}

// -----------------------------------------------------------------------------
// Command dispatcher
// -----------------------------------------------------------------------------

static void dispatch(char * line) {
    trim(line);
    if (line[0] == 0) {
        return;
    }

    char * argv[32];
    int    argc = 0;

    char * tok = strtok(line, " ");
    while (tok && argc < 32) {
        argv[argc++] = tok;
        tok          = strtok(NULL, " ");
    }
    if (argc == 0) {
        return;
    }

    const char * cmd = argv[0];

    if (strcmp(cmd, "OPEN") == 0) {
        cmd_open(argc, argv);
    } else if (strcmp(cmd, "CLOSE") == 0) {
        cmd_close(argc, argv);
    } else if (strcmp(cmd, "INFER") == 0) {
        cmd_infer(argc, argv);
    } else if (strcmp(cmd, "PING") == 0) {
        cmd_ping(argc, argv);
    } else if (strcmp(cmd, "STATS") == 0) {
        cmd_stats(argc, argv);
    } else {
        printf("ERR unknown_command\n");
    }
}

// -----------------------------------------------------------------------------
// Main loop
// -----------------------------------------------------------------------------

int main(void) {
    char line[65536];

    printf("[main] Starting...\n");

   while (fgets(line, sizeof(line), stdin)) {
        dispatch(line);

        printf("ai> ");
        fflush(stdout);
    }

    if (g_engine) {
        engine_close(g_engine);
    }
       
    return 0;
}
