// websearch.c — unified Wikipedia + DuckDuckGo web search plugin (structured, image-aware)
#include <windows.h>
#include <winhttp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../include/plugin.h"
#include "cJSON.h"
#include <stdbool.h>

#pragma comment(lib, "winhttp.lib")

// Forward declaration of DDG plugin (from ddg.c)
extern char* plugin_ddg(int argc, char** argv);

// ------------------------------------------------------------
// Simple HTTP GET (UTF-8)
// ------------------------------------------------------------
static char* http_get_utf8(const wchar_t* host, const wchar_t* path) {
    HINTERNET s = WinHttpOpen(
        L"Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/124.0",
        WINHTTP_ACCESS_TYPE_DEFAULT_PROXY,
        WINHTTP_NO_PROXY_NAME,
        WINHTTP_NO_PROXY_BYPASS,
        0);
    if (!s) return NULL;

    HINTERNET c = WinHttpConnect(s, host, INTERNET_DEFAULT_HTTPS_PORT, 0);
    if (!c) {
        WinHttpCloseHandle(s);
        return NULL;
    }

    HINTERNET r = WinHttpOpenRequest(
        c, L"GET", path, NULL, WINHTTP_NO_REFERER,
        WINHTTP_DEFAULT_ACCEPT_TYPES, WINHTTP_FLAG_SECURE);
    if (!r) {
        WinHttpCloseHandle(c);
        WinHttpCloseHandle(s);
        return NULL;
    }

    // ⭐ FULL PRODUCTION HEADER BLOCK
    LPCWSTR headers =
        L"User-Agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        L"AppleWebKit/537.36 (KHTML, like Gecko) "
        L"Chrome/124.0 Safari/537.36\r\n"
        L"Accept: application/json,text/html;q=0.9,*/*;q=0.8\r\n"
        L"Accept-Language: en-US,en;q=0.9\r\n"
        L"Cache-Control: no-cache\r\n"
        L"Pragma: no-cache\r\n"
        L"Connection: keep-alive\r\n";

    WinHttpAddRequestHeaders(r, headers, -1, WINHTTP_ADDREQ_FLAG_ADD);

    BOOL ok = WinHttpSendRequest(
        r,
        WINHTTP_NO_ADDITIONAL_HEADERS,
        0,
        WINHTTP_NO_REQUEST_DATA,
        0,
        0,
        0);
    if (!ok) {
        WinHttpCloseHandle(r);
        WinHttpCloseHandle(c);
        WinHttpCloseHandle(s);
        return NULL;
    }

    ok = WinHttpReceiveResponse(r, NULL);
    if (!ok) {
        WinHttpCloseHandle(r);
        WinHttpCloseHandle(c);
        WinHttpCloseHandle(s);
        return NULL;
    }

    DWORD size = 0, downloaded = 0;
    char* buf = malloc(1);
    size_t total = 0;
    buf[0] = 0;

    do {
        if (!WinHttpQueryDataAvailable(r, &size)) break;
        if (size == 0) break;

        char* chunk = malloc(size + 1);
        if (!WinHttpReadData(r, chunk, size, &downloaded)) {
            free(chunk);
            break;
        }

        chunk[downloaded] = 0;

        buf = realloc(buf, total + downloaded + 1);
        memcpy(buf + total, chunk, downloaded);
        total += downloaded;
        buf[total] = 0;

        free(chunk);
    } while (size > 0);

    WinHttpCloseHandle(r);
    WinHttpCloseHandle(c);
    WinHttpCloseHandle(s);

    return buf;
}


// ------------------------------------------------------------
// Simple query cache
// ------------------------------------------------------------
#define CACHE_SIZE 16
typedef struct {
    char query[256];
    char* json;
} CacheEntry;

static CacheEntry g_cache[CACHE_SIZE];

static char* cache_get(const char* q) {
    for (int i = 0; i < CACHE_SIZE; i++) {
        if (g_cache[i].json && _stricmp(g_cache[i].query, q) == 0)
            return _strdup(g_cache[i].json);
    }
    return NULL;
}

static void cache_put(const char* q, const char* json) {
    static int idx = 0;
    int i = idx++ % CACHE_SIZE;

    free(g_cache[i].json);
    strncpy(g_cache[i].query, q, 255);
    g_cache[i].query[255] = 0;
    g_cache[i].json = _strdup(json);
}

// ------------------------------------------------------------
// Wikipedia search → summary (raw JSON)
// ------------------------------------------------------------
static char* wikipedia_summary_raw(const char* query) {
    wchar_t encoded[2048];
    int ei = 0;

    for (int i = 0; query[i] && ei < 2040; i++) {
        if (query[i] == ' ') {
            encoded[ei++] = L'%'; encoded[ei++] = L'2'; encoded[ei++] = L'0';
        }
        else {
            encoded[ei++] = (wchar_t)query[i];
        }
    }
    encoded[ei] = 0;

    wchar_t search_path[1024];
    swprintf(search_path, 1024,
        L"/w/api.php?action=query&list=search&srsearch=%ls&format=json&utf8=1",
        encoded);

    char* search_json = http_get_utf8(L"en.wikipedia.org", search_path);
    if (!search_json) return NULL;

    cJSON* root = cJSON_Parse(search_json);
    free(search_json);
    if (!root) return NULL;

    cJSON* qobj = cJSON_GetObjectItem(root, "query");
    cJSON* arr = qobj ? cJSON_GetObjectItem(qobj, "search") : NULL;
    cJSON* first = (arr && cJSON_IsArray(arr)) ? cJSON_GetArrayItem(arr, 0) : NULL;
    cJSON* title = first ? cJSON_GetObjectItem(first, "title") : NULL;

    if (!cJSON_IsString(title)) {
        cJSON_Delete(root);
        return NULL;
    }

    wchar_t wtitle[1024];
    int wi = 0;
    for (; title->valuestring[wi] && wi < 1023; wi++)
        wtitle[wi] = (wchar_t)title->valuestring[wi];
    wtitle[wi] = 0;

    cJSON_Delete(root);

    wchar_t summary_path[1024];
    swprintf(summary_path, 1024,
        L"/api/rest_v1/page/summary/%ls", wtitle);

    return http_get_utf8(L"en.wikipedia.org", summary_path);
}

// ------------------------------------------------------------
// Wikipedia structured output
// ------------------------------------------------------------
static char* wikipedia_structured(const char* query) {
    char* raw = wikipedia_summary_raw(query);
    if (!raw) return NULL;

    cJSON* root = cJSON_Parse(raw);
    free(raw);
    if (!root) return NULL;

    const char* title = cJSON_GetStringValue(cJSON_GetObjectItem(root, "title"));
    const char* desc = cJSON_GetStringValue(cJSON_GetObjectItem(root, "description"));
    const char* extract = cJSON_GetStringValue(cJSON_GetObjectItem(root, "extract"));

    cJSON* urls = cJSON_GetObjectItem(root, "content_urls");
    cJSON* desk = urls ? cJSON_GetObjectItem(urls, "desktop") : NULL;
    const char* page_url = desk ? cJSON_GetStringValue(cJSON_GetObjectItem(desk, "page")) : "";

    cJSON* thumb = cJSON_GetObjectItem(root, "thumbnail");
    const char* thumb_src = thumb ? cJSON_GetStringValue(cJSON_GetObjectItem(thumb, "source")) : NULL;

    cJSON* orig = cJSON_GetObjectItem(root, "originalimage");
    const char* orig_src = orig ? cJSON_GetStringValue(cJSON_GetObjectItem(orig, "source")) : NULL;

    // Fallback logic
    const char* final_thumb = NULL;
    if (thumb_src && thumb_src[0])
        final_thumb = thumb_src;
    else if (orig_src && orig_src[0])
        final_thumb = orig_src;

    // Build structured JSON
    cJSON* out = cJSON_CreateObject();
    cJSON_AddStringToObject(out, "source", "wikipedia");
    cJSON_AddStringToObject(out, "title", title ? title : "");
    cJSON_AddStringToObject(out, "description", desc ? desc : "");
    cJSON_AddStringToObject(out, "summary", extract ? extract : "");
    cJSON_AddStringToObject(out, "page_url", page_url ? page_url : "");

    if (final_thumb)
        cJSON_AddStringToObject(out, "thumbnail", final_thumb);
    if (orig_src)
        cJSON_AddStringToObject(out, "original_image", orig_src);

    cJSON* images = cJSON_CreateArray();
    if (orig_src)
        cJSON_AddItemToArray(images, cJSON_CreateString(orig_src));
    else if (final_thumb)
        cJSON_AddItemToArray(images, cJSON_CreateString(final_thumb));
    cJSON_AddItemToObject(out, "images", images);

    char* out_str = cJSON_PrintUnformatted(out);

    // ALSO output plain text URLs for the UI to detect
    // Append plain URL at the end so UI can render it
    if (orig_src) {
        size_t len = strlen(out_str) + strlen(orig_src) + 4;
        char* merged = malloc(len);
        snprintf(merged, len, "%s\n%s", out_str, orig_src);
        free(out_str);
        out_str = merged;
    }
    else if (final_thumb) {
        size_t len = strlen(out_str) + strlen(final_thumb) + 4;
        char* merged = malloc(len);
        snprintf(merged, len, "%s\n%s", out_str, final_thumb);
        free(out_str);
        out_str = merged;
    }

    cJSON_Delete(out);
    cJSON_Delete(root);

    return out_str;
}

// ------------------------------------------------------------
// DuckDuckGo fallback (structured)
// ------------------------------------------------------------
static char* ddg_structured(int argc, char** argv) {
    char* raw = plugin_ddg(argc, argv);
    if (!raw) return NULL;

    cJSON* root = cJSON_Parse(raw);
    free(raw);

    if (!root) {
        cJSON* out = cJSON_CreateObject();
        cJSON_AddStringToObject(out, "source", "duckduckgo");
        cJSON_AddStringToObject(out, "title", "");
        cJSON_AddStringToObject(out, "description", "");
        cJSON_AddStringToObject(out, "summary", "");
        cJSON_AddStringToObject(out, "page_url", "");
        cJSON_AddItemToObject(out, "images", cJSON_CreateArray());
        char* s = cJSON_PrintUnformatted(out);
        cJSON_Delete(out);
        return s;
    }

    const char* heading = cJSON_GetStringValue(cJSON_GetObjectItem(root, "Heading"));
    const char* abstract = cJSON_GetStringValue(cJSON_GetObjectItem(root, "Abstract"));
    const char* image = cJSON_GetStringValue(cJSON_GetObjectItem(root, "Image"));

    cJSON* out = cJSON_CreateObject();
    cJSON_AddStringToObject(out, "source", "duckduckgo");
    cJSON_AddStringToObject(out, "title", heading ? heading : "");
    cJSON_AddStringToObject(out, "description", abstract ? abstract : "");
    cJSON_AddStringToObject(out, "summary", abstract ? abstract : "");
    cJSON_AddStringToObject(out, "page_url", "");

    if (image)
        cJSON_AddStringToObject(out, "thumbnail", image);

    cJSON* images = cJSON_CreateArray();
    if (image)
        cJSON_AddItemToArray(images, cJSON_CreateString(image));
    cJSON_AddItemToObject(out, "images", images);

    char* s = cJSON_PrintUnformatted(out);
    cJSON_Delete(out);
    cJSON_Delete(root);
    return s;
}

// ------------------------------------------------------------
// Better Image Search (Wikipedia + DuckDuckGo + Fallback)
// ------------------------------------------------------------
static char* websearch_images(const char* query) {
    char* wiki = wikipedia_structured(query);
    char* ddg = ddg_structured(0, NULL);   // you can improve this later

    cJSON* out = cJSON_CreateObject();
    cJSON_AddStringToObject(out, "source", "websearch");
    cJSON_AddStringToObject(out, "query", query);

    cJSON* images = cJSON_CreateArray();

    // Add Wikipedia images
    if (wiki) {
        cJSON* wiki_root = cJSON_Parse(wiki);
        if (wiki_root) {
            cJSON* wiki_images = cJSON_GetObjectItem(wiki_root, "images");
            if (wiki_images && cJSON_IsArray(wiki_images)) {
                cJSON* img;
                cJSON_ArrayForEach(img, wiki_images) {
                    if (cJSON_IsString(img))
                        cJSON_AddItemToArray(images, cJSON_CreateString(img->valuestring));
                }
            }
            cJSON_Delete(wiki_root);
        }
        free(wiki);
    }

    // Add DuckDuckGo image if available
    if (ddg) {
        cJSON* ddg_root = cJSON_Parse(ddg);
        if (ddg_root) {
            const char* img = cJSON_GetStringValue(cJSON_GetObjectItem(ddg_root, "Image"));
            if (img && img[0]) {
                cJSON_AddItemToArray(images, cJSON_CreateString(img));
            }
            cJSON_Delete(ddg_root);
        }
        free(ddg);
    }

    // TODO: Later add real image search (Unsplash, Pexels, Pixabay, etc.)

    cJSON_AddItemToObject(out, "images", images);

    // Also return a clean message for the LLM
    cJSON_AddStringToObject(out, "message",
        "Here are some real images I found for you.");

    char* result = cJSON_PrintUnformatted(out);
    cJSON_Delete(out);
    return result;
}

// ------------------------------------------------------------
// MAIN ENTRY
// ------------------------------------------------------------
//char* plugin_websearch(int argc, char** argv) {
//    if (argc < 1)
//        return _strdup("{\"error\":\"no query\"}");
//
//    char query[256] = { 0 };
//    for (int i = 0; i < argc; i++) {
//        strncat(query, argv[i], sizeof(query) - strlen(query) - 2);
//        if (i + 1 < argc)
//            strncat(query, " ", sizeof(query) - strlen(query) - 1);
//    }
//
//    // Cache
//    char* cached = cache_get(query);
//    if (cached) return cached;
//
//    // Wikipedia first
//    char* wiki = wikipedia_structured(query);
//    if (wiki) {
//        cache_put(query, wiki);
//        return wiki;
//    }
//
//    // DuckDuckGo fallback
//    char* ddg = ddg_structured(argc, argv);
//    if (ddg) {
//        cache_put(query, ddg);
//        return ddg;
//    }
//
//    return _strdup("{\"error\":\"websearch_failed\"}");
//}

// ------------------------------------------------------------
// MAIN ENTRY - Improved with better image support
// ------------------------------------------------------------
char* plugin_websearch(int argc, char** argv) {
    if (argc < 1)
        return _strdup("{\"error\":\"no query\"}");

    char query[512] = { 0 };
    for (int i = 0; i < argc; i++) {
        size_t current_len = strlen(query);
        size_t space_left = sizeof(query) - current_len - 1;
        if (space_left > 0) {
            strncat(query, argv[i], space_left);
        }

        current_len = strlen(query);
        space_left = sizeof(query) - current_len - 1;
        if (i + 1 < argc && space_left > 0) {
            strncat(query, " ", space_left);
        }
    }

    // Cache lookup checkpoint
    char* cached = cache_get(query);
    if (cached) return cached;

    // ==================== IMAGE REQUEST DETECTION ====================
    bool is_image_request = false;

    char* lower_query = _strdup(query);
    if (lower_query != NULL) {
        for (char* p = lower_query; *p; p++) *p = tolower((unsigned char)*p);

        const char* visual_tokens[] = {
            "image", "picture", "photo", "pic", "graphic",
            "illustration", "diagram", "drawing", "sketch",
            "screenshot", "wallpaper", "show me", "look at",
            "view", "display", "render", "search for", "draw",
            ".jpg", ".png", ".gif", ".jpeg"
        };
        size_t num_tokens = sizeof(visual_tokens) / sizeof(visual_tokens[0]);

        for (size_t i = 0; i < num_tokens; i++) {
            if (strstr(lower_query, visual_tokens[i]) != NULL) {
                is_image_request = true;
                break;
            }
        }
        free(lower_query);
    }

    // ==================== WIKIPEDIA ENTRY ====================
    char* wiki = wikipedia_structured(query);

    if (wiki) {
        if (is_image_request) {
            cJSON* root = cJSON_Parse(wiki);
            if (root) {
                cJSON* images = cJSON_GetObjectItem(root, "images");

                if (!images || !cJSON_IsArray(images) || cJSON_GetArraySize(images) == 0) {
                    images = cJSON_CreateArray();
                    cJSON_AddItemToObject(root, "images", images);
                }

                const char* orig = cJSON_GetStringValue(cJSON_GetObjectItem(root, "original_image"));
                if (orig) {
                    cJSON* img_array = cJSON_GetObjectItem(root, "images");
                    if (img_array && cJSON_IsArray(img_array)) {
                        cJSON_AddItemToArray(img_array, cJSON_CreateString(orig));
                    }
                }

                cJSON_AddBoolToObject(root, "image_mode", true);
                cJSON_AddStringToObject(root, "message", "Here are some images I found:");

                // FIX 1: Safely allocate a new buffer string from cJSON
                char* formatted_json = cJSON_PrintUnformatted(root);

                // Free the original wikipedia text segment pointer memory safely
                free(wiki);

                // Re-assign your unified tracking variable pointer cleanly
                wiki = formatted_json;

                cJSON_Delete(root);
            }
        }

        cache_put(query, wiki);
        return wiki;
    }

    // ==================== DUCKDUCKGO FALLBACK ====================
    char* ddg = ddg_structured(argc, argv);
    if (ddg) {
        if (is_image_request) {
            cJSON* root = cJSON_Parse(ddg);
            if (root) {
                cJSON_AddBoolToObject(root, "image_mode", true);
                cJSON_AddStringToObject(root, "message", "Here are some images:");

                // FIX 2: Prevent memory leaks inside the DuckDuckGo pipeline path
                char* formatted_json = cJSON_PrintUnformatted(root);
                free(ddg);
                ddg = formatted_json;

                cJSON_Delete(root);
            }
        }
        cache_put(query, ddg);
        return ddg;
    }

    // ==================== FIX 3: SAFE, NON-OVERFLOW ERROR GENERATION ====================
    size_t error_buf_size = strlen(query) + 128;
    char* error = (char*)malloc(error_buf_size);
    if (error) {
        snprintf(error, error_buf_size, "{\"error\":\"websearch_failed\",\"query\":\"%s\"}", query);
        cache_put(query, error);
        return error;
    }

    return _strdup("{\"error\":\"allocation_failed\"}");
}
