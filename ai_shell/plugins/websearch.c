// websearch.c — unified Wikipedia + DuckDuckGo web search plugin (structured, image-aware)
#include <windows.h>
#include <winhttp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../include/plugin.h"
#include "cJSON.h"

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
// MAIN ENTRY
// ------------------------------------------------------------
char* plugin_websearch(int argc, char** argv) {
    if (argc < 1)
        return _strdup("{\"error\":\"no query\"}");

    char query[256] = { 0 };
    for (int i = 0; i < argc; i++) {
        strncat(query, argv[i], sizeof(query) - strlen(query) - 2);
        if (i + 1 < argc)
            strncat(query, " ", sizeof(query) - strlen(query) - 1);
    }

    // Cache
    char* cached = cache_get(query);
    if (cached) return cached;

    // Wikipedia first
    char* wiki = wikipedia_structured(query);
    if (wiki) {
        cache_put(query, wiki);
        return wiki;
    }

    // DuckDuckGo fallback
    char* ddg = ddg_structured(argc, argv);
    if (ddg) {
        cache_put(query, ddg);
        return ddg;
    }

    return _strdup("{\"error\":\"websearch_failed\"}");
}