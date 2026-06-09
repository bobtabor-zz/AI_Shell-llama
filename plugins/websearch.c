// websearch.c — unified Wikipedia + DuckDuckGo web search plugin
#include <windows.h>
#include <winhttp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../include/plugin.h"

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

    LPCWSTR headers =
        L"User-Agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        L"AppleWebKit/537.36 (KHTML, like Gecko) "
        L"Chrome/124.0 Safari/537.36\r\n"
        L"Accept: text/html,application/json;q=0.9,*/*;q=0.8\r\n"
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
// Wikipedia summary API
// ------------------------------------------------------------
// ------------------------------------------------------------
// FINAL Wikipedia search → summary function (2026‑ready)
// ------------------------------------------------------------
static char* wikipedia_summary(const char* query) {
    wchar_t search_path[1024];
    wchar_t encoded[2048];

    // Encode query for URL (spaces → %20)
    int ei = 0;
    for (int i = 0; query[i] && ei < 2040; i++) {
        if (query[i] == ' ') {
            encoded[ei++] = L'%';
            encoded[ei++] = L'2';
            encoded[ei++] = L'0';
        }
        else {
            encoded[ei++] = (wchar_t)query[i];
        }
    }
    encoded[ei] = 0;

    // Build search API path
    swprintf(search_path, 1024,
        L"/w/api.php?action=query&list=search&srsearch=%ls&format=json&utf8=1",
        encoded);

    // Fetch search results
    char* search_json = http_get_utf8(L"en.wikipedia.org", search_path);
    if (!search_json)
        return NULL;

    // Find first title
    char* t = strstr(search_json, "\"title\":\"");
    if (!t) {
        free(search_json);
        return NULL;
    }
    t += 9;

    char* end = strchr(t, '"');
    if (!end) {
        free(search_json);
        return NULL;
    }

    size_t len = end - t;
    char* title = malloc(len + 1);
    memcpy(title, t, len);
    title[len] = 0;

    free(search_json);

    // Convert title to wide string
    wchar_t wtitle[1024];
    int wi = 0;
    for (; wi < len && wi < 1023; wi++)
        wtitle[wi] = (wchar_t)title[wi];
    wtitle[wi] = 0;

    free(title);

    // Build summary API path
    wchar_t summary_path[1024];
    swprintf(summary_path, 1024,
        L"/api/rest_v1/page/summary/%ls", wtitle);

    // Fetch summary JSON
    char* summary_json = http_get_utf8(L"en.wikipedia.org", summary_path);
    if (!summary_json)
        return NULL;

    // Validate summary
    if (!strstr(summary_json, "\"extract\"")) {
        free(summary_json);
        return NULL;
    }

    return summary_json;
}

// ------------------------------------------------------------
// MAIN PLUGIN ENTRY
// ------------------------------------------------------------
char* plugin_websearch(int argc, char** argv) {
    if (argc < 1)
        return _strdup("{\"error\":\"no query\"}");

    // Build query string
    char query[1024] = { 0 };
    for (int i = 0; i < argc; i++) {
        strcat(query, argv[i]);
        if (i + 1 < argc) strcat(query, " ");
    }

    // 1) Try Wikipedia first
    char* wiki = wikipedia_summary(query);
    if (wiki) {
        // Return Wikipedia JSON directly
        return wiki;
    }

    // 2) Fall back to DuckDuckGo
    char* ddg = plugin_ddg(argc, argv);
    if (!ddg)
        return _strdup("{\"error\":\"ddg_failed\"}");

    return ddg;
}
