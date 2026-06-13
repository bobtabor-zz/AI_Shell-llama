#include "vmm.h"

#include <windows.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>   // for true/false

#ifdef __cplusplus
extern "C" {
#endif
#include "gguf.h"
#ifdef __cplusplus
}
#endif


typedef struct vmm_state {
    HANDLE    file;
    HANDLE    mapping;
    uint64_t  size;
    uint64_t  used;
    int       default_numa_node;
    uint8_t* window_base;
    uint64_t  window_offset;
    uint64_t  window_size;
} vmm_state_t;

static vmm_state_t g_vmm;

struct vmm {
    vmm_config_t cfg;
    size_t       used;
};

vmm_model_t* vmm_model_open(
    const char* gguf_path,
    const char* vmm_path,
    size_t      vmm_budget_bytes)
{
    // ---- 1. If vmm.bin does NOT exist → build it ----
    FILE* f = fopen(vmm_path, "rb");
    if (!f) {
        fprintf(stderr, "[VMM] vmm.bin missing → building\n");

        int rc = vmm_model_build_from_gguf(
            gguf_path,
            vmm_path,
            vmm_budget_bytes);

        if (rc != 0) {
            fprintf(stderr, "[VMM] build_from_gguf FAILED rc=%d\n", rc);
            return NULL;
        }
    }
    else {
        fclose(f);
    }

    // ---- 2. Open existing vmm.bin (NO TRUNCATE) ----
    vmm_t* vmm = vmm_create(vmm_path, 0, VMM_MODE_OPEN_EXISTING);
    if (!vmm) {
        fprintf(stderr, "[VMM] vmm_create(OPEN_EXISTING) FAILED\n");
        return NULL;
    }

    // ---- 3. Map header ----
    void* hdr_ptr = vmm_map(vmm, 0, 64 * 1024);
    if (!hdr_ptr) {
        fprintf(stderr, "[VMM] vmm_map header FAILED\n");
        vmm_destroy(vmm);
        return NULL;
    }

    vmm_model_header_t* hdr = (vmm_model_header_t*)hdr_ptr;
    if (hdr->magic != VMM_MODEL_MAGIC) {
        fprintf(stderr, "[VMM] bad magic in vmm.bin — rebuilding\n");
        vmm_unmap(vmm, hdr_ptr, 64 * 1024);
        vmm_destroy(vmm);

        // force rebuild
        int rc = vmm_model_build_from_gguf(
            gguf_path,
            vmm_path,
            vmm_budget_bytes);
        if (rc != 0) {
            fprintf(stderr, "[VMM] rebuild_from_gguf FAILED rc=%d\n", rc);
            return NULL;
        }

        // reopen after rebuild
        vmm = vmm_create(vmm_path, 0, VMM_MODE_OPEN_EXISTING);
        if (!vmm) {
            fprintf(stderr, "[VMM] vmm_create(OPEN_EXISTING) FAILED after rebuild\n");
            return NULL;
        }

        hdr_ptr = vmm_map(vmm, 0, 64 * 1024);
        if (!hdr_ptr) {
            fprintf(stderr, "[VMM] vmm_map header FAILED after rebuild\n");
            vmm_destroy(vmm);
            return NULL;
        }
        hdr = (vmm_model_header_t*)hdr_ptr;

        if (hdr->magic != VMM_MODEL_MAGIC) {
            fprintf(stderr, "[VMM] bad magic even after rebuild\n");
            vmm_unmap(vmm, hdr_ptr, 64 * 1024);
            vmm_destroy(vmm);
            return NULL;
        }
    }

    //if (vmm_sanity_check(vmm, vmm_path, hdr) != 0) {
    //    fprintf(stderr, "[VMM] sanity failed — rebuilding\n");
    //    vmm_unmap(vmm, hdr_ptr, 64 * 1024);
    //    vmm_destroy(vmm);

    //    int rc = vmm_model_build_from_gguf(
    //        gguf_path,
    //        vmm_path,
    //        vmm_budget_bytes);

    //    if (rc != 0) {
    //        fprintf(stderr, "[VMM] rebuild_from_gguf FAILED rc=%d\n", rc);
    //        return NULL;
    //    }

    //    // reopen
    //    vmm = vmm_create(vmm_path, 0, VMM_MODE_OPEN_EXISTING);
    //    hdr_ptr = vmm_map(vmm, 0, 64 * 1024);
    //    hdr = (vmm_model_header_t*)hdr_ptr;

    //    if (vmm_sanity_check(vmm, vmm_path, hdr) != 0) {
    //        fprintf(stderr, "[VMM] sanity STILL failing after rebuild\n");
    //        vmm_unmap(vmm, hdr_ptr, 64 * 1024);
    //        vmm_destroy(vmm);
    //        return NULL;
    //    }
    //}

    fprintf(stderr, "[VMM] USING vmm.bin (magic OK, tensors=%u)\n", hdr->tensor_count);

    // ---- 4. Map meta table ----
    size_t meta_bytes = hdr->tensor_count * sizeof(vmm_tensor_meta_t);
    void* meta_ptr = vmm_map(vmm, hdr->meta_offset, meta_bytes);
    if (!meta_ptr) {
        fprintf(stderr, "[VMM] vmm_map meta FAILED\n");
        vmm_unmap(vmm, hdr_ptr, 64 * 1024);
        vmm_destroy(vmm);
        return NULL;
    }

    // ---- 5. Build model struct ----
    vmm_model_t* m = calloc(1, sizeof(vmm_model_t));
    m->vmm = vmm;
    m->tensor_count = hdr->tensor_count;
    m->meta_offset = hdr->meta_offset;
    m->data_offset = hdr->data_offset;
    m->meta = (vmm_tensor_meta_t*)meta_ptr;

    fprintf(stderr, "[VMM] open OK: tensors=%u\n", m->tensor_count);
    return m;
}

void vmm_inspect(vmm_model_t* m) {
    if (!m || !m->vmm || !m->meta) {
        fprintf(stderr, "[VMM] inspect: invalid model\n");
        return;
    }

    int real_count = 0;
    uint64_t file_size = vmm_budget(m->vmm);

    fprintf(stderr, "\n==================== VMM INSPECTOR ====================\n");
    fprintf(stderr, "File: vmm.bin\n");
    fprintf(stderr, "File size: %llu bytes\n", (unsigned long long)file_size);
    fprintf(stderr, "Tensor count: %u\n", m->tensor_count);
    fprintf(stderr, "Meta offset: %llu\n", (unsigned long long)m->meta_offset);
    fprintf(stderr, "Data offset: %llu\n", (unsigned long long)m->data_offset);
    fprintf(stderr, "--------------------------------------------------------\n");

    uint64_t last_end = m->data_offset;

    for (uint32_t i = 0; i < m->tensor_count; i++) {
        vmm_tensor_meta_t* t = &m->meta[i];

        // ⭐ SKIP EMPTY GGUF TENSORS ⭐
        if (t->size_bytes == 0 || t->offset == 0)
            continue;

        real_count++;

        uint64_t start = t->offset;
        uint64_t end = t->offset + t->size_bytes;

        // gap detection
        if (start > last_end) {
            fprintf(stderr,
                "  GAP: %llu bytes (0x%llX - 0x%llX)\n",
                (unsigned long long)(start - last_end),
                (unsigned long long)last_end,
                (unsigned long long)start);
        }

        fprintf(stderr,
            "Tensor %4u | %-40s\n"
            "  dtype=%u  dims=%u  size=%llu\n"
            "  offset=0x%llX  end=0x%llX\n"
            "  shape=[%llu %llu %llu %llu]\n"
            "--------------------------------------------------------\n",
            i,
            t->name,
            t->dtype,
            t->n_dims,
            (unsigned long long)t->size_bytes,
            (unsigned long long)start,
            (unsigned long long)end,
            (unsigned long long)t->ne[0],
            (unsigned long long)t->ne[1],
            (unsigned long long)t->ne[2],
            (unsigned long long)t->ne[3]
        );

        last_end = end;
    }

    fprintf(stderr, "[VMM] real tensors = %u\n", real_count);

    fprintf(stderr, "================== END VMM INSPECTOR ==================\n\n");
}




static int vmm_model_build_from_gguf(
    const char* gguf_path,
    const char* vmm_path,
    size_t      vmm_budget_bytes)
{
    // ---- 1. Load GGUF (no tensor alloc) ----
    struct gguf_init_params params = {
        .no_alloc = true,
        .ctx = NULL,
    };

    struct gguf_context* ctx = gguf_init_from_file(gguf_path, params);
    if (!ctx) {
        fprintf(stderr, "[VMM] gguf_init_from_file FAILED for '%s'\n", gguf_path);
        return -1;
    }

    int n = (int)gguf_get_n_tensors(ctx);
    gguf_tensor_desc_t* tensors =
        (gguf_tensor_desc_t*)calloc(n, sizeof(gguf_tensor_desc_t));

    if (!tensors) {
        gguf_free(ctx);
        return -2;
    }

    for (int i = 0; i < n; i++) {
        gguf_tensor_desc_t* td = &tensors[i];

        const char* name = gguf_get_tensor_name(ctx, i);
        uint64_t    off = gguf_get_tensor_offset(ctx, i);
        size_t      size = gguf_get_tensor_size(ctx, i);
        int         type = gguf_get_tensor_type(ctx, i);

        strncpy(td->name, name, sizeof(td->name) - 1);
        td->dtype = (uint32_t)type;
        td->file_offset = off;
        td->size_bytes = size;

        td->n_dims = 1;
        td->ne[0] = size;
        td->ne[1] = 1;
        td->ne[2] = 1;
        td->ne[3] = 1;
    }

    // ---- 2. Create/truncate vmm.bin ----
    vmm_t* vmm = vmm_create(vmm_path, vmm_budget_bytes, VMM_MODE_CREATE_TRUNCATE);
    if (!vmm) {
        fprintf(stderr, "[VMM] vmm_create(TRUNCATE) FAILED\n");
        free(tensors);
        gguf_free(ctx);
        return -3;
    }

    // ---- 3. Allocate header + meta ----
    size_t meta_bytes = n * sizeof(vmm_tensor_meta_t);
    size_t header_bytes = sizeof(vmm_model_header_t) + meta_bytes;

    vmm_region_t hdr_region;
    if (vmm_alloc(vmm, header_bytes, 0, -1, &hdr_region) != 0) {
        fprintf(stderr, "[VMM] header alloc FAILED\n");
        vmm_destroy(vmm);
        free(tensors);
        gguf_free(ctx);
        return -4;
    }

    uint8_t* hdr_base = (uint8_t*)hdr_region.ptr;
    vmm_model_header_t* hdr = (vmm_model_header_t*)hdr_base;
    vmm_tensor_meta_t* meta = (vmm_tensor_meta_t*)(hdr_base + sizeof(*hdr));

    memset(hdr, 0, sizeof(*hdr));
    hdr->magic = VMM_MODEL_MAGIC;
    hdr->version = 1;
    hdr->tensor_count = 0;  // will set after loop
    hdr->meta_offset = hdr_region.offset + sizeof(*hdr);
    hdr->data_offset = hdr_region.offset + header_bytes;

    // ---- 4. Copy tensor data ----
    FILE* f = fopen(gguf_path, "rb");
    if (!f) {
        fprintf(stderr, "[VMM] fopen GGUF FAILED\n");
        vmm_destroy(vmm);
        free(tensors);
        gguf_free(ctx);
        return -5;
    }

    uint32_t written = 0;

    for (int i = 0; i < n; i++) {
        gguf_tensor_desc_t* td = &tensors[i];
        vmm_region_t r;

        if (vmm_alloc(vmm, td->size_bytes, 0, -1, &r) != 0) {
            fprintf(stderr, "[VMM] data alloc FAILED i=%d (stopping)\n", i);
            break;
        }

        if (fseek(f, (long)td->file_offset, SEEK_SET) != 0 ||
            fread(r.ptr, 1, td->size_bytes, f) != td->size_bytes) {
            fprintf(stderr, "[VMM] fread FAILED i=%d (stopping)\n", i);
            break;
        }

        vmm_tensor_meta_t* m = &meta[written];
        memset(m, 0, sizeof(*m));

        strncpy(m->name, td->name, sizeof(m->name) - 1);
        m->dtype = td->dtype;
        m->n_dims = td->n_dims;
        m->size_bytes = td->size_bytes;
        m->offset = r.offset;

        for (uint32_t d = 0; d < td->n_dims; d++)
            m->ne[d] = td->ne[d];

        written++;
    }

    hdr->tensor_count = written;

    fclose(f);
    vmm_destroy(vmm);
    free(tensors);
    gguf_free(ctx);

    fprintf(stderr, "[VMM] build_from_gguf OK → %s (tensors=%u)\n",
        vmm_path, hdr->tensor_count);
    return 0;
}



static void vmm_state_reset(void) {
    memset(&g_vmm, 0, sizeof(g_vmm));
}

static int vmm_disk_init(const char* path, uint64_t size_bytes, vmm_mode_t mode) {
    vmm_state_reset();

    DWORD create_disp;
    switch (mode) {
    case VMM_MODE_CREATE_TRUNCATE:
        create_disp = CREATE_ALWAYS;   // always recreate/truncate
        break;
    case VMM_MODE_OPEN_EXISTING:
        create_disp = OPEN_EXISTING;   // must already exist
        break;
    default:
        return -1;
    }

    g_vmm.file = CreateFileA(path,
        GENERIC_READ | GENERIC_WRITE,
        FILE_SHARE_READ | FILE_SHARE_WRITE,
        NULL,
        create_disp,
        FILE_ATTRIBUTE_TEMPORARY | FILE_FLAG_RANDOM_ACCESS,
        NULL);
    if (g_vmm.file == INVALID_HANDLE_VALUE) {
        return -2;
    }

    LARGE_INTEGER li;
    li.QuadPart = size_bytes;

    if (mode == VMM_MODE_CREATE_TRUNCATE) {
        if (!SetFilePointerEx(g_vmm.file, li, NULL, FILE_BEGIN) ||
            !SetEndOfFile(g_vmm.file)) {
            return -3;
        }
    }
    else {
        // OPEN_EXISTING: infer size from file if size_bytes == 0
        if (size_bytes == 0) {
            LARGE_INTEGER sz;
            if (!GetFileSizeEx(g_vmm.file, &sz)) {
                return -4;
            }
            li = sz;
        }
    }

    g_vmm.size = (uint64_t)li.QuadPart;

    g_vmm.mapping = CreateFileMappingA(g_vmm.file,
        NULL,
        PAGE_READWRITE,
        li.HighPart,
        li.LowPart,
        NULL);
    if (!g_vmm.mapping) {
        return -5;
    }

    // initial window: map whole file (you can later shrink to sliding window)
    g_vmm.window_base = (uint8_t*)MapViewOfFile(g_vmm.mapping,
        FILE_MAP_ALL_ACCESS,
        0,
        0,
        g_vmm.size);
    if (!g_vmm.window_base) {
        return -6;
    }

    g_vmm.window_offset = 0;
    g_vmm.window_size = g_vmm.size;
    g_vmm.used = 0;

    return 0;
}

static vmm_t* vmm_init(const vmm_config_t* cfg, const char* path, vmm_mode_t mode) {
    if (!cfg || !path) {
        return NULL;
    }

    if (vmm_disk_init(path, cfg->budget_bytes, mode) != 0) {
        return NULL;
    }

    g_vmm.default_numa_node = cfg->default_numa_node;

    vmm_t* v = (vmm_t*)calloc(1, sizeof(vmm_t));
    if (!v) {
        return NULL;
    }

    v->cfg = *cfg;
    v->used = 0;

    return v;
}

vmm_t* vmm_create(const char* path, size_t size_bytes, vmm_mode_t mode) {
    vmm_config_t cfg;
    cfg.budget_bytes = size_bytes;
    cfg.prefer_largepages = 0;
    cfg.prefetch_on_alloc = 0;
    cfg.default_numa_node = -1;

    return vmm_init(&cfg, path, mode);
}

static void vmm_shutdown(vmm_t* v) {
    (void)v;

    if (g_vmm.window_base) {
        UnmapViewOfFile(g_vmm.window_base);
        g_vmm.window_base = NULL;
        g_vmm.window_offset = 0;
        g_vmm.window_size = 0;
    }

    if (g_vmm.mapping) {
        CloseHandle(g_vmm.mapping);
        g_vmm.mapping = NULL;
    }

    if (g_vmm.file && g_vmm.file != INVALID_HANDLE_VALUE) {
        CloseHandle(g_vmm.file);
        g_vmm.file = NULL;
    }

    vmm_state_reset();
}

void vmm_destroy(vmm_t* v) {
    if (v) {
        vmm_shutdown(v);
        free(v);
    }
}

size_t vmm_budget(vmm_t* v) {
    (void)v;
    return (size_t)g_vmm.size;
}

size_t vmm_used(vmm_t* v) {
    (void)v;
    return (size_t)g_vmm.used;
}

int vmm_alloc(vmm_t* v, size_t bytes, vmm_flags_t flags, int numa, vmm_region_t* out) {
    (void)v;

    if (!out || bytes == 0) {
        return -1;
    }

    uint64_t align = 64ull * 1024ull;
    uint64_t current = g_vmm.used;
    uint64_t aligned = (current + align - 1) & ~(align - 1);

    if (aligned + bytes > g_vmm.size) {
        return -2;
    }

    // ensure fits in current window
    if (!g_vmm.window_base ||
        aligned < g_vmm.window_offset ||
        aligned + bytes > g_vmm.window_offset + g_vmm.window_size) {
        // for now, fail; you can later implement sliding window remap here
        return -3;
    }

    out->offset = aligned;
    out->size = bytes;
    out->flags = flags;
    out->numa_node = numa;
    out->ptr = g_vmm.window_base + (aligned - g_vmm.window_offset);

    g_vmm.used = aligned + bytes;

    return 0;
}

int vmm_free(vmm_t* v, vmm_region_t* r) {
    (void)v;
    if (!r) return -1;
    memset(r, 0, sizeof(*r));
    return 0;
}

static void* vmm_map_window(size_t offset, size_t length) {
    SYSTEM_INFO si;
    GetSystemInfo(&si);
    size_t gran = si.dwAllocationGranularity;

    size_t aligned = offset & ~(gran - 1);
    size_t delta = offset - aligned;
    size_t map_sz = length + delta;

    if (g_vmm.window_base &&
        aligned >= g_vmm.window_offset &&
        aligned + map_sz <= g_vmm.window_offset + g_vmm.window_size) {
        return g_vmm.window_base + (aligned - g_vmm.window_offset) + delta;
    }

    if (g_vmm.window_base) {
        UnmapViewOfFile(g_vmm.window_base);
        g_vmm.window_base = NULL;
        g_vmm.window_offset = 0;
        g_vmm.window_size = 0;
    }

    uint32_t lo = (uint32_t)(aligned & 0xFFFFFFFFull);
    uint32_t hi = (uint32_t)(aligned >> 32);

    g_vmm.window_base = (uint8_t*)MapViewOfFile(g_vmm.mapping,
        FILE_MAP_ALL_ACCESS,
        hi,
        lo,
        map_sz);
    if (!g_vmm.window_base) {
        return NULL;
    }

    g_vmm.window_offset = aligned;
    g_vmm.window_size = map_sz;

    return g_vmm.window_base + delta;
}

void* vmm_map(vmm_t* v, uint64_t offset, size_t length) {
    (void)v;
    return vmm_map_window((size_t)offset, length);
}

void vmm_unmap(vmm_t* v, void* ptr, size_t length) {
    (void)v;
    (void)ptr;
    (void)length;
    // single-window design: drop it
    if (g_vmm.window_base) {
        UnmapViewOfFile(g_vmm.window_base);
        g_vmm.window_base = NULL;
        g_vmm.window_offset = 0;
        g_vmm.window_size = 0;
    }
}

int vmm_pin(vmm_t* v, vmm_region_t* r) {
#ifdef _WIN32
    (void)v;
    if (!r || !r->ptr || r->size == 0) return -1;
    return VirtualLock(r->ptr, r->size) ? 0 : -1;
#else
    (void)v; (void)r;
    return -99;
#endif
}

int vmm_unpin(vmm_t* v, vmm_region_t* r) {
#ifdef _WIN32
    (void)v;
    if (!r || !r->ptr || r->size == 0) return 0;
    return VirtualUnlock(r->ptr, r->size) ? 0 : -1;
#else
    (void)v; (void)r;
    return -99;
#endif
}

int vmm_prefetch(vmm_t* v, void* ptr, size_t bytes) {
#ifdef _WIN32
    (void)v;
    WIN32_MEMORY_RANGE_ENTRY range = { ptr, (SIZE_T)bytes };
    PrefetchVirtualMemory(GetCurrentProcess(), 1, &range, 0);
    return 0;
#else
    (void)v; (void)ptr; (void)bytes;
    return -99;
#endif
}

void vmm_set_default_numa(vmm_t* v, int node) {
    if (!v) return;
    v->cfg.default_numa_node = node;
    g_vmm.default_numa_node = node;
}

int vmm_get_default_numa(vmm_t* v) {
    (void)v;
    return g_vmm.default_numa_node;
}


void vmm_model_close(vmm_model_t* m) {
    if (!m) return;
    if (m->vmm && m->meta) {
        vmm_unmap(m->vmm, m->meta, m->tensor_count * sizeof(vmm_tensor_meta_t));
    }
    if (m->vmm) {
        vmm_destroy(m->vmm);
    }
    free(m);
}


