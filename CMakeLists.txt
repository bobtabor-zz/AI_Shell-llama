cmake_minimum_required(VERSION 3.16)
project(ai_shell_minimal C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_EXTENSIONS OFF)

# =============================================================
# llama.cpp Build Configuration
# =============================================================
set(LLAMA_C_API        ON  CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SHARED ON  CACHE BOOL "" FORCE)

set(LLAMA_VULKAN       ON  CACHE BOOL "" FORCE)
set(LLAMA_VULKAN_F16   ON  CACHE BOOL "" FORCE)
set(LLAMA_FLASH_ATTN   ON  CACHE BOOL "" FORCE)
set(LLAMA_NATIVE       ON  CACHE BOOL "" FORCE)

set(LLAMA_CUDA         OFF CACHE BOOL "" FORCE)

# =============================================================
# Add llama.cpp
# =============================================================
add_subdirectory(llama.cpp ${CMAKE_BINARY_DIR}/llama_build)

# =============================================================
# Engine static library
# =============================================================
add_library(engine STATIC
    ai_shell/engine/engine.c
    ai_shell/engine/engine_kv.c
    ai_shell/loader/gguf_loader.c
    ai_shell/vmm/vmm.c
    ai_shell/plugins/ddg.c
    ai_shell/plugins/summarize_file.c
    ai_shell/plugins/websearch.c
    ai_shell/include/plugin.h
    ai_shell/core/plugin.c
)

target_include_directories(engine PUBLIC
    ai_shell/engine
    ai_shell/loader
    ai_shell/vmm
    ai_shell/plugins
    ai_shell/include
    ai_shell/core
    ${CMAKE_SOURCE_DIR}/llama.cpp/include
)

target_link_libraries(engine PUBLIC llama)

# =============================================================
# Main executable
# =============================================================
add_executable(ai_shell
    ai_shell/main/ai_shell.c
)

target_include_directories(ai_shell PUBLIC
    ai_shell/engine
    ai_shell/loader
    ai_shell/vmm
    ${CMAKE_SOURCE_DIR}/llama.cpp/include
)

target_link_libraries(ai_shell PRIVATE engine)

# =============================================================
# Copy ALL important files: .dll, .lib, .exp
# =============================================================
#add_custom_command(TARGET ai_shell POST_BUILD
    # Copy DLLs
#    COMMAND ${CMAKE_COMMAND} -E copy_if_different
#        $<TARGET_FILE:llama> $<TARGET_FILE_DIR:ai_shell>

#    COMMAND ${CMAKE_COMMAND} -E copy_if_different
#        $<TARGET_FILE:ggml> $<TARGET_FILE_DIR:ai_shell>

#    COMMAND ${CMAKE_COMMAND} -E copy_if_different
#        $<TARGET_FILE:ggml-base> $<TARGET_FILE_DIR:ai_shell>

#    COMMAND ${CMAKE_COMMAND} -E copy_if_different
#        $<TARGET_FILE:ggml-cpu> $<TARGET_FILE_DIR:ai_shell>

    # Copy .lib files (Import Libraries)
#    COMMAND ${CMAKE_COMMAND} -E copy_if_different
#        $<TARGET_LINKER_FILE:llama> $<TARGET_FILE_DIR:ai_shell>

#    COMMAND ${CMAKE_COMMAND} -E copy_if_different
 #       $<TARGET_LINKER_FILE:ggml> $<TARGET_FILE_DIR:ai_shell>

 #   COMMAND ${CMAKE_COMMAND} -E copy_if_different
 #       $<TARGET_LINKER_FILE:ggml-base> $<TARGET_FILE_DIR:ai_shell>

 #   COMMAND ${CMAKE_COMMAND} -E copy_if_different
 #       $<TARGET_LINKER_FILE:ggml-cpu> $<TARGET_FILE_DIR:ai_shell>

    # Copy .exp files (optional but useful)
 #   COMMAND ${CMAKE_COMMAND} -E copy_if_different
#        ${CMAKE_BINARY_DIR}/llama_build/bin/llama.exp $<TARGET_FILE_DIR:ai_shell> 
 #       COMMAND_ERROR_IS_FATAL ANY

 #   COMMENT "Copying DLLs + .lib + .exp files to output directory"
#)