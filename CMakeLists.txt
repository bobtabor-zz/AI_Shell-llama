cmake_minimum_required(VERSION 3.16)
project(ai_shell_minimal C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_EXTENSIONS OFF)

# -------------------------------------------------------------
# Bring in upstream llama.cpp (unchanged)
# -------------------------------------------------------------
add_subdirectory(llama.cpp ${CMAKE_BINARY_DIR}/llama_build)

# -------------------------------------------------------------
# Build your custom engine as a static library
# -------------------------------------------------------------
add_library(engine STATIC
    ai_shell/engine/engine.c
    ai_shell/loader/gguf_loader.c
    ai_shell/vmm/vmm.c
)

    target_include_directories(engine PUBLIC
    ai_shell/engine
    ai_shell/loader
    ai_shell/vmm
    llama.cpp/include
)

target_link_libraries(engine PUBLIC llama)

# -------------------------------------------------------------
# Build ai_shell executable
# -------------------------------------------------------------
add_executable(ai_shell
    ai_shell/ai_shell/ai_shell.c
)

target_include_directories(ai_shell PUBLIC
    ai_shell/engine
    ai_shell/loader
    ai_shell/vmm

)

target_link_libraries(ai_shell PRIVATE engine)
