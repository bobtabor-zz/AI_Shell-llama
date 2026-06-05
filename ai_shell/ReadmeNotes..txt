_CRT_SECURE_NO_WARNINGS
WIN32
_WINDOWS
_DEBUG


$(WindowsSDK_IncludePath);$(VC_IncludePath);



cmake -B build
cmake --build build --config Release



ai_shell/
    CMakeLists.txt
    src/
        ai_shell.c        
    llama/
        src/
        include/
            llama.h
            llama.cpp
        ggml/
           include/
           src/
	   src/ggml-cpu


llama/ggml/include/ggml-backend-impl.h
llama/ggml/include/ggml-common.h


ai_shell/
    CMakeLists.txt
    src/
        ai_shell.c
        engine.cpp
	extmem.c
	http_winhttp.c
	pager.c
	server.c
	util.c
	vmm.c
    llama/
        src/
        include/
            llama.h
            llama.cpp
        ggml/
            include/
                ggml.h
                
            src/
		ggml-backend-impl.h
                ggml-common.h
                ggml.c
                ggml-quants.c
                ggml-backend.cpp
                ggml-cpu/
                    ggml-cpu.c
                    quants.c

new and tested:

llama.cpp/
    CMakeLists.txt        ← minimalized
    ggml/                 ← full backend (unchanged)
    src/                  ← llama core library (unchanged)
    include/              ← llama public headers (unchanged)

    ai_shell/
        CMakeLists.txt
        src/
            ai_shell.c
            engine.cpp
            extmem.c
            http_winhttp.c
            pager.c
            server.c
            util.c
            vmm.c
            *.h


ggml-backend.c

ggml-backend-ops.cpp  <

ggml-backend-reg.cpp

ggml-backend-device.cpp /ggml-virtgpu

ggml-backend-sched.cpp <

ggml-backend-cpu.c <

ggml-backend-cpu-alloc.c

ggml-backend-cpu-ops.c

ggml-opt.c

gguf.cpp

gguf.h

ggml-backend-ops.cpp ggml-backend-sched.cpp  ggml-backend-cpu.c

git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
git submodule update --init --recursive



Severity	Code	Description	Project	File	Line	Suppression State
Error	LNK2019	unresolved external symbol engine_infer_text referenced in function server_thread	ai_shell	C:\Users\btabor\source\repos\ai_shell-3\server.obj	1	
a clean separation between upstream llama.cpp and your custom engine




C:\Users\btabor\source\
    CMakeLists.txt        
    llama.cpp\           
    ai_shell\
        ai_shell\
		ai_shell.c
    	engine\
        	engine.c
        	engine.h
    	loader\
        	gguf_loader.c
        	gguf_loader.h
    	vmm\
        	vmm.c
        	vmm.h







