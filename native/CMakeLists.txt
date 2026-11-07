# Builds a single self-contained yapzip binary: llama.cpp linked statically, optionally with the model inside.
#   git clone --depth 1 --branch b11803 https://github.com/ggml-org/llama.cpp native/llama.cpp
#   cmake -S native -B build -DCMAKE_BUILD_TYPE=Release -DYAPZIP_MODEL=$PWD/models/yapzip-qwen3.5-0.8b-q8_0.gguf   # Mac (Metal)
#   cmake -S native -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=80 -DYAPZIP_MODEL=...                       # Linux + NVIDIA
#   cmake --build build --target yapzip -j
cmake_minimum_required(VERSION 3.18)
project(yapzip C CXX)
set(CMAKE_CXX_STANDARD 17)

set(LLAMA_DIR "${CMAKE_CURRENT_SOURCE_DIR}/llama.cpp" CACHE PATH "llama.cpp source checkout")
set(YAPZIP_MODEL "" CACHE FILEPATH "GGUF to embed in the binary (empty: load with -m at run time)")
set(YAPZIP_DEFAULT_MODEL "" CACHE STRING "model file a build without an embedded model loads by default")

set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)
set(LLAMA_CURL OFF CACHE BOOL "" FORCE)
if(APPLE)
  set(GGML_METAL_EMBED_LIBRARY ON CACHE BOOL "" FORCE)   # Metal shaders inside the binary
endif()

add_subdirectory(${LLAMA_DIR} llama.cpp EXCLUDE_FROM_ALL)
add_executable(yapzip yapzip.cpp)
target_link_libraries(yapzip PRIVATE llama)
if(YAPZIP_DEFAULT_MODEL)
  target_compile_definitions(yapzip PRIVATE YAPZIP_DEFAULT_MODEL="${YAPZIP_DEFAULT_MODEL}")
endif()

if(YAPZIP_MODEL)
  set_property(TARGET yapzip APPEND PROPERTY LINK_DEPENDS ${YAPZIP_MODEL})   # relink when the model file changes
  if(APPLE)
    # Model in its own Mach-O section: the code signature stays valid (appending bytes would break it).
    target_link_options(yapzip PRIVATE "LINKER:-sectcreate,__YAPZIP,__gguf,${YAPZIP_MODEL}"
                                       "LINKER:-sectalign,__YAPZIP,__gguf,0x4000")
  else()
    # ELF does not mind trailing data: append the model plus a "YZGGUF01"+offset trailer.
    add_custom_command(TARGET yapzip POST_BUILD
                       COMMAND sh ${CMAKE_CURRENT_SOURCE_DIR}/embed.sh $<TARGET_FILE:yapzip> ${YAPZIP_MODEL})
  endif()
endif()
