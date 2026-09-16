file(GLOB OP_HOST_SRCS *.cpp)
add_library(moe_gating_top_k_host STATIC ${OP_HOST_SRCS})
