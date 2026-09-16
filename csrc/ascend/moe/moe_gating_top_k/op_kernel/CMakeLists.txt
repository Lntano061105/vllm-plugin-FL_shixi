file(GLOB OP_KERNEL_SRCS *.cpp)
add_library(moe_gating_top_k_kernel STATIC ${OP_KERNEL_SRCS})
