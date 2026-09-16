add_subdirectory(op_host)

# 如果有 op_kernel/CMakeLists.txt 则添加
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/op_kernel/CMakeLists.txt")
    add_subdirectory(op_kernel)
endif()
