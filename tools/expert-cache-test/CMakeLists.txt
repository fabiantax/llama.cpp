add_executable(expert-cache-test expert-cache-test.cpp)
target_link_libraries(expert-cache-test PRIVATE common llama ${CMAKE_THREAD_LIBS_INIT})
target_include_directories(expert-cache-test PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/common
    ${CMAKE_SOURCE_DIR}/src
)
install(TARGETS expert-cache-test RUNTIME)
