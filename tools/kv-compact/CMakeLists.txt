set(TARGET llama-kv-compact)
add_executable(${TARGET} kv-compact.cpp)
target_link_libraries(${TARGET} PRIVATE common llama ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_17)

if(LLAMA_TOOLS_INSTALL)
    install(TARGETS ${TARGET} RUNTIME)
endif()

set(TARGET_PROFILE llama-kv-compact-profile)
add_executable(${TARGET_PROFILE} kv-compact-profile.cpp)
target_link_libraries(${TARGET_PROFILE} PRIVATE common llama ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET_PROFILE} PRIVATE cxx_std_17)

if(LLAMA_TOOLS_INSTALL)
    install(TARGETS ${TARGET_PROFILE} RUNTIME)
endif()
