add_subdirectory(lib)
if(!WIN32)
##currently, tests are supported on Unix only
add_subdirectory(tests)
endif()
add_subdirectory(apps)
