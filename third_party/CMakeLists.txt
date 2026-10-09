add_subdirectory(simsimd)
add_subdirectory(hdrhistogram_c)
add_subdirectory(hnswlib)
add_subdirectory(snowball)
add_subdirectory(icu)

# SVS C API, x86_64 Linux only; elsewhere vector_svs.cc compiles as a stub.
# Default: compile the public C API from the svs submodule (no LVQ/LeanVec).
# SVS_C_API_PREBUILT links the proprietary prebuilt archive instead. The C ABI
# is identical in both, so LVQ/LeanVec availability is probed at runtime.
option(SVS_C_API_PREBUILT
       "Link the prebuilt SVS C API (LVQ/LeanVec) instead of compiling third_party/svs" OFF)
set(SVS_C_API_URL
    "https://github.com/intel/ScalableVectorSearch/releases/download/nightly/svs-c-api-nightly-2026-10-09-99.tar.gz"
    CACHE STRING "Prebuilt SVS C API tarball: a release URL or a local path")

if(NOT (UNIX AND NOT APPLE
        AND CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|amd64|AMD64)$"))
  if(SVS_C_API_PREBUILT)
    message(FATAL_ERROR "The prebuilt SVS C API requires x86_64 Linux")
  endif()
  message(STATUS "SVS C API: unavailable on this platform")
  return()
endif()

find_package(OpenMP REQUIRED)

if(SVS_C_API_PREBUILT)
  include(FetchContent)
  if(POLICY CMP0135)
    cmake_policy(SET CMP0135 NEW)
  endif()
  FetchContent_Declare(svs_c_api URL "${SVS_C_API_URL}")
  FetchContent_MakeAvailable(svs_c_api)
  set(SVS_C_API_ROOT "${svs_c_api_SOURCE_DIR}")
  # The archive bundles fmt/spdlog/MKL; --exclude-libs,ALL on libsearch keeps
  # those symbols out of the module's dynamic table.
  find_package(svs_c_api_static CONFIG REQUIRED
               PATHS "${SVS_C_API_ROOT}" NO_DEFAULT_PATH)
  # Imported targets are scoped to this directory; src/indexes links it too.
  set_target_properties(svs::svs_c_api_static PROPERTIES IMPORTED_GLOBAL TRUE)
  # Wrapped under one name so src/indexes links the same target either way.
  add_library(svs_c_api_static INTERFACE)
  target_link_libraries(svs_c_api_static INTERFACE svs::svs_c_api_static)
  target_compile_definitions(svs_c_api_static INTERFACE
      VALKEY_SEARCH_SVS_PREBUILT=1)
  message(STATUS "SVS C API: prebuilt (LVQ/LeanVec) from ${SVS_C_API_ROOT}")
else()
  # bindings/c is added EXCLUDE_FROM_ALL only so it generates svs_c_version.h;
  # its shared library (and libgomp dependency) stays out of the link set.
  add_subdirectory(svs ${CMAKE_CURRENT_BINARY_DIR}/svs-build)
  add_subdirectory(svs/bindings/c ${CMAKE_CURRENT_BINARY_DIR}/svs-c-api-build
                   EXCLUDE_FROM_ALL)
  # The C API sources compile with SVS's strict warnings-as-errors flags
  # (svs_compile_options); a dedicated library keeps them off vector_svs.cc.
  set(SVS_C_API_DIR "${CMAKE_CURRENT_SOURCE_DIR}/svs/bindings/c")
  set(SVS_C_API_INCLUDE_DIR "${SVS_C_API_DIR}/include")
  set(SRCS_SVS_C_API
      ${SVS_C_API_DIR}/src/error.cpp
      ${SVS_C_API_DIR}/src/svs_c.cpp
      ${SVS_C_API_DIR}/src/dispatcher_vamana.cpp
      ${SVS_C_API_DIR}/src/dispatcher_dynamic_vamana.cpp
      ${SVS_C_API_DIR}/src/leanvec_training_data.cpp
      ${SVS_C_API_DIR}/src/data_builder.cpp)
  valkey_search_add_static_library(svs_c_api_static "${SRCS_SVS_C_API}")
  target_include_directories(svs_c_api_static PUBLIC
      ${SVS_C_API_INCLUDE_DIR}
      ${CMAKE_CURRENT_BINARY_DIR}/svs-c-api-build/include)
  target_include_directories(svs_c_api_static PRIVATE ${SVS_C_API_DIR}/src)
  target_compile_definitions(svs_c_api_static PRIVATE SVS_ENABLE_OMP=1)
  target_compile_options(svs_c_api_static PRIVATE -fvisibility=hidden)
  target_link_libraries(svs_c_api_static PRIVATE
      svs::svs
      svs_compile_options
      svs_x86_options_base)
  target_link_libraries(svs_c_api_static PUBLIC OpenMP::OpenMP_CXX)
  target_compile_definitions(svs_c_api_static INTERFACE
      VALKEY_SEARCH_SVS_PREBUILT=0)
  message(STATUS "SVS C API: source (public, no LVQ/LeanVec)")
endif()
