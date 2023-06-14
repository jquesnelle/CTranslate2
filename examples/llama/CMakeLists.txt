cmake_minimum_required(VERSION 3.7)
project(llama)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Release)
find_package(Threads)
add_executable(llama main.cc)
target_include_directories(llama PUBLIC "$<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/third_party/sentencepiece/src>" "$<INSTALL_INTERFACE:include>")
target_link_libraries(llama PRIVATE
  ${CMAKE_THREAD_LIBS_INIT}
  ctranslate2
  sentencepiece
  ${CMAKE_DL_LIBS}
  )
