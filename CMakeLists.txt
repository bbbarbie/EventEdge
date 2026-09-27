cmake_minimum_required(VERSION 3.16)

project(EventEdge LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(eventedge_core STATIC
    src/event_probability.cpp
    src/market_maker.cpp
    src/order_book.cpp
    src/trader_agents.cpp
)
target_include_directories(eventedge_core PUBLIC include)

add_executable(eventedge src/main.cpp)
target_link_libraries(eventedge PRIVATE eventedge_core)

enable_testing()
add_executable(eventedge_tests tests/test_eventedge.cpp)
target_link_libraries(eventedge_tests PRIVATE eventedge_core)
add_test(NAME unit_tests COMMAND eventedge_tests)

# Python data-layer tests (Kalshi normalisation + replay through the binary).
find_package(Python3 COMPONENTS Interpreter QUIET)
if(Python3_Interpreter_FOUND)
    add_test(NAME kalshi_python
             COMMAND ${Python3_EXECUTABLE} -m unittest tests.test_kalshi
             WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
endif()
