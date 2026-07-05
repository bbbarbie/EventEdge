cmake_minimum_required(VERSION 3.16)

project(EventEdge LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_executable(eventedge
    src/event_probability.cpp
    src/main.cpp
    src/market_maker.cpp
    src/order_book.cpp
)

target_include_directories(eventedge PRIVATE include)
