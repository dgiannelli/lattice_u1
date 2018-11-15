find_package(PythonLibs REQUIRED)
find_package(Boost COMPONENTS python REQUIRED)

include_directories(${PYTHON_INCLUDE_DIRS})
include_directories(${Boost_INCLUDE_DIRS})

add_library(
    lattice SHARED
    lattice.cpp
    )

target_link_libraries(
    lattice
    ${PYTHON_LIBRARIES}
    ${Boost_LIBRARIES}
    )

set_target_properties(lattice PROPERTIES PREFIX "")

