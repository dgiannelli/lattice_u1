//#include <string>
//#include <sstream>

#include "lattice.hpp"
#include "simulation.hpp"


PYBIND11_MODULE(simulation, m)
{
    py::class_<Simulation>(m, "Simulation")
        .def(py::init<double,int,int,int,string>())
        .def("run", &Simulation::run)
        .def("analysis", &Simulation::analysis)
        .def("write_state", &Simulation::write_state);
}

