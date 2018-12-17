//#include <string>
//#include <sstream>

#include "lattice.hpp"
#include "simulation.hpp"


PYBIND11_MODULE(lattice, m)
{
    py::class_<Simulation>(m, "Simulation")
        .def(py::init<double,int,int,int>())
        .def("data", &Simulation::data)
        .def("run", &Simulation::run)
        .def(py::pickle(
                [](const Simulation &sim){return sim.data();},
                [](const py::dict &data)
                {
                    double beta = data[py::cast("beta")].cast<double>();
                    int L1 = data[py::cast("L1")].cast<int>();
                    int L2 = data[py::cast("L2")].cast<int>();
                    int seed = data[py::cast("seed")].cast<int>();

                    Simulation sim(beta,L1,L2,seed);
                    sim.set_state(data);

                    return sim;
                }));
}

