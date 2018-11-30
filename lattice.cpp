#include <string>
#include <pybind11/pybind11.h>

using namespace std;
namespace py = pybind11;

class Simulation
{
    public:
        Simulation(double beta, int L1, int L2, int seed) :
            lat(beta,L1,L2),
            rng(seed)
        {}
        //Simulation(py::dict data) :
        //    get_energy{true}
        //    //get_charge{1}
        //{
        //    for (auto [key,value] : data)
        //    {
        //        string key_c = key.cast<string>();
        //        if (key_c == "beta") beta = value.cast<double>();
        //        else if (key_c == "L1") L1 = value.cast<int>();
        //        else if (key_c == "L2") L2 = value.cast<int>();
        //        else if (key_c == "get_energy") get_energy = value.cast<bool>();
        //        else throw runtime_error("Invalid key: "+key_c);
        //    }

        //    if (!data.contains("beta")) throw runtime_error("Missing beta");
        //    if (!data.contains("L1")) throw runtime_error("Missing L1");
        //    if (!data.contains("L2")) L2=L1;
        //}

        py::dict data() const
        {
            py::dict data_;

            data_[py::cast("beta")] = py::cast(lat.beta());
            data_[py::cast("L1")] = py::cast(lat.L1());
            data_[py::cast("L2")] = py::cast(lat.L2());

            data_[py::cast("energies")] = py::cast(energies);
            //data_[py::cast("energy_mean")] = py::cast(energy_mean);
            //data_[py::cast("energy_err")] = py::cast(energy_err);

            data_[py::cast("charge")] = py::cast(charges);
            //data_[py::cast("susc_mean")] = py::cast(susc_mean);
            //data_[py::cast("susc_err")] = py::cast(susc_err);

            return data_;
        }

        void run(int iters)
        {
            for (int i=0; i<iters; i++)
            {
                double local_acc = sweep();
                double cluster_acc = cluster();
            }
        }

    private:
        Lattice lat;
        mt19937 rng;

        vector<double> energies;
        vector<double> charges;
};

PYBIND11_MODULE(prova, m)
{
    py::class_<Simulation>(m, "Simulation")
        .def(py::init<py::dict>())
        .def("data", &Simulation::data)
        .def(py::pickle(
                [](const Simulation &sim){return sim.data();},
                [](const py::dict &data)
                {
                    return Simulation(data);
                }));
}


