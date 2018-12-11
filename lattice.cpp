#include <string>
#include <sstream>

#include <pybind11/pybind11.h>

#include "lattice.hpp"

using namespace std;
namespace py = pybind11;

class Simulation
{
    public:
        Simulation(double beta, int L1, int L2, int seed) :
            lat(beta,L1,L2),
            rng(seed),
            seed{seed}
        {}

        py::dict data() const
        {
            py::dict data_;

            data_[py::cast("beta")] = py::cast(lat.beta());
            data_[py::cast("L1")] = py::cast(lat.L1());
            data_[py::cast("L2")] = py::cast(lat.L2());

            data_[py::cast("seed")] = py::cast(seed);

            stringstream rng_stream; rng_stream << rng;
            data_[py::cast("rng_state")] = py::cast(rng_stream.str());

            data_[py::cast("config")] = py::cast(lat.config());

            data_[py::cast("energies")] = py::cast(energies);
            //data_[py::cast("energy_mean")] = py::cast(energy_mean);
            //data_[py::cast("energy_err")] = py::cast(energy_err);

            data_[py::cast("charge")] = py::cast(charges);
            //data_[py::cast("susc_mean")] = py::cast(susc_mean);
            //data_[py::cast("susc_err")] = py::cast(susc_err);

            data_[py::cast("local_accs")] = py::cast(local_accs);
            //data_[py::cast("local_acc_mean")] = py::cast(local_acc_mean);
            //data_[py::cast("local_acc_err")] = py::cast(local_acc_err);

            data_[py::cast("cluster_accs")] = py::cast(cluster_accs);

            return data_;
        }

        void set_state(const py::dict &data)
        {
            iters = data[py::cast("iters")].cast<int>();

            stringstream((data[py::cast("rng_state")].cast<string>())) >> rng;
            lat.set_config(data[py::cast("config")].cast<vector<double>>());

            energies = data[py::cast("energies")].cast<vector<double>>();
            //energy_mean = data[py::cast("energy_mean")].cast<double>();
            //energy_err = data[py::cast("energy_err")].cast<double>();

            charges = data[py::cast("charges")].cast<vector<double>>();
            //charge_mean = data[py::cast("charge_mean")].cast<double>();
            //charge_err = data[py::cast("charge_err")].cast<double>();
            
            sweep_acc = data[py::cast("local_accs")].cast<vector<double>>();

            cluster_acc = data[py::cast("cluster_accs")].cast<vector<double>>();

            //for (auto [key,value] : data)
            //{
            //    string key_c = key.cast<string>();
            //    if (key_c == "beta") beta = value.cast<double>();
            //    else if (key_c == "L1") L1 = value.cast<int>();
            //    else if (key_c == "L2") L2 = value.cast<int>();
            //    else if (key_c == "get_energy") get_energy = value.cast<bool>();
            //    else throw runtime_error("Invalid key: "+key_c);
            //}

            //if (!data.contains("beta")) throw runtime_error("Missing beta");
            //if (!data.contains("L1")) throw runtime_error("Missing L1");
            //if (!data.contains("L2")) L2=L1;
        }

        void set_hot()
        {
            for (int mu : {1,2}) {
                for (Site s : lat.sites()) {
                    double theta = UniformDouble(0.,2.*pi)(rng);
                    lat.set_link(Link{s,mu},exp(1i*theta));
                }
            }
        }

        void run(int iters)
        {
            for (int i=0; i<iters; i++)
            {
                local_accs.push_back(local_sweep());
                cluster_accs.push_back(gauss_cluster());
            }
            this->iters += iters;
        }


    private:
        Lattice lat;
        mt19937 rng;

        int seed;
        int iters;

        vector<double> energies;
        vector<double> charges;

        vector<double> local_accs;
        vector<double> cluster_accs;

        template <class URNG>
        double gauss_angle(double k)
        {   
            // y1 unif in [0,1], y2 unif in (0,1)
            double y1 = (double)(rng()-rng.min())/(rng.max()-rng.min());
            double y2 = (double)(rng()-rng.min()+1lu)
                / (rng.max()-rng.min()+2lu);

            double r = sqrt(-2./k*log(1.-y1*(1.-exp(-0.5*k*pi*pi))));
            double theta = 2.*pi*(y2-0.5);

            return r*cos(theta);
        }
};

PYBIND11_MODULE(lattice, m)
{
    py::class_<Simulation>(m, "Simulation")
        .def(py::init<double,int,int,int>())
        .def("data", &Simulation::data)
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

