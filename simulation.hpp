#pragma once

#include <iostream>
#include <string>
#include <sstream>
#include <vector>
#include <tuple>
#include <algorithm>

#include <experimental/filesystem>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <highfive/H5File.hpp>
#include <highfive/H5DataSet.hpp>
#include <highfive/H5DataSpace.hpp>
#include <highfive/H5Attribute.hpp>

#include "lattice.hpp"

using namespace std;
using namespace std::experimental::filesystem;
namespace py = pybind11;
namespace H5 = HighFive;

class Simulation
{
    public:
        Simulation(double beta, int L1, int L2, int seed) :
            lat(beta,L1,L2),
            rng(seed)
        {
            create_directory("data");

            stringstream filename_stream;
            filename_stream.precision(3);
            filename_stream << "data/" << beta << '_' << L1 << '_' 
                            << L2 << '_' << seed << ".h5";
            filename = filename_stream.str();

            if (is_regular_file(filename))
            {
                cout << "Restoring from file " << filename << endl;
                auto file = H5::File(filename, H5::File::ReadWrite);

                auto config = lat.config();
                file.getAttribute("config").read(config);
                lat.config(config);

                string rng_state;
                file.getAttribute("rng_state").read(rng_state);
                stringstream(rng_state) >> rng;

                file.getAttribute("iters").read(iters);

                file.getDataSet("energies").read(energies);
                file.getAttribute("energy_mean").read(energy_mean);
                file.getAttribute("energy_err").read(energy_err);

                file.getDataSet("charges").read(charges);
                file.getAttribute("susc_mean").read(susc_mean);
                file.getAttribute("susc_err").read(susc_err);

                file.getDataSet("local_accs").read(local_accs);
                file.getAttribute("local_acc_mean").read(local_acc_mean);
                file.getAttribute("local_acc_err").read(local_acc_err);

                file.getDataSet("cluster_accs").read(cluster_accs);
                file.getAttribute("cluster_acc_mean").read(cluster_acc_mean);
                file.getAttribute("cluster_acc_err").read(cluster_acc_err);
            }

            else
            {
                cout << "Creating new file " << filename << endl;
                auto file = H5::File(filename, H5::File::Create);
                file.createAttribute<double>("beta",H5::DataSpace(1)).write(beta);
                file.createAttribute<int>("L1",H5::DataSpace(1)).write(L1);
                file.createAttribute<int>("L2",H5::DataSpace(1)).write(L2);
                file.createAttribute<int>("seed",H5::DataSpace(1)).write(seed);

                set_hot();

                auto config = lat.config();
                file.createAttribute<int>("config",H5::DataSpace::From(config)).write(config);

                stringstream rng_state_stream;
                rng_state_stream << rng;
                string rng_state = rng_state_stream.str();
                file.createAttribute<std::string>("rng_state",H5::DataSpace::From(rng_state))
                     .write(rng_state);

                file.createAttribute<int>("iters",H5::DataSpace(1)).write(0);

                auto dataspace = H5::DataSpace({0},{H5::DataSpace::UNLIMITED});
                H5::DataSetCreateProps props;
                props.add(H5::Chunking({int(1e7)}));

                file.createDataSet<double>("energies",dataspace,props);
                file.createAttribute<double>("energy_mean",H5::DataSpace(1)).write(0.);
                file.createAttribute<double>("energy_err",H5::DataSpace(1)).write(0.);

                file.createDataSet<double>("charges",dataspace,props);
                file.createAttribute<double>("susc_mean",H5::DataSpace(1)).write(0.);
                file.createAttribute<double>("susc_err",H5::DataSpace(1)).write(0.);

                file.createDataSet<double>("local_accs",dataspace,props);
                file.createAttribute<double>("local_acc_mean",H5::DataSpace(1)).write(0.);
                file.createAttribute<double>("local_acc_err",H5::DataSpace(1)).write(0.);

                file.createDataSet<double>("cluster_accs",dataspace,props);
                file.createAttribute<double>("cluster_acc_mean",H5::DataSpace(1)).write(0.);
                file.createAttribute<double>("cluster_acc_err",H5::DataSpace(1)).write(0.);
            }
        }

        void run(int iters);
        void analysis();
        void write_state();

    private:
        Lattice lat;
        mt19937 rng;

        string filename;

        int iters;

        vector<double> energies;
        double energy_mean, energy_err;

        vector<double> charges;
        double susc_mean, susc_err;

        vector<double> local_accs;
        double local_acc_mean, local_acc_err;

        vector<double> cluster_accs;
        double cluster_acc_mean, cluster_acc_err;

        void set_hot();

        double gauss_angle(double k);

        // Return 1.0 if move is accepted, 0.0 if not
        double local_update(Link link);

        // Return acceptance ratio
        double local_sweep();

		// Return 1.0 if move is accepted, 0.0 if not
		double gauss_cluster(int L1_cluster, int L2_cluster);

};

void Simulation::run(int iters_new)
{
    int diff = iters_new - iters;
    if (diff > 0)
    {
        for (int i=0; i<diff; i++)
        {
            local_accs.push_back(local_sweep());
            cluster_accs.push_back(gauss_cluster((lat.L1()+1)/2,(lat.L2()+1)/2));
            energies.push_back(lat.energy());
            charges.push_back(lat.total_charge());
        }
        iters = iters_new;
    }
}

void Simulation::analysis()
{
    if (iters<1000) throw runtime_error("Too few iters");
    py::object resampling = py::module::import("resampling");
    py::object binning = resampling.attr("binning");
    //py::object tau_jack = resampling.attr("tau_jack");

    tie(energy_mean,energy_err) = py::cast<tuple<double,double>>(binning(energies));

    vector<double> suscs(charges.size());
    transform(charges.begin(),charges.end(),suscs.begin(),
              [&](double q){return q*q/lat.L1()/lat.L2()*lat.beta();});

    tie(susc_mean,susc_err) = py::cast<tuple<double,double>>(binning(charges));

    tie(local_acc_mean,local_acc_err) = py::cast<tuple<double,double>>(binning(local_accs));
    tie(cluster_acc_mean,cluster_acc_err) = py::cast<tuple<double,double>>(binning(local_accs));
}

void Simulation::write_state()
{
    auto file = H5::File(filename, H5::File::ReadWrite);

    auto config = lat.config();
    file.getAttribute("config").write(config);

    stringstream rng_state_stream;
    rng_state_stream << rng;
    file.getAttribute("rng_state").write(rng_state_stream.str());

    file.getAttribute("iters").write(iters);

    auto dataset = file.getDataSet("energies");
    dataset.resize({iters}); dataset.write(energies);
    file.getAttribute("energy_mean").write(energy_mean);
    file.getAttribute("energy_err").write(energy_err);

    dataset = file.getDataSet("charges");
    dataset.resize({iters}); dataset.write(charges);
    file.getAttribute("susc_mean").write(susc_mean);
    file.getAttribute("susc_err").write(susc_err);

    dataset = file.getDataSet("local_accs");
    dataset.resize({iters}); dataset.write(local_accs);
    file.getAttribute("local_acc_mean").write(local_acc_mean);
    file.getAttribute("local_acc_err").write(local_acc_err);

    dataset = file.getDataSet("cluster_accs");
    dataset.resize({iters}); dataset.write(cluster_accs);
    file.getAttribute("cluster_acc_mean").write(cluster_acc_mean);
    file.getAttribute("cluster_acc_err").write(cluster_acc_err);
}

void Simulation::set_hot()
{
    for (int mu : {1,2}) {
        for (Site s : lat.sites()) {
            double theta = UniformDouble(0.,2.*pi)(rng);
            lat.set_link(Link{s,mu},exp(1i*theta));
        }
    }
}

double Simulation::gauss_angle(double k)
{   
    // y1 unif in [0,1], y2 unif in (0,1)
    double y1 = (double)(rng()-rng.min())/(rng.max()-rng.min());
    double y2 = (double)(rng()-rng.min()+1lu)
        / (rng.max()-rng.min()+2lu);

    double r = sqrt(-2./k*log(1.-y1*(1.-exp(-0.5*k*pi*pi))));
    double theta = 2.*pi*(y2-0.5);

    return r*cos(theta);
}

double Simulation::local_update(Link link)
{
    auto [S_1,S_2] = conn_staples(link);
    
    cmplx W = lat.s_line(S_1) + lat.s_line(S_2);
    double k = lat.beta()*abs(W);
    double x_old = arg(W*lat.s_line(link));
    
    double x_new = gauss_angle(k);
    
    double p = exp(k*(cos(x_new)+pow(x_new,2)/2.
                     -cos(x_old)-pow(x_old,2)/2.));
    
    if (UniformDouble()(rng)<p) {
        cmplx u_new = exp(1i*(x_new-arg(W)));
        lat.set_link(link,u_new);
        return 1.;
    }
    else return 0.;
}

double Simulation::local_sweep()
{
    double accept = 0.;
    for (int mu : {1,2}) {
        for (Site s : lat.sites()) {
            accept += local_update(Link{s,mu});
        }
    }
    return accept/lat.L1()/lat.L2()/2.;
}

class Cluster
{   
    public:
        Cluster(int L1_cluster, int L2_cluster,
                Site corner, int mu, int offset);

        Link gate() const {return gate_;};
        const vector<Link> &path() const {return path_;};
        const Staple &estaple() const {return estaple_;};
        const Staple &istaple() const {return istaple_;};
        vector<Link> links() const;
    private:
        int L1_cluster;
        int L2_cluster;
        Site corner;

        Link gate_;
        vector<Link> path_;
        Staple estaple_, istaple_;
};

Cluster::Cluster(int L1_cluster, int L2_cluster,
                        Site corner, int mu, int offset) :
    L1_cluster{L1_cluster},
    L2_cluster{L2_cluster},
    corner{corner}
{

    // Build path
    int nu;
    int L_mu, L_nu;
    switch (mu) {
        case 1:
            nu = 2;
            L_mu = L1_cluster;
            L_nu = L2_cluster;
            break;
        case 2:
            nu = 1;
            L_mu = L2_cluster;
            L_nu = L1_cluster;
            break;
        default: throw runtime_error("Invalid mu");
    }

    Site s = corner;
    array<int,4> dirs = {mu,nu,-mu,-nu};
    array<int,4> sides = {L_mu,L_nu,L_mu,L_nu};
    for (int i=0; i<4; i++) {
        for (int j=0; j<sides[i]; j++) {
            path_.push_back(Link{s,dirs[i]});
            s = s + hat(dirs[i]);
        }
    }

    // Rotate path and extract gate
    rotate(path_.begin(), path_.begin()+offset, path_.end());
    gate_ = path_[0]; path_.erase(path_.begin());
    
    // Identify staples
    int mu_e; // External direction
    if (offset<L_mu) mu_e = -nu;
    else if (offset<L_mu+L_nu) mu_e = mu;
    else if (offset<2*L_mu+L_nu) mu_e = nu;
    else if (offset<2*(L_mu+L_nu)) mu_e = -mu;
    else throw runtime_error("Invalid offset");

    estaple_ = conn_staple(gate_,mu_e);
    istaple_ = conn_staple(gate_,-mu_e);
}

/* Select internal links.
   This function is lazily evaluated
   only if the inversion move is accepted */
vector<Link> Cluster::links() const
{
    vector<Link> vec;

    int x1, x2;
    x2 = corner.x2+1;
    for (; x2<corner.x2+L2_cluster; x2++) {
        x1 = corner.x1;
        for (; x1<corner.x1+L1_cluster; x1++) {
            vec.push_back(Link{Site{x1,x2},1});
        }
    }
    x2 = corner.x2;
    for (; x2<corner.x2+L2_cluster; x2++) {
        x1 = corner.x1+1;
        for (; x1<corner.x1+L1_cluster; x1++) {
            vec.push_back(Link{Site{x1,x2},2});
        }
    }
    return vec;
}

// Return 1.0 if move is accepted, 0.0 if not
double Simulation::gauss_cluster(int L1_cluster, int L2_cluster)
{
    /* Select random cluster corner,
       starting path direction
       and the path position of the gate */

    int L1 = lat.L1();
    int L2 = lat.L2();
    int ran_pos = UniformInt(0,L1*L2-1)(rng);
    int x1 = ran_pos/L2; // unif in [0,L1-1]
    int x2 = ran_pos%L2; // unif in [0,L2-1]
    Site corner{x1,x2}; // lower left corner

    int ran_muoff = UniformInt(0,2*(L1_cluster+L2_cluster)-1)(rng);
    int mu = ran_muoff%2 + 1; // unif in [1,2]
    int offset = ran_muoff/2; // unif in [0,2*(L1_cluster+L2_cluster)-1]

    Cluster cluster(L1_cluster,L2_cluster,corner,mu,offset);

    auto path = cluster.path();
    auto link_it = path.rbegin();
    for (; link_it<path.rend(); link_it++) {
        Link link = *link_it;
        cmplx u = lat.s_line(link);
        lat.local_gauge(link.s, conj(u));
    }
    
    cmplx US_e = lat.s_line(cluster.estaple());
    cmplx US_i = lat.s_line(cluster.istaple());
    cmplx W_new = US_e + conj(US_i);
    cmplx W_old = US_e + US_i;
    
    double k_old = lat.beta()*abs(W_old);
    double k_new = lat.beta()*abs(W_new);

    cmplx u_old = lat.s_line(cluster.gate());
    double x_old = arg(W_old*u_old);
    
    double x_new = gauss_angle(k_new);
    
    double p = exp(k_new*(cos(x_new)+pow(x_new,2.)/2.)
                  -k_old*(cos(x_old)+pow(x_old,2.)/2.))
              *erf(pi*sqrt(k_new/2.))/erf(pi*sqrt(k_old/2.))
              *sqrt(k_old/k_new);
    
    if (UniformDouble()(rng)<p) {
        cmplx u_new = exp(1i*(x_new-arg(W_new)));
        lat.set_link(cluster.gate(),u_new);
        for (Link link : cluster.links()) {
            cmplx u = lat.s_line(link);
            lat.set_link(link,conj(u));
        }
        return 1.;
    }
    else return 0.;
}

