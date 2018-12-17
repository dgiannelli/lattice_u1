#pragma once

#include <string>
#include <sstream>

#include "lattice.hpp"

class Simulation
{
    public:
        Simulation(double beta, int L1, int L2, int seed) :
            lat(beta,L1,L2),
            rng(seed),
            seed{seed}
        {}

        py::dict data() const;
        void set_state(const py::dict &data);

        void set_hot();
        void run(int iters)

    private:
        Lattice lat;
        mt19937 rng;

        int seed;
        int iters;

        vector<double> energies;
        vector<double> charges;

        vector<double> local_accs;
        vector<double> cluster_accs;

        double gauss_angle(double k) const;

        // Return 1.0 if move is accepted, 0.0 if not
        double local_update(Link link);

        // Return acceptance ratio
        double local_sweep();

		// Return 1.0 if move is accepted, 0.0 if not
		double gauss_cluster(int L1_cluster, int L2_cluster);
};

inline py::dict Simulation::data() const
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

inline void Simulation::set_state(const py::dict &data)
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

inline void Simulation::set_hot()
{
    for (int mu : {1,2}) {
        for (Site s : lat.sites()) {
            double theta = UniformDouble(0.,2.*pi)(rng);
            lat.set_link(Link{s,mu},exp(1i*theta));
        }
    }
}

inline void Simulation::run(int iters)
{
    for (int i=0; i<iters; i++)
    {
        local_accs.push_back(local_sweep());
        cluster_accs.push_back(gauss_cluster());
    }
    this->iters += iters;
}

inline double Simulation::gauss_angle(double k) const
{   
    // y1 unif in [0,1], y2 unif in (0,1)
    double y1 = (double)(rng()-rng.min())/(rng.max()-rng.min());
    double y2 = (double)(rng()-rng.min()+1lu)
        / (rng.max()-rng.min()+2lu);

    double r = sqrt(-2./k*log(1.-y1*(1.-exp(-0.5*k*pi*pi))));
    double theta = 2.*pi*(y2-0.5);

    return r*cos(theta);
}

inline double Simulation::local_update(Link link)
{
    auto [S_1,S_2] = conn_staples(link);
    
    cmplx W = lat.s_line(S_1) + lat.s_line(S_2);
    double k = lat.beta()*abs(W);
    double x_old = arg(W*lat.s_line(link));
    
    double x_new = gauss_angle(k,rng);
    
    double p = exp(k*(cos(x_new)+pow(x_new,2)/2.
                     -cos(x_old)-pow(x_old,2)/2.));
    
    if (UniformDouble()(rng)<p) {
        cmplx u_new = exp(1i*(x_new-arg(W)));
        lat.set_link(link,u_new);
        return 1.;
    }
    else return 0.;
}

inline double Simulation::local_sweep()
{
    double accept = 0.;
    for (int mu : {1,2}) {
        for (Site s : lat.sites()) {
            accept += local_update(lat,Link{s,mu},rng);
        }
    }
    return accept/lat.L1()/lat.L2()/2.;
}

class Cluster
{   
    public:
        template <class URNG>
        Cluster(int N, int side, URNG&);

        Link gate() const {return gate_;};
        const vector<Link> &path() const {return path_;};
        const Staple &estaple() const {return estaple_;};
        const Staple &istaple() const {return istaple_;};
        vector<Link> links() const;
    private:
        int L1_cluster;
        int L2_cluster
        Site corner;

        Link gate_;
        vector<Link> path_;
        Staple estaple_, istaple_;
};

template <class URNG>
inline Cluster::Cluster(int L1_cluster, int L2_cluster,
                        Site corner, int offset) :
    L1_cluster{L1_cluster},
    L2_cluster{L2_cluster},
    corner{corner}
{

    // Build path
    int nu;
    switch (mu) {
        case 1: nu = 2; break;
        case 2: nu = 1; break;
        default: throw runtime_error("Invalid mu");
    }

    Site s = corner;
    for (int rho : {mu,nu,-mu,-nu}) {
        for (int i=0; i<side; i++) {
            path_.push_back(Link{s,rho});
            s = s + hat(rho);
        }
    }

    // Rotate path and extract gate
    rotate(path_.begin(), path_.begin()+offset, path_.end());
    gate_ = path_[0]; path_.erase(path_.begin());
    
    // Identify staples
    int mu_e; // External direction
    switch (offset/side) {
        case 0: mu_e = -nu; break;
        case 1: mu_e = mu; break;
        case 2: mu_e = nu; break;
        case 3: mu_e = -mu; break;
        default: throw runtime_error("invalid mu");
    }
    estaple_ = conn_staple(gate_,mu_e);
    istaple_ = conn_staple(gate_,-mu_e);
}

/* Select internal links.
   This function is lazily evaluated
   only if the inversion move is accepted */
inline vector<Link> Cluster::links() const
{
    vector<Link> vec;

    int x1, x2;
    x2 = corner.x2+1;
    for (; x2<corner.x2+side; x2++) {
        x1 = corner.x1;
        for (; x1<corner.x1+side; x1++) {
            vec.push_back(Link{Site{x1,x2},1});
        }
    }
    x2 = corner.x2;
    for (; x2<corner.x2+side; x2++) {
        x1 = corner.x1+1;
        for (; x1<corner.x1+side; x1++) {
            vec.push_back(Link{Site{x1,x2},2});
        }
    }
    return vec;
}

// Return 1.0 if move is accepted, 0.0 if not
double gauss_cluster(int L1_cluster, int L2_cluster)
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
    
    double x_new = gauss_angle(k_new,rng);
    
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

