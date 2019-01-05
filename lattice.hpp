#pragma once

//#include <iostream>
#include <vector>
#include <array>
#include <complex>
#include <random>
#include <numeric>
#include <algorithm>

using namespace std;
using cmplx = complex<double>;
using UniformDouble = uniform_real_distribution<double>;
using UniformInt = uniform_int_distribution<int>;
const double pi = acos(-1.);

struct Site
{
    int x1;
    int x2;

    Site operator+(Site s) {
        return Site{x1+s.x1,x2+s.x2};
    }
};

Site hat(int mu)
{
    switch (mu) {
        case 1:; return Site{1,0};
        case -1: return Site{-1,0};
        case 2: return Site{0,1};
        case -2: return Site{0,-1};
        default: throw runtime_error("invalid mu");
    }
}

struct Link
{
    Site s;
    int mu;
};

using Staple = array<Link,3>;

// nu is the first staple link direction
Staple conn_staple(Link link, int nu)
{
    auto [s,mu] = link;
    return Staple{Link{s+hat(mu),nu},
                  Link{s+hat(mu)+hat(nu),-mu},
                  Link{s+hat(nu),-nu}};
}

// The order of returned staples is not considered
array<Staple,2> conn_staples(Link link)
{
    int nu;
    switch (link.mu) {
        case 1:; case -1: nu = 2; break;
        case 2:; case -2: nu = 1; break;
        default: throw runtime_error("Invalid mu");
    }
    return array<Staple,2>{conn_staple(link,nu),
                           conn_staple(link,-nu)};
}

using Plaq = array<Link,4>;

Plaq plaq_12(Site s)
{
    return Plaq{Link{s,1},
                Link{s+hat(1),2},
                Link{s+Site{1,1},-1},
                Link{s+hat(2),-2}};
}

using Staple = array<Link,3>;

class Lattice
{
    public:
        /* Lattice is initialized with its square side size N
           and the value of the action parameter beta */
        Lattice(double beta, int L1, int L2);
        double beta() const {return beta_;}
        int L1() const {return L1_;}
        int L2() const {return L2_;}
        const vector<double> &config() const {return link_vars;}
        void config(const vector<double> &config) {link_vars=config;}
        
        /* Vector of all lattice sites.
           Useful for iterating over them: */
        const vector<Site> &sites() const {return sites_;}
        
        // Observables: 
        double energy() const;
        double charge(Site) const;
        double total_charge() const;
        
        /* All the following methods are aware of
           boundary conditions and treat appropriately
           values out of bounds */
        
        // Compute Schwinger line over a generic Path:
        template <class Path>
        cmplx s_line(Path) const;
        
        // Set link variable link to value u:
        void set_link(Link link, cmplx u);
        
        // Local gauge transformation G(s):
        void local_gauge(Site s, cmplx G);
        
    private:
        double beta_;
        int L1_;
        int L2_;
        vector<Site> sites_;
        vector<double> link_vars; // Link variables data
        
        /* link_vars indexing:
           (boundaries are implemented here) */
        int idx(Link) const;
};

Lattice::Lattice(double beta, int L1, int L2) :
    beta_{beta}, L1_{L1}, L2_{L2}, link_vars(2*L1*L2,0.)
{
    for (int x2=0; x2<L2; x2++) {
        for (int x1=0; x1<L1; x1++) {
            sites_.push_back(Site{x1,x2});
        }
    }
}

double Lattice::energy() const
{
    double sum = 0.;
    for (Site s : sites())
    {
        sum += 1.-real(s_line(plaq_12(s)));
    }
    return beta()*sum/L1()/L2();
}

double Lattice::charge(Site s) const
{
    return arg(s_line(plaq_12(s)))/2./pi;
}

double Lattice::total_charge() const
{
    double sum = 0.;
    for (Site s : sites())
    {
        sum += charge(s);
    }
    return sum;
}

template <>
cmplx Lattice::s_line(Link link) const
{
    cmplx u = exp(1i*link_vars[idx(link)]);
    if (link.mu>0) return u;
    else return conj(u);
}

template <class Path>
cmplx Lattice::s_line(Path path) const
{
    cmplx prod = 1.;
    for (auto link : path)
    {
        prod *= s_line(link);
    }
    return prod;
}

void Lattice::set_link(Link link, cmplx u)
{
    if (link.mu < 0) u = conj(u);
    link_vars[idx(link)] = arg(u);
}

void Lattice::local_gauge(Site s, cmplx G)
{
    for (int mu : {1,2,-1,-2}) {
        Link link = Link{s,mu};
        set_link(link, G*s_line(link));
    }
}


/* Only link variables pointing to positive directions
   are stored, i.e. top and right links.
   Links in the opposite directions
   are computed in terms of them taking the complex
   conjugate inside s_line and set_link methods. */


/* Apply periodic boundary conditions and return
   the corresponding index of link_vars.
   If link points to a negative direction,
   the index of the opposite link is returned.
   Links with contiguous x1 are contiguous in memory.
   The successive axis is x2.
   Links pointing to direction x1 are first in the ordering,
   then the ones pointing to x2. */
int Lattice::idx(Link link) const
{
    auto [x1,x2] = link.s;
    
    /* axis 0: points to x1
       axis 1: points to x2 */
    int axis = 0;
    /* Select right axis.
       If link points to negative directions,
       select the opposite one */
    switch (link.mu) {
        case 1: break;
        case -1: x1--; break;
        case 2: axis = 1; break;
        case -2: x2--; axis = 1; break;
        default: throw runtime_error("Invalid mu");
    }

    // Apply periodic boundary to coord x:
    x1 = (x1%L1()+L1()) % L1();
    x2 = (x2%L2()+L2()) % L2();
    
    return x1 + x2*L1() + axis*L1()*L2();
}

