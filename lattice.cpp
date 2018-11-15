#include <boost/python.hpp>

#include "lattice.hpp"

BOOST_PYTHON_MODULE(lattice)
{
    using namespace boost::python;
    def("lattice", lattice);
}
