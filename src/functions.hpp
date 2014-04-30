#ifndef FUNCTIONS_HPP_
#define FUNCTIONS_HPP_

#include <math.h>

double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

#endif
