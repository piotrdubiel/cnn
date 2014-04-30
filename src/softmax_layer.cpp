#include "softmax_layer.h"
#include <math.h>

namespace CNN {
    SoftmaxLayer::SoftmaxLayer(unsigned inputs, unsigned outputs) : super(inputs, outputs) {}
    SoftmaxLayer::SoftmaxLayer(Matrix* weights, Vector* biases) : super(weights, biases) {}

    Vector SoftmaxLayer::forward(Vector input)
    {
        Vector output = super::forward(input);
        Vector::iterator it;
        double sum = 0.0;
        for (it=output.begin(); it!=output.end(); ++it) {
            sum += exp(*it);
        }

        for (it=output.begin(); it!=output.end(); ++it) {
            *it = exp(*it) / sum;
        }
        _last_output = output;
        return output;
    }

    unsigned SoftmaxLayer::prediction() const
    {
        double max = -MAXFLOAT;
        unsigned index = -1;
        for (unsigned i=0; i<_last_output.size(); ++i) {
            if (_last_output[i] > max) {
                max = _last_output[i];
                index = i;
            }
        }
        return index;
    }
}
