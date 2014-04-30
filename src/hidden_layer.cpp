#include "hidden_layer.h"
#include <assert.h>
#include <math.h>

namespace CNN {
    HiddenLayer::HiddenLayer(unsigned inputs, unsigned outputs)
    {
        _weights = new Matrix(inputs, outputs);
        for (int b=0; b<outputs; ++b) {
            for (int a=0; a<inputs; ++a) {
                (*_weights)(a,b) = static_cast<double>(rand()) / static_cast<double>(RAND_MAX) / 10.0;
            }
            _biases->push_back(static_cast<double>(rand()) / static_cast<double>(RAND_MAX) / 10.0);
        }
    }

    HiddenLayer::HiddenLayer(Matrix* weights, Vector* biases) : _weights(weights), _biases(biases) {}

    HiddenLayer::~HiddenLayer()
    {
        delete _weights;
    }

    Vector HiddenLayer::forward(Vector& input)
    {
        assert(input.size() == _weights->getHeight());
        Vector output;
        for (int i=0; i<_weights->getWidth(); ++i) {
            double sum = 0.0;
            for (int j=0; j<_weights->getHeight(); ++j) {
                sum += (*_weights)(i,j) * input[j];
            }
            output.push_back(tanh(sum + (*_biases)[i]));
        }
        return output;
    }

}
