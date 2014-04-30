#ifndef SOFTMAX_LAYER_H
#define SOFTMAX_LAYER_H

#include "hidden_layer.h"
#include "matrix.h"

namespace CNN {
    class SoftmaxLayer : public HiddenLayer {
    public:
        SoftmaxLayer(unsigned inputs, unsigned outputs);
        SoftmaxLayer(Matrix* weights, Vector* biases);
        Vector forward(Vector input);
        unsigned prediction() const;
    private:
        typedef HiddenLayer super;
        Vector _last_output;
    };
}

#endif // SOFTMAX_LAYER_H
