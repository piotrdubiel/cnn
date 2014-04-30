#ifndef HIDDEN_LAYER_H
#define HIDDEN_LAYER_H

#include "layer.h"
#include "matrix.h"

namespace CNN {
    class HiddenLayer : public Layer<Vector, Vector> {
    public:
        HiddenLayer(unsigned inputs, unsigned outputs);
        HiddenLayer(Matrix* _weights, Vector* _biases);
        ~HiddenLayer();
        Vector forward(Vector& input);
        unsigned getInputSize() const { return _weights->getHeight(); }
        unsigned getOutputSize() const { return _weights->getWidth(); }
    private:
        Matrix* _weights;
        Vector* _biases;
    };
}

#endif // HIDDEN_LAYER_H
