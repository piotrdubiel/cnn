#ifndef CNNCONVOLUTIONLAYER_H
#define CNNCONVOLUTIONLAYER_H

#include "layer.h"
#include "matrix.h"
#include "size.h"

namespace CNN {
    class ConvolutionLayer : public Layer<Matrices, Matrices>
    {
    public:
        ConvolutionLayer(unsigned map_count, Size filter);
        ConvolutionLayer(const Matrices& weights, const Vector& biases);
        Matrices forward(const Matrices& input);
        Matrices forward(const Matrix& input);
        unsigned getMapCount() const { return _weights.size(); }


    private:
        Matrices _weights;
        Vector _biases;
    };
}

#endif // CNNCONVOLUTIONLAYER_H
