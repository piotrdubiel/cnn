#include "convolution_layer.h"
#include <math.h>
#include <stdlib.h>
#include <iostream>
#include "functions.hpp"


namespace CNN {
    ConvolutionLayer::ConvolutionLayer(unsigned map_count, Size filter_size) {
        for (int i = 0; i < map_count; ++i) {
            Matrix filter(filter_size.getWidth(), filter_size.getHeight());
            for (int a = 0; a < filter.getWidth(); ++a)
                for (int b = 0; b < filter.getHeight(); ++b) {
                    filter(a, b) = static_cast<double>(rand()) / static_cast<double>(RAND_MAX) / 10.0;
                }
            _weights.push_back(filter);
            _biases.push_back(static_cast<double>(rand()) / static_cast<double>(RAND_MAX) / 10.0);
        }
    }

    ConvolutionLayer::ConvolutionLayer(const Matrices& weights, const Vector& biases) {
        _weights = weights;
        _biases = biases;
    }

    Matrices ConvolutionLayer::forward(const Matrices& inputs) {
        if (inputs.empty()) {
            throw std::exception();
        };
        int map_index = 0;
        int connection_index = 0;
        unsigned input_count = inputs.size();
        unsigned map_count = _weights.size() / input_count;
        double bias;
        while (map_index < map_count) {
            Matrices y;
            Matrices::const_iterator input;
            for (input = inputs.begin(); input != inputs.end(); ++input) {
                Matrix& weight = _weights[connection_index + map_index * input_count];
                y.push_back((*input).convolve(weight));
                connection_index++;
            }
            bias = _biases[map_index];

            const Matrix & v = Matrix::sumFrom(y).add(bias);
            std::cout << "Y:" << v << std::endl;
            _output.push_back(v);
            map_index++;
            connection_index = 0;
        }
        return _output;
    }

    Matrices ConvolutionLayer::forward(const Matrix& input) {
        for (int i = 0; i < _weights.size(); ++i) {
            Matrix& weight = _weights[i];
            double bias = _biases[i];
            _output.push_back(input.convolve(weight).add(bias));
        }
        return _output;
    }
}
