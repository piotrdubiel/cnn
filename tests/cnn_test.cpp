#include <iostream>
#include <fstream>
#include "convolution_layer.h"
#include "subsampling_layer.h"
#include "hidden_layer.h"
#include "softmax_layer.h"
#include "matrix.h"
#include "x.hpp"
#include "size.h"
#include <assert.h>

using namespace std;

int main() {
//    CNN::ConvolutionLayer conv1(CNN::Matrix::matricesFrom("/Users/piotrekd/remote/magisterka/net/l0W.txt"),
//                                CNN::vectorFrom("/Users/piotrekd/remote/magisterka/net/l0b.txt"));
//    CNN::SubsamplingLayer sub1(conv1.getMapCount(), CNN::Size(2, 2));

//    CNN::ConvolutionLayer conv2(CNN::Matrix::matricesFrom("/Users/piotrekd/remote/magisterka/net2/l1W.txt"),
//                                CNN::vectorFrom("/Users/piotrekd/remote/magisterka/net/l1b.txt"));
//    CNN::SubsamplingLayer sub2(conv2.getMapCount(), CNN::Size(2, 2));
//    CNN::HiddenLayer hidden(&CNN::Matrix::matrixFrom("/Users/piotrekd/remote/magisterka/net/l2W.txt"),
//                            CNN::vectorFrom("/Users/piotrekd/remote/magisterka/net/l2b.txt"));
//    CNN::SoftmaxLayer soft(&CNN::Matrix::matrixFrom("/Users/piotrekd/remote/magisterka/net/l3W.txt"),
//                           CNN::vectorFrom("/Users/piotrekd/remote/magisterka/net/l3b.txt"));

    CNN::ConvolutionLayer conv1(CNN::Matrix::matricesFrom("/Users/piotrekd/remote/magisterka/python/net/l0W.txt"),
            CNN::vectorFrom("/Users/piotrekd/remote/magisterka/python/net/l0b.txt"));
    CNN::SubsamplingLayer sub1(conv1.getMapCount(), CNN::Size(2, 2));

//    CNN::ConvolutionLayer conv2(CNN::Matrix::matricesFrom("/Users/piotrekd/remote/magisterka/python/net/l1W.txt"),
//                                CNN::vectorFrom("/Users/piotrekd/remote/magisterka/python/net/l1b.txt"));
//    CNN::SubsamplingLayer sub2(conv2.getMapCount(), CNN::Size(2, 2));
    CNN::Matrix l2W = CNN::Matrix::matrixFrom("/Users/piotrekd/remote/magisterka/python/net/l2W.txt");
    CNN::Vector l2b = CNN::vectorFrom("/Users/piotrekd/remote/magisterka/python/net/l2b.txt");
    CNN::HiddenLayer hidden(&l2W, &l2b);
    CNN::Matrix l3W = CNN::Matrix::matrixFrom("/Users/piotrekd/remote/magisterka/python/net/l3W.txt");
    CNN::Vector l3b = CNN::vectorFrom("/Users/piotrekd/remote/magisterka/python/net/l3b.txt");
    CNN::SoftmaxLayer soft(&l3W, &l3b);

    assert(conv1.getMapCount() == 20);
    //assert(conv2.getMapCount()==50 * conv1.getMapCount());
    assert(hidden.getInputSize() == 20 * 12 * 12);
    assert(hidden.getOutputSize() == 500);

    CNN::Matrix test = CNN::Matrix::matrixFrom("/Users/piotrekd/remote/magisterka/python/val2/0.txt");
    CNN::Matrices input = conv1.forward(test);
    CNN::Matrices l1 = sub1.forward(input);
    CNN::Vector v = CNN::Matrix::flatten(l1);
    //assert(v->size()==800);
    cout << v << endl;
    CNN::Vector l3 = hidden.forward(v);
    CNN::Vector l4 = soft.forward(l3);

    //cout << l4 << endl;
    //cout << soft.prediction() << endl;
}
