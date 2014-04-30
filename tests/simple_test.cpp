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
    CNN::Matrices weights = CNN::Matrix::matricesFrom("/Users/piotrekd/remote/magisterka/python/test/w.txt");
    CNN::Vector biases = CNN::vectorFrom("/Users/piotrekd/remote/magisterka/python/test/b.txt");
    cout << "Weights" << endl;
    cout << weights[0] << endl << endl;
    cout << weights[1] << endl << endl;
    cout << "Biases" << endl;
    cout << biases << endl << endl;
    CNN::ConvolutionLayer conv1(weights,biases);


//    CNN::SubsamplingLayer sub1(conv1.getMapCount(), CNN::Size(2, 2));

//    CNN::ConvolutionLayer conv2(CNN::Matrix::matricesFrom("/Users/piotrekd/remote/magisterka/net2/l1W.txt"),
//                                CNN::vectorFrom("/Users/piotrekd/remote/magisterka/net/l1b.txt"));
//    CNN::SubsamplingLayer sub2(conv2.getMapCount(), CNN::Size(2, 2));
//    CNN::HiddenLayer hidden(&CNN::Matrix::matrixFrom("/Users/piotrekd/remote/magisterka/net/l2W.txt"),
//                            CNN::vectorFrom("/Users/piotrekd/remote/magisterka/net/l2b.txt"));
//    CNN::SoftmaxLayer soft(&CNN::Matrix::matrixFrom("/Users/piotrekd/remote/magisterka/net/l3W.txt"),
//                           CNN::vectorFrom("/Users/piotrekd/remote/magisterka/net/l3b.txt"));

//    assert(conv1.getMapCount()==20);
//    assert(conv2.getMapCount()==50 * conv1.getMapCount());
//    assert(hidden.getInputSize()==50*4*4);
//    assert(hidden.getOutputSize()==500);

    CNN::Matrices test = CNN::Matrix::matricesFrom("/Users/piotrekd/remote/magisterka/python/test/in.txt");
    cout << "In 1:" <<endl;
    cout << test[0] << endl;
    cout << "In 2:" <<endl;
    cout << test[1] << endl;

    CNN::Matrices l1 = conv1.forward(test);

    cout << "Out 1:" <<endl;
    cout << l1[0] << endl;
    cout << "Out 2:" <<endl;
    cout << l1[1] << endl;

    return 0;

//    CNN::Matrices& l1 = sub1.forward(conv1.forward(test));
//    cout << *conv2.forward(l1)[20] << endl;
//    return 0;
//    CNN::Matrices& l2 = sub2.forward(conv2.forward(l1));
//    CNN::Vector* v = CNN::Matrix::flatten(l2);
//    assert(v->size()==800);
//    cout << *v << endl;
//    CNN::Vector l3 = hidden.forward(*v);
//    CNN::Vector l4 = soft.forward(l3);
//    cout << l4 << endl;
}

