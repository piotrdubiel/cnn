#include <iostream>
#include "convolution_layer.h"
#include "subsampling_layer.h"
#include "hidden_layer.h"
#include "softmax_layer.h"

using namespace std;

int main() {
    CNN::Matrices weights = CNN::Matrix::matricesFrom("/Users/piotrekd/remote/magisterka/python/test2/w1a.txt");
    CNN::Vector biases = CNN::vectorFrom("/Users/piotrekd/remote/magisterka/python/test2/b.txt");
    cout << "Weights" << endl;
    cout << weights[0] << endl << endl;
    cout << weights[1] << endl << endl;
    cout << weights[2] << endl << endl;
    cout << weights[3] << endl << endl;
    cout << "Biases" << endl;
    cout << biases << endl << endl;
    CNN::Matrices weights0 = CNN::Matrix::matricesFrom("/Users/piotrekd/remote/magisterka/python/test2/w0a.txt");
    CNN::ConvolutionLayer conv1(weights0,biases);
    CNN::SubsamplingLayer sub1(conv1.getMapCount(), CNN::Size(2,2));
    CNN::ConvolutionLayer conv2(CNN::Matrix::matricesFrom("/Users/piotrekd/remote/magisterka/python/test2/w1a.txt"),biases);

    CNN::Matrices test = CNN::Matrix::matricesFrom("/Users/piotrekd/remote/magisterka/python/test2/in.txt");
    cout << "In 1:" <<endl;
    cout << test[0] << endl;

    CNN::Matrices l1 = conv1.forward(test);
    CNN::Matrices s1 = sub1.forward(l1);
    CNN::Matrices l2 = conv2.forward(s1);

    cout << "Out 1:" <<endl;
    cout << l1[0] << endl;
    cout << l1[1] << endl;

    cout << "Sub 1:" <<endl;
    cout << s1[0] << endl;
    cout << s1[1] << endl;

    cout << "Out 2:" <<endl;
    cout << l2[0] << endl;
    cout << l2[1] << endl;

    return 0;

    //    CNN::Matrices& l1 = sub1.forward(conv1.forward(test));
    //    cout << *conv2.forward(l1)[20] << endl;
    //    return 0;
    //    CNN::Matrices& l2 = sub2.forward(conv2.forward(l1));
    //    CNN::Vector* v = CNN::Matrix::flatten(l2);
    //    assert(v->size()==800);
    ////    cout << *v << endl;
    //    CNN::Vector l3 = hidden.forward(*v);
    //    CNN::Vector l4 = soft.forward(l3);
    //    cout << l4 << endl;
    }


