#include "subsampling_layer.h"

#include <limits>
#include <math.h>

namespace CNN {
    SubsamplingLayer::SubsamplingLayer(unsigned feature_count, Size pool) :
        _pool_size(pool),
        _feature_count(feature_count) {}

    Matrices& SubsamplingLayer::forward(Matrices &input)
    {
        Matrices::const_iterator it;
        for (it=input.begin(); it!=input.end(); ++it) {
            Matrix subsampled((*it).getWidth()/2, (*it).getHeight()/2);
            for (int i=0; i<subsampled.getWidth(); ++i)
            for (int j=0; j<subsampled.getHeight(); ++j) {
               double max = std::numeric_limits<double>::lowest();
               for (int a=0; a<_pool_size.getWidth(); ++a)
               for (int b=0; b<_pool_size.getHeight(); ++b) {
                   const double& val = (*it)(a+i*_pool_size.getWidth(), b+j*_pool_size.getHeight());
                   if (val > max) {
                       max = val;
                   }
               }
               subsampled(i, j) = tanh(max);
            }
            //std::cout << *subsampled << std::endl << std::endl;
            _output.push_back(subsampled);
        }
        return _output;
    }
}
