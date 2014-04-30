#ifndef SUBSAMPLING_LAYER_H
#define SUBSAMPLING_LAYER_H

#include "layer.h"
#include "matrix.h"
#include "size.h"

namespace CNN {
    class SubsamplingLayer: public Layer<Matrices, Matrices> {
    public:
        SubsamplingLayer(unsigned map_count, Size pool);
        Matrices& forward(Matrices& input);
    private:
        Size _pool_size;
        unsigned _feature_count;
    };
}
#endif // SUBSUMPLING_LAYER_H
