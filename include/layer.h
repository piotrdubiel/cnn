#ifndef CNNLAYER_H
#define CNNLAYER_H

namespace CNN {
    template<class In, class Out>
    class Layer {
    public:
        Layer() {};
        const Out& getOutput() const { return _output; }

    protected:
        Out _output;
    };
}

#endif // CNNLAYER_H
