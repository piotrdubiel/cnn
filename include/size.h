#ifndef SIZE_H
#define SIZE_H

namespace CNN {
    class Size {
    public:
        Size(unsigned width, unsigned height) : _width(width), _height(height) {}
        unsigned getWidth() const { return _width; }
        unsigned getHeight() const { return _height; }

    private:
        unsigned _width;
        unsigned _height;
    };
}

#endif // SIZE_H
