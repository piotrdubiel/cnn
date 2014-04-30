#ifndef MATRIX_H
#define MATRIX_H

#include <iostream>
#include <deque>
#include <vector>

namespace CNN {
    class Matrix;

    typedef std::deque<Matrix> Matrices;
    typedef std::vector<double> Vector;

    class Matrix {
    public:
        Matrix();

        Matrix(unsigned width, unsigned height, unsigned channel = 1);

        ~Matrix();

        double& operator()(unsigned i, unsigned j, unsigned k = 0);

        double operator()(unsigned i, unsigned j, unsigned k = 0) const;

        double& get(unsigned i, unsigned j, unsigned k = 0);

        double get(unsigned i, unsigned j, unsigned k = 0) const;

        void set(double v, unsigned i, unsigned j, unsigned k = 0);

        unsigned getWidth() const {
            return _width;
        }

        unsigned getHeight() const {
            return _height;
        }

        Matrix convolve(const Matrix& filter) const;

        const Matrix& apply(double (* func)(double));

        const Matrix& add(double);

        const Matrix& add(const Matrix&);

        void initWith(double v);

        static Vector flatten(const Matrices&);

        static Matrices matricesFrom(std::string filename);

        static Matrix matrixFrom(const std::string& filename);

        static Matrix matrixFrom(std::istream& in);

        static Matrix sumFrom(const Matrices& matrices);


        friend std::ostream& operator<<(std::ostream& stream, const Matrix& matrix);

    private:
        unsigned _width;
        unsigned _height;
        unsigned _channel;
        std::vector<double> _values;
    };

    const Vector& vectorFrom(std::string filename);
}

std::ostream& operator<<(std::ostream& stream, const CNN::Vector& vector);

#endif // MATRIX_H
