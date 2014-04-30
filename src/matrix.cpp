#include "matrix.h"
#include "utils.h"
#include <math.h>
#include <exception>
#include <fstream>
#include <sstream>
#include <stdlib.h>
#include <assert.h>

namespace CNN {
    Matrix::Matrix() {
    }

    Matrix::Matrix(unsigned width, unsigned height, unsigned channel) :
            _width(width),
            _height(height),
            _channel(channel) {
        _values.assign(_width * _height * _channel, 0.0);
    }

    Matrix::~Matrix() {
        //delete[] _values;
    }

    void Matrix::initWith(double v) {
        for (unsigned int c = 0; c < _channel; ++c)
            for (unsigned int i = 0; i < _width; ++i)
                for (unsigned int j = 0; j < _height; ++j) {
                    set(v, i, j, c);
                }
    }

    double& Matrix::operator()(unsigned i, unsigned j, unsigned k) {
        return get(i, j, k);
    }

    double Matrix::operator()(unsigned i, unsigned j, unsigned k) const {
        return get(i, j, k);
    }

    double& Matrix::get(unsigned i, unsigned j, unsigned k) {
        if (i >= _width || j >= _height) {
            throw std::exception();
        }
        return _values.at((_width * _height) * k + _width * j + i);
    }

    double Matrix::get(unsigned i, unsigned j, unsigned k) const {
        if (i >= _width || j >= _height || k >= _channel) {
            throw std::exception();
        }
        return _values.at((_width * _height) * k + _width * j + i);
    }

    void Matrix::set(double v, unsigned i, unsigned j, unsigned k) {
        _values.at((_width * _height) * k + _width * j + i) = v;
    }

    Matrix Matrix::convolve(const Matrix& filter) const {
        unsigned output_width = _width - filter.getWidth() + 1;
        unsigned output_height = _height - filter.getHeight() + 1;
        Matrix output(output_width, output_height);
        output.initWith(0.0);

        for (int c = 0; c < _channel; ++c) {
            Matrix t(output_width, output_height);
            for (unsigned int i = 0; i < output_width; ++i)
                for (unsigned int j = 0; j < output_height; ++j) {
                    double sum = 0.0;
                    for (int a = 0; a < filter.getWidth(); ++a)
                        for (int b = 0; b < filter.getHeight(); ++b) {
                            sum += get(a + i, b + j) * filter(a, b);
                        }
                    t(i, j) = sum;
                }

            for (int i = 0; i < output_width; ++i)
                for (int j = 0; j < output_height; ++j) {
                    output.set(output(i, j, 0) + t(i, j, 0), i, j);
                }
        }

        return output;
    }

    const Matrix& Matrix::apply(double (* func)(double)) {
        for (int i = 0; i < _width; ++i)
            for (int j = 0; j < _height; ++j) {
                _values[_height * i + j] = func(_values[_height * i + j]);
            }
        return *this;
    }

    const Matrix& Matrix::add(double b) {
        for (int i = 0; i < _width; ++i)
            for (int j = 0; j < _height; ++j) {
                _values[_height * i + j] = _values[_height * i + j] + b;
            }
        return *this;
    }

    const Matrix& Matrix::add(const Matrix& m) {
        assert(m.getWidth() == _width);
        assert(m.getHeight() == _height);
        for (int i = 0; i < _width; ++i)
            for (int j = 0; j < _height; ++j) {
                _values[_height * i + j] = _values[_height * i + j] + m(i, j);
            }
        return *this;
    }

    Vector Matrix::flatten(const Matrices& input) {
        Vector output;
        Matrices::const_iterator it;
        for (it = input.begin(); it != input.end(); ++it) {
            const Matrix& matrix = *it;
            for (int j = 0; j < matrix.getHeight(); ++j)
                for (int i = 0; i < matrix.getWidth(); ++i) {
                    output.push_back(matrix(i, j));
                }
        }
        return output;
    }

    Matrices Matrix::matricesFrom(std::string filename) {
        std::ifstream in(filename);
        Matrices output;

        std::stringstream ss;
        while (in.good()) {
            char c = in.get();
            if (c != ' ' && c != '[' && c != '\n') {
                ss << c;
                while (in.good() && (c = in.get()) != ']') {
                    if (c != '\n' && c != '\r' && c != '[') {
                        ss << c;
                    }
                }
                ss << '\n';
                if ((c = in.get()) == ']') {
                    //std::cout << ss.str() << std::endl;
                    output.push_back(matrixFrom(ss));
                    ss.str("");
                    ss.clear();
                }
            }
        }
        in.close();
        return output;
    }

    Matrix Matrix::matrixFrom(const std::string& filename) {
        std::ifstream in(filename);
        Matrix output = matrixFrom(in);
        in.close();
        return output;
    }

    Matrix Matrix::matrixFrom(std::istream& in) {
        unsigned height = std::count(std::istreambuf_iterator<char>(in),
                std::istreambuf_iterator<char>(), '\n');

        in.seekg(0);
        std::string line;
        std::getline(in, line);
        line = Utils::trim(line);
        unsigned width = std::count(line.begin(), line.end(), '.');

        Matrix output(width, height);
        int i, j = 0;
        do {
            i = 0;
            std::vector<std::string> values = Utils::split(Utils::trim(line), ' ');
            std::vector<std::string>::iterator it;

            for (it = values.begin(); it != values.end(); ++it) {
                if (it->size() == 0) continue;

                output.set(atof(it->c_str()), i, j);
                i++;
            }
            j++;
        } while (std::getline(in, line));
        return output;
    }


    Matrix Matrix::sumFrom(const Matrices& matrices) {
        if (matrices.empty()) throw std::exception();
        unsigned int width = matrices.front().getWidth();
        unsigned int height = matrices.front().getHeight();
        Matrix sum(width, height);
        for (const Matrix& matrix : matrices) {
            sum.add(matrix);
        }
        return sum;
    }

    const Vector& vectorFrom(std::string filename) {
        std::ifstream in(filename);
        Vector* output = new Vector();
        double value;
        while (in >> value) {
            output->push_back(value);
        }

        in.close();
        return *output;
    }

    std::ostream& operator<<(std::ostream& stream, const Matrix& matrix) {
        for (unsigned int j = 0; j < matrix.getHeight(); ++j) {
            for (unsigned int i = 0; i < matrix.getWidth(); ++i) {
                stream << matrix.get(i, j);
                if (i != matrix.getWidth() - 1) {
                    stream << ", ";
                }
            }
            if (j != matrix.getHeight() - 1) {
                stream << "," << std::endl;
            }
        }
        return stream;
    }
}

std::ostream& operator<<(std::ostream& stream, const CNN::Vector& vector) {
    CNN::Vector::const_iterator it;
    for (it = vector.begin(); it != vector.end(); ++it) {
        stream << *it;
        if (it != vector.end() - 1) {
            stream << ",";
        }
    }
    return stream;
}
