#pragma once

#include <Eigen/Dense>
#include <cstddef>
#include <vector>

using namespace Eigen;

MatrixXd randomMatrix(size_t row, size_t col, double min = 0, double max = 1) {
    double range = max - min;

    MatrixXd m = MatrixXd::Random(row, col);
    m = (m + MatrixXd::Constant(row, col, 1.)) * range / 2.;
    m = (m + MatrixXd::Constant(row, col, min));

    return m;
}

class NeuralNetwork {
private:
    std::vector<MatrixXd> layers;
public:
    Matrix<double, 1, Dynamic> input; // input vector

    NeuralNetwork(size_t input_neurons = 3) {
        input.resize(1, input_neurons);
    }

    void add_layer(size_t neurons, double rand_min = 0, double rand_max = 1) {
        size_t layer_col = (this->layers.empty()) ? input.cols() : this->layers.back().rows();
        this->layers.push_back(randomMatrix(neurons, layer_col, rand_min, rand_max));
    }

    void add_layer(const MatrixXd &ref) {
        size_t cols = (this->layers.empty()) ? input.cols() : this->layers.back().rows();
        assert(ref.cols() == cols);

        this->layers.push_back(ref);
    }

    MatrixXd predict() const {
        MatrixXd out(this->input);

        for (const auto &layer : this->layers)
            out *= layer.transpose();

        return out;
    }
};