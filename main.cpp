#include <iostream>
#include <Eigen/Dense>
#include <vector>

#include "neural_network.h"

using namespace std;
using namespace Eigen;


int main() {
    Matrix<double, 3, 3> first, second;
    first << .1, .2, -.1,
            -.1, .1, .9,
            .1, .4, .1;

    second << .3, 1.1, -.3,
            .1, .2, 0,
            0, 1.3, .1;

    NeuralNetwork neural(3);
    neural.input << 8.5, 0.65, 1.2;


    neural.add_layer(first);
    neural.add_layer(second);
    cout << neural.predict();

//    Matrix<double, 1, 3> in;
//    Matrix<double, 3, 3> weight;
//
//    vector<Matrix<double, 3, 3>> test;
//    test.resize(2);
//
//    in << 8.5, 0.65, 1.2;
//
//

//
//
//    cout << deep_neutral_network(in, test);

    return 0;
}
