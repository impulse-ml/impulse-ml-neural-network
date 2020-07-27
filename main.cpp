#define EIGEN_USE_BLAS
#define EIGEN_USE_THREADS
/*
#define EIGEN_USE_MKL_ALL
#define EIGEN_USE_THREADS
#define MKL_LP64
#define EIGEN_USE_BLAS
#define EIGEN_USE_GPU
#define EIGEN_USE_SYCL
*/

#include <iostream>
#include <cstdlib>
#include <sstream>
#include <fstream>
#include <string>
#include <vector>
#include <ios>
#include <ctime>
#include <experimental/filesystem>
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>

#include "src/Vendor/impulse-ml-dataset/src/src/Impulse/Dataset/include.h"
#include "src/Impulse/NeuralNetwork/include.h"

using namespace std::chrono;
using namespace Impulse::NeuralNetwork;

Impulse::Dataset::SlicedDataset getDataset() {
    // create dataset
    Impulse::Dataset::DatasetBuilder::CSVBuilder datasetBuilder1(
            "/home/user/impulse-vectorized/data/ex4data1_x.csv");
    Impulse::Dataset::Dataset datasetInput = datasetBuilder1.build();

    Impulse::Dataset::DatasetBuilder::CSVBuilder datasetBuilder2(
            "/home/user/impulse-vectorized/data/ex4data1_y.csv");
    Impulse::Dataset::Dataset datasetOutput = datasetBuilder2.build();

    Impulse::Dataset::SlicedDataset dataset;
    dataset.input = datasetInput;
    dataset.output = datasetOutput;

    return dataset;
}

void test1() {
    Impulse::Dataset::SlicedDataset dataset = getDataset();

    Builder::ClassifierBuilder builder({400});
    builder.createLayer<Layer::Relu>([](auto * layer) {
        layer->setSize(100);
    });
    builder.createLayer<Layer::Relu>([](auto * layer) {
        layer->setSize(20);
    });
    builder.createLayer<Layer::Softmax>([](auto * layer) {
        layer->setSize(10);
    });

    Network::ClassifierNetwork net = builder.getNetwork();

    Trainer::GradientDescent trainer(net);
    trainer.setLearningIterations(20000);
    trainer.setVerboseStep(1);
    trainer.setRegularization(0.0);
    trainer.setVerbose(true);
    trainer.setLearningRate(0.05);

    Trainer::CostGradientResult cost = trainer.cost(dataset);
    std::cout << "Cost: " << cost.getCost() << std::endl;
    std::cout << "Forward:" << std::endl << net.forward(dataset.input.getSampleAt(0)->exportToEigen()) << std::endl;

    high_resolution_clock::time_point t1 = high_resolution_clock::now();
    trainer.train(dataset);
    high_resolution_clock::time_point t2 = high_resolution_clock::now();

    auto duration = duration_cast<seconds>(t2 - t1).count();
    std::cout << "Time: " << duration << std::endl;
    std::cout << "Forward:" << std::endl << net.forward(dataset.input.getSampleAt(0)->exportToEigen()) << std::endl;
    std::cout << "Cost: " << cost.getCost() << std::endl;
}

int main() {
    test1();
    return 0;
}