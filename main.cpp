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

    Trainer::MiniBatchGradientDescent trainer(net);
    trainer.setLearningIterations(2);
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
    std::cout << "Cost: " << trainer.cost(dataset).getCost() << std::endl;

    Serializer serializer(net);
    serializer.toJSON("../saved/test1.json");
}

void test_conv_mnist() {
    Impulse::Dataset::DatasetBuilder::CSVBuilder datasetBuilder1(
            "../data/mnist_test_1000.csv");
    Impulse::Dataset::Dataset dataset = datasetBuilder1.build();
    Impulse::Dataset::DatasetModifier::DatasetSlicer slicer(dataset);
    slicer.addOutputColumn(0);
    for (int i = 0; i < 28 * 28; i++) {
        slicer.addInputColumn(i + 1);
    }

    Impulse::Dataset::SlicedDataset slicedDataset = slicer.slice();

    Impulse::Dataset::DatasetModifier::Modifier::Category modifier2(slicedDataset.output);
    modifier2.applyToColumn(0);

    Builder::ConvBuilder builder({28, 28, 1});

    builder.createLayer<Layer::Conv>([](auto *layer) {
        layer->setFilterSize(4);
        layer->setPadding(1);
        layer->setStride(1);
        layer->setNumFilters(32);
    });

    builder.createLayer<Layer::MaxPool>([](auto *layer) {
        layer->setFilterSize(2);
        layer->setStride(2);
    });

    builder.createLayer<Layer::Conv>([](auto *layer) {
        layer->setFilterSize(3);
        layer->setPadding(1);
        layer->setStride(1);
        layer->setNumFilters(64);
    });

    builder.createLayer<Layer::MaxPool>([](auto *layer) {
        layer->setFilterSize(2);
        layer->setStride(2);
    });

    builder.createLayer<Layer::FullyConnected>([](auto *layer) {

    });

    builder.createLayer<Layer::Relu>([](auto *layer) {
        layer->setSize(1024);
    });

    builder.createLayer<Layer::Softmax>([](auto *layer) {
        layer->setSize(10);
    });

    Network::ConvNetwork net = builder.getNetwork();

    Trainer::ConjugateGradient trainer(net);
    trainer.setLearningIterations(10);
    trainer.setVerboseStep(1);
    trainer.setRegularization(0.1);
    trainer.setVerbose(true);
    trainer.setLearningRate(0.1);

    Trainer::CostGradientResult cost = trainer.cost(slicedDataset);
    std::cout << "Cost: " << cost.getCost() << std::endl;
    std::cout << "Forward:" << std::endl << net.forward(slicedDataset.input.getSampleAt(0)->exportToEigen()) << std::endl;

    high_resolution_clock::time_point t1 = high_resolution_clock::now();
    trainer.train(slicedDataset);
    high_resolution_clock::time_point t2 = high_resolution_clock::now();

    auto duration = duration_cast<seconds>(t2 - t1).count();
    std::cout << "Time: " << duration << std::endl;
    std::cout << "Forward:" << std::endl << net.forward(slicedDataset.input.getSampleAt(0)->exportToEigen()) << std::endl;
    std::cout << "Cost: " << trainer.cost(slicedDataset).getCost() << std::endl;

    Serializer serializer(net);
    serializer.toJSON("../saved/test_conv_mnist.json");
}

void test_mnist_minibatch_gradient_descent() {
    Impulse::Dataset::DatasetBuilder::CSVBuilder datasetBuilder1(
            "../data/mnist_test_1000.csv");
    Impulse::Dataset::Dataset dataset = datasetBuilder1.build();
    Impulse::Dataset::DatasetModifier::DatasetSlicer slicer(dataset);
    slicer.addOutputColumn(0);
    for (int i = 0; i < 28 * 28; i++) {
        slicer.addInputColumn(i + 1);
    }

    Impulse::Dataset::SlicedDataset slicedDataset = slicer.slice();

    Impulse::Dataset::DatasetModifier::Modifier::Category modifier2(slicedDataset.output);
    modifier2.applyToColumn(0);

    Builder::ClassifierBuilder builder({28*28});
    builder.createLayer<Layer::Tanh>([](auto * layer) {
        layer->setSize(100);
    });
    builder.createLayer<Layer::Tanh>([](auto * layer) {
        layer->setSize(20);
    });
    builder.createLayer<Layer::Softmax>([](auto * layer) {
        layer->setSize(10);
    });

    Network::ClassifierNetwork net = builder.getNetwork();

    Trainer::MiniBatchGradientDescent trainer(net);
    trainer.setLearningIterations(5);
    trainer.setVerboseStep(1);
    trainer.setRegularization(0.01);
    trainer.setVerbose(true);
    trainer.setLearningRate(0.005);
    trainer.setOptimizer("adam"); // you can comment this out

    Trainer::CostGradientResult cost = trainer.cost(slicedDataset);
    std::cout << "Cost: " << cost.getCost() << std::endl;
    std::cout << "Forward:" << std::endl << net.forward(slicedDataset.input.getSampleAt(0)->exportToEigen()) << std::endl;

    high_resolution_clock::time_point t1 = high_resolution_clock::now();
    trainer.train(slicedDataset);
    high_resolution_clock::time_point t2 = high_resolution_clock::now();

    auto duration = duration_cast<seconds>(t2 - t1).count();
    std::cout << "Time: " << duration << std::endl;
    std::cout << "Forward:" << std::endl << net.forward(slicedDataset.input.getSampleAt(0)->exportToEigen()) << std::endl;
    std::cout << "Cost: " << trainer.cost(slicedDataset).getCost() << std::endl;

    Serializer serializer(net);
    serializer.toJSON("../saved/test_mnist_minibatch_gradient_descent.json");
}

void test_mnist_minibatch_gradient_descent_restore() {
    Builder::ConvBuilder builder = Builder::ConvBuilder::fromJSON("../saved/test_mnist_minibatch_gradient_descent.json");
    Network::ConvNetwork network = builder.getNetwork();

    Impulse::Dataset::DatasetBuilder::CSVBuilder datasetBuilder1(
            "../data/mnist_test_1000.csv");
    Impulse::Dataset::Dataset dataset = datasetBuilder1.build();
    Impulse::Dataset::DatasetModifier::DatasetSlicer slicer(dataset);
    slicer.addOutputColumn(0);
    for (int i = 0; i < 28 * 28; i++) {
        slicer.addInputColumn(i + 1);
    }

    Impulse::Dataset::SlicedDataset slicedDataset = slicer.slice();

    Impulse::Dataset::DatasetModifier::Modifier::Category modifier2(slicedDataset.output);
    modifier2.applyToColumn(0);

    std::cout << network.forward(slicedDataset.input.getSampleAt(0)->exportToEigen()) << std::endl;
}

int main() {
    //test1();
    test_mnist_minibatch_gradient_descent();
    //test_mnist_minibatch_gradient_descent_restore();
    //test_conv_mnist();
    return 0;
}