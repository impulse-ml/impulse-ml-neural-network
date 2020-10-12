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
#include <experimental/filesystem>
#include <eigen3/Eigen/Core>

#include "src/Vendor/impulse-ml-dataset/src/src/Impulse/Dataset/include.h"
#include "src/Impulse/NeuralNetwork/include.h"

using namespace std::chrono;
using namespace Impulse::NeuralNetwork;

SlicedDataset getMnistDataset() {
    Impulse::Dataset::DatasetBuilder::CSVBuilder datasetBuilder1(
            "../data/mnist_test.csv");
    Impulse::Dataset::Dataset dataset = datasetBuilder1.build();

    Impulse::Dataset::DatasetModifier::DatasetSlicer slicer(dataset);
    slicer.addOutputColumn(0);
    for (int i = 0; i < 28 * 28; i++) {
        slicer.addInputColumn(i + 1);
    }

    Impulse::Dataset::SlicedDataset slicedDataset = slicer.slice();

    Impulse::Dataset::DatasetModifier::Modifier::Category modifier3(slicedDataset.output);
    modifier3.applyToColumn(0);

    return slicedDataset;
}

SlicedDataset getMnistTestDataset() {
    Impulse::Dataset::DatasetBuilder::CSVBuilder datasetBuilder1(
            "../data/mnist_test_1000.csv");
    Impulse::Dataset::Dataset dataset = datasetBuilder1.build();

    Impulse::Dataset::DatasetModifier::DatasetSlicer slicer(dataset);
    slicer.addOutputColumn(0);
    for (int i = 0; i < 28 * 28; i++) {
        slicer.addInputColumn(i + 1);
    }

    Impulse::Dataset::SlicedDataset slicedDataset = slicer.slice();

    Impulse::Dataset::DatasetModifier::Modifier::Category modifier3(slicedDataset.output);
    modifier3.applyToColumn(0);

    return slicedDataset;
}

void conv_mnist() {
    SlicedDataset dataset = getMnistDataset();
    SlicedDataset testDataset = getMnistTestDataset();

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

    builder.createLayer<Layer::Tanh>([](auto *layer) {
        layer->setSize(1024);
    });

    builder.createLayer<Layer::Tanh>([](auto *layer) {
        layer->setSize(256);
    });

    builder.createLayer<Layer::Softmax>([](auto *layer) {
        layer->setSize(10);
    });

    Network::ConvNetwork net = builder.getNetwork();

    Trainer::MiniBatch<Trainer::Optimizer::Adagrad> trainer(net);
    trainer.setLearningIterations(10);
    trainer.setVerboseStep(1);
    trainer.setRegularization(0.1);
    trainer.setVerbose(true);
    trainer.setLearningRate(0.005);
    trainer.setStepCallback([&trainer, &testDataset]() {
        Trainer::CostGradientResult cost = trainer.cost(testDataset);
        std::cout << "___STEP___" << std::endl;
        std::cout << "Cost: " << cost.getCost() << std::endl;
        std::cout << "Accuracy: " << cost.getAccuracy() << "%" << std::endl;
    });

    //Trainer::CostGradientResult cost = trainer.cost(dataset);
    //std::cout << "Cost: " << cost.getCost() << std::endl;
    //std::cout << "Accuracy: " << cost.getAccuracy() << std::endl;
    std::cout << "Forward:" << std::endl << net.forward(dataset.input.getSampleAt(0)->exportToEigen()) << std::endl;

    high_resolution_clock::time_point t1 = high_resolution_clock::now();
    trainer.train(dataset);
    high_resolution_clock::time_point t2 = high_resolution_clock::now();

    auto duration = duration_cast<seconds>(t2 - t1).count();
    std::cout << "Time: " << duration << std::endl;
    high_resolution_clock::time_point t3 = high_resolution_clock::now();
    std::cout << "Forward:" << std::endl << net.forward(dataset.input.getSampleAt(0)->exportToEigen()) << std::endl;
    high_resolution_clock::time_point t4 = high_resolution_clock::now();
    auto duration2 = duration_cast<milliseconds>(t4 - t3).count();
    std::cout << "Time forward: " << duration2 << std::endl;
    std::cout << "Cost: " << trainer.cost(dataset).getCost() << std::endl;
    high_resolution_clock::time_point t5 = high_resolution_clock::now();
    auto duration3 = duration_cast<milliseconds>(t5 - t4).count();
    std::cout << "Time cost: " << duration3 << std::endl;

    Serializer serializer(net);
    serializer.toJSON("../saved/test_conv_mnist.json");
}

void mnist_minibatch_gradient_descent() {
    SlicedDataset dataset = getMnistDataset();
    SlicedDataset testDataset = getMnistTestDataset();

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

    Trainer::MiniBatch<Trainer::Optimizer::Adagrad> trainer(net);
    trainer.setLearningIterations(3);
    trainer.setVerboseStep(1);
    trainer.setRegularization(0.05);
    trainer.setVerbose(true);
    trainer.setLearningRate(0.1);
    trainer.setStepCallback([&trainer, &testDataset]() {
        Trainer::CostGradientResult cost = trainer.cost(testDataset);
        std::cout << "___STEP___" << std::endl;
        std::cout << "Cost: " << cost.getCost() << std::endl;
        std::cout << "Accuracy: " << cost.getAccuracy() << "%" << std::endl;
    });

    Trainer::CostGradientResult cost = trainer.cost(dataset);
    std::cout << "Cost: " << cost.getCost() << std::endl;
    std::cout << "Accuracy: " << cost.getAccuracy() << "%" << std::endl;
    std::cout << "Forward:" << std::endl << net.forward(dataset.input.getSampleAt(0)->exportToEigen()) << std::endl;

    high_resolution_clock::time_point t1 = high_resolution_clock::now();
    trainer.train(dataset);
    high_resolution_clock::time_point t2 = high_resolution_clock::now();

    auto duration = duration_cast<seconds>(t2 - t1).count();
    std::cout << "Time: " << duration << std::endl;
    high_resolution_clock::time_point t3 = high_resolution_clock::now();
    std::cout << "Forward:" << std::endl << net.forward(testDataset.input.getSampleAt(0)->exportToEigen()) << std::endl;
    high_resolution_clock::time_point t4 = high_resolution_clock::now();
    auto duration2 = duration_cast<microseconds>(t4 - t3).count();
    std::cout << "Forward time: " << duration2 << " microseconds." << std::endl;
    std::cout << "Cost: " << trainer.cost(dataset).getCost() << std::endl;
    std::cout << "Accuracy: " << trainer.cost(dataset).getAccuracy() << "%" << std::endl;

    Serializer serializer(net);
    serializer.toJSON("../saved/test_mnist_minibatch_gradient_descent.json");
}

void mnist_minibatch_gradient_descent_restore() {
    Builder::ClassifierBuilder builder = Builder::ClassifierBuilder::fromJSON("../saved/test_mnist_minibatch_gradient_descent.json");
    Network::ClassifierNetwork network = builder.getNetwork();

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

    high_resolution_clock::time_point t1 = high_resolution_clock::now();
    std::cout << "Forward:" << std::endl << network.forward(slicedDataset.input.getSampleAt(0)->exportToEigen()) << std::endl;
    high_resolution_clock::time_point t2 = high_resolution_clock::now();
    auto duration2 = duration_cast<microseconds>(t2 - t1).count();
    std::cout << "Forward time: " << duration2 << " microseconds." << std::endl;
}

void test_lstm() {
    Impulse::Dataset::Dictionary::makeDictionary("../data/input.txt", "../data/output.txt", "../saved/dictionary.txt");
    Impulse::Dataset::Dictionary dic = Impulse::Dataset::Dictionary::load("../saved/dictionary.txt", "../data/input.txt", "../data/output.txt");
    std::vector<Eigen::VectorXd> input = dic.getInput();
    for (T_Size i = 0; i < input.size(); i += 1) {
        std::cout << input.at(i) << "\n\n\n";
    }
}

int main() {
    //mnist_minibatch_gradient_descent();
    //mnist_minibatch_gradient_descent_restore();
    //conv_mnist();
    test_lstm();
    return 0;
}
