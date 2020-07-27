#include "../include.h"

using namespace Impulse::NeuralNetwork;

namespace Impulse {

    namespace NeuralNetwork {

        namespace Trainer {

            MiniBatchGradientDescent::MiniBatchGradientDescent(Network::Abstract &net) : AbstractTrainer(net) {}

            void MiniBatchGradientDescent::setBatchSize(T_Size value) {
                this->batchSize = value;
            }

            void MiniBatchGradientDescent::train(Impulse::Dataset::SlicedDataset &dataSet) {
                Network::Abstract network = this->network;
                double learningRate = this->learningRate;
                T_Size iterations = this->learningIterations;
                T_Size batchSize = this->batchSize;
                auto numberOfExamples = (T_Size) dataSet.getInput().cols();
                high_resolution_clock::time_point beginTrain = high_resolution_clock::now();
                double beta1 = this->beta1;
                double beta2 = this->beta2;
                double epsilon = this->epsilon;

                for (T_Size i = 0; i < iterations; i++) {
                    high_resolution_clock::time_point beginIteration = high_resolution_clock::now();

                    for (T_Size batch = 0, offset = 0; batch < numberOfExamples; batch += batchSize, offset++) {
                        high_resolution_clock::time_point beginIterationBatch = high_resolution_clock::now();

                        network.backward(dataSet.getInput(offset, batchSize), dataSet.getOutput(offset, batchSize), network.forward(dataSet.getInput(offset, batchSize)), this->regularization);

                        for (T_Size j = 0; j < network.getSize(); j++) {
                            Layer::LayerPointer layer = network.getLayer(j);

                            if (layer->getType() == Layer::TYPE_MAXPOOL) {
                                continue;
                            }

                            layer->W = layer->W.array() - learningRate * (layer->gW.array());
                            layer->b = layer->b.array() - learningRate * (layer->gb.array());
                        }

                        if (this->verbose) {
                            high_resolution_clock::time_point endIterationBatch = high_resolution_clock::now();
                            auto durationBatch = duration_cast<milliseconds>(endIterationBatch - beginIterationBatch).count();
                            std::cout << "Batch: " << (offset + 1) << "/" << ceil((double) numberOfExamples / batchSize)
                                      << " | Time: " << durationBatch << "ms"
                                      << std::endl;
                        }
                    }

                    if (this->verbose) {
                        Trainer::CostGradientResult currentResult = this->cost(dataSet);

                        if ((i + 1) % this->verboseStep == 0) {
                            high_resolution_clock::time_point endIteration = high_resolution_clock::now();
                            auto duration = duration_cast<milliseconds>(endIteration - beginIteration).count();
                            std::cout << "Iteration: " << (i + 1)
                                      << " | Cost: " << currentResult.getCost()
                                      << " | Accuracy: " << currentResult.getAccuracy()
                                      << "% | Time: " << duration << "ms"
                                      << std::endl;
                        }
                    }
                }

                if (this->verbose) {
                    high_resolution_clock::time_point endTrain = high_resolution_clock::now();
                    auto duration = duration_cast<seconds>(endTrain - beginTrain).count();
                    std::cout << "Training end. " << duration << "s" << std::endl;
                }
            }
        }
    }
}
