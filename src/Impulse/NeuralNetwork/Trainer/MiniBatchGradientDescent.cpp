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

                T_Size t = 0;

                for (T_Size i = 0; i < iterations; i++) {
                    high_resolution_clock::time_point beginIteration = high_resolution_clock::now();

                    for (T_Size batch = 0, offset = 0; batch < numberOfExamples; batch += batchSize, offset++) {
                        high_resolution_clock::time_point beginIterationBatch = high_resolution_clock::now();

                        Eigen::MatrixXd input = dataSet.getInput(offset, batchSize);
                        Eigen::MatrixXd output = dataSet.getOutput(offset, batchSize);
                        Eigen::MatrixXd forward = network.forward(input);

                        network.backward(input, output, forward, this->regularization);

                        for (T_Size j = 0; j < network.getSize(); j++) {
                            Layer::LayerPointer layer = network.getLayer(j);

                            if (layer->getType() == Layer::TYPE_MAXPOOL) {
                                continue;
                            }

                            if (this->optimizer == "adam") {
                                t += 1;
                                Computation::factory().gradientAdam(layer->W, learningRate, layer->gW, layer->sW,
                                                                    layer->vW, t);
                                Computation::factory().gradientAdam(layer->b, learningRate, layer->gb, layer->sB,
                                                                    layer->vB, t);
                            } else if (this->optimizer == "rmsprop") {
                                Computation::factory().gradientRmsProp(layer->W, learningRate, layer->gW, layer->sW,
                                                                       batchSize);
                                Computation::factory().gradientRmsProp(layer->b, learningRate, layer->gb, layer->sB,
                                                                       batchSize);
                            } else if (this->optimizer == "adagrad") {
                                Computation::factory().gradientAdagrad(layer->W, learningRate, layer->gW, layer->sW,
                                                                       batchSize);
                                Computation::factory().gradientAdagrad(layer->b, learningRate, layer->gb, layer->sB,
                                                                       batchSize);
                            } else if (this->optimizer == "nesterov") {
                                Computation::factory().gradientNesterov(layer->W, learningRate, layer->gW, layer->sW,
                                                                       batchSize);
                                Computation::factory().gradientNesterov(layer->b, learningRate, layer->gb, layer->sB,
                                                                       batchSize);
                            } else if (this->optimizer == "momentum") {
                                Computation::factory().gradientMomentum(layer->W, learningRate, layer->gW, layer->sW,
                                                                        batchSize);
                                Computation::factory().gradientMomentum(layer->b, learningRate, layer->gb, layer->sB,
                                                                        batchSize);
                            } else {
                                Computation::factory().gradientDescent(layer->W, learningRate, layer->gW);
                                Computation::factory().gradientDescent(layer->b, learningRate, layer->gb);
                            }
                        }

                        if (this->verbose) {
                            high_resolution_clock::time_point endIterationBatch = high_resolution_clock::now();
                            auto durationBatch = duration_cast<milliseconds>(
                                    endIterationBatch - beginIterationBatch).count();
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

            void MiniBatchGradientDescent::setOptimizer(T_String optimizer) {
                this->optimizer = optimizer;
            }
        }
    }
}
