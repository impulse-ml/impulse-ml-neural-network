#include "../include.h"

using namespace Impulse::NeuralNetwork;

namespace Impulse {

    namespace NeuralNetwork {

        namespace Trainer {

            GradientDescent::GradientDescent(Network::Abstract &net) : AbstractTrainer(net) {}

            void GradientDescent::train(Impulse::Dataset::SlicedDataset &dataSet) {
                Network::Abstract network = this->network;
                double learningRate = this->learningRate;
                T_Size iterations = this->learningIterations;

                for (T_Size i = 0; i < iterations; i++) {
                    high_resolution_clock::time_point begin = high_resolution_clock::now();

                    network.backward(dataSet.getInput(), dataSet.getOutput(), network.forward(dataSet.getInput()), this->regularization);

                    Trainer::CostGradientResult result = this->cost(dataSet);

                    for (T_Size j = 0; j < network.getSize(); j++) {
                        Layer::LayerPointer layer = network.getLayer(j);

                        if (layer->getType() == Layer::TYPE_MAXPOOL) {
                            continue;
                        }

                        layer->W = layer->W.array() - learningRate * (layer->gW.array());
                        layer->b = layer->b.array() - learningRate * (layer->gb.array());
                    }

                    Trainer::CostGradientResult currentResult = this->cost(dataSet);

                    if (this->verbose) {
                        if ((i + 1) % this->verboseStep == 0) {
                            high_resolution_clock::time_point end = high_resolution_clock::now();
                            auto duration = duration_cast<milliseconds>(end - begin).count();
                            std::cout << "Iteration: " << (i + 1)
                                      << " | Cost: " << currentResult.getCost()
                                      << " | Accuracy: " << currentResult.getAccuracy()
                                      << "% | Time: " << duration
                                      << std::endl;
                        }
                    }

                    if (currentResult.getCost() > result.getCost()) {
                        std::cout << "Terminated." << std::endl;
                        break;
                    }
                }
            }
        }
    }
}
