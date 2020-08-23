#include "../../include.h"

namespace Impulse {

    namespace NeuralNetwork {

        namespace Layer {

            namespace BackPropagation {

                BackPropagationToMaxPool::BackPropagationToMaxPool(Layer::LayerPointer layer,
                                                                   Layer::LayerPointer previousLayer) : Abstract(layer,
                                                                                                                 previousLayer) {}

                Eigen::MatrixXd
                BackPropagationToMaxPool::propagate(const Eigen::MatrixXd &input, T_Size numberOfExamples,
                                                    double regularization, const Eigen::MatrixXd &sigma) {

                    auto *prevLayer = (Layer::MaxPool *) this->previousLayer.get();
                    Eigen::MatrixXd result(prevLayer->getComputation()->getVariable("Z").rows(),
                                           prevLayer->getComputation()->getVariable("Z").cols());
                    result.setZero();

                    T_Size filterSize = prevLayer->getFilterSize();
                    T_Size stride = prevLayer->getStride();
                    T_Size inputWidth = prevLayer->getWidth();
                    T_Size inputHeight = prevLayer->getHeight();
                    T_Size inputDepth = prevLayer->getDepth();
                    T_Size outputWidth = prevLayer->getOutputWidth();
                    T_Size outputHeight = prevLayer->getOutputHeight();
                    T_Size outputDepth = prevLayer->getOutputDepth();

#pragma omp parallel for collapse(4)
                    for (T_Size m = 0; m < numberOfExamples; m++) {
                        for (T_Size c = 0; c < outputDepth; c++) {
                            for (T_Size h = 0; h < outputHeight; h++) {
                                for (T_Size w = 0; w < outputWidth; w++) {
                                    T_Size vertStart = stride * h;
                                    T_Size vertEnd = vertStart + filterSize;
                                    T_Size horizStart = stride * w;
                                    T_Size horizEnd = horizStart + filterSize;

                                    double _max = -INFINITY;
                                    T_Size inputOffset = inputHeight * inputWidth * c;
                                    T_Size outputOffset = outputHeight * outputWidth * c;
                                    T_Size maxX = 0;
                                    T_Size maxY = 0;

                                    for (T_Size y = 0, vStart = vertStart; y < filterSize; y++, vStart++) {
                                        for (T_Size x = 0, hStart = horizStart; x < filterSize; x++, hStart++) {
                                            if (_max < prevLayer->getComputation()->getVariable("Z")(
                                                    inputOffset + (vStart * inputWidth) + hStart, m)) {
                                                _max = prevLayer->getComputation()->getVariable("Z")(
                                                        inputOffset + (vStart * inputWidth) + hStart, m);
                                                maxX = hStart;
                                                maxY = vStart;
                                            }
                                        }
                                    }

                                    result(inputOffset + (maxY * inputWidth) + maxX, m) = sigma(
                                            outputOffset + (h * outputWidth) + w, m);
                                }
                            }
                        }
                    }

                    return result;
                }
            }
        }
    }
}