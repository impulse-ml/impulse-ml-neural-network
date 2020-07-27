#include "../../include.h"

namespace Impulse {

    namespace NeuralNetwork {

        namespace Layer {

            namespace BackPropagation {

                BackPropagationToConv::BackPropagationToConv(Layer::LayerPointer layer, Layer::LayerPointer previousLayer) : Abstract(layer, previousLayer) {}

                Math::T_Matrix BackPropagationToConv::propagate(const Math::T_Matrix &input, T_Size numberOfExamples, double regularization, const Math::T_Matrix &sigma) {

                    auto *previousLayer = (Layer::Conv *) this->previousLayer.get();

                    int padding = previousLayer->getPadding();
                    int stride = previousLayer->getStride();
                    int filterSize = previousLayer->getFilterSize();
                    int outputWidth = previousLayer->getOutputWidth();
                    int outputHeight = previousLayer->getOutputHeight();
                    int outputDepth = previousLayer->getOutputDepth();
                    int inputWidth = previousLayer->getWidth();
                    int inputHeight = previousLayer->getHeight();
                    int inputDepth = previousLayer->getDepth();

                    Math::T_Matrix tmpResult((inputWidth + 2 * padding) * (inputHeight + 2 * padding) * inputDepth, numberOfExamples);
                    tmpResult.setZero();

                    Math::T_Matrix result(inputWidth * inputHeight * inputDepth, numberOfExamples);

                    Math::T_Matrix aPrev = previousLayer->derivative();

                    previousLayer->gW.setZero();
                    previousLayer->gb.setZero();

#pragma omp parallel
#pragma omp for
                    for (int m = 0; m < numberOfExamples; m++) {
                        for (int c = 0; c < outputDepth; c++) {
                            for (int h = 0; h < outputHeight; h++) {
                                for (int w = 0; w < outputWidth; w++) {
                                    int vertStart = stride * h;
                                    int vertEnd = vertStart + filterSize;
                                    int horizStart = stride * w;
                                    int horizEnd = horizStart + filterSize;

                                    // filter loop
                                    for (int d = 0; d < inputDepth; d++) {
                                        for (int y = 0, vertical = vertStart, verticalPad = -padding; y < filterSize; y++, vertical++, verticalPad++) {
                                            for (int x = 0, horizontal = horizStart, horizontalPad = -padding; x < filterSize; x++, horizontal++, horizontalPad++) {
                                                tmpResult(((d * (inputWidth + 2 * padding) * (inputHeight + 2 * padding)) + (vertical * (inputWidth + 2 * padding)) + horizontal), m) +=
                                                        previousLayer->W(c, (d * filterSize * filterSize) + (y * filterSize) + x) *
                                                        sigma((c * outputWidth * outputHeight) + (h * outputWidth) + w, m);

                                                double z = 0;
                                                if (padding == 0) {
                                                    z = previousLayer->Z((d * inputWidth * inputHeight) + (vertical * inputWidth) + horizontal, m);
                                                } else {
                                                    if (verticalPad >= 0 && horizontalPad >= 0 && verticalPad < inputHeight && horizontalPad < inputWidth) {
                                                        z = previousLayer->Z((d * inputWidth * inputHeight) + (verticalPad * inputWidth) + horizontalPad, m);
                                                    }
                                                }

                                                previousLayer->gW(c, (d * filterSize * filterSize) + (y * filterSize) + x) +=
                                                        (
                                                                z *
                                                                sigma(c * (outputWidth * outputHeight) + (h * outputWidth) + w, m)
                                                        ) / numberOfExamples;
                                            }
                                        }
                                    }

                                    previousLayer->gb(c, 0) += sigma(c * (outputWidth * outputHeight) + (h * outputWidth) + w, m) / numberOfExamples;
                                }
                            }
                        }

                        if (padding > 0) { // unpad
                            for (int c = 0; c < inputDepth; c++) {
                                for (int h = -padding, y = 0; h < inputHeight + padding; h++, y++) {
                                    for (int w = -padding, x = 0; w < inputWidth + padding; w++, x++) {
                                        if (w >= 0 && h >= 0 && w < inputWidth && h < inputHeight) {
                                            result((c * inputWidth * inputHeight) + (h * inputWidth) + w, m) = tmpResult((c * (inputWidth + 2 * padding) * (inputHeight + 2 * padding)) + (y * (inputWidth + 2 * padding)) + x, m);
                                        }
                                    }
                                }
                            }
                        }
                    }

                    if (padding > 0) {
                        return result;
                    }

                    return tmpResult;
                }
            }
        }
    }
}