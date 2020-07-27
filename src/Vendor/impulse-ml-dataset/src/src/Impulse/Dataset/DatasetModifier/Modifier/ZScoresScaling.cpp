#include "../../include.h"

namespace Impulse {

    namespace Dataset {

        namespace DatasetModifier {

            namespace Modifier {

                void ZScoresScaling::applyToColumn(int columnIndex) {
                    DatasetData samples = this->dataset.getSamples();
                    auto count = (double) samples.size();
                    double sum = 0.0;
                    double variance = 0.0;
                    double standardDeviation;
                    double mean;

                    for (auto &sample : samples) {
                        sum += sample->getColumnToDouble(columnIndex);
                    }

                    mean = sum / count;

                    for (auto &sample : samples) {
                        double value = sample->getColumnToDouble(columnIndex);
                        variance += pow(value - mean, 2.0);
                    }

                    standardDeviation = sqrt((1.0 / count) * variance);

                    for (auto &sample : samples) {
                        double value = sample->getColumnToDouble(columnIndex);
                        double newValue = (value - mean) / standardDeviation;
                        sample->setColumn(columnIndex, newValue);
                    }

                    this->data["mean"][columnIndex] = toString(mean);
                    this->data["standardDeviation"][columnIndex] = toString(
                            standardDeviation);
                }
            }
        }
    }
}
