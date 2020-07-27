#include "../../include.h"

namespace Impulse {

    namespace Dataset {

        namespace DatasetModifier {

            namespace Modifier {

                void MinMaxScaling::applyToColumn(int columnIndex) {
                    DatasetData samples = this->dataset.getSamples();
                    double min = std::numeric_limits<double>::max();
                    double max = std::numeric_limits<double>::min();

                    for (auto &sample : samples) {
                        double value = sample->getColumnToDouble(columnIndex);
                        min = MIN(value, min);
                        max = MAX(value, max);
                    }

                    for (auto &sample : samples) {
                        double value = sample->getColumnToDouble(columnIndex);
                        double newValue = (value - min) / (max - min);
                        sample->setColumn(columnIndex, newValue);
                    }
                }
            }
        }
    }
}
