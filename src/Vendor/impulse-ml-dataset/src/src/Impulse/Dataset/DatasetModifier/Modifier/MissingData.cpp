#include "../../include.h"

namespace Impulse {

    namespace Dataset {

        namespace DatasetModifier {

            namespace Modifier {

                void MissingData::setModificationType(T_String type) {
                    this->modificationType = std::move(type);
                }

                void MissingData::applyToColumn(int columnIndex) {
                    DatasetData samples = this->dataset.getSamples();
                    T_IntVector rowsToFill;
                    T_Size correctSamplesCount = 0;
                    double sum = 0.0;
                    double valueToFill = 0;
                    int rowIndex = 0;

                    for (auto &sample : samples) {
                        T_String columnValue = sample->getColumnToString(columnIndex);
                        if (columnValue.length() == 0) {
                            rowsToFill.push_back(rowIndex);
                        } else {
                            sum += sample->getColumnToDouble(columnIndex);
                            correctSamplesCount++;
                        }
                        rowIndex++;
                    }

                    if (this->modificationType == "mean") {
                        valueToFill = sum / (double) correctSamplesCount;
                    }

                    for (int i : rowsToFill) {
                        this->dataset.getSampleAt(i)->setColumn(columnIndex, valueToFill);
                    }
                }
            }
        }
    }
}
