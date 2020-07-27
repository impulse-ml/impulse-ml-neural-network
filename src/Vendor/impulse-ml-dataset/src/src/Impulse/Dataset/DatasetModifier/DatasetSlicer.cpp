#include "../include.h"

namespace Impulse {

    namespace Dataset {

        namespace DatasetModifier {

            DatasetSlicer::DatasetSlicer(Dataset &dataset) : dataset(dataset) {}

            SlicedDataset DatasetSlicer::slice() {
                Dataset input;
                Dataset output;
                DatasetData samples = this->dataset.getSamples();

                for (auto &oldSample : samples) {
                    T_StringVector inputSampleData;
                    T_StringVector outputSampleData;

                    for (int i : this->inputColumns) {
                        inputSampleData.push_back(oldSample->getColumnToString(i));
                    }
                    input.addSample(std::make_shared<Impulse::Dataset::DatasetSample>(std::move(inputSampleData)));

                    for (int i : this->outputColumns) {
                        outputSampleData.push_back(oldSample->getColumnToString(i));
                    }
                    output.addSample(std::make_shared<Impulse::Dataset::DatasetSample>(std::move(outputSampleData)));
                }

                SlicedDataset result;
                result.input = input;
                result.output = output;

                return result;
            }

            void DatasetSlicer::addInputColumn(int columnIndex) {
                this->inputColumns.push_back(columnIndex);
            }

            void DatasetSlicer::addOutputColumn(int columnIndex) {
                this->outputColumns.push_back(columnIndex);
            }
        }
    }
}
