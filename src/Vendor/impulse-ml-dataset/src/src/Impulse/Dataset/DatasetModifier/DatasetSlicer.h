#pragma once

#include "../include.h"

using namespace Impulse::Dataset;

namespace Impulse {

    namespace Dataset {

        struct SlicedDataset {
            Dataset input;
            Dataset output;

            T_Size getNumberOfExamples() {
                return this->input.getSize();
            }

            Eigen::MatrixXd getInput(T_Size offset, T_Size batchSize) {
                Eigen::MatrixXd input = this->input.exportToEigen();
                return input.block(offset, 0, batchSize, input.cols()).transpose();
            }

            Eigen::MatrixXd getOutput(T_Size offset, T_Size batchSize) {
                Eigen::MatrixXd output = this->output.exportToEigen();
                return output.block(offset, 0, batchSize, output.cols()).transpose();
            }

            Eigen::MatrixXd getInput() {
                return this->input.exportToEigen().transpose();
            }

            Eigen::MatrixXd getOutput() {
                return this->output.exportToEigen().transpose();
            }
        };

        namespace DatasetModifier {

            class DatasetSlicer {
            protected:
                Dataset &dataset;

                T_IntVector inputColumns;

                T_IntVector outputColumns;

            public:
                explicit DatasetSlicer(Dataset &dataset);

                void addInputColumn(int columnIndex);

                void addOutputColumn(int columnIndex);

                SlicedDataset slice();
            };
        }
    }
}
