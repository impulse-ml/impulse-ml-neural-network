#pragma once

#include "include.h"

using namespace Impulse::Dataset;

namespace Impulse {

    namespace Dataset {

        typedef std::shared_ptr<DatasetSample> DatasetSampleContainer;
        typedef std::vector<DatasetSampleContainer> DatasetData;

        class Dataset {
        protected:
            DatasetData data;
        public:
            ~Dataset();

            void addSample(DatasetSampleContainer sample);

            DatasetData &getSamples();

            DatasetSampleContainer getSampleAt(int index);

            void out(T_Size limit = 10);

            T_Size getSize();

            T_Size getColumnsSize();

            Eigen::MatrixXd exportToEigen();
        };
    }
}
