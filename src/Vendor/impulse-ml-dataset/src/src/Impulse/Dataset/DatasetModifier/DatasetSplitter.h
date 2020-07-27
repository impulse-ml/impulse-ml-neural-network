#pragma once

#include "../include.h"

using namespace Impulse::Dataset;

namespace Impulse {

    namespace Dataset {

        struct SplitDataset {
            Dataset primary;
            Dataset secondary;
        };

        namespace DatasetModifier {

            class DatasetSplitter {
            protected:
                Dataset &dataset;

            public:
                explicit DatasetSplitter(Dataset &slicedDataset);

                SplitDataset split(double ratio);
            };
        }
    }
}
