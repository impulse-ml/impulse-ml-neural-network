#pragma once

#include "../../include.h"

using namespace Impulse::Dataset;

namespace Impulse {

    namespace Dataset {

        namespace DatasetModifier {

            namespace Modifier {

                class MinMaxScaling : public Abstract {
                public:
                    explicit MinMaxScaling(Dataset &dataset) : Abstract(dataset) {}

                    void applyToColumn(int columnIndex) override;
                };
            }
        }
    }
}
