#pragma once

#include "../../include.h"

using namespace Impulse::Dataset;

namespace Impulse {

    namespace Dataset {

        namespace DatasetModifier {

            namespace Modifier {

                class CategoryId : public Abstract {
                public:
                    explicit CategoryId(Dataset &dataset) : Abstract(dataset) {}

                    void applyToColumn(int columnIndex) override;
                };
            }
        }
    }
}
