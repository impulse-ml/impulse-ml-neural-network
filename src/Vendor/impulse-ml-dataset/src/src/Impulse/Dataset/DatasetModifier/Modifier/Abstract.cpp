#include "../../include.h"

namespace Impulse {

    namespace Dataset {

        namespace DatasetModifier {

            namespace Modifier {

                Abstract::Abstract(Dataset &dataset) : dataset(dataset) {}

                void Abstract::apply() {
                    for (T_Size i = 0; i < this->dataset.getColumnsSize(); i++) {
                        this->applyToColumn(i);
                    }
                }
            }
        }
    }
}
