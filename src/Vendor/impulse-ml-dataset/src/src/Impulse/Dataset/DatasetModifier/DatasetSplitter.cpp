#include "../include.h"

namespace Impulse {

    namespace Dataset {

        namespace DatasetModifier {

            DatasetSplitter::DatasetSplitter(Dataset &dataset) : dataset(dataset) {
                this->dataset = dataset;
            }

            SplitDataset DatasetSplitter::split(double ratio) {
                SplitDataset result;

                for (T_Size i = 0; i < this->dataset.getSize(); i++) {
                    double stepRatio = (double) (i + 1) / (double) this->dataset.getSize();

                    if (ratio >= stepRatio) {
                        result.primary.addSample(this->dataset.getSampleAt(i));
                    } else {
                        result.secondary.addSample(this->dataset.getSampleAt(i));
                    }
                }

                return result;
            }
        }
    }
}
