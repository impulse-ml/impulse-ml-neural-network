#include "include.h"

namespace Impulse {

    namespace Dataset {

        DatasetSample::DatasetSample(T_StringVector vec) {
            this->rawData = std::move(vec);
        }

        DatasetSample::DatasetSample(T_StringVector *vec) {
            this->rawData = T_StringVector(vec->begin(), vec->end());
        }

        DatasetSample::DatasetSample(T_DoubleVector vec) {
            for (double i : vec) {
                this->rawData.push_back(toString(i));
            }
        }

        DatasetSample::DatasetSample(std::initializer_list<double> list) {
            for (double i : list) {
                this->rawData.push_back(toString(i));
            }
        }

        void DatasetSample::out() {
            for (T_Size i = 0; i < this->rawData.size() - 1; i++) {
                std::cout << this->rawData.at(i) << ',';
            }
            std::cout << this->rawData.at(this->rawData.size() - 1) << std::endl;
        }

        T_String DatasetSample::getColumnToString(int columnIndex) {
            return this->rawData.at(static_cast<unsigned long>(columnIndex));
        }

        double DatasetSample::getColumnToDouble(int columnIndex) {
            return toDouble(this->getColumnToString(columnIndex));
        }

        void DatasetSample::setColumn(int columnIndex, T_String value) {
            this->rawData.at(static_cast<unsigned long>(columnIndex)) = std::move(value);
        }

        void DatasetSample::setColumn(int columnIndex, double value) {
            this->rawData.at(static_cast<unsigned long>(columnIndex)) = toString(value);
        }

        void DatasetSample::setColumn(int columnIndex, std::function<T_String(T_String)> callback) {
            this->rawData.at(static_cast<unsigned long>(columnIndex)) = callback(this->rawData.at(
                    static_cast<unsigned long>(columnIndex)));
        }

        void DatasetSample::insertColumnAfter(int columnIndex, T_String value) {
            this->rawData.insert(this->rawData.begin() + columnIndex + 1, value);
        }

        void DatasetSample::removeColumnAt(int columnIndex) {
            this->rawData.erase(this->rawData.begin() + columnIndex);
        }

        T_Size DatasetSample::getSize() {
            return (T_Size) this->rawData.size();
        }

        Eigen::VectorXd DatasetSample::exportToEigen() {
            Eigen::VectorXd result(this->getSize());
            for (T_Size i = 0; i < this->getSize(); i++) {
                result(i) = this->getColumnToDouble(i);
            }
            return result;
        }
    }
}
