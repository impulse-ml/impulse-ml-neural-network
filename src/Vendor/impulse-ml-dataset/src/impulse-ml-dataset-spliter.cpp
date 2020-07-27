#include "src/Impulse/Dataset/include.h"

using namespace Impulse::Dataset;
using namespace std::chrono;

void test1() {
    auto start = std::chrono::system_clock::now();

    // build dataset from csv...
    DatasetBuilder::CSVBuilder builder("/home/user/impulse-ml-dataset/src/data/data.csv");
    Dataset dataset = builder.build();
    dataset.out();

    // slice dataset to input and output set
    DatasetModifier::DatasetSlicer datasetSlicer(dataset);
    datasetSlicer.addInputColumn(0);
    datasetSlicer.addInputColumn(1);
    datasetSlicer.addInputColumn(2);
    datasetSlicer.addOutputColumn(3);

    SlicedDataset slicedDataset = datasetSlicer.slice();
    Dataset input = slicedDataset.input;
    Dataset output = slicedDataset.output;

    // fill missing data by mean of fields
    DatasetModifier::Modifier::MissingData missingDataModifier(input);
    missingDataModifier.setModificationType("mean");
    missingDataModifier.applyToColumn(1);
    missingDataModifier.applyToColumn(2);
    input.out();

    // create categories
    DatasetModifier::Modifier::Category categoryModifier(input);
    categoryModifier.applyToColumn(0);
    input.out();


    // modify "Yes" and "No" to numbers
    DatasetModifier::Modifier::Callback callbackModifier(output);
    callbackModifier.setCallback([](T_String oldValue) -> T_String {
        if (oldValue.find("Yes") != T_String::npos) {
            return "1";
        }
        return "0";
    });
    callbackModifier.applyToColumn(0);
    output.out();

    // scale all input
    DatasetModifier::Modifier::ZScoresScaling zScoresModifier(input);
    zScoresModifier.applyToColumn(3);
    zScoresModifier.applyToColumn(4);
    input.out();

    auto end = std::chrono::system_clock::now();
    long elapsed_seconds = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    std::cout << elapsed_seconds << std::endl;
}

void test2() {
    DatasetBuilder::CSVBuilder builder("/home/hud/projekty/impulse-ml-dataset/src/data/data2.csv");
    Dataset dataset = builder.build();
    dataset.out();

    DatasetModifier::Modifier::CategoryId categoryId(dataset);
    categoryId.applyToColumn(0);
    categoryId.applyToColumn(1);
    dataset.out();
}

void test3() {
    DatasetBuilder::CSVBuilder builder("/home/hud/projekty/impulse-ml-dataset/src/data/data3.csv");
    Dataset dataset = builder.build();
    dataset.out();

    DatasetModifier::Modifier::MinMaxScaling minMaxScaling(dataset);
    minMaxScaling.apply();
    dataset.out();
}

void test4() {
    DatasetBuilder::CSVBuilder builder("/home/hud/projekty/impulse-ml-dataset/src/data/data3.csv");
    Dataset dataset = builder.build();
    std::cout << dataset.exportToEigen() << std::endl;
}

char *getCmdOption(char **begin, char **end, const std::string &option) {
    char **itr = std::find(begin, end, option);
    if (itr != end && ++itr != end) {
        return *itr;
    }
    return 0;
}

bool cmdOptionExists(char **begin, char **end, const std::string &option) {
    return std::find(begin, end, option) != end;
}

int main(int argc, char *argv[]) {
    auto start = std::chrono::system_clock::now();

    if (cmdOptionExists(argv, argv + argc, "-i")) {
        char *filenameIn = getCmdOption(argv, argv + argc, "-i");

        if (filenameIn) {
            DatasetBuilder::CSVBuilder builder(filenameIn);
            Dataset dataset = builder.build();

            if (cmdOptionExists(argv, argv + argc, "--ratio")) {
                char * ratio = getCmdOption(argv, argv + argc, "--ratio");

                if (ratio) {
                    // slice dataset to input and output set
                    DatasetModifier::DatasetSplitter datasetSplitter(dataset);
                    SplitDataset splitDataset = datasetSplitter.split(std::atof(ratio));

                    if (cmdOptionExists(argv, argv + argc, "-o")) {
                        char *filenameOut = getCmdOption(argv, argv + argc, "-o");
                        T_String filename = T_String(filenameIn).substr(T_String(filenameIn).find_last_of("/") + 1);

                        if (filenameOut) {
                            DatasetExporter exporter(splitDataset.primary);
                            exporter.exportToFile(T_String(filenameOut).append("/").append(filename).append(".PRIMARY.csv"));

                            DatasetExporter exporter2(splitDataset.secondary);
                            exporter2.exportToFile(T_String(filenameOut).append("/").append(filename).append(".SECONDARY.csv"));
                        }

                        if (cmdOptionExists(argv, argv + argc, "-v")) {
                            std::cout << "INPUT: " << std::endl;
                            std::cout << std::endl << "Number of rows: " << splitDataset.primary.getSize() << std::endl;
                            std::cout << "Number of cols: " << splitDataset.primary.getColumnsSize() << std::endl << "---"
                                      << std::endl;
                            splitDataset.primary.out();
                            std::cout << "OUTPUT: " << std::endl;
                            std::cout << std::endl << "Number of rows: " << splitDataset.secondary.getSize() << std::endl;
                            std::cout << "Number of cols: " << splitDataset.secondary.getColumnsSize() << std::endl << "---"
                                      << std::endl;
                            splitDataset.secondary.out();
                        }
                    }
                }
            }
        }
    }

    auto end = std::chrono::system_clock::now();
    long elapsed_seconds = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "Done at: " << elapsed_seconds << " ms." << std::endl;

    return 0;
}