# impulse-ml

## Description

This library can be used to manage datasets used in other impulse-ml
libraries.

It supports:

###### Importing:
 - Loading data from CSV

###### Column transformations:
 - Category transformation

Given string data stacked in examples i.e.: {"Cat1", "Cat2", "Cat3", "Cat1"}
are transformed into binary categories: {<0|1>, <0|1>, <0,1>}
(3 unique categories)

 - CategoryId transformation

Given string data stacked in examples i.e.: {"Cat1", "Cat2", "Cat3", "Cat1"}
are transformed into unique numbers: 0, 1, 2, 3

 - MinMaxScaling [https://en.wikipedia.org/wiki/Feature_scaling#Rescaling]
 - ZScoresScaling [https://en.wikipedia.org/wiki/Feature_scaling#Standardization]
 - MissingData - it can create mean values for missing data in example columns

###### Dataset transformations

 - slicing: for dividing input set and output set
 - splitting: i.e. for train set, dev set and test set

### TODO:
 - use OpenMP
 - implement export to file
 - fix demo paths

## Console usage:

### impulse-ml-dataset-slicer

It slice dataset to input and output .csv files.

```./impulse-ml-dataset-slicer -i ../src/data/data.csv --input-columns 0,1 --output-columns 2,3 -v -o .```

```-i``` - input path to .csv file

```---input-columns``` input columns from 0 to n - 1

```---output-columns``` output columns from 0 to n - 1

```-v``` verbose output (limited to 10 rows)

```-o``` output path

### impulse-ml-dataset-spliter

It split dataset to learning and test set.

```./impulse-ml-dataset-spliter -i ../src/data/data.csv --ratio 0.8 -v -o .```

```-i``` - input path to .csv file

```---ratio``` split ratio, i.e. ```0.8```

```-v``` verbose output (limited to 10 rows)

```-o``` output path

## Build

```git clone git@github.com:impulse-ml/impulse-ml-dataset```

```cd impulse-ml-dataset```

```mkdir build && cd build```

```cmake ..```