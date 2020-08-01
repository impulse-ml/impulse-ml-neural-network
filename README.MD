#impulse-ml-neural-network

### When the neural network coding is easy.

This is a neural classifier with convolutional layers option (not vectorized yet).

## Minimal classifier example

##### Step 1: Load data.

```c++
       Impulse::Dataset::DatasetBuilder::CSVBuilder datasetBuilder1(
               "/home/user/impulse-ml-neural-network/data/mnist_test_1000.csv");
       Impulse::Dataset::Dataset dataset = datasetBuilder1.build();
       Impulse::Dataset::DatasetModifier::DatasetSlicer slicer(dataset);
       slicer.addOutputColumn(0);
       for (int i = 0; i < 28 * 28; i++) {
           slicer.addInputColumn(i + 1);
       }
   
       Impulse::Dataset::SlicedDataset slicedDataset = slicer.slice();
   
       Impulse::Dataset::DatasetModifier::Modifier::Category modifier2(slicedDataset.output);
       modifier2.applyToColumn(0);
```

Image size is 28x28 pixels. Whole CSV has a 28*28 + 1 colums. We slice dataset here to an input and categorized output.

##### Step 2: Create neural network.

```c++
    Builder::ClassifierBuilder builder({28*28});
    builder.createLayer<Layer::Logistic>([](auto * layer) {
        layer->setSize(100);
    });
    builder.createLayer<Layer::Logistic>([](auto * layer) {
        layer->setSize(20);
    });
    builder.createLayer<Layer::Softmax>([](auto * layer) {
        layer->setSize(10);
    });

    Network::ClassifierNetwork net = builder.getNetwork();
```

As you see layers and be configured with size so the builder also.

##### Step 3: Create training method.

```c++
    Trainer::MiniBatchGradientDescent trainer(net);
    trainer.setLearningIterations(30);
    trainer.setVerboseStep(1);
    trainer.setRegularization(0.01);
    trainer.setVerbose(true);
    trainer.setLearningRate(0.01);
    trainer.setOptimizer("adam");
```

##### Step 4: Train the network.

```c++
    trainer.train(slicedDataset);
```

###### Cost and accuracy.

```c++
    Trainer::CostGradientResult cost = trainer.cost(slicedDataset);
    std::cout << "Cost: " << cost.getCost() << std::endl;
    std::cout << "Accuracy: " << cost.getAccuracy() << std::endl;
```

###### Save trained network to file.

```c++
    Serializer serializer(net);
    serializer.toJSON("/home/user/impulse-ml-neural-network/saved/test_mnist_minibatch_gradient_descent.json");
```

###### Restore trained network from file.

```c++
    Builder::ConvBuilder builder = Builder::ConvBuilder::fromJSON("/home/user/impulse-ml-neural-network/saved/test_mnist_minibatch_gradient_descent.json");
    Network::ConvNetwork network = builder.getNetwork();
```

## Author

Michał Baniowski (banit.pl)