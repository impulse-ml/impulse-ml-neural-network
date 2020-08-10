# impulse-ml-neural-network

### When the neural network coding is easy.

This is a neural classifier with convolutional layers option (which are not vectorized yet).

## Minimal classifier example

##### Step 1: Load data.

```c++
       Impulse::Dataset::DatasetBuilder::CSVBuilder datasetBuilder1(
               "../data/mnist_test_1000.csv");
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
    Trainer::MiniBatch<Trainer::Optimizer::Adam> trainer(net);
    trainer.setLearningIterations(30);
    trainer.setVerboseStep(1);
    trainer.setRegularization(0.01);
    trainer.setVerbose(true);
    trainer.setLearningRate(0.01);
```

Available trainers:
```c++
Trainer::MiniBatch
Trainer::Stochastic
```

Available optimizers:
```c++
Trainer::Optimizer::Adam
Trainer::Optimizer::Adadelta
Trainer::Optimizer::Adagrad
Trainer::Optimizer::GradientDescent
Trainer::Optimizer::Nesterov
Trainer::Optimizer::Rmsprop
```

##### Step 4: Train the network.

```c++
    trainer.train(slicedDataset);
```

## Other examples

##### Cost and accuracy.

```c++
    Trainer::CostGradientResult cost = trainer.cost(slicedDataset);
    std::cout << "Cost: " << cost.getCost() << std::endl;
    std::cout << "Accuracy: " << cost.getAccuracy() << std::endl;
```

##### Save trained network to file.

```c++
    Serializer serializer(net);
    serializer.toJSON("../saved/test_mnist_minibatch_gradient_descent.json");
```

##### Restore trained network from file.

```c++
    Builder::ClassifierBuilder builder = Builder::ClassifierBuilder::fromJSON("../saved/test_mnist_minibatch_gradient_descent.json");
    Network::ClassifierNetwork network = builder.getNetwork();
```

##### Forward propagation.

```c++
    std::cout << "Forward:" << std::endl << net.forward(slicedDataset.input.getSampleAt(0)->exportToEigen()) << std::endl;
```

##### Step callback.

```c++
trainer.setStepCallback([&trainer, &testDataset]() {
    Trainer::CostGradientResult cost = trainer.cost(testDataset);
    std::cout << "___STEP___" << std::endl;
    std::cout << "Cost: " << cost.getCost() << std::endl;
    std::cout << "Accuracy: " << cost.getAccuracy() << "%" << std::endl;
});
```

## Layer types

```c++
Layer::Conv             // in ConvBuilder only
Layer::FullyConnected   // in ConvBuilder only 
Layer::Logistic
Layer::MaxPool          // in ConvBuiler only
Layer::Purelin
Layer::Relu
Layer::Softmax          // must be last
Layer::Tanh
```

## Author

Michał Baniowski [banit.pl](https://banit.pl)