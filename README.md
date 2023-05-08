# DLstorm: Deep Learning Framework

## Pip install:
```sh
pip install DLstorm
```

## Layers & DL classes in framework:
- Conv2d
- MaxPool2d
- BatchNorm2d
- Flatten
- Dropout
- Linear
- ReLU
- Softmax
- SgdWithMomentum
- Adam
- CrossEntropyLoss
- Xavier
- He

## Model building:
```py
layers = [
    Conv2d(in_channels=1, out_channels=32,
           kernel_size=3, stride=1, padding='same'),
    BatchNorm2d(32),
    ReLU(),

    Conv2d(in_channels=32, out_channels=64,
           kernel_size=3, stride=1, padding='same'),
    BatchNorm2d(64),
    ReLU(),
    MaxPool2d(kernel_size=2, stride=2),

    Conv2d(in_channels=64, out_channels=64,
           kernel_size=3, stride=1, padding='same'),
    BatchNorm2d(64),
    ReLU(),
    MaxPool2d(kernel_size=2, stride=2),

    Flatten(),

    Linear(in_features=64*7*7, out_features=128),
    ReLU(),
    Linear(128, 64),
    ReLU(),
    Linear(64, 10),
    SoftMax(),
]

model = Model(layers)
```

Or

```py
model = Model()

model.append_layer(Conv2d(in_channels=1, out_channels=32,
                          kernel_size=3, stride=1, padding='same'))
model.append_layer(BatchNorm2d(32))
model.append_layer(ReLU())
model.append_layer(Conv2d(in_channels=32, out_channels=64,
                          kernel_size=3, stride=1, padding='same'))
model.append_layer(BatchNorm2d(64))
model.append_layer(ReLU())
model.append_layer(MaxPool2d(kernel_size=2, stride=2))

model.append_layer(Conv2d(in_channels=64, out_channels=64,
                          kernel_size=3, stride=1, padding='same'))
model.append_layer(BatchNorm2d(64))
model.append_layer(ReLU())
model.append_layer(MaxPool2d(kernel_size=2, stride=2))
model.append_layer(Flatten())
model.append_layer(Linear(in_features=64*7*7, out_features=128))
model.append_layer(ReLU())
model.append_layer(Linear(in_features=128, out_features=64))
model.append_layer(ReLU())
model.append_layer(Linear(in_features=64, out_features=10))
model.append_layer(SoftMax())
```

## Model compile:


```py
model.compile(optimizer=Adam(5e-3, 0.98, 0.999), loss=CrossEntropyLoss(), batch_size=batch_size, metrics=['accuracy'])
```

## Model training:
```py
epochs = 25
history = model.fit(x_train=train_images, y_train=train_labels, x_val=val_images, y_val=val_labels, epochs=epochs)
```