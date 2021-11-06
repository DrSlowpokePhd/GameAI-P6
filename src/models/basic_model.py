from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers

# Define a sequential model here
model = models.Sequential()
# add more layers to the model...
# CNN with one conv2d layer and one maxpool2d layer
model.add(layers.Conv2D(9, (7, 7), activation='relu', strides=(3, 3)))
model.add(layers.MaxPooling2D(pool_size=(5, 5), strides=(3, 3)))
model.add(layers.Conv2D(27, (7, 7), activation='relu', strides=(3, 3)))
model.add(layers.Flatten())
model.add(layers.Dense(4, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

# Then, call model.compile()
model.compile(
    loss='binary_crossentropy',
    optimizer=optimizers.RMSprop(lr=1e-4),
    metrics=['acc']
)

# Finally, train this compiled model by running:
# python train.py
