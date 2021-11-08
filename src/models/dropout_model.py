from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers

# Define your dropout model here
model = models.Sequential()

model.add(layers.Conv2D(16, (5, 5), activation='relu', strides=(1, 1)))
model.add(layers.MaxPooling2D(pool_size=(7, 7), strides=(1, 1)))
model.add(layers.Conv2D(32, (3, 3), activation='relu', strides=(1, 1)))
model.add(layers.MaxPooling2D(pool_size=(7, 7)))
# add a dropout layer here
model.add(layers.Dropout(0.15))
model.add(layers.Flatten())
model.add(layers.Dense(4, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))


# Then, call model.compile()
model.compile(
    loss='binary_crossentropy',
    optimizer=optimizers.RMSprop(lr=1e-4),
    metrics=['acc']
)

# Train this compiled model by modifying basic_train 
# to import this model, then run:
#   python train.py