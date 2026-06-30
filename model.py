from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv3D, MaxPooling3D, UpSampling3D, Concatenate
from tensorflow.keras.optimizers import Adam


def build_model(input_shape=(128, 128, 3)):
    inputs = Input(shape=input_shape)

    # Encoder
    conv1 = Conv3D(32, (3, 3), activation="relu", padding="same")(inputs)
    conv1 = Conv3D(32, (3, 3), activation="relu", padding="same")(conv1)
    pool1 = MaxPooling3D((2, 2))(conv1)

    conv2 = Conv3D(64, (3, 3), activation="relu", padding="same")(pool1)
    conv2 = Conv3D(64, (3, 3), activation="relu", padding="same")(conv2)
    pool2 = MaxPooling3D((2, 2))(conv2)

    conv3 = Conv3D(128, (3, 3), activation="relu", padding="same")(pool2)
    conv3 = Conv3D(128, (3, 3), activation="relu", padding="same")(conv3)
    pool3 = MaxPooling3D((2, 2))(conv3)

    # Bottleneck
    conv4 = Conv3D(256, (3, 3), activation="relu", padding="same")(pool3)
    conv4 = Conv3D(256, (3, 3), activation="relu", padding="same")(conv4)

    # Decoder
    up5 = UpSampling3D((2, 2))(conv4)
    up5 = Conv3D(128, (2, 2), activation="relu", padding="same")(up5)
    merge5 = Concatenate(axis=3)([conv3, up5])
    conv5 = Conv3D(128, (3, 3), activation="relu", padding="same")(merge5)
    conv5 = Conv3D(128, (3, 3), activation="relu", padding="same")(conv5)

    up6 = UpSampling3D((2, 2))(conv5)
    up6 = Conv3D(64, (2, 2), activation="relu", padding="same")(up6)
    merge6 = Concatenate(axis=3)([conv2, up6])
    conv6 = Conv3D(64, (3, 3), activation="relu", padding="same")(merge6)
    conv6 = Conv3D(64, (3, 3), activation="relu", padding="same")(conv6)

    up7 = UpSampling3D((2, 2))(conv6)
    up7 = Conv3D(32, (2, 2), activation="relu", padding="same")(up7)
    merge7 = Concatenate(axis=3)([conv1, up7])
    conv7 = Conv3D(32, (3, 3), activation="relu", padding="same")(merge7)
    conv7 = Conv3D(32, (3, 3), activation="relu", padding="same")(conv7)

    outputs = Conv3D(1, (1, 1), activation="sigmoid")(conv7)

    model = Model(inputs=inputs, outputs=outputs)

    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return model


if __name__ == "__main__":
    model = build_model()
    model.summary()