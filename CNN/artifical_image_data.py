
from keras.preprocessing.image import ImageDataGenerator

img_width, img_height = 150, 150

train_data_dir = 'home/akshay/data' # 650 samples

train_datagen_augmented = ImageDataGenerator(
        rescale=1./255,        # normalize pixel values to [0,1]
        shear_range=0.2,       # randomly applies shearing transformation
        zoom_range=0.2,        # randomly applies shearing transformation
        horizontal_flip=True)  # randomly flip the images

# same code as before
train_generator_augmented = train_datagen_augmented.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=32,
        class_mode='binary')

# We can fit the same model on new data that is generated.

model.fit_generator(
        train_generator_augmented,
        samples_per_epoch=nb_train_samples,
        nb_epoch=nb_epoch,
        validation_data=validation_generator,
        nb_val_samples=nb_validation_samples)

# Other ways of adding artificial data 

# mean image subtraction - works well if your color or intensity distributions is not consistent throughout the image (e.g. only centered objects)
# per-channel normalization (subtract mean, divide by standard deviation), pretty standard, useful for variable sized input where you can't use 1.
# per-channel mean subtraction - good for variable sized input where you can't use 1 and don't want to make too many assumptions about the distribution.
# whitening (turn the distribution into a normal distribution, sometimes as easy as normalization but only if it's already normally distributed). Maybe others can weigh in on cases where whitening is not a good idea.
# Dimensionality Reduction (e.g. Principal component analysis). You're basically transforming your data into a compressed space with less dimensions, you control the amount of loss and use that as your input to your network. 


# - Data Augmentation

# rotation: random with angle between 0° and 360° (uniform)
# translation: random with shift between -10 and 10 pixels (uniform)
# rescaling: random with scale factor between 1/1.6 and 1.6 (log-uniform)
# flipping: yes or no (bernoulli)
# shearing: random with angle between -20° and 20° (uniform)
# stretching: random with stretch factor between 1/1.3 and 1.3 (log-uniform)


# datagen = ImageDataGenerator(
#         rotation_range=40,
#         width_shift_range=0.2,
#         height_shift_range=0.2,
#         rescale=1./255,
#         shear_range=0.2,
#         zoom_range=0.2,
#         horizontal_flip=True,
#         fill_mode='nearest')

# rotation_range is a value in degrees (0-180), a range within which to randomly rotate pictures
# width_shift and height_shift are ranges (as a fraction of total width or height) within which to randomly translate pictures vertically or horizontally
# rescale is a value by which we will multiply the data before any other processing. Our original images consist in RGB coefficients in the 0-255, but such values would be too high for our models to process (given a typical learning rate), so we target values between 0 and 1 instead by scaling with a 1/255. factor.
# shear_range is for randomly applying shearing transformations
# zoom_range is for randomly zooming inside pictures
# horizontal_flip is for randomly flipping half of the images horizontally --relevant when there are no assumptions of horizontal assymetry (e.g. real-world pictures).
# fill_mode is the strategy used for filling in newly created pixels, which can appear after a rotation or a width/height shift.

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img


# img = load_img('data/train/cats/cat.0.jpg')  # this is a PIL image
# x = img_to_array(img)  # this is a Numpy array with shape (3, 150, 150)
# x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, 150, 150)

# the .flow() command below generates batches of randomly transformed images

# i = 0
# for batch in datagen.flow(x, batch_size=1,
#                           save_to_dir='preview', save_prefix='cat', save_format='jpeg'):
#     i += 1
#     if i > 20:
#         break  # otherwise the generator would loop indefinitely

datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')
