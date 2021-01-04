import numpy as np
import matplotlib.pyplot as pp
import os
import tensorflow as tf
from tensorflow.keras import datasets, layers, models, Model, optimizers, preprocessing, backend
from tensorflow.keras.preprocessing import image

class SemiSupervisedGan:
    # This class implements a semi-supervised GAN model in order to detect topographic features with a small training dataset.
    # num_classes: number of classification categories
    # input_shape: the shape of the input file (image_height, image_width, dim)
    # latent_space_sz: the size of the GAN latent space
    # train_path/test_path: the path to the train/test dir
    # lr: the learning rate

    def __init__(self, num_classes, num_samples, input_shape, latent_space_sz = 100, train_path = "data/train/", test_path = "data/test/", lr = 0.0001):
        self.num_classes = num_classes
        self.num_samples = num_samples
        self.input_shape = input_shape
        self.train_path = train_path
        self.latent_space_sz = latent_space_sz
        self.lr = lr
        self.test_path = test_path

        # Training images and labels datasets:
        self.img_dataset = np.empty((0, self.input_shape[0], self.input_shape[1], 1))
        self.labels = np.array([])

        # Testing images and labels datasets:
        self.img_dataset_test = np.empty((0, self.input_shape[0], self.input_shape[1], 1))
        self.labels_test = np.array([])

        # Define all models
        # Supervised and unsupervides discriminators:
        self.sup_disc_model = models.Sequential()
        self.sup_disc_model = models.Sequential()
        # Generator:
        self.generator_model = models.Sequential()
        # Finally, the GAN model:
        self.gan_model = models.Sequential()

    ###
    # This method loads the training data from the self.train_path dir.
    def load_training_data(self):
        # Initialize image and label arrays
        img_dataset = np.empty((0, self.input_shape[0], self.input_shape[1], 1))
        labels = np.array([])

        # Get names of classes (as subdirs in the train_path dir)
        classes = os.listdir(self.train_path)
        print('Loading training data...')

        for class_idx, class_name in enumerate(classes):
        # For each class, read images up to the number of samples (total number of training samples will be num_classes * num_samples)
            for filename in np.sort(os.listdir(self.train_path + "/" + class_name))[:1000]:
                labels = np.append(labels, class_idx)

                # Load the image and convert to array:
                img = image.load_img(self.train_path + class_name + "/" + filename, target_size = (self.input_shape[0], self.input_shape[1]), color_mode = 'grayscale')
                img = image.img_to_array(img)
                # Expand into a tensor and put in array:
                img = np.expand_dims(img, axis = 0)
                img_dataset = np.append(img_dataset, img, axis = 0)

        img_dataset = (img_dataset - (255.0 / 2)) / (255.0 / 2)
        self.img_dataset = img_dataset
        self.labels = labels
        print('Done loading training data.')
        return [img_dataset, labels]

        ###
    # This method loads the training data from the self.train_path dir.
    def load_testing_data(self):
        # Initialize image and label arrays
        img_dataset_test = np.empty((0, self.input_shape[0], self.input_shape[1], 1))
        labels_test = np.array([])

        # Get names of classes (as subdirs in the train_path dir)
        classes = os.listdir(self.test_path)
        print('Loading testing data...')

        for class_idx, class_name in enumerate(classes):
        # For each class, read images up to the number of samples (total number of training samples will be num_classes * num_samples)
            for filename in np.sort(os.listdir(self.test_path + "/" + class_name))[:1000]:
                labels_test = np.append(labels_test, class_idx)

                # Load the image and convert to array:
                img = image.load_img(self.test_path + class_name + "/" + filename, target_size = (self.input_shape[0], self.input_shape[1]), color_mode = 'grayscale')
                img = image.img_to_array(img)
                # Expand into a tensor and put in array:
                img = np.expand_dims(img, axis = 0)
                img_dataset_test = np.append(img_dataset_test, img, axis = 0)

        img_dataset_test = (img_dataset_test - (255.0 / 2)) / (255.0 / 2)
        self.img_dataset_test = img_dataset_test
        self.labels_test = labels_test
        print('Done loading testing data.')

    ###
    # A log-sum activation function for the unsupervised discriminator
    def logsum_activation_fun(self, x):
        buff_sum = backend.sum(backend.exp(x), keepdims = True, axis = -1)
        return buff_sum / (1 + buff_sum)

    ###
    # This method creates the unsupervised and the supervised discriminator models.
    def create_discriminator_models(self):
        # Initialize supervised discriminator model:
        sup_disc_model = models.Sequential()

        sup_disc_model.add(layers.Conv2D(128, (3, 3), strides = (2, 2), padding = "same", input_shape = self.input_shape))
        sup_disc_model.add(layers.LeakyReLU(0.2))
        sup_disc_model.add(layers.Conv2D(128, (3, 3), strides = (2, 2), padding = "same"))
        sup_disc_model.add(layers.LeakyReLU(0.2))
        sup_disc_model.add(layers.Conv2D(128, (3, 3), strides = (2, 2), padding = "same"))
        sup_disc_model.add(layers.LeakyReLU(0.2))
        sup_disc_model.add(layers.Flatten())
        sup_disc_model.add(layers.Dropout(0.4))
        sup_disc_model.add(layers.Dense(self.num_classes, activation = 'softmax'))

        sup_disc_model.compile(optimizer = optimizers.Adam(learning_rate = self.lr, beta_1 = 0.5), loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])

        # Initialize unsupervised discriminator model:
        unsup_disc_model = models.Sequential()
        
        unsup_disc_model.add(layers.Conv2D(128, (3, 3), strides = (2, 2), padding = "same", input_shape = self.input_shape))
        unsup_disc_model.add(layers.LeakyReLU(0.2))
        unsup_disc_model.add(layers.Conv2D(128, (3, 3), strides = (2, 2), padding = "same"))
        unsup_disc_model.add(layers.LeakyReLU(0.2))
        unsup_disc_model.add(layers.Conv2D(128, (3, 3), strides = (2, 2), padding = "same"))
        unsup_disc_model.add(layers.LeakyReLU(0.2))
        unsup_disc_model.add(layers.Flatten())
        unsup_disc_model.add(layers.Dropout(0.4))
        unsup_disc_model.add(layers.Dense(self.num_classes, activation = layers.Lambda(self.logsum_activation_fun)))

        unsup_disc_model.compile(optimizer = optimizers.Adam(learning_rate = self.lr, beta_1 = 0.5), loss = 'binary_crossentropy')

        self.unsup_disc_model = unsup_disc_model
        self.sup_disc_model = sup_disc_model

        return unsup_disc_model, sup_disc_model

    ###
    # This method created the generator (standard sequential) and incroporates the discriminator to the generator to create the GAN
    def create_gan_model(self):
        # First, create the generator:
        # Define generator model:
        generator_model = models.Sequential()

        # Start the generator with 8x8, and increase the resolution two-folds, twice:
        generator_model.add(layers.Dense(8 * 8 * 128, input_shape = (self.latent_space_sz, )))
        generator_model.add(layers.LeakyReLU(0.2))
        generator_model.add(layers.Reshape((8, 8, 128)))
        generator_model.add(layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding = "same"))
        generator_model.add(layers.LeakyReLU(0.2))
        generator_model.add(layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding = "same"))
        generator_model.add(layers.LeakyReLU(0.2))
        generator_model.add(layers.Conv2D(1, (8, 8), activation = 'tanh', padding = "same"))

        self.generator_model = generator_model

        # Second, create the GAN by connecting the generator and discriminators:
        # Set the unsupervised discriminator weights to not-trainable:
        self.unsup_disc_model.trainable = False

        # # Connect models:
        gan_model = Model(generator_model.input, self.unsup_disc_model(generator_model.output))
        # Compile:
        gan_model.compile(optimizers.Adam(learning_rate = self.lr, beta_1 = 0.5), loss = 'binary_crossentropy')

        # Set class variable:
        self.generator_model = generator_model
        self.gan_model = gan_model

        return generator_model, gan_model

    def get_data_subset(self, images, labels, num_samples):
        idx_to_load = np.random.choice(np.shape(images)[0], num_samples, replace = False)

        subset_images = images[idx_to_load,:,:,:]
        subset_labels = labels[idx_to_load]

        return subset_images, subset_labels

    ###
    # This method trains the model using labeled and unlabeled samples.
    def train(self, num_samples_in_batch, num_epoches = 10, verbose = False, print_latent_space = False):
        # First, load the labeled dataset:
        labeled_img_dataset = np.empty((0, self.input_shape[0], self.input_shape[1], 1))
        labeled_img_labels = np.array([])
        # The number of samples per class to load:
        num_samples_per_class = np.int(self.num_samples / self.num_classes)
        print("Number of samples/class:" + str(num_samples_per_class))
        # For each class (label...), load num_samples_per_class samples:
        for lbl_idx in range(self.num_classes):
            data_in_class = self.img_dataset[self.labels == lbl_idx]
            labels_in_class = self.labels[self.labels == lbl_idx]

            class_imgs, class_lbls = self.get_data_subset(data_in_class, labels_in_class, num_samples_per_class)

            labeled_img_dataset = np.append(labeled_img_dataset, class_imgs, axis = 0)
            labeled_img_labels = np.append(labeled_img_labels, class_lbls)


        ########################
        # Now, start training: #
        ########################
        num_training_steps = num_epoches * np.int(np.shape(self.img_dataset)[0] / num_samples_in_batch)

        print("Training...")
        print("Training parameters: " + str(num_epoches) + " epoches, " + str(num_samples_in_batch) + " samples per batch.")
        half_batch = np.int(num_samples_in_batch / 2)

        if verbose == True:
            print("Training step; Sup. discriminator loss; Unsup. discriminator loss (real); Unsup. discriminator (fake) loss; GAN loss:")

        for training_step in range(num_training_steps):
            # Real samples to train the supervised model
            sup_real_imgs, sup_real_lbls = self.get_data_subset(labeled_img_dataset, labeled_img_labels, half_batch)

            # Real samples to train the unsupervised model
            unsup_real_imgs = self.get_data_subset(self.img_dataset, self.labels, half_batch)[0]
            unsup_real_lbls = np.ones([half_batch, 1])
            
            # Fake samples to train the unsupervised model
            latent_space_points = np.random.randn(half_batch, self.latent_space_sz)
            unsup_fake_imgs = self.generator_model.predict(latent_space_points)
            unsup_fake_lbls = np.zeros([half_batch, 1])

            # Train:
            sup_loss, sup_acc = self.sup_disc_model.train_on_batch(sup_real_imgs, sup_real_lbls)
            unsup_loss_real = self.unsup_disc_model.train_on_batch(unsup_real_imgs, unsup_real_lbls)
            unsup_loss_fake = self.unsup_disc_model.train_on_batch(unsup_fake_imgs, unsup_fake_lbls)

            # Update GAN and generator
            gan_data = np.random.randn(num_samples_in_batch, self.latent_space_sz)
            gan_lbls = np.ones([num_samples_in_batch, 1])
            gan_loss = self.gan_model.train_on_batch(gan_data, gan_lbls)

            # Show losses if verbose is True:
            if verbose:
                print("%d: %.2f; %.2f; %.2f; %.2f" % (training_step + 1, sup_loss, unsup_loss_real, unsup_loss_fake, gan_loss))

            # Print performence:
            if (training_step + 1) % (np.int(np.shape(self.img_dataset)[0] / num_samples_in_batch)) == 0:
                print("Training performence:")
                self.sup_disc_model.evaluate(self.img_dataset, self.labels, verbose = True)
            
                # Print fake samples from latent space
                if print_latent_space:
                    curr_ls = self.generator_model.predict(np.random.randn(num_samples_in_batch, self.latent_space_sz))

                    for idx in range(np.int(np.sqrt(num_samples_in_batch)) ** 2):
                        pp.subplot(np.int(np.sqrt(num_samples_in_batch)), np.int(np.sqrt(num_samples_in_batch)), idx + 1)
                        pp.imshow(np.squeeze(curr_ls[idx, :, :, :]), cmap=pp.get_cmap('Greys'))

                    # Save to file:
                    filename = "./fake_samples/fake_samples_at_ts_" + str(training_step + 1) + ".png"
                    pp.savefig(filename)
                    pp.close()

                # Save the generator and the supervised (classifier) models:
                generator_model_file = "generator_model_" + str(training_step + 1) + ".h5"
                unsupervised_disc_model_file = "classifier_model_" + str(training_step + 1) + ".h5"
                self.generator_model.save(generator_model_file)
                self.unsup_disc_model.save(unsupervised_disc_model_file)

    # Test model
    def test(self):
        self.sup_disc_model.evaluate(self.img_dataset_test, self.labels_test, verbose = True)

