#include "ImageRecognition.h"
#include "CIFAR10.h"
#include "Image.h"

#include <cstdlib>

unsigned* ImageRecognition::trainingLabels = nullptr;
unsigned* ImageRecognition::testingLabels = nullptr;

float** ImageRecognition::trainingImages = nullptr;
float** ImageRecognition::testingImages = nullptr;

ImageRecognition ImageRecognition::helper;

ImageRecognition::ImageRecognition()
{
	testingImages = (float**)calloc(N_TEST_DATA, sizeof(float*));
	testingLabels = (unsigned*)calloc(N_TEST_DATA, sizeof(unsigned));

	trainingImages = (float**)calloc(N_TRAIN_DATA, sizeof(float*));
	trainingLabels = (unsigned*)calloc(N_TRAIN_DATA, sizeof(unsigned));

	uint8_t* labels = CIFAR10::getBatchLabels(0);
	for (unsigned i = 0; i < IMAGES_PER_BATCH; i++)
		testingLabels[i] = unsigned(labels[i]);
	CIFAR10::freeBatch(0);

	unsigned j = 0;
	for (unsigned batch = 1; batch <= 5; batch++)
	{
		labels = CIFAR10::getBatchLabels(batch);
		for (unsigned i = 0; i < IMAGES_PER_BATCH; i++)
			trainingLabels[j++] = unsigned(labels[i]);

		CIFAR10::freeBatch(batch);
	}
}

ImageRecognition::~ImageRecognition()
{
	freeCache(TESTING);
	freeCache(TRAINING);

	free(trainingImages);
	free(testingImages);
	free(trainingLabels);
	free(testingLabels);
}

void ImageRecognition::apply_normalizations(float* image)
{
	return; // No normalisations so far
}

void ImageRecognition::freeCache(Set test_train)
{
	float** cache = test_train == TRAINING ? trainingImages : testingImages;
	unsigned size = test_train == TRAINING ? N_TRAIN_DATA : N_TEST_DATA;

	for (unsigned i = 0; i < size; i++)
		if (cache[i])
		{
			free(cache[i]);
			cache[i] = nullptr;
		}
}

float** ImageRecognition::getImages(Set test_train, size_t start_idx, size_t end_idx)
{
	if (end_idx > (test_train == TRAINING ? N_TRAIN_DATA : N_TEST_DATA) || start_idx >= end_idx)
		return nullptr;

	uint8_t** raw_images;

	switch (test_train)
	{
	case TESTING:
		for (unsigned i = (unsigned)start_idx; i < end_idx; i++)
		{
			if (!testingImages[i])
			{
				raw_images = CIFAR10::getBatchImages(0);
				testingImages[i] = (float*)calloc(IMAGE_DIM, sizeof(float));
				for (unsigned p = 0; p < IMAGE_DIM; p++)
					testingImages[i][p] = float(raw_images[i][p]) / 255.f;

				apply_normalizations(testingImages[i]);
			}
		}
		CIFAR10::freeBatch(0);
		return &testingImages[start_idx];

	case TRAINING:
		
		for (unsigned i = (unsigned)start_idx; i < end_idx; i++)
		{
			if (!trainingImages[i])
			{
				raw_images = CIFAR10::getBatchImages(i / IMAGES_PER_BATCH + 1u);
				trainingImages[i] = (float*)calloc(IMAGE_DIM, sizeof(float));
				for (unsigned p = 0; p < IMAGE_DIM; p++)
					trainingImages[i][p] = float(raw_images[i % IMAGES_PER_BATCH][p]) / 255.f;

				apply_normalizations(trainingImages[i]);
			}
		}
		for(unsigned batch = 1; batch <= 5; batch++)
			CIFAR10::freeBatch(batch);			

		return &trainingImages[start_idx];

	default:
		return nullptr;
	}
}

unsigned* ImageRecognition::getLabels(Set test_train, size_t start_idx, size_t end_idx)
{
	if (end_idx > (test_train == TRAINING ? N_TRAIN_DATA : N_TEST_DATA) || start_idx >= end_idx)
		return nullptr;

	return test_train == TRAINING ? &trainingLabels[start_idx] : &testingLabels[start_idx];
}

void ImageRecognition::saveImage(Set test_train, size_t idx)
{
	if (idx >= (test_train == TRAINING ? N_TRAIN_DATA : N_TEST_DATA))
		return;

	uint8_t* buffer;

	if (test_train == TESTING)
		buffer = CIFAR10::getBatchImages(0)[idx];
	else
		buffer = CIFAR10::getBatchImages((unsigned)idx / IMAGES_PER_BATCH + 1u)[(unsigned)idx % IMAGES_PER_BATCH];


	Image image(IMG_COLS * 10, IMG_ROWS * 10);
	Color* pixels = image.getPixels();

	for (unsigned p = 0; p < IMG_PIXELS; p++)
		for (unsigned i = 0; i < 100; i++)
		{
			unsigned col = 10 * (p % IMG_COLS) + (i % 10);
			unsigned row = 10 * (p / IMG_COLS) + (i / 10);
			pixels[10 * IMG_COLS * row + col] = { buffer[p], buffer[IMG_PIXELS + p], buffer[2 * IMG_PIXELS + p], 255 };
		}

	image.save("images/image %u from %s set", idx, test_train == TESTING ? "testing" : "training");

	if (test_train == TESTING)
		CIFAR10::freeBatch(0);
	else
		CIFAR10::freeBatch((unsigned)idx / IMAGES_PER_BATCH + 1u);
}