#include "CIFAR10.h"

#include <cstdlib>
#include <cstdio>

// Paths for the CIFAR10 files

#define BATCH_1_PATH CIFAR_PATH "data_batch_1.bin"
#define BATCH_2_PATH CIFAR_PATH "data_batch_2.bin"
#define BATCH_3_PATH CIFAR_PATH "data_batch_3.bin"
#define BATCH_4_PATH CIFAR_PATH "data_batch_4.bin"
#define BATCH_5_PATH CIFAR_PATH "data_batch_5.bin"
#define TESTING_PATH CIFAR_PATH "test_batch.bin"

// Static variables initialization

uint8_t** CIFAR10::Batch1Images = nullptr;
uint8_t** CIFAR10::Batch2Images = nullptr;
uint8_t** CIFAR10::Batch3Images = nullptr;
uint8_t** CIFAR10::Batch4Images = nullptr;
uint8_t** CIFAR10::Batch5Images = nullptr;
uint8_t** CIFAR10::testingImages = nullptr;

uint8_t* CIFAR10::Batch1Labels = nullptr;
uint8_t* CIFAR10::Batch2Labels = nullptr;
uint8_t* CIFAR10::Batch3Labels = nullptr;
uint8_t* CIFAR10::Batch4Labels = nullptr;
uint8_t* CIFAR10::Batch5Labels = nullptr;
uint8_t* CIFAR10::testingLabels = nullptr;

CIFAR10 CIFAR10::helper;

// When the program ends the helper will call the destructor,
// freeing the batches that were loaded.

CIFAR10::~CIFAR10()
{
	freeBatch(0);
	freeBatch(1);
	freeBatch(2);
	freeBatch(3);
	freeBatch(4);
	freeBatch(5);
}

// Loads the batch from the files into the pointers.

void CIFAR10::initializeBatch(unsigned batch)
{
	uint8_t** images = (uint8_t**)calloc(IMAGES_PER_BATCH, sizeof(uint8_t*));
	uint8_t* labels = (uint8_t*)calloc(IMAGES_PER_BATCH, sizeof(uint8_t));
	const char* filepath;

	switch (batch)
	{
	case 0:
		filepath = TESTING_PATH;
		testingImages = images;
		testingLabels = labels;
		break;

	case 1:
		filepath = BATCH_1_PATH;
		Batch1Images = images;
		Batch1Labels = labels;
		break;

	case 2:
		filepath = BATCH_2_PATH;
		Batch2Images = images;
		Batch2Labels = labels;
		break;

	case 3:
		filepath = BATCH_3_PATH;
		Batch3Images = images;
		Batch3Labels = labels;
		break;

	case 4:
		filepath = BATCH_4_PATH;
		Batch4Images = images;
		Batch4Labels = labels;
		break;

	case 5:
		filepath = BATCH_5_PATH;
		Batch5Images = images;
		Batch5Labels = labels;
		break;

	default:
		return;
	}

	FILE* file = fopen(filepath, "rb");
	if (!file)
		goto cleanup;

	for (unsigned i = 0; i < IMAGES_PER_BATCH; i++)
	{
		if (fread(&labels[i], sizeof(uint8_t), 1, file) != 1)
			goto cleanup;

		images[i] = (uint8_t*)calloc(IMG_TOTAL_UINT8, sizeof(uint8_t));
		if (fread(images[i], sizeof(uint8_t), IMG_TOTAL_UINT8, file) != IMG_TOTAL_UINT8)
			goto cleanup;
	}
	fclose(file);
	return;

cleanup:
	if(file)
		fclose(file);
	free(labels);
	for (unsigned i = 0; i < IMAGES_PER_BATCH && images[i]; i++)
		free(images[i]);
	free(images);

	switch (batch)
	{
	case 0:
		testingImages = nullptr;
		testingLabels = nullptr;
		return;

	case 1:
		Batch1Images = nullptr;
		Batch1Labels = nullptr;
		return;

	case 2:
		Batch2Images = nullptr;
		Batch2Labels = nullptr;
		return;

	case 3:
		Batch3Images = nullptr;
		Batch3Labels = nullptr;
		return;

	case 4:
		Batch4Images = nullptr;
		Batch4Labels = nullptr;
		return;

	case 5:
		Batch5Images = nullptr;
		Batch5Labels = nullptr;
		return;

	default:
		return;
	}
}

// Returns the pointer to the training images as a uint8_t**. -1 for test batch.

uint8_t** CIFAR10::getBatchImages(unsigned batch)
{
	switch (batch)
	{
	case 0:
		if (!testingImages)
			initializeBatch(batch);
		return testingImages;

	case 1:
		if (!Batch1Images)
			initializeBatch(batch);
		return Batch1Images;

	case 2:
		if (!Batch2Images)
			initializeBatch(batch);
		return Batch2Images;

	case 3:
		if (!Batch3Images)
			initializeBatch(batch);
		return Batch3Images;

	case 4:
		if (!Batch4Images)
			initializeBatch(batch);
		return Batch4Images;

	case 5:
		if (!Batch5Images)
			initializeBatch(batch);
		return Batch5Images;

	default:
		return nullptr;
	}
}

// Returns the pointer to the training labels as an uint8_t*. -1 for test batch.

uint8_t* CIFAR10::getBatchLabels(unsigned batch)
{
	switch (batch)
	{
	case 0:
		if (!testingImages)
			initializeBatch(batch);
		return testingLabels;

	case 1:
		if (!Batch1Images)
			initializeBatch(batch);
		return Batch1Labels;

	case 2:
		if (!Batch2Images)
			initializeBatch(batch);
		return Batch2Labels;

	case 3:
		if (!Batch3Images)
			initializeBatch(batch);
		return Batch3Labels;

	case 4:
		if (!Batch4Images)
			initializeBatch(batch);
		return Batch4Labels;

	case 5:
		if (!Batch5Images)
			initializeBatch(batch);
		return Batch5Labels;

	default:
		return nullptr;
	}
}

// Frees the allocated space for the batch (pointers no longer valid). -1 for test batch.

void CIFAR10::freeBatch(unsigned batch)
{
	switch (batch)
	{
	case 0:
		if (testingImages)
		{
			for (unsigned i = 0; i < IMAGES_PER_BATCH; i++)
				free(testingImages[i]);

			free(testingImages);
			free(testingLabels);
			testingImages = nullptr;
			testingLabels = nullptr;
		}
		return;

	case 1:
		if (Batch1Images)
		{
			for (unsigned i = 0; i < IMAGES_PER_BATCH; i++)
				free(Batch1Images[i]);

			free(Batch1Images);
			free(Batch1Labels);
			Batch1Images = nullptr;
			Batch1Labels = nullptr;
		}
		return;

	case 2:
		if (Batch2Images)
		{
			for (unsigned i = 0; i < IMAGES_PER_BATCH; i++)
				free(Batch2Images[i]);

			free(Batch2Images);
			free(Batch2Labels);
			Batch2Images = nullptr;
			Batch2Labels = nullptr;
		}
		return;

	case 3:
		if (Batch3Images)
		{
			for (unsigned i = 0; i < IMAGES_PER_BATCH; i++)
				free(Batch3Images[i]);

			free(Batch3Images);
			free(Batch3Labels);
			Batch3Images = nullptr;
			Batch3Labels = nullptr;
		}
		return;

	case 4:
		if (Batch4Images)
		{
			for (unsigned i = 0; i < IMAGES_PER_BATCH; i++)
				free(Batch4Images[i]);

			free(Batch4Images);
			free(Batch4Labels);
			Batch4Images = nullptr;
			Batch4Labels = nullptr;
		}
		return;

	case 5:
		if (Batch5Images)
		{
			for (unsigned i = 0; i < IMAGES_PER_BATCH; i++)
				free(Batch5Images[i]);

			free(Batch5Images);
			free(Batch5Labels);
			Batch5Images = nullptr;
			Batch5Labels = nullptr;
		}
		return;

	default:
		return;
	}
}