#pragma once
#include <cstdint>

/* CIFAR10 DATABASE HELPER CLASS HEADER
-------------------------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------------------------
This class is a set of handy static functions to deal with the CIFAR10 database.
This is stored in files in the data/ folder of the project and contains a big
set of training data for image recognintion.

This class is meant for easy manipulation of the data in such set. Just call the 
functions getBatchImages and getBatchLabels for batches from 0 to 6 with 10'000 
pictures each, where 0 is for the test batch.

To free some space in RAM you can call freeBatch() when you no longer need 
the pointer to the data, the data is given raw as it appears on the binary files.
-------------------------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------------------------
*/

// Expected rows and columns to read from the dataset

#define IMG_CHANNELS		3
#define IMG_ROWS			32u
#define IMG_COLS			32u
#define IMG_PIXELS			(IMG_ROWS * IMG_COLS)
#define IMG_TOTAL_UINT8		(IMG_PIXELS * IMG_CHANNELS)
#define IMAGES_PER_BATCH	10000

// Simple static class to handle the loading of the CIFAR 10 database for image recognintion.
class CIFAR10
{
private:

	static uint8_t** Batch1Images;		// CIFAR pointer to the batch 1 images
	static uint8_t** Batch2Images;		// CIFAR pointer to the batch 2 images
	static uint8_t** Batch3Images;		// CIFAR pointer to the batch 3 images
	static uint8_t** Batch4Images;		// CIFAR pointer to the batch 4 images
	static uint8_t** Batch5Images;		// CIFAR pointer to the batch 5 images
	static uint8_t** testingImages;		// CIFAR pointer to the testing images

	static uint8_t* Batch1Labels;		// CIFAR pointer to the batch 1 labels
	static uint8_t* Batch2Labels;		// CIFAR pointer to the batch 2 labels
	static uint8_t* Batch3Labels;		// CIFAR pointer to the batch 3 labels
	static uint8_t* Batch4Labels;		// CIFAR pointer to the batch 4 labels
	static uint8_t* Batch5Labels;		// CIFAR pointer to the batch 5 labels
	static uint8_t* testingLabels;		// CIFAR pointer to the testing labels

	static CIFAR10 helper;	// Helper to call the destructor at the end of the program

	// When the program ends the helper will call the destructor,
	// freeing the batches that were loaded.
	~CIFAR10();
	CIFAR10() = default;

	// Loads the batch from the files into the pointers.
	static void initializeBatch(unsigned batch);
public:

	// Returns the pointer to the training images as a uint8_t**. 0 for test batch.
	static uint8_t** getBatchImages(unsigned batch);

	// Returns the pointer to the training labels as an uint8_t*. 0 for test batch.
	static uint8_t* getBatchLabels(unsigned batch);

	// Frees the allocated space for the batch (pointers no longer valid). 0 for test batch.
	static void freeBatch(unsigned batch);
	
};
