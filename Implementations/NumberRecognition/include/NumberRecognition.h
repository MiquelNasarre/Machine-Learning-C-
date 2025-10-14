#pragma once
#define IMAGE_DIM 784
#define VALUES_DIM 15

enum Set
{
	TESTING,
	TRAINING
};

class NumberRecognition
{
private:

	static unsigned* trainingLabels;
	static unsigned* testingLabels;

	static float** trainingImages;
	static float** testingImages;

	static float** trainingValues;
	static float** testingValues;

	static unsigned n_training;
	static unsigned n_testing;

	static NumberRecognition loader;

	NumberRecognition();
	~NumberRecognition();
public:

	static float** getValues(Set test_train, size_t start_idx, size_t end_idx);

	static float** getImages(Set test_train, size_t start_idx, size_t end_idx);

	static unsigned* getLabels(Set test_train, size_t start_idx, size_t end_idx);

	static void printImage(Set test_train, size_t idx);

	static unsigned getSize(Set test_train);
};