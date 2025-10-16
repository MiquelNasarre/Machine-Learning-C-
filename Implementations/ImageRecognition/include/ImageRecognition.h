#pragma once
#define IMAGE_DIM 3072

#define N_TEST_DATA  10000
#define N_TRAIN_DATA 50000


enum Set
{
	TESTING,
	TRAINING
};

class ImageRecognition
{
private:

	static unsigned* trainingLabels;
	static unsigned* testingLabels;

	static float** trainingImages;
	static float** testingImages;

	static ImageRecognition helper;

	ImageRecognition();
	~ImageRecognition();

	static void apply_normalizations(float* image);
public:

	static void freeCache(Set test_train);

	static float** getImages(Set test_train, size_t start_idx, size_t end_idx);

	static unsigned* getLabels(Set test_train, size_t start_idx, size_t end_idx);


	static void saveImage(Set test_train, size_t idx);
};