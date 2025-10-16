#include "ImageRecognition.h"
#include "NeuralNetwork.h"
#include "Timer.h"

#include <cstdio>

int main()
{
	const unsigned training_set_size = 50000;
	const unsigned training_set_start = 0;

	const unsigned testing_set_size = 10000;
	const unsigned testing_set_start = 0;

	const unsigned num_layers = 4;
	const unsigned layers[] = { IMAGE_DIM, 64, 32, 10 };
	const char* NN_name = "Training set";

	const unsigned epoch = 15;
	const float learning_rate = 0.0001f;
	const float weight_decay = 0.0f;
	const float momentum = 0.93f;
	const float lr_alpha = 0.011f;

	float* const* training_data = ImageRecognition::getImages(TRAINING, training_set_start, training_set_size + training_set_start);
	unsigned const* training_labels = ImageRecognition::getLabels(TRAINING, training_set_start, training_set_size + training_set_start);

	float* const* testing_data = ImageRecognition::getImages(TESTING, testing_set_start, testing_set_size + testing_set_start);
	unsigned const* testing_labels = ImageRecognition::getLabels(TESTING, testing_set_start, testing_set_size + testing_set_start);

	NeuralNetwork NN(num_layers, layers);
	NN.setLearningRate(learning_rate);
	NN.setWeightDecay(weight_decay);
	NN.setMomentum(momentum);
	NN.setLRalpha(lr_alpha);
	NN.loadWeights(NN_name);

	printf("Neural Network correctly generated, Initial weights:\n");
	NN.printWeights();

	printf("\nInitial prediction Error: %.4f", NN.computePredictionError(testing_set_size, testing_data, testing_labels));
	printf("\nInitial Prediction Rate:  %.4f\n", NN.computePredictionRate(testing_set_size, testing_data, testing_labels));

	//const char* meta[] =
	//{
	//		"airplane",
	//		"automobile",
	//		"bird",
	//		"cat",
	//		"deer",
	//		"dog",
	//		"frog",
	//		"horse",
	//		"ship",
	//		"truck",
	//};
	//
	//while (true)
	//{
	//	int i;
	//	printf("Select an example: ");
	//	scanf("%i", &i);
	//	if (i < 0 || i >= 10000)
	//		return 0;
	//
	//	ImageRecognition::saveImage(TESTING, i);
	//
	//	float* p = NN.predictCase(testing_data[i]);
	//	printf("\nlabel = %i (%s)", testing_labels[i], meta[testing_labels[i]]);
	//	printf("\nprediction = { %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f }\n\n", p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8], p[9]);
	//
	//}

	NN.feedData(training_set_size, training_data, training_labels);

	ImageRecognition::freeCache(TRAINING);

	printf("\nStarted training...\n");

	Timer timer;
	NN.trainWeights(epoch);
	float time = timer.check();

	printf("\nFinished training in %.4fs, updated weights:\n", time);
	NN.printWeights();

	printf("\nUpdated Prediction Error: %.4f", NN.computePredictionError(testing_set_size, testing_data, testing_labels));
	printf("\nUpdated Prediction Rate:  %.4f\n\n", NN.computePredictionRate(testing_set_size, testing_data, testing_labels));

	NN.storeWeights(NN_name);
}