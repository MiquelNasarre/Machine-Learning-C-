#include "NeuralNetwork.h"
#include "NumberRecognition.h"
#include "Timer.h"

#include <cstdio>
#include <random>

int main()
{
	Timer timer;

	const unsigned training_set_size = 60000;
	const unsigned training_set_start = 0;

	const unsigned testing_set_size = 10000;
	const unsigned testing_set_start = 0;

	const unsigned num_layers = 4;
	const unsigned layers[] = { IMAGE_DIM, 16, 16, 10 };
	const char* NN_name = "3B1B";
	
	const unsigned epoch = 5;
	const float learning_rate = 0.000001f;
	const float weight_decay = 1e-8f;
	const float momentum = 0.9f;

	//float* const* training_values = NumberRecognition::getValues(TRAINING, training_set_start, training_set_size + training_set_start);
	float* const* training_images = NumberRecognition::getImages(TRAINING, training_set_start, training_set_size + training_set_start);
	unsigned const* training_labels = NumberRecognition::getLabels(TRAINING, training_set_start, training_set_size + training_set_start);

	//float* const* testing_values = NumberRecognition::getValues(TESTING, testing_set_start, testing_set_size + testing_set_start);
	float* const* testing_images = NumberRecognition::getImages(TESTING, testing_set_start, testing_set_size + testing_set_start);
	unsigned const* testing_labels = NumberRecognition::getLabels(TESTING, testing_set_start, testing_set_size + testing_set_start);

	float* const* training_data = training_images;
	float* const* testing_data = testing_images;

	NeuralNetwork NN(num_layers, layers);
	NN.setLearningRate(learning_rate);
	NN.setWeightDecay(weight_decay);
	NN.setMomentum(momentum);
	NN.loadWeights(NN_name);

	printf("Neural Network correctly generated, Initial weights:\n");
	NN.printWeights();

	printf("\nInitial prediction Error: %.4f", NN.computePredictionError(testing_set_size, testing_data, testing_labels));
	printf("\nInitial Prediction Rate:  %.4f\n", NN.computePredictionRate(testing_set_size, testing_data, testing_labels));

	NN.feedData(training_set_size, training_data, training_labels);

	printf("\nStarted training for %u epoch ...\n", epoch);

	timer.mark();
	NN.trainWeights(epoch);
	float time = timer.check();

	printf("\nFinished training for %u epoch in %.4fs, updated weights:\n", epoch, time);
	NN.printWeights();

	printf("\nUpdated Prediction Error: %.4f", NN.computePredictionError(testing_set_size, testing_data, testing_labels));
	printf("\nUpdated Prediction Rate:  %.4f\n\n", NN.computePredictionRate(testing_set_size, testing_data, testing_labels));

	NN.storeWeights(NN_name);

	printf("Do you wanna see some examples? Y:N\n");
	char c;
	scanf("%c", &c);
	if (c != 'y' && c != 'Y')
		return 0;

	srand((uint32_t)Timer::get_system_time_ns());
	for (int i = 0; i < 10; i++)
	{
		unsigned int idx = rand() % 10000;
		printf("\n\nImage %i (label = %hhu):\n\n", idx, testing_labels[idx]);
		NumberRecognition::printImage(TESTING, idx);

		float* p = NN.predictCase(testing_data[idx]);
		printf("\nprediction = { %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f }", p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8], p[9]);
	}

	return 0;
}