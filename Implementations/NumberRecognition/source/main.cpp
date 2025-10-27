#include "NeuralNetwork.h"
#include "NumberRecognition.h"
#include "Image.h"
#include "Timer.h"

#include <cstdio>
#include <random>

//#define _TRAINING
#define SHOWCASE "96.25%", "CV_3B1B"

static void storeWeightsImages(float*** weights, unsigned n_neurons)
{
	Image image(280, 280);
	Color* pixels = image.getPixels();
	for (unsigned i = 0; i < 16; i++)
	{
		for (unsigned row = 0; row < 28; row++)
			for (unsigned col = 0; col < 28; col++)
			{
				Color c = Color::Black;

				float weight = 5 * weights[0][i][row * 28 + col];

				if (weight < -1.f)weight = -1.f;
				if (weight > 1.f)weight = 1.f;

				weight = (weight + 1) / 2;

				c.R = unsigned char((1 - weight) * 255);
				c.B = unsigned char(weight * 255);

				for (unsigned p = 0; p < 100; p++)
					pixels[((10 * row) + (p / 10)) * 280 + (10 * col + p % 10)] = c;
			}

		image.save("images/weights_%u", i);
	}
}

int main()
{
#ifdef _TRAINING
	Timer timer;

	const unsigned training_set_size = 60000;
	const unsigned training_set_start = 0;

	const unsigned testing_set_size = 10000;
	const unsigned testing_set_start = 0;

	const unsigned num_layers = 4;
	const unsigned layers[] = { IMAGE_DIM, 16, 16, 10 };
	const char* NN_name = "Training set";
	
	const unsigned epoch = 10;
	const float learning_rate = 0.0004f;
	const float weight_decay = 0.f;
	const float momentum = 0.85f;
	const float lr_alpha = 0.01f;

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
	NN.setLRalpha(lr_alpha);

	//NN.loadWeights(NN_name);

	printf("Neural Network correctly generated, Initial weights:\n");
	NN.printWeights();

	printf("\nInitial Prediction Error: %.4f", NN.computePredictionError(testing_set_size, testing_data, testing_labels));
	printf("\nInitial Prediction Rate:  %.2f%c\n", NN.computePredictionRate(testing_set_size, testing_data, testing_labels) * 100.f, 37);

	NN.feedData(training_set_size, training_data, training_labels);

	printf("\nStarted training weights ...\n");

	NN.trainWeights(epoch);
	//NN.train_HyperParamCalibration(50, 20);
	float time = timer.check();

	printf("\nFinished training in %.4fs, updated weights:\n", time);
	NN.printWeights();

	printf("\nUpdated Prediction Error: %.4f", NN.computePredictionError(testing_set_size, testing_data, testing_labels));
	printf("\nUpdated Prediction Rate:  %.2f%c\n\n", NN.computePredictionRate(testing_set_size, testing_data, testing_labels) * 100.f, 37);

	NN.storeWeights(NN_name);

#else

	NeuralNetwork NN(SHOWCASE);

	printf("Neural Network '%s' from file '%s' has been successfully loaded. Weights:\n", SHOWCASE);
	NN.printWeights();
	//storeWeightsImages(NN.getWeights(), 16);

	const unsigned testing_set_size = NumberRecognition::getSize(TESTING);
	float* const* testing_data = NumberRecognition::getImages(TESTING, 0, testing_set_size);
	unsigned const* testing_labels = NumberRecognition::getLabels(TESTING, 0, testing_set_size);

	printf("\nTest Set Prediction Error: %.4f", NN.computePredictionError(testing_set_size, testing_data, testing_labels));
	printf("\nTest Set Prediction Rate:  %.4f\n\n", NN.computePredictionRate(testing_set_size, testing_data, testing_labels));

#endif

repeat:
	printf("\n\nDo you wanna see some examples? Y:N\n");
	char c;
	scanf("\n%c", &c);
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
	goto repeat;
}