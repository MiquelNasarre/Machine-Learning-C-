#include "NeuralNetwork.h"
#include "LinearAlgebra.h"
#include "Timer.h"

#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cstdio>
#include <cmath>

// Keeps track of the version for storage file compatibility
constexpr uint32_t VERSION = 0x02UL;

// This struct contains all the matrix needed for the Neural Network to
// perform operations, it is stored inside the class masked as a void*.
struct MATRIX_DATA
{
	unsigned num_layers = 0u;				// Stores the number of layers
	unsigned* layer_sizes = nullptr;		// Stores the size for each layer

	
	Matrix* Weights = nullptr;				// Stores the weights matrix for each layer
	Matrix* Velocity = nullptr;				// Stores the velocity matrix for each layer
	Matrix* Gradient = nullptr;				// Stores the gradients for the weights

	Vector* biases = nullptr;				// Stores the biases vector for each layer
	Vector* b_velocity = nullptr;			// Stores the velocity vector for each layer bias
	Vector* b_gradient = nullptr;			// Stores the gradient for the biases

	Vector* error_signal = nullptr;			// Stores the error signal vector for each layer
	Vector* layers = nullptr;				// Stores the nodes vector for each Layer

	Vector& output_layer;					// Output layer reference for convenience
	Vector& input_layer;					// Input layer reference for convenience

	// Initialises all the matrices and vectors
	MATRIX_DATA(const unsigned new_num_layers, const unsigned* new_layer_sizes) :
		num_layers		{ new_num_layers >= 2 ? new_num_layers : throw("The number of layers to create a NeuralNetwork must be at least 2") },
		layer_sizes		{ (unsigned*)calloc(new_num_layers, sizeof(unsigned)) },
		Weights			{ new Matrix[new_num_layers - 1] },
		Velocity		{ new Matrix[new_num_layers - 1] },
		Gradient		{ new Matrix[new_num_layers - 1] },
		biases			{ new Vector[new_num_layers - 1] },
		b_velocity		{ new Vector[new_num_layers - 1] },
		b_gradient		{ new Vector[new_num_layers - 1] },
		layers			{ new Vector[new_num_layers] },
		error_signal	{ new Vector[new_num_layers] },
		input_layer		{ layers[0] },
		output_layer	{ layers[new_num_layers - 1] }
	{

		// Copies the layer sizes for internal storage
		for (unsigned l = 0; l < num_layers; l++)
			layer_sizes[l] = new_layer_sizes[l];

		// Initializes the weights and velocity vector
		for (unsigned layer = 0u; layer < num_layers - 1; layer++)
			Weights[layer].init(layer_sizes[layer + 1], layer_sizes[layer]),
			Velocity[layer].init(layer_sizes[layer + 1], layer_sizes[layer]),
			Gradient[layer].init(layer_sizes[layer + 1], layer_sizes[layer]),
			biases[layer].init(layer_sizes[layer + 1]),
			b_velocity[layer].init(layer_sizes[layer + 1]),
			b_gradient[layer].init(layer_sizes[layer + 1]);

		// Initializes the layer nodes, and error signal matrices. Input layer is excluded,
		// since it is provided by the training data and does not apply for error signal.
		for (unsigned layer = 1; layer < num_layers; layer++)
			layers[layer].init(layer_sizes[layer]),
			error_signal[layer].init(layer_sizes[layer]);
	}

	// Deletes all the matrices and vectors
	~MATRIX_DATA()
	{
		delete[] Weights;
		delete[] Velocity;
		delete[] Gradient;
		delete[] error_signal;
		delete[] layers;
		delete[] biases;
		delete[] b_velocity;
		delete[] b_gradient;

		free(layer_sizes);
	}
};

/*
-------------------------------------------------------------------------------------------------------
Utilities for binary file storage
-------------------------------------------------------------------------------------------------------
*/

// Header on any set of wieghts stored in file.
struct binary_descriptor
{
	static constexpr uint32_t default_magic = 0x00CA0200UL + VERSION;

	uint32_t magic;
	uint32_t n_layers;
	char name[40] = {};
};

// On match: returns a descriptor with proper magic and leaves `file`
// positioned immediately after the descriptor (before layer sizes).
static inline binary_descriptor find_in_file(const char* stored_name, FILE*& file)
{
	binary_descriptor desc;
	while (fread(&desc, sizeof(binary_descriptor), 1u, file) == 1u)
	{
		// Check wether the magic holds, for file corruption
		if (desc.magic != binary_descriptor::default_magic)
			return { 0,1 };

		// Checks if the name is the same
		for (unsigned i = 0; i < 40 && stored_name[i] == desc.name[i]; i++)
			if (stored_name[i] == '\0')
				return desc;

		// Stores the layer sizes
		uint32_t* file_layer_sizes = (uint32_t*)calloc(desc.n_layers, sizeof(uint32_t));
		if(fread(file_layer_sizes, sizeof(uint32_t), desc.n_layers, file) != desc.n_layers)
			return { 0,1 };

		// Skip to next set of weights
		for (unsigned i = 0; i < desc.n_layers - 1; i++)
			if (fseek(file, (file_layer_sizes[i] + 1) * file_layer_sizes[i + 1] * sizeof(float), SEEK_CUR) != 0)
				return { 0,1 };
		free(file_layer_sizes);

	}
	return { 0,0 };
}

/*
-------------------------------------------------------------------------------------------------------
Inline private helpers
-------------------------------------------------------------------------------------------------------
*/

// Sets the velocity vector to zero for new training round.

inline void NeuralNetwork::resetVelocity()
{
	MATRIX_DATA& M = *((MATRIX_DATA*)matrixData);

	for (unsigned layer = 0u; layer < M.num_layers - 1; layer++)
		M.Velocity[layer].zero(),
		M.b_velocity[layer].zero();
}

// Applies back propagation gradient descent given a single training example.

inline void NeuralNetwork::doBackPropagation(unsigned answer)
{
	MATRIX_DATA& M = *((MATRIX_DATA*)matrixData);

	// Assigns the initial error signals given by the formula D_Loss = p_i - y_i
	
	M.error_signal[M.num_layers - 1].copy_from(M.output_layer);
	M.error_signal[M.num_layers - 1](answer) -= 1.f;

	// Iterates through the layers from output to input doing back propagation
	for (int layer = M.num_layers - 2; layer >= 0; layer--)
	{
		// If not on the last step compute previous layer error signal
		if (layer > 0)
		{
			gemm(M.Weights[layer], M.error_signal[layer + 1], M.error_signal[layer], 1.f, 0.f, true /* transpose Weights */);
			M.error_signal[layer].relu_grad_inplace(M.layers[layer]);
		}

		// Compute the gradient for the current layer including weight decay
		gemm(M.error_signal[layer + 1], M.layers[layer], M.Gradient[layer]);
		axpy(M.Weights[layer], M.Gradient[layer], weight_decay_lambda); // Weight decay

		M.b_gradient[layer].copy_from(M.error_signal[layer + 1]); // Biases

		// Update the velocity matrix with the gradient, learning rate and momentum
		axpby(M.Gradient[layer], M.Velocity[layer], -learning_rate, gradient_momentum);

		axpby(M.b_gradient[layer], M.b_velocity[layer], -learning_rate, gradient_momentum); // Biases

		// Update weights with velocity
		add(M.Weights[layer], M.Velocity[layer], M.Weights[layer]);

		add(M.biases[layer], M.b_velocity[layer], M.biases[layer]); // Biases
	}
}

// Applies a forward pass of the Neural Network given an input.

inline void NeuralNetwork::doForwardPass()
{
	MATRIX_DATA& M = *((MATRIX_DATA*)matrixData);

	// Iterates through the layers computing the dot product and applying ReLU to inner layers
	for (unsigned layer = 0; layer < M.num_layers - 1; layer++)
	{
		axpb(M.Weights[layer], M.layers[layer], M.biases[layer], M.layers[layer + 1]);

		if (layer < M.num_layers - 2)
			M.layers[layer + 1]._relu();
	}
	// applies softmax to the output layer
	M.output_layer._softmax();
}

/*
-------------------------------------------------------------------------------------------------------
Other helpers
-------------------------------------------------------------------------------------------------------
*/

// Used for learning rate decrease with epoch

static inline float lr_cosine(float lr0, float alpha, unsigned t, unsigned T) 
{
	const float c = 0.5f * (1.0f + cosf(3.14159265f * (float)t / (float)T));
	return lr0 * (alpha + (1.0f - alpha) * c);
}

// Mixes up the randomizer seed

static inline unsigned long long& splitmix(unsigned long long& seed)
{
	seed += 0x9E3779B97F4A7C15ull;
	seed = (seed ^ (seed >> 30)) * 0xBF58476D1CE4E5B9ull;
	seed = (seed ^ (seed >> 27)) * 0x94D049BB133111EBull;
	seed ^= (seed >> 31);
	return seed;
}

// Generates random values between 0 and 1

static inline float random_0_1()
{
	static unsigned long long seed = Timer::get_system_time_ns() 
		^ (unsigned long long)(uintptr_t)&seed;

	return (splitmix(seed) >> 8) * (1.0f / 72057594037927936.0f); // 2^56
}

/*
-------------------------------------------------------------------------------------------------------
Constructor / Destructor
-------------------------------------------------------------------------------------------------------
*/

// Creates the linear model and randomly initialises the weights or initialises them to zero.
// The type, number of inputs and number of outputs must be specified for creation.

NeuralNetwork::NeuralNetwork(unsigned num_layers, unsigned const* layer_sizes):
	matrixData{ (void*)new MATRIX_DATA(num_layers, layer_sizes) }
{
	// Randomize the weights
	randomizeWeights();
}

// Replicates the neural network stored in file with that name, if it does not exist it will 
// throw a message, so make sure you only call it on a name existing in storage.

NeuralNetwork::NeuralNetwork(const char* stored_name, const char* filename)
{
	// If invalid name fail
	if (!stored_name || !*stored_name)
		throw("Invalid storage name");

	// IF invalid file fail
	if (!filename || !*filename)
		throw("Invalid file name");

	// If cannot open file fail
	FILE* weights_file = fopen(filename, "rb");
	if (!weights_file)
		throw("Cannot access weights file");

	// try to find the name in file
	binary_descriptor desc = find_in_file(stored_name, weights_file);

	// If not found fail
	if (!desc.magic)
	{
		fclose(weights_file);

		if(desc.n_layers == 1)
			throw("File corrupted or from older version");

		throw("Neural Network not found in file");
	}

	// Initializes the matrix data with the layer layout found in the file
	unsigned* layer_sizes = (unsigned*)calloc(desc.n_layers, sizeof(unsigned));
	fread(layer_sizes, sizeof(uint32_t), desc.n_layers, weights_file);

	matrixData = (void*)new MATRIX_DATA(desc.n_layers, layer_sizes);
	MATRIX_DATA& M = *((MATRIX_DATA*)matrixData);
	free(layer_sizes);

	// Copies the weights from the array into the object stored weights
	for (unsigned layer = 0; layer < M.num_layers - 1; layer++)
	{
		size_t num_weights = M.layer_sizes[layer];
		size_t num_biases = M.layer_sizes[layer + 1];

		for (unsigned output_node = 0; output_node < M.layer_sizes[layer + 1]; output_node++)
			if (fread(M.Weights[layer].data() + output_node * M.Weights[layer].ld(), sizeof(float), num_weights, weights_file) != num_weights)
			{
				fclose(weights_file);
				delete& M;
				throw("Unable to read data from file, may be corrupted");
			}

		if (fread(M.biases[layer].data(), sizeof(float), num_biases, weights_file) != num_biases)
		{
			fclose(weights_file);
			delete& M;
			throw("Unable to read data from file, may be corrupted");
		}
	}

	fclose(weights_file);
}

// Frees the weight values and training data, unless stored, the current weights will be forgotten.

NeuralNetwork::~NeuralNetwork()
{
	freeData();

	delete (MATRIX_DATA*)matrixData;
}

/*
-------------------------------------------------------------------------------------------------------
User end functions
-------------------------------------------------------------------------------------------------------
*/

// Randomizes the weights, unless stored, the current weights will be overwritten.

void NeuralNetwork::randomizeWeights()
{
	MATRIX_DATA& M = *((MATRIX_DATA*)matrixData);

	// Set those randomized weights
	for (unsigned layer = 0; layer < M.num_layers - 1; layer++)
	{
		// Establish a good range for the randomized weights
		float range;
		if (layer == M.num_layers - 2)
			// Xavier/Glorot for last layer (softmax)
			range = sqrtf(6.f / float(M.layer_sizes[layer] + M.layer_sizes[M.num_layers - 1]));
		else
			// He uniform ReLU for hidden layers
			range = sqrtf(6.0f / float(M.layer_sizes[layer]));

		// Iterate throught the matrix parameters and randomize them
		for (unsigned output_node = 0; output_node < M.layer_sizes[layer + 1]; output_node++)
			for(unsigned input_node = 0; input_node < M.layer_sizes[layer]; input_node++)
				M.Weights[layer](output_node,input_node) = (2.f * random_0_1() - 1.f) * range;

		// Zero the biases
		memset(M.biases[layer].data(), 0, sizeof(float) * M.layer_sizes[layer + 1]);
	}
}

// Stores the weights in a binary file for future loading. They are stored
// with the name, the number of layers and the size of each layer.
// If the name already exists and the layout is the same it will override it.
// If the name already exists and the layout is not the same it will fail.

bool NeuralNetwork::storeWeights(const char* stored_name, const char* filename) const
{
	MATRIX_DATA& M = *((MATRIX_DATA*)matrixData);

	// Okay this function works but is a bit of a mess, just trust it
	if (!stored_name || !*stored_name)
		return false;

	if (!filename || !*filename)
		return false;

	// open weights file if failed create it
	bool success = false;
	binary_descriptor desc;
	FILE* weights_file = fopen(filename, "rb+");
	if (!weights_file)
		goto for_new_file;

	// try to find the name in file
	desc = find_in_file(stored_name, weights_file);

	// if not go to the end of the file and write a new entry
	if (!desc.magic)
	{
		fclose(weights_file);

		// File corrupted or from previous version
		if (desc.n_layers == 1)
			return false;

	for_new_file:

		weights_file = fopen(filename, "ab");
		if (!weights_file)
			return false;
	
		// create the binary descriptor
		desc = { binary_descriptor::default_magic, M.num_layers };
		unsigned i = 0;
		for (; i < 40 && stored_name[i]; i++)
			desc.name[i] = stored_name[i];

		if (i == 40) // name too long
			goto cleanup;

		// push the binary descriptor and layer sizes into the file
		if (
			fwrite(&desc, sizeof(binary_descriptor), 1, weights_file) != 1 ||
			fwrite(M.layer_sizes, sizeof(uint32_t), M.num_layers, weights_file) != M.num_layers
			) goto cleanup;

		goto push_weights;	
	}

	// Check you have the same amout of layers
	if (desc.n_layers != M.num_layers)
		goto cleanup;

	// Check the layer sizes are the same
	for (uint32_t layer_size, i = 0; i < desc.n_layers; i++)
		if (fread(&layer_size, sizeof(uint32_t), 1, weights_file) != 1 || layer_size != M.layer_sizes[i])
			goto cleanup;

	fseek(weights_file, 0, SEEK_CUR); // otherwise the debugger complains

push_weights:

	// Everything alright, override the previous weight values
	for (unsigned layer = 0; layer < M.num_layers - 1; layer++)
	{
		for (unsigned output_node = 0; output_node < M.layer_sizes[layer + 1]; output_node++)
			fwrite(M.Weights[layer].data() + output_node * M.Weights[layer].ld(), sizeof(float), M.layer_sizes[layer], weights_file);

		fwrite(M.biases[layer].data(), sizeof(float), M.layer_sizes[layer + 1], weights_file);
	}
	success = true;
cleanup:

	// Clean up before leaving
	fclose(weights_file);
	return success;
}

// Loads weights previously stored with the same name. If the neural network
// structure does not match to the stored one with the same name it will fail.

bool NeuralNetwork::loadWeights(const char* stored_name, const char* filename)
{
	MATRIX_DATA& M = *((MATRIX_DATA*)matrixData);

	if (!stored_name || !*stored_name)
		return false;

	if (!filename || !*filename)
		return false;

	FILE* weights_file = fopen(filename, "rb");
	if (!weights_file)
		return false;

	// try to find the name in file
	binary_descriptor desc = find_in_file(stored_name, weights_file);

	// If not found or mismatch in amount of layers fail
	if (!desc.magic || desc.n_layers != M.num_layers)
	{
		fclose(weights_file);
		return false;
	}

	// Check the layer sizes are the same, else fail
	for (uint32_t layer_size, i = 0; i < desc.n_layers; i++)
		if (fread(&layer_size, sizeof(uint32_t), 1, weights_file) != 1 || layer_size != M.layer_sizes[i])
		{
			fclose(weights_file);
			return false;
		}

	// Copies the weights from the array into the object stored weights
	for (unsigned layer = 0; layer < M.num_layers - 1; layer++)
	{
		size_t num_weights = M.layer_sizes[layer];
		size_t num_biases = M.layer_sizes[layer + 1];

		for (unsigned output_node = 0; output_node < M.layer_sizes[layer + 1]; output_node++)
			if(fread(M.Weights[layer].data() + output_node * M.Weights[layer].ld(), sizeof(float), num_weights, weights_file) != num_weights)
			{
				fclose(weights_file);
				return false;
			}

		if(fread(M.biases[layer].data(), sizeof(float), num_biases, weights_file) != num_biases)
		{
			fclose(weights_file);
			return false;
		}
	}

	fclose(weights_file);
	return true;
}

// Copies the new training inputs and outputs and adds it to its array. They must be ordered
// as follows: inputs[num_data][num_input_nodes], outputs[num_data][num_output_nodes].

void NeuralNetwork::feedData(unsigned num_data, float* const* new_training_inputs, unsigned const* correct_answers)
{
	MATRIX_DATA& M = *((MATRIX_DATA*)matrixData);

	float** temp_inputs = (float**)calloc(num_data + num_training_data, sizeof(float*));
	unsigned* temp_answers = (unsigned*)calloc(num_data + num_training_data, sizeof(unsigned));

	for (unsigned i = 0; i < num_training_data; i++)
	{
		temp_inputs[i] = this->training_inputs[i];
		temp_answers[i] = this->training_answers[i];
	}

	for (unsigned i = num_training_data; i < num_training_data + num_data; i++)
	{
		temp_inputs[i] = (float*)calloc(M.layer_sizes[0], sizeof(float));
		for (unsigned j = 0; j < M.layer_sizes[0]; j++)
			temp_inputs[i][j] = new_training_inputs[i - num_training_data][j];

		temp_answers[i] = correct_answers[i - num_training_data];
	}

	if (num_training_data)
	{
		free(this->training_inputs);
		free(this->training_answers);
	}

	num_training_data += num_data;
	this->training_inputs = temp_inputs;
	this->training_answers = temp_answers;
}

// Releases all the training examples stored in the neural network data,
// for better memory management. New data can be feeded normally.

void NeuralNetwork::freeData()
{
	if (num_training_data)
	{
		for (unsigned i = 0; i < num_training_data; i++)
			free(training_inputs[i]);

		free(training_inputs);
		free(training_answers);

		training_inputs = nullptr;
		training_answers = nullptr;
		num_training_data = 0;
	}
}

// Using the current weights for the model outputs a prediction given an input.
// If an output pointer is set, it will write it there, otherwise it creates a new one.

float* NeuralNetwork::predictCase(float const* input)
{
	MATRIX_DATA& M = *((MATRIX_DATA*)matrixData);

	// The input case is bound to the input laywe. Has to be 
	// recasted but the data inside will not be modified.
	M.layers[0].set_data((float*)input, M.layer_sizes[0]);

	// Does the forward pass
	doForwardPass();

	// Returns the last layer vector data
	return M.output_layer.data();
}

// Using the current weights for the model computes the prediction error on a single data point.

float NeuralNetwork::outputError(float const* input, const unsigned correct_answer)
{
	MATRIX_DATA& M = *((MATRIX_DATA*)matrixData);

	// What do you think?
	float p = predictCase(input)[correct_answer];

	// We dont want infinities here
	if (p < 1e-12f) 
		p = 1e-12f;

	// Return loss function
	return -logf(p);
}

// Using the current weights for the model computes the average prediction error on a
// set of test data. For generalization the test data must not be part of the training data.

float NeuralNetwork::computePredictionError(unsigned num_data, float* const* test_inputs, unsigned const* correct_answers)
{
	float total_error = 0.f;
	for (unsigned n = 0; n < num_data; n++)
		total_error += outputError(test_inputs[n], correct_answers[n]);

	return total_error / num_data;
}

// Using the current weights for the model computes the average prediction rate on a
// set of test data. Considers a guess the output with strictly highest probability.
// For generalization the test data must not be part of the training data.

float NeuralNetwork::computePredictionRate(unsigned num_data, float* const* test_inputs, unsigned const* correct_answers)
{
	MATRIX_DATA& M = *((MATRIX_DATA*)matrixData);

	unsigned answer, success_count = 0u, output_size = M.layer_sizes[M.num_layers - 1];
	float* prediction = M.output_layer.data();

	for (unsigned n = 0; n < num_data; n++)
	{
		predictCase(test_inputs[n]);
		answer = correct_answers[n];

		unsigned c;
		for (c = 0; c < output_size && (c == answer || prediction[c] < prediction[answer]); c++);

		if (c == output_size)success_count++;
	}
	return float(success_count) / num_data;
}

// Trains the weights with the current weight decay lambda using stochastic gradient descent.

void NeuralNetwork::trainWeights(unsigned epoch)
{
	if (!num_training_data)
		return;

	resetVelocity();
	float initial_lr = learning_rate;

	for (unsigned iteration = 0; iteration < epoch; iteration++)
	{
		// Cosine decay for learning rate
		learning_rate = lr_cosine(initial_lr, learning_rate_alpha, iteration, epoch);

		for (unsigned n = 0; n < num_training_data; n++)
		{
			predictCase(training_inputs[n]);
			doBackPropagation(training_answers[n]);
		}
	}

	learning_rate = initial_lr;
}

// Trains the weights using cross validation (10% testing batches) and outputs expected error.

float NeuralNetwork::trainCrossValidation(unsigned epoch, unsigned patience, unsigned n_batches)
{
	float error = 0.f;
	float initial_lr = learning_rate;
	for (unsigned batch = 0; batch < n_batches; batch++)
	{
		// Reset weights and velocity vector
		randomizeWeights();
		resetVelocity();

		// Set the boundries for the validation set
		const unsigned start_validation_set = batch * num_training_data / n_batches;
		const unsigned end_validation_set = (batch + 1) * num_training_data / n_batches;

		// Train the neural network for a certain amount of epoch skipping the validation set
		float current_best_error = INFINITY;
		unsigned patience_counter = 0U;
		for (unsigned iteration = 0; iteration < epoch; iteration++)
		{
			// Cosine decay for learning rate
			learning_rate = lr_cosine(initial_lr, learning_rate_alpha, iteration, epoch);

			// Train
			for (unsigned n = 0; n < num_training_data; n++)
			{
				if (n >= start_validation_set && n < end_validation_set)
					continue;

				predictCase(training_inputs[n]);
				doBackPropagation(training_answers[n]);
			}

			// Compute the error for the validation set
			float validation_error = 0.f;
			for (unsigned n = start_validation_set; n < end_validation_set; n++)
				validation_error += outputError(training_inputs[n], training_answers[n]);

			// Check if you've done better than before
			if (validation_error < current_best_error)
				current_best_error = validation_error, patience_counter = 0;

			// If you run out of patience stop
			else if (++patience_counter == patience)
				break;

		}

		// Add validation error accumulates untill full mean error is computed
		error += current_best_error / num_training_data;
	}
	learning_rate = initial_lr;
	return error;
}

// Trains the weights using cross validation (10% testing batches) for different values
// of lambda and learning rate, then trains the full set and outputs the expected error.

float NeuralNetwork::train_HyperParamCalibration(unsigned tries, unsigned epoch)
{
	if (!num_training_data || !tries || !epoch)
		return INFINITY;

	constexpr float min_learning_rate = 1e-5f;
	constexpr float max_learning_rate = 1e-1f;

	constexpr float min_lr_alpha = 0.005f;
	constexpr float max_lr_alpha = 0.04f;

	constexpr float min_weight_decay = 1e-10f;
	constexpr float max_weight_decay = 1e-4f;

	constexpr float min_momentum = 0.65f;
	constexpr float max_momentum = 0.95f;

	constexpr unsigned n_batches = 10;

	constexpr unsigned patience = 3;

	float best_learning_rate;
	float best_weight_decay;
	float best_momentum;
	float best_lr_alpha;
	float best_mean_error = INFINITY;

	for (unsigned t = 0; t < tries; t++)
	{
		// Randomize the hyperparameters for new try
		float rand = random_0_1();
		learning_rate = expf(rand * logf(min_learning_rate) + (1 - rand) * logf(max_learning_rate));

		rand = random_0_1();
		weight_decay_lambda = expf(rand * logf(min_weight_decay) + (1 - rand) * logf(max_weight_decay));

		rand = random_0_1();
		gradient_momentum = rand * min_momentum + (1 - rand) * max_momentum;

		rand = random_0_1();
		learning_rate_alpha = rand * min_lr_alpha + (1 - rand) * max_lr_alpha;

		// Train for the total number of validation batches and store error
		float error = trainCrossValidation(epoch, patience, n_batches);

		// If surpassed best score store the hyperparameters
		if (error < best_mean_error)
		{
			best_mean_error = error;
			best_learning_rate = learning_rate;
			best_lr_alpha = learning_rate_alpha;
			best_momentum = gradient_momentum;
			best_weight_decay = weight_decay_lambda;
		}
#ifdef _CONSOLE
		printf("\n\nTraning try %u finished with parameters:\n  lambda = %.8f\n  learning_rate = %.5f\n  momentum = %.3f\n  lr_alpha = %.3f\nMean validation error was %.4f",
			t, weight_decay_lambda, learning_rate, gradient_momentum, learning_rate_alpha, error);
#endif
	}
	
	// Set the best results for training and start the real training!
	learning_rate = best_learning_rate;
	learning_rate_alpha = best_lr_alpha;
	gradient_momentum = best_momentum;
	weight_decay_lambda = best_weight_decay;

	randomizeWeights();
	trainWeights(epoch);
#ifdef _CONSOLE
	printf("\n\nTraning session finished with winning parameters:\n  lambda = %.8f\n  learning_rate = %.5f\n  momentum = %.3f\n  lr_alpha = %.3f\nMean expected error is %.4f",
		best_weight_decay, best_learning_rate, best_momentum, best_lr_alpha, best_mean_error);
#endif
	return best_mean_error;
}

#ifdef _CONSOLE
// Prints the weights to the console.

void NeuralNetwork::printWeights()
{
	MATRIX_DATA& M = *((MATRIX_DATA*)matrixData);

	for (unsigned layer = 0; layer < M.num_layers - 1; layer++)
	{
		if (!layer)
			printf("\nInput layer (%u nodes) -> ", M.layer_sizes[layer]);
		else
			printf("\nLayer %u (%u nodes) -> ", layer, M.layer_sizes[layer]);

		if (layer == M.num_layers - 2)
			printf("Output layer (%u nodes):\n", M.layer_sizes[layer + 1]);
		else
			printf("Layer %u (%u nodes):\n", layer + 1, M.layer_sizes[layer + 1]);

		for (unsigned output_node = 0; output_node < M.layer_sizes[layer + 1]; output_node++)
		{
			if (M.layer_sizes[layer + 1] > 8 && output_node > 2 && output_node < M.layer_sizes[layer + 1] - 3)
				continue;

			printf("\tOutput %u: { ", output_node);
			for (unsigned i = 0; i < M.layer_sizes[layer]; i++)
			{
				if (M.layer_sizes[layer] > 10 && i > 3 && i < M.layer_sizes[layer] - 3)
					continue;

				printf("%+1.3f, ", M.Weights[layer](output_node, i));

				if (M.layer_sizes[layer] > 10 && i == 3)
					printf("... , ");
			}

			printf("bias = %+1.3f }\n", M.biases[layer](output_node));

			if (M.layer_sizes[layer + 1] > 8 && output_node == 2) 
				printf("\t ...\n");
		}
	}
}
#endif

/*
-------------------------------------------------------------------------------------------------------
Getters and setters
-------------------------------------------------------------------------------------------------------
*/

// Returns a pointer to the set of weights

float* NeuralNetwork::getWeights(unsigned layer)
{
	MATRIX_DATA& M = *((MATRIX_DATA*)matrixData);
	return M.Weights[layer].data();
}

// Returns a pointer to the biases of a specific layer

float* NeuralNetwork::getBiases(unsigned layer)
{
	MATRIX_DATA& M = *((MATRIX_DATA*)matrixData);
	return M.biases[layer].data();
}

// Returns the lambda used for weight decay

float NeuralNetwork::getWeightDecayLambda() const
{
	return weight_decay_lambda;
}

// Sets the lambda used for weight decay

void NeuralNetwork::setWeightDecay(float lambda)
{
	weight_decay_lambda = lambda;
}

// Returns the learning rate used for training

float NeuralNetwork::getLearningRate() const
{
	return learning_rate;
}

// Sets the learning rate used for training

void NeuralNetwork::setLearningRate(float rate)
{
	learning_rate = rate;
}

// Returns the momentum variable used for weight step

float NeuralNetwork::getMomentum() const
{
	return gradient_momentum;
}

// Sets the momentum variable used for weight step

void NeuralNetwork::setMomentum(float momentum)
{
	gradient_momentum = momentum;
}

// Returns the alpha variable used for learning rate decay

float NeuralNetwork::getLRalpha() const
{
	return learning_rate_alpha;
}

// Sets the alpha variable used for learning rate decay

void NeuralNetwork::setLRalpha(float alpha)
{
	learning_rate_alpha = alpha;
}
