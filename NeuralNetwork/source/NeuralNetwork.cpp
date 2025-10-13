#include "NeuralNetwork.h"
#include "Timer.h"

#include <cstdlib>
#include <cstdint>
#include <cstdio>
#include <cmath>

/*
-------------------------------------------------------------------------------------------------------
Utilities for binary file storage
-------------------------------------------------------------------------------------------------------
*/

// Header on any set of wieghts stored in file.

struct binary_descriptor
{
	static constexpr uint32_t default_magic = 0x00CA0020UL;
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
			return { 0,0 };

		// Checks if the name is the same
		for (unsigned i = 0; i < 40 && stored_name[i] == desc.name[i]; i++)
			if (stored_name[i] == '\0')
				return desc;

		// Stores the layer sizes
		uint32_t* file_layer_sizes = (uint32_t*)calloc(desc.n_layers, sizeof(uint32_t));
		fread(file_layer_sizes, sizeof(uint32_t), desc.n_layers, file);

		// Skip to next set of weights
		for (unsigned i = 0; i < desc.n_layers - 1; i++)
			fseek(file, (file_layer_sizes[i] + 1) * file_layer_sizes[i + 1] * sizeof(float), SEEK_CUR);
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
	for (unsigned layer = 0u; layer < num_layers - 1; layer++)
		for (unsigned output = 0u; output < layer_sizes[layer + 1]; output++)
			memset(velocity[layer][output], 0, (layer_sizes[layer] + 1) + sizeof(float));
}

// Helper inline function to calculate the dot product between a set of weights and a layer.

inline float NeuralNetwork::_dot_prod(const unsigned input_layer, const unsigned output_node)
{
	float* o_weights = weights[input_layer][output_node];
	float dot_prod = 0;

	for (unsigned i = 0; i < layer_sizes[input_layer]; i++)
		dot_prod += o_weights[i] * layers[input_layer][i];

	return dot_prod + o_weights[layer_sizes[input_layer]];
}

// Helper inline function to update a set of weights for one training iteration.
// Updates the weights from one layer to a specific node by w[i_layer][o_node] += multiplier * layers[i_layer]

inline void NeuralNetwork::doWeightStep(const float error_signal, const unsigned input_layer, const unsigned output_node)
{
	float* node_weights = weights[input_layer][output_node];
	float* node_velocity = velocity[input_layer][output_node];
	const float const* layer = layers[input_layer];
	const float factor = -learning_rate * error_signal;
	const unsigned layer_size = layer_sizes[input_layer];

	unsigned i = 0;
	for (; i < layer_size; i++)
		node_weights[i] += (node_velocity[i] = node_velocity[i] * gradient_momentum + factor * layer[i]);

	node_weights[i] += (node_velocity[i] = node_velocity[i] * gradient_momentum + factor);
}

// Helper inline function that performs a weight decay step on a set of weights.

inline void NeuralNetwork::doWeightDecay(const unsigned input_layer, const unsigned output_node)
{
	float* o_weights = weights[input_layer][output_node];
	float factor = 1.f - weight_decay_lambda * learning_rate;

	for (unsigned i = 0; i < layer_sizes[input_layer]; i++)
		o_weights[i] *= factor;
}

// Normalizes the output values using the softmax activation function, returns the pointer to the output.

inline float* NeuralNetwork::_softmax()
{
	float max = output_layer[0];
	for (unsigned node = 1; node < layer_sizes[num_layers - 1]; node++)
		if (max < output_layer[node])
			max = output_layer[node];

	float total = 0.f;
	for (unsigned node = 0; node < layer_sizes[num_layers - 1]; node++)
	{
		output_layer[node] = expf(output_layer[node] - max);
		total += output_layer[node];
	}

	for (unsigned node = 0; node < layer_sizes[num_layers - 1]; node++)
		output_layer[node] /= total;

	return output_layer;
}

// Applies ReLU to a given layer, simple function for convenience, returns the pointer to the layer.

inline float* NeuralNetwork::_ReLU(unsigned layer)
{
	for (unsigned node = 0; node < layer_sizes[layer]; node++)
		if (layers[layer][node] < 0.f) 
			layers[layer][node] = 0.f;

	return layers[layer];
}

// Applies back propagation gradient descent given a single training example.

inline void NeuralNetwork::doBackPropagation(unsigned answer)
{
	// Assigns the initial error signals given by the formula D_Loss = p_i - y_i
	for (unsigned node = 0; node < layer_sizes[num_layers - 1]; node++)
		error_signal[num_layers - 1][node] = output_layer[node];
	error_signal[num_layers - 1][answer]--;

	// Iterates through the layers from output to input doing back propagation
	for (unsigned layer = num_layers - 1; layer > 0; layer--)
	{
		// Sets the error signals to work with in the current layer
		float* current_error_signal = error_signal[layer];
		// Sets the error signals for the next iteration and resets them
		float* next_error_signal = error_signal[layer - 1];
		if(layer > 1)
			memset(next_error_signal, 0, layer_sizes[layer - 1] * sizeof(float));

		// Iterates through all the nodes of the layer applying weight decay and gradient descent
		// and computing next error signal by adding the error of each independent weight
		for (unsigned node = 0; node < layer_sizes[layer]; node++)
		{
			// If not on the last step compute previous layer error signal
			if (layer > 1)
				for (unsigned i = 0; i < layer_sizes[layer - 1]; i++)
					// Only compute error signal if previous node is activated
					if (layers[layer - 1][i])
						next_error_signal[i] += current_error_signal[node] * weights[layer - 1][node][i];

			// Applies weight decay and stochastic gradient descent to every weight connected to this node
			doWeightDecay(layer - 1, node);
			doWeightStep(current_error_signal[node], layer - 1, node);
		}
	}
}

/*
-------------------------------------------------------------------------------------------------------
Constructor / Destructor
-------------------------------------------------------------------------------------------------------
*/

// Creates the linear model and randomly initialises the weights or initialises them to zero.
// The type, number of inputs and number of outputs must be specified for creation.

NeuralNetwork::NeuralNetwork(unsigned num_layers, unsigned const* layer_sizes):
	num_layers{num_layers}, layer_sizes{ (unsigned*)calloc(num_layers, sizeof(unsigned)) }
{
	if (num_layers < 2)
		throw("The number of layers to create a NeuralNetwork must be at least 2");

	// Copies the layer sizes for internal storage
	for (unsigned l = 0; l < num_layers; l++)
		this->layer_sizes[l] = layer_sizes[l];

	// Initializes the weights and velocity vector
	weights = (float***)calloc(num_layers - 1, sizeof(float**));
	velocity = (float***)calloc(num_layers - 1, sizeof(float**));

	for (unsigned layer = 0u; layer < num_layers - 1; layer++)
	{
		weights[layer] = (float**)calloc(layer_sizes[layer + 1], sizeof(float*));
		velocity[layer] = (float**)calloc(layer_sizes[layer + 1], sizeof(float*));

		for (unsigned output = 0u; output < layer_sizes[layer + 1]; output++)
			weights[layer][output] = (float*)calloc(layer_sizes[layer] + 1, sizeof(float)),
			velocity[layer][output] = (float*)calloc(layer_sizes[layer] + 1, sizeof(float));
	}

	// Randomize the weights
	randomizeWeights();

	// Initializes the layer nodes, and error signal array. Input layer is excluded,
	// since it is provided by the training data and does not apply for error signal.
	layers = (float**)calloc(num_layers, sizeof(float*));
	error_signal = (float**)calloc(num_layers, sizeof(float*));

	for (unsigned layer = 1; layer < num_layers; layer++)
		layers[layer] = (float*)calloc(layer_sizes[layer], sizeof(float)), 
		error_signal[layer] = (float*)calloc(layer_sizes[layer], sizeof(float));

	output_layer = layers[num_layers - 1];
}

// Replicates the neural network stored in file with that name, if it does not exist it will 
// throw a message, so make sure you only call it on a name existing in storage.

NeuralNetwork::NeuralNetwork(const char* stored_name)
{
	// If invalid name fail
	if (!stored_name || !*stored_name)
		throw("Invalid storage name name");

	// If cannot open file fail
	FILE* weights_file = fopen("weights", "rb");
	if (!weights_file)
		throw("Cannot access weights file");

	// try to find the name in file
	binary_descriptor desc = find_in_file(stored_name, weights_file);

	// If not found fail
	if (!desc.magic)
	{
		fclose(weights_file);
		throw("Neural Network not found in file");
	}

	// Copies the layer layout for internal storage
	num_layers = desc.n_layers;
	layer_sizes = (unsigned*)calloc(num_layers, sizeof(unsigned));
	fread(layer_sizes, sizeof(uint32_t), num_layers, weights_file);

	// Initializes the velocity vector and weights and copies them from the file
	velocity = (float***)calloc(num_layers - 1, sizeof(float**));
	weights = (float***)calloc(num_layers - 1, sizeof(float**));

	for (unsigned layer = 0u; layer < num_layers - 1; layer++)
	{
		velocity[layer] = (float**)calloc(layer_sizes[layer + 1], sizeof(float*));
		weights[layer] = (float**)calloc(layer_sizes[layer + 1], sizeof(float*));

		for (unsigned output = 0u; output < layer_sizes[layer + 1]; output++)
		{
			velocity[layer][output] = (float*)calloc(layer_sizes[layer] + 1, sizeof(float));
			weights[layer][output] = (float*)calloc(layer_sizes[layer] + 1, sizeof(float));
			fread(weights[layer][output], sizeof(float), layer_sizes[layer] + 1, weights_file);
		}
	}

	// Initializes the layer nodes, and error signal array. Input layer is excluded,
	// since it is provided by the training data and does not apply for error signal.
	layers = (float**)calloc(num_layers, sizeof(float*));
	error_signal = (float**)calloc(num_layers, sizeof(float*));

	for (unsigned layer = 1; layer < num_layers; layer++)
		layers[layer] = (float*)calloc(layer_sizes[layer], sizeof(float)),
		error_signal[layer] = (float*)calloc(layer_sizes[layer], sizeof(float));

	output_layer = layers[num_layers - 1];

}

// Frees the weight values and training data, unless stored, the current weights will be forgotten.

NeuralNetwork::~NeuralNetwork()
{

	for (unsigned layer = 0u; layer < num_layers - 1; layer++)
	{
		for (unsigned output = 0u; output < layer_sizes[layer + 1]; output++)
			free(weights[layer][output]), free(velocity[layer][output]);

		free(weights[layer]);
		free(velocity[layer]);
	}
	free(weights);
	free(velocity);

	if (num_training_data)
	{
		for (unsigned i = 0; i < num_training_data; i++)
			free(training_inputs[i]);

		free(training_inputs);
		free(training_answers);
	}

	for (unsigned layer = 1; layer < num_layers; layer++)
		free(layers[layer]), free(error_signal[layer]);

	free(error_signal);
	free(layers);
	free(layer_sizes);
}

/*
-------------------------------------------------------------------------------------------------------
User end functions
-------------------------------------------------------------------------------------------------------
*/

// Randomizes the weights, unless stored, the current weights will be overwritten.

static unsigned long long& splitmix(unsigned long long& seed)
{
	seed += 0x9E3779B97F4A7C15ull;
	seed = (seed ^ (seed >> 30)) * 0xBF58476D1CE4E5B9ull;
	seed = (seed ^ (seed >> 27)) * 0x94D049BB133111EBull;
	seed ^= (seed >> 31);
	return seed;
}

void NeuralNetwork::randomizeWeights()
{
	// Generate a random seed with splitmix
	unsigned long long seed = Timer::get_system_time_ns() ^ (unsigned long long)(uintptr_t) & seed;
	splitmix(seed), splitmix(seed), splitmix(seed);

	// Set those randomized weights
	for (unsigned layer = 0; layer < num_layers - 1; layer++)
		for (unsigned output = 0; output < layer_sizes[layer + 1]; ++output)
		{
			// Establish a good range for the randomized weights
			float range;
			if (layer == num_layers - 2)
				// Xavier/Glorot for last layer (sorfmax)
				range = sqrtf(6.f / float(layer_sizes[layer] + layer_sizes[num_layers - 1]));
			else
				// He uniform ReLU for hidden layers
				range = sqrtf(6.0f / float(layer_sizes[layer]));

			// Iterate through weights and randomize [-range,range)
			for (unsigned input = 0; input < layer_sizes[layer] + 1u; ++input)
			{
				// simple float from 64-bit: map [0,2^64) -> [0,1)
				float u = (splitmix(seed) >> 8) * (1.0f / 72057594037927936.0f); // 2^56
				weights[layer][output][input] = (2.0f * u - 1.0f) * range;
			}
		}
}

// Stores the weights in a binary file for future loading. They are stored
// with the name, the number of layers and the size of each layer.
// If the name already exists and the layout is the same it will override it.
// If the name already exists and the layout is not the same it will fail.

bool NeuralNetwork::storeWeights(const char* stored_name) const
{
	// Okay this function works but is a bit of a mess, just trust it
	if (!stored_name || !*stored_name)
		return false;

	// open weights file if failed create it
	bool success = false;
	binary_descriptor desc;
	FILE* weights_file = fopen("weights", "rb+");
	if (!weights_file)
		goto for_new_file;

	// try to find the name in file
	desc = find_in_file(stored_name, weights_file);

	// if not go to the end of the file and write a new entry
	if (!desc.magic)
	{
		fclose(weights_file);

	for_new_file:

		weights_file = fopen("weights", "ab");
		if (!weights_file)
			return false;
	
		// create the binary descriptor
		desc = { binary_descriptor::default_magic, num_layers };
		unsigned i = 0;
		for (; i < 40 && stored_name[i]; i++)
			desc.name[i] = stored_name[i];

		if (i == 40) // name too long
			goto cleanup;

		// push the binary descriptor and layer sizes into the file
		fwrite(&desc, sizeof(binary_descriptor), 1, weights_file);
		fwrite(layer_sizes, sizeof(uint32_t), num_layers, weights_file);

		goto push_weights;	
	}

	// Check you have the same amout of layers
	if (desc.n_layers != num_layers)
		goto cleanup;

	// Check the layer sizes are the same
	for (uint32_t layer_size, i = 0; i < desc.n_layers; i++)
		if (fread(&layer_size, sizeof(uint32_t), 1, weights_file) != 1 || layer_size != layer_sizes[i])
			goto cleanup;

	fseek(weights_file, 0, SEEK_CUR); // otherwise the debugger complains

push_weights:

	// Everything alright, override the previous weight values
	for (unsigned layer = 0; layer < num_layers - 1; layer++)
		for (unsigned output_node = 0; output_node < layer_sizes[layer + 1]; output_node++)
			fwrite(weights[layer][output_node], sizeof(float), layer_sizes[layer] + 1, weights_file);
	success = true;
cleanup:

	// Clean up before leaving
	fclose(weights_file);
	return success;
}

// Loads weights previously stored with the same name. If the neural network
// structure does not match to the stored one with the same name it will fail.

bool NeuralNetwork::loadWeights(const char* stored_name)
{
	if (!stored_name || !*stored_name)
		return false;

	FILE* weights_file = fopen("weights", "rb");
	if (!weights_file)
		return false;

	// try to find the name in file
	binary_descriptor desc = find_in_file(stored_name, weights_file);

	// If not found or mismatch in amount of layers fail
	if (!desc.magic || desc.n_layers != num_layers)
	{
		fclose(weights_file);
		return false;
	}

	// Check the layer sizes are the same, else fail
	for (uint32_t layer_size, i = 0; i < desc.n_layers; i++)
		if (fread(&layer_size, sizeof(uint32_t), 1, weights_file) != 1 || layer_size != layer_sizes[i])
		{
			fclose(weights_file);
			return false;
		}

	// Copies the weights from the array into the object stored weights
	for (unsigned layer = 0; layer < num_layers - 1; layer++)
		for (unsigned output_node = 0; output_node < layer_sizes[layer + 1]; output_node++)
			fread(weights[layer][output_node], sizeof(float), layer_sizes[layer] + 1, weights_file);

	fclose(weights_file);
	return true;
}

// Copies the new training inputs and outputs and adds it to its array. They must be ordered
// as follows: inputs[num_data][num_input_nodes], outputs[num_data][num_output_nodes].

void NeuralNetwork::feedData(unsigned num_data, float* const* new_training_inputs, unsigned const* correct_answers)
{
	float** temp_inputs = (float**)calloc(num_data + num_training_data, sizeof(float*));
	unsigned* temp_answers = (unsigned*)calloc(num_data + num_training_data, sizeof(unsigned));

	for (unsigned i = 0; i < num_training_data; i++)
	{
		temp_inputs[i] = this->training_inputs[i];
		temp_answers[i] = this->training_answers[i];
	}

	for (unsigned i = num_training_data; i < num_training_data + num_data; i++)
	{
		temp_inputs[i] = (float*)calloc(layer_sizes[0], sizeof(float));
		for (unsigned j = 0; j < layer_sizes[0]; j++)
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

// Using the current weights for the model outputs a prediction given an input.
// If an output pointer is set, it will write it there, otherwise it creates a new one.

float* NeuralNetwork::predictCase(float const* input)
{
	// This here is the only line that binds the input layer to any actual input
	layers[0] = (float*)input;

	// Iterates through the layers computing the dot product and applying ReLU to inner layers
	for (unsigned next = 1; next < num_layers; next++)
	{
		float* _layer = layers[next];
		unsigned _layer_size = layer_sizes[next];

		for (unsigned output_node = 0; output_node < _layer_size; output_node++)
			_layer[output_node] = _dot_prod(next - 1, output_node);

		if (_layer != output_layer)
			_ReLU(next);
	}
	// applies softmax and retunrs output layer
	return _softmax();
}

// Using the current weights for the model computes the prediction error on a single data point.

float NeuralNetwork::outputError(float const* input, const unsigned correct_answer)
{
	// What do you think?
	predictCase(input);
	float p = output_layer[correct_answer];
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
	unsigned answer, success_count = 0u, output_size = layer_sizes[num_layers - 1];

	for (unsigned n = 0; n < num_data; n++)
	{
		predictCase(test_inputs[n]);
		answer = correct_answers[n];

		unsigned c;
		for (c = 0; c < output_size && (c == answer || output_layer[c] < output_layer[answer]); c++);

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

	for (unsigned iteration = 0; iteration < epoch; iteration++)
		for (unsigned n = 0; n < num_training_data; n++)
		{
			predictCase(training_inputs[n]);
			doBackPropagation(training_answers[n]);
		}
}

// Trains the weights using cross validation (10% testing batches) for different values
// of lambda and learning rate, then trains the full set and outputs the expected error.

float NeuralNetwork::train_CrossValidation()
{
	return 0.f; // Not implemented yet :/
}

#ifdef _CONSOLE
// Prints the weights to the console.

void NeuralNetwork::printWeights()
{
	for (unsigned layer = 0; layer < num_layers - 1; layer++)
	{
		if (!layer)
			printf("\nInput layer (%u nodes) -> ", layer_sizes[layer]);
		else
			printf("\nLayer %u (%u nodes) -> ", layer, layer_sizes[layer]);

		if (layer == num_layers - 2)
			printf("Output layer (%u nodes):\n", layer_sizes[layer + 1]);
		else
			printf("Layer %u (%u nodes):\n", layer + 1, layer_sizes[layer + 1]);

		for (unsigned output_node = 0; output_node < layer_sizes[layer + 1]; output_node++)
		{
			if (layer_sizes[layer + 1] > 8 && output_node > 2 && output_node < layer_sizes[layer + 1] - 3) 
				continue;

			printf("\tOutput %u: { ", output_node);
			for (unsigned i = 0; i < layer_sizes[layer]; i++)
			{
				if (layer_sizes[layer] > 10 && i > 3 && i < layer_sizes[layer] - 3)
					continue;

				printf("%+1.3f, ", weights[layer][output_node][i]);

				if (layer_sizes[layer] > 10 && i == 3)
					printf("... , ");
			}

			printf("bias = %+1.3f }\n", weights[layer][output_node][layer_sizes[layer]]);

			if (layer_sizes[layer + 1] > 8 && output_node == 2) 
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

float*** NeuralNetwork::getWeights() const
{
	return weights;
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