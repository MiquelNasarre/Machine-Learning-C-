#pragma once

/* NEURAL NETWORK CLASS HEADER
-------------------------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------------------------
This class is my first attempt at creating a working Neural Network algorithm.
It is based on the handwritten number classification problem, therefore it outputs
a set of probabilities using softmax, and uses ReLU for the hidden layers.

Supports any arbitrary number of layers with any size, but is not fully optimized 
therefore is intended for smaller neural networks. It supports training with weight 
decay, and cross validation to obtimise the weight decay variable.
-------------------------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------------------------
*/

// Creates a Neural Network with the specified structure and takes in training examples 
// to train its weights and predict the output. It is a classification algorithm so the 
// output will be a probabilistic set of each possible outcome that adds up to one.
class NeuralNetwork
{
private:
	unsigned	num_layers = 0u;			// Stores the number of layers
	unsigned* layer_sizes = nullptr;		// Stores the size for each layer

	float* output_layer;					// Pointer to the last layer for convenience
	float** error_signal = nullptr;			// Stores the error signal to each layer
	float** layers = nullptr;				// Stores the nodes of the neural network
	float*** weights = nullptr;				// Stores the weights of the neural network

	float learning_rate = 0.005f;			// Learning rate used for GD and perceptron
	float weight_decay_lambda = 1e-8f;		// Stores the lambda for weight decay training

	float** training_inputs = nullptr;		// Stores inputs for training.
	unsigned* training_answers = nullptr;	// Stores correct answers for training.
	unsigned num_training_data = 0;			// Stores the amount of training data

	// Inline helpers

	// Helper inline function to calculate the dot product between a set of weights and a layer.
	inline float _dot_prod(const unsigned input_layer, const unsigned output_node) const;
	// Helper inline function to update a set of weights for one training iteration.
	// Updates the weights from one layer to a specific node by w[i_layer][o_node] += multiplier * layers[i_layer]
	inline void addToWeights(const float multiplier, const unsigned input_layer, const unsigned output_node);
	// Helper inline function that performs a weight decay step on a set of weights.
	inline void doWeightDecay(const unsigned input_layer, const unsigned output_node);
	// Normalizes the output values using the softmax activation function, returns the pointer to the output.
	inline float* _softmax() const;
	// Applies ReLU to a given layer, simple function for convenience, returns the pointer to the layer.
	inline float* _ReLU(unsigned layer) const;
	// Applies back propagation gradient descent given a single training example.
	inline void doBackPropagation(unsigned answer);

	// No copies allowed
	NeuralNetwork(const NeuralNetwork& other) = delete;
	NeuralNetwork operator=(const NeuralNetwork& other) = delete;

public:
	// Creates the neural network and randomly initialises the weights or initialises them to zero.
	// Number of layers and size of each layer -- inlcuding input and output layer -- must be
	// specified for creation, bias will be added to weights.
	NeuralNetwork(unsigned num_layers, const unsigned* layer_sizes);

	// Replicates the neural network stored in file with that name, if it does not exist it will 
	// throw a message, so make sure you only call it on a name existing in storage.
	NeuralNetwork(const char* stored_name);

	// Frees the weight values and training data, unless stored, the current weights will be forgotten.
	~NeuralNetwork();

	// Randomizes the weights, unless stored, the current weights will be overwritten.
	void randomizeWeights();

	// Stores the weights in a binary file for future loading. They are stored
	// with the name, the number of layers and the size of each layer.
	// If the name already exists and the layout is the same it will override it.
	// If the name already exists and the layout is not the same it will return false.
	bool storeWeights(const char* stored_name) const;

	// Loads weights previously stored with the same name. If the neural network
	// structure does not match to the stored one with the same name it will fail.
	bool loadWeights(const char* stored_name) const;

	// Copies the new training inputs and outputs and adds it to its array. They must be ordered
	// as follows: inputs[num_data][num_input_nodes], outputs[num_data][num_output_nodes].
	void feedData(unsigned num_data, float* const* new_training_inputs, unsigned const* correct_answers);

	// Using the current weights for the model outputs a prediction given an input.
	// If an output pointer is set, it will write it there, otherwise it creates a new one.
	float* predictCase(float const* input);

	// Using the current weights for the model computes the prediction error on a single data point.
	float outputError(float const* input, const unsigned correct_answer);

	// Using the current weights for the model computes the average prediction error on a
	// set of test data. For generalization the test data must not be part of the training data.
	float computePredictionError(unsigned num_data, float* const* test_inputs, unsigned const* correct_answers);

	// Using the current weights for the model computes the average prediction rate on a
	// set of test data. Considers a guess the output with strictly highest probability.
	// For generalization the test data must not be part of the training data.
	float computePredictionRate(unsigned num_data, float* const* test_inputs, unsigned const* correct_answers);

	// Trains the weights with the current weight decay lambda using stochastic gradient descent.
	void trainWeights(unsigned epoch);

	// Trains the weights using cross validation (10% testing batches) for different values
	// of lambda and learning rate, then trains the full set and outputs the expected error.
	float train_CrossValidation();

#ifdef _CONSOLE
	// Prints the weights to the console.
	void printWeights();
#endif

	// Getters and Setters

	float*** getWeights() const;			// Returns a pointer to the set of weights
	float	 getWeightDecayLambda() const;	// Returns the lambda used for weight decay
	void	 setWeightDecay(float lambda);	// Sets the lambda used for weight decay
	float	 getLearningRate() const;		// Returns the learning rate used for training
	void	 setLearningRate(float rate);	// Sets the learning rate used for training
};