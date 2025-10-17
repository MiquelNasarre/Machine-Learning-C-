#pragma region // Don't look in here
class layer
{
public:
	friend bool bind(layer, layer, size_t, size_t, size_t, size_t);
};
enum ActivationFunction
{
	_RELU,
	_SOFTMAX
};
class NeuralNetwork
{
public:
	NeuralNetwork(size_t, size_t);

	layer& input();
	layer& output();

	bool build();
	bool storeWeights();

	void feedData(size_t, float**, float**);
	void train();
};
class NormalLayer : public layer
{
public:
	NormalLayer(size_t, size_t, ActivationFunction);
};
class ConvolutionLayer : public layer
{
public:
	ConvolutionLayer(size_t, size_t*, size_t, size_t, ActivationFunction);
};
bool bind(layer, layer, size_t, size_t, size_t, size_t);

float** data_for_input_provider(size_t);
float** data_for_output_provider(size_t);

#define DIM 2
#define IMAGE_ROWS 32
#define IMAGE_COLS 32

#define OUTPUT_LAYER_SIZE 1024 * 3
#define LAST_NODE -1
#define ERROR_CHECK(a) a
#define DATASET_SIZE 50000

#pragma endregion

// This is a little showcase of how I would like my Neural Network infrastructure
// to be, of course it does not work but the idea is a lot of freedom of how you 
// want to build the NN, and a lot of abstraction, while everything is running on
// your GPU you always have access to what is going on in the way of pointers.
// 
// This is just the brainstorming of a night and a lot of changes might be needed 
// to make it possible or more user friendly, but I think is a promising idea.

#define INPUT_LAYER_SIZE IMAGE_ROWS * IMAGE_COLS * 3
#define OUTPUT_LAYER_SIZE 100

int main()
{
	// Creates the neural network with the desired amount of inputs
	// and outputs but the desing still needs to be provided.
	NeuralNetwork NN(INPUT_LAYER_SIZE, OUTPUT_LAYER_SIZE);

	// Initializes 3 convolution layers that will operate in paralel 
	// over every channel of the input image.
	size_t image_dim[DIM] = { IMAGE_ROWS, IMAGE_COLS };
	ConvolutionLayer convLayer[3] = {
		{ DIM, image_dim, 3ULL /*Grid size*/, 1ULL /*Offset*/, _RELU },
		{ DIM, image_dim, 3ULL /*Grid size*/, 1ULL /*Offset*/, _RELU },
		{ DIM, image_dim, 3ULL /*Grid size*/, 1ULL /*Offset*/, _RELU },
	};

	// Binds each convolution layer to the input nodes corresponding to 
	// their channel in the image.
	for (unsigned c = 0; c < 3; c++)
		ERROR_CHECK(bind(NN.input(), convLayer[c], c * IMAGE_ROWS * IMAGE_COLS, (c + 1) * IMAGE_ROWS * IMAGE_COLS, 0, LAST_NODE));

	// Initializes the reduction layer that will take as input the three
	// convolution layer outputs and output a 256 sized vector.
	NormalLayer reductionLayer(INPUT_LAYER_SIZE, 256, _RELU);

	// Binds each convolution layer output to their corresponding
	// input nodes in the reduction layer.
	for (unsigned c = 0; c < 3; c++)
		ERROR_CHECK(bind(convLayer[c], reductionLayer, 0, LAST_NODE, c * IMAGE_ROWS * IMAGE_COLS, (c + 1) * IMAGE_ROWS * IMAGE_COLS));

	// Initializes t the layer that will do the final computation
	// that will write to the neural network output.
	NormalLayer outputBindLayer(256, OUTPUT_LAYER_SIZE, _SOFTMAX);

	// Binds the input of the last layer to the output of the reduction layer.
	ERROR_CHECK(bind(reductionLayer, outputBindLayer, 0, LAST_NODE, 0, LAST_NODE));

	// Binds the output of the last layer to the output of the neural network
	ERROR_CHECK(bind(outputBindLayer, NN.output(), 0, LAST_NODE, 0, LAST_NODE));

	// Generates the entire neural network infrastructure with the specified
	// layer layout, creating all the necessary functions in the GPU and the 
	// paths for forward passes and back propagation.
	ERROR_CHECK(NN.build());

	// Gets the training data.
	float** training_input = data_for_input_provider(DATASET_SIZE);
	float** training_output = data_for_output_provider(DATASET_SIZE);

	// Feeds the training data to the NN.
	NN.feedData(DATASET_SIZE, training_input, training_output);

	// Starts the training...
	NN.train();

	// Thera are may more requirements and featuresof the NN, for example each
	// output (or the NN input layer) could bind to multiple inputs, making it so
	// that you can branch multiple outputs and study them separately, as long as 
	// they are eventually combined somewhere and there are no hanging nodes.
	// 
	// Also each layer you created will hold pointers to their data in the GPU.
	// So you could destroy the layer object without an issue, because the NN
	// will have access to all the necessarey data, but you can also keep them
	// to allow yourself interaction with the GPU data of each individual layer,
	// Including nodes, weights and biases.
	// 
	// This will also allow for easy implementation of any other kind of layers,
	// since it promotes some type of build it yourself neural network. 
	// 
	// There are many cool ideas of what you can do with this kind of layout
	// but due to amount of time and effort it would require building all the
	// code to run this neural networks it stays as a project for the furute.

	NN.storeWeights();
	return 0;
}