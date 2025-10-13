#include "NumberRecognition.h"
#include "MNIST_Dataset.h"
#include <cstdlib>

NumberRecognition NumberRecognition::loader;

unsigned* NumberRecognition::trainingLabels;
unsigned* NumberRecognition::testingLabels;

float** NumberRecognition::trainingImages;
float** NumberRecognition::testingImages;

float** NumberRecognition::trainingValues;
float** NumberRecognition::testingValues;

unsigned NumberRecognition::n_training;
unsigned NumberRecognition::n_testing;

/*
-------------------------------------------------------------------------------------------------------
inline  helpers
-------------------------------------------------------------------------------------------------------
*/

static inline uint32_t* computeCenter(uint8_t const* image)
{
    uint32_t total_mass = 0u;
    uint32_t mass_rows = 0u;
    uint32_t mass_cols = 0u;

    uint32_t mass;
    for (uint32_t row = 0; row < IMG_ROWS; row++)
        for (uint32_t col = 0; col < IMG_COLS; col++)
        {
            mass = image[row * IMG_COLS + col];

            total_mass += mass;
            mass_rows += row * mass;
            mass_cols += col * mass;
        }

    uint32_t* center = (uint32_t*)calloc(3, sizeof(uint32_t));
    center[0] = mass_rows / total_mass;
    center[1] = mass_cols / total_mass;
    center[2] = total_mass;

    return center;
}

static inline float computeMassProportion(uint8_t* image, uint32_t* center, uint8_t idx)
{
    uint8_t row_i, row_f, col_i, col_f;

    switch (idx)
    {
    case 0:
        row_i = 0;
        row_f = center[0];
        col_i = 0;
        col_f = center[1];
        break;

    case 1:
        row_i = 0;
        row_f = center[0];
        col_i = center[1];
        col_f = IMG_COLS;
        break;

    case 2:
        row_i = center[0];
        row_f = IMG_ROWS;
        col_i = 0;
        col_f = center[1];
        break;

    case 3:
        row_i = center[0];
        row_f = IMG_ROWS;
        col_i = center[1];
        col_f = IMG_COLS;
        break;

    default:
        return 0;

    }

    uint32_t total_mass = 0;

    for (uint32_t row = row_i; row < row_f; row++)
        for (uint32_t col = col_i; col < col_f; col++)
            total_mass += image[row * IMG_COLS + col];

    return float(total_mass) / center[2];
}

static inline float computeInertiaMoment(uint8_t* image, uint32_t* center, uint8_t idx)
{
    uint8_t row_i, row_f, col_i, col_f;

    switch (idx)
    {
    case 0:
        row_i = 0;
        row_f = center[0];
        col_i = 0;
        col_f = center[1];
        break;

    case 1:
        row_i = 0;
        row_f = center[0];
        col_i = center[1];
        col_f = IMG_COLS;
        break;

    case 2:
        row_i = center[0];
        row_f = IMG_ROWS;
        col_i = 0;
        col_f = center[1];
        break;

    case 3:
        row_i = center[0];
        row_f = IMG_ROWS;
        col_i = center[1];
        col_f = IMG_COLS;
        break;

    default:
        return 0;

    }

    uint32_t mass, r_2;
    uint32_t totalInertia = 0u;

    for (uint32_t row = row_i; row < row_f; row++)
        for (uint32_t col = col_i; col < col_f; col++)
        {
            r_2 = uint32_t((int(row) - int(center[0])) * (int(row) - int(center[0])) + (int(col) - int(center[1])) * (int(col) - int(center[1])));
            mass = image[row * IMG_COLS + col];

            totalInertia += r_2 * mass;
        }

    return float(totalInertia);
}

static inline float computeVerticalSymmetry(uint8_t* image, uint32_t* center, uint8_t idx)
{
    uint8_t col_i, col_f;

    switch (idx)
    {
    case 0:
        col_i = 0;
        col_f = center[1];
        break;

    case 1:
        col_i = center[1];
        col_f = IMG_COLS;
        break;

    default:
        return 0;
    }

    int mass_up, mass_down;
    uint32_t symmetry = 0u;
    for (int8_t row_up = center[0] - 1, row_down = center[0]; row_up >= 0 && row_down < IMG_ROWS; row_up--, row_down++)
    {
        for (uint8_t col = col_i; col < col_f; col++)
        {
            mass_up = image[col + IMG_COLS * row_up];
            mass_down = image[col + IMG_COLS * row_down];

            symmetry += uint32_t((mass_down - mass_up) * (mass_down - mass_up));
        }
    }
    return float(symmetry);
}

static inline float computeHorizontalSymmetry(uint8_t* image, uint32_t* center, uint8_t idx)
{
    uint8_t row_i, row_f;

    switch (idx)
    {
    case 0:
        row_i = 0;
        row_f = center[0];
        break;

    case 1:
        row_i = center[0];
        row_f = IMG_ROWS;
        break;

    default:
        return 0;
    }

    int mass_left, mass_right;
    uint32_t symmetry = 0u;
    for (uint8_t row = row_i; row < row_f; row++)
    {
        for (int8_t col_left = center[1] - 1, col_right = center[1]; col_left >= 0 && col_right < IMG_COLS; col_left--, col_right++)
        {
            mass_left = image[col_left + IMG_COLS * row];
            mass_right = image[col_right + IMG_COLS * row];

            symmetry += uint32_t((mass_left - mass_right) * (mass_left - mass_right));
        }
    }
    return float(symmetry);
}

static inline float computeDiagonalSymmetry(uint8_t* image, uint32_t* center, uint8_t idx)
{
    uint8_t col_i_up, col_i_down, col_f_up, col_f_down;

    switch (idx)
    {
    case 0:
        col_i_up = 0;
        col_f_up = center[1];
        col_i_down = IMG_COLS;
        col_f_down = center[1];
        break;

    case 1:
        col_i_up = center[1];
        col_f_up = IMG_COLS;
        col_i_down = center[1];
        col_f_down = 0;
        break;

    default:
        return 0;
    }

    int mass_up, mass_down;
    uint32_t symmetry = 0u;
    for (int8_t row_up = center[0] - 1, row_down = center[0]; row_up >= 0 && row_down < IMG_ROWS; row_up--, row_down++)
    {
        for (int8_t col_up = col_i_up, col_down = col_i_down - 1; col_up < col_f_up && col_down >= col_f_down; col_up++, col_down--)
        {
            mass_up = image[col_up + IMG_COLS * row_up];
            mass_down = image[col_down + IMG_COLS * row_down];

            symmetry += uint32_t((mass_up - mass_down) * (mass_up - mass_down));
        }
    }
    return float(symmetry);
}

/*
-------------------------------------------------------------------------------------------------------
Constructor / Destructor
-------------------------------------------------------------------------------------------------------
*/

NumberRecognition::NumberRecognition()
{
    MNIST::loadDataSet();
    if (!MNIST::getTrainingSetImages())
        abort();

    n_training = (unsigned)MNIST::get_n_training_images();
    n_testing = (unsigned)MNIST::get_n_testing_images();

    trainingImages = (float**)calloc(n_training, sizeof(float*));
    trainingValues = (float**)calloc(n_training, sizeof(float*));
    trainingLabels = (unsigned*)calloc(n_training, sizeof(unsigned));

    testingImages = (float**)calloc(n_testing, sizeof(float*));
    testingValues = (float**)calloc(n_testing, sizeof(float*));
    testingLabels = (unsigned*)calloc(n_testing, sizeof(unsigned));

    uint8_t* training_labels = MNIST::getTrainingSetLabels();
    uint8_t* testing_labels = MNIST::getTestingSetLabels();

    for (unsigned i = 0; i < n_training; i++)
        trainingLabels[i] = (unsigned)training_labels[i];

    for (unsigned i = 0; i < n_testing; i++)
        testingLabels[i] = (unsigned)testing_labels[i];
}

NumberRecognition::~NumberRecognition()
{

    free(trainingLabels);
    free(testingLabels);

    for (unsigned i = 0; i < n_training; i++)
    {
        if (trainingValues[i])
            free(trainingValues[i]);

        if (trainingImages[i])
            free(trainingImages[i]);
    }

    for (unsigned i = 0; i < n_testing; i++)
    {
        if (testingValues[i])
            free(testingValues[i]);

        if (testingImages[i])
            free(testingImages[i]);
    }


    free(trainingImages);
    free(trainingValues);

    free(testingImages);
    free(testingValues);

}

/*
-------------------------------------------------------------------------------------------------------
User end functions
-------------------------------------------------------------------------------------------------------
*/

float** NumberRecognition::getValues(Set test_train, size_t start_idx, size_t end_idx)
{
    uint8_t** raw_images;
    float** my_values;

    switch (test_train)
    {
    case TESTING:
        raw_images = MNIST::getTestingSetImages();
        my_values = testingValues;
        break;

    case TRAINING:
        raw_images = MNIST::getTrainingSetImages();
        my_values = trainingValues;
        break;

    default:
        return nullptr;
    }

    size_t n_data = end_idx - start_idx;

    for (size_t i = 0; i < n_data; i++)
    {
        size_t idx = i + start_idx;
        uint8_t* image = raw_images[idx];

        if (my_values[idx])
            continue;

        my_values[idx] = (float*)calloc(VALUES_DIM, sizeof(float));

        uint32_t* center = computeCenter(image);
        float totalMass_2 = float(center[2] * center[2]);

        my_values[idx][0] = computeInertiaMoment(image, center, 0) / totalMass_2 * 1000.f;
        my_values[idx][1] = computeInertiaMoment(image, center, 1) / totalMass_2 * 1000.f;
        my_values[idx][2] = computeInertiaMoment(image, center, 2) / totalMass_2 * 1000.f;
        my_values[idx][3] = computeInertiaMoment(image, center, 3) / totalMass_2 * 1000.f;
        my_values[idx][4] = computeVerticalSymmetry(image, center, 0) / totalMass_2 * 1000.f;
        my_values[idx][5] = computeVerticalSymmetry(image, center, 1) / totalMass_2 * 1000.f;
        my_values[idx][6] = computeHorizontalSymmetry(image, center, 0) / totalMass_2 * 1000.f;
        my_values[idx][7] = computeHorizontalSymmetry(image, center, 1) / totalMass_2 * 1000.f;
        my_values[idx][8] = computeDiagonalSymmetry(image, center, 0) / totalMass_2 * 1000.f;
        my_values[idx][9] = computeDiagonalSymmetry(image, center, 1) / totalMass_2 * 1000.f;

        my_values[idx][10] = float(center[2]) / (256.f * IMAGE_DIM);
        my_values[idx][11] = computeMassProportion(image, center, 0);
        my_values[idx][12] = computeMassProportion(image, center, 1);
        my_values[idx][13] = computeMassProportion(image, center, 2);
        my_values[idx][14] = computeMassProportion(image, center, 3);


        free(center);
    }

    return &my_values[start_idx];
}

float** NumberRecognition::getImages(Set test_train, size_t start_idx, size_t end_idx)
{
    uint8_t** raw_images;
    float** my_images;

    switch (test_train)
    {
    case TESTING:
        raw_images = MNIST::getTestingSetImages();
        my_images = testingImages;
        break;

    case TRAINING:
        raw_images = MNIST::getTrainingSetImages();
        my_images = trainingImages;
        break;

    default:
        return nullptr;
    }

    size_t n_data = end_idx - start_idx;

    for (size_t i = 0; i < n_data; i++)
    {
        size_t idx = i + start_idx;
        uint8_t* image = raw_images[idx];

        if (my_images[idx])
            continue;

        my_images[idx] = (float*)calloc(IMAGE_DIM, sizeof(float));

        uint32_t* center = computeCenter(image);
        float averageMasspp = float(center[2]) / (256.f * IMAGE_DIM);
        free(center);

        for (unsigned i = 0; i < IMAGE_DIM; i++)
            my_images[idx][i] = float(image[i]) / 256.f;
    }

    return &my_images[start_idx];
}

unsigned* NumberRecognition::getLabels(Set test_train, size_t start_idx, size_t end_idx)
{
    switch (test_train)
    {
    case TESTING:
        return testingLabels;

    case TRAINING:
        return trainingLabels;

    default:
        return nullptr;
    }
}

void NumberRecognition::printImage(Set test_train, size_t idx)
{
    switch (test_train)
    {
    case TESTING:
        MNIST::consolePrint(MNIST::getTestingSetImages()[idx]);
        return;

    case TRAINING:
        MNIST::consolePrint(MNIST::getTrainingSetImages()[idx]);
        return;

    default:
        return;
    }
}
