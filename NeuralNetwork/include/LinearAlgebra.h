#pragma once
#include <cstdint>
#include <cstddef>

/* LINEAR ALGEBRA CLASSES HEADER
-------------------------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------------------------
This header contains linear algebra objects to abstract fast
operations from the Neural Network class.

In particular it contains the Vector class and the Matrix class.
Both allow for arbitrary sized objects and multiple friend operations
are defined to allow for fast computation of typical operations used
for Machine Learning algorightms.
-------------------------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------------------------
*/

class Matrix;
class Vector;

// This class stores a vector of arbitrary size n and allows for
// fast vector operations through his friend functions. 
// Any value can be accessed and modified via Vector(i).
class Vector
{
private:
    float* data_ = nullptr;         // Stores the vector indexs
    size_t  size_ = 0ULL;           // Stores the size of the vector
    bool    owner_ = false;         // Stores whether the data is owned or borrowed

public:

    // Default constructor for an empty vector.
    Vector() = default;

    // Allocates a null vector of size n.
    Vector(size_t n);
    void init(size_t n);

    // Generates a vector from the data provided, copying
    // or borrowing the pointer as indicated.
    Vector(float* data, size_t n, bool own = false);
    void set_data(float* new_data, size_t size);  // Non-owning

    // Deep copies.
    Vector(const Vector& other);
    Vector& operator=(const Vector& other);
    void copy_from(const Vector& src);


    // Referencing.
    Vector(Vector&& other) noexcept;
    Vector& operator=(Vector&& other) noexcept;

    // Frees the vector data.
    ~Vector();

    // Getters

    inline float* data();               // Returns the data pointer
    inline const float* data() const;   // Returns the data pointer as const
    inline size_t size() const;         // Returns the vector size
    float norm2() const;                // Computes the squared norm of the vector

    // Returns the reference to the vector index i.
    inline float& operator()(size_t i);
    // Returns the const reference to the vector index i.
    inline const float& operator()(size_t i) const;

    // Fill helpers

    void zero();                    // Sets all values to zero
    void set_constant(float v);     // Sets all valuea constant
    void scale(float v);            // Scales all values by a factor

    // Activator functions

    void relu_grad_inplace(const Vector& grad); // multiply by ReLU'(self)

    void _relu();       // Applies relu to the vector values
    void _softmax();    // Applies softmax to the vector values

#pragma region // Friend operations
    friend float dot(const Vector&, const Vector&);
    friend void axpy(const Vector&, Vector&, float);
    friend void gemm(const Matrix&, const Matrix&, Matrix&, float, float, bool, bool);
    friend void gemm(const Matrix&, const Vector&, Vector&, float, float, bool);
    friend void gemm(const Vector&, const Vector&, Matrix&, float, float);
    friend void axpb(const Matrix&, const Vector&, const Vector&, Vector&);
    friend void axpy(const Matrix&, Matrix&, float);
    friend void axpby(const Matrix&, Matrix&, float, float);
    friend void add(const Matrix&, const Matrix&, Matrix&);
#pragma endregion
};

// This class stores a matrix of arbitrary size nxm and allows for fast matrix
// operations through his friend functions. The matrix layout is row major so
// the indexation is idx = i(row) * ld(leading dimension) + j(col).
// Any value can be accessed and modified via Matrix(i,j).
class Matrix
{
private:
    float* data_ = nullptr;     // Stores the matrix indices
    size_t  rows_ = 0ULL;       // Stores the number of rows
    size_t cols_ = 0ULL;        // Stores the number of columns
    size_t ld_ = 0ULL;          // Stores the leading dimension (stride in elements)
    bool owner_ = false;        // Stores whether the data is own or inherited

public:

    // Default constructor for an empty matrix.
    Matrix() = default;

    // Aligned allocate (ld_ >= cols, row-major).
    Matrix(size_t rows, size_t cols);
    void init(size_t rows, size_t cols);

    // Generates a matrix from the data provided, copying
    // or borrowing the pointer as indicated.
    Matrix(float* data, size_t rows, size_t cols, size_t ld, bool own = false);

    // Deep copies.
    Matrix(const Matrix&);
    Matrix& operator=(const Matrix&);
    void copy_from(const Matrix& other);

    // Referencing.
    Matrix(Matrix&&) noexcept;
    Matrix& operator=(Matrix&&) noexcept;

    // Frees the matrix data.
    ~Matrix();

    // Getters

    inline float* data();               // Returns the data pointer
    inline const float* data() const;   // Returns the data pointer as const
    inline size_t rows() const;         // Returns the number of rows
    inline size_t cols() const;         // Returns the number of columns
    inline size_t ld() const;           // Returns the leading dimension

    // Returns the reference to the matrix position i,j.
    inline float& operator()(size_t i, size_t j);
    // Returns the const reference to the matrix position i,j.
    inline const float& operator()(size_t i, size_t j) const;

    // Fill helpers

    void zero();                    // Sets all values to zero
    void set_constant(float v);     // Sets all valuea constant
    void scale(float v);            // Scales all values by a factor

#pragma region // Friend operations
    friend float dot(const Vector&, const Vector&);
    friend void axpy(const Vector&, Vector&, float);
    friend void gemm(const Matrix&, const Matrix&, Matrix&, float, float, bool, bool);
    friend void gemm(const Matrix&, const Vector&, Vector&, float, float, bool);
    friend void gemm(const Vector&, const Vector&, Matrix&, float, float);
    friend void axpb(const Matrix&, const Vector&, const Vector&, Vector&);
    friend void axpy(const Matrix&, Matrix&, float);
    friend void axpby(const Matrix&, Matrix&, float, float);
    friend void add(const Matrix&, const Matrix&, Matrix&);
#pragma endregion
};

// Friend operations declaration

// Returns the dot product between two vectors.
float dot(const Vector& x, const Vector& y);

// C <- alpha * AxB + beta * C
void gemm(const Matrix& A, const Matrix& B, Matrix& C,
    float alpha = 1.f, float beta = 0.f, bool transA = false, bool transB = false);

// c <- alpha * Axb + beta * c
void gemm(const Matrix& A, const Vector& b, Vector& c,
    float alpha = 1.f, float beta = 0.f, bool transA = false);

// C <- alpha * axb^T + beta * C
void gemm(const Vector& a, const Vector& b, Matrix& C,
    float alpha = 1.f, float beta = 0.f);

// z <- Xxa + b
void axpb(const Matrix& X, const Vector& a, const Vector& b, Vector& z);

// Y <- alpha * X + Y
void axpy(const Matrix& X, Matrix& Y, float alpha = 1.f);

// y <- alpha * x + y
void axpy(const Vector& x, Vector& y, float alpha);

// Y <- alpha * X + beta * Y
void axpby(const Matrix& X, Matrix& Y, float alpha = 1.f, float beta = 0.f);

// y <- alpha * x + beta * y
void axpby(const Vector& x, Vector& y, float alpha = 1.f, float beta = 0.f);

// C <- A + B
void add(const Matrix& A, const Matrix& B, Matrix& C);

// c <- a + b
void add(const Vector& a, const Vector& b, Vector& c);