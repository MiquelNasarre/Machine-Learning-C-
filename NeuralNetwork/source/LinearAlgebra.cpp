#include "LinearAlgebra.h"

#include <math.h>
#include <stdlib.h>

//-----------------
// VECTOR FUNCTIONS
// ----------------

/*
-------------------------------------------------------------------------------------------------------
Initializers and copy functions
-------------------------------------------------------------------------------------------------------
*/

// Allocates a null vector of size n.

Vector::Vector(size_t n)
{
	init(n);
}

void Vector::init(size_t n)
{
	if (data_ && owner_)
		free(data_);

	size_ = n;
	owner_ = true;
	data_ = (float*)calloc(size_, sizeof(float));
}

// Generates a vector from the data provided, copying
// or borrowing the pointer as indicated.

Vector::Vector(float* data, size_t n, bool own)
{
	set_data(data, n, own);
}

void Vector::set_data(float* new_data, size_t size, bool own)
{
	if (data_ && owner_)
		free(data_);

	if (!own)
	{
		data_ = new_data;
		size_ = size;
		owner_ = false;
	}
	else
	{
		data_ = (float*)calloc(size, sizeof(float));
		for (size_t i = 0; i < size; i++)
			data_[i] = new_data[i];
		size_ = size;
		owner_ = true;
	}
}

// Deep copies.

Vector::Vector(const Vector& other)
{
	*this = other;
}

Vector& Vector::operator=(const Vector& other)
{
	copy_from(other);
	return *this;
}

void Vector::copy_from(const Vector& src)
{
	if (data_ && owner_)
		free(data_);

	size_ = src.size_;
	data_ = (float*)calloc(size_, sizeof(float));

	for (size_t i = 0; i < size_; i++)
		data_[i] = src.data_[i];

	owner_ = true;
}

// Referencing.

Vector::Vector(Vector&& other) noexcept
{
	if (data_ && owner_)
		free(data_);

	data_ = other.data_;
	size_ = other.size_;
	owner_ = false;
}

Vector& Vector::operator=(Vector&& other) noexcept
{
	if (data_ && owner_)
		free(data_);

	data_ = other.data_;
	size_ = other.size_;
	owner_ = false;

	return *this;
}

// Frees the vector data.

Vector::~Vector()
{
	if (data_ && owner_)
		free(data_);
}

/*
-------------------------------------------------------------------------------------------------------
Getters
-------------------------------------------------------------------------------------------------------
*/

// Returns the data pointer

float* Vector::data()
{
	return data_;
}

// Returns the data pointer as const

const float* Vector::data() const
{
	return data_;
}

// Returns the vector size

size_t Vector::size() const
{
	return size_;
}

// Computes the squared norm of the vector

float Vector::norm2() const
{
#ifdef _DEBUG
	if (!data_) throw("Trying to calculate the norm of an empty vector");
#endif
	float result = 0.f;

	for (size_t i = 0; i < size_; i++)
		result += data_[i] * data_[i];

	return result;
}

// Returns the reference to the vector index i.

float& Vector::operator()(size_t i)
{
#ifdef _DEBUG
	if (!data_) throw("Trying to access a component of an empty vector");
#endif
	return data_[i];
}

// Returns the const reference to the vector index i.

const float& Vector::operator()(size_t i) const
{
#ifdef _DEBUG
	if (!data_) throw("Trying to access a component of an empty vector");
#endif
	return data_[i];
}

/*
-------------------------------------------------------------------------------------------------------
Fill helpers
-------------------------------------------------------------------------------------------------------
*/

// Sets all values to zero

void Vector::zero()
{
#ifdef _DEBUG
	if (!data_) throw("Trying to zero an empty vector");
#endif
	for (size_t i = 0; i < size_; i++)
		data_[i] = 0.f;
}

// Sets all valuea constant

void Vector::set_constant(float v)
{
#ifdef _DEBUG
	if (!data_) throw("Trying to set constant an empty vector");
#endif
	for (size_t i = 0; i < size_; i++)
		data_[i] = v;
}

// Scales all values by a factor

void Vector::scale(float v)
{
#ifdef _DEBUG
	if (!data_) throw("Trying to scale an empty vector");
#endif
	for (size_t i = 0; i < size_; i++)
		data_[i] *= v;
}

/*
-------------------------------------------------------------------------------------------------------
Activator functions
-------------------------------------------------------------------------------------------------------
*/

// multiply by ReLU'(self)

void Vector::relu_grad_inplace(const Vector& grad)
{
	for (size_t i = 0; i < size_; i++)
		if (grad.data_[i] <= 0.f)
			data_[i] = 0.f;
}

// Applies relu to the vector values

void Vector::_relu()
{
	for (size_t i = 0; i < size_; i++)
		if (data_[i] < 0.f)
			data_[i] = 0.f;
}

// Applies softmax to the vector values

void Vector::_softmax()
{
	float max = data_[0];
	for (size_t i = 1; i < size_; i++)
		if (max < data_[i])
			max = data_[i];

	float total = 0.f;
	for (unsigned i = 0; i < size_; i++)
	{
		data_[i] = expf(data_[i] - max);
		total += data_[i];
	}

	for (unsigned i = 0; i < size_; i++)
		data_[i] /= total;
}

//-----------------
// MATRIX FUNCTIONS
// ----------------

/*
-------------------------------------------------------------------------------------------------------
Initializers and copy functions
-------------------------------------------------------------------------------------------------------
*/

// Aligned allocate (ld_ >= cols, row-major).

Matrix::Matrix(size_t rows, size_t cols)
{
	init(rows, cols);
}

void Matrix::init(size_t rows, size_t cols)
{
	if (data_ && owner_) 
		free(data_);

	rows_ = rows; cols_ = cols;
	ld_ = cols;
	owner_ = true;
	data_ = (float*)calloc(rows_ * ld_, sizeof(float));
}

// Generates a matrix from the data provided, copying
// or borrowing the pointer as indicated.

Matrix::Matrix(float* data, size_t rows, size_t cols, size_t ld, bool own)
{
	set_data(data, rows, cols, ld, own);
}

void Matrix::set_data(float* new_data, size_t rows, size_t cols, size_t ld, bool own)
{
	if (data_ && owner_)
		free(data_);

	if (!own)
	{
		data_ = new_data;
		rows_ = rows;
		cols_ = cols;
		ld_ = ld;
		owner_ = false;
	}
	else
	{
		rows_ = rows;
		cols_ = cols;
		ld_ = ld;
		data_ = (float*)calloc(rows_ * ld_, sizeof(float));

		for (size_t i = 0; i < rows_; i++)
			for (size_t j = 0; j < cols_; j++)
				data_[i * ld_ + j] = new_data[i * ld_ + j];

		owner_ = true;
	}
}

// Deep copies.

Matrix::Matrix(const Matrix& other)
{
	*this = other;
}

Matrix& Matrix::operator=(const Matrix& other)
{
	copy_from(other);
	return *this;
}

void Matrix::copy_from(const Matrix& src)
{
	if (&src == this)
		return;
	if (data_ && owner_)
		free(data_);

	rows_ = src.rows_;
	cols_ = src.cols_;
	ld_ = src.ld_;
	
	data_ = (float*)calloc(rows_ * ld_, sizeof(float));
	
	for (size_t i = 0; i < rows_; i++)
		for (size_t j = 0; j < cols_; j++)
			data_[i * ld_ + j] = src(i, j);

	owner_ = true;
}

// Referencing.

Matrix::Matrix(Matrix&& other) noexcept
{
	*this = other;
}

Matrix& Matrix::operator=(Matrix&& other) noexcept
{
	if (&other == this)
		return *this;
	if (data_ && owner_)
		free(data_);

	owner_ = false;
	data_ = other.data_;
	cols_ = other.cols_;
	rows_ = other.rows_;
	ld_ = other.ld_;
	return *this;
}

// Frees the matrix data.

Matrix::~Matrix()
{
	if (data_ && owner_)
		free(data_);
}

/*
-------------------------------------------------------------------------------------------------------
Getters
-------------------------------------------------------------------------------------------------------
*/

// Returns the data pointer

float* Matrix::data()
{
	return data_;
}

// Returns the data pointer as const

const float* Matrix::data() const
{
	return data_;
}

// Returns the number of rows

size_t Matrix::rows() const
{
	return rows_;
}

// Returns the number of columns

size_t Matrix::cols() const
{
	return cols_;
}

// Returns the leading dimension

size_t Matrix::ld() const
{
	return ld_;
}

// Returns the reference to the matrix position i,j.

float& Matrix::operator()(size_t i, size_t j)
{
#ifdef _DEBUG
	if (!data_) throw("Trying to access a parameter in an empty matrix");
#endif
	return data_[i * ld_ + j];
}

// Returns the const reference to the matrix position i,j.

const float& Matrix::operator()(size_t i, size_t j) const
{
#ifdef _DEBUG
	if (!data_) throw("Trying to access a parameter in an empty matrix");
#endif
	return data_[i * ld_ + j];
}

/*
-------------------------------------------------------------------------------------------------------
Fill helpers
-------------------------------------------------------------------------------------------------------
*/

// Sets all values to zero

void Matrix::zero()
{
#ifdef _DEBUG
	if (!data_) throw("Trying to set to zero an empty matrix");
#endif
	for (size_t i = 0; i < rows_; i++)
		for (size_t j = 0; j < cols_; j++)
			data_[i * ld_ + j] = 0.f;
}

// Sets all valuea constant

void Matrix::set_constant(float v)
{
#ifdef _DEBUG
	if (!data_) throw("Trying to set constant an empty matrix");
#endif
	for (size_t i = 0; i < rows_; i++)
		for (size_t j = 0; j < cols_; j++)
			data_[i * ld_ + j] = v;
}

// Scales all values by a factor

void Matrix::scale(float v)
{
#ifdef _DEBUG
	if (!data_) throw("Trying to scale an empty matrix");
#endif
	for (size_t i = 0; i < rows_; i++)
		for (size_t j = 0; j < cols_; j++)
			data_[i * ld_ + j] *= v;
}

/*
-------------------------------------------------------------------------------------------------------
Matrix and vector operations
-------------------------------------------------------------------------------------------------------
*/

// Returns the dot product between two vectors.

float dot(const Vector& x, const Vector& y)
{
#ifdef _DEBUG
	if (y.size_ != x.size_) throw("dot product vector size mismatch");
#endif

	float dotpr = 0.f;
	for (size_t i = 0; i < y.size_; i++)
		dotpr += x.data_[i] * y.data_[i];

	return dotpr;
}

// C <- alpha * AxB + beta * C

void gemm(const Matrix& A, const Matrix& B, Matrix& C,
	float alpha, float beta, bool transA, bool transB)
{
	const size_t Am = transA ? A.cols_ : A.rows_;
	const size_t Ak = transA ? A.rows_ : A.cols_;
	const size_t Bk = transB ? B.cols_ : B.rows_;
	const size_t Bn = transB ? B.rows_ : B.cols_;

#ifdef _DEBUG
	if (Ak != Bk || C.rows_ != Am || C.cols_ != Bn) throw("gemm shape mismatch");
#endif

	// scale C by beta
	if (beta == 0.f) C.zero();
	else if (beta != 1.f) C.scale(beta);

	float* data_A = A.data_;
	float* data_B = B.data_;
	float* data_C = C.data_;

	size_t pa_mul = transA ? A.ld_ : 1;
	size_t pb_mul = transB ? 1 : B.ld_;
	size_t ia_mul = transA ? 1 : A.ld_;
	size_t jb_mul = transB ? B.ld_ : 1;
	size_t ic_mul = C.ld_;

	for (size_t i = 0; i < Am; ++i) {
		for (size_t j = 0; j < Bn; ++j) {
			float sum = 0.f;
			for (size_t p = 0; p < Ak; ++p)
				sum += data_A[p * pa_mul + i * ia_mul] * data_B[j * jb_mul + p * pb_mul];

			data_C[i * ic_mul + j] += alpha * sum;
		}
	}
}

// c <- alpha * Axb + beta * c

void gemm(const Matrix& A, const Vector& b, Vector& c,
	float alpha, float beta, bool transA)
{
	const size_t Am = transA ? A.cols_ : A.rows_;
	const size_t Ak = transA ? A.rows_ : A.cols_;

#ifdef _DEBUG
	if (b.size_ != Ak || c.size_ != Am) throw("gemv shape mismatch");
#endif

	float* data_A = A.data_;
	float* data_b = b.data_;
	float* data_c = c.data_;

	size_t pa_mul = transA ? A.ld_ : 1;
	size_t ia_mul = transA ? 1 : A.ld_;

	if (beta == 0.f) c.zero();
	else if (beta != 1.f) c.scale(beta);

	for (size_t i = 0; i < Am; ++i) {
		float sum = 0.f;
		for (size_t p = 0; p < Ak; ++p)
			sum += data_A[p * pa_mul + i * ia_mul] * data_b[p];

		data_c[i] += alpha * sum;
	}
}

// C <- alpha * axb^T + beta * C

void gemm(const Vector& a, const Vector& b, Matrix& C,
	float alpha, float beta)
{
#ifdef _DEBUG
	if (C.rows_ != a.size_ || C.cols_ != b.size_) throw("outer size mismatch");
#endif

	if (beta == 0.f) C.zero();
	else if (beta != 1.f) C.scale(beta);

	float* data_a = a.data_;
	float* data_b = b.data_;
	float* data_C = C.data_;

	for (size_t i = 0; i < a.size_; ++i) {
		const float ai = alpha * data_a[i];
		float* row = data_C + i * C.ld_;
		for (size_t j = 0; j < b.size_; ++j)
			row[j] += ai * data_b[j];
	}
}

// z <- Xxa + b

void axpb(const Matrix& X, const Vector& a, const Vector& b, Vector& z)
{
#ifdef _DEBUG
	if (X.cols_ != a.size_ || X.rows_ != b.size_ || z.size_ != b.size_)
		throw("axpb shape mismatch");
#endif
	// z = X*a
	gemm(X, a, z, 1.f, 0.f, false);
	// z += b
	axpy(b, z, 1.f);
}

// Y <- alpha * X + Y

void axpy(const Matrix& X, Matrix& Y, float alpha)
{
#ifdef _DEBUG
	if (X.rows_ != Y.rows_ || X.cols_ != Y.cols_) throw("axpy shape mismatch");
#endif
	for (size_t i = 0; i < X.rows_; ++i) {
		const float* xrow = X.data_ + i * X.ld_;
		float* yrow = Y.data_ + i * Y.ld_;
		for (size_t j = 0; j < X.cols_; ++j)
			yrow[j] += alpha * xrow[j];
	}
}

// y <- alpha * x + y

void axpy(const Vector& x, Vector& y, float alpha)
{
	for (size_t i = 0; i < y.size_; i++)
		y.data_[i] += alpha * x.data_[i];
}

// Y <- alpha * X + beta * Y

void axpby(const Matrix& X, Matrix& Y, float alpha, float beta)
{
#ifdef _DEBUG
	if (X.rows_ != Y.rows_ || X.cols_ != Y.cols_) throw("axpby shape mismatch");
#endif
	if (beta == 0.f) {
		for (size_t i = 0; i < X.rows_; ++i) {
			const float* xrow = X.data_ + i * X.ld_;
			float* yrow = Y.data_ + i * Y.ld_;
			for (size_t j = 0; j < X.cols_; ++j)
				yrow[j] = alpha * xrow[j];
		}
	}
	else {
		const size_t xrows = X.rows_;
		const size_t xcols = X.cols_;
		const size_t xld = X.ld_;
		const size_t yld = Y.ld_;
		const float* xdata = X.data_;
		float* ydata = Y.data_;
		
		for (size_t i = 0; i < xrows; ++i) {
			const float* xrow = xdata + i * xld;
			float* yrow = ydata + i * yld;
			for (size_t j = 0; j < xcols; ++j, ++yrow, ++xrow)
			{
				*yrow *= beta;
				*yrow += alpha * *xrow;
			}

		}
	}
}

// y <- alpha * x + beta * y

void axpby(const Vector& x, Vector& y, float alpha, float beta)
{
	for (size_t i = 0; i < y.size_; i++)
		y.data_[i] = alpha * x.data_[i] + beta * y.data_[i];
}

// C <- A + B

void add(const Matrix& A, const Matrix& B, Matrix& C)
{
#ifdef _DEBUG
	if (A.rows_ != B.rows_ || A.cols_ != B.cols_ ||
		C.rows_ != A.rows_ || C.cols_ != A.cols_) throw("add shape mismatch");
#endif
	for (size_t i = 0; i < A.rows_; ++i) {
		const float* arow = A.data_ + i * A.ld_;
		const float* brow = B.data_ + i * B.ld_;
		float* crow = C.data_ + i * C.ld_;
		for (size_t j = 0; j < A.cols_; ++j)
			crow[j] = arow[j] + brow[j];
	}
}

// c <- a + b

void add(const Vector& a, const Vector& b, Vector& c)
{
	for (size_t i = 0; i < c.size_; i++)
		c.data_[i] = a.data_[i] + b.data_[i];
}