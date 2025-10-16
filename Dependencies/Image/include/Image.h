#pragma once
#include "Color.h"

/* IMAGE CLASS HEADER
-------------------------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------------------------
Inherited library from my 3D renderer project, ment for image handling.
I changed it so that it does not require external dependencies but only 
stores images as bitmaps, enough for the needs of this project.

Stores a color array and allows for basic functionalities, but mostly 
relevant for image file handling.
-------------------------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------------------------
*/

// Simple class for handling images, and storing and loading them as BMP files
class Image
{
private:
	Color* Pixels = nullptr;	// Pointer to the image pixels as a color array

	unsigned int width	= 0u;	// Stores the width of the image
	unsigned int height	= 0u;	// Stores the height of the image

public:
	// Initializes the image as stored in the bitmap file
	Image(const char* fmt_filename, ...);

	// Copies the other image
	Image(const Image& other);

	// Copies the other image
	Image& operator=(const Image& other);

	// Stores a copy of the color pointer
	Image(Color* pixels, unsigned int width, unsigned int height);

	// Creates an image with the specified size and color
	Image(unsigned int width, unsigned int height, Color color = Color::Transparent);

	// Frees the pixel pointer
	~Image();

	// Returns the pointer to the image pixels as a color array
	Color* getPixels();

	// Returns the image width
	unsigned getWidth() const;

	// Returns the image height
	unsigned getHeight() const;

	// Loads an image from the specified file path
	bool load(const char* fmt_filename, ...);

	// Saves the image to the specified file path
	bool save(const char* fmt_filename, ...) const;
};