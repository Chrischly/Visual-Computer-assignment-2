/*
 * Texture.hpp
 *
 *  Class for representing a texture.
 *  by Stefanie Zollmann
 *
 */
#ifndef TEXTURE_HPP
#define TEXTURE_HPP
// Include standard headers
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <vector>

class Texture {
public:
    Texture();
    Texture(std::string filename);
    Texture(int w, int h);
    Texture(unsigned char* data, int width, int height, bool bgrFormat = true);
    ~Texture();

    void bindTexture();
    GLuint getTextureID();
    void update(unsigned char* data, int width, int height, bool bgrFormat = true);


private:
    GLuint loadBMP_custom(const char* imagepath);
    GLuint loadDDS(const char* imagepath);

    GLuint m_textureID;
};

#endif