#include <vector>

class BasicTensor {
private:
    std::vector<float> data;
    size_t width;
    size_t height;

public:
    // --- Constructors ---
    BasicTensor(size_t h, size_t w);
    BasicTensor(std::vector<float> d, size_t h, size_t w);

    // --- Getters ---
    size_t getWidth() const;
    size_t getHeight() const;

    // --- Math Operations ---
    BasicTensor operator*(const BasicTensor& other) const;
    BasicTensor operator+(const BasicTensor& other) const;

    // --- Standard Iterators (Read/Write) ---
    std::vector<float>::iterator begin();
    std::vector<float>::iterator end();

    // --- Const Iterators (Read-Only) ---
    std::vector<float>::const_iterator begin() const;
    std::vector<float>::const_iterator end() const;

    // --- Utility ---
    void display() const;
};
