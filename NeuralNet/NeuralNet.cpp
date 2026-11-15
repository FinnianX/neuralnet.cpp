#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <map>
#include <algorithm>
#include <cmath>
#include <numbers>

using namespace std;
random_device dev;
mt19937 rng(dev());
uniform_real_distribution<float> dist6(-0.05f, 0.05f);

map<int, string> tokenizer = {
    { 1, "a" },{ 2, "b" },{ 3, "c" },{ 4, "d" },{ 5, "e" },
    { 6, "f" },{ 7, "g" },{ 8, "h" },{ 9, "i" },{ 10, "j" },
    { 11, "k" },{ 12, "l" },{ 13, "m" },{ 14, "n" },{ 15, "o" },
    { 16, "p" },{ 17, "q" },{ 18, "r" },{ 19, "s" },{ 20, "t" },
    { 21, "u" },{ 22, "v" },{ 23, "w" },{ 24, "x" },{ 25, "y" },
    { 26, "z" },{27, " "}
};

float ReLU(float input) {
    if (input > 0.0) { return input; }
    else { return 0.0; }
}
float sigmoid(float input) {
    return 1 / (1 + pow(numbers::e, -1 * input));
}
float BCELoss(float truth, float predicted) { 
    // Truth must be 1.0 or 0.0
    // Requires a sigmoided input
    return -1 * (truth * log(predicted) + (1 - truth) * log(1 - predicted));
}
vector<vector<float>> transposeMatrix(vector<vector<float>> mat1) {
    //Create opposite of mat1, its dimensions are swapped
    vector<vector<float>> mat2(mat1[0].size(), vector<float>(mat1.size()));
    for (int i = 0; i < mat1.size(); i++) {
        for (int j = 0; j < mat2.size(); j++) {
            // swapping values, super confusing just dont touch because it works
            mat2[j][i] = mat1[i][j];
        }
    }
    return mat2;
}
vector<vector<float>> multiplyMatrixByFloat(float factor, vector<vector<float>> matrix) {
    for (int i = 0; i < matrix.size(); i++) {
        for (int j = 0; j < matrix[i].size(); j++) {
            matrix[i][j] *= factor;
        }
    }
    return matrix;
}
class Neuron {
public:
    vector<float> weights;
    float bias;
    float output = 0.0f;
    float outputWithActivation = 0.0f;
    Neuron(int inputCount) {
        for (int i = 0; i < inputCount; i++) {
            weights.push_back(dist6(rng));
        }
        this->bias = dist6(rng);
    }
    float pass(vector<float> &inputs) {
        float weightedSum = 0.0;
        for (int i = 0; i < inputs.size(); i++) {
            weightedSum += inputs[i] * weights[i];
        }
        weightedSum += bias;
        output = weightedSum;
        outputWithActivation = ReLU(weightedSum);
        return outputWithActivation;
    }
};

class Layer {
public:
    int width;
    int inputs;
    vector<Neuron> Neurons;
    Layer(int inputs,int width) {
        this->width = width;
        this->inputs = inputs;
        for (int i = 0; i < width; i++) {
            Neurons.emplace_back(Neuron(inputs));
        }
    }
    vector<float> pass(vector<float> &inputs) {
        vector<float> outputs;
        for (int i = 0; i < Neurons.size(); i++) {    
            outputs.push_back(Neurons[i].pass(inputs));
        }
        return outputs;
    }
};
class FFNLinear {
public:
    vector<Layer> Layers;
    FFNLinear(vector<Layer> Layers) {
        this->Layers = Layers;
    }
    vector<float> Forward(const vector<float> &input) {
        vector<float> output = input;
        for (int i = 0; i < Layers.size(); i++) {
            output = Layers[i].pass(output);
        }
        return output;
    }
    vector<float> Backward(float groundTruth, float output) {
        // None of this works
        float loss = BCELoss(groundTruth, output);
        float finalOutputError = output - groundTruth;
        vector<vector<float>> weightedErrorContributions(Layers.size());
        for (int i = Layers.size()-1; i > 0; i--) {
            vector<vector<float>> weightsToNextLayer;
            for (int j = 0; j < Layers[i+1].Neurons.size(); j++) {
                weightsToNextLayer.push_back(Layers[i+1].Neurons[j].weights);
            }
            //weightedErrorContributions[i].push_back(multiplyMatrixByFloat(finalOutputError, weightsToNextLayer));
        }
    }
};
int main()
{
    FFNLinear binaryClassifier({ Layer(1,8), Layer(8,16),Layer(16,8),Layer(8,1) });
    vector<vector<float>> matrix2transpose = { { 1.0, 1.0, 2.0 },
                              { 2.0, 1.0, 1.0 }};
    vector<vector<float>> final = transposeMatrix(matrix2transpose);
    for (int i = 0; i < final.size(); i++) {
        for (int j = 0; j < final[i].size(); j++) {
            cout << final[i][j] << " ";
        }
        cout << endl;
    }
}