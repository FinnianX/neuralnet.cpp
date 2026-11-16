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
vector<float> addFloatToTensor(vector<float> t, float f) {
    for (int i = 0; i < t.size(); i++) {
        t[i] += f;
    }
    return t;
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
        this->Layers = move(Layers);
    }
    vector<float> Forward(const vector<float> &input) {
        vector<float> output = input;
        for (int i = 0; i < Layers.size(); i++) {
            output = Layers[i].pass(output);
        }
        return output;
    }
    void Backward(const vector<float>& input, float groundTruth, float output) {
        float loss = BCELoss(groundTruth, output);
        FFNLinear mutation = createMutation();
        float mutationOutput = mutation.Forward(input)[0];
        if (BCELoss(groundTruth,mutationOutput) < loss) {
            Layers = mutation.Layers;
        }
    }
    FFNLinear createMutation() {
        FFNLinear mutation = *this;
        for (int i = 0; i < Layers.size(); i++) {
            for (int j = 0; j < Layers[i].Neurons.size(); j++) {
                mutation.Layers[i].Neurons[j].weights = addFloatToTensor(Layers[i].Neurons[j].weights, dist6(rng));
            }
        }
        return mutation;
    }
};
int main()
{
    FFNLinear binaryClassifier({ Layer(1,8), Layer(8,1)});
    vector<float> inputs = {1.0, 2.0, 10.0};
    vector<float> input1 = { 1.0 };
    vector<float> outputs = { 0.5, 1.0, 5.0 };
    float untrainedOutput = binaryClassifier.Forward(input1)[0];
    binaryClassifier.Backward(input1, 0.5, untrainedOutput);
    for (int i = 0; i < 100; i++) {
        binaryClassifier.Backward(input1, 0.5, binaryClassifier.Forward(input1)[0]);;
        cout << binaryClassifier.Forward(input1)[0];
    }
    cout << binaryClassifier.Forward(input1)[0];
}
