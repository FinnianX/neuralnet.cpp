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
uniform_real_distribution<float> dist6(-0.5f, 0.5f);
uniform_real_distribution<float> percent(0, 1);


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
float MSELoss(float truth, float predicted) {
    float difference = truth - predicted;
    return difference * difference;
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
    float (*activation)(float);
    Neuron(int inputCount, float (*activation)(float) = ReLU) {
        for (int i = 0; i < inputCount; i++) {
            weights.push_back(dist6(rng));
        }
        this->bias = dist6(rng);
        this->activation = activation;
    }
    float pass(vector<float> &inputs) {
        float weightedSum = 0.0;
        for (int i = 0; i < inputs.size(); i++) {
            weightedSum += inputs[i] * weights[i];
        }
        weightedSum += bias;
        output = weightedSum;
        outputWithActivation = activation(weightedSum);
        return outputWithActivation;
    }
};

class Layer {
public:
    int width;
    int inputs;
    vector<Neuron> Neurons;
    float (*activation)(float);
    Layer(int inputs,int width, float (*activation)(float) = ReLU) {
        this->width = width;
        this->inputs = inputs;
        for (int i = 0; i < width; i++) {
            Neurons.emplace_back(Neuron(inputs,activation));
        }
        this->activation = activation;
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
    FFNLinear(vector<Layer> Layers = {Layer(1,1)}) {
        this->Layers = move(Layers);
    }
    vector<float> Forward(const vector<float> &input) {
        vector<float> output = input;
        for (int i = 0; i < Layers.size(); i++) {
            output = Layers[i].pass(output);
        }
        return output;
    }
    void Backward(const vector<float>& input, const vector<float>& groundTruths, float lr=0.015, float mutationChance=0.3, int population=40) {
        float averageFinalLoss = 0.0;
        for (int i = 0; i < input.size(); i++) {
            averageFinalLoss += MSELoss(groundTruths[i],Forward({ input[i] })[0]);
        }
        averageFinalLoss /= input.size();
        float bestLoss = averageFinalLoss;
        FFNLinear survivor = *this;
        for (int i = 0; i < population; i++) {
            FFNLinear mutation = createMutation(lr, mutationChance);
            float averageLoss = 0.0;
            for (int i = 0; i < input.size(); i++) {
                averageLoss += MSELoss(groundTruths[i], mutation.Forward({ input[i] })[0]);
            }
            averageLoss /= input.size();
            if (averageLoss < bestLoss) {
                survivor.Layers = mutation.Layers;
                bestLoss = averageLoss;
            }
        }
        if (bestLoss < averageFinalLoss) {
            Layers = survivor.Layers;
            cout << bestLoss << endl;
        }
    }
    FFNLinear createMutation(float lr, float chance) {
        uniform_real_distribution<float> learning(-lr, lr);
        FFNLinear mutation = *this;
        for (int i = 0; i < Layers.size(); i++) {
            for (int j = 0; j < Layers[i].Neurons.size(); j++) {
                if (percent(rng) < chance) {
                    mutation.Layers[i].Neurons[j].weights = addFloatToTensor(Layers[i].Neurons[j].weights, learning(rng));
                    mutation.Layers[i].Neurons[j].bias = Layers[i].Neurons[j].bias + learning(rng);
                }
            }
        }
        return mutation;
    }
};
class trainer {
public:
    FFNLinear student;
    map<float, float> dataset;
    int steps;
    trainer(FFNLinear student){
        this->student = student;
    }
    void train() {
        vector<float> inputs;
        vector<float> outputs;

        for (auto& data : dataset) {
            inputs.push_back(data.first);
            outputs.push_back(data.second);
        }
        for (int i = 0; i < steps; i++) {
            student.Backward(inputs,outputs);
        }
    }
};
int main()
{
    trainer test(FFNLinear({ Layer(1,8),Layer(8,8),Layer(8,8),Layer(8,1) }));
    for (int i = 0; i <= 100; i++) {
        test.dataset[i] = i * i;
    }
    test.steps = 90000.0;
    test.train();
    cout << test.student.Forward({ 2 })[0] << " Excpected: 4" << endl;
    cout << test.student.Forward({ 4 })[0] << " Excpected: 16" << endl;
};
