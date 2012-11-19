#include "ann.h"


using namespace std;

NeuralNetwork::NeuralNetwork(int numInputs, int numHidden) {


  hidden.inputs = numInputs; hidden.numNodes = numHidden;
  hidden.weights = new double*[hidden.numNodes];
  for (int i = 0; i < hidden.numNodes; i++) 
    hidden.weights[i] = new double[hidden.inputs];
  hidden.outputs = new double[hidden.numNodes];

  output.inputs = hidden.numNodes; output.numNodes = 7;
  output.weights = new double*[output.numNodes];
  for (int i = 0; i < output.numNodes; i++)
    output.weights[i] = new double[output.inputs];
  output.outputs = new double[output.numNodes];

  inputs = new double[numInputs];

}

void NeuralNetwork::init() {

  //initializing weights to 0.5 each. TODO: make random initial weights
  
  for (int i = 0; i < hidden.numNodes; i++) {
    for (int j = 0; j < hidden.inputs; j++) {
      hidden.weights[i][j] = (double)std::rand() / (double)RAND_MAX;
      hidden.weights[i][j] /= (double)hidden.numNodes;
    }
    hidden.outputs[i] = 0;
  }

  for (int i = 0; i < output.numNodes; i++) {
    for (int j = 0; j < output.inputs; j++) {
      output.weights[i][j] = (double)std::rand() / (double)RAND_MAX;
      output.weights[i][j] /= (double)output.numNodes;
    }
    output.outputs[i] = 0;
  }
}

double NeuralNetwork::sigma(double weightedSum) {
  //this function is sigma(x) = 1/(1+e^(-x))

  weightedSum *= -1;
  double answer = 1 + exp(weightedSum);
  answer = 1/answer;

  return answer;
}

double NeuralNetwork::sigmaPrime(double weightedSum) {
  double sig = sigma(weightedSum);

  return sig*(1-sig);
}

vector<double> NeuralNetwork::forwardProp(vector<double> lineIn) {
  
  if (lineIn.size() != 54) {
    cerr << "line in not correct size\n";
    exit(1);
  } 
  //putting data in input nodes
  for (int i = 0; i < lineIn.size(); i++) {
    inputs[i] = lineIn[i];
  }

  //calculating hidden node outputs.
  for (int node = 0; node < hidden.numNodes; node++) {
    double sum = 0;
    for (int in = 0; in < hidden.inputs; in++) {
      sum += (inputs[in] * hidden.weights[node][in]);
    }
    hidden.outputs[node] = sum;
  }
  
  //now for output node calculations
  for (int node = 0; node < output.numNodes; node++) {
    double sum = 0;
    for (int in = 0; in < output.inputs; in++) {
      sum += (sigma(hidden.outputs[in]) * output.weights[node][in]);
    }
    output.outputs[node] = sum;
  }
  vector<double> answer(output.numNodes, 0);
  for (int i = 0; i < answer.size(); i++)
    answer[i] = sigma(output.outputs[i]);
  return answer;
}

vector<double> NeuralNetwork::findErrorVector(vector<double> output, int trainer) {

  vector<double> error(output.size(), 0);
  for (int i = 0; i < error.size(); i++) {
    if (trainer == i)
      error[i] = 1.0 - output[i];
    else
      error[i] = 0.0 - output[i];
  }
  return error;
}

int main() {

  Data *d = new Data();

  cout << "reading data\n";
  d->readData();
  cout << "data read\nnow making net\n";
  NeuralNetwork *net = new NeuralNetwork(NUM_ATTRIBUTES, 120);
  net->init();
  cout << "net initialized with random weights\n";
  
  for (int i = 0; i < 40; i++) {
    vector<double> out = net->forwardProp(d->getData()[i]);
    vector<double> err = net->findErrorVector(out, d->getCover(i));
  }
}
