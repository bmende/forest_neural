#include "ann.h"


using namespace std;

NeuralNetwork::NeuralNetwork(int numInputs) {

  output.inputs = numInputs; output.numNodes = 7;
  output.weights = new double*[output.numNodes];
  for (int i = 0; i < output.numNodes; i++)
    output.weights[i] = new double[output.inputs];
  output.outputs = new double[output.numNodes];
  output.summed = new double[output.numNodes];
  output.primed = new double[output.numNodes];

  
  inputs = new double[numInputs];

}

void NeuralNetwork::init(double alpha) {

  learnRate = alpha;


  std::srand( std::time( NULL ) );

  for (int i = 0; i < output.numNodes; i++) {
    for (int j = 0; j < output.inputs; j++) {
      output.weights[i][j] = (double)std::rand() / (double)RAND_MAX;
      output.weights[i][j] /= (double)output.numNodes;
    }
    output.outputs[i] = 0;
    output.summed[i] = 0;
    output.primed[i] = 0;
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

vector<double> NeuralNetwork::forwardProp(const vector<double>& lineIn) {
  
  if (lineIn.size() != 54) {
    cerr << "line in not correct size\n";
    exit(1);
  } 
  //putting data in input nodes
  for (int i = 0; i < lineIn.size(); i++) {
    inputs[i] = lineIn[i];
  }

  //now for output node calculations
  for (int node = 0; node < output.numNodes; node++) {
    double sum = 0;
    for (int in = 0; in < output.inputs; in++) {
      sum += (inputs[in] * output.weights[node][in]);
    }
    output.outputs[node] = sigma(sum);
    output.summed[node] = sum;
    output.primed[node] = sigmaPrime(sum);
  }
  vector<double> answer(output.numNodes, 0);
  for (int i = 0; i < answer.size(); i++)
    answer[i] = output.outputs[i];
  return answer;
}

vector<double> NeuralNetwork::findErrorVector(vector<double> trainee, int trainer) {

  vector<double> error(trainee.size(), 0);
  for (int i = 0; i < error.size(); i++) {
    if ((trainer-1) == i)
      error[i] = 1.0 - trainee[i];
    else
      error[i] = 0.0 - trainee[i];
  }
  return error;
}


void NeuralNetwork::backProp(const vector<double>& lineIn, int trainer) {

  vector<double> out = forwardProp(lineIn);
  vector<double> err = findErrorVector(out, trainer);
  
  //so now we adjust the error de = err * sigma'
  for (int i = 0; i < err.size(); i++) {
    err[i] *= output.primed[i];
  }

  //now to adjust input -> output weights
  //W_ji <- W_ji + learn*out_j*err_i
  for (int outNode = 0; outNode < output.numNodes; outNode++) {
    for (int inNode = 0; inNode < output.inputs; inNode++) {
      double old_weight = output.weights[outNode][inNode];
      double new_weight = learnRate * lineIn[inNode] * err[outNode];
      new_weight += old_weight;
      output.weights[outNode][inNode] = new_weight;
    }
  }

}


void NeuralNetwork::readWeightsToFile(string fileName)
{

  ofstream saving;
  saving.open(fileName.c_str());

  for (int i = 0; i < output.numNodes; i++) {
    for (int j = 0; j < output.inputs; j++) {
      saving << output.weights[i][j] << " ";
    }
    saving << endl;
  }

  saving.close();

}

void NeuralNetwork::readWeightsFromFile(string fileName)
{


  ifstream saved;
  saved.open(fileName.c_str());

  string line;

  for (int i = 0; i < output.numNodes; i++) {
    std::getline(saved, line);
    istringstream stream(line);
    char value[100];
    for (int j = 0; j < output.inputs; j++) {
      stream.getline(value, 100, ' ');
      output.weights[i][j] = std::atof(value);
    }
  }

}
