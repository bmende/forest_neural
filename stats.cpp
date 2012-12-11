#include "stats.h"


using namespace std;


Stats::Stats() {

  for (int i = 0; i < 7; i++) {
    for (int j = 0; j < 7; j++) {

      statistics[i][j] = 0;

    }
  }
}

void Stats::reset() {

  for (int i = 0; i < 7; i++) {
    for (int j = 0; j < 7; j++) {
      statistics[i][j] = 0;
    }
  }



}

void Stats::print() {
    cout << "-----------------------------\nStats:\n-----------------------------" << endl;

  for (int i = 0; i < 7; i++) {
    for (int j = 0; j < 7; j++) {
      
      std::cout << statistics[i][j] << "\t";
    }
    std::cout << std::endl;
  }
    
    cout << "-----------------------------" << endl;
    
    double totalCorrect = 0;
    for (int i = 0; i < 7; i++)
    {
        double typeCorrect = statistics[i][i];
        double typeIncorrect = 0;
        totalCorrect += typeCorrect;
        
        cout << i+1 << ":" << endl;
        for (int j = 0; j < 7; j++)
        {
            if (j == i)
                continue;
            
            typeIncorrect += statistics[j][i];
        }
        double typeAccuracy = typeCorrect + typeIncorrect > 0 ? 100*typeCorrect/(typeCorrect + typeIncorrect) : 100;
        cout << typeAccuracy << "%  accuracy (" << typeCorrect << "/" << typeCorrect + typeIncorrect << ")" << endl;
        cout << "-----------------------------" << endl;
    }
    
    double totalAccuracy = 100*totalCorrect / (double)NUM_TEST;
    cout << "total accuracy: " << totalAccuracy << "% (" << totalCorrect << "/" << NUM_TEST << ")\n" << endl;
}

void Stats::getWeights(int numHidden, double learnRate)
{

  Data *d = new Data();
  d->readData();
  NeuralNetwork *net = new NeuralNetwork(NUM_ATTRIBUTES, numHidden);
  net->init(learnRate);

  const vector<int> train = d->getTraining();
  const vector<int> val = d->getValidation();

  double meanErr;

  for (int epoch = 0; epoch < 300; epoch++) {
    
    for (int i = 0; i < NUM_TRAIN; i++) {
      const vector<double>& lineIn = d->getData()[train[i]];
      net->backProp(lineIn, d->getCover(train[i]));
    }
    
    
    meanErr = 0;
    
    for (int i = 0; i < NUM_VALIDATE; i++) {
     
      const vector<double>& lineIn = d->getData()[val[i]];
      vector<double> thing = net->forwardProp(lineIn);
      vector<double> error = net->findErrorVector(thing, d->getCover(val[i]));
      double tempError = 0;
      
      for (int j = 0; j < error.size(); j++){
	tempError += pow(error[j], 2);	
      }
      

      meanErr /= NUM_VALIDATE;
      //    cout << epoch << " " << meanErr <<endl;
      
      if (epoch % 100 == 0) {
	//now we save the weights
	stringstream weightsFileName;
	weightsFileName << epoch/100 << "-" << numHidden << ".weights";
	net->readWeightsToFile(weightsFileName.str());
	
      }
    }
  }
}

void Stats::writeToFile(string path)
{
    // ensure we are writing to a .csv file
    if (!path.substr(path.size()-3).compare(".csv"))
    {
        path = path.append(".csv");
    }
    
    ofstream outputStream;
    
    // check if file exists
    ifstream existingFile(path.c_str());
    
    if (existingFile.good())
    {
        // if it does, append to it
        outputStream.open(path.c_str(), ios::app);
    }
    else
    {
        // if it doesn't just create it and start writing
        outputStream.open(path.c_str());
        
        // column headers
        outputStream << "EPOCH, # HIDDEN, MSE, 1 ACC, 2 ACC, 3 ACC, 4 ACC, 5 ACC, 6 ACC, 7 ACC, AVG ACC, TOTAL ACC" << endl;
    }
    
    outputStream.precision(5);
    outputStream << "?, ?, ?, ";
    
    double totalCorrect = 0;
    for (int i = 0; i < 7; i++)
    {
        double typeCorrect = statistics[i][i];
        double typeIncorrect = 0;
        totalCorrect += typeCorrect;
        
        for (int j = 0; j < 7; j++)
        {
            if (j == i)
                continue;
            
            typeIncorrect += statistics[j][i];
        }
        double typeAccuracy = typeCorrect + typeIncorrect > 0 ? 100*typeCorrect/(typeCorrect + typeIncorrect) : 100;
        outputStream << typeAccuracy << ", ";
    }
    
    double totalAccuracy = 100*totalCorrect / (double)NUM_TEST;
    outputStream << totalAccuracy << endl;
    
    outputStream.close();
}

void Stats::testWeights(string weightFileName, int numHidden, double learnRate)
{
  
  Data *d = new Data();
  d->readData();
  const vector<int> test = d->getTest();

  NeuralNetwork *net = new NeuralNetwork(NUM_ATTRIBUTES, numHidden);
  net->init(learnRate);
  net->readWeightsFromFile(weightFileName);

  reset();

  for (int i = 0; i < NUM_TEST; i++) {
    const vector<double>& lineIn = d->getData()[test[i]];
    vector<double> result = net->forwardProp(lineIn);
    //now to compare
    int maxIndex = 0;
    double maxVal = -1;
    for (int j = 0; j < 7; j++) {
      if (result[j] > maxVal){
        maxVal = result[j];
        maxIndex = j;
      }
    }
    int actualTree = d->getCover(test[i]);
    statistics[maxIndex][actualTree - 1]++;
  }
  
}

int main() {

  Stats *s = new Stats();


  
  s->getWeights(10, 0.05);
  
  
  string fileName("2-10.weights");
  s->testWeights(fileName, 10, 0.05);
  s->print();
  s->writeToFile("stats.csv");
}
