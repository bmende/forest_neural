#include "data.h"


using namespace std;


Data::Data() :
  dataArray(vector<vector<double> >(NUM_EXAMPLES, vector<double>(NUM_ATTRIBUTES))),
  coverType(vector<int>(NUM_EXAMPLES))
{
}

void Data::readData() {


  ifstream dataFile;
  dataFile.open("covtype.data");


  string line;
  int outer = 0, inner = 0;
  double max[NUM_ATTRIBUTES], min[NUM_ATTRIBUTES];
  for (int i = 0; i < NUM_ATTRIBUTES; i++) {
    max[i] = 0;
    min[i] = 100000;
  }

  while (std::getline(dataFile, line)) {
    
    istringstream stream(line);
    char value[100];
    inner = 0;
   
    while (stream.getline(value, 100, ',')) {
      double number = std::atof(value);
      if (inner < NUM_ATTRIBUTES) {
       	dataArray[outer][inner] = number;
      	if (number > max[inner])
      	  max[inner] = number;
      	else if (number < min[inner])
      	  min[inner] = number;
 
      }
      else
      	coverType[outer] = number;

      inner++;
    }
    outer++;
  }

  for (int i = 0; i < NUM_EXAMPLES; i++) {
    for (int j = 0; j < NUM_ATTRIBUTES; j++) {
      dataArray[i][j] = (dataArray[i][j] - min[j])/(max[j] - min[j]);
      if (dataArray[i][j] > 1 || dataArray[i][j] < 0) {
	
      }
    }
  }


  dataFile.close(); //we no longer need the file to be open.

}


// int main() {

//   Data *d = new Data();

//   d->readData();

// }
