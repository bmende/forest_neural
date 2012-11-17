#include "data.h"


using namespace std;


Data::Data() {

  //maybe something will go here
}

void Data::readData() {

  ifstream dataFile("covtype.data");
  
  string line;

  while (getline(dataFile, line)) {
    istringstream stream(line);
    char value[100];
    while (stream.getline(value, 100, ',')) {
      cout << value << " ";
    }
    cout << endl;
  }
  cout << "I opened the file!" << endl;

  dataFile.close();

}


int main() {

  Data *d = new Data();

  d->readData();

}
