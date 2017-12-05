#include <opencv2/opencv.hpp>
#include "tld_utils.h"
#include <iostream>
#include <sstream>
#include "TLD.h"
#include <stdio.h>
using namespace cv;
using namespace std;
//Global variables
Rect box;
bool drawing_box = false;
bool gotBB = false;
bool tl = true;
bool rep = false;
bool fromfile=false;
string video;

typedef struct {
	float x1;   //bottom left
	float y1;
	float x2;   //top left
	float y2;
	float x3;   //top right
	float y3;
	float x4;   //bottom right
	float y4;
} VOTPolygon;

void readBB(char* file) {

	VOTPolygon p;
	ifstream bb_file(file);
	string line;
	std::getline(bb_file, line);
	std::vector<float> numbers;
	std::istringstream s(line);
	float x;
	char ch;
	while (s >> x) {
		numbers.push_back(x);
		s >> ch;
	}
	if (numbers.size() == 4) {
		float x = numbers[0], y = numbers[1], w = numbers[2], h = numbers[3];
		p.x1 = x;
		p.y1 = y + h;
		p.x2 = x;
		p.y2 = y;
		p.x3 = x + w;
		p.y3 = y;
		p.x4 = x + w;
		p.y4 = y + h;
	}
	else if (numbers.size() == 8) {
		p.x1 = numbers[0];
		p.y1 = numbers[1];
		p.x2 = numbers[2];
		p.y2 = numbers[3];
		p.x3 = numbers[4];
		p.y3 = numbers[5];
		p.x4 = numbers[6];
		p.y4 = numbers[7];
	}
	else {
		std::cerr << "Error loading initial region in file - unknow format " << file << "!" << std::endl;
		p.x1 = 0;
		p.y1 = 0;
		p.x2 = 0;
		p.y2 = 0;
		p.x3 = 0;
		p.y3 = 0;
		p.x4 = 0;
		p.y4 = 0;
	}
	VOTPolygon initPolygon = p;
	float x1 = std::min(initPolygon.x1, std::min(initPolygon.x2, std::min(initPolygon.x3, initPolygon.x4)));
	float x2 = std::max(initPolygon.x1, std::max(initPolygon.x2, std::max(initPolygon.x3, initPolygon.x4)));
	float y1 = std::min(initPolygon.y1, std::min(initPolygon.y2, std::min(initPolygon.y3, initPolygon.y4)));
	float y2 = std::max(initPolygon.y1, std::max(initPolygon.y2, std::max(initPolygon.y3, initPolygon.y4)));
	box = cv::Rect(x1, y1, x2 - x1, y2 - y1);

}
//bounding box mouse callback
void mouseHandler(int event, int x, int y, int flags, void *param){
  switch( event ){
  case CV_EVENT_MOUSEMOVE:
    if (drawing_box){
        box.width = x-box.x;
        box.height = y-box.y;
    }
    break;
  case CV_EVENT_LBUTTONDOWN:
    drawing_box = true;
    box = Rect( x, y, 0, 0 );
    break;
  case CV_EVENT_LBUTTONUP:
    drawing_box = false;
    if( box.width < 0 ){
        box.x += box.width;
        box.width *= -1;
    }
    if( box.height < 0 ){
        box.y += box.height;
        box.height *= -1;
    }
    gotBB = true;
    break;
  }
}

void print_help(char** argv){
  printf("use:\n     %s -p /path/parameters.yml\n",argv[0]);
  printf("-s    source video\n-b        bounding box file\n-tl  track and learn\n-r     repeat\n");
}

void read_options(int argc, char** argv,VideoCapture& capture,FileStorage &fs){
  for (int i=0;i<argc;i++){
      if (strcmp(argv[i],"-b")==0){
          if (argc>i){
              readBB(argv[i+1]);
              gotBB = true;
          }
          else
            print_help(argv);
      }
      if (strcmp(argv[i],"-s")==0){
          if (argc>i){
              video = string(argv[i+1]);
              capture.open(video);
              fromfile = true;
          }
          else
            print_help(argv);

      }
	  if (strcmp(argv[i], "-a") == 0)
	  {
		  if (argc>i) {
			  video = "input/img/%04d.jpg";
			  capture.open(video);
			  fromfile = true;
		  }
		  else
			  print_help(argv);
	  }
	  if (strcmp(argv[i], "-v") == 0)
	  {
		  if (argc>i) {
			  video = "input/%08d.jpg";
			  capture.open(video);
			  fromfile = true;
		  }
		  else
			  print_help(argv);
	  }	  
      if (strcmp(argv[i],"-p")==0){
          if (argc>i){
              fs.open(argv[i+1], FileStorage::READ);
          }
          else
            print_help(argv);
      }
      if (strcmp(argv[i],"-no_tl")==0){
          tl = false;
      }
      if (strcmp(argv[i],"-r")==0){
          rep = true;
      }
  }
}

int main(int argc, char * argv[]){
  VideoCapture capture;
  capture.open(0);
  FileStorage fs;
  //Read options
  read_options(argc,argv,capture,fs);
  //Init camera
  if (!capture.isOpened())
  {
	cout << "capture device failed to open!" << endl;
    return 1;
  }
  //Register mouse callback to draw the bounding box
  namedWindow("TLD",CV_WINDOW_AUTOSIZE);
  setMouseCallback( "TLD", mouseHandler, NULL );
  //TLD framework
  TLD tld;
  //Read parameters file
  tld.read(fs.getFirstTopLevelNode());
  Mat frame;
  Mat last_gray;
  Mat first;
  if (fromfile){
      capture >> frame;
      cvtColor(frame, last_gray, COLOR_RGB2GRAY);
      frame.copyTo(first);
  }else{
      capture.set(CV_CAP_PROP_FRAME_WIDTH,340);
      capture.set(CV_CAP_PROP_FRAME_HEIGHT,240);
  }

  ///Initialization
GETBOUNDINGBOX:
  while(!gotBB)
  {
    if (!fromfile){
      capture >> frame;
    }
    else
      first.copyTo(frame);
    cvtColor(frame, last_gray, COLOR_RGB2GRAY);
    drawBox(frame,box,CV_RGB(255,0,0));
    imshow("TLD", frame);
    if (waitKey(33) == 'q')
	    return 0;
  }
  if (min(box.width,box.height)<(int)fs.getFirstTopLevelNode()["min_win"]){
      cout << "Bounding box too small, try again." << endl;
      gotBB = false;
      goto GETBOUNDINGBOX;
  }
  //Remove callback
  setMouseCallback( "TLD", NULL, NULL );
  printf("Initial Bounding Box = x:%d y:%d h:%d w:%d\n",box.x,box.y,box.width,box.height);
  //Output file
  FILE  *bb_file = fopen("mstld_boxes.txt","w"); 
  //TLD initialization
  tld.init(last_gray,frame,box,bb_file);

  ///Run-time
  Mat current_gray;
  BoundingBox pbox;
  bool status=true;
  int frames = 1;
  int detections = 1;
  double avg_time = 0.;
REPEAT:
  while(capture.read(frame)){
    //get frame
	double time_profile_counter = cv::getCPUTickCount();
    cvtColor(frame, current_gray, COLOR_RGB2GRAY);
    //Process Frame
    tld.processFrame(last_gray,current_gray,frame,pbox,status,tl,bb_file);  

	time_profile_counter = cv::getCPUTickCount() - time_profile_counter;
	//std::cout << "  -> speed : " <<  time_profile_counter/((double)cvGetTickFrequency()*1000) << "ms. per frame" << std::endl;
	avg_time += time_profile_counter / ((double)cvGetTickFrequency() * 1000);

    if (status){
      //drawPoints(frame,pts1);
      //drawPoints(frame,pts2,Scalar(0,255,0));
      drawBox(frame,pbox,CV_RGB(255,0,0),2);
      detections++;
    }
    //Display
    imshow("TLD", frame);
    //swap points and images
    swap(last_gray,current_gray);  
    frames++;
	pbox = (Rect(0,0,0,0));    
    if (cvWaitKey(33) == 'q')
      break;
  }
  if (rep){
    rep = false;
    tl = false;
    fclose(bb_file);
    bb_file = fopen("final_detector.txt","w");
    //capture.set(CV_CAP_PROP_POS_AVI_RATIO,0);
    capture.release();
    capture.open(video);
    goto REPEAT;
  }

  std::cout << "Average processing speed " << avg_time / frames << "ms. (" << 1. / (avg_time / frames) * 1000 << " fps)" << std::endl;
  waitKey(0);
  
  fclose(bb_file);
  return 0;
}
