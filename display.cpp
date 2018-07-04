#include "opencv2/objdetect.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/core.hpp"
#include "opencv2/face.hpp"
#include <iostream>
#include <stdio.h>
#include <string>
#include <sstream>
#include <fstream>

//including namespaces
using namespace std;
using namespace cv;
using namespace cv::face;

fstream pca,info;

static Mat norm_0_255(InputArray _src) {
    Mat src = _src.getMat();
    // Create and return normalized image:
    Mat dst;
    switch(src.channels()) {
    case 1:
        cv::normalize(_src, dst, 0, 255, NORM_MINMAX, CV_8UC1);
        break;
    case 3:
        cv::normalize(_src, dst, 0, 255, NORM_MINMAX, CV_8UC3);
        break;
    default:
        src.copyTo(dst);
        break;
    }
    return dst;
}

// FaceRecognition Object
Ptr<BasicFaceRecognizer> model = createEigenFaceRecognizer(); // the facerecognizer pointer to class...


CascadeClassifier faceDetect;
CascadeClassifier eyeDetect;
const string CCName("haarcascade_frontalface_alt.xml");
const string CCName1("haarcascade_eye_tree_eyeglasses.xml");


//prototype
void frame_inverse(Mat& frame,Mat& frame_inv);
void GET_NAME(char*,vector<string>&,vector<int>&,int&);
void face_detect(Mat&,Mat&,std::vector<Rect>&);

//two point whose needed
Point p1=Point(200,40);
Point p2=Point(490,480-40);

//global names
vector<string> names;
vector<int>	   lbls;




int main()
{
					//load classifier

			if(!faceDetect.load(CCName)){cout<<endl<<"The Classifier doesnt load "<<endl;return -1;}
			if(!eyeDetect.load(CCName1)){cout<<endl<<"The Classifier doesnt load "<<endl;return -1;}

int select=1;

//Menu While
 
while(select)
{


						int Menu;

		cout<<"\n########## choose : 0 -> for capture into directory 								##########"<<endl;
		cout<<"\n########## choose : 1 -> for recognize face       								##########"<<endl;

cin>>Menu;
cout<<endl;

	
switch(Menu)
{
	//case 1
		case 1:
				{
				pca.open("pca.txt",ios::in);
				info.open("info.txt",ios::in);
				if(!pca.is_open())
							{cout<<"no pca.txt \n";return -1;}
				if(!info.is_open())
							{cout<<"no info.txt \n";return -1;}
				VideoCapture cap(0);
				Mat frame;
				Mat frame_inv;
				string name_window="captured_frame";
				
				// recognition initial
				vector<string> names1;
				vector<int> lables1;
				int number;
				GET_NAME((char*)"pca.txt",names1,lables1,number);
				// 
				vector<Mat> images;
				vector<int> lbls;
				for(int i=0;i<number;i++)
				{images.push_back(imread(names1[i],CV_LOAD_IMAGE_GRAYSCALE));lbls.push_back(lables1[i]);}
				
				model->train(images,lbls);

				vector<string> nm;
				vector<int> lb;
				int num;
				GET_NAME((char*)"info.txt",nm,lb,num);
				
							vector<int> lookup;
							for(int i=0;i<20;i++)
								{
										int count=0;
											for(int j=0;j<lb.size();j++)			
										{

										if(lb[j]==i){count++;}

										}
										lookup.push_back(count);
								}

				for(int i=0;i<lookup.size();i++)
				{
						stringstream ps;
						ps<<"label "<<i;
						model->setLabelInfo(i,ps.str());

				}

				while(1)
		{
					//capture_frame
					cap>>frame;
					if(!cap.isOpened()){cout<<"capture error\n";return -1;}
					//inverse frame

					Mat inv;
					frame_inverse(frame,inv);
					
					Mat out=frame.clone();
					std::vector<Rect> faces1;
					face_detect(inv,out,faces1);
		if(faces1.size()>0)
			{
					for(int z=0;z<faces1.size();z++)
				{
					Mat face_Mat_1=inv(Rect(faces1[z].x,faces1[z].y,faces1[z].width,faces1[z].height));
					std::vector<Rect> faces;
					face_detect(face_Mat_1,out,faces);
					double thresh;
						if(faces.size()>0)
							{
								for(int i=0;i<faces.size();i++){
									Mat face_Mat=face_Mat_1(Rect(faces[i].x,faces[i].y,faces[i].width,faces[i].height));

				
													Mat dst;			
													Mat dst_gray;
													cvtColor(face_Mat,dst_gray,COLOR_BGR2GRAY);
													equalizeHist(dst_gray,dst_gray);			
													dst=norm_0_255(dst_gray);	
								//recognition
					stringstream ds;
					ds<<"face"<<z;
								
								int num_label_detected;
								
								double conf=0.0;
								double sum=0;
								for(int d=0;d<face_Mat.rows;d++)
								for(int r=0;r<face_Mat.cols;r++)
								for(int c=0;c<face_Mat.channels();c++)
								{sum=sum+face_Mat.at<Vec3b>(d,r)[c];}
								thresh=sum/750;
								model->setThreshold(thresh);
				Mat dst1;
				resize(dst,dst1,Size(270,270),0,0,2);
imshow(ds.str(),dst1);
								model->predict(dst1,num_label_detected,conf);
				stringstream ls;
				for(int j=0;j<lb.size();j++)
				{
				if(lb[j]==num_label_detected)
				ls<<nm[j]<<" ";
				}
								stringstream text1,text2,text3;





								if(num_label_detected!=-1)
									{text1<<"for "<<ds.str()<<" the recognized person is : "<<model->getLabelInfo(num_label_detected)<<ls.str();
									text2<<"THRESHOLD = "<<model->getThreshold();
									text3<<"Conf = "<<conf;
									putText(inv,text1.str(),Point(2,20),FONT_HERSHEY_PLAIN ,1.0,CV_RGB(255,0,0),0.2,1,false);
									putText(inv,text2.str(),Point(2,40),FONT_HERSHEY_PLAIN ,1.0,CV_RGB(255,0,0),0.2,1,false);
									putText(inv,text3.str(),Point(2,60),FONT_HERSHEY_PLAIN ,1.0,CV_RGB(255,0,0),0.2,1,false);
									}
								else
									{
									text1<<"No one recognized\n thresh = "<<thresh;
									putText(inv,text1.str(),Point(2,20),FONT_HERSHEY_PLAIN ,1.0,CV_RGB(255,0,0),0.2,1,false);
									}
              }

		}
				else
				{
				stringstream text1;
				imshow(name_window,inv);
				text1<<"No face detected \n thresh = "<<thresh;
								putText(inv,text1.str(),Point(2,20),FONT_HERSHEY_COMPLEX_SMALL,0.2,CV_RGB(255,0,0),0.2,1,false);
				}
	
								

}// end of for

imshow(name_window,inv);
} // end  if faces1
else{imshow(name_window,inv);}
char wk=waitKey(1);
				if(wk==27){destroyWindow(name_window);break;}	
				} // end while
				break;
				}//Cace 1 end



// Case 0
	case 0:
			{
			
			VideoCapture cap(0);
			
			string name_window="captured_frame";
	cout<<"########## press c -> capture , press x -> exit 					"<<endl;
			while(1)
			{
Mat frame;
				//capture_frame
				cap>>frame;
				if(!cap.isOpened()){cout<<"capture error\n";return 					-1;}
	
Mat inv;
frame_inverse(frame,inv);
imshow(name_window,inv);

Mat out=frame;
std::vector<Rect> faces1;
face_detect(inv,out,faces1);

if(faces1.size()>0)
{
Mat face_Mat_1=inv(Rect(faces1[0].x,faces1[0].y,faces1[0].width,faces1[0].height));
	std::vector<Rect> faces;
Mat out1=frame;
	face_detect(face_Mat_1,out1,faces);
if(faces.size()>0)
{
	Mat face_Mat=face_Mat_1(Rect(faces[0].x,faces[0].y,faces[0].width,faces[0].height));
	
						
						Mat dst;			
						Mat dst_gray;
						cvtColor(face_Mat,dst_gray,COLOR_BGR2GRAY);
						equalizeHist(dst_gray,dst_gray);			
						dst=norm_0_255(dst_gray);			
						Mat dst1;
						resize(dst,dst1,Size(270,270),0,0,2);
						imshow("face",dst1);

					char wk=waitKey(1);
					if(wk=='c')
					{


					stringstream os;
						cout<<"\n########## Enter the person name     ########"<<endl;
						string name;						
						cin>>name;
						os<<name<<".jpg";
						names.push_back(name);
						cout<<"\n########## Enter the person label     #######"<<endl;
						int label;
						cin>>label;
						pca.open("pca.txt",ios::out | ios::app);
						pca<<"/home/behzad/Desktop/opencv_project/tow_lect_another_detect/"<<name<<".jpg;"<<label<<endl;
						pca.close();

						imwrite(os.str(),dst1);
						
						info.open("info.txt",ios::out | ios::app);
						info<<name<<';'<<label<<endl;
						info.close();
					}
					else if(wk=='x')
					{
					destroyWindow(name_window);
					break;
					}
		}
			}
else
{
cout<<"no face "<<endl;
}

}

			break;
			}// end of Case 0



} // end switch case
cout<<"\n########## Are u want to continue ?                  ##########"<<endl;
cin>>select;

} // end while select
destroyAllWindows();
return 0;}


//frame_inverse
void frame_inverse(Mat& frame,Mat& frame_inv)
{
frame.copyTo(frame_inv);
	for(int i=0;i<frame.rows;i++)
		for(int j=0;j<frame.cols;j++)
			for(int c=0;c<frame.channels();c++)
				
				{
	frame_inv.at<Vec3b>(i,j)[c]=frame.at<Vec3b>(i,frame.cols-j)[c];
				}

}

void GET_NAME(char* file_name,vector<string>& name,vector<int>& labels,int& counter){
	ifstream pca_txt(file_name);
	string line;
	counter=0;
	if(pca_txt.is_open())
	{
		while(getline(pca_txt,line))
				{		
					string name1;int label1;		
					stringstream lines;
					lines<<line;
					getline(lines,name1,';');
					name.push_back(name1);
					lines>>label1;
					labels.push_back(label1);
					counter++;
				}
	}
}

//face_detect
void face_detect(Mat& img,Mat& out,std::vector<Rect>& faces){

Mat frame_gray;
cvtColor( img, frame_gray, COLOR_BGR2GRAY );
equalizeHist( frame_gray, frame_gray );
faceDetect.detectMultiScale(frame_gray,faces,1.1,2,0 |CASCADE_SCALE_IMAGE,Size(100,100));
img.copyTo(out);
}




//ok
