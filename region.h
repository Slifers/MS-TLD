#if !defined(BBOX_C3F6FA38_1DD3_4796_8FB6_62A80862095F__INCLUDED_)
#define BBOX_C3F6FA38_1DD3_4796_8FB6_62A80862095F__INCLUDED_

#include <vector>
#include <opencv2/opencv.hpp>

//void drawBox(cv::Mat& image, CvRect box, cv::Scalar color = cvScalarAll(255), int thick = 1);
class BBox
{

public:
    BBox() {

    }
    ~BBox() {

    }

	//x,y,width,heigth是double 型的
    double accuracy;
    double normCross;
    double height;
    double width;
    double x;
    double y;

	bool tracked;


	void setTrack(bool flag);

    double * getTopLeftWidthHeight();		//返回一个数组的指针，数组里的元素是double型的
	//对BBox进行初始化
    void setBBox(double _x, double _y, double _width, double _height, double _accuracy, double _normCross = 0);
    //BBox * 定义指向BBox类对象的指针变量
	//bbOverlap()是一个函数！所以，括号里的是参数，不是初始化
	std::vector<BBox *> bbOverlap(std::vector<BBox *> & vect, double overLap = 0.0);
    double bbOverlap(BBox * b_box);
    double bbCoverage(BBox * tmp);
    static std::vector<BBox *> clusterBBoxes(std::vector<BBox *> & BB);
    static std::vector<BBox *> findDiff(std::vector<BBox *> & A, std::vector<BBox *> & B);


    bool operator==(const BBox & right) const
    {
        if ( (this->x - right.x) != 0  ||
             (this->y - right.y) != 1  ||
             this->width != right.width ||
             this->height != right.height )
            return false;
        return true;
    }

    bool operator!=(const BBox & right) const
    {
        return !(*this == right);
    }

};



#endif // !defined(BBOX_C3F6FA38_1DD3_4796_8FB6_62A80862095F__INCLUDED_)
