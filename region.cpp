#include "region.h"
#include <algorithm>

using namespace cv;
using namespace std;
/*
void drawBox(Mat& image, CvRect box, Scalar color, int thick) {
	rectangle(image, cvPoint(box.x, box.y), cvPoint(box.x + box.width, box.y + box.height), color, thick);
}
*/

void BBox::setTrack(bool flag) {

	tracked = flag;
}

void BBox::setBBox(double _x, double _y, double _width, double _height, double _accuracy, double _normCross){
	accuracy = _accuracy;
	height = _height;
	width = _width;
	x = _x;
	y = _y;
    normCross = _normCross;
}


//将x,y,width,heigth的值赋予ret，并返回ret
double * BBox::getTopLeftWidthHeight()
{
    double * ret = new double[4];
    ret[0] = x;
    ret[1] = y;
    ret[2] = width;
    ret[3] = height;

    return ret;
}

//这TM竟然是个函数，我都没反应过来。道理类似于   int xxx() {return int;}
std::vector<BBox *> BBox::bbOverlap(std::vector<BBox *> & vect, double overLap)
{
    std::vector<BBox *> ret;		//存储overlap<0.7的BBox
    std::vector<BBox *> retP;		//存储overlap>0.7的BBox
    double x1, y1;
    BBox * tmp;		//tmp是一个指向BBox类的指针
    double intersection;
    double over = overLap;

    if (over == 0)		//头文件声明时，overlap=0.0，意思就是默认是0.7呗
        over = 0.7;

		//使用迭代器，遍历整个vector<BBox *>
    for(std::vector<BBox *>::iterator it = vect.begin(); it != vect.end(); ++it){
        tmp = *it;
        x1 =  std::min(x + width, tmp->x + tmp->width) - std::max(x, tmp->x) + 1;
        if (x1 <= 0)
        {
            ret.push_back(*it);
            continue;
        }

        y1 =  std::min(y + height, tmp->y + tmp->height) - std::max(y, tmp->y) + 1;

        if (y1 <= 0)
        {
            ret.push_back(*it);
            continue;
        }

        intersection = x1 * y1;

		//计算重叠度，大于阈值的，存入retP，咸鱼阈值的，存入ret
        if ( (intersection / (width * height + tmp->width * tmp->height - intersection)) >= over)
            retP.push_back(*it);
        else
            ret.push_back(*it);
    }
	
	//vect中全部为正样本？
    vect = retP;

	//返回负样本？
    return (ret);
}
//计算tmp  和类中box（调用函数时，那个类中的box信息）的重叠度
double BBox::bbOverlap(BBox * tmp)
{
    double x1, y1;
    double intersection;

    x1 =  std::min(x + width, tmp->x + tmp->width) - std::max(x, tmp->x) + 1;
    if (x1 <= 0)
        return 0;
    y1 =  std::min(y + height, tmp->y + tmp->height) - std::max(y, tmp->y) + 1;

    if (y1 <= 0)
        return 0;

    intersection = x1 * y1;
	double area1 = (width+1)*(height+1);
	double area2 = (tmp->width+1)*(tmp->height+1);
    return (intersection / (area1 + area2 - intersection));
}

//计算tmp和 类中Box（调用函数时，那个类中的box信息）的重叠区域
double BBox::bbCoverage(BBox * tmp)
{
    double x1, y1;
    double intersection;

    x1 =  std::min(x + width, tmp->x + tmp->width) - std::max(x, tmp->x) + 1;
    if (x1 <= 0)
        return 0;
    y1 =  std::min(y + height, tmp->y + tmp->height) - std::max(y, tmp->y) + 1;

    if (y1 <= 0)
        return 0;

    intersection = x1 * y1;

    return (intersection);

}
//对BB进行聚类。
//运行次函数后，先把和BB[0]相似的归类，求出平均值。然后再剩下的再归类求平均值。
//最后会返回一个ret(std::vector<BBox *>)，存的是各个类的均值，同时BB被清空。
std::vector<BBox *> BBox::clusterBBoxes(std::vector<BBox *> & BB)
{
    std::vector<BBox *> ret;
    std::vector<BBox *> tmpV1;
    std::vector<BBox *> tmpV2;
    std::vector<BBox *> tmpV3;

    BBox * tmp;

    if (BB.size() == 0)
        return ret;

    while(1){
        tmp = BB[0];
        tmpV1 = tmp->bbOverlap(BB);			//tmpv1中存入tmp（BB[0]）和BB中box的重叠度小于阈值（0.7）的负样本
        tmpV3 = BB;							//BB中剩余样本为正样本，赋给tmpV3

		//遍历整个BB。所有正样本分别于负样本再次比对，tmpV3存入大于阈值的正样本，tmpV1（和tmpV2）存入所有比对完剩余的负样本
        for (std::vector<BBox *>::iterator it = BB.begin(); it != BB.end(); ++it){
            tmpV2 = (*it)->bbOverlap(tmpV1);
            for (unsigned int i=0; i<tmpV1.size(); ++i)
                tmpV3.push_back(tmpV1[i]);
            tmpV1.swap(tmpV2);
        }

		
        if (tmpV3.size() != 0){
            BBox * bbox = new BBox();
            bbox->setBBox(0,0,0,0,0);
            bbox->normCross = 0;
			//找出正样本库tmpV3中，normCross和accuracy的最大值（存在bbox中）。x,y,width,heigth全部累加
			
            for (std::vector<BBox *>::iterator it = tmpV3.begin(); it != tmpV3.end(); ++it){
                bbox->x += (*it)->x;
                bbox->y += (*it)->y;
                bbox->width += (*it)->width;
                bbox->height += (*it)->height;
                if ((*it)->normCross > bbox->normCross)
                    bbox->normCross = (*it)->normCross;
                if ((*it)->accuracy > bbox->accuracy)
                    bbox->accuracy = (*it)->accuracy;
                delete *it;
            }
			//算出正样本库中x,y,width,heigth的平均值
            bbox->x /= tmpV3.size();
            bbox->y /= tmpV3.size();
            bbox->width /= tmpV3.size();
            bbox->height /= tmpV3.size();

			//bbox存入ret。
            ret.push_back(bbox);
			//清空正样本库（tmpV3）和负样本库（tmpV2），负样本库V1没有清空
            tmpV3.clear();
            tmpV2.clear();
        }
        BB.swap(tmpV1);		//BB中存入的是负样本

        if (BB.size() == 0)
            break;
    }

    return ret;
}
//A和B中的BOX进行比对。将A中和B中BOX重复的删除（返回的ret）。
std::vector<BBox *> BBox::findDiff(std::vector<BBox *> & A, std::vector<BBox *> & B)
{
    bool equal;
    std::vector<BBox *> ret;
    if (B.size()==0){
        ret = A;
        return (ret);
    }

    for(std::vector<BBox *>::iterator it = A.begin(); it != A.end(); ++it){
        equal = false;
        for (std::vector<BBox *>::iterator it2 = B.begin(); it2 != B.end(); ++it2)
            if (*it == *it2) {
                equal = true;
                break;
            }

        if (!equal)
            ret.push_back(*it);
    }

    return (ret);
}


