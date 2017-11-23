#include "histogram.h"
#include <cstring>
#include <assert.h>
#include <iostream>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <fstream>
//构造函数，有默认值   Histogram(int _dimSize = 16, int _range = 256);
Histogram::Histogram(int _dimSize, int _range)
{
    dimSize = _dimSize;
    range = _range;
}

//插入值
//input: 图像三个通道的灰度图，每个对应像素的权重
//output: data   其中存入三个灰度图对应像素的值的权重
//				  data可以理解为一个直方图。x轴大小为16*16*16（默认），为三个灰度图对应于一个像素点的
//				  三个像素值经过特定运算产生的一个值。y轴对应位，这个像素点的权重在总权重中的权重
//				  （归一化后的权重），为一个0~1中的值。
void Histogram::insertValues(std::vector<int> & data1, std::vector<int> & data2, std::vector<int> & data3, std::vector<double> &weight)
{
    if (data.size() < (unsigned int)dimSize*dimSize*dimSize)
        data.resize(dimSize*dimSize*dimSize);

    bool useWeights = true;
    if (weight.size() != data1.size())
        useWeights = false;

    rangePerBin = range/dimSize;				//int rangePerBin;  根据默认值为16
    rangePerBinInv = 1./(float)rangePerBin;		//float rangePerBinInv;  根据默认值为0.0625（1/16）
    double sum = 0;
	
	//
    for (unsigned int i=0; i < data1.size(); ++i){
        int id1 = rangePerBinInv*data1[i];
        int id2 = rangePerBinInv*data2[i];
        int id3 = rangePerBinInv*data3[i];
        int id = id1*dimSize*dimSize + id2*dimSize + id3;	//一个像素的三个通道的值分别乘以一个权重相加	
			//id最大值为4368   256*16+256+16
        double w = useWeights ? weight[i] : 1;		//如果weigth个data1的容量相等，则一一对应。否则全设置为1

		//data的大小为16*16*16,但是id 的最大值要比这个大啊
        data[id] += w;
        sum += w;				//sum是weight的和
    }

	//归一化。将data中的权值相加，每个权值更新为  之前的每个权值在权值和中的比重，
    normalize();
}
//计算相似度。两个直方图中  所有对应的data[i]的积 的平方根  的和
double Histogram::computeSimilarity(Histogram * hist)
{
    double conf = 0;
    for (unsigned int i=0; i < data.size(); ++i) {
        conf += sqrt(data[i]*hist->data[i]);
    }
    return conf;
}
//获取对应的权值
double Histogram::getValue(int val1, int val2, int val3)
{
    int id1 = rangePerBinInv*val1;
    int id2 = rangePerBinInv*val2;
    int id3 = rangePerBinInv*val3;
    int id = id1*dimSize*dimSize + id2*dimSize + id3;
    return data[id];
}

void Histogram::transformToWeights()
{
    double min = 0;
/*    std::ifstream alfa_file;
    alfa_file.open("param.txt");
    if (alfa_file.is_open()){
        double sum = 0;
        for (unsigned int i=0; i < data.size(); ++i) {
            sum += data[i];
        }
        double alfa;
        alfa_file >> alfa;
        min = (alfa/100.0)*sum;
        alfa_file.close();
    }else*/
        min = getMin();			//找出data中的最小值

    transformByWeight(min);
}
//再归一化？data中的最小值分别除以data中的数据
void Histogram::transformByWeight(double min)
{
    for (unsigned int i=0; i < data.size(); ++i){
        if (data[i] > 0){
            data[i] = min/data[i];
            if (data[i] > 1)
                data[i] = 1;
        }else
            data[i] = 1;
    }

}

void Histogram::multiplyByWeights(Histogram * hist)
{
    double sum = 0;
    for (unsigned int i=0; i < data.size(); ++i) {
        data[i] *= hist->data[i];
        sum += data[i];
    }

    normalize();
}
void Histogram::clear()
{
    for (unsigned int i=0; i < data.size(); ++i)
        data[i] = 0;
}

//标准化（归一化）
void Histogram::normalize()
{
    double sum = 0;
    for (unsigned int i=0; i < data.size(); ++i)
        sum += data[i];
    for (unsigned int i=0; i < data.size(); ++i)
        data[i] /= sum;
}
//找出data中的最小值
double Histogram::getMin()
{
    double min = 1;
    for (unsigned int i=0; i < data.size(); ++i) {
        if (data[i] < min && data[i] != 0)
            min = data[i];
    }
    return min;
}

void Histogram::addExpHist(double alpha, Histogram & hist)
{
    double beta = 1-alpha;
    for (unsigned int i=0; i < data.size(); ++i){
        data[i] = beta*data[i] + alpha*hist.data[i];
    }
}

