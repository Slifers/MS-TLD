#include "colotracker.h"
#include <iostream>
#include <fstream>

void ColorTracker::init(cv::Mat & img, int x1, int y1, int x2, int y2)
{
	//建立三个灰度图像
    im1 = cv::Mat( img.rows, img.cols, CV_8UC1 );
    im2 = cv::Mat( img.rows, img.cols, CV_8UC1 );
    im3 = cv::Mat( img.rows, img.cols, CV_8UC1 );

    //boundary checks
	//检查box是否超出了图像边界
    y1 = std::max(0, y1);
    y2 = std::min(img.rows-1, y2);
    x1 = std::max(0, x1);
    x2 = std::min(img.cols-1, x2);

	//图像预处理（将图像拆分出三通道的灰度图）
    preprocessImage(img);

	//提取前景直方图，赋给q_orig_hist
    extractForegroundHistogram(x1, y1, x2, y2, q_hist);
    q_orig_hist = q_hist;

	//提取背景直方图，赋给b_weights
    extractBackgroundHistogram(x1, y1, x2, y2, b_hist);
    Histogram b_weights = b_hist;

    b_weights.transformToWeights();			//这个是对背景直方图的处理。处理过后data中全为1？？？
    q_hist.multiplyByWeights(&b_weights);	//运行后，等于是又归一化了一次。b_weights.data中都是1

    lastPosition.setBBox(x1, y1, x2-x1, y2-y1, 1, 1);
    defaultWidth = x2-x1;
    defaultHeight = y2-y1;
    sumIter = 0;
    frame = 0;


    double w2 = (x2-x1)/2.;
    double h2 = (y2-y1)/2.;
	//这里是坐标中点
    double cx = x1 + w2;
    double cy = y1 + h2;
	//这应该是，宽高的偏移步长吧
    double wh = w2+5.;
    double hh = h2+5.;

	
    double Sbg = 0, Sfg = 0;
    for(int i = y1; i < y2+1; i++) {
        const uchar *Mi1 = im1.ptr<uchar>(i);
        const uchar *Mi2 = im2.ptr<uchar>(i);
        const uchar *Mi3 = im3.ptr<uchar>(i);
        double tmp_y = std::pow((cy - i) / hh, 2);
        for (int j = x1; j < x1 + 1; j++) {
            double arg = std::pow((cx - j) / wh, 2) + tmp_y;
            if (arg > 1)
                continue;
            //likelihood weights
            double wqi = 1.0;
			//背景对应权重/前景对应权重  ，如果大于1，累加进Sbg,如果小于1，则为0
            double wbi = sqrt(b_hist.getValue(Mi1[j], Mi2[j], Mi3[j]) /
                              q_orig_hist.getValue(Mi1[j], Mi2[j], Mi3[j]));
			//如果背景权重大于前景权重，累计额前景权重
            Sbg += (wqi < wbi) ? q_orig_hist.getValue(Mi1[j], Mi2[j], Mi3[j])
                               : 0.0;
            Sfg += q_orig_hist.getValue(Mi1[j], Mi2[j], Mi3[j]);	//累加前景权重
        }
    }

    //wAvgBg = 0.5
	//0.1-0.5之间
    wAvgBg = std::max(0.1, std::min(Sbg/Sfg, 0.5));
    bound1 = 0.05;
    bound2 = 0.1;

}

cv::Point ColorTracker::histMeanShift(double x1, double y1, double x2, double y2)
{
    int maxIter = 15;

    double w2 = (x2-x1+1)/2;
    double h2 = (y2-y1+1)/2;

    double borderX = 5;
    double borderY = 5;

    double cx = x1 + w2;
    double cy = y1 + h2;

    Histogram y1hist;

    int ii;
    for (ii = 0; ii < maxIter; ++ii){

        double wh = (w2+borderX);
        double hh = (h2+borderY);
        int rowMin = std::max(0, (int)(cy-hh));
        int rowMax = std::min(im1.rows, (int)(cy+hh));
        int colMin = std::max(0, (int)(cx-wh));
        int colMax = std::min(im1.cols, (int)(cx+wh));

        extractForegroundHistogram(colMin, rowMin, colMax, rowMax, y1hist);

        double batta_q = y1hist.computeSimilarity(&q_orig_hist);
        double batta_b = y1hist.computeSimilarity(&b_hist);


        //MeanShift Vector
        double m0 = 0, m1x = 0, m1y = 0;
        for(int i = rowMin; i < rowMax; i++){
            const uchar* Mi1 = im1.ptr<uchar>(i);
            const uchar* Mi2 = im2.ptr<uchar>(i);
            const uchar* Mi3 = im3.ptr<uchar>(i);
            double tmp_y = std::pow((cy-(float)i)/hh,2);
            for(int j = colMin; j < colMax; j++){
                double arg = std::pow((cx-(float)j)/wh,2) + tmp_y;
                if (arg>1)
                    continue;

                double wqi = sqrt(q_orig_hist.getValue(Mi1[j], Mi2[j], Mi3[j])/y1hist.getValue(Mi1[j], Mi2[j], Mi3[j]));
                double wbi = sqrt(b_hist.getValue(Mi1[j], Mi2[j], Mi3[j])/y1hist.getValue(Mi1[j], Mi2[j], Mi3[j]));
                
                double wg = std::max(wqi/batta_q - wbi/batta_b, 0.0)*(-kernelProfile_EpanechnikovDeriv(arg));     

                m0 += wg;
                m1x += (j-cx)*wg;
                m1y += (i-cy)*wg;
            }
        }

        double xn_1 = m1x/m0 + cx;
        double yn_1 = m1y/m0 + cy;

        if (std::pow(xn_1 - cx,2) + std::pow(yn_1 - cy,2) < 0.1)
            break;

        if (m0==m0 && !isinf(m0) && m0 > 0){
            cx = xn_1;
            cy = yn_1;
        }
    }

    return  cv::Point(cx, cy);
}

cv::Point ColorTracker::histMeanShiftIsotropicScale(double x1, double y1, double x2, double y2, double * scale, int * iter)
{
    int maxIter = 15;

    double w2 = (x2-x1)/2;
    double h2 = (y2-y1)/2;

	//border边界
    double borderX = 5;
    double borderY = 5;

	//中点
    double cx = x1 + w2;
    double cy = y1 + h2;

    Histogram y1hist;
    double h0 = 1;

    int ii;
    for (ii = 0; ii < maxIter; ++ii){

		//原来的框box，向外扩展，以默认参数为例。每次扩展五个像素，扩展15次，或者是到达图像的边界
		//展示出来的效果大概就是，以box为中心，均匀的一圈一圈扩展。
        double wh = h0*w2+borderX;
        double hh = h0*h2+borderY;
		//下边是扩展后的box，左上角和右下角的坐标
        int rowMin = std::max(0, (int)(cy-hh));					//y1
        int rowMax = std::min(im1.rows-1, (int)(cy+hh));		//y2
        int colMin = std::max(0, (int)(cx-wh));					//x1
        int colMax = std::min(im1.cols-1, (int)(cx+wh));		//x2

		//提取前景直方图
        extractForegroundHistogram(colMin, rowMin, colMax, rowMax, y1hist);

		//计算新的前景直方图与之前的 前景/背景 直方图的相似度
        double batta_q = y1hist.computeSimilarity(&q_orig_hist);
        double batta_b = y1hist.computeSimilarity(&b_hist);


        //MeanShift Vector
        double m0 = 0, m1x = 0, m1y = 0;

        double wg_dist_sum = 0, wk_sum = 0, Sbg = 0, Sfg = 0;

		//对新的(尺度放大后的)box的处理
        for(int i = rowMin; i < rowMax; i++){
            const uchar* Mi1 = im1.ptr<uchar>(i);
            const uchar* Mi2 = im2.ptr<uchar>(i);
            const uchar* Mi3 = im3.ptr<uchar>(i);
            double tmp_y = std::pow((cy-i)/hh,2);
            for(int j = colMin; j < colMax; j++){
                double arg = std::pow((cx-j)/wh,2) + tmp_y;
                if (arg>1)
                    continue;

                //likelihood weights  可能性权重
                double wqi = sqrt(q_orig_hist.getValue(Mi1[j], Mi2[j], Mi3[j])/y1hist.getValue(Mi1[j], Mi2[j], Mi3[j]));
                double wbi = sqrt(b_hist.getValue(Mi1[j], Mi2[j], Mi3[j])/y1hist.getValue(Mi1[j], Mi2[j], Mi3[j]));
                double w = std::max(wqi/batta_q - wbi/batta_b, 0.0);

                //weights based on "Robust mean-shift tracking with corrected background-weighted histogram"
                // double w = sqrt(q_hist.getValue(Mi1[j], Mi2[j], Mi3[j])/y1hist.getValue(Mi1[j], Mi2[j], Mi3[j]));

                //orig weights
                // double w = sqrt(q_orig_hist.getValue(Mi1[j], Mi2[j], Mi3[j])/y1hist.getValue(Mi1[j], Mi2[j], Mi3[j]));

                double wg = w*(-kernelProfile_EpanechnikovDeriv(arg));     
                double dist = std::sqrt(std::pow((j-cx)/w2,2) + std::pow((i-cy)/h2,2));

                wg_dist_sum += wg*dist;
                wk_sum += w*(kernelProfile_Epanechnikov(arg));

                //orig weights
                // Sbg += (q_orig_hist.getValue(Mi1[j], Mi2[j], Mi3[j]) == 0) ? y1hist.getValue(Mi1[j], Mi2[j], Mi3[j]) : 0;
                // Sfg += q_orig_hist.getValue(Mi1[j], Mi2[j], Mi3[j]);

                //likelihood
                Sbg += (wqi < wbi) ? y1hist.getValue(Mi1[j], Mi2[j], Mi3[j]) : 0;
                Sfg += q_orig_hist.getValue(Mi1[j], Mi2[j], Mi3[j]);

                //SCIA
                // Sbg += (w == 0) ? y1hist.getValue(Mi1[j], Mi2[j], Mi3[j]) : 0;
                // Sfg += q_hist.getValue(Mi1[j], Mi2[j], Mi3[j]);

                m0 += wg;
                m1x += (j-cx)*wg;
                m1y += (i-cy)*wg;
            }
        }

        double xn_1 = m1x/m0 + cx;
        double yn_1 = m1y/m0 + cy;

        //Rebularization
        double reg2 = 0, reg1 = 0;
        reg1 = (wAvgBg - Sbg/Sfg);
        if (std::abs(reg1) > bound1)
            reg1 = reg1 > 0 ? bound1 : -bound1;

        reg2 = -(log(h0));
        if (std::abs(reg2) > bound2)
            reg2 = reg2 > 0 ? bound2 : -bound2;
        
        double h_tmp = (1.0 - wk_sum/m0)*h0 + (1.0/h0)*(wg_dist_sum/m0) + reg1 + reg2;
                
        if (std::pow(xn_1 - cx,2) + std::pow(yn_1 - cy,2) < 0.1)
            break;

        if (m0==m0 && !isinf(m0) && m0 > 0){
            cx = xn_1;
            cy = yn_1;
            h0 = 0.7*h0 + 0.3*h_tmp;
            if (borderX > 5){
                borderX /= 3;
                borderY /= 3;
            }
        }else if (ii == 0){     
            //if in first iteration is m0 not valid => fail (maybe too fast movement) 
            //  try to enlarge the search region
            borderX = 3*borderX;
            borderY = 3*borderY;
        }
    }


    *scale = h0;
    if (iter != NULL)
        *iter = ii;
    return  cv::Point(cx, cy);
}

cv::Point ColorTracker::histMeanShiftAnisotropicScale(double x1, double y1, double x2, double y2, double * width, double * height)
{
    int maxIter = 15;

    double w2 = (x2-x1)/2.;
    double h2 = (y2-y1)/2.;

    double borderX = 5.;
    double borderY = 5.;

    double cx = x1 + w2;
    double cy = y1 + h2;
    double h0_1 = 1.;
    double h0_2 = 1.;

    double wh;
    double hh;

    Histogram y1hist;

    int ii;
    for (ii = 0; ii < maxIter; ++ii){

        wh = w2*h0_1+borderX;
        hh = h2*h0_2+borderY;
        int rowMin = std::max(0, (int)(cy-hh));
        int rowMax = std::min(im1.rows-1, (int)(cy+hh));
        int colMin = std::max(0, (int)(cx-wh));
        int colMax = std::min(im1.cols-1, (int)(cx+wh));

        extractForegroundHistogram(colMin, rowMin, colMax, rowMax, y1hist);

        double batta_q = y1hist.computeSimilarity(&q_orig_hist);
        double batta_b = y1hist.computeSimilarity(&b_hist);

        //MeanShift Vector
        double m0 = 0, m1x = 0, m1y = 0;
        double Swigdist_1 = 0, Swigdist_2 = 0;
        double wk_sum = 0, Sbg = 0, Sfg = 0;

        for(int i = rowMin; i < rowMax; i++){
            const uchar* Mi1 = im1.ptr<uchar>(i);
            const uchar* Mi2 = im2.ptr<uchar>(i);
            const uchar* Mi3 = im3.ptr<uchar>(i);
            double tmp_y = std::pow((cy-(float)i)/hh,2);

            for(int j = colMin; j < colMax; j++){
                double arg = std::pow((cx-(float)j)/wh,2) + tmp_y;
                if (arg>1)
                    continue;

                //likelihood weights
                double wqi = sqrt(q_orig_hist.getValue(Mi1[j], Mi2[j], Mi3[j])/y1hist.getValue(Mi1[j], Mi2[j], Mi3[j]));
                double wbi = sqrt(b_hist.getValue(Mi1[j], Mi2[j], Mi3[j])/y1hist.getValue(Mi1[j], Mi2[j], Mi3[j]));
                double w = std::max(wqi/batta_q - wbi/batta_b, 0.0);

                double wg = w*(-kernelProfile_EpanechnikovDeriv(arg));

                wk_sum += (w * kernelProfile_Epanechnikov(arg));
                Swigdist_1 += wg*std::pow((cx-(float)j),2);
                Swigdist_2 += wg*std::pow((cy-(float)i),2);

                //likelihood
                Sbg += (wqi < wbi) ? y1hist.getValue(Mi1[j], Mi2[j], Mi3[j]) : 0;
                Sfg += q_orig_hist.getValue(Mi1[j], Mi2[j], Mi3[j]);

                m0 += wg;
                m1x += (j-cx)*wg;
                m1y += (i-cy)*wg;
            }
        }

        double a2 = std::pow(w2,2);
        double b2 = std::pow(h2,2);

        float mx = (h0_2/h0_1)*(m1x/m0);
        float my = (h0_1/h0_2)*(m1y/m0);

        double reg1 = (wAvgBg - Sbg/Sfg);
        if (std::abs(reg1) > bound1)
            reg1 = reg1 > 0 ? bound1 : -bound1;

        double reg2_1 = -(log(h0_1));
        if (std::abs(reg2_1) > bound2)
            reg2_1 = reg2_1 > 0 ? bound2 : -bound2;
        double reg2_2 = -(log(h0_2));
        if (std::abs(reg2_2) > bound2)
            reg2_2 = reg2_2 > 0 ? bound2 : -bound2;

        double h_1 = h0_1 - (h0_2/2)*(wk_sum / m0) + (h0_2 / (h0_1 * h0_1 * a2)) * (Swigdist_1 / m0) + reg1 + reg2_1;
        double h_2 = h0_2 - (h0_1/2)*(wk_sum / m0) + (h0_1 / (h0_2 * h0_2 * b2)) * (Swigdist_2 / m0) + reg1 + reg2_2;

        if (std::pow(mx,2) + std::pow(my,2) < 0.1)
            break;

        if (m0==m0 && !isinf(m0) && m0 > 0){
            cx = cx + mx;
            cy = cy + my;
            h0_1 = 0.7*h0_1 + 0.3*h_1;
            h0_2 = 0.7*h0_2 + 0.3*h_2;
            if (borderX > 5){
                borderX /= 3;
                borderY /= 3;
            }
        }else if (ii == 0){
            //if in first iteration is m0 not valid => fail (maybe too fast movement)
            //  try to enlarge the search region
            borderX = 3*borderX;
            borderY = 3*borderY;
        }
    }

    *width = 2*w2*h0_1;
    *height = 2*h2*h0_2;

    return  cv::Point(cx, cy);
}


//跟踪模块
BBox * ColorTracker::track(cv::Mat & img, double x1, double y1, double x2, double y2)
{
    double width = x2-x1;
    double height = y2-y1;

    im1_old = im1.clone();
    im2_old = im2.clone();
    im3_old = im3.clone();
    preprocessImage(img);

    //MS with scale estimation
    double scale = 1; 
    int iter = 0;
	//获得模型中点
    cv::Point modeCenter = histMeanShiftIsotropicScale(x1, y1, x2, y2, &scale, &iter);
    width = 0.7*width + 0.3*width*scale;
    height = 0.7*height + 0.3*height*scale;
	bool tr = true;

    //MS with anisotropic scale estimation
    // double scale = 1.; 
    // double new_width = 1., new_height = 1.;
    // cv::Point modeCenter = histMeanShiftAnisotropicScale(x1, y1, x2, y2, &new_width, &new_height);
    // width = new_width;
    // height = new_height;
    
    //Forward-Backward validation
    if (std::abs(std::log(scale)) > 0.05){
        cv::Mat tmp_im1 = im1;
        cv::Mat tmp_im2 = im2;
        cv::Mat tmp_im3 = im3;
        im1 = im1_old;
        im2 = im2_old;
        im3 = im3_old;
        double scaleB = 1;
        histMeanShiftIsotropicScale(modeCenter.x - width/2, modeCenter.y - height/2, modeCenter.x + width/2, modeCenter.y + height/2, &scaleB);
        im1 = tmp_im1;
        im2 = tmp_im2;
        im3 = tmp_im3;

        if (std::abs(std::log(scale*scaleB)) > 0.1){
            double alfa = 0.1*(defaultWidth/(float)(x2-x1));
            width = (0.9 - alfa)*(x2-x1) + 0.1*(x2-x1)*scale + alfa*defaultWidth;
            height = (0.9 - alfa)*(y2-y1) + 0.1*(y2-y1)*scale + alfa*defaultHeight;
        }
		tr = false;
    }

    BBox * retBB = new BBox();
    retBB->setBBox(modeCenter.x - width/2, modeCenter.y - height/2, width, height, 1, 1);
    lastPosition.setBBox(modeCenter.x - width/2, modeCenter.y - height/2, width, height, 1, 1);
	retBB->setTrack(tr);
    frame++;

    return retBB;
}
//预处理图像，将三个通道拆开
void ColorTracker::preprocessImage(cv::Mat &img)
{
    cv::Mat ra[3] = {im1, im2, im3};
    cv::split(img, ra);
}

//提取背景直方图
//在目标框周围选一圈框，记录这圈框中不包含box，返回的data里weights全为1/data.size()
//距离中心坐标越远，权重越低；距离中心越近，权重越高。远到一定范围，则为0
void ColorTracker::extractBackgroundHistogram(int x1, int y1, int x2, int y2, Histogram & hist)
{
	//width和heigth 的中点
    int offsetX = (x2-x1)/2;
    int offsetY = (y2-y1)/2;

	//
    int rowMin = std::max(0, (int)(y1-offsetY));
    int rowMax = std::min(im1.rows, (int)(y2+offsetY+1));
    int colMin = std::max(0, (int)(x1-offsetX));
    int colMax = std::min(im1.cols, (int)(x2+offsetX+1));

    int numData = (rowMax-rowMin)*(colMax-colMin) - (y2-y1)*(x2-x1);

    if (numData < 1)
        numData = (rowMax-rowMin)*(colMax-colMin)/2 + 1;

    std::vector<int> d1, d2, d3;
    std::vector<double> weights;
    d1.reserve(numData);
    d2.reserve(numData);
    d3.reserve(numData);

    for (int y = rowMin; y < rowMax; ++y){
        const uchar * M1 = im1.ptr<uchar>(y);
        const uchar * M2 = im2.ptr<uchar>(y);
        const uchar * M3 = im3.ptr<uchar>(y);
        for (int x = colMin; x < colMax; ++x){
            if (x >= x1 && x <= x2 && y >= y1 && y <= y2)
                continue;
            d1.push_back(M1[x]);
            d2.push_back(M2[x]);
            d3.push_back(M3[x]);
        }
    }
    hist.clear();    
    hist.insertValues(d1, d2, d3, weights);
}

//提取前景直方图
//归一化权重
void ColorTracker::extractForegroundHistogram(int x1, int y1, int x2, int y2, Histogram & hist)
{
    hist.clear();						//清除data中的数据
    std::vector<int> data1;
    std::vector<int> data2;
    std::vector<int> data3;
    std::vector<double> weights;

    int numData = (y2-y1)*(x2-x1);		//box的面积（包含像素个数）

    if (numData <= 0){
        return;
    }

    data1.reserve(numData);
    data2.reserve(numData);
    data3.reserve(numData);
    weights.reserve(numData);			//权重

    double w2 = (x2-x1)/2;
    double h2 = (y2-y1)/2;

	//这俩是box的中点坐标
    double cx = x1 + w2;
    double cy = y1 + h2;

    double wh_i = 1.0/(w2*1.4142+1);  //sqrt(2)
    double hh_i = 1.0/(h2*1.4142+1);

	//权重有两个值，0和另外一个。  权重的计算与中点到各个像素点的位置有关，和像素值无关。  
	//即权重是一个空间的属性
    for (int y = y1; y < y2+1; ++y){
        const uchar * M1 = im1.ptr<uchar>(y);		//获得im1第y行的指针
        const uchar * M2 = im2.ptr<uchar>(y);
        const uchar * M3 = im3.ptr<uchar>(y);
        double tmp_y = std::pow((cy-y)*hh_i,2);
        for (int x = x1; x < x2+1; ++x){
            data1.push_back(M1[x]);
            data2.push_back(M2[x]);
            data3.push_back(M3[x]);
            weights.push_back(kernelProfile_Epanechnikov(std::pow((cx-x)*wh_i,2) + tmp_y));
        }
    }


    hist.clear();
	//hist中data存入了归一化后的权重值
    hist.insertValues(data1, data2, data3, weights);
}