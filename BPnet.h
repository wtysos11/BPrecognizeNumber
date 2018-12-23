#ifndef BPNET
#define BPNET
/*
为了方便起见，只有一个隐层的BP神经网络
*/
#include <cmath>
#include <vector>
#include <cstdlib>
#include <cstring>
#include <ctime>
using namespace std;

#define IN_NUM 784 //输入784维向量
#define HID_NUM 20 //隐层20维向量
#define OUT_NUM 10 //输出10维向量
#define ALPHA 0.2//学习率

inline double sigmoid(double x)
{
    return 1/(1+exp(-1*x));
}

//返回-1 ~ 1之间的随机值
inline double getRandom()
{
    return ((2.0*(double)rand()/RAND_MAX) - 1);
}

struct BPnet{
public:
    double dest[OUT_NUM];//目标输出
    double output[OUT_NUM];//输出
    double hidden[HID_NUM];
    double input[IN_NUM];

    double V[IN_NUM][HID_NUM];
    double W[HID_NUM][OUT_NUM];

    double etajy[HID_NUM]; //输入层与隐层间误差
    double etako[OUT_NUM]; //输出层与隐层间误差

    
    void init();
    //设置输入层与输出层（提供的是没有归一化的数据）
    void setData(vector<int> in,int label);

    void forward();

    void backward();

    void update();
//根据输入层与权重计算输出层，如果与实际输出相符合，则返回true
    bool predict();
};

#endif