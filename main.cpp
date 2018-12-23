#include <iostream>
#include <vector>
#include <fstream>
#include "BPnet.h"

using namespace std;

#define TRAIN_NUM 60000
#define TEST_NUM 10000

int main(void)
{
    srand((unsigned)time(NULL));
//读取训练数据
    ifstream tr_image("train_image"),tr_label("train_label");
    vector<vector<int>> train_image;
    vector<int> train_label;
    for(int i = 0;i<TRAIN_NUM;i++)
    {
        vector<int> cache;
        for(int j = 0;j<784;j++)
        {
            int x;
            tr_image>>x;
            cache.push_back(x);
        }
        train_image.push_back(cache);
        int y;
        tr_label>>y;
        train_label.push_back(y);
    }
    tr_image.close();
    tr_label.close();
//进行训练
//多次训练，对于每一个训练数据
    int times = 100;
    BPnet bp;
    bp.init();//初始化各个参数
    for(int n = 0;n<times;n++)
    {
//读入，初始化BPnet的输入层与标准输出
        for(int m = 0;m<TRAIN_NUM;m++)
        {  
            bp.setData(train_image[m],train_label[m]);
//前向传播
            bp.forward();
//反向传播
            bp.backward();
//更新权重
            bp.update();
        }
    }
//清空，防止出问题
    train_image.clear();
    train_label.clear();
//读取测试数据
    ifstream ts_image("test_image"),ts_label("test_label");
    vector<vector<int>> test_image;
    vector<int> test_label;
    for(int i = 0;i<TEST_NUM;i++)
    {
        vector<int> cache;
        for(int j = 0;j<784;j++)
        {
            int x;
            ts_image>>x;
            cache.push_back(x);
        }
        test_image.push_back(cache);
        int y;
        ts_label>>y;
        test_label.push_back(y);
    }
    ts_image.close();
    ts_label.close();
//进行测试
    int ac_num = 0;
    for(int m = 0;m<TEST_NUM;m++)
    {
        bp.setData()
        bool isOk = bp.predict();
        if(isOk)
        {
            ac_num++;
        }
    }
    cout<<"accuray:"<<(double)ac_num/TEST_NUM<<endl;

    return 0;
}
