#include "fbow.h"
#include <chrono>
#include <opencv2/opencv.hpp>

#define KITTI_DATADIR   "/home/symao/data/kitti/odometry/dataset/sequences/"
#define SEQUENCE_IDX    0

std::vector<double> readtimes(const char* file) {
    std::vector<double> res;
    std::ifstream fin(file);
    if(!fin.is_open()) {
        printf("[ERROR] cannot open file %s\n", file);
        return res;
    }
    while(!fin.eof()) {
        double t;
        fin>>t;
        res.push_back(t);
    }
    return res;
}

std::vector<cv::Affine3f> readtraj(const char* file) {
    std::vector<cv::Affine3f> traj;
    std::ifstream fin(file);
    if(!fin.is_open()) {
        printf("[ERROR] cannot open file %s\n", file);
        return traj;
    }
    while(!fin.eof()) {
        float t[16] = {0};
        for(int i=0; i<12; i++) fin>>t[i];
        traj.push_back(cv::Affine3f(cv::Mat(4, 4, CV_32FC1, t)));
    }
    return traj;
}

int main(int argc, char **argv) {
    std::string voc_file = "../vocabularies/orb_mur.fbow";

    fbow::Vocabulary fvoc;
    fvoc.readFromFile(voc_file);

    char ts_file[256] = {0};
    sprintf(ts_file, "%s/%02d/times.txt", KITTI_DATADIR, SEQUENCE_IDX);
    auto timestamps = readtimes(ts_file);
    int n_img = timestamps.size();

    char traj_file[256] = {0};
    sprintf(traj_file, "%s/../poses/%02d.txt", KITTI_DATADIR, SEQUENCE_IDX);
    auto trajectory = readtraj(traj_file);

    cv::Ptr<cv::Feature2D> fdetector = cv::ORB::create(2000);
    std::vector<fbow::fBow> words;
    std::vector<cv::Mat> imgs;
    std::vector<int> idxs;

    cv::Mat img;
    for(int i=0; i<n_img; i+=10) {
        // load img
        char fl[256] = {0};
        sprintf(fl, "%s/%02d/image_0/%06d.png", KITTI_DATADIR, SEQUENCE_IDX, i);
        img = cv::imread(fl, cv::IMREAD_GRAYSCALE);
        //detect feature
        std::vector<cv::KeyPoint> keypoints;
        cv::Mat descriptors;
        fdetector->detectAndCompute(img, cv::Mat(), keypoints, descriptors);
        fbow::fBow word = fvoc.transform(descriptors);

        double max_score = -1;
        int max_k = -1;
        for(int k=0; k<words.size()&&idxs[k]+300<i; k++) {
            double score = fbow::fBow::score(word, words[k]);
            if(score>max_score) {
                max_score = score;
                max_k = k;
            }
        }
        char key;
        cv::imshow("img", img);
        if(max_score>0.04) {
            printf("#%d max:%f idx:%d\n", i, max_score, idxs[max_k]);
            cv::imshow("loop",imgs[max_k]);
            key = cv::waitKey();
        } else
        {
            printf("#%d\n", i);
            key = cv::waitKey(20);
        }
        if(key == 27) break;
        words.push_back(word);
        cv::Mat small_img;
        cv::resize(img, small_img, cv::Size(img.cols/4,img.rows/4));
        imgs.push_back(small_img);
        idxs.push_back(i);
    }
}
