// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2022 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#include <android/asset_manager_jni.h>
#include <android/bitmap.h>
#include <android/log.h>

#include <jni.h>

#include "layer.h"
#include "net.h"
#include "benchmark.h"

static ncnn::UnlockedPoolAllocator g_blob_pool_allocator;
static ncnn::PoolAllocator g_workspace_pool_allocator;

static ncnn::Net yolov5;
static ncnn::Net classifier;

//#if defined(USE_NCNN_SIMPLEOCV)
//#include "simpleocv.h"
//#else
//#include <opencv2/core/core.hpp>
//#include <opencv2/highgui/highgui.hpp>
//#include <opencv2/imgproc/imgproc.hpp>
//#endif
#include <float.h>
#include <stdio.h>
#include <vector>

//struct Object
//{
//    cv::Rect_<float> rect;
//    int label;
//    float prob;
//};

struct Object
{
    float x;
    float y;
    float w;
    float h;
    int label;
    int _value;
    int _color;
    float prob;
};


static inline float intersection_area(const Object& a, const Object& b)
{
    if (a.x > b.x + b.w || a.x + a.w < b.x || a.y > b.y + b.h || a.y + a.h < b.y)
    {
        // no intersection
        return 0.f;
    }

    float inter_width = std::min(a.x + a.w, b.x + b.w) - std::max(a.x, b.x);
    float inter_height = std::min(a.y + a.h, b.y + b.h) - std::max(a.y, b.y);

    return inter_width * inter_height;
}


static void qsort_descent_inplace(std::vector<Object>& faceobjects, int left, int right)
{
    int i = left;
    int j = right;
    float p = faceobjects[(left + right) / 2].prob;

    while (i <= j)
    {
        while (faceobjects[i].prob > p)
            i++;

        while (faceobjects[j].prob < p)
            j--;

        if (i <= j)
        {
            // swap
            std::swap(faceobjects[i], faceobjects[j]);

            i++;
            j--;
        }
    }

    #pragma omp parallel sections
    {
        #pragma omp section
        {
            if (left < j) qsort_descent_inplace(faceobjects, left, j);
        }
        #pragma omp section
        {
            if (i < right) qsort_descent_inplace(faceobjects, i, right);
        }
    }
}

static void qsort_descent_inplace(std::vector<Object>& faceobjects)
{
    if (faceobjects.empty())
        return;

    qsort_descent_inplace(faceobjects, 0, faceobjects.size() - 1);
}

static void nms_sorted_bboxes(const std::vector<Object>& faceobjects, std::vector<int>& picked, float nms_threshold)
{
    picked.clear();

    const int n = faceobjects.size();

    std::vector<float> areas(n);
    for (int i = 0; i < n; i++)
    {
        areas[i] = faceobjects[i].w * faceobjects[i].h;
    }

    for (int i = 0; i < n; i++)
    {
        const Object& a = faceobjects[i];

        int keep = 1;
        for (int j = 0; j < (int)picked.size(); j++)
        {
            const Object& b = faceobjects[picked[j]];

            // intersection over union
            float inter_area = intersection_area(a, b);
            float union_area = areas[i] + areas[picked[j]] - inter_area;
            // float IoU = inter_area / union_area
            if (inter_area / union_area > nms_threshold)
                keep = 0;
        }

        if (keep)
            picked.push_back(i);
    }
}

static inline float sigmoid(float x)
{
    return static_cast<float>(1.f / (1.f + exp(-x)));
}

static void generate_proposals(const ncnn::Mat& anchors, int stride, const ncnn::Mat& in_pad, const ncnn::Mat& feat_blob, float prob_threshold, std::vector<Object>& objects)
{
    const int num_grid_x = feat_blob.w;
    const int num_grid_y = feat_blob.h;

    const int num_anchors = anchors.w / 2;

    const int num_class = feat_blob.c / num_anchors - 5;

    const int feat_offset = num_class + 5;

    for (int q = 0; q < num_anchors; q++)
    {
        const float anchor_w = anchors[q * 2];
        const float anchor_h = anchors[q * 2 + 1];

        for (int i = 0; i < num_grid_y; i++)
        {
            for (int j = 0; j < num_grid_x; j++)
            {
                // find class index with max class score
                int color_index = 0;
                float color_score = -FLT_MAX;
                for (int k = 0; k < 4; k++)
                {
                    float score = feat_blob.channel(q * feat_offset + 6 + k).row(i)[j];
                    if (score > color_score)
                    {
                        color_index = k;
                        color_score = score;
                    }
                }
                int value_index = 0;
                float value_score = -FLT_MAX;
                for (int k = 0; k < 14; k++)
                {
                    float score = feat_blob.channel(q * feat_offset + 10 + k).row(i)[j];
                    if (score > value_score)
                    {
                        value_index = k;
                        value_score = score;
                    }
                }

                int class_index = 4 * value_index + color_index + 1;

                float box_score = feat_blob.channel(q * feat_offset + 4).row(i)[j];

                float confidence_box = sigmoid(box_score);
                float confidence_class = 0.5 * (sigmoid(value_score) + sigmoid(color_score));

                if (confidence_box >= 0.1 && confidence_class >= prob_threshold)
                {
                    // yolov5/models/yolo.py Detect forward
                    // y = x[i].sigmoid()
                    // y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i].to(x[i].device)) * self.stride[i]  # xy
                    // y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh

                    float dx = sigmoid(feat_blob.channel(q * feat_offset + 0).row(i)[j]);
                    float dy = sigmoid(feat_blob.channel(q * feat_offset + 1).row(i)[j]);
                    float dw = sigmoid(feat_blob.channel(q * feat_offset + 2).row(i)[j]);
                    float dh = sigmoid(feat_blob.channel(q * feat_offset + 3).row(i)[j]);

                    float pb_cx = (dx * 2.f - 0.5f + j) * stride;
                    float pb_cy = (dy * 2.f - 0.5f + i) * stride;

                    float pb_w = pow(dw * 2.f, 2) * anchor_w;
                    float pb_h = pow(dh * 2.f, 2) * anchor_h;

                    float x0 = pb_cx - pb_w * 0.5f;
                    float y0 = pb_cy - pb_h * 0.5f;
                    float x1 = pb_cx + pb_w * 0.5f;
                    float y1 = pb_cy + pb_h * 0.5f;

                    Object obj;
                    obj.x = x0;
                    obj.y = y0;
                    obj.w = x1 - x0;
                    obj.h = y1 - y0;
                    obj.prob = confidence_class;
                    obj.label = class_index;
                    obj._value = value_index;
                    obj._color = color_index;

                    objects.push_back(obj);
                }
            }
        }
    }
}
extern "C" {

// FIXME DeleteGlobalRef is missing for objCls
static jclass objCls = NULL;
static jmethodID constructortorId;
static jfieldID xId;
static jfieldID yId;
static jfieldID wId;
static jfieldID hId;
static jfieldID labelId;
static jfieldID probId;
static jfieldID _valueId;
static jfieldID _colorId;

JNIEXPORT jint JNI_OnLoad(JavaVM* vm, void* reserved)
{
__android_log_print(ANDROID_LOG_DEBUG, "YoloV5Ncnn", "JNI_OnLoad");

ncnn::create_gpu_instance();

return JNI_VERSION_1_4;
}

JNIEXPORT void JNI_OnUnload(JavaVM* vm, void* reserved)
{
    __android_log_print(ANDROID_LOG_DEBUG, "YoloV5Ncnn", "JNI_OnUnload");

    ncnn::destroy_gpu_instance();
}

// public native boolean Init(AssetManager mgr);
JNIEXPORT jboolean JNICALL Java_com_tencent_yolov5ncnn_YoloV5Ncnn_Init(JNIEnv* env, jobject thiz, jobject assetManager)
{
    ncnn::Option opt;
    opt.lightmode = true;
    opt.num_threads = 2;
    opt.blob_allocator = &g_blob_pool_allocator;
    opt.workspace_allocator = &g_workspace_pool_allocator;
    opt.use_packing_layout = true;

    // use vulkan compute
    if (ncnn::get_gpu_count() != 0)
        opt.use_vulkan_compute = true;

    AAssetManager* mgr = AAssetManager_fromJava(env, assetManager);

    yolov5.opt = opt;
    classifier.opt = opt;
    // init param
    {
        int ret = yolov5.load_param(mgr, "multilabel_yolov5n6.ncnn.param");
        ret = classifier.load_param(mgr, "rummi.param");
        if (ret != 0)
        {
            __android_log_print(ANDROID_LOG_DEBUG, "YoloV5Ncnn", "load_param failed");
            return JNI_FALSE;
        }
    }

    // init bin
    {
        int ret = yolov5.load_model(mgr, "multilabel_yolov5n6.ncnn.bin");
        ret = classifier.load_model(mgr, "rummi.bin");
        if (ret != 0)
        {
            __android_log_print(ANDROID_LOG_DEBUG, "YoloV5Ncnn", "load_model failed");
            return JNI_FALSE;
        }
    }

    // init jni glue
    jclass localObjCls = env->FindClass("com/tencent/yolov5ncnn/YoloV5Ncnn$Obj");
    objCls = reinterpret_cast<jclass>(env->NewGlobalRef(localObjCls));

    constructortorId = env->GetMethodID(objCls, "<init>", "(Lcom/tencent/yolov5ncnn/YoloV5Ncnn;)V");

    xId = env->GetFieldID(objCls, "x", "F");
    yId = env->GetFieldID(objCls, "y", "F");
    wId = env->GetFieldID(objCls, "w", "F");
    hId = env->GetFieldID(objCls, "h", "F");
    labelId = env->GetFieldID(objCls, "label", "Ljava/lang/String;");
    probId = env->GetFieldID(objCls, "prob", "F");
    _valueId = env->GetFieldID(objCls, "_value", "I");
    _colorId = env->GetFieldID(objCls, "_color", "I");

    return JNI_TRUE;
}

// public native Obj[] Detect(Bitmap bitmap, boolean use_gpu);
JNIEXPORT jobjectArray JNICALL Java_com_tencent_yolov5ncnn_YoloV5Ncnn_Detect(JNIEnv* env, jobject thiz, jobject bitmap, jboolean use_gpu)
{
    if (use_gpu == JNI_TRUE && ncnn::get_gpu_count() == 0)
    {
        __android_log_print(ANDROID_LOG_DEBUG, "YoloV5Ncnn", "no gpu");
        return NULL;
        //return env->NewStringUTF("no vulkan capable gpu");
    }

    double start_time = ncnn::get_current_time();

    AndroidBitmapInfo info;
    AndroidBitmap_getInfo(env, bitmap, &info);
    const int width = info.width;
    const int height = info.height;
    if (info.format != ANDROID_BITMAP_FORMAT_RGBA_8888)
        return NULL;

    // ncnn from bitmap
//    const int target_size = 640;
    const int target_size = 1280;

    // letterbox pad to multiple of 32
    int w = width;
    int h = height;
    float scale = 1.f;
    if (w > h)
    {
        scale = (float)target_size / w;
        w = target_size;
        h = h * scale;
    }
    else
    {
        scale = (float)target_size / h;
        h = target_size;
        w = w * scale;
    }

    ncnn::Mat in = ncnn::Mat::from_android_bitmap_resize(env, bitmap, ncnn::Mat::PIXEL_RGB, w, h);

    // pad to target_size rectangle
    // yolov5/utils/datasets.py letterbox
//    int wpad = (w + 31) / 32 * 32 - w;
//    int hpad = (h + 31) / 32 * 32 - h;
    int wpad = (w + 63) / 64 * 64 - w;
    int hpad = (h + 63) / 64 * 64 - h;
    ncnn::Mat in_pad;
    ncnn::copy_make_border(in, in_pad, hpad / 2, hpad - hpad / 2, wpad / 2, wpad - wpad / 2, ncnn::BORDER_CONSTANT, 114.f);
//    __android_log_print(ANDROID_LOG_DEBUG, "YoloV5Ncnn", "img_w: %d, img_h: %d", in_pad.row(0)[], in_pad.h);
    // yolov5
    std::vector<Object> objects;
    {
        const float prob_threshold = 0.5f;
        const float nms_threshold = 0.45f;

        const float norm_vals[3] = {1 / 255.f, 1 / 255.f, 1 / 255.f};
//        const float norm_vals[3] = {1.f, 1.f, 1.f};
        in_pad.substract_mean_normalize(0, norm_vals);

        ncnn::Extractor ex = yolov5.create_extractor();

        ex.set_vulkan_compute(use_gpu);

        int res = ex.input("in0", in_pad);
//        for (int i = 0; i < in_pad.w; i++)
//        {
//            for (int j = 0; j < in_pad.h; j++)
//            {
//                __android_log_print(ANDROID_LOG_DEBUG, "YoloV5Ncnn", "in_pad[%d][%d] = %f", i, j, in_pad.row(i)[j]);
//            }
//        }

        std::vector<Object> proposals;

        // anchor setting from yolov5/models/yolov5s.yaml
        double start_time1 = ncnn::get_current_time();

        // stride 8
        {
            ncnn::Mat out;
            int res = ex.extract("out0", out, 1);
//            for (int i = 0; i < in_pad.w; i++)
//            {
//                for (int j = 0; j < in_pad.h; j++)
//                {
//                    __android_log_print(ANDROID_LOG_DEBUG, "YoloV5Ncnn", "out[%d][%d] = %f", i, j, out.row(i)[j]);
//                }
//            }

            ncnn::Mat anchors(6);
            anchors[0] = 19.f;
            anchors[1] = 27.f;
            anchors[2] = 44.f;
            anchors[3] = 40.f;
            anchors[4] = 38.f;
            anchors[5] = 94.f;

            std::vector<Object> objects8;
            generate_proposals(anchors, 8, in_pad, out, prob_threshold, objects8);

            proposals.insert(proposals.end(), objects8.begin(), objects8.end());
        }

        // stride 16
        {
            ncnn::Mat out;
            ex.extract("out1", out, 1);

            ncnn::Mat anchors(6);
            anchors[0] = 96.f;
            anchors[1] = 68.f;
            anchors[2] = 86.f;
            anchors[3] = 152.f;
            anchors[4] = 180.f;
            anchors[5] = 137.f;

            std::vector<Object> objects16;
            generate_proposals(anchors, 16, in_pad, out, prob_threshold, objects16);

            proposals.insert(proposals.end(), objects16.begin(), objects16.end());
        }

        // stride 32
        {
            ncnn::Mat out;
            ex.extract("out2", out, 1);

            ncnn::Mat anchors(6);
            anchors[0] = 140.f;
            anchors[1] = 301.f;
            anchors[2] = 303.f;
            anchors[3] = 264.f;
            anchors[4] = 238.f;
            anchors[5] = 542.f;

            std::vector<Object> objects32;
            generate_proposals(anchors, 32, in_pad, out, prob_threshold, objects32);

            proposals.insert(proposals.end(), objects32.begin(), objects32.end());
        }

        // stride 64
        {
            ncnn::Mat out;
            int res = ex.extract("out3", out, 1);
            __android_log_print(ANDROID_LOG_DEBUG, "YoloV5Ncnn", "64 result %d", res);

            ncnn::Mat anchors(6);
            anchors[0] = 436.f;
            anchors[1] = 615.f;
            anchors[2] = 739.f;
            anchors[3] = 380.f;
            anchors[4] = 925.f;
            anchors[5] = 792.f;

            std::vector<Object> objects64;
            generate_proposals(anchors, 64, in_pad, out, prob_threshold, objects64);

            proposals.insert(proposals.end(), objects64.begin(), objects64.end());
        }

        double elasped1 = ncnn::get_current_time() - start_time1;
        __android_log_print(ANDROID_LOG_DEBUG, "YoloV5Ncnn", "%.2fms   detect1", elasped1);

        // sort all proposals by score from highest to lowest
        qsort_descent_inplace(proposals);

        // apply nms with nms_threshold
        std::vector<int> picked;
        nms_sorted_bboxes(proposals, picked, nms_threshold);

        int count = picked.size();

        objects.resize(count);
        for (int i = 0; i < count; i++)
        {
            objects[i] = proposals[picked[i]];

            // adjust offset to original unpadded
            float x0 = (objects[i].x - (wpad / 2)) / scale;
            float y0 = (objects[i].y - (hpad / 2)) / scale;
            float x1 = (objects[i].x + objects[i].w - (wpad / 2)) / scale;
            float y1 = (objects[i].y + objects[i].h - (hpad / 2)) / scale;

            // clip
            x0 = std::max(std::min(x0, (float)(width - 1)), 0.f);
            y0 = std::max(std::min(y0, (float)(height - 1)), 0.f);
            x1 = std::max(std::min(x1, (float)(width - 1)), 0.f);
            y1 = std::max(std::min(y1, (float)(height - 1)), 0.f);

            objects[i].x = x0;
            objects[i].y = y0;
            objects[i].w = x1 - x0;
            objects[i].h = y1 - y0;
        }
    }

    // objects to Obj[]
    static const char* class_names[] = {
            "empty","1-blue","1-black","1-orange","1-red","2-blue","2-black","2-orange","2-red",
            "3-blue","3-black","3-orange","3-red","4-blue","4-black","4-orange","4-red",
            "5-blue","5-black","5-orange","5-red","6-blue","6-black","6-orange","6-red",
            "7-blue","7-black","7-orange","7-red","8-blue","8-black","8-orange","8-red",
            "9-blue","9-black","9-orange","9-red","10-blue","10-black","10-orange","10-red",
            "11-blue","11-black","11-orange","11-red","12-blue","12-black","12-orange","12-red",
            "13-blue","13-black","13-orange","13-red","j-blue","j-black","j-orange","j-red",
    };

    jobjectArray jObjArray = env->NewObjectArray(objects.size(), objCls, NULL);

    for (size_t i=0; i<objects.size(); i++)
    {
        jobject jObj = env->NewObject(objCls, constructortorId, thiz);

        env->SetFloatField(jObj, xId, objects[i].x);
        env->SetFloatField(jObj, yId, objects[i].y);
        env->SetFloatField(jObj, wId, objects[i].w);
        env->SetFloatField(jObj, hId, objects[i].h);
        env->SetObjectField(jObj, labelId, env->NewStringUTF(class_names[objects[i].label]));
        env->SetFloatField(jObj, probId, objects[i].prob);
        env->SetIntField(jObj, _valueId, objects[i]._value);
        env->SetIntField(jObj, _colorId, objects[i]._color);

        env->SetObjectArrayElement(jObjArray, i, jObj);
    }

    double elasped = ncnn::get_current_time() - start_time;
    __android_log_print(ANDROID_LOG_DEBUG, "YoloV5Ncnn", "%.2fms   detect", elasped);

    return jObjArray;
}

}