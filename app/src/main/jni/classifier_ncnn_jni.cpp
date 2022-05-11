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

static ncnn::Net classifier;

#include <float.h>
#include <stdio.h>
#include <vector>
#include <jni.h>
#include <jni.h>

class Object {
    int value;
    int color;
    float prob;
};

extern "C" {

static jclass objCls = NULL;

JNIEXPORT jint JNI_OnLoad(JavaVM* vm, void* reserved)
{
    __android_log_print(ANDROID_LOG_DEBUG, "ClassifierNcnn", "JNI_OnLoad");
    ncnn::create_gpu_instance();
    return JNI_VERSION_1_4;
}

JNIEXPORT void JNI_OnUnload(JavaVM* vm, void* reserved)
{
    __android_log_print(ANDROID_LOG_DEBUG, "ClassifierNcnn", "JNI_OnUnload");
    ncnn::destroy_gpu_instance();
}

jboolean Java_com_tencent_yolov5ncnn_ClassifierNcnn_Init(JNIEnv* env, jobject thiz, jobject assetManager)
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

    classifier.opt = opt;
    // init param
    {
        int ret = classifier.load_param(mgr, "rummi.param");
        if (ret != 0)
        {
            __android_log_print(ANDROID_LOG_DEBUG, "ClassifierNcnn", "load_param failed");
            return JNI_FALSE;
        }
    }

    // init bin
    {
        int ret = classifier.load_model(mgr, "rummi.bin");
        if (ret != 0)
        {
            __android_log_print(ANDROID_LOG_DEBUG, "ClassifierNcnn", "load_model failed");
            return JNI_FALSE;
        }
    }

    // init jni glue
    jclass localObjCls = env->FindClass("com/tencent/yolov5ncnn/ClassifierNcnn$Obj");
    objCls = reinterpret_cast<jclass>(env->NewGlobalRef(localObjCls));

//    constructortorId = env->GetMethodID(objCls, "<init>", "(Lcom/tencent/yolov5ncnn/ClassifierNcnn;)V");
//    labelId = env->GetFieldID(objCls, "label", "Ljava/lang/String;");
//    probId = env->GetFieldID(objCls, "prob", "F");
//    valueId = env->GetFieldID(objCls, "_value", "I");
//    colorId = env->GetFieldID(objCls, "_color", "I");

    return JNI_TRUE;
}

JNIEXPORT jobjectArray JNICALL Java_com_tencent_yolov5ncnn_ClassifierNcnn_Detect(JNIEnv* env, jobject thiz, jobject bitmap, jboolean use_gpu)
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

    const int target_size = 32;

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

    int hpad = target_size - h;
    int wpad = target_size - w;
    ncnn::Mat in_pad;
    ncnn::copy_make_border(in, in_pad, hpad / 2, hpad - hpad / 2, wpad / 2, wpad - wpad / 2, ncnn::BORDER_CONSTANT, 114.f);
    std::vector<Object> objects;
    {
        const float prob_threshold = 0.5f;
        const float norm_vals[3] = {1 / 255.f, 1 / 255.f, 1 / 255.f};
        const float mean_vals[3] = {127.5f, 127.5f, 127.5f};
        in_pad.substract_mean_normalize(mean_vals, norm_vals);

        ncnn::Extractor ex = classifier.create_extractor();
        ex.set_vulkan_compute(use_gpu);
        int ret = ex.input("input_1", in_pad);

        if (ret != 0)
        {
            __android_log_print(ANDROID_LOG_DEBUG, "ClassifierNcnn", "input failed");
            return NULL;
        }

        {
            ncnn::Mat out;
            int ret = ex.extract("tf.reshape_2", out, 1);
            if (ret != 0)
            {
                __android_log_print(ANDROID_LOG_DEBUG, "ClassifierNcnn", "output failed");
                return NULL;
            }
        }

//        objects.resize(count);
//        for (int i = 0; i < count; i++)
//        {
//            objects[i] = proposals[picked[i]];
//
//            // adjust offset to original unpadded
//            float x0 = (objects[i].x - (wpad / 2)) / scale;
//            float y0 = (objects[i].y - (hpad / 2)) / scale;
//            float x1 = (objects[i].x + objects[i].w - (wpad / 2)) / scale;
//            float y1 = (objects[i].y + objects[i].h - (hpad / 2)) / scale;
//
//            // clip
//            x0 = std::max(std::min(x0, (float)(width - 1)), 0.f);
//            y0 = std::max(std::min(y0, (float)(height - 1)), 0.f);
//            x1 = std::max(std::min(x1, (float)(width - 1)), 0.f);
//            y1 = std::max(std::min(y1, (float)(height - 1)), 0.f);
//
//            objects[i].x = x0;
//            objects[i].y = y0;
//            objects[i].w = x1 - x0;
//            objects[i].h = y1 - y0;
//        }
    }

    // objects to Obj[]
//    static const char* class_names[] = {
//            "empty","1-blue","1-black","1-orange","1-red","2-blue","2-black","2-orange","2-red",
//            "3-blue","3-black","3-orange","3-red","4-blue","4-black","4-orange","4-red",
//            "5-blue","5-black","5-orange","5-red","6-blue","6-black","6-orange","6-red",
//            "7-blue","7-black","7-orange","7-red","8-blue","8-black","8-orange","8-red",
//            "9-blue","9-black","9-orange","9-red","10-blue","10-black","10-orange","10-red",
//            "11-blue","11-black","11-orange","11-red","12-blue","12-black","12-orange","12-red",
//            "13-blue","13-black","13-orange","13-red","j-blue","j-black","j-orange","j-red",
//    };
//
//    jobjectArray jObjArray = env->NewObjectArray(objects.size(), objCls, NULL);
//
//    for (size_t i=0; i<objects.size(); i++)
//    {
//        jobject jObj = env->NewObject(objCls, constructortorId, thiz);
//
//        env->SetFloatField(jObj, xId, objects[i].x);
//        env->SetFloatField(jObj, yId, objects[i].y);
//        env->SetFloatField(jObj, wId, objects[i].w);
//        env->SetFloatField(jObj, hId, objects[i].h);
//        env->SetObjectField(jObj, labelId, env->NewStringUTF(class_names[objects[i].label]));
//        env->SetFloatField(jObj, probId, objects[i].prob);
//        env->SetIntField(jObj, _valueId, objects[i]._value);
//        env->SetIntField(jObj, _colorId, objects[i]._color);
//
//        env->SetObjectArrayElement(jObjArray, i, jObj);
//    }
//
//    double elasped = ncnn::get_current_time() - start_time;
//    __android_log_print(ANDROID_LOG_DEBUG, "YoloV5Ncnn", "%.2fms   detect", elasped);
//
//    return jObjArray;
    return NULL;
}

}
