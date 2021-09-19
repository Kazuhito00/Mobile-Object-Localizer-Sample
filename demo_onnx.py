#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import copy
import time
import argparse

import cv2 as cv
import numpy as np
import onnxruntime


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--movie", type=str, default=None)
    parser.add_argument("--width", help='cap width', type=int, default=960)
    parser.add_argument("--height", help='cap height', type=int, default=540)

    parser.add_argument(
        "--model",
        type=str,
        default=
        'model/object_detection_mobile_object_localizer_v1_1_default_1.onnx',
    )
    parser.add_argument("--score", type=float, default=0.2)

    args = parser.parse_args()

    return args


def run_inference(onnx_session, input_size, image):
    # 前処理
    input_image = cv.resize(image, dsize=(input_size, input_size))  # リサイズ
    input_image = cv.cvtColor(input_image, cv.COLOR_BGR2RGB)  # BGR→RGB変換
    input_image = input_image.reshape(-1, input_size, input_size, 3)  # リシェイプ
    input_image = input_image.astype('uint8')  # uint8へキャスト

    # 推論
    input_name = onnx_session.get_inputs()[0].name
    output_detail = onnx_session.get_outputs()
    output_names = []
    output_names.append(output_detail[0].name)
    output_names.append(output_detail[1].name)
    output_names.append(output_detail[2].name)
    output_names.append(output_detail[3].name)
    outputs = onnx_session.run(output_names, {input_name: input_image})

    # 推論結果取り出し
    bboxes = outputs[0]
    classes = outputs[1]
    scores = outputs[2]
    num = outputs[3]

    bboxes = np.squeeze(bboxes)
    classes = np.squeeze(classes)
    scores = np.squeeze(scores)
    num = int(num[0])

    return bboxes, classes, scores, num


def main():
    # 引数解析 #################################################################
    args = get_args()
    cap_device = args.device
    cap_width = args.width
    cap_height = args.height

    if args.movie is not None:
        cap_device = args.movie

    model_path = args.model
    score_th = args.score

    # カメラ準備 ###############################################################
    cap = cv.VideoCapture(cap_device)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)

    # モデルロード #############################################################
    onnx_session = onnxruntime.InferenceSession(model_path)

    input_size = 192

    while True:
        start_time = time.time()

        # カメラキャプチャ #####################################################
        ret, frame = cap.read()
        if not ret:
            break
        debug_image = copy.deepcopy(frame)

        # 検出実施 ##############################################################
        bboxes, classes, scores, num = run_inference(
            onnx_session,
            input_size,
            frame,
        )

        elapsed_time = time.time() - start_time

        # デバッグ描画
        debug_image = draw_debug(
            debug_image,
            elapsed_time,
            score_th,
            num,
            bboxes,
            classes,
            scores,
        )

        # キー処理(ESC：終了) ##################################################
        key = cv.waitKey(1)
        if key == 27:  # ESC
            break

        # 画面反映 #############################################################
        cv.imshow('Mobile Object Localizer Demo', debug_image)

    cap.release()
    cv.destroyAllWindows()


def draw_debug(
    image,
    elapsed_time,
    score_th,
    num,
    bboxes,
    classes,
    scores,
):
    image_width, image_height = image.shape[1], image.shape[0]
    debug_image = copy.deepcopy(image)

    for i in range(num):
        score = scores[i]
        bbox = bboxes[i]
        # class_id = classes[i].astype(np.int) + 1

        if score < score_th:
            continue

        # 検出結果可視化 ###################################################
        x1, y1 = int(bbox[1] * image_width), int(bbox[0] * image_height)
        x2, y2 = int(bbox[3] * image_width), int(bbox[2] * image_height)

        cv.putText(debug_image, '{:.3f}'.format(score), (x1, y1 - 15),
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv.LINE_AA)
        cv.rectangle(debug_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # 処理時間
    cv.putText(debug_image,
               "Elapsed Time : " + '{:.1f}'.format(elapsed_time * 1000) + "ms",
               (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2,
               cv.LINE_AA)

    return debug_image


if __name__ == '__main__':
    main()