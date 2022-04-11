import argparse
import os
import platform
import shutil
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from .models.experimental import attempt_load
from .utils.datasets import LoadStreams, LoadImages
from .utils.plots import plot_one_box
from .utils.general import (
    check_img_size, non_max_suppression, apply_classifier, scale_coords,
    xyxy2xywh, strip_optimizer, set_logging)
from .utils.torch_utils import select_device, load_classifier, time_synchronized
from tensorflow import keras
#from keras.applications.vgg16 import preprocess_input
from PIL import Image
# keras, pillow讀檔案也可以!
from keras.preprocessing import image
import numpy as np
from matplotlib import pyplot as plt
from keras.models import load_model
from .models.yolo import Model

weights = './best.pt'
device=torch.device('cpu')
model = attempt_load(weights, map_location=device)
parser = argparse.ArgumentParser()
parser.add_argument('--weights', nargs='+', type=str, default='best.pt', help='model.pt path(s)')
parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
parser.add_argument('--output', type=str, default='inference/output', help='output folder')  # output folder
parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
parser.add_argument('--conf-thres', type=float, default=0.4, help='object confidence threshold')
parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
parser.add_argument('--device', default='cpu', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
parser.add_argument('--view-img', action='store_true', help='display results')
parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
parser.add_argument('--augment', action='store_true', help='augmented inference')
parser.add_argument('--update', action='store_true', help='update all models')

indices = {0: '吉娃娃', 1: '日本狆', 2: '馬爾濟斯', 3: '獅子狗', 4: '西施犬', 5: '查理王小獵犬', 6: '蝴蝶犬', 7: '玩具梗犬', 8: '羅德西亞脊背犬', 9: '阿富汗獵犬', 10: '巴吉度獵犬', 11: '比格犬', 12: '尋血獵犬', 13: '布鲁特克浣熊犬', 14: '黑褐色獵浣熊犬', 15: '競走者獵浣熊犬', 16: '英國獵狐犬', 17: '美洲赤狗', 18: '蘇俄牧羊犬', 19: '愛爾蘭獵狼犬', 20: '格雷伊獵犬', 21: '惠比特犬', 22: '依比沙獵犬', 23: '挪威獵鷹犬', 24: '奧達水懶獵犬', 25: '薩路基獵犬', 26: '蘇格蘭獵鹿犬', 27: '威瑪犬', 28: '斯塔福郡鬥牛梗', 29: '美國史特富郡梗', 30: '貝林登㹴', 31: '邊境㹴', 32: '凱利藍㹴', 33: '愛爾蘭㹴', 34: '諾福克㹴', 35: '挪利其㹴', 36: '約克夏㹴', 37: '剛毛獵狐㹴', 38: '湖畔㹴', 39: '西里漢㹴', 40: '萬能㹴', 41: '凱恩㹴', 42: '澳洲㹴', 43: '丹第丁蒙㹴', 44: '波士頓㹴', 45: '迷你雪納瑞', 46: '巨型雪納瑞犬', 47: '標準型雪納瑞', 48: '蘇格蘭㹴犬', 49: '西藏㹴', 50: '澳洲絲毛㹴', 51: '愛爾蘭軟毛㹴 ', 52: '西高地白㹴', 53: '拉薩犬', 54: '平毛尋回犬', 55: '捲毛尋回犬', 56: '黃金獵犬', 57: '拉布拉多犬', 58: '乞沙比克獵犬', 59: '德國短毛指示犬', 60: '維茲拉犬', 61: '英國蹲獵犬', 62: '愛爾蘭雪達犬', 63: '戈登蹲獵犬', 64: '不列塔尼獵犬', 65: '克倫伯獵犬', 66: '史賓格犬', 67: '威爾斯激飛獵犬', 68: '可卡獵犬', 69: '薩塞克斯獵犬', 70: '愛爾蘭水獵犬', 71: '庫瓦茲犬', 72: '史奇派克犬', 73: '格羅安達犬', 74: '比利時瑪連萊犬', 75: '伯瑞犬', 76: '澳洲卡爾比犬', 77: '可蒙犬', 78: '英國古代牧羊犬', 79: '喜樂蒂牧羊犬', 80: '可麗牧羊犬', 81: '邊境牧羊犬', 82: '法蘭德斯畜牧犬', 83: '羅威那', 84: '德國牧羊犬', 85: '杜賓犬', 86: '迷你品犬', 87: '大瑞士山地犬', 88: '伯恩山犬', 89: '阿彭策爾山犬', 90: '恩特布山犬', 91: '拳師犬', 92: '鬥牛獒', 93: '藏獒', 94: '法國鬥牛犬', 95: '大丹犬', 96: '聖伯納犬', 97: '美國愛斯基摩犬', 98: '阿拉斯加雪橇犬', 99: '西伯利亞哈士奇', 100: '猴㹴', 101: '貝生吉犬', 102: '巴哥犬', 103: '蘭伯格犬', 104: '紐芬蘭犬', 105: '大白熊犬', 106: '薩摩耶犬', 107: '博美犬', 108: '鬆獅犬', 109: '凱斯犬', 110: '布魯塞爾格林芬犬', 111: '潘布魯克威爾斯柯基犬', 112: '卡提根威爾斯柯基犬', 113: '貴賓犬', 114: '迷你貴賓犬', 115: '標準貴賓犬', 116: '墨西哥無毛犬', 117: '澳洲野犬', 118: '豺狗', 119: '非洲野犬'}

dogs_model = load_model('./mainView/code/best_ResNet50_dogClass_model.hdf5')

def img_read(fname):
    IMG_WIDTH = 250
    IMG_HEIGHT = 250
    img = Image.open(fname)
    img = img.resize((IMG_WIDTH, IMG_HEIGHT)) # 注意:習慣寫法 width, height
    x = np.asarray(img, dtype='float32')
    x = x.reshape(1,IMG_HEIGHT,IMG_WIDTH,3)
    x = x/255
    return x

def predictImg(img):
    x = img_read(img)
    predict = dogs_model.predict(x)
    predict = np.argmax(predict,axis=1)
    return indices[predict[0]]

def tec_detect(source):
    opt = parser.parse_args(args=[ 
    '--img-size', '416', 
    '--conf-thres', '0.5', 
    '--iou-thres','0.5', 
    '--source', source, 
    '--output','./templates/static/output'])
    # 獲取輸出資料夾，輸入源，權重，參數等參數
    out, source, weights, view_img, save_txt, imgsz = opt.output, opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
    
    # 來源若為鏡頭與影片，不儲存處理結果圖片
    save_img = False
    
    #webcam = source == '0' or source.startswith('rtsp') or source.startswith('http') or source.endswith('.txt')
    webcam = source.isnumeric() or source.startswith(('rtsp://', 'rtmp://', 'http://')) or source.endswith('.txt')
    
    # Initialize
    # 獲取設備
    device = select_device(opt.device)
    
    # 如果要儲存偵測到的物件圖形 save_img = True的話
    # 移除之前的輸出資料夾 如果不存檔    
    if os.path.exists(out):
        shutil.rmtree(out)  # delete output folder
    os.makedirs(out)  # make new output folder
    # 如果設備為gpu，使用Float16
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    # 載入Float32模型，確保使用者設定的輸入圖片解析度能整除32(如不能則調整為能整除並返回)
    # 在前面已經有載入模型，不要重複載入，浪費時間
    # model = attempt_load(weights, map_location=device)  # load FP32 model
    # 輸入圖片size必須為32的倍數，若不是，會自動調整為32的倍數
    imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size 
    # 設置Float16
    if half:
        model.half()  # to FP16

    # Second-stage classifier
    # 設置第二次分類，默認不使用
    # 尚未實驗
    #classify = False
    #if classify:
    #    modelc = load_classifier(name='resnet101', n=2)  # initialize
    #    modelc.load_state_dict(torch.load('./resnet101.pt', map_location=device)['model'])  # load weights
    #    modelc.to(device).eval()
    model.to(device).float()
    # Set Dataloader
    # 通過不同的輸入源來設置不同的資料載入方式
    vid_path, vid_writer = None, None
    if webcam:
        view_img = True
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz)
    else:
        #存檔於否?
        save_img = True
        #save_img = False
        
        # 如果檢測視頻的時候想顯示出來，可以在這里加一行view_img = True
        # 要顯示opencv會掛掉! 
        # view_img = True
        dataset = LoadImages(source, img_size=imgsz)

    # Get names and colors
    # 獲取類別名字
    names = model.module.names if hasattr(model, 'module') else model.names
    # 設置畫框的顏色
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

    # Run inference
    t0 = time.time()
    
    # 進行一次前向推理,測試程式是否正常
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once
    
    """
    path 圖片/視頻路徑
    img 進行resize+pad之後的圖片
    img0 原size圖片
    cap 當讀取圖片時為None，讀取視頻時為視頻源
    """
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        # 圖片也設置為Float16
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        # 沒有batch_size的話則在最前面添加一個軸
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        # print("preprocess_image:", t1 - t0)
        # t1 = time.time()
        """
        前向傳播 返回pred的shape是(1, num_boxes, 5+num_class)
        h,w為傳入網路圖片的長和寬，注意dataset在檢測時使用了矩形推理，所以這裡h不一定等於w
        num_boxes = h/32 * w/32 + h/16 * w/16 + h/8 * w/8
        pred[..., 0:4]為預測框座標
        預測框座標為xywh(中心點+寬長)格式
        pred[..., 4]為objectness置信度
        pred[..., 5:-1]為分類結果
        """
        pred = model(img, augment=opt.augment)[0]
        # t1_ = time_synchronized()
        # print('inference:', t1_ - t1) #印出預測所花的時間

        # Apply NMS
        # 進行NMS
        """
        pred:前向傳播的輸出
        conf_thres:置信度閾值
        iou_thres:iou閾值
        classes:是否只保留特定的類別
        agnostic:進行nms是否也去除不同類別之間的框
        經過nms之後，預測框格式：xywh-->xyxy(左上角右下角)
        pred是一個列表list[torch.tensor]，長度為batch_size
        每一個torch.tensor的shape為(num_boxes, 6),內容為box+conf+cls
        """
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t2 = time_synchronized()
        
        print(f'predResult:{pred}')
        # t2 = time.time()

        # Apply Classifier
        # 添加二次分類，默認不使用
        #if classify:
        #    pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
        # 對每一張圖片作處理
        # im0是im0s的複製，用來畫偵測到物件外框
        
        for i, det in enumerate(pred):  # detections per image
            # 如果輸入源是webcam，則batch_size不為1，取出dataset中的一張圖片
            if webcam:  # batch_size >= 1
                p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()
            else:
                p, s, im0 = path, '', im0s
            # 設置保存圖片/視頻的路徑
            save_path = str(Path(out) / Path(p).name)
            # 設置保存框座標txt檔的路徑
            txt_path = str(Path(out) / Path(p).stem) + ('_%g' % dataset.frame if dataset.mode == 'video' else '')
            # 設置列印資訊(圖片長寬)
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            
            if det is None:
                print("沒有偵測到物件")
            print(f'det:{det}, lenDet:{len(det)}')
            if det is not None and len(det): #det不是None，len(det)就不會是0?
                
                # Rescale boxes from img_size to im0 size
                # 調整預測框的座標：基於resize+pad的圖片的座標-->基於原size圖片的座標
                # 此時座標格式為xyxy

                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                # 列印檢測到的類別數量
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += '%g %ss, ' % (n, names[int(c)])  # add to string
                    
                
                # Write results
                # 保存預測結果
                for *xyxy, conf, cls in det:
                    if save_txt:  # Write to file
                        # 將xyxy(左上角+右下角)格式轉為xywh(中心點+寬長)格式，並除上w，h做歸一化，轉化為列表再保存
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * 5 + '\n') % (cls, *xywh))  # label format
                    # 在原圖上畫框
                    if save_img or view_img:  # Add bbox to image
                        label = '%s %.2f' % (names[int(cls)], conf)
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=2)
                        

            # Print time (inference + NMS)
            # 列印前向傳播+nms時間
            print('%sDone. (%.3fs)' % (s, t2 - t1))

            # Stream results
            # 如果設置展示，則show圖片/視頻
            if view_img:
                cv2.imshow(p, im0)
                if cv2.waitKey(1) == ord('q'):  # q to quit
                    raise StopIteration

            # Save results (image with detections)
            # 設置保存圖片/視頻
            if save_img:
                if dataset.mode == 'image':
                    # 存圖片
                    cv2.imwrite(save_path, im0)
                else:
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer

                        fourcc = 'mp4v'  # output video codec
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
                    vid_writer.write(im0)

    if save_txt or save_img:
        print('Results saved to %s' % Path(out))
        # 打開保存圖片和txt的路徑(適用於MacOS系統)
        if platform == 'darwin' and not opt.update:  # MacOS
            os.system('open ' + save_path)
    # 列印總時間
    # print('Done. (%.3fs)' % (time.time() - t0))
    output_path = save_path.replace('templates\\','')
    return output_path

def detect(source):
    # 獲取輸出資料夾，輸入源，權重，參數等參數
    opt = parser.parse_args(args=[ 
    '--img', '416', 
    '--conf', '0.5', 
    '--iou-thres','0.5', 
    '--source', source, 
    '--output','./output'])
    out = './templates/static/output'
    imgsz = 416
    view_img = opt.view_img
    save_txt = opt.save_txt
    # 來源若為鏡頭與影片，不儲存處理結果圖片
    save_img = True
    # Initialize
    
    # 獲取設備 
    # 如果要儲存偵測到的物件圖形 save_img = True的話
    # 移除之前的輸出資料夾 如果不存檔    
    if os.path.exists(out):
        shutil.rmtree(out)  # delete output folder
    os.makedirs(out)  # make new output folder
    
    # 如果設備為gpu，使用Float16
    half = device.type != 'cpu'  # half precision only supported on CUDA
    # Load model
    # 載入Float32模型，確保使用者設定的輸入圖片解析度能整除32(如不能則調整為能整除並返回)
    # 在前面已經有載入模型，不要重複載入，浪費時間
    # model = attempt_load(weights, map_location=device)  # load FP32 model
    # 輸入圖片size必須為32的倍數，若不是，會自動調整為32的倍數
    imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size 
    # 設置Float16
    if half:
        model.half()  # to FP16

    # Second-stage classifier
    # 設置第二次分類，默認不使用
    # 尚未實驗
    #classify = False
    #if classify:
    #    modelc = load_classifier(name='resnet101', n=2)  # initialize
    #    modelc.load_state_dict(torch.load('./resnet101.pt', map_location=device)['model'])  # load weights
    #    modelc.to(device).eval()

    # Set Dataloader
    # 通過不同的輸入源來設置不同的資料載入方式
    vid_path, vid_writer = None, None
    
    save_img = True
    #save_img = False

    # 如果檢測視頻的時候想顯示出來，可以在這里加一行view_img = True
    # 要顯示opencv會掛掉! 
    # view_img = True
    dataset = LoadImages(source, img_size=imgsz)
    # Get names and colors
    # 獲取類別名字
    names = model.module.names if hasattr(model, 'module') else model.names
    # 設置畫框的顏色
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

    # Run inference
    t0 = time.time()
    
    # 進行一次前向推理,測試程式是否正常
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once
    
    """
    path 圖片/視頻路徑
    img 進行resize+pad之後的圖片
    img0 原size圖片
    cap 當讀取圖片時為None，讀取視頻時為視頻源
    """
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        # 圖片也設置為Float16
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        # 沒有batch_size的話則在最前面添加一個軸
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        # print("preprocess_image:", t1 - t0)
        # t1 = time.time()
        """
        前向傳播 返回pred的shape是(1, num_boxes, 5+num_class)
        h,w為傳入網路圖片的長和寬，注意dataset在檢測時使用了矩形推理，所以這裡h不一定等於w
        num_boxes = h/32 * w/32 + h/16 * w/16 + h/8 * w/8
        pred[..., 0:4]為預測框座標
        預測框座標為xywh(中心點+寬長)格式
        pred[..., 4]為objectness置信度
        pred[..., 5:-1]為分類結果
        """
        pred = model(img, augment=opt.augment)[0]
        print(pred)
        # t1_ = time_synchronized()
        # print('inference:', t1_ - t1) #印出預測所花的時間

        # Apply NMS
        # 進行NMS
        """
        pred:前向傳播的輸出
        conf_thres:置信度閾值
        iou_thres:iou閾值
        classes:是否只保留特定的類別
        agnostic:進行nms是否也去除不同類別之間的框
        經過nms之後，預測框格式：xywh-->xyxy(左上角右下角)
        pred是一個列表list[torch.tensor]，長度為batch_size
        每一個torch.tensor的shape為(num_boxes, 6),內容為box+conf+cls
        """
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t2 = time_synchronized()
        # t2 = time.time()

        # Apply Classifier
        # 添加二次分類，默認不使用
        #if classify:
        #    pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
        # 對每一張圖片作處理
        # im0是im0s的複製，用來畫偵測到物件外框
        
        for i, det in enumerate(pred):  # detections per image
            # 如果輸入源是webcam，則batch_size不為1，取出dataset中的一張圖片
            
            p, s, im0 = path, '', im0s
            # 設置保存圖片/視頻的路徑
            save_path = str(Path(out) / Path(p).name)
            # 設置保存框座標txt檔的路徑
            txt_path = str(Path(out) / Path(p).stem) + ('_%g' % dataset.frame if dataset.mode == 'video' else '')
            # 設置列印資訊(圖片長寬)
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            
            if det is None:
                print("沒有偵測到物件")
            
            if det is not None and len(det): #det不是None，len(det)就不會是0?
                
                # Rescale boxes from img_size to im0 size
                # 調整預測框的座標：基於resize+pad的圖片的座標-->基於原size圖片的座標
                # 此時座標格式為xyxy

                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                # 列印檢測到的類別數量
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += '%g %ss, ' % (n, names[int(c)])  # add to string

                # Write results
                # 保存預測結果
                for *xyxy, conf, cls in det:
                    if save_txt:  # Write to file
                        # 將xyxy(左上角+右下角)格式轉為xywh(中心點+寬長)格式，並除上w，h做歸一化，轉化為列表再保存
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * 5 + '\n') % (cls, *xywh))  # label format
                    # 在原圖上畫框
                    if save_img or view_img:  # Add bbox to image
                        label = '%s %.2f' % (names[int(cls)], conf)
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=2)

            # Print time (inference + NMS)
            # 列印前向傳播+nms時間
            print('%sDone. (%.3fs)' % (s, t2 - t1))
            # Stream results
            # 如果設置展示，則show圖片/視頻
            if view_img:
                cv2.imshow(p, im0)
                if cv2.waitKey(1) == ord('q'):  # q to quit
                    raise StopIteration

            # Save results (image with detections)
            # 設置保存圖片/視頻
            if save_img:
                if dataset.mode == 'image':
                    # 存圖片
                    cv2.imwrite(save_path, im0)

    if save_txt or save_img:
        print('Results saved to %s' % Path(out))
        # 打開保存圖片和txt的路徑(適用於MacOS系統)
        if platform == 'darwin' and not opt.update:  # MacOS
            os.system('open ' + save_path)
    # 列印總時間
    # print('Done. (%.3fs)' % (time.time() - t0))
    output_path = save_path.replace('templates\\','')
    return output_path