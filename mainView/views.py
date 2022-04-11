import cv2,numpy,os,base64
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
from django.conf import settings
from django.shortcuts import render,HttpResponse
from mainView.code import yoloDetect as yd
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
def loadindex(request):
    return render(request, 'index.html', locals())

# Create your views here.
@csrf_exempt
def yoloPredict(request):
    if request.method == "POST": 
        img = request.FILES['image']
        filename = img._get_name()
        filepath = os.path.join("./tmp", filename)
        path = default_storage.save(filepath, ContentFile(img.read()))
        whatDog = yd.predictImg(path)
        #whatDog = '吉娃娃'
        result_img = yd.tec_detect(path)
        return JsonResponse({'data':result_img,
                            'dogKinds' : whatDog})
        