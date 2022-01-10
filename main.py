from imageai.Detection import ObjectDetection
from ebaysdk.finding import Connection
import os

api_key = 'EBAY API KEY HERE'

execution_path = os.getcwd()

detector = ObjectDetection()
detector.setModelTypeAsRetinaNet()
# Download resnet50_coco_best_v2.0.1.h5 from "https://github.com/OlafenwaMoses/ImageAI/releases/tag/1.0"
# HDF5 is too large for github
detector.setModelPath(os.path.join(execution_path, "resnet50_coco_best_v2.1.0.h5"))
detector.loadModel()
detections = detector.detectObjectsFromImage(input_image=os.path.join(execution_path, "image.jpg"),
                                             output_image_path=os.path.join(execution_path, "imagenew.jpg"))

api = Connection(appid=api_key, config_file=None, siteid="EBAY-US")


def get_eBay_data(name):
    response = api.execute('findItemsByKeywords', {
        'keywords': name
    })
    data = response.reply.searchResult.item
    return data


def calculate_average(items_data):
    total_price = 0
    num_items = 0
    for items in items_data:
        item_price = items.sellingStatus.currentPrice.value
        total_price += float(item_price)
        num_items += 1
    average = round((total_price / num_items), 2)
    return average


for eachObject in detections:
    items_data = get_eBay_data(eachObject["name"])

    average_price = calculate_average(items_data)
    print(eachObject["name"], " : ", eachObject["percentage_probability"], "- Price: " + str(average_price))
