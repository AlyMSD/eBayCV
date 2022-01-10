from imageai.Detection import ObjectDetection
from ebaysdk.finding import Connection
import os

# Paste eBay Production API key here
api_key = 'EBAY API KEY HERE'

# variable to store where RetinaNet model file and images are
execution_path = os.getcwd()

# define object detection class
detector = ObjectDetection()
# Set model type to RetinaNet
detector.setModelTypeAsRetinaNet()
# Download resnet50_coco_best_v2.0.1.h5 from "https://github.com/OlafenwaMoses/ImageAI/releases/tag/1.0"
# HDF5 is too large for github
# Set file path for model
detector.setModelPath(os.path.join(execution_path, "resnet50_coco_best_v2.1.0.h5"))
# Load model into object detection class
detector.loadModel()
# Parse input image and input image paths
detections = detector.detectObjectsFromImage(input_image=os.path.join(execution_path, "image.jpg"),
                                             output_image_path=os.path.join(execution_path, "imagenew.jpg"))
# Connect to eBay API
api = Connection(appid=api_key, config_file=None, siteid="EBAY-US")


# Function to get eBay listings for the given item
def get_eBay_data(name):
    # Pass in item name to get listing data
    response = api.execute('findItemsByKeywords', {
        'keywords': name
    })
    # Save search results for item to variable and return the results
    data = response.reply.searchResult.item
    return data


# Function to calculate the average price of a given item by passing in listing data
def calculate_average(items_data):
    total_price = 0
    num_items = 0
    # loop through all items and add to total_price
    for items in items_data:
        item_price = items.sellingStatus.currentPrice.value
        total_price += float(item_price)
        num_items += 1
    # Calculate average and round to 2 decimals
    average = round((total_price / num_items), 2)
    return average


# loop through each item identifies and output name, confidence, and average price
for eachObject in detections:
    items_data = get_eBay_data(eachObject["name"])

    average_price = calculate_average(items_data)
    print(eachObject["name"], " : ", eachObject["percentage_probability"], "- Price: " + str(average_price))
