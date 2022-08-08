import yolov5
from os import listdir
from pandas import DataFrame
import csv
def prediction_to_csv(path_to_model: str = "model/best.pt", image_path: str = "/Users/hashem/Desktop/KI/Im"):

    train_model = yolov5.load(path_to_model)
    loaded_images = listdir(image_path)

    predictions = []

    object = [2, 4, 5, 3, 9, 6, 7, 10, 8, 1]

    for image in loaded_images:

        full_path = image_path+"/"+image
        print(image_path+"/"+image)

        prediction = train_model("{}".format(full_path))
        x: DataFrame

        for x in prediction.pandas().xywhn:
            elements = []

            for y in x.to_numpy():
                final = [
                   object[y[5]], y[0], y[1], y[2], y[3]
                ]
                elements.append(final)
            image_name = image.split(".",1)[0]
            result_file = open("Output/"+ image_name +".MartinHashem.csv", "w")
            csv_writer = csv.writer(result_file)
            csv_writer.writerow(["ObjectID", "x", "y", "w", "h"])
            csv_writer.writerows(elements)
