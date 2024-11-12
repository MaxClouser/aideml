import aide
import logging
from ultra import DOCS

logging.basicConfig(
    level=logging.DEBUG,
    format='[%(asctime)s] %(name)s - %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("aide")
logger.setLevel(logging.DEBUG)


def main():
    data_description = """The boxes_variants .csv file contains the image name (without the extension), the bbox coordinates, and the class name (aircraft variant). The images/ dir has all of the images. The image names have leading zeros so make sure to treat them as strings and not integers otherwise you will not be able to find the images"""
    # goal=f"Detect and classify different types of airplanes in images using YOLOv8 Nano. Here is info about the provided data: {data_description}"

    # experiment_description = f"""We have a pretrained YOLOv8 Nano model that classifies different types of airplanes in images. 
    # We identified the main weaknesses of the `yolov8n.pt` model, and saved the model weights in the data folder for you. 
    # Specifically, the model does a poor job classifying Boeing 737 aircraft that are parked on the tarmac from the side view in cloudy weather. 
    # Your task is to fine-tune this model using the provided dataset to improve this weakness. You also have access to synthetic data which 
    # contains synthetically generated images which you may use during your evaluation or training. Keep in mind that the synthetic data is not real
    # so it should be used carefully. Here is a description of the provided data: {data_description}.
    # Be sure to save the model weights of the best model so that we can use it for our downstream task."""

    experiment_description = f"""Your task is to train a YOLOv8 Nano model using the provided dataset to classify different types of aircraft. You also have access to synthetic data which 
    contains synthetically generated images which you may use during your evaluation or training. Keep in mind that the synthetic data is not real
    so it should be used carefully. Here is a description of the provided data: {data_description}.
    Be sure to save the model weights of the best model so that we can use it for our downstream task.
    You have access to a single A10 GPU.
    Here is some documentation from ultralytics on YOLO {DOCS}"""

    # experiment_description = f"""Your task is to train an EfficientNet model (specifically, EfficientNet-B0) using the provided dataset to classify different types of aircraft. 
    # You also have access to synthetic data, which contains synthetically generated images. However, synthetic data may not perfectly match the distribution of real-world data, 
    # so it should be used carefully. 
    # Here is a description of the provided data: {data_description}.
    # Be sure to leverage pretrained weights for EfficientNet-B0 to improve efficiency and save the model weights of the best-performing model so that we can use it for our downstream task."""

    experiment_description = f"""Your task is to fine-tune a small image classification model such as EfficientNet or MobileNetV2 using the provided FGVC dataset to classify different types of aircraft. 
    You also have access to synthetic data, which contains synthetically generated images. However, synthetic data may not perfectly match the distribution of real-world data, 
    so it should be used carefully. You have access to a single T4 GPU.
    Here is a description of the provided data: {data_description}.
    Be sure to leverage pretrained weights to improve efficiency and to save the model weights so that we can use it for our downstream task."""

    exp = aide.Experiment(
        data_dir="data/fgvc/",
        goal=experiment_description,
        eval="Maximize the precision"
    )
    # exp = aide.Experiment(
    #     data_dir="example_tasks/bitcoin_price",  # replace this with your own directory
    #     goal="Build a timeseries forcasting model for bitcoin close price.",  # replace with your own goal description
    #     eval="RMSLE"  # replace with your own evaluation metric
    # )

    best_solution = exp.run(steps=100)

    print(f"Best solution has validation metric: {best_solution.valid_metric}")
    print(f"Best solution code: {best_solution.code}")


if __name__ == '__main__':
    main()
