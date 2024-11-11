import aide
import logging

# Configure root logger
logging.basicConfig(
    level=logging.DEBUG,
    format='[%(asctime)s] %(name)s - %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("aide")
logger.setLevel(logging.DEBUG)


def main():
    data_description = "The bounding box text file includes the image name followed by a spaced followed by the four bounding box numbers which are each seperated by a space.\
        The variant_train text file maps the image name to it's class of airplane. Finally, the images/ dir has all of the images with image_name.jpg."
    
    exp = aide.Experiment(
        data_dir="data/fgvc/",
        goal=f"Detect and classify different types of airplanes in images using YOLOv8 Nano. Here is info about the provided data: {data_description}",
        eval="Precision"
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
