import os
import json


def create_json_for_omniglot(dataset_dir, output_file):
    # Initialize the dictionary to store the data
    data = {
        'label_names': [],
        'image_names': [],
        'image_labels': []
    }

    # Iterate through each class directory
    class_directories = sorted(os.listdir(dataset_dir))

    for label_index, class_name in enumerate(class_directories):
        class_path = os.path.join(dataset_dir, class_name)

        # Add class name to label_names
        data['label_names'].append(class_name)

        # Check if it's a directory
        if os.path.isdir(class_path):
            # Iterate through subdirectories in each class directory
            for sub_dir in os.listdir(class_path):
                sub_dir_path = os.path.join(class_path, sub_dir)

                # Check if sub_dir is a directory
                if os.path.isdir(sub_dir_path):
                    # Iterate through each image file in the subdirectory
                    for image_file in os.listdir(sub_dir_path):
                        image_path = os.path.join(sub_dir_path, image_file)

                        # Add image path and label to the respective lists
                        if os.path.isfile(image_path):
                            data['image_names'].append(image_path)
                            data['image_labels'].append(label_index)

    # Write the data to a JSON file
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=0)


# Usage
dataset_dir = '/data/pikey/dataset/FewshotData/omniglot-master/python/images_evaluation'  # Replace with the path to your dataset directory
output_file = '/data/pikey/dataset/FewshotData/omniglot-master/python/val.json'  # The name of the output JSON file
create_json_for_omniglot(dataset_dir, output_file)
