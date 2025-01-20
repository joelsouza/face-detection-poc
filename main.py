from utils.face_utils import process_image_and_store_faces


def main():
    import os
    
    output_folder = os.path.join(os.path.dirname(__file__), "output")
    images_dir = os.path.join(os.path.dirname(__file__), "data")
    for image_index, image_path in enumerate(os.listdir(images_dir)):
        image_path = os.path.join(images_dir, image_path)
        
        process_image_and_store_faces(image_path, output_folder, image_index)

if __name__ == "__main__":
    main()
