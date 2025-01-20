from deepface import DeepFace
import cv2
import numpy as np
import os

def process_image_and_store_faces(image_path, output_folder, image_index):
    """
    Process an image to detect and store multiple faces.

    Args:
        image_path (str): Path to the input image.
        output_folder (str): Folder to save the cropped faces and embeddings.
    """
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Load the image using OpenCV
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Unable to load image at {image_path}")
        return

    # Detect faces in the image
    try:
        # Extract faces (cropped images and their regions)
        face_objs = DeepFace.extract_faces(img_path=image_path, detector_backend="mtcnn", enforce_detection=False)
    except Exception as e:
        print(f"Error detecting faces: {e}")
        return

    # Process each detected face
    for i, face_obj in enumerate(face_objs):
        # Get the cropped face image
        face_image = face_obj["face"]
        face_region = face_obj["facial_area"]

        # Save the cropped face image
        face_filename = os.path.join(output_folder, f"face_{i + 1}.jpg")
        cv2.imwrite(face_filename, face_image)
        print(f"Saved face {i + 1} to {face_filename}")

        # Extract the face embedding
        try:
            embedding = DeepFace.represent(img_path=face_image, model_name="Facenet", enforce_detection=False)[0]["embedding"]
            # Save the embedding to a file
            embedding_filename = os.path.join(output_folder, f"face_{i + 1}_embedding.txt")
            with open(embedding_filename, "w") as f:
                f.write(",".join(map(str, embedding)))
            print(f"Saved embedding for face {i + 1} to {embedding_filename}")
        except Exception as e:
            print(f"Error extracting embedding for face {i + 1}: {e}")

        # Optional: Draw bounding box on the original image
        x, y, w, h = face_region["x"], face_region["y"], face_region["w"], face_region["h"]
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Save the original image with bounding boxes
    output_image_path = os.path.join(output_folder, f"annotated_image_{image_index}.jpg")
    cv2.imwrite(output_image_path, image)
    print(f"Saved annotated image to {output_image_path}")
