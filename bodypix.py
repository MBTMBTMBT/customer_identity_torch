import cv2
import numpy as np
from tf_bodypix.api import download_model, load_model, BodyPixModelPaths

def extract_mask_region(frame, mask, expand_x=0.5, expand_y=0.5):
    """
    Extracts the face region from the image and expands the region by the specified amount.
    
    :param frame: The source image.
    :param mask: The mask with the face part.
    :param expand_x: The percentage to expand the width of the bounding box.
    :param expand_y: The percentage to expand the height of the bounding box.
    :return: The extracted face region as a numpy array, or None if not found.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)

        # Expand the bounding box
        new_w = w * (1 + expand_x)
        new_h = h * (1 + expand_y)
        x -= (new_w - w) // 2
        y -= (new_h - h) // 2

        # Ensure the new bounding box is within the frame dimensions
        x = int(max(0, x))
        y = int(max(0, y))
        new_w = min(frame.shape[1] - x, new_w)
        new_h = min(frame.shape[0] - y, new_h)

        face_region = frame[y:y+int(new_h), x:x+int(new_w)]
        return face_region
    return None


class BodyPixDetector:
    def __init__(self) -> None:
        model_path = download_model(BodyPixModelPaths.MOBILENET_FLOAT_75_STRIDE_8)
        self.bodypix_model = load_model(model_path)

    def detect(self, frame: np.ndarray, return_masks=False) -> tuple[np.ndarray, np.ndarray]:
        # run prediction on camera frame
        result = self.bodypix_model.predict_single(frame)

        # extract mask with minimum confidence
        mask = result.get_mask(threshold=0.75)
        face_mask = result.get_part_mask(mask, ['left_face', 'right_face'])
        torso_mask = result.get_part_mask(mask, ['torso_front', 'torso_back'])
        # Extract the regions
        face_region = extract_mask_region(frame, face_mask.astype(np.uint8), expand_x=0.4, expand_y=0.5)
        torso_region = extract_mask_region(frame, torso_mask.astype(np.uint8), expand_x=0.2, expand_y=0.0)

        if return_masks:
            return face_region, torso_region, face_mask, torso_mask
        
        return face_region, torso_region


if __name__ == "__main__":
    # load model
    bodypix_model = load_model(download_model(BodyPixModelPaths.MOBILENET_FLOAT_75_STRIDE_8))

    # capture from webcam
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        success, frame = cap.read()

        if success:
            # run prediction on camera frame
            result = bodypix_model.predict_single(frame)

            # extract mask with minimum confidence
            mask = result.get_mask(threshold=0.75)
            face_mask = result.get_part_mask(mask, ['left_face', 'right_face'])
            torso_mask = result.get_part_mask(mask, ['torso_front', 'torso_back'])

            # Extract the regions
            face_region = extract_mask_region(frame, face_mask.astype(np.uint8), expand_x=0.25, expand_y=0.5)
            torso_region = extract_mask_region(frame, torso_mask.astype(np.uint8), expand_x=0.2, expand_y=0.0)

            if face_region is not None:
                cv2.imshow("Face Region", face_region)
            if torso_region is not None:
                cv2.imshow("Torso Region", torso_region)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            print("Failure capturing frame")
            break

    cap.release()
    cv2.destroyAllWindows()
