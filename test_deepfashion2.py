import torch

if __name__ == "__main__":
    import torch.optim as optim
    from torch.utils.tensorboard import SummaryWriter

    from models import *
    from datasets import *
    from train import *

    cap = cv2.VideoCapture(0)

    device = torch.device('cpu')

    num_classes = len(DeepFashion2Dataset.categories)
    model = SegmentPredictorBbox(num_masks=num_classes + 4, num_labels=num_classes + 4, num_bbox_classes=4)
    model.to(device)

    optimizer = None

    # early stopping params
    early_stopping_patience = 5
    early_stopping_counter = 0

    # check model saving dir
    model_dir = "deepfashion2-segpredbbox-general-small"
    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)

    # Check for the latest saved model
    latest_checkpoint = find_latest_checkpoint(model_dir)
    if latest_checkpoint:
        print(f"Loading model from {latest_checkpoint}")
        model, optimizer, start_epoch, best_acc, counter = load_model(model, optimizer, path=latest_checkpoint)
        start_epoch += 1
        counter += 1
    else:
        start_epoch = 0
        counter = 0
        best_acc = 0.0

    plt.ion()  # Enable interactive mode
    fig, axs = None, None
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
        cv2.imshow('Frame', frame)
        cv2.waitKey(1)

        # Convert the image from BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (256, 192))

        # Convert the image to a PyTorch tensor and rearrange color channels
        frame = torch.from_numpy(frame).permute(2, 0, 1).float().unsqueeze(0).to(device) / 255

        with torch.no_grad():
            pred_masks, pred_classes, pred_bboxes = model(frame)

        frame, pred_masks, pred_classes, pred_bboxes = frame[0].permute(1, 2, 0).cpu().numpy(), pred_masks[
            0].cpu().numpy(), \
            pred_classes[0].cpu().numpy(), pred_bboxes[0].cpu().numpy()

        bboxes_list = []
        for idx, label in enumerate(pred_classes[:4]):
            if label >= 0.3:
                bboxes_list.append((idx, pred_bboxes[idx,].tolist()))

        fig, axs = show_deepfashion2_image_masks_and_labels(frame, pred_masks, pred_classes, bboxes_list, fig, axs)
        plt.pause(0.1)

    plt.ioff()  # Disable interactive mode
