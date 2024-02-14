import torch.optim as optim
from torch.utils.data import DataLoader, random_split, ConcatDataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import random

from models import *
from datasets import *
from utils import *


def _scale_images_uniformly(images: torch.Tensor, scale_factor: float):
    """
    Scale a batch of images with a uniform scale factor, ensuring even dimensions.

    :param images: A torch tensor of shape (batch, channel, h, w)
    :param scale_factor: Scale factor
    :return: Scaled images with even dimensions
    """
    assert 0 < scale_factor < 1

    # Compute new size and ensure it's even
    new_h, new_w = int(images.shape[2] * scale_factor), int(images.shape[3] * scale_factor)
    new_h = new_h + 1 if new_h % 2 != 0 else new_h
    new_w = new_w + 1 if new_w % 2 != 0 else new_w

    # Resize all images in the batch
    scaled_images = F.interpolate(images, size=(new_h, new_w), mode='bilinear', align_corners=False)

    return scaled_images


def train(model, optimizer, train_loader, criterion_mask, criterion_pred, epoch, device, mode=0):
    model.train()
    running_loss = 0.0
    running_accuracy = 0.0
    running_loss = 0.0
    mask_running_loss = 0.0
    pred_running_loss = 0.0
    progress_bar = tqdm(train_loader, desc=f'Training Epoch {epoch}')
    for i, batch in enumerate(progress_bar):
        inputs, mask_labels, attributes, _ = batch

        # img = inputs.clone().squeeze(0).permute(1, 2, 0).numpy().clip(0, 1)
        # # _img = _inputs.clone().squeeze(0).permute(1, 2, 0).numpy().clip(0, 1)
        # masks = mask_labels.clone().squeeze(0).numpy().astype(np.uint8)
        # mask_list = [masks[i,:,:] for i in range(masks.shape[0])]
        # _classes = classes.clone().detach().squeeze(0).numpy().astype(np.uint8)
        # class_list = [_classes[i].item() > 0.5 for i in range(_classes.shape[0])]
        # # _colour_labels = (colour_labels.clone().detach().squeeze(0)*255).numpy().astype(np.uint8)
        # colour_list = [np.array([255,255,255], dtype=np.uint8) for i in range(len(class_list))]
        # combined_image = plot_with_matplotlib(
        #     img, 
        #     ['hair', 'hat', 'eye_g', 'skin',],  # 'brow', 'eye', 'mouth', 'nose', ],
        #     mask_list,
        #     class_list + [None for _ in range(len(mask_list) - len(class_list))],
        #     colour_list,
        #     )
        # cv2.namedWindow('Display', cv2.WINDOW_NORMAL)
        # cv2.resizeWindow('Display', 800, 400)  # adjust the size as needed
        # combined_image = cv2.cvtColor(combined_image, cv2.COLOR_BGR2RGB)
        # cv2.imshow('Display', combined_image)

        attributes = attributes.to(device)
        inputs, mask_labels = inputs.to(device), mask_labels.to(device)

        # Select a uniform scale for the entire batch
        scale_factor = random.uniform(0.5, 1)
        inputs, mask_labels = _scale_images_uniformly(inputs, scale_factor), _scale_images_uniformly(mask_labels, scale_factor)

        optimizer.zero_grad()
        
        pred_masks, pred_classes = model(inputs)
        mask_loss = criterion_mask(pred_masks, mask_labels)
        # pred_loss, cl_loss, rg_loss = criterion_pred(pred_classes, classes, pred_colours)  #, colour_labels)
        pred_loss = criterion_pred(pred_classes, attributes)
        loss = mask_loss + pred_loss
        
        # Assume `predictions` and `classes` are your model's predictions and true class labels respectively
        predictions = pred_classes > 0.5
        # Create a mask for where the true class labels are 1 (positive class)
        positive_class_mask = (attributes == 1)
        # Select predictions and true labels where true labels are 1
        positive_predictions = predictions[positive_class_mask]
        positive_true_labels = attributes[positive_class_mask]
        # Calculate correct predictions for positive class
        correct_positives = (positive_predictions == positive_true_labels).float()
        # Calculate accuracy for positive class
        if correct_positives.numel() > 0:  # Check to make sure we have positive samples
            positive_accuracy = correct_positives.mean()
        else:
            positive_accuracy = torch.tensor(0.0)  # If no positive samples, set accuracy to 0
        # Now `positive_accuracy` will be the accuracy only for the class with label 1
        accuracy = positive_accuracy

        if mode == 1:
            model.unfreeze_segment_model()
            mask_loss.backward()
        else:
            model.freeze_segment_model()
            loss.backward()

        optimizer.step()

        running_loss += loss.item()
        mask_running_loss += mask_loss.item()
        pred_running_loss += pred_loss.item()
        # cl_running_loss += cl_loss.item()
        # rg_running_loss += rg_loss.item()
        
        running_accuracy += accuracy
        progress_bar.set_description(f'Train E{epoch}: ML:{mask_running_loss/(i+1):.3f} PL:{pred_running_loss/(i+1):.3f} Acc:{running_accuracy/(i+1):.2f}')

    train_loss = running_loss / len(train_loader)
    mask_train_loss = mask_running_loss / len(train_loader)
    pred_train_loss = pred_running_loss / len(train_loader)
    return train_loss, mask_train_loss, pred_train_loss


def validate(model, val_loader, criterion_mask, criterion_pred, epoch, device):
    model.eval()
    running_loss = 0.0
    running_accuracy = 0.0
    running_loss = 0.0
    mask_running_loss = 0.0
    pred_running_loss = 0.0
    progress_bar = tqdm(val_loader, desc=f'Validation Epoch {epoch}')
    for i, batch in enumerate(progress_bar):
        inputs, mask_labels, attributes, _ = batch
        attributes = attributes.to(device)
        # colour_labels = colour_labels.to(device)
        inputs, mask_labels = inputs.to(device), mask_labels.to(device)

        pred_masks, pred_classes = model(inputs)
        mask_loss = criterion_mask(pred_masks, mask_labels)
        # pred_loss, cl_loss, rg_loss = criterion_pred(pred_classes, classes, pred_colours)  #, colour_labels)
        pred_loss = criterion_pred(pred_classes, attributes)
        loss = mask_loss + pred_loss

        # Assume `predictions` and `classes` are your model's predictions and true class labels respectively
        predictions = pred_classes > 0.5
        # Create a mask for where the true class labels are 1 (positive class)
        positive_class_mask = (attributes == 1)
        # Select predictions and true labels where true labels are 1
        positive_predictions = predictions[positive_class_mask]
        positive_true_labels = attributes[positive_class_mask]
        # Calculate correct predictions for positive class
        correct_positives = (positive_predictions == positive_true_labels).float()
        # Calculate accuracy for positive class
        if correct_positives.numel() > 0:  # Check to make sure we have positive samples
            positive_accuracy = correct_positives.mean()
        else:
            positive_accuracy = torch.tensor(0.0)  # If no positive samples, set accuracy to 0
        # Now `positive_accuracy` will be the accuracy only for the class with label 1
        accuracy = positive_accuracy

        running_loss += loss.item()
        mask_running_loss += mask_loss.item()
        pred_running_loss += pred_loss.item()
        # cl_running_loss += cl_loss.item()
        # rg_running_loss += rg_loss.item()
        running_accuracy += accuracy
        progress_bar.set_description(f'Val E{epoch}:  ML:{mask_running_loss/(i+1):.3f} PL:{pred_running_loss/(i+1):.3f} Acc:{running_accuracy/(i+1):.2f}')

    val_loss = running_loss / len(val_loader)
    mask_val_loss = mask_running_loss / len(val_loader)
    pred_val_loss = pred_running_loss / len(val_loader)
    return val_loss, mask_val_loss, pred_val_loss


def test(model, test_loader, criterion_mask, criterion_pred, epoch, device):
    model.eval()
    running_loss = 0.0
    running_accuracy = 0.0
    running_loss = 0.0
    mask_running_loss = 0.0
    pred_running_loss = 0.0
    progress_bar = tqdm(test_loader, desc=f'Testing Epoch {epoch}')
    for i, batch in enumerate(progress_bar):
        inputs, mask_labels, attributes = batch
        attributes = attributes.to(device)
        # colour_labels = colour_labels.to(device)
        inputs, mask_labels = inputs.to(device), mask_labels.to(device)

        pred_masks, pred_classes = model(inputs)
        mask_loss = criterion_mask(pred_masks, mask_labels)
        # pred_loss, cl_loss, rg_loss = criterion_pred(pred_classes, classes, pred_colours)  #, colour_labels)
        pred_loss = criterion_pred(pred_classes, attributes)
        loss = mask_loss + pred_loss

        # Assume `predictions` and `classes` are your model's predictions and true class labels respectively
        predictions = pred_classes > 0.5
        # Create a mask for where the true class labels are 1 (positive class)
        positive_class_mask = (attributes == 1)
        # Select predictions and true labels where true labels are 1
        positive_predictions = predictions[positive_class_mask]
        positive_true_labels = attributes[positive_class_mask]
        # Calculate correct predictions for positive class
        correct_positives = (positive_predictions == positive_true_labels).float()
        # Calculate accuracy for positive class
        if correct_positives.numel() > 0:  # Check to make sure we have positive samples
            positive_accuracy = correct_positives.mean()
        else:
            positive_accuracy = torch.tensor(0.0)  # If no positive samples, set accuracy to 0
        # Now `positive_accuracy` will be the accuracy only for the class with label 1
        accuracy = positive_accuracy

        running_loss += loss.item()
        mask_running_loss += mask_loss.item()
        pred_running_loss += pred_loss.item()
        # cl_running_loss += cl_loss.item()
        # rg_running_loss += rg_loss.item()
        running_accuracy += accuracy
        progress_bar.set_description(f'Test E{epoch}: ML:{mask_running_loss/(i+1):.3f} PL:{pred_running_loss/(i+1):.3f} Acc:{running_accuracy/(i+1):.2f}')

    test_loss = running_loss / len(test_loader)
    mask_test_loss = mask_running_loss / len(test_loader)
    pred_test_loss = pred_running_loss / len(test_loader)
    # cl_test_loss = cl_running_loss / len(test_loader)
    # rg_test_loss = rg_running_loss / len(test_loader)
    return test_loss, mask_test_loss, pred_test_loss  # , rg_test_loss


if __name__ == "__main__":
    seed = 0  # random.randint(0, 1024)
    torch.manual_seed(seed)

    image_size = (192, 192)
    # datasets
    full_dataset = MergedCelebAMaskHQDataset(root_dir=r"/home/bentengma/work_space/CelebAMask-HQ", output_size=image_size)
    train_size = int(0.9 * len(full_dataset))
    val_size = int(0.09 * len(full_dataset))
    test_size = len(full_dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_size, val_size, test_size])  # [train_size, val_size, test_size]) [1, 1, len(full_dataset)-2])
    train_dataset = AugmentedDataset(
        dataset_source=train_dataset, output_size=image_size, 
        flip_prob=0.5, crop_ratio=(1, 1), scale_factor=(0.9, 1.1),
        noise_level=(0, 1), blur_radius=(0, 2), brightness_factor=(0.85, 1.25),
        seed=seed,
    )  # replace with augmented dataset
    val_dataset = AugmentedDataset(
        dataset_source=val_dataset, output_size=image_size, 
        flip_prob=0.5, crop_ratio=(1, 1), scale_factor=(1, 1),
        noise_level=(0, 1), blur_radius=(0, 1), brightness_factor=(0.85, 1.25), 
        seed=seed,
    )

    # lip_dataset = LIPDataset(root_dir=r"/home/bentengma/work_space/LIP")
    # lip_dataset = AugmentedDataset(
    #     dataset_source=lip_dataset, output_size=image_size,
    #     flip_prob=0.5, crop_ratio=(1, 1), scale_factor=(0.9, 1.1),
    #     noise_level=(0, 1), blur_radius=(0, 1), brightness_factor=(0.75, 1.25),
    #     seed=0, pil=True,
    # )
    # train_size = int(0.9 * len(lip_dataset))
    # val_size = len(lip_dataset) - train_size
    # lip_train_dataset, lip_val_dataset = random_split(lip_dataset, [train_size, val_size])
    # train_dataset = ConcatDataset([train_dataset, lip_train_dataset])
    # val_dataset = ConcatDataset([val_dataset, lip_val_dataset])

    # dataLoaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=24)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=16)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=16)

    from categories_and_attributes import CelebAMaskHQCategoriesAndAttributes
    # model
    cat_layers = CelebAMaskHQCategoriesAndAttributes.merged_categories.keys().__len__()
    # segment_model = DeepLabV3PlusMobileNetV3(num_classes=4)  # ['hair', 'hat', 'eye_g', 'skin', ] 'brow', 'eye', 'mouth', 'nose', ]
    segment_model = UNetWithResnet18Encoder(num_classes=cat_layers)
    # predict_model = MultiLabelMobileNetV3Large(4, 7)   # 'hair', 'hat', 'glasses', 'face', ; first three with colours, rgb
    predictions = len(CelebAMaskHQCategoriesAndAttributes.attributes) - len(CelebAMaskHQCategoriesAndAttributes.avoided_attributes) + len(CelebAMaskHQCategoriesAndAttributes.mask_labels)
    predict_model = MultiLabelResNet(num_labels=predictions, input_channels=cat_layers+3)
    model = CombinedModelNoRegression(segment_model, predict_model, cat_layers=cat_layers)

    # choose device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = 'cpu'
    model.to(device)

    # optimizer
    criterion_mask = nn.BCELoss()
    criterion_pred = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # TensorBoard writer
    writer = SummaryWriter('runs/24-2-14/freeze-half-32')

    # early stopping params
    early_stopping_patience = 5
    early_stopping_counter = 0

    # check model saving dir
    model_dir = "saved-models"
    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)

    # Check for the latest saved model
    latest_checkpoint = find_latest_checkpoint(model_dir)
    if latest_checkpoint:
        print(f"Loading model from {latest_checkpoint}")
        model, optimizer, start_epoch, best_val_loss = load_model(model, optimizer, path=latest_checkpoint)
        start_epoch += 1
    else:
        start_epoch = 0
        best_val_loss = float('inf')

    # train loop
    num_epochs = 60
    mode = 1
    for epoch in range(start_epoch, num_epochs):
        if epoch >= 30:
            mode = 0
        print(f'Epoch {epoch+1}/{num_epochs}, mode={mode}')
        print('-' * 10)

        # train, validate, test
        train_loss, mask_train_loss, pred_train_loss = train(model, optimizer, train_loader, criterion_mask, criterion_pred, epoch, device, mode=mode)
        val_loss, mask_val_loss, pred_val_loss = validate(model, val_loader, criterion_mask, criterion_pred, epoch, device)
        test_loss, mask_test_loss, pred_test_loss = test(model, test_loader, criterion_mask, criterion_pred, epoch, device)

        # write to TensorBoard
        writer.add_scalar('Loss/Train', train_loss, epoch)
        writer.add_scalar('Loss/Validation', val_loss, epoch)
        writer.add_scalar('Loss/Test', test_loss, epoch)
        writer.add_scalar('LossMask/Train', mask_train_loss, epoch)
        writer.add_scalar('LossMask/Validation', mask_val_loss, epoch)
        writer.add_scalar('LossMask/Test', mask_test_loss, epoch)
        writer.add_scalar('LossPred/Train', pred_train_loss, epoch)
        writer.add_scalar('LossPred/Validation', pred_val_loss, epoch)
        writer.add_scalar('LossPred/Test', pred_test_loss, epoch)

        # save the model
        if mode == 0:
            if val_loss < best_val_loss:
                print(f"Validation loss decreased ({best_val_loss:.6f} --> {val_loss:.6f}).  Saving model ...")
                save_model(epoch, model, optimizer, val_loss, path=f"{model_dir}/model_epoch_{epoch}.pth")
                best_val_loss = val_loss
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1
                print(f"Validation loss did not improve. EarlyStopping counter: {early_stopping_counter}/{early_stopping_patience}")
            
            if early_stopping_counter >= early_stopping_patience:
                print("Early stopping")
                break
        else:
            save_model(epoch, model, optimizer, val_loss, path=f"{model_dir}/model_epoch_{epoch}.pth")
            best_val_loss = val_loss
            early_stopping_counter = 0

    writer.close()
