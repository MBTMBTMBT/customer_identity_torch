if __name__ == "__main__":
    import torch.optim as optim
    from torch.utils.data import Subset, DataLoader  # random_split
    from torch.utils.tensorboard import SummaryWriter

    from models import *
    from datasets import *
    from train import *

    seed = 0  # random.randint(0, 1024)
    torch.manual_seed(seed)

    image_size = (256, 256)
    # datasets
    full_dataset = MergedCelebAMaskHQDataset(root_dir=r"/home/bentengma/work_space/CelebAMask-HQ",
                                             output_size=image_size, replay=1)
    train_size = int(0.9 * len(full_dataset))
    val_size = int(0.09 * len(full_dataset))
    test_size = len(full_dataset) - train_size - val_size
    # Create indices for training, validation, and test sets
    train_indices = list(range(0, train_size))
    val_indices = list(range(train_size, train_size + val_size))
    test_indices = list(range(train_size + val_size, len(full_dataset)))
    # Create non-random subsets
    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)
    test_dataset = Subset(full_dataset, test_indices)
    # train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_size, val_size, test_size], seed=0)  # [train_size, val_size, test_size]) [1, 1, len(full_dataset)-2])
    train_dataset = AugmentedDataset(
        dataset_source=train_dataset, output_size=image_size,
        flip_prob=0.5, crop_ratio=(1, 1), scale_factor=(0.9, 1.1),
        noise_level=(0, 1), blur_radius=(0, 2), brightness_factor=(0.85, 1.25),
        seed=seed,
    )  # replace with augmented dataset
    val_dataset = AugmentedDataset(
        dataset_source=val_dataset, output_size=image_size,
        flip_prob=0.5, crop_ratio=(1, 1), scale_factor=(1, 1),
        noise_level=(0, 0), blur_radius=(0, 0), brightness_factor=(1, 1),
        seed=seed,
    )

    # dataLoaders
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=24)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=24)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=24)

    from categories_and_attributes import CelebAMaskHQCategoriesAndAttributes

    # model
    cat_layers = CelebAMaskHQCategoriesAndAttributes.merged_categories.keys().__len__()
    # segment_model = DeepLabV3PlusMobileNetV3(num_classes=4)  # ['hair', 'hat', 'eye_g', 'skin', ] 'brow', 'eye', 'mouth', 'nose', ]
    segment_model = UNetWithResnetEncoder(num_classes=cat_layers)
    # predict_model = MultiLabelMobileNetV3Large(4, 7)   # 'hair', 'hat', 'glasses', 'face', ; first three with colours, rgb
    predictions = len(CelebAMaskHQCategoriesAndAttributes.attributes) - len(
        CelebAMaskHQCategoriesAndAttributes.avoided_attributes) + len(CelebAMaskHQCategoriesAndAttributes.mask_labels)
    predict_model = MultiLabelResNet(num_labels=predictions, input_channels=cat_layers + 3)
    model = CombinedModel(segment_model, predict_model, cat_layers=cat_layers)

    # choose device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = 'cpu'
    model.to(device)

    # optimizer
    criterion_mask = nn.BCELoss()
    criterion_pred = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # TensorBoard writer
    writer = SummaryWriter('runs/24-2-25/freeze-half-8')

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
        if epoch >= 15:
            mode = 2
        else:
            mode = 1
        print(f'Epoch {epoch + 1}/{num_epochs}, mode={mode}')
        print('-' * 10)

        # train, validate, test
        train_loss, mask_train_loss, pred_train_loss = train_CelebA(model, optimizer, train_loader, criterion_mask,
                                                             criterion_pred, epoch, device, mode=mode)
        val_loss, mask_val_loss, pred_val_loss = validate_CelebA(model, val_loader, criterion_mask, criterion_pred, epoch,
                                                          device)
        test_loss, mask_test_loss, pred_test_loss = test_CelebA(model, test_loader, criterion_mask, criterion_pred, epoch,
                                                         device)

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
        if mode != 1:
            if val_loss < best_val_loss:
                print(f"Validation loss decreased ({best_val_loss:.6f} --> {val_loss:.6f}).  Saving model ...")
                save_model(epoch, model, optimizer, val_loss, path=f"{model_dir}/model_epoch_{epoch}.pth")
                best_val_loss = val_loss
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1
                print(
                    f"Validation loss did not improve. EarlyStopping counter: {early_stopping_counter}/{early_stopping_patience}")

            if early_stopping_counter >= early_stopping_patience:
                print("Early stopping")
                break
        else:
            save_model(epoch, model, optimizer, val_loss, path=f"{model_dir}/model_epoch_{epoch}.pth")
            best_val_loss = val_loss
            early_stopping_counter = 0

    writer.close()
