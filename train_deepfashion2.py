if __name__ == "__main__":
    import torch.optim as optim
    from torch.utils.data import Subset, DataLoader  # random_split
    from torch.utils.tensorboard import SummaryWriter

    from models import *
    from datasets import *
    from train import *

    seed = 0  # random.randint(0, 1024)
    torch.manual_seed(seed)

    image_size = (256, 192)
    scale_range = (0.5, 1.0)
    # datasets
    train_dataset = DeepFashion2Dataset(
        image_dir='/home/bentengma/work_space/deepfashion2/train/image',
        anno_dir='/home/bentengma/work_space/deepfashion2/train/annos',
        output_size=image_size,
    )
    val_dataset = DeepFashion2Dataset(
        image_dir='/home/bentengma/work_space/deepfashion2/validation/image',
        anno_dir='/home/bentengma/work_space/deepfashion2/validation/annos',
        output_size=image_size,
    )
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
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=16)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=16)

    # model
    segment_model = UNetWithResnetEncoder(num_classes=len(DeepFashion2Dataset.categories))
    predict_model = MultiLabelResNet(num_labels=len(DeepFashion2Dataset.categories), input_channels=len(DeepFashion2Dataset.categories) + 3)
    model = CombinedModel(segment_model, predict_model, cat_layers=len(DeepFashion2Dataset.categories))

    # choose device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = 'cpu'
    model.to(device)

    # optimizer
    criterion_mask = nn.BCELoss()
    criterion_pred = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # TensorBoard writer
    writer = SummaryWriter('runs/24-4-16/deepfashion2')

    # early stopping params
    early_stopping_patience = 5
    early_stopping_counter = 0

    # check model saving dir
    model_dir = "saved-models-deepfashion2"
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
    mode = 'mix'
    for epoch in range(start_epoch, num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}, mode={mode}')
        print('-' * 10)

        # train, validate, test
        train_loss, mask_train_loss, pred_train_loss = train(model, optimizer, train_loader, criterion_mask,
                                                             criterion_pred, scale_range, epoch, device, mode=mode)
        val_loss, mask_val_loss, pred_val_loss = validate(model, val_loader, criterion_mask, criterion_pred, epoch,
                                                          device)

        # write to TensorBoard
        writer.add_scalar('Loss/Train', train_loss, epoch)
        writer.add_scalar('Loss/Validation', val_loss, epoch)
        writer.add_scalar('LossMask/Train', mask_train_loss, epoch)
        writer.add_scalar('LossMask/Validation', mask_val_loss, epoch)
        writer.add_scalar('LossPred/Train', pred_train_loss, epoch)
        writer.add_scalar('LossPred/Validation', pred_val_loss, epoch)

        # save the model
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

    writer.close()
