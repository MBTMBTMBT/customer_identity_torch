if __name__ == "__main__":
    import torch.optim as optim
    from torch.utils.tensorboard import SummaryWriter

    from models import *
    from datasets import *
    from train import *

    seed = None  # random.randint(0, 1024)
    # torch.manual_seed(seed)

    image_size = (192, 256)
    scale_range = (0.5, 1.0)
    # datasets
    train_dataset = DeepFashion2Dataset(
        image_dir='../deepfashion2/train/image',
        anno_dir='../deepfashion2/train/annos',
        output_size=image_size,
        return_bbox=False,
    )
    val_dataset = DeepFashion2Dataset(
        image_dir='../deepfashion2/validation/image',
        anno_dir='../deepfashion2/validation/annos',
        output_size=image_size,
        return_bbox=False,
    )
    # train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_size, val_size, test_size], seed=0)  # [train_size, val_size, test_size]) [1, 1, len(full_dataset)-2])
    train_dataset = AugmentedDeepFashion2Dataset(
        dataset_source=train_dataset, output_size=image_size,
        flip_prob=0.5, crop_ratio=(1, 1), scale_factor=(0.9, 1.1),
        noise_level=(0, 1), blur_radius=(0, 2), brightness_factor=(0.85, 1.25),
        seed=seed,
    )
    # replace with augmented dataset
    # val_dataset = AugmentedDataset(
    #     dataset_source=val_dataset, output_size=image_size,
    #     flip_prob=0.5, crop_ratio=(1, 1), scale_factor=(1, 1),
    #     noise_level=(0, 0), blur_radius=(0, 0), brightness_factor=(1, 1),
    #     seed=seed,
    # )

    # dataLoaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=8, )
    # collate_fn=collate_fn_DeepFashion2)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=8, )
    # collate_fn=collate_fn_DeepFashion2)

    # choose device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # model
    num_classes = len(DeepFashion2Dataset.categories)
    model = SegmentPredictor(num_masks=num_classes, num_labels=num_classes, )
    model.to(device)

    # optimizer
    # optimizer
    criterion_mask = nn.BCELoss()
    criterion_pred = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)


    # early stopping params
    early_stopping_patience = 5
    early_stopping_counter = 0

    # check model saving dir
    model_dir = "deepfashion2-segpred"
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

    # TensorBoard writer
    writer = SummaryWriter(model_dir)

    # train loop
    num_epochs = 60
    for epoch in range(start_epoch, num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}')
        print('-' * 10)

        train_loss, mask_train_loss, pred_train_loss, train_acc, counter = train(model, optimizer, train_loader,
                                                                                 criterion_mask, criterion_pred,
                                                                                 scale_range, epoch, device, mode='mix',
                                                                                 tb_writer=writer, counter=counter)
        val_loss, mask_val_loss, pred_val_loss, val_acc = validate(model, val_loader, criterion_mask, criterion_pred,
                                                                   epoch, device)

        # write to TensorBoard
        # writer.add_scalar('Loss/Train', train_loss, epoch)
        writer.add_scalar('Loss/Validation', val_loss, epoch)
        # writer.add_scalar('LossMask/Train', mask_train_loss, epoch)
        writer.add_scalar('LossMask/Validation', mask_val_loss, epoch)
        # writer.add_scalar('LossPred/Train', pred_train_loss, epoch)
        writer.add_scalar('LossPred/Validation', pred_val_loss, epoch)
        writer.add_scalar('Accuracy/Validation', val_acc, epoch)

        # train, validate, test
        # train_loss, mask_train_loss, pred_train_loss, avrg_mAP, avrg_f1, avrg_iou, counter = train_DeepFashion2(model, optimizer, train_loader, scale_range, epoch, device, tb_writer=writer, counter=counter)
        # val_loss, mask_val_loss, pred_val_loss, val_avrg_mAP, val_avrg_f1, val_avrg_iou = val_DeepFashion2(model, val_loader, (1, 1), epoch, device)

        # # write to TensorBoard
        # val_acc = 0.333 * val_avrg_mAP + 0.333 * val_avrg_f1 + 0.333 * val_avrg_iou
        # writer.add_scalar('Loss/Validation', val_loss, epoch)
        # writer.add_scalar('LossMask/Validation', mask_val_loss, epoch)
        # writer.add_scalar('LossPred/Validation', pred_val_loss, epoch)
        # writer.add_scalar('Accuracy/Validation', val_acc, epoch)
        # writer.add_scalar('MAP/Validation', val_avrg_mAP, epoch)
        # writer.add_scalar('F1/Validation', val_avrg_f1, epoch)
        # writer.add_scalar('IOU/Validation', val_avrg_iou, epoch)

        # save the model
        if val_acc >= best_acc:
            print(f"Accuracy increased ({best_acc:.2f} --> {val_acc:.2f}).  Saving model ...")
            save_model(epoch, model, optimizer, val_acc, path=f"{model_dir}/model_epoch_{epoch}.pth", counter=counter)
            best_acc = val_acc
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
            print(
                f"Validation loss did not improve. EarlyStopping counter: {early_stopping_counter}/{early_stopping_patience}")

        if early_stopping_counter >= early_stopping_patience:
            print("Early stopping")
            break

    writer.close()
