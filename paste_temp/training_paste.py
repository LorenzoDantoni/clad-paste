def train_model(
    model: Stfpm,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    device: torch.device,
    category: str,
    model_save_path: str,
    log_dirpath=None,
    seed=None,
    early_stopping: Union[float, bool] = False,
):
    """
    Train the student-teacher feature-pyramid model and save checkpoints
    for each category.

    Args:
        model: stfpm model
        train_loader: torch dataloader for the training dataset
        val_loader: torch dataloader for the validation dataset
        epochs: number of epochs to train the model
        device: where to run the model
        category: name of the mvtec category where to save the model
        model_save_path: directory where to create the category subdirectory and save the model
        log_dirpath: directory where to save the training logs
        seed: seed for reproducibility
        early_stopping: if a float is provided, the training will stop if the validation
            loss difference between the current and the previous epoch is less than the
            provided value.
    """
    model.seed = seed
    model.epochs = epochs
    model.category = category

    if model.seed is not None:
        torch.manual_seed(model.seed)

    min_err = 10000
    prev_val_loss = 100000

    if "micronet" in model.student.model_name:
        # optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        # optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        optimizer = torch.optim.SGD(
            model.student.parameters(), 0.04, momentum=0.9, weight_decay=1e-4
        )
    else:
        optimizer = torch.optim.SGD(
            model.student.parameters(), 0.4, momentum=0.9, weight_decay=1e-4
        )

    # simple loss function of STFPM
    def loss_fn(t_feat, s_feat):
        return torch.sum((t_feat - s_feat) ** 2, 1).mean()

    logs = []
    for epoch in trange(epochs, desc="Train stfpm"):
        model.train()
        mean_loss = 0

        # train the model
        for batch_data in train_loader:
            # the original loader returns a tuple of two lists, one contains the paths
            # and the other the images
            _, batch_img = batch_data
            batch_img = batch_img.to(device)

            t_feat, s_feat = model(batch_img)

            loss = loss_fn(t_feat[0], s_feat[0])
            for i in range(1, len(t_feat)):
                t_feat[i] = F.normalize(t_feat[i], dim=1)
                s_feat[i] = F.normalize(s_feat[i], dim=1)
                loss += loss_fn(t_feat[i], s_feat[i])

            print("[%d/%d] loss: %f" % (epoch, epochs, loss.item()))
            mean_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        mean_loss /= len(train_loader)

        # evaluate the model
        with torch.no_grad():
            model.eval()
            val_loss = torch.zeros(1, device=device)
            for _, batch_imgs in val_loader:
                anomaly_maps, _ = model(batch_imgs.to(device))
                val_loss += anomaly_maps.mean()
            val_loss /= len(val_loader)

        log_dict = {
            "epochs": epoch,
            "val_loss": val_loss.cpu(),
            "train_loss": mean_loss,
        }
        logs.append(log_dict)

        # save best checkpoint
        if val_loss < min_err:
            min_err = val_loss

            model_filename = model.model_filename()
            save_path = os.path.join(model_save_path, model.category, model_filename)
            dir_name = os.path.dirname(save_path)
            if dir_name and not os.path.exists(dir_name):
                os.makedirs(dir_name)
            torch.save(model.state_dict(), save_path)

        if early_stopping not in [False, None] and epoch > 0:
            if np.abs(val_loss.cpu() - prev_val_loss) < early_stopping:
                print(f"Early stopping at epoch {epoch+1}/{epochs}")
                break
        prev_val_loss = val_loss.cpu()

    logs_df = pd.DataFrame(logs)
    if log_dirpath is not None:
        assert model.category is not None
        logs_path = os.path.join(
            log_dirpath,
            model.category,
            "train_logs",
            model.model_filename() + "_train_logs.csv",
        )
        dirpath = os.path.dirname(logs_path)
        if not os.path.exists(dirpath):
            os.makedirs(dirpath)
        logs_df.to_csv(logs_path, index=False)

    return logs_df, save_path
