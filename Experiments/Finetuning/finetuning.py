# -----------------------------
# Train / Evaluate
# -----------------------------
def adjust_learning_rate(optimizer, epoch, args):
    """
    Cosine LR schedule with warmup (per MAE fine-tuning setting).
    """
    warmup_epochs = args.warmup_epochs
    if epoch < warmup_epochs:
        lr = args.lr * float(epoch + 1) / float(warmup_epochs)
    else:
        # cosine decay after warmup
        t = (epoch - warmup_epochs) / float(max(1, args.epochs - warmup_epochs))
        lr = args.lr * 0.5 * (1.0 + torch.cos(torch.tensor(t * 3.1415926535)))

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr.item() if isinstance(lr, torch.Tensor) else lr


def train_one_epoch(model, criterion, optimizer, data_loader, device, epoch, args):
    model.train()
    criterion.train()

    running_loss = 0.0
    running_correct = 0
    running_total = 0

    t0 = time.time()
    for i, (images, targets) in enumerate(data_loader):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        outputs = model(images)  # full end-to-end ViT
        loss = criterion(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        _, preds = torch.max(outputs, dim=1)
        running_loss += loss.item() * images.size(0)
        running_correct += torch.sum(preds == targets).item()
        running_total += images.size(0)

        if is_main_process(args) and (i + 1) % args.log_freq == 0:
            avg_loss = running_loss / running_total
            avg_acc = 100.0 * running_correct / running_total
            print(
                f"Epoch [{epoch}][{i+1}/{len(data_loader)}] "
                f"Loss: {avg_loss:.4f}  Acc: {avg_acc:.2f}%"
            )

    epoch_loss = running_loss / running_total
    epoch_acc = 100.0 * running_correct / running_total
    if is_main_process(args):
        print(
            f"Train epoch {epoch}: "
            f"Loss {epoch_loss:.4f}, Acc {epoch_acc:.2f}%, "
            f"time {time.time() - t0:.1f}s"
        )
    return epoch_loss, epoch_acc


@torch.no_grad()
def evaluate(model, criterion, data_loader, device, args, epoch=None):
    model.eval()
    criterion.eval()

    running_loss = 0.0
    running_correct = 0
    running_total = 0

    for images, targets in data_loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        outputs = model(images)
        loss = criterion(outputs, targets)

        _, preds = torch.max(outputs, dim=1)
        running_loss += loss.item() * images.size(0)
        running_correct += torch.sum(preds == targets).item()
        running_total += images.size(0)

    epoch_loss = running_loss / running_total
    epoch_acc = 100.0 * running_correct / running_total
    if is_main_process(args):
        if epoch is not None:
            print(
                f"Val epoch {epoch}: Loss {epoch_loss:.4f}, "
                f"Acc {epoch_acc:.2f}%"
            )
        else:
            print(
                f"Val: Loss {epoch_loss:.4f}, "
                f"Acc {epoch_acc:.2f}%"
            )
    return epoch_loss, epoch_acc


