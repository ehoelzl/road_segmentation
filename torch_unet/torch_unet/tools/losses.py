import torch
from torch.autograd import Function


class DiceCoeff(Function):
    """Dice coeff for individual examples"""
    
    def forward(self, input, target):
        self.save_for_backward(input, target)
        eps = 0.0001
        self.inter = torch.dot(input.view(-1), target.view(-1))
        self.union = torch.sum(input) + torch.sum(target) + eps
        
        t = (2 * self.inter.float() + eps) / self.union.float()
        return t
    
    # This function has only a single output, so it gets only one gradient
    def backward(self, grad_output):
        
        input, target = self.saved_variables
        grad_input = grad_target = None
        
        if self.needs_input_grad[0]:
            grad_input = grad_output * 2 * (target * self.union - self.inter) \
                         / (self.union * self.union)
        if self.needs_input_grad[1]:
            grad_target = None
        
        return grad_input, grad_target


def f1_score(y_true, y_pred, threshold, eps=1e-9):
    y_pred = torch.ge(y_pred.float(), threshold).float()
    y_true = y_true.float()
    
    true_positive = (y_pred * y_true).sum(dim=1)
    precision = true_positive.div(y_pred.sum(dim=1).add(eps))
    recall = true_positive.div(y_true.sum(dim=1).add(eps))
    
    return torch.mean((precision * recall).div(precision + recall + eps).mul(2))


def dice_loss(input, target):
    smooth = 1.
    
    iflat = input.view(-1)
    tflat = target.view(-1)
    intersection = (iflat * tflat).sum()
    
    return 1 - ((2. * intersection + smooth) /
                (iflat.sum() + tflat.sum() + smooth))


def dice_loss_withlogits(logits, true, eps=1e-7):
    """Computes the Sørensen–Dice loss.
    Note that PyTorch optimizers minimize a loss. In this
    case, we would like to maximize the dice loss so we
    return the negated dice loss.
    Args:
        true: a tensor of shape [B, 1, H, W].
        logits: a tensor of shape [B, C, H, W]. Corresponds to
            the raw output or logits of the model.
        eps: added to the denominator for numerical stability.
    Returns:
        dice_loss: the Sørensen–Dice loss.
    """

    num_classes = logits.shape[1]
    true_1_hot = torch.eye(num_classes + 1)[true.type(torch.int64).squeeze(1)]
    true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
    true_1_hot_f = true_1_hot[:, 0:1, :, :]
    true_1_hot_s = true_1_hot[:, 1:2, :, :]
    true_1_hot = torch.cat([true_1_hot_s, true_1_hot_f], dim=1)
    pos_prob = torch.sigmoid(logits)
    neg_prob = 1 - pos_prob
    probas = torch.cat([pos_prob, neg_prob], dim=1)
    
    true_1_hot = true_1_hot.type(logits.type())
    dims = (0,) + tuple(range(2, true.ndimension()))
    intersection = torch.sum(probas * true_1_hot, dims)
    cardinality = torch.sum(probas + true_1_hot, dims)
    dice_loss = (2. * intersection / (cardinality + eps)).mean()
    return 1 - dice_loss


def dice_coeff(input, target):
    """Dice coeff for batches"""
    if input.is_cuda:
        s = torch.FloatTensor(1).cuda().zero_()
    else:
        s = torch.FloatTensor(1).zero_()
    
    for i, c in enumerate(zip(input, target)):
        s = s + DiceCoeff().forward(c[0], c[1])
    
    return s / (i + 1)
