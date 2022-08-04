import torch
import torch.nn as nn

import framework.configbase
import framework.ops

def cosine_sim(im, s):
  '''cosine similarity between all the image and sentence pairs
  '''
  inner_prod = im.mm(s.t())
  im_norm = torch.sqrt((im**2).sum(1).view(-1, 1) + 1e-18)
  s_norm = torch.sqrt((s**2).sum(1).view(1, -1) + 1e-18)
  sim = inner_prod / (im_norm * s_norm)
  return sim

class ContrastiveLoss(nn.Module):
  '''compute contrastive loss
  '''
  def __init__(self, margin=0, max_violation=False, direction='bi', topk=1):
    '''Args:
      direction: i2t for negative sentence, t2i for negative image, bi for both
    '''
    super(ContrastiveLoss, self).__init__()
    self.margin = margin
    self.max_violation = max_violation
    self.direction = direction
    self.topk = topk

  def forward(self, scores, margin=None, average_batch=True, batch_relevance=None, threshold_pos=1.):
    '''
    Args:
      scores: image-sentence score matrix, (batch, batch)
        the same row of im and s are positive pairs, different rows are negative pairs

      batch_relevance: image-sentence relevancy matrix (batch, batch)
    '''

    if margin is None:
      margin = self.margin

    batch_size = scores.size(0)
    diagonal = scores.diag().view(batch_size, 1) # positive pairs

    # mask to clear diagonals which are positive pairs
    pos_masks = torch.eye(batch_size).bool().to(scores.device)

    batch_topk = min(batch_size, self.topk)
    if self.direction == 'i2t' or self.direction == 'bi':
      d1 = diagonal.expand_as(scores) # same collumn for im2s (negative sentence)
      # compare every diagonal score to scores in its collumn
      # caption retrieval
      cost_s = (margin + scores - d1).clamp(min=0)
      cost_s = cost_s.masked_fill(pos_masks, 0)

      if batch_relevance is not None:
        cost_s[batch_relevance >= threshold_pos] = 0

      if self.max_violation:
        cost_s, _ = torch.topk(cost_s, batch_topk, dim=1)
        cost_s = cost_s / batch_topk
        if average_batch:
          cost_s = cost_s / batch_size
      else:
        if average_batch:
          cost_s = cost_s / (batch_size * (batch_size - 1))
      cost_s = torch.sum(cost_s)

    if self.direction == 't2i' or self.direction == 'bi':
      d2 = diagonal.t().expand_as(scores) # same row for s2im (negative image)
      # compare every diagonal score to scores in its row
      cost_im = (margin + scores - d2).clamp(min=0)
      cost_im = cost_im.masked_fill(pos_masks, 0)

      if batch_relevance is not None:
        cost_im[batch_relevance >= threshold_pos] = 0

      if self.max_violation:
        cost_im, _ = torch.topk(cost_im, batch_topk, dim=0)
        cost_im = cost_im / batch_topk
        if average_batch:
          cost_im = cost_im / batch_size
      else:
        if average_batch:
          cost_im = cost_im / (batch_size * (batch_size - 1))
      cost_im = torch.sum(cost_im)

    if self.direction == 'i2t':
      return cost_s
    elif self.direction == 't2i':
      return cost_im
    else:
      return cost_s + cost_im

class ContrastiveLossHP(nn.Module):
  '''compute contrastive loss
  '''
  def __init__(self, margin=0, margin_pos=0, max_violation=False, direction='bi', topk=1):
    '''Args:
      direction: i2t for negative sentence, t2i for negative image, bi for both
    '''
    super(ContrastiveLossHP, self).__init__()
    self.margin = margin
    self.margin_pos = margin_pos
    self.max_violation = max_violation
    self.direction = direction
    self.topk = topk

  def forward(self, scores, margin=None, average_batch=True, batch_relevance=None, threshold_pos=1., margin_pos=None):
    '''
    Args:
      scores: image-sentence score matrix, (batch, batch)
        the same row of im and s are positive pairs, different rows are negative pairs

      batch_relevance: image-sentence relevancy matrix (batch, batch)
    '''

    if margin is None:
      margin = self.margin
    if margin_pos is None:
      margin_pos = self.margin_pos

    batch_size = scores.size(0)
    diagonal = scores.diag().view(batch_size, 1) # positive pairs

    # mask to clear diagonals which are positive pairs
    pos_masks = torch.eye(batch_size).bool().to(scores.device)

    batch_topk = min(batch_size, self.topk)
    if self.direction == 'i2t' or self.direction == 'bi':
      d1 = diagonal.expand_as(scores) # same collumn for im2s (negative sentence)
      # compare every diagonal score to scores in its collumn
      # caption retrieval
      cost_s = (margin + scores - d1).clamp(min=0)
      cost_s = cost_s.masked_fill(pos_masks, 0)

      if batch_relevance is not None:
        cost_s[batch_relevance >= threshold_pos] = 0

      if self.max_violation:
        cost_s, _ = torch.topk(cost_s, batch_topk, dim=1)
        cost_s = cost_s / batch_topk
        if average_batch:
          cost_s = cost_s / batch_size
      else:
        if average_batch:
          cost_s = cost_s / (batch_size * (batch_size - 1))
      cost_s = torch.sum(cost_s)

      # we want a copy of scores in order to compute the negative-masked version as well
      hp_scores = scores.clone()  # scores are the (v_i, q_j)
      hp_scores[batch_relevance < threshold_pos] = 1  # mask the negatives; positives have now score <= 1
      hp_scores = hp_scores.masked_fill(pos_masks, 1)
      # we want to pick the argmin (hardest positive)
      hp_scores, _ = torch.topk(hp_scores, batch_topk, dim=1, largest=False)
      # -> s(q, v+)
      hp_cost_s = (margin_pos + scores - hp_scores).clamp(min=0)
      # we need to mask hp_cost again because 'scores' is not pos-masked
      hp_cost_s = hp_cost_s.masked_fill(pos_masks, 0)
      if batch_relevance is not None:
        hp_cost_s[batch_relevance >= threshold_pos] = 0
      hp_cost_s, _ = torch.topk(hp_cost_s, batch_topk, dim=1)

      hp_cost_s = hp_cost_s / batch_topk
      if average_batch:
        hp_cost_s = hp_cost_s / batch_size
      hp_cost_s = torch.sum(hp_cost_s)

    if self.direction == 't2i' or self.direction == 'bi':
      d2 = diagonal.t().expand_as(scores) # same row for s2im (negative image)
      # compare every diagonal score to scores in its row
      cost_im = (margin + scores - d2).clamp(min=0)
      cost_im = cost_im.masked_fill(pos_masks, 0)

      if batch_relevance is not None:
        cost_im[batch_relevance >= threshold_pos] = 0

      if self.max_violation:
        cost_im, _ = torch.topk(cost_im, batch_topk, dim=0)
        cost_im = cost_im / batch_topk
        if average_batch:
          cost_im = cost_im / batch_size
      else:
        if average_batch:
          cost_im = cost_im / (batch_size * (batch_size - 1))
      cost_im = torch.sum(cost_im)

      # we want a copy of scores in order to compute the negative-masked version as well
      hp_scores_im = scores.clone()  # scores are the (v_i, q_j)
      hp_scores_im[batch_relevance < threshold_pos] = 1  # mask the negatives; positives have now score <= 1
      hp_scores_im = hp_scores_im.masked_fill(pos_masks, 1)
      # we want to pick the argmin (hardest positive)
      hp_scores_im, _ = torch.topk(hp_scores_im, batch_topk, dim=0, largest=False)
      # -> s(q, v+)
      hp_cost_im = (margin_pos + scores - hp_scores_im).clamp(min=0)
      # we need to mask hp_cost again because 'scores' is not pos-masked
      hp_cost_im = hp_cost_im.masked_fill(pos_masks, 0)
      if batch_relevance is not None:
        hp_cost_im[batch_relevance >= threshold_pos] = 0
      hp_cost_im, _ = torch.topk(hp_cost_im, batch_topk, dim=0)

      hp_cost_im = hp_cost_im / batch_topk
      if average_batch:
        hp_cost_im = hp_cost_im / batch_size
      hp_cost_im = torch.sum(hp_cost_im)

    if self.direction == 'i2t':
      return cost_s, hp_cost_s
    elif self.direction == 't2i':
      return cost_im, hp_cost_im
    else:
      return cost_s + cost_im, hp_cost_s + hp_cost_im
