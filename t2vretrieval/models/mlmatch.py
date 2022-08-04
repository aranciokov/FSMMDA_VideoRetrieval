import numpy as np
import torch

import framework.ops

import t2vretrieval.encoders.mlsent
import t2vretrieval.encoders.mlvideo

import t2vretrieval.models.globalmatch
from t2vretrieval.models.criterion import cosine_sim

from t2vretrieval.models.globalmatch import VISENC, TXTENC

class RoleGraphMatchModelConfig(t2vretrieval.models.globalmatch.GlobalMatchModelConfig):
  def __init__(self):
    super().__init__()
    self.num_verbs = 4
    self.num_nouns = 6
    
    self.attn_fusion = 'embed' # sim, embed
    self.simattn_sigma = 4

    self.hard_topk = 1
    self.max_violation = True

    self.loss_weights = None

    self.subcfgs[VISENC] = t2vretrieval.encoders.mlvideo.MultilevelEncoderConfig()
    self.subcfgs[TXTENC] = t2vretrieval.encoders.mlsent.RoleGraphEncoderConfig()


class RoleGraphMatchModel(t2vretrieval.models.globalmatch.GlobalMatchModel):
  def build_submods(self):
    return {
      VISENC: t2vretrieval.encoders.mlvideo.MultilevelEncoder(self.config.subcfgs[VISENC]),
      TXTENC: t2vretrieval.encoders.mlsent.RoleGraphEncoder(self.config.subcfgs[TXTENC])
    }

  def print_comp_graph(self, _loss):
    import torchviz
    print("Computing and rendering the computational graph")
    render_params = dict(self.submods[VISENC].named_parameters(),
                         **dict(self.submods[TXTENC].named_parameters()))
    dot = torchviz.make_dot(_loss, params=render_params)
    dot.render()
    exit(0)

  def forward_video_embed(self, batch_data):
    vid_fts = torch.FloatTensor(batch_data['attn_fts']).to(self.device)
    vid_lens = torch.LongTensor(batch_data['attn_lens']).to(self.device)
    # (batch, max_vis_len, dim_embed)
    vid_sent_embeds, vid_verb_embeds, vid_noun_embeds = self.submods[VISENC](vid_fts, vid_lens)
    return {
      'vid_sent_embeds': vid_sent_embeds,
      'vid_verb_embeds': vid_verb_embeds, 
      'vid_noun_embeds': vid_noun_embeds,
      'vid_lens': vid_lens,
    }

  def forward_text_embed(self, batch_data):
    sent_ids = torch.LongTensor(batch_data['sent_ids']).to(self.device)
    sent_lens = torch.LongTensor(batch_data['sent_lens']).to(self.device)
    verb_masks = torch.BoolTensor(batch_data['verb_masks']).to(self.device)
    noun_masks = torch.BoolTensor(batch_data['noun_masks']).to(self.device)
    node_roles = torch.LongTensor(batch_data['node_roles']).to(self.device)
    rel_edges = torch.FloatTensor(batch_data['rel_edges']).to(self.device)
    verb_lens = torch.sum(verb_masks, 2)
    noun_lens = torch.sum(noun_masks, 2)
    # sent_embeds: (batch, dim_embed)
    # verb_embeds, noun_embeds: (batch, num_xxx, dim_embed)
    sent_embeds, verb_embeds, noun_embeds = self.submods[TXTENC](
      sent_ids, sent_lens, verb_masks, noun_masks, node_roles, rel_edges)
    return {
      'sent_embeds': sent_embeds, 'sent_lens': sent_lens,
      'verb_embeds': verb_embeds, 'verb_lens': verb_lens,
      'noun_embeds': noun_embeds, 'noun_lens': noun_lens,
      }

  def generate_phrase_scores(self, vid_embeds, vid_masks, phrase_embeds, phrase_masks):
    '''Args:
      - vid_embeds: (batch, num_frames, embed_size)
      - vid_masks: (batch, num_frames)
      - phrase_embeds: (batch, num_phrases, embed_size)
      - phrase_masks: (batch, num_phrases)
    '''
    batch_vids, num_frames, _ = vid_embeds.size()
    vid_pad_masks = (vid_masks == 0).unsqueeze(1).unsqueeze(3)
    batch_phrases, num_phrases, dim_embed = phrase_embeds.size()

    # compute component-wise similarity
    vid_2d_embeds = vid_embeds.view(-1, dim_embed)
    phrase_2d_embeds = phrase_embeds.view(-1, dim_embed)
    # size = (batch_vids, batch_phrases, num_frames, num_phrases)
    ground_sims = cosine_sim(vid_2d_embeds, phrase_2d_embeds).view(
      batch_vids, num_frames, batch_phrases, num_phrases).transpose(1, 2)

    vid_attn_per_word = ground_sims.masked_fill(vid_pad_masks, 0)
    vid_attn_per_word[vid_attn_per_word < 0] = 0
    vid_attn_per_word = framework.ops.l2norm(vid_attn_per_word, dim=2)
    vid_attn_per_word = vid_attn_per_word.masked_fill(vid_pad_masks, -1e18)
    vid_attn_per_word = torch.softmax(self.config.simattn_sigma * vid_attn_per_word, dim=2)
    
    if self.config.attn_fusion == 'embed':
      vid_attned_embeds = torch.einsum('abcd,ace->abde', vid_attn_per_word, vid_embeds)
      word_attn_sims = torch.einsum('abde,bde->abd',
        framework.ops.l2norm(vid_attned_embeds),
        framework.ops.l2norm(phrase_embeds))
    elif self.config.attn_fusion == 'sim':
      # (batch_vids, batch_phrases, num_phrases)
      word_attn_sims = torch.sum(ground_sims * vid_attn_per_word, dim=2) 

    # sum: (batch_vid, batch_phrases)
    phrase_scores = torch.sum(word_attn_sims * phrase_masks.float().unsqueeze(0), 2) \
                   / torch.sum(phrase_masks, 1).float().unsqueeze(0).clamp(min=1)
    return phrase_scores

  def generate_scores(self, **kwargs):
    ##### shared #####
    vid_lens = kwargs['vid_lens'] # (batch, )
    num_frames = kwargs['vid_verb_embeds'].size(1)
    vid_masks = framework.ops.sequence_mask(vid_lens, num_frames, inverse=False)

    ##### sentence-level scores #####
    sent_scores = cosine_sim(kwargs['vid_sent_embeds'], kwargs['sent_embeds'])

    ##### verb-level scores #####
    vid_verb_embeds = kwargs['vid_verb_embeds'] # (batch, num_frames, dim_embed)
    verb_embeds = kwargs['verb_embeds'] # (batch, num_verbs, dim_embed)
    verb_lens = kwargs['verb_lens'] # (batch, num_verbs)
    verb_masks = framework.ops.sequence_mask(torch.sum(verb_lens > 0, 1).long(), 
      self.config.num_verbs, inverse=False)
    # sum: (batch_vids, batch_sents)
    verb_scores = self.generate_phrase_scores(vid_verb_embeds, vid_masks, verb_embeds, verb_masks)

    ##### noun-level scores #####
    vid_noun_embeds = kwargs['vid_noun_embeds'] # (batch, num_frames, dim_embed)
    noun_embeds = kwargs['noun_embeds'] # (batch, num_nouns, dim_embed)
    noun_lens = kwargs['noun_lens'] # (batch, num_nouns)
    noun_masks = framework.ops.sequence_mask(torch.sum(noun_lens > 0, 1).long(), 
      self.config.num_nouns, inverse=False)
    # sum: (batch_vids, batch_sents)
    noun_scores = self.generate_phrase_scores(vid_noun_embeds, vid_masks, noun_embeds, noun_masks)

    return sent_scores, verb_scores, noun_scores

  def forward_loss(self, batch_data, step=None):
    enc_outs = self.forward_video_embed(batch_data)
    cap_enc_outs = self.forward_text_embed(batch_data)
    enc_outs.update(cap_enc_outs)

    sent_scores, verb_scores, noun_scores = self.generate_scores(**enc_outs)
    scores = (sent_scores + verb_scores + noun_scores) / 3

    from .ndcg_map_helpers import get_relevances_single_caption, get_relevances_multi_caption

    threshold_pos = batch_data["threshold_pos"]
    if threshold_pos < 1:
      noun_classes = batch_data['noun_classes']
      verb_class = batch_data['verb_class']
      # get_relevances should be dataset-dependent
      #  (for MSR->multi-captions-per-clip->
      #  it should not be "batch captions"x"batch captions", but "batch-videos' 'pooled-captions'"x"batch caption")
      #  --> video_verb_classes, video_noun_classes from batch_data
      if "video_verb_classes" in batch_data:
        video_verb_classes = batch_data["video_verb_classes"]
        video_noun_classes = batch_data["video_noun_classes"]

        sent_rel = get_relevances_multi_caption(video_verbs=video_verb_classes, video_nouns=video_noun_classes,
                                                batch_verbs=verb_class, batch_nouns=noun_classes).cuda()
        noun_rel = get_relevances_multi_caption(video_nouns=video_noun_classes,
                                                batch_nouns=noun_classes).cuda()
        verb_rel = get_relevances_multi_caption(video_verbs=video_verb_classes,
                                                batch_verbs=verb_class).cuda()

      else:
        sent_rel = get_relevances_single_caption(batch_verbs=verb_class, batch_nouns=noun_classes).cuda()
        noun_rel = get_relevances_single_caption(batch_nouns=noun_classes).cuda()
        verb_rel = get_relevances_single_caption(batch_verbs=verb_class).cuda()

      sent_loss = self.criterion(sent_scores, batch_relevance=sent_rel, threshold_pos=threshold_pos)
      verb_loss = self.criterion(verb_scores, batch_relevance=verb_rel, threshold_pos=threshold_pos)
      noun_loss = self.criterion(noun_scores, batch_relevance=noun_rel, threshold_pos=threshold_pos)
      fusion_loss = self.criterion(scores, batch_relevance=sent_rel, threshold_pos=threshold_pos)

    else:
      sent_loss = self.criterion(sent_scores)
      verb_loss = self.criterion(verb_scores)
      noun_loss = self.criterion(noun_scores)
      fusion_loss = self.criterion(scores)

    if self.config.loss_weights is None:
      loss = fusion_loss
    else:
      loss = self.config.loss_weights[0] * fusion_loss + \
             self.config.loss_weights[1] * sent_loss + \
             self.config.loss_weights[2] * verb_loss + \
             self.config.loss_weights[3] * noun_loss

    if step is not None and self.config.monitor_iter > 0 and step % self.config.monitor_iter == 0:
      neg_scores = scores.masked_fill(torch.eye(len(scores), dtype=torch.bool).to(self.device), -1e10)
      self.print_fn('\tstep %d: pos mean scores %.2f, hard neg mean scores i2t %.2f, t2i %.2f'%(
        step, torch.mean(torch.diag(scores)), torch.mean(torch.max(neg_scores, 1)[0]), 
        torch.mean(torch.max(neg_scores, 0)[0])))
      self.print_fn('\tstep %d: sent_loss %.4f, verb_loss %.4f, noun_loss %.4f, fusion_loss %.4f'%(
        step, sent_loss.data.item(), verb_loss.data.item(), noun_loss.data.item(), fusion_loss.data.item()))

    def get_hard_neg_relevance(_scores, _computed_relevance, delta=0.2, threshold_pos=1.):  # -> only relevance, no 1-relevance
      # here we want to compute (1) the indices of the hard-negatives, and (2) their relevance
      # this should be related to what happens during training:
      # so if above we mask the items which have relevance>threshold, we should do the same here as well
      batch_size = _scores.size(0)
      diagonal = _scores.diag().view(batch_size, 1)  # positive pairs

      # mask to clear diagonals which are positive pairs
      pos_masks = torch.eye(batch_size).bool().to(_scores.device)

      batch_topk = 1
      d1 = diagonal.expand_as(_scores)  # same collumn for im2s (negative sentence)
      # compare every diagonal score to scores in its collumn
      # caption retrieval
      cost_s = (delta + _scores - d1).clamp(min=0)
      cost_s = cost_s.masked_fill(pos_masks, 0)
      if threshold_pos < 1:
        cost_s[_computed_relevance >= threshold_pos] = 0
      cost_s, indices_s = torch.topk(cost_s, batch_topk, dim=1)

      d2 = diagonal.t().expand_as(_scores)  # same row for s2im (negative image)
      # compare every diagonal score to scores in its row
      cost_im = (delta + _scores - d2).clamp(min=0)
      cost_im = cost_im.masked_fill(pos_masks, 0)
      if threshold_pos < 1:
        cost_im[_computed_relevance >= threshold_pos] = 0
      cost_im, indices_im = torch.topk(cost_im, batch_topk, dim=0)

      #print(_computed_relevance[indices_s], _computed_relevance[indices_im])
      #print(_computed_relevance[indices_s].shape, _computed_relevance[indices_im].shape)
      indices_s = indices_s.squeeze(1)
      indices_im = indices_im.squeeze(0)
      return _computed_relevance[torch.arange(len(indices_im)), indices_im], \
             _computed_relevance[indices_s, torch.arange(len(indices_s))]

    self.logger.add_scalar("train/loss", loss.item(), step)
    self.logger.add_scalar("train_other/sentence_loss", sent_loss.item(), step)
    self.logger.add_scalar("train_other/verb_loss", verb_loss.item(), step)
    self.logger.add_scalar("train_other/noun_loss", noun_loss.item(), step)

    with torch.no_grad():
      noun_classes = batch_data['noun_classes']
      verb_class = batch_data['verb_class']
      if threshold_pos >= 1:
        if "video_verb_classes" in batch_data:
          video_verb_classes = batch_data["video_verb_classes"]
          video_noun_classes = batch_data["video_noun_classes"]

          sent_rel = get_relevances_multi_caption(video_verbs=video_verb_classes, video_nouns=video_noun_classes,
                                                  batch_verbs=verb_class, batch_nouns=noun_classes).cuda()
          noun_rel = get_relevances_multi_caption(video_nouns=video_noun_classes,
                                                  batch_nouns=noun_classes).cuda()
          verb_rel = get_relevances_multi_caption(video_verbs=video_verb_classes,
                                                  batch_verbs=verb_class).cuda()

        else:
          sent_rel = get_relevances_single_caption(batch_verbs=verb_class, batch_nouns=noun_classes).cuda()
          noun_rel = get_relevances_single_caption(batch_nouns=noun_classes).cuda()
          verb_rel = get_relevances_single_caption(batch_verbs=verb_class).cuda()

      debug_hn_rels = []
      for n, v in [("sentence", sent_rel), ("noun", noun_rel), ("verb", verb_rel)]:
        self.logger.add_scalar(f"train_other/{n}_positives_relevance", torch.mean(v.diag()).item(), step)
        if threshold_pos < 1:
          hn_rel_tv, hn_rel_vt = get_hard_neg_relevance(scores, v, delta=self.criterion.margin, threshold_pos=threshold_pos)
        else:
          hn_rel_tv, hn_rel_vt = get_hard_neg_relevance(scores, v, delta=self.criterion.margin)
        self.logger.add_scalar(f"train_other/{n}_hard-negatives-t2v_relevance",
                               torch.mean(hn_rel_tv).item(), step)
        self.logger.add_scalar(f"train_other/{n}_hard-negatives-v2t_relevance",
                               torch.mean(hn_rel_vt).item(), step)
        self.logger.add_histogram(f"train_other/{n}_hard-negatives-hist-t2v_relevance",
                                  hn_rel_tv, step)
        self.logger.add_histogram(f"train_other/{n}_hard-negatives-hist-v2t_relevance",
                                  hn_rel_vt, step)
        self.logger.add_scalar(f"train_other/{n}_hard-negatives-t2v_%rel>0",
                               len(hn_rel_tv[hn_rel_tv > 0])/len(hn_rel_tv), step)
        self.logger.add_scalar(f"train_other/{n}_hard-negatives-v2t_%rel>0",
                               len(hn_rel_vt[hn_rel_vt > 0])/len(hn_rel_vt), step)

        if n == "sentence":
          debug_hn_rels.append(hn_rel_vt.cpu().numpy())

      """import h5py
      import os
      file_name = f"epic100_debug_count_hardneg_relevances_vt{f'_thr{threshold_pos}' if threshold_pos < 1 else ''}.hdf5"
      mod = "w"
      if os.path.exists(file_name):
        mod = "a"
      with h5py.File(file_name, mod) as f:
        f.create_dataset(f"step{step}", data=np.concatenate(debug_hn_rels))"""

    return loss
      
  def evaluate_scores(self, tst_reader):
    K = self.config.subcfgs[VISENC].num_levels
    vid_names, all_scores = [], [[] for _ in range(K)]
    cap_names = tst_reader.dataset.captions
    for vid_data in tst_reader:
      vid_names.extend(vid_data['names'])
      vid_enc_outs = self.forward_video_embed(vid_data)
      for k in range(K):
        all_scores[k].append([])
      for cap_data in tst_reader.dataset.iterate_over_captions(self.config.tst_batch_size):
        cap_enc_outs = self.forward_text_embed(cap_data)
        cap_enc_outs.update(vid_enc_outs)
        indv_scores = self.generate_scores(**cap_enc_outs)
        for k in range(K):
          all_scores[k][-1].append(indv_scores[k].data.cpu().numpy())
      for k in range(K):
        all_scores[k][-1] = np.concatenate(all_scores[k][-1], axis=1)
      #break
    for k in range(K):
      all_scores[k] = np.concatenate(all_scores[k], axis=0) # (n_img, n_cap)
    all_scores = np.array(all_scores) # (3, n_img, n_cap)
    return vid_names, cap_names, all_scores

  def evaluate(self, tst_reader, return_outs=False):
    vid_names, cap_names, scores = self.evaluate_scores(tst_reader)

    i2t_gts = []
    for vid_name in vid_names:
      i2t_gts.append([])
      for i, cap_name in enumerate(cap_names):
        if cap_name in tst_reader.dataset.ref_captions[vid_name]:
          i2t_gts[-1].append(i)

    t2i_gts = {}
    for i, t_gts in enumerate(i2t_gts):
      for t_gt in t_gts:
        t2i_gts.setdefault(t_gt, [])
        t2i_gts[t_gt].append(i)

    fused_scores = np.mean(scores, 0)

    metrics = self.calculate_metrics(fused_scores, i2t_gts, t2i_gts)

    if tst_reader.dataset.is_test:
      import pickle as pkl
      rel_mat = tst_reader.dataset.relevance_matrix
      print(f"using relevance matrix with shape {rel_mat.shape}")

      from .ndcg_map_helpers import calculate_k_counts, calculate_mAP, calculate_nDCG, calculate_IDCG
      vis_k_counts = calculate_k_counts(rel_mat)
      txt_k_counts = calculate_k_counts(rel_mat.T)

      idcg_v = calculate_IDCG(rel_mat, vis_k_counts)
      idcg_t = calculate_IDCG(rel_mat.T, txt_k_counts)

      vis_nDCG = calculate_nDCG(fused_scores, rel_mat, vis_k_counts, IDCG=idcg_v)
      txt_nDCG = calculate_nDCG(fused_scores.T, rel_mat.T, txt_k_counts, IDCG=idcg_t)
      print('nDCG: VT:{:.3f} TV:{:.3f} AVG:{:.3f}'.format(vis_nDCG, txt_nDCG, (vis_nDCG + txt_nDCG) / 2))

      vis_mAP = calculate_mAP(fused_scores, rel_mat)
      txt_mAP = calculate_mAP(fused_scores.T, rel_mat.T)
      print('mAP: VT:{:.3f} TV:{:.3f} AVG:{:.3f}'.format(vis_mAP, txt_mAP, (vis_mAP + txt_mAP) / 2))

      if tst_reader.dataset.dname == "youcook":
        metrics["nDCG_vt"] = vis_nDCG
        metrics["nDCG_tv"] = txt_nDCG
        metrics["nDCG_avg"] = (vis_nDCG + txt_nDCG) / 2
        metrics["mAP_vt"] = vis_mAP
        metrics["mAP_tv"] = txt_mAP
        metrics["mAP_avg"] = (vis_mAP + txt_mAP) / 2

    else:
      import pickle as pkl
      rel_mat = tst_reader.dataset.relevance_matrix
      print(f"using relevance matrix with shape {rel_mat.shape}")

      from .ndcg_map_helpers import calculate_k_counts, calculate_mAP, calculate_nDCG, calculate_IDCG
      vis_k_counts = calculate_k_counts(rel_mat)
      txt_k_counts = calculate_k_counts(rel_mat.T)

      idcg_v = calculate_IDCG(rel_mat, vis_k_counts)
      idcg_t = calculate_IDCG(rel_mat.T, txt_k_counts)

      vis_nDCG = calculate_nDCG(fused_scores, rel_mat, vis_k_counts, IDCG=idcg_v)
      txt_nDCG = calculate_nDCG(fused_scores.T, rel_mat.T, txt_k_counts, IDCG=idcg_t)
      print('nDCG: VT:{:.3f} TV:{:.3f} AVG:{:.3f}'.format(vis_nDCG, txt_nDCG, (vis_nDCG + txt_nDCG) / 2))

      vis_mAP = calculate_mAP(fused_scores, rel_mat)
      txt_mAP = calculate_mAP(fused_scores.T, rel_mat.T)
      print('mAP: VT:{:.3f} TV:{:.3f} AVG:{:.3f}'.format(vis_mAP, txt_mAP, (vis_mAP + txt_mAP) / 2))

      metrics["nDCG_vt"] = vis_nDCG
      metrics["nDCG_tv"] = txt_nDCG
      metrics["nDCG_avg"] = (vis_nDCG + txt_nDCG) / 2
      metrics["mAP_vt"] = vis_mAP
      metrics["mAP_tv"] = txt_mAP
      metrics["mAP_avg"] = (vis_mAP + txt_mAP) / 2

    if return_outs:
      outs = {
        'vid_names': vid_names,
        'cap_names': cap_names,
        'scores': scores,
      }
      return metrics, outs
    else:
      return metrics


class MultisentRoleGraphMatchModel(RoleGraphMatchModel):

  def forward_text_embed(self, batch_data):
    # during eval we still want to use the original inference
    if not isinstance(batch_data["sent_lens"][0], tuple):
      return super().forward_text_embed(batch_data)
    else:
      fst_sent_data = {}
      fst_sent_data['sent_lens'] = [re[0] for re in batch_data['sent_lens']]
      fst_sent_data['sent_ids'] = batch_data['sent_ids'][0] #[:, 0]
      fst_sent_data['noun_masks'] = batch_data['noun_masks'][0] #[:, 0, :, :]
      fst_sent_data['verb_masks'] = batch_data['verb_masks'][0] #[:, 0, :, :]
      fst_sent_data['node_roles'] = [re[0] for re in batch_data['node_roles']]
      fst_sent_data['rel_edges'] = [re[0] for re in batch_data['rel_edges']]

      snd_sent_data = {}
      snd_sent_data['sent_lens'] = [re[1] for re in batch_data['sent_lens']]
      snd_sent_data['sent_ids'] = batch_data['sent_ids'][1] #[:, 1]
      snd_sent_data['noun_masks'] = batch_data['noun_masks'][1] #[:, 1, :, :]
      snd_sent_data['verb_masks'] = batch_data['verb_masks'][1] #[:, 1, :, :]
      snd_sent_data['node_roles'] = [re[1] for re in batch_data['node_roles']]
      snd_sent_data['rel_edges'] = [re[1] for re in batch_data['rel_edges']]

      sent_ids = torch.LongTensor(fst_sent_data['sent_ids']).to(self.device)
      fst_sent_lens = torch.LongTensor(fst_sent_data['sent_lens']).to(self.device)
      verb_masks = torch.BoolTensor(fst_sent_data['verb_masks']).to(self.device)
      noun_masks = torch.BoolTensor(fst_sent_data['noun_masks']).to(self.device)
      node_roles = torch.LongTensor(fst_sent_data['node_roles']).to(self.device)
      rel_edges = torch.FloatTensor(fst_sent_data['rel_edges']).to(self.device)
      #print(sent_ids.shape, fst_sent_lens.shape, verb_masks.shape, noun_masks.shape, node_roles.shape, rel_edges.shape)

      fst_verb_lens = torch.sum(verb_masks, 2)
      fst_noun_lens = torch.sum(noun_masks, 2)
      # sent_embeds: (batch, dim_embed)
      # verb_embeds, noun_embeds: (batch, num_xxx, dim_embed)
      fst_sent_embeds, fst_verb_embeds, fst_noun_embeds = self.submods[TXTENC](
        sent_ids, fst_sent_lens, verb_masks, noun_masks, node_roles, rel_edges)

      sent_ids = torch.LongTensor(snd_sent_data['sent_ids']).to(self.device)
      snd_sent_lens = torch.LongTensor(snd_sent_data['sent_lens']).to(self.device)
      verb_masks = torch.BoolTensor(snd_sent_data['verb_masks']).to(self.device)
      noun_masks = torch.BoolTensor(snd_sent_data['noun_masks']).to(self.device)
      node_roles = torch.LongTensor(snd_sent_data['node_roles']).to(self.device)
      rel_edges = torch.FloatTensor(snd_sent_data['rel_edges']).to(self.device)
      snd_verb_lens = torch.sum(verb_masks, 2)
      snd_noun_lens = torch.sum(noun_masks, 2)
      # sent_embeds: (batch, dim_embed)
      # verb_embeds, noun_embeds: (batch, num_xxx, dim_embed)
      snd_sent_embeds, snd_verb_embeds, snd_noun_embeds = self.submods[TXTENC](
        sent_ids, snd_sent_lens, verb_masks, noun_masks, node_roles, rel_edges)

      from numpy.random import beta
      lambdas = torch.FloatTensor(beta(1, 1, size=len(fst_sent_embeds))).view(-1, 1, 1).to(self.device)
      lambdas_sent = lambdas.squeeze(-1)

      return {
        'sent_embeds': fst_sent_embeds * lambdas_sent + (1 - lambdas_sent) * snd_sent_embeds,
        'sent_lens': (fst_sent_lens + snd_sent_lens).clamp(max=1),
        'verb_embeds': fst_verb_embeds * lambdas + (1 - lambdas) * snd_verb_embeds,
        'verb_lens': (fst_verb_lens + snd_verb_lens).clamp(max=1),
        'noun_embeds': fst_noun_embeds * lambdas + (1 - lambdas) * snd_noun_embeds,
        'noun_lens': (fst_noun_lens + snd_noun_lens).clamp(max=1),
        }


class HPRoleGraphMatchModel(RoleGraphMatchModel):
  def build_loss(self):
    criterion = t2vretrieval.models.criterion.ContrastiveLossHP(
      margin=self.config.margin,
      margin_pos=self.config.margin_pos,
      max_violation=self.config.max_violation,
      topk=self.config.hard_topk,
      direction=self.config.loss_direction)
    return criterion


  def get_hard_neg_pos_relevance(self, _scores, _computed_relevance, delta=0.2, threshold_pos=1.):  # -> only relevance, no 1-relevance
      # here we want to compute (1) the indices of the hard-negatives, and (2) their relevance
      # same for hard positives
      # this should be related to what happens during training:
      # so if above we mask the items which have relevance>threshold, we should do the same here as well
      batch_size = _scores.size(0)
      diagonal = _scores.diag().view(batch_size, 1)  # positive pairs

      # mask to clear diagonals which are positive pairs
      pos_masks = torch.eye(batch_size).bool().to(_scores.device)

      batch_topk = 1
      d1 = diagonal.expand_as(_scores)  # same collumn for im2s (negative sentence)
      # compare every diagonal score to scores in its collumn
      # caption retrieval
      cost_s = (delta + _scores - d1).clamp(min=0)
      cost_s = cost_s.masked_fill(pos_masks, 0)
      hp_scores = _scores.clone()
      if threshold_pos < 1:
        cost_s[_computed_relevance >= threshold_pos] = 0
        hp_scores[_computed_relevance < threshold_pos] = 1  # mask the negatives; positives have now score <= 1
      cost_s, indices_s = torch.topk(cost_s, batch_topk, dim=1)

      hp_scores = hp_scores.masked_fill(pos_masks, 1)
      # we want to pick the argmin (hardest positive)
      hp_scores, hp_indices_s = torch.topk(hp_scores, batch_topk, dim=1, largest=False)
      # -> s(q, v+)
      hp_cost_s = (delta + _scores - hp_scores).clamp(min=0)
      hp_cost_s = hp_cost_s.masked_fill(pos_masks, 0)
      if _computed_relevance is not None:
        hp_cost_s[_computed_relevance >= threshold_pos] = 0
      hp_cost_s, _ = torch.topk(hp_cost_s, batch_topk, dim=1)

      d2 = diagonal.t().expand_as(_scores)  # same row for s2im (negative image)
      # compare every diagonal score to scores in its row
      cost_im = (delta + _scores - d2).clamp(min=0)
      cost_im = cost_im.masked_fill(pos_masks, 0)
      hp_scores_im = _scores.clone()
      if threshold_pos < 1:
        cost_im[_computed_relevance >= threshold_pos] = 0
        hp_scores_im[_computed_relevance < threshold_pos] = 1
      cost_im, indices_im = torch.topk(cost_im, batch_topk, dim=0)
      hp_scores_im = hp_scores_im.masked_fill(pos_masks, 1)
      # we want to pick the argmin (hardest positive)
      hp_scores_im, hp_indices_im = torch.topk(hp_scores_im, batch_topk, dim=0, largest=False)
      # -> s(q, v+)
      hp_cost_im = (delta + _scores - hp_scores_im).clamp(min=0)
      hp_cost_im = hp_cost_im.masked_fill(pos_masks, 0)
      if _computed_relevance is not None:
        hp_cost_im[_computed_relevance >= threshold_pos] = 0
      hp_cost_im, _ = torch.topk(hp_cost_im, batch_topk, dim=0)

      #print(_computed_relevance[indices_s], _computed_relevance[indices_im])
      #print(_computed_relevance[indices_s].shape, _computed_relevance[indices_im].shape)
      indices_s = indices_s.squeeze(1)
      indices_im = indices_im.squeeze(0)
      hp_indices_s = hp_indices_s.squeeze(1)
      hp_indices_im = hp_indices_im.squeeze(0)
      return _computed_relevance[torch.arange(len(indices_im)), indices_im], \
             _computed_relevance[torch.arange(len(indices_s)), indices_s], \
             _computed_relevance[torch.arange(len(hp_indices_im)), hp_indices_im], \
             _computed_relevance[torch.arange(len(hp_indices_s)), hp_indices_s]


  def forward_loss(self, batch_data, step=None):
    enc_outs = self.forward_video_embed(batch_data)
    cap_enc_outs = self.forward_text_embed(batch_data)
    enc_outs.update(cap_enc_outs)

    sent_scores, verb_scores, noun_scores = self.generate_scores(**enc_outs)
    scores = (sent_scores + verb_scores + noun_scores) / 3

    from .ndcg_map_helpers import get_relevances_single_caption, get_relevances_multi_caption

    threshold_pos = batch_data["threshold_pos"]
    assert threshold_pos < 1
    noun_classes = batch_data['noun_classes']
    verb_class = batch_data['verb_class']
    # get_relevances should be dataset-dependent
    #  (for MSR->multi-captions-per-clip->
    #  it should not be "batch captions"x"batch captions", but "batch-videos' 'pooled-captions'"x"batch caption")
    #  --> video_verb_classes, video_noun_classes from batch_data
    if "video_verb_classes" in batch_data:
      video_verb_classes = batch_data["video_verb_classes"]
      video_noun_classes = batch_data["video_noun_classes"]

      sent_rel = get_relevances_multi_caption(video_verbs=video_verb_classes, video_nouns=video_noun_classes,
                                              batch_verbs=verb_class, batch_nouns=noun_classes).cuda()
      noun_rel = get_relevances_multi_caption(video_nouns=video_noun_classes,
                                              batch_nouns=noun_classes).cuda()
      verb_rel = get_relevances_multi_caption(video_verbs=video_verb_classes,
                                              batch_verbs=verb_class).cuda()

    else:
      sent_rel = get_relevances_single_caption(batch_verbs=verb_class, batch_nouns=noun_classes).cuda()
      noun_rel = get_relevances_single_caption(batch_nouns=noun_classes).cuda()
      verb_rel = get_relevances_single_caption(batch_verbs=verb_class).cuda()

    sent_loss, hp_sent_loss = self.criterion(sent_scores, batch_relevance=sent_rel, threshold_pos=threshold_pos)
    verb_loss, hp_verb_loss = self.criterion(verb_scores, batch_relevance=verb_rel, threshold_pos=threshold_pos)
    noun_loss, hp_noun_loss = self.criterion(noun_scores, batch_relevance=noun_rel, threshold_pos=threshold_pos)
    fusion_loss, hp_fusion_loss = self.criterion(scores, batch_relevance=sent_rel, threshold_pos=threshold_pos)

    if self.config.loss_weights is None:
      loss = fusion_loss + hp_fusion_loss
    else:
      loss = self.config.loss_weights[0] * fusion_loss + \
             self.config.loss_weights[1] * sent_loss + \
             self.config.loss_weights[2] * verb_loss + \
             self.config.loss_weights[3] * noun_loss

      loss += self.config.loss_weights[0] * hp_fusion_loss + \
             self.config.loss_weights[1] * hp_sent_loss + \
             self.config.loss_weights[2] * hp_verb_loss + \
             self.config.loss_weights[3] * hp_noun_loss

    if step is not None and self.config.monitor_iter > 0 and step % self.config.monitor_iter == 0:
      neg_scores = scores.masked_fill(torch.eye(len(scores), dtype=torch.bool).to(self.device), -1e10)
      self.print_fn('\tstep %d: pos mean scores %.2f, hard neg mean scores i2t %.2f, t2i %.2f'%(
        step, torch.mean(torch.diag(scores)), torch.mean(torch.max(neg_scores, 1)[0]),
        torch.mean(torch.max(neg_scores, 0)[0])))
      self.print_fn('\tstep %d: sent_loss %.4f, verb_loss %.4f, noun_loss %.4f, fusion_loss %.4f'%(
        step, sent_loss.data.item(), verb_loss.data.item(), noun_loss.data.item(), fusion_loss.data.item()))

    self.logger.add_scalar("train/loss", loss.item(), step)
    self.logger.add_scalar("train_other/sentence_loss", sent_loss.item(), step)
    self.logger.add_scalar("train_other/verb_loss", verb_loss.item(), step)
    self.logger.add_scalar("train_other/noun_loss", noun_loss.item(), step)
    self.logger.add_scalar("train_other/hard-pos_fusion_loss", hp_fusion_loss.item(), step)
    self.logger.add_scalar("train_other/hard-pos_sentence_loss", hp_sent_loss.item(), step)
    self.logger.add_scalar("train_other/hard-pos_verb_loss", hp_verb_loss.item(), step)
    self.logger.add_scalar("train_other/hard-pos_noun_loss", hp_noun_loss.item(), step)

    with torch.no_grad():
      noun_classes = batch_data['noun_classes']
      verb_class = batch_data['verb_class']
      if threshold_pos >= 1:
        if "video_verb_classes" in batch_data:
          video_verb_classes = batch_data["video_verb_classes"]
          video_noun_classes = batch_data["video_noun_classes"]

          sent_rel = get_relevances_multi_caption(video_verbs=video_verb_classes, video_nouns=video_noun_classes,
                                                  batch_verbs=verb_class, batch_nouns=noun_classes).cuda()
          noun_rel = get_relevances_multi_caption(video_nouns=video_noun_classes,
                                                  batch_nouns=noun_classes).cuda()
          verb_rel = get_relevances_multi_caption(video_verbs=video_verb_classes,
                                                  batch_verbs=verb_class).cuda()

        else:
          sent_rel = get_relevances_single_caption(batch_verbs=verb_class, batch_nouns=noun_classes).cuda()
          noun_rel = get_relevances_single_caption(batch_nouns=noun_classes).cuda()
          verb_rel = get_relevances_single_caption(batch_verbs=verb_class).cuda()

      for n, v in [("sentence", sent_rel), ("noun", noun_rel), ("verb", verb_rel)]:
        self.logger.add_scalar(f"train_other/{n}_positives_relevance", torch.mean(v.diag()).item(), step)
        if threshold_pos < 1:
          hn_rel_tv, hn_rel_vt, hp_rel_tv, hp_rel_vt = self.get_hard_neg_pos_relevance(scores, v, delta=self.criterion.margin, threshold_pos=threshold_pos)
        else:
          hn_rel_tv, hn_rel_vt, hp_rel_tv, hp_rel_vt = self.get_hard_neg_pos_relevance(scores, v, delta=self.criterion.margin)
        self.logger.add_scalar(f"train_other/{n}_hard-negatives-t2v_relevance",
                               torch.mean(hn_rel_tv).item(), step)
        self.logger.add_scalar(f"train_other/{n}_hard-negatives-v2t_relevance",
                               torch.mean(hn_rel_vt).item(), step)
        self.logger.add_histogram(f"train_other/{n}_hard-negatives-hist-t2v_relevance",
                                  hn_rel_tv, step)
        self.logger.add_histogram(f"train_other/{n}_hard-negatives-hist-v2t_relevance",
                                  hn_rel_vt, step)
        self.logger.add_scalar(f"train_other/{n}_hard-positives-t2v_relevance",
                               torch.mean(hp_rel_tv).item(), step)
        self.logger.add_scalar(f"train_other/{n}_hard-positives-v2t_relevance",
                               torch.mean(hp_rel_vt).item(), step)
        self.logger.add_histogram(f"train_other/{n}_hard-positives-hist-t2v_relevance",
                                  hp_rel_tv, step)
        self.logger.add_histogram(f"train_other/{n}_hard-positives-hist-v2t_relevance",
                                  hp_rel_vt, step)
        self.logger.add_scalar(f"train_other/{n}_hard-negatives-t2v_%rel>0",
                               len(hn_rel_tv[hn_rel_tv > 0])/len(hn_rel_tv), step)
        self.logger.add_scalar(f"train_other/{n}_hard-negatives-v2t_%rel>0",
                               len(hn_rel_vt[hn_rel_vt > 0])/len(hn_rel_vt), step)

    return loss


class HPMultisentRoleGraphMatchModel(HPRoleGraphMatchModel):

  def forward_text_embed(self, batch_data):
    # during eval we still want to use the original inference
    if not isinstance(batch_data["sent_lens"][0], tuple):
      return super().forward_text_embed(batch_data)
    else:
      fst_sent_data = {}
      fst_sent_data['sent_lens'] = [re[0] for re in batch_data['sent_lens']]
      fst_sent_data['sent_ids'] = batch_data['sent_ids'][0] #[:, 0]
      fst_sent_data['noun_masks'] = batch_data['noun_masks'][0] #[:, 0, :, :]
      fst_sent_data['verb_masks'] = batch_data['verb_masks'][0] #[:, 0, :, :]
      fst_sent_data['node_roles'] = [re[0] for re in batch_data['node_roles']]
      fst_sent_data['rel_edges'] = [re[0] for re in batch_data['rel_edges']]

      snd_sent_data = {}
      snd_sent_data['sent_lens'] = [re[1] for re in batch_data['sent_lens']]
      snd_sent_data['sent_ids'] = batch_data['sent_ids'][1] #[:, 1]
      snd_sent_data['noun_masks'] = batch_data['noun_masks'][1] #[:, 1, :, :]
      snd_sent_data['verb_masks'] = batch_data['verb_masks'][1] #[:, 1, :, :]
      snd_sent_data['node_roles'] = [re[1] for re in batch_data['node_roles']]
      snd_sent_data['rel_edges'] = [re[1] for re in batch_data['rel_edges']]

      sent_ids = torch.LongTensor(fst_sent_data['sent_ids']).to(self.device)
      fst_sent_lens = torch.LongTensor(fst_sent_data['sent_lens']).to(self.device)
      verb_masks = torch.BoolTensor(fst_sent_data['verb_masks']).to(self.device)
      noun_masks = torch.BoolTensor(fst_sent_data['noun_masks']).to(self.device)
      node_roles = torch.LongTensor(fst_sent_data['node_roles']).to(self.device)
      rel_edges = torch.FloatTensor(fst_sent_data['rel_edges']).to(self.device)
      #print(sent_ids.shape, fst_sent_lens.shape, verb_masks.shape, noun_masks.shape, node_roles.shape, rel_edges.shape)

      fst_verb_lens = torch.sum(verb_masks, 2)
      fst_noun_lens = torch.sum(noun_masks, 2)
      # sent_embeds: (batch, dim_embed)
      # verb_embeds, noun_embeds: (batch, num_xxx, dim_embed)
      fst_sent_embeds, fst_verb_embeds, fst_noun_embeds = self.submods[TXTENC](
        sent_ids, fst_sent_lens, verb_masks, noun_masks, node_roles, rel_edges)

      sent_ids = torch.LongTensor(snd_sent_data['sent_ids']).to(self.device)
      snd_sent_lens = torch.LongTensor(snd_sent_data['sent_lens']).to(self.device)
      verb_masks = torch.BoolTensor(snd_sent_data['verb_masks']).to(self.device)
      noun_masks = torch.BoolTensor(snd_sent_data['noun_masks']).to(self.device)
      node_roles = torch.LongTensor(snd_sent_data['node_roles']).to(self.device)
      rel_edges = torch.FloatTensor(snd_sent_data['rel_edges']).to(self.device)
      snd_verb_lens = torch.sum(verb_masks, 2)
      snd_noun_lens = torch.sum(noun_masks, 2)
      # sent_embeds: (batch, dim_embed)
      # verb_embeds, noun_embeds: (batch, num_xxx, dim_embed)
      snd_sent_embeds, snd_verb_embeds, snd_noun_embeds = self.submods[TXTENC](
        sent_ids, snd_sent_lens, verb_masks, noun_masks, node_roles, rel_edges)

      from numpy.random import beta
      lambdas = torch.FloatTensor(beta(1, 1, size=len(fst_sent_embeds))).view(-1, 1, 1).to(self.device)
      lambdas_sent = lambdas.squeeze(-1)

      return {
        'sent_embeds': fst_sent_embeds * lambdas_sent + (1 - lambdas_sent) * snd_sent_embeds,
        'sent_lens': (fst_sent_lens + snd_sent_lens).clamp(max=1),
        'verb_embeds': fst_verb_embeds * lambdas + (1 - lambdas) * snd_verb_embeds,
        'verb_lens': (fst_verb_lens + snd_verb_lens).clamp(max=1),
        'noun_embeds': fst_noun_embeds * lambdas + (1 - lambdas) * snd_noun_embeds,
        'noun_lens': (fst_noun_lens + snd_noun_lens).clamp(max=1),
        }


