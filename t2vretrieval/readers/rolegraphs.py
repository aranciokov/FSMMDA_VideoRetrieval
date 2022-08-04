import os
import json
import numpy as np
import h5py
import collections
import torch

import t2vretrieval.readers.mpdata

def extract_video_classes(dataset_file, video_captions_keys, name, percent=0.25):
  # dataset_file -> dict containing the annotations
  # video_captions_keys -> list of caption keys for a given video (likely in a format {video_id}_{caption_id})
  # name -> name of the column (noun_classes, verb_class, etc)
  # percent -> each video has C captions, we want to keep the classes appearing in >=percent*C captions
  #all_vcs = [self.dataset_file[__k]["annotations"][0]["verb_class"] for __k in video_captions_keys]
  all_classes = [dataset_file[__k]["annotations"][0][name] for __k in video_captions_keys]
  count = {}
  for vcs in all_classes:
    for c in vcs:
      if c not in count.keys():
        count[c] = 0
      count[c] += 1

  kept_classes = []
  for cl, cn in count.items():
    if cn >= percent * len(video_captions_keys):
      kept_classes.append(cl)

  return kept_classes

ROLES = ['V', 'ARG1', 'ARG0', 'ARG2', 'ARG3', 'ARG4',
 'ARGM-LOC', 'ARGM-MNR', 'ARGM-TMP', 'ARGM-DIR', 'ARGM-ADV', 
 'ARGM-PRP', 'ARGM-PRD', 'ARGM-COM', 'ARGM-MOD', 'NOUN']

class RoleGraphDataset(t2vretrieval.readers.mpdata.MPDataset):
  def __init__(self, name_file, attn_ft_files, word2int_file,
    max_words_in_sent, num_verbs, num_nouns, ref_caption_file, ref_graph_file, 
    max_attn_len=20, load_video_first=False, is_train=False, _logger=None,
               dataset_file='', is_test=False, dname="", rel_mat_path="", threshold_pos=1):
    if _logger is None:
      self.print_fn = print
    else:
      from torch.utils.tensorboard import SummaryWriter
      self.print_fn = _logger.info if not isinstance(_logger, SummaryWriter) else print

    self.max_words_in_sent = max_words_in_sent
    self.is_train = is_train
    self.attn_ft_files = attn_ft_files
    self.max_attn_len = max_attn_len
    self.load_video_first = load_video_first

    self.names = np.load(name_file)
    self.word2int = json.load(open(word2int_file))

    self.num_videos = len(self.names)
    self.print_fn('num_videos %d' % (self.num_videos))
    self.dname = dname
    self.print_fn('working on dataset: %s' % (self.dname))

    if ref_caption_file is None:
      self.ref_captions = None
    else:
      self.ref_captions = json.load(open(ref_caption_file))
      self.captions = list()
      self.pair_idxs = []
      for i, name in enumerate(self.names):
        for j, sent in enumerate(self.ref_captions[name]):
          self.captions.append(sent)
          self.pair_idxs.append((i, j))

      # for val/test here we may also load the relevance matrix if we want to compute nDCG&mAP
      if is_test or not is_train:  # test or validation
        print(f"rel mat path {rel_mat_path}")
        import pandas
        if is_test:
          assert rel_mat_path != ""
        if rel_mat_path != "":
          if "hdf5" in rel_mat_path:
            import h5py
            with h5py.File(rel_mat_path, "r") as f:
              self.relevance_matrix = np.array(f["rel_mat"])
          else:
            self.relevance_matrix = pandas.read_pickle(rel_mat_path)

        if is_test and dname == "epic":
            print("reading epic100 unique caps")
            self.captions = pandas.read_csv("annotation/epic100RET/EPIC_100_retrieval_test_sentence.csv")['narration'].values

      self.num_pairs = len(self.pair_idxs)
      self.print_fn('captions size %d' % self.num_pairs)

    if self.load_video_first:
      self.all_attn_fts, self.all_attn_lens = [], []
      for name in self.names:
        attn_fts = self.load_attn_ft_by_name(name, self.attn_ft_files)
        attn_fts, attn_len = self.pad_or_trim_feature(attn_fts, self.max_attn_len, trim_type='select')
        self.all_attn_fts.append(attn_fts)
        self.all_attn_lens.append(attn_len)
      self.all_attn_fts = np.array(self.all_attn_fts)
      self.all_attn_lens = np.array(self.all_attn_lens)

    self.num_verbs = num_verbs
    self.num_nouns = num_nouns
    
    self.role2int = {}
    for i, role in enumerate(ROLES):
      self.role2int[role] = i
      self.role2int['C-%s'%role] = i
      self.role2int['R-%s'%role] = i

    self.ref_graphs = json.load(open(ref_graph_file))

    if is_train:
      print(f"using a threshold of {threshold_pos} [0, 1] to distinguish positives (>= thr) from negatives (< thr)")
      self.threshold_pos = threshold_pos
      assert 0 < threshold_pos and threshold_pos <= 1

    self.is_test = is_test
    self.dataset_file_path = dataset_file
    if dataset_file != '':
      self.dataset_file = json.load(open(dataset_file, "r"))['database']
      self.sent2classes = {v["annotations"][0]["sentence"]: {"verb_class": v["annotations"][0]["verb_class"],
                                                             "noun_classes": v["annotations"][0]["noun_classes"]}
                           for v in self.dataset_file.values()}

  def load_attn_ft_by_name(self, name, attn_ft_files):
    attn_fts = []
    for i, attn_ft_file in enumerate(attn_ft_files):
      with h5py.File(attn_ft_file, 'r') as f:
        key = name.replace('/', '_')
        attn_ft = f[key][...]
        attn_fts.append(attn_ft)
    attn_fts = np.concatenate([attn_ft for attn_ft in attn_fts], axis=-1)
    return attn_fts

  def pad_or_trim_feature(self, attn_ft, max_attn_len, trim_type='top'):
    if len(attn_ft.shape) == 2:
      seq_len, dim_ft = attn_ft.shape
    else:
      sqz, seq_len, dim_ft = attn_ft.shape
      assert sqz == 1
      attn_ft = attn_ft.squeeze(0)
    attn_len = min(seq_len, max_attn_len)

    # pad
    if seq_len < max_attn_len:
      new_ft = np.zeros((max_attn_len, dim_ft), np.float32)
      new_ft[:seq_len] = attn_ft
    # trim
    else:
      if trim_type == 'top':
        new_ft = attn_ft[:max_attn_len]
      elif trim_type == 'select':
        idxs = np.round(np.linspace(0, seq_len-1, max_attn_len)).astype(np.int32)
        new_ft = attn_ft[idxs]
    return new_ft, attn_len

  def get_caption_outs(self, out, sent, graph):
    graph_nodes, graph_edges = graph
    #print(graph)

    verb_node2idxs, noun_node2idxs = {}, {}
    edges = []
    out['node_roles'] = np.zeros((self.num_verbs + self.num_nouns, ), np.int32)

    # root node
    sent_ids, sent_len = self.process_sent(sent, self.max_words_in_sent)
    out['sent_ids'] = sent_ids
    out['sent_lens'] = sent_len

    # graph: add verb nodes
    node_idx = 1
    out['verb_masks'] = np.zeros((self.num_verbs, self.max_words_in_sent), np.bool)
    for knode, vnode in graph_nodes.items():
      k = node_idx - 1
      if k >= self.num_verbs:
        break
      if vnode['role'] == 'V' and np.min(vnode['spans']) < self.max_words_in_sent:
        verb_node2idxs[knode] = node_idx
        for widx in vnode['spans']:
          if widx < self.max_words_in_sent:
            out['verb_masks'][k][widx] = True
        out['node_roles'][node_idx - 1] = self.role2int['V']
        # add root to verb edge
        edges.append((0, node_idx))
        node_idx += 1
        
    # graph: add noun nodes
    node_idx = 1 + self.num_verbs
    out['noun_masks'] = np.zeros((self.num_nouns, self.max_words_in_sent), np.bool)
    for knode, vnode in graph_nodes.items():
      k = node_idx - self.num_verbs - 1
      if k >= self.num_nouns:
          break
      if vnode['role'] not in ['ROOT', 'V'] and np.min(vnode['spans']) < self.max_words_in_sent:
        noun_node2idxs[knode] = node_idx
        for widx in vnode['spans']:
          if widx < self.max_words_in_sent:
            out['noun_masks'][k][widx] = True
        out['node_roles'][node_idx - 1] = self.role2int.get(vnode['role'], self.role2int['NOUN'])
        node_idx += 1

    # graph: add verb_node to noun_node edges
    for e in graph_edges:
      if e[0] in verb_node2idxs and e[1] in noun_node2idxs:
        edges.append((verb_node2idxs[e[0]], noun_node2idxs[e[1]]))
        edges.append((noun_node2idxs[e[1]], verb_node2idxs[e[0]]))

    num_nodes = 1 + self.num_verbs + self.num_nouns
    rel_matrix = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    for src_nodeidx, tgt_nodeidx in edges:
      rel_matrix[tgt_nodeidx, src_nodeidx] = 1
    # row norm
    for i in range(num_nodes):
      s = np.sum(rel_matrix[i])
      if s > 0:
        rel_matrix[i] /= s

    out['rel_edges'] = rel_matrix
    return out

  def __getitem__(self, idx):
    out = {}
    if self.is_train:
      video_idx, cap_idx = self.pair_idxs[idx]
      name = self.names[video_idx]
      sent = self.ref_captions[name][cap_idx]
      out = self.get_caption_outs(out, sent, self.ref_graphs[sent])
    else:
      video_idx = idx
      name = self.names[idx]
    
    if self.load_video_first:
      attn_fts, attn_len = self.all_attn_fts[video_idx], self.all_attn_lens[video_idx]
    else:
      attn_fts = self.load_attn_ft_by_name(name, self.attn_ft_files)
      attn_fts, attn_len = self.pad_or_trim_feature(attn_fts, self.max_attn_len, trim_type='select')
    
    out['names'] = name
    out['attn_fts'] = attn_fts
    out['attn_lens'] = attn_len
    if self.is_train:
      out['threshold_pos'] = self.threshold_pos

    if self.dataset_file_path != '' and self.is_train:
      _key = name
      nc_str = self.dataset_file[_key]["annotations"][0]["noun_classes"]
      if isinstance(nc_str, str):
        nc_str = list(map(int, nc_str.replace("[", "").replace("]", "").split(",")))
      out['noun_classes'] = nc_str
      out['verb_class'] = self.dataset_file[_key]["annotations"][0]["verb_class"]

    return out

  def iterate_over_captions(self, batch_size):
    # the sentence order is the same as self.captions
    for s in range(0, len(self.captions), batch_size):
      e = s + batch_size
      data = []
      for sent in self.captions[s: e]:
        out = self.get_caption_outs({}, sent, self.ref_graphs[sent])

        data.append(out)
      outs = collate_graph_fn(data)
      yield outs


class AugmentedRoleGraphDataset(RoleGraphDataset):

  def __init__(self, name_file, attn_ft_files, word2int_file,
    max_words_in_sent, num_verbs, num_nouns, ref_caption_file, ref_graph_file,
    max_attn_len=20, load_video_first=False, is_train=False, _logger=None,
               dataset_file='', is_test=False, dname="", rel_mat_path="", threshold_pos=1,
               verb_classes=None, noun_classes=None, video_verb_similars=None, video_noun_similars=None,
               aug_chance=0.5, original_dataframe="", fix_lambda=False, aug_chance_txt=-1.):
    super().__init__(name_file, attn_ft_files, word2int_file,
                     max_words_in_sent, num_verbs, num_nouns, ref_caption_file, ref_graph_file,
                     max_attn_len=max_attn_len, load_video_first=load_video_first, is_train=is_train, _logger=_logger,
                     dataset_file=dataset_file, is_test=is_test, dname=dname, rel_mat_path=rel_mat_path, threshold_pos=threshold_pos)

    self.aug_chance = aug_chance
    self.aug_chance_txt = aug_chance_txt
    self.fix_lambda = fix_lambda
    self.perform_txt_aug = False
    self.perform_vid_aug = False
    if is_train:
      # noun/verb -> [synonym]
      # e.g. synonyms(classID-of-take) = [pick up, grab, ...]
      import pandas as pd

      if video_verb_similars is not None:
        self.perform_vid_aug = True
        # video_name -> [different video_name with shared verb/noun]
        # e.g. similars(vidX) = [vid2, vid18, ...] where vidX,vid2,... share the same verb/noun
        if os.path.exists(video_verb_similars):
          self.video_verb_similars = json.load(open(video_verb_similars))
          self.video_noun_similars = json.load(open(video_noun_similars))
        else:
          self.video_verb_similars = [set() for i in range(len(self.names))]
          self.video_noun_similars = [set() for i in range(len(self.names))]

          def uncoherent_check(vc1, vc2, nc1, nc2):
            # only checks that v1 and v2 are sharing some verb (or noun) classes
            if not isinstance(vc1, set):
              vc1 = set(vc1)
            if not isinstance(vc2, set):
              vc2 = set(vc2)
            return len(vc1.intersection(vc2)) > 0

          def coherent_check(vc1, vc2, nc1, nc2):
            # checks both that v1 and v2 are sharing some verb (or noun) classes,
            # and that the same holds for noun (or verb) classes
            if not isinstance(nc1, set):
              nc1 = set(nc1)
            if not isinstance(nc2, set):
              nc2 = set(nc2)
            return len(nc1.intersection(nc2)) > 0 and uncoherent_check(vc1, vc2, nc1, nc2)

          if "coherent" in video_noun_similars:
            test_fn = coherent_check
          else:
            test_fn = uncoherent_check

          from tqdm import tqdm
          for i, name in tqdm(enumerate(self.names)):
            # 1) collect classes for this video
            vid_verb_classes = []
            vid_noun_classes = []
            for j, sent in enumerate(self.ref_captions[name]):
              vc = self.sent2classes[sent]["verb_class"]
              if not isinstance(vc, list):
                vc = [vc]
              vid_verb_classes += vc
              nc_str = self.sent2classes[sent]["noun_classes"]
              if isinstance(nc_str, str):
                nc_str = list(map(int, nc_str.replace("[", "").replace("]", "").split(",")))
              vid_noun_classes += nc_str
            vid_verb_classes = set(vid_verb_classes)
            vid_noun_classes = set(vid_noun_classes)

            # 2) find videos with shared classes
            for i2, name2 in enumerate(self.names):
              if i2 > i:
                for j2, sent2 in enumerate(self.ref_captions[name2]):
                  vc2 = self.sent2classes[sent2]["verb_class"]
                  nc_str = self.sent2classes[sent2]["noun_classes"]
                  if isinstance(nc_str, str):
                    nc_str = list(map(int, nc_str.replace("[", "").replace("]", "").split(",")))
                  if not isinstance(vc2, list):
                    vc2 = [vc2]
                  if test_fn(vid_verb_classes, vc2, vid_noun_classes, nc_str):  #len(set(vc2).intersection(vid_verb_classes)) > 0:
                    self.video_verb_similars[i].add(i2)
                    self.video_verb_similars[i2].add(i)
                  if test_fn(vid_noun_classes, nc_str, vid_verb_classes, vc2):  #len(set(nc_str).intersection(vid_noun_classes)) > 0:
                    self.video_noun_similars[i].add(i2)
                    self.video_noun_similars[i2].add(i)

          tmp = []
          for v in self.video_verb_similars:
            tmp.append(list(v))
          json.dump(tmp, open(video_verb_similars, "w"))
          self.video_verb_similars = tmp
          tmp = []
          for v in self.video_noun_similars:
            tmp.append(list(v))
          json.dump(tmp, open(video_noun_similars, "w"))
          self.video_noun_similars = tmp

        print(f"--- Loaded dictionaries for similar videos: {len(self.video_noun_similars)} nouns, {len(self.video_verb_similars)} verbs ---")

      if original_dataframe != "":
        parse = lambda s, t: s.replace("[", "").replace("]", "").replace("'", "").replace(" ", "").replace(t, " ").split(",")
        self.parse_fn = parse
        tmp = pd.read_csv(verb_classes)
        self.verb_synonyms = tmp.instances.apply(
          lambda s: parse(s, '-')).values  # [parse(r["instances"], '-') for i, r in tmp.iterrows()]
        # self.verb_classes_mapping = tmp
        tmp = pd.read_csv(noun_classes)
        self.noun_synonyms = tmp.instances.apply(
          lambda s: parse(s, ':')).values  # [parse(r["instances"], ':') for i, r in tmp.iterrows()]
        # self.noun_classes_mapping = tmp
        print(f"--- Loaded synonyms' lists: {len(self.noun_synonyms)} nouns, {len(self.verb_synonyms)} verbs ---")

        self.original_dataframe = pd.read_csv(original_dataframe)
        self.perform_txt_aug = True
        print(f"--- Loaded original dataframe: {len(self.original_dataframe)} rows ---")

  def pick_syn_from_excluding(self, src, old, old_tks):
    from random import randint
    candidates = [c for c in src[old] if c != old_tks]  # avoiding replacement with same token
    if len(candidates) > 0:
      return candidates[randint(0, len(candidates) - 1)]
    else:
      return None

  def replace_noun(self, sent, old_tks, old_tks_cls, s=":"):
    return self.replace_tk(sent, old_tks, old_tks_cls, self.noun_synonyms, s)

  def replace_verb(self, sent, old_tks, old_tks_cls, s="-"):
    return self.replace_tk(sent, old_tks, old_tks_cls, self.verb_synonyms, s)

  def replace_tk(self, sent, old_tks, old_tks_cls, new_tks_source, s):
    new_tks = self.pick_syn_from_excluding(new_tks_source, old_tks_cls, old_tks.replace(s, ' '))
    if new_tks is not None:
      #print(f"replacing '{old_tks.replace(s, ' ')}' ({old_tks_cls}) with '{new_tks}' within '{sent}'")
      if old_tks.replace(s, ' ') in sent:
        return sent.replace(old_tks.replace(s, ' '), new_tks)

      old_tokens = old_tks.strip().split(s)
      new_tokens = new_tks.strip().split(s)
      try:
        first_occ = min([sent.split().index(_tk) for _tk in old_tokens])
        new_sent = [stk for stk in sent.split() if stk not in old_tokens]
        new_sent = new_sent[:first_occ] + new_tokens + new_sent[first_occ:]
        return " ".join(new_sent)
      except:
        #print(f"replace_tk failed on {sent}")
        return sent
    else:
      return sent

  def mix_fn(self, vid1, vid1_att, vid2, vid2_att, alpha=1):
    # idea based on Mixup (https://arxiv.org/pdf/1710.09412.pdf)
    from numpy.random import beta
    if self.fix_lambda:
      lam = 0.5
    else:
      lam = beta(alpha, alpha)
    v = lam*vid1 + (1-lam)*vid2
    #print(vid1.shape, vid2.shape)  # (Len, Feats)

    return v, (vid1_att + vid2_att).clip(max=max(vid1.shape[0], vid2.shape[0]))

  def __getitem__(self, idx):
    out = {}
    from random import randint
    coin_S = randint(0, 99)  # decide whether to change part of the 'sentence'
    perform_aug = coin_S >= (1 - self.aug_chance)*100
    if self.aug_chance_txt > 0:
      coin_S_txt = randint(0, 99)  # decide whether to change part of the 'sentence'
      perform_aug_txt = coin_S_txt >= (1 - self.aug_chance_txt)*100
    else:
      perform_aug_txt = perform_aug
    coin_N_or_V = randint(0, 1)  # if perform_aug, decide whether to change noun or verb (used both for txt and vid)

    if self.is_train:
      video_idx, cap_idx = self.pair_idxs[idx]
      name = self.names[video_idx]
      sent = self.ref_captions[name][cap_idx]
      if perform_aug_txt and self.perform_txt_aug:
        # perform replacement during training
        nc_str = self.original_dataframe[self.original_dataframe["narration_id"] == name]
        if coin_N_or_V == 0:  # replace noun
          repl_fn = self.replace_noun
          nc_tk = nc_str["all_nouns"].values[0]
          nc_cl = nc_str["all_noun_classes"].values[0]
          if isinstance(nc_cl, str):
            nc_cl = list(map(int, nc_cl.replace("[", "").replace("]", "").split(",")))
          if isinstance(nc_tk, str):
            nc_tk = nc_tk.replace("[", "").replace("]", "").replace("'", "").split(",")
          #print(nc_tk, type(nc_tk[0]), len(nc_tk), nc_cl, type(nc_cl[0]), len(nc_cl))
          # ^ more than one noun *classes*; must select ONE to replace
          sel_nn = randint(0, len(nc_tk) - 1)
          old_tokens = nc_tk[sel_nn]
          old_tokens_classes = nc_cl[sel_nn]
          #print(old_tokens, type(old_tokens), old_tokens_classes, type(old_tokens_classes))
        else:
          repl_fn = self.replace_verb
          assert len(nc_str["verb"].values) > 0, f"replacing verb in {name}"
          old_tokens = nc_str["verb"].values[0]
          old_tokens_classes = nc_str["verb_class"].values[0]
          if isinstance(old_tokens_classes, str):
            old_tokens_classes = int(old_tokens_classes)

        #print(old_tokens, old_tokens_classes)
        sent = repl_fn(sent, old_tokens, old_tokens_classes)
        #print(f"original sentence '{self.ref_captions[name][cap_idx]}', new sentence '{new_sent}', obtained by replacing '{old_tokens}'")
        #input()
      out = self.get_caption_outs(out, sent, self.ref_graphs[sent])
    else:
      video_idx = idx
      name = self.names[idx]

    if self.is_train and perform_aug and self.perform_vid_aug:
      # find the index for a similar video
      if coin_N_or_V == 0:
        video_source = self.video_noun_similars
      else:
        video_source = self.video_verb_similars

      valid_videos = video_source[video_idx]
      #print(valid_videos)
      #input()
      if len(valid_videos) > 0:
        other_video = valid_videos[randint(0, len(valid_videos)-1)]
        if self.load_video_first:
          first_attn_fts, first_attn_len = self.all_attn_fts[video_idx], self.all_attn_lens[video_idx]
          other_attn_fts, other_attn_len = self.all_attn_fts[other_video], self.all_attn_lens[other_video]
        else:
          first_attn_fts = self.load_attn_ft_by_name(name, self.attn_ft_files)
          first_attn_fts, first_attn_len = self.pad_or_trim_feature(first_attn_fts, self.max_attn_len, trim_type='select')
          other_name = self.names[other_video]
          other_attn_fts = self.load_attn_ft_by_name(other_name, self.attn_ft_files)
          other_attn_fts, other_attn_len = self.pad_or_trim_feature(other_attn_fts, self.max_attn_len, trim_type='select')

        attn_fts, attn_len = self.mix_fn(first_attn_fts, first_attn_len, other_attn_fts, other_attn_len)
      else:
        # can not do the mixing
        if self.load_video_first:
          attn_fts, attn_len = self.all_attn_fts[video_idx], self.all_attn_lens[video_idx]
        else:
          attn_fts = self.load_attn_ft_by_name(name, self.attn_ft_files)
          attn_fts, attn_len = self.pad_or_trim_feature(attn_fts, self.max_attn_len, trim_type='select')

    else:
      if self.load_video_first:
        attn_fts, attn_len = self.all_attn_fts[video_idx], self.all_attn_lens[video_idx]
      else:
        attn_fts = self.load_attn_ft_by_name(name, self.attn_ft_files)
        attn_fts, attn_len = self.pad_or_trim_feature(attn_fts, self.max_attn_len, trim_type='select')

    out['names'] = name
    out['attn_fts'] = attn_fts
    out['attn_lens'] = attn_len
    if self.is_train:
      out['threshold_pos'] = self.threshold_pos

    if self.dataset_file_path != '' and self.is_train:
      _key = name
      nc_str = self.dataset_file[_key]["annotations"][0]["noun_classes"]
      if isinstance(nc_str, str):
        nc_str = list(map(int, nc_str.replace("[", "").replace("]", "").split(",")))
      out['noun_classes'] = nc_str
      out['verb_class'] = self.dataset_file[_key]["annotations"][0]["verb_class"]

    return out


class VidNoiseAugmentedRoleGraphDataset(AugmentedRoleGraphDataset):
  # idea based on https://arxiv.org/abs/2004.03815,
  # code from https://github.com/danieljf24/cbvr/blob/master/data_augmenter.py
  def __init__(self, name_file, attn_ft_files, word2int_file,
               max_words_in_sent, num_verbs, num_nouns, ref_caption_file, ref_graph_file,
               max_attn_len=20, load_video_first=False, is_train=False, _logger=None,
               dataset_file='', is_test=False, dname="", rel_mat_path="", threshold_pos=1,
               verb_classes=None, noun_classes=None, video_verb_similars=None, video_noun_similars=None,
               aug_chance=0.5, original_dataframe="", fix_lambda=False, aug_chance_txt=-1.):
    super().__init__(name_file, attn_ft_files, word2int_file,
                     max_words_in_sent, num_verbs, num_nouns, ref_caption_file, ref_graph_file,
                     max_attn_len=max_attn_len, load_video_first=load_video_first, is_train=is_train, _logger=_logger,
                     dataset_file=dataset_file, is_test=is_test, dname=dname, rel_mat_path=rel_mat_path,
                     threshold_pos=threshold_pos,
                     verb_classes=verb_classes, noun_classes=noun_classes, video_verb_similars=video_verb_similars, video_noun_similars=video_noun_similars,
                     aug_chance=aug_chance, original_dataframe=original_dataframe, fix_lambda=fix_lambda, aug_chance_txt=aug_chance_txt)

    if self.is_train:
      self.n_dims = 3072  # assuming TBN features
      self.step_size = 500
      self.perturb_prob = 0.5
      self.perturb_intensity = 1
      self.mean, self.std = self.__get_mean_std()
      self.__init_mask()

  def __get_mean_std(self):
    mean = []
    std = []
    for i in range(0, self.n_dims, self.step_size):
      vec_list = []
      for idx in range(len(self.names)):
        video_idx, cap_idx = self.pair_idxs[idx]
        attn_fts, attn_len = self.all_attn_fts[video_idx], self.all_attn_lens[video_idx]
        #feat_vec = self.feat_reader.read_one(video)
        # using the subvec to accelerate calculation
        vec_list.append(np.mean(attn_fts[:, i:min(self.step_size + i, self.n_dims)], 0))
      mean.extend(np.mean(vec_list, 0))
      std.extend(np.std(vec_list, 0))
    return np.array(mean), np.array(std)

  def __init_mask(self):
    self.mask = np.zeros(self.n_dims)
    self.mask[:int(self.n_dims * self.perturb_prob)] = 1

  def __shuffle_mask(self):
    from random import shuffle
    shuffle(self.mask)

  def get_aug_feat(self, vid_feat):
    self.__shuffle_mask()
    perturbation = (np.random.randn(self.n_dims) * self.std + self.mean) * self.perturb_intensity * self.mask
    aug_feat = vid_feat + perturbation
    return aug_feat

  def mix_fn(self, vid1, vid1_att, vid2, vid2_att, alpha=1):
    #print(vid1[0, :5], self.get_aug_feat(vid1)[0, :5])
    return self.get_aug_feat(vid1), vid1_att


class MultisentAugmentedRoleGraphDataset(AugmentedRoleGraphDataset):
  def pick_alternative_sent(self, df, noun_classes, verb_class):
    data = df.query('noun_class in @noun_classes & verb_class == @verb_class')
    data = list(set(data.narration.values))
    from random import randint
    new_sent = data[randint(0, len(data) - 1)]
    return new_sent

  def __getitem__(self, idx):
    out = {}
    from random import randint
    coin_S = randint(0, 99)  # decide whether to change part of the 'sentence'
    perform_aug = coin_S >= (1 - self.aug_chance)*100
    if self.aug_chance_txt > 0:
      coin_S_txt = randint(0, 99)  # decide whether to change part of the 'sentence'
      perform_aug_txt = coin_S_txt >= (1 - self.aug_chance_txt)*100
    else:
      perform_aug_txt = perform_aug
    coin_N_or_V = randint(0, 1)  # if perform_aug, decide whether to change noun or verb (used both for txt and vid)

    if self.is_train:
      video_idx, cap_idx = self.pair_idxs[idx]
      name = self.names[video_idx]
      sent = self.ref_captions[name][cap_idx]
      out = self.get_caption_outs(out, sent, self.ref_graphs[sent])
      out_alt = {}
      if self.perform_txt_aug and perform_aug_txt:
        nc_str = self.original_dataframe[self.original_dataframe["narration_id"] == name]
        noun_classes = nc_str["all_noun_classes"].values[0]
        if isinstance(noun_classes, str):
          noun_classes = list(map(int, noun_classes.replace("[", "").replace("]", "").split(",")))
        verb_class = nc_str["verb_class"].values[0]
        if isinstance(verb_class, str):
          verb_class = int(verb_class)
        alt_sent = self.pick_alternative_sent(self.original_dataframe, noun_classes, verb_class)
        out_alt = self.get_caption_outs(out_alt, alt_sent, self.ref_graphs[alt_sent])
        #print(f"alt-sent of '{sent}' -> '{alt_sent}'") #; replaced in 'out' the keys: {out.keys()}; out['sent_lens']={out['sent_lens'][0]}")
      else:
        out_alt = out
      out = {k: (v, out_alt[k]) for k, v in out.items()}

    else:
      video_idx = idx
      name = self.names[idx]

    if self.is_train and perform_aug and self.perform_vid_aug:
      # find the index for a similar video
      if coin_N_or_V == 0:
        video_source = self.video_noun_similars
      else:
        video_source = self.video_verb_similars

      valid_videos = video_source[video_idx]
      #print(valid_videos)
      #input()
      if len(valid_videos) > 0:
        other_video = valid_videos[randint(0, len(valid_videos)-1)]
        if self.load_video_first:
          first_attn_fts, first_attn_len = self.all_attn_fts[video_idx], self.all_attn_lens[video_idx]
          other_attn_fts, other_attn_len = self.all_attn_fts[other_video], self.all_attn_lens[other_video]
        else:
          first_attn_fts = self.load_attn_ft_by_name(name, self.attn_ft_files)
          first_attn_fts, first_attn_len = self.pad_or_trim_feature(first_attn_fts, self.max_attn_len, trim_type='select')
          other_name = self.names[other_video]
          other_attn_fts = self.load_attn_ft_by_name(other_name, self.attn_ft_files)
          other_attn_fts, other_attn_len = self.pad_or_trim_feature(other_attn_fts, self.max_attn_len, trim_type='select')

        attn_fts, attn_len = self.mix_fn(first_attn_fts, first_attn_len, other_attn_fts, other_attn_len)
      else:
        # can not do the mixing
        if self.load_video_first:
          attn_fts, attn_len = self.all_attn_fts[video_idx], self.all_attn_lens[video_idx]
        else:
          attn_fts = self.load_attn_ft_by_name(name, self.attn_ft_files)
          attn_fts, attn_len = self.pad_or_trim_feature(attn_fts, self.max_attn_len, trim_type='select')

    else:
      if self.load_video_first:
        attn_fts, attn_len = self.all_attn_fts[video_idx], self.all_attn_lens[video_idx]
      else:
        attn_fts = self.load_attn_ft_by_name(name, self.attn_ft_files)
        attn_fts, attn_len = self.pad_or_trim_feature(attn_fts, self.max_attn_len, trim_type='select')

    out['names'] = name
    out['attn_fts'] = attn_fts
    out['attn_lens'] = attn_len
    if self.is_train:
      out['threshold_pos'] = self.threshold_pos

    if self.dataset_file_path != '' and self.is_train:
      _key = name
      nc_str = self.dataset_file[_key]["annotations"][0]["noun_classes"]
      if isinstance(nc_str, str):
        nc_str = list(map(int, nc_str.replace("[", "").replace("]", "").split(",")))
      out['noun_classes'] = nc_str
      out['verb_class'] = self.dataset_file[_key]["annotations"][0]["verb_class"]

    return out

class YC2AugRoleGraphDataset(AugmentedRoleGraphDataset):
  def __init__(self, name_file, attn_ft_files, word2int_file,
    max_words_in_sent, num_verbs, num_nouns, ref_caption_file, ref_graph_file,
    max_attn_len=20, load_video_first=False, is_train=False, _logger=None,
               dataset_file='', is_test=False, dname="", rel_mat_path="", threshold_pos=1,
               verb_classes=None, noun_classes=None, video_verb_similars=None, video_noun_similars=None,
               aug_chance=0.5, original_dataframe="", fix_lambda=False, aug_chance_txt=-1.):
    super().__init__(name_file, attn_ft_files, word2int_file,
                     max_words_in_sent, num_verbs, num_nouns, ref_caption_file, ref_graph_file,
                     max_attn_len=max_attn_len, load_video_first=load_video_first, is_train=is_train, _logger=_logger,
                     dataset_file=dataset_file, is_test=is_test, dname=dname, rel_mat_path=rel_mat_path, threshold_pos=threshold_pos)

    self.aug_chance = aug_chance
    self.aug_chance_txt = aug_chance_txt
    self.fix_lambda = fix_lambda
    self.perform_txt_aug = False
    self.perform_vid_aug = False
    if is_train:
      # noun/verb -> [synonym]
      # e.g. synonyms(classID-of-take) = [pick up, grab, ...]
      import pandas as pd

      if video_verb_similars is not None:
        self.perform_vid_aug = True
        # video_name -> [different video_name with shared verb/noun]
        # e.g. similars(vidX) = [vid2, vid18, ...] where vidX,vid2,... share the same verb/noun
        if os.path.exists(video_verb_similars):
          self.video_verb_similars = json.load(open(video_verb_similars))
          self.video_noun_similars = json.load(open(video_noun_similars))
        else:
          self.video_verb_similars = [set() for i in range(len(self.names))]
          self.video_noun_similars = [set() for i in range(len(self.names))]

          def uncoherent_check(vc1, vc2, nc1, nc2):
            # only checks that v1 and v2 are sharing some verb (or noun) classes
            if not isinstance(vc1, set):
              vc1 = set(vc1)
            if not isinstance(vc2, set):
              vc2 = set(vc2)
            return len(vc1.intersection(vc2)) > 0

          def coherent_check(vc1, vc2, nc1, nc2):
            # checks both that v1 and v2 are sharing some verb (or noun) classes,
            # and that the same holds for noun (or verb) classes
            if not isinstance(nc1, set):
              nc1 = set(nc1)
            if not isinstance(nc2, set):
              nc2 = set(nc2)
            return len(nc1.intersection(nc2)) > 0 and uncoherent_check(vc1, vc2, nc1, nc2)

          if "coherent" in video_noun_similars:
            test_fn = coherent_check
          else:
            test_fn = uncoherent_check

          from tqdm import tqdm
          for i, name in tqdm(enumerate(self.names)):
            # 1) collect classes for this video
            vid_verb_classes = []
            vid_noun_classes = []
            
            for j, sent in enumerate(self.ref_captions[name]):
              vc = self.sent2classes[sent]["verb_class"]
              if not isinstance(vc, list):
                vc = [vc]
              vid_verb_classes += vc
              nc_str = self.sent2classes[sent]["noun_classes"]
              if isinstance(nc_str, str):
                nc_str = list(map(int, nc_str.replace("[", "").replace("]", "").split(",")))
              vid_noun_classes += nc_str
            vid_verb_classes = set(vid_verb_classes)
            vid_noun_classes = set(vid_noun_classes)

            # 2) find videos with shared classes
            for i2, name2 in enumerate(self.names):
              if i2 > i:
                for j2, sent2 in enumerate(self.ref_captions[name2]):
                  vc2 = self.sent2classes[sent2]["verb_class"]
                  nc_str = self.sent2classes[sent2]["noun_classes"]
                  if isinstance(nc_str, str):
                    nc_str = list(map(int, nc_str.replace("[", "").replace("]", "").split(",")))
                  if not isinstance(vc2, list):
                    vc2 = [vc2]
                  if test_fn(vid_verb_classes, vc2, vid_noun_classes, nc_str):  #len(set(vc2).intersection(vid_verb_classes)) > 0:
                    self.video_verb_similars[i].add(i2)
                    self.video_verb_similars[i2].add(i)
                  if test_fn(vid_noun_classes, nc_str, vid_verb_classes, vc2):  #len(set(nc_str).intersection(vid_noun_classes)) > 0:
                    self.video_noun_similars[i].add(i2)
                    self.video_noun_similars[i2].add(i)

          tmp = []
          for v in self.video_verb_similars:
            tmp.append(list(v))
          json.dump(tmp, open(video_verb_similars, "w"))
          self.video_verb_similars = tmp
          tmp = []
          for v in self.video_noun_similars:
            tmp.append(list(v))
          json.dump(tmp, open(video_noun_similars, "w"))
          self.video_noun_similars = tmp

        print(f"--- Loaded dictionaries for similar videos: {len(self.video_noun_similars)} nouns, {len(self.video_verb_similars)} verbs ---")

      if original_dataframe != "":
        # want to precompute the vid->sentence_candidates
        parse = lambda s, t: s.replace("[", "").replace("]", "").replace("'", "").replace(" ", "").replace(t, " ").split(",")
        self.parse_fn = parse

        max_nc, max_vc = 0, 0
        for snt, clss in self.sent2classes.items():
          nc = clss["noun_classes"]
          vc = clss["verb_class"]
          if len(nc) > 0:
            max_nc = max(max_nc, max(nc))
          if len(vc) > 0:
            max_vc = max(max_vc, max(vc))

        self.verbcls2sents = [[] for i in range(max_vc+1)]
        self.nouncls2sents = [[] for i in range(max_nc+1)]
        for snt, clss in self.sent2classes.items():
          nc = clss["noun_classes"]
          vc = clss["verb_class"]
          for v in vc:
            self.verbcls2sents[v].append(snt)
          for n in nc:
            self.nouncls2sents[n].append(snt)

        self.original_dataframe = pd.read_csv(original_dataframe)
        self.perform_txt_aug = True
        print(f"--- Loaded original dataframe: {len(self.original_dataframe)} rows ---")

  def _pick_alternative_sent(self, important_class, _type, shared_classes):
    # we may pick a sentence from all the captions in the dataset, and those sharing at least ...
    if _type == "noun":
      src_sents = self.nouncls2sents
      oth_sents = self.verbcls2sents
      cl_imp, cl_shared = "all_noun_classes", "verb_class"
    else:
      src_sents = self.verbcls2sents
      oth_sents = self.nouncls2sents
      cl_imp, cl_shared = "verb_class", "all_noun_classes"
    all_cand = []

    for sh_cls in shared_classes:
      #print(f"query: {important_class} in {cl_imp} & {sh_cls} in {cl_shared}")
      all_cand.extend(list(set(src_sents[important_class]).intersection(set(oth_sents[sh_cls]))))
      #all_cand.append(self.original_dataframe.query('@important_class in @cl_imp & @sh_cls in @cl_shared').narration.values)
    #print(all_cand)
    #input()
    all_cand = list(np.unique(np.array(all_cand)))
    from random import randint
    if len(all_cand) > 0:
      new_sent = all_cand[randint(0, len(all_cand) - 1)]
      return new_sent
    return ""

  def pick_alternative_sent_verb(self, verb_class, noun_classes):
    return self._pick_alternative_sent(verb_class, "verb", noun_classes)

  def pick_alternative_sent_noun(self, noun_class, verb_classes):
    return self._pick_alternative_sent(noun_class, "noun", verb_classes)


  def _pick_alternative_sent_lesser(self, _type, important_class=None, shared_classes=None):
    # one of the two (N/V) is empty, so we only have one set of classes to use
    if _type == "noun":
      src_sents = self.nouncls2sents
      oth_sents = self.verbcls2sents
      cl_imp, cl_shared = "all_noun_classes", "verb_class"
    else:
      src_sents = self.verbcls2sents
      oth_sents = self.nouncls2sents
      cl_imp, cl_shared = "verb_class", "all_noun_classes"
    
    if important_class is not None:
        all_cand = list(set(src_sents[important_class]))
    else:
        all_cand = []
        for sh_cls in shared_classes:
            all_cand.extend(oth_sents[sh_cls])
  
    all_cand = list(np.unique(np.array(all_cand)))
    from random import randint
    if len(all_cand) > 0:
      new_sent = all_cand[randint(0, len(all_cand) - 1)]
      return new_sent
    return ""


  def __getitem__(self, idx):
    out = {}
    from random import randint
    coin_S = randint(0, 99)  # decide whether to change part of the 'sentence'
    perform_aug = coin_S >= (1 - self.aug_chance)*100
    if self.aug_chance_txt > 0:
      coin_S_txt = randint(0, 99)  # decide whether to change part of the 'sentence'
      perform_aug_txt = coin_S_txt >= (1 - self.aug_chance_txt)*100
    else:
      perform_aug_txt = perform_aug
    coin_N_or_V = randint(0, 1)  # if perform_aug, decide whether to change noun or verb (used both for txt and vid)

    if self.is_train:
      video_idx, cap_idx = self.pair_idxs[idx]
      name = self.names[video_idx]
      sent = self.ref_captions[name][cap_idx]
      out = self.get_caption_outs(out, sent, self.ref_graphs[sent])
      out_alt = {}
      if self.perform_txt_aug and perform_aug_txt:
        # nc_str contains all the sentences' data related to the video
        nc_str = self.original_dataframe[self.original_dataframe["narration_id"].str.contains(name.replace(".mp4", ""))]
        noun_classes = nc_str["all_noun_classes"].values  # list of string! (each string->list of classes for that sentence)
        verb_class = nc_str["verb_class"].values
        if isinstance(noun_classes[0], str):
          noun_classes = [list(map(int, nc.replace("[", "").replace("]", "").split(","))) for nc in noun_classes if nc != '[]']
        if isinstance(verb_class[0], str):
          verb_class = [list(map(int, nc.replace("[", "").replace("]", "").split(","))) for nc in verb_class if nc != '[]']
        if coin_N_or_V == 0:
          #print(noun_classes)
          if len(noun_classes) > 0 and len(verb_class) > 0:
            chosen_noun_classes = noun_classes[randint(0, len(noun_classes)-1)]
            alt_sent = self.pick_alternative_sent_noun(chosen_noun_classes[randint(0, len(chosen_noun_classes)-1)], verb_class[randint(0, len(verb_class)-1)])
          elif len(noun_classes) > 0:
            chosen_noun_classes = noun_classes[randint(0, len(noun_classes)-1)]
            alt_sent = self._pick_alternative_sent_lesser("noun", important_class=chosen_noun_classes[randint(0, len(chosen_noun_classes)-1)])
          elif len(verb_class) > 0:
            alt_sent = self._pick_alternative_sent_lesser("noun", shared_classes=verb_class[randint(0, len(verb_class)-1)])
          else:
            alt_sent = sent
        else:
          #print(verb_class)
          if len(verb_class) > 0 and len(noun_classes) > 0:
            chosen_verb_classes = verb_class[randint(0, len(verb_class)-1)]
            alt_sent = self.pick_alternative_sent_verb(chosen_verb_classes[randint(0, len(chosen_verb_classes)-1)], noun_classes[randint(0, len(noun_classes)-1)])
          elif len(verb_class) > 0:
            chosen_verb_classes = verb_class[randint(0, len(verb_class)-1)]
            alt_sent = self._pick_alternative_sent_lesser("verb", important_class=chosen_verb_classes[randint(0, len(chosen_verb_classes)-1)])
          elif len(noun_classes) > 0:
            alt_sent = self._pick_alternative_sent_lesser("verb", shared_classes=noun_classes[randint(0, len(noun_classes)-1)])
          else:
            alt_sent = sent
        if alt_sent == "":
          alt_sent = sent
        out_alt = self.get_caption_outs(out_alt, alt_sent, self.ref_graphs[alt_sent])
        """print(f"alt-sent of '{sent}' (nouns {noun_classes[cap_idx]}, verbs {verb_class[cap_idx]}) "
              f"-> '{alt_sent}'") #; replaced in 'out' the keys: {out.keys()}; out['sent_lens']={out['sent_lens'][0]}")"""
      else:
        out_alt = out
      out = {k: (v, out_alt[k]) for k, v in out.items()}

    else:
      video_idx = idx
      name = self.names[idx]

    if self.is_train and perform_aug and self.perform_vid_aug:
      # find the index for a similar video
      if coin_N_or_V == 0:
        video_source = self.video_noun_similars
      else:
        video_source = self.video_verb_similars

      valid_videos = video_source[video_idx]
      #print(valid_videos)
      #input()
      if len(valid_videos) > 0:
        other_video = valid_videos[randint(0, len(valid_videos)-1)]
        if self.load_video_first:
          first_attn_fts, first_attn_len = self.all_attn_fts[video_idx], self.all_attn_lens[video_idx]
          other_attn_fts, other_attn_len = self.all_attn_fts[other_video], self.all_attn_lens[other_video]
        else:
          first_attn_fts = self.load_attn_ft_by_name(name, self.attn_ft_files)
          first_attn_fts, first_attn_len = self.pad_or_trim_feature(first_attn_fts, self.max_attn_len, trim_type='select')
          other_name = self.names[other_video]
          other_attn_fts = self.load_attn_ft_by_name(other_name, self.attn_ft_files)
          other_attn_fts, other_attn_len = self.pad_or_trim_feature(other_attn_fts, self.max_attn_len, trim_type='select')

        attn_fts, attn_len = self.mix_fn(first_attn_fts, first_attn_len, other_attn_fts, other_attn_len)
      else:
        # can not do the mixing
        if self.load_video_first:
          attn_fts, attn_len = self.all_attn_fts[video_idx], self.all_attn_lens[video_idx]
        else:
          attn_fts = self.load_attn_ft_by_name(name, self.attn_ft_files)
          attn_fts, attn_len = self.pad_or_trim_feature(attn_fts, self.max_attn_len, trim_type='select')

    else:
      if self.load_video_first:
        attn_fts, attn_len = self.all_attn_fts[video_idx], self.all_attn_lens[video_idx]
      else:
        attn_fts = self.load_attn_ft_by_name(name, self.attn_ft_files)
        attn_fts, attn_len = self.pad_or_trim_feature(attn_fts, self.max_attn_len, trim_type='select')

    out['names'] = name
    out['attn_fts'] = attn_fts
    out['attn_lens'] = attn_len
    if self.is_train:
      out['threshold_pos'] = self.threshold_pos

    if self.dataset_file_path != '' and self.is_train:
      _key = name
      nc_str = self.dataset_file[_key]["annotations"][0]["noun_classes"]
      if isinstance(nc_str, str):
        nc_str = list(map(int, nc_str.replace("[", "").replace("]", "").split(",")))
      out['noun_classes'] = nc_str
      out['verb_class'] = self.dataset_file[_key]["annotations"][0]["verb_class"]

    return out


def collate_graph_fn(data):
  outs = {}
  for key in ['names', 'attn_fts', 'attn_lens', 'sent_ids', 'sent_lens',
              'verb_masks', 'noun_masks', 'node_roles', 'rel_edges']:
    if key in data[0]:
      outs[key] = [x[key] for x in data]

  batch_size = len(data)

  # reduce attn_lens
  if 'attn_fts' in outs:
    max_len = np.max(outs['attn_lens'])
    outs['attn_fts'] = np.stack(outs['attn_fts'], 0)[:, :max_len]

  # reduce caption_ids lens
  if 'sent_lens' in outs:
    max_cap_len = np.max(outs['sent_lens'])
    outs['sent_ids'] = np.array(outs['sent_ids'])[:, :max_cap_len]
    outs['verb_masks'] = np.array(outs['verb_masks'])[:, :, :max_cap_len]
    outs['noun_masks'] = np.array(outs['noun_masks'])[:, :, :max_cap_len]

  if 'noun_classes' in data[0]:
    outs['noun_classes'] = [x['noun_classes'] for x in data]
    outs['verb_class'] = [x['verb_class'] for x in data]

  if 'threshold_pos' in data[0]:
    outs["threshold_pos"] = data[0]["threshold_pos"]

  if 'video_noun_classes' in data[0]:
    outs["video_noun_classes"] = [x['video_noun_classes'] for x in data]
    outs["video_verb_classes"] = [x['video_verb_classes'] for x in data]

  if 'action_classes' in data[0]:
    outs["action_classes"] = [x['action_classes'] for x in data]
    outs["entity_classes"] = [x['entity_classes'] for x in data]
  return outs

def collate_graph_fn_multisent(data):
  outs = {}
  for key in ['names', 'attn_fts', 'attn_lens', 'sent_ids', 'sent_lens',
              'verb_masks', 'noun_masks', 'node_roles', 'rel_edges']:
    if key in data[0]:
      outs[key] = [x[key] for x in data]

  batch_size = len(data)

  #print("collate", outs.keys())
  if not "sent_lens" in outs.keys(): # and not isinstance(outs['sent_lens'][0], tuple):
    return collate_graph_fn(data)

  # reduce attn_lens
  if 'attn_fts' in outs:
    max_len = np.max(outs['attn_lens'])
    outs['attn_fts'] = np.stack(outs['attn_fts'], 0)[:, :max_len]

  # reduce caption_ids lens
  if 'sent_lens' in outs:
    max_cap_len_0 = np.max([a[0] for a in outs['sent_lens']])
    max_cap_len_1 = np.max([a[1] for a in outs['sent_lens']])
    outs['sent_ids'] = (np.array(outs['sent_ids'])[:, 0, :max_cap_len_0],
                        np.array(outs['sent_ids'])[:, 1, :max_cap_len_1])
    # (B, 2, N) since we have two sentences
    outs['verb_masks'] = (np.array(outs['verb_masks'])[:, 0, :, :max_cap_len_0],
                          np.array(outs['verb_masks'])[:, 1, :, :max_cap_len_1])
    outs['noun_masks'] = (np.array(outs['noun_masks'])[:, 0, :, :max_cap_len_0],
                          np.array(outs['noun_masks'])[:, 1, :, :max_cap_len_1])
    #print(outs['sent_ids'].shape, outs['noun_masks'].shape)

  if 'noun_classes' in data[0]:
    outs['noun_classes'] = [x['noun_classes'] for x in data]
    outs['verb_class'] = [x['verb_class'] for x in data]

  if 'threshold_pos' in data[0]:
    outs["threshold_pos"] = data[0]["threshold_pos"]

  if 'video_noun_classes' in data[0]:
    outs["video_noun_classes"] = [x['video_noun_classes'] for x in data]
    outs["video_verb_classes"] = [x['video_verb_classes'] for x in data]

  if 'action_classes' in data[0]:
    outs["action_classes"] = [x['action_classes'] for x in data]
    outs["entity_classes"] = [x['entity_classes'] for x in data]
  return outs
