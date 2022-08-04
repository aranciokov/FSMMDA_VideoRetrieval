import os
import argparse
import json

#from allennlp.predictors.predictor import Predictor

def replace_noun(sent, old_tks, old_tks_cls, noun_synonyms, new_tok, s=":"):
  return replace_tk(sent, old_tks, old_tks_cls, noun_synonyms, new_tok, s)


def replace_verb(sent, old_tks, old_tks_cls, verb_synonyms, new_tok, s="-"):
  return replace_tk(sent, old_tks, old_tks_cls, verb_synonyms, new_tok, s)


def replace_tk(sent, old_tks, old_tks_cls, new_tks_source, new_tks, s):
  # print(f"replacing '{old_tks.replace(s, ' ')}' ({old_tks_cls}) with '{new_tks}' within '{sent}'")
  if old_tks.replace(s, ' ') in sent:
    return sent.replace(old_tks.replace(s, ' '), new_tks), 0

  old_tokens = old_tks.strip().split(s)
  new_tokens = new_tks.strip().split(s)
  try:
    first_occ = min([sent.split().index(_tk) for _tk in old_tokens])
    new_sent = [stk for stk in sent.split() if stk not in old_tokens]
    new_sent = new_sent[:first_occ] + new_tokens + new_sent[first_occ:]
    return " ".join(new_sent), 0
  except:
    print(f"replace_tk failed on {sent} ({old_tks} -> {new_tks})")
    #input()
    return sent, 1

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('out_file')
  parser.add_argument('--cuda_device', default=-1, type=int)
  opts = parser.parse_args()

  # predictor = Predictor.from_path('https://storage.googleapis.com/allennlp-public-models/structured-prediction-srl-bert.2020.12.15.tar.gz', cuda_device=opts.cuda_device)
  # bert-base-srl-2020.11.19.tar.gz'
  # original "https://s3-us-west-2.amazonaws.com/allennlp/models/bert-base-srl-2019.06.17.tar.gz"
  from allennlp_models import pretrained
  predictor = pretrained.load_predictor("structured-prediction-srl-bert", cuda_device=opts.cuda_device)

  uniq_sents = set()
  from tqdm import tqdm
  import pandas as pd
  parse = lambda s, t: s.replace("[", "").replace("]", "").replace("'", "").replace(" ", "").replace(t, " ").split(
    ",")
  tmp = pd.read_csv("annotation/epic100RET/EPIC_100_verb_classes.csv")
  verb_synonyms = tmp.instances.apply(
    lambda s: parse(s, '-')).values  # [parse(r["instances"], '-') for i, r in tmp.iterrows()]
  tmp = pd.read_csv("annotation/epic100RET/EPIC_100_noun_classes.csv")
  noun_synonyms = tmp.instances.apply(
    lambda s: parse(s, ':')).values  # [parse(r["instances"], ':') for i, r in tmp.iterrows()]
  original_dataframe = pd.read_csv("annotation/epic100RET/EPIC_100_retrieval_train.csv")
  errs = 0
  _tot = 0
  for ix, row in tqdm(original_dataframe.iterrows()):
    sent = row["narration"]
    uniq_sents.add(sent)
    if isinstance(row["all_noun_classes"], str):
      nc_cl = list(map(int, row["all_noun_classes"].replace("[", "").replace("]", "").split(",")))
    if isinstance(row["all_nouns"], str):
      nc_tk = row["all_nouns"].replace("[", "").replace("]", "").replace("'", "").split(",")
    for nc, nn in zip(nc_cl, nc_tk):
      for cand in noun_synonyms[nc]:
        new_sent, ok_ko = replace_noun(sent, nn, nc, noun_synonyms, cand)
        errs += ok_ko
        #print(nc, nn, cand)
        uniq_sents.add(new_sent)
        _tot += 1

    nc_cl = int(row["verb_class"])
    nc_tk = row["verb"]
    for cand in verb_synonyms[nc_cl]:
      new_sent, ok_ko = replace_verb(sent, nc_tk, nc_cl, verb_synonyms, cand)
      errs += ok_ko
      #print(nc_cl, nc_tk, cand)
      uniq_sents.add(new_sent)
      _tot += 1

  print(f"errors -> {errs} (out of {_tot})")
  uniq_sents = list(uniq_sents)
  print('unique sents', len(uniq_sents))

  outs = {}
  if os.path.exists(opts.out_file):
    outs = json.load(open(opts.out_file))
  for i, sent in tqdm(enumerate(uniq_sents)):
    if sent in outs:
      continue
    try:
      out = predictor.predict_tokenized(sent.split())
    except KeyboardInterrupt:
      break
    except:
      continue
    outs[sent] = out
    if i % 1000 == 0:
      print('finish %d / %d = %.2f%%' % (i, len(uniq_sents), i / len(uniq_sents) * 100))

  with open(opts.out_file, 'w') as f:
    json.dump(outs, f)

if __name__ == '__main__':
  main()
