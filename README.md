# A Feature-space Multimodal Data Augmentation Technique for Text-video Retrieval
In this repo, we provide code and pretrained models for the paper ["**A Feature-space Multimodal Data Augmentation Technique for Text-video Retrieval**"](https://arxiv.org/abs/2208.02080) which has been accepted for presentation at the [**30th ACM International Conference on Multimedia (ACM MM)**](https://2022.acmmm.org/).

![Overview of the proposed multimodal data augmentation technique working on latent representations.](https://github.com/aranciokov/FSMMDA_VideoRetrieval/blob/images/teaser.png?raw=true)

#### Python environment
Requirements: python 3, allennlp 2.8.0, h5py 3.6.0, pandas 1.3.5, spacy 2.3.5, torch 1.7.0 (also tested with 1.8)
```
# clone the repository
cd FSMMDA_VideoRetrieval
export PYTHONPATH=$(pwd):${PYTHONPATH}
```

#### Data
- Features: 
    - TBN **EPIC-Kitchens-100** [**features**](https://drive.google.com/file/d/16_WXNg2aziVBsWjc1_egE4YjnJ_aKbrM/view?usp=sharing) from [JPoSE's repo](https://github.com/mwray/Joint-Part-of-Speech-Embeddings). 
    - S3D **YouCook2** [**features**](https://drive.google.com/file/d/1Bp-uY_rFvNv3f6TrD8TfGHrfB-dwkywy/view?usp=sharing) from the [VALUE benchmark](https://value-benchmark.github.io/).
- Additional:
    - pre-extracted annotations for [EPIC-Kitchens-100](https://drive.google.com/file/d/1XiRE-dF7EHqouWx8oHptODiQuYCHAqh6/view?usp=sharing) and [YouCook2](https://drive.google.com/file/d/19FP8fWpGiv_y9iewDcYKIOYck8Fjy-rw/view?usp=sharing)
    - split folders for [EPIC-Kitchens-100](https://drive.google.com/file/d/1eYxzyCb2Jl0oeHP_y2awZhTTNz5th7X2/view?usp=sharing) and [YouCook2](https://drive.google.com/file/d/1CZTpMer2eHHC6vxl-gCcs4lc9HPc3Fjw/view?usp=sharing)
    - GloVe checkpoints for [EPIC-Kitchens-100](https://drive.google.com/file/d/1q7viOUp_kByPc3-y8PIZw1A7BZcLdtAD/view?usp=sharing) and [YouCook2](https://drive.google.com/file/d/1p2Nhvd6XJwXoc8d01fkmirH6nfLnjlpw/view?usp=sharing)

#### Training
To launch a training, first select a configuration file (e.g. ``prepare_mlmatch_configs_EK100_TBN_augmented_VidTxtLate.py``) and execute the following:

``python t2vretrieval/driver/configs/prepare_mlmatch_configs_EK100_TBN_augmented_VidTxtLate.py .``

This will return a folder name (where config, models, logs, etc will be saved). Let that folder be ``$resdir``. Then, execute the following to start a training:

``python t2vretrieval/driver/multilevel_match.py $resdir/model.json $resdir/path.json --is_train --load_video_first --resume_file glove_checkpoint_path``

#### About the config files
Config files are used to define details of the model and of the paths containing the annotations, features, etc. By running a "prepare_*" script, a folder containing two .json files is created. 
- HGR baseline: *prepare_mlmatch_configs_EK100_TBN_baseline.py*
- Coarse-grained video selection with variable lambda (λ~β(1, 1)): *prepare_mlmatch_configs_EK100_TBN_augmented_Vid_coarse*
- Fine-grained video selection with fixed lambda (λ=0.5): *prepare_mlmatch_configs_EK100_TBN_augmented_fixLambda.py*
- Video augmentation by noise addition (Dong et al.): *prepare_mlmatch_configs_EK100_TBN_augmented_thrPos_VidNoise.py* 
- Text augmentation by synonym replacement: *prepare_mlmatch_configs_EK100_TBN_augmented_Txt.py*
- Video augmentation by the proposed feature-space technique: *prepare_mlmatch_configs_EK100_TBN_augmented_Vid.py*
- Text augmentation by the proposed feature-space technique: *prepare_mlmatch_configs_EK100_TBN_augmented_TxtLate.py*
- **Augmentation by the proposed feature-space multi-modal technique**: *prepare_mlmatch_configs_EK100_TBN_augmented_VidTxtLate.py*
- Cooperation of the proposed FSMMDA with RAN: *prepare_mlmatch_configs_EK100_TBN_augmented_thrPos_VidTxtLate.py*
- Cooperation of the proposed FSMMDA with RANP: *prepare_mlmatch_configs_EK100_TBN_augmented_thrPos_HP_VidTxtLate.py*
- HGR baseline on YouCook2: *prepare_mlmatch_configs_YC2-S3D.py*
- **Augmentation by the proposed feature-space multi-modal technique** on YouCook2: *prepare_mlmatch_configs_YC2_augVidTxt-S3D.py*

#### Evaluating
To automatically check for the best checkpoint (after a training run):

``python t2vretrieval/driver/multilevel_match.py $resdir/model.json $resdir/path.json --eval_set tst``

To resume one of the checkpoints provided:

``python t2vretrieval/driver/multilevel_match.py $resdir/model.json $resdir/path.json --eval_set tst --resume_file checkpoint.th``

For instance, by unzipping the archive for the augmented HGR on EPIC-Kitchens-100, the following folder is obtained:
```
results/RET.released/mlmatch/ek100_TBN_aug0.5_VcTLate_thrPos0.15_mPos0.2_m0.2.vis.TBN.pth.txt.bigru.16role.gcn.1L.attn.1024.loss.bi.af.embed.4.glove.init.50ep/
```

Therefore, the evaluation can be done by running the following:
```
python t2vretrieval/driver/multilevel_match.py \
  results/RET.released/mlmatch/ek100_TBN_aug0.5_VcTLate_thrPos0.15_mPos0.2_m0.2.vis.TBN.pth.txt.bigru.16role.gcn.1L.attn.1024.loss.bi.af.embed.4.glove.init.50ep/model.json \
  results/RET.released/mlmatch/ek100_TBN_aug0.5_VcTLate_thrPos0.15_mPos0.2_m0.2.vis.TBN.pth.txt.bigru.16role.gcn.1L.attn.1024.loss.bi.af.embed.4.glove.init.50ep/path.json \
  --eval_set tst \
  --resume_file results/RET.released/mlmatch/ek100_TBN_aug0.5_VcTLate_thrPos0.15_mPos0.2_m0.2.vis.TBN.pth.txt.bigru.16role.gcn.1L.attn.1024.loss.bi.af.embed.4.glove.init.50ep/model/epoch.42.th
```

#### Pretrained models
*On EPIC-Kitchens-100:*
- Baseline model (HGR): [(35.9 nDCG, 39.5 mAP)](https://drive.google.com/file/d/1uIiUVQhrfI3GBXmNpr8jQNNI6NEWPqdU/view?usp=sharing) 
- Augmented HGR with the proposed **FSMMDA**: [thr=0.15 (59.3 nDCG, 47.1 mAP)](https://drive.google.com/file/d/1P22GZFFh_RkkTHn-KnuoDfZVK1v4DGKz/view?usp=sharing)

*On YouCook2:*
- Baseline model: [(49.9 nDCG, 44.6 mAP)](https://drive.google.com/file/d/1ghq-xmmmW3vbwF4rLMGaCnMTX8cRFvN7/view?usp=sharing) 
- With the proposed **FSMMDA**: [(51.0 nDCG, 44.7 mAP)](https://drive.google.com/file/d/1MGFg7hCFvj25r-3f8J8XuhQgYtrctcAX/view?usp=sharing)

#### Acknowledgements
We thank the authors of 
 [Chen et al. (CVPR, 2020)](https://arxiv.org/abs/2003.00392) ([github](https://github.com/cshizhe/hgr_v2t)),
 [Wray et al. (ICCV, 2019)](https://openaccess.thecvf.com/content_ICCV_2019/papers/Wray_Fine-Grained_Action_Retrieval_Through_Multiple_Parts-of-Speech_Embeddings_ICCV_2019_paper.pdf) ([github](https://github.com/mwray/Joint-Part-of-Speech-Embeddings)),
 [Wray et al. (CVPR, 2021)](https://arxiv.org/abs/2103.10095) ([github](https://github.com/mwray/Semantic-Video-Retrieval)),
 [Falcon et al. (ICIAP, 2022)](https://arxiv.org/abs/2203.08688) ([github](https://github.com/aranciokov/ranp))
 for the release of their codebases. 

## Citations
If you use this code as part of any published research, we'd really appreciate it if you could cite the following paper:
```text
@article{falcon2022fsmmda,
  title={A Feature-space Multimodal Data Augmentation Technique for Text-video Retrieval},
  author={Falcon, Alex and Serra, Giuseppe and Lanz, Oswald},
  journal={ACM MM},
  year={2022}
}
```

## License

MIT License
