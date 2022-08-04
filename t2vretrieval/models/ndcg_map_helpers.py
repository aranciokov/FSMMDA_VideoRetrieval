import numpy as np
import torch as th


def calculate_mAP(sim_mat, relevancy_matrix, diff_rels=None):
    """
    Computes the mean average precision according to the following formula of
    average precision:
    \frac{\sum_{k=1}^n p(k) x rel(k)}{num_rel_docs}

    where p(k) is the precision at k, rel(k) is an indicator function
    determining whether the kth returned item is relevant or not and
    num_rel_docs is the number of relevant items to find within the search.

    The mean average precision is the mean of the average precision for each
    query item (i.e row in the matrix)

    This function takes in two parameters:
        - sim_mat: a NxM matrix which represents the similarity between two
        modalities (with modality 1 being of size N and modality 2 of size M).
        - relevancy_matrix: an NxM matrix which represents the relevancy between two
        modalities of items (with modality 1 being of size N and modality 2 of
        size M).
    """
    #Find the order of the items in modality 2 according to modality 1
    ranked_order = (-sim_mat).argsort()
    ranked_sim_mat = sim_mat[np.arange(sim_mat.shape[0])[:, None], ranked_order]
    #re-order the relevancy matrix to accommodate the proposals
    ranked_rel_mat = relevancy_matrix[np.arange(relevancy_matrix.shape[0])[:, None], ranked_order]

    #find the number of relevant items found at each k
    cumulative_rel_mat = np.cumsum(ranked_rel_mat, axis=1)
    #Mask this ensuring that it is non zero if the kth term is 1 (rel(k) above)
    if diff_rels is None:
        cumulative_rel_mat[ranked_rel_mat != 1] = 0
    else:
        print(ranked_rel_mat.shape, diff_rels.shape)
        cumulative_rel_mat[ranked_rel_mat < diff_rels] = 0

    #find the divisor for p(k)
    divisor = np.arange(ranked_rel_mat.shape[1]) + 1

    #find the number of relevant docs per query item
    if diff_rels is None:
        number_rel_docs = np.sum(ranked_rel_mat == 1, axis=1)
    else:
        number_rel_docs = np.sum(ranked_rel_mat >= diff_rels, axis=1)

    #find the average precision per query, within np.sum finds p(k) * rel(k)
    avg_precision = np.sum(cumulative_rel_mat / divisor, axis=1) / np.maximum(number_rel_docs, 1)
    mAP = np.mean(avg_precision)
    return mAP


def relevance(v_i, v_j, N_i, N_j):
  assert not isinstance(N_i, str) and not isinstance(N_j, str)
  verb_iou = float(v_i == v_j) if not isinstance(v_i, list) \
    else len(set(v_i).intersection(set(v_j))) / max(1, len(set(v_i).union(set(v_j))))
  noun_iou = len(set(N_i).intersection(set(N_j))) / max(1, len(set(N_i).union(set(N_j))))
  return 0.5 * (verb_iou + noun_iou)

def get_relevances_single_caption(batch_verbs=None, batch_nouns=None, rel_f=relevance):
  if batch_verbs is None:
    rel_mat = th.tensor([[rel_f(0, 0, n1, n2) for n2 in batch_nouns]
                            for n1 in batch_nouns])
  elif batch_nouns is None:
    rel_mat = th.tensor([[rel_f(v1, v2, [0], [0]) for v2 in batch_verbs]
                            for v1 in batch_verbs])
  else:
    rel_mat = th.tensor([[rel_f(v1, v2, n1, n2) for v2, n2 in zip(batch_verbs, batch_nouns)]
                            for v1, n1 in zip(batch_verbs, batch_nouns)])
  return rel_mat

def get_relevances_multi_caption(video_verbs=None, video_nouns=None,
                                 batch_verbs=None, batch_nouns=None, rel_f=relevance):
  if batch_verbs is None:
    rel_mat = th.tensor([[rel_f(0, 0, n1, n2) for n2 in batch_nouns]
                            for n1 in video_nouns])
  elif batch_nouns is None:
    rel_mat = th.tensor([[rel_f(v1, v2, [0], [0]) for v2 in batch_verbs]
                            for v1 in video_verbs])
  else:
    rel_mat = th.tensor([[rel_f(v1, v2, n1, n2) for v2, n2 in zip(batch_verbs, batch_nouns)]
                            for v1, n1 in zip(video_verbs, video_nouns)])
  return rel_mat

def calculate_DCG(similarity_matrix, relevancy_matrix, k_counts):
    """
    Calculates the Discounted Cumulative Gain (DCG) between two modalities for
    the first modality.
    DCG = \sum_{i=1}^k \frac{rel_i}{log_2(i + 1)}
    i.e. the sum of the k relevant retrievals which is calculated as the scaled
    relevancy for the ith item. The scale is designed such that early
    retrievals are more important than later retrievals.
    Params:
        - similarity_matrix: matrix of size n1 x n2 where n1 is the number of
          items in the first modality and n2 is the number of items in the
          second modality. The [ith,jth] element is the predicted similarity
          between the ith item from the first modality and the jth item from
          the second modality.
        - relevancy_matrix: matrix of size n1 x n2 (see similarity_matrix
          above). The [ith, jth] element is the semantic relevancy between the
          ith item from the first modality and the jth item from the second
          modality.
        - k_counts: matrix of size n1 x n2 (see similarity_matrix above) which
          includes information on which items to use to calculate the DCG for
          (see calculate_k_counts for more info on this matrix).
    Returns:
        - The DCG for each item in the first modality, a n1 length vector.
    """
    x_sz, y_sz = similarity_matrix.shape
    ranks = np.argsort(similarity_matrix)[:, ::-1]
    #Create vector of size (n,) where n is the length of the last dimension in
    #similarity matrix
    #This vector is of the form log(i+1)
    logs = np.log2(np.arange(y_sz) + 2)
    #Convert logs into the divisor for the DCG calculation, of size similarity
    #matrix
    divisors = np.repeat(np.expand_dims(logs, axis=0), x_sz, axis=0)

    #mask out the sorted relevancy matrix to only use the first k relevant
    #retrievals for each item.
    columns = np.repeat(np.expand_dims(np.arange(x_sz), axis=1), y_sz, axis=1)
    numerators = relevancy_matrix[columns, ranks] * k_counts
    #Calculate the final DCG score (note that this isn't expected to sum to 1)
    return np.sum(numerators / divisors, axis=1)

def calculate_k_counts(relevancy_matrix):
    """
    Works out the maximum number of allowed retrievals when working out the
    Discounted Cumulative Gain. For each query the DCG only uses the first k
    items retrieved which constitute the k relevant items for that query
    (otherwise the nDCG scores can be deceptively high for bad rankings).
    Params:
        - relevancy_matrix: matrix of size n1 x n2 where n1 is the number of
          items in the first modality and n2 is the number of items in the
          second modality.  The [ith, jth] element is the semantic relevancy
          between the ith item from the first modality and the jth item from
          the second modality.
    Returns:
        - Matrix of size n1 x n2 (see relevancy matrix for more info). This is
          created as a mask such that if the [ith, jth] element is 1 it
          represents a valid item to use for the calculation of DCG for the
          ith item after sorting. For example, if relevancy matrix of:
        [[1, 0.5, 0],
          [0, 0  , 1]]
          is given, then the k_counts matrix will be:
        [[1, 1, 0],
         [1, 0, 0]]
         i.e. the first row has 2 non-zero items, so the first two retrieved
         items should be used in the calculation. In the second row there is
         only 1 relevant item, therefore only the first retrieved item should
         be used for the DCG calculation.
    """
    return (np.sort(relevancy_matrix)[:, ::-1] > 0).astype(int)

def calculate_IDCG(relevancy_matrix, k_counts):
    """
    Calculates the Ideal Discounted Cumulative Gain (IDCG) which is the value
    of the Discounted Cumulative Gain (DCG) for a perfect retrieval, i.e. the
    items in the second modality were retrieved in order of their descending
    relevancy.
    Params:
        - relevancy_matrix: matrix of size n1 x n2 where n1 is the number of
          items in the first modality and n2 is the number of items in the
          second modality. The [ith, jth] element is the semantic relevancy
          between the ith item from the first modality and the jth item from
          the second modality.
        - k_counts: matrix of size n1 x n2 (see similarity_matrix above) which
          includes information on which items to use to calculate the DCG for
          (see calculate_k_counts for more info on this matrix).
    """
    return calculate_DCG(relevancy_matrix, relevancy_matrix, k_counts)

def calculate_nDCG(similarity_matrix, relevancy_matrix, k_counts=None, IDCG=None, reduction='mean'):
    """
    Calculates the normalised Discounted Cumulative Gain (nDCG) between two
    modalities for the first modality using the Discounted Cumulative Gain
    (DCG) and the Ideal Discounted Cumulative Gain (IDCG).

    nDCG = \frac{DCG}{IDCG}

    Params:
        - similarity_matrix: matrix of size n1 x n2 where n1 is the number of
          items in the first modality and n2 is the number of items in the second
          modality. The [ith,jth] element is the predicted similarity between
          the ith item from the first modality and the jth item from the second
          modality.
        - relevancy_matrix: matrix of size n1 x n2 (see similarity_matrix
          above). The [ith, jth] element is the semantic relevancy between the
          ith item from the first modality and the jth item from the second
          modality.
        - k_counts: optional parameter: matrix of size n1 x n2 (see
          similarity_matrix above) which includes information on which items to
          use to calculate the DCG for (see calculate_k_counts for more info on
          this matrix). This will be calculated using calculate_IDCG if not
          present, but should be pre-processed for efficiency.
        - IDCG: Optional parameter which includes the pre-processed Ideal
          Discounted Cumulative Gain (IDCG). This is a vector of size n1 (see
          similarity_matrix above) which contains the IDCG value for each item
          from the first modality. This will be calculated using calculate_IDCG
          if not present, but should be pre-processed for efficiency.
        - reduction: what to use to reduce the different nDCG scores. By
          default this applies np.mean across all different queries.
    Returns:
        - The nDCG values for the first modality.
    """
    if k_counts is None:
        k_counts = calculate_k_counts(relevancy_matrix)
    DCG = calculate_DCG(similarity_matrix, relevancy_matrix, k_counts)
    if IDCG is None:
        IDCG = calculate_IDCG(relevancy_matrix, k_counts)
    nDCG = DCG /IDCG

    nDCG[np.isnan(nDCG)] = 0
    nDCG[np.isinf(nDCG)] = 0

    if reduction == 'mean':
        return np.mean(nDCG)
    elif reduction is None:
        return nDCG

if __name__ == "__main__":
    precompute_epic_rel_data()