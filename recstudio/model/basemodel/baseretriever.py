from typing import Dict, List, Optional, Tuple, Union
import inspect
import recstudio.eval as eval
import torch
import torch.nn.functional as F
from ..scorer import *
from . import Recommender
from ..loss_func import FullScoreLoss
from recstudio.ann.sampler import *
from recstudio.data import UserDataset, SeqDataset


class BaseRetriever(Recommender):
    def __init__(self, config: Dict = None, **kwargs):
        super(BaseRetriever, self).__init__(config, **kwargs)

        if 'item_encoder' in kwargs:
            assert isinstance(kwargs['item_encoder'], torch.nn.Module), \
                "item_encoder must be torch.nn.Module"
            self.item_encoder = kwargs['item_encoder']
        else:
            self.item_encoder = None

        if 'query_encoder' in kwargs:
            assert isinstance(kwargs['query_encoder'], torch.nn.Module), \
                "query_encoder must be torch.nn.Module"
            self.query_encoder = kwargs['query_encoder']
        else:
            self.query_encoder = None

        if 'scorer' in kwargs:
            assert isinstance(kwargs['scorer'], torch.nn.Module), \
                "scorer must be torch.nn.Module"
            self.score_func = kwargs['scorer']
        else:
            self.score_func = self._get_score_func()

        if 'sampler' in kwargs:
            assert isinstance(kwargs['sampler'], Sampler), \
                "sampler must be recstudio.ann.sampler.Sampler"
            self.sampler = kwargs['sampler']
        else:
            self.sampler = None

        self.use_index = self.config['train']['ann'] is not None and (not config['model']['item_bias'] or
                (isinstance(self.score_func, InnerProductScorer) or isinstance(self.score_func, EuclideanScorer)))

    def _set_data_field(self, data):
        data.use_field = set([data.fuid, data.fiid, data.frating])
        if hasattr(self, 'logger'):
            self.logger.info("The default fields to be used is set as [user_id, item_id, rating]. "
                             "If more fields are needed, please use `self._set_data_field()` to reset.")

    def _init_model(self, train_data, drop_unused_field=True):
        super()._init_model(train_data, drop_unused_field)
        self.query_fields = set(train_data.user_feat.fields).intersection(train_data.use_field)
        if isinstance(train_data, UserDataset) or isinstance(train_data, SeqDataset):
            self.query_fields = self.query_fields | set(["in_"+f for f in self.item_fields])
            if isinstance(train_data, SeqDataset):
                self.query_fields = self.query_fields | set(['seqlen'])

        self.fiid = train_data.fiid
        self.fuid = train_data.fuid
        assert self.fiid in self.item_fields, 'item id is required to use.'

        self.item_encoder = self._get_item_encoder(train_data) if not self.item_encoder else self.item_encoder
        self.query_encoder = self._get_query_encoder(train_data) if not self.query_encoder else self.query_encoder
        self.sampler = self._get_sampler(train_data) if not self.sampler else self.sampler

    def _get_item_feat(self, data):
        if isinstance(data, dict):  # batch
            if len(self.item_fields) == 1:
                return data[self.fiid]
            else:
                return dict((field, value) for field, value in data.items() if field in self.item_fields)
        else:  # neg_item_idx
            if len(self.item_fields) == 1:
                return data
            else:
                device = next(self.parameters()).device
                return self._to_device(self.item_feat[data], device)

    def _get_item_encoder(self, train_data):
        return torch.nn.Embedding(train_data.num_items, self.embed_dim, padding_idx=0)

    def _get_query_feat(self, data):
        if isinstance(data, dict):  # batch
            if (len(self.query_fields) == 1):
                return data[list(self.query_fields)[0]]
            else:
                return dict((field, value) for field, value in data.items() if field in self.query_fields)
        else:  # neg_user as query?
            if len(self.query_fields) == 1:
                return data
            else:
                device = next(self.parameters()).device
                return self._to_device(self.user_feat[data], device)

    def _get_query_encoder(self, train_data):
        if self.fuid in self.query_fields:
            self.logger.warning("No specific query_encoder is configured, "
                                "query_encoder is set as Embedding for user id "
                                "by default due to detect user id is in use_fields")
            return torch.nn.Embedding(train_data.num_users, self.embed_dim, padding_idx=0)
        else:
            self.logger.error("query_encoder missing. Please configure query_encoder. "
                              "If you want to use Embedding for user id as query_encoder, "
                              "please add user id field name in use_field in your configuration.")
            raise ValueError("query_encoder missing.")

    def _get_score_func(self):
        return InnerProductScorer()

    def _get_sampler(self, train_data):
        return UniformSampler(train_data.num_items)

    def _get_item_vector(self):
        # if self.item_encoder is None:
        #     assert hasattr(self, 'item_vector') and self.item_vector is not None, \
        #         'model without item_encoder should have item_vector.'
        #     return self.item_vector
        if len(self.item_fields) == 1 and isinstance(self.item_encoder, torch.nn.Embedding):
            return self.item_encoder.weight[1:]
        else:
            device = next(self.parameters()).device
            output = [self.item_encoder(self._get_item_feat(self._to_device(batch, device)))
                      for batch in self.item_feat.loader(batch_size=self.config['train'].get('item_batch_size', 1024))]
            output = torch.cat(output, dim=0)
            return output[1:]

    def _update_item_vector(self):  # TODO: update frequency setting
        item_vector = self._get_item_vector()
        if not hasattr(self, "item_vector"):
            self.register_buffer('item_vector', item_vector.detach().clone() if isinstance(item_vector, torch.Tensor) \
                else item_vector.copy())
        else:
            self.item_vector = item_vector

        if self.use_index:
            self.ann_index = self.build_index()

    def forward(
            self,
            batch: Dict,
            full_score: bool = False,
            return_query: bool = False,
            return_item: bool = False,
            return_neg_item: bool = False,
            return_neg_id: bool = False
        ):
        # query_vec, pos_item_vec, neg_item_vec,
        output = {}
        pos_items = self._get_item_feat(batch)
        pos_item_vec = self.item_encoder(pos_items)
        if self.sampler is not None:
            if self.neg_count is None:
                raise ValueError("`negative_count` value is required when "
                                 "`sampler` is not none.")

            (log_pos_prob, neg_item_idx, log_neg_prob), query = self.sampling(batch=batch, num_neg=self.neg_count,
                                                                              excluding_hist=self.config['train'].get('excluding_hist', False),
                                                                              method=self.config['train'].get('sampling_method', 'none'), return_query=True)
            pos_score = self.score_func(query, pos_item_vec)
            if batch[self.fiid].dim() > 1:
                pos_score[batch[self.fiid] == 0] = -float('inf')  # padding

            neg_items = self._get_item_feat(neg_item_idx)
            neg_item_vec = self.item_encoder(neg_items)
            neg_score = self.score_func(query, neg_item_vec)
            output['score'] = {'pos_score': pos_score, 'log_pos_prob': log_pos_prob,
                               'neg_score': neg_score, 'log_neg_prob': log_neg_prob}

            if return_neg_item:
                output['neg_item'] = neg_item_vec
            if return_neg_id:
                output['neg_id'] = neg_item_idx
        else:
            query = self.query_encoder(self._get_query_feat(batch))
            pos_score = self.score_func(query, pos_item_vec)
            if batch[self.fiid].dim() > 1:
                pos_score[batch[self.fiid] == 0] = -float('inf')  # padding
            output['score'] = {'pos_score': pos_score}
            if full_score:
                item_vectors = self._get_item_vector()
                all_item_scores = self.score_func(query, item_vectors)
                output['score']['all_score'] = all_item_scores

        if return_query:
            output['query'] = query
        if return_item:
            output['item'] = pos_item_vec
        return output

    def score(self, batch, query=None, neg_id=None):
        # designed for cascade models like RankFlow or CoRR
        if query is None:
            query = self.scorer.query_encoder(self.scorer._get_query_feat(batch))
        if neg_id is not None:
            item_vec = self.scorer.item_encoder(self.scorer_get_item_feat(neg_id))
        else:   # score on positive items in batch
            item_vec = self.scorer.item_encoder(self.scorer_get_item_feat(batch))
        return self.scorer.score_func(query, item_vec)

    def _sample(
        self,
        batch,
        neg: int = 1,
        excluding_hist: bool = False,
        return_query: bool = True
    ):
        query = self.query_encoder(self._get_query_feat(batch))
        pos_items = batch.get(self.fiid, None)
        if excluding_hist:
            user_hist = batch.get('user_hist', None)
            if user_hist is None:
                user_hist = batch.get(self.fiid, None)
        else:
            user_hist = None

        if isinstance(self.sampler, Sampler):
            kwargs = {
                'num_neg': neg,
                'pos_items': pos_items,
            }
            if 'excluding_hist' in inspect.signature(self.sampler.forward).parameters:
                kwargs['excluding_hist'] = excluding_hist
            if 'user_hist' in inspect.signature(self.sampler.forward).parameters:
                kwargs['user_hist'] = user_hist
            if isinstance(self.sampler, RetrieverSampler):
                # assert not _is_query, "RetreiverSampler expected a batch of data instead of queries."
                kwargs.update({
                    'batch': batch if self.sampler.retriever is not None else query,
                })

            else:
                kwargs['query'] = query

            pos_prob, neg_id, neg_prob = self.sampler(**kwargs)

        else:
            raise TypeError("`sampler` only support Sampler type.")

        if return_query:
            return pos_prob, neg_id, neg_prob, query
        else:
            return pos_prob, neg_id, neg_prob

    def sampling(
            self, batch, num_neg, method='none', excluding_hist=False, t=1,
            return_query=False, query=None):
        pos_items = batch.get(self.fiid, None)
        # TODO: consider the case of multi positive items, then the negatives should be sampled for
        # each postive.
        if pos_items.dim() == 1:
            # In order to keep consistent with multi positives, otherwise gather function
            # in brute method will report error.
            pos_items = pos_items.view(-1, 1)

        user_hist = batch.get('user_hist', None)
        if user_hist is None:
            user_hist = batch.get(self.fiid, None)

        if isinstance(num_neg, int):
            num_neg = [num_neg, num_neg]
        elif isinstance(num_neg, (List, Tuple)):
            assert len(num_neg) == 2, "length of negative_count must be 2 \
                when it's list type for retriever_dns sampler."
            assert num_neg[0] >= num_neg[1], "the first element of \
                negative_count must be larger than the second element."
        else:
            raise TypeError("num_neg only support int and List/Tuple type.")

        if method in ("none", "sir", "dns", "toprand", "brute", "top&rand"):

            if method == 'none':
                assert self.sampler is not None, "excepted sampler of retriever to be Sampler, but get None."
                log_pos_prob, neg_id, log_neg_prob, query = self._sample(
                    batch, num_neg[1], excluding_hist, True)

            elif method == 'toprand':
                scores, topk_items, query = self.topk(batch, k=num_neg[0], user_h=user_hist, return_query=True)
                rand_idx = torch.randint(0, num_neg[0], (topk_items.size(0), num_neg[1]),
                                         device=topk_items.device)
                neg_id = torch.gather(topk_items, -1, rand_idx)
                log_neg_prob = torch.zeros_like(neg_id)
                log_pos_prob = None if pos_items is None else torch.zeros_like(pos_items)

            elif method == 'top&rand':
                num_neg_0 = num_neg[1] // 2
                scores, neg_id, query = self.topk(batch, k=num_neg_0, user_h=user_hist, return_query=True)
                num_queries = np.prod(query.shape[:-1])
                rand_sampled_idx = torch.randint(1, self.item_vector.size(0)+1,
                                                 size=(num_queries, num_neg[1]-num_neg_0),
                                                 device=query.device)
                neg_id = torch.cat((neg_id, rand_sampled_idx), dim=-1)
                log_neg_prob = torch.zeros_like(neg_id)
                log_pos_prob = None if pos_items is None else torch.zeros_like(pos_items)

            elif method == 'brute':  # brute force sampling
                query = self.query_encoder(self._get_query_feat(batch)) if query is None else query

                item_vectors = self.item_vector
                if ('item_batch_size' in self.config['train']) and (self.config['train']['item_batch_size'] is not None):
                    # if the items are enormous, the scores of all items should be split into steps to calculate.
                    item_vectors = torch.split(self.item_vector, self.config['train']['item_batch_size'], dim=0)
                    all_score = []
                    for vec in item_vectors:
                        score = self.score_func(query, vec) / t
                        all_score.append(score)
                    all_score = torch.cat(all_score, dim=1)
                else:
                    all_score = self.score_func(query, item_vectors) / t
                all_prob = torch.softmax(all_score, dim=-1)
                all_prob = F.pad(all_prob, pad=(1, 0))   # padding
                sampling_prob = all_prob

                num_pos = 1
                if pos_items is not None:
                    log_pos_prob = torch.log(torch.gather(all_prob, dim=-1, index=pos_items))
                    num_pos = pos_items.size(-1)

                if excluding_hist:
                    from recstudio.utils.utils import mask_with_hist
                    sampling_prob = mask_with_hist(sampling_prob, user_hist, 0)

                sampled_idx = torch.multinomial(sampling_prob, num_neg[1] * num_pos, replacement=True)
                neg_id = sampled_idx
                log_neg_prob = torch.log(torch.gather(sampling_prob, dim=-1, index=sampled_idx))

            elif method in ('sir', 'dns'):
                if pos_items is not None:
                    log_pos_prob, neg_id_pool, _, query = self._sample(
                        batch, num_neg[0], excluding_hist, True)
                else:
                    neg_id_pool, _, query = self._sample(batch,
                                                         num_neg[0], excluding_hist, True)

                neg_item_vector = self.item_encoder(self._get_item_feat(neg_id_pool))
                scores_on_pool_items = self.score_func(query, neg_item_vector)

                if method == 'dns':
                    _, topk_id = torch.topk(scores_on_pool_items, num_neg[1])
                    neg_id = torch.gather(neg_id_pool, -1, topk_id)
                    log_neg_prob = torch.zeros_like(neg_id)
                    log_pos_prob = None if pos_items is None else torch.zeros_like(pos_items)
                else:
                    if pos_items is not None:
                        pos_item_vector = self.item_encoder(self._get_item_feat(batch))
                        pos_score = self.score_func(query, pos_item_vector)
                        log_pos_prob = pos_score
                    probs_on_pool_items = torch.softmax(
                        scores_on_pool_items+torch.finfo(torch.float32).eps, dim=-1)
                    resampled_id = torch.multinomial(probs_on_pool_items, num_neg[1], replacement=True)
                    neg_id = torch.gather(neg_id_pool, dim=-1, index=resampled_id)
                    log_neg_prob = torch.gather(scores_on_pool_items, dim=-1, index=resampled_id)
        else:
            raise NotImplementedError(
                'sampling method only support one of none/brute/is/dns/top/toprand/top&rand')

        if pos_items is not None:
            log_pos_prob = log_pos_prob.view_as(batch.get(self.fiid))
            sampled_result = (log_pos_prob.detach(), neg_id, log_neg_prob.detach())
        else:
            sampled_result = (None, neg_id, log_neg_prob.detach())

        if return_query:
            return sampled_result, query
        else:
            return sampled_result, None

    def build_index(self):
        import faiss
        item_vector = self.item_vector
        if isinstance(self.score_func, scorer.InnerProductScorer):
            metric = faiss.METRIC_INNER_PRODUCT
        elif isinstance(self.score_func, scorer.CosineScorer):
            metric = faiss.METRIC_INNER_PRODUCT
            item_vector = F.normalize(item_vector, dim=1)
        elif isinstance(self.score_func, scorer.EuclideanScorer):
            metric = faiss.METRIC_L2
        else:
            raise ValueError(f'ANN index do not support the {type(self.score_func).__name__}')
        num_item, dim = item_vector.shape
        if 'HNSWx' in self.config['ann']['index']:
            ann_search = self.config['ann']['index'].replace('x', '32')
        elif 'IVFx' in self.config['ann']['index']:
            ann_search = self.config['ann']['index'].replace('x', str(int(4*np.sqrt(num_item))))
        else:   # IndexPQ, IndexLSH, etc
            ann_search = self.config['ann']['index']
        index = faiss.index_factory(dim, ann_search, metric)
        if 'parameter' in self.config['ann'] and self.config['ann']['parameter'] is not None:
            for k, v in self.config['ann']['parameter'].items():
                faiss.ParameterSpace().set_index_parameter(index, k, v)
        if not index.is_trained:
            index.train(item_vector.numpy())
        index.add(item_vector.numpy())
        return index

    def topk(self, batch, k, user_h=None, return_query=False):
        query = self.query_encoder(self._get_query_feat(batch))
        more = user_h.size(1) if user_h is not None else 0
        if self.use_index:
            if isinstance(self.score_func, CosineScorer):
                score, topk_items = self.ann_index.search(
                    torch.nn.functional.normalize(query, dim=1).numpy(), k + more)
            else:
                score, topk_items = self.ann_index.search(query.numpy(), k + more)
        else:
            score, topk_items = torch.topk(self.score_func(query, self.item_vector), k + more)
        topk_items = topk_items + 1
        if user_h is not None:
            existing, _ = user_h.sort()
            idx_ = torch.searchsorted(existing, topk_items)
            idx_[idx_ == existing.size(1)] = existing.size(1) - 1
            score[torch.gather(existing, 1, idx_) == topk_items] = -float('inf')
            score, idx = score.topk(k)
            topk_items = torch.gather(topk_items, 1, idx)

        if return_query:
            return score, topk_items, query
        else:
            return score, topk_items

    def training_step(self, batch):
        output = self.forward(batch, isinstance(self.loss_fn, FullScoreLoss))
        score = output['score']
        score['label'] = batch[self.frating]
        loss_value = self.loss_fn(**score)
        return loss_value

    def validation_step(self, batch):
        eval_metric = self.config['eval']['val_metrics']
        cutoff = self.config['eval']['cutoff'][0] if isinstance(self.config['eval']['cutoff'], list) else self.config['eval']['cutoff']
        return self._test_step(batch, eval_metric, [cutoff])

    def test_step(self, batch):
        eval_metric = self.config['eval']['test_metrics']
        cutoffs = self.config['eval']['cutoff'] if isinstance(self.config['eval']['cutoff'], list) else [self.config['eval']['cutoff']]
        return self._test_step(batch, eval_metric, cutoffs)

    def _test_step(self, batch, metric, cutoffs):
        rank_m = eval.get_rank_metrics(metric)
        topk = self.config['eval']['topk']
        bs = batch[self.frating].size(0)
        assert len(rank_m) > 0
        score, topk_items = self.topk(batch, topk, batch['user_hist'])
        if batch[self.fiid].dim() > 1:
            target, _ = batch[self.fiid].sort()
            idx_ = torch.searchsorted(target, topk_items)
            idx_[idx_ == target.size(1)] = target.size(1) - 1
            label = torch.gather(target, 1, idx_) == topk_items
            pos_rating = batch[self.frating]
        else:
            label = batch[self.fiid].view(-1, 1) == topk_items
            pos_rating = batch[self.frating].view(-1, 1)
        return {f"{name}@{cutoff}": func(label, pos_rating, cutoff) for cutoff in cutoffs for name, func in rank_m}, bs
