'Score registration hypotheses in bounded chunks.'

from typing import Optional

import torch

from pareconv.modules.registration.combineRegisraition import (HypothesisProposer,
                                                              LocalGlobalRegistration,
                                                              combineRegisraition)

from protassem.fitting.hypothesis_scoring import select_best_hypothesis


class ChunkedLocalGlobalRegistration(LocalGlobalRegistration):

    def __init__(self, *args, chunk_size=0, **kwargs):
        super().__init__(*args, **kwargs)
        self.chunk_size = int(chunk_size)

    def local_to_global_registration(self, ref_knn_points, src_knn_points, score_mat, corr_mat):
        batch_indices, ref_indices, src_indices = torch.nonzero(corr_mat, as_tuple=True)
        global_ref_corr_points = ref_knn_points[batch_indices, ref_indices]
        global_src_corr_points = src_knn_points[batch_indices, src_indices]
        global_corr_scores = score_mat[batch_indices, ref_indices, src_indices]

        if self.correspondence_limit is not None and global_corr_scores.shape[0] > self.correspondence_limit:
            corr_scores, sel_indices = global_corr_scores.topk(k=self.correspondence_limit, largest=True)
            ref_corr_points = global_ref_corr_points[sel_indices]
            src_corr_points = global_src_corr_points[sel_indices]
        else:
            ref_corr_points = global_ref_corr_points
            src_corr_points = global_src_corr_points
            corr_scores = global_corr_scores

        unique_masks = torch.ne(batch_indices[1:], batch_indices[:-1])
        unique_indices = torch.nonzero(unique_masks, as_tuple=True)[0] + 1
        unique_indices = unique_indices.detach().cpu().numpy().tolist()
        unique_indices = [0] + unique_indices + [batch_indices.shape[0]]
        chunks = [
            (x, y) for x, y in zip(unique_indices[:-1], unique_indices[1:])
            if y - x >= self.correspondence_threshold
        ]

        batch_size = len(chunks)
        if batch_size > 0:
            batch_ref_corr_points, batch_src_corr_points, batch_corr_scores = self.convert_to_batch(
                global_ref_corr_points, global_src_corr_points, global_corr_scores, chunks
            )
            batch_transforms = self.procrustes(batch_src_corr_points, batch_ref_corr_points,
                                               batch_corr_scores)
            _, best_inlier_mask = select_best_hypothesis(
                ref_corr_points, src_corr_points, batch_transforms,
                self.acceptance_radius, self.chunk_size
            )
            cur_corr_scores = corr_scores * best_inlier_mask.float()
        else:
            estimated_transform = self.procrustes(src_corr_points, ref_corr_points, corr_scores)
            cur_corr_scores = self.recompute_correspondence_scores(
                ref_corr_points, src_corr_points, corr_scores, estimated_transform
            )

        estimated_transform = self.procrustes(src_corr_points, ref_corr_points, cur_corr_scores)
        for _ in range(self.num_refinement_steps - 1):
            cur_corr_scores = self.recompute_correspondence_scores(
                ref_corr_points, src_corr_points, corr_scores, estimated_transform
            )
            estimated_transform = self.procrustes(src_corr_points, ref_corr_points, cur_corr_scores)

        return (global_ref_corr_points, global_src_corr_points, global_corr_scores,
                estimated_transform)


class ChunkedHypothesisProposer(HypothesisProposer):

    def __init__(self, *args, chunk_size=0, **kwargs):
        super().__init__(*args, **kwargs)
        self.chunk_size = int(chunk_size)

    def feature_based_hypothesis_proposer(self, ref_knn_points, src_knn_points,
                                          ref_knn_feats, src_knn_feats, score_mat, corr_mat,
                                          voter_corr_mat):
        batch_indices, ref_indices, src_indices = torch.nonzero(corr_mat, as_tuple=True)
        global_ref_corr_points = ref_knn_points[batch_indices, ref_indices]
        global_src_corr_points = src_knn_points[batch_indices, src_indices]
        global_corr_scores = score_mat[batch_indices, ref_indices, src_indices]
        ref_corr_feats = ref_knn_feats[batch_indices, ref_indices]
        src_corr_feats = src_knn_feats[batch_indices, src_indices]

        batch_v_indices, ref_v_indices, src_v_indices = torch.nonzero(voter_corr_mat, as_tuple=True)
        ref_corr_points = ref_knn_points[batch_v_indices, ref_v_indices]
        src_corr_points = src_knn_points[batch_v_indices, src_v_indices]
        corr_scores = score_mat[batch_v_indices, ref_v_indices, src_v_indices]

        transformation_hypotheses = self.extract_fine_transforms(
            ref_corr_feats, src_corr_feats, global_ref_corr_points, global_src_corr_points)

        _, best_inlier_mask = select_best_hypothesis(
            ref_corr_points, src_corr_points, transformation_hypotheses,
            self.acceptance_radius, self.chunk_size
        )
        cur_corr_scores = corr_scores * best_inlier_mask.float()

        estimated_transform = self.procrustes(src_corr_points, ref_corr_points, cur_corr_scores)
        for _ in range(self.num_refinement_steps - 1):
            cur_corr_scores = self.recompute_correspondence_scores(
                ref_corr_points, src_corr_points, corr_scores, estimated_transform
            )
            estimated_transform = self.procrustes(src_corr_points, ref_corr_points, cur_corr_scores)

        return (global_ref_corr_points, global_src_corr_points, global_corr_scores,
                estimated_transform, transformation_hypotheses, ref_corr_feats, src_corr_feats)


def build_registration(hypothesis_chunk=0, **kwargs):
    registration = combineRegisraition(**kwargs)
    chunk = int(hypothesis_chunk)
    if chunk <= 0:
        return registration

    def lgr_kwargs():
        return {
            "k": registration.lgr.k,
            "acceptance_radius": registration.lgr.acceptance_radius,
            "mutual": registration.lgr.mutual,
            "confidence_threshold": registration.lgr.confidence_threshold,
            "use_dustbin": registration.lgr.use_dustbin,
            "use_global_score": registration.lgr.use_global_score,
            "correspondence_threshold": registration.lgr.correspondence_threshold,
            "correspondence_limit": registration.lgr.correspondence_limit,
            "num_refinement_steps": registration.lgr.num_refinement_steps,
        }

    def hp_kwargs():
        return {
            "k": registration.hp.k,
            "acceptance_radius": registration.hp.acceptance_radius,
            "confidence_threshold": registration.hp.confidence_threshold,
            "num_hypotheses": registration.hp.num_hypotheses,
            "num_refinement_steps": registration.hp.num_refinement_steps,
        }

    registration.lgr = ChunkedLocalGlobalRegistration(chunk_size=chunk, **lgr_kwargs())
    registration.hp = ChunkedHypothesisProposer(chunk_size=chunk, **hp_kwargs())
    return registration
