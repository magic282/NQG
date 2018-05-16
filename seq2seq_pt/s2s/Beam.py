from __future__ import division

# Class for managing the internals of the beam search process.
#
#
#         hyp1#-hyp1---hyp1 -hyp1
#                 \             /
#         hyp2 \-hyp2 /-hyp2#hyp2
#                               /      \
#         hyp3#-hyp3---hyp3 -hyp3
#         ========================
#
# Takes care of beams, back pointers, and scores.

import torch
import s2s

try:
    import ipdb
except ImportError:
    pass


class Beam(object):
    def __init__(self, size, cuda=False):

        self.size = size
        self.done = False

        self.tt = torch.cuda if cuda else torch

        # The score for each translation on the beam.
        self.scores = self.tt.FloatTensor(size).zero_()
        self.all_scores = []
        self.all_length = []

        # The backpointers at each time-step.
        self.prevKs = []

        # The outputs at each time-step.
        self.nextYs = [self.tt.LongTensor(size).fill_(s2s.Constants.PAD)]
        self.nextYs[0][0] = s2s.Constants.BOS
        self.nextYs_true = [self.tt.LongTensor(size).fill_(s2s.Constants.PAD)]
        self.nextYs_true[0][0] = s2s.Constants.BOS

        # The attentions (matrix) for each time.
        self.attn = []

        # is copy for each time
        self.isCopy = []

    # Get the outputs for the current timestep.
    def getCurrentState(self):
        return self.nextYs[-1]

    # Get the backpointers for the current timestep.
    def getCurrentOrigin(self):
        return self.prevKs[-1]

    #  Given prob over words for every last beam `wordLk` and attention
    #   `attnOut`: Compute and update the beam search.
    #
    # Parameters:
    #
    #     * `wordLk`- probs of advancing from the last step (K x words)
    #     * `attnOut`- attention at the last step
    #
    # Returns: True if beam search is complete.
    def advance(self, wordLk, copyLk, attnOut):
        numWords = wordLk.size(1)
        numSrc = copyLk.size(1)
        numAll = numWords + numSrc
        allScores = torch.cat((wordLk, copyLk), dim=1)

        # self.length += 1  # TODO: some is finished so do not acc length for them
        if len(self.prevKs) > 0:
            finish_index = self.nextYs[-1].eq(s2s.Constants.EOS)
            if any(finish_index):
                # wordLk.masked_fill_(finish_index.unsqueeze(1).expand_as(wordLk), -float('inf'))
                allScores.masked_fill_(finish_index.unsqueeze(1).expand_as(allScores), -float('inf'))
                for i in range(self.size):
                    if self.nextYs[-1][i] == s2s.Constants.EOS:
                        # wordLk[i][s2s.Constants.EOS] = 0
                        allScores[i][s2s.Constants.EOS] = 0
            # set up the current step length
            cur_length = self.all_length[-1]
            for i in range(self.size):
                cur_length[i] += 0 if self.nextYs[-1][i] == s2s.Constants.EOS else 1

        # Sum the previous scores.
        if len(self.prevKs) > 0:
            prev_score = self.all_scores[-1]
            # now_acc_score = wordLk + prev_score.unsqueeze(1).expand_as(wordLk)
            # beamLk = now_acc_score / cur_length.unsqueeze(1).expand_as(now_acc_score)
            now_acc_score = allScores + prev_score.unsqueeze(1).expand_as(allScores)
            beamLk = now_acc_score / cur_length.unsqueeze(1).expand_as(now_acc_score)
        else:
            self.all_length.append(self.tt.FloatTensor(self.size).fill_(1))
            # beamLk = wordLk[0]
            beamLk = allScores[0]

        flatBeamLk = beamLk.view(-1)

        bestScores, bestScoresId = flatBeamLk.topk(self.size, 0, True, True)
        self.scores = bestScores

        # bestScoresId is flattened beam x word array, so calculate which
        # word and beam each score came from
        prevK = bestScoresId / numAll
        # predict = bestScoresId - prevK * numWords
        predict = bestScoresId - prevK * numAll
        isCopy = predict.ge(self.tt.LongTensor(self.size).fill_(numWords)).long()
        final_predict = predict * (1 - isCopy) + isCopy * s2s.Constants.UNK

        if len(self.prevKs) > 0:
            self.all_length.append(cur_length.index_select(0, prevK))
            self.all_scores.append(now_acc_score.view(-1).index_select(0, bestScoresId))
        else:
            self.all_scores.append(self.scores)

        self.prevKs.append(prevK)
        self.nextYs.append(final_predict)
        self.nextYs_true.append(predict)
        self.isCopy.append(isCopy)
        self.attn.append(attnOut.index_select(0, prevK))

        # End condition is when every one is EOS.
        if all(self.nextYs[-1].eq(s2s.Constants.EOS)):
            self.done = True

        return self.done

    def sortBest(self):
        return torch.sort(self.scores, 0, True)

    # Get the score of the best in the beam.
    def getBest(self):
        scores, ids = self.sortBest()
        return scores[1], ids[1]

    # Walk back to construct the full hypothesis.
    #
    # Parameters.
    #
    #     * `k` - the position in the beam to construct.
    #
    # Returns.
    #
    #     1. The hypothesis
    #     2. The attention at each time step.
    def getHyp(self, k):
        hyp, attn = [], []
        isCopy, copyPos = [], []
        # print(len(self.prevKs), len(self.nextYs), len(self.attn))
        for j in range(len(self.prevKs) - 1, -1, -1):
            hyp.append(self.nextYs[j + 1][k])
            attn.append(self.attn[j][k])
            isCopy.append(self.isCopy[j][k])
            copyPos.append(self.nextYs_true[j + 1][k])
            k = self.prevKs[j][k]

        return hyp[::-1],  isCopy[::-1], copyPos[::-1],torch.stack(attn[::-1])
