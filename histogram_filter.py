import numpy as np


class HistogramFilter(object):
    """
    Class HistogramFilter implements the Bayes Filter on a discretized grid space.
    """


    def histogram_filter(self, cmap, belief, action, observation):
        '''
        Takes in a prior belief distribution, a colormap, action, and observation, and returns the posterior
        belief distribution according to the Bayes Filter.
        :param cmap: The binary NxM colormap known to the robot.
        :param belief: An NxM numpy ndarray representing the prior belief.
        :param action: The action as a numpy ndarray. [(1, 0), (-1, 0), (0, 1), (0, -1)]
        :param observation: The observation from the color sensor. [0 or 1].
        :return: The posterior distribution.
        '''

        ### Your Algorithm goes Below.
        correct_obs=0.9
        wrong_obs=0.1
        Moving_prob = 0.9
        Staying_prob = 0.1
        uniform = 1 / (cmap.shape[0] * cmap.shape[1])
        Prior_Bel = np.zeros([cmap.shape[0], cmap.shape[1]])

        for i in range(cmap.shape[0]):
            for j in range(cmap.shape[1]):
                if i+action[1] >= 0 and i+action[1] < cmap.shape[0] and j+action[0] >=0 and j+action[0] < cmap.shape[1]:
                    Prior_Bel[i+action[1]][j+action[0]] +=Moving_prob*belief[i][j]
                    Prior_Bel[i][j] += Staying_prob*belief[i][j]
                else:
                    Prior_Bel[i][j] += belief[i][j]
                   
        normalize=0
        for i in range(cmap.shape[0]):
            for j in range(cmap.shape[1]):
              if observation == cmap[i][j]:
                Prior_Bel[i][j] = correct_obs*Prior_Bel[i][j]
              else:
                Prior_Bel[i][j] = wrong_obs*Prior_Bel[i][j]
              normalize = normalize+Prior_Bel[i][j]
        Prior_Bel = Prior_Bel/normalize

        return Prior_Bel