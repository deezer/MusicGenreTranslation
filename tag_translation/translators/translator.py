class Translator():

    def train_and_evaluate(self, train_data, target_data, eval_data, eval_target, score_function):
        raise NotImplementedError()

    def predict_scores(self, eval_data):
        raise NotImplementedError()

    def _evaluate(self, eval_data, eval_target, score_function, **kwargs):
        scores = self.predict_scores(eval_data)
        return score_function(eval_target, scores, **kwargs)


class LogRegTranslator(Translator):

    def predict_scores(self, eval_data):
        return eval_data.dot(self.W.T) + self.b.reshape((1, -1))