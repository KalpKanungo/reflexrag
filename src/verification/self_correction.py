class SelfCorrector:
    def refine_query(self, query):
        if "how" in query:
            return query + " explanation mechanism"
        if "what" in query:
            return query + " definition details"
        return query + " detailed explanation"

    def correct(self, query, retriever, generator, checker):
        results = retriever.retrieve_multi_hop(query, top_k=5)
        answer, sources = generator.generate(query, results)

        is_valid, score = checker.is_grounded(answer, results)

        if is_valid:
            return answer, sources, score, False

        new_query = self.refine_query(query)

        results = retriever.retrieve_multi_hop(new_query, top_k=5)
        answer, sources = generator.generate(new_query, results)

        is_valid, score = checker.is_grounded(answer, results)

        return answer, sources, score, True